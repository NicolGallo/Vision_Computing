"""
Estudio 4: Ablaciones de Arquitectura para FCN8s
- 4.1 Backbone: ResNet50 (base) vs ResNet18 (ligero)
- 4.2 Upsampling: ConvTranspose2d (base) vs Upsample bilinear + Conv

Incluye reproducibilidad, logging, métricas, figuras, carga segura de checkpoints
con weights_only=True, y modo eval_only para usar checkpoints existentes.
"""

import argparse
import json
import os
import random
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from lab05_utils import plot_segmentation_results

# Permitir carga segura de checkpoints con weights_only=True
torch.serialization.add_safe_globals([
    np.core.multiarray.scalar,
    np.dtype,
    np.ndarray,
    np.dtypes.Float64DType,
    np.dtypes.Float32DType,
    np.dtypes.Int64DType,
    np.dtypes.Int32DType,
])

# ===============================#
#  Reproducibilidad global       #
# ===============================#
SEED = 69415

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
except Exception:
    pass


def worker_init_fn(worker_id: int) -> None:
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ================== Logging helpers ==================#
def setup_logger(log_path: str):
    import logging

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger(log_path)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ================== FCN8s variants ==================#
class _BilinearInitMixin:
    @staticmethod
    def _get_bilinear_filter(kernel_h: int, kernel_w: int, in_channels: int, out_channels: int) -> torch.Tensor:
        factor = (kernel_h + 1) // 2
        center = factor - 1 if kernel_h % 2 == 1 else factor - 0.5
        og = np.ogrid[:kernel_h, :kernel_w]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        filt = torch.from_numpy(filt).float()
        weight = torch.zeros(in_channels, out_channels, kernel_h, kernel_w)
        for i in range(min(in_channels, out_channels)):
            weight[i, i, :, :] = filt
        return weight

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                in_ch, out_ch, h, w = m.weight.data.size()
                weight = self._get_bilinear_filter(h, w, in_ch, out_ch)
                m.weight.data.copy_(weight)


def _make_backbone(name: str):
    if name == "resnet50":
        resnet = models.resnet50(weights="DEFAULT")
        c3, c4, c5 = 512, 1024, 2048
    elif name == "resnet18":
        resnet = models.resnet18(weights="DEFAULT")
        c3, c4, c5 = 128, 256, 512
    else:
        raise ValueError(f"Backbone no soportado: {name}")
    layers = {
        "conv1": resnet.conv1,
        "bn1": resnet.bn1,
        "relu": resnet.relu,
        "maxpool": resnet.maxpool,
        "layer1": resnet.layer1,
        "layer2": resnet.layer2,
        "layer3": resnet.layer3,
        "layer4": resnet.layer4,
    }
    return layers, (c3, c4, c5)


class FCN8sResNet50(_BilinearInitMixin, nn.Module):
    def __init__(self, n_classes: int = 21):
        super().__init__()
        layers, (c3, c4, c5) = _make_backbone("resnet50")
        self.conv1 = layers["conv1"]
        self.bn1 = layers["bn1"]
        self.relu = layers["relu"]
        self.maxpool = layers["maxpool"]
        self.layer1 = layers["layer1"]
        self.layer2 = layers["layer2"]
        self.layer3 = layers["layer3"]
        self.layer4 = layers["layer4"]

        self.score_pool3 = nn.Conv2d(c3, n_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(c4, n_classes, kernel_size=1)
        self.score_fr = nn.Conv2d(c5, n_classes, kernel_size=1)

        self.upscore2 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.upscore2_pool4 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.upscore8 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=16, stride=8, padding=4, bias=False)
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        pool3 = self.layer2(x)
        pool4 = self.layer3(pool3)
        x = self.layer4(pool4)

        score_fr = self.score_fr(x)
        upscore2 = self.upscore2(score_fr)

        pool4_score = self.score_pool4(pool4)
        fuse_pool4 = upscore2 + pool4_score
        upscore_pool4 = self.upscore2_pool4(fuse_pool4)

        pool3_score = self.score_pool3(pool3)
        fuse = upscore_pool4 + pool3_score
        x = self.upscore8(fuse)
        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x


class FCN8sResNet18(FCN8sResNet50):
    def __init__(self, n_classes: int = 21):
        super().__init__(n_classes=n_classes)
        layers, (c3, c4, c5) = _make_backbone("resnet18")
        self.conv1 = layers["conv1"]
        self.bn1 = layers["bn1"]
        self.relu = layers["relu"]
        self.maxpool = layers["maxpool"]
        self.layer1 = layers["layer1"]
        self.layer2 = layers["layer2"]
        self.layer3 = layers["layer3"]
        self.layer4 = layers["layer4"]

        self.score_pool3 = nn.Conv2d(c3, n_classes, kernel_size=1)  # 128
        self.score_pool4 = nn.Conv2d(c4, n_classes, kernel_size=1)  # 256
        self.score_fr = nn.Conv2d(c5, n_classes, kernel_size=1)     # 512
        self._initialize_weights()


class FCN8sBilinear(FCN8sResNet50):
    def __init__(self, n_classes: int = 21):
        super().__init__(n_classes=n_classes)
        # Reemplazar deconvs por Upsample + Conv para evitar artefactos
        self.upscore2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(n_classes, n_classes, kernel_size=3, padding=1, bias=False),
        )
        self.upscore2_pool4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(n_classes, n_classes, kernel_size=3, padding=1, bias=False),
        )
        self.upscore8 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False),
            nn.Conv2d(n_classes, n_classes, kernel_size=3, padding=1, bias=False),
        )


# ================== Losses ==================#
class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, ignore_index: int = 255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = F.softmax(pred, dim=1)
        target_one_hot = torch.zeros_like(pred)
        valid_mask = (target != self.ignore_index).unsqueeze(1)
        target_filtered = torch.where(valid_mask.squeeze(1), target, torch.zeros_like(target))
        target_one_hot.scatter_(1, target_filtered.unsqueeze(1), 1)
        target_one_hot = target_one_hot * valid_mask
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, ignore_index: int = 255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(pred, target, reduction="none", ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        valid_mask = (target != self.ignore_index).float()
        focal_loss = focal_loss * valid_mask
        valid_elements = valid_mask.sum()
        if valid_elements == 0:
            return torch.tensor(0.0, device=pred.device)
        return focal_loss.sum() / valid_elements


class CombinedLoss(nn.Module):
    def __init__(self, weights: Dict[str, float] = None, ignore_index: int = 255):
        super().__init__()
        self.weights = weights or {"ce": 0.3, "dice": 0.5, "focal": 0.2}
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        self.focal_loss = FocalLoss(ignore_index=ignore_index)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        return self.weights["ce"] * ce + self.weights["dice"] * dice + self.weights["focal"] * focal


# ================== Métricas ==================#
def calculate_miou(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 255):
    mask = target != ignore_index
    pred = torch.where(mask, pred, torch.full_like(pred, ignore_index))
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    mask_np = mask.cpu().numpy()
    pred_np = np.where(mask_np, pred_np, -1)
    target_np = np.where(mask_np, target_np, -1)
    ious = []
    for c in range(num_classes):
        pm = pred_np == c
        tm = target_np == c
        inter = np.logical_and(pm, tm).sum()
        union = np.logical_or(pm, tm).sum()
        ious.append(float("nan") if union == 0 else inter / union)
    ious = np.array(ious)
    return np.nanmean(ious), ious


def calculate_pixel_accuracy(pred: torch.Tensor, target: torch.Tensor, ignore_index: int = 255) -> float:
    valid_mask = target != ignore_index
    correct = ((pred == target) & valid_mask).float().sum()
    total = valid_mask.float().sum()
    if total == 0:
        return 0.0
    return (correct / total).item()


# ================== Entrenamiento y validación ==================#
def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    total_miou = 0.0
    for images, masks in tqdm(dataloader, desc="Training"):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        pred = outputs.argmax(dim=1)
        num_classes = outputs.shape[1]
        miou, _ = calculate_miou(pred, masks, num_classes=num_classes, ignore_index=255)
        total_loss += loss.item()
        total_miou += miou
    return total_loss / len(dataloader), total_miou / len(dataloader)


def validate(model, dataloader, device, num_classes, criterion=None, return_class_iou=False):
    model.eval()
    total_miou = 0.0
    total_pa = 0.0
    total_loss = 0.0
    intersections = np.zeros(num_classes, dtype=np.float64)
    unions = np.zeros(num_classes, dtype=np.float64)
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            if criterion is not None:
                total_loss += criterion(outputs, masks).item()
            pred = outputs.argmax(dim=1)
            miou, _ = calculate_miou(pred, masks, num_classes=num_classes, ignore_index=255)
            pa = calculate_pixel_accuracy(pred, masks, ignore_index=255)
            total_miou += miou
            total_pa += pa

            valid_mask = masks != 255
            pred_valid = torch.where(valid_mask, pred, torch.full_like(pred, -1))
            target_valid = torch.where(valid_mask, masks, torch.full_like(masks, -1))
            for c in range(num_classes):
                pc = pred_valid == c
                tc = target_valid == c
                intersections[c] += torch.logical_and(pc, tc).sum().item()
                unions[c] += torch.logical_or(pc, tc).sum().item()
    class_ious = np.full(num_classes, np.nan)
    valid_union = unions > 0
    class_ious[valid_union] = intersections[valid_union] / unions[valid_union]
    avg_miou = total_miou / len(dataloader)
    avg_pa = total_pa / len(dataloader)
    avg_loss = total_loss / len(dataloader) if criterion is not None else float("nan")
    if return_class_iou:
        return avg_miou, avg_pa, avg_loss, class_ious
    return avg_miou, avg_pa, avg_loss, None


def visualize_predictions(model, dataloader, device, num_samples, save_path):
    model.eval()
    with torch.no_grad():
        images_batch, masks_batch = next(iter(dataloader))
        images_batch, masks_batch = images_batch.to(device), masks_batch.to(device)
        outputs = model(images_batch)
        predictions = outputs.argmax(dim=1)
        num_to_plot = min(num_samples, images_batch.size(0))
        fig = plot_segmentation_results(
            images=images_batch[:num_to_plot],
            masks=masks_batch[:num_to_plot],
            predictions=predictions[:num_to_plot],
            title=f"Predicciones (primeros {num_to_plot} ejemplos)",
        )
        plt.show()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")


# ================== Dataset ==================#
class VOCSegmentationDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train", image_size: int = 256, use_augmentation: bool = True):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.use_augmentation = use_augmentation and split == "train"
        if not os.path.isabs(root_dir):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.abspath(os.path.join(script_dir, root_dir))
        self.images_dir = os.path.join(root_dir, "JPEGImages")
        self.masks_dir = os.path.join(root_dir, "SegmentationClass")
        split_file = os.path.join(root_dir, "ImageSets", "Segmentation", f"{split}.txt")
        if not os.path.exists(self.images_dir) or not os.path.exists(self.masks_dir):
            print(f"Warning: Could not find VOC dataset at {root_dir}. Using synthetic data.")
            self.use_synthetic = True
            self.length = 100 if split == "train" else 20
            return
        if os.path.exists(split_file):
            with open(split_file, "r") as f:
                self.image_ids = [line.strip() for line in f.readlines()]
            print(f"✓ Loaded {len(self.image_ids)} {split} images from VOC")
        else:
            print(f"Warning: Split file not found: {split_file}. Using synthetic data.")
            self.use_synthetic = True
            self.length = 100 if split == "train" else 20

    def __len__(self) -> int:
        if hasattr(self, "use_synthetic"):
            return self.length
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        if hasattr(self, "use_synthetic"):
            image = torch.randn(3, self.image_size, self.image_size)
            mask = torch.randint(0, 21, (self.image_size, self.image_size)).long()
            image = TF.normalize(
                image,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
            return image, mask

        image_id = self.image_ids[idx]
        image_path = os.path.join(self.images_dir, f"{image_id}.jpg")
        mask_path = os.path.join(self.masks_dir, f"{image_id}.png")
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        if self.use_augmentation:
            image, mask = self._apply_augmentation(image, mask)
        else:
            image = TF.resize(image, [self.image_size, self.image_size], interpolation=Image.BILINEAR)
            mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=Image.NEAREST)
        image = TF.to_tensor(image)
        mask = torch.from_numpy(np.array(mask)).long()
        image = TF.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        return image, mask

    def _apply_augmentation(self, image: Image.Image, mask: Image.Image):
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        scale = random.uniform(0.5, 2.0)
        w, h = image.size
        new_w, new_h = int(w * scale), int(h * scale)
        image = TF.resize(image, [new_h, new_w], interpolation=Image.BILINEAR)
        mask = TF.resize(mask, [new_h, new_w], interpolation=Image.NEAREST)
        w, h = image.size
        if w < self.image_size or h < self.image_size:
            pad_h = max(self.image_size - h, 0)
            pad_w = max(self.image_size - w, 0)
            image = TF.pad(image, [0, 0, pad_w, pad_h], fill=0)
            mask = TF.pad(mask, [0, 0, pad_w, pad_h], fill=255)
            w, h = image.size
        i = random.randint(0, h - self.image_size)
        j = random.randint(0, w - self.image_size)
        image = TF.crop(image, i, j, self.image_size, self.image_size)
        mask = TF.crop(mask, i, j, self.image_size, self.image_size)
        if random.random() > 0.5:
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
        if random.random() > 0.5:
            image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
        if random.random() > 0.5:
            image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            image = TF.rotate(image, angle, interpolation=Image.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=Image.NEAREST)
        return image, mask


# ================== Experimentos ==================#
def build_model(name: str, n_classes: int) -> nn.Module:
    name = name.lower()
    if name == "fcn8s":
        return FCN8sResNet50(n_classes=n_classes)
    if name == "fcn8s_resnet18":
        return FCN8sResNet18(n_classes=n_classes)
    if name == "fcn8s_bilinear":
        return FCN8sBilinear(n_classes=n_classes)
    raise ValueError(f"Modelo no soportado: {name}")


def run_experiment(
    model_name: str,
    epochs: int,
    batch_size: int,
    image_size: int,
    learning_rate: float,
    data_dir: str,
    output_dir: str = None,
    eval_only: bool = False,
) -> None:
    config = {
        "model": model_name,
        "n_classes": 21,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "device": device,
        "image_size": image_size,
        "data_dir": data_dir,
        "num_workers": min(8, os.cpu_count()),
        "use_amp": True,
        "weight_decay": 1e-4,
        "warmup_epochs": 3,
        "use_augmentation": True,
    }
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_out_dir = os.path.abspath(output_dir) if output_dir else os.path.abspath(os.path.join(script_dir, "..", ".."))
    os.makedirs(base_out_dir, exist_ok=True)
    log_dir = os.path.join(base_out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"log_{config['model']}.txt")
    logger = setup_logger(log_path)

    if eval_only:
        logger.info(f"Evaluación (sin entrenamiento) de {config['model']}")
    else:
        logger.info(f"Entrenando {config['model']} | Backbone/Up: {config['model']}")
    logger.info("=" * 60)

    train_dataset = VOCSegmentationDataset(
        root_dir=config["data_dir"],
        split="train",
        image_size=config["image_size"],
        use_augmentation=config["use_augmentation"],
    )
    val_dataset = VOCSegmentationDataset(
        root_dir=config["data_dir"],
        split="val",
        image_size=config["image_size"],
        use_augmentation=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
        generator=g,
    )

    model = build_model(config["model"], config["n_classes"]).to(config["device"])
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parámetros: {num_params:,}")

    criterion = CombinedLoss()
    best_miou = 0.0
    best_checkpoint_path = os.path.join(base_out_dir, f"best_{config['model']}_model.pth")
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_mious: list[float] = []
    val_pas: list[float] = []
    lr_history: list[float] = []
    metrics_path = os.path.join(base_out_dir, f"metrics_{config['model']}.jsonl")

    if not eval_only:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
            betas=(0.9, 0.999),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-7,
        )
        scaler = torch.amp.GradScaler("cuda") if config["use_amp"] and torch.cuda.is_available() else None
        patience = 30
        patience_counter = 0

        for epoch in range(config["epochs"]):
            logger.info(f"Epoch [{epoch + 1}/{config['epochs']}]")
            if epoch < config["warmup_epochs"]:
                warmup_factor = (epoch + 1) / config["warmup_epochs"]
                for param_group in optimizer.param_groups:
                    param_group["lr"] = config["learning_rate"] * warmup_factor
                logger.info(f"Warm-up: LR = {optimizer.param_groups[0]['lr']:.6f}")

            train_loss, train_miou = train_epoch(model, train_loader, optimizer, criterion, config["device"], scaler)
            val_miou, val_pa, val_loss, _ = validate(
                model, val_loader, config["device"], num_classes=config["n_classes"], criterion=criterion
            )
            if epoch >= config["warmup_epochs"]:
                scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Train Loss: {train_loss:.4f} | Train mIoU: {train_miou:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f} | Val PA: {val_pa:.4f} | LR: {current_lr:.6f}"
            )
            with open(metrics_path, "a") as mf:
                mf.write(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "train_loss": train_loss,
                            "train_miou": train_miou,
                            "val_loss": val_loss,
                            "val_miou": val_miou,
                            "val_pa": val_pa,
                            "lr": current_lr,
                            "model": config["model"],
                        }
                    )
                    + "\n"
                )
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_mious.append(val_miou)
            val_pas.append(val_pa)
            lr_history.append(current_lr)

            if val_miou > best_miou:
                best_miou = val_miou
                patience_counter = 0
                torch.save(
                    {"epoch": epoch, "model_state_dict": model.state_dict(), "best_miou": best_miou},
                    best_checkpoint_path,
                )
                logger.info(f"✓ Guardado mejor modelo con mIoU: {best_miou:.4f}")
                logger.info(f"   Path: {best_checkpoint_path}")
            else:
                patience_counter += 1
                logger.info(f"Sin mejora por {patience_counter}/{patience} épocas")
                if patience_counter >= patience:
                    logger.info("Early stopping activado.")
                    break

        logger.info(f"Entrenamiento finalizado. Mejor mIoU: {best_miou:.4f}")

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        axes[0, 0].plot(train_losses, label="Train Loss", linewidth=2)
        axes[0, 0].plot(val_losses, label="Val Loss", linewidth=2, linestyle="--", color="C1")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Loss (Train vs Val)")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        axes[0, 1].plot(val_mious, label="Val mIoU", color="green", linewidth=2)
        axes[0, 1].axhline(y=best_miou, color="r", linestyle="--", label=f"Best: {best_miou:.4f}")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("mIoU")
        axes[0, 1].set_title("Validation mIoU")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

        axes[1, 0].plot(val_pas, label="Val Pixel Acc", color="orange", linewidth=2)
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("PA")
        axes[1, 0].set_title("Validation Pixel Accuracy")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

        ax_lr = axes[1, 1]
        ax_lr.plot(lr_history, label="Learning Rate", color="purple", linewidth=2)
        ax_lr.set_xlabel("Epoch")
        ax_lr.set_ylabel("LR")
        ax_lr.set_title("LR Schedule")
        ax_lr.grid(True, alpha=0.3)
        ax_lr.legend()

        plt.tight_layout()
        plot_path = os.path.join(base_out_dir, f"training_curves_{config['model']}.png")
        plt.savefig(plot_path, dpi=150)
        logger.info(f"Saved training curves to {plot_path}")

    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location=config["device"])
        model.load_state_dict(checkpoint["model_state_dict"])
        best_miou = checkpoint.get("best_miou", best_miou)
        logger.info(f"Cargado checkpoint para reporte final: {best_checkpoint_path}")
    else:
        logger.warning(f"No se encontró checkpoint en {best_checkpoint_path}; se omite evaluación final.")
        return

    logger.info("Generando métricas por clase...")
    final_miou, final_pa, final_loss, class_ious = validate(
        model,
        val_loader,
        config["device"],
        num_classes=config["n_classes"],
        criterion=criterion,
        return_class_iou=True,
    )
    class_names = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]
    fig_iou, ax_iou = plt.subplots(figsize=(12, 5))
    ax_iou.bar(range(len(class_ious)), class_ious * 100)
    ax_iou.set_xticks(range(len(class_ious)))
    ax_iou.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax_iou.set_ylabel("IoU (%)")
    ax_iou.set_title("IoU por clase (validación)")
    ax_iou.grid(True, axis="y", alpha=0.3)
    per_class_path = os.path.join(base_out_dir, f"per_class_iou_{config['model']}.png")
    fig_iou.tight_layout()
    fig_iou.savefig(per_class_path, dpi=150)
    logger.info(f"Saved per-class IoU barplot to {per_class_path}")

    predictions_path = os.path.join(base_out_dir, f"predictions_{config['model']}.png")
    logger.info("Generando predicciones de muestra...")
    visualize_predictions(
        model, val_loader, config["device"], num_samples=4, save_path=predictions_path
    )

    summary_path = os.path.join(base_out_dir, "summary_results.jsonl")
    summary = {
        "model": config["model"],
        "best_miou": best_miou,
        "eval_miou": final_miou,
        "eval_pa": final_pa,
        "eval_loss": final_loss,
        "num_params": num_params,
        "checkpoint": best_checkpoint_path,
        "per_class_iou_path": per_class_path,
        "predictions_path": predictions_path,
    }
    with open(summary_path, "a") as sf:
        sf.write(json.dumps(summary) + "\n")
    logger.info(f"Resumen guardado en {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Ablaciones de Arquitectura para FCN8s")
    parser.add_argument(
        "--model",
        type=str,
        default="fcn8s",
        choices=["fcn8s", "fcn8s_resnet18", "fcn8s_bilinear"],
        help="Arquitectura a entrenar/evaluar",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../../voc/VOC2012_train_val/VOC2012_train_val",
        help="Path al dataset VOC",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directorio donde guardar checkpoints, curvas, métricas y logs.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="No entrena; carga el checkpoint existente y genera métricas/figuras.",
    )
    parser.add_argument(
        "--run_all",
        action="store_true",
        help="Entrena/evalúa secuencialmente fcn8s, fcn8s_resnet18, fcn8s_bilinear.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.run_all:
        for m in ["fcn8s", "fcn8s_resnet18", "fcn8s_bilinear"]:
            run_experiment(
                model_name=m,
                epochs=args.epochs,
                batch_size=args.batch_size,
                image_size=args.image_size,
                learning_rate=args.lr,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                eval_only=args.eval_only,
            )
    else:
        run_experiment(
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            learning_rate=args.lr,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            eval_only=args.eval_only,
        )
