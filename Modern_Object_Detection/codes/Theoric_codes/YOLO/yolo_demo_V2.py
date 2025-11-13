# Save as: yolo_complete.py
"""
YOLO (You Only Look Once) Complete Implementation — Patched
Patches:
  (1) Responsible-box selection + class masking (YOLOv1-faithful)
  (2) Confidence target for positives = IoU (not 1.0)
  (3) w,h positivity via sigmoid
  (4) Avoid multiple objects per grid cell in synthetic dataset
  (5) Simple IoU-NMS in decode for cleaner visuals
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict


# ============================================================================
# YOLO MODEL
# ============================================================================

class SimpleBackbone(nn.Module):
    """Simple CNN backbone for YOLO"""
    def __init__(self, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            # Conv1: 224x224 -> 112x112
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv2: 112x112 -> 56x56
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv3: 56x56 -> 28x28
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv4: 28x28 -> 14x14
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.out_channels = 512

    def forward(self, x):
        return self.features(x)


class YOLODetector(nn.Module):
    """
    YOLO object detector (YOLOv1-style head on a tiny backbone)
    """
    def __init__(self, num_classes=3, grid_size=7, num_boxes=2):
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.num_boxes = num_boxes

        # Backbone
        self.backbone = SimpleBackbone(in_channels=3)
        feature_dim = self.backbone.out_channels

        # Detection head: outputs S × S × (B × (5 + C))
        # NOTE: We keep per-box class scores for simplicity,
        # but we will compute class loss only for the responsible box.

        #Per cell and per box, the model predicts
        # 4 geom (xcell, ycell, w, h)
        # 1 objectness/confidence 
        # C class scores 
        #  per cell we need B x (5+C)

        
        output_channels = num_boxes * (5 + num_classes)

        self.detection_head = nn.Sequential(
            nn.Conv2d(feature_dim, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(1024, output_channels, kernel_size=1)
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        self._init_weights()

    def _init_weights(self):
        for m in self.detection_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Returns:
            B (batch size);  S (grid size), C (num classes); nb (boxes per cell)
            predictions: [B, S, S, B, 5+C] with activations applied:
              - (x, y) in (0,1) within cell
              - (w, h) in (0,1) relative to image (positivity guaranteed)
              - conf in (0,1)
              - class probs softmax per box
        """
        # -------------------------------
        # Aliases for readability
        # -------------------------------
        B = x.size(0)                              # batch size
        S, C, nb = self.grid_size, self.num_classes, self.num_boxes

        # -------------------------------
        # 1) Backbone feature extraction
        # -------------------------------
        # x: [B, 3, H, W]
        feats = self.backbone(x)                   # -> [B, 512, H/stride, W/stride]
        # stride ~ 32 in this tiny backbone

        # -----------------------------------------
        # 2) Fix spatial size to the S×S grid (pool)
        # -----------------------------------------
        # AdaptiveAvgPool2d partitions each channel into S×S bins and averages each bin:
        # Y_{c,i,j} = (1/|R_ij|) * sum_{(h,w) in R_ij} X_{c,h,w}
        feats = self.adaptive_pool(feats)          # -> [B, 512, S, S]

        # -------------------------------
        # 3) Detection head (per-cell vec)
        # -------------------------------
        # 3×3 conv (mix local context) -> BN -> LeakyReLU -> Dropout -> 1×1 projection to B*(5+C)
        raw = self.detection_head(feats)           # -> [B, B*(5+C), S, S]

        # ---------------------------------------
        # 4) Reorder and reshape to per-box layout
        # ---------------------------------------
        # Put spatial first for easier indexing, then split the channel dim into [B, 5+C]
        raw = raw.permute(0, 2, 3, 1).contiguous() # -> [B, S, S, B*(5+C)]
        raw = raw.view(B, S, S, nb, 5 + C)         # -> [B, S, S, B, 5+C]

        # ---------------------------------------------------------
        # 5) Apply activations to map raw numbers to valid semantics
        # ---------------------------------------------------------
        # (We keep a clone so 'raw' stays available if you want logits later.)
        preds = raw.clone()

        # (x_cell, y_cell): offsets within the cell in (0,1)
        # Absolute conversion at decode time:
        #   x_abs = (j + x_cell) / S,  y_abs = (i + y_cell) / S
        preds[..., 0:2] = torch.sigmoid(raw[..., 0:2])  # x,y in (0,1) inside cell

        # (w, h): relative to whole image in (0,1), ensures positivity
        # (Anchor-free demo. Anchor-based YOLOs use exp(t_w,t_h)*anchor.)
        preds[..., 2:4] = torch.sigmoid(raw[..., 2:4])  # w,h in (0,1)

        # Confidence (objectness): probability in (0,1)
        # Loss target: IoU for the responsible box; 0 for others.
        preds[..., 4:5] = torch.sigmoid(raw[..., 4:5])  # conf in (0,1)

        # Class probabilities: softmax per box over C classes
        # If you prefer CrossEntropyLoss, remove this softmax and feed logits.
        preds[..., 5:]  = F.softmax(raw[..., 5:], dim=-1)  # class probs sum to 1

        # --------------------------------
        # 6) Return structured predictions
        # --------------------------------
        # Shape is convenient for loss/decoding:
        #   predictions[b, i, j, a, :] = [x,y,w,h, conf, p_1..p_C]
        return preds


# ============================================================================
# YOLO LOSS (YOLOv1-style with responsible-box selection)
# ============================================================================

class YOLOLoss(nn.Module):
    """
    L = λ_coord * L_coord +
        L_conf(responsible; target=IoU) +
        λ_noobj * L_conf(non-responsible and empty cells; target=0) +
        L_class (responsible box only)
    """
    def __init__(self, grid_size=7, num_boxes=2, num_classes=3,
                 lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.S = grid_size
        self.B = num_boxes
        self.C = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    @staticmethod
    def _to_corners(box):
        # box: [..., 4] where [x,y,w,h], x,y in (0,1) absolute, w,h in (0,1)
        cx, cy, w, h = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return x1, y1, x2, y2

    def compute_iou(self, box1, box2):
        """
        IoU between boxes given as [x, y, w, h] in absolute image coords (0..1).
        box1/box2 shapes should be broadcastable to a common shape.
        """
        b1_x1, b1_y1, b1_x2, b1_y2 = self._to_corners(box1)
        b2_x1, b2_y1, b2_x2, b2_y2 = self._to_corners(box2)

        inter_x1 = torch.maximum(b1_x1, b2_x1)
        inter_y1 = torch.maximum(b1_y1, b2_y1)
        inter_x2 = torch.minimum(b1_x2, b2_x2)
        inter_y2 = torch.minimum(b1_y2, b2_y2)

        inter_w = torch.clamp(inter_x2 - inter_x1, min=0.0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0.0)
        inter_area = inter_w * inter_h

        area1 = torch.clamp(b1_x2 - b1_x1, min=0.0) * torch.clamp(b1_y2 - b1_y1, min=0.0)
        area2 = torch.clamp(b2_x2 - b2_x1, min=0.0) * torch.clamp(b2_y2 - b2_y1, min=0.0)
        union = area1 + area2 - inter_area + 1e-6
        return inter_area / union

    def forward(self, predictions, targets):
        """
        predictions: [B, S, S, B, 5+C] (post-activation as returned by model)
        targets:     [B, S, S, B, 5+C]
          - Convention (dataset):
              * If a cell has an object, GT box/class stored in box 0,
                target[...,0,4]=1 ; others 0.
              * If empty cell, all zeros.
        """
        B, S, nb, C = predictions.size(0), self.S, self.B, self.C

        # Extract predictions
        pred_xy   = predictions[..., 0:2]      # [B,S,S,B,2] (cell-relative x,y)
        pred_wh   = predictions[..., 2:4]      # [B,S,S,B,2] (0..1 relative to image)
        pred_conf = predictions[..., 4:5]      # [B,S,S,B,1]
        pred_cls  = predictions[..., 5:]       # [B,S,S,B,C]

        # Get target per cell from box 0 (dataset encodes it there)
        tgt_xy_cell = targets[..., 0, 0:2]     # [B,S,S,2] (cell-relative)
        tgt_wh      = targets[..., 0, 2:4]     # [B,S,S,2]
        tgt_conf_any= targets[..., :, :, :, 4].sum(dim=-1, keepdim=True)  # [B,S,S,1], >0 if object in cell
        tgt_cls_onehot = targets[..., 0, 5:]   # [B,S,S,C]

        # Build absolute coords for IoU:
        # Convert pred cell-relative (x_cell,y_cell) into absolute (0..1) center coords
        grid_y, grid_x = torch.meshgrid(
            torch.arange(S, device=predictions.device),
            torch.arange(S, device=predictions.device), indexing='ij'
        )  # [S,S]
        grid_x = grid_x[None, :, :, None, None].float()  # [1,S,S,1,1]
        grid_y = grid_y[None, :, :, None, None].float()

        pred_cx_abs = (grid_x + pred_xy[..., 0:1]) / S
        pred_cy_abs = (grid_y + pred_xy[..., 1:2]) / S
        pred_abs = torch.cat([pred_cx_abs, pred_cy_abs, pred_wh], dim=-1)  # [B,S,S,B,4]

        # GT absolute center from cell-relative target in box 0
        tgt_cx_abs = (grid_x.squeeze(-1) + tgt_xy_cell[..., 0:1]) / S   # [B,S,S,1]
        tgt_cy_abs = (grid_y.squeeze(-1) + tgt_xy_cell[..., 1:2]) / S
        gt_abs = torch.cat([tgt_cx_abs, tgt_cy_abs, tgt_wh], dim=-1)     # [B,S,S,4]

        # IoU between each predicted box and the GT box for cells that have objects
        gt_abs_expanded = gt_abs[..., None, :]  # [B,S,S,1,4]
        ious = self.compute_iou(pred_abs, gt_abs_expanded)  # [B,S,S,B]

        # Determine responsible box per positive cell: argmax IoU
        # If cell has no object, argmax is arbitrary; we'll mask it out.
        resp_idx = torch.argmax(ious, dim=-1, keepdim=True)  # [B,S,S,1]
        resp_mask = torch.zeros_like(pred_conf)              # [B,S,S,B,1]
        resp_mask.scatter_(-2, resp_idx.unsqueeze(-1), 1.0)  # one-hot along B

        # Positive (has object) cell mask
        cell_obj_mask = (tgt_conf_any > 0).float()           # [B,S,S,1]
        cell_obj_mask_broadcast = cell_obj_mask.unsqueeze(-2)  # [B,S,S,1,1]

        # Responsible positive mask
        pos_mask = resp_mask * cell_obj_mask_broadcast       # [B,S,S,B,1]

        # Negative mask for confidence (everything that's NOT the responsible positive)
        neg_mask = 1.0 - pos_mask

        # --- Localization loss (responsible boxes only) ---
        # Compare predicted (x_cell,y_cell,w,h) to target
        # Note: targets store GT in box 0; broadcast to all B for easy masking
        tgt_xy_cell_b = tgt_xy_cell.unsqueeze(-2).expand(-1, -1, -1, nb, -1)  # [B,S,S,B,2]
        tgt_wh_b      = tgt_wh.unsqueeze(-2).expand(-1, -1, -1, nb, -1)       # [B,S,S,B,2]

        loc_xy = (pred_xy - tgt_xy_cell_b) ** 2
        # sqrt parameterization for sizes
        loc_wh = (torch.sqrt(torch.clamp(pred_wh, min=1e-6)) -
                  torch.sqrt(torch.clamp(tgt_wh_b,  min=1e-6))) ** 2

        loss_xy = self.lambda_coord * torch.sum(pos_mask * loc_xy[..., 0:1] + pos_mask * loc_xy[..., 1:2])
        loss_wh = self.lambda_coord * torch.sum(pos_mask * loc_wh[..., 0:1] + pos_mask * loc_wh[..., 1:2])

        # --- Confidence loss ---
        # For responsible positives: target = IoU(pred_resp, GT)
        # Gather IoU of the responsible box at each (B,S,S)
        iou_resp = torch.gather(ious, dim=-1, index=resp_idx).unsqueeze(-1)  # [B,S,S,1,1]
        loss_conf_pos = torch.sum(pos_mask * (pred_conf - iou_resp) ** 2)

        # For all others (including other boxes in positive cells and all boxes in empty cells): target = 0
        loss_conf_neg = self.lambda_noobj * torch.sum(neg_mask * (pred_conf - 0.0) ** 2)

        # --- Class loss (responsible box only) ---
        # Use class predictions from the responsible box only
        # Gather per-box class probabilities at resp_idx
        idx_cls = resp_idx.expand(-1, -1, -1, 1, C)  # [B,S,S,1,C]
        pred_cls_resp = torch.gather(pred_cls, dim=-2, index=idx_cls).squeeze(-2)  # [B,S,S,C]
        # YOLOv1 used MSE to one-hot. For CE, switch to logits and nn.CrossEntropyLoss.
        loss_class = torch.sum(cell_obj_mask * (pred_cls_resp - tgt_cls_onehot) ** 2)

        total = (loss_xy + loss_wh + loss_conf_pos + loss_conf_neg + loss_class) / B

        return total, {
            'loss_xy':        (loss_xy / B).item(),
            'loss_wh':        (loss_wh / B).item(),
            'loss_conf_obj':  (loss_conf_pos / B).item(),
            'loss_conf_noobj':(loss_conf_neg / B).item(),
            'loss_class':     (loss_class / B).item()
        }


# ============================================================================
# SYNTHETIC DATASET (avoid multiple objects per cell)
# ============================================================================

class SyntheticObjectDataset(Dataset):
    """
    Generates synthetic images with geometric shapes.
    Ensures at most ONE object per grid cell (YOLOv1-compatible).
    """
    def __init__(self, num_samples=1000, image_size=224, grid_size=7,
                 num_boxes=2, num_classes=3, max_objects=3, max_retries=50):
        self.num_samples = num_samples
        self.image_size = image_size
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.max_objects = max_objects
        self.max_retries = max_retries

        self.class_names = ['circle', 'square', 'triangle']
        self.class_colors = [
            np.array([0.2, 0.4, 0.8]),
            np.array([0.8, 0.2, 0.2]),
            np.array([0.2, 0.8, 0.3]),
        ]

    def __len__(self):
        return self.num_samples

    def draw_shape(self, image, x, y, size, class_id):
        color = self.class_colors[class_id]
        if class_id == 0:  # Circle
            y_grid, x_grid = np.ogrid[:self.image_size, :self.image_size]
            mask = (x_grid - x) ** 2 + (y_grid - y) ** 2 <= (size / 2) ** 2
            image[mask] = color
        elif class_id == 1:  # Square
            x1 = max(0, int(x - size / 2)); y1 = max(0, int(y - size / 2))
            x2 = min(self.image_size, int(x + size / 2))
            y2 = min(self.image_size, int(y + size / 2))
            image[y1:y2, x1:x2] = color
        elif class_id == 2:  # Triangle
            pts = np.array([
                [x, y - size / 2],
                [x - size / 2, y + size / 2],
                [x + size / 2, y + size / 2]
            ], dtype=np.int32)
            from matplotlib.path import Path
            path = Path(pts)
            yy, xx = np.meshgrid(np.arange(self.image_size),
                                 np.arange(self.image_size), indexing='ij')
            points = np.stack([xx.ravel(), yy.ravel()], axis=1)
            mask = path.contains_points(points).reshape(self.image_size, self.image_size)
            image[mask] = color

    def __getitem__(self, idx):
        image = np.ones((self.image_size, self.image_size, 3), dtype=np.float32) * 0.95

        num_objects = np.random.randint(1, self.max_objects + 1)
        objects = []
        occupied = set()  # (gx, gy) taken

        for _ in range(num_objects):
            placed = False
            for _ in range(self.max_retries):
                class_id = np.random.randint(0, self.num_classes)
                size = np.random.randint(30, 80)
                x = np.random.randint(size, self.image_size - size)
                y = np.random.randint(size, self.image_size - size)

                gx = int((x / self.image_size) * self.grid_size)
                gy = int((y / self.image_size) * self.grid_size)
                gx = min(gx, self.grid_size - 1)
                gy = min(gy, self.grid_size - 1)

                if (gx, gy) in occupied:
                    continue

                # place
                self.draw_shape(image, x, y, size, class_id)
                objects.append({
                    'x': x / self.image_size,
                    'y': y / self.image_size,
                    'w': size / self.image_size,
                    'h': size / self.image_size,
                    'class': class_id
                })
                occupied.add((gx, gy))
                placed = True
                break
            if not placed:
                # skip if we can't find a free cell
                pass

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        target = self.encode_target(objects)
        return image_tensor, target, objects

    def encode_target(self, objects):
        """
        Encode objects into [S,S,B,5+C].
        - If a cell has an object, we store GT in box 0 (standard YOLOv1 didactics).
        - Only box 0 has conf=1 to signal presence; others 0.
        """
        S, B, C = self.grid_size, self.num_boxes, self.num_classes
        target = torch.zeros(S, S, B, 5 + C)

        for obj in objects:
            gx = int(obj['x'] * S); gy = int(obj['y'] * S)
            gx = min(gx, S - 1); gy = min(gy, S - 1)

            x_cell = obj['x'] * S - gx
            y_cell = obj['y'] * S - gy

            # Store in box 0
            target[gy, gx, 0, 0] = x_cell
            target[gy, gx, 0, 1] = y_cell
            target[gy, gx, 0, 2] = obj['w']
            target[gy, gx, 0, 3] = obj['h']
            target[gy, gx, 0, 4] = 1.0
            target[gy, gx, 0, 5 + obj['class']] = 1.0

        return target


# ============================================================================
# TRAINING
# ============================================================================

def train_yolo(num_epochs=20, batch_size=16, learning_rate=1e-3):
    """Train YOLO on synthetic dataset"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    dataset = SyntheticObjectDataset(num_samples=1000, grid_size=7, num_boxes=2)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = YOLODetector(num_classes=3, grid_size=7, num_boxes=2).to(device)
    criterion = YOLOLoss(grid_size=7, num_boxes=2, num_classes=3)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("=" * 70)
    print("YOLO TRAINING")
    print("=" * 70)

    history = {'total_loss': [], 'xy_loss': [], 'wh_loss': [],
               'conf_obj_loss': [], 'conf_noobj_loss': [], 'class_loss': []}

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        epoch_components = {'xy': [], 'wh': [], 'conf_obj': [], 'conf_noobj': [], 'class': []}

        for images, targets, _ in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            preds = model(images)
            loss, ld = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_components['xy'].append(ld['loss_xy'])
            epoch_components['wh'].append(ld['loss_wh'])
            epoch_components['conf_obj'].append(ld['loss_conf_obj'])
            epoch_components['conf_noobj'].append(ld['loss_conf_noobj'])
            epoch_components['class'].append(ld['loss_class'])

        avg = float(np.mean(epoch_losses))
        history['total_loss'].append(avg)
        for k in epoch_components:
            history[f'{k}_loss'].append(float(np.mean(epoch_components[k])))

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg:.4f} "
              f"| xy: {history['xy_loss'][-1]:.3f} "
              f"| wh: {history['wh_loss'][-1]:.3f} "
              f"| obj: {history['conf_obj_loss'][-1]:.3f} "
              f"| noobj: {history['conf_noobj_loss'][-1]:.3f} "
              f"| class: {history['class_loss'][-1]:.3f}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    return model, history, dataset


# ============================================================================
# VISUALIZATION
# ============================================================================

def nms_iou(boxes, scores, iou_thresh=0.45):
    """
    boxes: [N, 4] in (x,y,w,h) absolute 0..1
    scores: [N]
    """
    if len(boxes) == 0:
        return []
    boxes = torch.tensor(boxes)
    scores = torch.tensor(scores)

    # Convert to corners
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w/2; y1 = cy - h/2; x2 = cx + w/2; y2 = cy + h/2
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]

        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])

        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-6)
        order = rest[iou <= iou_thresh]
    return keep


def visualize_predictions(model, dataset, num_samples=4):
    """Visualize YOLO predictions with simple NMS"""
    device = next(model.parameters()).device
    model.eval()

    rows = 2
    cols = max(1, num_samples // 2)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 8))
    axes = np.array(axes).reshape(-1)
    with torch.no_grad():
        for i in range(num_samples):
            image, target, gt_objects = dataset[i]
            pred = model(image.unsqueeze(0).to(device)).squeeze(0).cpu()

            detections = decode_predictions(pred, conf_threshold=0.3, nms=True)

            ax = axes[i]
            img_np = image.permute(1, 2, 0).numpy()
            ax.imshow(img_np)

            # Ground truth (dashed)
            for obj in gt_objects:
                x, y, w, h = obj['x'], obj['y'], obj['w'], obj['h']
                x1 = (x - w / 2) * dataset.image_size
                y1 = (y - h / 2) * dataset.image_size
                w_px = w * dataset.image_size
                h_px = h * dataset.image_size
                color = dataset.class_colors[obj['class']]
                rect = patches.Rectangle((x1, y1), w_px, h_px, linewidth=2,
                                         edgecolor=color, facecolor='none',
                                         linestyle='--')
                ax.add_patch(rect)
                ax.text(x1, y1 - 5, f"GT: {dataset.class_names[obj['class']]}",
                        color=color, fontsize=8, fontweight='bold')

            # Predictions (solid)
            for det in detections:
                x, y, w, h = det['x'], det['y'], det['w'], det['h']
                x1 = (x - w / 2) * dataset.image_size
                y1 = (y - h / 2) * dataset.image_size
                w_px = w * dataset.image_size
                h_px = h * dataset.image_size

                color = dataset.class_colors[det['class']]
                rect = patches.Rectangle((x1, y1), w_px, h_px, linewidth=2,
                                         edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1 + h_px + 15,
                        f"{dataset.class_names[det['class']]} {det['conf']:.2f}",
                        color=color, fontsize=8, fontweight='bold')

            ax.set_title(f"Sample {i+1}")
            ax.axis('off')

    plt.tight_layout()
    plt.savefig('yolo_predictions.png', dpi=150, bbox_inches='tight')
    print("\nPredictions saved to 'yolo_predictions.png'")
    plt.show()


def decode_predictions(predictions, conf_threshold=0.3, nms=True, iou_thresh=0.45):
    """
    Decode YOLO predictions to boxes.
    predictions: [S,S,B,5+C]
    Returns list of dicts with keys: x,y,w,h,conf,class
    """
    S = predictions.shape[0]
    B = predictions.shape[2]

    dets = []
    for i in range(S):
        for j in range(S):
            for b in range(B):
                pred = predictions[i, j, b]
                x_cell, y_cell = pred[0].item(), pred[1].item()
                w, h = max(1e-6, pred[2].item()), max(1e-6, pred[3].item())
                conf = pred[4].item()
                class_probs = pred[5:]
                class_id = torch.argmax(class_probs).item()
                class_conf = class_probs[class_id].item()
                score = conf * class_conf
                if score < conf_threshold:
                    continue
                x = (j + x_cell) / S
                y = (i + y_cell) / S
                dets.append({'x': x, 'y': y, 'w': w, 'h': h,
                             'conf': score, 'class': class_id})

    if not nms or len(dets) == 0:
        return dets

    # NMS per class
    final = []
    dets_by_class = {}
    for d in dets:
        dets_by_class.setdefault(d['class'], []).append(d)

    for cls, cds in dets_by_class.items():
        boxes = [[d['x'], d['y'], d['w'], d['h']] for d in cds]
        scores = [d['conf'] for d in cds]
        keep = nms_iou(boxes, scores, iou_thresh=iou_thresh)
        final.extend([cds[k] for k in keep])
    return final


def plot_training_curves(history):
    """Plot training loss curves"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history['total_loss'], linewidth=2)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Total YOLO Loss'); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['xy_loss'], label='Localization (xy)', linewidth=2)
    axes[1].plot(history['wh_loss'], label='Localization (wh)', linewidth=2)
    axes[1].plot(history['conf_obj_loss'], label='Confidence (obj)', linewidth=2)
    axes[1].plot(history['conf_noobj_loss'], label='Confidence (noobj)', linewidth=2)
    axes[1].plot(history['class_loss'], label='Classification', linewidth=2)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].set_title('YOLO Loss Components')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('yolo_training_curves.png', dpi=150, bbox_inches='tight')
    print("Training curves saved to 'yolo_training_curves.png'")
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("YOLO: YOU ONLY LOOK ONCE")
    print("Complete PyTorch Implementation with Synthetic Data (Patched)")
    print("=" * 70 + "\n")

    model, history, dataset = train_yolo(num_epochs=20, batch_size=16, learning_rate=1e-3)
    plot_training_curves(history)
    visualize_predictions(model, dataset, num_samples=4)

    print("\n" + "=" * 70)
    print("YOLO Training and Visualization Complete!")
    print("=" * 70)
    print("\nKey Components Demonstrated:")
    print("  ✓ Grid-based detection (S×S cells)")
    print("  ✓ Multi-task loss (localization + confidence + classification)")
    print("  ✓ Responsibility assignment (YOLOv1: one predictor per object)")
    print("  ✓ Confidence target = IoU (better calibration)")
    print("  ✓ End-to-end training on synthetic shapes")
    print("  ✓ NMS decoding for cleaner visualization")
    print("=" * 70 + "\n")
