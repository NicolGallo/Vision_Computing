"""
Mini-SAM Implementation - Complete Solution
Based on Lab Slides Architecture

Author: Prof. David Olivieri
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ================== Mini-SAM Architecture (from slides) ==================

class TinyEncoder(nn.Module):
    """Tiny image encoder - simple CNN for lab"""
    
    def __init__(self, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, stride=2), nn.ReLU(),   # 1/2
            nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.ReLU(),  # 1/4
            nn.Conv2d(128, out_dim, 3, padding=1, stride=2), nn.ReLU()  # 1/8
        )
    
    def forward(self, x):  # x: [B, 3, H, W]
        return self.net(x)  # [B, 256, H/8, W/8]


class PromptEncoder(nn.Module):
    """Prompt encoder: points & boxes to feature maps at 1/8 resolution"""
    
    def __init__(self, feat_h, feat_w, d=256):
        super().__init__()
        self.d = d
        self.pt_embed = nn.Embedding(2, d)  # pos/neg point labels
        self.box_token = nn.Parameter(torch.randn(1, d))
        self.conv = nn.Conv2d(d, d, 3, padding=1)
        
        # Precompute relative position grid
        yy, xx = torch.meshgrid(
            torch.arange(feat_h), 
            torch.arange(feat_w), 
            indexing="ij"
        )
        self.register_buffer("yy", yy.float() / max(1, feat_h - 1))
        self.register_buffer("xx", xx.float() / max(1, feat_w - 1))
    
    def rasterize_points(self, B, points):
        """Rasterize points onto feature map"""
        feat = []
        for b in range(B):
            canvas = torch.zeros(
                self.d, self.yy.shape[0], self.yy.shape[1],
                device=self.yy.device
            )
            if points is not None and len(points[b]) > 0:
                for (px, py, pl) in points[b]:  # normalized coords
                    iy = int(py * (self.yy.shape[0] - 1))
                    ix = int(px * (self.yy.shape[1] - 1))
                    canvas[:, iy, ix] += self.pt_embed.weight[int(pl)].view(-1)
            feat.append(canvas)
        return torch.stack(feat, 0)  # [B, d, H', W']
    
    def rasterize_box(self, B, boxes):
        """Rasterize box onto feature map"""
        feat = []
        for b in range(B):
            canvas = torch.zeros(
                self.d, self.yy.shape[0], self.yy.shape[1],
                device=self.yy.device
            )
            if boxes is not None and len(boxes[b]) == 4:
                x1, y1, x2, y2 = boxes[b]
                mask = (self.xx >= x1) & (self.xx <= x2) & (self.yy >= y1) & (self.yy <= y2)
                canvas[:, mask] = self.box_token.view(-1, 1)
            feat.append(canvas)
        return torch.stack(feat, 0)
    
    def forward(self, points=None, boxes=None):
        B = len(points) if points is not None else len(boxes)
        pmap = self.rasterize_points(B, points) if points is not None else 0
        bmap = self.rasterize_box(B, boxes) if boxes is not None else 0
        fmap = pmap + bmap  # [B, d, H', W']
        return self.conv(fmap)  # [B, d, H', W']


class MiniDecoder(nn.Module):
    """Mini decoder: fuse enc+prompt, produce 3 masks + IoU scores"""
    
    def __init__(self, d=256, n_masks=3):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(d * 2, d, 3, padding=1), nn.ReLU(),
            nn.Conv2d(d, d, 3, padding=1), nn.ReLU()
        )
        self.mask_head = nn.Conv2d(d, n_masks, 1)  # [B, 3, H', W']
        self.iou_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(d, n_masks, 1)
        )  # [B, 3, 1, 1]
    
    def forward(self, fim, fpr):  # both [B, d, H', W']
        z = self.fuse(torch.cat([fim, fpr], dim=1))
        masks = self.mask_head(z)
        ious = self.iou_head(z).squeeze(-1).squeeze(-1)  # [B, 3]
        return masks, ious


class MiniSAM(nn.Module):
    """Full Mini-SAM from slides"""
    
    def __init__(self, in_size=(256, 256), n_masks=3, d=256):
        super().__init__()
        self.enc = TinyEncoder(d)
        H, W = in_size[0] // 8, in_size[1] // 8
        self.penc = PromptEncoder(H, W, d)
        self.dec = MiniDecoder(d, n_masks)
    
    def forward(self, img, points=None, boxes=None, upsample=True):
        fim = self.enc(img)                  # [B, d, H', W']
        fpr = self.penc(points, boxes)       # [B, d, H', W']
        masks, ious = self.dec(fim, fpr)     # [B, 3, H', W'], [B, 3]
        if upsample:
            masks = F.interpolate(
                masks, size=img.shape[-2:], 
                mode="bilinear", align_corners=False
            )
        return masks.sigmoid(), ious


# ================== Helper Functions ==================

def sample_points_from_mask(masks, n_points=5):
    """
    Sample points from ground truth masks
    Args:
        masks: (B, H, W) ground truth binary masks
        n_points: number of points to sample
    Returns:
        points: list of lists [(x, y, label), ...]
    """
    B, H, W = masks.shape
    points_list = []
    
    for b in range(B):
        mask = masks[b]
        points = []
        
        # Sample foreground points
        fg_indices = torch.nonzero(mask > 0)
        if len(fg_indices) > 0:
            n_fg = min(n_points // 2, len(fg_indices))
            fg_samples = fg_indices[torch.randperm(len(fg_indices))[:n_fg]]
            for idx in fg_samples:
                y, x = idx[0].item(), idx[1].item()
                # Normalize to [0, 1]
                points.append((x / (W - 1), y / (H - 1), 1))  # label 1 = foreground
        
        # Sample background points
        bg_indices = torch.nonzero(mask == 0)
        if len(bg_indices) > 0:
            n_bg = min(n_points // 2, len(bg_indices))
            bg_samples = bg_indices[torch.randperm(len(bg_indices))[:n_bg]]
            for idx in bg_samples:
                y, x = idx[0].item(), idx[1].item()
                # Normalize to [0, 1]
                points.append((x / (W - 1), y / (H - 1), 0))  # label 0 = background
        
        points_list.append(points)
    
    return points_list


def iou_per_mask(pred_masks, gt_mask):
    """
    Calculate IoU between each predicted mask and ground truth
    Args:
        pred_masks: (B, 3, H, W) - 3 candidate masks
        gt_mask: (B, H, W) - ground truth
    Returns:
        ious: (B, 3) - IoU for each candidate
    """
    B, n_masks = pred_masks.shape[0], pred_masks.shape[1]
    ious = torch.zeros(B, n_masks, device=pred_masks.device)
    
    for b in range(B):
        for m in range(n_masks):
            pred = (pred_masks[b, m] > 0.5).float()
            gt = (gt_mask[b] > 0.5).float()
            
            intersection = (pred * gt).sum()
            union = ((pred + gt) > 0).float().sum()
            
            if union > 0:
                ious[b, m] = intersection / union
            else:
                ious[b, m] = 0.0
    
    return ious


def select_mask(pred_masks, best_idx):
    """Select best mask from 3 candidates"""
    B = pred_masks.shape[0]
    selected = []
    for b in range(B):
        selected.append(pred_masks[b, best_idx[b]])
    return torch.stack(selected, 0)


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """Focal loss for binary segmentation"""
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce
    return focal.mean()


def dice_loss(pred, target, smooth=1e-6):
    """Dice loss for binary segmentation"""
    pred_flat = pred.view(pred.shape[0], -1)
    target_flat = target.view(target.shape[0], -1)
    
    intersection = (pred_flat * target_flat).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (
        pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth
    )
    
    return 1 - dice.mean()


# ================== Training Loop ==================

def train_epoch_minisam(model, dataloader, optimizer, device):
    """Train Mini-SAM for one epoch"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc='Training'):
        img, gt_mask = batch["image"].to(device), batch["mask"].to(device)  # [B, 1, H, W]
        
        # Sample points from ground truth
        points = sample_points_from_mask(gt_mask.squeeze(1), n_points=5)
        boxes = None  # Start with points-only
        
        # Forward pass
        pred_masks, pred_iou = model(img, points=points, boxes=boxes)
        
        # Choose best of 3 masks by lowest loss or highest IoU to GT
        with torch.no_grad():
            iou_to_gt = iou_per_mask(pred_masks, gt_mask.squeeze(1))  # [B, 3]
            best_idx = iou_to_gt.argmax(dim=1)  # [B]
        
        # Compute losses
        best_masks = select_mask(pred_masks, best_idx)
        gt_mask_float = gt_mask.squeeze(1).float()
        
        focal = focal_loss(best_masks, gt_mask_float)
        dice = dice_loss(best_masks, gt_mask_float)
        
        # IoU prediction loss
        iou_mse = F.mse_loss(
            pred_iou.gather(1, best_idx.view(-1, 1)).squeeze(1),
            iou_to_gt.gather(1, best_idx.view(-1, 1)).squeeze(1)
        )
        
        loss = 0.5 * focal + 0.5 * dice + 1.0 * iou_mse
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate_minisam(model, dataloader, device):
    """Validate Mini-SAM"""
    model.eval()
    total_iou = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            img, gt_mask = batch["image"].to(device), batch["mask"].to(device)
            
            points = sample_points_from_mask(gt_mask.squeeze(1), n_points=5)
            
            pred_masks, pred_iou = model(img, points=points, boxes=None)
            
            # Take best mask by predicted IoU
            best_idx = pred_iou.argmax(dim=1)
            best_masks = select_mask(pred_masks, best_idx)
            
            # Calculate true IoU
            iou_to_gt = iou_per_mask(pred_masks, gt_mask.squeeze(1))
            total_iou += iou_to_gt.gather(1, best_idx.view(-1, 1)).mean().item()
    
    return total_iou / len(dataloader)


# ================== Main Training Script ==================

def main():
    """Main training pipeline for Mini-SAM"""
    
    config = {
        'in_size': (256, 256),
        'n_masks': 3,
        'embed_dim': 256,
        'batch_size': 8,
        'learning_rate': 3e-4,
        'weight_decay': 1e-2,
        'epochs': 30,
        'device': device
    }
    
    print("Mini-SAM Training")
    print("=" * 60)
    
    # Create model
    model = MiniSAM(
        in_size=config['in_size'],
        n_masks=config['n_masks'],
        d=config['embed_dim']
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # NOTE: You need to implement your own dataset loader
    # train_loader = DataLoader(your_dataset, batch_size=config['batch_size'])
    # val_loader = DataLoader(your_val_dataset, batch_size=config['batch_size'])
    
    print("\nTraining loop ready!")
    print("Note: Implement your dataset loader to start training.")
    print("=" * 60)
    
    # Training loop example
    # for epoch in range(config['epochs']):
    #     train_loss = train_epoch_minisam(model, train_loader, optimizer, device)
    #     val_iou = validate_minisam(model, val_loader, device)
    #     print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Val IoU={val_iou:.4f}")


if __name__ == "__main__":
    main()