# Save as: ssd_tiny_pedagogical.py
"""
SSD (Single Shot MultiBox Detector) - Educational Implementation
================================================================
A simplified SSD implementation for teaching object detection concepts.

Key Concepts Demonstrated:
- Multi-scale feature maps (P3 @ stride 16, P4 @ stride 32)
- Anchor boxes (default boxes) at multiple aspect ratios and scales
- IoU-based matching: positive (≥0.5), negative (<0.4), ignored (between)
- Multi-task loss: Smooth L1 (localization) + CrossEntropy (classification)
- Hard Negative Mining (3:1 ratio) for class imbalance
- Non-Maximum Suppression (NMS) for post-processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple


# ============================================================================
# SYNTHETIC DATASET (Geometric Shapes)
# ============================================================================

class ShapesDataset(Dataset):
    """
    Generates synthetic images with geometric shapes (circles, squares, triangles).
    Each image contains 1-4 randomly placed and sized objects.
    """
    def __init__(self, num_samples=1000, image_size=224, num_classes=3, max_objects=4):
        self.num_samples = num_samples
        self.H = self.W = image_size
        self.num_classes = num_classes
        self.max_objects = max_objects
        
        self.class_names = ['circle', 'square', 'triangle']
        self.class_colors = [
            np.array([0.2, 0.4, 0.8]),  # Blue for circles
            np.array([0.8, 0.2, 0.2]),  # Red for squares
            np.array([0.2, 0.8, 0.3]),  # Green for triangles
        ]
    
    def __len__(self):
        return self.num_samples
    
    def _draw_shape(self, img, x, y, size, class_id):
        """Draw a shape on the image"""
        color = self.class_colors[class_id]
        
        if class_id == 0:  # Circle
            yy, xx = np.ogrid[:self.H, :self.W]
            mask = (xx - x)**2 + (yy - y)**2 <= (size/2)**2
            img[mask] = color
            
        elif class_id == 1:  # Square
            x1, y1 = max(0, int(x - size/2)), max(0, int(y - size/2))
            x2, y2 = min(self.W, int(x + size/2)), min(self.H, int(y + size/2))
            img[y1:y2, x1:x2] = color
            
        else:  # Triangle
            pts = np.array([
                [x, y - size/2],
                [x - size/2, y + size/2],
                [x + size/2, y + size/2]
            ], dtype=np.int32)
            from matplotlib.path import Path
            path = Path(pts)
            Y, X = np.meshgrid(np.arange(self.H), np.arange(self.W), indexing='ij')
            pts2 = np.stack([X.ravel(), Y.ravel()], 1)
            mask = path.contains_points(pts2).reshape(self.H, self.W)
            img[mask] = color
    
    def __getitem__(self, idx):
        # Create white background
        img = np.ones((self.H, self.W, 3), dtype=np.float32) * 0.95
        
        # Random number of objects
        num_objects = np.random.randint(1, self.max_objects + 1)
        boxes, labels = [], []
        
        for _ in range(num_objects):
            class_id = np.random.randint(0, self.num_classes)
            size = np.random.randint(28, 80)
            x = np.random.randint(size, self.W - size)
            y = np.random.randint(size, self.H - size)
            
            self._draw_shape(img, x, y, size, class_id)
            
            # Store in normalized (cx, cy, w, h) format
            cx_norm = x / self.W
            cy_norm = y / self.H
            w_norm = h_norm = size / self.W
            
            boxes.append([cx_norm, cy_norm, w_norm, h_norm])
            labels.append(class_id)
        
        # Convert to tensors
        image_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return image_tensor, boxes_tensor, labels_tensor


# ============================================================================
# BACKBONE NETWORK (Feature Extraction)
# ============================================================================

class TinyBackbone(nn.Module):
    """
    Lightweight backbone that produces two feature levels:
    - P3: stride 16 (14x14 for 224x224 input)
    - P4: stride 32 (7x7 for 224x224 input)
    """
    def __init__(self):
        super().__init__()
        
        # Stem: progressively downsample to stride 8
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),      # stride 2
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.MaxPool2d(2, 2),              # stride 4
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.MaxPool2d(2, 2),              # stride 8
            
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True)
        )
        
        # P3 level: stride 16
        self.p3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True)
        )
        
        # P4 level: stride 32
        self.p4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True)
        )
    
    def forward(self, x):
        x = self.stem(x)     # stride 8
        p3 = self.p3(x)      # stride 16 (H/16, W/16)
        p4 = self.p4(p3)     # stride 32 (H/32, W/32)
        return p3, p4


# ============================================================================
# ANCHOR GENERATION
# ============================================================================

def make_anchors(HW_list, strides, ratios=(0.5, 1.0, 2.0), scales=(1.0, 1.6)):
    """
    Generate anchor boxes for each feature level.
    
    Args:
        HW_list: List of (H, W) tuples for each feature map
        strides: List of stride values for each level
        ratios: Aspect ratios for anchors
        scales: Scale factors for anchors
    
    Returns:
        List of anchor tensors [N, 4] in normalized (cx, cy, w, h) format
    """
    all_anchors = []
    
    for (H, W), stride in zip(HW_list, strides):
        anchors = []
        
        # Generate anchor at each grid cell
        for i in range(H):
            for j in range(W):
                # Center of the cell in pixel coordinates
                cx = (j + 0.5) * stride
                cy = (i + 0.5) * stride
                
                # Generate anchors with different ratios and scales
                for ratio in ratios:
                    for scale in scales:
                        w = scale * stride * np.sqrt(1.0 / ratio)
                        h = scale * stride * np.sqrt(ratio)
                        anchors.append([cx, cy, w, h])
        
        # Convert to array and normalize
        anchors = np.array(anchors, dtype=np.float32)
        anchors[:, 0] /= (W * stride)  # normalize cx
        anchors[:, 1] /= (H * stride)  # normalize cy
        anchors[:, 2] /= (W * stride)  # normalize w
        anchors[:, 3] /= (H * stride)  # normalize h
        
        all_anchors.append(torch.from_numpy(anchors))
    
    return all_anchors


def xywh_to_xyxy(xywh):
    """Convert from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2)"""
    cx, cy, w, h = xywh.unbind(-1)
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2
    return torch.stack([x1, y1, x2, y2], -1)


def compute_iou(boxes_a, boxes_b):
    """
    Compute IoU between two sets of boxes.
    
    Args:
        boxes_a: [N, 4] in (x1, y1, x2, y2) format
        boxes_b: [M, 4] in (x1, y1, x2, y2) format
    
    Returns:
        iou: [N, M] matrix of IoU values
    """
    N, M = boxes_a.size(0), boxes_b.size(0)
    
    # Expand for broadcasting
    a = boxes_a[:, None, :].expand(N, M, 4)
    b = boxes_b[None, :, :].expand(N, M, 4)
    
    # Compute intersection
    x1 = torch.max(a[..., 0], b[..., 0])
    y1 = torch.max(a[..., 1], b[..., 1])
    x2 = torch.min(a[..., 2], b[..., 2])
    y2 = torch.min(a[..., 3], b[..., 3])
    
    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter_area = inter_w * inter_h
    
    # Compute areas
    area_a = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)
    
    # Compute IoU
    union = area_a + area_b - inter_area + 1e-6
    return inter_area / union


# ============================================================================
# SSD MODEL
# ============================================================================

class SSDTiny(nn.Module):
    """
    Simplified SSD detector with two feature levels.
    Each level predicts class scores and box offsets for multiple anchors.
    """
    def __init__(self, num_classes=3, ratios=(0.5, 1.0, 2.0), scales=(1.0, 1.6)):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = TinyBackbone()
        self.ratios = ratios
        self.scales = scales
        self.num_anchors = len(ratios) * len(scales)  # anchors per location
        
        # Detection heads for each level
        # Output: num_anchors × (num_classes + 1) for classification (including background)
        #         num_anchors × 4 for box regression
        self.cls_head_p3 = nn.Conv2d(256, self.num_anchors * (num_classes + 1), 3, 1, 1)
        self.box_head_p3 = nn.Conv2d(256, self.num_anchors * 4, 3, 1, 1)
        
        self.cls_head_p4 = nn.Conv2d(256, self.num_anchors * (num_classes + 1), 3, 1, 1)
        self.box_head_p4 = nn.Conv2d(256, self.num_anchors * 4, 3, 1, 1)
    
    def forward(self, x):
        B = x.size(0)
        
        # Extract features
        p3, p4 = self.backbone(x)  # [B, 256, H3, W3], [B, 256, H4, W4]
        
        def process_level(features, cls_head, box_head):
            """Process a single feature level"""
            H, W = features.shape[2], features.shape[3]
            
            # Classification: [B, A*(C+1), H, W] -> [B, H*W*A, C+1]
            cls_logits = cls_head(features)
            cls_logits = cls_logits.permute(0, 2, 3, 1).contiguous()
            cls_logits = cls_logits.view(B, H * W * self.num_anchors, self.num_classes + 1)
            
            # Box regression: [B, A*4, H, W] -> [B, H*W*A, 4]
            box_preds = box_head(features)
            box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
            box_preds = box_preds.view(B, H * W * self.num_anchors, 4)
            
            return cls_logits, box_preds, H, W
        
        # Process both levels
        cls3, box3, H3, W3 = process_level(p3, self.cls_head_p3, self.box_head_p3)
        cls4, box4, H4, W4 = process_level(p4, self.cls_head_p4, self.box_head_p4)
        
        # Concatenate predictions from all levels
        cls_logits = torch.cat([cls3, cls4], dim=1)  # [B, N_total, C+1]
        box_preds = torch.cat([box3, box4], dim=1)   # [B, N_total, 4]
        
        # Generate anchors
        strides = [16, 32]
        anchors = make_anchors([(H3, W3), (H4, W4)], strides, self.ratios, self.scales)
        anchors = torch.cat(anchors, dim=0).to(x.device)
        anchors = anchors.unsqueeze(0).expand(B, -1, -1)  # [B, N_total, 4]
        
        return cls_logits, box_preds, anchors


# ============================================================================
# ANCHOR MATCHING & LOSS
# ============================================================================

def match_anchors(anchors_xywh, gt_xywh, gt_labels, pos_iou=0.5, neg_iou=0.4):
    """
    Match anchors to ground truth boxes based on IoU.
    
    Returns:
        matches: [N] tensor with:
            - index of matched GT (≥0) for positive anchors
            - -2 for negative anchors (IoU < neg_iou)
            - -1 for ignored anchors (neg_iou ≤ IoU < pos_iou)
        matched_boxes: [N, 4] matched GT boxes (for positive anchors)
    """
    if gt_xywh.numel() == 0:
        # No ground truth objects
        matches = torch.full((anchors_xywh.size(0),), -1, dtype=torch.long)
        return matches, torch.zeros_like(anchors_xywh)
    
    # Convert to corner format for IoU computation
    anchors_xyxy = xywh_to_xyxy(anchors_xywh)
    gt_xyxy = xywh_to_xyxy(gt_xywh)
    
    # Compute IoU matrix [N_anchors, N_gt]
    ious = compute_iou(anchors_xyxy, gt_xyxy)
    
    # Find best GT for each anchor
    max_iou, max_idx = ious.max(dim=1)
    
    # Initialize all as ignored (-1)
    matches = torch.full((anchors_xywh.size(0),), -1, dtype=torch.long)
    
    # Positive: IoU >= pos_iou
    matches[max_iou >= pos_iou] = max_idx[max_iou >= pos_iou]
    
    # Negative: IoU < neg_iou
    matches[max_iou < neg_iou] = -2
    
    # Ensure each GT is matched at least once (best anchor)
    gt_best_iou, gt_best_anchor = ious.max(dim=0)
    matches[gt_best_anchor] = torch.arange(gt_xywh.size(0), device=gt_xywh.device)
    
    # Get matched boxes
    matched_boxes = gt_xywh[matches.clamp(min=0)]
    
    return matches, matched_boxes


def smooth_l1_loss(pred, target, beta=1/9):
    """Smooth L1 loss for localization"""
    diff = (pred - target).abs()
    loss = torch.where(diff < beta, 0.5 * diff**2 / beta, diff - 0.5 * beta)
    return loss


def ssd_loss(cls_logits, box_preds, anchors, gt_boxes_list, gt_labels_list, neg_pos_ratio=3):
    """
    SSD multi-task loss with hard negative mining.
    
    Args:
        cls_logits: [B, N, C+1] classification logits
        box_preds: [B, N, 4] box predictions
        anchors: [B, N, 4] anchor boxes
        gt_boxes_list: List of [M, 4] ground truth boxes per image
        gt_labels_list: List of [M] ground truth labels per image
        neg_pos_ratio: Ratio of negatives to positives in classification loss
    
    Returns:
        cls_loss: Classification loss
        loc_loss: Localization loss
    """
    B, N, _ = anchors.shape
    cls_loss_total = 0.0
    loc_loss_total = 0.0
    
    for b in range(B):
        gt_boxes = gt_boxes_list[b]
        gt_labels = gt_labels_list[b]
        
        # Match anchors to GT
        matches, matched_boxes = match_anchors(anchors[b], gt_boxes, gt_labels)
        
        # Prepare classification targets (0 = background, 1..C = classes)
        cls_targets = torch.zeros(N, dtype=torch.long, device=cls_logits.device)
        pos_mask = matches >= 0
        neg_mask = matches == -2
        
        cls_targets[pos_mask] = gt_labels[matches[pos_mask]] + 1  # shift by 1 for background
        
        # Classification loss (per anchor)
        cls_loss_all = F.cross_entropy(
            cls_logits[b], cls_targets, reduction='none'
        )
        
        # Hard Negative Mining
        num_pos = pos_mask.sum().item()
        num_neg = min(int(neg_pos_ratio * num_pos), neg_mask.sum().item())
        
        if num_neg > 0:
            # Select hardest negatives
            neg_losses = cls_loss_all[neg_mask]
            hard_neg_idx = torch.argsort(neg_losses, descending=True)[:num_neg]
            
            # Create selection mask
            selected = pos_mask.clone()
            neg_indices = torch.nonzero(neg_mask, as_tuple=False).squeeze(1)
            selected[neg_indices[hard_neg_idx]] = True
        else:
            selected = pos_mask
        
        # Classification loss (positives + hard negatives)
        cls_loss_total += cls_loss_all[selected].sum() / max(1, num_pos)
        
        # Localization loss (positives only)
        if num_pos > 0:
            loc_pred = box_preds[b][pos_mask]
            loc_target = matched_boxes[pos_mask]
            loc_loss = smooth_l1_loss(loc_pred, loc_target).sum()
            loc_loss_total += loc_loss / num_pos
    
    return cls_loss_total / B, loc_loss_total / B


# ============================================================================
# DECODING & NMS
# ============================================================================

def simple_nms(boxes, scores, iou_threshold=0.45):
    """Simple NMS implementation"""
    if len(boxes) == 0:
        return []
    
    boxes = torch.tensor(boxes) if not isinstance(boxes, torch.Tensor) else boxes
    scores = torch.tensor(scores) if not isinstance(scores, torch.Tensor) else scores
    
    # Convert to corners
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    
    order = scores.argsort(descending=True)
    keep = []
    
    while order.numel() > 0:
        i = int(order[0])
        keep.append(i)
        
        if order.numel() == 1:
            break
        
        rest = order[1:]
        
        # Compute IoU with remaining boxes
        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])
        
        inter_w = (xx2 - xx1).clamp(min=0)
        inter_h = (yy2 - yy1).clamp(min=0)
        inter = inter_w * inter_h
        
        iou = inter / (areas[i] + areas[rest] - inter + 1e-6)
        order = rest[iou <= iou_threshold]
    
    return keep


def decode_predictions(cls_logits, box_preds, anchors, conf_threshold=0.3, nms_threshold=0.45):
    """
    Decode SSD predictions to final detections.
    
    Returns:
        List of detections per image, each detection is (box, class_id, score)
    """
    probs = F.softmax(cls_logits, dim=-1)  # [B, N, C+1]
    B = probs.size(0)
    
    detections = []
    
    for b in range(B):
        p = probs[b]  # [N, C+1]
        boxes = box_preds[b]  # [N, 4]
        anc = anchors[b]  # [N, 4]
        
        # Get best class (excluding background at index 0)
        conf, cls = torch.max(p[:, 1:], dim=-1)  # [N]
        
        # Filter by confidence
        keep = conf >= conf_threshold
        boxes_kept = boxes[keep].clamp(0, 1)
        conf_kept = conf[keep]
        cls_kept = cls[keep]
        
        # Convert to corner format for NMS
        boxes_xyxy = xywh_to_xyxy(boxes_kept)
        
        # NMS per class
        dets = []
        for c in range(p.shape[1] - 1):  # exclude background
            mask = (cls_kept == c)
            if mask.sum() == 0:
                continue
            
            class_boxes = boxes_xyxy[mask]
            class_scores = conf_kept[mask]
            
            keep_idx = simple_nms(class_boxes, class_scores, nms_threshold)
            
            for idx in keep_idx:
                box = class_boxes[idx]
                score = float(class_scores[idx])
                dets.append((box, int(c), score))
        
        detections.append(dets)
    
    return detections


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_predictions(model, dataset, num_samples=4):
    """Visualize SSD predictions on sample images"""
    device = next(model.parameters()).device
    model.eval()
    
    rows = 2
    cols = max(1, num_samples // 2)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 8))
    axes = np.array(axes).reshape(-1)
    
    with torch.no_grad():
        for i in range(num_samples):
            img, gt_boxes, gt_labels = dataset[i]
            
            # Get predictions
            cls_logits, box_preds, anchors = model(img.unsqueeze(0).to(device))
            detections = decode_predictions(cls_logits, box_preds, anchors, 0.3, 0.45)[0]
            
            # Plot
            ax = axes[i]
            img_np = img.permute(1, 2, 0).numpy()
            ax.imshow(img_np)
            
            # Draw ground truth (dashed)
            for (cx, cy, w, h), label in zip(gt_boxes, gt_labels):
                x1 = (cx - w/2) * dataset.W
                y1 = (cy - h/2) * dataset.H
                w_px = w * dataset.W
                h_px = h * dataset.H
                
                color = dataset.class_colors[label]
                rect = patches.Rectangle(
                    (x1, y1), w_px, h_px, linewidth=2,
                    edgecolor=color, facecolor='none', linestyle='--'
                )
                ax.add_patch(rect)
                ax.text(x1, y1 - 5, f'GT: {dataset.class_names[label]}',
                       color=color, fontsize=8, fontweight='bold')
            
            # Draw predictions (solid)
            for box, class_id, score in detections:
                x1, y1, x2, y2 = (box * dataset.W).tolist()
                w_px = x2 - x1
                h_px = y2 - y1
                
                color = dataset.class_colors[class_id]
                rect = patches.Rectangle(
                    (x1, y1), w_px, h_px, linewidth=2,
                    edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(x1, y2 + 15,
                       f'{dataset.class_names[class_id]} {score:.2f}',
                       color=color, fontsize=8, fontweight='bold')
            
            ax.set_title(f'Sample {i+1}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('ssd_predictions.png', dpi=150, bbox_inches='tight')
    print("\nPredictions saved to 'ssd_predictions.png'")
    plt.show()


def plot_training_curves(history):
    """Plot training loss curves"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Total loss
    axes[0].plot(history['total_loss'], linewidth=2, color='#2E86AB')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title('Total SSD Loss', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Loss components
    axes[1].plot(history['cls_loss'], label='Classification', linewidth=2, color='#A23B72')
    axes[1].plot(history['loc_loss'], label='Localization', linewidth=2, color='#F18F01')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Loss', fontsize=11)
    axes[1].set_title('SSD Loss Components', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ssd_training_curves.png', dpi=150, bbox_inches='tight')
    print("Training curves saved to 'ssd_training_curves.png'")
    plt.show()


# ============================================================================
# TRAINING
# ============================================================================

def train_ssd(num_epochs=20, batch_size=16, learning_rate=1e-3):
    """Train SSD on synthetic dataset"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create dataset and dataloader
    dataset = ShapesDataset(num_samples=1000)
    
    def collate_fn(batch):
        """Custom collate to handle variable number of objects"""
        images, boxes, labels = zip(*batch)
        return torch.stack(images), list(boxes), list(labels)
    
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn
    )
    
    # Create model
    model = SSDTiny(num_classes=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {'total_loss': [], 'cls_loss': [], 'loc_loss': []}
    
    print("=" * 70)
    print("SSD TRAINING")
    print("=" * 70)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_cls_loss = 0.0
        epoch_loc_loss = 0.0
        num_batches = 0
        
        for images, gt_boxes, gt_labels in dataloader:
            images = images.to(device)
            gt_boxes = [b.to(device) for b in gt_boxes]
            gt_labels = [l.to(device) for l in gt_labels]
            
            # Forward pass
            cls_logits, box_preds, anchors = model(images)
            
            # Compute loss
            cls_loss, loc_loss = ssd_loss(
                cls_logits, box_preds, anchors, gt_boxes, gt_labels
            )
            total_loss = cls_loss + loc_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Accumulate losses
            epoch_cls_loss += cls_loss.item()
            epoch_loc_loss += loc_loss.item()
            num_batches += 1
        
        # Average losses
        avg_cls = epoch_cls_loss / num_batches
        avg_loc = epoch_loc_loss / num_batches
        avg_total = avg_cls + avg_loc
        
        history['total_loss'].append(avg_total)
        history['cls_loss'].append(avg_cls)
        history['loc_loss'].append(avg_loc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Loss: {avg_total:.4f} | "
              f"Cls: {avg_cls:.3f} | "
              f"Loc: {avg_loc:.3f}")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    
    return model, history, dataset


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("SSD: SINGLE SHOT MULTIBOX DETECTOR")
    print("Educational PyTorch Implementation with Synthetic Data")
    print("=" * 70 + "\n")
    
    # Train model
    model, history, dataset = train_ssd(
        num_epochs=20,
        batch_size=16,
        learning_rate=1e-3
    )
    
    # Plot training curves
    plot_training_curves(history)
    
    # Visualize predictions
    visualize_predictions(model, dataset, num_samples=4)
    
    print("\n" + "=" * 70)
    print("SSD Training and Visualization Complete!")
    print("=" * 70)
    print("\nKey Components Demonstrated:")
    print("  ✓ Multi-scale feature pyramids (P3 @ stride 16, P4 @ stride 32)")
    print("  ✓ Anchor boxes with multiple aspect ratios and scales")
    print("  ✓ IoU-based anchor matching (pos ≥ 0.5, neg < 0.4)")
    print("  ✓ Multi-task loss (classification + localization)")
    print("  ✓ Hard Negative Mining (3:1 ratio)")
    print("  ✓ Non-Maximum Suppression for clean detections")
    print("  ✓ End-to-end training on synthetic shapes")
    print("=" * 70 + "\n")