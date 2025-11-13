# Save as: retinanet_tiny_pedagogical.py
"""
RetinaNet - Educational Implementation
======================================
A simplified RetinaNet implementation for teaching object detection concepts.

Key Concepts Demonstrated:
- Feature Pyramid Network (FPN) for multi-scale detection
- Focal Loss to address class imbalance (γ=2, α=0.25)
- One-stage detection without background class (sigmoid per class)
- Shared classification and regression subnets across pyramid levels
- Anchor-based detection with multiple aspect ratios and scales
- Non-Maximum Suppression (NMS) for post-processing

Key Differences from SSD:
- Uses Focal Loss instead of Hard Negative Mining
- No explicit background class (uses sigmoid instead of softmax)
- Feature Pyramid Network with top-down pathway
- Deeper prediction subnets (4 conv layers vs 1)
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
# BACKBONE + FEATURE PYRAMID NETWORK (FPN)
# ============================================================================

class TinyBackboneFPN(nn.Module):
    """
    Lightweight backbone with Feature Pyramid Network.
    
    Architecture:
    1. Stem: Progressive downsampling to create base features
    2. Bottom-up pathway: Creates features at different scales (C3, C4)
    3. Top-down pathway: Builds pyramid features (P3, P4) by combining
       high-level semantic features with low-level spatial details
    
    Output:
    - P3: stride 16 (14x14 for 224x224 input) - finer details
    - P4: stride 32 (7x7 for 224x224 input) - coarser, semantic features
    """
    def __init__(self):
        super().__init__()
        
        # Bottom-up pathway (backbone)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),      # stride 2: 224 -> 112
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.MaxPool2d(2, 2),              # stride 4: 112 -> 56
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True)
        )
        
        # Additional downsampling for multi-scale features
        self.c3_down = nn.Conv2d(256, 256, 3, 2, 1)  # stride 8 -> 16
        self.c4_down = nn.Conv2d(256, 256, 3, 2, 1)  # stride 16 -> 32
        
        # Top-down pathway (FPN)
        # 1x1 convs to reduce channels for lateral connections
        self.p3_lateral = nn.Conv2d(256, 256, 1)
        self.p4_lateral = nn.Conv2d(256, 256, 1)
        
        # 3x3 convs to reduce aliasing after upsampling
        self.p3_output = nn.Conv2d(256, 256, 3, 1, 1)
        self.p4_output = nn.Conv2d(256, 256, 3, 1, 1)
    
    def forward(self, x):
        # Bottom-up pathway
        c = self.stem(x)          # stride 4
        c3 = self.c3_down(c)      # stride 16
        c4 = self.c4_down(c3)     # stride 32
        
        # Top-down pathway with lateral connections
        # P4: just process C4
        p4 = self.p4_output(F.relu(self.p4_lateral(c4)))
        
        # P3: upsample P4 and add to C3 (lateral connection)
        p4_up = F.interpolate(self.p4_lateral(c4), scale_factor=2, mode='nearest')
        p3 = self.p3_output(F.relu(self.p3_lateral(c3) + p4_up))
        
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
        
        for i in range(H):
            for j in range(W):
                cx = (j + 0.5) * stride
                cy = (i + 0.5) * stride
                
                for ratio in ratios:
                    for scale in scales:
                        w = scale * stride * np.sqrt(1.0 / ratio)
                        h = scale * stride * np.sqrt(ratio)
                        anchors.append([cx, cy, w, h])
        
        # Normalize coordinates
        anchors = np.array(anchors, dtype=np.float32)
        anchors[:, 0] /= (W * stride)
        anchors[:, 1] /= (H * stride)
        anchors[:, 2] /= (W * stride)
        anchors[:, 3] /= (H * stride)
        
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
    """Compute IoU between two sets of boxes"""
    N, M = boxes_a.size(0), boxes_b.size(0)
    
    a = boxes_a[:, None, :].expand(N, M, 4)
    b = boxes_b[None, :, :].expand(N, M, 4)
    
    x1 = torch.max(a[..., 0], b[..., 0])
    y1 = torch.max(a[..., 1], b[..., 1])
    x2 = torch.min(a[..., 2], b[..., 2])
    y2 = torch.min(a[..., 3], b[..., 3])
    
    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter_area = inter_w * inter_h
    
    area_a = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)
    
    union = area_a + area_b - inter_area + 1e-6
    return inter_area / union


# ============================================================================
# RETINANET MODEL
# ============================================================================

class RetinaNetTiny(nn.Module):
    """
    Simplified RetinaNet detector.
    
    Key components:
    1. FPN backbone for multi-scale features
    2. Shared classification subnet (4 conv layers)
    3. Shared box regression subnet (4 conv layers)
    4. Multiple anchors per spatial location
    """
    def __init__(self, num_classes=3, ratios=(0.5, 1.0, 2.0), scales=(1.0, 1.6)):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = len(ratios) * len(scales)
        self.ratios = ratios
        self.scales = scales
        
        # Backbone with FPN
        self.backbone = TinyBackboneFPN()
        
        # Shared classification subnet (deeper than SSD)
        # Uses 4 conv layers as in original RetinaNet
        self.cls_subnet = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(256, self.num_anchors * num_classes, 3, 1, 1)
        )
        
        # Shared box regression subnet
        self.box_subnet = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(256, self.num_anchors * 4, 3, 1, 1)
        )
        
        # Initialize with proper bias for classification
        # (helps with training stability when using focal loss)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize final classification layer with prior bias"""
        for m in self.cls_subnet.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Prior bias for rare object classes (helps focal loss)
        prior_prob = 0.01
        bias_value = -np.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_subnet[-1].bias, bias_value)
    
    def forward(self, x):
        B = x.size(0)
        
        # Extract FPN features
        p3, p4 = self.backbone(x)
        
        def process_level(features):
            """Process a single pyramid level"""
            H, W = features.shape[2], features.shape[3]
            
            # Classification: [B, A*C, H, W] -> [B, H*W*A, C]
            cls = self.cls_subnet(features)
            cls = cls.permute(0, 2, 3, 1).contiguous()
            cls = cls.view(B, H * W * self.num_anchors, self.num_classes)
            
            # Box regression: [B, A*4, H, W] -> [B, H*W*A, 4]
            box = self.box_subnet(features)
            box = box.permute(0, 2, 3, 1).contiguous()
            box = box.view(B, H * W * self.num_anchors, 4)
            
            return cls, box, H, W
        
        # Process both pyramid levels
        cls3, box3, H3, W3 = process_level(p3)
        cls4, box4, H4, W4 = process_level(p4)
        
        # Concatenate predictions
        cls_logits = torch.cat([cls3, cls4], dim=1)  # [B, N, C]
        box_preds = torch.cat([box3, box4], dim=1)   # [B, N, 4]
        
        # Generate anchors
        strides = [16, 32]
        anchors = make_anchors([(H3, W3), (H4, W4)], strides, self.ratios, self.scales)
        anchors = torch.cat(anchors, dim=0).to(x.device)
        anchors = anchors.unsqueeze(0).expand(B, -1, -1)
        
        return cls_logits, box_preds, anchors


# ============================================================================
# FOCAL LOSS & MATCHING
# ============================================================================

def match_anchors(anchors, gt_boxes, gt_labels, pos_iou=0.5, neg_iou=0.4):
    """
    Match anchors to ground truth boxes based on IoU.
    
    Returns:
        matches: Indices of matched GT (-2 for negatives, -1 for ignored)
        matched_boxes: Matched GT boxes
        matched_labels: Matched GT labels
    """
    if gt_boxes.numel() == 0:
        matches = torch.full((anchors.size(0),), -1, dtype=torch.long)
        return matches, torch.zeros_like(anchors), torch.zeros(anchors.size(0), dtype=torch.long)
    
    anchors_xyxy = xywh_to_xyxy(anchors)
    gt_xyxy = xywh_to_xyxy(gt_boxes)
    
    ious = compute_iou(anchors_xyxy, gt_xyxy)
    
    max_iou, max_idx = ious.max(dim=1)
    
    matches = torch.full((anchors.size(0),), -1, dtype=torch.long)
    matches[max_iou >= pos_iou] = max_idx[max_iou >= pos_iou]
    matches[max_iou < neg_iou] = -2
    
    # Ensure each GT matched at least once
    gt_best_iou, gt_best_anchor = ious.max(dim=0)
    matches[gt_best_anchor] = torch.arange(gt_boxes.size(0), device=gt_boxes.device)
    
    matched_boxes = gt_boxes[matches.clamp(min=0)]
    matched_labels = gt_labels[matches.clamp(min=0)]
    
    return matches, matched_boxes, matched_labels


def focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction='sum'):
    """
    Focal Loss for addressing class imbalance.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        inputs: [N, C] logits
        targets: [N, C] binary targets (0 or 1)
        alpha: Weighting factor (default 0.25)
        gamma: Focusing parameter (default 2.0)
        
    The focal loss down-weights easy examples and focuses on hard negatives.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    
    focal_weight = (1 - p_t).pow(gamma)
    loss = alpha_t * focal_weight * ce_loss
    
    if reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean()
    return loss


def retinanet_loss(cls_logits, box_preds, anchors, gt_boxes_list, gt_labels_list):
    """
    RetinaNet loss: Focal loss for classification + Smooth L1 for localization.
    
    Key differences from SSD:
    - Uses focal loss instead of hard negative mining
    - No background class (uses sigmoid per class)
    """
    B, N, C = cls_logits.shape
    total_focal = 0.0
    total_loc = 0.0
    
    for b in range(B):
        gt_boxes = gt_boxes_list[b]
        gt_labels = gt_labels_list[b]
        
        # Match anchors to GT
        matches, matched_boxes, matched_labels = match_anchors(
            anchors[b], gt_boxes, gt_labels
        )
        
        pos_mask = matches >= 0
        neg_mask = matches == -2
        
        # Classification targets (one-hot encoding)
        cls_targets = torch.zeros((N, C), device=cls_logits.device)
        cls_targets[pos_mask, matched_labels[pos_mask]] = 1.0
        
        # Focal loss (includes both positives and negatives)
        focal = focal_loss(cls_logits[b], cls_targets, alpha=0.25, gamma=2.0, reduction='sum')
        num_pos = max(1, pos_mask.sum().item())
        total_focal += focal / num_pos
        
        # Localization loss (positives only)
        if pos_mask.any():
            loc_pred = box_preds[b][pos_mask]
            loc_target = matched_boxes[pos_mask]
            total_loc += F.smooth_l1_loss(loc_pred, loc_target, reduction='sum') / num_pos
    
    return total_focal / B, total_loc / B


# ============================================================================
# DECODING & NMS
# ============================================================================

def simple_nms(boxes, scores, iou_threshold=0.45):
    """Simple NMS implementation"""
    if len(boxes) == 0:
        return []
    
    boxes = boxes if isinstance(boxes, torch.Tensor) else torch.tensor(boxes)
    scores = scores if isinstance(scores, torch.Tensor) else torch.tensor(scores)
    
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


def decode_predictions(cls_logits, box_preds, conf_threshold=0.3, nms_threshold=0.45):
    """
    Decode RetinaNet predictions to final detections.
    
    Note: Uses sigmoid (not softmax) since there's no background class.
    Each class is predicted independently.
    """
    B, N, C = cls_logits.shape
    detections = []
    
    for b in range(B):
        probs = torch.sigmoid(cls_logits[b])  # [N, C]
        boxes = box_preds[b].clamp(0, 1)       # [N, 4]
        
        dets = []
        
        # Process each class independently
        for c in range(C):
            conf = probs[:, c]
            mask = conf >= conf_threshold
            
            if mask.sum() == 0:
                continue
            
            class_boxes = xywh_to_xyxy(boxes[mask])
            class_scores = conf[mask]
            
            # Apply NMS
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
    """Visualize RetinaNet predictions on sample images"""
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
            detections = decode_predictions(cls_logits, box_preds, 0.3, 0.45)[0]
            
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
    plt.savefig('retinanet_predictions.png', dpi=150, bbox_inches='tight')
    print("\nPredictions saved to 'retinanet_predictions.png'")
    plt.show()


def plot_training_curves(history):
    """Plot training loss curves"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Total loss
    axes[0].plot(history['total_loss'], linewidth=2, color='#6A4C93')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title('Total RetinaNet Loss', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Loss components
    axes[1].plot(history['focal_loss'], label='Focal Loss (Classification)', linewidth=2, color='#C1666B')
    axes[1].plot(history['loc_loss'], label='Smooth L1 (Localization)', linewidth=2, color='#48A9A6')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Loss', fontsize=11)
    axes[1].set_title('RetinaNet Loss Components', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('retinanet_training_curves.png', dpi=150, bbox_inches='tight')
    print("Training curves saved to 'retinanet_training_curves.png'")
    plt.show()


# ============================================================================
# TRAINING
# ============================================================================

def train_retinanet(num_epochs=20, batch_size=16, learning_rate=1e-3):
    """Train RetinaNet on synthetic dataset"""
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
    model = RetinaNetTiny(num_classes=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {'total_loss': [], 'focal_loss': [], 'loc_loss': []}
    
    print("=" * 70)
    print("RETINANET TRAINING")
    print("=" * 70)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_focal_loss = 0.0
        epoch_loc_loss = 0.0
        num_batches = 0
        
        for images, gt_boxes, gt_labels in dataloader:
            images = images.to(device)
            gt_boxes = [b.to(device) for b in gt_boxes]
            gt_labels = [l.to(device) for l in gt_labels]
            
            # Forward pass
            cls_logits, box_preds, anchors = model(images)
            
            # Compute loss
            focal_loss_val, loc_loss_val = retinanet_loss(
                cls_logits, box_preds, anchors, gt_boxes, gt_labels
            )
            total_loss = focal_loss_val + loc_loss_val
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Accumulate losses
            epoch_focal_loss += focal_loss_val.item()
            epoch_loc_loss += loc_loss_val.item()
            num_batches += 1
        
        # Average losses
        avg_focal = epoch_focal_loss / num_batches
        avg_loc = epoch_loc_loss / num_batches
        avg_total = avg_focal + avg_loc
        
        history['total_loss'].append(avg_total)
        history['focal_loss'].append(avg_focal)
        history['loc_loss'].append(avg_loc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Loss: {avg_total:.4f} | "
              f"Focal: {avg_focal:.3f} | "
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
    print("RETINANET: FOCAL LOSS FOR DENSE OBJECT DETECTION")
    print("Educational PyTorch Implementation with Synthetic Data")
    print("=" * 70 + "\n")
    
    # Train model
    model, history, dataset = train_retinanet(
        num_epochs=20,
        batch_size=16,
        learning_rate=1e-3
    )
    
    # Plot training curves
    plot_training_curves(history)
    
    # Visualize predictions
    visualize_predictions(model, dataset, num_samples=4)
    
    print("\n" + "=" * 70)
    print("RetinaNet Training and Visualization Complete!")
    print("=" * 70)
    print("\nKey Components Demonstrated:")
    print("  ✓ Feature Pyramid Network (FPN) with top-down pathway")
    print("  ✓ Focal Loss to address class imbalance (γ=2, α=0.25)")
    print("  ✓ No background class (sigmoid per class)")
    print("  ✓ Shared classification and regression subnets (4 conv layers)")
    print("  ✓ Multi-scale anchor-based detection")
    print("  ✓ End-to-end training on synthetic shapes")
    print("\nKey Differences from SSD:")
    print("  • Uses Focal Loss instead of Hard Negative Mining")
    print("  • Deeper prediction heads (4 layers vs 1)")
    print("  • FPN for better multi-scale feature fusion")
    print("  • Sigmoid activation (no explicit background class)")
    print("=" * 70 + "\n")