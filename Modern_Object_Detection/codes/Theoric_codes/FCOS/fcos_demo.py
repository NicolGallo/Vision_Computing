"""
FCOS (Fully Convolutional One-Stage) Complete Implementation
Anchor-free detector with FPN and centerness branch
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
# BACKBONE & FPN
# ============================================================================

class SimpleBackbone(nn.Module):
    """Simple ResNet-like backbone"""
    
    def __init__(self, in_channels=3):
        super().__init__()
        
        # C1: 224 -> 112
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # C2: 112 -> 56 (stride 4)
        self.layer1 = self._make_layer(64, 64, 2)
        
        # C3: 56 -> 28 (stride 8)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        
        # C4: 28 -> 14 (stride 16)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # C5: 14 -> 7 (stride 32)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(1, num_blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        
        return {'C3': c3, 'C4': c4, 'C5': c5}


class FPN(nn.Module):
    """Feature Pyramid Network"""
    
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        
        # Lateral connections (1x1 conv to reduce channels)
        self.lateral_c5 = nn.Conv2d(in_channels_list[2], out_channels, kernel_size=1)
        self.lateral_c4 = nn.Conv2d(in_channels_list[1], out_channels, kernel_size=1)
        self.lateral_c3 = nn.Conv2d(in_channels_list[0], out_channels, kernel_size=1)
        
        # Output convolutions (3x3 conv to reduce aliasing)
        self.output_p5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.output_p4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.output_p3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, features):
        c3, c4, c5 = features['C3'], features['C4'], features['C5']
        
        # Top-down pathway
        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4) + F.interpolate(p5, scale_factor=2, mode='nearest')
        p3 = self.lateral_c3(c3) + F.interpolate(p4, scale_factor=2, mode='nearest')
        
        # Output convolutions
        p5 = self.output_p5(p5)
        p4 = self.output_p4(p4)
        p3 = self.output_p3(p3)
        
        return [p3, p4, p5]


# ============================================================================
# FCOS MODEL
# ============================================================================

class Scale(nn.Module):
    """Learnable scale parameter"""
    
    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))
    
    def forward(self, x):
        return x * self.scale


class FCOSHead(nn.Module):
    """FCOS detection head (shared across FPN levels)"""
    
    def __init__(self, in_channels, num_classes, num_convs=4):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Classification tower
        cls_layers = []
        for i in range(num_convs):
            cls_layers.extend([
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.GroupNorm(32, in_channels),
                nn.ReLU(inplace=True)
            ])
        self.cls_tower = nn.Sequential(*cls_layers)
        
        # Classification output
        self.cls_logits = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        
        # Regression tower
        reg_layers = []
        for i in range(num_convs):
            reg_layers.extend([
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.GroupNorm(32, in_channels),
                nn.ReLU(inplace=True)
            ])
        self.reg_tower = nn.Sequential(*reg_layers)
        
        # Regression output (l, t, r, b)
        self.reg_pred = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        
        # Centerness branch
        self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        
        # Scales for regression (learnable per-level scaling)
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(3)])
        
        self._init_weights()
    
    def _init_weights(self):
        for modules in [self.cls_tower, self.reg_tower]:
            for m in modules.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        
        # Initialize classification head with bias for focal loss
        nn.init.normal_(self.cls_logits.weight, std=0.01)
        nn.init.constant_(self.cls_logits.bias, -2.19)  # -log((1-π)/π) for π=0.01
        
        nn.init.normal_(self.reg_pred.weight, std=0.01)
        nn.init.constant_(self.reg_pred.bias, 0)
        
        nn.init.normal_(self.centerness.weight, std=0.01)
        nn.init.constant_(self.centerness.bias, 0)
    
    def forward(self, x, level_idx=0):
        # Classification branch
        cls_feat = self.cls_tower(x)
        cls_logits = self.cls_logits(cls_feat)
        
        # Regression branch
        reg_feat = self.reg_tower(x)
        reg_pred = self.scales[level_idx](self.reg_pred(reg_feat))
        reg_pred = F.relu(reg_pred)  # Ensure positive distances
        
        # Centerness
        centerness = self.centerness(reg_feat)
        
        return cls_logits, reg_pred, centerness


class FCOSDetector(nn.Module):
    """
    FCOS anchor-free object detector
    Reference: "FCOS: Fully Convolutional One-Stage Object Detection" (Tian et al., 2019)
    """
    
    def __init__(self, num_classes=3, fpn_channels=256, num_convs=4):
        super().__init__()
        
        self.num_classes = num_classes
        self.strides = [8, 16, 32]  # FPN strides
        
        # Backbone + FPN
        self.backbone = SimpleBackbone()
        self.fpn = FPN(in_channels_list=[128, 256, 512], out_channels=fpn_channels)
        
        # FCOS head (shared across FPN levels)
        self.head = FCOSHead(fpn_channels, num_classes, num_convs)
    
    def forward(self, x):
        # Extract multi-scale features
        features = self.backbone(x)  # Dict with 'C3', 'C4', 'C5'
        
        # FPN
        fpn_features = self.fpn(features)  # List [P3, P4, P5]
        
        # FCOS heads
        cls_logits_list = []
        reg_preds_list = []
        centerness_list = []
        
        for level_idx, feat in enumerate(fpn_features):
            cls_logits, reg_pred, centerness = self.head(feat, level_idx)
            cls_logits_list.append(cls_logits)
            reg_preds_list.append(reg_pred)
            centerness_list.append(centerness)
        
        return cls_logits_list, reg_preds_list, centerness_list


# ============================================================================
# FCOS LOSS
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        """
        Args:
            pred: [N, C] predicted logits
            target: [N, C] one-hot targets
        """
        pred_sigmoid = torch.sigmoid(pred)
        pt = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
        focal_weight = (1 - pt) ** self.gamma
        
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        loss = self.alpha * focal_weight * bce_loss
        
        return loss.sum()


class IoULoss(nn.Module):
    """IoU Loss for bounding box regression"""
    
    def __init__(self, loss_type='iou'):
        super().__init__()
        self.loss_type = loss_type
    
    def forward(self, pred, target):
        """
        Args:
            pred: [N, 4] predicted ltrb
            target: [N, 4] target ltrb
        """
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]
        
        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]
        
        # Predicted and target areas
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)
        target_area = (target_left + target_right) * (target_top + target_bottom)
        
        # Intersection
        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_top, target_top) + torch.min(pred_bottom, target_bottom)
        area_intersect = w_intersect * h_intersect
        
        # Union
        area_union = pred_area + target_area - area_intersect
        
        # IoU
        iou = area_intersect / (area_union + 1e-6)
        
        # IoU loss
        loss = 1 - iou
        
        return loss.sum()


class FCOSLoss(nn.Module):
    """
    FCOS Loss: Focal Loss + IoU Loss + Centerness Loss
    """
    
    def __init__(self, num_classes=3, strides=[8, 16, 32]):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.iou_loss = IoULoss(loss_type='iou')
    
    def compute_centerness_targets(self, reg_targets):
        """
        Compute centerness targets from regression targets
        centerness = sqrt((min(l,r) / max(l,r)) * (min(t,b) / max(t,b)))
        """
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        
        centerness = (left_right.min(dim=-1)[0] / (left_right.max(dim=-1)[0] + 1e-6)) * \
                     (top_bottom.min(dim=-1)[0] / (top_bottom.max(dim=-1)[0] + 1e-6))
        
        return torch.sqrt(centerness)
    
    def forward(self, cls_logits_list, reg_preds_list, centerness_list, targets_list):
        """
        Args:
            cls_logits_list: List of [B, C, H, W] classification logits
            reg_preds_list: List of [B, 4, H, W] regression predictions (ltrb)
            centerness_list: List of [B, 1, H, W] centerness predictions
            targets_list: List of target dicts with 'labels', 'reg_targets', 'locations'
        
        Returns:
            total_loss, loss_dict
        """
        cls_losses = []
        reg_losses = []
        centerness_losses = []
        
        for level_idx, (cls_logits, reg_preds, centerness_pred) in enumerate(
            zip(cls_logits_list, reg_preds_list, centerness_list)
        ):
            B, C, H, W = cls_logits.shape
            
            # Reshape predictions
            cls_logits = cls_logits.permute(0, 2, 3, 1).reshape(-1, C)
            reg_preds = reg_preds.permute(0, 2, 3, 1).reshape(-1, 4)
            centerness_pred = centerness_pred.permute(0, 2, 3, 1).reshape(-1)
            
            # Get targets for this level
            labels = targets_list[level_idx]['labels'].reshape(-1)
            reg_targets = targets_list[level_idx]['reg_targets'].reshape(-1, 4)
            
            # Positive samples mask
            pos_mask = labels >= 0
            num_pos = pos_mask.sum().clamp(min=1).float()
            
            # Classification loss (all locations)
            labels_one_hot = F.one_hot(
                labels.clamp(min=0), 
                num_classes=self.num_classes
            ).float()
            labels_one_hot[~pos_mask] = 0  # Set background to all zeros
            
            cls_loss = self.focal_loss(cls_logits, labels_one_hot) / num_pos
            cls_losses.append(cls_loss)
            
            # Regression loss (only positive samples)
            if pos_mask.sum() > 0:
                reg_loss = self.iou_loss(
                    reg_preds[pos_mask], 
                    reg_targets[pos_mask]
                ) / num_pos
                reg_losses.append(reg_loss)
                
                # Centerness loss
                centerness_targets = self.compute_centerness_targets(reg_targets[pos_mask])
                centerness_loss = F.binary_cross_entropy_with_logits(
                    centerness_pred[pos_mask],
                    centerness_targets,
                    reduction='sum'
                ) / num_pos
                centerness_losses.append(centerness_loss)
        
        # Aggregate losses
        total_cls_loss = sum(cls_losses) / len(cls_losses)
        total_reg_loss = sum(reg_losses) / len(reg_losses) if reg_losses else torch.tensor(0.0)
        total_centerness_loss = sum(centerness_losses) / len(centerness_losses) if centerness_losses else torch.tensor(0.0)
        
        total_loss = total_cls_loss + total_reg_loss + total_centerness_loss
        
        return total_loss, {
            'cls_loss': total_cls_loss.item(),
            'reg_loss': total_reg_loss.item() if isinstance(total_reg_loss, torch.Tensor) else 0.0,
            'centerness_loss': total_centerness_loss.item() if isinstance(total_centerness_loss, torch.Tensor) else 0.0
        }


# ============================================================================
# SYNTHETIC DATASET
# ============================================================================

class SyntheticObjectDataset(Dataset):
    """Generates synthetic images with geometric shapes for FCOS"""
    
    def __init__(self, num_samples=1000, image_size=224, max_objects=3):
        self.num_samples = num_samples
        self.image_size = image_size
        self.max_objects = max_objects
        self.strides = [8, 16, 32]
        
        # Class names and colors
        self.class_names = ['circle', 'square', 'triangle']
        self.num_classes = len(self.class_names)
        self.class_colors = [
            np.array([0.2, 0.4, 0.8]),  # blue
            np.array([0.8, 0.2, 0.2]),  # red
            np.array([0.2, 0.8, 0.3]),  # green
        ]
    
    def __len__(self):
        return self.num_samples
    
    def draw_shape(self, image, x, y, size, class_id):
        """Draw a shape on the image"""
        color = self.class_colors[class_id]
        
        if class_id == 0:  # Circle
            y_grid, x_grid = np.ogrid[:self.image_size, :self.image_size]
            mask = (x_grid - x) ** 2 + (y_grid - y) ** 2 <= (size / 2) ** 2
            image[mask] = color
        
        elif class_id == 1:  # Square
            x1 = max(0, int(x - size / 2))
            y1 = max(0, int(y - size / 2))
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
            y_grid, x_grid = np.meshgrid(np.arange(self.image_size), 
                                         np.arange(self.image_size), indexing='ij')
            points = np.stack([x_grid.ravel(), y_grid.ravel()], axis=1)
            mask = path.contains_points(points).reshape(self.image_size, self.image_size)
            image[mask] = color
    
    def __getitem__(self, idx):
        # Create blank image
        image = np.ones((self.image_size, self.image_size, 3), dtype=np.float32) * 0.95
        
        # Generate random objects
        num_objects = np.random.randint(1, self.max_objects + 1)
        objects = []
        
        for _ in range(num_objects):
            class_id = np.random.randint(0, self.num_classes)
            size = np.random.randint(30, 80)
            x = np.random.randint(size, self.image_size - size)
            y = np.random.randint(size, self.image_size - size)
            
            self.draw_shape(image, x, y, size, class_id)
            
            objects.append({
                'x1': x - size / 2,
                'y1': y - size / 2,
                'x2': x + size / 2,
                'y2': y + size / 2,
                'class': class_id
            })
        
        # Convert to tensor [3, H, W]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Create FCOS targets for each FPN level
        targets = self.encode_targets(objects)
        
        return image_tensor, targets, objects
    
    def encode_targets(self, objects):
        """Encode objects into FCOS targets for each FPN level"""
        targets_list = []
        
        for stride in self.strides:
            H = W = self.image_size // stride
            
            labels = torch.full((H, W), -1, dtype=torch.long)  # -1 = ignore
            reg_targets = torch.zeros((H, W, 4), dtype=torch.float32)
            
            # Create location grid
            for i in range(H):
                for j in range(W):
                    loc_x = (j + 0.5) * stride
                    loc_y = (i + 0.5) * stride
                    
                    # Find if this location is inside any object
                    for obj in objects:
                        if (obj['x1'] <= loc_x <= obj['x2'] and 
                            obj['y1'] <= loc_y <= obj['y2']):
                            
                            # Compute ltrb distances
                            l = loc_x - obj['x1']
                            t = loc_y - obj['y1']
                            r = obj['x2'] - loc_x
                            b = obj['y2'] - loc_y
                            
                            labels[i, j] = obj['class']
                            reg_targets[i, j] = torch.tensor([l, t, r, b])
                            break
            
            targets_list.append({
                'labels': labels,
                'reg_targets': reg_targets
            })
        
        return targets_list


# ============================================================================
# TRAINING
# ============================================================================

def train_fcos(num_epochs=30, batch_size=8, learning_rate=1e-3):
    """Train FCOS on synthetic dataset"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create dataset and dataloader
    dataset = SyntheticObjectDataset(num_samples=1000, image_size=224)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           collate_fn=collate_fn)
    
    # Create model
    model = FCOSDetector(num_classes=3, fpn_channels=256, num_convs=4).to(device)
    
    # Loss and optimizer
    criterion = FCOSLoss(num_classes=3, strides=[8, 16, 32])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("=" * 70)
    print("FCOS TRAINING - Anchor-Free Detection with FPN")
    print("=" * 70)
    
    history = {'total_loss': [], 'cls_loss': [], 'reg_loss': [], 'centerness_loss': []}
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        epoch_components = {'cls': [], 'reg': [], 'centerness': []}
        
        for batch_idx, (images, targets_list, _) in enumerate(dataloader):
            images = images.to(device)
            targets_list = [[{k: v.to(device) for k, v in t.items()} for t in batch_targets] 
                           for batch_targets in targets_list]
            
            # Transpose targets from [B][L] to [L][B]
            targets_by_level = []
            for level_idx in range(3):
                level_targets = {
                    'labels': torch.stack([t[level_idx]['labels'] for t in targets_list]),
                    'reg_targets': torch.stack([t[level_idx]['reg_targets'] for t in targets_list])
                }
                targets_by_level.append(level_targets)
            
            # Forward pass
            cls_logits_list, reg_preds_list, centerness_list = model(images)
            
            # Compute loss
            loss, loss_dict = criterion(cls_logits_list, reg_preds_list, 
                                       centerness_list, targets_by_level)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Record losses
            epoch_losses.append(loss.item())
            epoch_components['cls'].append(loss_dict['cls_loss'])
            epoch_components['reg'].append(loss_dict['reg_loss'])
            epoch_components['centerness'].append(loss_dict['centerness_loss'])
        
        # Epoch summary
        avg_loss = np.mean(epoch_losses)
        history['total_loss'].append(avg_loss)
        history['cls_loss'].append(np.mean(epoch_components['cls']))
        history['reg_loss'].append(np.mean(epoch_components['reg']))
        history['centerness_loss'].append(np.mean(epoch_components['centerness']))
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} " +
              f"| cls: {history['cls_loss'][-1]:.3f} " +
              f"| reg: {history['reg_loss'][-1]:.3f} " +
              f"| center: {history['centerness_loss'][-1]:.3f}")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    
    return model, history, dataset


def collate_fn(batch):
    """Custom collate function for batching"""
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    objects = [item[2] for item in batch]
    return images, targets, objects


# ============================================================================
# VISUALIZATION
# ============================================================================

def decode_predictions(cls_logits_list, reg_preds_list, centerness_list, 
                      image_size=224, conf_threshold=0.3):
    """Decode FCOS predictions"""
    strides = [8, 16, 32]
    detections = []
    
    for level_idx, (cls_logits, reg_preds, centerness) in enumerate(
        zip(cls_logits_list, reg_preds_list, centerness_list)
    ):
        H, W = cls_logits.shape[2:]
        stride = strides[level_idx]
        
        # Reshape
        cls_scores = torch.sigmoid(cls_logits[0]).permute(1, 2, 0).reshape(-1, 3)
        reg_pred = reg_preds[0].permute(1, 2, 0).reshape(-1, 4)
        centerness_pred = torch.sigmoid(centerness[0]).permute(1, 2, 0).reshape(-1)
        
        # Get scores and labels
        scores, labels = cls_scores.max(dim=1)
        scores = scores * centerness_pred
        
        # Filter
        keep = scores > conf_threshold
        if keep.sum() == 0:
            continue
        
        scores = scores[keep]
        labels = labels[keep]
        reg_pred = reg_pred[keep]
        
        # Decode boxes
        for idx in range(len(scores)):
            i = idx // W
            j = idx % W
            
            loc_x = (j + 0.5) * stride
            loc_y = (i + 0.5) * stride
            
            l, t, r, b = reg_pred[idx]
            x1 = loc_x - l.item()
            y1 = loc_y - t.item()
            x2 = loc_x + r.item()
            y2 = loc_y + b.item()
            
            detections.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'conf': scores[idx].item(),
                'class': labels[idx].item()
            })
    
    return detections


def visualize_predictions(model, dataset, num_samples=4):
    """Visualize FCOS predictions"""
    
    device = next(model.parameters()).device
    model.eval()
    
    fig, axes = plt.subplots(2, num_samples // 2, figsize=(15, 8))
    axes = axes.flatten()
    
    with torch.no_grad():
        for i in range(num_samples):
            image, targets, gt_objects = dataset[i]
            
            # Get prediction
            cls_logits, reg_preds, centerness = model(image.unsqueeze(0).to(device))
            
            # Decode
            detections = decode_predictions(cls_logits, reg_preds, centerness,
                                           image_size=dataset.image_size,
                                           conf_threshold=0.3)
            
            # Plot
            ax = axes[i]
            img_np = image.permute(1, 2, 0).numpy()
            ax.imshow(img_np)
            
            # Draw ground truth (dashed)
            for obj in gt_objects:
                x1, y1, x2, y2 = obj['x1'], obj['y1'], obj['x2'], obj['y2']
                w, h = x2 - x1, y2 - y1
                
                color = dataset.class_colors[obj['class']]
                rect = patches.Rectangle((x1, y1), w, h, linewidth=2,
                                        edgecolor=color, facecolor='none',
                                        linestyle='--')
                ax.add_patch(rect)
                ax.text(x1, y1 - 5, f"GT: {dataset.class_names[obj['class']]}",
                       color=color, fontsize=8, fontweight='bold')
            
            # Draw predictions (solid)
            for det in detections:
                x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
                w, h = x2 - x1, y2 - y1
                
                color = dataset.class_colors[det['class']]
                rect = patches.Rectangle((x1, y1), w, h, linewidth=2,
                                        edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1 + h + 15,
                       f"{dataset.class_names[det['class']]} {det['conf']:.2f}",
                       color=color, fontsize=8, fontweight='bold')
            
            ax.set_title(f"Sample {i+1}")
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('fcos_predictions.png', dpi=150, bbox_inches='tight')
    print("\nPredictions saved to 'fcos_predictions.png'")
    plt.show()


def plot_training_curves(history):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Total loss
    axes[0].plot(history['total_loss'], 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total FCOS Loss')
    axes[0].grid(True, alpha=0.3)
    
    # Component losses
    axes[1].plot(history['cls_loss'], label='Classification (Focal)', linewidth=2)
    axes[1].plot(history['reg_loss'], label='Regression (IoU)', linewidth=2)
    axes[1].plot(history['centerness_loss'], label='Centerness', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('FCOS Loss Components')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fcos_training_curves.png', dpi=150, bbox_inches='tight')
    print("Training curves saved to 'fcos_training_curves.png'")
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("FCOS: FULLY CONVOLUTIONAL ONE-STAGE OBJECT DETECTION")
    print("Anchor-Free Detection with FPN and Centerness Branch")
    print("=" * 70 + "\n")
    
    # Train model
    model, history, dataset = train_fcos(num_epochs=30, batch_size=8, learning_rate=1e-3)
    
    # Plot training curves
    plot_training_curves(history)
    
    # Visualize predictions
    visualize_predictions(model, dataset, num_samples=4)
    
    print("\n" + "=" * 70)
    print("FCOS Training and Visualization Complete!")
    print("=" * 70)
    print("\nKey FCOS Innovations:")
    print("  ✓ Anchor-free detection (no predefined anchor boxes)")
    print("  ✓ Per-pixel predictions (ltrb distances)")
    print("  ✓ FPN multi-scale features (P3, P4, P5)")
    print("  ✓ Centerness branch (suppress low-quality boxes)")
    print("  ✓ Focal Loss (handle class imbalance)")
    print("  ✓ IoU Loss (better localization)")
    print("\nFCOS vs YOLO:")
    print("  • YOLO: Grid-based with anchor boxes")
    print("  • FCOS: Anchor-free with per-pixel predictions")
    print("  • FCOS: Uses FPN for multi-scale features")
    print("  • FCOS: Centerness for quality estimation")
    print("=" * 70 + "\n")