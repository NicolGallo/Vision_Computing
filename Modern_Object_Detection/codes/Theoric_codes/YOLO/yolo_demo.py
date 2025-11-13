# Save this as: yolo_complete.py

"""
YOLO (You Only Look Once) Complete Implementation
Includes: Model, Loss, Training, Synthetic Dataset, Visualization
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
    YOLO object detector
    Reference: "You Only Look Once: Unified, Real-Time Object Detection"
    """
    
    def __init__(self, num_classes=3, grid_size=7, num_boxes=2):
        super().__init__()
        
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        
        # Backbone
        self.backbone = SimpleBackbone(in_channels=3)
        feature_dim = self.backbone.out_channels
        
        # Detection head
        # Output: S × S × (B × 5 + C)
        output_channels = num_boxes * 5 + num_classes
        
        self.detection_head = nn.Sequential(
            nn.Conv2d(feature_dim, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(1024, output_channels, kernel_size=1)
        )
        
        # Adaptive pooling to grid size
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
        Args:
            x: [B, 3, H, W] input images
        
        Returns:
            predictions: [B, S, S, B, 5+C]
                where each box is [x, y, w, h, conf, class_probs...]
        """
        batch_size = x.size(0)
        
        # Extract features
        features = self.backbone(x)  # [B, 512, H/16, W/16]
        
        # Pool to grid size
        features = self.adaptive_pool(features)  # [B, 512, S, S]
        
        # Detection head
        out = self.detection_head(features)  # [B, B*5+C, S, S]
        
        # Reshape: [B, S, S, B*5+C]
        out = out.permute(0, 2, 3, 1).contiguous()
        
        # Reshape: [B, S, S, B, 5+C]
        out = out.view(batch_size, self.grid_size, self.grid_size,
                      self.num_boxes, 5 + self.num_classes)
        
        # Apply activations
        predictions = out.clone()
        predictions[..., 0:2] = torch.sigmoid(out[..., 0:2])  # x, y ∈ [0,1]
        predictions[..., 4:5] = torch.sigmoid(out[..., 4:5])  # confidence
        predictions[..., 5:] = F.softmax(out[..., 5:], dim=-1)  # classes
        
        return predictions


# ============================================================================
# YOLO LOSS (from your slides)
# ============================================================================

class YOLOLoss(nn.Module):
    """
    YOLO loss function with all components:
    L = λ_coord * L_coord + L_conf^obj + λ_noobj * L_conf^noobj + L_class
    """
    
    def __init__(self, grid_size=7, num_boxes=2, num_classes=3,
                 lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.S = grid_size
        self.B = num_boxes
        self.C = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
    
    def compute_iou(self, box1, box2):
        """Compute IoU between two boxes [x, y, w, h]"""
        # Convert to corners
        box1_x1 = box1[..., 0:1] - box1[..., 2:3] / 2
        box1_y1 = box1[..., 1:2] - box1[..., 3:4] / 2
        box1_x2 = box1[..., 0:1] + box1[..., 2:3] / 2
        box1_y2 = box1[..., 1:2] + box1[..., 3:4] / 2
        
        box2_x1 = box2[..., 0:1] - box2[..., 2:3] / 2
        box2_y1 = box2[..., 1:2] - box2[..., 3:4] / 2
        box2_x2 = box2[..., 0:1] + box2[..., 2:3] / 2
        box2_y2 = box2[..., 1:2] + box2[..., 3:4] / 2
        
        # Intersection
        inter_x1 = torch.max(box1_x1, box2_x1)
        inter_y1 = torch.max(box1_y1, box2_y1)
        inter_x2 = torch.min(box1_x2, box2_x2)
        inter_y2 = torch.min(box1_y2, box2_y2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                     torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Union
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        union_area = box1_area + box2_area - inter_area
        
        iou = inter_area / (union_area + 1e-6)
        return iou
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: [B, S, S, B, 5+C]
            targets: [B, S, S, B, 5+C] (same format, with responsibility already set)
        
        Returns:
            loss: scalar loss value
        """
        batch_size = predictions.size(0)
        
        # Extract components
        pred_xy = predictions[..., 0:2]      # [B, S, S, B, 2]
        pred_wh = predictions[..., 2:4]      # [B, S, S, B, 2]
        pred_conf = predictions[..., 4:5]    # [B, S, S, B, 1]
        pred_class = predictions[..., 5:]    # [B, S, S, B, C]
        
        target_xy = targets[..., 0:2]
        target_wh = targets[..., 2:4]
        target_conf = targets[..., 4:5]
        target_class = targets[..., 5:]
        
        # Responsibility mask (where target_conf > 0)
        obj_mask = target_conf > 0  # [B, S, S, B, 1]
        noobj_mask = target_conf == 0
        
        # 1. LOCALIZATION LOSS (only for responsible boxes)
        loss_xy = self.lambda_coord * torch.sum(
            obj_mask * (pred_xy - target_xy) ** 2
        )
        
        # Square root for w, h
        loss_wh = self.lambda_coord * torch.sum(
            obj_mask * (torch.sqrt(torch.abs(pred_wh) + 1e-6) - 
                       torch.sqrt(torch.abs(target_wh) + 1e-6)) ** 2
        )
        
        # 2. CONFIDENCE LOSS (object present)
        loss_conf_obj = torch.sum(
            obj_mask * (pred_conf - target_conf) ** 2
        )
        
        # 3. CONFIDENCE LOSS (no object)
        loss_conf_noobj = self.lambda_noobj * torch.sum(
            noobj_mask * (pred_conf - 0) ** 2
        )
        
        # 4. CLASSIFICATION LOSS
        # Sum over cells that have objects (any box responsible)
        cell_obj_mask = (target_conf.sum(dim=3, keepdim=True) > 0).float()  # [B, S, S, 1, 1]
        loss_class = torch.sum(
            cell_obj_mask * ((pred_class - target_class) ** 2).sum(dim=-1, keepdim=True)
        )
        
        # Total loss
        total_loss = loss_xy + loss_wh + loss_conf_obj + loss_conf_noobj + loss_class
        
        # Normalize by batch size
        total_loss = total_loss / batch_size
        
        return total_loss, {
            'loss_xy': loss_xy.item() / batch_size,
            'loss_wh': loss_wh.item() / batch_size,
            'loss_conf_obj': loss_conf_obj.item() / batch_size,
            'loss_conf_noobj': loss_conf_noobj.item() / batch_size,
            'loss_class': loss_class.item() / batch_size
        }


# ============================================================================
# SYNTHETIC DATASET
# ============================================================================

class SyntheticObjectDataset(Dataset):
    """
    Generates synthetic images with geometric shapes
    """
    
    def __init__(self, num_samples=1000, image_size=224, grid_size=7,
                 num_boxes=2, num_classes=3, max_objects=3):
        self.num_samples = num_samples
        self.image_size = image_size
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.max_objects = max_objects
        
        # Class names and colors
        self.class_names = ['circle', 'square', 'triangle']
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
            
            # Rasterize triangle
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
            
            # Normalized coordinates
            objects.append({
                'x': x / self.image_size,
                'y': y / self.image_size,
                'w': size / self.image_size,
                'h': size / self.image_size,
                'class': class_id
            })
        
        # Convert to tensor [3, H, W]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Create YOLO target tensor [S, S, B, 5+C]
        target = self.encode_target(objects)
        
        return image_tensor, target, objects
    
    def encode_target(self, objects):
        """
        Encode objects into YOLO target format [S, S, B, 5+C]
        Implements responsibility assignment from slides
        """
        S = self.grid_size
        B = self.num_boxes
        C = self.num_classes
        
        target = torch.zeros(S, S, B, 5 + C)
        
        for obj in objects:
            # Find which grid cell contains the object center
            grid_x = int(obj['x'] * S)
            grid_y = int(obj['y'] * S)
            
            # Clip to grid
            grid_x = min(grid_x, S - 1)
            grid_y = min(grid_y, S - 1)
            
            # Cell-relative coordinates
            x_cell = obj['x'] * S - grid_x
            y_cell = obj['y'] * S - grid_y
            
            # Assign to first box (simplified: in practice, use IoU)
            # Only first box is "responsible" for each object
            box_idx = 0
            
            # Set box parameters
            target[grid_y, grid_x, box_idx, 0] = x_cell
            target[grid_y, grid_x, box_idx, 1] = y_cell
            target[grid_y, grid_x, box_idx, 2] = obj['w']
            target[grid_y, grid_x, box_idx, 3] = obj['h']
            target[grid_y, grid_x, box_idx, 4] = 1.0  # confidence
            
            # Set class (one-hot)
            target[grid_y, grid_x, box_idx, 5 + obj['class']] = 1.0
        
        return target


# ============================================================================
# TRAINING
# ============================================================================

def train_yolo(num_epochs=20, batch_size=16, learning_rate=1e-3):
    """Train YOLO on synthetic dataset"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create dataset and dataloader
    dataset = SyntheticObjectDataset(num_samples=1000, grid_size=7, num_boxes=2)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    model = YOLODetector(num_classes=3, grid_size=7, num_boxes=2).to(device)
    
    # Loss and optimizer
    criterion = YOLOLoss(grid_size=7, num_boxes=2, num_classes=3)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("=" * 70)
    print("YOLO TRAINING")
    print("=" * 70)
    
    history = {'total_loss': [], 'xy_loss': [], 'wh_loss': [], 
               'conf_obj_loss': [], 'conf_noobj_loss': [], 'class_loss': []}
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        epoch_components = {'xy': [], 'wh': [], 'conf_obj': [], 'conf_noobj': [], 'class': []}
        
        for batch_idx, (images, targets, _) in enumerate(dataloader):
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            predictions = model(images)
            
            # Compute loss
            loss, loss_dict = criterion(predictions, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Record losses
            epoch_losses.append(loss.item())
            for key in ['loss_xy', 'loss_wh', 'loss_conf_obj', 'loss_conf_noobj', 'loss_class']:
                epoch_components[key.replace('loss_', '')].append(loss_dict[key])
        
        # Epoch summary
        avg_loss = np.mean(epoch_losses)
        history['total_loss'].append(avg_loss)
        for key in epoch_components:
            history[f'{key}_loss'].append(np.mean(epoch_components[key]))
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} " +
              f"| xy: {history['xy_loss'][-1]:.3f} " +
              f"| wh: {history['wh_loss'][-1]:.3f} " +
              f"| obj: {history['conf_obj_loss'][-1]:.3f} " +
              f"| noobj: {history['conf_noobj_loss'][-1]:.3f} " +
              f"| class: {history['class_loss'][-1]:.3f}")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    
    return model, history, dataset


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_predictions(model, dataset, num_samples=4):
    """Visualize YOLO predictions"""
    
    device = next(model.parameters()).device
    model.eval()
    
    fig, axes = plt.subplots(2, num_samples // 2, figsize=(15, 8))
    axes = axes.flatten()
    
    with torch.no_grad():
        for i in range(num_samples):
            image, target, gt_objects = dataset[i]
            
            # Get prediction
            pred = model(image.unsqueeze(0).to(device))
            pred = pred.squeeze(0).cpu()
            
            # Decode predictions
            detections = decode_predictions(pred, conf_threshold=0.3)
            
            # Plot
            ax = axes[i]
            img_np = image.permute(1, 2, 0).numpy()
            ax.imshow(img_np)
            
            # Draw ground truth (dashed)
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
            
            # Draw predictions (solid)
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


def decode_predictions(predictions, conf_threshold=0.3):
    """Decode YOLO predictions to bounding boxes"""
    S = predictions.shape[0]
    B = predictions.shape[2]
    
    detections = []
    
    for i in range(S):
        for j in range(S):
            for b in range(B):
                pred = predictions[i, j, b]
                
                x_cell, y_cell = pred[0].item(), pred[1].item()
                w, h = pred[2].item(), pred[3].item()
                conf = pred[4].item()
                class_probs = pred[5:]
                
                # Get class
                class_id = torch.argmax(class_probs).item()
                class_conf = class_probs[class_id].item()
                
                score = conf * class_conf
                
                if score < conf_threshold:
                    continue
                
                # Convert to absolute coordinates
                x = (j + x_cell) / S
                y = (i + y_cell) / S
                
                detections.append({
                    'x': x, 'y': y, 'w': abs(w), 'h': abs(h),
                    'conf': score, 'class': class_id
                })
    
    return detections


def plot_training_curves(history):
    """Plot training loss curves"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Total loss
    axes[0].plot(history['total_loss'], 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total YOLO Loss')
    axes[0].grid(True, alpha=0.3)
    
    # Component losses
    axes[1].plot(history['xy_loss'], label='Localization (xy)', linewidth=2)
    axes[1].plot(history['wh_loss'], label='Localization (wh)', linewidth=2)
    axes[1].plot(history['conf_obj_loss'], label='Confidence (obj)', linewidth=2)
    axes[1].plot(history['conf_noobj_loss'], label='Confidence (noobj)', linewidth=2)
    axes[1].plot(history['class_loss'], label='Classification', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('YOLO Loss Components')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
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
    print("Complete PyTorch Implementation with Synthetic Data")
    print("=" * 70 + "\n")
    
    # Train model
    model, history, dataset = train_yolo(num_epochs=20, batch_size=16, learning_rate=1e-3)
    
    # Plot training curves
    plot_training_curves(history)
    
    # Visualize predictions
    visualize_predictions(model, dataset, num_samples=4)
    
    print("\n" + "=" * 70)
    print("YOLO Training and Visualization Complete!")
    print("=" * 70)
    print("\nKey Components Demonstrated:")
    print("  ✓ Grid-based detection (S×S cells)")
    print("  ✓ Multi-task loss (localization + confidence + classification)")
    print("  ✓ Responsibility assignment (cell ownership)")
    print("  ✓ End-to-end training on synthetic shapes")
    print("  ✓ Prediction decoding and visualization")
    print("=" * 70 + "\n")