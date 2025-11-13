"""
DETR (Detection Transformer) Complete Implementation
End-to-end object detection with transformers and Hungarian matching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict


# ============================================================================
# POSITIONAL ENCODING
# ============================================================================

class PositionalEncoding2D(nn.Module):
    """2D Positional Encoding for image features"""
    
    def __init__(self, d_model, max_h=100, max_w=100):
        super().__init__()
        
        # Create position encodings
        pe = torch.zeros(d_model, max_h, max_w)
        
        d_model_half = d_model // 2
        div_term = torch.exp(torch.arange(0, d_model_half, 2).float() * 
                            (-np.log(10000.0) / d_model_half))
        
        # Height encodings
        pos_h = torch.arange(0, max_h).unsqueeze(1)
        pe[0:d_model_half:2, :, :] = torch.sin(pos_h * div_term).unsqueeze(2).repeat(1, 1, max_w)
        pe[1:d_model_half:2, :, :] = torch.cos(pos_h * div_term).unsqueeze(2).repeat(1, 1, max_w)
        
        # Width encodings
        pos_w = torch.arange(0, max_w).unsqueeze(1)
        pe[d_model_half::2, :, :] = torch.sin(pos_w * div_term).unsqueeze(1).repeat(1, max_h, 1)
        pe[d_model_half+1::2, :, :] = torch.cos(pos_w * div_term).unsqueeze(1).repeat(1, max_h, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, C, H, W] with positional encoding added
        """
        return x + self.pe[:, :x.size(2), :x.size(3)].unsqueeze(0)


# ============================================================================
# TRANSFORMER
# ============================================================================

class TransformerEncoder(nn.Module):
    """Transformer encoder for image features"""
    
    def __init__(self, d_model=256, nhead=8, num_layers=6, dim_feedforward=2048):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, src):
        """
        Args:
            src: [B, HW, C] flattened features
        Returns:
            [B, HW, C] encoded features
        """
        return self.encoder(src)


class TransformerDecoder(nn.Module):
    """Transformer decoder with object queries"""
    
    def __init__(self, d_model=256, nhead=8, num_layers=6, dim_feedforward=2048):
        super().__init__()
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
    
    def forward(self, tgt, memory):
        """
        Args:
            tgt: [B, N, C] object queries
            memory: [B, HW, C] encoder output
        Returns:
            [B, N, C] decoded queries
        """
        return self.decoder(tgt, memory)


# ============================================================================
# DETR MODEL
# ============================================================================

class SimpleBackbone(nn.Module):
    """Simple CNN backbone"""
    
    def __init__(self, d_model=256):
        super().__init__()
        
        self.features = nn.Sequential(
            # 224 -> 112
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # 112 -> 56
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 56 -> 28
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Project to d_model
        self.conv_proj = nn.Conv2d(256, d_model, kernel_size=1)
    
    def forward(self, x):
        x = self.features(x)
        x = self.conv_proj(x)
        return x


class DETR(nn.Module):
    """
    DETR: End-to-End Object Detection with Transformers
    Reference: "End-to-End Object Detection with Transformers" (Carion et al., 2020)
    """
    
    def __init__(self, num_classes=3, num_queries=100, d_model=256, 
                 nhead=8, num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_queries = num_queries
        
        # Backbone
        self.backbone = SimpleBackbone(d_model=d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding2D(d_model)
        
        # Transformer
        self.transformer_encoder = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=2048
        )
        
        self.transformer_decoder = TransformerDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=2048
        )
        
        # Learnable object queries
        self.query_embed = nn.Embedding(num_queries, d_model)
        
        # Prediction heads
        self.class_embed = nn.Linear(d_model, num_classes + 1)  # +1 for "no object"
        self.bbox_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] input images
        
        Returns:
            pred_logits: [B, num_queries, num_classes+1]
            pred_boxes: [B, num_queries, 4]
        """
        B = x.size(0)
        
        # Backbone features
        features = self.backbone(x)  # [B, d_model, H', W']
        
        # Add positional encoding
        features = self.pos_encoding(features)  # [B, d_model, H', W']
        
        # Flatten for transformer
        B, C, H, W = features.shape
        features_flat = features.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        
        # Transformer encoder
        memory = self.transformer_encoder(features_flat)  # [B, HW, C]
        
        # Object queries
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, N, C]
        
        # Transformer decoder
        hs = self.transformer_decoder(query_embed, memory)  # [B, N, C]
        
        # Prediction heads
        pred_logits = self.class_embed(hs)  # [B, N, num_classes+1]
        pred_boxes = self.bbox_embed(hs).sigmoid()  # [B, N, 4] in [0,1]
        
        return pred_logits, pred_boxes


# ============================================================================
# HUNGARIAN MATCHING
# ============================================================================

class HungarianMatcher:
    """
    Hungarian matching algorithm for optimal assignment
    """
    
    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
    
    def compute_giou(self, boxes1, boxes2):
        """Compute Generalized IoU"""
        # boxes format: [x_center, y_center, width, height]
        
        # Convert to [x1, y1, x2, y2]
        boxes1_xyxy = torch.zeros_like(boxes1)
        boxes1_xyxy[:, 0] = boxes1[:, 0] - boxes1[:, 2] / 2
        boxes1_xyxy[:, 1] = boxes1[:, 1] - boxes1[:, 3] / 2
        boxes1_xyxy[:, 2] = boxes1[:, 0] + boxes1[:, 2] / 2
        boxes1_xyxy[:, 3] = boxes1[:, 1] + boxes1[:, 3] / 2
        
        boxes2_xyxy = torch.zeros_like(boxes2)
        boxes2_xyxy[:, 0] = boxes2[:, 0] - boxes2[:, 2] / 2
        boxes2_xyxy[:, 1] = boxes2[:, 1] - boxes2[:, 3] / 2
        boxes2_xyxy[:, 2] = boxes2[:, 0] + boxes2[:, 2] / 2
        boxes2_xyxy[:, 3] = boxes2[:, 1] + boxes2[:, 3] / 2
        
        # Intersection
        x1 = torch.max(boxes1_xyxy[:, None, 0], boxes2_xyxy[:, 0])
        y1 = torch.max(boxes1_xyxy[:, None, 1], boxes2_xyxy[:, 1])
        x2 = torch.min(boxes1_xyxy[:, None, 2], boxes2_xyxy[:, 2])
        y2 = torch.min(boxes1_xyxy[:, None, 3], boxes2_xyxy[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Areas
        area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]) * (boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1])
        area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1])
        
        union = area1[:, None] + area2 - intersection
        iou = intersection / (union + 1e-6)
        
        # Enclosing box
        x1_c = torch.min(boxes1_xyxy[:, None, 0], boxes2_xyxy[:, 0])
        y1_c = torch.min(boxes1_xyxy[:, None, 1], boxes2_xyxy[:, 1])
        x2_c = torch.max(boxes1_xyxy[:, None, 2], boxes2_xyxy[:, 2])
        y2_c = torch.max(boxes1_xyxy[:, None, 3], boxes2_xyxy[:, 3])
        
        area_c = (x2_c - x1_c) * (y2_c - y1_c)
        
        giou = iou - (area_c - union) / (area_c + 1e-6)
        
        return giou
    
    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict with 'pred_logits' [B, N, C+1] and 'pred_boxes' [B, N, 4]
            targets: list of dicts with 'labels' and 'boxes'
        
        Returns:
            List of (pred_idx, tgt_idx) tuples for each batch
        """
        B, N = outputs['pred_logits'].shape[:2]
        
        # Flatten batch dimension
        pred_logits = outputs['pred_logits'].flatten(0, 1)  # [B*N, C+1]
        pred_boxes = outputs['pred_boxes'].flatten(0, 1)  # [B*N, 4]
        
        # Concatenate all target labels and boxes
        tgt_labels = torch.cat([t['labels'] for t in targets])
        tgt_boxes = torch.cat([t['boxes'] for t in targets])
        
        # Classification cost (negative log probability)
        pred_probs = pred_logits.softmax(-1)
        cost_class = -pred_probs[:, tgt_labels]
        
        # L1 cost
        cost_bbox = torch.cdist(pred_boxes, tgt_boxes, p=1)
        
        # GIoU cost
        cost_giou = -self.compute_giou(pred_boxes, tgt_boxes)
        
        # Final cost matrix
        C = self.cost_class * cost_class + \
            self.cost_bbox * cost_bbox + \
            self.cost_giou * cost_giou
        
        C = C.view(B, N, -1).cpu()
        
        # Hungarian matching for each batch
        indices = []
        sizes = [len(t['labels']) for t in targets]
        
        start_idx = 0
        for i, size in enumerate(sizes):
            cost_matrix = C[i, :, start_idx:start_idx + size]
            pred_idx, tgt_idx = linear_sum_assignment(cost_matrix)
            indices.append((pred_idx, tgt_idx))
            start_idx += size
        
        return indices


# ============================================================================
# DETR LOSS
# ============================================================================

class DETRLoss(nn.Module):
    """
    DETR set-based loss with Hungarian matching
    """
    
    def __init__(self, num_classes=3, weight_class=1.0, weight_bbox=5.0, weight_giou=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.weight_class = weight_class
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou
        
        self.matcher = HungarianMatcher(
            cost_class=weight_class,
            cost_bbox=weight_bbox,
            cost_giou=weight_giou
        )
    
    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict with 'pred_logits' and 'pred_boxes'
            targets: list of dicts with 'labels' and 'boxes'
        """
        # Hungarian matching
        indices = self.matcher.forward(outputs, targets)
        
        # Get matched predictions and targets
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        
        B, N = pred_logits.shape[:2]
        
        # Prepare targets
        target_classes = torch.full((B, N), self.num_classes, 
                                   dtype=torch.long, device=pred_logits.device)
        target_boxes = torch.zeros((B, N, 4), device=pred_boxes.device)
        
        for batch_idx, (pred_idx, tgt_idx) in enumerate(indices):
            target_classes[batch_idx, pred_idx] = targets[batch_idx]['labels'][tgt_idx]
            target_boxes[batch_idx, pred_idx] = targets[batch_idx]['boxes'][tgt_idx]
        
        # Classification loss (cross entropy)
        loss_class = F.cross_entropy(
            pred_logits.flatten(0, 1),
            target_classes.flatten(0, 1),
            reduction='mean'
        )
        
        # Box loss (only for matched objects)
        num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = max(num_boxes, 1)
        
        obj_mask = target_classes != self.num_classes
        
        if obj_mask.sum() > 0:
            # L1 loss
            loss_bbox = F.l1_loss(
                pred_boxes[obj_mask],
                target_boxes[obj_mask],
                reduction='sum'
            ) / num_boxes
            
            # GIoU loss
            giou = self.matcher.compute_giou(
                pred_boxes[obj_mask],
                target_boxes[obj_mask]
            )
            loss_giou = (1 - giou.diagonal()).sum() / num_boxes
        else:
            loss_bbox = torch.tensor(0.0, device=pred_boxes.device)
            loss_giou = torch.tensor(0.0, device=pred_boxes.device)
        
        # Total loss
        total_loss = (self.weight_class * loss_class + 
                     self.weight_bbox * loss_bbox + 
                     self.weight_giou * loss_giou)
        
        return total_loss, {
            'loss_class': loss_class.item(),
            'loss_bbox': loss_bbox.item(),
            'loss_giou': loss_giou.item()
        }


# ============================================================================
# SYNTHETIC DATASET
# ============================================================================

class SyntheticObjectDataset(Dataset):
    """Generates synthetic images for DETR"""
    
    def __init__(self, num_samples=1000, image_size=224, max_objects=5):
        self.num_samples = num_samples
        self.image_size = image_size
        self.max_objects = max_objects
        
        self.class_names = ['circle', 'square', 'triangle']
        self.num_classes = len(self.class_names)
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
        image = np.ones((self.image_size, self.image_size, 3), dtype=np.float32) * 0.95
        
        num_objects = np.random.randint(1, self.max_objects + 1)
        objects = []
        
        for _ in range(num_objects):
            class_id = np.random.randint(0, self.num_classes)
            size = np.random.randint(30, 80)
            x = np.random.randint(size, self.image_size - size)
            y = np.random.randint(size, self.image_size - size)
            
            self.draw_shape(image, x, y, size, class_id)
            
            # Normalized center coordinates and size
            objects.append({
                'x': x / self.image_size,
                'y': y / self.image_size,
                'w': size / self.image_size,
                'h': size / self.image_size,
                'class': class_id
            })
        
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # DETR format: list of boxes [x_center, y_center, width, height]
        labels = torch.tensor([obj['class'] for obj in objects], dtype=torch.long)
        boxes = torch.tensor([[obj['x'], obj['y'], obj['w'], obj['h']] for obj in objects], 
                            dtype=torch.float32)
        
        target = {'labels': labels, 'boxes': boxes}
        
        return image_tensor, target, objects


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    objects = [item[2] for item in batch]
    return images, targets, objects


# ============================================================================
# TRAINING
# ============================================================================

def train_detr(num_epochs=50, batch_size=8, learning_rate=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Dataset
    dataset = SyntheticObjectDataset(num_samples=1000, image_size=224)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           collate_fn=collate_fn)
    
    # Model
    model = DETR(num_classes=3, num_queries=100, d_model=256, 
                 nhead=8, num_encoder_layers=3, num_decoder_layers=3).to(device)
    
    # Loss and optimizer
    criterion = DETRLoss(num_classes=3)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    print("=" * 70)
    print("DETR TRAINING - Transformer-Based Object Detection")
    print("=" * 70)
    
    history = {'total_loss': [], 'class_loss': [], 'bbox_loss': [], 'giou_loss': []}
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        epoch_components = {'class': [], 'bbox': [], 'giou': []}
        
        for batch_idx, (images, targets, _) in enumerate(dataloader):
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward
            pred_logits, pred_boxes = model(images)
            outputs = {'pred_logits': pred_logits, 'pred_boxes': pred_boxes}
            
            # Loss
            loss, loss_dict = criterion(outputs, targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            epoch_components['class'].append(loss_dict['loss_class'])
            epoch_components['bbox'].append(loss_dict['loss_bbox'])
            epoch_components['giou'].append(loss_dict['loss_giou'])
        
        avg_loss = np.mean(epoch_losses)
        history['total_loss'].append(avg_loss)
        history['class_loss'].append(np.mean(epoch_components['class']))
        history['bbox_loss'].append(np.mean(epoch_components['bbox']))
        history['giou_loss'].append(np.mean(epoch_components['giou']))
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} " +
              f"| cls: {history['class_loss'][-1]:.3f} " +
              f"| bbox: {history['bbox_loss'][-1]:.3f} " +
              f"| giou: {history['giou_loss'][-1]:.3f}")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    
    return model, history, dataset


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_predictions(model, dataset, num_samples=4):
    device = next(model.parameters()).device
    model.eval()
    
    fig, axes = plt.subplots(2, num_samples // 2, figsize=(15, 8))
    axes = axes.flatten()
    
    with torch.no_grad():
        for i in range(num_samples):
            image, target, gt_objects = dataset[i]
            
            # Predict
            pred_logits, pred_boxes = model(image.unsqueeze(0).to(device))
            
            # Get predictions (top-k by confidence)
            pred_logits = pred_logits[0].cpu()
            pred_boxes = pred_boxes[0].cpu()
            
            probs = F.softmax(pred_logits, dim=-1)
            scores, labels = probs[:, :-1].max(dim=-1)  # Exclude "no object" class
            
            # Filter by confidence
            keep = scores > 0.7
            pred_boxes_filtered = pred_boxes[keep]
            scores_filtered = scores[keep]
            labels_filtered = labels[keep]
            
            # Plot
            ax = axes[i]
            img_np = image.permute(1, 2, 0).numpy()
            ax.imshow(img_np)
            
            # Ground truth (dashed)
            for obj in gt_objects:
                x_c, y_c, w, h = obj['x'], obj['y'], obj['w'], obj['h']
                x1 = (x_c - w/2) * dataset.image_size
                y1 = (y_c - h/2) * dataset.image_size
                w_px = w * dataset.image_size
                h_px = h * dataset.image_size
                
                color = dataset.class_colors[obj['class']]
                rect = patches.Rectangle((x1, y1), w_px, h_px, linewidth=2,
                                        edgecolor=color, facecolor='none', linestyle='--')
                ax.add_patch(rect)
                ax.text(x1, y1 - 5, f"GT: {dataset.class_names[obj['class']]}",
                       color=color, fontsize=8, fontweight='bold')
            
            # Predictions (solid)
            for box, score, label in zip(pred_boxes_filtered, scores_filtered, labels_filtered):
                x_c, y_c, w, h = box
                x1 = (x_c - w/2) * dataset.image_size
                y1 = (y_c - h/2) * dataset.image_size
                w_px = w * dataset.image_size
                h_px = h * dataset.image_size
                
                color = dataset.class_colors[label.item()]
                rect = patches.Rectangle((x1, y1), w_px, h_px, linewidth=2,
                                        edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1 + h_px + 15,
                       f"{dataset.class_names[label.item()]} {score:.2f}",
                       color=color, fontsize=8, fontweight='bold')
            
            ax.set_title(f"Sample {i+1}")
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('detr_predictions.png', dpi=150, bbox_inches='tight')
    print("\nPredictions saved to 'detr_predictions.png'")
    plt.show()


def plot_training_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history['total_loss'], 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total DETR Loss')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['class_loss'], label='Classification', linewidth=2)
    axes[1].plot(history['bbox_loss'], label='L1 Box', linewidth=2)
    axes[1].plot(history['giou_loss'], label='GIoU', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('DETR Loss Components')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('detr_training_curves.png', dpi=150, bbox_inches='tight')
    print("Training curves saved to 'detr_training_curves.png'")
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("DETR: DETECTION TRANSFORMER")
    print("End-to-End Object Detection with Transformers")
    print("=" * 70 + "\n")
    
    model, history, dataset = train_detr(num_epochs=50, batch_size=8, learning_rate=1e-4)
    
    plot_training_curves(history)
    visualize_predictions(model, dataset, num_samples=4)
    
    print("\n" + "=" * 70)
    print("DETR Training Complete!")
    print("=" * 70)
    print("\nKey DETR Innovations:")
    print("  ✓ Transformer encoder-decoder architecture")
    print("  ✓ Learnable object queries (100 queries)")
    print("  ✓ Hungarian matching for bipartite assignment")
    print("  ✓ Set-based loss (no NMS needed!)")
    print("  ✓ End-to-end training")
    print("\nDETR vs Previous Methods:")
    print("  • No anchors (like FCOS)")
    print("  • No NMS post-processing")
    print("  • No hand-crafted components")
    print("  • Direct set prediction")
    print("  • Global reasoning with self-attention")
    print("=" * 70 + "\n")