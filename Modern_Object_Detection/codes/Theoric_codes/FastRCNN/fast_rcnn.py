"""
Fast R-CNN: Efficient object detection with shared features
Reference: "Fast R-CNN" (Girshick, 2015)

Key Innovation: Share CNN computation across all proposals
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet50_Weights
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict


class RoIPooling(nn.Module):
    """
    Region of Interest Pooling Layer
    
    Converts variable-size RoIs to fixed-size feature maps
    """
    
    def __init__(self, output_size=(7, 7)):
        """
        Args:
            output_size: (height, width) of output feature map
        """
        super().__init__()
        self.output_size = output_size
    
    def forward(self, feature_map, rois, image_size):
        """
        Args:
            feature_map: [B, C, H, W] - shared CNN features
            rois: [N, 5] - [batch_idx, x1, y1, x2, y2] in image coordinates
            image_size: (H, W) - original image size
        
        Returns:
            pooled_features: [N, C, P, P] - RoI pooled features
        """
        batch_size, channels, feat_h, feat_w = feature_map.shape
        img_h, img_w = image_size
        
        # Calculate spatial scale
        spatial_scale_h = feat_h / img_h
        spatial_scale_w = feat_w / img_w
        
        num_rois = rois.size(0)
        output_h, output_w = self.output_size
        
        pooled_features = torch.zeros(
            num_rois, channels, output_h, output_w,
            device=feature_map.device
        )
        
        for roi_idx in range(num_rois):
            batch_idx = int(rois[roi_idx, 0])
            x1, y1, x2, y2 = rois[roi_idx, 1:].tolist()
            
            # Map to feature map coordinates
            x1_feat = int(x1 * spatial_scale_w)
            y1_feat = int(y1 * spatial_scale_h)
            x2_feat = int(x2 * spatial_scale_w)
            y2_feat = int(y2 * spatial_scale_h)
            
            # Handle edge cases
            x1_feat = max(0, x1_feat)
            y1_feat = max(0, y1_feat)
            x2_feat = min(feat_w, x2_feat)
            y2_feat = min(feat_h, y2_feat)
            
            roi_width = max(x2_feat - x1_feat, 1)
            roi_height = max(y2_feat - y1_feat, 1)
            
            # Compute bin size
            bin_h = roi_height / output_h
            bin_w = roi_width / output_w
            
            # Pool each bin
            for i in range(output_h):
                for j in range(output_w):
                    # Compute bin boundaries
                    y_start = int(y1_feat + i * bin_h)
                    y_end = int(y1_feat + (i + 1) * bin_h)
                    x_start = int(x1_feat + j * bin_w)
                    x_end = int(x1_feat + (j + 1) * bin_w)
                    
                    # Clip to feature map bounds
                    y_start = max(0, min(y_start, feat_h - 1))
                    y_end = max(0, min(y_end, feat_h))
                    x_start = max(0, min(x_start, feat_w - 1))
                    x_end = max(0, min(x_end, feat_w))
                    
                    # Max pooling over the bin
                    if y_end > y_start and x_end > x_start:
                        roi_bin = feature_map[batch_idx, :, y_start:y_end, x_start:x_end]
                        pooled_features[roi_idx, :, i, j] = F.adaptive_max_pool2d(
                            roi_bin.unsqueeze(0), (1, 1)
                        ).squeeze()
        
        return pooled_features


class FastRCNNHead(nn.Module):
    """
    Fast R-CNN detection head with multi-task loss
    """
    
    def __init__(self, in_features, num_classes):
        """
        Args:
            in_features: Input feature dimension
            num_classes: Number of object classes (+ background)
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # FC layers
        self.fc6 = nn.Linear(in_features, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        
        # Classification head
        self.cls_score = nn.Linear(4096, num_classes + 1)  # +1 for background
        
        # Bounding box regression head
        self.bbox_pred = nn.Linear(4096, 4 * (num_classes + 1))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in [self.fc6, self.fc7, self.cls_score, self.bbox_pred]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, roi_features):
        """
        Args:
            roi_features: [N, C*P*P] - flattened RoI pooled features
        
        Returns:
            cls_scores: [N, num_classes+1]
            bbox_deltas: [N, 4*(num_classes+1)]
        """
        x = F.relu(self.fc6(roi_features))
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = F.relu(self.fc7(x))
        x = F.dropout(x, p=0.5, training=self.training)
        
        cls_scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        
        return cls_scores, bbox_deltas


class FastRCNN(nn.Module):
    """
    Fast R-CNN: Efficient detection with shared CNN features
    """
    
    def __init__(self, num_classes=20, roi_output_size=(7, 7)):
        """
        Args:
            num_classes: Number of object classes
            roi_output_size: RoI pooling output size

            RESNET-50 ARCHITECTURE
            -----------------------
            Child	Type				Output shape (example for 224×224 input)   Purpose
            conv1	nn.Conv2d(3, 64, 7×7, stride=2)	[64, 112, 112]	Initial feature extractor
            bn1	nn.BatchNorm2d(64)		same	Normalization
            relu	nn.ReLU()			same	Activation
            maxpool	nn.MaxPool2d(3×3, stride=2)	[64, 56, 56]	Downsample
            layer1	nn.Sequential (3 bottleneck blocks)	[256, 56, 56]	Stage 1
            layer2	nn.Sequential (4 blocks)	[512, 28, 28]	Stage 2
            layer3	nn.Sequential (6 blocks)	[1024, 14, 14]	Stage 3
            layer4	nn.Sequential (3 blocks)	[2048, 7, 7]	Stage 4
            avgpool	nn.AdaptiveAvgPool2d(1)		[2048, 1, 1]	Global pooling (removed)
            fc	nn.Linear(2048, 1000)		[1000]	Classifier (removed)


        """
        super().__init__()
        
        self.num_classes = num_classes
        self.roi_output_size = roi_output_size
        
        # Shared CNN backbone (ResNet-50)
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Remove final pooling and fc layers
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Backbone output channels
        self.backbone_out_channels = 2048
        
        # RoI pooling
        self.roi_pool = RoIPooling(output_size=roi_output_size)
        
        # Detection head
        in_features = self.backbone_out_channels * roi_output_size[0] * roi_output_size[1]
        self.head = FastRCNNHead(in_features, num_classes)
        
        # Freeze early layers (optional)
        for param in list(self.backbone.parameters())[:30]:
            param.requires_grad = False
    
    def forward(self, images, proposals):
        """
        Args:
            images: [B, 3, H, W]
            proposals: [N, 5] - [batch_idx, x1, y1, x2, y2]
        
        Returns:
            cls_scores: [N, num_classes+1]
            bbox_deltas: [N, 4*(num_classes+1)]
        """
        # Shared CNN features (SINGLE forward pass!)
        features = self.backbone(images)
        
        # RoI pooling
        image_size = (images.size(2), images.size(3))
        roi_features = self.roi_pool(features, proposals, image_size)
        
        # Flatten
        roi_features = roi_features.view(roi_features.size(0), -1)
        
        # Detection head
        cls_scores, bbox_deltas = self.head(roi_features)
        
        return cls_scores, bbox_deltas


def smooth_l1_loss(pred, target, beta=1.0):
    """
    Smooth L1 loss (Huber loss)
    
    L(x) = 0.5 * x^2                  if |x| < beta
           |x| - 0.5 * beta           otherwise
    
    Args:
        pred: Predictions
        target: Targets
        beta: Threshold (default=1.0)
    """
    diff = torch.abs(pred - target)
    loss = torch.where(
        diff < beta,
        0.5 * diff ** 2 / beta,
        diff - 0.5 * beta
    )
    return loss.sum()


class FastRCNNLoss:
    """
    Multi-task loss for Fast R-CNN
    
    L = L_cls + λ * L_bbox
    """
    
    def __init__(self, num_classes, lambda_bbox=1.0):
        self.num_classes = num_classes
        self.lambda_bbox = lambda_bbox
    
    def __call__(self, cls_scores, bbox_deltas, labels, bbox_targets, bbox_inside_weights):
        """
        Args:
            cls_scores: [N, num_classes+1]
            bbox_deltas: [N, 4*(num_classes+1)]
            labels: [N] - class labels (0 = background)
            bbox_targets: [N, 4] - regression targets
            bbox_inside_weights: [N, 4] - weights (1 for foreground, 0 for background)
        """
        # Classification loss (cross-entropy)
        cls_loss = F.cross_entropy(cls_scores, labels)
        
        # Bounding box regression loss (only for foreground)
        # Select deltas for the true class
        batch_size = bbox_deltas.size(0)
        bbox_deltas = bbox_deltas.view(batch_size, -1, 4)
        
        # Gather deltas for ground truth classes
        bbox_deltas_selected = bbox_deltas[torch.arange(batch_size), labels]
        
        # Smooth L1 loss with inside weights
        bbox_loss = smooth_l1_loss(
            bbox_deltas_selected * bbox_inside_weights,
            bbox_targets * bbox_inside_weights
        ) / batch_size
        
        # Total loss
        total_loss = cls_loss + self.lambda_bbox * bbox_loss
        
        return total_loss, cls_loss, bbox_loss


class SimplifiedProposalGenerator:
    """Generate proposals for Fast R-CNN (simplified)"""
    
    def __init__(self, num_proposals=300):
        self.num_proposals = num_proposals
    
    def generate_proposals(self, image):
        """Generate random proposals"""
        h, w = image.shape[:2]
        proposals = []
        
        for _ in range(self.num_proposals):
            x1 = np.random.randint(0, w - 20)
            y1 = np.random.randint(0, h - 20)
            x2 = np.random.randint(x1 + 20, min(x1 + 100, w))
            y2 = np.random.randint(y1 + 20, min(y1 + 100, h))
            proposals.append([x1, y1, x2, y2])
        
        return proposals


class FastRCNNTrainer:
    """Training pipeline for Fast R-CNN"""
    
    def __init__(self, model, num_classes, device='cuda'):
        self.model = model.to(device)
        self.num_classes = num_classes
        self.device = device
        self.loss_fn = FastRCNNLoss(num_classes)
        self.optimizer = optim.SGD(
            model.parameters(), 
            lr=0.001, 
            momentum=0.9, 
            weight_decay=0.0005
        )
    
    def compute_iou(self, box1, box2):
        """Compute IoU between two boxes [x1, y1, x2, y2]"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def compute_bbox_targets(self, proposal, gt_box):
        """Compute bbox regression targets"""
        px1, py1, px2, py2 = proposal
        gx1, gy1, gx2, gy2 = gt_box
        
        pw = px2 - px1
        ph = py2 - py1
        px = px1 + 0.5 * pw
        py = py1 + 0.5 * ph
        
        gw = gx2 - gx1
        gh = gy2 - gy1
        gx = gx1 + 0.5 * gw
        gy = gy1 + 0.5 * gh
        
        tx = (gx - px) / pw
        ty = (gy - py) / ph
        tw = np.log(gw / pw)
        th = np.log(gh / ph)
        
        return np.array([tx, ty, tw, th], dtype=np.float32)
    
    def prepare_training_data(self, images, annotations, proposals_list):
        """Prepare training batch"""
        all_rois = []
        all_labels = []
        all_bbox_targets = []
        all_bbox_weights = []
        
        for batch_idx, (proposals, annot) in enumerate(zip(proposals_list, annotations)):
            gt_boxes = annot['boxes']
            gt_labels = annot['labels']
            
            # Convert to [x1, y1, x2, y2] format
            gt_boxes_xyxy = []
            for box in gt_boxes:
                x, y, w, h = box
                gt_boxes_xyxy.append([x, y, x + w, y + h])
            
            for proposal in proposals:
                # Find best matching GT
                max_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt_box in enumerate(gt_boxes_xyxy):
                    iou = self.compute_iou(proposal, gt_box)
                    if iou > max_iou:
                        max_iou = iou
                        best_gt_idx = gt_idx
                
                # Label assignment
                if max_iou >= 0.5:
                    # Foreground
                    label = gt_labels[best_gt_idx] + 1  # +1 because 0 is background
                    bbox_target = self.compute_bbox_targets(proposal, gt_boxes_xyxy[best_gt_idx])
                    bbox_weight = np.ones(4, dtype=np.float32)
                else:
                    # Background
                    label = 0
                    bbox_target = np.zeros(4, dtype=np.float32)
                    bbox_weight = np.zeros(4, dtype=np.float32)
                
                all_rois.append([batch_idx] + proposal)
                all_labels.append(label)
                all_bbox_targets.append(bbox_target)
                all_bbox_weights.append(bbox_weight)
        
        return (
            torch.tensor(all_rois, dtype=torch.float32),
            torch.tensor(all_labels, dtype=torch.long),
            torch.tensor(all_bbox_targets, dtype=torch.float32),
            torch.tensor(all_bbox_weights, dtype=torch.float32)
        )
    
    def train_epoch(self, images, annotations):
        """Train one epoch"""
        self.model.train()
        
        # Generate proposals
        proposal_gen = SimplifiedProposalGenerator(num_proposals=200)
        proposals_list = [proposal_gen.generate_proposals(img) for img in images]
        
        # Prepare batch
        images_tensor = torch.stack([
            torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            for img in images
        ]).to(self.device)
        
        rois, labels, bbox_targets, bbox_weights = self.prepare_training_data(
            images, annotations, proposals_list
        )
        rois = rois.to(self.device)
        labels = labels.to(self.device)
        bbox_targets = bbox_targets.to(self.device)
        bbox_weights = bbox_weights.to(self.device)
        
        # Forward pass
        cls_scores, bbox_deltas = self.model(images_tensor, rois)
        
        # Compute loss
        total_loss, cls_loss, bbox_loss = self.loss_fn(
            cls_scores, bbox_deltas, labels, bbox_targets, bbox_weights
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item(), cls_loss.item(), bbox_loss.item()


def create_synthetic_dataset(num_images=5):
    """Create synthetic dataset for demo"""
    images = []
    annotations = []
    
    for _ in range(num_images):
        # Create image
        img = np.ones((224, 224, 3), dtype=np.uint8) * 255
        
        boxes = []
        labels = []
        
        # Add 2-3 objects
        num_objects = np.random.randint(2, 4)
        for i in range(num_objects):
            w = np.random.randint(30, 80)
            h = np.random.randint(30, 80)
            x = np.random.randint(0, 224 - w)
            y = np.random.randint(0, 224 - h)
            
            color = tuple(np.random.randint(50, 200, 3).tolist())
            cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
            
            boxes.append([x, y, w, h])
            labels.append(i % 3)
        
        images.append(img)
        annotations.append({'boxes': boxes, 'labels': labels})
    
    return images, annotations


def demo_fast_rcnn():
    """Run Fast R-CNN demo"""
    print("="*70)
    print("Fast R-CNN Demo: Shared CNN Features for Efficient Detection")
    print("="*70)
    
    # Create dataset
    print("\n1. Creating synthetic dataset...")
    images, annotations = create_synthetic_dataset(num_images=5)
    print(f"   Created {len(images)} training images")
    
    # Initialize model
    print("\n2. Initializing Fast R-CNN...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FastRCNN(num_classes=3)
    trainer = FastRCNNTrainer(model, num_classes=3, device=device)
    print(f"   Device: {device}")
    
    # Training
    print("\n3. Training Fast R-CNN...")
    print("   " + "-"*60)
    
    start_time = time.time()
    for epoch in range(10):
        total_loss, cls_loss, bbox_loss = trainer.train_epoch(images, annotations)
        print(f"   Epoch {epoch+1:2d} | Loss: {total_loss:.4f} "
              f"(cls: {cls_loss:.4f}, bbox: {bbox_loss:.4f})")
    
    train_time = time.time() - start_time
    print(f"\n   Training completed in {train_time:.2f} seconds")
    
    # Comparison with R-CNN
    print("\n4. Computational Comparison:")
    print("   " + "-"*60)
    print("   R-CNN:      2000 proposals × 5 images = 10,000 CNN passes")
    print("   Fast R-CNN: 1 CNN pass × 5 images = 5 CNN passes")
    print(f"   Speed-up:   ~{10000/5:.0f}× fewer CNN forward passes!")
    
    # Test inference
    print("\n5. Testing inference...")
    model.eval()
    test_img = images[0]
    
    # Generate proposals
    proposal_gen = SimplifiedProposalGenerator(num_proposals=100)
    proposals = proposal_gen.generate_proposals(test_img)
    
    # Prepare for inference
    img_tensor = torch.from_numpy(test_img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)
    
    rois = torch.tensor([[0] + prop for prop in proposals], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        cls_scores, bbox_deltas = model(img_tensor, rois)
        probs = F.softmax(cls_scores, dim=1)
        max_probs, predicted_classes = probs.max(dim=1)
    
    # Filter detections
    detections = []
    for i, (prob, cls) in enumerate(zip(max_probs, predicted_classes)):
        if prob > 0.5 and cls > 0:  # Not background
            detections.append((proposals[i], cls.item() - 1, prob.item()))
    
    print(f"   Found {len(detections)} detections (threshold=0.5)")
    
    print("\n" + "="*70)
    print("Fast R-CNN Demo Complete!")
    print("="*70)
    
    return model, images, annotations


if __name__ == "__main__":
    demo_fast_rcnn()