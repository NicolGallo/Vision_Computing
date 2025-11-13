"""
Region Proposal Network (RPN) and Faster R-CNN
The final piece: Learned proposals replace Selective Search
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import List, Tuple


class AnchorGeneratorTorch(nn.Module):
    """PyTorch anchor generator for RPN"""
    
    def __init__(self, scales=[8, 16, 32], aspect_ratios=[0.5, 1.0, 2.0], stride=16):
        super().__init__()
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.stride = stride
        self.num_anchors = len(scales) * len(aspect_ratios)
        
        # Pre-compute base anchors
        self.register_buffer('base_anchors', self._generate_base_anchors())
    
    def _generate_base_anchors(self):
        """Generate base anchor templates"""
        base_anchors = []
        for scale in self.scales:
            for ratio in self.aspect_ratios:
                area = scale ** 2
                w = np.sqrt(area / ratio)
                h = w * ratio
                base_anchors.append([-w/2, -h/2, w/2, h/2])
        return torch.tensor(base_anchors, dtype=torch.float32)
    
    def forward(self, feature_map, image_size):
        """
        Generate anchors for a feature map
        
        Args:
            feature_map: [B, C, H, W]
            image_size: (H_img, W_img)
        
        Returns:
            anchors: [H*W*num_anchors, 4] in (x1, y1, x2, y2) format
        """
        B, C, H, W = feature_map.shape
        device = feature_map.device
        
        # Create grid of anchor centers
        shifts_x = torch.arange(0, W, device=device) * self.stride + self.stride / 2
        shifts_y = torch.arange(0, H, device=device) * self.stride + self.stride / 2
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        shifts = torch.stack([shift_x.reshape(-1), shift_y.reshape(-1),
                             shift_x.reshape(-1), shift_y.reshape(-1)], dim=1)
        
        # Shift base anchors to all locations
        anchors = (shifts.view(-1, 1, 4) + self.base_anchors.view(1, -1, 4).to(device)).reshape(-1, 4)
        
        # Clip to image boundaries
        anchors[:, 0].clamp_(min=0, max=image_size[1])
        anchors[:, 1].clamp_(min=0, max=image_size[0])
        anchors[:, 2].clamp_(min=0, max=image_size[1])
        anchors[:, 3].clamp_(min=0, max=image_size[0])
        
        return anchors


class RPN(nn.Module):
    """
    Region Proposal Network
    
    Takes shared CNN features and predicts:
    1. Objectness (is there an object?)
    2. Box refinement (how to adjust anchor?)
    """
    
    def __init__(self, in_channels=512, num_anchors=9):
        """
        Args:
            in_channels: Number of input channels from backbone
            num_anchors: Number of anchors per location
        """
        super().__init__()
        
        self.num_anchors = num_anchors
        
        # 3x3 conv for feature transformation
        self.conv = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        
        # Objectness classification (object vs background)
        self.cls_logits = nn.Conv2d(512, num_anchors, kernel_size=1)
        
        # Bounding box regression (4 values per anchor)
        self.bbox_pred = nn.Conv2d(512, num_anchors * 4, kernel_size=1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for layer in [self.conv, self.cls_logits, self.bbox_pred]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)
    
    def forward(self, features):
        """
        Forward pass
        
        Args:
            features: [B, C, H, W] - shared CNN features
        
        Returns:
            objectness: [B, H, W, num_anchors] - object vs background scores
            bbox_deltas: [B, H, W, num_anchors*4] - box refinements
        """
        # Shared 3x3 conv
        x = F.relu(self.conv(features))
        
        # Objectness scores
        objectness = self.cls_logits(x)  # [B, num_anchors, H, W]
        objectness = objectness.permute(0, 2, 3, 1)  # [B, H, W, num_anchors]
        
        # Bbox deltas
        bbox_deltas = self.bbox_pred(x)  # [B, num_anchors*4, H, W]
        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1)  # [B, H, W, num_anchors*4]
        
        return objectness, bbox_deltas


class RPNLoss:
    """RPN loss computation"""
    
    def __init__(self, pos_iou_threshold=0.7, neg_iou_threshold=0.3,
                 batch_size=256, pos_fraction=0.5):
        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_threshold = neg_iou_threshold
        self.batch_size = batch_size
        self.pos_fraction = pos_fraction
    
    def compute_iou(self, boxes1, boxes2):
        """Compute IoU between two sets of boxes"""
        # boxes1: [N, 4], boxes2: [M, 4]
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
        
        wh = (rb - lt).clamp(min=0)  # [N, M, 2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
        
        union = area1[:, None] + area2 - inter
        iou = inter / union
        return iou
    
    def assign_anchors(self, anchors, gt_boxes):
        """
        Assign anchors to ground truth boxes
        
        Returns:
            labels: [N] - 1=positive, 0=negative, -1=ignore
            bbox_targets: [N, 4] - regression targets for positive anchors
        """
        N = len(anchors)
        labels = torch.full((N,), -1, dtype=torch.long, device=anchors.device)
        bbox_targets = torch.zeros((N, 4), dtype=torch.float32, device=anchors.device)
        
        if len(gt_boxes) == 0:
            return labels, bbox_targets
        
        # Compute IoU
        iou = self.compute_iou(anchors, gt_boxes)  # [N, M]
        max_iou, max_idx = iou.max(dim=1)
        
        # Assign labels
        labels[max_iou < self.neg_iou_threshold] = 0  # Negative
        labels[max_iou >= self.pos_iou_threshold] = 1  # Positive
        
        # Compute regression targets for positive anchors
        pos_mask = labels == 1
        if pos_mask.sum() > 0:
            matched_gt = gt_boxes[max_idx[pos_mask]]
            bbox_targets[pos_mask] = self.compute_bbox_targets(
                anchors[pos_mask], matched_gt
            )
        
        return labels, bbox_targets
    
    def compute_bbox_targets(self, anchors, gt_boxes):
        """Compute bbox regression targets"""
        # anchors: [N, 4], gt_boxes: [N, 4]
        ax1, ay1, ax2, ay2 = anchors.unbind(dim=1)
        gx1, gy1, gx2, gy2 = gt_boxes.unbind(dim=1)
        
        aw = ax2 - ax1
        ah = ay2 - ay1
        ax = ax1 + 0.5 * aw
        ay = ay1 + 0.5 * ah
        
        gw = gx2 - gx1
        gh = gy2 - gy1
        gx = gx1 + 0.5 * gw
        gy = gy1 + 0.5 * gh
        
        tx = (gx - ax) / aw
        ty = (gy - ay) / ah
        tw = torch.log(gw / aw)
        th = torch.log(gh / ah)
        
        return torch.stack([tx, ty, tw, th], dim=1)
    
    def __call__(self, objectness, bbox_deltas, anchors, gt_boxes_list):
        """
        Compute RPN loss
        
        Args:
            objectness: [B, H, W, num_anchors]
            bbox_deltas: [B, H, W, num_anchors*4]
            anchors: [H*W*num_anchors, 4]
            gt_boxes_list: List of [M, 4] tensors (one per image)
        """
        B = objectness.size(0)
        device = objectness.device
        
        # Flatten predictions
        objectness = objectness.reshape(B, -1)  # [B, N]
        bbox_deltas = bbox_deltas.reshape(B, -1, 4)  # [B, N, 4]
        
        total_cls_loss = 0
        total_reg_loss = 0
        
        for b in range(B):
            # Assign anchors
            labels, bbox_targets = self.assign_anchors(anchors, gt_boxes_list[b])
            
            # Sample mini-batch
            pos_idx = torch.where(labels == 1)[0]
            neg_idx = torch.where(labels == 0)[0]
            
            num_pos = min(len(pos_idx), int(self.batch_size * self.pos_fraction))
            num_neg = min(len(neg_idx), self.batch_size - num_pos)
            
            if len(pos_idx) > num_pos:
                pos_idx = pos_idx[torch.randperm(len(pos_idx))[:num_pos]]
            if len(neg_idx) > num_neg:
                neg_idx = neg_idx[torch.randperm(len(neg_idx))[:num_neg]]
            
            sampled_idx = torch.cat([pos_idx, neg_idx])
            
            # Classification loss
            cls_loss = F.binary_cross_entropy_with_logits(
                objectness[b, sampled_idx],
                labels[sampled_idx].float()
            )
            total_cls_loss += cls_loss
            
            # Regression loss (only for positives)
            if num_pos > 0:
                reg_loss = F.smooth_l1_loss(
                    bbox_deltas[b, pos_idx],
                    bbox_targets[pos_idx]
                )
                total_reg_loss += reg_loss
        
        return total_cls_loss / B, total_reg_loss / B


class FasterRCNN(nn.Module):
    """
    Faster R-CNN: Fast R-CNN + RPN
    
    End-to-end trainable object detector
    """
    
    def __init__(self, num_classes=20, backbone_channels=512):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Simplified backbone (in practice, use ResNet)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, backbone_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Anchor generator
        self.anchor_generator = AnchorGeneratorTorch(
            scales=[8, 16, 32],
            aspect_ratios=[0.5, 1.0, 2.0],
            stride=16
        )
        
        # RPN
        self.rpn = RPN(in_channels=backbone_channels, num_anchors=9)
        
        # Detection head (simplified - would use RoI pooling + FC in practice)
        # For demo purposes, we'll just show RPN proposals
    
    def forward(self, images, gt_boxes=None):
        """
        Args:
            images: [B, 3, H, W]
            gt_boxes: List of [M, 4] tensors (for training)
        
        Returns:
            If training: losses
            If inference: proposals
        """
        B, _, H, W = images.shape
        
        # Shared CNN features
        features = self.backbone(images)
        
        # Generate anchors
        anchors = self.anchor_generator(features, (H, W))
        
        # RPN predictions
        objectness, bbox_deltas = self.rpn(features)
        
        if self.training and gt_boxes is not None:
            # Compute RPN loss
            rpn_loss = RPNLoss()
            cls_loss, reg_loss = rpn_loss(objectness, bbox_deltas, anchors, gt_boxes)
            return {'rpn_cls_loss': cls_loss, 'rpn_reg_loss': reg_loss}
        else:
            # Generate proposals from RPN predictions
            proposals = self.decode_proposals(anchors, objectness, bbox_deltas)
            return proposals
    
    def decode_proposals(self, anchors, objectness, bbox_deltas, top_k=2000):
        """Decode RPN outputs to proposals"""
        B = objectness.size(0)
        device = objectness.device
        
        # Flatten
        objectness = objectness.reshape(B, -1)
        bbox_deltas = bbox_deltas.reshape(B, -1, 4)
        
        proposals_list = []
        
        for b in range(B):
            # Get top-k by objectness
            scores = torch.sigmoid(objectness[b])
            top_scores, top_idx = scores.topk(min(top_k, len(scores)))
            
            # Get corresponding anchors and deltas
            top_anchors = anchors[top_idx]
            top_deltas = bbox_deltas[b, top_idx]
            
            # Apply deltas to anchors
            proposals = self.apply_deltas(top_anchors, top_deltas)
            proposals_list.append(proposals)
        
        return proposals_list
    
    def apply_deltas(self, anchors, deltas):
        """Apply bbox regression deltas to anchors"""
        ax1, ay1, ax2, ay2 = anchors.unbind(dim=1)
        dx, dy, dw, dh = deltas.unbind(dim=1)
        
        aw = ax2 - ax1
        ah = ay2 - ay1
        ax = ax1 + 0.5 * aw
        ay = ay1 + 0.5 * ah
        
        gw = aw * torch.exp(dw)
        gh = ah * torch.exp(dh)
        gx = ax + aw * dx
        gy = ay + ah * dy
        
        x1 = gx - 0.5 * gw
        y1 = gy - 0.5 * gh
        x2 = gx + 0.5 * gw
        y2 = gy + 0.5 * gh
        
        return torch.stack([x1, y1, x2, y2], dim=1)


def demo_faster_rcnn():
    """Demo Faster R-CNN with RPN"""
    print("="*70)
    print("Faster R-CNN Demo: End-to-End Detection with RPN")
    print("="*70)
    
    # Create model
    model = FasterRCNN(num_classes=20)
    model.train()
    
    # Dummy data
    images = torch.randn(2, 3, 224, 224)
    gt_boxes = [
        torch.tensor([[50, 50, 100, 100], [120, 80, 180, 150]], dtype=torch.float32),
        torch.tensor([[30, 40, 90, 120]], dtype=torch.float32)
    ]
    
    print("\n1. Model Architecture")
    print("-" * 70)
    print(f"Backbone: Simplified CNN")
    print(f"RPN: 3 scales × 3 ratios = 9 anchors/location")
    print(f"Classes: {model.num_classes}")
    
    print("\n2. Training RPN")
    print("-" * 70)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(5):
        losses = model(images, gt_boxes)
        
        total_loss = losses['rpn_cls_loss'] + losses['rpn_reg_loss']
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}: Total={total_loss:.4f}, "
              f"Cls={losses['rpn_cls_loss']:.4f}, "
              f"Reg={losses['rpn_reg_loss']:.4f}")
    
    print("\n3. Inference: Generating Proposals")
    print("-" * 70)
    
    model.eval()
    with torch.no_grad():
        proposals = model(images)
    
    print(f"Image 0: Generated {len(proposals[0])} proposals")
    print(f"Image 1: Generated {len(proposals[1])} proposals")
    
    print("\n4. Evolution Comparison")
    print("-" * 70)
    print("""
    R-CNN:       Selective Search (hand-crafted) → CNN → SVM
    Fast R-CNN:  Selective Search (hand-crafted) → Shared CNN → FC
    Faster R-CNN: RPN (learned!) → Shared CNN → FC
    
    Key Innovation: RPN replaces Selective Search
    - End-to-end trainable
    - Faster proposal generation
    - Better proposals (learned from data)
    """)
    
    print("\n5. Computational Comparison")
    print("-" * 70)
    print(f"Selective Search: ~2 seconds per image")
    print(f"RPN: ~0.01 seconds per image (200× faster!)")
    print(f"Plus: RPN proposals are learned → better quality")
    
    print("\n" + "="*70)
    print("Demo Complete!")
    print("="*70)


if __name__ == "__main__":
    demo_faster_rcnn()