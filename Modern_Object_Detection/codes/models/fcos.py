"""
FCOS (Fully Convolutional One-Stage) object detector implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import ResNetBackbone
from .fpn import FPN


class FCOSHead(nn.Module):
    """
    FCOS detection head (shared across FPN levels)
    """
    
    def __init__(self, in_channels, num_classes, num_convs=4):
        """
        Args:
            in_channels: Input channels from FPN
            num_classes: Number of object classes
            num_convs: Number of conv layers in each tower
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # Classification tower
        cls_layers = []
        for i in range(num_convs):
            cls_layers.extend([
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
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
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(32, in_channels),
                nn.ReLU(inplace=True)
            ])
        self.reg_tower = nn.Sequential(*reg_layers)
        
        # Regression output (l, t, r, b)
        self.reg_pred = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        
        # Center-ness branch
        self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        
        # Scales for regression (learnable per-level scaling)
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
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
        """
        Forward pass
        
        Args:
            x: Feature map [B, C, H, W]
            level_idx: FPN level index for scale adjustment
        
        Returns:
            cls_logits: [B, num_classes, H, W]
            reg_pred: [B, 4, H, W]
            centerness: [B, 1, H, W]
        """
        # Classification branch
        cls_feat = self.cls_tower(x)
        cls_logits = self.cls_logits(cls_feat)
        
        # Regression branch
        reg_feat = self.reg_tower(x)
        reg_pred = self.scales[level_idx](self.reg_pred(reg_feat))
        reg_pred = F.relu(reg_pred)  # Ensure positive distances
        
        # Center-ness
        centerness = self.centerness(reg_feat)
        
        return cls_logits, reg_pred, centerness


class Scale(nn.Module):
    """Learnable scale parameter"""
    
    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))
    
    def forward(self, x):
        return x * self.scale


class FCOSDetector(nn.Module):
    """
    FCOS anchor-free object detector
    
    Reference: "FCOS: Fully Convolutional One-Stage Object Detection" (Tian et al., 2019)
    """
    
    def __init__(self, num_classes=80, backbone='resnet50', fpn_channels=256, num_convs=4):
        """
        Args:
            num_classes: Number of object classes
            backbone: Backbone network ('resnet18' or 'resnet50')
            fpn_channels: FPN output channels
            num_convs: Number of conv layers in detection head
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.strides = [8, 16, 32]
        
        # Backbone + FPN
        self.backbone = ResNetBackbone(backbone, pretrained=True)
        feature_dims = self.backbone.get_feature_dims()
        self.fpn = FPN(feature_dims, out_channels=fpn_channels)
        
        # FCOS head (shared across FPN levels)
        self.head = FCOSHead(fpn_channels, num_classes, num_convs)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input images [B, 3, H, W]
        
        Returns:
            cls_logits_list: List of [B, C, H_i, W_i]
            reg_preds_list: List of [B, 4, H_i, W_i]
            centerness_list: List of [B, 1, H_i, W_i]
        """
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


class FCOSDecoder:
    """
    Decoder to convert FCOS predictions to bounding boxes
    """
    
    def __init__(self, strides=[8, 16, 32], num_classes=80,
                 conf_threshold=0.05, nms_threshold=0.5, input_size=448):
        """
        Args:
            strides: FPN level strides
            num_classes: Number of classes
            conf_threshold: Confidence threshold
            nms_threshold: NMS IoU threshold
            input_size: Input image size
        """
        self.strides = strides
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
    
    def decode(self, cls_logits_list, reg_preds_list, centerness_list):
        """
        Decode FCOS predictions to bounding boxes
        
        Args:
            cls_logits_list: List of [B, C, H, W]
            reg_preds_list: List of [B, 4, H, W]
            centerness_list: List of [B, 1, H, W]
        
        Returns:
            detections: List of B lists, each containing (box, score, label) tuples
        """
        from utils.nms import nms
        
        batch_size = cls_logits_list[0].size(0)
        device = cls_logits_list[0].device
        
        detections = []
        
        for b in range(batch_size):
            boxes_list = []
            scores_list = []
            labels_list = []
            
            for level_idx, (cls_logits, reg_preds, centerness) in enumerate(
                zip(cls_logits_list, reg_preds_list, centerness_list)
            ):
                H, W = cls_logits.shape[2:]
                stride = self.strides[level_idx]
                
                # Reshape predictions
                cls_scores = torch.sigmoid(cls_logits[b]).permute(1, 2, 0).reshape(-1, self.num_classes)
                reg_pred = reg_preds[b].permute(1, 2, 0).reshape(-1, 4)
                centerness_pred = torch.sigmoid(centerness[b]).permute(1, 2, 0).reshape(-1)
                
                # Generate locations
                shifts_x = torch.arange(0, W, dtype=torch.float32, device=device) * stride
                shifts_y = torch.arange(0, H, dtype=torch.float32, device=device) * stride
                shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
                locations = torch.stack([shift_x.reshape(-1), shift_y.reshape(-1)], dim=1) + stride // 2
                
                # Get class predictions
                scores, labels = cls_scores.max(dim=1)
                scores = scores * centerness_pred  # Multiply by centerness
                
                # Filter by confidence
                keep = scores > self.conf_threshold
                if keep.sum() == 0:
                    continue
                
                scores = scores[keep]
                labels = labels[keep]
                reg_pred = reg_pred[keep]
                locations = locations[keep]
                
                # Convert ltrb to xyxy (normalized)
                l, t, r, b = reg_pred[:, 0], reg_pred[:, 1], reg_pred[:, 2], reg_pred[:, 3]
                x1 = (locations[:, 0] - l) / self.input_size
                y1 = (locations[:, 1] - t) / self.input_size
                x2 = (locations[:, 0] + r) / self.input_size
                y2 = (locations[:, 1] + b) / self.input_size
                
                boxes = torch.stack([x1, y1, x2, y2], dim=1)
                
                # Clip to [0, 1]
                boxes = torch.clamp(boxes, 0, 1)
                
                boxes_list.append(boxes)
                scores_list.append(scores)
                labels_list.append(labels)
            
            if len(boxes_list) == 0:
                detections.append([])
                continue
            
            # Concatenate all levels
            boxes = torch.cat(boxes_list)
            scores = torch.cat(scores_list)
            labels = torch.cat(labels_list)
            
            # Apply NMS
            keep = nms(boxes, scores, self.nms_threshold)
            
            det = [
                (boxes[i].tolist(), scores[i].item(), int(labels[i].item()))
                for i in keep
            ]
            detections.append(det)
        
        return detections