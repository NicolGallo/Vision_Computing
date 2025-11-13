"""
YOLO (You Only Look Once) object detector implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import SimpleBackbone


class YOLODetector(nn.Module):
    """
    YOLO-style object detector
    
    Reference: "You Only Look Once: Unified, Real-Time Object Detection" (Redmon et al., 2016)
    """
    
    def __init__(self, num_classes=80, grid_size=7, num_boxes=2, backbone='resnet18'):
        """
        Args:
            num_classes: Number of object classes
            grid_size: Grid size S (image divided into S×S grid)
            num_boxes: Number of bounding boxes per grid cell
            backbone: Backbone network ('resnet18' or 'resnet50')
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        
        # Backbone
        self.backbone = SimpleBackbone(backbone, pretrained=True)
        feature_dim = self.backbone.get_out_channels()
        
        # Detection head
        # Output: S × S × (B × 5 + C)
        # Each box: (x, y, w, h, confidence) + class probabilities
        output_channels = num_boxes * 5 + num_classes
        
        self.detection_head = nn.Sequential(
            nn.Conv2d(feature_dim, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(1024, output_channels, kernel_size=1)
        )
        
        # Adaptive pooling to get desired grid size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        
        # Initialize detection head
        self._init_weights()
    
    def _init_weights(self):
        """Initialize detection head weights"""
        for m in self.detection_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input images [B, 3, H, W]
        
        Returns:
            predictions: Tensor [B, S, S, B, 5+C]
                where last dim = [x, y, w, h, conf, class_probs...]
                - x, y: box center coordinates relative to cell (0-1)
                - w, h: box width/height relative to image (0-1)
                - conf: objectness confidence (0-1)
                - class_probs: class probabilities (softmax, sum to 1)
        """
        batch_size = x.size(0)
        
        # Extract features
        features = self.backbone(x)  # [B, feature_dim, H/32, W/32]
        
        # Pool to grid size
        features = self.adaptive_pool(features)  # [B, feature_dim, S, S]
        
        # Detection head
        out = self.detection_head(features)  # [B, B*5+C, S, S]
        
        # Reshape to [B, S, S, B*5+C]
        out = out.permute(0, 2, 3, 1).contiguous()
        
        # Further reshape to [B, S, S, B, 5+C]
        out = out.view(batch_size, self.grid_size, self.grid_size, 
                      self.num_boxes, 5 + self.num_classes)
        
        # Apply activations
        predictions = out.clone()
        predictions[..., 0:2] = torch.sigmoid(out[..., 0:2])  # x, y in [0, 1]
        predictions[..., 2:4] = out[..., 2:4]  # w, h (keep as is for now)
        predictions[..., 4:5] = torch.sigmoid(out[..., 4:5])  # confidence
        predictions[..., 5:] = F.softmax(out[..., 5:], dim=-1)  # class probs
        
        return predictions


class YOLODecoder:
    """
    Decoder to convert YOLO predictions to bounding boxes
    """
    
    def __init__(self, grid_size=7, num_boxes=2, num_classes=80,
                 conf_threshold=0.05, nms_threshold=0.5):
        """
        Args:
            grid_size: YOLO grid size
            num_boxes: Number of boxes per cell
            num_classes: Number of classes
            conf_threshold: Confidence threshold for filtering
            nms_threshold: NMS IoU threshold
        """
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
    
    def decode(self, predictions):
        """
        Decode YOLO predictions to bounding boxes
        
        Args:
            predictions: [B, S, S, B, 5+C] tensor
        
        Returns:
            detections: List of B lists, each containing (box, score, label) tuples
                       box: [x1, y1, x2, y2] in normalized coordinates
        """
        from utils.nms import nms
        
        batch_size, S = predictions.shape[0], self.grid_size
        detections = []
        
        for b in range(batch_size):
            boxes_list = []
            scores_list = []
            labels_list = []
            
            for i in range(S):
                for j in range(S):
                    for box_idx in range(self.num_boxes):
                        pred = predictions[b, i, j, box_idx]
                        
                        x_cell = pred[0].item()
                        y_cell = pred[1].item()
                        w = pred[2].item()
                        h = pred[3].item()
                        conf = pred[4].item()
                        class_probs = pred[5:]
                        
                        # Convert to absolute coordinates
                        x = (j + x_cell) / S
                        y = (i + y_cell) / S
                        
                        # Get class prediction
                        class_conf, class_id = class_probs.max(dim=0)
                        score = conf * class_conf.item()
                        
                        if score < self.conf_threshold:
                            continue
                        
                        # Convert from xywh to xyxy (normalized)
                        x1 = x - w / 2
                        y1 = y - h / 2
                        x2 = x + w / 2
                        y2 = y + h / 2
                        
                        # Clip to [0, 1]
                        x1 = max(0, min(1, x1))
                        y1 = max(0, min(1, y1))
                        x2 = max(0, min(1, x2))
                        y2 = max(0, min(1, y2))
                        
                        boxes_list.append([x1, y1, x2, y2])
                        scores_list.append(score)
                        labels_list.append(class_id.item())
            
            if len(boxes_list) == 0:
                detections.append([])
                continue
            
            boxes = torch.tensor(boxes_list, device=predictions.device)
            scores = torch.tensor(scores_list, device=predictions.device)
            labels = torch.tensor(labels_list, device=predictions.device)
            
            # Apply NMS
            keep = nms(boxes, scores, self.nms_threshold)
            
            det = [
                (boxes[i].tolist(), scores[i].item(), int(labels[i].item()))
                for i in keep
            ]
            detections.append(det)
        
        return detections