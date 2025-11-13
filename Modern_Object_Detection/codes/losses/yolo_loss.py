"""
YOLO loss function implementation
"""
import torch
import torch.nn as nn
from utils.nms import compute_iou, xyxy_to_xywh


class YOLOLoss(nn.Module):
    """
    YOLO loss function
    
    Combines localization, confidence, and classification losses
    """
    
    def __init__(self, num_classes=80, grid_size=7, num_boxes=2,
                 lambda_coord=5.0, lambda_noobj=0.5):
        """
        Args:
            num_classes: Number of object classes
            grid_size: YOLO grid size
            num_boxes: Number of boxes per cell
            lambda_coord: Weight for coordinate loss
            lambda_noobj: Weight for no-object confidence loss
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
    
    def forward(self, predictions, targets_boxes, targets_labels):
        """
        Compute YOLO loss
        
        Args:
            predictions: Tensor [B, S, S, B, 5+C]
            targets_boxes: List of [N_i, 4] tensors (normalized xyxy)
            targets_labels: List of [N_i] tensors
        
        Returns:
            loss: Total loss (scalar)
            loss_dict: Dictionary with loss components
        """
        batch_size = predictions.size(0)
        device = predictions.device
        
        # Encode targets to grid format
        target_grids = self.encode_targets(
            targets_boxes, targets_labels, batch_size, device
        )  # [B, S, S, 5+C]
        
        # Extract target components
        target_xy = target_grids[..., :2]
        target_wh = target_grids[..., 2:4]
        target_conf = target_grids[..., 4:5]
        target_cls = target_grids[..., 5:]
        
        # Object mask [B, S, S]
        obj_mask = (target_conf[..., 0] > 0).float()
        noobj_mask = 1 - obj_mask
        
        # Find responsible boxes (highest IoU with GT)
        responsible_mask = self.get_responsible_boxes(
            predictions, target_xy, target_wh, target_conf
        )  # [B, S, S, B]
        
        # Extract predictions
        pred_xy = predictions[..., 0:2]  # [B, S, S, B, 2]
        pred_wh = predictions[..., 2:4]
        pred_conf = predictions[..., 4:5]
        pred_cls = predictions[..., 5:]
        
        # ========== Localization Loss ==========
        # Only for responsible boxes in cells with objects
        loss_xy = self.lambda_coord * torch.sum(
            obj_mask.unsqueeze(-1).unsqueeze(-1) * 
            responsible_mask.unsqueeze(-1) *
            (pred_xy - target_xy.unsqueeze(3)) ** 2
        ) / batch_size
        
        # Use sqrt for width and height (as in original YOLO)
        loss_wh = self.lambda_coord * torch.sum(
            obj_mask.unsqueeze(-1).unsqueeze(-1) * 
            responsible_mask.unsqueeze(-1) *
            (torch.sqrt(torch.clamp(pred_wh, min=1e-6)) - 
             torch.sqrt(torch.clamp(target_wh.unsqueeze(3), min=1e-6))) ** 2
        ) / batch_size
        
        # ========== Confidence Loss (Object) ==========
        loss_conf_obj = torch.sum(
            obj_mask.unsqueeze(-1).unsqueeze(-1) * 
            responsible_mask.unsqueeze(-1) *
            (pred_conf - target_conf.unsqueeze(3)) ** 2
        ) / batch_size
        
        # ========== Confidence Loss (No Object) ==========
        loss_conf_noobj = self.lambda_noobj * torch.sum(
            noobj_mask.unsqueeze(-1).unsqueeze(-1) * 
            (pred_conf - 0) ** 2
        ) / batch_size
        
        # ========== Classification Loss ==========
        # Only for cells with objects, use responsible box predictions
        pred_cls_resp = (pred_cls * responsible_mask.unsqueeze(-1)).sum(dim=3)
        loss_cls = torch.sum(
            obj_mask.unsqueeze(-1) * 
            (pred_cls_resp - target_cls) ** 2
        ) / batch_size
        
        # Total loss
        total_loss = loss_xy + loss_wh + loss_conf_obj + loss_conf_noobj + loss_cls
        
        loss_dict = {
            'loss_xy': loss_xy.item(),
            'loss_wh': loss_wh.item(),
            'loss_conf_obj': loss_conf_obj.item(),
            'loss_conf_noobj': loss_conf_noobj.item(),
            'loss_cls': loss_cls.item()
        }
        
        return total_loss, loss_dict
    
    def encode_targets(self, targets_boxes, targets_labels, batch_size, device):
        """
        Encode ground truth boxes to grid format
        
        Args:
            targets_boxes: List of [N_i, 4] tensors (xyxy format, normalized)
            targets_labels: List of [N_i] tensors
            batch_size: Batch size
            device: Device
        
        Returns:
            target_grid: Tensor [B, S, S, 5+C]
                        [x_cell, y_cell, w, h, conf, class_probs...]
        """
        S = self.grid_size
        C = self.num_classes
        
        target_grid = torch.zeros(batch_size, S, S, 5 + C, device=device)
        
        for b in range(batch_size):
            boxes = targets_boxes[b]  # [N, 4] xyxy format
            labels = targets_labels[b]  # [N]
            
            if len(boxes) == 0:
                continue
            
            # Convert to xywh (center format)
            boxes_xywh = xyxy_to_xywh(boxes)
            
            for box, label in zip(boxes_xywh, labels):
                x, y, w, h = box.tolist()
                
                # Find grid cell
                i = int(y * S)
                j = int(x * S)
                
                # Clip to valid range
                i = min(i, S - 1)
                j = min(j, S - 1)
                
                # Compute offsets relative to cell
                x_cell = x * S - j
                y_cell = y * S - i
                
                # Set target (only set once per cell)
                if target_grid[b, i, j, 4] == 0:  # Cell not yet assigned
                    target_grid[b, i, j, 0] = x_cell
                    target_grid[b, i, j, 1] = y_cell
                    target_grid[b, i, j, 2] = w
                    target_grid[b, i, j, 3] = h
                    target_grid[b, i, j, 4] = 1.0  # confidence
                    target_grid[b, i, j, 5 + label] = 1.0  # class
        
        return target_grid
    
    def get_responsible_boxes(self, predictions, target_xy, target_wh, target_conf):
        """
        Determine which box in each cell is responsible for prediction
        
        Args:
            predictions: [B, S, S, B, 5+C]
            target_xy: [B, S, S, 2]
            target_wh: [B, S, S, 2]
            target_conf: [B, S, S, 1]
        
        Returns:
            responsible_mask: [B, S, S, B] binary mask
        """
        B, S, _, num_boxes, _ = predictions.shape
        device = predictions.device
        
        # Extract predicted boxes
        pred_xy = predictions[..., 0:2]  # [B, S, S, B, 2]
        pred_wh = predictions[..., 2:4]
        
        # Prepare target boxes [B, S, S, 1, 4]
        target_boxes = torch.cat([target_xy, target_wh], dim=-1).unsqueeze(3)
        
        # Prepare pred boxes [B, S, S, B, 4]
        pred_boxes = torch.cat([pred_xy, pred_wh], dim=-1)
        
        # Compute IoU [B, S, S, B]
        ious = compute_iou(
            pred_boxes.reshape(-1, 4),
            target_boxes.expand_as(pred_boxes).reshape(-1, 4),
            box_format='xywh'
        ).reshape(B, S, S, num_boxes)
        
        # Find box with highest IoU for each cell
        _, best_box_idx = ious.max(dim=3)  # [B, S, S]
        
        # Create one-hot mask
        responsible_mask = torch.zeros(B, S, S, num_boxes, device=device)
        
        for b in range(B):
            for i in range(S):
                for j in range(S):
                    if target_conf[b, i, j, 0] > 0:
                        responsible_mask[b, i, j, best_box_idx[b, i, j]] = 1.0
        
        return responsible_mask