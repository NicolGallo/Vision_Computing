"""
Focal Loss and FCOS Loss implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.nms import compute_iou


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in dense object detection
    
    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    
    Formula: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Weighting factor in [0, 1] to balance positive/negative examples
            gamma: Focusing parameter γ ≥ 0 (γ=0 is equivalent to CE loss)
            reduction: 'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predicted logits [N, C] or [N]
            targets: Ground truth labels [N, C] (one-hot) or [N]
        
        Returns:
            loss: Focal loss
        """
        # Compute probabilities
        p = torch.sigmoid(inputs)
        
        # Compute cross-entropy loss
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Compute p_t
        p_t = p * targets + (1 - p) * (1 - targets)
        
        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        # Apply alpha weighting
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class FCOSLoss(nn.Module):
    """
    FCOS loss function combining classification, regression, and center-ness losses
    """
    
    def __init__(self, num_classes=80, strides=[8, 16, 32],
                 scale_ranges=[(0, 64), (64, 128), (128, float('inf'))],
                 focal_alpha=0.25, focal_gamma=2.0):
        """
        Args:
            num_classes: Number of object classes
            strides: FPN level strides
            scale_ranges: Scale ranges for each FPN level
            focal_alpha: Focal loss alpha parameter
            focal_gamma: Focal loss gamma parameter
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.strides = strides
        self.scale_ranges = scale_ranges
        
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='sum')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='sum')
    
    def forward(self, cls_logits_list, reg_preds_list, centerness_list,
                targets_boxes, targets_labels):
        """
        Compute FCOS loss
        
        Args:
            cls_logits_list: List of [B, C, H, W]
            reg_preds_list: List of [B, 4, H, W]
            centerness_list: List of [B, 1, H, W]
            targets_boxes: List of [N_i, 4] (xyxy, normalized)
            targets_labels: List of [N_i]
        
        Returns:
            total_loss: Total loss
            loss_dict: Dictionary of individual losses
        """
        batch_size = cls_logits_list[0].size(0)
        device = cls_logits_list[0].device
        
        # Assign targets to all FPN levels
        all_cls_targets = []
        all_reg_targets = []
        all_centerness_targets = []
        all_pos_masks = []
        
        for level_idx, (cls_logits, reg_preds, centerness) in enumerate(
            zip(cls_logits_list, reg_preds_list, centerness_list)
        ):
            B, C, H, W = cls_logits.shape
            stride = self.strides[level_idx]
            scale_range = self.scale_ranges[level_idx]
            
            # Generate locations for this level
            locations = self.compute_locations(H, W, stride, device)  # [H*W, 2]
            
            # Assign targets
            cls_targets, reg_targets, centerness_targets, pos_mask = \
                self.assign_targets_single_level(
                    locations, targets_boxes, targets_labels, stride, 
                    scale_range, batch_size, device
                )
            
            all_cls_targets.append(cls_targets)
            all_reg_targets.append(reg_targets)
            all_centerness_targets.append(centerness_targets)
            all_pos_masks.append(pos_mask)
        
        # Concatenate across levels
        cls_targets = torch.cat(all_cls_targets, dim=1)  # [B, sum(H*W)]
        reg_targets = torch.cat(all_reg_targets, dim=1)  # [B, sum(H*W), 4]
        centerness_targets = torch.cat(all_centerness_targets, dim=1)  # [B, sum(H*W)]
        pos_masks = torch.cat(all_pos_masks, dim=1)  # [B, sum(H*W)]
        
        # Reshape predictions
        cls_preds = torch.cat([
            x.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes)
            for x in cls_logits_list
        ], dim=1)
        
        reg_preds = torch.cat([
            x.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
            for x in reg_preds_list
        ], dim=1)
        
        centerness_preds = torch.cat([
            x.permute(0, 2, 3, 1).reshape(batch_size, -1)
            for x in centerness_list
        ], dim=1)
        
        # Number of positive samples
        num_pos = pos_masks.sum().clamp(min=1)
        
        # ========== Classification Loss (Focal Loss) ==========
        # Flatten for focal loss
        cls_preds_flat = cls_preds.reshape(-1, self.num_classes)
        cls_targets_flat = cls_targets.reshape(-1)
        
        # Create one-hot targets (including background class)
        cls_targets_one_hot = torch.zeros(
            cls_targets_flat.size(0), self.num_classes + 1,
            dtype=torch.float32, device=device
        )
        cls_targets_one_hot[torch.arange(cls_targets_flat.size(0)), cls_targets_flat] = 1.0
        cls_targets_one_hot = cls_targets_one_hot[:, :-1]  # Remove background class
        
        loss_cls = self.focal_loss(cls_preds_flat, cls_targets_one_hot) / num_pos
        
        # ========== Regression Loss (GIoU) ==========
        if pos_masks.sum() > 0:
            loss_reg = self.giou_loss(
                reg_preds[pos_masks],
                reg_targets[pos_masks]
            ).sum() / num_pos
        else:
            loss_reg = reg_preds.sum() * 0
        
        # ========== Center-ness Loss (BCE) ==========
        if pos_masks.sum() > 0:
            loss_centerness = self.bce_loss(
                centerness_preds[pos_masks],
                centerness_targets[pos_masks]
            ) / num_pos
        else:
            loss_centerness = centerness_preds.sum() * 0
        
        # Total loss
        total_loss = loss_cls + loss_reg + loss_centerness
        
        loss_dict = {
            'loss_cls': loss_cls.item(),
            'loss_reg': loss_reg.item(),
            'loss_centerness': loss_centerness.item(),
            'num_pos': num_pos.item()
        }
        
        return total_loss, loss_dict
    
    def compute_locations(self, h, w, stride, device):
        """
        Compute center locations for each position in feature map
        
        Args:
            h, w: Feature map height and width
            stride: Stride of this FPN level
            device: Device
        
        Returns:
            locations: [H*W, 2] tensor with (x, y) coordinates in input image space
        """
        shifts_x = torch.arange(0, w, dtype=torch.float32, device=device) * stride
        shifts_y = torch.arange(0, h, dtype=torch.float32, device=device) * stride
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack([shift_x, shift_y], dim=1) + stride // 2
        return locations
    
    def assign_targets_single_level(self, locations, targets_boxes, targets_labels,
                                    stride, scale_range, batch_size, device):
        """
        Assign targets to locations for a single FPN level
        
        Args:
            locations: [L, 2] location coordinates
            targets_boxes: List of [N_i, 4] boxes (xyxy, normalized)
            targets_labels: List of [N_i] labels
            stride: Stride for this level
            scale_range: (min_size, max_size) for this level
            batch_size: Batch size
            device: Device
        
        Returns:
            cls_targets: [B, L] class targets (num_classes = background)
            reg_targets: [B, L, 4] regression targets (l, t, r, b)
            centerness_targets: [B, L] center-ness targets
            pos_masks: [B, L] positive sample masks
        """
        num_locations = locations.size(0)
        
        cls_targets = torch.full((batch_size, num_locations), self.num_classes, 
                                 dtype=torch.long, device=device)
        reg_targets = torch.zeros(batch_size, num_locations, 4, device=device)
        centerness_targets = torch.zeros(batch_size, num_locations, device=device)
        pos_masks = torch.zeros(batch_size, num_locations, dtype=torch.bool, device=device)
        
        for b in range(batch_size):
            boxes = targets_boxes[b]  # [N, 4] xyxy normalized
            labels = targets_labels[b]  # [N]
            
            if len(boxes) == 0:
                continue
            
            # Normalize locations to [0, 1]
            locs_normalized = locations / 448.0  # Assuming input size 448
            
            # Compute l, t, r, b for all locations and all boxes
            # locations: [L, 2], boxes: [N, 4]
            l = locs_normalized[:, 0:1] - boxes[:, 0:1].T  # [L, N]
            t = locs_normalized[:, 1:2] - boxes[:, 1:2].T
            r = boxes[:, 2:3].T - locs_normalized[:, 0:1]
            b = boxes[:, 3:4].T - locs_normalized[:, 1:2]
            
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)  # [L, N, 4]
            
            # Check if location is inside box
            is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0  # [L, N]
            
            # Check scale range
            max_reg_targets = reg_targets_per_im.max(dim=2)[0]  # [L, N]
            max_reg_targets_in_pixels = max_reg_targets * 448.0
            is_in_scale = (max_reg_targets_in_pixels >= scale_range[0]) & \
                         (max_reg_targets_in_pixels <= scale_range[1])
            
            # Valid locations
            valid = is_in_boxes & is_in_scale  # [L, N]
            
            # For each location, find the box with minimum area
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # [N]
            areas = areas[None].repeat(num_locations, 1)  # [L, N]
            areas[~valid] = float('inf')
            
            min_area, min_area_inds = areas.min(dim=1)  # [L]
            
            # Positive locations
            pos_inds = min_area != float('inf')
            
            # Set targets for positive locations
            cls_targets[b, pos_inds] = labels[min_area_inds[pos_inds]]
            reg_targets[b, pos_inds] = reg_targets_per_im[pos_inds, min_area_inds[pos_inds]]
            
            # Compute center-ness
            left_right = reg_targets[b, pos_inds, [0, 2]]
            top_bottom = reg_targets[b, pos_inds, [1, 3]]
            centerness = torch.sqrt(
                (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0].clamp(min=1e-6)) *
                (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0].clamp(min=1e-6))
            )
            centerness_targets[b, pos_inds] = centerness
            
            pos_masks[b] = pos_inds
        
        return cls_targets, reg_targets, centerness_targets, pos_masks
    
    def giou_loss(self, pred, target):
        """
        Generalized IoU loss
        
        Args:
            pred: [N, 4] predicted boxes in ltrb format
            target: [N, 4] target boxes in ltrb format
        
        Returns:
            loss: [N] GIoU loss values
        """
        # Convert ltrb to xyxy (with arbitrary center at origin)
        pred_boxes = torch.zeros_like(pred)
        pred_boxes[:, 0] = -pred[:, 0]  # x1 = -l
        pred_boxes[:, 1] = -pred[:, 1]  # y1 = -t
        pred_boxes[:, 2] = pred[:, 2]   # x2 = r
        pred_boxes[:, 3] = pred[:, 3]   # y2 = b
        
        target_boxes = torch.zeros_like(target)
        target_boxes[:, 0] = -target[:, 0]
        target_boxes[:, 1] = -target[:, 1]
        target_boxes[:, 2] = target[:, 2]
        target_boxes[:, 3] = target[:, 3]
        
        # IoU
        iou = compute_iou(pred_boxes, target_boxes, box_format='xyxy')
        
        # Enclosing box
        enclose_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        enclose_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        enclose_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        enclose_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
        
        enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
        
        # Union
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        union_area = pred_area + target_area - iou * pred_area
        
        # GIoU
        giou = iou - (enclose_area - union_area) / enclose_area.clamp(min=1e-6)
        
        return 1 - giou