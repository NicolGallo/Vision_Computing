"""
Non-Maximum Suppression and box utility functions
"""
import torch


def compute_iou(box1, box2, box_format='xyxy'):
    """
    Compute IoU (Intersection over Union) between boxes
    
    Args:
        box1: Tensor [..., 4] in format specified by box_format
        box2: Tensor [..., 4] in format specified by box_format
        box_format: 'xyxy' (x1, y1, x2, y2) or 'xywh' (x_center, y_center, w, h)
    
    Returns:
        iou: Tensor [...] with IoU values in [0, 1]
    """
    if box_format == 'xywh':
        # Convert to xyxy
        box1 = xywh_to_xyxy(box1)
        box2 = xywh_to_xyxy(box2)
    
    # Get coordinates
    x1_min, y1_min, x1_max, y1_max = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
    x2_min, y2_min, x2_max, y2_max = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]
    
    # Intersection area
    inter_xmin = torch.maximum(x1_min, x2_min)
    inter_ymin = torch.maximum(y1_min, y2_min)
    inter_xmax = torch.minimum(x1_max, x2_max)
    inter_ymax = torch.minimum(y1_max, y2_max)
    
    inter_w = torch.clamp(inter_xmax - inter_xmin, min=0)
    inter_h = torch.clamp(inter_ymax - inter_ymin, min=0)
    inter_area = inter_w * inter_h
    
    # Union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    # IoU
    iou = inter_area / (union_area + 1e-6)
    
    return iou


def xywh_to_xyxy(boxes):
    """
    Convert boxes from (x_center, y_center, w, h) to (x1, y1, x2, y2) format
    
    Args:
        boxes: Tensor [..., 4] in xywh format
    
    Returns:
        boxes: Tensor [..., 4] in xyxy format
    """
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def xyxy_to_xywh(boxes):
    """
    Convert boxes from (x1, y1, x2, y2) to (x_center, y_center, w, h) format
    
    Args:
        boxes: Tensor [..., 4] in xyxy format
    
    Returns:
        boxes: Tensor [..., 4] in xywh format
    """
    x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([x, y, w, h], dim=-1)


def nms(boxes, scores, iou_threshold=0.5):
    """
    Non-Maximum Suppression (NMS)
    
    Removes duplicate detections by suppressing boxes that have high IoU
    with a box of higher confidence.
    
    Args:
        boxes: Tensor [N, 4] in xyxy format
        scores: Tensor [N] with confidence scores
        iou_threshold: IoU threshold for suppression
    
    Returns:
        keep: Tensor with indices of boxes to keep
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=boxes.device)
    
    # Sort by score in descending order
    _, order = scores.sort(descending=True)
    
    keep = []
    
    while order.numel() > 0:
        # Keep box with highest score
        if order.numel() == 1:
            keep.append(order.item())
            break
        
        i = order[0]
        keep.append(i.item())
        
        # Compute IoU with remaining boxes
        ious = compute_iou(boxes[i].unsqueeze(0), boxes[order[1:]], box_format='xyxy')
        
        # Keep boxes with IoU less than threshold
        mask = ious <= iou_threshold
        order = order[1:][mask]
    
    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)


def soft_nms(boxes, scores, iou_threshold=0.5, sigma=0.5, score_threshold=0.001):
    """
    Soft Non-Maximum Suppression
    
    Instead of removing overlapping boxes, reduces their scores based on IoU.
    
    Args:
        boxes: Tensor [N, 4] in xyxy format
        scores: Tensor [N] with confidence scores
        iou_threshold: IoU threshold (for linear mode)
        sigma: Gaussian sigma parameter
        score_threshold: Minimum score to keep detection
    
    Returns:
        keep: Tensor with indices of boxes to keep
        scores: Updated scores after soft suppression
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=boxes.device), scores
    
    # Clone scores to avoid modifying original
    scores = scores.clone()
    
    # Sort by score
    _, order = scores.sort(descending=True)
    
    keep = []
    
    for i in range(len(order)):
        idx = order[i]
        
        if scores[idx] < score_threshold:
            continue
        
        keep.append(idx.item())
        
        if i == len(order) - 1:
            break
        
        # Compute IoU with remaining boxes
        ious = compute_iou(boxes[idx].unsqueeze(0), boxes[order[i+1:]], box_format='xyxy')
        
        # Gaussian suppression
        weights = torch.exp(-(ious ** 2) / sigma)
        
        # Alternative: Linear suppression
        # weights = torch.where(ious > iou_threshold, 1 - ious, torch.ones_like(ious))
        
        # Update scores
        scores[order[i+1:]] *= weights
    
    keep = torch.tensor(keep, dtype=torch.int64, device=boxes.device)
    return keep, scores


def batched_nms(boxes, scores, labels, iou_threshold=0.5):
    """
    Batched NMS for multiple classes
    
    Performs NMS independently for each class.
    
    Args:
        boxes: Tensor [N, 4] in xyxy format
        scores: Tensor [N] with confidence scores
        labels: Tensor [N] with class labels
        iou_threshold: IoU threshold for suppression
    
    Returns:
        keep: Tensor with indices of boxes to keep
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=boxes.device)
    
    # Get unique classes
    unique_labels = labels.unique()
    
    keep_all = []
    
    for label in unique_labels:
        # Filter by class
        mask = labels == label
        class_boxes = boxes[mask]
        class_scores = scores[mask]
        class_indices = torch.where(mask)[0]
        
        # Apply NMS
        keep_class = nms(class_boxes, class_scores, iou_threshold)
        
        # Map back to original indices
        keep_all.append(class_indices[keep_class])
    
    if len(keep_all) == 0:
        return torch.empty(0, dtype=torch.int64, device=boxes.device)
    
    keep = torch.cat(keep_all)
    
    # Sort by score (optional, for consistent ordering)
    keep = keep[scores[keep].argsort(descending=True)]
    
    return keep


def clip_boxes(boxes, image_size):
    """
    Clip boxes to image boundaries
    
    Args:
        boxes: Tensor [..., 4] in xyxy format (can be normalized or absolute)
        image_size: Tuple (height, width) or scalar (for square images)
    
    Returns:
        boxes: Clipped boxes
    """
    if isinstance(image_size, (int, float)):
        h = w = image_size
    else:
        h, w = image_size
    
    boxes = boxes.clone()
    boxes[..., 0] = boxes[..., 0].clamp(min=0, max=w)
    boxes[..., 1] = boxes[..., 1].clamp(min=0, max=h)
    boxes[..., 2] = boxes[..., 2].clamp(min=0, max=w)
    boxes[..., 3] = boxes[..., 3].clamp(min=0, max=h)
    
    return boxes


def box_area(boxes):
    """
    Compute area of boxes
    
    Args:
        boxes: Tensor [..., 4] in xyxy format
    
    Returns:
        area: Tensor [...] with box areas
    """
    return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])


def filter_boxes_by_area(boxes, scores, labels, min_area=0, max_area=float('inf')):
    """
    Filter boxes by area
    
    Args:
        boxes: Tensor [N, 4] in xyxy format
        scores: Tensor [N]
        labels: Tensor [N]
        min_area: Minimum box area
        max_area: Maximum box area
    
    Returns:
        boxes, scores, labels: Filtered tensors
    """
    areas = box_area(boxes)
    mask = (areas >= min_area) & (areas <= max_area)
    
    return boxes[mask], scores[mask], labels[mask]