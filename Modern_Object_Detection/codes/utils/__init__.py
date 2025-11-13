"""
Utilities package for object detection
"""
from .nms import (
    nms, soft_nms, batched_nms,
    compute_iou, xywh_to_xyxy, xyxy_to_xywh,
    clip_boxes, box_area, filter_boxes_by_area
)
from .metrics import (
    COCOEvaluator, DetectionMetrics, AverageMeter,
    calculate_confusion_matrix
)

__all__ = [
    # NMS functions
    'nms',
    'soft_nms',
    'batched_nms',
    # Box utilities
    'compute_iou',
    'xywh_to_xyxy',
    'xyxy_to_xywh',
    'clip_boxes',
    'box_area',
    'filter_boxes_by_area',
    # Metrics
    'COCOEvaluator',
    'DetectionMetrics',
    'AverageMeter',
    'calculate_confusion_matrix'
]