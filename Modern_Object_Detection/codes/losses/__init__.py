"""
Losses package for object detection
"""
from .yolo_loss import YOLOLoss
from .focal_loss import FocalLoss, FCOSLoss

__all__ = [
    'YOLOLoss',
    'FocalLoss',
    'FCOSLoss'
]