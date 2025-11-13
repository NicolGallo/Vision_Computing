"""
Datasets package for object detection
"""
from .coco_dataset import COCODetectionDataset, collate_fn
from .transforms import get_transforms

__all__ = [
    'COCODetectionDataset',
    'collate_fn',
    'get_transforms'
]