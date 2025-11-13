"""
COCO dataset loader for object detection
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from pycocotools.coco import COCO


class COCODetectionDataset(Dataset):
    """
    COCO Dataset for object detection
    
    Returns images with bounding boxes and labels
    """
    
    def __init__(self, img_dir, ann_file, transform=None, is_train=True, filter_empty=True):
        """
        Args:
            img_dir: Path to images directory
            ann_file: Path to COCO annotation JSON file
            transform: Data augmentation transforms
            is_train: Whether this is training set
            filter_empty: Whether to filter images without annotations
        """
        self.img_dir = Path(img_dir)
        self.coco = COCO(str(ann_file))
        self.transform = transform
        self.is_train = is_train
        
        # Get all image IDs
        self.img_ids = list(self.coco.imgs.keys())
        
        # Filter images without annotations (optional)
        if filter_empty and is_train:
            self.img_ids = [
                img_id for img_id in self.img_ids
                if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=False)) > 0
            ]
        
        # Create contiguous category ID mapping (COCO IDs are not contiguous)
        self.coco_ids = sorted(self.coco.getCatIds())
        self.coco_id_to_class = {coco_id: idx for idx, coco_id in enumerate(self.coco_ids)}
        self.class_to_coco_id = {idx: coco_id for coco_id, idx in self.coco_id_to_class.items()}
        
        print(f"Loaded COCO dataset: {len(self.img_ids)} images, {len(self.coco_ids)} classes")
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        """
        Get one sample from dataset
        
        Returns:
            image: Tensor [3, H, W]
            boxes: Tensor [N, 4] in XYXY format, normalized to [0, 1]
            labels: Tensor [N] with class indices [0, num_classes-1]
            image_id: int
        """
        img_id = self.img_ids[idx]
        
        # Load image
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.img_dir / img_info['file_name']
        image = Image.open(img_path).convert('RGB')
        
        img_w, img_h = image.size
        
        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)
        
        # Extract boxes and labels
        boxes = []
        labels = []
        
        for ann in anns:
            # Skip crowd annotations
            if ann.get('iscrowd', 0) == 1:
                continue
            
            # Get bbox in XYWH format
            x, y, w, h = ann['bbox']
            
            # Skip invalid boxes
            if w <= 0 or h <= 0:
                continue
            
            # Convert to XYXY
            x2, y2 = x + w, y + h
            
            # Normalize coordinates to [0, 1]
            x1_norm = x / img_w
            y1_norm = y / img_h
            x2_norm = x2 / img_w
            y2_norm = y2 / img_h
            
            # Clip to valid range
            x1_norm = max(0, min(1, x1_norm))
            y1_norm = max(0, min(1, y1_norm))
            x2_norm = max(0, min(1, x2_norm))
            y2_norm = max(0, min(1, y2_norm))
            
            # Skip degenerate boxes
            if x2_norm <= x1_norm or y2_norm <= y1_norm:
                continue
            
            boxes.append([x1_norm, y1_norm, x2_norm, y2_norm])
            labels.append(self.coco_id_to_class[ann['category_id']])
        
        # Convert to tensors
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, boxes, labels, img_id
    
    def get_img_info(self, idx):
        """Get image information"""
        img_id = self.img_ids[idx]
        return self.coco.loadImgs(img_id)[0]


def collate_fn(batch):
    """
    Custom collate function for variable number of boxes per image
    
    Args:
        batch: List of (image, boxes, labels, img_id) tuples
    
    Returns:
        images: Tensor [B, 3, H, W]
        boxes: List of [N_i, 4] tensors
        labels: List of [N_i] tensors
        img_ids: List of image IDs
    """
    images, boxes, labels, img_ids = zip(*batch)
    
    # Stack images into batch
    images = torch.stack(images, 0)
    
    # Keep boxes and labels as lists (different lengths per image)
    return images, list(boxes), list(labels), list(img_ids)