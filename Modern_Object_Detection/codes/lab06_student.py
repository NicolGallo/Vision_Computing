# ============================================================================
# Lab 6: Object Detection - Student Template
# Artificial Vision (VIAR25/26) - UVigo
# 
# INSTRUCTIONS:
# 1. Complete the TODOs in order
# 2. Test each component before moving to the next
# 3. Use the provided test functions to verify your implementation
# 4. Refer to the lecture notes for mathematical details
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.models import resnet18, resnet50

import json
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for training and evaluation"""
    # Data
    data_root = Path('data/coco')
    train_images = data_root / 'train2017'
    val_images = data_root / 'val2017'
    train_ann = data_root / 'annotations/instances_train2017.json'
    val_ann = data_root / 'annotations/instances_val2017.json'
    
    # Model
    num_classes = 80
    input_size = 448
    
    # YOLO specific
    grid_size = 7
    num_boxes = 2
    
    # FCOS specific
    fpn_strides = [8, 16, 32]
    fpn_scales = [(0, 64), (64, 128), (128, float('inf'))]
    
    # Training
    batch_size = 8
    num_epochs = 20
    learning_rate = 1e-4
    weight_decay = 5e-4
    num_workers = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Loss weights
    lambda_coord = 5.0
    lambda_noobj = 0.5
    lambda_cls = 1.0
    lambda_conf = 1.0
    
    # Focal loss
    focal_alpha = 0.25
    focal_gamma = 2.0
    
    # NMS
    nms_threshold = 0.5
    conf_threshold = 0.05
    
    # Logging
    log_interval = 100
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)

config = Config()

# ============================================================================
# DATASET
# ============================================================================

class COCODetectionDataset(Dataset):
    """COCO Dataset for object detection"""
    
    def __init__(self, img_dir, ann_file, transform=None, is_train=True):
        """
        Initialize COCO dataset
        
        Args:
            img_dir: Path to images directory
            ann_file: Path to COCO annotation JSON file
            transform: Data augmentation transforms
            is_train: Whether this is training set
        """
        self.img_dir = Path(img_dir)
        self.coco = COCO(str(ann_file))
        self.transform = transform
        self.is_train = is_train
        
        # TODO 1.1: Get all image IDs from COCO
        # Hint: Use self.coco.imgs.keys()
        self.img_ids = None  # REPLACE THIS
        
        # TODO 1.2: Filter images without annotations (for training only)
        # Hint: Use self.coco.getAnnIds(imgIds=img_id) to check if image has annotations
        if is_train:
            pass  # IMPLEMENT FILTERING
        
        # TODO 1.3: Create contiguous category ID mapping
        # COCO category IDs are not contiguous (1-90 with gaps)
        # We need to map them to [0, num_classes-1]
        # Hint: Get category IDs with self.coco.getCatIds(), sort them,
        # and create a dictionary mapping COCO ID to class index
        self.coco_ids = None  # REPLACE THIS
        self.coco_id_to_class = None  # REPLACE THIS
        
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
        
        # TODO 1.4: Load image
        # Hint: Use self.coco.loadImgs(img_id) to get image info
        # Then load image from self.img_dir / img_info['file_name']
        # Convert to RGB using PIL.Image
        img_info = None  # GET IMAGE INFO
        img_path = None  # CONSTRUCT PATH
        image = None  # LOAD AND CONVERT TO RGB
        
        # TODO 1.5: Get annotations for this image
        # Hint: Use self.coco.getAnnIds(imgIds=img_id) and self.coco.loadAnns()
        ann_ids = None  # GET ANNOTATION IDS
        anns = None  # LOAD ANNOTATIONS
        
        # TODO 1.6: Extract boxes and labels
        # For each annotation:
        #   - Skip if 'iscrowd' == 1
        #   - Get bbox in XYWH format from ann['bbox']
        #   - Convert to XYXY format: x2 = x + w, y2 = y + h
        #   - Normalize coordinates by image width and height
        #   - Map category_id to contiguous class index using self.coco_id_to_class
        boxes = []
        labels = []
        
        # IMPLEMENT BOX AND LABEL EXTRACTION HERE
        
        # TODO 1.7: Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, boxes, labels, img_id


def collate_fn(batch):
    """
    Custom collate function for variable number of boxes per image
    
    TODO 1.8: Implement collate function
    - Stack images into a batch tensor
    - Keep boxes and labels as lists (they have different lengths per image)
    
    Args:
        batch: List of (image, boxes, labels, img_id) tuples
    
    Returns:
        images: Tensor [B, 3, H, W]
        boxes: List of [N_i, 4] tensors
        labels: List of [N_i] tensors
        img_ids: List of image IDs
    """
    # IMPLEMENT HERE
    pass


def get_transforms(is_train=True, input_size=448):
    """Get data augmentation transforms"""
    if is_train:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_iou(box1, box2, box_format='xyxy'):
    """
    Compute IoU between two boxes or batches of boxes
    
    TODO 2.1: Implement IoU computation
    
    Steps:
    1. If box_format is 'xywh', convert to 'xyxy'
    2. Extract coordinates: x1_min, y1_min, x1_max, y1_max for box1
    3. Extract coordinates: x2_min, y2_min, x2_max, y2_max for box2
    4. Compute intersection rectangle:
       - inter_xmin = max(x1_min, x2_min)
       - inter_ymin = max(y1_min, y2_min)
       - inter_xmax = min(x1_max, x2_max)
       - inter_ymax = min(y1_max, y2_max)
    5. Compute intersection area:
       - inter_w = clamp(inter_xmax - inter_xmin, min=0)
       - inter_h = clamp(inter_ymax - inter_ymin, min=0)
       - inter_area = inter_w * inter_h
    6. Compute union area:
       - box1_area = (x1_max - x1_min) * (y1_max - y1_min)
       - box2_area = (x2_max - x2_min) * (y2_max - y2_min)
       - union_area = box1_area + box2_area - inter_area
    7. Return IoU = inter_area / (union_area + 1e-6)
    
    Args:
        box1: Tensor [..., 4]
        box2: Tensor [..., 4]
        box_format: 'xyxy' or 'xywh'
    
    Returns:
        iou: Tensor [...] with IoU values
    """
    # IMPLEMENT HERE
    pass


def xywh_to_xyxy(boxes):
    """
    Convert boxes from (x, y, w, h) to (x1, y1, x2, y2) format
    
    TODO 2.2: Implement conversion
    Where (x, y) is center, (w, h) is width and height
    
    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2
    """
    # IMPLEMENT HERE
    pass


def xyxy_to_xywh(boxes):
    """
    Convert boxes from (x1, y1, x2, y2) to (x, y, w, h) format
    
    TODO 2.3: Implement conversion
    Where (x, y) is center, (w, h) is width and height
    
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    """
    # IMPLEMENT HERE
    pass


def nms(boxes, scores, iou_threshold=0.5):
    """
    Non-maximum suppression
    
    TODO 2.4: Implement NMS algorithm
    
    Steps:
    1. Handle empty input case
    2. Sort boxes by scores in descending order
    3. Initialize keep list
    4. While there are remaining boxes:
       a. Take box with highest score, add to keep list
       b. Compute IoU between this box and all remaining boxes
       c. Remove boxes with IoU > threshold
    5. Return indices of kept boxes
    
    Args:
        boxes: Tensor [N, 4] in xyxy format
        scores: Tensor [N]
        iou_threshold: IoU threshold for suppression
    
    Returns:
        keep: Tensor with indices of kept boxes
    """
    # IMPLEMENT HERE
    pass


# ============================================================================
# YOLO MODEL
# ============================================================================

class YOLODetector(nn.Module):
    """YOLO-style object detector"""
    
    def __init__(self, num_classes=80, grid_size=7, num_boxes=2, backbone='resnet18'):
        """
        TODO 3.1: Initialize YOLO detector
        
        Architecture:
        1. Backbone: ResNet (remove final FC layer)
        2. Adaptive pooling to get grid_size x grid_size feature map
        3. Detection head: Conv layers to output (B*5 + C) channels
        
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
        
        # TODO 3.1.1: Create backbone
        # Hint: Load pretrained ResNet, remove last 2 layers (avgpool and fc)
        # Get feature dimension (512 for resnet18, 2048 for resnet50)
        
        # TODO 3.1.2: Create detection head
        # Output channels = num_boxes * 5 + num_classes
        # Use Conv2d layers with BatchNorm and LeakyReLU
        
        # TODO 3.1.3: Create adaptive pooling layer
        # Use nn.AdaptiveAvgPool2d to get grid_size x grid_size output
        
        # IMPLEMENT HERE
        pass
    
    def forward(self, x):
        """
        Forward pass
        
        TODO 3.2: Implement forward pass
        
        Steps:
        1. Pass through backbone to get features
        2. Apply adaptive pooling to get [B, C, S, S]
        3. Pass through detection head to get [B, B*5+C, S, S]
        4. Reshape to [B, S, S, B, 5+C]
        5. Apply activations:
           - Sigmoid on x, y (box center coordinates)
           - Sigmoid on confidence
           - Softmax on class probabilities
        
        Args:
            x: Input images [B, 3, H, W]
        
        Returns:
            predictions: Tensor [B, S, S, B, 5+C]
                where last dim = [x, y, w, h, conf, class_probs...]
        """
        # IMPLEMENT HERE
        pass


class YOLOLoss(nn.Module):
    """YOLO loss function"""
    
    def __init__(self, num_classes=80, grid_size=7, num_boxes=2,
                 lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
    
    def forward(self, predictions, targets_boxes, targets_labels):
        """
        Compute YOLO loss
        
        TODO 3.3: Implement YOLO loss function
        
        Steps:
        1. Encode target boxes to grid format using encode_targets()
        2. Create object and no-object masks
        3. Determine responsible boxes using get_responsible_boxes()
        4. Compute localization loss (x, y, sqrt(w), sqrt(h))
        5. Compute confidence loss for objects
        6. Compute confidence loss for no objects
        7. Compute classification loss
        8. Weight and sum all losses
        
        Loss formula:
        L = λ_coord * L_coord + L_conf_obj + λ_noobj * L_conf_noobj + L_class
        
        Args:
            predictions: Tensor [B, S, S, B, 5+C]
            targets_boxes: List of [N_i, 4] tensors (normalized xyxy)
            targets_labels: List of [N_i] tensors
        
        Returns:
            loss: Scalar tensor
            loss_dict: Dictionary with individual loss components
        """
        # IMPLEMENT HERE
        pass
    
    def encode_targets(self, targets_boxes, targets_labels, batch_size, device):
        """
        Encode ground truth boxes to grid format
        
        TODO 3.4: Implement target encoding
        
        Steps:
        1. Create target grid tensor [B, S, S, 5+C] initialized to zeros
        2. For each image in batch:
           a. For each ground truth box:
              - Convert box from xyxy to xywh format
              - Find which grid cell contains the center: i = int(y * S), j = int(x * S)
              - Compute offsets relative to cell: x_cell = x * S - j, y_cell = y * S - i
              - Set target_grid[b, i, j, 0:2] = (x_cell, y_cell)
              - Set target_grid[b, i, j, 2:4] = (w, h)
              - Set target_grid[b, i, j, 4] = 1.0 (confidence)
              - Set target_grid[b, i, j, 5 + label] = 1.0 (one-hot class)
        3. Return target grid
        
        Returns:
            target_grid: Tensor [B, S, S, 5+C]
        """
        # IMPLEMENT HERE
        pass
    
    def get_responsible_boxes(self, predictions, target_xy, target_wh, target_conf):
        """
        Determine which box in each cell is responsible for prediction
        
        TODO 3.5: Implement responsible box selection
        
        Steps:
        1. For each grid cell with an object:
           a. Compute IoU between each predicted box and ground truth
           b. Select box with highest IoU as responsible
        2. Create binary mask [B, S, S, B] with 1 for responsible boxes
        
        Returns:
            responsible_mask: Tensor [B, S, S, B]
        """
        # IMPLEMENT HERE
        pass


# ============================================================================
# FCOS MODEL
# ============================================================================

class FPN(nn.Module):
    """Feature Pyramid Network"""
    
    def __init__(self, in_channels_list, out_channels=256):
        """
        TODO 4.1: Initialize FPN
        
        Architecture:
        1. Lateral connections: 1x1 conv to reduce channels
        2. Output convolutions: 3x3 conv after upsampling
        
        Args:
            in_channels_list: List of input channels [C3, C4, C5]
            out_channels: Output channels for all levels
        """
        super().__init__()
        
        self.out_channels = out_channels
        
        # TODO 4.1.1: Create lateral convolutions
        # One 1x1 conv for each input level to reduce to out_channels
        
        # TODO 4.1.2: Create output convolutions
        # One 3x3 conv for each level after upsampling and addition
        
        # IMPLEMENT HERE
        pass
    
    def forward(self, inputs):
        """
        Forward pass through FPN
        
        TODO 4.2: Implement FPN forward pass
        
        Steps:
        1. Apply lateral convolutions to all inputs
        2. Build top-down pathway:
           a. Start from deepest level (last input)
           b. For each level from deep to shallow:
              - Upsample previous level (use F.interpolate with mode='nearest')
              - Add with lateral connection
              - Apply output convolution
        3. Return list of feature maps [P3, P4, P5]
        
        Args:
            inputs: List of feature maps [C3, C4, C5]
        
        Returns:
            outputs: List of FPN feature maps [P3, P4, P5]
        """
        # IMPLEMENT HERE
        pass


class FCOSHead(nn.Module):
    """FCOS detection head"""
    
    def __init__(self, in_channels, num_classes, num_convs=4):
        """
        TODO 4.3: Initialize FCOS head
        
        Architecture:
        1. Classification tower: num_convs layers of (Conv3x3 + GroupNorm + ReLU)
        2. Classification output: Conv3x3 to num_classes channels
        3. Regression tower: num_convs layers of (Conv3x3 + GroupNorm + ReLU)
        4. Regression output: Conv3x3 to 4 channels (l, t, r, b)
        5. Center-ness output: Conv3x3 to 1 channel
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # TODO 4.3.1: Build classification tower and head
        
        # TODO 4.3.2: Build regression tower and head
        
        # TODO 4.3.3: Build center-ness head
        
        # TODO 4.3.4: Initialize weights
        # Use normal initialization with std=0.01 for Conv2d
        
        # IMPLEMENT HERE
        pass
    
    def forward(self, x):
        """
        TODO 4.4: Implement FCOS head forward pass
        
        Steps:
        1. Pass through classification tower and get class logits
        2. Pass through regression tower and get box predictions (apply ReLU)
        3. Get center-ness prediction
        
        Returns:
            cls_logits: [B, num_classes, H, W]
            reg_pred: [B, 4, H, W]
            centerness: [B, 1, H, W]
        """
        # IMPLEMENT HERE
        pass


class FCOSDetector(nn.Module):
    """FCOS anchor-free object detector"""
    
    def __init__(self, num_classes=80, backbone='resnet50'):
        """
        TODO 4.5: Initialize FCOS detector
        
        Architecture:
        1. ResNet backbone to extract multi-scale features [C3, C4, C5]
        2. FPN to build feature pyramid [P3, P4, P5]
        3. Shared FCOS head for all pyramid levels
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.strides = [8, 16, 32]
        
        # TODO 4.5.1: Create backbone layers
        # Extract C3, C4, C5 from ResNet
        
        # TODO 4.5.2: Create FPN
        
        # TODO 4.5.3: Create FCOS head
        
        # IMPLEMENT HERE
        pass
    
    def forward(self, x):
        """
        TODO 4.6: Implement FCOS forward pass
        
        Steps:
        1. Extract multi-scale features from backbone [C3, C4, C5]
        2. Apply FPN to get [P3, P4, P5]
        3. Apply FCOS head to each pyramid level
        4. Return lists of predictions for each level
        
        Returns:
            cls_logits_list: List of [B, C, H_i, W_i]
            reg_preds_list: List of [B, 4, H_i, W_i]
            centerness_list: List of [B, 1, H_i, W_i]
        """
        # IMPLEMENT HERE
        pass


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        """
        Compute focal loss
        
        TODO 4.7: Implement focal loss
        
        Formula: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
        
        Steps:
        1. Compute probabilities: p = sigmoid(inputs)
        2. Compute p_t: p_t = p * targets + (1 - p) * (1 - targets)
        3. Compute cross-entropy loss
        4. Compute focal weight: (1 - p_t)^gamma
        5. Apply alpha weighting
        6. Return mean loss
        
        Args:
            inputs: Predictions [N, C]
            targets: Ground truth [N, C] (one-hot or soft labels)
        
        Returns:
            loss: Scalar
        """
        # IMPLEMENT HERE
        pass


class FCOSLoss(nn.Module):
    """FCOS loss function"""
    
    def __init__(self, num_classes=80, strides=[8, 16, 32],
                 scale_ranges=[(0, 64), (64, 128), (128, float('inf'))]):
        super().__init__()
        
        self.num_classes = num_classes
        self.strides = strides
        self.scale_ranges = scale_ranges
        
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
    
    def forward(self, cls_logits_list, reg_preds_list, centerness_list,
                targets_boxes, targets_labels):
        """
        Compute FCOS loss
        
        TODO 4.8: Implement FCOS loss computation
        
        Steps:
        1. For each FPN level:
           a. Compute locations (pixel coordinates in input image)
           b. Assign targets to locations based on:
              - Location inside box
              - Box size within scale range for this level
           c. Collect classification, regression, and centerness targets
        2. Compute classification loss (focal loss)
        3. Compute regression loss (GIoU loss) for positive samples
        4. Compute centerness loss (BCE) for positive samples
        5. Return total loss and loss dictionary
        
        Args:
            cls_logits_list: List of [B, C, H, W]
            reg_preds_list: List of [B, 4, H, W]
            centerness_list: List of [B, 1, H, W]
            targets_boxes: List of [N_i, 4] (xyxy, normalized)
            targets_labels: List of [N_i]
        
        Returns:
            loss: Total loss
            loss_dict: Dictionary of individual losses
        """
        # IMPLEMENT HERE
        pass
    
    def compute_locations(self, h, w, stride, device):
        """
        Compute center locations for each position in feature map
        
        TODO 4.9: Implement location computation
        
        Steps:
        1. Create meshgrid of positions: shifts_x = [0, stride, 2*stride, ...]
        2. Add stride//2 to get center coordinates
        3. Return [H*W, 2] tensor of (x, y) coordinates
        
        Returns:
            locations: [H*W, 2] (x, y in input image coordinates)
        """
        # IMPLEMENT HERE
        pass
    
    def assign_targets_single_level(self, locations, targets_boxes, targets_labels,
                                    stride, scale_range, batch_size):
        """
        Assign targets to locations for a single FPN level
        
        TODO 4.10: Implement target assignment
        
        Steps:
        1. For each image in batch:
           a. Compute (l, t, r, b) distances from each location to each box
           b. Check if location is inside box (all distances > 0)
           c. Check if box size is within scale range for this level
           d. For each location, assign to box with smallest area (if valid)
           e. Compute centerness target: sqrt((min(l,r)/max(l,r)) * (min(t,b)/max(t,b)))
        2. Return classification targets, regression targets, centerness targets, positive masks
        
        Returns:
            cls_targets: [B, L] with class indices (num_classes for background)
            reg_targets: [B, L, 4] with (l, t, r, b) distances
            centerness_targets: [B, L]
            pos_masks: [B, L] binary mask for positive samples
        """
        # IMPLEMENT HERE
        pass
    
    def giou_loss(self, pred, target):
        """
        Compute Generalized IoU loss
        
        TODO 4.11: Implement GIoU loss
        
        Formula: GIoU = IoU - (area_enclosing - area_union) / area_enclosing
        Loss = 1 - GIoU
        
        Steps:
        1. Convert ltrb format to xyxy
        2. Compute IoU
        3. Compute enclosing box
        4. Compute GIoU
        5. Return 1 - GIoU
        
        Args:
            pred: [N, 4] in ltrb format
            target: [N, 4] in ltrb format
        
        Returns:
            loss: [N] GIoU loss values
        """
        # IMPLEMENT HERE
        pass


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Train for one epoch
    
    TODO 5.1: Implement training loop
    
    Steps:
    1. Set model to train mode
    2. For each batch:
       a. Move data to device
       b. Forward pass
       c. Compute loss
       d. Backward pass
       e. Clip gradients (max_norm=10.0)
       f. Update weights
       g. Log progress
    3. Return average loss
    """
    # IMPLEMENT HERE
    pass


def evaluate_model(model, val_loader, coco_gt, device):
    """
    Evaluate model on validation set
    
    TODO 5.2: Implement evaluation
    
    Steps:
    1. Set model to eval mode
    2. For each batch:
       a. Get predictions
       b. Decode predictions to bounding boxes
       c. Convert to COCO format
    3. Use COCOeval to compute metrics
    4. Return metric dictionary
    """
    # IMPLEMENT HERE
    pass


def decode_yolo_predictions(predictions, conf_threshold=0.05, nms_threshold=0.5):
    """
    Decode YOLO predictions to bounding boxes
    
    TODO 5.3: Implement YOLO prediction decoding
    
    Steps:
    1. For each image in batch:
       a. For each grid cell and each box:
          - Extract x, y, w, h, confidence, class probabilities
          - Convert grid-relative coordinates to absolute
          - Compute final score = confidence * max_class_prob
          - Filter by confidence threshold
       b. Convert boxes from xywh to xyxy
       c. Apply NMS
    2. Return list of detections per image
    
    Args:
        predictions: [B, S, S, B, 5+C]
        conf_threshold: Confidence threshold
        nms_threshold: NMS IoU threshold
    
    Returns:
        detections: List of lists of (box, score, label) tuples
    """
    # IMPLEMENT HERE
    pass


def decode_fcos_predictions(cls_logits_list, reg_preds_list, centerness_list,
                           strides, conf_threshold=0.05, nms_threshold=0.5):
    """
    Decode FCOS predictions to bounding boxes
    
    TODO 5.4: Implement FCOS prediction decoding
    
    Steps:
    1. For each FPN level:
       a. Apply sigmoid to class logits and centerness
       b. Compute locations
       c. Convert ltrb to xyxy boxes
       d. Multiply class scores with centerness
       e. Filter by confidence threshold
    2. Concatenate detections from all levels
    3. Apply NMS
    4. Return detections
    
    Returns:
        detections: List of lists of (box, score, label) tuples
    """
    # IMPLEMENT HERE
    pass


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    """
    Main training function
    
    TODO 6.1: Complete the main training pipeline
    
    Steps:
    1. Create datasets and dataloaders
    2. Initialize model and loss function
    3. Create optimizer and scheduler
    4. Training loop:
       a. Train for one epoch
       b. Validate every N epochs
       c. Save best model
       d. Update learning rate
    5. Print final results
    """
    
    # TODO 6.1.1: Create datasets
    train_dataset = None  # CREATE TRAIN DATASET
    val_dataset = None  # CREATE VAL DATASET
    
    # TODO 6.1.2: Create dataloaders
    train_loader = None  # CREATE TRAIN LOADER
    val_loader = None  # CREATE VAL LOADER
    
    # TODO 6.1.3: Choose and initialize model
    model_type = 'fcos'  # or 'yolo'
    model = None  # CREATE MODEL
    criterion = None  # CREATE LOSS FUNCTION
    
    # TODO 6.1.4: Create optimizer
    optimizer = None  # CREATE OPTIMIZER
    
    # TODO 6.1.5: Create learning rate scheduler
    scheduler = None  # CREATE SCHEDULER
    
    # TODO 6.1.6: Implement training loop
    best_map = 0
    
    for epoch in range(config.num_epochs):
        # Train
        train_loss = None  # CALL train_one_epoch
        
        print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}')
        
        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0:
            metrics = None  # CALL evaluate_model
            print(f'Validation: mAP={metrics["mAP"]:.3f}')
            
            # Save best model
            if metrics['mAP'] > best_map:
                best_map = metrics['mAP']
                # SAVE CHECKPOINT
        
        # Update learning rate
        # CALL scheduler.step()
    
    print(f'Training complete. Best mAP: {best_map:.3f}')


# ============================================================================
# TEST FUNCTIONS (Use these to verify your implementations)
# ============================================================================

def test_dataset():
    """Test dataset implementation"""
    print("Testing dataset...")
    
    # Create dataset
    dataset = COCODetectionDataset(
        config.train_images,
        config.train_ann,
        transform=get_transforms(is_train=True),
        is_train=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get one sample
    image, boxes, labels, img_id = dataset[0]
    
    print(f"Image shape: {image.shape}")
    print(f"Number of boxes: {len(boxes)}")
    print(f"Boxes shape: {boxes.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image ID: {img_id}")
    
    # Verify boxes are in correct format
    assert image.shape == (3, 448, 448), "Image shape incorrect"
    assert boxes.ndim == 2 and boxes.shape[1] == 4, "Boxes shape incorrect"
    assert labels.ndim == 1, "Labels shape incorrect"
    assert torch.all((boxes >= 0) & (boxes <= 1)), "Boxes not normalized"
    
    print("✓ Dataset test passed!")


def test_iou():
    """Test IoU computation"""
    print("Testing IoU...")
    
    # Test case 1: Perfect overlap
    box1 = torch.tensor([[0.2, 0.2, 0.8, 0.8]])
    box2 = torch.tensor([[0.2, 0.2, 0.8, 0.8]])
    iou = compute_iou(box1, box2)
    assert torch.isclose(iou, torch.tensor(1.0)), f"Perfect overlap should be 1.0, got {iou}"
    
    # Test case 2: No overlap
    box1 = torch.tensor([[0.0, 0.0, 0.3, 0.3]])
    box2 = torch.tensor([[0.7, 0.7, 1.0, 1.0]])
    iou = compute_iou(box1, box2)
    assert torch.isclose(iou, torch.tensor(0.0)), f"No overlap should be 0.0, got {iou}"
    
    # Test case 3: Partial overlap
    box1 = torch.tensor([[0.0, 0.0, 0.5, 0.5]])
    box2 = torch.tensor([[0.25, 0.25, 0.75, 0.75]])
    iou = compute_iou(box1, box2)
    expected = 0.25 * 0.25 / (0.5 * 0.5 + 0.5 * 0.5 - 0.25 * 0.25)
    assert torch.isclose(iou, torch.tensor(expected), atol=1e-4), f"IoU incorrect: {iou} vs {expected}"
    
    print("✓ IoU test passed!")


def test_nms():
    """Test NMS implementation"""
    print("Testing NMS...")
    
    # Create overlapping boxes with different scores
    boxes = torch.tensor([
        [0.1, 0.1, 0.4, 0.4],  # High score
        [0.15, 0.15, 0.45, 0.45],  # Overlaps with first, lower score
        [0.6, 0.6, 0.9, 0.9],  # Different location, medium score
    ])
    scores = torch.tensor([0.9, 0.7, 0.8])
    
    keep = nms(boxes, scores, iou_threshold=0.5)
    
    # Should keep first and third box
    assert len(keep) == 2, f"Should keep 2 boxes, kept {len(keep)}"
    assert 0 in keep, "Should keep highest scoring box"
    assert 2 in keep, "Should keep non-overlapping box"
    
    print("✓ NMS test passed!")


def test_yolo_model():
    """Test YOLO model"""
    print("Testing YOLO model...")
    
    model = YOLODetector(num_classes=80, grid_size=7, num_boxes=2)
    
    # Test forward pass
    x = torch.randn(2, 3, 448, 448)
    predictions = model(x)
    
    assert predictions.shape == (2, 7, 7, 2, 85), f"Output shape incorrect: {predictions.shape}"
    
    # Check activations
    assert torch.all((predictions[..., 0:2] >= 0) & (predictions[..., 0:2] <= 1)), "x, y not in [0,1]"
    assert torch.all((predictions[..., 4] >= 0) & (predictions[..., 4] <= 1)), "confidence not in [0,1]"
    assert torch.allclose(predictions[..., 5:].sum(dim=-1), torch.ones(2, 7, 7, 2)), "class probs don't sum to 1"
    
    print("✓ YOLO model test passed!")


def test_fcos_model():
    """Test FCOS model"""
    print("Testing FCOS model...")
    
    model = FCOSDetector(num_classes=80)
    
    # Test forward pass
    x = torch.randn(2, 3, 448, 448)
    cls_logits, reg_preds, centerness = model(x)
    
    assert len(cls_logits) == 3, "Should have 3 FPN levels"
    assert len(reg_preds) == 3, "Should have 3 FPN levels"
    assert len(centerness) == 3, "Should have 3 FPN levels"
    
    for cls, reg, cent in zip(cls_logits, reg_preds, centerness):
        assert cls.size(1) == 80, "Classification channels incorrect"
        assert reg.size(1) == 4, "Regression channels incorrect"
        assert cent.size(1) == 1, "Centerness channels incorrect"
    
    print("✓ FCOS model test passed!")


def run_all_tests():
    """Run all test functions"""
    print("="*50)
    print("Running all tests...")
    print("="*50)
    
    test_dataset()
    test_iou()
    test_nms()
    test_yolo_model()
    test_fcos_model()
    
    print("="*50)
    print("All tests passed! ✓")
    print("="*50)


if __name__ == '__main__':
    # Uncomment to run tests
    # run_all_tests()
    
    # Uncomment to start training
    # main()
    
    print("Lab 6 Template Ready!")
    print("\nNext steps:")
    print("1. Complete the TODOs in order")
    print("2. Run tests after each major component: run_all_tests()")
    print("3. Start training: main()")
    print("\nGood luck!")