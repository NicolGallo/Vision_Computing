"""
Configuration file for Lab 6: Object Detection
"""
from pathlib import Path
import torch

class Config:
    """Configuration for training and evaluation"""
    
    # ==================== Paths ====================
    data_root = Path('data/coco')
    train_images = data_root / 'train2017'
    val_images = data_root / 'val2017'
    train_ann = data_root / 'annotations/instances_train2017.json'
    val_ann = data_root / 'annotations/instances_val2017.json'
    
    checkpoint_dir = Path('checkpoints')
    log_dir = Path('logs')
    output_dir = Path('outputs')
    
    # Create directories
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    # ==================== Model ====================
    num_classes = 80
    input_size = 448
    
    # YOLO specific
    grid_size = 7
    num_boxes = 2
    yolo_backbone = 'resnet18'  # 'resnet18' or 'resnet50'
    
    # FCOS specific
    fpn_strides = [8, 16, 32]
    fpn_scales = [(0, 64), (64, 128), (128, float('inf'))]
    fcos_backbone = 'resnet50'  # 'resnet18' or 'resnet50'
    fpn_channels = 256
    num_convs = 4
    
    # ==================== Training ====================
    batch_size = 8
    num_epochs = 20
    learning_rate = 1e-4
    weight_decay = 5e-4
    num_workers = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Learning rate schedule
    warmup_epochs = 1
    warmup_lr = 1e-6
    
    # Gradient clipping
    max_grad_norm = 10.0
    
    # ==================== Loss Weights ====================
    # YOLO loss weights
    lambda_coord = 5.0
    lambda_noobj = 0.5
    lambda_cls = 1.0
    lambda_conf = 1.0
    
    # FCOS loss weights
    lambda_cls_fcos = 1.0
    lambda_reg = 1.0
    lambda_centerness = 1.0
    
    # Focal loss parameters
    focal_alpha = 0.25
    focal_gamma = 2.0
    
    # ==================== Inference ====================
    nms_threshold = 0.5
    conf_threshold = 0.05
    max_detections = 100
    
    # ==================== Logging ====================
    log_interval = 100
    val_interval = 5  # Validate every N epochs
    save_interval = 5  # Save checkpoint every N epochs
    
    # ==================== Data Augmentation ====================
    # Image normalization (ImageNet stats)
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    
    # Training augmentation
    horizontal_flip_prob = 0.5
    color_jitter = {
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.2,
        'hue': 0.0
    }
    
    # ==================== Evaluation ====================
    eval_batch_size = 4
    visualize_predictions = True
    num_vis_samples = 10

# Create global config instance
config = Config()