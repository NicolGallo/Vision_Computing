"""
Data augmentation transforms for object detection
"""
from torchvision import transforms


def get_transforms(is_train=True, input_size=448, img_mean=None, img_std=None):
    """
    Get data augmentation transforms
    
    Args:
        is_train: Whether for training (includes augmentation)
        input_size: Input image size
        img_mean: Mean for normalization (default: ImageNet mean)
        img_std: Std for normalization (default: ImageNet std)
    
    Returns:
        transform: Composed transforms
    """
    if img_mean is None:
        img_mean = [0.485, 0.456, 0.406]
    if img_std is None:
        img_std = [0.229, 0.224, 0.225]
    
    if is_train:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean, std=img_std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean, std=img_std)
        ])