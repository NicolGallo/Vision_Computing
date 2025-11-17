"""
Lab 4: Utility functions for data loading and metrics
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import cv2

# ================== Metrics ==================

def calculate_metrics(pred, target, threshold=0.5):
    """
    Calculate multiple segmentation metrics
    
    Args:
        pred: Model predictions (logits)
        target: Ground truth masks
        threshold: Threshold for binary classification
    
    Returns:
        Dictionary with metrics
    """
    # Apply sigmoid and threshold
    pred = torch.sigmoid(pred)
    pred_binary = (pred > threshold).float()
    
    # Flatten tensors
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    
    # Calculate metrics
    tp = (pred_flat * target_flat).sum()
    fp = pred_flat.sum() - tp
    fn = target_flat.sum() - tp
    tn = (1 - pred_flat).sum() - fn
    
    # IoU
    iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
    
    # Dice
    dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    
    # Pixel accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # Precision and Recall
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    
    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    return {
        'iou': iou.item(),
        'dice': dice.item(),
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
    }


def boundary_f1_score(pred, target, threshold=0.5, dilation=2):
    """
    Calculate boundary F1 score for segmentation quality
    """
    # Convert to numpy
    pred = torch.sigmoid(pred).cpu().numpy()
    target = target.cpu().numpy()
    
    # Threshold prediction
    pred_binary = (pred > threshold).astype(np.uint8)
    target_binary = target.astype(np.uint8)
    
    # Get boundaries using morphological operations
    kernel = np.ones((dilation*2+1, dilation*2+1), np.uint8)
    
    pred_boundary = cv2.dilate(pred_binary[0, 0], kernel, iterations=1) - \
                   cv2.erode(pred_binary[0, 0], kernel, iterations=1)
    target_boundary = cv2.dilate(target_binary[0, 0], kernel, iterations=1) - \
                     cv2.erode(target_binary[0, 0], kernel, iterations=1)
    
    # Calculate F1 on boundaries
    tp = np.sum(pred_boundary * target_boundary)
    fp = np.sum(pred_boundary) - tp
    fn = np.sum(target_boundary) - tp
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return f1


# ================== Data Augmentation ==================

class SegmentationAugmentation:
    """
    Data augmentation for segmentation tasks
    Applies same transform to both image and mask
    """
    
    def __init__(self, image_size=128, mode='train'):
        self.image_size = image_size
        self.mode = mode
        
    def __call__(self, image, mask):
        # Resize
        image = transforms.functional.resize(image, (self.image_size, self.image_size))
        mask = transforms.functional.resize(mask, (self.image_size, self.image_size))
        
        if self.mode == 'train':
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)
            
            # Random rotation
            angle = torch.randint(-15, 15, (1,)).item()
            image = transforms.functional.rotate(image, angle)
            mask = transforms.functional.rotate(mask, angle)
            
            # Color jitter (only for image)
            jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                           saturation=0.2, hue=0.1)
            image = jitter(image)
        
        # Convert to tensor if not already
        if not isinstance(image, torch.Tensor):
            image = transforms.functional.to_tensor(image)
        if not isinstance(mask, torch.Tensor):
            mask = transforms.functional.to_tensor(mask)
        
        # Normalize image
        image = transforms.functional.normalize(image, 
                                               mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])
        
        return image, mask


# ================== Visualization ==================

def plot_segmentation_results(images, masks, predictions, title="Segmentation Results"):
    """
    Plot segmentation results in a grid
    
    Args:
        images: Input images (B, C, H, W)
        masks: Ground truth masks (B, 1, H, W)
        predictions: Model predictions (B, 1, H, W)
        title: Figure title
    """
    batch_size = images.shape[0]
    num_samples = min(4, batch_size)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Input image
        img = images[i].cpu()
        # Denormalize
        img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cpu()
        img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cpu()
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0).numpy()
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')
        
        # Ground truth
        mask_tensor = masks[i]
        if mask_tensor.dim() == 3 and mask_tensor.shape[0] == 1:
            mask_tensor = mask_tensor.squeeze(0)
        mask_np = mask_tensor.cpu().numpy()
        if mask_tensor.dtype.is_floating_point():
            axes[i, 1].imshow(mask_np, cmap='gray')
        else:
            mask_max = float(np.nanmax(mask_np)) if mask_np.size else 1.0
            vmax = max(1, int(mask_max))
            axes[i, 1].imshow(mask_np, cmap='tab20', vmin=0, vmax=vmax)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Prediction
        pred_tensor = predictions[i]
        if pred_tensor.dim() == 3 and pred_tensor.shape[0] == 1:
            pred_tensor = pred_tensor.squeeze(0)
        if pred_tensor.dtype.is_floating_point() and pred_tensor.dim() == 3:
            pred_tensor = torch.softmax(pred_tensor, dim=0).argmax(dim=0)
        if pred_tensor.dtype.is_floating_point() and pred_tensor.dim() == 2:
            pred_np = torch.sigmoid(pred_tensor).cpu().numpy()
            axes[i, 2].imshow(pred_np, cmap='gray')
        else:
            pred_np = pred_tensor.cpu().numpy()
            pred_max = float(np.nanmax(pred_np)) if pred_np.size else 1.0
            vmax = max(1, int(pred_max))
            axes[i, 2].imshow(pred_np, cmap='tab20', vmin=0, vmax=vmax)
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig


def visualize_feature_maps(model, image, layer_names=None):
    """
    Visualize intermediate feature maps from the model
    
    Args:
        model: The neural network model
        image: Input image (1, C, H, W)
        layer_names: List of layer names to visualize
    """
    activations = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks
    hooks = []
    if layer_names is None:
        layer_names = ['encoders.0', 'encoders.1', 'bottleneck', 'decoders.0']
    
    for name, layer in model.named_modules():
        if any(ln in name for ln in layer_names):
            hook = layer.register_forward_hook(hook_fn(name))
            hooks.append(hook)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(image)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Plot activations
    fig, axes = plt.subplots(len(activations), 8, figsize=(16, 2*len(activations)))
    
    for idx, (name, activation) in enumerate(activations.items()):
        # Select first 8 channels
        act = activation[0, :8].cpu()
        
        for i in range(min(8, act.shape[0])):
            ax = axes[idx, i] if len(activations) > 1 else axes[i]
            ax.imshow(act[i], cmap='viridis')
            ax.axis('off')
            if i == 0:
                ax.set_ylabel(name, rotation=90)
    
    plt.suptitle('Feature Map Visualization')
    plt.tight_layout()
    return fig


def plot_gradient_flow(model):
    """
    Plot gradient flow through the network layers
    Useful for debugging vanishing/exploding gradients
    """
    ave_grads = []
    layers = []
    
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
    
    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.7)
    plt.xticks(np.arange(len(ave_grads)), layers, rotation=45, ha='right')
    plt.xlabel('Layers')
    plt.ylabel('Average Gradient Magnitude')
    plt.title('Gradient Flow Through Network')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ================== Memory Profiling ==================

def profile_memory(model, input_shape=(1, 3, 128, 128), device='cuda'):
    """
    Profile memory usage of the model
    
    Args:
        model: The neural network model
        input_shape: Shape of input tensor
        device: Device to run on
    
    Returns:
        Dictionary with memory statistics
    """
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    model = model.to(device)
    dummy_input = torch.randn(input_shape).to(device)
    
    # Forward pass
    start_mem = torch.cuda.memory_allocated() if device == 'cuda' else 0
    output = model(dummy_input)
    forward_mem = torch.cuda.memory_allocated() if device == 'cuda' else 0
    
    # Backward pass
    loss = output.mean()
    loss.backward()
    backward_mem = torch.cuda.max_memory_allocated() if device == 'cuda' else 0
    
    # Calculate statistics
    stats = {
        'model_params': sum(p.numel() for p in model.parameters()),
        'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2,
        'forward_memory_mb': (forward_mem - start_mem) / 1024**2 if device == 'cuda' else 0,
        'peak_memory_mb': backward_mem / 1024**2 if device == 'cuda' else 0,
        'output_shape': list(output.shape)
    }
    
    return stats


# ================== Learning Rate Scheduling ==================

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """
    Create a learning rate scheduler with linear warmup and cosine decay
    
    Args:
        optimizer: The optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
    
    Returns:
        Learning rate scheduler
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / \
                   float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ================== Export Functions ==================

def export_onnx(model, save_path='unet_model.onnx', input_shape=(1, 3, 128, 128)):
    """
    Export model to ONNX format for deployment
    
    Args:
        model: PyTorch model
        save_path: Path to save ONNX model
        input_shape: Input tensor shape
    """
    model.eval()
    dummy_input = torch.randn(input_shape)
    
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model exported to {save_path}")


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test metrics
    pred = torch.randn(1, 1, 128, 128)
    target = torch.randint(0, 2, (1, 1, 128, 128)).float()
    metrics = calculate_metrics(pred, target)
    print("Metrics:", metrics)
    
    # Test augmentation
    aug = SegmentationAugmentation()
    img = torch.randn(3, 256, 256)
    mask = torch.randint(0, 2, (1, 256, 256)).float()
    img_aug, mask_aug = aug(img, mask)
    print(f"Augmented shapes - Image: {img_aug.shape}, Mask: {mask_aug.shape}")
    
    print("Utilities tested successfully!")
