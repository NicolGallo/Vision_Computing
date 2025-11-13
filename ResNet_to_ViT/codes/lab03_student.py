"""
Lab 3: Architecture Implementation Mastery - Student Template
VIAR25/26 - Artificial Vision - UVigo
Prof. David Olivieri

This template provides the structure for implementing:
1. ResNet architectures (Basic and Bottleneck blocks)
2. Attention mechanisms (SE and CBAM)
3. Vision Transformer components
4. Benchmarking and comparison tools

TODO: Complete all functions marked with "TODO" comments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: RESNET IMPLEMENTATION
# ============================================================================

class BasicBlock(nn.Module):
    """
    Basic ResNet block with two 3x3 convolutions
    Used in ResNet-18 and ResNet-34
    """
    expansion = 1
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, 
                 downsample: Optional[nn.Module] = None):
        super(BasicBlock, self).__init__()
        
        # Implement the basic block
        # Hint: You need:
        # - First 3x3 conv with given stride
        # - BatchNorm + ReLU
        # - Second 3x3 conv with stride=1
        # - BatchNorm (no ReLU here)
        # - Store downsample for skip connection
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)  # TODO: Replace with actual implementation
        self.bn1 = nn.BatchNorm2d(out_channels)    # TODO: Replace with actual implementation
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)  # TODO: Replace with actual implementation
        self.bn2 = nn.BatchNorm2d(out_channels)    # TODO: Replace with actual implementation
        self.downsample = downsample
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement forward pass
        # Remember: output = F(x) + x (or F(x) + downsample(x))
        
        identity = x
        
        # TODO: Apply first conv + bn + relu
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # TODO: Apply second conv + bn (no relu)
        out = self.conv2(out)
        out = self.bn2(out)
        
        # TODO: Apply skip connection
        if self.downsample is not None:
            identity = self.downsample(x)  # TODO: Apply downsample to identity
            
        # TODO: Add identity and apply final ReLU
        out += identity
        out = F.relu(out)
        
        return out


class BottleneckBlock(nn.Module):
    """
    Bottleneck ResNet block with 1x1 -> 3x3 -> 1x1 convolutions
    Used in ResNet-50, ResNet-101, ResNet-152
    """
    expansion = 4
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        super(BottleneckBlock, self).__init__()
        
        # TODO: Implement the bottleneck block
        # Hint: You need:
        # - 1x1 conv to reduce channels (in_channels -> out_channels)
        # - 3x3 conv with given stride (out_channels -> out_channels)  
        # - 1x1 conv to expand channels (out_channels -> out_channels * expansion)
        # - BatchNorm + ReLU after each conv (except last)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)  # TODO: 1x1 conv
        self.bn1 = nn.BatchNorm2d(out_channels)    # TODO: BatchNorm
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)  # TODO: 3x3 conv with stride
        self.bn2 = nn.BatchNorm2d(out_channels)    # TODO: BatchNorm
        self.relu = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)  # TODO: 1x1 conv (expansion)
        self.bn3 = self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)    # TODO: BatchNorm
        self.downsample = downsample
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement forward pass for bottleneck block
        identity = x
        
        # TODO: Apply conv1 + bn1 + relu
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # TODO: Apply conv2 + bn2 + relu  
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        # TODO: Apply conv3 + bn3 (no relu)
        out = self.conv3(out)
        out = self.bn3(out)
        
        # TODO: Apply skip connection
        if self.downsample is not None:
            identity = self.downsample(x)  # TODO: Apply downsample
            
        # TODO: Add identity and apply final ReLU
        out += identity
        out = F.relu(out)
        
        return out


class ResNet(nn.Module):
    """
    ResNet architecture that can be configured for different depths
    """
    def __init__(self, block, layers: List[int], num_classes: int = 10, 
                 input_channels: int = 3):
        super(ResNet, self).__init__()
        
        self.in_channels = 64
        
        # TODO: Implement initial layers
        # Hint: Start with 7x7 conv, bn, relu, maxpool for ImageNet
        # For CIFAR: use 3x3 conv, bn, relu (no maxpool)
        
        if input_channels==3 and layers[0]==2: # CIFAR-10/100 case
            self.conv1 = nn.Conv2d(input_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)     # TODO: Initial convolution
            self.bn1 = nn.BatchNorm2d(64)       # TODO: BatchNorm
            self.maxpool = None 

        else: # ImageNet case
            self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # TODO: Implement residual layers
        # Hint: Use _make_layer helper function
        self.layer1 = self._make_layer(block, 64, layers[0])    # TODO: First residual layer
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)    # TODO: Second residual layer  
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)    # TODO: Third residual layer
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)    # TODO: Fourth residual layer
        
        # TODO: Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))   # TODO: Global average pooling
        self.fc = nn.Linear(512 * block.expansion, num_classes)        # TODO: Final classification layer
        
    def _make_layer(self, block, out_channels: int, blocks: int, 
                    stride: int = 1) -> nn.Sequential:
        """
        Create a residual layer with multiple blocks
        """
        downsample = None
        
        # TODO: Create downsample layer if needed
        # Hint: Need downsample when stride != 1 or channels change
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False), 
                          nn.BatchNorm2d(out_channels * block.expansion),)  # TODO: Implement downsample
            
        layers = []
        # TODO: Add first block with potential downsample
        layers.append(block(self.in_channels, out_channels, stride, downsample))  # TODO: First block
        
        # TODO: Update in_channels for subsequent blocks
        self.in_channels = out_channels * block.expansion
        
        # TODO: Add remaining blocks
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))  # TODO: Additional blocks
            
        return nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement forward pass through entire network
        
        # TODO: Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        if self.maxpool is not None:
            x = self.maxpool(x)
        
        # TODO: Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # TODO: Final layers
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


def resnet18(num_classes: int = 10) -> ResNet:
    """Create ResNet-18"""
    # TODO: Return ResNet with BasicBlock and [2, 2, 2, 2] layers
    resnet18 = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes)
    return resnet18

def resnet34(num_classes: int = 10) -> ResNet:
    """Create ResNet-34"""
    # TODO: Return ResNet with BasicBlock and [3, 4, 6, 3] layers
    resnet34 = ResNet(block=BasicBlock, layers=[3, 4, 6, 3], num_classes=num_classes)
    return resnet34

def resnet50(num_classes: int = 10) -> ResNet:
    """Create ResNet-50"""
    resnet50 = ResNet(block=BottleneckBlock, layers=[3, 4, 6, 3], num_classes=num_classes)
    return resnet50

def resnet101(num_classes: int = 10) -> ResNet:
    """Create ResNet-101"""
    resnet101 = ResNet(block=BottleneckBlock, layers=[3, 4, 23, 3], num_classes=num_classes)
    return resnet101


# ============================================================================
# PART 2: ATTENTION MECHANISMS
# ============================================================================

class SEModule(nn.Module):
    """
    Squeeze-and-Excitation module for channel attention
    """
    def __init__(self, channels: int, reduction: int = 16):
        super(SEModule, self).__init__()
        
        # TODO: Implement SE module
        # Hint: You need:
        # - Global average pooling (adaptive)
        # - Two FC layers with reduction ratio
        # - ReLU after first FC, Sigmoid after second
        
        self.global_pool = None    # TODO: Global average pooling
        self.fc1 = None           # TODO: First FC layer (reduce channels)
        self.fc2 = None           # TODO: Second FC layer (restore channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement forward pass
        # Hint: 
        # 1. Global average pool: (B, C, H, W) -> (B, C, 1, 1)
        # 2. Squeeze: (B, C, 1, 1) -> (B, C//reduction)
        # 3. Excitation: (B, C//reduction) -> (B, C) 
        # 4. Scale: multiply input by attention weights
        
        batch_size, channels, _, _ = x.size()
        
        # TODO: Global average pooling
        y = None
        
        # TODO: Squeeze
        y = None
        
        # TODO: Excitation  
        y = None
        
        # TODO: Scale input
        return x * y.view(batch_size, channels, 1, 1)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (Channel + Spatial attention)
    """
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super(CBAM, self).__init__()
        
        # TODO: Implement CBAM
        # Hint: You need:
        # - Channel attention (similar to SE)
        # - Spatial attention (avg + max pool, then conv)
        
        # Channel attention
        self.channel_attention = None  # TODO: Implement or reuse SEModule
        
        # Spatial attention  
        self.spatial_conv = None       # TODO: Conv layer for spatial attention
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement forward pass
        # Hint:
        # 1. Apply channel attention
        # 2. Apply spatial attention  
        # 3. Return final attended features
        
        # TODO: Channel attention
        x = None
        
        # TODO: Spatial attention
        # Hint: Concatenate avg and max pooled features along channel dim
        avg_pool = None  # TODO: Average pool along channel
        max_pool = None  # TODO: Max pool along channel
        spatial_input = None  # TODO: Concatenate
        spatial_attention = None  # TODO: Apply conv + sigmoid
        
        # TODO: Apply spatial attention
        x = x * spatial_attention
        
        return x


class SEResNet(ResNet):
    """
    ResNet with Squeeze-and-Excitation modules
    """
    def __init__(self, block, layers: List[int], num_classes: int = 10):
        super(SEResNet, self).__init__(block, layers, num_classes)
        
        # TODO: Add SE modules to each residual block
        # Hint: You can modify the _make_layer method or add SE after each layer
        pass


# ============================================================================
# PART 3: VISION TRANSFORMER
# ============================================================================

class PatchEmbedding(nn.Module):
    """
    Convert image patches to embeddings
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, 
                 in_channels: int = 3, embed_dim: int = 768):
        super(PatchEmbedding, self).__init__()
        
        # TODO: Implement patch embedding
        # Hint: You can use Conv2d with kernel_size=patch_size, stride=patch_size
        # Or use Unfold + Linear
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.projection = None  # TODO: Implement patch projection
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement forward pass
        # Input: (B, C, H, W)
        # Output: (B, num_patches, embed_dim)
        
        B, C, H, W = x.shape
        
        # TODO: Project patches to embeddings
        x = None
        
        # TODO: Reshape to (B, num_patches, embed_dim)
        x = None
        
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # TODO: Implement linear projections for Q, K, V
        self.qkv = None        # TODO: Combined QKV projection
        self.proj = None       # TODO: Output projection
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement multi-head attention
        # Input: (B, N, embed_dim) where N = num_patches + 1 (for CLS token)
        
        B, N, C = x.shape
        
        # TODO: Compute Q, K, V
        qkv = None  # TODO: Apply QKV projection
        qkv = None  # TODO: Reshape for multi-head: (B, N, 3, num_heads, head_dim)
        qkv = None  # TODO: Permute to (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # TODO: Compute attention scores
        attn = None  # TODO: (q @ k.transpose(-2, -1)) * self.scale
        attn = None  # TODO: Apply softmax
        attn = self.dropout(attn)
        
        # TODO: Apply attention to values
        x = None    # TODO: attn @ v
        x = None    # TODO: Transpose and reshape
        x = None    # TODO: Apply output projection
        
        return x


class TransformerBlock(nn.Module):
    """
    Transformer encoder block with self-attention and MLP
    """
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        
        # TODO: Implement transformer block
        # Hint: You need:
        # - Layer normalization (applied before attention and MLP)
        # - Multi-head self-attention
        # - MLP with GELU activation
        # - Residual connections
        
        self.norm1 = None       # TODO: Layer norm before attention
        self.attn = None        # TODO: Multi-head attention
        self.norm2 = None       # TODO: Layer norm before MLP
        
        # MLP
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = None         # TODO: Implement MLP (Linear -> GELU -> Linear)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement forward pass with residual connections
        # Hint: x = x + attention(norm(x))
        #       x = x + mlp(norm(x))
        
        # TODO: Attention block with residual
        x = None
        
        # TODO: MLP block with residual  
        x = None
        
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer implementation
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, 
                 in_channels: int = 3, num_classes: int = 1000,
                 embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super(VisionTransformer, self).__init__()
        
        # TODO: Implement ViT components
        # Hint: You need:
        # - Patch embedding
        # - Class token (learnable parameter)
        # - Position embeddings (learnable)
        # - Transformer blocks
        # - Classification head
        
        self.patch_embed = None     # TODO: Patch embedding
        
        num_patches = self.patch_embed.num_patches
        
        # TODO: Learnable parameters
        self.cls_token = None       # TODO: Class token
        self.pos_embed = None       # TODO: Position embeddings
        
        # TODO: Transformer blocks
        self.blocks = None          # TODO: Stack of transformer blocks
        
        # TODO: Final normalization and classification
        self.norm = None            # TODO: Final layer norm
        self.head = None            # TODO: Classification head
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement forward pass
        # Hint:
        # 1. Patch embedding
        # 2. Add class token
        # 3. Add position embeddings
        # 4. Apply transformer blocks
        # 5. Extract class token and classify
        
        B = x.shape[0]
        
        # TODO: Patch embedding
        x = None
        
        # TODO: Add class token
        cls_tokens = None  # TODO: Expand class token for batch
        x = None          # TODO: Concatenate class token with patches
        
        # TODO: Add position embeddings
        x = None
        
        # TODO: Apply transformer blocks
        x = None
        
        # TODO: Classification head (use class token)
        x = None  # TODO: Extract class token (first token)
        x = None  # TODO: Apply final norm and head
        
        return x


def vit_tiny(num_classes: int = 10) -> VisionTransformer:
    """Create ViT-Tiny"""
    # TODO: Return ViT with appropriate parameters for tiny model
    return None

def vit_small(num_classes: int = 10) -> VisionTransformer:
    """Create ViT-Small"""  
    # TODO: Return ViT with appropriate parameters for small model
    return None

def vit_base(num_classes: int = 10) -> VisionTransformer:
    """Create ViT-Base"""
    # TODO: Return ViT with appropriate parameters for base model
    return None


# ============================================================================
# PART 4: BENCHMARKING AND ANALYSIS
# ============================================================================

class ModelProfiler:
    """
    Tool for profiling model performance
    """
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model.to(device)
        self.device = device
        
    def count_parameters(self) -> int:
        """Count total number of trainable parameters"""
        # TODO: Implement parameter counting
        return 0
        
    def measure_flops(self, input_size: Tuple[int, ...]) -> int:
        """Estimate FLOPs for forward pass"""
        # TODO: Implement FLOP counting
        # Hint: You can use torchprofile library or implement manually
        return 0
        
    def measure_memory(self, batch_size: int, input_size: Tuple[int, ...]) -> Dict[str, float]:
        """Measure memory usage during forward/backward pass"""
        # TODO: Implement memory profiling
        # Hint: Use torch.cuda.memory_allocated() and torch.cuda.max_memory_allocated()
        return {'forward': 0.0, 'backward': 0.0, 'peak': 0.0}
        
    def measure_latency(self, input_size: Tuple[int, ...], batch_size: int = 1, 
                       num_runs: int = 100) -> float:
        """Measure inference latency"""
        # TODO: Implement latency measurement
        # Hint: Average over multiple runs, use torch.cuda.synchronize()
        return 0.0
        
    def benchmark_training(self, dataloader: DataLoader, epochs: int = 1) -> Dict[str, float]:
        """Benchmark training performance"""
        # TODO: Implement training benchmark
        # Hint: Measure time per epoch, convergence rate
        return {'time_per_epoch': 0.0, 'samples_per_second': 0.0}


class ArchitectureComparator:
    """
    Compare different architectures on the same task
    """
    def __init__(self, models: Dict[str, nn.Module], device: torch.device):
        self.models = {name: model.to(device) for name, model in models.items()}
        self.device = device
        self.results = {}
        
    def compare_accuracy(self, test_loader: DataLoader) -> Dict[str, float]:
        """Compare test accuracy of all models"""
        # TODO: Implement accuracy comparison
        # Hint: Evaluate each model on test set
        accuracies = {}
        for name, model in self.models.items():
            # TODO: Evaluate model and store accuracy
            accuracies[name] = 0.0
        return accuracies
        
    def compare_efficiency(self, input_size: Tuple[int, ...]) -> Dict[str, Dict[str, float]]:
        """Compare computational efficiency of all models"""
        # TODO: Implement efficiency comparison
        # Hint: Use ModelProfiler for each model
        efficiency = {}
        for name, model in self.models.items():
            profiler = ModelProfiler(model, self.device)
            # TODO: Profile each model
            efficiency[name] = {
                'parameters': 0,
                'flops': 0,
                'latency': 0.0,
                'memory': 0.0
            }
        return efficiency
        
    def plot_comparison(self, save_path: Optional[str] = None):
        """Create visualization comparing all models"""
        # TODO: Implement comparison plots
        # Hint: Create subplots for accuracy, efficiency metrics
        pass


def create_data_loaders(dataset_name: str = 'CIFAR10', batch_size: int = 128,
                       subset_size: Optional[int] = None) -> Tuple[DataLoader, DataLoader]:
    """Create train and test data loaders"""
    # TODO: Implement data loader creation
    # Hint: Support CIFAR-10, CIFAR-100, and subset functionality
    
    # TODO: Define transforms
    transform_train = None
    transform_test = None
    
    # TODO: Load datasets
    if dataset_name == 'CIFAR10':
        train_dataset = None
        test_dataset = None
    elif dataset_name == 'CIFAR100':
        train_dataset = None
        test_dataset = None
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # TODO: Create subset if specified
    if subset_size is not None:
        train_dataset = None  # TODO: Create subset
        
    # TODO: Create data loaders
    train_loader = None
    test_loader = None
    
    return train_loader, test_loader


def train_model(model: nn.Module, train_loader: DataLoader, 
                val_loader: DataLoader, epochs: int = 10,
                device: torch.device = torch.device('cpu')) -> Dict[str, List[float]]:
    """Train a model and return training history"""
    # TODO: Implement training loop
    # Hint: Include validation, learning rate scheduling, early stopping
    
    model.to(device)
    
    # TODO: Define optimizer and criterion
    optimizer = None
    criterion = None
    scheduler = None  # Optional: learning rate scheduler
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # TODO: Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # TODO: Implement training step
            pass
            
        # TODO: Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                # TODO: Implement validation step
                pass
                
        # TODO: Update history and print progress
        
    return history


# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

def test_resnet_implementation():
    """Test ResNet implementation"""
    print("Testing ResNet implementation...")
    
    # TODO: Create test cases for different ResNet variants
    models = {
        'ResNet-18': resnet18(num_classes=10),
        'ResNet-34': resnet34(num_classes=10), 
        'ResNet-50': resnet50(num_classes=10)
    }
    
    test_input = torch.randn(2, 3, 32, 32)  # CIFAR-10 input size
    
    for name, model in models.items():
        if model is not None:
            try:
                output = model(test_input)
                print(f"✓ {name}: Output shape {output.shape}")
            except Exception as e:
                print(f"✗ {name}: Error - {e}")
        else:
            print(f"✗ {name}: Not implemented")


def test_attention_implementation():
    """Test attention mechanism implementation"""
    print("\nTesting attention mechanisms...")
    
    # TODO: Test SE module
    se_module = SEModule(channels=64)
    test_input = torch.randn(2, 64, 32, 32)
    
    try:
        if se_module.global_pool is not None:
            output = se_module(test_input)
            print(f"✓ SE Module: Output shape {output.shape}")
        else:
            print("✗ SE Module: Not implemented")
    except Exception as e:
        print(f"✗ SE Module: Error - {e}")
    
    # TODO: Test CBAM module
    cbam_module = CBAM(channels=64)
    try:
        if cbam_module.channel_attention is not None:
            output = cbam_module(test_input)
            print(f"✓ CBAM Module: Output shape {output.shape}")
        else:
            print("✗ CBAM Module: Not implemented")
    except Exception as e:
        print(f"✗ CBAM Module: Error - {e}")


def test_vit_implementation():
    """Test Vision Transformer implementation"""
    print("\nTesting Vision Transformer...")
    
    # TODO: Test patch embedding
    patch_embed = PatchEmbedding(img_size=32, patch_size=4, embed_dim=192)
    test_input = torch.randn(2, 3, 32, 32)
    
    try:
        if patch_embed.projection is not None:
            output = patch_embed(test_input)
            print(f"✓ Patch Embedding: Output shape {output.shape}")
        else:
            print("✗ Patch Embedding: Not implemented")
    except Exception as e:
        print(f"✗ Patch Embedding: Error - {e}")
    
    # TODO: Test ViT models
    models = {
        'ViT-Tiny': vit_tiny(num_classes=10),
        'ViT-Small': vit_small(num_classes=10),
        'ViT-Base': vit_base(num_classes=10)
    }
    
    for name, model in models.items():
        if model is not None:
            try:
                output = model(test_input)
                print(f"✓ {name}: Output shape {output.shape}")
            except Exception as e:
                print(f"✗ {name}: Error - {e}")
        else:
            print(f"✗ {name}: Not implemented")


def main():
    """Main function to run tests and example experiments"""
    print("Lab 3: Architecture Implementation Mastery")
    print("=" * 50)
    
    # Test implementations
    test_resnet_implementation()
    test_attention_implementation()
    test_vit_implementation()
    
    print("\n" + "=" * 50)
    print("Implementation tests complete!")
    print("TODO: Complete the implementations and run your experiments")
    
    # TODO: Add example experiments here
    # Example:
    # 1. Train ResNet-18 vs ResNet-50 on CIFAR-10
    # 2. Compare with and without attention mechanisms
    # 3. Train ViT and compare with CNNs
    # 4. Create visualizations and analysis


if __name__ == "__main__":
    main()