"""
Lab 3: Architecture Implementation Mastery - Student Template
VIAR25/26 - Artificial Vision - UVigo
Prof. David Olivieri

This template provides the structure for implementing:
1. ResNet architectures (Basic and Bottleneck blocks)
2. Attention mechanisms (SE and CBAM)
3. Vision Transformer components
4. Benchmarking and comparison tools

All required components are implemented for reference architectures and transformers
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
from torch.nn.init import trunc_normal_
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

        # Hint: You need:
        # - First 3x3 conv with given stride
        # - BatchNorm + ReLU
        # - Second 3x3 conv with stride=1
        # - BatchNorm (no ReLU here)
        # - Store downsample for skip connection

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = None
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Remember: output = F(x) + x (or F(x) + downsample(x))

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

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

        # Hint: You need:
        # - 1x1 conv to reduce channels (in_channels -> out_channels)
        # - 3x3 conv with given stride (out_channels -> out_channels)
        # - 1x1 conv to expand channels (out_channels -> out_channels * expansion)
        # - BatchNorm + ReLU after each conv (except last)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se = None
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet architecture that can be configured for different depths
    """
    def __init__(self, block, layers: List[int], num_classes: int = 10,
                 input_channels: int = 3):
        super(ResNet, self).__init__()

        self.in_channels = 64

        # Hint: Start with 7x7 conv, bn, relu, maxpool for ImageNet
        # For CIFAR: use 3x3 conv, bn, relu (no maxpool)

        self.conv1 = nn.Conv2d(input_channels, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Identity()

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels: int, blocks: int,
                    stride: int = 1) -> nn.Sequential:
        """
        Create a residual layer with multiple blocks
        """
        downsample = None

        # Hint: Need downsample when stride != 1 or channels change
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))

        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18(num_classes: int = 10) -> ResNet:
    """Create ResNet-18"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def resnet34(num_classes: int = 10) -> ResNet:
    """Create ResNet-34"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

def resnet50(num_classes: int = 10) -> ResNet:
    """Create ResNet-50"""
    return ResNet(BottleneckBlock, [3, 4, 6, 3], num_classes=num_classes)

def resnet101(num_classes: int = 10) -> ResNet:
    """Create ResNet-101"""
    return ResNet(BottleneckBlock, [3, 4, 23, 3], num_classes=num_classes)


# ============================================================================
# PART 2: ATTENTION MECHANISMS
# ============================================================================

class SEModule(nn.Module):
    """
    Squeeze-and-Excitation module for channel attention
    """
    def __init__(self, channels: int, reduction: int = 16):
        super(SEModule, self).__init__()

        # Hint: You need:
        # - Global average pooling (adaptive)
        # - Two FC layers with reduction ratio
        # - ReLU after first FC, Sigmoid after second

        reduced_channels = max(channels // reduction, 1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, reduced_channels)
        self.fc2 = nn.Linear(reduced_channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Hint:
        # 1. Global average pool: (B, C, H, W) -> (B, C, 1, 1)
        # 2. Squeeze: (B, C, 1, 1) -> (B, C//reduction)
        # 3. Excitation: (B, C//reduction) -> (B, C)
        # 4. Scale: multiply input by attention weights

        batch_size, channels, _, _ = x.size()

        y = self.global_pool(x).view(batch_size, channels)
        y = F.relu(self.fc1(y), inplace=True)
        y = torch.sigmoid(self.fc2(y))

        return x * y.view(batch_size, channels, 1, 1)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (Channel + Spatial attention)
    """
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super(CBAM, self).__init__()

        # Hint: You need:
        # - Channel attention (similar to SE)
        # - Spatial attention (avg + max pool, then conv)

        # Channel attention
        self.channel_attention = SEModule(channels, reduction)

        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Hint:
        # 1. Apply channel attention
        # 2. Apply spatial attention
        # 3. Return final attended features

        x = self.channel_attention(x)

        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_attention = torch.sigmoid(self.spatial_conv(spatial_input))

        x = x * spatial_attention

        return x


class SEResNet(ResNet):
    """
    ResNet with Squeeze-and-Excitation modules
    """
    def __init__(self, block, layers: List[int], num_classes: int = 10, reduction: int = 16):
        super(SEResNet, self).__init__(block, layers, num_classes)

        self._inject_se(reduction)

    def _inject_se(self, reduction: int) -> None:
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                if hasattr(block, 'bn3'):
                    channels = block.bn3.num_features
                else:
                    channels = block.bn2.num_features
                block.se = SEModule(channels, reduction=reduction)


class CBAMResNet(ResNet):
    """
    ResNet with Convolutional Block Attention Modules (CBAM)
    """
    def __init__(self, block, layers: List[int], num_classes: int = 10,
                 reduction: int = 16, kernel_size: int = 7):
        super(CBAMResNet, self).__init__(block, layers, num_classes)

        self._inject_cbam(reduction, kernel_size)

    def _inject_cbam(self, reduction: int, kernel_size: int) -> None:
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                if hasattr(block, 'bn3'):
                    channels = block.bn3.num_features
                else:
                    channels = block.bn2.num_features
                block.se = CBAM(channels, reduction=reduction, kernel_size=kernel_size)




def seresnet18(num_classes: int = 10, reduction: int = 16) -> SEResNet:
    """Create SE-ResNet-18"""
    return SEResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, reduction=reduction)

def seresnet34(num_classes: int = 10, reduction: int = 16) -> SEResNet:
    """Create SE-ResNet-34"""
    return SEResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, reduction=reduction)

def seresnet50(num_classes: int = 10, reduction: int = 16) -> SEResNet:
    """Create SE-ResNet-50"""
    return SEResNet(BottleneckBlock, [3, 4, 6, 3], num_classes=num_classes, reduction=reduction)

def seresnet101(num_classes: int = 10, reduction: int = 16) -> SEResNet:
    """Create SE-ResNet-101"""
    return SEResNet(BottleneckBlock, [3, 4, 23, 3], num_classes=num_classes, reduction=reduction)

def cbam_resnet18(num_classes: int = 10, reduction: int = 16, kernel_size: int = 7) -> CBAMResNet:
    """Create CBAM-ResNet-18"""
    return CBAMResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes,
                      reduction=reduction, kernel_size=kernel_size)

def cbam_resnet34(num_classes: int = 10, reduction: int = 16, kernel_size: int = 7) -> CBAMResNet:
    """Create CBAM-ResNet-34"""
    return CBAMResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes,
                      reduction=reduction, kernel_size=kernel_size)

def cbam_resnet50(num_classes: int = 10, reduction: int = 16, kernel_size: int = 7) -> CBAMResNet:
    """Create CBAM-ResNet-50"""
    return CBAMResNet(BottleneckBlock, [3, 4, 6, 3], num_classes=num_classes,
                      reduction=reduction, kernel_size=kernel_size)

def cbam_resnet101(num_classes: int = 10, reduction: int = 16, kernel_size: int = 7) -> CBAMResNet:
    """Create CBAM-ResNet-101"""
    return CBAMResNet(BottleneckBlock, [3, 4, 23, 3], num_classes=num_classes,
                      reduction=reduction, kernel_size=kernel_size)
# ============================================================================
# PART 3: VISION TRANSFORMER
# ============================================================================

class PatchEmbedding(nn.Module):
    """Convert image patches to embeddings"""
    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, embed_dim: int = 768,
                 flatten: bool = True):
        super(PatchEmbedding, self).__init__()

        if img_size % patch_size != 0:
            raise ValueError(f"img_size ({img_size}) debe ser divisible por patch_size ({patch_size})")

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.flatten = flatten

        # Conv2d con stride = patch_size actua como extractor de parches y proyeccion lineal
        self.projection = nn.Conv2d(in_channels, embed_dim,
                                    kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Entrada: (B, C, H, W)
        B, _, H, W = x.shape
        if H != self.img_size or W != self.img_size:
            raise ValueError(f"Entrada con tamano espacial {(H, W)} incompatible con img_size={self.img_size}")

        x = self.projection(x)
        if self.flatten:
            # Salida: (B, num_patches, embed_dim)
            x = x.flatten(2).transpose(1, 2)
        else:
            # Forma: (B, embed_dim, grid_size, grid_size)
            x = x.view(B, -1, self.grid_size, self.grid_size)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim debe ser divisible por num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Entrada: (B, N, embed_dim) donde N = num_patches + 1 (token CLS)
        B, N, _ = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, self.embed_dim)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block con autoatencion y MLP"""
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super(TransformerBlock, self).__init__()

        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio debe ser mayor que cero")

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual: x = x + atencion(norm(x)) y x = x + mlp(norm(x))
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer implementation"""
    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, num_classes: int = 1000,
                 embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super(VisionTransformer, self).__init__()

        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self.apply(self._init_weights)
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed[:, :x.size(1)]
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x


def vit_tiny(num_classes: int = 10) -> VisionTransformer:
    """Create ViT-Tiny"""
    return VisionTransformer(img_size=32, patch_size=4, in_channels=3,
                             num_classes=num_classes, embed_dim=192, depth=12,
                             num_heads=3, mlp_ratio=4.0, dropout=0.1)


def vit_small(num_classes: int = 10) -> VisionTransformer:
    """Create ViT-Small"""
    return VisionTransformer(img_size=32, patch_size=4, in_channels=3,
                             num_classes=num_classes, embed_dim=384, depth=12,
                             num_heads=6, mlp_ratio=4.0, dropout=0.1)


def vit_base(num_classes: int = 10) -> VisionTransformer:
    """Create ViT-Base"""
    return VisionTransformer(img_size=32, patch_size=4, in_channels=3,
                             num_classes=num_classes, embed_dim=768, depth=12,
                             num_heads=12, mlp_ratio=4.0, dropout=0.1)

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
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def measure_flops(self, input_size: Tuple[int, ...]) -> int:
        """Estimate FLOPs for forward pass"""
        flops: float = 0.0
        handles = []
        training_mode = self.model.training
        self.model.eval()

        def conv_hook(module, inputs, outputs):
            input_tensor = inputs[0]
            batch_size = input_tensor.shape[0]
            out_channels = module.out_channels
            kernel_h, kernel_w = module.kernel_size
            out_h, out_w = outputs.shape[2], outputs.shape[3]
            groups = module.groups
            in_channels = module.in_channels
            kernel_ops = (in_channels / groups) * kernel_h * kernel_w
            bias_ops = 1 if module.bias is not None else 0
            total_ops = batch_size * out_channels * out_h * out_w * (kernel_ops + bias_ops)
            nonlocal flops
            flops += total_ops

        def linear_hook(module, inputs, outputs):
            input_tensor = inputs[0]
            batch_size = input_tensor.shape[0]
            in_features = module.in_features
            out_features = module.out_features
            bias_ops = out_features if module.bias is not None else 0
            total_ops = batch_size * (in_features * out_features + bias_ops)
            nonlocal flops
            flops += total_ops

        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                handles.append(module.register_forward_hook(conv_hook))
            elif isinstance(module, nn.Linear):
                handles.append(module.register_forward_hook(linear_hook))

        dummy_input = torch.randn((1,) + input_size, device=self.device)
        with torch.no_grad():
            self.model(dummy_input)

        for handle in handles:
            handle.remove()

        self.model.train(training_mode)
        return int(flops)

    def measure_memory(self, batch_size: int, input_size: Tuple[int, ...]) -> Dict[str, float]:
        """Measure memory usage during forward/backward pass"""
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            return {'forward': 0.0, 'backward': 0.0, 'peak': 0.0}

        original_mode = self.model.training
        self.model.to(self.device)

        forward_input = torch.randn(batch_size, *input_size, device=self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        with torch.no_grad():
            self.model.eval()
            self.model(forward_input)
            torch.cuda.synchronize(self.device)
        forward_mem = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)

        backward_input = torch.randn(batch_size, *input_size, device=self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        self.model.train()
        outputs = self.model(backward_input)
        if outputs.dim() > 2:
            outputs = outputs.view(outputs.size(0), -1)
        num_classes = outputs.size(-1)
        targets = torch.randint(0, num_classes, (batch_size,), device=self.device)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        loss.backward()
        torch.cuda.synchronize(self.device)
        peak_mem = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        backward_mem = max(peak_mem - forward_mem, 0.0)

        self.model.zero_grad(set_to_none=True)
        self.model.train(original_mode)

        return {'forward': forward_mem, 'backward': backward_mem, 'peak': peak_mem}

    def measure_latency(self, input_size: Tuple[int, ...], batch_size: int = 1,
                       num_runs: int = 100) -> float:
        """Measure inference latency"""
        original_mode = self.model.training
        self.model.eval()
        input_tensor = torch.randn(batch_size, *input_size, device=self.device)

        with torch.no_grad():
            for _ in range(min(10, num_runs)):
                self.model(input_tensor)
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            for _ in range(num_runs):
                self.model(input_tensor)
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - start

        self.model.train(original_mode)
        return elapsed / max(num_runs, 1)

    def benchmark_training(self, dataloader: DataLoader, epochs: int = 1) -> Dict[str, float]:
        """Benchmark training performance"""
        original_mode = self.model.training
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        total_time = 0.0
        total_samples = 0

        for _ in range(epochs):
            start = time.perf_counter()
            for data, target in dataloader:
                data = data.to(self.device)
                target = target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_samples += data.size(0)
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            total_time += time.perf_counter() - start

        avg_epoch_time = total_time / max(epochs, 1)
        samples_per_second = total_samples / total_time if total_time > 0 else 0.0

        self.model.train(original_mode)
        return {'time_per_epoch': avg_epoch_time, 'samples_per_second': samples_per_second}

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
        accuracies: Dict[str, float] = {}
        original_modes = {name: model.training for name, model in self.models.items()}

        for model in self.models.values():
            model.eval()

        with torch.no_grad():
            for name, model in self.models.items():
                correct = 0
                total = 0
                for data, target in test_loader:
                    data = data.to(self.device)
                    target = target.to(self.device)
                    outputs = model(data)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == target).sum().item()
                    total += target.size(0)
                accuracies[name] = correct / total if total > 0 else 0.0

        for name, model in self.models.items():
            model.train(original_modes[name])

        self.results['accuracy'] = accuracies
        return accuracies

    def compare_efficiency(self, input_size: Tuple[int, ...]) -> Dict[str, Dict[str, float]]:
        """Compare computational efficiency of all models"""
        efficiency: Dict[str, Dict[str, float]] = {}
        for name, model in self.models.items():
            profiler = ModelProfiler(model, self.device)
            memory_metrics = profiler.measure_memory(batch_size=1, input_size=input_size)
            efficiency[name] = {
                'parameters': float(profiler.count_parameters()),
                'flops': float(profiler.measure_flops(input_size)),
                'latency': float(profiler.measure_latency(input_size)),
                'memory': float(memory_metrics['peak'])
            }

        self.results['efficiency'] = efficiency
        return efficiency

    def plot_comparison(self, save_path: Optional[str] = None):
        """Create visualization comparing all models"""
        if 'accuracy' not in self.results or 'efficiency' not in self.results:
            raise RuntimeError('Run compare_accuracy and compare_efficiency before plotting.')

        accuracy = self.results['accuracy']
        efficiency = self.results['efficiency']
        model_names = list(accuracy.keys())

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        axes[0].bar(model_names, [accuracy[name] * 100 for name in model_names])
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].set_title('Test Accuracy')

        axes[1].bar(model_names, [efficiency[name]['parameters'] / 1e6 for name in model_names])
        axes[1].set_ylabel('Parameters (Millions)')
        axes[1].set_title('Model Size')

        axes[2].bar(model_names, [efficiency[name]['latency'] * 1000 for name in model_names])
        axes[2].set_ylabel('Latency (ms)')
        axes[2].set_title('Inference Latency')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()


def create_data_loaders(dataset_name: str = 'CIFAR10', batch_size: int = 128,
                       subset_size: Optional[int] = None) -> Tuple[DataLoader, DataLoader]:
    """Create train and test data loaders"""
    if dataset_name not in {'CIFAR10', 'CIFAR100'}:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if dataset_name == 'CIFAR10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    dataset_cls = torchvision.datasets.CIFAR10 if dataset_name == 'CIFAR10' else torchvision.datasets.CIFAR100

    train_dataset = dataset_cls(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = dataset_cls(root='./data', train=False, download=True, transform=transform_test)

    if subset_size is not None and subset_size < len(train_dataset):
        indices = torch.randperm(len(train_dataset))[:subset_size]
        train_dataset = Subset(train_dataset, indices.tolist())

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin_memory)

    return train_loader, test_loader


def train_model(model: nn.Module, train_loader: DataLoader,
                val_loader: DataLoader, epochs: int = 10,
                device: torch.device = torch.device('cpu')) -> Dict[str, List[float]]:
    """Train a model and return training history"""
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 3, 1), gamma=0.1) if epochs > 1 else None

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == target).sum().item()
            train_total += target.size(0)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                target = target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                val_loss += loss.item() * data.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == target).sum().item()
                val_total += target.size(0)

        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)
        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if scheduler is not None:
            scheduler.step()

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    return history


# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

def test_resnet_implementation():
    """Test ResNet implementation"""
    print("Testing ResNet implementation...")

    models = {
        'ResNet-18': resnet18(num_classes=10),
        'ResNet-34': resnet34(num_classes=10),
        'ResNet-50': resnet50(num_classes=10)
    }

    test_input = torch.randn(2, 3, 32, 32)  # CIFAR-10 input size

    for name, model in models.items():
        if model is None:
            print(f'[SKIP] {name}: Not implemented')
            continue
        try:
            output = model(test_input)
            print(f'[OK] {name}: Output shape {output.shape}')
        except Exception as e:
            print(f'[FAIL] {name}: Error - {e}')



def test_attention_implementation():
    """Test attention mechanism implementation"""
    print("\nTesting attention mechanisms...")

    se_module = SEModule(channels=64)
    test_input = torch.randn(2, 64, 32, 32)

    try:
        output = se_module(test_input)
        print(f'[OK] SE Module: Output shape {output.shape}')
    except Exception as e:
        print(f'[FAIL] SE Module: Error - {e}')

    cbam_module = CBAM(channels=64)
    try:
        output = cbam_module(test_input)
        print(f'[OK] CBAM Module: Output shape {output.shape}')
    except Exception as e:
        print(f'[FAIL] CBAM Module: Error - {e}')

    # Validate full networks with attention modules
    models = {
        'SE-ResNet-18': seresnet18(num_classes=10),
        'CBAM-ResNet-18': cbam_resnet18(num_classes=10)
    }

    image_input = torch.randn(2, 3, 32, 32)
    for name, model in models.items():
        try:
            output = model(image_input)
            print(f'[OK] {name}: Output shape {output.shape}')
        except Exception as e:
            print(f'[FAIL] {name}: Error - {e}')



def test_vit_implementation():
    """Test Vision Transformer implementation"""
    print("\nTesting Vision Transformer...")

    patch_embed = PatchEmbedding(img_size=32, patch_size=4, embed_dim=192)
    test_input = torch.randn(2, 3, 32, 32)

    try:
        output = patch_embed(test_input)
        print(f'[OK] Patch Embedding: Output shape {output.shape}')
    except Exception as e:
        print(f'[FAIL] Patch Embedding: Error - {e}')

    models = {
        'ViT-Tiny': vit_tiny(num_classes=10),
        'ViT-Small': vit_small(num_classes=10),
        'ViT-Base': vit_base(num_classes=10)
    }

    for name, model in models.items():
        if model is None:
            print(f'[SKIP] {name}: Not implemented')
            continue
        try:
            output = model(test_input)
            print(f'[OK] {name}: Output shape {output.shape}')
        except Exception as e:
            print(f'[FAIL] {name}: Error - {e}')



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
    print("Implementations complete! Customize and run your experiments.")

    # Example:
    # 1. Train ResNet-18 vs ResNet-50 on CIFAR-10
    # 2. Compare with and without attention mechanisms
    # 3. Train ViT and compare with CNNs
    # 4. Create visualizations and analysis


if __name__ == "__main__":
    main()