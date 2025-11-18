"""
Lab 5: Advanced Image Segmentation
Student Template - Complete all TODOs

This lab implements and compares 5 state-of-the-art semantic segmentation architectures:
- FCN-32s/16s/8s: Fully Convolutional Networks with progressive skip connections
- DeepLabV3+: Advanced architecture with ASPP (Atrous Spatial Pyramid Pooling)
- MiniSAM: Lightweight Segment Anything Model with interactive prompts

Author: David y Nico
Date: November 2025
Dataset: PASCAL VOC 2012 (21 classes)
Hardware: NVIDIA RTX 3090 (24GB VRAM)
Framework: PyTorch 2.6 with Automatic Mixed Precision (AMP)

Key Features:
- Transfer learning from pretrained ResNet50 backbones
- Combined loss function (CrossEntropy + Dice + Focal) for better convergence
- Strong data augmentation pipeline (flip, scale, crop, color jitter, rotation)
- Learning rate warmup + cosine annealing scheduler
- Early stopping with patience mechanism
- Checkpoint management compatible with weights_only=True (PyTorch 2.6+)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
from tqdm import tqdm
import warnings
import random
from lab05_utils import plot_segmentation_results


# =============================== 
# GLOBAL SEED FOR FULL REPRODUCIBILITY
# ===============================

SEED = 69415  # el seed que tú quieras (69415 es perfecto)

# --- Python & NumPy ---
random.seed(SEED)
np.random.seed(SEED)

# --- PyTorch (CPU & GPU) ---
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # múltiple GPUs (por si acaso)

# --- CUDNN determinism ---
# ⚠ Estos dos aseguran reproducibilidad BIT-A-BIT en GPU.
# ⚠ benchmark=False es necesario: si está True, PyTorch selecciona
#    diferentes algoritmos no deterministas en cada run.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- Dataloader reproducible (más importante de lo que parece) ---
def worker_init_fn(worker_id):
    """
    Cada worker recibe un seed distinto pero reproducible,
    derivado del seed global.
    """
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(SEED)




# Register numpy types as safe globals for weights_only=True loading
torch.serialization.add_safe_globals([
    np.core.multiarray.scalar,
    np.dtype,
    np.ndarray,
    np.dtypes.Float64DType,
    np.dtypes.Float32DType,
    np.dtypes.Int64DType,
    np.dtypes.Int32DType,
])

#warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ================== Part 1: FCN Architecture ==================



class KCCSContextBlock(nn.Module):
    """
    Bloque de contexto KCCS para mapas de características 2D (B, C, H, W).

    Divide el espacio de características en tres subespacios:
        - S: semántico
        - E: episódico
        - I: intencional

    Entre ellos aplica proyecciones 1x1 y afinidades tipo kernel gaussiano
    por píxel, de forma muy parecida a la versión de GridWorld pero aquí
    vectorizada en H×W.

    Entrada / salida:
        x:  [B, C, H, W]  (C = in_channels)
        y:  [B, C, H, W]  (mismo shape, con residual: y = x + f_KCCS(x))
    """
    def __init__(self, in_channels, s_dim=64, e_dim=64, i_dim=64, sigma_min=0.5):
        super().__init__()
        self.s_dim = s_dim
        self.e_dim = e_dim
        self.i_dim = i_dim
        self.sigma_min = sigma_min

        # Proyección inicial a subespacios S, E, I (1x1 conv)
        self.to_S = nn.Conv2d(in_channels, s_dim, kernel_size=1, bias=False)
        self.to_E = nn.Conv2d(in_channels, e_dim, kernel_size=1, bias=False)
        self.to_I = nn.Conv2d(in_channels, i_dim, kernel_size=1, bias=False)

        # Proyecciones entre subespacios (por canal, compartidas en HxW)
        self.W_e_to_s = nn.Conv2d(e_dim, s_dim, kernel_size=1, bias=False)
        self.W_i_to_s = nn.Conv2d(i_dim, s_dim, kernel_size=1, bias=False)

        self.W_s_to_e = nn.Conv2d(s_dim, e_dim, kernel_size=1, bias=False)
        self.W_i_to_e = nn.Conv2d(i_dim, e_dim, kernel_size=1, bias=False)

        self.W_s_to_i = nn.Conv2d(s_dim, i_dim, kernel_size=1, bias=False)
        self.W_e_to_i = nn.Conv2d(e_dim, i_dim, kernel_size=1, bias=False)

        # Sigmas (anchura del kernel) aprendibles, como en GridWorld
        self.rho_s_e = nn.Parameter(torch.tensor(0.0))
        self.rho_s_i = nn.Parameter(torch.tensor(0.0))
        self.rho_e_i = nn.Parameter(torch.tensor(0.0))

        # Proyección de vuelta a canales originales + residual
        self.out_proj = nn.Conv2d(s_dim + e_dim + i_dim, in_channels, kernel_size=1)

    def _sigma(self, rho):
        # Evita sigma <= 0
        return F.softplus(rho) + self.sigma_min

    def _gauss_aff(self, a, b, sigma):
        """
        Kernel gaussiano por píxel:
            a, b: [B, C, H, W]
            sigma: escalar
        Devuelve affinities: [B, 1, H, W]
        """
        diff = a - b
        dist2 = torch.sum(diff * diff, dim=1, keepdim=True)  # suma en canales
        return torch.exp(-dist2 / (sigma * sigma + 1e-8))

    def forward(self, x):
        # x: [B, C, H, W]
        # Proyección a subespacios
        S_prev = self.to_S(x)  # [B, s_dim, H, W]
        E_prev = self.to_E(x)  # [B, e_dim, H, W]
        I_prev = self.to_I(x)  # [B, i_dim, H, W]

        # Proyecciones entre subespacios
        S_from_E = self.W_e_to_s(E_prev)
        S_from_I = self.W_i_to_s(I_prev)

        E_from_S = self.W_s_to_e(S_prev)
        E_from_I = self.W_i_to_e(I_prev)

        I_from_S = self.W_s_to_i(S_prev)
        I_from_E = self.W_e_to_i(E_prev)

        # Sigmas
        sigma_s_e = self._sigma(self.rho_s_e)
        sigma_s_i = self._sigma(self.rho_s_i)
        sigma_e_i = self._sigma(self.rho_e_i)

        # Afinidades gaussianas (por píxel)
        aff_ES = self._gauss_aff(S_prev, S_from_E, sigma_s_e)
        aff_IS = self._gauss_aff(S_prev, S_from_I, sigma_s_i)

        aff_SE = self._gauss_aff(E_prev, E_from_S, sigma_s_e)
        aff_IE = self._gauss_aff(E_prev, E_from_I, sigma_e_i)

        aff_SI = self._gauss_aff(I_prev, I_from_S, sigma_s_i)
        aff_EI = self._gauss_aff(I_prev, I_from_E, sigma_e_i)

        # Actualizaciones tipo GridWorld pero en 2D
        S_new = torch.tanh(
            S_prev
            + aff_ES * self.W_e_to_s(E_prev)
            + aff_IS * self.W_i_to_s(I_prev)
        )

        E_new = torch.tanh(
            E_prev
            + aff_SE * self.W_s_to_e(S_prev)
            + aff_IE * self.W_i_to_e(I_prev)
        )

        I_new = torch.tanh(
            I_prev
            + aff_SI * self.W_s_to_i(S_prev)
            + aff_EI * self.W_e_to_i(E_prev)
        )

        # Fusionar subespacios y proyectar de vuelta
        out = torch.cat([S_new, E_new, I_new], dim=1)  # [B, s+e+i, H, W]
        out = self.out_proj(out)                       # [B, C, H, W]

        # Residual: no rompemos el backbone
        return x + out




class FCN32s(nn.Module):
    """Fully Convolutional Network without skip connections (baseline).
    
    This is the simplest FCN variant that directly upsamples the final feature map
    from stride 32 back to the original resolution using a single transposed convolution.
    
    Architecture:
        Input (3×H×W) → ResNet50 Encoder (stride 32) → Score Layer (1×1 conv to n_classes)
        → Upsample 32× → Output (n_classes×H×W)
    
    Advantages:
        - Simple architecture, fast inference (~1.57ms/image on RTX 3090)
        - Fewer parameters than skip connection variants
        - Good baseline for comparison
    
    Disadvantages:
        - Poor spatial detail due to aggressive downsampling without skip connections
        - Lower mIoU (~55-60%) compared to FCN-16s/8s
    
    Parameters:
        n_classes (int): Number of segmentation classes (default: 21 for PASCAL VOC)
    
    Expected Performance:
        - mIoU: 55-60% (with augmentation, 60 epochs)
        - Pixel Accuracy: ~75%
        - Parameters: ~25.36M
    """
    
    def __init__(self, n_classes=21):
        super().__init__()
        
        # Task 1.1: Load pretrained ResNet50 and extract layers
        # 1. Load models.resnet50(weights='DEFAULT')
        resnet = models.resnet50(weights='DEFAULT')

        # 2. Extract conv1, bn1, relu, maxpool
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # 3. Extract layer1, layer2, layer3, layer4
        self.layer1 = resnet.layer1  # Output stride 4 (1/4 of input size)
        self.layer2 = resnet.layer2  # Output stride 8 (1/8 of input size)
        self.layer3 = resnet.layer3  # Output stride 16 (1/16 of input size)
        self.layer4 = resnet.layer4  # Output stride 32 (1/32 of input size)
        
        # Task 1.2: Add score layer
        # 1. Create 1x1 convolution: nn.Conv2d(2048, n_classes, kernel_size=1)
        self.score_fr = nn.Conv2d(2048, n_classes, kernel_size=1)
        
        # Task 1.3: Add upsampling layer
        # 1. Create transposed convolution for 32x upsampling
        # 2. Use nn.ConvTranspose2d(n_classes, n_classes, kernel_size=64, stride=32, bias=False)

        self.upscore32 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=64, stride=32, padding=16, bias=False)
        
        # Initialize upsampling layer with bilinear weights
        self._initialize_weights()
        
    def forward(self, x):
        # Task 1.4: Implement forward pass
        # 1. Pass through conv1, bn1, relu, maxpool
        # 2. Pass through layer1, layer2, layer3, layer4
        # 3. Apply score layer (1x1 conv)
        # 4. Apply 32x upsampling
        # 5. Return output
        
        # Save input size for final resize if needed
        input_size = x.shape[2:]  # (H, W)

        # Initial convolution block (stride 2 after conv1, stride 4 after maxpool)
        x = self.conv1(x)  # Stride 2: H/2 × W/2
        x = self.bn1(x)    # Batch normalization
        x = self.relu(x)   # ReLU activation
        x = self.maxpool(x) # Stride 4: H/4 × W/4 (factor 2 más)

        # ResNet bottleneck blocks (progressive downsampling)
        x = self.layer1(x)  # Stride 4: H/4 × W/4 (256 channels)
        x = self.layer2(x)  # Stride 8: H/8 × W/8 (512 channels)
        x = self.layer3(x)  # Stride 16: H/16 × W/16 (1024 channels)
        x = self.layer4(x)  # Stride 32: H/32 × W/32 (2048 channels)

        # Score layer: map 2048 feature channels to n_classes (e.g., 21 for VOC)
        # This predicts class scores for each pixel at stride 32
        x = self.score_fr(x)  # Shape: (B, n_classes, H/32, W/32)
        
        # Upsample by 32× to restore original resolution
        # Uses learned bilinear interpolation weights
        x = self.upscore32(x)  # Shape: (B, n_classes, H, W)
        
        # Ensure output exactly matches input size (handle any rounding errors)
        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
     
        return x
    
    def _initialize_weights(self):
        """Initialize ConvTranspose2d layers with bilinear weights"""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                # Initialize with bilinear upsampling weights
                # ConvTranspose2d weight shape: (in_channels, out_channels, kernel_h, kernel_w)
                in_ch, out_ch, h, w = m.weight.data.size()
                weight = self._get_bilinear_filter(h, w, in_ch, out_ch)
                m.weight.data.copy_(weight)
    
    def _get_bilinear_filter(self, kernel_h, kernel_w, in_channels, out_channels):
        """Generate bilinear interpolation weights (per-channel upsampling)
        
        Creates a filter that implements bilinear interpolation for smooth upsampling.
        This is better than random initialization as it preserves spatial structure.
        """
        # Calculate the center of the filter
        factor = (kernel_h + 1) // 2
        if kernel_h % 2 == 1:
            center = factor - 1  # Odd kernel size: center is an integer
        else:
            center = factor - 0.5  # Even kernel size: center is between pixels
        
        # Create a grid of coordinates for the filter
        og = np.ogrid[:kernel_h, :kernel_w]
        
        # Compute bilinear weights: closer to center = higher weight
        # Weight decreases linearly with distance from center
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        filt = torch.from_numpy(filt).float()
        
        # Create per-channel identity mapping (channel i maps to channel i)
        weight = torch.zeros(in_channels, out_channels, kernel_h, kernel_w)
        for i in range(min(in_channels, out_channels)):
            weight[i, i, :, :] = filt  # Each channel gets the same bilinear filter
        return weight



class FCN16s(nn.Module):
    """Fully Convolutional Network with one skip connection from pool4 (layer3).
    
    Improves upon FCN-32s by adding a skip connection from layer3 (stride 16),
    which helps preserve mid-level spatial details for better boundary delineation.
    
    Architecture:
        Input → ResNet50 Encoder
        ├─ layer3 (pool4, stride 16) → Score Layer → ┐
        └─ layer4 (stride 32) → Score Layer → Upsample 2× → Element-wise Add
                                                               ↓
                                                       Upsample 16× → Output
    
    Fusion Strategy:
        1. Upsample layer4 predictions by 2× (stride 32 → 16)
        2. Add with layer3 predictions (element-wise)
        3. Upsample fused result by 16× to original resolution
    
    Advantages:
        - Better spatial detail than FCN-32s
        - Faster inference than FCN-8s (~1.25ms/image)
        - Good balance of accuracy and speed
    
    Parameters:
        n_classes (int): Number of segmentation classes
    
    Expected Performance:
        - mIoU: 58-62%
        - Parameters: ~24.03M
    """
    
    def __init__(self, n_classes=21):
        super().__init__()
        
        # Task 1.1: Load pretrained ResNet50 and extract layers
        # (Same as FCN32s)
        resnet = models.resnet50(weights='DEFAULT')

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2 
        self.layer3 = resnet.layer3 
        self.layer4 = resnet.layer4  
        
        # Task 1.2: Add score layers
        # 1. score_pool4: nn.Conv2d(1024, n_classes, 1) for layer3 output
        # 2. score_fr: nn.Conv2d(2048, n_classes, 1) for layer4 output
        
        self.score_pool4 = nn.Conv2d(1024, n_classes, kernel_size=1)
        self.score_fr = nn.Conv2d(2048, n_classes, kernel_size=1)
        
        # Task 1.3: Add upsampling layers
        # 1. upscore2: 2x upsampling with stride=2
        # 2. upscore16: 16x upsampling with stride=16
        
        self.upscore2 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.upscore16 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=32, stride=16, padding=8, bias=False)
        
        # Initialize upsampling layers with bilinear weights
        self._initialize_weights()
        
    def forward(self, x):
        # Task 1.4: Implement forward pass with one skip connection
        # 1. Pass through initial layers until layer3 (save as pool4)
        # 2. Continue to layer4
        # 3. Apply score_fr to layer4 output
        # 4. Apply score_pool4 to pool4
        # 5. Upsample score_fr by 2x (use upscore2)
        # 6. Add upsampled score_fr + score_pool4 (element-wise addition)
        # 7. Upsample fused result by 16x
        # 8. Return output
        
        # Save input size for final adjustment
        input_size = x.shape[2:]
        
        # Encoder: ResNet50 backbone
        x = self.conv1(x)     # Stride 2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)   # Stride 4

        x = self.layer1(x)    # Stride 4 (256 channels)
        x = self.layer2(x)    # Stride 8 (512 channels)
        
        # SKIP CONNECTION: Save pool4 (layer3 output) for later fusion
        pool4 = self.layer3(x)  # Stride 16 (1024 channels) - MID-LEVEL FEATURES
        
        # Continue to deepest layer
        x = self.layer4(pool4)  # Stride 32 (2048 channels) - HIGH-LEVEL FEATURES

        # Score layers: convert to class predictions
        x = self.score_fr(x)           # Deep features: stride 32, n_classes channels
        pool4 = self.score_pool4(pool4)  # Mid features: stride 16, n_classes channels

        # Upsample deep features by 2× (32 → 16) to match pool4 resolution
        x = self.upscore2(x)
        
        # Align shapes if needed (handle any size mismatches)
        if x.shape != pool4.shape:
            x = F.interpolate(x, size=pool4.shape[2:], mode='bilinear', align_corners=False)
        
        # FUSION: Combine deep (coarse) and mid (detailed) features
        # Element-wise addition merges semantic and spatial information
        x = x + pool4
        
        # Final upsample by 16× to restore original resolution
        x = self.upscore16(x)
        
        # Ensure output exactly matches input size
        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)

        return x
    
    def _initialize_weights(self):
        """Initialize upsampling and score layers (don't touch pretrained ResNet)"""
        # Initialize upsampling layers with bilinear interpolation
        for layer in [self.upscore2, self.upscore16]:
            if isinstance(layer, nn.ConvTranspose2d):
                in_ch, out_ch, h, w = layer.weight.data.size()
                weight = self._get_bilinear_filter(h, w, in_ch, out_ch)
                layer.weight.data.copy_(weight)
        
        # Initialize score layers with small random weights
        for score_layer in [self.score_pool4, self.score_fr]:
            nn.init.normal_(score_layer.weight, std=0.01)
            if score_layer.bias is not None:
                nn.init.constant_(score_layer.bias, 0)
    
    def _get_bilinear_filter(self, kernel_h, kernel_w, in_channels, out_channels):
        """Generate bilinear interpolation weights"""
        factor = (kernel_h + 1) // 2
        if kernel_h % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:kernel_h, :kernel_w]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        filt = torch.from_numpy(filt).float()
        weight = torch.zeros(in_channels, out_channels, kernel_h, kernel_w)
        for i in range(min(in_channels, out_channels)):
            weight[i, i, :, :] = filt
        return weight



class FCN8s(nn.Module):
    """Fully Convolutional Network with two skip connections (pool3 and pool4).
    
    Most advanced FCN variant with two progressive skip connections from layer2 (pool3)
    and layer3 (pool4), providing the best spatial detail among FCN family.
    
    Architecture:
        Input → ResNet50 Encoder
        ├─ layer2 (pool3, stride 8) → Score Layer → ──────────┐
        ├─ layer3 (pool4, stride 16) → Score Layer → ────┐    │
        └─ layer4 (stride 32) → Score → Up 2× → Add → Up 2× → Add → Up 8× → Output
    
    Progressive Fusion:
        1. First fusion: Upsample layer4 (32→16) + Add with layer3
        2. Second fusion: Upsample result (16→8) + Add with layer2
        3. Final upsample: 8× to original resolution
    
    CRITICAL IMPLEMENTATION NOTE:
        - DO NOT reinitialize ALL Conv2d layers (destroys pretrained ResNet weights)
        - ONLY initialize new layers: score_pool3, score_pool4, score_fr, upscore layers
        - Use small std (0.01) for score layers and bilinear filters for upsampling
    
    Common Issues:
        - If training collapses (NaNs, mIoU→0): check initialization doesn't touch ResNet
        - If predictions are all background: verify skip connections are properly aligned
    
    Parameters:
        n_classes (int): Number of segmentation classes
    
    Expected Performance:
        - mIoU: 60-65% (best among FCN variants)
        - Parameters: ~23.71M (fewer than FCN-32s due to optimization)
    """
    
    def __init__(self, n_classes=21):
        super().__init__()
        
        # Task 1.1: Load pretrained ResNet50 and extract layers
        # 1. Load models.resnet50(weights='DEFAULT')
        # 2. Extract conv1, bn1, relu, maxpool
        # 3. Extract layer1 (stride 4)
        # 4. Extract layer2 (stride 8, will be pool3)
        # 5. Extract layer3 (stride 16, will be pool4)
        # 6. Extract layer4 (stride 32)

        resnet = models.resnet50(weights='DEFAULT')

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2 
        self.layer3 = resnet.layer3 
        self.layer4 = resnet.layer4
        
        # Task 1.2: Add score layers (1x1 convolutions)
        # 1. score_pool3: nn.Conv2d(512, n_classes, 1) - layer2 outputs 512 channels
        # 2. score_pool4: nn.Conv2d(1024, n_classes, 1) - layer3 outputs 1024 channels
        # 3. score_fr: nn.Conv2d(2048, n_classes, 1) - layer4 outputs 2048 channels
        self.score_pool3 = nn.Conv2d(512, n_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(1024, n_classes, kernel_size=1)
        self.score_fr = nn.Conv2d(2048, n_classes, kernel_size=1)

        # Task 1.3: Add upsampling layers
        # 1. upscore2: nn.ConvTranspose2d for 2x upsampling (32 -> 16)
        # 2. upscore_pool4: nn.ConvTranspose2d for 2x upsampling (16 -> 8)
        # 3. upscore8: nn.ConvTranspose2d for 8x upsampling (8 -> 1)
        # All should have: (n_classes, n_classes, kernel_size=4 or 16, stride=2 or 8, bias=False)
        
        self.upscore2 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.upscore8 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=16, stride=8, padding=4, bias=False)
        
        # Initialize upsampling layers with bilinear weights
        self._initialize_weights()
        
    def forward(self, x):
        # Task 1.4: Implement forward pass with progressive skip fusion
        # ENCODER PATH:
        input_size = x.shape[2:]
        
        # Initial layers (combined for efficiency)
        x = self.relu(self.bn1(self.conv1(x)))  # Stride 2
        x = self.maxpool(x)  # Stride 4
        x = self.layer1(x)   # Stride 4 (256 channels)
        
        # SKIP CONNECTION 1: Save pool3 (layer2 output) - FINE DETAILS
        pool3 = self.layer2(x)  # Stride 8 (512 channels) - captures fine spatial details
        
        # SKIP CONNECTION 2: Save pool4 (layer3 output) - MID-LEVEL FEATURES  
        pool4 = self.layer3(pool3)  # Stride 16 (1024 channels) - semantic understanding
        
        # Deepest layer - HIGH-LEVEL SEMANTICS
        x = self.layer4(pool4)  # Stride 32 (2048 channels) - what objects are present
        
        # SCORE LAYERS: Convert multi-scale features to class predictions
        score_fr = self.score_fr(x)          # Deep: stride 32 (semantic, coarse)
        score_pool4 = self.score_pool4(pool4)  # Mid: stride 16 (structure)
        score_pool3 = self.score_pool3(pool3)  # Shallow: stride 8 (boundaries)
        
        # PROGRESSIVE UPSAMPLING WITH SKIP CONNECTIONS:
        # Strategy: gradually recover spatial detail by fusing features at multiple scales
        
        # First fusion: Combine deep (stride 32) with mid-level (stride 16)
        upscore2 = self.upscore2(score_fr)  # Upsample 32 → 16

        # Align shapes if needed (handle any size mismatches from conv/deconv)
        if upscore2.shape != score_pool4.shape:
            upscore2 = F.interpolate(upscore2, size=score_pool4.shape[2:], mode='bilinear', align_corners=False)
        
        # Fuse: deep semantics + mid-level structure
        fuse_pool4 = upscore2 + score_pool4  # Element-wise addition
        
        # Second fusion: Combine fused features (stride 16) with shallow (stride 8)
        upscore_pool4 = self.upscore_pool4(fuse_pool4)  # Upsample 16 → 8

        # Align shapes
        if upscore_pool4.shape != score_pool3.shape:
            upscore_pool4 = F.interpolate(upscore_pool4, size=score_pool3.shape[2:], mode='bilinear', align_corners=False)

        # Fuse: semantic understanding + fine spatial details
        fuse_pool3 = upscore_pool4 + score_pool3  # Element-wise addition
        
        # Final upsampling: Restore original resolution
        out = self.upscore8(fuse_pool3)  # Upsample 8 → 1 (full resolution)
        
        # Ensure output exactly matches input size
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        
        return out
    
    def _initialize_weights(self):
        """Initialize ConvTranspose2d and score layers with proper weights"""
        # Initialize upsampling layers with bilinear interpolation
        for m in [self.upscore2, self.upscore_pool4, self.upscore8]:
            if isinstance(m, nn.ConvTranspose2d):
                in_ch, out_ch, h, w = m.weight.data.size()
                weight = self._get_bilinear_filter(h, w, in_ch, out_ch)
                m.weight.data.copy_(weight)
        
        # Initialize score layers with small random weights (don't touch ResNet!)
        for m in [self.score_pool3, self.score_pool4, self.score_fr]:
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def _get_bilinear_filter(self, kernel_h, kernel_w, in_channels, out_channels):
        """Generate bilinear interpolation weights"""
        factor = (kernel_h + 1) // 2
        if kernel_h % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:kernel_h, :kernel_w]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        filt = torch.from_numpy(filt).float()
        weight = torch.zeros(in_channels, out_channels, kernel_h, kernel_w)
        for i in range(min(in_channels, out_channels)):
            weight[i, i, :, :] = filt
        return weight
    


class FCN8sKCCS(FCN8s):
    """
    FCN-8s + bloque de contexto KCCS en el mapa de características más profundo (layer4).

    Igual que tu FCN8s:
        - Mismo backbone ResNet50
        - Mismos skips pool3/pool4
        - Mismos upsampling y score layers

    Única diferencia:
        - Antes de aplicar score_fr, pasamos x por un bloque KCCSContextBlock
          que implementa la interacción S/E/I con kernels gaussianos.
    """
    def __init__(self, n_classes=21, kccs_dims=(64, 64, 64)):
        super().__init__(n_classes=n_classes)
        s_dim, e_dim, i_dim = kccs_dims

        # Mapa profundo de ResNet50 layer4 tiene 2048 canales
        self.kccs_block = KCCSContextBlock(
            in_channels=2048,
            s_dim=s_dim,
            e_dim=e_dim,
            i_dim=i_dim
        )

    def forward(self, x):
        # Calc input size para el upsampling final
        input_size = x.shape[2:]

        # ENCODER (igual que FCN8s)
        x = self.relu(self.bn1(self.conv1(x)))  # Stride 2
        x = self.maxpool(x)                    # Stride 4
        x = self.layer1(x)                     # Stride 4 (256 canales)

        # SKIP 1: pool3 (layer2 output) - FINE DETAILS
        pool3 = self.layer2(x)                 # Stride 8 (512 canales)

        # SKIP 2: pool4 (layer3 output) - MID-LEVEL
        pool4 = self.layer3(pool3)             # Stride 16 (1024 canales)

        # Deepest layer - HIGH-LEVEL SEMANTICS
        x = self.layer4(pool4)                 # Stride 32 (2048 canales)

        # 🔵 AQUÍ entra KCCS: razonamiento S/E/I sobre el mapa profundo
        x = self.kccs_block(x)                 # Mantiene shape: [B, 2048, H/32, W/32]

        # SCORE LAYERS (igual que FCN8s)
        score_fr = self.score_fr(x)            # Deep: stride 32
        score_pool4 = self.score_pool4(pool4)  # Mid: stride 16
        score_pool3 = self.score_pool3(pool3)  # Shallow: stride 8

        # PROGRESSIVE UPSAMPLING (igual que FCN8s)

        # 1) Fusion profunda: 32 → 16 + pool4
        upscore2 = self.upscore2(score_fr)     # Upsample 32 → 16

        if upscore2.shape != score_pool4.shape:
            upscore2 = F.interpolate(
                upscore2,
                size=score_pool4.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        fuse_pool4 = upscore2 + score_pool4

        # 2) Fusion media: 16 → 8 + pool3
        upscore_pool4 = self.upscore_pool4(fuse_pool4)  # 16 → 8

        if upscore_pool4.shape != score_pool3.shape:
            upscore_pool4 = F.interpolate(
                upscore_pool4,
                size=score_pool3.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        fuse_pool3 = upscore_pool4 + score_pool3

        # 3) Upsampling final: 8 → 1 (resolución original)
        out = self.upscore8(fuse_pool3)

        if out.shape[2:] != input_size:
            out = F.interpolate(
                out,
                size=input_size,
                mode='bilinear',
                align_corners=False
            )

        return out




# ================== Part 2: DeepLabV3+ Architecture ==================

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (ASPP) module for multi-scale feature extraction.
    
    ASPP is the core component of DeepLabV3/V3+ that captures multi-scale context
    by applying parallel atrous (dilated) convolutions with different dilation rates.
    
    Architecture (5 parallel branches):
        Branch 1: 1×1 conv (captures fine details)
        Branch 2: 3×3 atrous conv, rate=6 (receptive field ~13×13)
        Branch 3: 3×3 atrous conv, rate=12 (receptive field ~25×25)
        Branch 4: 3×3 atrous conv, rate=18 (receptive field ~37×37)
        Branch 5: Global average pooling + 1×1 conv (captures global context)
    
    All branches output 'out_channels' feature maps, which are concatenated
    (5 × out_channels total) and then projected back to out_channels.
    
    Why ASPP works:
        - Multi-scale context: Different dilation rates capture objects at various scales
        - Maintains resolution: Unlike pooling, atrous conv doesn't reduce spatial size
        - Global + local: Combines pixel-level detail with image-level context
    
    Parameters:
        in_channels (int): Number of input channels (typically 2048 for ResNet50 layer4)
        out_channels (int): Number of output channels for each branch (default: 256)
        rates (list): Dilation rates for atrous convolutions (default: [6, 12, 18])
    
    Input/Output:
        Input: (B, in_channels, H, W) - typically stride 16 or 32
        Output: (B, out_channels, H, W) - same spatial dimensions
    """
    
    def __init__(self, in_channels, out_channels=256, rates=[6, 12, 18]):
        super().__init__()
        
        # Task 2.1: Implement Branch 1 - 1x1 convolution
        # 1. Create nn.Sequential with:
        #    - nn.Conv2d(in_channels, out_channels, 1, bias=False)
        #    - nn.BatchNorm2d(out_channels)
        #    - nn.ReLU(inplace=True)

        self.conv1x1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(inplace=True)
                                    )
        # Task 2.1: Implement Branches 2-4 - Atrous convolutions
        # 1. Create nn.ModuleList()
        # 2. For each rate in rates [6, 12, 18]:
        #    - Append nn.Sequential with:
        #      * nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False)
        #      * nn.BatchNorm2d(out_channels)
        #      * nn.ReLU(inplace=True)

        self.atrous_convs = nn.ModuleList()
        for rate in rates:
            self.atrous_convs.append(
                nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False),
                              nn.BatchNorm2d(out_channels),
                              nn.ReLU(inplace=True)
                            )
                        )
        
        # Task 2.1: Implement Branch 5 - Global average pooling
        # 1. Create nn.Sequential with:
        #    - nn.AdaptiveAvgPool2d(1) - Pools to 1x1
        #    - nn.Conv2d(in_channels, out_channels, 1, bias=False) ; Proyectar a out_channels
        #    - nn.BatchNorm2d(out_channels)
        #    - nn.ReLU(inplace=True)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), 
                                            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), 
                                            nn.BatchNorm2d(out_channels),
                                            nn.ReLU(inplace=True)
                                            )
        
        # Task 2.2: Implement fusion layer
        # 1. Create nn.Sequential with:
        #    - nn.Conv2d(5 * out_channels, out_channels, 1, bias=False)
        # Los canales de entrada son 5 * out_channels porque hay 5 ramas concatenadas (1x1 + 3 atrous + 1 global)
        #    - nn.BatchNorm2d(out_channels)
        #    - nn.ReLU(inplace=True)
        #    - nn.Dropout(0.1) para regularización o reducción de overfitting
        
        self.conv_out = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, kernel_size=1, bias=False),
                                      nn.BatchNorm2d(out_channels),
                                      nn.ReLU(inplace=True),
                                      nn.Dropout(0.1)
                                    )
        
    def forward(self, x):
        # Task 2.2: Apply all branches and concatenate
        # 1. Save spatial dimensions: size = x.shape[2:]
        size = x.shape[2:]  # (H, W) - needed to upsample global features back
        
        # BRANCH 1: 1×1 convolution (local, fine-grained features)
        # Captures pixel-level detail without spatial context
        feat1 = self.conv1x1(x)  # Shape: (B, 256, H, W)
        
        # BRANCHES 2-4: Atrous convolutions with different dilation rates
        # Each rate captures context at a different scale:
        #   rate=6:  receptive field ~13×13 (small objects, local context)
        #   rate=12: receptive field ~25×25 (medium objects)
        #   rate=18: receptive field ~37×37 (large objects, wider context)
        # All maintain spatial resolution (no downsampling)
        feat_atrous = [conv(x) for conv in self.atrous_convs]  # List of 3 tensors, each (B, 256, H, W)

        # BRANCH 5: Global Average Pooling (image-level context)
        # Captures the overall scene context (e.g., "outdoor", "indoor")
        # Pools entire feature map to 1×1, then projects to 256 channels
        feat_global = self.global_avg_pool(x)  # Shape: (B, 256, 1, 1)

        # Upsample global features back to original spatial size
        # This broadcasts the global context to every spatial location
        feat_global_upsampled = F.interpolate(feat_global, size=size, mode='bilinear', align_corners=False)
        # Shape: (B, 256, H, W)

        # CONCATENATION: Combine all 5 branches along channel dimension
        # Result: [local, scale1, scale2, scale3, global] = 5 × 256 = 1280 channels
        all_features = [feat1] + feat_atrous + [feat_global_upsampled]
        feat = torch.cat(all_features, dim=1)  # Shape: (B, 1280, H, W)
        
        # FUSION: Project concatenated features back to out_channels (256)
        # Learns optimal combination of multi-scale information
        # Includes dropout (0.1) for regularization
        return self.conv_out(feat)  # Shape: (B, 256, H, W)


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ with encoder-decoder architecture and ASPP.
    
    State-of-the-art semantic segmentation architecture that combines:
    - ASPP module for multi-scale context in the encoder
    - Lightweight decoder that recovers spatial details via skip connections
    
    Architecture Overview:
        Encoder Path:
            Input → ResNet50 (conv1, layer1-4)
                   ├─ layer1 (low-level features, stride 4) ───────┐
                   └─ layer4 (high-level features, stride 32) → ASPP → Up 4× ─┐
                                                                            │
        Decoder Path:                                                       │
            low-level (256 ch) → 1×1 conv (256→48) ────────────────┘
                                                                            ↓
            Concatenate [ASPP:256, low-level:48] = 304 channels
                                                                            ↓
            3×3 conv (304→256) → 3×3 conv (256→256) → Classifier (256→n_classes)
                                                                            ↓
            Upsample 4× → Output (original resolution)
    
    Key Design Choices:
        - Output stride 16: Balance between speed and accuracy (vs stride 8 or 32)
        - Low-level features from layer1: Captures fine spatial details
        - 48 channels for low-level: Prevents low-level features from dominating
        - Two 3×3 convs in decoder: Refines combined features
    
    Advantages:
        - Excellent boundary delineation (better than FCN-8s)
        - Handles multi-scale objects well (thanks to ASPP)
        - Good speed/accuracy trade-off
    
    Parameters:
        n_classes (int): Number of segmentation classes
        backbone (str): Backbone architecture (default: 'resnet50')
    
    Expected Performance:
        - mIoU: 70-75% (with ResNet50, VOC 2012)
        - Parameters: ~40.35M
        - Inference: ~2.14ms/image on RTX 3090
    
    Paper: "Encoder-Decoder with Atrous Separable Convolution for Semantic
            Image Segmentation" (Chen et al., ECCV 2018)
    """
    
    def __init__(self, n_classes=21, backbone='resnet50'):
        super().__init__()
        
        # Task 2.3: Load backbone and extract encoder layers
        # 1. Load models.resnet50(weights='DEFAULT')
        # 2. Extract conv1, bn1, relu, maxpool
        # 3. Extract layer1 (low-level features, stride 4)
        # 4. Extract layer2, layer3, layer4
        
        # 1. Cargar ResNet50
        if backbone == 'resnet50':
            resnet = models.resnet50(weights='DEFAULT')
            low_level_channels = 256  # Salida de ResNet layer1
            high_level_channels = 2048 # Salida de ResNet layer4
        else:
            raise NotImplementedError("Backbone no soportado")

        # 2. Capas iniciales
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # 3. Extraer layer1 (características de bajo nivel, stride 4)
        self.layer1 = resnet.layer1
        
        # 4. Extraer el resto del backbone (encoder profundo)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Task 2.3: Create ASPP module
        # 1. self.aspp = ASPP(2048, 256) - ResNet50 layer4 outputs 2048 channels

        # La entrada son los canales de alto nivel (2048), la salida son 256
        self.aspp = ASPP(high_level_channels, 256)
        
        # Task 2.3: Create low-level feature projection
        # 1. Create nn.Sequential with:
        #    - nn.Conv2d(256, 48, 1, bias=False) - layer1 outputs 256 channels
        #    - nn.BatchNorm2d(48)
        #    - nn.ReLU(inplace=True)

        # Proyecta las características de bajo nivel de 256 -> 48 canales
        self.low_level_conv = nn.Sequential(nn.Conv2d(low_level_channels, 48, kernel_size=1, bias=False),
                                            nn.BatchNorm2d(48),
                                            nn.ReLU(inplace=True)
                                            )
        
        # Task 2.3: Create decoder
        # 1. Create nn.Sequential with:
        #    - nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False) - ASPP (256) + low-level (48)
        #    - nn.BatchNorm2d(256)
        #    - nn.ReLU(inplace=True)
        #    - nn.Conv2d(256, 256, 3, padding=1, bias=False)
        #    - nn.BatchNorm2d(256)
        #    - nn.ReLU(inplace=True)

        # Canales de entrada = 256 (ASPP) + 48 (low-level) = 304
        self.decoder = nn.Sequential(nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True)
                                     )
        
        # Task 2.3: Create classification head
        # 1. nn.Conv2d(256, n_classes, 1)

        # Mapea los 256 canales del decodificador a n_classes
        self.classifier = nn.Conv2d(256, n_classes, kernel_size=1)

        
    def forward(self, x):
        # TODO Task 2.4: Implement forward pass
        # 1. Save input size: size = x.shape[2:]
        input_size = x.shape[2:]  # (H, W) for final upsampling
        
        # ===== ENCODER PATH =====
        # Initial layers: reduce spatial resolution while building features
        x = self.relu(self.bn1(self.conv1(x)))  # Stride 2: H/2 × W/2
        x = self.maxpool(x)  # Stride 4: H/4 × W/4
        
        # LOW-LEVEL FEATURES: capture fine spatial details (edges, textures)
        # These are essential for precise boundary delineation
        low_level_feat = self.layer1(x)  # Stride 4: H/4 × W/4, 256 channels
        # SAVE for skip connection in decoder
        
        # Continue encoder: build semantic understanding
        x = self.layer2(low_level_feat)  # Stride 8: H/8 × W/8, 512 channels
        x = self.layer3(x)  # Stride 16: H/16 × W/16, 1024 channels
        
        # HIGH-LEVEL FEATURES: capture semantic meaning (what objects are present)
        x = self.layer4(x)  # Stride 32: H/32 × W/32, 2048 channels
        
        # ===== ASPP MODULE =====
        # Multi-scale context aggregation with parallel atrous convolutions
        # Captures objects at different scales (small, medium, large)
        x = self.aspp(x)  # Stride 32: H/32 × W/32, 256 channels
        
        # ===== DECODER PATH =====
        # Strategy: combine high-level semantics with low-level spatial details
        
        # Upsample ASPP output by 4×: stride 32 → stride 4
        # This brings semantic features to the same resolution as low-level features
        x = F.interpolate(x, size=low_level_feat.shape[2:], mode='bilinear', align_corners=False)
        # Shape: (B, 256, H/4, W/4)

        # Project low-level features: 256 → 48 channels
        # Reduces dimensionality to prevent low-level features from dominating
        low_level_feat = self.low_level_conv(low_level_feat)
        # Shape: (B, 48, H/4, W/4)

        # CONCATENATE: merge semantic (256) + spatial (48) = 304 channels
        # This fusion combines "what" (semantics) with "where" (boundaries)
        x = torch.cat([x, low_level_feat], dim=1)
        # Shape: (B, 304, H/4, W/4)

        # REFINE: two 3×3 convolutions to blend fused features
        # Learns optimal combination of semantic and spatial information
        x = self.decoder(x)
        # Shape: (B, 256, H/4, W/4)
        
        # ===== CLASSIFICATION =====
        # Final 1×1 conv: map 256 features to n_classes logits
        x = self.classifier(x)
        # Shape: (B, n_classes, H/4, W/4)

        # Final upsample by 4×: restore original resolution
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        # Shape: (B, n_classes, H, W)

        return x


# ================== Part 3: Mini-SAM Architecture ==================

class MiniSAM(nn.Module):
    """Simplified Segment Anything Model (MiniSAM) with interactive prompts.
    
    A lightweight, trainable-from-scratch version of Meta's Segment Anything Model (SAM),
    designed for interactive segmentation with user prompts (points, boxes).
    
    Architecture Components:
        1. Image Encoder (MobileNetV3-Small backbone):
           - Lightweight CNN for feature extraction (~96 channels, stride 8)
           - Projected to embed_dim (256) channels
        
        2. Prompt Encoders (learns to encode user interactions):
           a) Point Encoder:
              - Position encoding: MLP(2→128→embed_dim) for (x,y) coordinates
              - Type encoding: Embedding(2→embed_dim) for fg/bg labels
              - Combined via addition
           
           b) Box Encoder:
              - MLP(4→128→embed_dim) for (x1,y1,x2,y2) coordinates
        
        3. Feature Fusion:
           - Broadcast prompt features to match image feature spatial dims
           - Concatenate: [image_features:256, prompt_features:256] = 512 channels
        
        4. Decoder (4 conv blocks with upsampling):
           - Progressive upsampling from stride 8 to stride 1
           - Outputs 64-channel feature map
        
        5. Output Heads:
           a) Mask Head: Conv 64→n_classes (segmentation logits)
           b) IoU Head: Global pool + Linear → Sigmoid (predicts mask quality)
    
    Interactive Workflow:
        1. User provides initial prompts (e.g., 3 foreground points)
        2. Model generates initial segmentation mask
        3. Model predicts IoU score (mask quality)
        4. User adds correction prompts if needed
        5. Model refines segmentation iteratively
    
    Training Strategy:
        - Simulated prompts: Sample random points from ground truth masks
        - 50% foreground points (from object pixels)
        - 50% background points (from background pixels)
        - Multi-objective loss:
          * Segmentation: CrossEntropy + Dice
          * Quality prediction: MSE(predicted_IoU, true_IoU)
    
    Advantages:
        - Interactive: Allows user refinement unlike fully automatic methods
        - Lightweight: Only ~2.22M parameters (10× smaller than FCN-8s)
        - Fast: ~1.86ms/image inference
        - Quality-aware: Predicts its own mask quality
    
    Disadvantages:
        - Requires prompts (can't work fully automatically)
        - Lower accuracy without good prompts (~45-50% mIoU with auto prompts)
        - Performance depends on prompt quality
    
    Parameters:
        n_classes (int): Number of segmentation classes (default: 21)
        embed_dim (int): Embedding dimension for features and prompts (default: 256)
    
    Use Cases:
        - Interactive annotation tools
        - Few-shot segmentation
        - Scenarios where user can provide clicks/boxes
        - Edge devices (lightweight model)
    
    Paper inspiration: "Segment Anything" (Kirillov et al., ICCV 2023)
    Note: This is a simplified, trainable-from-scratch version, not the full SAM.
    """
    
    def __init__(self, n_classes=21, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Task 3.1: Create lightweight image encoder
        # 1. Load models.mobilenet_v3_small(weights='DEFAULT')
        backbone = models.mobilenet_v3_small(weights='DEFAULT')

        # 2. Extract features: nn.Sequential(*list(backbone.features))
        #    Extraer características hasta antes de la última capa
        #    MobileNetV3-Small tiene 12 bloques, usamos hasta el bloque 11 (índice -2)
        #    Esto nos da 96 canales de salida, no 576
        self.image_encoder = nn.Sequential(*list(backbone.features)[:-2])

        # 3. Create projection: nn.Conv2d(96, embed_dim, 1) - Adjusted for correct channels
        #    Crear proyección de 96 -> embed_dim (ej. 256)
        self.img_proj = nn.Conv2d(96, embed_dim, kernel_size=1)

        # Task 3.2: Create prompt encoders
        # Point type embedding:
        # 1. nn.Embedding(2, embed_dim) - 2 types: foreground (1) and background (0)
        self.point_type_embed = nn.Embedding(2, embed_dim)
        
        # Point position embedding:
        # 2. Create nn.Sequential with:
        #    - nn.Linear(2, 128) - Input is (x, y) coordinates
        #    - nn.ReLU()
        #    - nn.Linear(128, embed_dim)
        self.point_pos_embed = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
        
        # Box embedding:
        # 3. Create nn.Sequential with:
        #    - nn.Linear(4, 128) - Input is (x1, y1, x2, y2)
        #    - nn.ReLU()
        #    - nn.Linear(128, embed_dim)
        self.box_embed = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
        
        # Task 3.3: Create decoder
        # 1. Create nn.Sequential with:
        #    - nn.Conv2d(embed_dim * 2, 256, 3, padding=1) - Image + prompt features
        #    - nn.BatchNorm2d(256)
        #    - nn.ReLU()
        #    - nn.Conv2d(256, 128, 3, padding=1)
        #    - nn.BatchNorm2d(128)
        #    - nn.ReLU()
        #    - nn.Conv2d(128, 64, 3, padding=1)
        #    - nn.BatchNorm2d(64)
        #    - nn.ReLU()
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim * 2, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Task 3.3: Create output heads
        # Mask head:
        # 1. nn.Conv2d(64, n_classes, 1)
        self.mask_head = nn.Conv2d(64, n_classes, kernel_size=1)
        
        # IoU prediction head:
        # 2. Create nn.Sequential with:
        #    - nn.AdaptiveAvgPool2d(1)
        #    - nn.Flatten()
        #    - nn.Linear(64, 1)
        #    - nn.Sigmoid()
        self.iou_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Upsampling:
        # 3. nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        
    def encode_image(self, x):
        """Extract image features using lightweight MobileNetV3 encoder.
        
        Args:
            x: Input image tensor (B, 3, H, W)
            
        Returns:
            features: Encoded features (B, embed_dim, H/8, W/8)
        """
        # Task 3.4: Encode image
        # Extract features using MobileNetV3-Small backbone
        # This is much lighter than ResNet50: ~2M params vs ~25M
        # Output stride: 8 (less aggressive downsampling than FCN/DeepLab)
        features = self.image_encoder(x)  # Shape: (B, 96, H/8, W/8)
        
        # Project to embedding dimension (96 → embed_dim, typically 256)
        # This creates a common feature space for fusion with prompts
        features = self.img_proj(features)  # Shape: (B, embed_dim, H/8, W/8)
        
        return features
    
    def encode_prompts(self, points=None, point_labels=None, boxes=None, img_size=None):
        """
        Encode point and/or box prompts
        Args:
            points: (B, N, 2) normalized [0,1] coordinates
            point_labels: (B, N) with 0=bg, 1=fg
            boxes: (B, 4) normalized [0,1] as [x1, y1, x2, y2]
            img_size: (H, W)
        Returns:
            prompt_features: (B, embed_dim, H/8, W/8)
        """
        # Task 3.4: Encode prompts
        # 1. Get batch size B from points or boxes
        if points is not None:
            B = points.shape[0]
        elif boxes is not None:
            B = boxes.shape[0]
        else:
            raise ValueError("Either points or boxes must be provided")
        
        # 2. Get H, W from img_size
        H, W = img_size
        
        # 3. Create empty list: prompt_features = []
        prompt_features = []
        
        # IF POINTS PROVIDED:
        if points is not None:
            # 4. pos_enc = self.point_pos_embed(points) - Shape: B x N x embed_dim
            pos_enc = self.point_pos_embed(points)
            
            # 5. type_enc = self.point_type_embed(point_labels) - Shape: B x N x embed_dim
            # Convert point_labels to Long for embedding layer
            type_enc = self.point_type_embed(point_labels.long())
            
            # 6. point_enc = pos_enc + type_enc
            point_enc = pos_enc + type_enc
            
            # 7. point_enc = point_enc.mean(dim=1, keepdim=True) - Average over N points
            point_enc = point_enc.mean(dim=1, keepdim=True)
            
            # 8. Append point_enc to prompt_features
            prompt_features.append(point_enc)
        
        # IF BOXES PROVIDED:
        if boxes is not None:
            # 9. box_enc = self.box_embed(boxes).unsqueeze(1) - Shape: B x 1 x embed_dim
            box_enc = self.box_embed(boxes).unsqueeze(1)
            
            # 10. Append box_enc to prompt_features
            prompt_features.append(box_enc)
        
        # COMBINE PROMPTS:
        # 11. If prompt_features not empty:
        if prompt_features:
            #     - Concatenate along dim=1 and take mean: prompt_enc = torch.cat(...).mean(dim=1)
            prompt_enc = torch.cat(prompt_features, dim=1).mean(dim=1)
            
            #     - Reshape to (B, embed_dim, 1, 1)
            prompt_enc = prompt_enc.view(B, self.embed_dim, 1, 1)
            
            #     - Expand to (B, embed_dim, H//8, W//8)
            prompt_enc = prompt_enc.expand(B, self.embed_dim, H // 8, W // 8)
        # 12. Else: create zero tensor of shape (B, embed_dim, H//8, W//8)
        else:
            prompt_enc = torch.zeros(B, self.embed_dim, H // 8, W // 8, device=points.device if points is not None else boxes.device)
        
        # 13. Return prompt_enc
        return prompt_enc
    
    def forward(self, images, points=None, point_labels=None, boxes=None):
        """
        Forward pass
        Args:
            images: B x 3 x H x W
            points: B x N x 2
            point_labels: B x N
            boxes: B x 4
        Returns:
            mask_logits: B x n_classes x H x W
            iou_pred: B x 1
        """
        # Task 3.4: Implement forward pass
        # 1. Get B, C, H, W = images.shape
        B, C, H, W = images.shape
        
        # ENCODE IMAGE:
        # 2. img_features = self.encode_image(images)
        img_features = self.encode_image(images)
        
        # Get actual feature map size from encoder output
        feat_h, feat_w = img_features.shape[2], img_features.shape[3]
        
        # ENCODE PROMPTS:
        # 3. prompt_features = self.encode_prompts(points, point_labels, boxes, img_size=(H, W))
        prompt_features = self.encode_prompts(points, point_labels, boxes, img_size=(H, W))
        
        # Resize prompt features to match image features spatial dimensions
        if prompt_features.shape[2:] != img_features.shape[2:]:
            prompt_features = F.interpolate(prompt_features, size=(feat_h, feat_w), mode='bilinear', align_corners=False)
        
        # FUSE FEATURES:
        # 4. fused = torch.cat([img_features, prompt_features], dim=1)
        fused = torch.cat([img_features, prompt_features], dim=1)
        
        # DECODE:
        # 5. decoded = self.decoder(fused)
        decoded = self.decoder(fused)
        
        # OUTPUT MASK:
        # 6. mask_logits = self.mask_head(decoded)
        mask_logits = self.mask_head(decoded)
        
        # 7. mask_logits = self.upsample(mask_logits)
        mask_logits = self.upsample(mask_logits)
        
        # Ensure output matches input size
        if mask_logits.shape[2:] != (H, W):
            mask_logits = F.interpolate(mask_logits, size=(H, W), mode='bilinear', align_corners=False)
        
        # PREDICT IoU:
        # 8. iou_pred = self.iou_head(decoded)
        iou_pred = self.iou_head(decoded)
        
        # 9. Return mask_logits, iou_pred
        return mask_logits, iou_pred


def sample_points_from_mask(masks, n_points=5):
    """Sample foreground/background points from ground truth masks (simulates user clicks).
    
    This function is crucial for training and evaluating MiniSAM. It simulates
    interactive user prompts by randomly sampling points from ground truth masks.
    
    Sampling Strategy:
        - 50% foreground points: sampled from pixels where mask > 0 (objects)
        - 50% background points: sampled from pixels where mask == 0 (background)
        - Points are normalized to [0, 1] range
        - Labels: 1 for foreground, 0 for background
    
    Why this strategy?
        - Mimics realistic user behavior (clicks on object + corrections on background)
        - Balanced supervision prevents bias toward one class
        - Random sampling ensures diverse prompt configurations
    
    Edge Cases Handled:
        - Image with no foreground: samples only background points
        - Image with no background: samples only foreground points
        - Total points < n_points: pads with zeros
    
    Args:
        masks (torch.Tensor): Ground truth class indices, shape (B, H, W)
                             Values: 0=background, 1-20=objects, 255=ignore
        n_points (int): Total number of points to sample per image (default: 5)
    
    Returns:
        points (torch.Tensor): Normalized coordinates, shape (B, n_points, 2)
                              Values in [0, 1], order: [row/H, col/W]
        labels (torch.Tensor): Point labels, shape (B, n_points)
                              Values: 0=background, 1=foreground
    
    Implementation Details:
        - Uses torch.nonzero() to find valid pixel coordinates
        - torch.randint() for random sampling
        - Handles ignore_index (255) by using valid_mask
        - Normalizes coordinates by dividing by [H, W]
    
    Example:
        >>> masks = torch.tensor([[[0,0,1], [1,1,0]]])  # B=1, H=2, W=3
        >>> points, labels = sample_points_from_mask(masks, n_points=4)
        >>> print(f"Points: {points.shape}, Labels: {labels.shape}")
        Points: torch.Size([1, 4, 2]), Labels: torch.Size([1, 4])
        >>> print(f"Sample point: {points[0,0]}, label: {labels[0,0]}")
        Sample point: tensor([0.5, 0.33]), label: 1  # example values
    
    Use Cases:
        1. Training: Generate prompts for each batch
        2. Validation: Evaluate with simulated user prompts
        3. Interactive demo: Simulate initial user clicks
    
    Performance:
        - Sampling is fast (~1ms for batch of 32 images)
        - Negligible overhead compared to forward pass
    """
    # Task 3.5: Implement point sampling
    # 1. Get B, H, W = masks.shape
    B, H, W = masks.shape  # Batch size, Height, Width
    device = masks.device  # Ensure tensors are on same device (cuda/cpu)
    
    # 2. Create empty lists: points_list = [], labels_list = []
    points_list = []  # Will store (n_points, 2) for each image
    labels_list = []  # Will store (n_points,) for each image
    
    # FOR EACH IMAGE IN BATCH (process independently):
    # 3. For b in range(B):
    for b in range(B):
        #    - Get mask = masks[b]
        mask = masks[b]  # Single image mask: (H, W) with class indices
        valid_mask = (mask != 255)  # Exclude ignore pixels (borders/uncertain regions)
        
        #    SAMPLE FOREGROUND POINTS (50%):
        #    4. fg_indices = torch.nonzero(mask > 0) - Find all foreground pixels
        # Foreground = any object class (1-20 for VOC), excluding background (0) and ignore (255)
        fg_indices = torch.nonzero((mask > 0) & valid_mask, as_tuple=False)  # Shape: (num_fg_pixels, 2) - [row, col] coords
        
        #    5. If fg_indices not empty:
        if len(fg_indices) > 0:
            #       - Randomly sample n_points//2 indices
            n_fg = n_points // 2  # Half of points are foreground (e.g., 2-3 for n_points=5)
            sampled_idx = torch.randint(0, len(fg_indices), (n_fg,), device=device)  # Random indices into fg_indices
            fg_points = fg_indices[sampled_idx].float()  # Sample n_fg coordinates: (n_fg, 2)
            
            #       - Normalize to [0,1]: divide by [H, W]
            # Model expects normalized coordinates (0=top/left, 1=bottom/right)
            fg_points[:, 0] /= H  # Normalize row (y-coordinate)
            fg_points[:, 1] /= W  # Normalize column (x-coordinate)
            
            #       - Create labels as ones
            fg_labels = torch.ones(n_fg, dtype=torch.long, device=masks.device)  # Label=1 means foreground click
        #    6. Else: create empty tensors
        else:
            # Edge case: image has no foreground (e.g., all background)
            fg_points = torch.zeros(0, 2, device=device)
            fg_labels = torch.zeros(0, dtype=torch.long, device=device)
        
        #    SAMPLE BACKGROUND POINTS (50%):
        #    7. bg_indices = torch.nonzero(mask == 0) - Find all background pixels
        bg_indices = torch.nonzero((mask == 0) & valid_mask, as_tuple=False)  # Background = class 0
        
        #    8. If bg_indices not empty:
        if len(bg_indices) > 0:
            #       - Randomly sample n_points//2 indices
            n_bg = n_points - (n_points // 2)  # Remaining points (e.g., 2-3 for n_points=5)
            sampled_idx = torch.randint(0, len(bg_indices), (n_bg,), device=device)
            bg_points = bg_indices[sampled_idx].float()  # Sample n_bg coordinates: (n_bg, 2)
            
            #       - Normalize to [0,1]: divide by [H, W]
            bg_points[:, 0] /= H  # Normalize row
            bg_points[:, 1] /= W  # Normalize column
            
            #       - Create labels as zeros
            bg_labels = torch.zeros(n_bg, dtype=torch.long, device=masks.device)  # Label=0 means background click
        #    9. Else: create empty tensors
        else:
            # Edge case: image has no background (rare)
            bg_points = torch.zeros(0, 2, device=device)
            bg_labels = torch.zeros(0, dtype=torch.long, device=device)
        
        #    COMBINE AND PAD:
        #    10. Concatenate fg_points and bg_points
        combined_points = torch.cat([fg_points, bg_points], dim=0)  # Stack foreground + background points
        
        #    11. Concatenate fg_labels and bg_labels
        combined_labels = torch.cat([fg_labels, bg_labels], dim=0)  # Stack corresponding labels
        
        #    12. If total points < n_points: pad with zeros
        # Handle edge case where image doesn't have enough fg/bg pixels
        if len(combined_points) < n_points:
            pad_size = n_points - len(combined_points)
            pad_points = torch.zeros(pad_size, 2, device=device)  # Dummy points (will be ignored by model)
            pad_labels = torch.zeros(pad_size, dtype=torch.long, device=device)  # Dummy labels
            combined_points = torch.cat([combined_points, pad_points], dim=0)
            combined_labels = torch.cat([combined_labels, pad_labels], dim=0)
        
        #    13. Append to lists
        points_list.append(combined_points)  # Add to batch: (n_points, 2)
        labels_list.append(combined_labels)  # Add to batch: (n_points,)
    
    # 14. Stack lists and return torch.stack(points_list), torch.stack(labels_list)
    # Convert list of tensors to batched tensors: (B, n_points, 2) and (B, n_points)
    return torch.stack(points_list), torch.stack(labels_list)


# ================== Loss Functions ==================

class DiceLoss(nn.Module):
    """Dice Loss for semantic segmentation.
    
    Dice loss directly optimizes the Dice coefficient (F1-score), which is closely
    related to IoU. Unlike CrossEntropy, it handles class imbalance naturally.
    
    Formula:
        Dice Coefficient: DC = (2 * |X ∩ Y|) / (|X| + |Y|)
        Dice Loss: L = 1 - DC
    
    where X is the predicted mask and Y is the ground truth.
    
    Advantages:
        - Handles class imbalance (large background vs small objects)
        - Differentiable approximation of IoU
        - Works well for segmentation tasks
    
    Parameters:
        smooth (float): Smoothing factor to avoid division by zero (default: 1e-6)
        ignore_index (int): Index to ignore in target (default: 255 for VOC)
    
    Input:
        pred: (B, C, H, W) - logits (before softmax)
        target: (B, H, W) - class indices
    
    Output:
        loss: scalar tensor
    
    Implementation Notes:
        - Applies softmax internally to get probabilities
        - Converts target to one-hot encoding
        - Computes Dice per class, then averages
        - Handles ignore_index by masking
    """
    
    def __init__(self, smooth=1e-6, ignore_index=255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        
    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) logits
            target: (B, H, W) class indices
        """    
        pred = torch.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target_processed = target.clone()
        valid_mask = (target_processed != self.ignore_index).unsqueeze(1)
        target_processed = target_processed.masked_fill(valid_mask.squeeze(1) == 0, 0)
        target_processed = target_processed.clamp(min=0, max=num_classes - 1)
        target_one_hot = F.one_hot(target_processed, num_classes=num_classes).permute(0, 3, 1, 2).float()

        valid_mask = valid_mask.float()
        pred = pred * valid_mask
        target_one_hot = target_one_hot * valid_mask

        B, C, H, W = pred.shape
        pred_flat = pred.reshape(B, C, -1)
        target_flat = target_one_hot.reshape(B, C, -1)

        intersection = (pred_flat * target_flat).sum(dim=2)
        denominator = pred_flat.sum(dim=2) + target_flat.sum(dim=2) + self.smooth
        dice = (2. * intersection + self.smooth) / denominator
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """Focal Loss for handling extreme class imbalance.
    
    Focal Loss down-weights easy examples and focuses training on hard examples.
    Particularly effective for datasets with extreme class imbalance (e.g., small
    objects in large backgrounds).
    
    Formula:
        FL(p_t) = -α (1 - p_t)^γ log(p_t)
    
    where:
        - p_t: probability of the correct class
        - α (alpha): balancing factor for positive/negative classes
        - γ (gamma): focusing parameter (higher = more focus on hard examples)
    
    Intuition:
        - Easy examples (p_t → 1): (1-p_t)^γ → 0, loss ≈ 0 (ignored)
        - Hard examples (p_t → 0): (1-p_t)^γ → 1, loss is standard CE (focused)
    
    Example:
        - p_t = 0.9 (easy), γ=2: weight = (1-0.9)^2 = 0.01 (99% reduction)
        - p_t = 0.5 (hard), γ=2: weight = (1-0.5)^2 = 0.25 (75% reduction)
        - p_t = 0.1 (very hard), γ=2: weight = (1-0.1)^2 = 0.81 (minimal reduction)
    
    Parameters:
        alpha (float): Balancing factor (default: 0.25, favors positives)
        gamma (float): Focusing parameter (default: 2.0, standard value)
        ignore_index (int): Index to ignore (default: 255)
    
    Input:
        pred: (B, C, H, W) - logits
        target: (B, H, W) - class indices
    
    Output:
        loss: scalar tensor
    
    Use Cases:
        - Extreme class imbalance (e.g., 99% background, 1% objects)
        - Small object detection/segmentation
        - Complement to Dice loss
    
    Paper: "Focal Loss for Dense Object Detection" (Lin et al., ICCV 2017)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        
    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) logits
            target: (B, H, W) class indices
        """
        ce_loss = F.cross_entropy(pred, target, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        valid_mask = (target != self.ignore_index).float()
        focal_loss = focal_loss * valid_mask
        valid_elements = valid_mask.sum()
        if valid_elements == 0:
            return torch.tensor(0.0, device=pred.device)
        return focal_loss.sum() / valid_elements


class CombinedLoss(nn.Module):
    """Combined loss function: CrossEntropy + Dice + Focal.
    
    Combines three complementary loss functions for robust segmentation training:
    
    1. CrossEntropy (30%): Standard pixel-wise classification loss
       - Penalizes incorrect class predictions
       - Well-established, stable training signal
    
    2. Dice Loss (50%): Directly optimizes IoU-like metric
       - Handles class imbalance
       - Aligns loss with evaluation metric (mIoU)
    
    3. Focal Loss (20%): Focuses on hard examples
       - Down-weights easy examples
       - Helps with small objects and boundaries
    
    Why combine?
        - CE provides stable gradients and pixel-wise accuracy
        - Dice optimizes for segmentation quality (IoU)
        - Focal handles extreme imbalance and hard cases
        - Together, they cover different aspects of the segmentation task
    
    Default Weights Rationale:
        - Dice 50%: Primary focus on segmentation quality
        - CE 30%: Stable baseline, prevents collapse
        - Focal 20%: Fine-tunes on hard cases
    
    Parameters:
        weights (dict): Loss weights, e.g., {'ce': 0.3, 'dice': 0.5, 'focal': 0.2}
        ignore_index (int): Index to ignore in target (default: 255)
    
    Input:
        pred: (B, C, H, W) - logits
        target: (B, H, W) - class indices
    
    Output:
        loss: scalar tensor (weighted sum of all three losses)
    
    Tuning Tips:
        - Increase Dice weight if mIoU is low but pixel accuracy is high
        - Increase Focal weight if small objects are poorly segmented
        - Increase CE weight if training is unstable
    """
    
    def __init__(self, weights={'ce': 0.3, 'dice': 0.5, 'focal': 0.2}, ignore_index=255):
        super().__init__()
        self.weights = weights
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        self.focal_loss = FocalLoss(ignore_index=ignore_index)
        
    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        return self.weights['ce'] * ce + self.weights['dice'] * dice + self.weights['focal'] * focal


# ================== Evaluation Metrics ==================

def calculate_miou(pred, target, num_classes, ignore_index=255):
    """Calculate mean Intersection over Union (mIoU) for semantic segmentation.
    
    mIoU is the primary evaluation metric for semantic segmentation. It measures
    the average overlap between predicted and ground truth masks across all classes.
    
    Formula (per class):
        IoU_c = TP_c / (TP_c + FP_c + FN_c)
              = intersection_c / union_c
    
    where:
        - TP: True Positives (correctly predicted pixels of class c)
        - FP: False Positives (incorrectly predicted as class c)
        - FN: False Negatives (missed pixels of class c)
    
    mIoU = mean(IoU across all classes)
    
    Interpretation:
        - mIoU = 0.0: No overlap (worst)
        - mIoU = 0.5: Moderate overlap
        - mIoU = 0.7: Good segmentation
        - mIoU = 0.9+: Excellent (near state-of-the-art)
    
    Args:
        pred (torch.Tensor): Predicted class indices, shape (B, H, W)
        target (torch.Tensor): Ground truth class indices, shape (B, H, W)
        num_classes (int): Number of classes (e.g., 21 for PASCAL VOC)
        ignore_index (int): Index to ignore in target (default: 255)
    
    Returns:
        miou (float): Mean IoU across all classes (ignoring NaN for absent classes)
        class_iou (np.ndarray): IoU per class, shape (num_classes,)
                                NaN for classes not present in ground truth
    
    Implementation Notes:
        - Handles ignore_index by masking those pixels
        - Uses np.nanmean to ignore classes not present in the batch
        - Returns both global mIoU and per-class IoU for detailed analysis
    
    Example:
        >>> pred = torch.tensor([[0, 1, 1], [1, 2, 2]])
        >>> target = torch.tensor([[0, 1, 2], [1, 2, 2]])
        >>> miou, class_ious = calculate_miou(pred, target, num_classes=3)
        >>> print(f"mIoU: {miou:.2%}, Class IoUs: {class_ious}")
    """
    # Task 4.4: Implement mIoU
    # 1. Create empty list: ious = []
    ious = []

    # 2. Convert pred and target to numpy
    mask = (target != ignore_index)
    pred = pred.clone()
    pred = torch.where(mask, pred, torch.full_like(pred, ignore_index))
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    mask_np = mask.cpu().numpy()
    pred = np.where(mask_np, pred, -1)
    target = np.where(mask_np, target, -1)

    # 3. For each class c in range(num_classes):
    for c in range(num_classes):
        #    - pred_mask = (pred == c)
        #    - target_mask = (target == c)
        #    - intersection = logical_and(pred_mask, target_mask).sum()
        #    - union = logical_or(pred_mask, target_mask).sum()
        #    - If union == 0: iou = nan (class not present)
        #    - Else: iou = intersection / union
        #    - Append iou to ious
        pred_mask = (pred == c)
        target_mask = (target == c)
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        if union == 0:
            iou = float('nan')  # Class not present
        else:
            iou = intersection / union
        ious.append(iou)

    # 4. Convert ious to numpy array
    ious = np.array(ious)

    # 5. Compute miou = nanmean(ious) - Ignores NaN values
    miou = np.nanmean(ious)

    # 6. Return miou, ious
    return miou, ious

def calculate_pixel_accuracy(pred, target, ignore_index=255):
    """Calculate pixel-wise accuracy for semantic segmentation.
    
    Pixel Accuracy (PA) is the simplest segmentation metric, measuring the
    percentage of correctly classified pixels.
    
    Formula:
        PA = (Number of correctly classified pixels) / (Total valid pixels)
           = (TP + TN) / (TP + TN + FP + FN)
    
    Advantages:
        - Simple, intuitive metric
        - Easy to compute
        - Useful as a secondary metric
    
    Disadvantages:
        - Dominated by majority classes (e.g., background)
        - Can be high even with poor segmentation of small objects
        - Not as informative as mIoU for segmentation quality
    
    Example:
        - Image with 90% background, 10% objects
        - Predicting all background: PA = 90% (but mIoU ≈ 0%!)
        - This is why mIoU is preferred for evaluation
    
    Args:
        pred (torch.Tensor): Predicted class indices, shape (B, H, W)
        target (torch.Tensor): Ground truth class indices, shape (B, H, W)
        ignore_index (int): Index to ignore (default: 255)
    
    Returns:
        pa (float): Pixel accuracy as a fraction in [0, 1]
    
    Note:
        - Always use alongside mIoU for complete evaluation
        - High PA but low mIoU indicates class imbalance issues
    """
    # Task 4.5: Implement pixel accuracy
    # 1. correct = (pred == target).sum()
    valid_mask = (target != ignore_index)
    correct = ((pred == target) & valid_mask).float().sum()
    total = valid_mask.float().sum()
    if total == 0:
        return 0.0
    return (correct / total).item()


def compute_batch_iou(pred, target, ignore_index=255):
    """Compute IoU for each image in a batch (used for MiniSAM IoU head supervision).
    
    Unlike calculate_miou which averages across classes, this function computes
    a single IoU score per image, treating all non-background pixels as positive.
    
    Purpose:
        - Supervise MiniSAM's IoU prediction head
        - Provide per-image quality metric
        - Enable mask quality ranking
    
    Formula (per image):
        IoU = intersection / union
        where:
            - intersection = pixels correctly predicted as foreground
            - union = all pixels predicted or labeled as foreground
    
    Args:
        pred (torch.Tensor): Predicted class indices, shape (B, H, W)
        target (torch.Tensor): Ground truth class indices, shape (B, H, W)
        ignore_index (int): Index to ignore (default: 255)
    
    Returns:
        iou (torch.Tensor): IoU per image, shape (B,)
    
    Use Case (MiniSAM training):
        1. Generate segmentation mask
        2. Compute true IoU using this function
        3. Compare with IoU head's prediction
        4. Supervise IoU head with MSE loss
    
    Example:
        >>> pred = torch.tensor([[[0,1,1], [1,1,0]]])  # B=1
        >>> target = torch.tensor([[[0,1,0], [1,1,1]]])
        >>> iou = compute_batch_iou(pred, target)
        >>> print(f"Image IoU: {iou.item():.2%}")  # e.g., 60%
    """
    # Task 4.6: Implement batch IoU
    # Get batch size
    B = pred.shape[0] 

    # Create list to store IoU for each image in the batch
    ious = []

    # Compute IoU independently for each image
    # This allows per-image quality assessment (needed for MiniSAM IoU head)
    for b in range(B):  # Process each image separately
        # Create valid mask (exclude ignore_index pixels)
        valid_mask = (target[b] != ignore_index)
        
        # Binary masks: foreground (any class > 0) vs background (class 0)
        pred_pos = (pred[b] > 0) & valid_mask  # Predicted foreground pixels
        target_pos = (target[b] > 0) & valid_mask  # Actual foreground pixels
        
        # Compute intersection: pixels correctly predicted as foreground
        intersection = ((pred[b] == target[b]) & target_pos).float().sum()
        
        # Compute union: all pixels that are foreground in pred OR target
        union = (pred_pos | target_pos).float().sum()
        
        # Handle edge case: both pred and target have no foreground
        if union == 0:
            iou = torch.tensor(1.0, device=pred.device)  # Perfect agreement (both empty)
        else:
            iou = intersection / (union + 1e-6)  # Add epsilon to prevent division by zero
        
        ious.append(iou)

    # Stack individual IoUs into a batch tensor
    return torch.stack(ious)  # Shape: (B,)


# ================== Training Functions ==================

def train_epoch_fcn(model, dataloader, optimizer, criterion, device, scaler=None):
    """Train FCN/DeepLabV3+ models for one epoch.
    
    This function implements a standard training loop for fully automatic
    segmentation models (FCN-32s/16s/8s, DeepLabV3+) that don't require prompts.
    
    Training Pipeline:
        1. For each batch:
           a) Forward pass (with AMP if scaler provided)
           b) Compute loss (CombinedLoss: CE + Dice + Focal)
           c) Backward pass (with gradient scaling if AMP)
           d) Optimizer step
           e) Calculate batch mIoU for monitoring
        2. Return epoch averages
    
    Automatic Mixed Precision (AMP):
        - If scaler is provided: uses torch.amp.autocast('cuda')
        - Forward and loss computation in fp16/bf16
        - Backward and optimizer steps with gradient scaling
        - 2× faster training, 50% less VRAM, minimal accuracy loss
    
    Args:
        model (nn.Module): Segmentation model (FCN or DeepLabV3+)
        dataloader (DataLoader): Training data loader
        optimizer (torch.optim.Optimizer): Optimizer (typically AdamW)
        criterion (nn.Module): Loss function (typically CombinedLoss)
        device (torch.device): Device to train on (cuda or cpu)
        scaler (torch.amp.GradScaler, optional): Gradient scaler for AMP
    
    Returns:
        avg_loss (float): Average loss over the epoch
        avg_miou (float): Average mIoU over the epoch
    
    Implementation Details:
        - Uses tqdm for progress bar
        - Accumulates metrics for each batch
        - Handles AMP vs non-AMP training transparently
        - Computes mIoU on-the-fly for monitoring (not just loss)
    
    Performance Notes:
        - With AMP on RTX 3090: ~0.9s/epoch (vs 1.8s without AMP)
        - Batch size 32, image size 512×512
        - ~1500 iterations for full VOC training set
    """
    model.train()  # Enable training mode (dropout, batchnorm in training mode)
    total_loss = 0  # Accumulate loss for entire epoch
    total_miou = 0  # Accumulate mIoU for monitoring training progress
    
    # Task 5.1: Implement training loop
    # 1. For images, masks in dataloader:
    # 2. Move to device: images, masks = images.to(device), masks.to(device)
    for images, masks in tqdm(dataloader, desc='Training'):
        # Move batch to GPU (async transfer for speed)
        images, masks = images.to(device), masks.to(device)  # images: (B, 3, H, W), masks: (B, H, W)

        # 3. Zero gradients: optimizer.zero_grad()
        optimizer.zero_grad()  # Clear gradients from previous iteration
        
        # Mixed Precision Training (AMP) - 2× faster, 50% less VRAM
        if scaler is not None:
            with torch.amp.autocast('cuda'):  # Enable automatic mixed precision (fp16/bf16)
                # 4. Forward pass: outputs = model(images)
                outputs = model(images)  # outputs: (B, num_classes, H, W) - logits per pixel
                # 5. Compute loss: loss = criterion(outputs, masks)
                loss = criterion(outputs, masks)  # CombinedLoss: CE + Dice + Focal
            
            # 6. Backward with gradient scaling (prevents underflow in fp16)
            scaler.scale(loss).backward()  # Scale loss before backward to prevent gradient underflow
            # 7. Update weights with unscaling
            scaler.step(optimizer)  # Unscale gradients, then optimizer step
            scaler.update()  # Update scaler for next iteration (dynamic loss scaling)
        else:
            # Standard fp32 training (no AMP)
            # 4. Forward pass: outputs = model(images)
            outputs = model(images)  # outputs: (B, num_classes, H, W)
            # 5. Compute loss: loss = criterion(outputs, masks)
            loss = criterion(outputs, masks)
            # 6. Backward: loss.backward()
            loss.backward()  # Compute gradients
            # 7. Update weights: optimizer.step()
            optimizer.step()  # Update model parameters

        # 8. Calculate metrics: pred = outputs.argmax(dim=1), then miou = calculate_miou(pred, masks, num_classes)
        pred = outputs.argmax(dim=1)  # pred: (B, H, W) - class indices (0-20)
        num_classes = outputs.shape[1]  # Infer num_classes from output channels (21 for VOC)
        miou, _ = calculate_miou(pred, masks, num_classes=num_classes, ignore_index=255)  # ignore border pixels

        # 9. Accumulate: total_loss += loss.item(), total_miou += miou
        total_loss = total_loss + loss.item()  # loss.item() converts to Python float
        total_miou = total_miou + miou  # Batch mIoU (averaged over images in batch)

    # 10. Return averages: total_loss / len(dataloader), total_miou / len(dataloader)
    return total_loss / len(dataloader), total_miou / len(dataloader)  # Epoch averages


def train_epoch_minisam(model, dataloader, optimizer, criterion, device, n_points=5, scaler=None):
    """Train Mini-SAM model for one epoch with simulated prompts.
    
    Training MiniSAM is different from FCN/DeepLab because it requires prompts
    (points or boxes). During training, we simulate user interactions by sampling
    points from ground truth masks.
    
    Training Pipeline:
        1. For each batch:
           a) Sample prompts from ground truth masks:
              - 50% foreground points (from object pixels)
              - 50% background points (from background pixels)
           b) Forward pass with prompts: model(images, points, point_labels)
           c) Compute multi-objective loss:
              - Segmentation: CrossEntropy + Dice (mask quality)
              - Quality: MSE(predicted_IoU, true_IoU) (IoU head supervision)
           d) Backward and optimizer step
           e) Calculate mIoU for monitoring
        2. Return epoch averages
    
    Prompt Simulation Strategy:
        - Simulates realistic user clicks during training
        - Teaches model to segment from sparse point annotations
        - Enables zero-shot generalization to real user prompts at test time
    
    Multi-Objective Loss:
        Total Loss = CE + Dice + 0.1 × IoU_MSE
        
        Where:
        - CE + Dice: Standard segmentation losses
        - IoU_MSE: Supervises IoU prediction head
        - Weight 0.1: IoU loss is auxiliary (prevents overfitting to quality prediction)
    
    Args:
        model (nn.Module): MiniSAM model
        dataloader (DataLoader): Training data loader
        optimizer (torch.optim.Optimizer): Optimizer (typically AdamW)
        criterion (nn.Module): Primary loss (typically CombinedLoss)
        device (torch.device): Device to train on
        n_points (int): Number of points to sample per image (default: 5)
        scaler (torch.amp.GradScaler, optional): Gradient scaler for AMP
    
    Returns:
        avg_loss (float): Average total loss over the epoch
        avg_miou (float): Average mIoU over the epoch
    
    Differences from train_epoch_fcn:
        - Calls sample_points_from_mask() to generate prompts
        - Model forward takes (images, points, point_labels)
        - Additional IoU head loss
        - Slightly slower due to prompt sampling
    
    Performance Notes:
        - Prompt sampling adds ~10% overhead vs FCN training
        - More points (n_points) = better segmentation but slower training
        - Typical n_points: 5-10 for training, 1-3 for fast inference
    """
    model.train()  # Enable training mode
    total_loss = 0  # Accumulate total loss (segmentation + IoU)
    total_miou = 0  # Track segmentation quality
    dice_fn = DiceLoss(ignore_index=255)  # Pre-instantiate Dice loss for efficiency
    
    # Task 5.2: Implement Mini-SAM training
    # 1. For images, masks in dataloader:
    for images, masks in tqdm(dataloader, desc='Training Mini-SAM'):
        # 2. Move to device
        images, masks = images.to(device), masks.to(device)  # images: (B, 3, H, W), masks: (B, H, W)
        
        # 3. Sample points from masks: points, point_labels = sample_points_from_mask(masks, n_points)
        # Simulate user clicks: 50% foreground + 50% background points
        points, point_labels = sample_points_from_mask(masks, n_points)  # points: (B, n_points, 2), labels: (B, n_points)
        
        # 4. Zero gradients
        optimizer.zero_grad()  # Clear previous gradients
        
        # Mixed Precision Training (AMP)
        if scaler is not None:
            with torch.amp.autocast('cuda'):  # Enable fp16/bf16 for speed
                # 5. Forward pass: mask_logits, iou_pred = model(images, points, point_labels)
                # MiniSAM returns: mask predictions + quality prediction (IoU head)
                mask_logits, iou_pred = model(images, points, point_labels)  # logits: (B, num_classes, H, W), iou: (B, 1)
                
                # 6. Compute losses:
                #    SEGMENTATION LOSSES:
                #    - ce_loss = F.cross_entropy(mask_logits, masks)
                ce_loss = F.cross_entropy(mask_logits, masks, ignore_index=255)  # Pixel-wise classification loss
                
                #    - dice_loss = DiceLoss()(mask_logits, masks)
                dice_loss = dice_fn(mask_logits, masks)  # Region-based overlap loss
                
                #    - pred_masks = mask_logits.argmax(dim=1)
                pred_masks = mask_logits.argmax(dim=1)  # Convert logits to class indices: (B, H, W)
                
                #    IOU HEAD LOSS (quality prediction):
                #    - true_iou = compute_batch_iou(pred_masks, masks)
                true_iou = compute_batch_iou(pred_masks, masks)  # Ground truth IoU: (B,) - one value per image
                
                #    - iou_loss = F.mse_loss(iou_pred.squeeze(), true_iou)
                iou_loss = F.mse_loss(iou_pred.squeeze(), true_iou)  # Supervise IoU prediction head
                
                # 7. Combined loss: loss = ce_loss + dice_loss + 0.1 * iou_loss
                # Multi-objective: segment well + predict quality accurately
                loss = ce_loss + dice_loss + 0.1 * iou_loss  # Weight 0.1: IoU loss is auxiliary
            
            # 8. Backward with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 5. Forward pass: mask_logits, iou_pred = model(images, points, point_labels)
            mask_logits, iou_pred = model(images, points, point_labels)
            
            # 6. Compute losses:
            ce_loss = F.cross_entropy(mask_logits, masks, ignore_index=255)
            dice_loss = dice_fn(mask_logits, masks)
            pred_masks = mask_logits.argmax(dim=1)
            true_iou = compute_batch_iou(pred_masks, masks)
            iou_loss = F.mse_loss(iou_pred.squeeze(), true_iou)
            
            # 7. Combined loss
            loss = ce_loss + dice_loss + 0.1 * iou_loss
            
            # 8. Backward and update
            loss.backward()
            optimizer.step()
        
        # 9. Calculate metrics
        num_classes = mask_logits.shape[1]
        miou, _ = calculate_miou(pred_masks, masks, num_classes=num_classes, ignore_index=255)
        
        # Accumulate
        total_loss += loss.item()
        total_miou += miou
    
    # 10. Return averages
    return total_loss / len(dataloader), total_miou / len(dataloader)


def validate(model, dataloader, device, num_classes=None, is_minisam=False):
    """Validate segmentation model on validation set.
    
    Evaluates model performance using mIoU and pixel accuracy metrics.
    Handles both automatic models (FCN, DeepLabV3+) and interactive models (MiniSAM).
    
    Validation Process:
        1. Set model to eval mode (disables dropout, batchnorm training)
        2. Disable gradients (torch.no_grad() for speed and memory)
        3. For each batch:
           a) Generate predictions:
              - FCN/DeepLabV3+: direct forward pass
              - MiniSAM: sample prompts from GT, then forward
           b) Compute mIoU and pixel accuracy
           c) Accumulate metrics
        4. Return average metrics
    
    Why validate regularly?
        - Monitor overfitting (train mIoU >> val mIoU)
        - Early stopping criterion
        - Model selection (save best model by val mIoU)
        - Track learning progress
    
    Args:
        model (nn.Module): Segmentation model to validate
        dataloader (DataLoader): Validation data loader
        device (torch.device): Device to run validation on
        num_classes (int, optional): Number of classes (inferred from model output if None)
        is_minisam (bool): True if model is MiniSAM (requires prompts)
    
    Returns:
        avg_miou (float): Average mIoU over validation set
        avg_pa (float): Average pixel accuracy over validation set
    
    Implementation Notes:
        - Uses torch.no_grad() to save memory and speed up inference
        - For MiniSAM: simulates user prompts from ground truth
        - Computes metrics batch-wise for efficiency
        - No augmentation during validation (deterministic results)
    
    Typical Usage:
        >>> val_miou, val_pa = validate(model, val_loader, device, num_classes=21)
        >>> print(f"Validation - mIoU: {val_miou:.2%}, PA: {val_pa:.2%}")
        >>> if val_miou > best_miou:
        >>>     torch.save(model.state_dict(), 'best_model.pth')
    
    Performance:
        - RTX 3090, batch_size=32, VOC val set (~1500 images): ~30 seconds
        - No AMP during validation (reproducibility)
    """
    model.eval()  # Disable training mode (dropout off, batchnorm uses running stats)
    total_miou = 0  # Accumulate mIoU across all batches
    total_pa = 0  # Accumulate pixel accuracy
    
    # Task 5.3: Implement validation
    # 1. Use torch.no_grad() context
    # 2. For images, masks in dataloader:
    # 3. Move to device
    with torch.no_grad():  # Disable gradient computation (saves memory, speeds up inference)
        for images, masks in tqdm(dataloader, desc='Validation'):
            images, masks = images.to(device), masks.to(device)  # Move batch to GPU
    
            # 4. If is_minisam:
            #    - Sample points (simulate user prompts for fair evaluation)
            #    - outputs, _ = model(images, points, point_labels)
            #    Else:
            #    - outputs = model(images) (automatic models: no prompts needed)

            if is_minisam:
                # MiniSAM requires prompts even during validation
                points, point_labels = sample_points_from_mask(masks, n_points=5)  # Simulate 5 user clicks
                outputs, _ = model(images, points, point_labels)  # outputs: (B, num_classes, H, W), discard iou_pred
            else:
                # FCN/DeepLabV3+: fully automatic (no prompts)
                outputs = model(images)  # outputs: (B, num_classes, H, W)

            # 5. Get predictions: pred = outputs.argmax(dim=1)
            pred = outputs.argmax(dim=1)  # Convert logits to class indices: (B, H, W)

            # 6. Calculate metrics (ignore border pixels with value 255)
            eval_num_classes = outputs.shape[1] if num_classes is None else num_classes  # Infer from output channels
            miou, _ = calculate_miou(pred, masks, eval_num_classes, ignore_index=255)  # Mean IoU across classes
            pa = calculate_pixel_accuracy(pred, masks, ignore_index=255)  # Percentage of correct pixels

            # 7. Accumulate metrics for averaging
            total_miou = total_miou + miou  # Batch mIoU
            total_pa = total_pa + pa  # Batch pixel accuracy

    # 8. Return averages over entire validation set
    return total_miou / len(dataloader), total_pa / len(dataloader)  # Epoch-level metrics


def visualize_predictions(model, dataloader, device, num_samples=4, is_minisam=False, save_path='predictions.png'):
    """Visualize model predictions alongside input images and ground truth.
    
    Creates a figure with side-by-side comparisons of:
    - Input images (denormalized for viewing)
    - Ground truth segmentation masks
    - Model predictions
    
    This visualization is essential for:
    - Qualitative evaluation (what does the model see?)
    - Debugging (are predictions reasonable?)
    - Progress tracking (improvements over epochs)
    - Identifying failure modes (which objects/scenes are problematic?)
    
    Visualization Details:
        - Uses tab20 colormap for distinct class colors
        - Denormalizes images (reverses ImageNet normalization)
        - Clamps values to [0,1] for proper display
        - Saves high-resolution figure (300 dpi)
    
    Args:
        model (nn.Module): Trained segmentation model
        dataloader (DataLoader): Data loader (typically validation set)
        device (torch.device): Device to run inference on
        num_samples (int): Number of samples to visualize (default: 4)
        is_minisam (bool): True if model requires prompts (MiniSAM)
        save_path (str): Path to save visualization (default: 'predictions.png')
    
    Returns:
        fig (matplotlib.figure.Figure): Figure object (can be further customized)
    
    Implementation:
        - Gets one batch from dataloader
        - Limits to first num_samples images
        - Generates predictions with torch.no_grad()
        - Uses plot_segmentation_results() from lab05_utils for actual plotting
    
    Usage Example:
        >>> # After training
        >>> fig = visualize_predictions(
        ...     model, val_loader, device,
        ...     num_samples=8,
        ...     save_path='fcn8s_predictions.png'
        ... )
        >>> plt.show()  # Display interactively
    
    What to Look For:
        - Good: Clean boundaries, correct object classification
        - Bad: Noisy predictions, missing small objects, wrong classes
        - Common errors: Confusion between similar classes (dog vs cat),
                        missing thin structures, boundary inaccuracies
    """
    model.eval()  # Set to evaluation mode
    
    # Task 5.4: Implement visualization
    # 1. Get one batch: images_batch, masks_batch = next(iter(dataloader))
    # 2. Take first num_samples and move to device
    with torch.no_grad():  # No gradients needed for visualization
        images_batch, masks_batch = next(iter(dataloader))  # Get first batch from dataloader
        images_batch, masks_batch = images_batch.to(device), masks_batch.to(device)  # Move to GPU
    
    # 3. Generate predictions (with or without prompts based on is_minisam)
        if is_minisam:
            # MiniSAM: needs prompts even for visualization
            points, point_labels = sample_points_from_mask(masks_batch, n_points=5)  # Simulate user clicks
            outputs, _ = model(images_batch, points, point_labels)  # Get predictions + discard IoU pred
        else:
            # FCN/DeepLabV3+: automatic segmentation
            outputs = model(images_batch)  # Direct forward pass, no prompts
        predictions = outputs.argmax(dim=1)  # Convert logits to class indices: (B, H, W)

        # Limit number of samples for visualization (avoid cluttered plots)
        batch_size = images_batch.size(0)
        num_to_plot = min(num_samples, batch_size)  # Don't exceed batch size
        
        # Select subset of batch to visualize
        images_to_plot = images_batch[:num_to_plot]  # First N images: (N, 3, H, W)
        masks_to_plot = masks_batch[:num_to_plot]  # Ground truth: (N, H, W)
        predictions_to_plot = predictions[:num_to_plot]  # Model predictions: (N, H, W)

    # 4. Create figure with subplots: (num_samples, 3)
    # 5. For each sample, plot:
    #    - Column 0: Input image (denormalized from ImageNet stats)
    #    - Column 1: Ground truth mask (colored by class)
    #    - Column 2: Predicted mask (colored by class)
    # 6. Save figure
    # plot_segmentation_results from lab05_utils handles:
    #   - Denormalization: reverses mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    #   - Colormap: tab20 for distinct class colors
    #   - Layout: side-by-side comparison (Input | GT | Prediction)
    fig = plot_segmentation_results(
        images=images_to_plot,
        masks=masks_to_plot,
        predictions=predictions_to_plot,
        title=f"Segmentation Results (First {num_to_plot} Samples)"
    )
    plt.show()  # Display interactively (if running in notebook/GUI)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')  # Save high-res image
    print(f"Saved predictions to {save_path}")

    return fig  # Return figure object for further customization


# ================== Dataset Class ==================

class VOCSegmentationDataset(Dataset):
    """PASCAL VOC 2012 Semantic Segmentation Dataset with strong data augmentation.
    
    PASCAL VOC 2012 is a standard benchmark for semantic segmentation with:
    - 21 classes: background + 20 object categories
    - ~1,464 training images
    - ~1,449 validation images
    - Variable image sizes (resized to fixed size for training)
    
    Classes:
        Background, aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair,
        cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa,
        train, tvmonitor
    
    Data Augmentation Pipeline (training only):
        1. Random Horizontal Flip (50% probability)
           - Mirrors image left-right
           - Preserves semantic content
        
        2. Random Scale (0.5× to 2.0×)
           - Handles objects at different scales
           - Critical for scale invariance
        
        3. Random Crop to target size
           - Extracts diverse viewpoints
           - Pads if image is too small
        
        4. Color Jitter (50% each: brightness, contrast, saturation)
           - Range: ±20% per component
           - Robustness to lighting conditions
        
        5. Random Rotation (-10° to +10°, 50% probability)
           - Small rotations for robustness
           - Larger rotations could distort objects
    
    Normalization:
        - Mean: [0.485, 0.456, 0.406] (ImageNet statistics)
        - Std: [0.229, 0.224, 0.225]
        - Applied to images, not masks
    
    Special Values:
        - ignore_index = 255: Pixels to ignore (boundaries, void regions)
    
    Args:
        root_dir (str): Path to VOC dataset root
                       (e.g., 'voc/VOC2012_train_val/VOC2012_train_val')
        split (str): 'train' or 'val'
        image_size (int): Target image size (e.g., 512)
        transform (deprecated): Use use_augmentation instead
        use_augmentation (bool): Enable data augmentation (only for training)
    
    Returns:
        image (Tensor): Normalized image, shape (3, H, W)
        mask (Tensor): Class indices, shape (H, W), dtype=long
    
    Fallback Behavior:
        - If VOC dataset not found: generates synthetic data for testing
        - Prints warning and uses random tensors
    
    Usage Example:
        >>> train_dataset = VOCSegmentationDataset(
        ...     root_dir='voc/VOC2012_train_val/VOC2012_train_val',
        ...     split='train',
        ...     image_size=512,
        ...     use_augmentation=True
        ... )
        >>> val_dataset = VOCSegmentationDataset(
        ...     root_dir='voc/VOC2012_train_val/VOC2012_train_val',
        ...     split='val',
        ...     image_size=512,
        ...     use_augmentation=False  # IMPORTANT: No augmentation for validation!
        ... )
    
    Performance Impact:
        - Without augmentation: ~50-55% mIoU (poor generalization)
        - With augmentation: ~65-75% mIoU (+15-20% absolute improvement!)
        - Data augmentation is CRITICAL for good segmentation results
    """
    
    def __init__(self, root_dir, split='train', image_size=256, transform=None, use_augmentation=True):
        """
        Args:
            root_dir: Path to VOC dataset root (e.g., path/to/VOC2012_train_val/VOC2012_train_val)
            split: 'train' or 'val'
            image_size: Target image size
            transform: Optional transforms (deprecated, use use_augmentation)
            use_augmentation: Enable data augmentation (only for training)
        """
        import os
        
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.use_augmentation = use_augmentation and (split == 'train')
        
        # Convert to absolute path
        if not os.path.isabs(root_dir):
            # Get script directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.abspath(os.path.join(script_dir, root_dir))
        
        # Define paths
        self.images_dir = os.path.join(root_dir, 'JPEGImages')
        self.masks_dir = os.path.join(root_dir, 'SegmentationClass')
        split_file = os.path.join(root_dir, 'ImageSets', 'Segmentation', f'{split}.txt')
        
        # Check if dataset exists
        if not os.path.exists(self.images_dir) or not os.path.exists(self.masks_dir):
            print(f"Warning: Could not find VOC dataset at {root_dir}")
            print(f"  Looking for: {self.images_dir}")
            print(f"  Looking for: {self.masks_dir}")
            print(f"  Using synthetic data instead.")
            self.use_synthetic = True
            self.length = 100 if split == 'train' else 20
            return
        
        # Load image IDs from split file
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                self.image_ids = [line.strip() for line in f.readlines()]
            print(f"✓ Loaded {len(self.image_ids)} {split} images from VOC dataset")
        else:
            print(f"Warning: Split file not found: {split_file}")
            print(f"  Using synthetic data instead.")
            self.use_synthetic = True
            self.length = 100 if split == 'train' else 20
    
    def __len__(self):
        if hasattr(self, 'use_synthetic'):
            return self.length
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        if hasattr(self, 'use_synthetic'):
            # Generate synthetic data for testing
            image = torch.randn(3, self.image_size, self.image_size)
            mask = torch.randint(0, 21, (self.image_size, self.image_size)).long()
            
            # Normalize image
            image = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )(image)
            
            return image, mask
        
        # Load real VOC data
        import os
        
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.images_dir, f'{image_id}.jpg')
        mask_path = os.path.join(self.masks_dir, f'{image_id}.png')
        
        # Load image and mask
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path)
        
        # Apply data augmentation if enabled
        if self.use_augmentation:
            image, mask = self._apply_augmentation(image, mask)
        else:
            # Just resize for validation
            image = TF.resize(image, [self.image_size, self.image_size], interpolation=Image.BILINEAR)
            mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=Image.NEAREST)
        
        # Convert to tensor
        image = TF.to_tensor(image)
        mask = torch.from_numpy(np.array(mask)).long()
        
        # Normalize image
        image = TF.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        return image, mask
    
    def _apply_augmentation(self, image, mask):
        """Apply data augmentation transforms to image and mask"""
        
        # ==================== DATA AUGMENTATION PIPELINE ====================
        # Apply strong augmentation to training images for better generalization
        # All transformations applied to BOTH image and mask (synchronized)
        
        # 1. Random horizontal flip (50% probability)
        # - Mirrors image left-right (person facing left → person facing right)
        # - Preserves semantic content (objects still recognizable)
        if random.random() > 0.5:
            image = TF.hflip(image)  # Flip image
            mask = TF.hflip(mask)  # Flip mask (must match image flip)
        
        # 2. Random scale (0.5× to 2.0×)
        # - Handles objects at different scales (near vs far)
        # - Critical for scale invariance (recognize cats at any size)
        # - BILINEAR for image (smooth), NEAREST for mask (preserve class indices)
        scale = random.uniform(0.5, 2.0)  # Random zoom factor
        w, h = image.size  # Original dimensions
        new_w, new_h = int(w * scale), int(h * scale)  # Scaled dimensions
        image = TF.resize(image, [new_h, new_w], interpolation=Image.BILINEAR)  # Smooth resize
        mask = TF.resize(mask, [new_h, new_w], interpolation=Image.NEAREST)  # No interpolation (class indices)
        
        # 3. Random crop to target size (with padding if needed)
        # - Extracts diverse viewpoints (top-left corner vs center vs bottom-right)
        # - Pads with zeros (image) or 255/ignore (mask) if too small
        w, h = image.size  # Current dimensions after scaling
        if w < self.image_size or h < self.image_size:
            # Pad if image is smaller than target (happens with scale < 1.0)
            pad_h = max(self.image_size - h, 0)  # How much to pad vertically
            pad_w = max(self.image_size - w, 0)  # How much to pad horizontally
            image = TF.pad(image, [0, 0, pad_w, pad_h], fill=0)  # Pad image with black (0)
            mask = TF.pad(mask, [0, 0, pad_w, pad_h], fill=255)  # Pad mask with ignore (255)
            w, h = image.size  # Update dimensions after padding
        
        # Random crop (extract target_size × target_size patch from random location)
        i = random.randint(0, h - self.image_size)  # Random top coordinate
        j = random.randint(0, w - self.image_size)  # Random left coordinate
        image = TF.crop(image, i, j, self.image_size, self.image_size)  # Crop image
        mask = TF.crop(mask, i, j, self.image_size, self.image_size)  # Crop mask (same location)
        
        # 4. Color jitter (brightness, contrast, saturation)
        # - Robustness to lighting conditions (sunny vs cloudy, indoor vs outdoor)
        # - Range: ±20% per component (subtle changes, not unrealistic)
        # - Applied independently with 50% probability each
        if random.random() > 0.5:
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))  # Brightness: 80%-120%
        if random.random() > 0.5:
            image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))  # Contrast: 80%-120%
        if random.random() > 0.5:
            image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))  # Saturation: 80%-120%
        
        # 5. Random rotation (-10° to +10°)
        # - Small rotations for robustness (objects not always perfectly upright)
        # - Larger rotations (e.g., 90°) could distort objects unrealistically
        # - BILINEAR for image, NEAREST for mask (preserve class indices)
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)  # Random angle in degrees
            image = TF.rotate(image, angle, interpolation=Image.BILINEAR)  # Smooth rotation
            mask = TF.rotate(mask, angle, interpolation=Image.NEAREST)  # No interpolation
        
        return image, mask  # Return augmented image and mask


# ================== Main Training Script ==================

def main(model_name=None):
    """Main training pipeline for semantic segmentation models.
    
    This function orchestrates the complete training workflow:
    1. Setup: Load configuration, create datasets and dataloaders
    2. Model: Initialize architecture, move to GPU
    3. Optimizer: Setup AdamW + LR scheduler (cosine annealing with warm restarts)
    4. Training loop: Train for N epochs with validation after each epoch
    5. Tracking: Monitor losses, mIoU, pixel accuracy, save best model
    6. Visualization: Plot training curves and example predictions
    
    Configuration (optimized for RTX 3090, 24GB VRAM):
        - batch_size: 32 (4× larger than baseline)
        - image_size: 512 (higher resolution for better accuracy)
        - learning_rate: 3e-4 (scaled with batch size)
        - epochs: 100 (sufficient for convergence with early stopping)
        - use_amp: True (Automatic Mixed Precision for 2× speedup)
        - use_augmentation: True (critical for good generalization)
    
    Training Techniques:
        1. Transfer Learning: Pretrained ResNet50 backbone (ImageNet)
        2. Learning Rate Warmup: 3 epochs linear ramp (0 → lr)
        3. Cosine Annealing: Smooth LR decay with periodic restarts
        4. Early Stopping: Patience=30 epochs (stops if no improvement)
        5. AMP: Mixed precision training (fp16/fp32) for speed
        6. Combined Loss: CE + Dice + Focal for robust optimization
    
    Learning Rate Schedule:
        Epochs 0-3:   Linear warmup (0 → 3e-4)
        Epochs 3-10:  Cosine decay (3e-4 → 1e-7)
        Epoch 10:     Restart to 3e-4
        Epochs 10-30: Cosine decay
        Epoch 30:     Restart to 3e-4
        ... (continues with T_mult=2, so periods double each time)
    
    Checkpoint Management:
        - Saves best model based on validation mIoU
        - Checkpoint contains: epoch, model_state_dict, best_miou
        - Format compatible with weights_only=True (PyTorch 2.6+ secure loading)
        - File: best_{model_name}_model.pth
    
    Args:
        model_name (str, optional): Model to train
                                   Options: 'fcn32s', 'fcn16s', 'fcn8s',
                                           'deeplabv3plus', 'minisam'
                                   Default: 'fcn8s'
    
    Outputs:
        - Best model checkpoint: best_{model_name}_model.pth
        - Training curves: training_curves_{model_name}.png
        - Sample predictions: predictions_{model_name}.png
    
    Typical Training Time (RTX 3090):
        - FCN-32s/16s/8s: ~60 mins for 60 epochs
        - DeepLabV3+: ~90 mins for 60 epochs (larger model)
        - MiniSAM: ~75 mins for 60 epochs (prompt sampling overhead)
    
    Expected Results (after 60 epochs with augmentation):
        - FCN-32s: 55-60% mIoU
        - FCN-16s: 58-62% mIoU
        - FCN-8s: 60-65% mIoU
        - DeepLabV3+: 70-75% mIoU
        - MiniSAM: 45-50% mIoU (depends on prompt quality)
    
    Usage:
        >>> # Train a specific model
        >>> main(model_name='fcn8s')
        
        >>> # Train all models sequentially
        >>> for model in ['fcn32s', 'fcn16s', 'fcn8s', 'deeplabv3plus', 'minisam']:
        >>>     main(model_name=model)
    
    Troubleshooting:
        - CUDA OOM: Reduce batch_size or image_size
        - Slow training: Verify use_amp=True and num_workers>0
        - Poor results: Ensure use_augmentation=True, train longer
        - NaN loss: Check initialization (don't reinitialize pretrained layers)
    """
    
    # CONFIGURATION DICTIONARY - all training hyperparameters
    config = {
        'model': model_name or 'fcn8s',  # Options: 'fcn32s', 'fcn16s', 'fcn8s', 'deeplabv3plus', 'minisam'
        'n_classes': 21,  # PASCAL VOC 2012: background + 20 object classes
        'batch_size': 32,  # RTX 3090: 8→32 (4× increase with 24GB VRAM, adjust down if OOM)
        'learning_rate': 3e-4,  # Increased LR for larger batch size (linear scaling rule)
        'epochs': 100,  # Increased from 30 for better convergence with augmentation
        'device': device,  # Detected earlier: cuda if available, else cpu
        'image_size': 512,  # RTX 3090: 256→512 (higher resolution for better accuracy)
        'data_dir': '../../voc/VOC2012_train_val/VOC2012_train_val',  # VOC dataset root
        'num_workers': min(8, os.cpu_count()),  # Multi-threaded data loading (adjust based on CPU cores)
        'use_amp': True,  # Automatic Mixed Precision for 2× faster training (fp16/bf16)
        'weight_decay': 1e-4,  # L2 regularization (prevents overfitting)
        'warmup_epochs': 3,  # Learning rate warm-up period (prevents early instability)
        'use_augmentation': True,  # Enable data augmentation for better generalization
    }
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get current script directory for saving outputs
    
    print(f"Training {config['model']} for {config['epochs']} epochs")
    
    # ==================== TASK 6.1: SETUP DATA LOADERS ====================
    print("\n[1/5] Setting up data loaders...")
    
    # Create datasets (handles loading images and masks)
    train_dataset = VOCSegmentationDataset(
        root_dir=config['data_dir'],
        split='train',  # Training split (~1,464 images)
        image_size=config['image_size'],
        use_augmentation=config.get('use_augmentation', True)  # Enable augmentation for training
    )
    
    val_dataset = VOCSegmentationDataset(
        root_dir=config['data_dir'],
        split='val',  # Validation split (~1,449 images)
        image_size=config['image_size'],
        use_augmentation=False  # NO augmentation for validation (deterministic evaluation)
    )
    
    # Create data loaders (handles batching, shuffling, parallel loading)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],  # 32 images per batch
        shuffle=True,  # Randomize order each epoch (prevents overfitting to sequence)
        num_workers=config.get('num_workers', 4),  # Parallel data loading (4-8 workers recommended)
        pin_memory=True if torch.cuda.is_available() else False,  # Faster GPU transfer
        worker_init_fn=worker_init_fn,   # ← IMPORTANTÍSIMO
        generator=g 
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],  # Same batch size as training
        shuffle=False,  # NO shuffling for validation (reproducible results)
        num_workers=config.get('num_workers', 4),
        pin_memory=True if torch.cuda.is_available() else False,
        worker_init_fn=worker_init_fn,
        generator=g
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # ==================== TASK 6.2: CREATE MODEL ====================
    print("\n[2/5] Creating model...")
    
    # Initialize model based on config (factory pattern)
    if config['model'] == 'fcn32s':
        model = FCN32s(n_classes=config['n_classes'])  # Baseline FCN (32× downsampling)
    elif config['model'] == 'fcn16s':
        model = FCN16s(n_classes=config['n_classes'])  # FCN with one skip connection
    elif config['model'] == 'fcn8s':
        model = FCN8s(n_classes=config['n_classes'])  # FCN with two skip connections (best FCN)
    elif config['model'] == 'deeplabv3plus':
        model = DeepLabV3Plus(n_classes=config['n_classes'])  # State-of-the-art with ASPP
    elif config['model'] == 'minisam':
        model = MiniSAM(n_classes=config['n_classes'])  # Interactive model with prompts
    elif config['model'] == 'fcn8s_kccs':
        model = FCN8sKCCS(n_classes=config['n_classes'])

    else:
        raise ValueError(f"Unknown model: {config['model']}")
    
    # Move model to device (GPU if available, CPU otherwise)
    model = model.to(config['device'])  # Transfer all parameters and buffers to GPU
    
    # Count parameters (useful for comparing model complexity)
    num_params = sum(p.numel() for p in model.parameters())  # Total trainable params
    print(f"Model: {config['model']}, Parameters: {num_params:,}")
    
    # ==================== TASK 6.3: SETUP OPTIMIZER AND LOSS ====================
    print("\n[3/5] Setting up optimizer and loss...")
    
    # Create optimizer with weight decay (L2 regularization)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],  # Initial learning rate (will be warmed up)
        weight_decay=config.get('weight_decay', 1e-4),  # L2 penalty on weights
        betas=(0.9, 0.999)  # Default Adam momentum parameters
    )
    
    # Create learning rate scheduler with cosine annealing for smoother convergence
    # CosineAnnealingWarmRestarts: smooth decay with periodic restarts (prevents local minima)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # Restart every 10 epochs (first cycle)
        T_mult=2,  # Double the period after each restart (10, 20, 40, ...)
        eta_min=1e-7  # Minimum learning rate (prevents LR from going to zero)
    )
    
    # Create loss function (multi-objective: CE + Dice + Focal)
    criterion = CombinedLoss()  # 30% CE + 50% Dice + 20% Focal
    
    # Initialize GradScaler for Automatic Mixed Precision (AMP)
    # AMP: uses fp16/bf16 for speed, fp32 for stability (2× faster, 50% less VRAM)
    scaler = torch.amp.GradScaler('cuda') if config.get('use_amp', False) and torch.cuda.is_available() else None
    if scaler:
        print("✓ Using Automatic Mixed Precision (AMP) for faster training")
    
    # ==================== TASK 6.4: TRAINING LOOP ====================
    print("\n[4/5] Starting training...")
    
    # Initialize tracking variables (for monitoring and checkpointing)
    best_miou = 0  # Best validation mIoU achieved (for saving checkpoints)
    train_losses = []  # Training loss per epoch (for plotting curves)
    val_mious = []  # Validation mIoU per epoch
    val_pas = []  # Validation pixel accuracy per epoch
    
    # Early stopping configuration (prevents overfitting)
    patience = 30  # Stop if no improvement for 30 consecutive epochs
    patience_counter = 0  # Count epochs without improvement
    
    # Determine if model is MiniSAM (needs different training function)
    is_minisam = (config['model'] == 'minisam')
    
    # Training loop
    # Training loop - iterate over all epochs
    for epoch in range(config['epochs']):
        print(f"\nEpoch [{epoch+1}/{config['epochs']}]")
        
        # LEARNING RATE WARM-UP (first 3 epochs)
        # Prevents early training instability by gradually increasing LR from 0
        if epoch < config.get('warmup_epochs', 0):
            warmup_factor = (epoch + 1) / config['warmup_epochs']  # Linear ramp: 0.33, 0.67, 1.0 for 3 epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = config['learning_rate'] * warmup_factor  # Gradually increase to target LR
            print(f"Warm-up: LR = {optimizer.param_groups[0]['lr']:.6f}")
        
        # TRAINING PHASE
        # Choose appropriate training function based on model type
        if is_minisam:
            # MiniSAM requires prompt sampling (50% fg + 50% bg points)
            train_loss, train_miou = train_epoch_minisam(
                model, train_loader, optimizer, criterion, config['device'],
                scaler=scaler  # AMP gradient scaler (None if AMP disabled)
            )
        else:
            # FCN/DeepLabV3+ are fully automatic (no prompts needed)
            train_loss, train_miou = train_epoch_fcn(
                model, train_loader, optimizer, criterion, config['device'],
                scaler=scaler
            )
        
        # VALIDATION PHASE
        # Evaluate on validation set without gradient computation (faster, less memory)
        val_miou, val_pa = validate(
            model, val_loader, config['device'], 
            num_classes=config['n_classes'],
            is_minisam=is_minisam  # Determines if prompts are needed during validation
        )
        
        # LR SCHEDULER STEP (after warm-up period)
        # CosineAnnealingWarmRestarts: smooth decay with periodic restarts
        if epoch >= config.get('warmup_epochs', 0):
            scheduler.step()  # Update learning rate for next epoch
        
        # PRINT EPOCH SUMMARY
        current_lr = optimizer.param_groups[0]['lr']  # Get current LR (changes each epoch)
        print(f"Train Loss: {train_loss:.4f}, Train mIoU: {train_miou:.4f}")
        print(f"Val mIoU: {val_miou:.4f}, Val PA: {val_pa:.4f}, LR: {current_lr:.6f}")
        
        # TRACKING (for plotting training curves later)
        train_losses.append(train_loss)  # Loss trajectory
        val_mious.append(val_miou)  # Validation mIoU trajectory
        val_pas.append(val_pa)  # Pixel accuracy trajectory
        
        # CHECKPOINT MANAGEMENT (save best model based on validation mIoU)
        if val_miou > best_miou:
            best_miou = val_miou  # Update best score
            patience_counter = 0  # Reset early stopping counter (model is improving)
            checkpoint_path = os.path.join(script_dir, '..', '..', f'best_{config["model"]}_model.pth')
            # Save only essential data for secure loading (weights_only=True compatible in PyTorch 2.6+)
            torch.save({
                'epoch': epoch,  # Which epoch achieved this score
                'model_state_dict': model.state_dict(),  # Model weights
                'best_miou': best_miou,  # Best validation mIoU
            }, checkpoint_path)
            print(f"✓ Saved new best model with mIoU: {best_miou:.4f}")
            print(f"   Path: {checkpoint_path}")
        else:
            # EARLY STOPPING (prevent overfitting by stopping if no improvement)
            patience_counter += 1  # Increment counter (no improvement this epoch)
            print(f"No improvement for {patience_counter}/{patience} epochs")
            if patience_counter >= patience:
                print(f"\n⚠ Early stopping triggered after {epoch+1} epochs (no improvement for {patience} epochs)")
                break  # Exit training loop early
    
    print(f"\nTraining completed! Best validation mIoU: {best_miou:.4f}")
    
    # ==================== TASK 6.5: PLOT TRAINING CURVES ====================
    print("\n[5/5] Plotting training curves...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))  # Create 3 subplots side-by-side
    
    # SUBPLOT 1: Training Loss
    # Shows how well the model is fitting the training data
    # Should decrease over time (if increasing → diverging/unstable training)
    axes[0].plot(train_losses, label='Training Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Over Epochs')
    axes[0].grid(True, alpha=0.3)  # Add grid for readability
    axes[0].legend()
    
    # SUBPLOT 2: Validation mIoU
    # Primary metric for segmentation quality
    # Should increase over time and plateau (if decreasing → overfitting)
    axes[1].plot(val_mious, label='Validation mIoU', color='green', linewidth=2)
    axes[1].axhline(y=best_miou, color='r', linestyle='--', label=f'Best mIoU: {best_miou:.4f}')  # Mark best score
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('mIoU')
    axes[1].set_title('Validation mIoU Over Epochs')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # SUBPLOT 3: Validation Pixel Accuracy
    # Secondary metric (less informative than mIoU for segmentation)
    # Pixel accuracy can be misleading with class imbalance (e.g., large background)
    axes[2].plot(val_pas, label='Validation Pixel Accuracy', color='orange', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Pixel Accuracy')
    axes[2].set_title('Validation Pixel Accuracy Over Epochs')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()  # Adjust spacing between subplots
    plot_path = os.path.join(script_dir, '..', '..', f'training_curves_{config["model"]}.png')
    plt.savefig(plot_path, dpi=150)  # Save high-res version
    print(f"Saved training curves to {plot_path}")
    plt.show()  # Display interactively (if running in GUI/notebook)
    
    # VISUALIZE PREDICTIONS (qualitative evaluation)
    # Shows actual predictions vs ground truth on sample images
    # Useful for debugging and understanding failure modes
    print("\nGenerating prediction visualizations...")
    predictions_path = os.path.join(script_dir, '..', '..', f'predictions_{config["model"]}.png')
    visualize_predictions(
        model, val_loader, config['device'],
        num_samples=4,  # Show 4 examples (adjustable)
        is_minisam=is_minisam,  # Handle prompts if needed
        save_path=predictions_path
    )
    
    print("\n" + "="*60)
    print("Training pipeline completed successfully!")
    print("="*60)


def compare_models():
    """Compare all implemented models on validation set with comprehensive metrics.
    
    This function provides a complete comparative analysis of all 5 architectures:
    FCN-32s, FCN-16s, FCN-8s, DeepLabV3+, and MiniSAM.
    
    Comparison Metrics:
        1. Parameters (M): Model size in millions of parameters
        2. mIoU (%): Mean Intersection over Union (primary metric)
        3. Pixel Accuracy (%): Percentage of correctly classified pixels
        4. Inference Time (ms): Average time per image on RTX 3090
        5. Model Size (MB): Checkpoint file size on disk
    
    Analysis Outputs:
        1. Quantitative Table: Prints formatted table with all metrics
        2. Qualitative Visualization: Side-by-side predictions on sample images
           - Saved as model_comparison.png
           - Shows: Input | GT | FCN32s | FCN16s | FCN8s | DeepLabV3+ | MiniSAM
    
    Use Cases:
        - Final evaluation after training all models
        - Model selection for deployment (speed vs accuracy trade-off)
        - Understanding architectural differences qualitatively
        - Preparing results for reports/papers
    
    Expected Comparison (approximate, depends on training):
        Model       | Params | mIoU  | PA   | Time | Size
        ------------|--------|-------|------|------|-------
        FCN-32s     | 25.36M | 57%   | 75%  | 1.57 | 96.7
        FCN-16s     | 24.03M | 60%   | 76%  | 1.25 | 91.7
        FCN-8s      | 23.71M | 63%   | 77%  | 1.26 | 90.5
        DeepLabV3+  | 40.35M | 72%   | 82%  | 2.14 | 153.9
        MiniSAM     | 2.22M  | 48%   | 72%  | 1.86 | 8.5
    
    Key Insights:
        - DeepLabV3+: Best accuracy, but largest and slowest
        - FCN-8s: Best FCN variant, good balance
        - MiniSAM: Smallest, fast, but needs prompts
        - Skip connections: FCN-8s > FCN-16s > FCN-32s (clear progression)
    
    Interpretation Guide:
        - mIoU vs Parameters: DeepLabV3+ has 1.7× params but 1.14× mIoU of FCN-8s
        - Speed vs Accuracy: FCN-8s is 1.7× faster than DeepLabV3+ with only 9% lower mIoU
        - Size: MiniSAM is 10× smaller, suitable for mobile/edge deployment
    
    Returns:
        results (dict): Dictionary with all metrics for further analysis
    
    Usage:
        >>> # After training all models
        >>> results = compare_models()
        >>> # Outputs:
        >>> # - Prints comparison table
        >>> # - Saves model_comparison.png
        >>> # - Returns results dict
    
    Notes:
        - Loads trained checkpoints (best_{model}_model.pth)
        - If checkpoint not found: uses random weights (prints warning)
        - Evaluates on same validation set for fair comparison
        - Uses deterministic settings (no augmentation, fixed seed)
    """
    # Task 7.1: Compare FCN-32s, FCN-16s, FCN-8s, DeepLabV3+, and Mini-SAM
    
    print("="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    # 1. Create results dictionary
    results = {
        'Model': [],
        'Parameters (M)': [],
        'mIoU (%)': [],
        'Pixel Acc (%)': [],
        'Inference Time (ms)': [],
        'Model Size (MB)': []
    }
    
    models_to_compare = ['fcn32s', 'fcn16s', 'fcn8s', 'fcn8s_kccs', 'deeplabv3plus', 'minisam']
    n_classes = 21
    image_size = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create validation dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    voc_path = os.path.join(script_dir, '..', '..', 'voc', 'VOC2012_train_val', 'VOC2012_train_val')
    val_dataset = VOCSegmentationDataset(
        root_dir=voc_path,
        split='val',
        image_size=image_size
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )
    
    # 2. For each model: evaluate
    for model_name in models_to_compare:
        print(f"\n{'='*80}")
        print(f"Evaluating {model_name.upper()}...")
        print(f"{'='*80}")
        
        try:
            # Initialize model
            if model_name == 'fcn32s':
                model = FCN32s(n_classes=n_classes)
            elif model_name == 'fcn16s':
                model = FCN16s(n_classes=n_classes)
            elif model_name == 'fcn8s':
                model = FCN8s(n_classes=n_classes)
            elif model_name == 'deeplabv3plus':
                model = DeepLabV3Plus(n_classes=n_classes)
            elif model_name == 'minisam':
                model = MiniSAM(n_classes=n_classes)
            elif model_name == 'fcn8s_kccs':
                model = FCN8sKCCS(n_classes=n_classes)

            
            model = model.to(device)
            
            # Try to load trained checkpoint
            script_dir = os.path.dirname(os.path.abspath(__file__))
            checkpoint_path = os.path.join(script_dir, '..', '..', f'best_{model_name}_model.pth')
            
            if os.path.exists(checkpoint_path):
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"✓ Loaded checkpoint from {checkpoint_path}")
                except Exception as e:
                    print(f"⚠ Error loading checkpoint: {e}")
                    print(f"⚠ Using random weights for {model_name}")
            else:
                print(f"⚠ Checkpoint not found at: {checkpoint_path}")
                print(f"⚠ Using random weights for {model_name}")
            
            # Count parameters
            num_params = sum(p.numel() for p in model.parameters())
            params_m = num_params / 1e6
            
            # Calculate model size
            model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
            
            # Evaluate on test set
            is_minisam = (model_name == 'minisam')
            
            # Measure inference time
            model.eval()
            inference_times = []
            
            with torch.no_grad():
                # Warm up
                dummy_input = torch.randn(1, 3, image_size, image_size).to(device)
                if is_minisam:
                    dummy_points = torch.rand(1, 5, 2).to(device)
                    dummy_labels = torch.randint(0, 2, (1, 5)).to(device)
                    _ = model(dummy_input, dummy_points, dummy_labels)
                else:
                    _ = model(dummy_input)
                
                # Measure on first few batches
                for i, (images, masks) in enumerate(val_loader):
                    if i >= 10:  # Test on 10 batches
                        break
                    
                    images = images.to(device)
                    
                    start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                    end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                    
                    if torch.cuda.is_available():
                        start_time.record()
                    else:
                        import time
                        start = time.time()
                    
                    if is_minisam:
                        points, point_labels = sample_points_from_mask(masks, n_points=5)
                        points, point_labels = points.to(device), point_labels.to(device)
                        _ = model(images, points, point_labels)
                    else:
                        _ = model(images)
                    
                    if torch.cuda.is_available():
                        end_time.record()
                        torch.cuda.synchronize()
                        inference_times.append(start_time.elapsed_time(end_time) / images.size(0))
                    else:
                        end = time.time()
                        inference_times.append((end - start) * 1000 / images.size(0))
            
            avg_inference_time = np.mean(inference_times)
            
            # Validate
            val_miou, val_pa = validate(model, val_loader, device, n_classes, is_minisam)
            
            # Record results
            results['Model'].append(model_name.upper())
            results['Parameters (M)'].append(f"{params_m:.2f}")
            results['mIoU (%)'].append(f"{val_miou*100:.2f}")
            results['Pixel Acc (%)'].append(f"{val_pa*100:.2f}")
            results['Inference Time (ms)'].append(f"{avg_inference_time:.2f}")
            results['Model Size (MB)'].append(f"{model_size_mb:.2f}")
            
            print(f"✓ {model_name.upper()}: mIoU={val_miou*100:.2f}%, PA={val_pa*100:.2f}%, "
                  f"Params={params_m:.2f}M, Time={avg_inference_time:.2f}ms")
            
        except Exception as e:
            print(f"✗ Error evaluating {model_name}: {e}")
            results['Model'].append(model_name.upper())
            results['Parameters (M)'].append("N/A")
            results['mIoU (%)'].append("N/A")
            results['Pixel Acc (%)'].append("N/A")
            results['Inference Time (ms)'].append("N/A")
            results['Model Size (MB)'].append("N/A")
    
    # 3. Create comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    
    # Print header
    header = f"{'Model':<15} {'Params(M)':<12} {'mIoU(%)':<10} {'PA(%)':<10} {'Time(ms)':<12} {'Size(MB)':<10}"
    print(header)
    print("-"*80)
    
    # Print rows
    for i in range(len(results['Model'])):
        row = f"{results['Model'][i]:<15} "
        row += f"{results['Parameters (M)'][i]:<12} "
        row += f"{results['mIoU (%)'][i]:<10} "
        row += f"{results['Pixel Acc (%)'][i]:<10} "
        row += f"{results['Inference Time (ms)'][i]:<12} "
        row += f"{results['Model Size (MB)'][i]:<10}"
        print(row)
    
    print("="*80)
    
    # 4. Generate side-by-side qualitative comparisons
    print("\nGenerating qualitative comparisons...")
    
    # Get one batch for visualization
    images_batch, masks_batch = next(iter(val_loader))
    num_samples = min(2, images_batch.size(0))
    
    fig, axes = plt.subplots(num_samples, len(models_to_compare) + 2, 
                            figsize=(4*(len(models_to_compare)+2), 4*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for sample_idx in range(num_samples):
        # Show input image
        img = images_batch[sample_idx].cpu()
        img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        img = torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()
        
        axes[sample_idx, 0].imshow(img)
        axes[sample_idx, 0].set_title('Input')
        axes[sample_idx, 0].axis('off')
        
        # Show ground truth
        gt_mask = masks_batch[sample_idx].cpu().numpy()
        axes[sample_idx, 1].imshow(gt_mask, cmap='tab20', vmin=0, vmax=20)
        axes[sample_idx, 1].set_title('Ground Truth')
        axes[sample_idx, 1].axis('off')
        
        # Show predictions from each model
        for model_idx, model_name in enumerate(models_to_compare):
            try:
                # Load model
                if model_name == 'fcn32s':
                    model = FCN32s(n_classes=n_classes)
                elif model_name == 'fcn16s':
                    model = FCN16s(n_classes=n_classes)
                elif model_name == 'fcn8s':
                    model = FCN8s(n_classes=n_classes)
                elif model_name == 'deeplabv3plus':
                    model = DeepLabV3Plus(n_classes=n_classes)
                elif model_name == 'minisam':
                    model = MiniSAM(n_classes=n_classes)
                
                model = model.to(device)
                
                # Try to load checkpoint
                try:
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    checkpoint_path = os.path.join(script_dir, '..', '..', f'best_{model_name}_model.pth')
                    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
                    model.load_state_dict(checkpoint['model_state_dict'])
                except Exception:
                    pass  # Silently continue with untrained model for visualization
                
                model.eval()
                
                # Generate prediction
                with torch.no_grad():
                    img_input = images_batch[sample_idx:sample_idx+1].to(device)
                    mask_input = masks_batch[sample_idx:sample_idx+1]
                    
                    if model_name == 'minisam':
                        points, point_labels = sample_points_from_mask(mask_input, n_points=5)
                        points, point_labels = points.to(device), point_labels.to(device)
                        output, _ = model(img_input, points, point_labels)
                    else:
                        output = model(img_input)
                    
                    pred = output.argmax(dim=1)[0].cpu().numpy()
                
                axes[sample_idx, model_idx + 2].imshow(pred, cmap='tab20', vmin=0, vmax=20)
                axes[sample_idx, model_idx + 2].set_title(model_name.upper())
                axes[sample_idx, model_idx + 2].axis('off')
                
            except Exception as e:
                axes[sample_idx, model_idx + 2].text(0.5, 0.5, f'Error:\n{model_name}',
                                                      ha='center', va='center')
                axes[sample_idx, model_idx + 2].axis('off')
    
    plt.tight_layout()
    comparison_path = os.path.join(script_dir, '..', '..', 'model_comparison.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison to '{comparison_path}'")
    plt.show()
    
    print("\n" + "="*80)
    print("Model comparison completed!")
    print("="*80)
    
    return results


def interactive_minisam_demo():
    """Interactive Mini-SAM demonstration with iterative prompt refinement.
    
    This demo simulates the interactive segmentation workflow of SAM-style models:
    1. User provides initial prompts (points/boxes)
    2. Model generates initial segmentation
    3. Model predicts mask quality (IoU score)
    4. User adds correction prompts
    5. Model refines segmentation
    6. Compare before/after results
    
    Demo Workflow:
        [1/7] Load trained MiniSAM model (or use random weights)
        [2/7] Load random test image from VOC validation set
        [3/7] Simulate initial user prompts:
              - 2 foreground points (on object)
              - 1 background point (on background)
        [4/7] Run initial segmentation
        [5/7] Visualize: Input | Prompts | Prediction (with IoU score)
        [6/7] Add correction prompts:
              - 1 additional foreground point
              - 1 additional background point
        [7/7] Run refined segmentation and compare
    
    Simulated Prompts:
        Initial (3 points):
        - (0.3, 0.3): Foreground (green star)
        - (0.5, 0.5): Foreground (green star)
        - (0.1, 0.1): Background (red X)
        
        Refinement (+2 points):
        - (0.7, 0.7): Foreground
        - (0.2, 0.8): Background
    
    Visualization Output:
        2 rows, 3 columns:
        Row 1 (Initial):
        - Col 1: Input image
        - Col 2: Image with initial prompts overlaid
        - Col 3: Initial prediction (with predicted IoU)
        
        Row 2 (Refined):
        - Col 1: Ground truth mask
        - Col 2: Image with all prompts (initial + corrections)
        - Col 3: Refined prediction (with updated IoU)
    
    Expected Behavior:
        - More prompts → better segmentation (higher IoU)
        - IoU prediction should correlate with actual mask quality
        - Corrections fix errors from initial segmentation
    
    Real Interactive Implementation (not in this demo):
        In a production system, you would:
        1. Use matplotlib event handlers: fig.canvas.mpl_connect('button_press_event', ...)
        2. Capture user clicks: event.xdata, event.ydata
        3. Update prompts dynamically
        4. Re-run model in real-time
        5. Display updated segmentation immediately
        6. Support box drawing (click-drag rectangle)
    
    Returns:
        pred_mask_v1 (np.ndarray): Initial prediction mask
        pred_mask_v2 (np.ndarray): Refined prediction mask
    
    Outputs:
        - Saved visualization: minisam_interactive_demo.png
        - Prints summary: IoU improvement from refinement
    
    Use Cases:
        - Demonstrating interactive segmentation capabilities
        - Understanding SAM-style prompt-based models
        - Comparing with fully automatic methods (FCN, DeepLabV3+)
        - Prototyping interactive annotation tools
    
    Insights:
        - Interactive models require fewer training images (prompt provides strong signal)
        - Trade-off: Better zero-shot but needs user input
        - IoU head helps users decide if more prompts are needed
    """
    # Task 7.2: Create interactive demo
    
    print("="*80)
    print("INTERACTIVE MINI-SAM DEMO")
    print("="*80)
    
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load trained Mini-SAM model
    print("\n[1/7] Loading Mini-SAM model...")
    model = MiniSAM(n_classes=21)
    model = model.to(device)
    
    # Try to load checkpoint
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(script_dir, '..', '..', 'best_minisam_model.pth')
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Loaded trained checkpoint")
    except Exception as e:
        print(f"⚠ Error loading checkpoint: {e}")
        print("⚠ Using random weights")
    
    model.eval()
    
    # 2. Load test image
    print("\n[2/7] Loading test image...")
    voc_path = os.path.join(script_dir, '..', '..', 'voc', 'VOC2012_train_val', 'VOC2012_train_val')
    val_dataset = VOCSegmentationDataset(
        root_dir=voc_path,
        split='val',
        image_size=256
    )
    
    # Get a random test image
    idx = np.random.randint(0, len(val_dataset))
    image, gt_mask = val_dataset[idx]
    
    print(f"✓ Loaded test image {idx}")
    
    # 3. Display image and use pre-defined points (simulating user clicks)
    print("\n[3/7] Setting up prompts...")
    
    # Prepare image for model
    image_tensor = image.unsqueeze(0).to(device)  # Add batch dimension
    
    # Simulate initial user clicks (foreground and background points)
    # These would normally come from user interaction
    initial_points = torch.tensor([
        [0.3, 0.3],  # Foreground point
        [0.5, 0.5],  # Foreground point
        [0.1, 0.1],  # Background point
    ]).unsqueeze(0).to(device)  # Shape: (1, 3, 2)
    
    initial_labels = torch.tensor([1, 1, 0]).unsqueeze(0).to(device)  # Shape: (1, 3)
    
    print(f"✓ Initial prompts: {initial_points.shape[1]} points")
    print(f"  - Foreground points: {(initial_labels == 1).sum().item()}")
    print(f"  - Background points: {(initial_labels == 0).sum().item()}")
    
    # 4. Run model with point prompts
    print("\n[4/7] Running initial segmentation...")
    with torch.no_grad():
        mask_logits_v1, iou_pred_v1 = model(image_tensor, initial_points, initial_labels)
        pred_mask_v1 = mask_logits_v1.argmax(dim=1)[0].cpu().numpy()
    
    print(f"✓ Initial segmentation complete (Predicted IoU: {iou_pred_v1.item():.3f})")
    
    # 5. Display segmentation result
    print("\n[5/7] Visualizing initial result...")
    
    # Denormalize image for display
    img_display = image.cpu()
    img_display = img_display * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_display = img_display + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    img_display = torch.clamp(img_display, 0, 1).permute(1, 2, 0).numpy()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Initial segmentation
    axes[0, 0].imshow(img_display)
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')
    
    # Plot initial points
    points_np = initial_points[0].cpu().numpy()
    labels_np = initial_labels[0].cpu().numpy()
    fg_points = points_np[labels_np == 1]
    bg_points = points_np[labels_np == 0]
    
    axes[0, 1].imshow(img_display)
    if len(fg_points) > 0:
        axes[0, 1].scatter(fg_points[:, 1] * 256, fg_points[:, 0] * 256, 
                          c='green', s=200, marker='*', edgecolors='white', linewidths=2,
                          label='Foreground')
    if len(bg_points) > 0:
        axes[0, 1].scatter(bg_points[:, 1] * 256, bg_points[:, 0] * 256, 
                          c='red', s=200, marker='x', linewidths=3,
                          label='Background')
    axes[0, 1].set_title(f'Initial Prompts ({len(points_np)} points)')
    axes[0, 1].legend()
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(pred_mask_v1, cmap='tab20', vmin=0, vmax=20)
    axes[0, 2].set_title(f'Initial Prediction (IoU: {iou_pred_v1.item():.3f})')
    axes[0, 2].axis('off')
    
    # 6. Add correction points (simulating refinement)
    print("\n[6/7] Adding correction points...")
    
    # Add more points to refine the segmentation
    correction_points = torch.tensor([
        [0.7, 0.7],  # Additional foreground
        [0.2, 0.8],  # Additional background
    ]).unsqueeze(0).to(device)
    
    correction_labels = torch.tensor([1, 0]).unsqueeze(0).to(device)
    
    # Combine with initial points
    refined_points = torch.cat([initial_points, correction_points], dim=1)
    refined_labels = torch.cat([initial_labels, correction_labels], dim=1)
    
    print(f"✓ Refined prompts: {refined_points.shape[1]} points")
    print(f"  - Foreground points: {(refined_labels == 1).sum().item()}")
    print(f"  - Background points: {(refined_labels == 0).sum().item()}")
    
    # 7. Re-run and display refined result
    print("\n[7/7] Running refined segmentation...")
    with torch.no_grad():
        mask_logits_v2, iou_pred_v2 = model(image_tensor, refined_points, refined_labels)
        pred_mask_v2 = mask_logits_v2.argmax(dim=1)[0].cpu().numpy()
    
    print(f"✓ Refined segmentation complete (Predicted IoU: {iou_pred_v2.item():.3f})")
    print(f"  IoU improvement: {(iou_pred_v2.item() - iou_pred_v1.item()):.3f}")
    
    # Row 2: Refined segmentation
    axes[1, 0].imshow(gt_mask.cpu().numpy(), cmap='tab20', vmin=0, vmax=20)
    axes[1, 0].set_title('Ground Truth')
    axes[1, 0].axis('off')
    
    # Plot refined points
    points_np_refined = refined_points[0].cpu().numpy()
    labels_np_refined = refined_labels[0].cpu().numpy()
    fg_points_refined = points_np_refined[labels_np_refined == 1]
    bg_points_refined = points_np_refined[labels_np_refined == 0]
    
    axes[1, 1].imshow(img_display)
    if len(fg_points_refined) > 0:
        axes[1, 1].scatter(fg_points_refined[:, 1] * 256, fg_points_refined[:, 0] * 256, 
                          c='green', s=200, marker='*', edgecolors='white', linewidths=2,
                          label='Foreground')
    if len(bg_points_refined) > 0:
        axes[1, 1].scatter(bg_points_refined[:, 1] * 256, bg_points_refined[:, 0] * 256, 
                          c='red', s=200, marker='x', linewidths=3,
                          label='Background')
    axes[1, 1].set_title(f'Refined Prompts ({len(points_np_refined)} points)')
    axes[1, 1].legend()
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(pred_mask_v2, cmap='tab20', vmin=0, vmax=20)
    axes[1, 2].set_title(f'Refined Prediction (IoU: {iou_pred_v2.item():.3f})')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    demo_path = os.path.join(script_dir, '..', '..', 'minisam_interactive_demo.png')
    plt.savefig(demo_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved demo visualization to '{demo_path}'")
    plt.show()
    
    # Summary
    print("\n" + "="*80)
    print("DEMO SUMMARY")
    print("="*80)
    print(f"Initial segmentation - Points: {initial_points.shape[1]}, IoU: {iou_pred_v1.item():.3f}")
    print(f"Refined segmentation - Points: {refined_points.shape[1]}, IoU: {iou_pred_v2.item():.3f}")
    print(f"Improvement: {(iou_pred_v2.item() - iou_pred_v1.item()):.3f}")
    print("="*80)
    
    print("\nNote: In a real interactive demo, you would:")
    print("  1. Use matplotlib event handlers to capture user clicks")
    print("  2. Update the visualization in real-time")
    print("  3. Allow multiple refinement iterations")
    print("  4. Support both point and box prompts")
    
    return pred_mask_v1, pred_mask_v2


if __name__ == "__main__":
    print("Lab 5: Advanced Image Segmentation")
    print("=" * 60)
    print("\nComplete all TODOs in the order they appear!")
    print("\nRecommended implementation order:")
    print("1. Part 1: FCN-32s (simplest, no skip connections)")
    print("2. Part 1: FCN-16s (add one skip connection)")
    print("3. Part 1: FCN-8s (add two skip connections)")
    print("4. Part 2: ASPP module")
    print("5. Part 2: DeepLabV3+")
    print("6. Part 3: Mini-SAM (most challenging!)")
    print("7. Part 4: Loss functions")
    print("8. Part 5: Training and evaluation")
    print("\n" + "=" * 60)
    
    # ==================== CONFIGURATION ====================
    # Set which operations to run
    TRAIN_MODELS = []  # List of models: ['fcn32s', 'fcn16s', 'fcn8s', 'deeplabv3plus', 'minisam']
                              # Or use 'all' to train all models sequentially
    RUN_COMPARISON = True     # Compare all trained models
    RUN_INTERACTIVE_DEMO = False  # Run Mini-SAM interactive demo
    # =======================================================
    
    # Train models
    if TRAIN_MODELS:
        if TRAIN_MODELS == 'all' or (isinstance(TRAIN_MODELS, list) and 'all' in TRAIN_MODELS):
            models_to_train = ['fcn32s', 'fcn16s', 'fcn8s', 'deeplabv3plus', 'minisam']
        else:
            models_to_train = TRAIN_MODELS if isinstance(TRAIN_MODELS, list) else [TRAIN_MODELS]
        
        for idx, model_name in enumerate(models_to_train, 1):
            print("\n" + "=" * 80)
            print(f"TRAINING MODEL {idx}/{len(models_to_train)}: {model_name.upper()}")
            print("=" * 80)
            main(model_name=model_name)
    
    # Run comparisons
    if RUN_COMPARISON:
        print("\n" + "=" * 80)
        print("RUNNING MODEL COMPARISON")
        print("=" * 80)
        compare_models()
    
    # Run interactive demo
    if RUN_INTERACTIVE_DEMO:
        print("\n" + "=" * 80)
        print("RUNNING INTERACTIVE DEMO")
        print("=" * 80)
        interactive_minisam_demo()
