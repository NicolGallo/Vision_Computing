"""
Lab 5: Advanced Image Segmentation
Student Template - Complete all TODOs

Author: [Your Name]
Date: [Current Date]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm
import warnings
from lab05_utils import plot_segmentation_results
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ================== Part 1: FCN Architecture ==================

class FCN32s(nn.Module):
    """FCN without skip connections (baseline)"""
    
    def __init__(self, n_classes=21):
        super().__init__()
        
        # Task 1.1: Load pretrained ResNet50 and extract layers
        # 1. Load models.resnet50(pretrained=True)
        resnet = models.resnet50(pretrained=True)

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

        self.upscore32 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=64, stride=32, bias=False)
        
    def forward(self, x):
        # Task 1.4: Implement forward pass
        # 1. Pass through conv1, bn1, relu, maxpool
        # 2. Pass through layer1, layer2, layer3, layer4
        # 3. Apply score layer (1x1 conv)
        # 4. Apply 32x upsampling
        # 5. Return output

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # Aquí x tiene stride 4 (1/4 del tamaño original)

        x = self.layer1(x)  # Aquí x tiene stride 4 (1/4 del tamaño original)
        x = self.layer2(x)  # Aquí x tiene stride 8 (1/8 del tamaño original)
        x = self.layer3(x)  # Aquí x tiene stride 16 (1/16 del tamaño original)
        x = self.layer4(x)  # Aquí x tiene stride 32 (1/32 del tamaño original)

        # Esto mapea los 2048 canales a las n_classes (ej. 21)
        x = self.score_fr(x)  # Aplicar la capa de puntuación (1x1 conv). x sigue teniendo stride 32 y n_classes canales
        x = self.upscore32(x)  # Upsampling 32x para volver al tamaño original
     
        return x



class FCN16s(nn.Module):
    """FCN with one skip connection from pool4"""
    
    def __init__(self, n_classes=21):
        super().__init__()
        
        # Task 1.1: Load pretrained ResNet50 and extract layers
        # (Same as FCN32s)
        resnet = models.resnet50(pretrained=True)

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
        
        self.upscore2 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=2, bias=False)
        self.upscore16 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=32, stride=16, bias=False)
        
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
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)  
        pool4 = self.layer3(x)  
        x = self.layer4(pool4)

        x = self.score_fr(x)
        pool4 = self.score_pool4(pool4)

        x = self.upscore2(x)
        if x.shape != pool4.shape:
            x = F.interpolate(x, size=pool4.shape[2:], mode='bilinear', align_corners=False)
        
        x = x + pool4
        x = self.upscore16(x)

        return x



class FCN8s(nn.Module):
    """Fully Convolutional Network with two skip connections"""
    
    def __init__(self, n_classes=21):
        super().__init__()
        
        # Task 1.1: Load pretrained ResNet50 and extract layers
        # 1. Load models.resnet50(pretrained=True)
        # 2. Extract conv1, bn1, relu, maxpool
        # 3. Extract layer1 (stride 4)
        # 4. Extract layer2 (stride 8, will be pool3)
        # 5. Extract layer3 (stride 16, will be pool4)
        # 6. Extract layer4 (stride 32)

        resnet = models.resnet50(pretrained=True)

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
        
        self.upscore2 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=2, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=16, stride=8, bias=False)
        
    def forward(self, x):
        # Task 1.4: Implement forward pass with progressive skip fusion
        # ENCODER PATH:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x) # stride 4
        pool3 = self.layer2(x) # Stride 8, save for skip connection
        pool4 = self.layer3(pool3)  # Stride 16, save for skip connection
        x = self.layer4(pool4) #Stride 32
        
        # SCORE LAYERS:
        score_fr = self.score_fr(x)
        score_pool4 = self.score_pool4(pool4)
        score_pool3 = self.score_pool3(pool3)
        
        # PROGRESSIVE UPSAMPLING WITH SKIP CONNECTIONS:
        # First skip (pool4 at stride 16):
        upscore2 = self.upscore2(score_fr)  # Upsample 32 -> 16

        # 11. If shapes don't match, use F.interpolate to resize
        if upscore2.shape != score_pool4.shape:
            upscore2 = F.interpolate(upscore2, size=score_pool4.shape[2:], mode='bilinear', align_corners=False)
        
        fuse_pool4 = upscore2 + score_pool4  # Element-wise addition
        
        # Second skip (pool3 at stride 8):
        upscore_pool4 = self.upscore_pool4(fuse_pool4)  # Upsample 16 -> 8

        # 14. If shapes don't match, use F.interpolate to resize
        if upscore_pool4.shape != score_pool3.shape:
            upscore_pool4 = F.interpolate(upscore_pool4, size=score_pool3.shape[2:], mode='bilinear', align_corners=False)

        fuse_pool3 = upscore_pool4 + score_pool3  # Element-wise addition
        
        # Final upsampling:
        out = self.upscore8(fuse_pool3) # Upsample 8 -> 1 (original resolution)
        
        return out



# ================== Part 2: DeepLabV3+ Architecture ==================

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module"""
    
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
        size = x.shape[2:]
        
        # Branch 1:
        # 2. feat1 = self.conv1(x)
        feat1 = self.conv1x1(x)
        
        # Branches 2-4:
        # 3. feat_atrous = [conv(x) for conv in self.atrous_convs]
        feat_atrous = [conv(x) for conv in self.atrous_convs]

        # Branch 5:
        # 4. feat_global = self.global_avg_pool(x)
        feat_global = self.global_avg_pool(x)

        # 5. Interpolate feat_global back to original size using F.interpolate
        feat_global_upsampled = F.interpolate(feat_global, size=size, mode='bilinear', align_corners=False)

        # Concatenation:
        # 6. feat = torch.cat([feat1] + feat_atrous + [feat_global], dim=1)
        all_features = [feat1] + feat_atrous + [feat_global_upsampled]
        feat = torch.cat(all_features, dim=1)
        
        # Output projection:
        return self.conv_out(feat)


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ architecture with ASPP"""
    
    def __init__(self, n_classes=21, backbone='resnet50'):
        super().__init__()
        
        # Task 2.3: Load backbone and extract encoder layers
        # 1. Load models.resnet50(pretrained=True)
        # 2. Extract conv1, bn1, relu, maxpool
        # 3. Extract layer1 (low-level features, stride 4)
        # 4. Extract layer2, layer3, layer4
        
        # 1. Cargar ResNet50
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=True)
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
        input_size = x.shape[2:]
        
        # ENCODER:
        # 2. x = relu(bn1(conv1(x)))
        # 3. x = maxpool(x)
        # 4. low_level_feat = layer1(x) - Save for decoder
        # 5. x = layer2(low_level_feat)
        # 6. x = layer3(x)
        # 7. x = layer4(x)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        low_level_feat = self.layer1(x)  # Guardar características de bajo nivel (stride 4)
        x = self.layer2(low_level_feat)
        x = self.layer3(x)
        x = self.layer4(x)  # Salida a stride 32
        
        # ASPP:
        # 8. x = self.aspp(x)
        x = self.aspp(x) # Salida a stride 32, 256 canales
        
        # DECODER:
        # 9. Upsample x to match low_level_feat size using F.interpolate
        #    Upsample de x (stride 32) a low_level_feat (stride 4)
        x = F.interpolate(x, size=low_level_feat.shape[2:], mode='bilinear', align_corners=False)

        # 10. low_level_feat = self.low_level_conv(low_level_feat)
        #     Proyectar características de bajo nivel
        low_level_feat = self.low_level_conv(low_level_feat) # Salida a stride 4, 48 canales

        # 11. x = torch.cat([x, low_level_feat], dim=1) - Concatenate along channel dimension
        #     Concatenar características
        x = torch.cat([x, low_level_feat], dim=1) # Salida: 256 + 48 = 304 canales

        # 12. x = self.decoder(x)
        #     Pasar por el decodificador
        x = self.decoder(x) # Salida: 256 canales, stride 4
        
        # CLASSIFICATION:
        # 13. x = self.classifier(x)
        x = self.classifier(x) # Salida: n_classes, stride 4

        # 14. Upsample x to original input size using F.interpolate
        #     Upsample final a tamaño original
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)

        # 15. Return x
        return x


# ================== Part 3: Mini-SAM Architecture ==================

class MiniSAM(nn.Module):
    """Simplified SAM architecture trainable from scratch (~5M parameters)"""
    
    def __init__(self, n_classes=21, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Task 3.1: Create lightweight image encoder
        # 1. Load models.mobilenet_v3_small(pretrained=True)
        backbone = models.mobilenet_v3_small(pretrained=True)

        # 2. Extract features: nn.Sequential(*list(backbone.features))
        #    Extraer características, EXCLUYENDO el pooling final. Esto nos da una salida de [B, 576, H/16, W/16]
        self.image_encoder = nn.Sequential(*list(backbone.features)[:-1])

        # 3. Create projection: nn.Conv2d(576, embed_dim, 1) - MobileNetV3-Small outputs 576 channels
        #    Crear proyección de 576 -> embed_dim (ej. 256)
        self.img_proj = nn.Conv2d(576, embed_dim, kernel_size=1)

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
        """Extract image features"""
        # Task 3.4: Encode image
        # 1. features = self.image_encoder(x) - Output: B x 576 x H/8 x W/8
        features = self.image_encoder(x)
        
        # 2. features = self.img_proj(features) - Output: B x embed_dim x H/8 x W/8
        features = self.img_proj(features)
        
        # 3. Return features
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
            type_enc = self.point_type_embed(point_labels)
            
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
        
        # ENCODE PROMPTS:
        # 3. prompt_features = self.encode_prompts(points, point_labels, boxes, img_size=(H, W))
        prompt_features = self.encode_prompts(points, point_labels, boxes, img_size=(H, W))
        
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
        
        # PREDICT IoU:
        # 8. iou_pred = self.iou_head(decoded)
        iou_pred = self.iou_head(decoded)
        
        # 9. Return mask_logits, iou_pred
        return mask_logits, iou_pred


def sample_points_from_mask(masks, n_points=5):
    """
    Sample points from ground truth masks (simulates user clicks)
    Args:
        masks: (B, H, W) ground truth class indices
        n_points: number of points to sample
    Returns:
        points: (B, n_points, 2) normalized coordinates
        labels: (B, n_points) with 0=bg, 1=fg
    """
    # Task 3.5: Implement point sampling
    # 1. Get B, H, W = masks.shape
    B, H, W = masks.shape
    
    # 2. Create empty lists: points_list = [], labels_list = []
    points_list = []
    labels_list = []
    
    # FOR EACH IMAGE IN BATCH:
    # 3. For b in range(B):
    for b in range(B):
        #    - Get mask = masks[b]
        mask = masks[b]
        
        #    SAMPLE FOREGROUND POINTS (50%):
        #    4. fg_indices = torch.nonzero(mask > 0) - Find all foreground pixels
        fg_indices = torch.nonzero(mask > 0, as_tuple=False)
        
        #    5. If fg_indices not empty:
        if len(fg_indices) > 0:
            #       - Randomly sample n_points//2 indices
            n_fg = n_points // 2
            sampled_idx = torch.randint(0, len(fg_indices), (n_fg,))
            fg_points = fg_indices[sampled_idx].float()
            
            #       - Normalize to [0,1]: divide by [H, W]
            fg_points[:, 0] /= H
            fg_points[:, 1] /= W
            
            #       - Create labels as ones
            fg_labels = torch.ones(n_fg, dtype=torch.long, device=masks.device)
        #    6. Else: create empty tensors
        else:
            fg_points = torch.zeros(0, 2, device=masks.device)
            fg_labels = torch.zeros(0, dtype=torch.long, device=masks.device)
        
        #    SAMPLE BACKGROUND POINTS (50%):
        #    7. bg_indices = torch.nonzero(mask == 0) - Find all background pixels
        bg_indices = torch.nonzero(mask == 0, as_tuple=False)
        
        #    8. If bg_indices not empty:
        if len(bg_indices) > 0:
            #       - Randomly sample n_points//2 indices
            n_bg = n_points - (n_points // 2)  # Remaining points
            sampled_idx = torch.randint(0, len(bg_indices), (n_bg,))
            bg_points = bg_indices[sampled_idx].float()
            
            #       - Normalize to [0,1]: divide by [H, W]
            bg_points[:, 0] /= H
            bg_points[:, 1] /= W
            
            #       - Create labels as zeros
            bg_labels = torch.zeros(n_bg, dtype=torch.long, device=masks.device)
        #    9. Else: create empty tensors
        else:
            bg_points = torch.zeros(0, 2, device=masks.device)
            bg_labels = torch.zeros(0, dtype=torch.long, device=masks.device)
        
        #    COMBINE AND PAD:
        #    10. Concatenate fg_points and bg_points
        combined_points = torch.cat([fg_points, bg_points], dim=0)
        
        #    11. Concatenate fg_labels and bg_labels
        combined_labels = torch.cat([fg_labels, bg_labels], dim=0)
        
        #    12. If total points < n_points: pad with zeros
        if len(combined_points) < n_points:
            pad_size = n_points - len(combined_points)
            pad_points = torch.zeros(pad_size, 2, device=masks.device)
            pad_labels = torch.zeros(pad_size, dtype=torch.long, device=masks.device)
            combined_points = torch.cat([combined_points, pad_points], dim=0)
            combined_labels = torch.cat([combined_labels, pad_labels], dim=0)
        
        #    13. Append to lists
        points_list.append(combined_points)
        labels_list.append(combined_labels)
    
    # 14. Stack lists and return torch.stack(points_list), torch.stack(labels_list)
    return torch.stack(points_list), torch.stack(labels_list)


# ================== Loss Functions ==================

class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) logits
            target: (B, H, W) class indices
        """    
        # Task 4.1: Implement Dice loss
        # 1. Apply softmax to pred: pred = torch.softmax(pred, dim=1)
        pred = torch.softmax(pred, dim=1) 

        # 2. Convert target to one-hot: target_one_hot = F.one_hot(target, num_classes=pred.shape[1])
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]) # Shape: (B, H, W, C)

        # 3. Permute target_one_hot to (B, C, H, W) and convert to float
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float() # Shape: (B, C, H, W)

        # 4. Flatten spatial dimensions: pred_flat = pred.view(B, C, -1)
        B, C, H, W = pred.shape
        pred_flat = pred.view(B, C, -1) # Shape: (B, C, H*W)
        
        # 5. Flatten target: target_flat = target_one_hot.view(B, C, -1)
        target_flat = target_one_hot.view(B, C, -1) # Shape: (B, C, H*W)

        # 6. Compute intersection: (pred_flat * target_flat).sum(dim=2)
        intersection = (pred_flat * target_flat).sum(dim=2) # Shape: (B, C)

        # 7. Compute dice = (2 * intersection + smooth) / (pred_flat.sum(dim=2) + target_flat.sum(dim=2) + smooth)
        dice = (2. * intersection + self.smooth) / (pred_flat.sum(dim=2) + target_flat.sum(dim=2) + self.smooth) # Shape: (B, C)

        # 8. Return 1 - dice.mean()
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) logits
            target: (B, H, W) class indices
        """
        # Task 4.2: Implement Focal loss
        # 1. Compute cross-entropy: ce_loss = F.cross_entropy(pred, target, reduction='none')
        ce_loss = F.cross_entropy(pred, target, reduction='none')

        # 2. Compute pt = torch.exp(-ce_loss) - Probability of correct class
        pt = torch.exp(-ce_loss)

        # 3. Apply focal term: focal_loss = alpha * (1 - pt)^gamma * ce_loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        # 4. Return focal_loss.mean()
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """Combined loss: CE + Dice + Focal"""
    
    def __init__(self, weights={'ce': 0.3, 'dice': 0.5, 'focal': 0.2}):
        super().__init__()
        self.weights = weights
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        
    def forward(self, pred, target):
        # Task 4.3: Compute weighted combination
        # 1. Compute: ce = self.ce_loss(pred, target)
        # 2. Compute: dice = self.dice_loss(pred, target)
        # 3. Compute: focal = self.focal_loss(pred, target)
        # 4. Return: weights['ce'] * ce + weights['dice'] * dice + weights['focal'] * focal
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        return self.weights['ce'] * ce + self.weights['dice'] * dice + self.weights['focal'] * focal


# ================== Evaluation Metrics ==================

def calculate_miou(pred, target, num_classes):
    """
    Calculate mean Intersection over Union
    Args:
        pred: (B, H, W) predicted class indices
        target: (B, H, W) ground truth class indices
        num_classes: int
    Returns:
        miou: float
        class_iou: numpy array (num_classes,)
    """
    # Task 4.4: Implement mIoU
    # 1. Create empty list: ious = []
    ious = []

    # 2. Convert pred and target to numpy
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

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

def calculate_pixel_accuracy(pred, target):
    """Calculate pixel accuracy"""
    # Task 4.5: Implement pixel accuracy
    # 1. correct = (pred == target).sum()
    correct = (pred == target).sum()

    # 2. total = pred.numel()
    total = pred.numel()

    # 3. Return (correct / total).item()
    return (correct / total).item()


def compute_batch_iou(pred, target):
    """
    Compute IoU for each image in batch
    Args:
        pred: (B, H, W)
        target: (B, H, W)
    Returns:
        iou: (B,) tensor
    """
    # Task 4.6: Implement batch IoU
    # 1. Get B = pred.shape[0]
    B = pred.shape[0] 

    # 2. Create empty list: ious = []
    ious = []

    # 3. For each b in range(B):
        #    - intersection = ((pred[b] == target[b]) & (target[b] > 0)).float().sum()
        #    - union = ((pred[b] > 0) | (target[b] > 0)).float().sum()
        #    - iou = intersection / (union + 1e-6)
        #    - Append iou
    for b in range(B): #cada imagen del batch
        intersection = ((pred[b] == target[b]) & (target[b] > 0)).float().sum()
        union = ((pred[b] > 0) | (target[b] > 0)).float().sum()
        if union == 0:
            iou = torch.tensor(1.0, device=pred.device)  # Both empty
        else:
            iou = intersection / (union + 1e-6)
        ious.append(iou)

    # 4. Return torch.stack(ious)
    return torch.stack(ious)


# ================== Training Functions ==================

def train_epoch_fcn(model, dataloader, optimizer, criterion, device):
    """Train FCN/DeepLab for one epoch"""
    model.train()
    total_loss = 0
    total_miou = 0
    
    # Task 5.1: Implement training loop
    # 1. For images, masks in dataloader:
    # 2. Move to device: images, masks = images.to(device), masks.to(device)
    for images, masks in tqdm(dataloader, desc='Training'):
        images, masks = images.to(device), masks.to(device)

    # 3. Zero gradients: optimizer.zero_grad()
    # 4. Forward pass: outputs = model(images)
    # 5. Compute loss: loss = criterion(outputs, masks)
    # 6. Backward: loss.backward()
    # 7. Update weights: optimizer.step()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

    # 8. Calculate metrics: pred = outputs.argmax(dim=1), then miou = calculate_miou(pred, masks, num_classes)
        pred = outputs.argmax(dim=1)
        miou, _ = calculate_miou(pred, masks, num_classes=21)
        loss_iou = torch.tensor(miou, device=device)

    # 9. Accumulate: total_loss += loss.item(), total_miou += miou
        total_loss = total_loss + loss.item()
        total_iou = total_iou + loss_iou

    # 10. Return averages: total_loss / len(dataloader), total_miou / len(dataloader)
    return total_loss / len(dataloader), total_miou / len(dataloader)


def train_epoch_minisam(model, dataloader, optimizer, criterion, device, n_points=5):
    """Train Mini-SAM for one epoch with simulated prompts"""
    model.train()
    total_loss = 0
    total_miou = 0
    
    # Task 5.2: Implement Mini-SAM training
    # 1. For images, masks in dataloader:
    for images, masks in tqdm(dataloader, desc='Training Mini-SAM'):
        # 2. Move to device
        images, masks = images.to(device), masks.to(device)
        
        # 3. Sample points from masks: points, point_labels = sample_points_from_mask(masks, n_points)
        points, point_labels = sample_points_from_mask(masks, n_points)
        
        # 4. Zero gradients
        optimizer.zero_grad()
        
        # 5. Forward pass: mask_logits, iou_pred = model(images, points, point_labels)
        mask_logits, iou_pred = model(images, points, point_labels)
        
        # 6. Compute losses:
        #    - ce_loss = F.cross_entropy(mask_logits, masks)
        ce_loss = F.cross_entropy(mask_logits, masks)
        
        #    - dice_loss = DiceLoss()(mask_logits, masks)
        dice_loss = DiceLoss()(mask_logits, masks)
        
        #    - pred_masks = mask_logits.argmax(dim=1)
        pred_masks = mask_logits.argmax(dim=1)
        
        #    - true_iou = compute_batch_iou(pred_masks, masks)
        true_iou = compute_batch_iou(pred_masks, masks)
        
        #    - iou_loss = F.mse_loss(iou_pred.squeeze(), true_iou)
        iou_loss = F.mse_loss(iou_pred.squeeze(), true_iou)
        
        # 7. Combined loss: loss = ce_loss + dice_loss + 0.1 * iou_loss
        loss = ce_loss + dice_loss + 0.1 * iou_loss
        
        # 8. Backward and update
        loss.backward()
        optimizer.step()
        
        # 9. Calculate metrics
        miou, _ = calculate_miou(pred_masks, masks, num_classes=21)
        
        # Accumulate
        total_loss += loss.item()
        total_miou += miou
    
    # 10. Return averages
    return total_loss / len(dataloader), total_miou / len(dataloader)


def validate(model, dataloader, device, num_classes=21, is_minisam=False):
    """Validate the model"""
    model.eval()
    total_miou = 0
    total_pa = 0 # pixel accuracy
    
    # Task 5.3: Implement validation
    # 1. Use torch.no_grad() context
    # 2. For images, masks in dataloader:
    # 3. Move to device
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc='Validation'):
            images, masks = images.to(device), masks.to(device)
    
    # 4. If is_minisam:
    #    - Sample points
    #    - outputs, _ = model(images, points, point_labels)
    #    Else:
    #    - outputs = model(images)

            if is_minisam:
                points, point_labels = sample_points_from_mask(masks, n_points=5)
                outputs, _ = model(images, points, point_labels)
            else:
                outputs = model(images)

    # 5. Get predictions: pred = outputs.argmax(dim=1)
            pred = outputs.argmax(dim=1)

    # 6. Calculate metrics
            miou, _ = calculate_miou(pred, masks, num_classes)
            pa = calculate_pixel_accuracy(pred, masks)
            loss_iou = torch.tensor(miou, device=device)
            loss_pa = torch.tensor(pa, device=device)

    # 7. Accumulate
            total_miou = total_miou + loss_iou
            total_pa = total_pa + loss_pa

    # 8. Return averages
    return total_miou / len(dataloader), total_pa / len(dataloader)


def visualize_predictions(model, dataloader, device, num_samples=4, is_minisam=False, save_path='predictions.png'):
    """Visualize model predictions"""
    model.eval()
    
    # Task 5.4: Implement visualization
    # 1. Get one batch: images_batch, masks_batch = next(iter(dataloader))
    # 2. Take first num_samples and move to device
    with torch.no_grad():
        images_batch, masks_batch = next(iter(dataloader))
        images_batch, masks_batch = images_batch.to(device), masks_batch.to(device)
    
    # 3. Generate predictions (with or without prompts based on is_minisam)
        if is_minisam:
            points, point_labels = sample_points_from_mask(masks_batch, n_points=5)
            outputs, _ = model(images_batch, points, point_labels)
        else:
            outputs = model(images_batch)
        predictions = outputs.argmax(dim=1)

        # Opcional: limitar el número de muestras para la visualización
        batch_size = images_batch.size(0)
        num_to_plot = min(num_samples, batch_size)
        
        # Seleccionar solo las muestras que se van a visualizar
        images_to_plot = images_batch[:num_to_plot]
        masks_to_plot = masks_batch[:num_to_plot]
        predictions_to_plot = predictions[:num_to_plot]

    # 4. Create figure with subplots: (num_samples, 3)
    # 5. For each sample, plot:
    #    - Column 0: Input image
    #    - Column 1: Ground truth mask
    #    - Column 2: Predicted mask
    # 6. Save figure
    fig = plot_segmentation_results(images=images_to_plot, masks=masks_to_plot, predictions=predictions_to_plot, title=f"Segmentation Results (First {num_to_plot} Samples)")
    plt.show()
    fig.savefig(save_path)
    print(f"Saved predictions to {save_path}")

    return fig


# ================== Main Training Script ==================

def main():
    """Main training pipeline"""
    
    config = {
        'model': 'fcn8s',  # Options: 'fcn32s', 'fcn16s', 'fcn8s', 'deeplabv3plus', 'minisam'
        'n_classes': 21,
        'batch_size': 8,
        'learning_rate': 1e-4,
        'epochs': 30,
        'device': device
    }
    
    print(f"Training {config['model']} for {config['epochs']} epochs")
    
    # TODO Task 6.1: Setup data loaders
    # 1. Create dataset class or use existing PASCAL VOC dataset
    # 2. Create train_loader and val_loader with appropriate transforms
    
    # TODO Task 6.2: Create model
    # 1. Initialize model based on config['model']
    # 2. Move model to device
    
    # TODO Task 6.3: Setup optimizer and loss
    # 1. optimizer = torch.optim.AdamW(model.parameters(), lr=..., weight_decay=1e-4)
    # 2. scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(...)
    # 3. criterion = CombinedLoss()
    
    # TODO Task 6.4: Training loop
    # 1. Initialize best_miou = 0
    # 2. For each epoch:
    #    - Train: train_loss, train_miou = train_epoch_*(...) based on model type
    #    - Validate: val_miou, val_pa = validate(...)
    #    - Update scheduler: scheduler.step(val_miou)
    #    - Print metrics
    #    - If val_miou > best_miou: save model
    
    # TODO Task 6.5: Plot training curves
    # 1. Plot training loss over epochs
    # 2. Plot validation mIoU over epochs
    # 3. Save plots
    
    pass


def compare_models():
    """Compare all implemented models"""
    # TODO Task 7.1: Compare FCN-32s, FCN-16s, FCN-8s, DeepLabV3+, and Mini-SAM
    # 1. Create results dictionary
    # 2. For each model:
    #    - Load trained checkpoint
    #    - Evaluate on test set
    #    - Record mIoU, number of parameters, inference time
    # 3. Create comparison table (use pandas or print nicely)
    # 4. Generate side-by-side qualitative comparisons
    
    pass


def interactive_minisam_demo():
    """Interactive Mini-SAM demo with point/box prompting"""
    # TODO Task 7.2: Create interactive demo
    # 1. Load trained Mini-SAM model
    # 2. Load test image
    # 3. Display image and get user clicks (or use pre-defined points)
    # 4. Run model with point prompts
    # 5. Display segmentation result
    # 6. Allow user to add correction points
    # 7. Re-run and display refined result
    
    pass


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
    
    # Uncomment to run training
    # main()
    
    # Uncomment to run comparisons
    # compare_models()
    
    # Uncomment to run interactive demo
    # interactive_minisam_demo()