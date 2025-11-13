"""
ResNet backbone for feature extraction
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights


class ResNetBackbone(nn.Module):
    """
    ResNet backbone that outputs multi-scale features
    """
    
    def __init__(self, backbone_name='resnet50', pretrained=True):
        """
        Args:
            backbone_name: 'resnet18' or 'resnet50'
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        self.backbone_name = backbone_name
        
        # Load ResNet
        if backbone_name == 'resnet18':
            if pretrained:
                resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            else:
                resnet = resnet18(weights=None)
            self.feature_dims = [128, 256, 512]  # C3, C4, C5
        elif backbone_name == 'resnet50':
            if pretrained:
                resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            else:
                resnet = resnet50(weights=None)
            self.feature_dims = [512, 1024, 2048]  # C3, C4, C5
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        
        # Extract layers
        # C1: conv1 + bn1 + relu + maxpool (stride 4)
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        
        # C2: layer1 (stride 4)
        self.layer1 = resnet.layer1
        
        # C3: layer2 (stride 8)
        self.layer2 = resnet.layer2
        
        # C4: layer3 (stride 16)
        self.layer3 = resnet.layer3
        
        # C5: layer4 (stride 32)
        self.layer4 = resnet.layer4
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [B, 3, H, W]
        
        Returns:
            features: Dictionary with keys 'C3', 'C4', 'C5'
        """
        c1 = self.layer0(x)  # stride 4
        c2 = self.layer1(c1)  # stride 4
        c3 = self.layer2(c2)  # stride 8
        c4 = self.layer3(c3)  # stride 16
        c5 = self.layer4(c4)  # stride 32
        
        return {
            'C3': c3,
            'C4': c4,
            'C5': c5
        }
    
    def get_feature_dims(self):
        """Get output channels for each feature level"""
        return self.feature_dims


class SimpleBackbone(nn.Module):
    """
    Simplified backbone for YOLO (single-scale output)
    """
    
    def __init__(self, backbone_name='resnet18', pretrained=True):
        """
        Args:
            backbone_name: 'resnet18' or 'resnet50'
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        if backbone_name == 'resnet18':
            if pretrained:
                resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            else:
                resnet = resnet18(weights=None)
            self.out_channels = 512
        elif backbone_name == 'resnet50':
            if pretrained:
                resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            else:
                resnet = resnet50(weights=None)
            self.out_channels = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        
        # Remove avgpool and fc layers
        self.features = nn.Sequential(*list(resnet.children())[:-2])
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [B, 3, H, W]
        
        Returns:
            features: Feature tensor [B, C, H/32, W/32]
        """
        return self.features(x)
    
    def get_out_channels(self):
        """Get output channels"""
        return self.out_channels