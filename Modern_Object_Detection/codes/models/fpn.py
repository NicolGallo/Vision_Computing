"""
Feature Pyramid Network (FPN) implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    """
    Feature Pyramid Network for multi-scale feature fusion
    
    Reference: "Feature Pyramid Networks for Object Detection" (Lin et al., 2017)
    """
    
    def __init__(self, in_channels_list, out_channels=256):
        """
        Args:
            in_channels_list: List of input channels for each level [C3, C4, C5]
                             e.g., [512, 1024, 2048] for ResNet50
            out_channels: Output channels for all FPN levels (typically 256)
        """
        super().__init__()
        
        self.out_channels = out_channels
        self.num_levels = len(in_channels_list)
        
        # Lateral connections (1x1 conv to reduce channels)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])
        
        # Output convolutions (3x3 conv after upsampling + addition)
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize convolutional layers"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features):
        """
        Forward pass through FPN
        
        Args:
            features: List or dict of feature maps
                     If dict: expects keys 'C3', 'C4', 'C5'
                     If list: [C3, C4, C5]
        
        Returns:
            outputs: List of FPN feature maps [P3, P4, P5]
                    All with out_channels channels
        """
        # Handle both list and dict inputs
        if isinstance(features, dict):
            feature_list = [features[f'C{i+3}'] for i in range(self.num_levels)]
        else:
            feature_list = features
        
        # Apply lateral convolutions
        laterals = [
            lateral_conv(feature)
            for lateral_conv, feature in zip(self.lateral_convs, feature_list)
        ]
        
        # Build top-down pathway
        # Start from the deepest level (P5)
        outputs = [laterals[-1]]
        
        # Iterate from deep to shallow
        for i in range(self.num_levels - 2, -1, -1):
            # Upsample previous level
            upsampled = F.interpolate(
                outputs[0],
                size=laterals[i].shape[2:],
                mode='nearest'
            )
            
            # Add with lateral connection
            fused = upsampled + laterals[i]
            
            # Insert at beginning to maintain order
            outputs.insert(0, fused)
        
        # Apply output convolutions
        outputs = [
            output_conv(feature)
            for output_conv, feature in zip(self.output_convs, outputs)
        ]
        
        return outputs


class FPNWithExtraLevels(FPN):
    """
    FPN with additional coarser levels (P6, P7) for detecting larger objects
    Used in some detection frameworks like RetinaNet
    """
    
    def __init__(self, in_channels_list, out_channels=256, num_extra_levels=2):
        """
        Args:
            in_channels_list: Input channels [C3, C4, C5]
            out_channels: Output channels
            num_extra_levels: Number of extra coarser levels to add
        """
        super().__init__(in_channels_list, out_channels)
        
        # Extra levels (P6, P7, ...) are created by strided convolution
        self.extra_convs = nn.ModuleList()
        in_ch = out_channels
        
        for i in range(num_extra_levels):
            self.extra_convs.append(
                nn.Conv2d(in_ch, out_channels, kernel_size=3, stride=2, padding=1)
            )
            in_ch = out_channels
    
    def forward(self, features):
        """
        Forward pass with extra levels
        
        Returns:
            outputs: [P3, P4, P5, P6, P7, ...]
        """
        # Get standard FPN outputs [P3, P4, P5]
        outputs = super().forward(features)
        
        # Add extra levels
        x = outputs[-1]  # Start from P5
        for extra_conv in self.extra_convs:
            x = F.relu(extra_conv(x))
            outputs.append(x)
        
        return outputs