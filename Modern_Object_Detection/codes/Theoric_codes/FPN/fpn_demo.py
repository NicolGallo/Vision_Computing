"""
Feature Pyramid Network (FPN) Demo
Educational demonstration of multi-scale feature fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


class SimpleFPN(nn.Module):
    """
    Simplified FPN for educational demo
    Shows the core concept clearly
    """
    
    def __init__(self, in_channels_list=[512, 1024, 2048], out_channels=256):
        """
        Args:
            in_channels_list: [C3, C4, C5] channels from backbone
            out_channels: Unified output channels (typically 256)
        """
        super().__init__()
        
        self.out_channels = out_channels
        
        # Lateral 1x1 convolutions (reduce channels)
        self.lateral_conv3 = nn.Conv2d(in_channels_list[0], out_channels, 1)
        self.lateral_conv4 = nn.Conv2d(in_channels_list[1], out_channels, 1)
        self.lateral_conv5 = nn.Conv2d(in_channels_list[2], out_channels, 1)
        
        # Output 3x3 convolutions (smooth features)
        self.output_conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.output_conv4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.output_conv5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
    
    def forward(self, features):
        """
        FPN forward pass with detailed logging
        
        Args:
            features: Dict with 'C3', 'C4', 'C5' or list [C3, C4, C5]
        
        Returns:
            outputs: [P3, P4, P5] - multi-scale features
        """
        # Handle both dict and list inputs
        if isinstance(features, dict):
            c3, c4, c5 = features['C3'], features['C4'], features['C5']
        else:
            c3, c4, c5 = features
        
        print("\n📊 FPN Forward Pass:")
        print(f"  Input C3: {c3.shape} (stride 8, high resolution, weak semantics)")
        print(f"  Input C4: {c4.shape} (stride 16, medium resolution)")
        print(f"  Input C5: {c5.shape} (stride 32, low resolution, strong semantics)")
        
        # Step 1: Lateral connections (1x1 conv to align channels)
        lat3 = self.lateral_conv3(c3)
        lat4 = self.lateral_conv4(c4)
        lat5 = self.lateral_conv5(c5)
        
        print(f"\n  After lateral 1x1 convs (channel alignment):")
        print(f"    lat3: {lat3.shape}")
        print(f"    lat4: {lat4.shape}")
        print(f"    lat5: {lat5.shape}")
        
        # Step 2: Top-down pathway
        # Start from P5 (deepest)
        p5 = lat5
        
        # Build P4: upsample P5 and add to lat4
        p5_upsampled = F.interpolate(p5, size=lat4.shape[2:], mode='nearest')
        p4 = p5_upsampled + lat4
        
        print(f"\n  Top-down pathway:")
        print(f"    P5 (seed): {p5.shape}")
        print(f"    P5 upsampled: {p5_upsampled.shape}")
        print(f"    P4 (fused): {p4.shape}")
        
        # Build P3: upsample P4 and add to lat3
        p4_upsampled = F.interpolate(p4, size=lat3.shape[2:], mode='nearest')
        p3 = p4_upsampled + lat3
        
        print(f"    P4 upsampled: {p4_upsampled.shape}")
        print(f"    P3 (fused): {p3.shape}")
        
        # Step 3: Apply 3x3 smoothing convolutions
        p3_final = self.output_conv3(p3)
        p4_final = self.output_conv4(p4)
        p5_final = self.output_conv5(p5)
        
        print(f"\n  After 3x3 smoothing convs:")
        print(f"    P3: {p3_final.shape} ✅ (high-res + strong semantics)")
        print(f"    P4: {p4_final.shape} ✅ (medium-res + strong semantics)")
        print(f"    P5: {p5_final.shape} ✅ (low-res + strong semantics)")
        
        return [p3_final, p4_final, p5_final]


class SimpleBackbone(nn.Module):
    """Simplified backbone to generate C3, C4, C5"""
    
    def __init__(self):
        super().__init__()
        
        # Simplified ResNet-like backbone
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)  # /2
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)       # /4
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 512, 3, padding=1),
            nn.ReLU()
        )  # C3: stride 8
        
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.ReLU()
        )  # C4: stride 16
        
        self.layer4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(1024, 2048, 3, padding=1),
            nn.ReLU()
        )  # C5: stride 32
    
    def forward(self, x):
        """Extract multi-scale features"""
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        
        return {'C3': c3, 'C4': c4, 'C5': c5}


def visualize_fpn_architecture():
    """Visualize FPN architecture diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'Feature Pyramid Network (FPN) Architecture', 
            ha='center', fontsize=16, fontweight='bold')
    
    # Bottom-up pathway (backbone)
    levels = [
        {'name': 'C3', 'y': 3, 'size': 2.0, 'channels': 512, 'stride': 8, 'color': 'lightblue'},
        {'name': 'C4', 'y': 5, 'size': 1.5, 'channels': 1024, 'stride': 16, 'color': 'lightgreen'},
        {'name': 'C5', 'y': 7, 'size': 1.0, 'channels': 2048, 'stride': 32, 'color': 'lightcoral'},
    ]
    
    # Draw bottom-up pathway
    ax.text(0.5, 8.5, 'Bottom-Up\n(Backbone)', ha='center', fontsize=11, fontweight='bold')
    
    for level in levels:
        # Backbone features
        x = 1.5
        size = level['size']
        box = FancyBboxPatch((x - size/2, level['y'] - size/2), size, size,
                             boxstyle="round,pad=0.05", 
                             facecolor=level['color'], edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, level['y'], f"{level['name']}\n{level['channels']}ch", 
                ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(x, level['y'] - size/2 - 0.3, f"stride {level['stride']}", 
                ha='center', fontsize=8)
    
    # Arrows between backbone levels
    for i in range(len(levels) - 1):
        arrow = FancyArrowPatch((1.5, levels[i]['y'] + levels[i]['size']/2), 
                               (1.5, levels[i+1]['y'] - levels[i+1]['size']/2),
                               arrowstyle='->', lw=2, color='black')
        ax.add_patch(arrow)
    
    # Lateral connections (1x1 conv)
    ax.text(3.5, 8.5, 'Lateral\n(1×1 conv)', ha='center', fontsize=11, fontweight='bold')
    
    for i, level in enumerate(levels):
        # 1x1 conv
        box = FancyBboxPatch((3, level['y'] - 0.3), 1, 0.6,
                             boxstyle="round,pad=0.02",
                             facecolor='yellow', edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(3.5, level['y'], '1×1', ha='center', va='center', fontsize=9)
        
        # Arrow from backbone to lateral
        arrow = FancyArrowPatch((2.5, level['y']), (3, level['y']),
                               arrowstyle='->', lw=1.5, color='blue')
        ax.add_patch(arrow)
    
    # Top-down pathway
    ax.text(5.5, 8.5, 'Top-Down\n(Upsample + Add)', ha='center', fontsize=11, fontweight='bold')
    
    for i, level in enumerate(levels):
        # Top-down features
        size = 1.0
        box = FancyBboxPatch((5 - size/2, level['y'] - size/2), size, size,
                             boxstyle="round,pad=0.05",
                             facecolor='plum', edgecolor='black', linewidth=2)
        ax.add_patch(box)
        
        # Arrow from lateral to top-down
        arrow = FancyArrowPatch((4, level['y']), (4.5, level['y']),
                               arrowstyle='->', lw=1.5, color='green')
        ax.add_patch(arrow)
    
    # Upsampling arrows (top-down)
    for i in range(len(levels) - 1, 0, -1):
        # Upsample arrow
        arrow = FancyArrowPatch((5, levels[i]['y'] - 0.6), 
                               (5, levels[i-1]['y'] + 0.6),
                               arrowstyle='->', lw=2, color='red', linestyle='--')
        ax.add_patch(arrow)
        ax.text(5.3, (levels[i]['y'] + levels[i-1]['y'])/2, 'upsample', 
                fontsize=8, color='red', rotation=270, va='center')
    
    # Output convolutions (3x3)
    ax.text(7, 8.5, 'Output\n(3×3 conv)', ha='center', fontsize=11, fontweight='bold')
    
    pyramid_features = []
    for i, level in enumerate(levels):
        # 3x3 conv
        box = FancyBboxPatch((6.5, level['y'] - 0.3), 1, 0.6,
                             boxstyle="round,pad=0.02",
                             facecolor='lightgray', edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(7, level['y'], '3×3', ha='center', va='center', fontsize=9)
        
        # Arrow from top-down to output
        arrow = FancyArrowPatch((5.5, level['y']), (6.5, level['y']),
                               arrowstyle='->', lw=1.5, color='purple')
        ax.add_patch(arrow)
        
        # Final pyramid features
        size = 1.2
        box = FancyBboxPatch((8.5 - size/2, level['y'] - size/2), size, size,
                             boxstyle="round,pad=0.05",
                             facecolor='orange', edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(8.5, level['y'], f"P{i+3}\n256ch", 
                ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Arrow to final
        arrow = FancyArrowPatch((7.5, level['y']), (8, level['y']),
                               arrowstyle='->', lw=1.5, color='black')
        ax.add_patch(arrow)
        
        pyramid_features.append((8.5, level['y'], f"P{i+3}"))
    
    # Detection heads
    ax.text(11, 8.5, 'Detection\nHeads', ha='center', fontsize=11, fontweight='bold')
    
    for x, y, name in pyramid_features:
        # Arrow to detection
        arrow = FancyArrowPatch((x + 0.6, y), (10, y),
                               arrowstyle='->', lw=1.5, color='black')
        ax.add_patch(arrow)
        
        # Detection head
        box = FancyBboxPatch((10, y - 0.4), 1.5, 0.8,
                             boxstyle="round,pad=0.02",
                             facecolor='lightblue', edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(10.75, y, f'{name}\nHead', ha='center', va='center', fontsize=9)
    
    # Legend
    legend_y = 1.5
    ax.text(0.5, legend_y + 0.5, 'Key Benefits:', fontsize=11, fontweight='bold')
    ax.text(0.5, legend_y, '• High resolution (P3) + Strong semantics (from P5)', fontsize=9)
    ax.text(0.5, legend_y - 0.4, '• Multi-scale detection (P3→P5 for small→large)', fontsize=9)
    ax.text(0.5, legend_y - 0.8, '• Unified 256 channels across all levels', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('fpn_architecture.pdf', dpi=200, bbox_inches='tight')
    plt.savefig('fpn_architecture.png', dpi=200, bbox_inches='tight')
    print("Saved: fpn_architecture.pdf/.png")
    plt.show()


def demonstrate_fpn_features():
    """Demonstrate FPN feature fusion"""
    print("="*70)
    print("FPN Demo: Multi-Scale Feature Fusion")
    print("="*70)
    
    # Create models
    backbone = SimpleBackbone()
    fpn = SimpleFPN(in_channels_list=[512, 1024, 2048], out_channels=256)
    
    # Dummy input
    x = torch.randn(1, 3, 224, 224)
    
    print("\n🔍 Input Image:")
    print(f"  Shape: {x.shape}")
    print(f"  Size: 224×224 pixels")
    
    # Extract backbone features
    print("\n🏗️  Backbone Feature Extraction:")
    features = backbone(x)
    
    for name, feat in features.items():
        stride = 8 if name == 'C3' else (16 if name == 'C4' else 32)
        print(f"  {name}: {feat.shape} (stride {stride})")
    
    # Apply FPN
    pyramid = fpn(features)
    
    print("\n✅ FPN Output - Unified Multi-Scale Features:")
    for i, feat in enumerate(pyramid):
        level_name = f"P{i+3}"
        stride = 8 * (2 ** i)
        print(f"  {level_name}: {feat.shape} (stride {stride})")
    
    print("\n💡 Key Insight:")
    print("  All pyramid levels have 256 channels (unified!)")
    print("  P3: High resolution (28×28) + Strong semantics (from top-down)")
    print("  P4: Medium resolution (14×14) + Strong semantics")
    print("  P5: Low resolution (7×7) + Strong semantics")
    
    print("\n📊 What Each Level Detects:")
    print("  P3 (stride 8):  Small objects (8-64 pixels)")
    print("  P4 (stride 16): Medium objects (64-128 pixels)")
    print("  P5 (stride 32): Large objects (128+ pixels)")
    
    return backbone, fpn, pyramid


def compare_with_without_fpn():
    """Compare detection with and without FPN"""
    print("\n" + "="*70)
    print("Comparison: With vs Without FPN")
    print("="*70)
    
    print("\n❌ Without FPN (Single-Scale):")
    print("  • Use only C5 (stride 32, 7×7 resolution)")
    print("  • Problem: Small objects too small in 7×7 feature map")
    print("  • Result: Poor small object detection")
    
    print("\n✅ With FPN (Multi-Scale):")
    print("  • Use P3, P4, P5 (strides 8, 16, 32)")
    print("  • P3 has 28×28 resolution → detects small objects")
    print("  • All levels have strong semantics (from top-down)")
    print("  • Result: Excellent multi-scale detection")
    
    print("\n📈 Performance Gain:")
    print("  • Small objects AP: +8.0% improvement")
    print("  • Overall AP: +2-3% improvement")
    print("  • Key: Top-down pathway brings semantics to high-res features")


def demo_fpn_complete():
    """Complete FPN demonstration"""
    print("\n" + "="*70)
    print("COMPLETE FPN DEMONSTRATION")
    print("="*70)
    
    # 1. Visualize architecture
    print("\n1. Generating FPN architecture diagram...")
    visualize_fpn_architecture()
    
    # 2. Demonstrate feature fusion
    print("\n2. Demonstrating feature fusion...")
    backbone, fpn, pyramid = demonstrate_fpn_features()
    
    # 3. Compare with/without FPN
    compare_with_without_fpn()
    
    print("\n" + "="*70)
    print("FPN Demo Complete!")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. FPN combines high resolution + strong semantics")
    print("2. Top-down pathway brings semantic info to shallow layers")
    print("3. Lateral connections preserve spatial details")
    print("4. Multi-scale detection without image pyramids")
    print("5. Foundation for modern detectors (RetinaNet, Mask R-CNN, etc.)")


if __name__ == "__main__":
    demo_fpn_complete()