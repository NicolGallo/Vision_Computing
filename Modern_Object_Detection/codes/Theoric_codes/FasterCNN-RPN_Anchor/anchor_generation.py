"""
Standalone Anchor Generation Demo
Educational demonstration of how anchors are generated in Faster R-CNN
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
#import torch


class AnchorGenerator:
    """
    Generate anchors at multiple scales and aspect ratios
    
    This is the foundation of RPN in Faster R-CNN
    """
    
    def __init__(self, scales=[8, 16, 32], aspect_ratios=[0.5, 1.0, 2.0], stride=16):
        """
        Args:
            scales: Anchor scales in pixels (relative to stride)
            aspect_ratios: Anchor aspect ratios (height/width)
            stride: Feature map stride (how much we downsample)
        """
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.stride = stride
        self.num_anchors = len(scales) * len(aspect_ratios)
    
    def generate_base_anchors(self):
        """
        Generate base anchor templates centered at (0, 0)
        
        Returns:
            base_anchors: [num_anchors, 4] in (x1, y1, x2, y2) format
        """
        base_anchors = []
        
        for scale in self.scales:
            for ratio in self.aspect_ratios:
                # Compute anchor dimensions
                area = scale ** 2
                w = np.sqrt(area / ratio)
                h = w * ratio
                
                # Center at origin, create box
                x1 = -w / 2
                y1 = -h / 2
                x2 = w / 2
                y2 = h / 2
                
                base_anchors.append([x1, y1, x2, y2])
        
        return np.array(base_anchors)
    
    def generate_anchors(self, feature_map_size, image_size):
        """
        Generate all anchors for a feature map
        
        Args:
            feature_map_size: (H_feat, W_feat) - size of feature map
            image_size: (H_img, W_img) - size of input image
        
        Returns:
            anchors: [H_feat * W_feat * num_anchors, 4] - all anchors
        """
        H_feat, W_feat = feature_map_size
        
        # Generate base anchors
        base_anchors = self.generate_base_anchors()
        
        all_anchors = []
        
        # Tile anchors across feature map
        for i in range(H_feat):
            for j in range(W_feat):
                # Compute center in image coordinates
                center_x = j * self.stride + self.stride / 2
                center_y = i * self.stride + self.stride / 2
                
                # Shift base anchors to this location
                for base_anchor in base_anchors:
                    x1, y1, x2, y2 = base_anchor
                    anchor = [
                        center_x + x1,
                        center_y + y1,
                        center_x + x2,
                        center_y + y2
                    ]
                    all_anchors.append(anchor)
        
        anchors = np.array(all_anchors)
        
        # Clip to image boundaries
        anchors[:, 0] = np.clip(anchors[:, 0], 0, image_size[1])
        anchors[:, 1] = np.clip(anchors[:, 1], 0, image_size[0])
        anchors[:, 2] = np.clip(anchors[:, 2], 0, image_size[1])
        anchors[:, 3] = np.clip(anchors[:, 3], 0, image_size[0])
        
        return anchors
    
    def visualize_base_anchors(self):
        """Visualize base anchor templates"""
        base_anchors = self.generate_base_anchors()
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(base_anchors)))
        
        # Draw all base anchors centered at origin
        for i, (anchor, color) in enumerate(zip(base_anchors, colors)):
            x1, y1, x2, y2 = anchor
            w = x2 - x1
            h = y2 - y1
            
            rect = Rectangle((x1, y1), w, h, linewidth=2,
                           edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # Label
            scale_idx = i // len(self.aspect_ratios)
            ratio_idx = i % len(self.aspect_ratios)
            label = f'S={self.scales[scale_idx]}, R={self.aspect_ratios[ratio_idx]:.1f}'
            ax.text(x2 + 2, y1, label, fontsize=9, color=color, fontweight='bold')
        
        # Draw center point
        ax.plot(0, 0, 'r*', markersize=20, label='Anchor center')
        
        ax.set_xlim(-60, 60)
        ax.set_ylim(-60, 60)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=1)
        ax.axvline(x=0, color='k', linewidth=1)
        ax.legend(loc='upper right')
        ax.set_title(f'Base Anchors: {len(self.scales)} scales × {len(self.aspect_ratios)} ratios = {self.num_anchors} anchors per location',
                    fontsize=13, fontweight='bold')
        ax.set_xlabel('Width (pixels)', fontsize=12)
        ax.set_ylabel('Height (pixels)', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('base_anchors.pdf', dpi=200, bbox_inches='tight')
        plt.savefig('base_anchors.png', dpi=200, bbox_inches='tight')
        print("Saved: base_anchors.pdf/.png")
        plt.show()
    
    def visualize_anchors_on_image(self, image_size=(224, 224), feature_map_size=(14, 14)):
        """Visualize anchors tiled across an image"""
        anchors = self.generate_anchors(feature_map_size, image_size)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Panel 1: All anchors at one location
        ax = axes[0]
        ax.set_xlim(0, image_size[1])
        ax.set_ylim(image_size[0], 0)
        
        # Draw image background
        ax.add_patch(Rectangle((0, 0), image_size[1], image_size[0],
                              facecolor='lightgray', alpha=0.3))
        
        # Pick center location
        center_i = feature_map_size[0] // 2
        center_j = feature_map_size[1] // 2
        start_idx = (center_i * feature_map_size[1] + center_j) * self.num_anchors
        
        colors = plt.cm.Set3(np.linspace(0, 1, self.num_anchors))
        
        for i in range(self.num_anchors):
            anchor = anchors[start_idx + i]
            x1, y1, x2, y2 = anchor
            w = x2 - x1
            h = y2 - y1
            
            rect = Rectangle((x1, y1), w, h, linewidth=2,
                           edgecolor=colors[i], facecolor='none')
            ax.add_patch(rect)
        
        # Mark center
        center_x = center_j * self.stride + self.stride / 2
        center_y = center_i * self.stride + self.stride / 2
        ax.plot(center_x, center_y, 'r*', markersize=15)
        
        ax.set_title(f'{self.num_anchors} Anchors at One Location', fontsize=13, fontweight='bold')
        ax.set_xlabel('X (pixels)', fontsize=11)
        ax.set_ylabel('Y (pixels)', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Sample of anchors across image
        ax = axes[1]
        ax.set_xlim(0, image_size[1])
        ax.set_ylim(image_size[0], 0)
        
        # Draw image background
        ax.add_patch(Rectangle((0, 0), image_size[1], image_size[0],
                              facecolor='lightgray', alpha=0.3))
        
        # Draw grid of anchor centers
        for i in range(feature_map_size[0]):
            for j in range(feature_map_size[1]):
                center_x = j * self.stride + self.stride / 2
                center_y = i * self.stride + self.stride / 2
                ax.plot(center_x, center_y, 'b.', markersize=6)
        
        # Draw sample anchors (one per location, largest scale, ratio=1)
        sample_anchor_idx = len(self.aspect_ratios) * (len(self.scales) - 1) + 1  # Largest scale, ratio=1
        
        for i in range(0, feature_map_size[0], 2):  # Every other location
            for j in range(0, feature_map_size[1], 2):
                anchor_idx = (i * feature_map_size[1] + j) * self.num_anchors + sample_anchor_idx
                if anchor_idx < len(anchors):
                    anchor = anchors[anchor_idx]
                    x1, y1, x2, y2 = anchor
                    w = x2 - x1
                    h = y2 - y1
                    
                    rect = Rectangle((x1, y1), w, h, linewidth=1,
                                   edgecolor='red', facecolor='none', alpha=0.5)
                    ax.add_patch(rect)
        
        ax.set_title(f'Anchors Tiled Across Image\n({feature_map_size[0]}×{feature_map_size[1]} locations, stride={self.stride})',
                    fontsize=13, fontweight='bold')
        ax.set_xlabel('X (pixels)', fontsize=11)
        ax.set_ylabel('Y (pixels)', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('anchors_on_image.pdf', dpi=200, bbox_inches='tight')
        plt.savefig('anchors_on_image.png', dpi=200, bbox_inches='tight')
        print("Saved: anchors_on_image.pdf/.png")
        plt.show()
        
        # Print statistics
        print(f"\nAnchor Statistics:")
        print(f"  Feature map: {feature_map_size[0]}×{feature_map_size[1]}")
        print(f"  Anchors per location: {self.num_anchors}")
        print(f"  Total anchors: {len(anchors)}")
        print(f"  Stride: {self.stride}")
        print(f"  Scales: {self.scales}")
        print(f"  Aspect ratios: {self.aspect_ratios}")


def demo_anchor_properties():
    """Demonstrate key properties of anchors"""
    
    print("="*70)
    print("Anchor Generation Demo - Understanding Faster R-CNN Anchors")
    print("="*70)
    
    # Standard Faster R-CNN configuration
    anchor_gen = AnchorGenerator(
        scales=[8, 16, 32],           # 3 scales
        aspect_ratios=[0.5, 1.0, 2.0], # 3 aspect ratios
        stride=16                      # Typical for VGG/ResNet
    )
    
    print("\n1. Base Anchor Generation")
    print("-" * 70)
    base_anchors = anchor_gen.generate_base_anchors()
    print(f"Generated {len(base_anchors)} base anchors:")
    print(f"{'Scale':<8} {'Ratio':<8} {'Width':<10} {'Height':<10} {'Area':<10}")
    print("-" * 70)
    
    for i, anchor in enumerate(base_anchors):
        scale_idx = i // len(anchor_gen.aspect_ratios)
        ratio_idx = i % len(anchor_gen.aspect_ratios)
        scale = anchor_gen.scales[scale_idx]
        ratio = anchor_gen.aspect_ratios[ratio_idx]
        
        x1, y1, x2, y2 = anchor
        w = x2 - x1
        h = y2 - y1
        area = w * h
        
        print(f"{scale:<8} {ratio:<8.1f} {w:<10.1f} {h:<10.1f} {area:<10.1f}")
    
    print("\n2. Anchor Properties")
    print("-" * 70)
    print(f"• Covers multiple scales: {anchor_gen.scales}")
    print(f"• Covers multiple shapes: {anchor_gen.aspect_ratios}")
    print(f"• Can detect objects of various sizes and shapes")
    print(f"• Each location has {anchor_gen.num_anchors} candidate boxes")
    
    print("\n3. Full Image Anchors")
    print("-" * 70)
    image_size = (224, 224)
    feature_map_size = (14, 14)  # 224/16 = 14
    
    anchors = anchor_gen.generate_anchors(feature_map_size, image_size)
    print(f"Image size: {image_size}")
    print(f"Feature map: {feature_map_size} (stride={anchor_gen.stride})")
    print(f"Total locations: {feature_map_size[0] * feature_map_size[1]}")
    print(f"Anchors per location: {anchor_gen.num_anchors}")
    print(f"Total anchors: {len(anchors)}")
    
    print("\n4. Visualizing Anchors...")
    print("-" * 70)
    
    # Visualize base anchors
    print("Creating base anchor visualization...")
    anchor_gen.visualize_base_anchors()
    
    # Visualize anchors on image
    print("Creating full image anchor visualization...")
    anchor_gen.visualize_anchors_on_image(image_size, feature_map_size)
    
    print("\n" + "="*70)
    print("Why Anchors Matter for Faster R-CNN:")
    print("="*70)
    print("""
1. MULTI-SCALE DETECTION:
   - Different scales (8, 16, 32) detect small, medium, large objects
   - No need for image pyramid!

2. MULTI-SHAPE DETECTION:
   - Different ratios (0.5, 1.0, 2.0) detect tall, square, wide objects
   - Handles diverse object shapes

3. DENSE COVERAGE:
   - Anchors at every spatial location
   - Ensures no object is missed
   - RPN learns which anchors likely contain objects

4. EFFICIENCY:
   - Pre-defined templates
   - RPN just predicts: "Is there an object?" and "How to adjust?"
   - Much faster than sliding windows or selective search
    """)
    
    print("\n" + "="*70)
    print("Demo Complete!")
    print("="*70)


if __name__ == "__main__":
    demo_anchor_properties()