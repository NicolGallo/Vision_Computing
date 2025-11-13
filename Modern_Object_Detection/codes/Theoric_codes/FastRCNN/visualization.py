"""
Visualization script to compare R-CNN vs Fast R-CNN
Creates figures for lecture slides
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def visualize_roi_pooling():
    """Visualize RoI pooling operation"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Panel 1: Feature map with RoI
    ax = axes[0]
    ax.set_title('1. Feature Map + RoI', fontsize=14, fontweight='bold')
    
    # Draw feature map
    feat_map = np.random.rand(16, 16)
    ax.imshow(feat_map, cmap='viridis', alpha=0.6)
    
    # Draw RoI
    roi_rect = patches.Rectangle((3, 4), 9, 7, linewidth=3, 
                                 edgecolor='red', facecolor='none')
    ax.add_patch(roi_rect)
    ax.text(7.5, 2, 'RoI (variable size)', ha='center', fontsize=11,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7))
    
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(15.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Feature Map (H × W × D)', fontsize=11)
    
    # Panel 2: Grid division
    ax = axes[1]
    ax.set_title('2. Divide into 7×7 Grid', fontsize=14, fontweight='bold')
    
    # Zoomed RoI
    ax.set_xlim(0, 7)
    ax.set_ylim(7, 0)
    
    # Draw grid
    for i in range(8):
        ax.axhline(i, color='blue', linewidth=1)
        ax.axvline(i, color='blue', linewidth=1)
    
    # Highlight one bin
    highlight = patches.Rectangle((2, 3), 1, 1, linewidth=2,
                                 edgecolor='orange', facecolor='yellow', alpha=0.5)
    ax.add_patch(highlight)
    ax.text(2.5, 2.5, 'Bin\n(max pool)', ha='center', va='center', fontsize=10,
           fontweight='bold')
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('P × P bins (e.g., 7×7)', fontsize=11)
    
    # Panel 3: Output
    ax = axes[2]
    ax.set_title('3. Fixed-Size Output', fontsize=14, fontweight='bold')
    
    # Draw output grid
    output = np.random.rand(7, 7)
    im = ax.imshow(output, cmap='plasma')
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Output: 7×7×D (fixed size)', fontsize=11)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('roi_pooling_visualization.pdf', dpi=200, bbox_inches='tight')
    plt.savefig('roi_pooling_visualization.png', dpi=200, bbox_inches='tight')
    print("Saved: roi_pooling_visualization.pdf/.png")
    plt.show()


def visualize_architecture_comparison():
    """Compare R-CNN vs Fast R-CNN architectures"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # R-CNN
    ax = axes[0]
    ax.set_title('R-CNN: Sequential Processing', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 3)
    ax.axis('off')
    
    # Image
    img_box = FancyBboxPatch((0.5, 1), 1.5, 1, boxstyle="round,pad=0.05",
                            facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(img_box)
    ax.text(1.25, 1.5, 'Image', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Proposals
    prop_box = FancyBboxPatch((2.5, 1), 1.5, 1, boxstyle="round,pad=0.05",
                             facecolor='lightyellow', edgecolor='black', linewidth=2)
    ax.add_patch(prop_box)
    ax.text(3.25, 1.5, '~2000\nproposals', ha='center', va='center', fontsize=9)
    
    # CNN (repeated)
    for i in range(3):
        cnn_box = FancyBboxPatch((4.5 + i*0.3, 1 + i*0.15), 1.2, 0.8, 
                                boxstyle="round,pad=0.05",
                                facecolor='lightcoral', edgecolor='black', linewidth=1.5)
        ax.add_patch(cnn_box)
    ax.text(5.1, 1.9, 'CNN\n×2000', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # SVM
    svm_box = FancyBboxPatch((7, 1), 1.5, 1, boxstyle="round,pad=0.05",
                            facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(svm_box)
    ax.text(7.75, 1.5, 'SVM', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Bbox
    bbox_box = FancyBboxPatch((9, 1), 1.5, 1, boxstyle="round,pad=0.05",
                             facecolor='plum', edgecolor='black', linewidth=2)
    ax.add_patch(bbox_box)
    ax.text(9.75, 1.5, 'Bbox\nReg', ha='center', va='center', fontsize=10)
    
    # Output
    out_box = FancyBboxPatch((11, 1), 1.5, 1, boxstyle="round,pad=0.05",
                            facecolor='lightgray', edgecolor='black', linewidth=2)
    ax.add_patch(out_box)
    ax.text(11.75, 1.5, 'Detections', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows
    for x in [2, 4, 6.5, 8.5, 10.5]:
        arrow = FancyArrowPatch((x, 1.5), (x+0.4, 1.5),
                               arrowstyle='->', lw=2, color='black')
        ax.add_patch(arrow)
    
    # Timing
    ax.text(7, 0.3, '⏱ ~47 seconds per image', ha='center', fontsize=12,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.3))
    
    # Fast R-CNN
    ax = axes[1]
    ax.set_title('Fast R-CNN: Shared Features', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 3)
    ax.axis('off')
    
    # Image
    img_box = FancyBboxPatch((0.5, 1), 1.5, 1, boxstyle="round,pad=0.05",
                            facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(img_box)
    ax.text(1.25, 1.5, 'Image', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # CNN (single!)
    cnn_box = FancyBboxPatch((2.5, 1), 1.8, 1, boxstyle="round,pad=0.05",
                            facecolor='lightcoral', edgecolor='black', linewidth=2)
    ax.add_patch(cnn_box)
    ax.text(3.4, 1.5, 'CNN\n×1', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(3.4, 0.5, '(Shared!)', ha='center', fontsize=9, style='italic',
           color='red', fontweight='bold')
    
    # RoI Pool
    roi_box = FancyBboxPatch((4.8, 1), 1.5, 1, boxstyle="round,pad=0.05",
                            facecolor='lightyellow', edgecolor='black', linewidth=2)
    ax.add_patch(roi_box)
    ax.text(5.55, 1.5, 'RoI\nPool', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # FC
    fc_box = FancyBboxPatch((6.8, 1), 1.5, 1, boxstyle="round,pad=0.05",
                           facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(fc_box)
    ax.text(7.55, 1.5, 'FC\nLayers', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Two heads
    cls_box = FancyBboxPatch((9, 1.6), 1.3, 0.6, boxstyle="round,pad=0.05",
                            facecolor='plum', edgecolor='black', linewidth=2)
    ax.add_patch(cls_box)
    ax.text(9.65, 1.9, 'Class', ha='center', va='center', fontsize=9)
    
    bbox_box = FancyBboxPatch((9, 0.8), 1.3, 0.6, boxstyle="round,pad=0.05",
                             facecolor='plum', edgecolor='black', linewidth=2)
    ax.add_patch(bbox_box)
    ax.text(9.65, 1.1, 'Bbox', ha='center', va='center', fontsize=9)
    
    # Output
    out_box = FancyBboxPatch((11, 1), 1.5, 1, boxstyle="round,pad=0.05",
                            facecolor='lightgray', edgecolor='black', linewidth=2)
    ax.add_patch(out_box)
    ax.text(11.75, 1.5, 'Detections', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows
    for x in [2, 4.3, 6.3]:
        arrow = FancyArrowPatch((x, 1.5), (x+0.4, 1.5),
                               arrowstyle='->', lw=2, color='black')
        ax.add_patch(arrow)
    
    # Two arrows from FC
    arrow1 = FancyArrowPatch((8.3, 1.5), (9, 1.9),
                            arrowstyle='->', lw=2, color='black')
    ax.add_patch(arrow1)
    arrow2 = FancyArrowPatch((8.3, 1.5), (9, 1.1),
                            arrowstyle='->', lw=2, color='black')
    ax.add_patch(arrow2)
    
    # Arrows to output
    arrow3 = FancyArrowPatch((10.3, 1.9), (11, 1.6),
                            arrowstyle='->', lw=2, color='black')
    ax.add_patch(arrow3)
    arrow4 = FancyArrowPatch((10.3, 1.1), (11, 1.4),
                            arrowstyle='->', lw=2, color='black')
    ax.add_patch(arrow4)
    
    # Timing
    ax.text(7, 0.3, '⏱ ~0.3 seconds per image (150× faster!)', ha='center', fontsize=12,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='green', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('architecture_comparison.pdf', dpi=200, bbox_inches='tight')
    plt.savefig('architecture_comparison.png', dpi=200, bbox_inches='tight')
    print("Saved: architecture_comparison.pdf/.png")
    plt.show()


def visualize_smooth_l1_loss():
    """Visualize smooth L1 loss function"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.linspace(-3, 3, 1000)
    
    # Panel 1: Loss functions
    ax = axes[0]
    
    # L2 loss
    l2 = 0.5 * x**2
    ax.plot(x, l2, 'b-', linewidth=2.5, label='L2 loss (0.5x²)')
    
    # L1 loss
    l1 = np.abs(x)
    ax.plot(x, l1, 'r-', linewidth=2.5, label='L1 loss (|x|)')
    
    # Smooth L1
    smooth_l1 = np.where(np.abs(x) < 1, 0.5 * x**2, np.abs(x) - 0.5)
    ax.plot(x, smooth_l1, 'g-', linewidth=3, label='Smooth L1', linestyle='--')
    
    ax.set_xlabel('Prediction Error (x)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=13, fontweight='bold')
    ax.set_title('Loss Function Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, loc='upper center')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 4)
    
    # Highlight smooth transition
    ax.axvline(-1, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    ax.axvline(1, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    ax.text(0, 3.5, 'Quadratic\n(|x| < 1)', ha='center', fontsize=11,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    ax.text(-2, 3.5, 'Linear\n(|x| ≥ 1)', ha='center', fontsize=11,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.5))
    
    # Panel 2: Gradients
    ax = axes[1]
    
    # Gradients
    grad_l2 = x
    grad_l1 = np.sign(x)
    grad_smooth = np.where(np.abs(x) < 1, x, np.sign(x))
    
    ax.plot(x, grad_l2, 'b-', linewidth=2.5, label='∂L2/∂x = x')
    ax.plot(x, grad_l1, 'r-', linewidth=2.5, label='∂L1/∂x = sign(x)')
    ax.plot(x, grad_smooth, 'g-', linewidth=3, label='∂(Smooth L1)/∂x', linestyle='--')
    
    ax.set_xlabel('Prediction Error (x)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Gradient', fontsize=13, fontweight='bold')
    ax.set_title('Gradient Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    
    # Annotations
    ax.text(0, -2.5, 'Smooth L1 gradient:\n• Quadratic when |x| < 1 (no outlier sensitivity)\n• Linear when |x| ≥ 1 (stable gradients)',
           ha='center', fontsize=10,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('smooth_l1_visualization.pdf', dpi=200, bbox_inches='tight')
    plt.savefig('smooth_l1_visualization.png', dpi=200, bbox_inches='tight')
    print("Saved: smooth_l1_visualization.pdf/.png")
    plt.show()


if __name__ == "__main__":
    print("Generating Fast R-CNN visualizations...")
    print("-" * 60)
    
    print("\n1. RoI Pooling Operation...")
    visualize_roi_pooling()
    
    print("\n2. Architecture Comparison...")
    visualize_architecture_comparison()
    
    print("\n3. Smooth L1 Loss...")
    visualize_smooth_l1_loss()
    
    print("\n" + "=" * 60)
    print("All visualizations saved!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - roi_pooling_visualization.pdf/.png")
    print("  - architecture_comparison.pdf/.png")
    print("  - smooth_l1_visualization.pdf/.png")
    print("\nUse these in your lecture slides!")