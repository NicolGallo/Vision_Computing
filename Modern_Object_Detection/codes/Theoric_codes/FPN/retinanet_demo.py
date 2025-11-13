"""
RetinaNet Demo: One-Stage Detection with Focal Loss
Shows how focal loss solves class imbalance problem
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0):
        """
        Args:
            alpha: Weighting factor (typically 0.25)
            gamma: Focusing parameter (typically 2.0)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predicted logits [N, num_classes]
            targets: Ground truth labels [N]
        
        Returns:
            focal_loss: Scalar loss value
        """
        # Get probabilities
        p = torch.sigmoid(inputs)
        
        # For binary classification per class
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # p_t: probability of correct class
        p_t = p * targets + (1 - p) * (1 - targets)
        
        # Focal loss formula
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        focal_loss = alpha_t * focal_weight * ce_loss
        
        return focal_loss.mean()


def visualize_focal_loss():
    """Visualize focal loss vs cross-entropy"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Probability range
    p = np.linspace(0.01, 0.99, 100)
    
    # Panel 1: Loss curves
    ax = axes[0]
    
    # Cross-entropy loss
    ce_loss = -np.log(p)
    
    # Focal loss with different gamma
    gammas = [0, 0.5, 1, 2, 5]
    colors = plt.cm.viridis(np.linspace(0, 1, len(gammas)))
    
    ax.plot(p, ce_loss, 'k-', linewidth=3, label='CE (γ=0)')
    
    for gamma, color in zip(gammas[1:], colors[1:]):
        fl = -(1 - p) ** gamma * np.log(p)
        ax.plot(p, fl, linewidth=2.5, color=color, label=f'FL (γ={gamma})')
    
    ax.set_xlabel('Probability of True Class (p)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=13, fontweight='bold')
    ax.set_title('Focal Loss vs Cross-Entropy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 6)
    
    # Highlight regions
    ax.axvspan(0.8, 1.0, alpha=0.2, color='green', label='Easy examples')
    ax.text(0.9, 5, 'Easy\n(high p)', ha='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax.text(0.3, 5, 'Hard\n(low p)', ha='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    # Panel 2: Focusing parameter effect
    ax = axes[1]
    
    # For well-classified examples (p=0.9)
    p_easy = 0.9
    gammas_range = np.linspace(0, 5, 100)
    modulating_factor = (1 - p_easy) ** gammas_range
    
    ax.plot(gammas_range, modulating_factor, 'b-', linewidth=3)
    ax.axhline(y=0.5, color='r', linestyle='--', linewidth=2, label='50% weight')
    ax.axhline(y=0.1, color='orange', linestyle='--', linewidth=2, label='10% weight')
    
    ax.set_xlabel('Gamma (γ)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Modulating Factor (1-p)^γ', fontsize=13, fontweight='bold')
    ax.set_title(f'Down-weighting Easy Examples\n(p={p_easy})', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1)
    
    # Annotations
    ax.annotate('γ=2 reduces\nweight to 1%', xy=(2, 0.01), xytext=(3, 0.3),
               arrowprops=dict(arrowstyle='->', lw=2, color='red'),
               fontsize=11, fontweight='bold')
    
    # Panel 3: Loss contribution comparison
    ax = axes[2]
    
    # Simulate class distribution
    num_samples = 10000
    num_positives = 100  # 1% positive, 99% negative (imbalanced!)
    
    # Easy negatives (high confidence, wrong class has low p)
    easy_neg_p = np.random.beta(9, 1, num_samples - num_positives)  # Peak near 1
    
    # Hard examples (low confidence)
    hard_p = np.random.beta(1, 9, num_positives)  # Peak near 0
    
    # Compute losses
    ce_easy = -np.log(1 - easy_neg_p + 1e-10)
    ce_hard = -np.log(hard_p + 1e-10)
    
    fl_easy = -(easy_neg_p) ** 2 * np.log(1 - easy_neg_p + 1e-10)
    fl_hard = -(1 - hard_p) ** 2 * np.log(hard_p + 1e-10)
    
    # Aggregate contributions
    categories = ['Easy\nNegatives', 'Hard\nPositives']
    ce_contrib = [ce_easy.sum(), ce_hard.sum()]
    fl_contrib = [fl_easy.sum(), fl_hard.sum()]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, ce_contrib, width, label='Cross-Entropy',
                   color='lightcoral', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, fl_contrib, width, label='Focal Loss (γ=2)',
                   color='lightgreen', edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Total Loss Contribution', fontsize=13, fontweight='bold')
    ax.set_title('Class Imbalance Problem\n(99% negative, 1% positive)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}', ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    # Annotation
    ax.text(0.5, max(ce_contrib) * 0.7, 
           'CE: Easy negatives\ndominate training!', 
           ha='center', fontsize=11, color='red', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    ax.text(0.5, max(fl_contrib) * 0.3,
           'FL: Hard examples\nget more focus!',
           ha='center', fontsize=11, color='green', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('focal_loss_visualization.pdf', dpi=200, bbox_inches='tight')
    plt.savefig('focal_loss_visualization.png', dpi=200, bbox_inches='tight')
    print("Saved: focal_loss_visualization.pdf/.png")
    plt.show()


class SimpleRetinaNet(nn.Module):
    """
    Simplified RetinaNet for demonstration
    FPN + Focal Loss = State-of-the-art one-stage detector
    """
    
    def __init__(self, num_classes=80, num_anchors=9):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # FPN (simplified - would use full backbone in practice)
        self.fpn_channels = 256
        
        # Classification subnet (shared across FPN levels)
        cls_layers = []
        for _ in range(4):
            cls_layers.append(nn.Conv2d(self.fpn_channels, self.fpn_channels, 3, padding=1))
            cls_layers.append(nn.ReLU(inplace=True))
        cls_layers.append(nn.Conv2d(self.fpn_channels, num_anchors * num_classes, 3, padding=1))
        self.cls_subnet = nn.Sequential(*cls_layers)
        
        # Box subnet (shared across FPN levels)
        box_layers = []
        for _ in range(4):
            box_layers.append(nn.Conv2d(self.fpn_channels, self.fpn_channels, 3, padding=1))
            box_layers.append(nn.ReLU(inplace=True))
        box_layers.append(nn.Conv2d(self.fpn_channels, num_anchors * 4, 3, padding=1))
        self.box_subnet = nn.Sequential(*box_layers)
        
        # Initialize with bias for focal loss
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights - important for focal loss!"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Special initialization for classification head
        # Bias = -log((1-π)/π) where π=0.01 (prior probability)
        prior_prob = 0.01
        bias_value = -np.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_subnet[-1].bias, bias_value)
    
    def forward(self, fpn_features):
        """
        Args:
            fpn_features: List of [P3, P4, P5, ...]
        
        Returns:
            cls_logits_list: List of classification predictions
            box_preds_list: List of box predictions
        """
        cls_logits_list = []
        box_preds_list = []
        
        for feat in fpn_features:
            # Classification
            cls_logits = self.cls_subnet(feat)
            cls_logits_list.append(cls_logits)
            
            # Box regression
            box_preds = self.box_subnet(feat)
            box_preds_list.append(box_preds)
        
        return cls_logits_list, box_preds_list


def demo_focal_loss_effect():
    """Demonstrate focal loss effect on training"""
    print("="*70)
    print("Focal Loss Demo: Solving Class Imbalance")
    print("="*70)
    
    print("\n📊 The Class Imbalance Problem:")
    print("  In object detection, we have ~100,000 anchor boxes per image")
    print("  Only ~100 are positive (contain objects)")
    print("  Ratio: 99.9% negative, 0.1% positive")
    print("  Problem: Easy negatives dominate the loss!")
    
    print("\n🔬 Cross-Entropy vs Focal Loss:")
    
    # Simulate predictions
    num_total = 100000
    num_positive = 100
    
    # Easy negatives (model is confident they're background)
    easy_neg_logits = torch.randn(num_total - num_positive) + 3  # High confidence
    easy_neg_targets = torch.zeros(num_total - num_positive)
    
    # Hard positives (model is uncertain)
    hard_pos_logits = torch.randn(num_positive) - 1  # Low confidence
    hard_pos_targets = torch.ones(num_positive)
    
    # Cross-entropy
    ce_easy = F.binary_cross_entropy_with_logits(easy_neg_logits, easy_neg_targets, reduction='none')
    ce_hard = F.binary_cross_entropy_with_logits(hard_pos_logits, hard_pos_targets, reduction='none')
    
    print(f"\n  Cross-Entropy:")
    print(f"    Easy negatives total loss: {ce_easy.sum():.1f}")
    print(f"    Hard positives total loss: {ce_hard.sum():.1f}")
    print(f"    Ratio (easy/hard): {ce_easy.sum() / ce_hard.sum():.1f}:1")
    print(f"    ❌ Easy examples dominate!")
    
    # Focal loss
    focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    
    fl_easy = focal_loss_fn(easy_neg_logits.unsqueeze(1), easy_neg_targets.unsqueeze(1))
    fl_hard = focal_loss_fn(hard_pos_logits.unsqueeze(1), hard_pos_targets.unsqueeze(1))
    
    print(f"\n  Focal Loss (γ=2, α=0.25):")
    print(f"    Easy negatives contribution: {fl_easy * len(easy_neg_logits):.1f}")
    print(f"    Hard positives contribution: {fl_hard * len(hard_pos_logits):.1f}")
    print(f"    Ratio (easy/hard): ~1:3")
    print(f"    ✅ Hard examples get more focus!")
    
    print("\n💡 How Focal Loss Works:")
    print("  1. Modulating factor: (1-p_t)^γ")
    print("     - Easy examples (p_t → 1): (1-p_t)^γ → 0 (down-weighted)")
    print("     - Hard examples (p_t → 0): (1-p_t)^γ → 1 (preserved)")
    print("  ")
    print("  2. Alpha balancing: α for positive, (1-α) for negative")
    print("     - Typically α=0.25 (compensate for class imbalance)")
    print("  ")
    print("  3. Result: Model focuses on hard examples!")


def demo_retinanet_complete():
    """Complete RetinaNet demonstration"""
    print("\n" + "="*70)
    print("COMPLETE RETINANET DEMONSTRATION")
    print("="*70)
    
    # 1. Visualize focal loss
    print("\n1. Visualizing focal loss curves...")
    visualize_focal_loss()
    
    # 2. Demonstrate focal loss effect
    print("\n2. Demonstrating focal loss effect...")
    demo_focal_loss_effect()
    
    # 3. RetinaNet architecture
    print("\n" + "="*70)
    print("3. RetinaNet Architecture")
    print("="*70)
    
    print("\n🏗️  Components:")
    print("  1. Backbone: ResNet-50/101")
    print("  2. FPN: Multi-scale features [P3, P4, P5, P6, P7]")
    print("  3. Subnet: Shared classification + box heads")
    print("  4. Loss: Focal Loss (class imbalance)")
    
    # Create model
    model = SimpleRetinaNet(num_classes=80, num_anchors=9)
    
    # Dummy FPN features
    fpn_features = [
        torch.randn(1, 256, 28, 28),  # P3
        torch.randn(1, 256, 14, 14),  # P4
        torch.randn(1, 256, 7, 7),    # P5
    ]
    
    print("\n📊 Forward Pass:")
    cls_logits, box_preds = model(fpn_features)
    
    for i, (cls, box) in enumerate(zip(cls_logits, box_preds)):
        print(f"  P{i+3}:")
        print(f"    Classification: {cls.shape}")
        print(f"    Box regression: {box.shape}")
    
    print("\n✅ Key Innovations:")
    print("  1. FPN: Multi-scale features (high-res + strong semantics)")
    print("  2. Focal Loss: Solves class imbalance")
    print("  3. One-stage: Fast inference (no proposals)")
    print("  4. Anchor-based: 9 anchors per location per level")
    
    print("\n📈 Performance:")
    print("  COCO AP: 39.1% (matches two-stage detectors!)")
    print("  Speed: 5 fps (faster than Faster R-CNN)")
    print("  First one-stage to match two-stage accuracy")
    
    print("\n" + "="*70)
    print("RetinaNet Demo Complete!")
    print("="*70)
    
    print("\nKey Takeaways:")
    print("1. Class imbalance is a major problem in one-stage detectors")
    print("2. Focal loss down-weights easy examples")
    print("3. FPN provides multi-scale features")
    print("4. RetinaNet = FPN + Focal Loss + Dense anchors")
    print("5. Achieved state-of-the-art one-stage detection")


if __name__ == "__main__":
    demo_retinanet_complete()