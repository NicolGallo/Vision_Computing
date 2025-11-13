"""
Standalone Anchor Generation Demo (Enhanced)
Educational demonstration of how anchors are generated in Faster R-CNN,
and how the lattice relates to a YOLO SxS grid.
Adds:
  • YOLO grid overlay
  • GT→anchor IoU heatmap + best anchor highlight
  • Parameterization explainer (RPN deltas vs YOLO direct)
  • Micro-NMS on objectness ≈ IoU with GT
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle


# -----------------------------
# Utilities
# -----------------------------
def iou_xyxy(a, b):
    """
    Compute IoU between sets of boxes a[N,4], b[M,4] in (x1,y1,x2,y2).
    Returns [N,M].
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    N, M = a.shape[0], b.shape[0]
    ious = np.zeros((N, M), dtype=np.float32)

    for i in range(N):
        ax1, ay1, ax2, ay2 = a[i]
        aw = max(0.0, ax2 - ax1); ah = max(0.0, ay2 - ay1)
        aarea = aw * ah
        for j in range(M):
            bx1, by1, bx2, by2 = b[j]
            # intersection
            inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
            inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
            iw = max(0.0, inter_x2 - inter_x1)
            ih = max(0.0, inter_y2 - inter_y1)
            inter = iw * ih
            barea = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            union = aarea + barea - inter + 1e-6
            ious[i, j] = inter / union
    return ious


def nms_iou(boxes, scores, iou_thresh=0.5, topk=None):
    """
    Simple IoU-based NMS.
    boxes: [N,4], scores: [N]
    Returns indices to keep (list).
    """
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    x1 = boxes[:, 0]; y1 = boxes[:, 1]
    x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if topk is not None and len(keep) >= topk:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        iw = np.maximum(0.0, xx2 - xx1)
        ih = np.maximum(0.0, yy2 - yy1)
        inter = iw * ih
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        order = order[1:][iou <= iou_thresh]
    return keep


def draw_yolo_grid(ax, image_size=(224, 224), S=14, color='k', alpha=0.25, lw=0.8, mark_centers=True):
    H, W = image_size
    cell_w, cell_h = W / S, H / S
    # Vertical grid lines
    for j in range(S + 1):
        x = j * cell_w
        ax.plot([x, x], [0, H], color=color, alpha=alpha, linewidth=lw)
    # Horizontal grid lines
    for i in range(S + 1):
        y = i * cell_h
        ax.plot([0, W], [y, y], color=color, alpha=alpha, linewidth=lw)
    # Optional centers
    if mark_centers:
        for i in range(S):
            for j in range(S):
                cx = j * cell_w + cell_w / 2
                cy = i * cell_h + cell_h / 2
                ax.plot(cx, cy, 'k.', alpha=0.2, markersize=3)


def print_parametrization_explainer():
    print("\nParameterization:")
    print("RPN / anchor-based (per anchor a with center (ax,ay), size (aw,ah)):")
    print("  tx = (x - ax) / aw")
    print("  ty = (y - ay) / ah")
    print("  tw = log(w / aw)")
    print("  th = log(h / ah)")
    print("Decode:")
    print("  x = tx*aw + ax,  y = ty*ah + ay")
    print("  w = exp(tw)*aw,  h = exp(th)*ah")
    print("\nYOLOv1 (anchor-free, per cell (i,j), SxS grid):")
    print("  x = (j + sigmoid(t_x_cell)) / S")
    print("  y = (i + sigmoid(t_y_cell)) / S")
    print("  w = sigmoid(t_w),  h = sigmoid(t_h)   (relative to image)")
    print("Assignment difference:")
    print("  • RPN/YOLOv2+: GT → best-IoU anchor(s)")
    print("  • YOLOv1: one responsible predictor per positive cell")


# -----------------------------
# Anchor generator
# -----------------------------
class AnchorGenerator:
    """
    Generate anchors at multiple scales and aspect ratios
    (Foundation of RPN in Faster R-CNN)
    """

    def __init__(self, scales=[8, 16, 32], aspect_ratios=[0.5, 1.0, 2.0], stride=16):
        """
        Args:
            scales: Anchor scales in pixels (relative to stride)
            aspect_ratios: Anchor aspect ratios (height/width)
            stride: Feature-map stride (downsampling factor)
        """
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.stride = stride
        self.num_anchors = len(scales) * len(aspect_ratios)

    def generate_base_anchors(self):
        """
        Base anchor templates centered at (0,0), as (x1,y1,x2,y2).
        """
        base_anchors = []
        for scale in self.scales:
            for ratio in self.aspect_ratios:
                area = scale ** 2
                w = np.sqrt(area / ratio)
                h = w * ratio
                x1 = -w / 2; y1 = -h / 2
                x2 = +w / 2; y2 = +h / 2
                base_anchors.append([x1, y1, x2, y2])
        return np.array(base_anchors, dtype=np.float32)

    def generate_anchors(self, feature_map_size, image_size):
        """
        Tile anchors (shift base anchors to every lattice location).
        Returns [H_feat * W_feat * A, 4].
        """
        H_feat, W_feat = feature_map_size
        base_anchors = self.generate_base_anchors()
        A = base_anchors.shape[0]

        all_anchors = np.zeros((H_feat * W_feat * A, 4), dtype=np.float32)
        idx = 0
        for i in range(H_feat):
            for j in range(W_feat):
                cx = j * self.stride + self.stride / 2
                cy = i * self.stride + self.stride / 2
                shifts = base_anchors + np.array([cx, cy, cx, cy], dtype=np.float32)
                all_anchors[idx:idx + A] = shifts
                idx += A

        # Clip to image boundaries
        H_img, W_img = image_size
        all_anchors[:, 0] = np.clip(all_anchors[:, 0], 0, W_img)
        all_anchors[:, 1] = np.clip(all_anchors[:, 1], 0, H_img)
        all_anchors[:, 2] = np.clip(all_anchors[:, 2], 0, W_img)
        all_anchors[:, 3] = np.clip(all_anchors[:, 3], 0, H_img)
        return all_anchors

    # -----------------------------
    # Visualizations
    # -----------------------------
    def visualize_base_anchors(self):
        """Visualize base anchor templates at origin."""
        base_anchors = self.generate_base_anchors()
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        colors = plt.cm.Set3(np.linspace(0, 1, len(base_anchors)))

        for i, (anchor, color) in enumerate(zip(base_anchors, colors)):
            x1, y1, x2, y2 = anchor
            w = x2 - x1; h = y2 - y1
            rect = Rectangle((x1, y1), w, h, linewidth=2,
                             edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            scale_idx = i // len(self.aspect_ratios)
            ratio_idx = i % len(self.aspect_ratios)
            label = f'S={self.scales[scale_idx]}, R={self.aspect_ratios[ratio_idx]:.1f}'
            ax.text(x2 + 2, y1, label, fontsize=9, color=color, fontweight='bold')

        ax.plot(0, 0, 'r*', markersize=20, label='Anchor center')
        ax.set_xlim(-60, 60); ax.set_ylim(-60, 60); ax.set_aspect('equal')
        ax.grid(True, alpha=0.3); ax.axhline(0, color='k', linewidth=1); ax.axvline(0, color='k', linewidth=1)
        ax.legend(loc='upper right')
        ax.set_title(f'Base Anchors: {len(self.scales)} scales × {len(self.aspect_ratios)} ratios = {self.num_anchors} anchors/location',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Width (px)'); ax.set_ylabel('Height (px)')
        plt.tight_layout()
        plt.savefig('base_anchors.pdf', dpi=200, bbox_inches='tight')
        plt.savefig('base_anchors.png', dpi=200, bbox_inches='tight')
        print("Saved: base_anchors.pdf/.png")
        plt.show()

    def visualize_anchors_on_image(self, image_size=(224, 224), feature_map_size=(14, 14)):
        """Visualize anchors tiled across an image + YOLO grid overlay."""
        anchors = self.generate_anchors(feature_map_size, image_size)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Panel 1: All anchors at one center
        ax = axes[0]
        ax.set_xlim(0, image_size[1]); ax.set_ylim(image_size[0], 0)
        ax.add_patch(Rectangle((0, 0), image_size[1], image_size[0],
                               facecolor='lightgray', alpha=0.3))

        # Overlay YOLO grid (same lattice as feature map)
        draw_yolo_grid(ax, image_size=image_size, S=feature_map_size[0], color='k', alpha=0.25)

        Hf, Wf = feature_map_size
        A = self.num_anchors
        ci = Hf // 2; cj = Wf // 2
        start_idx = (ci * Wf + cj) * A
        colors = plt.cm.Set3(np.linspace(0, 1, A))
        for i in range(A):
            x1, y1, x2, y2 = anchors[start_idx + i]
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                             edgecolor=colors[i], facecolor='none')
            ax.add_patch(rect)

        cx = cj * self.stride + self.stride / 2
        cy = ci * self.stride + self.stride / 2
        ax.plot(cx, cy, 'r*', markersize=15)
        ax.set_title(f'{A} Anchors at One Lattice Location\n(stride={self.stride})', fontsize=13, fontweight='bold')
        ax.set_xlabel('X (px)'); ax.set_ylabel('Y (px)'); ax.grid(True, alpha=0.3)

        # Panel 2: Sample anchors across image
        ax = axes[1]
        ax.set_xlim(0, image_size[1]); ax.set_ylim(image_size[0], 0)
        ax.add_patch(Rectangle((0, 0), image_size[1], image_size[0],
                               facecolor='lightgray', alpha=0.3))
        draw_yolo_grid(ax, image_size=image_size, S=Hf, color='k', alpha=0.25)

        # Plot lattice centers
        for i in range(Hf):
            for j in range(Wf):
                cx = j * self.stride + self.stride / 2
                cy = i * self.stride + self.stride / 2
                ax.plot(cx, cy, 'b.', markersize=6, alpha=0.6)

        # Draw one sample anchor per 2×2 positions: largest scale, ratio=1
        sample_anchor_idx = len(self.aspect_ratios) * (len(self.scales) - 1) + 1  # largest scale, ratio=1
        for i in range(0, Hf, 2):
            for j in range(0, Wf, 2):
                idx = (i * Wf + j) * A + sample_anchor_idx
                if idx < len(anchors):
                    x1, y1, x2, y2 = anchors[idx]
                    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1,
                                           edgecolor='red', facecolor='none', alpha=0.5))

        ax.set_title(f'Anchors Tiled Across Image\n({Hf}×{Wf} positions, stride={self.stride})',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('X (px)'); ax.set_ylabel('Y (px)'); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('anchors_on_image.pdf', dpi=200, bbox_inches='tight')
        plt.savefig('anchors_on_image.png', dpi=200, bbox_inches='tight')
        print("Saved: anchors_on_image.pdf/.png")
        plt.show()

        # Stats
        print(f"\nAnchor Statistics:")
        print(f"  Feature map: {Hf}×{Wf}")
        print(f"  Anchors per location: {A}")
        print(f"  Total anchors: {len(anchors)}")
        print(f"  Stride: {self.stride}")
        print(f"  Scales: {self.scales}")
        print(f"  Aspect ratios: {self.aspect_ratios}")

    def visualize_assignment(self, image_size=(224, 224), feature_map_size=(14, 14),
                             gt_box=(60, 60, 160, 170), iou_vis_thresh=0.5):
        """
        Visualize GT→anchor assignment:
          • IoU heat at lattice centers (max IoU over anchors at each location)
          • Best anchor outlined
          • All anchors with IoU >= threshold (thin red)
        """
        anchors = self.generate_anchors(feature_map_size, image_size)
        ious = iou_xyxy([gt_box], anchors)[0]  # [num_anchors_total]
        Hf, Wf = feature_map_size
        A = self.num_anchors

        # Per-location max IoU (over the A anchors)
        iou_loc = ious.reshape(Hf * Wf, A).max(axis=1).reshape(Hf, Wf)

        best_idx = int(np.argmax(ious))
        best_anchor = anchors[best_idx]

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.set_xlim(0, image_size[1]); ax.set_ylim(image_size[0], 0)
        ax.add_patch(Rectangle((0, 0), image_size[1], image_size[0], facecolor='lightgray', alpha=0.2))
        draw_yolo_grid(ax, image_size=image_size, S=Hf, color='k', alpha=0.25)

        # IoU heat as colored dots at centers
        for i in range(Hf):
            for j in range(Wf):
                cx = j * self.stride + self.stride / 2
                cy = i * self.stride + self.stride / 2
                c = plt.cm.viridis(iou_loc[i, j])
                ax.plot(cx, cy, 'o', color=c, markersize=6, alpha=0.95)

        # Draw GT
        x1, y1, x2, y2 = gt_box
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='magenta',
                               facecolor='none', linewidth=2))
        ax.text(x1, y1 - 5, 'GT', color='magenta', fontsize=10, fontweight='bold')

        # Best anchor
        bx1, by1, bx2, by2 = best_anchor
        ax.add_patch(Rectangle((bx1, by1), bx2 - bx1, by2 - by1, edgecolor='lime',
                               facecolor='none', linewidth=2))
        ax.text(bx1, by1 - 5, f'Best IoU={ious[best_idx]:.2f}', color='lime',
                fontsize=10, fontweight='bold')

        # All anchors above threshold (thin red)
        high = np.where(ious >= iou_vis_thresh)[0]
        for idx in high[:300]:  # cap to keep the plot readable
            x1, y1, x2, y2 = anchors[idx]
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red',
                                   facecolor='none', linewidth=1, alpha=0.35))

        ax.set_title('GT→Anchor Assignment (IoU heat at centers; best anchor in lime)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('anchor_assignment.png', dpi=200, bbox_inches='tight')
        print("Saved: anchor_assignment.png")
        plt.show()

        return anchors, ious

    def micro_nms_demo(self, anchors, scores, image_size=(224, 224), iou_thresh=0.5, topk=200):
        """
        Run micro-NMS on provided scores (e.g., IoU with a GT) and visualize reduction.
        """
        keep = nms_iou(anchors, scores, iou_thresh=iou_thresh, topk=topk)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        titles = [f'Pre-NMS (N={len(anchors)})', f'Post-NMS (kept={len(keep)})']

        for ax, show_keep in zip(axes, [False, True]):
            ax.set_xlim(0, image_size[1]); ax.set_ylim(image_size[0], 0)
            ax.add_patch(Rectangle((0, 0), image_size[1], image_size[0],
                                   facecolor='lightgray', alpha=0.2))
            ax.grid(True, alpha=0.3)
            if show_keep:
                sel = keep
                color = 'lime'
                alpha = 0.9
                lw = 1.5
            else:
                # Show top-N by score for visibility
                order = scores.argsort()[::-1]
                sel = order[:400]  # cap for plotting
                color = 'red'
                alpha = 0.25
                lw = 1.0
            for idx in sel:
                x1, y1, x2, y2 = anchors[idx]
                ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       edgecolor=color, facecolor='none',
                                       linewidth=lw, alpha=alpha))
            ax.set_title(titles[0] if not show_keep else titles[1])

        plt.tight_layout()
        outname = f'micro_nms_iou{int(iou_thresh*100)}.png'
        plt.savefig(outname, dpi=200, bbox_inches='tight')
        print(f"Saved: {outname}")
        plt.show()

        # Print quick stats
        print("\nMicro-NMS stats:")
        print(f"  Total boxes in: {len(anchors)}")
        print(f"  Kept after NMS: {len(keep)} (iou_thresh={iou_thresh}, topk={topk})")
        return keep


# -----------------------------
# Demo flow
# -----------------------------
def demo_anchor_properties():
    """Demonstrate key properties of anchors + YOLO lattice connection + micro-NMS."""
    print("=" * 70)
    print("Anchor Generation Demo - Understanding Faster R-CNN Anchors")
    print("=" * 70)

    # Standard Faster R-CNN-ish configuration
    anchor_gen = AnchorGenerator(
        scales=[8, 16, 32],            # 3 scales
        aspect_ratios=[0.5, 1.0, 2.0], # 3 aspect ratios
        stride=16                       # Typical for VGG/ResNet C4
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
        w = x2 - x1; h = y2 - y1; area = w * h
        print(f"{scale:<8} {ratio:<8.1f} {w:<10.1f} {h:<10.1f} {area:<10.1f}")

    print("\n2. Anchor Properties and Lattice (YOLO grid overlay)")
    print("-" * 70)
    image_size = (224, 224)
    feature_map_size = (14, 14)  # 224/16 = 14
    anchors = anchor_gen.generate_anchors(feature_map_size, image_size)
    print(f"Image size: {image_size}")
    print(f"Feature map: {feature_map_size} (stride={anchor_gen.stride})")
    print(f"Total locations: {feature_map_size[0] * feature_map_size[1]}")
    print(f"Anchors per location: {anchor_gen.num_anchors}")
    print(f"Total anchors: {len(anchors)}")

    # Visuals: base anchors, tiled anchors + YOLO grid
    anchor_gen.visualize_base_anchors()
    anchor_gen.visualize_anchors_on_image(image_size, feature_map_size)

    print("\n3. GT→Anchor Assignment Visualization (IoU heat + best anchor)")
    print("-" * 70)
    gt_box = (60, 60, 160, 170)  # (x1,y1,x2,y2) for demo
    anchors, ious = anchor_gen.visualize_assignment(
        image_size=image_size,
        feature_map_size=feature_map_size,
        gt_box=gt_box,
        iou_vis_thresh=0.5
    )

    print_parametrization_explainer()

    print("\n4. Micro-NMS on 'objectness' ≈ IoU with GT")
    print("-" * 70)
    # Treat IoU with the GT as a stand-in objectness score
    scores = ious
    keep = anchor_gen.micro_nms_demo(
        anchors=anchors, scores=scores,
        image_size=image_size,
        iou_thresh=0.5, topk=200
    )

    print("\n" + "=" * 70)
    print("Why Anchors Matter for Faster R-CNN:")
    print("=" * 70)
    print("""
1) MULTI-SCALE & MULTI-SHAPE:
   - Scales (8,16,32) and ratios (0.5,1,2) cover size/shape diversity.

2) DENSE COVERAGE:
   - Anchors at every lattice cell (stride s). Assignment via IoU.

3) EFFICIENCY:
   - RPN predicts objectness + deltas from fixed templates.
   - Micro-NMS reduces thousands of candidates to a compact set.

4) CONNECTION TO YOLO:
   - Same lattice as YOLO's S×S grid; YOLOv2/3 attach anchors per cell,
     while YOLOv1 regresses boxes directly from the cell without priors.
    """)

    print("\nDemo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    demo_anchor_properties()
