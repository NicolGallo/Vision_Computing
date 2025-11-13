# Cross-correlation demo: locate a template inside a search image
# Run: pip install opencv-python matplotlib numpy
import numpy as np
import cv2
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# 1) Create a synthetic "template" (e.g., a blob + edges)
h, w = 31, 31
template = np.zeros((h, w), np.float32)
cv2.circle(template, center=(w//2, h//2), radius=9, color=1.0, thickness=-1)
cv2.GaussianBlur(template, (0, 0), 2.0, dst=template)

# Add a simple edge-like pattern to make it more "HOG-friendly"
cv2.rectangle(template, (3, 3), (w-4, h-4), 0.5, 2)

# 2) Create a "search" image (next frame), insert the template at an unknown shift + noise
H, W = 128, 128
search = (rng.normal(0.0, 0.05, (H, W))).astype(np.float32)  # background noise
true_top, true_left = 70, 45                                # ground-truth placement
search[true_top:true_top+h, true_left:true_left+w] += template

# 3) Cross-correlation (normalized) using OpenCV (TM_CCOEFF_NORMED)
#    This is equivalent to sliding the template over all positions at once (conceptually),
#    producing a response map where the peak = best match.
res = cv2.matchTemplate(search, template, method=cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
pred_top, pred_left = max_loc[1], max_loc[0]

# 4) Visualize: template, search with GT + pred boxes, and response heatmap
fig = plt.figure(figsize=(10, 3.2))

ax1 = plt.subplot(1, 3, 1)
ax1.imshow(template, cmap='gray', vmin=0, vmax=1)
ax1.set_title('Template'); ax1.axis('off')

ax2 = plt.subplot(1, 3, 2)
ax2.imshow(search, cmap='gray')
# Draw ground-truth (green) and prediction (red)
gt_rect = plt.Rectangle((true_left, true_top), w, h, fill=False, linewidth=2)
gt_rect.set_edgecolor('g')
pred_rect = plt.Rectangle((pred_left, pred_top), w, h, fill=False, linewidth=2)
pred_rect.set_edgecolor('r')
ax2.add_patch(gt_rect); ax2.add_patch(pred_rect)
ax2.set_title('Search image\nGT (green) vs Pred (red)')
ax2.axis('off')

ax3 = plt.subplot(1, 3, 3)
im = ax3.imshow(res, cmap='viridis')
ax3.plot(pred_left, pred_top, 'r+', markersize=12, markeredgewidth=2)
ax3.set_title(f'Response map (peak={max_val:.3f})')
ax3.axis('off')
plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

print(f"True (top,left)=({true_top},{true_left}) | Predicted=({pred_top},{pred_left})")
