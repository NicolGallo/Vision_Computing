# RPN and Faster R-CNN Classroom Demo Guide

## 📚 Understanding the Evolution

### The Story So Far:
1. **R-CNN** (2014): Selective Search → CNN (slow, 2000 CNN passes)
2. **Fast R-CNN** (2015): Selective Search → Shared CNN (10× faster, still uses Selective Search)
3. **Faster R-CNN** (2015): **RPN** → Shared CNN (200× faster proposals!)

### The Key Innovation: RPN (Region Proposal Network)
- Replaces hand-crafted Selective Search with **learned proposals**
- Uses **anchors** as templates at multiple scales/ratios
- Shares CNN features (no separate proposal generation)
- End-to-end trainable

---

## 🎯 Teaching Flow (30-40 minutes)

### Part 1: Anchor Generation (10 min)

**Concept**: Before RPN, understand how anchors work

```python
from anchor_generation_demo import demo_anchor_properties

# Run standalone anchor demo
demo_anchor_properties()
```

**What students will see:**
1. Base anchor templates (9 templates: 3 scales × 3 ratios)
2. Anchors tiled across image (1,764 total for 14×14 feature map)
3. Visualizations showing multi-scale, multi-shape coverage

**Key Teaching Points:**
- "Anchors are pre-defined boxes at different scales and shapes"
- "At each location, we place 9 candidate boxes"
- "RPN's job: decide which anchors contain objects"
- "No image pyramid needed - anchors handle multiple scales!"

**Generated Files:**
- `base_anchors.pdf` - Show base anchor templates
- `anchors_on_image.pdf` - Show tiling across image

---

### Part 2: RPN Network (10 min)

**Concept**: How RPN uses anchors to generate proposals

```python
from rpn_faster_rcnn import demo_faster_rcnn

# Run RPN demo
demo_faster_rcnn()
```

**What students will see:**
```
RPN Training:
  Epoch 1: Total=1.8234, Cls=1.2156, Reg=0.6078
  ...
  Epoch 10: Total=0.5421, Cls=0.3012, Reg=0.2409

Inference:
  Generated 2000 proposals per image
```

**Key Teaching Points:**
- "RPN has 2 jobs: objectness + box refinement"
- "Objectness: Is there an object at this anchor?"
- "Box refinement: How should we adjust the anchor?"
- "Training uses IoU thresholds (>0.7 positive, <0.3 negative)"

**Architecture Diagram on Board:**
```
Input Image
    ↓
Shared CNN Features
    ↓
RPN (3×3 conv)
    ├→ Objectness (1 × num_anchors)
    └→ Box Deltas (4 × num_anchors)
    ↓
Proposals → Detection Head
```

---

### Part 3: Faster R-CNN Integration (10 min)

**Concept**: Putting it all together

**Show code evolution:**

```python
# R-CNN: Separate components
proposals = selective_search(image)  # 2s
features = [cnn(crop(image, p)) for p in proposals]  # 2000 CNN passes

# Fast R-CNN: Shared CNN
proposals = selective_search(image)  # Still 2s
features = cnn(image)  # 1 CNN pass
roi_features = roi_pool(features, proposals)

# Faster R-CNN: RPN replaces Selective Search
features = cnn(image)  # 1 CNN pass
proposals = rpn(features)  # 0.01s (200× faster!)
roi_features = roi_pool(features, proposals)
detections = detection_head(roi_features)
```

**Timing Comparison Table:**
| Component | R-CNN | Fast R-CNN | Faster R-CNN |
|-----------|-------|------------|--------------|
| Proposals | 2s | 2s | **0.01s** |
| CNN | 47s | 0.3s | 0.2s |
| **Total** | **49s** | **2.3s** | **0.21s** |
| Speed-up | 1× | 21× | **233×** |

---

### Part 4: Live Coding Demo (10 min)

**Interactive demonstration:**

```python
# 1. Show anchor generation
from anchor_generation_demo import AnchorGenerator

gen = AnchorGenerator(scales=[8,16,32], aspect_ratios=[0.5,1.0,2.0], stride=16)

# Visualize what happens at ONE location
base = gen.generate_base_anchors()
print(f"At each location, we have {len(base)} anchors")
print("Different sizes and shapes to catch different objects")

# 2. Show RPN in action
from rpn_faster_rcnn import RPN
import torch

rpn = RPN(in_channels=512, num_anchors=9)

# Dummy features (what CNN produces)
features = torch.randn(1, 512, 14, 14)

# RPN predictions
objectness, bbox_deltas = rpn(features)

print(f"Objectness shape: {objectness.shape}")  # [1, 14, 14, 9]
print(f"Box deltas shape: {bbox_deltas.shape}")  # [1, 14, 14, 36]

print("\nFor each of 1764 anchors (14×14×9), RPN predicts:")
print("  • Is there an object? (objectness)")
print("  • How to adjust the anchor? (4 deltas)")

# 3. Show full Faster R-CNN
model = FasterRCNN(num_classes=20)
# ... train and show results
```

---

## 📊 Visualizations to Show

### 1. Anchor Templates (`base_anchors.pdf`)
- 9 boxes at origin showing different scales/ratios
- Label each with scale and ratio
- **Point out:** "These are templates, copied to every location"

### 2. Anchors on Image (`anchors_on_image.pdf`)
- Left panel: 9 anchors at one location
- Right panel: Sample across entire image
- **Point out:** "Dense coverage ensures no object is missed"

### 3. Architecture Comparison (draw on board)
```
R-CNN:       [Image] → Selective Search → [CNN ×2000]
Fast R-CNN:  [Image] → Selective Search → [Shared CNN] → RoI Pool
Faster R-CNN:[Image] → [Shared CNN] → RPN → RoI Pool
                              ↓
                          Anchors!
```

---

## 🔧 Demo Scripts

### Quick Anchor Demo (5 min)
```bash
python anchor_generation_demo.py
```
Shows: Base anchors, tiling, statistics

### Full RPN Demo (5 min)
```bash
python rpn_faster_rcnn.py
```
Shows: RPN training, proposal generation, comparison

### Combined Demo (10 min)
```python
# In Python/Jupyter
from anchor_generation_demo import AnchorGenerator
from rpn_faster_rcnn import FasterRCNN

# 1. Understand anchors
gen = AnchorGenerator()
gen.visualize_base_anchors()
gen.visualize_anchors_on_image()

# 2. See RPN in action
model = FasterRCNN(num_classes=20)
# ... train and show results
```

---

## 💡 Key Concepts to Emphasize

### 1. **Why Anchors?**
- **Problem**: Objects have different sizes and shapes
- **Solution**: Put multiple templates at each location
- **Benefit**: RPN just needs to say "yes/no" and "adjust slightly"

### 2. **Anchor Parameters**
- **Scales [8, 16, 32]**: Handle small, medium, large objects
- **Ratios [0.5, 1.0, 2.0]**: Handle tall, square, wide objects
- **Stride 16**: Feature map downsampling factor
- **Total per location**: 3 scales × 3 ratios = 9 anchors

### 3. **RPN Training**
- **Positive anchors**: IoU > 0.7 with any GT box
- **Negative anchors**: IoU < 0.3 with all GT boxes
- **Neutral anchors**: 0.3 ≤ IoU ≤ 0.7 (ignored)
- **Mini-batch**: 256 anchors (128 pos + 128 neg)

### 4. **RPN Output**
- **Objectness**: Binary classification (object vs background)
- **Box deltas**: 4 values (tx, ty, tw, th) to refine anchor
- **Same parameterization** as Fast R-CNN bbox regression

---

## 🎓 Student Questions & Answers

### Q: "Why do we need anchors? Can't we just predict boxes directly?"
**A:** We could (that's what YOLO does), but anchors help training:
- Provide good initial guesses
- Reduce the learning problem (predict small adjustments, not full boxes)
- Enable multi-scale detection without image pyramids

### Q: "How many anchors do we really need?"
**A:** For 224×224 image with stride=16:
- Feature map: 14×14 = 196 locations
- Anchors per location: 9
- **Total: 1,764 anchors**
- After NMS: Keep top ~2000 proposals

### Q: "Is RPN trained separately or jointly?"
**A:** In practice, 4-step alternating training:
1. Train RPN
2. Train Fast R-CNN with RPN proposals
3. Fine-tune RPN (freeze shared layers)
4. Fine-tune Fast R-CNN (freeze shared layers)

But modern implementations use **approximate joint training** (easier, works well)

### Q: "Why not just use YOLO instead?"
**A:** Trade-offs:
- **Faster R-CNN**: More accurate (two-stage refinement)
- **YOLO**: Faster inference (one-stage)
- **Faster R-CNN**: Better for small objects (anchors at multiple scales)
- **YOLO**: Simpler architecture

---

## 📝 Homework Ideas

1. **Implement Different Anchor Configurations**
   - Try [4, 8, 16, 32] scales
   - Try [0.33, 0.5, 1.0, 2.0, 3.0] ratios
   - Compare coverage and performance

2. **Visualize RPN Predictions**
   - Show which anchors have high objectness
   - Visualize box refinements
   - Compare to ground truth

3. **Analyze Anchor Assignment**
   - How many anchors are positive/negative/neutral?
   - Does class imbalance matter?
   - Try different IoU thresholds

4. **Compare Proposal Methods**
   - RPN vs Selective Search
   - Speed comparison
   - Quality comparison (recall at different IoU)

---

## 🔍 Assessment Questions

1. **Why did Faster R-CNN replace Selective Search with RPN?**
   - Speed (200× faster)
   - Learned from data
   - End-to-end training
   - Better proposals

2. **Explain anchor generation at one location**
   - 3 scales × 3 ratios = 9 anchors
   - Different sizes and shapes
   - Centered at feature map location
   - Mapped to image coordinates

3. **What does RPN predict for each anchor?**
   - Objectness score (is object?)
   - 4 box deltas (how to adjust?)

4. **How are anchors assigned during training?**
   - Positive: IoU > 0.7
   - Negative: IoU < 0.3
   - Ignore: in between

---

## 📦 Files Included

### Demo Scripts:
- `anchor_generation_demo.py` - Standalone anchor visualization
- `rpn_faster_rcnn.py` - Complete RPN and Faster R-CNN

### Generated Visualizations:
- `base_anchors.pdf` - Base anchor templates
- `anchors_on_image.pdf` - Anchors tiled on image

### Slides (Beamer):
- Evolution comparison (R-CNN → Fast → Faster)
- Anchor generation with code
- RPN architecture with code
- Training procedure with code
- Complete demo results

---

## ⏱️ Suggested Lecture Timeline

**0:00 - 0:05**: Recap Fast R-CNN limitations (still uses Selective Search)
**0:05 - 0:15**: Anchor generation demo (standalone, visual)
**0:15 - 0:25**: RPN architecture and training
**0:25 - 0:35**: Full Faster R-CNN demo
**0:35 - 0:40**: Q&A and wrap-up

**Total: 40 minutes** (fits in one lecture slot)

---

## 🎯 Learning Outcomes

By the end of the demo, students should:
1. ✅ Understand what anchors are and why they're used
2. ✅ Explain how RPN generates proposals
3. ✅ Know the difference between R-CNN, Fast R-CNN, and Faster R-CNN
4. ✅ Appreciate the end-to-end learning paradigm
5. ✅ Be ready for modern detectors (RetinaNet, FCOS, etc.)

---

## 🚀 Next Steps

After Faster R-CNN, students are ready for:
1. **Feature Pyramid Networks (FPN)** - Multi-scale features
2. **Focal Loss (RetinaNet)** - Class imbalance solution
3. **Anchor-free detectors (FCOS)** - Beyond anchors
4. **Transformers (DETR)** - Set prediction paradigm

The anchor concept is fundamental to understanding modern detection!