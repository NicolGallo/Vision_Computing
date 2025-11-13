# Classical Tracking Methods Demo

This directory contains OpenCV implementations of classical (pre-deep learning) tracking methods.

## Methods Implemented

### 1. KLT Tracker (`klt_demo.py`)

**Kanade-Lucas-Tomasi** tracker using sparse optical flow.

**Key Features:**
- Shi-Tomasi corner detection for feature selection
- Pyramidal Lucas-Kanade optical flow
- Automatic feature re-detection when points are lost
- Trajectory visualization

**Usage:**
```bash
# Track with webcam
python klt_demo.py

# Track video file with ROI selection
python klt_demo.py --video input.mp4 --roi

# Save output
python klt_demo.py --video input.mp4 --output output.mp4

# Adjust parameters
python klt_demo.py --max_corners 200
```

**Strengths:**
- Fast (sparse point tracking)
- Works well for small inter-frame motion
- Good for structure from motion applications

**Weaknesses:**
- No object identity (just tracks points)
- Brittle under large motion or occlusion
- Requires good features (corners)

### 2. MeanShift/CamShift Tracker (`meanshift_demo.py`)

**Color histogram-based** tracking in HSV space.

**Key Features:**
- MeanShift: Fixed window size
- CamShift: Adaptive window (scale + orientation)
- Back-projection visualization
- Interactive ROI selection

**Usage:**
```bash
# MeanShift tracking with webcam
python meanshift_demo.py

# CamShift (adaptive) with video
python meanshift_demo.py --video input.mp4 --camshift

# Specify initial bounding box
python meanshift_demo.py --video input.mp4 --bbox 100 100 80 120

# Adjust histogram resolution
python meanshift_demo.py --bins 32 --camshift
```

**Strengths:**
- Robust to partial occlusion
- Handles rotation (CamShift)
- Adapts to scale changes (CamShift)
- Simple and fast

**Weaknesses:**
- Fails with background clutter of similar color
- No appearance model update
- Can drift over time

## Algorithm Complexity

| Method | Time Complexity | Space | Real-time |
|--------|----------------|-------|-----------|
| KLT | O(n·w²·L) | O(n) | Yes (30+ FPS) |
| MeanShift | O(k·w·h) | O(bins) | Yes (60+ FPS) |
| CamShift | O(k·w·h) | O(bins) | Yes (60+ FPS) |

where:
- n = number of points
- w = window size
- L = pyramid levels
- k = iterations to converge
- bins = histogram bins

## Tips for Good Results

### KLT:
1. Ensure sufficient lighting and texture
2. Use pyramid levels (2-3) for larger motions
3. Re-detect features periodically
4. Filter out points with high error

### MeanShift/CamShift:
1. Choose distinctive color object
2. Good lighting important
3. Adjust histogram bins (16-32 typical)
4. Use mask to exclude very dark/bright regions
5. CamShift better for scale/rotation

## Comparison with Modern Deep Learning

These classical methods form the **conceptual foundation** for modern trackers:

- **KLT** → Optical flow networks (FlowNet, RAFT)
- **MeanShift** → Attention mechanisms (soft selection)
- **Template Matching** → Siamese networks (SiamFC)
- **Color Histograms** → Learned embeddings

**Evolution:**
```
Hand-crafted features → Learned features
Explicit models → End-to-end learning
Single scale → Multi-scale pyramids
Fixed templates → Adaptive templates
```

## References

1. Shi & Tomasi (1994). "Good Features to Track"
2. Lucas & Kanade (1981). "An Iterative Image Registration Technique"
3. Comaniciu & Meer (2002). "Mean Shift: A Robust Approach Toward Feature Space Analysis"
4. Bradski (1998). "Computer Vision Face Tracking For Use in a Perceptual User Interface"