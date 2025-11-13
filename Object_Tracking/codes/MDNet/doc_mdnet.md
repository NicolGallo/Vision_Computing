# MDNet: Multi-Domain Convolutional Neural Network Tracker

## Paper Information
**Title:** Learning Multi-Domain Convolutional Neural Networks for Visual Tracking  
**Authors:** Hyeonseob Nam, Bohyung Han  
**Conference:** CVPR 2016  
**Paper:** [arXiv:1510.07945](https://arxiv.org/abs/1510.07945)

## Overview

MDNet is one of the **first successful deep learning trackers**, achieving state-of-the-art performance through:
1. **Multi-domain learning**: Pre-training on multiple tracking sequences
2. **Online adaptation**: Fine-tuning during tracking
3. **Binary classification**: Distinguishing target from background
4. **Shared + Domain-specific architecture**: Generalizable features + sequence-specific adaptation

**Impact:** Won VOT2015 challenge, established deep learning as viable for tracking.

## Architecture

### Network Structure

```
Input: 107×107×3 RGB patch

Shared Convolutional Layers (φ_shared):
├── Conv1: 3→96 channels, kernel 7×7, stride 2
│   ├── ReLU activation
│   ├── LRN (Local Response Normalization)
│   └── MaxPool 3×3, stride 2
│   Output: 96×24×24
│
├── Conv2: 96→256 channels, kernel 5×5, stride 2
│   ├── ReLU activation
│   ├── LRN
│   └── MaxPool 3×3, stride 2
│   Output: 256×5×5
│
└── Conv3: 256→512 channels, kernel 3×3, stride 1
    └── ReLU activation
    Output: 512×3×3 = 4,608 dimensions

Domain-Specific FC Layers (for domain k):
├── FC4_k: 4,608 → 512
│   ├── ReLU activation
│   └── Dropout(0.5)
│
└── FC5_k: 512 → 512
    ├── ReLU activation
    └── Dropout(0.5)

Shared Classification Layer:
└── FC6: 512 → 2 (target/background)
    └── Softmax activation
```

### Parameter Count

| Component | Parameters | Description |
|-----------|-----------|-------------|
| Conv1-3 (shared) | ~6M | Feature extraction |
| FC4-5 (per domain) | ~2.6M | Domain-specific adaptation |
| FC6 (shared) | ~1K | Binary classification |
| **Total (K domains)** | **6M + 2.6M×K** | Scales with domains |

### Why This Architecture?

**Shared Convolutional Layers:**
- Extract generic visual features
- Learn edges, textures, patterns
- Invariant across different sequences

**Domain-Specific FC Layers:**
- Adapt to sequence-specific appearance
- Handle different object categories
- Capture sequence-level variations

**Shared Classification Layer:**
- Universal target vs background decision
- Trained on all domains jointly

## Multi-Domain Learning

### Training Strategy

```
Training Data: K video sequences (domains)
  - K = 50-100 sequences
  - Each sequence = one domain

For each training iteration:
  1. Sample domain k randomly
  2. Sample positive/negative patches from domain k
  3. Forward pass using:
     - Shared conv layers (φ_shared)
     - Domain k's FC layers (φ_k)
     - Shared classification layer
  4. Compute loss and backpropagate
  5. Update:
     - φ_shared (always)
     - φ_k (only for domain k)
     - Classification layer (always)
```

### Loss Function

**Binary Cross-Entropy:**

```
L = -(1/N) Σ [y_i log(p_i) + (1-y_i) log(1-p_i)]
           i=1..N

where:
  y_i = 1 for positive samples (target)
  y_i = 0 for negative samples (background)
  p_i = softmax probability for target class
  N = batch size
```

### Positive and Negative Sampling

**Positive Samples:**
- IoU > 0.7 with ground truth
- ~50 samples per frame
- Small translation (±10% of target size)
- Small scale variation (±5%)

**Negative Samples:**
- IoU < 0.5 with ground truth
- ~200 samples per frame
- Large translation (50-200% of target size)
- Various scales

**Hard Negative Mining:**
```python
# During training
negative_scores = model(negative_samples)
hard_negatives = negative_samples[negative_scores > 0.5]

# Focus training on hard negatives
loss = cross_entropy(hard_negatives, labels=0)
```

## Online Adaptation During Tracking

### Tracking Algorithm

```
Initialization (Frame 1):
  1. Extract patch around initial bbox
  2. Sample positives (50) and negatives (200)
  3. Train model for 30 epochs
     - Fine-tune FC4, FC5, FC6
     - Keep Conv1-3 fixed initially
  4. Store as initialized model

Tracking (Frame t > 1):
  1. Sample N=256 candidate patches around previous location
     - Translation: ±60% of target size
     - Scale: ±5% of target size
  
  2. Forward pass all candidates through network
     scores = softmax(model(candidates))[:, 1]  # Target class
  
  3. Select best candidate:
     best_idx = argmax(scores)
     current_bbox = candidates[best_idx].bbox
  
  4. Online update (every 10 frames):
     if frame_idx % 10 == 0 and score > 0.5:
       - Sample new positives around current_bbox
       - Short training (5 iterations)
       - Update only FC6 (classification layer)
       - Small learning rate (0.0001)
```

### Why Online Adaptation?

**Problem:** Target appearance changes over time
- Illumination changes
- Pose variations
- Partial occlusions
- Deformations

**Solution:** Adapt model to current appearance

**Trade-off:**
- ✓ Handles appearance changes
- ✗ Risk of drift (adapting to wrong target)
- ✗ Slower (needs training time)

### Preventing Drift

**Strategies:**
1. **Conservative updates**: Update only when confident (score > 0.5)
2. **Infrequent updates**: Every 10 frames, not every frame
3. **Limited adaptation**: Update only FC6, keep Conv1-5 fixed
4. **Small learning rate**: 0.0001 vs 0.001 for initial training
5. **Short training**: 5 iterations vs 30 for initialization

## Training Details

### Pre-training Phase

**Datasets:**
- VOT2014 training set
- VOT2015 training set
- OTB-100 sequences
- **Total:** ~50 sequences as domains

**Training Configuration:**
| Parameter | Value |
|-----------|-------|
| Batch size | 128 (50 pos + 200 neg) |
| Learning rate | 0.0001 |
| Weight decay | 0.0005 |
| Momentum | 0.9 |
| Epochs | 100 |
| Optimizer | SGD with momentum |
| Dropout | 0.5 |

**Data Augmentation:**
- Random Gaussian noise
- Color jittering
- Random translation (during sampling)

### Online Fine-tuning

**Initial Frame:**
| Parameter | Value |
|-----------|-------|
| Positive samples | 50 |
| Negative samples | 200 |
| Epochs | 30 |
| Learning rate | 0.0001 |
| Layers updated | FC4, FC5, FC6 |

**During Tracking:**
| Parameter | Value |
|-----------|-------|
| Update interval | 10 frames |
| Positive samples | 20 |
| Iterations | 5 |
| Learning rate | 0.00001 |
| Layers updated | FC6 only |

## Performance

### Benchmark Results

**OTB-100:**
- Success rate: **67.8%**
- Precision: **90.9%**
- Rank: **#1** (2016)

**VOT2015:**
- Accuracy: **0.60**
- Robustness: **1.16** (failures per sequence)
- **Winner** of VOT2015 challenge

**Comparison (2016):**
| Method | OTB Success | OTB Precision | VOT2015 EAO |
|--------|-------------|---------------|-------------|
| SRDCF | 62.6% | 83.8% | 0.238 |
| LCT | 57.8% | 84.8% | 0.227 |
| **MDNet** | **67.8%** | **90.9%** | **0.257** |

### Speed Analysis

| Component | Time (ms) | Notes |
|-----------|-----------|-------|
| Candidate sampling | 50 | Generate 256 patches |
| Feature extraction | 600 | Conv1-3 for all |
| Classification | 50 | FC4-6 |
| Online update | 300 | Every 10 frames |
| **Total per frame** | **~1000** | **~1 FPS** |

**Bottleneck:** Processing 256 candidates through network

## Ablation Studies

### Impact of Multi-Domain Learning

| Configuration | OTB Success | VOT EAO |
|--------------|-------------|---------|
| Single domain | 62.3% | 0.223 |
| 10 domains | 65.1% | 0.241 |
| **50 domains** | **67.8%** | **0.257** |

More domains → Better generalization!

### Impact of Online Adaptation

| Configuration | OTB Success |
|--------------|-------------|
| No online update | 61.2% |
| Update every frame | 64.5% |
| **Update every 10 frames** | **67.8%** |

Balanced update frequency optimal.

### Impact of Network Depth

| Architecture | OTB Success | Speed |
|-------------|-------------|-------|
| 3 conv + 1 FC | 58.3% | 3 FPS |
| **3 conv + 3 FC** | **67.8%** | 1 FPS |
| 5 conv + 3 FC | 68.1% | 0.5 FPS |

Diminishing returns with more layers.

## Strengths and Limitations

### Strengths

✓ **State-of-the-art accuracy (2016)**: Outperformed all competitors  
✓ **Multi-domain learning**: Good generalization  
✓ **Online adaptation**: Handles appearance changes  
✓ **Binary classification**: Simple and effective  
✓ **Robust**: Handles occlusions, deformations  
✓ **Influential**: Established deep learning for tracking

### Limitations

✗ **Very slow**: ~1 FPS (1000× slower than real-time)  
✗ **Heavy online training**: Requires GPU  
✗ **Drift risk**: Online adaptation can fail  
✗ **No temporal modeling**: Treats frames independently  
✗ **Large model**: Many domain-specific parameters  
✗ **Bounding box only**: No segmentation

## Implementation Details

### Candidate Sampling Strategy

```python
def sample_candidates(prev_bbox, image_size, n_samples=256):
    """Sample candidate patches around previous location"""
    x, y, w, h = prev_bbox
    candidates = []
    
    for i in range(n_samples):
        # Sample translation
        dx = np.random.uniform(-0.6*w, 0.6*w)
        dy = np.random.uniform(-0.6*h, 0.6*h)
        
        # Sample scale
        scale = np.random.uniform(0.95, 1.05)
        
        # New bbox
        new_x = x + dx
        new_y = y + dy
        new_w = w * scale
        new_h = h * scale
        
        candidates.append([new_x, new_y, new_w, new_h])
    
    return candidates
```

### Patch Extraction

```python
def extract_patch(image, bbox, output_size=107):
    """Extract and resize patch from image"""
    x, y, w, h = bbox
    
    # Add context (padding)
    pad = 0.3  # 30% padding
    x1 = max(0, int(x - pad*w))
    y1 = max(0, int(y - pad*h))
    x2 = min(image.shape[1], int(x + w + pad*w))
    y2 = min(image.shape[0], int(y + h + pad*h))
    
    # Crop
    patch = image[y1:y2, x1:x2]
    
    # Resize to 107×107
    patch = cv2.resize(patch, (output_size, output_size))
    
    return patch
```

### Hard Negative Mining

```python
def hard_negative_mining(negatives, scores, threshold=0.3):
    """Select hard negatives for training"""
    # Hard negatives: high score but actually negative
    hard_negatives = negatives[scores > threshold]
    
    # If too many, sample randomly
    if len(hard_negatives) > 50:
        indices = np.random.choice(len(hard_negatives), 50, replace=False)
        hard_negatives = hard_negatives[indices]
    
    return hard_negatives
```

## Usage Example

```python
from mdnet_demo import MDNetTracker
import cv2

# Initialize tracker
tracker = MDNetTracker()

# First frame
first_frame = cv2.imread('frame_0001.jpg')
init_bbox = [x, y, width, height]

print("Initializing... (may take 30 seconds)")
tracker.init(first_frame, init_bbox)

# Track video
cap = cv2.VideoCapture('video.mp4')
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_idx += 1
    
    # Track
    bbox, score = tracker.update(frame)
    
    # Visualize
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(frame, f'Score: {score:.2f}', (x, y-10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('MDNet', frame)
    
    if cv2.waitKey(30) == ord('q'):
        break
```

## Extensions and Variants

### RT-MDNet (Real-Time MDNet)

**Key Changes:**
- Lightweight backbone (MobileNet)
- Fewer candidates (64 vs 256)
- Faster inference (~25 FPS)
- Slight accuracy drop (65% vs 67.8%)

### MDNet with Correlation Filters

**Hybrid Approach:**
- MDNet for feature extraction
- Correlation filter for localization
- Faster tracking (~10 FPS)

### MDNet++

**Improvements:**
- Better sampling strategy
- Improved online update
- Multi-scale search
- +2% accuracy improvement

## Historical Context

### Before MDNet (2015)

**Dominant paradigm:** Correlation filters
- Fast (100+ FPS)
- Limited by hand-crafted features
- Accuracy plateau

### MDNet's Impact (2016)

**Paradigm shift:** Deep learning for tracking
- First deep tracker to win major challenge
- Showed CNNs work for tracking
- Inspired numerous follow-ups

### After MDNet (2016-2020)

**New direction:** Siamese networks
- SiamFC (2016): Faster but less accurate
- SiamRPN (2018): Fast AND accurate
- Eventually superseded MDNet's accuracy/speed trade-off

## Key Takeaways

### For Researchers

1. **Multi-domain learning works**: Pre-training on diverse data crucial
2. **Online adaptation necessary**: Target appearance changes
3. **Balance updates carefully**: Too frequent → drift, too rare → stale
4. **Binary classification sufficient**: Don't need complex outputs
5. **Speed matters**: 1 FPS too slow for practical use

### For Practitioners

1. **Use for accuracy, not speed**: When accuracy is paramount
2. **Need GPU**: Too slow on CPU
3. **Initial training important**: Spend time on first frame
4. **Monitor confidence**: Use scores to detect failures
5. **Consider alternatives**: Modern trackers are faster and better

## References

1. [Original Paper](https://arxiv.org/abs/1510.07945)
2. [Project Page](http://cvlab.postech.ac.kr/research/mdnet/)
3. [Official MATLAB Code](https://github.com/HyeonseobNam/MDNet)
4. [PyTorch Implementation](https://github.com/HyeonseobNam/py-MDNet)
5. [VOT Challenge](https://www.votchallenge.net/)

## Citation

```bibtex
@inproceedings{nam2016mdnet,
  title={Learning Multi-domain Convolutional Neural Networks for Visual Tracking},
  author={Nam, Hyeonseob and Han, Bohyung},
  booktitle={CVPR},
  year={2016}
}
```

---

**Legacy:** MDNet proved deep learning viable for tracking, despite speed limitations. It established multi-domain learning and online adaptation as key techniques, influencing tracker design for years to come.