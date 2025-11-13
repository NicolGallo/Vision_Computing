# Computer Vision Lab 1: Camera Calibration and Image Transformations

## Overview

Lab 1 transforms the mathematical foundations from Lecture 1 into practical implementations. Students will implement core computer vision algorithms including Zhang's camera calibration method, homography estimation, and fundamental image processing operations.

## Learning Objectives

By the end of this lab, students will be able to:

1. **Implement Calibration Code in OpenCV**
2. **Implement Zhang's camera calibration algorithm from scratch**
3. **Advanced:  Estimate homographies using Direct Linear Transform (DLT) and RANSAC**
4. **Advanced/Optional: Apply lens distortion correction and image transformations**
5. **Advanced/optional: Perform color space conversions and histogram equalization**
6. **Advanced/Optional: Analyze calibration quality and algorithm robustness**
7. **Connect theoretical concepts to practical implementations**
8. **Pytorch  for next time**


## Lab Structure

### Materials Provided

1. **Lab Notes** (`lab1_notes.tex`): LaTeX Beamer presentation with:
   - Step-by-step algorithm explanations
   - Code demonstrations
   - Exercises and questions
   - Theoretical connections

2. **Demo Code** (`lab1_demo_code.py`): Complete working implementation:
   - All algorithms from the lecture
   - Visualization utilities
   - Synthetic data generation
   - Example workflows

3. **Student Exercises** (`lab1_exercises.py`): Structured assignments:
   - 7 main exercises with increasing difficulty
   - Challenge problems for extra credit
   - Theoretical questions
   - Experimental tasks

4. **Setup Instructions** (this file): Complete environment setup and usage guide

## 🚀 Setup Instructions

### Prerequisites

```bash
# Python 3.8+ required
python --version  # Should be 3.8 or higher
```

### Environment Setup

#### Option 1: Conda Environment (Recommended)

```bash
# Create new environment
conda create -n cv_lab1 python=3.9

# Activate environment
conda activate cv_lab1

# Install packages
conda install pytorch torchvision opencv matplotlib numpy scipy scikit-learn
```

#### Option 2: Virtual Environment

```bash
# Create virtual environment
python -m venv cv_lab1

# Activate (Linux/Mac)
source cv_lab1/bin/activate
# Activate (Windows)
cv_lab1\Scripts\activate

# Install packages
pip install torch torchvision opencv-python matplotlib numpy scipy scikit-learn
```

### Verify Installation

```python
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

print("✓ All packages installed successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"OpenCV version: {cv2.__version__}")
```

## Lab Session Structure
### Hands-on Implementation  

Students work through exercises in order:

1. **Exercise 1.1**: OpenCV Calibration Code (15 min)

Extra/Additional:  
1. **Exercise E1.1**: PyTorch tensor operations  
2. **Exercise E1.2**: Geometric transformations 
3. **Exercise E1.3**: Corner detection analysis 
4. **Exercise E1.4**: Calibration quality assessment  
5. **Exercise E1.5**: Homography robustness  
6. **Exercise E1.6**: Color space analysis 


### Recommended Approach

1. **Read the theory** in lab notes before starting each exercise
2. **Study the demo code** to understand the implementation pattern
3. **Implement step-by-step** using the TODO comments as guidance
4. **Test incrementally** with provided test cases
5. **Answer questions** to deepen understanding
6. **Experiment** with different parameters and edge cases




## Challenge Problems (Advanced and optional)

### Challenge 1: Stereo Camera Calibration
- Implement two-camera calibration system
- Validate epipolar geometry constraints
- Create rectification pipeline

### Challenge 2: Real-time Calibration
- Build live webcam calibration interface
- Implement automatic quality assessment
- Add real-time parameter visualization

### Challenge 3: Custom Calibration Patterns
- Implement circular dot pattern detection
- Compare accuracy vs. checkerboard patterns
- Handle partial pattern visibility

## Connections to Course Materials

### From Lecture 1
- **Pinhole camera model** → Camera calibration implementation
- **Homogeneous coordinates** → Geometric transformation matrices


### To Lecture 2 (Preview)
- **Projective geometry** → Homography estimation algorithms
- **Optimization theory** → Bundle adjustment and RANSAC
- **Image filtering** builds on histogram equalization concepts
- **Convolution operations** extend transformation matrices
- **Feature extraction** generalizes corner detection methods
- **Deep learning pipelines** use these preprocessing techniques

## Additional Resources

### Reference Materials
- [Zhang's Camera Calibration Paper](https://www.microsoft.com/en-us/research/publication/a-flexible-new-technique-for-camera-calibration/)
- [Multiple View Geometry (Hartley & Zisserman)](https://www.robots.ox.ac.uk/~vgg/hzbook/)
- [OpenCV Calibration Documentation](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)

### Debugging Tools
```python
# Visualization helpers
def debug_corners(img, corners):
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.plot(corners[:, 0], corners[:, 1], 'ro', markersize=3)
    plt.title('Detected Corners')
    plt.show()

def debug_reprojection(img_points, proj_points):
    errors = np.linalg.norm(img_points - proj_points, axis=1)
    print(f"Reprojection errors: mean={errors.mean():.3f}, std={errors.std():.3f}")
    print(f"Max error: {errors.max():.3f} pixels")
```

### Performance Optimization
```python
# Use vectorized operations
def fast_transform_points(points, H):
    """Vectorized point transformation"""
    points_hom = np.column_stack([points, np.ones(len(points))])
    transformed = (H @ points_hom.T).T
    return transformed[:, :2] / transformed[:, [2]]
```

## Support and Help

### During Lab Session
- **TA support**: Available for implementation questions
- **Peer collaboration**: Encouraged for concept discussion
- **Instructor consultation**: For theoretical clarifications

### Common Issues
1. **Import errors**: Check environment activation
2. **Numerical instability**: Ensure point normalization
3. **Calibration failure**: Verify checkerboard detection
4. **Poor results**: Check parameter ranges and data quality

### Getting Help
```python
# Include this information when asking for help:
print(f"Python version: {sys.version}")
print(f"OpenCV version: {cv2.__version__}")
print(f"Error message: {error}")
print(f"Input data shape: {data.shape}")
print(f"Parameter values: {params}")
```

## Learning Outcomes Assessment

Upon completion, students should be able to:

- [ ] Set up and configure a computer vision development environment
- [ ] Implement camera calibration from mathematical first principles
- [ ] Apply robust estimation techniques to computer vision problems
- [ ] Analyze and validate the quality of computer vision algorithms
- [ ] Connect theoretical concepts to practical implementations
- [ ] Debug and optimize computer vision code effectively

## Submission Requirements

### Required Deliverables
1. **Completed exercise file** with all TODO sections implemented
2. **Analysis report** (2-3 pages) discussing:
   - Implementation challenges and solutions
   - Experimental results and insights
   - Comparison of different approaches
   - Theoretical connections and learning outcomes

3. **Calibration results** from your own dataset:
   - Camera parameters
   - Reprojection error analysis
   - Undistorted image examples

### Optional Deliverables (Extra Credit)
1. **Challenge problem implementations**
2. **Novel experimental analysis**
3. **Performance optimization improvements**
4. **Extended theoretical discussions**
