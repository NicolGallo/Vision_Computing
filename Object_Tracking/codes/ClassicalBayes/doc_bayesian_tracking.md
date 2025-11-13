# Bayesian Tracking Methods: Kalman and Particle Filters

This directory contains OpenCV implementations of classical Bayesian state estimation methods for tracking.

## Methods Implemented

### 1. Kalman Filter (`kalman_demo.py`)

**Linear-Gaussian Bayesian filtering** for state estimation.

**Key Features:**
- Constant velocity motion model
- Optimal for linear-Gaussian systems
- Prediction during missing measurements
- Uncertainty visualization (covariance ellipse)
- Real-time performance

**Theory:**
The Kalman filter recursively estimates state `x_t = [x, y, vx, vy]` from noisy measurements `z_t = [x_meas, y_meas]`.

**State Equations:**
```
Predict:  x̂(t|t-1) = A·x̂(t-1|t-1)
          P(t|t-1) = A·P(t-1|t-1)·A' + Q

Update:   K(t) = P(t|t-1)·H' / (H·P(t|t-1)·H' + R)
          x̂(t|t) = x̂(t|t-1) + K(t)·(z(t) - H·x̂(t|t-1))
          P(t|t) = (I - K(t)·H)·P(t|t-1)
```

**Usage:**
```bash
# Manual mode: click to provide measurements
python kalman_demo.py --manual

# Automatic detection mode
python kalman_demo.py --video input.mp4

# Adjust noise parameters
python kalman_demo.py --process_noise 0.01 --measurement_noise 0.1

# Save output
python kalman_demo.py --video input.mp4 --output output.mp4
```

**Visualization:**
- **Blue circle**: Predicted position
- **Green circle**: Measured position
- **Red filled circle**: Corrected (filtered) position
- **Yellow ellipse**: Uncertainty (2-sigma covariance)
- **Magenta arrow**: Velocity vector

**Strengths:**
- Optimal for linear systems
- Closed-form solution (fast)
- Handles missing measurements via prediction
- Provides uncertainty quantification

**Limitations:**
- Assumes linear dynamics
- Assumes Gaussian noise
- Single-modal posterior only

---

### 2. Particle Filter (`particle_demo.py`)

**Sequential Monte Carlo** for nonlinear/non-Gaussian tracking.

**Key Features:**
- Color histogram-based observation model
- No linearity/Gaussianity assumptions
- Handles multimodal distributions
- Systematic resampling
- Particle cloud visualization

**Theory:**
Represents posterior `p(x_t | z_{1:t})` as weighted particles:
```
p(x_t | z_{1:t}) ≈ Σ w_t^(i) δ(x_t - x_t^(i))
```

**Algorithm:**
```
For each time step:
  1. Predict: Sample x_t^(i) ~ p(x_t | x_{t-1}^(i))
  2. Weight:  w_t^(i) ∝ p(z_t | x_t^(i))
  3. Normalize: Σ w_t^(i) = 1
  4. Estimate: x̂_t = Σ w_t^(i) · x_t^(i)
  5. Resample: If ESS < threshold
```

**Observation Model:**
Color histogram similarity (Bhattacharyya coefficient):
```
ρ(h1, h2) = Σ sqrt(h1(b) · h2(b))
likelihood = exp(-λ · (1 - ρ))
```

**Usage:**
```bash
# Interactive ROI selection
python particle_demo.py --video input.mp4

# Specify number of particles
python particle_demo.py --num_particles 500

# Provide initial bounding box
python particle_demo.py --video input.mp4 --bbox 100 100 50 80

# Save output
python particle_demo.py --video input.mp4 --output output.mp4
```

**Visualization:**
- **Green dots**: Particles (colored by weight)
- **Green rectangle**: Estimated bounding box
- **Red circle**: Estimated center

**Strengths:**
- Handles nonlinear dynamics
- Non-Gaussian noise
- Multimodal posteriors (e.g., occlusion ambiguity)
- No Jacobians needed (unlike EKF)

**Limitations:**
- Computationally expensive (many particles needed)
- Particle degeneracy if not resampled properly
- Requires good motion model

---

## Comparison: Kalman vs Particle Filter

| Aspect | Kalman Filter | Particle Filter |
|--------|---------------|-----------------|
| **Computational Cost** | O(d³) | O(N·d) |
| **Optimality** | Optimal (linear-Gaussian) | Approximate |
| **Nonlinearity** | Poor (linearization) | Excellent |
| **Multimodal** | No | Yes |
| **Real-time** | Yes (~500 FPS) | Yes (~30-100 FPS) |
| **Tuning** | Q, R matrices | N, motion noise |

where d = state dim, N = number of particles

---

## Extended Kalman Filter (EKF)

For nonlinear systems, the **Extended Kalman Filter** linearizes via Jacobians:
```python
# Nonlinear motion model
x_t = f(x_{t-1}, u_t) + w_t

# Linearization
F_t = ∂f/∂x |_{x̂(t-1|t-1)}

# Then use F_t in place of A in Kalman equations
```

**When to use:**
- Moderate nonlinearity (e.g., bearing-only tracking)
- Gaussian noise assumption holds
- Need real-time performance

**Alternatives:**
- **Unscented Kalman Filter (UKF)**: Deterministic sampling, better for highly nonlinear
- **Ensemble Kalman Filter**: Use particles but with Kalman-like updates

---

## Connection to Modern Tracking

These classical methods laid the foundation for modern approaches:

### Motion Models → Neural Networks
```
Kalman:        x_t = A·x_{t-1} + w_t
Deep Learning: x_t = f_θ(x_{t-1}, I_t) + w_t
```

### Particle Filters → Attention Mechanisms
```
Particles:     Discrete weighted hypotheses
Transformers:  Continuous attention weights over spatial locations
```

### State Estimation → Embedding Learning
```
Bayesian:      p(x_t | z_{1:t}) via recursive update
Deep:          φ(I_t) ≈ φ(I_{t-1}) via learned similarity
```

---

## Implementation Notes

### Kalman Filter Tips:
1. **Initialize P_0 large** if uncertain about initial state
2. **Tune Q**: Higher = trust motion less, track measurements more closely
3. **Tune R**: Higher = trust measurements less, rely more on prediction
4. **Constant Velocity** works for smooth motion; use Constant Acceleration for aggressive maneuvers

### Particle Filter Tips:
1. **Number of particles**: 100-500 for 2D tracking; 1000+ for high-dimensional
2. **Resample threshold**: ESS < N/2 is typical
3. **Add noise after resampling** to maintain particle diversity
4. **Systematic resampling** preferred (lowest variance)
5. **Observation model**: Color works well for distinct objects; use edges/templates for texture

---

## References

1. **Kalman, R.E.** (1960). "A New Approach to Linear Filtering and Prediction Problems"
2. **Julier & Uhlmann** (1997). "New Extension of the Kalman Filter to Nonlinear Systems" (UKF)
3. **Isard & Blake** (1998). "CONDENSATION—Conditional Density Propagation for Visual Tracking"
4. **Gordon et al.** (1993). "Novel Approach to Nonlinear/Non-Gaussian Bayesian State Estimation"

---

## Further Reading

- **UKF**: No Jacobians, sigma points, better than EKF for nonlinear
- **JPDA**: Joint Probabilistic Data Association for multi-target tracking
- **PHD Filter**: Probability Hypothesis Density for unknown number of targets
- **SORT/DeepSORT**: Kalman + Hungarian matching for MOT (bridge to deep learning)