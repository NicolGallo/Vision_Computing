"""
Particle Filter (Condensation) Tracker Demo
Sequential Monte Carlo for nonlinear/non-Gaussian tracking

Features:
- Color histogram-based observation model
- Systematic resampling
- Multimodal posterior handling
- Visualization of particle cloud
"""

import cv2
import numpy as np
import argparse
from pathlib import Path


class ParticleFilterTracker:
    """
    Particle Filter for color-based tracking
    State: [x, y, vx, vy] (position and velocity)
    """
    def __init__(self, num_particles=300, state_dims=4, dt=1.0):
        """
        Args:
            num_particles: Number of particles
            state_dims: State dimensionality (4 for [x, y, vx, vy])
            dt: Time step
        """
        self.num_particles = num_particles
        self.state_dims = state_dims
        self.dt = dt
        
        # Particles and weights
        self.particles = np.zeros((num_particles, state_dims))
        self.weights = np.ones(num_particles) / num_particles
        
        # Motion model noise
        self.process_std = np.array([2.0, 2.0, 0.5, 0.5])  # [x, y, vx, vy]
        
        # Target histogram (HSV)
        self.target_hist = None
        self.hist_bins = 16
        
        # Track state
        self.initialized = False
        self.estimate = None
        
        # Resampling threshold
        self.resample_threshold = 0.5
        
    def init_tracking(self, frame, bbox):
        """
        Initialize particle filter with target appearance
        
        Args:
            frame: First frame (BGR)
            bbox: Bounding box (x, y, w, h)
        """
        x, y, w, h = bbox
        
        # Extract target histogram
        roi = frame[y:y+h, x:x+w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Compute HSV histogram (only Hue channel for simplicity)
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        self.target_hist = cv2.calcHist([hsv_roi], [0], mask, [self.hist_bins], [0, 180])
        cv2.normalize(self.target_hist, self.target_hist, 0, 1, cv2.NORM_MINMAX)
        
        # Initialize particles around initial position
        center_x = x + w / 2
        center_y = y + h / 2
        
        # Sample particles from Gaussian around initial position
        self.particles[:, 0] = np.random.normal(center_x, w / 4, self.num_particles)  # x
        self.particles[:, 1] = np.random.normal(center_y, h / 4, self.num_particles)  # y
        self.particles[:, 2] = np.random.normal(0, 1, self.num_particles)  # vx
        self.particles[:, 3] = np.random.normal(0, 1, self.num_particles)  # vy
        
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.estimate = np.array([center_x, center_y, w, h])
        
        self.initialized = True
        print(f"Particle filter initialized with {self.num_particles} particles")
    
    def predict(self):
        """
        Prediction step: propagate particles through motion model
        """
        if not self.initialized:
            return
        
        # Constant velocity model with noise
        # x_t = x_{t-1} + vx * dt + noise
        # y_t = y_{t-1} + vy * dt + noise
        # vx_t = vx_{t-1} + noise
        # vy_t = vy_{t-1} + noise
        
        self.particles[:, 0] += self.particles[:, 2] * self.dt + \
                                np.random.normal(0, self.process_std[0], self.num_particles)
        self.particles[:, 1] += self.particles[:, 3] * self.dt + \
                                np.random.normal(0, self.process_std[1], self.num_particles)
        self.particles[:, 2] += np.random.normal(0, self.process_std[2], self.num_particles)
        self.particles[:, 3] += np.random.normal(0, self.process_std[3], self.num_particles)
    
    def compute_observation_likelihood(self, frame, particle, window_size=40):
        """
        Compute observation likelihood for a particle using color histogram
        
        Args:
            frame: Current frame (BGR)
            particle: Particle state [x, y, vx, vy]
            window_size: Size of window to extract histogram
            
        Returns:
            likelihood: p(z_t | x_t)
        """
        x, y = int(particle[0]), int(particle[1])
        
        # Extract window around particle
        h, w = frame.shape[:2]
        x1 = max(0, x - window_size // 2)
        y1 = max(0, y - window_size // 2)
        x2 = min(w, x + window_size // 2)
        y2 = min(h, y + window_size // 2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Compute histogram
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        hist = cv2.calcHist([hsv_roi], [0], mask, [self.hist_bins], [0, 180])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        
        # Bhattacharyya coefficient
        similarity = cv2.compareHist(self.target_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
        
        # Convert to likelihood (Bhattacharyya distance -> similarity)
        likelihood = np.exp(-10 * similarity)
        
        return likelihood
    
    def update(self, frame):
        """
        Update step: weight particles based on observation likelihood
        
        Args:
            frame: Current frame (BGR)
        """
        if not self.initialized:
            return
        
        # Compute weights for all particles
        for i in range(self.num_particles):
            self.weights[i] = self.compute_observation_likelihood(frame, self.particles[i])
        
        # Normalize weights
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            # All particles have zero weight - reinitialize uniformly
            self.weights = np.ones(self.num_particles) / self.num_particles
    
    def estimate_state(self):
        """
        Compute state estimate as weighted average of particles
        
        Returns:
            estimate: [x, y, w, h] bounding box
        """
        if not self.initialized:
            return None
        
        # Weighted mean
        state_mean = np.average(self.particles, weights=self.weights, axis=0)
        
        # Estimate window size from particle spread
        particle_std = np.std(self.particles[:, :2], axis=0)
        w = max(20, int(particle_std[0] * 4))
        h = max(20, int(particle_std[1] * 4))
        
        self.estimate = np.array([state_mean[0], state_mean[1], w, h])
        return self.estimate
    
    def effective_sample_size(self):
        """
        Compute effective sample size (ESS)
        ESS = 1 / sum(w_i^2)
        """
        return 1.0 / np.sum(self.weights ** 2)
    
    def systematic_resample(self):
        """
        Systematic resampling (lowest variance)
        """
        cumsum = np.cumsum(self.weights)
        
        # Draw starting point
        u = np.random.uniform(0, 1.0 / self.num_particles)
        
        # Deterministic samples
        indices = np.zeros(self.num_particles, dtype=int)
        i = 0
        for j in range(self.num_particles):
            u_j = u + j / self.num_particles
            while u_j > cumsum[i]:
                i += 1
            indices[j] = i
        
        # Resample particles
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles
        
        # Add small noise to maintain diversity
        self.particles[:, :2] += np.random.normal(0, 1.0, (self.num_particles, 2))
    
    def track(self, frame):
        """
        Full tracking cycle: predict, update, resample, estimate
        
        Args:
            frame: Current frame (BGR)
            
        Returns:
            bbox: Estimated bounding box [x, y, w, h]
        """
        if not self.initialized:
            return None
        
        # Predict
        self.predict()
        
        # Update
        self.update(frame)
        
        # Estimate
        bbox = self.estimate_state()
        
        # Resample if needed
        ess = self.effective_sample_size()
        if ess < self.num_particles * self.resample_threshold:
            self.systematic_resample()
        
        return bbox


def draw_particles(frame, particles, weights, max_particles=100):
    """
    Draw particle cloud
    
    Args:
        frame: Frame to draw on
        particles: Particle array
        weights: Particle weights
        max_particles: Maximum number of particles to draw (for visualization)
    """
    # Draw only top particles
    indices = np.argsort(weights)[-max_particles:]
    
    for i in indices:
        x, y = int(particles[i, 0]), int(particles[i, 1])
        
        # Color based on weight (green = high, red = low)
        weight_normalized = weights[i] / weights[indices].max()
        color = (0, int(255 * weight_normalized), int(255 * (1 - weight_normalized)))
        
        cv2.circle(frame, (x, y), 2, color, -1)


def main():
    parser = argparse.ArgumentParser(description='Particle Filter Tracker Demo')
    parser.add_argument('--video', type=str, default='0',
                       help='Video file or camera index (default: 0 for webcam)')
    parser.add_argument('--num_particles', type=int, default=300,
                       help='Number of particles (default: 300)')
    parser.add_argument('--bbox', type=int, nargs=4, default=None,
                       metavar=('X', 'Y', 'W', 'H'),
                       help='Initial bounding box (x y w h). If not provided, select interactively')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video file (optional)')
    args = parser.parse_args()
    
    # Open video
    if args.video.isdigit():
        cap = cv2.VideoCapture(int(args.video))
    else:
        cap = cv2.VideoCapture(args.video)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        return
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        return
    
    # Get initial bounding box
    if args.bbox:
        bbox = tuple(args.bbox)
    else:
        print("Select object to track, then press SPACE or ENTER")
        bbox = cv2.selectROI("Select Object", frame, fromCenter=False)
        cv2.destroyWindow("Select Object")
        if bbox[2] == 0 or bbox[3] == 0:
            print("Error: Invalid ROI")
            return
        bbox = tuple(map(int, bbox))
    
    # Initialize tracker
    tracker = ParticleFilterTracker(num_particles=args.num_particles)
    tracker.init_tracking(frame, bbox)
    
    # Setup video writer if output specified
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    print(f"Particle Filter tracking started with {args.num_particles} particles")
    print("Press 'q' to quit, 'r' to reset")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video")
            break
        
        # Track
        bbox = tracker.track(frame)
        
        # Visualization
        vis_frame = frame.copy()
        
        if bbox is not None:
            # Draw bounding box
            x, y, w, h = map(int, bbox)
            cv2.rectangle(vis_frame, (x - w//2, y - h//2), (x + w//2, y + h//2),
                         (0, 255, 0), 2)
            
            # Draw center
            cv2.circle(vis_frame, (x, y), 5, (0, 0, 255), -1)
        
        # Draw particles
        draw_particles(vis_frame, tracker.particles, tracker.weights)
        
        # Info overlay
        cv2.putText(vis_frame, 'Particle Filter Tracker', (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        ess = tracker.effective_sample_size()
        info_text = f'Particles: {tracker.num_particles}, ESS: {ess:.1f}'
        cv2.putText(vis_frame, info_text, (20, vis_frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display
        cv2.imshow('Particle Filter Tracker', vis_frame)
        
        # Write frame if output specified
        if writer:
            writer.write(vis_frame)
        
        # Handle keys
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("Resetting tracker - select new ROI...")
            bbox = cv2.selectROI("Select Object", frame, fromCenter=False)
            if bbox[2] > 0 and bbox[3] > 0:
                tracker = ParticleFilterTracker(num_particles=args.num_particles)
                tracker.init_tracking(frame, tuple(map(int, bbox)))
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Done!")


if __name__ == '__main__':
    main()