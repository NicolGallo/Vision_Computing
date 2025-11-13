"""
Kalman Filter Tracker Demo
Demonstrates Bayesian state estimation for object tracking with OpenCV

Features:
- Constant velocity motion model
- Gaussian noise handling
- Prediction during occlusion
- Visualization of prediction vs measurement
"""

import cv2
import numpy as np
import argparse
from pathlib import Path


class KalmanTracker:
    """
    Kalman Filter for 2D object tracking
    State: [x, y, vx, vy] (position and velocity)
    """
    def __init__(self, dt=1.0, process_noise=1e-2, measurement_noise=1e-1):
        """
        Args:
            dt: Time step (seconds per frame)
            process_noise: Process noise covariance (Q)
            measurement_noise: Measurement noise covariance (R)
        """
        self.dt = dt
        
        # Create Kalman filter (state_dim=4, measurement_dim=2)
        self.kf = cv2.KalmanFilter(4, 2, 0)
        
        # State transition matrix (Constant Velocity model)
        # [x, y, vx, vy]' = [1, 0, dt, 0; 0, 1, 0, dt; 0, 0, 1, 0; 0, 0, 0, 1] * [x, y, vx, vy]
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix (we only observe position)
        # [x_meas, y_meas]' = [1, 0, 0, 0; 0, 1, 0, 0] * [x, y, vx, vy]
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance Q
        # Assume small random accelerations
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        
        # Measurement noise covariance R
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        
        # Initial covariance P
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0
        
        # Track state
        self.initialized = False
        self.predicted_state = None
        self.corrected_state = None
        
    def init_tracking(self, measurement):
        """
        Initialize tracker with first measurement
        
        Args:
            measurement: [x, y] position
        """
        x, y = measurement
        
        # Initialize state [x, y, vx, vy] with zero velocity
        self.kf.statePre = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
        
        self.initialized = True
        self.corrected_state = measurement
        
        print(f"Kalman filter initialized at position ({x:.1f}, {y:.1f})")
    
    def predict(self):
        """
        Prediction step (time update)
        
        Returns:
            predicted_pos: Predicted [x, y] position
        """
        if not self.initialized:
            return None
        
        # Predict next state
        predicted = self.kf.predict()
        
        # Extract position
        self.predicted_state = np.array([predicted[0, 0], predicted[1, 0]])
        
        return self.predicted_state
    
    def update(self, measurement):
        """
        Update step (measurement update)
        
        Args:
            measurement: [x, y] measured position
            
        Returns:
            corrected_pos: Corrected [x, y] position
        """
        if not self.initialized:
            self.init_tracking(measurement)
            return measurement
        
        # Correct with measurement
        measurement_matrix = np.array([[measurement[0]], [measurement[1]]], dtype=np.float32)
        corrected = self.kf.correct(measurement_matrix)
        
        # Extract position
        self.corrected_state = np.array([corrected[0, 0], corrected[1, 0]])
        
        return self.corrected_state
    
    def get_velocity(self):
        """Get current velocity estimate"""
        if not self.initialized:
            return None
        
        state = self.kf.statePost
        return np.array([state[2, 0], state[3, 0]])
    
    def get_state_covariance(self):
        """Get uncertainty ellipse parameters"""
        if not self.initialized:
            return None
        
        # Extract position covariance (2x2 submatrix)
        P = self.kf.errorCovPost[:2, :2]
        
        # Compute eigenvalues and eigenvectors for uncertainty ellipse
        eigenvalues, eigenvectors = np.linalg.eig(P)
        
        # Major axis angle
        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
        
        # Axis lengths (2-sigma)
        axis_length = 2 * np.sqrt(eigenvalues)
        
        return angle, axis_length


def draw_uncertainty_ellipse(frame, center, angle, axis_length, color=(0, 255, 255)):
    """Draw uncertainty ellipse"""
    if axis_length is None:
        return
    
    center = tuple(map(int, center))
    axes = tuple(map(int, axis_length))
    angle_deg = int(np.degrees(angle))
    
    cv2.ellipse(frame, center, axes, angle_deg, 0, 360, color, 2)


def detect_object_simple(frame, prev_pos, search_radius=50):
    """
    Simple blob detector for demo
    In practice, use proper detection (e.g., background subtraction, blob detection)
    
    Returns:
        detected_pos: [x, y] or None if no detection
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Threshold to find bright regions (assuming tracking a bright object)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    # Find largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get centroid
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None
    
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    return np.array([cx, cy], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description='Kalman Filter Tracker Demo')
    parser.add_argument('--video', type=str, default='0',
                       help='Video file or camera index (default: 0 for webcam)')
    parser.add_argument('--dt', type=float, default=1.0,
                       help='Time step (inverse of fps)')
    parser.add_argument('--process_noise', type=float, default=1e-2,
                       help='Process noise covariance')
    parser.add_argument('--measurement_noise', type=float, default=1e-1,
                       help='Measurement noise covariance')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video file (optional)')
    parser.add_argument('--manual', action='store_true',
                       help='Manual mode: click to provide measurements')
    args = parser.parse_args()
    
    # Open video
    if args.video.isdigit():
        cap = cv2.VideoCapture(int(args.video))
    else:
        cap = cv2.VideoCapture(args.video)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps > 0:
        dt = 1.0 / fps
    else:
        dt = args.dt
    
    # Initialize tracker
    tracker = KalmanTracker(dt=dt, 
                           process_noise=args.process_noise,
                           measurement_noise=args.measurement_noise)
    
    # Setup video writer if output specified
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # For manual mode
    mouse_pos = None
    def mouse_callback(event, x, y, flags, param):
        nonlocal mouse_pos
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_pos = np.array([x, y], dtype=np.float32)
    
    if args.manual:
        cv2.namedWindow('Kalman Filter Tracker')
        cv2.setMouseCallback('Kalman Filter Tracker', mouse_callback)
        print("Manual mode: Click to provide measurements")
    
    print("Kalman Filter tracking started. Press 'q' to quit, 'r' to reset")
    print(f"Process noise: {args.process_noise}, Measurement noise: {args.measurement_noise}")
    
    trajectory_predicted = []
    trajectory_corrected = []
    trajectory_measured = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video")
            break
        
        vis_frame = frame.copy()
        
        # Get measurement
        if args.manual:
            measurement = mouse_pos
            mouse_pos = None  # Reset
        else:
            measurement = detect_object_simple(frame, 
                                               tracker.corrected_state if tracker.initialized else None)
        
        # Kalman filter cycle
        if tracker.initialized:
            # Predict
            predicted_pos = tracker.predict()
            
            if predicted_pos is not None:
                trajectory_predicted.append(predicted_pos.copy())
                
                # Draw prediction
                pred_pt = tuple(map(int, predicted_pos))
                cv2.circle(vis_frame, pred_pt, 8, (255, 0, 0), 2)  # Blue prediction
                cv2.putText(vis_frame, 'Predicted', (pred_pt[0] + 10, pred_pt[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                # Draw uncertainty ellipse
                angle, axis_length = tracker.get_state_covariance()
                draw_uncertainty_ellipse(vis_frame, predicted_pos, angle, axis_length, (255, 255, 0))
        
        # Update if measurement available
        if measurement is not None:
            trajectory_measured.append(measurement.copy())
            
            # Draw measurement
            meas_pt = tuple(map(int, measurement))
            cv2.circle(vis_frame, meas_pt, 6, (0, 255, 0), 2)  # Green measurement
            cv2.putText(vis_frame, 'Measured', (meas_pt[0] + 10, meas_pt[1] + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Correct
            corrected_pos = tracker.update(measurement)
            trajectory_corrected.append(corrected_pos.copy())
            
            # Draw corrected
            corr_pt = tuple(map(int, corrected_pos))
            cv2.circle(vis_frame, corr_pt, 7, (0, 0, 255), -1)  # Red corrected (filled)
            cv2.putText(vis_frame, 'Corrected', (corr_pt[0] + 10, corr_pt[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Draw velocity vector
            velocity = tracker.get_velocity()
            if velocity is not None:
                vel_scale = 10
                vel_end = corrected_pos + velocity * vel_scale
                cv2.arrowedLine(vis_frame, corr_pt, tuple(map(int, vel_end)),
                              (255, 0, 255), 2)
        else:
            # No measurement - show prediction only
            if tracker.initialized and tracker.predicted_state is not None:
                cv2.putText(vis_frame, 'NO MEASUREMENT - PREDICTING', (20, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw trajectories
        if len(trajectory_predicted) > 1:
            pts_pred = np.array(trajectory_predicted, dtype=np.int32)
            cv2.polylines(vis_frame, [pts_pred], False, (255, 0, 0), 1)
        
        if len(trajectory_corrected) > 1:
            pts_corr = np.array(trajectory_corrected, dtype=np.int32)
            cv2.polylines(vis_frame, [pts_corr], False, (0, 0, 255), 2)
        
        if len(trajectory_measured) > 1:
            pts_meas = np.array(trajectory_measured, dtype=np.int32)
            cv2.polylines(vis_frame, [pts_meas], False, (0, 255, 0), 1)
        
        # Info overlay
        cv2.putText(vis_frame, 'Kalman Filter Tracker', (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if tracker.initialized:
            state = tracker.kf.statePost
            info_text = f'State: x={state[0,0]:.1f}, y={state[1,0]:.1f}, vx={state[2,0]:.2f}, vy={state[3,0]:.2f}'
            cv2.putText(vis_frame, info_text, (20, vis_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display
        cv2.imshow('Kalman Filter Tracker', vis_frame)
        
        # Write frame if output specified
        if writer:
            writer.write(vis_frame)
        
        # Handle keys
        key = cv2.waitKey(30 if not args.manual else 1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("Resetting tracker...")
            tracker = KalmanTracker(dt=dt,
                                   process_noise=args.process_noise,
                                   measurement_noise=args.measurement_noise)
            trajectory_predicted.clear()
            trajectory_corrected.clear()
            trajectory_measured.clear()
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Done!")


if __name__ == '__main__':
    main()