"""
KLT (Kaasade-Lucas-Tomasi) Tracker Demo
Uses Shi-Tomasi corner detection + pyramidal Lucas-Kanade optical flow

Features:
- Sparse feature point tracking
- Automatic feature re-detection when points are lost
- Visualization of trajectories
"""

import cv2
import numpy as np
import argparse
from pathlib import Path

class KLTTracker:
    """
    KLT Tracker using OpenCV's goodFeaturesToTrack and calcOpticalFlowPyrLK
    """
    def __init__(self, 
                 max_corners=100,
                 quality_level=0.3,
                 min_distance=7,
                 block_size=7,
                 lk_win_size=(15, 15),
                 lk_max_level=2,
                 lk_criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)):
        """
        Args:
            max_corners: Maximum number of corners to detect
            quality_level: Quality level for Shi-Tomasi (0-1)
            min_distance: Minimum distance between corners
            block_size: Size of averaging block for corner detection
            lk_win_size: Window size for Lucas-Kanade
            lk_max_level: Number of pyramid levels
            lk_criteria: Termination criteria for LK
        """
        # Shi-Tomasi parameters
        self.feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=block_size
        )
        
        # Lucas-Kanade parameters
        self.lk_params = dict(
            winSize=lk_win_size,
            maxLevel=lk_max_level,
            criteria=lk_criteria
        )
        
        # Tracking state
        self.prev_gray = None
        self.prev_points = None
        self.trajectories = []
        self.colors = None
        self.min_points = 20  # Re-detect if below this
        
    def detect_features(self, gray_frame):
        """Detect good features to track using Shi-Tomasi corner detector"""
        return cv2.goodFeaturesToTrack(gray_frame, mask=None, **self.feature_params)
    
    def init_tracking(self, frame, roi=None):
        """
        Initialize tracking on first frame
        
        Args:
            frame: First frame (BGR)
            roi: Optional region of interest (x, y, w, h)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if roi is not None:
            # Create mask for ROI
            mask = np.zeros_like(gray)
            x, y, w, h = roi
            mask[y:y+h, x:x+w] = 255
            self.prev_points = cv2.goodFeaturesToTrack(
                gray, mask=mask, **self.feature_params
            )
        else:
            self.prev_points = self.detect_features(gray)
        
        self.prev_gray = gray
        
        # Initialize random colors for trajectories
        if self.prev_points is not None:
            n_points = len(self.prev_points)
            self.colors = np.random.randint(0, 255, (n_points, 3)).tolist()
            self.trajectories = [[p] for p in self.prev_points.reshape(-1, 2)]
    
    def track(self, frame):
        """
        Track features in new frame
        
        Args:
            frame: Current frame (BGR)
            
        Returns:
            frame: Frame with visualization
            n_tracked: Number of successfully tracked points
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis_frame = frame.copy()
        
        if self.prev_points is None or len(self.prev_points) == 0:
            # No points to track
            return vis_frame, 0
        
        # Calculate optical flow (Lucas-Kanade)
        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.prev_points,
            None,
            **self.lk_params
        )
        
        # Select good points
        if next_points is not None:
            good_new = next_points[status == 1]
            good_old = self.prev_points[status == 1]
            
            # Update trajectories and draw
            new_trajectories = []
            new_colors = []
            
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                a, b, c, d = int(a), int(b), int(c), int(d)
                
                # Find original index
                orig_idx = np.where(status == 1)[0][i]
                color = tuple(map(int, self.colors[orig_idx]))
                
                # Draw trajectory line
                cv2.line(vis_frame, (a, b), (c, d), color, 2)
                # Draw current point
                cv2.circle(vis_frame, (a, b), 5, color, -1)
                
                # Update trajectory
                new_trajectories.append(self.trajectories[orig_idx] + [(a, b)])
                new_colors.append(self.colors[orig_idx])
            
            self.trajectories = new_trajectories
            self.colors = new_colors
            self.prev_points = good_new.reshape(-1, 1, 2)
            
            # Re-detect features if too few
            if len(self.prev_points) < self.min_points:
                new_features = self.detect_features(gray)
                if new_features is not None:
                    # Add new features
                    self.prev_points = np.vstack([self.prev_points, new_features])
                    n_new = len(new_features)
                    new_colors_list = np.random.randint(0, 255, (n_new, 3)).tolist()
                    self.colors.extend(new_colors_list)
                    self.trajectories.extend([[p] for p in new_features.reshape(-1, 2)])
        
        self.prev_gray = gray.copy()
        
        # Draw info
        n_tracked = len(self.prev_points) if self.prev_points is not None else 0
        cv2.putText(vis_frame, f'Tracking {n_tracked} points', (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return vis_frame, n_tracked


def main():
    parser = argparse.ArgumentParser(description='KLT Tracker Demo')
    parser.add_argument('--video', type=str, default='0',
                       help='Video file or camera index (default: 0 for webcam)')
    parser.add_argument('--roi', action='store_true',
                       help='Select ROI for initial feature detection')
    parser.add_argument('--max_corners', type=int, default=100,
                       help='Maximum number of corners to detect')
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
    
    # Optional: Select ROI
    roi = None
    if args.roi:
        print("Select ROI for feature detection, then press SPACE or ENTER")
        roi = cv2.selectROI("Select ROI", frame, fromCenter=False)
        cv2.destroyWindow("Select ROI")
        if roi[2] > 0 and roi[3] > 0:  # Valid ROI
            roi = tuple(map(int, roi))
        else:
            roi = None
    
    # Initialize tracker
    tracker = KLTTracker(max_corners=args.max_corners)
    tracker.init_tracking(frame, roi)
    
    # Setup video writer if output specified
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    print("Tracking started. Press 'q' to quit, 'r' to reset")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame")
            break
        
        # Track
        vis_frame, n_tracked = tracker.track(frame)
        
        # Display
        cv2.imshow('KLT Tracker', vis_frame)
        
        # Write frame if output specified
        if writer:
            writer.write(vis_frame)
        
        # Handle keys
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset tracking
            print("Resetting tracker...")
            tracker.init_tracking(frame, roi)
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Done!")


if __name__ == '__main__':
    main()