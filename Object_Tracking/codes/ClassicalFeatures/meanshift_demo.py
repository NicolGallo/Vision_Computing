"""
MeanShift and CamShift Tracker Demo
Demonstrates color histogram-based tracking using OpenCV

MeanShift: Tracks object by finding local maximum of color probability distribution
CamShift: Adaptive version that also handles scale and orientation changes
"""

import cv2
import numpy as np
import argparse
from pathlib import Path


class MeanShiftTracker:
    """
    MeanShift tracker using color histograms (HSV space)
    """
    def __init__(self, use_camshift=False, hist_bins=16):
        """
        Args:
            use_camshift: If True, use CamShift (adaptive); otherwise MeanShift
            hist_bins: Number of bins for histogram
        """
        self.use_camshift = use_camshift
        self.hist_bins = hist_bins
        
        # Termination criteria for meanShift/camShift
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        
        # Histogram and tracking state
        self.roi_hist = None
        self.track_window = None
        
    def init_tracking(self, frame, bbox):
        """
        Initialize tracker with first frame and bounding box
        
        Args:
            frame: First frame (BGR)
            bbox: Initial bounding box (x, y, w, h)
        """
        x, y, w, h = bbox
        self.track_window = tuple(bbox)
        
        # Extract ROI
        roi = frame[y:y+h, x:x+w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Create mask to ignore very dark/light regions
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        
        # Compute histogram in HSV space (Hue channel primarily)
        self.roi_hist = cv2.calcHist(
            [hsv_roi], [0], mask, [self.hist_bins], [0, 180]
        )
        cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)
        
        print(f"Initialized {'CamShift' if self.use_camshift else 'MeanShift'} tracker")
        print(f"ROI histogram shape: {self.roi_hist.shape}")
    
    def track(self, frame):
        """
        Track object in new frame
        
        Args:
            frame: Current frame (BGR)
            
        Returns:
            vis_frame: Frame with tracking visualization
            bbox: Updated bounding box (x, y, w, h) or None if tracking failed
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Back-project the histogram onto the frame
        dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)
        
        # Apply meanShift or camShift
        if self.use_camshift:
            # CamShift returns rotated rectangle
            ret, self.track_window = cv2.CamShift(dst, self.track_window, self.term_crit)
            
            # Extract box from rotated rectangle
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            
            # Draw rotated rectangle
            vis_frame = frame.copy()
            cv2.polylines(vis_frame, [pts], True, (0, 255, 0), 2)
            
            # Also draw axis for orientation
            center = (int(ret[0][0]), int(ret[0][1]))
            angle = ret[2]
            
            # Draw orientation line
            length = int(min(ret[1]) / 2)
            dx = int(length * np.cos(np.radians(angle)))
            dy = int(length * np.sin(np.radians(angle)))
            cv2.arrowedLine(vis_frame, center, (center[0] + dx, center[1] + dy),
                          (255, 0, 0), 2)
            
            # Extract regular bbox for return
            x, y, w, h = cv2.boundingRect(pts)
            bbox = (x, y, w, h)
        else:
            # MeanShift returns updated window
            ret, self.track_window = cv2.meanShift(dst, self.track_window, self.term_crit)
            
            x, y, w, h = self.track_window
            vis_frame = frame.copy()
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            bbox = (x, y, w, h)
        
        # Draw back-projection as overlay (optional visualization)
        dst_vis = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
        dst_vis = cv2.addWeighted(vis_frame, 0.7, dst_vis, 0.3, 0)
        
        # Add text info
        method = "CamShift" if self.use_camshift else "MeanShift"
        cv2.putText(dst_vis, f'{method} Tracking', (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return dst_vis, bbox


def select_roi_interactive(frame):
    """
    Allow user to select ROI by drawing a rectangle
    """
    print("Select ROI by dragging a rectangle, then press SPACE or ENTER")
    bbox = cv2.selectROI("Select Object to Track", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Object to Track")
    
    if bbox[2] > 0 and bbox[3] > 0:
        return tuple(map(int, bbox))
    else:
        return None


def main():
    parser = argparse.ArgumentParser(description='MeanShift/CamShift Tracker Demo')
    parser.add_argument('--video', type=str, default='0',
                       help='Video file or camera index (default: 0 for webcam)')
    parser.add_argument('--camshift', action='store_true',
                       help='Use CamShift instead of MeanShift (adapts scale/rotation)')
    parser.add_argument('--bbox', type=int, nargs=4, default=None,
                       metavar=('X', 'Y', 'W', 'H'),
                       help='Initial bounding box (x y w h). If not provided, select interactively')
    parser.add_argument('--bins', type=int, default=16,
                       help='Number of histogram bins (default: 16)')
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
        bbox = select_roi_interactive(frame)
        if bbox is None:
            print("Error: Invalid ROI selected")
            return
    
    # Initialize tracker
    tracker = MeanShiftTracker(use_camshift=args.camshift, hist_bins=args.bins)
    tracker.init_tracking(frame, bbox)
    
    # Setup video writer if output specified
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    method = "CamShift" if args.camshift else "MeanShift"
    print(f"{method} tracking started. Press 'q' to quit, 'r' to reset ROI")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame")
            break
        
        # Track
        vis_frame, bbox = tracker.track(frame)
        
        # Display
        cv2.imshow(f'{method} Tracker', vis_frame)
        
        # Write frame if output specified
        if writer:
            writer.write(vis_frame)
        
        # Handle keys
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset tracking with new ROI
            print("Resetting tracker - select new ROI...")
            bbox = select_roi_interactive(frame)
            if bbox:
                tracker.init_tracking(frame, bbox)
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Done!")


if __name__ == '__main__':
    main()