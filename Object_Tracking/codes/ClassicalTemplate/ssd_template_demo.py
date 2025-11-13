"""
SSD Template Matching Tracker Demo
Simple sliding-window baseline using OpenCV matchTemplate

Features:
- SSD/SQDIFF-NORMED score (lower is better)
- Search in a padded window around last bbox
- Optional heatmap visualization
"""

import cv2
import numpy as np
import argparse

class SSDTemplateTracker:
    def __init__(self, method="SQDIFF_NORMED", search_factor=2.0):
        """
        Args:
            method: 'SQDIFF' or 'SQDIFF_NORMED'
            search_factor: search region expansion relative to bbox
        """
        self.method = method
        self.search_factor = float(search_factor)
        self.template = None
        self.bbox = None  # (x, y, w, h)
        self.initialized = False

        if method == "SQDIFF":
            self.cv_method = cv2.TM_SQDIFF
        else:
            self.cv_method = cv2.TM_SQDIFF_NORMED  # default

    @staticmethod
    def _clamp_rect(x, y, w, h, W, H):
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(1, min(w, W - x))
        h = max(1, min(h, H - y))
        return x, y, w, h

    def init_tracking(self, frame, bbox):
        x, y, w, h = bbox
        H, W = frame.shape[:2]
        x, y, w, h = self._clamp_rect(int(x), int(y), int(w), int(h), W, H)

        # Use grayscale for robustness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.template = gray[y:y+h, x:x+w].copy()
        self.bbox = (x, y, w, h)
        self.initialized = True
        print(f"[SSD] Initialized with template size {self.template.shape[::-1]}")

    def _search_rect(self, frame):
        """Expand around current bbox."""
        x, y, w, h = self.bbox
        cx, cy = x + w // 2, y + h // 2
        sw, sh = int(self.search_factor * w), int(self.search_factor * h)

        H, W = frame.shape[:2]
        x1 = max(0, cx - sw // 2); y1 = max(0, cy - sh // 2)
        x2 = min(W, cx + sw // 2);  y2 = min(H, cy + sh // 2)
        if x2 <= x1 or y2 <= y1:
            return 0, 0, W, H
        return x1, y1, x2 - x1, y2 - y1

    def track(self, frame, visualize_heatmap=False):
        if not self.initialized:
            return None, frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tx, ty, tw, th = self.bbox
        sx, sy, sw, sh = self._search_rect(frame)

        search = gray[sy:sy+sh, sx:sx+sw]
        if search.shape[0] < th or search.shape[1] < tw:
            # If search too small, fall back to whole image
            search = gray
            sx, sy = 0, 0

        # matchTemplate: for SQDIFF, min is best
        res = cv2.matchTemplate(search, self.template, self.cv_method)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)
        best = minLoc if self.cv_method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED) else maxLoc

        # Map back to frame coords
        new_x = sx + best[0]
        new_y = sy + best[1]
        self.bbox = (new_x, new_y, tw, th)

        vis = frame.copy()
        cv2.rectangle(vis, (new_x, new_y), (new_x + tw, new_y + th), (0, 255, 0), 2)
        cv2.putText(vis, f"SSD score: {minVal:.4f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        if visualize_heatmap:
            # Normalize response for display
            norm = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            heat = cv2.applyColorMap(255 - norm, cv2.COLORMAP_JET)  # invert so red=best
            # Resize heatmap to search region size (res is (sh-th+1, sw-tw+1))
            hh, ww = heat.shape[:2]
            canvas = vis.copy()
            # place heatmap in top-right corner
            hp = min(240, vis.shape[0]//3)
            wp = min(320, vis.shape[1]//3)
            heat_small = cv2.resize(heat, (wp, hp))
            canvas[5:5+hp, vis.shape[1]-wp-5:vis.shape[1]-5] = heat_small
            vis = canvas

        return self.bbox, vis


def main():
    ap = argparse.ArgumentParser(description="SSD Template Matching Tracker Demo")
    ap.add_argument("--video", type=str, default="0",
                    help="Video file or camera index (default: 0)")
    ap.add_argument("--bbox", type=int, nargs=4, default=None, metavar=("X","Y","W","H"),
                    help="Initial bounding box")
    ap.add_argument("--search_factor", type=float, default=2.0,
                    help="Search window expansion factor (default: 2.0)")
    ap.add_argument("--method", type=str, default="SQDIFF_NORMED",
                    choices=["SQDIFF","SQDIFF_NORMED"], help="SSD variant")
    ap.add_argument("--heatmap", action="store_true", help="Show match heatmap")
    ap.add_argument("--output", type=str, default=None, help="Optional output mp4")
    args = ap.parse_args()

    cap = cv2.VideoCapture(int(args.video)) if args.video.isdigit() else cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: cannot open {args.video}")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: empty video")
        return

    if args.bbox:
        bbox = tuple(args.bbox)
    else:
        print("Select ROI, press ENTER/SPACE")
        bbox = cv2.selectROI("Select Object", frame, fromCenter=False)
        cv2.destroyWindow("Select Object")
        if bbox[2] == 0 or bbox[3] == 0:
            print("Invalid ROI")
            return
        bbox = tuple(map(int, bbox))

    tracker = SSDTemplateTracker(method=args.method, search_factor=args.search_factor)
    tracker.init_tracking(frame, bbox)

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        H, W = frame.shape[:2]
        writer = cv2.VideoWriter(args.output, fourcc, fps, (W, H))

    print("SSD template tracking started. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        bbox, vis = tracker.track(frame, visualize_heatmap=args.heatmap)
        cv2.putText(vis, "SSD Template Tracker", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imshow("SSD Tracker", vis)
        if writer: writer.write(vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): break

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
