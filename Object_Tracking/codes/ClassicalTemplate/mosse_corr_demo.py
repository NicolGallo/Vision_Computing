"""
MOSSE Correlation Filter Tracker Demo
Fast Fourier-domain correlation with online updates

Features:
- Grayscale, cosine (Hann) window, log+z-norm preprocessing
- Frequency-domain training: A=conj(F)*G, B=conj(F)*F
- Online update with learning rate
- Robust, real-time capable on CPU for modest sizes
"""

import cv2
import numpy as np
import argparse

def hann2d(w, h):
    return np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)

def gaussian_peak(size, sigma=2.0):
    h, w = size
    xs = np.arange(w) - (w // 2)
    ys = np.arange(h) - (h // 2)
    X, Y = np.meshgrid(xs, ys)
    g = np.exp(-(X**2 + Y**2) / (2 * sigma * sigma))
    # Center at (h/2,w/2); roll so peak is at (0,0) in spatial domain
    return np.roll(np.roll(g, -h//2, axis=0), -w//2, axis=1).astype(np.float32)

def preprocess(x, eps=1e-5):
    # x in grayscale float32 [0,255]
    x = np.log(x + 1.0)
    x = (x - x.mean()) / (x.std() + eps)
    return x

class MOSSETracker:
    def __init__(self, lr=0.125, reg=1e-4, peak_sigma=2.0):
        self.lr = float(lr)     # learning rate eta
        self.reg = float(reg)   # lambda
        self.peak_sigma = float(peak_sigma)

        self.A = None  # numerator accumulator (complex)
        self.B = None  # denominator accumulator (real/complex)
        self.H = None  # filter in freq
        self.win = None
        self.size = None  # (h,w)
        self.bbox = None  # (x,y,w,h)
        self.initialized = False

    def _crop_gray(self, frame, bbox):
        x, y, w, h = bbox
        H, W = frame.shape[:2]
        x = max(0, min(int(x), W - 1))
        y = max(0, min(int(y), H - 1))
        w = max(1, min(int(w), W - x))
        h = max(1, min(int(h), H - y))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return gray[y:y+h, x:x+w], (x, y, w, h)

    def _fft2(self, x):  return np.fft.fft2(x)
    def _ifft2(self, X): return np.fft.ifft2(X)

    def init_tracking(self, frame, bbox):
        patch, bbox = self._crop_gray(frame, bbox)
        h, w = patch.shape
        self.size = (h, w)
        self.win = hann2d(w, h)

        x = preprocess(patch) * self.win
        X = self._fft2(x)
        G = self._fft2(gaussian_peak((h, w), sigma=self.peak_sigma))

        # Initialize accumulators
        self.A = np.conj(X) * G
        self.B = np.conj(X) * X
        self.H = self.A / (self.B + self.reg)
        self.bbox = bbox
        self.initialized = True
        print(f"[MOSSE] Initialized with patch {w}x{h}")

    def _detect(self, frame, bbox):
        patch, bbox = self._crop_gray(frame, bbox)
        if patch.shape != self.size:
            patch = cv2.resize(patch, (self.size[1], self.size[0])).astype(np.float32)
        x = preprocess(patch) * self.win
        Z = self._fft2(x)
        R = Z * self.H  # frequency-domain correlation
        r = np.real(self._ifft2(R))
        (dy, dx) = np.unravel_index(np.argmax(r), r.shape)

        # Convert peak position to offset from center
        h, w = r.shape
        dx = dx if dx < w/2 else dx - w
        dy = dy if dy < h/2 else dy - h
        return int(dx), int(dy), r

    def _update(self, frame):
        # Update filter at current bbox
        patch, _ = self._crop_gray(frame, self.bbox)
        if patch.shape != self.size:
            patch = cv2.resize(patch, (self.size[1], self.size[0])).astype(np.float32)
        x = preprocess(patch) * self.win
        X = self._fft2(x)
        G = self._fft2(gaussian_peak(self.size, sigma=self.peak_sigma))

        self.A = (1 - self.lr) * self.A + self.lr * (np.conj(X) * G)
        self.B = (1 - self.lr) * self.B + self.lr * (np.conj(X) * X)
        self.H = self.A / (self.B + self.reg)

    def track(self, frame, return_response=False):
        if not self.initialized:
            return None, frame, None
        x, y, w, h = self.bbox

        dx, dy, r = self._detect(frame, self.bbox)
        # Shift bbox by (dx,dy)
        x += dx; y += dy
        self.bbox = (x, y, w, h)

        # Update model at new location
        self._update(frame)

        # Visualization
        vis = frame.copy()
        cv2.rectangle(vis, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
        cv2.putText(vis, f"MOSSE (dx,dy)=({dx},{dy})", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        if return_response:
            # Put small response map in a corner
            r_norm = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            r_color = cv2.applyColorMap(r_norm, cv2.COLORMAP_JET)
            hh, ww = r_color.shape[:2]
            hp = min(240, vis.shape[0]//3)
            wp = min(320, vis.shape[1]//3)
            r_small = cv2.resize(r_color, (wp, hp))
            vis[5:5+hp, vis.shape[1]-wp-5:vis.shape[1]-5] = r_small

        return self.bbox, vis, r


def main():
    ap = argparse.ArgumentParser(description="MOSSE Correlation Filter Tracker Demo")
    ap.add_argument("--video", type=str, default="0",
                    help="Video file or camera index (default: 0)")
    ap.add_argument("--bbox", type=int, nargs=4, default=None, metavar=("X","Y","W","H"),
                    help="Initial bounding box")
    ap.add_argument("--lr", type=float, default=0.125, help="Learning rate (eta)")
    ap.add_argument("--reg", type=float, default=1e-4, help="Regularization (lambda)")
    ap.add_argument("--sigma", type=float, default=2.0, help="Desired peak sigma")
    ap.add_argument("--response", action="store_true", help="Show response heatmap")
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

    tracker = MOSSETracker(lr=args.lr, reg=args.reg, peak_sigma=args.sigma)
    tracker.init_tracking(frame, bbox)

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        H, W = frame.shape[:2]
        writer = cv2.VideoWriter(args.output, fourcc, fps, (W, H))

    print("MOSSE tracking started. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        bbox, vis, _ = tracker.track(frame, return_response=args.response)
        cv2.putText(vis, "MOSSE Correlation Filter", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imshow("MOSSE Tracker", vis)
        if writer: writer.write(vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): break

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
