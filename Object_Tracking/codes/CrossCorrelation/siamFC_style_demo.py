# SiamFC-style cross-correlation demo (PyTorch) vs OpenCV matchTemplate
# pip install torch torchvision opencv-python numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

# ----------------------------
# Tiny backbone (SiamFC-style)
# ----------------------------
class TinyBackbone(nn.Module):
    """
    A very small conv stack to mimic SiamFC feature extractor.
    Output stride ~8 (depends on the exact config you want).
    """
    def __init__(self, in_ch=3, out_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 11, stride=2, padding=0), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),  # big early downsample
            nn.Conv2d(32, 32, 5, stride=1, padding=0),     nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, out_ch, 3, stride=1, padding=0), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)  # [N, C, H', W']

# -------------------------------------------
# SiamFC-style xcorr via depthwise conv trick
# -------------------------------------------
def xcorr_depthwise(z, x):
    """
    Cross-correlate template features z with search features x.
    Shapes:
      z: [N, C, Hz, Wz]   (template features; cached across frames)
      x: [N, C, Hx, Wx]   (search features; per frame)
    Returns:
      response: [N, 1, Hx-Hz+1, Wx-Wz+1]
    Implementation detail:
      PyTorch conv2d is actually cross-correlation (no kernel flip),
      so we can use grouped conv to perform per-channel correlation
      and then sum across channels.
    """
    N, C, Hz, Wz = z.shape
    _, _, Hx, Wx = x.shape

    # Reshape to use groups = N*C so each (N,C) channel correlates independently
    x_ = x.view(1, N * C, Hx, Wx)          # input channels = N*C
    z_ = z.view(N * C, 1, Hz, Wz)          # weights as (out=N*C, in=1, kH=Hz, kW=Wz)
    out = F.conv2d(x_, z_, groups=N * C)   # -> [1, N*C, Hout, Wout]
    out = out.view(N, C, out.size(-2), out.size(-1))
    # Sum over channel dimension to get the final response map per sample
    return out.sum(dim=1, keepdim=True)     # [N, 1, Hout, Wout]

# --------------------------
# Full SiamFC-style tracker
# --------------------------
class SiamFCDemo(nn.Module):
    def __init__(self, backbone=None):
        super().__init__()
        self.backbone = backbone or TinyBackbone(in_ch=3, out_ch=32)
        self._z = None  # cached template features

    @torch.no_grad()
    def set_template(self, z_img):
        """
        z_img: [1,3,127,127] (typical SiamFC template size; can vary)
        Cache template features for the whole sequence.
        """
        self._z = self.backbone(z_img)

    @torch.no_grad()
    def track(self, x_img):
        """
        x_img: [1,3,255,255] (typical SiamFC search size; can vary)
        Returns:
          response: [1,1,Hr,Wr], peak location as (row, col)
        """
        assert self._z is not None, "Call set_template() first."
        x = self.backbone(x_img)
        response = xcorr_depthwise(self._z, x)  # [1,1,*,*]
        # Argmax for location (coarse, pixel in response space)
        peak = torch.nonzero(response[0,0] == response.max())[0]
        peak_row, peak_col = int(peak[0]), int(peak[1])
        return response, (peak_row, peak_col)

# ---------------------------------------
# Minimal example + OpenCV juxtaposition
# ---------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build a synthetic template and search (as in the OpenCV example)
    def make_gray_template_and_search():
        rng = np.random.default_rng(0)
        th, tw = 63, 63
        H, W = 255, 255

        template = np.zeros((th, tw), np.float32)
        cv2.circle(template, (tw//2, th//2), 14, 1.0, -1)
        cv2.rectangle(template, (6, 6), (tw-7, th-7), 0.5, 3)
        cv2.GaussianBlur(template, (0, 0), 1.2, dst=template)

        search = (rng.normal(0.0, 0.05, (H, W))).astype(np.float32)
        gt_top, gt_left = 120, 80
        search[gt_top:gt_top+th, gt_left:gt_left+tw] += template
        return template, search, (gt_top, gt_left)

    tmpl_gray, search_gray, (gt_r, gt_c) = make_gray_template_and_search()

    # OpenCV normalized cross-correlation on raw grayscale (baseline)
    res = cv2.matchTemplate(search_gray, tmpl_gray, method=cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    cv_pred_r, cv_pred_c = max_loc[1], max_loc[0]

    print(f"[OpenCV] GT (r,c)=({gt_r},{gt_c}) | Pred=({cv_pred_r},{cv_pred_c}) | Peak={max_val:.3f}")

    # SiamFC-style: run through a tiny conv backbone, then xcorr in feature space
    # Convert grayscale to 3-channel (dummy) and to torch tensors
    def to_3ch_torch(img):
        # img: [H,W] float32 -> [1,3,H,W] normalized
        t = torch.from_numpy(img)[None, None, ...]  # [1,1,H,W]
        t = t.repeat(1, 3, 1, 1)                    # [1,3,H,W]
        # simple normalization to zero-mean per-channel
        t = (t - t.mean(dim=(2,3), keepdim=True)) / (t.std(dim=(2,3), keepdim=True) + 1e-6)
        return t

    # Crop a template and a search region akin to SiamFC sizes (can reuse the same sizes we created)
    z_img = to_3ch_torch(tmpl_gray).to(device)        # [1,3,th,tw] ~ [1,3,63,63]
    x_img = to_3ch_torch(search_gray).to(device)      # [1,3,255,255]

    model = SiamFCDemo().to(device)
    model.eval()
    model.set_template(z_img)                         # cache φ(z)
    response, (r_peak, c_peak) = model.track(x_img)  # compute φ(x) and xcorr

    # Map response-peak to image coords (coarse; stride depends on backbone).
    # For teaching, we just print response peak; in a real tracker you keep scale/stride bookkeeping.
    print(f"[SiamFC-PyTorch] Response shape: {tuple(response.shape)} | Peak index in response map: (r={r_peak}, c={c_peak})")

    # NOTE for lecture:
    # - OpenCV matchTemplate operates in the pixel domain.
    # - SiamFC operates in FEATURE space: φ(z) ⋆ φ(x), with φ shared and template features cached once.
    # - Depthwise conv trick computes all sliding positions in one shot, just like convolution.
