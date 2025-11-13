# dff_demo_adaptive_annot.py
"""
DFF demo with:
  (1) Adaptive keyframe resets (AKR)
  (2) Minimal box annotator + tiny COCO-style loss for toy head

Usage examples
-------------
# Just run DFF (fixed interval)
python dff_demo_adaptive_annot.py --video_path ../vfencing.mp4 --visualize

# Run with adaptive keyframes
python dff_demo_adaptive_annot.py --video_path ../vfencing.mp4 --adaptive --visualize

# Annotate a few frames (OpenCV UI) and save JSON
python dff_demo_adaptive_annot.py --video_path ../vfencing.mp4 --annotate --anno_out fencing_boxes.json

# Train toy head from your boxes, then DFF inference (+viz)
python dff_demo_adaptive_annot.py --video_path ../vfencing.mp4 --train_boxes --anno fencing_boxes.json --epochs 5 --visualize
"""

import os, json, math, time, argparse
from typing import List, Tuple, Dict, Optional
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights

# headless-safe plotting
if os.environ.get("MPLBACKEND","") == "":
    try:
        import tkinter  # noqa
    except Exception:
        matplotlib.use("Agg")

# ------------------------------
# Core DFF module
# ------------------------------
class DeepFeatureFlow(nn.Module):
    def __init__(self, backbone: nn.Module, key_frame_interval: int = 10, flow_method: str = 'farneback',
                 device: Optional[torch.device] = None):
        super().__init__()
        self.backbone = backbone.eval()
        self.key_frame_interval = max(1, int(key_frame_interval))
        self.flow_method = flow_method
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.backbone.to(self.device)

        # Toy head: 1 obj logit + 4 box values (x1,y1,x2,y2) normalized to image size
        self.detection_head = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 5, 1)
        ).to(self.device)

    def compute_optical_flow(self, bgr1: np.ndarray, bgr2: np.ndarray) -> np.ndarray:
        if self.flow_method != 'farneback':
            raise NotImplementedError
        g1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2GRAY)
        return cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    def preprocess_frame(self, bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        im = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(im).to(self.device, dtype=torch.float32)
        t = t.permute(2,0,1).unsqueeze(0)/255.0
        mean = torch.tensor([0.485,0.456,0.406], device=self.device).view(1,3,1,1)
        std  = torch.tensor([0.229,0.224,0.225], device=self.device).view(1,3,1,1)
        return (t-mean)/std

    @torch.no_grad()
    def extract_backbone(self, bgr: np.ndarray) -> torch.Tensor:
        return self.backbone(self.preprocess_frame(bgr))  # (1,2048,7,7) for resnet50@224

    def warp_features(self, features: torch.Tensor, flow: np.ndarray) -> torch.Tensor:
        _, _, Hf, Wf = features.shape
        Hi, Wi = flow.shape[:2]
        flow_rs = cv2.resize(flow, (Wf, Hf), interpolation=cv2.INTER_LINEAR)
        flow_rs[...,0] *= (Wf/float(Wi))
        flow_rs[...,1] *= (Hf/float(Hi))
        dev = features.device; dt = features.dtype
        gy, gx = torch.meshgrid(
            torch.linspace(-1,1,Hf, device=dev, dtype=dt),
            torch.linspace(-1,1,Wf, device=dev, dtype=dt),
            indexing='ij'
        )
        base = torch.stack([gx,gy], dim=-1).unsqueeze(0)  # (1,Hf,Wf,2)
        ft = torch.from_numpy(flow_rs).to(dev, dt)
        flow_norm = torch.empty_like(base)
        flow_norm[...,0] = ft[...,0]/(Wf/2.0)
        flow_norm[...,1] = ft[...,1]/(Hf/2.0)
        grid = base + flow_norm
        return F.grid_sample(features, grid, mode='bilinear', padding_mode='border', align_corners=True)

    def detect_objects(self, feats: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.detection_head(feats)               # (1,5,Hf,Wf)
        obj = torch.sigmoid(logits[:,0:1])                # (1,1,Hf,Wf)
        boxes = logits[:,1:5]                             # normalized [x1,y1,x2,y2]
        return {"objectness": obj, "logits": logits, "boxes": boxes}
    
    # --- NUEVO MÉTODO AÑADIDO (Paso 1.1) ---
    @torch.no_grad()
    def postprocess_detections(self,
                               feats: torch.Tensor,
                               img_width: int,
                               img_height: int,
                               score_thr: float = 0.5,
                               topk: int = 20) -> np.ndarray:
        """
        Toma los feature maps de entrada (1,C,Hf,Wf) y devuelve un array
        de cajas [N, 5] en coordenadas de imagen [x1, y1, x2, y2, score].
        """
        # 1. Obtener predicciones de la red (en GPU)
        out = self.detect_objects(feats)

        # 2. Mover a CPU y numpy para post-procesado
        # .squeeze(0) elimina el batch_size=1
        obj = out["objectness"].squeeze(0).squeeze(0).cpu().numpy() # (Hf,Wf)
        box = out["boxes"].squeeze(0).cpu().numpy()                 # (4,Hf,Wf)

        # 3. Encontrar celdas que superan el umbral de confianza
        idx = np.argwhere(obj >= score_thr)

        # 4. Si no hay detecciones, devolver un array vacío
        if idx.size == 0:
            return np.empty((0, 5), dtype=np.float32)

        # 5. Filtrar las mejores 'k' detecciones si hay demasiadas
        # (Esto ayuda a evitar miles de cajas si el umbral es bajo)
        scores_at_idx = obj[idx[:, 0], idx[:, 1]]
        if idx.shape[0] > topk:
            top_indices = scores_at_idx.argsort()[-topk:] # Índices de los mejores scores
            idx = idx[top_indices] # Filtra las coordenadas 'idx'
            scores_at_idx = scores_at_idx[top_indices] # Filtra los scores

        # 6. Extraer coordenadas y scores de las celdas seleccionadas
        detections = []
        # Iteramos sobre las coordenadas (idx) y sus scores ya filtrados
        for (iy, ix), score in zip(idx, scores_at_idx):
            # (4,Hf,Wf) -> b[canal, iy, ix]
            b = box[:, iy, ix] # (4,) -> [x1_norm, y1_norm, x2_norm, y2_norm]

            # 7. Convertir de coordenadas normalizadas [0,1] a coordenadas de píxel
            x1 = b[0] * img_width
            y1 = b[1] * img_height
            x2 = b[2] * img_width
            y2 = b[3] * img_height

            detections.append([x1, y1, x2, y2, score])

        return np.array(detections, dtype=np.float32)

    # --- FIN DEL NUEVO MÉTODO ---

    def forward(self, frames_bgr: List[np.ndarray], adaptive: bool=False,
                diff_thresh: float=900.0, flow_thresh: float=6.0) -> List[torch.Tensor]:
        feats_all = []
        key_feats = None
        key_idx = 0
        prev = frames_bgr[0]
        for i, frame in enumerate(frames_bgr):
            force_key = False
            if adaptive and i>0:
                # grayscale MSE
                g1 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY).astype(np.float32)
                g2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                mse = np.mean((g1-g2)**2)
                # rough flow magnitude
                flow = self.compute_optical_flow(prev, frame)
                mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
                mag_mean = float(np.mean(mag))
                if mse > diff_thresh or mag_mean > flow_thresh:
                    force_key = True
                prev = frame

            if (i % self.key_frame_interval == 0) or force_key:
                key_feats = self.extract_backbone(frame)
                key_idx = i
                feats_all.append(key_feats)
            else:
                flow = self.compute_optical_flow(frames_bgr[key_idx], frame)
                feats_all.append(self.warp_features(key_feats, flow))
        return feats_all

# ------------------------------
# Plotting & diagnostics
# ------------------------------
def visualize_feature_energy(feats: torch.Tensor, title: str, path: Optional[str]=None):
    energy = feats.norm(dim=1, keepdim=False)[0].detach().cpu().numpy()
    plt.figure(figsize=(6,6)); plt.imshow(energy, cmap='viridis'); plt.colorbar()
    plt.title(title); plt.axis('off'); plt.tight_layout()
    if path: plt.savefig(path, dpi=140); plt.close()
    else: plt.show()

def visualize_flow(flow: np.ndarray, title: str, path: Optional[str]=None):
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv = np.zeros((*flow.shape[:2],3), dtype=np.uint8)
    hsv[...,0] = (ang*180/np.pi/2).astype(np.uint8)
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(mag, None, 0,255, cv2.NORM_MINMAX).astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    plt.figure(figsize=(8,6)); plt.imshow(rgb); plt.title(title); plt.axis('off'); plt.tight_layout()
    if path: plt.savefig(path, dpi=140); plt.close()
    else: plt.show()

def print_stats(tag: str, t: torch.Tensor):
    x = t.detach().float().cpu()
    print(f"{tag}: mean={x.mean():.5f} std={x.std():.5f} min={x.min():.5f} max={x.max():.5f}")

# ------------------------------
# Simple annotator (OpenCV UI)
# ------------------------------
class BoxAnnotator:
    def __init__(self, window="Annotator"):
        self.window = window
        self.drawing = False
        self.x0 = self.y0 = 0
        self.boxes = []  # list of (x1,y1,x2,y2)
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.x0, self.y0 = x, y
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            pass
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1, y1 = min(self.x0,x), min(self.y0,y)
            x2, y2 = max(self.x0,x), max(self.y0,y)
            if (x2-x1) > 3 and (y2-y1) > 3:
                self.boxes.append((x1,y1,x2,y2))

    def annotate_frame(self, frame_bgr: np.ndarray) -> List[Tuple[int,int,int,int]]:
        self.boxes = []
        disp = frame_bgr.copy()
        while True:
            tmp = disp.copy()
            # draw existing boxes
            for (x1,y1,x2,y2) in self.boxes:
                cv2.rectangle(tmp, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.imshow(self.window, tmp)
            k = cv2.waitKey(10) & 0xFF
            if k == ord('n'):     # next frame
                break
            if k == ord('c'):     # clear
                self.boxes = []
            if k == ord('q'):
                break
        return self.boxes

def run_annotator(video_path: str, out_json: str, stride: int = 10, max_frames: int = 300):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    annot = BoxAnnotator()
    idx = 0
    ann = {}  # frame_index -> list of boxes
    cnt = 0
    while cnt < max_frames:
        ok, frame = cap.read()
        if not ok: break
        if idx % stride == 0:
            print(f"[anno] frame {idx}: draw boxes (keys: n=next, c=clear, q=quit)")
            boxes = annot.annotate_frame(frame)
            ann[str(idx)] = boxes
        idx += 1
        cnt += 1
    cap.release()
    cv2.destroyAllWindows()
    with open(out_json, "w") as f:
        json.dump(ann, f)
    print(f"[anno] Saved to {out_json}")

# ------------------------------
# Box training (tiny COCO-style)
# ------------------------------
@torch.no_grad()
def extract_all_backbone_features(dff: DeepFeatureFlow, frames_bgr: List[np.ndarray]) -> torch.Tensor:
    feats = []
    for f in frames_bgr:
        feats.append(dff.extract_backbone(f))
    return torch.cat(feats, dim=0)  # (N,C,Hf,Wf)

def load_video_frames(video_path: str, max_frames: int = 300) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    frames = []
    while len(frames) < max_frames:
        ok, f = cap.read()
        if not ok: break
        frames.append(f)
    cap.release()
    return frames

# def make_targets_from_boxes(ann: Dict[str, List[List[int]]], frames: List[np.ndarray],
#                             Hf: int, Wf: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     """
#     Returns:
#       obj_tgt: (N,1,Hf,Wf) in {0,1}
#       box_tgt: (N,4,Hf,Wf) normalized [x1,y1,x2,y2] (valid only where obj=1)
#       mask   : (N,1,Hf,Wf) bool mask for positive cells
#     Cell is positive if its center (in image coords) lies inside any gt box.
#     """
#     N = len(frames)
#     obj = np.zeros((N, 1, Hf, Wf), dtype=np.float32)
#     box = np.zeros((N, 4, Hf, Wf), dtype=np.float32)

#     for i in range(N):
#         H, W = frames[i].shape[:2]
#         boxes = ann.get(str(i), [])
#         if not boxes: continue
#         # grid of cell centers in image coords
#         ys = (np.arange(Hf) + 0.5) * (H / Hf)
#         xs = (np.arange(Wf) + 0.5) * (W / Wf)
#         yy, xx = np.meshgrid(ys, xs, indexing='ij')  # (Hf,Wf)
#         pos = np.zeros((Hf, Wf), dtype=bool)
#         # for regression, use the *nearest* box (here: first box that contains the cell)
#         reg = np.zeros((4, Hf, Wf), dtype=np.float32)
#         for (x1,y1,x2,y2) in boxes:
#             inside = (xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)
#             pos |= inside
#             # assign normalized box to those cells
#             reg[0][inside] = x1 / W
#             reg[1][inside] = y1 / H
#             reg[2][inside] = x2 / W
#             reg[3][inside] = y2 / H
#         obj[i,0] = pos.astype(np.float32)
#         box[i] = reg

#     mask = obj.astype(bool)
#     return (torch.from_numpy(obj), torch.from_numpy(box), torch.from_numpy(mask))


def make_targets_from_boxes(ann, frames, Hf, Wf):
    """
    Returns:
      obj_tgt: (N,1,Hf,Wf) in {0,1}
      box_tgt: (N,4,Hf,Wf) normalized [x1,y1,x2,y2]
      mask   : (N,1,Hf,Wf) bool mask for positive cells
      has_pos_idx: 1D numpy array of frame indices that have >=1 positive cell
    """
    N = len(frames)
    obj = np.zeros((N, 1, Hf, Wf), dtype=np.float32)
    box = np.zeros((N, 4, Hf, Wf), dtype=np.float32)
    has_pos = []

    for i in range(N):
        H, W = frames[i].shape[:2]
        boxes = ann.get(str(i), [])
        if not boxes:
            continue
        ys = (np.arange(Hf) + 0.5) * (H / Hf)
        xs = (np.arange(Wf) + 0.5) * (W / Wf)
        yy, xx = np.meshgrid(ys, xs, indexing='ij')  # (Hf,Wf)

        pos = np.zeros((Hf, Wf), dtype=bool)
        reg = np.zeros((4, Hf, Wf), dtype=np.float32)
        for (x1,y1,x2,y2) in boxes:
            inside = (xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)
            pos |= inside
            reg[0][inside] = x1 / W; reg[1][inside] = y1 / H
            reg[2][inside] = x2 / W; reg[3][inside] = y2 / H

        if pos.any():
            has_pos.append(i)
            obj[i,0] = pos.astype(np.float32)
            box[i]    = reg

    mask = obj.astype(bool)
    has_pos_idx = np.array(has_pos, dtype=np.int64)
    return (torch.from_numpy(obj), torch.from_numpy(box),
            torch.from_numpy(mask), has_pos_idx)






# def train_toy_head_from_boxes(dff: DeepFeatureFlow, frames: List[np.ndarray], anno_path: str,
#                               epochs: int = 5, lr: float = 1e-3, batch_size: int = 16):
#     with open(anno_path, "r") as f:
#         ann = json.load(f)

#     dff.backbone.eval()
#     for p in dff.backbone.parameters(): p.requires_grad_(False)
#     dff.detection_head.train()

#     with torch.no_grad():
#         feats = extract_all_backbone_features(dff, frames)  # (N,C,Hf,Wf)
#     N, C, Hf, Wf = feats.shape

#     obj_tgt, box_tgt, mask = make_targets_from_boxes(ann, frames, Hf, Wf)
#     device = dff.device
#     feats = feats.to(device)
#     obj_tgt = obj_tgt.to(device)
#     box_tgt = box_tgt.to(device)
#     mask = mask.to(device)

#     opt = torch.optim.Adam(dff.detection_head.parameters(), lr=lr, weight_decay=1e-5)
#     bce = nn.BCEWithLogitsLoss()
#     sl1 = nn.SmoothL1Loss(reduction='none')

#     idx = torch.arange(N, device=device)
#     for ep in range(1, epochs+1):
#         perm = idx[torch.randperm(N)]
#         loss_b, loss_r = 0.0, 0.0
#         t0 = time.time()
#         for s in range(0, N, batch_size):
#             b = perm[s:s+batch_size]
#             x = feats[b]                        # (B,C,Hf,Wf)
#             y_obj = obj_tgt[b]                  # (B,1,Hf,Wf)
#             y_box = box_tgt[b]                  # (B,4,Hf,Wf)
#             m = mask[b]                         # (B,1,Hf,Wf)

#             out = dff.detection_head(x)         # (B,5,Hf,Wf)
#             obj_logit = out[:,0:1]
#             box_pred  = out[:,1:5]

#             Lobj = bce(obj_logit, y_obj)
#             # SmoothL1 only where positives
#             m4 = m.expand_as(box_pred)
#             Lbox = sl1(box_pred, y_box)
#             Lbox = (Lbox * m4.float()).sum() / (m4.float().sum() + 1e-6)

#             loss = Lobj + 5.0*Lbox
#             opt.zero_grad(set_to_none=True)
#             loss.backward()
#             opt.step()

#             loss_b += float(Lobj.detach().cpu())
#             loss_r += float(Lbox.detach().cpu())
#         t1 = time.time()
#         print(f"[train boxes] epoch {ep:02d} | obj={loss_b:.4f} box={loss_r:.4f} | time={t1-t0:.2f}s")

#     dff.detection_head.eval()


def train_toy_head_from_boxes(dff, frames, anno_path, epochs=5, lr=1e-3, batch_size=16):
    import json
    with open(anno_path, "r") as f:
        ann = json.load(f)

    # freeze backbone
    dff.backbone.eval()
    for p in dff.backbone.parameters(): p.requires_grad_(False)
    dff.detection_head.train()

    with torch.no_grad():
        feats = extract_all_backbone_features(dff, frames)  # (N,C,Hf,Wf)
    N, C, Hf, Wf = feats.shape

    obj_tgt, box_tgt, mask, has_pos_idx = make_targets_from_boxes(ann, frames, Hf, Wf)
    if has_pos_idx.size == 0:
        print("[train boxes] no positive frames found in annotations — nothing to train.")
        dff.detection_head.eval()
        return

    # keep only frames with positives
    feats   = feats[has_pos_idx]
    obj_tgt = obj_tgt[has_pos_idx]
    box_tgt = box_tgt[has_pos_idx]
    mask    = mask[has_pos_idx]

    device = dff.device
    feats   = feats.to(device)
    obj_tgt = obj_tgt.to(device)
    box_tgt = box_tgt.to(device)
    mask    = mask.to(device)

    opt = torch.optim.Adam(dff.detection_head.parameters(), lr=lr, weight_decay=1e-5)
    sl1 = nn.SmoothL1Loss(reduction='none')

    idx = torch.arange(feats.size(0), device=device)
    for ep in range(1, epochs+1):
        perm = idx[torch.randperm(idx.numel())]
        loss_b, loss_r = 0.0, 0.0
        t0 = time.time()
        for s in range(0, perm.numel(), batch_size):
            b = perm[s:s+batch_size]
            x = feats[b]                        # (B,C,Hf,Wf)
            y_obj = obj_tgt[b]                  # (B,1,Hf,Wf)
            y_box = box_tgt[b]                  # (B,4,Hf,Wf)
            m = mask[b]                         # (B,1,Hf,Wf)

            out = dff.detection_head(x)         # (B,5,Hf,Wf)
            obj_logit = out[:,0:1]
            box_pred  = out[:,1:5]

            # ----- Balanced BCE -----
            # compute per-batch positive/negative counts
            pos = y_obj.sum()
            neg = y_obj.numel() - pos
            # avoid division by zero
            pos_w = (neg / (pos + 1e-6)).clamp(1.0, 100.0)
            bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
            Lobj = bce(obj_logit, y_obj)

            # box regression only on positives
            m4 = m.expand_as(box_pred).float()
            Lbox = sl1(box_pred, y_box)
            Lbox = (Lbox * m4).sum() / (m4.sum() + 1e-6)

            loss = Lobj + 5.0 * Lbox
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            loss_b += float(Lobj.detach().cpu())
            loss_r += float(Lbox.detach().cpu())
        t1 = time.time()
        print(f"[train boxes] epoch {ep:02d} | obj={loss_b:.4f} box={loss_r:.4f} | time={t1-t0:.2f}s")

    dff.detection_head.eval()


def _probe_fps(path, default=30):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    cap.release()
    return fps if fps > 0 else default

def save_overlay_video(frames_bgr, dff, feats, out_path="overlay.mp4", thr=0.6, topk=5, fps=30):
    H, W = frames_bgr[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")   # fallback: 'XVID'
    vw = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    with torch.no_grad():
        for f, F in zip(frames_bgr, feats):
            """
            out = dff.detect_objects(F)
            obj = out["objectness"][0,0].detach().cpu().numpy()   # (Hf,Wf)
            box = out["boxes"][0].detach().cpu().numpy()          # (4,Hf,Wf) in [0,1]
            idx = np.argwhere(obj >= thr)
            if idx.size == 0:
                flat = obj.reshape(-1)
                top = flat.argsort()[-topk:]
                idx = np.stack(np.unravel_index(top, obj.shape), axis=1)
            vis = f.copy()
            for (iy, ix) in idx[:topk]:
                x1 = int(box[0,iy,ix] * W); y1 = int(box[1,iy,ix] * H)
                x2 = int(box[2,iy,ix] * W); y2 = int(box[3,iy,ix] * H)
                cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(vis, f"{obj[iy,ix]:.2f}", (x1, max(0,y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
            """
            # --- NUEVA LÓGICA (Paso 1.1) ---
            vis = f.copy()
            # Usamos los 'thr' y 'topk' de los argumentos de la función
            detections = dff.postprocess_detections(F, W, H, score_thr=thr, topk=topk)

            for (x1, y1, x2, y2, score) in detections:
                # Convertimos a int solo para dibujar
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(vis, f"{score:.2f}", (x1, max(0,y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
            # --- FIN NUEVA LÓGICA ---

            vw.write(vis)
    vw.release()
    print(f"[DFF] Saved overlay video to: {out_path}")




# ------------------------------
# Main pipelines
# ------------------------------
def process_video(video_path: str, key_interval: int, adaptive: bool, visualize: bool,
                  max_frames: int, save_viz_dir: Optional[str]):
    # load frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise RuntimeError(f"Could not open {video_path}")
    frames = []
    while len(frames) < max_frames:
        ok, f = cap.read()
        if not ok: break
        frames.append(f)
    cap.release()
    print(f"Loaded {len(frames)} frames from {video_path}")

    resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    backbone = nn.Sequential(*list(resnet.children())[:-2])
    dff = DeepFeatureFlow(backbone=backbone, key_frame_interval=key_interval, flow_method='farneback')

    t0 = time.time()
    feats = dff.forward(frames, adaptive=adaptive)
    t1 = time.time()
    n = len(frames); n_key = math.floor((n-1)/key_interval) + 1
    print(f"Inference: {(t1-t0):.3f}s | frames={n} | theo speedup≈{n/max(1,n_key):.2f}x")
    print_stats("C5 key stats", feats[0])

    if visualize and n >= 2:
        mid = min(key_interval//2 if key_interval>1 else 1, n-1)
        os.makedirs(save_viz_dir or ".", exist_ok=True)
        visualize_feature_energy(feats[0], "Key C5 energy",
                                 None if not save_viz_dir else os.path.join(save_viz_dir, "c5_key.png"))
        visualize_feature_energy(feats[mid], f"Prop C5 energy (frame {mid})",
                                 None if not save_viz_dir else os.path.join(save_viz_dir, f"c5_prop_{mid}.png"))
        with torch.no_grad():
            o0 = dff.detect_objects(feats[0]); om = dff.detect_objects(feats[mid])
        for tag, obj in [("key", o0["objectness"]), ("prop", om["objectness"])]:
            m = obj[0,0].detach().cpu().numpy()
            plt.figure(figsize=(6,6)); plt.imshow(m, vmin=0, vmax=1); plt.colorbar()
            plt.title(f"Toy head objectness — {tag}"); plt.axis('off'); plt.tight_layout()
            if save_viz_dir: plt.savefig(os.path.join(save_viz_dir, f"obj_{tag}.png"), dpi=140); plt.close()
            else: plt.show()
        flow = dff.compute_optical_flow(frames[0], frames[mid])
        visualize_flow(flow, f"Optical Flow (0→{mid})",
                       None if not save_viz_dir else os.path.join(save_viz_dir, f"flow_0_{mid}.png"))

def main():
    ap = argparse.ArgumentParser("DFF demo with adaptive keyframes + annotation training")
    ap.add_argument("--video_path", type=str, required=True)
    ap.add_argument("--key_interval", type=int, default=10)
    ap.add_argument("--adaptive", action="store_true", help="enable adaptive keyframe resets")
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--max_frames", type=int, default=300)
    ap.add_argument("--save_viz_dir", type=str, default=None)

    # annotation options
    ap.add_argument("--annotate", action="store_true", help="open CV UI to draw boxes")
    ap.add_argument("--anno_out", type=str, default="boxes.json")
    ap.add_argument("--anno_stride", type=int, default=10)

    # training from boxes
    ap.add_argument("--train_boxes", action="store_true")
    ap.add_argument("--anno", type=str, default=None, help="path to boxes.json")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)

    ap.add_argument("--save_overlay", type=str, default=None,
                    help="Path to write overlay MP4 (e.g., vfencing_overlay.mp4)")
    ap.add_argument("--overlay_thr", type=float, default=0.6)
    ap.add_argument("--overlay_topk", type=int, default=5)




    args = ap.parse_args()

    if args.annotate:
        run_annotator(args.video_path, args.anno_out, stride=args.anno_stride, max_frames=args.max_frames)
        return

    # If training is requested, do it first, then run DFF
    if args.train_boxes:
        if not args.anno:
            raise SystemExit("Please pass --anno path/to/boxes.json")
        # load frames once
        frames = load_video_frames(args.video_path, max_frames=args.max_frames)
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        backbone = nn.Sequential(*list(resnet.children())[:-2])
        dff = DeepFeatureFlow(backbone=backbone, key_frame_interval=args.key_interval, flow_method='farneback')
        train_toy_head_from_boxes(dff, frames, args.anno, epochs=args.epochs, lr=args.lr)
        # After training, run DFF inference (with the trained head inside dff)
        # Reuse same frames for speed
        t0 = time.time()
        feats = dff.forward(frames, adaptive=args.adaptive)
        # OPTIONAL overlay
        if args.save_overlay:
            fps = _probe_fps(args.video_path, default=30)
            save_overlay_video(frames, dff, feats,
                            out_path=args.save_overlay,
                            thr=args.overlay_thr,
                            topk=args.overlay_topk,
                            fps=fps)





        t1 = time.time()
        print(f"Inference after training: {(t1-t0):.3f}s on {len(frames)} frames")
        # quick viz
        if args.visualize and len(frames) >= 2:
            os.makedirs(args.save_viz_dir or ".", exist_ok=True)
            mid = min(args.key_interval//2 if args.key_interval>1 else 1, len(frames)-1)
            with torch.no_grad():
                o0 = dff.detect_objects(feats[0]); om = dff.detect_objects(feats[mid])
            for tag, obj in [("key", o0["objectness"]), ("prop", om["objectness"])]:
                m = obj[0,0].detach().cpu().numpy()
                plt.figure(figsize=(6,6)); plt.imshow(m, vmin=0, vmax=1); plt.colorbar()
                plt.title(f"Toy head objectness — {tag}"); plt.axis('off'); plt.tight_layout()
                if args.save_viz_dir: plt.savefig(os.path.join(args.save_viz_dir, f"obj_{tag}.png"), dpi=140); plt.close()
                else: plt.show()
        return

    # Plain DFF run (optionally adaptive)
    process_video(args.video_path, args.key_interval, args.adaptive, args.visualize, args.max_frames, args.save_viz_dir)

if __name__ == "__main__":
    main()



"""
1)  Annotate first to train the head
(opens an OpenCV window every 8th frame; press n to advance, c to clear, q to quit)
python dff_demo_adaptive_annot.py --video_path ../vfencing.mp4 --annotate --anno_out fencing_boxes.json --anno_stride 8

2)  Train the toy head from your boxes, then run DFF with adaptive keyframes and visualization
python dff_demo_adaptive_annot.py --video_path ../vfencing.mp4 --train_boxes --anno fencing_boxes.json --epochs 5 --visualize --adaptive


3)  Just run inference later (no training), e.g. to demo speedup

python dff_demo_adaptive_annot.py --video_path ../vfencing.mp4  --adaptive  --key_interval 10  --visualize  --save_viz_dir ./viz_run

  """