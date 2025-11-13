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

    def forward(self, frames_bgr: List[np.ndarray], adaptive: bool=False,
                diff_thresh: float=900.0, flow_thresh: float=6.0) -> List[torch.Tensor]:
        """
        Runs DFF.
        Returns:
            feats_all (List[torch.Tensor]): List of (1,C,Hf,Wf) features for each frame.
            adaptive_cuts (List[int]): List of frame indices where adaptive logic triggered a keyframe.
        """

        feats_all = []
        adaptive_cuts = []  #Aspecto añadido
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
                    if i > 0:
                        adaptive_cuts.append(i)  #Aspecto añadido

                prev = frame

            if (i % self.key_frame_interval == 0) or force_key:
                key_feats = self.extract_backbone(frame)
                key_idx = i
                feats_all.append(key_feats)
            else:
                flow = self.compute_optical_flow(frames_bgr[key_idx], frame)
                feats_all.append(self.warp_features(key_feats, flow))
        
        return feats_all,adaptive_cuts  #Aspecto modificado

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
            vw.write(vis)
    vw.release()
    print(f"[DFF] Saved overlay video to: {out_path}")


# ------------------------------
# APLICACIÓN 2 - FUNCIONES DE AYUDA PARA FEATURE TRACKING
# ------------------------------

@torch.no_grad()
def find_best_match_cosine(target_vec_norm: torch.Tensor, feat_map: torch.Tensor) -> Tuple[int, int]:
    """
    Busca el vector más similar usando Similitud de Coseno (más robusto).
    - target_vec_norm: (1, C, 1, 1) - El vector objetivo, ya normalizado.
    - feat_map: (1, C, Hf, Wf) - El mapa de features propagado donde buscamos.
    """
    # 1. Normalizar todos los vectores en el mapa de features
    #    F.normalize(tensor, p=2, dim=1) normaliza a lo largo del canal C
    feat_map_norm = F.normalize(feat_map, p=2, dim=1)
    
    # 2. Calcular la similitud de coseno
    #    Esto es un truco: F.conv2d con el target_vec como "filtro"
    #    es equivalente a un producto punto denso (dot product).
    #    Como ambos están normalizados, dot(A, B) = Similitud de Coseno.
    #    feat_map_norm es (1, C, Hf, Wf)
    #    target_vec_norm es (1, C, 1, 1) -> actuará como un kernel de 1x1
    
    # torch.Size([1, C, 1, 1]) -> torch.Size([1, C, 1, 1])
    # DFF.py:537: UserWarning: A 'weights' tensor was provided, which will be used as the 'kernel'
    # for 'conv2d'. This scenario is going to be deprecated, please provide
    # the 'weights' as a 'kernel'. (Triggered internally at ..\aten\src\ATen\native\Convolution.cpp:831.)
    
    # Para evitar el 'Warning', podemos cambiar el orden:
    # 'weight' (el filtro) debe ser (out_channels, in_channels, kH, kW)
    # Nuestro filtro es 'target_vec_norm'
    # Lo reordenamos para que sea (1, C, 1, 1)
    
    # target_vec_norm ya está en formato (1, C, 1, 1), ¡perfecto!
    
    # El 'weight' (filtro) debe ser (out_channels, in_channels, kH, kW)
    #   out_channels = 1 (solo queremos 1 mapa de similitud)
    #   in_channels = C
    #   kH, kW = 1, 1
    # target_vec_norm tiene forma (1, C, 1, 1) lo que es (1, C, 1, 1)
    # ¡Perfecto!
    
    # O... más simple. El 'target_vec_norm' que pasamos es (C,).
    # Re-hagamos esta función para que sea más clara.
    
    # ---- INICIO DE LA NUEVA LÓGICA (MÁS SIMPLE) ----
    
    C, Hf, Wf = feat_map.shape[1:4]
    
    # 1. Normalizar el mapa de features (el target ya viene normalizado)
    feat_map_norm = F.normalize(feat_map, p=2, dim=1) # (1, C, Hf, Wf)
    
    # 2. Reformar ambos para el producto punto (dot product)
    #    Target: (C,) -> (C, 1)
    #    Mapa: (1, C, Hf, Wf) -> (C, Hf*Wf)
    target_for_dot = target_vec_norm.view(C, 1)
    map_for_dot = feat_map_norm.view(C, Hf * Wf)
    
    # 3. Calcular producto punto (que es Similitud de Coseno)
    #    (C, Hf*Wf).T @ (C, 1) -> (Hf*Wf, C) @ (C, 1) -> (Hf*Wf, 1)
    similarity_map_flat = torch.matmul(map_for_dot.T, target_for_dot)
    
    # 4. Encontrar el MÁXIMO (argmax), no el mínimo
    max_idx = similarity_map_flat.view(-1).argmax()
    
    # 5. Convertir índice plano a coordenadas (iy, ix)
    iy, ix = np.unravel_index(max_idx.cpu().numpy(), (Hf, Wf))
    return int(iy), int(ix)

    # ---- FIN DE LA NUEVA LÓGICA ----

def _coords_img_to_feat(x_img: int, y_img: int, H_img: int, W_img: int, 
                        Hf: int = 7, Wf: int = 7) -> Tuple[int, int]:
    """Mapea un clic (x,y) en la imagen a una celda (iy, ix) en el mapa de features."""
    iy = int((y_img / H_img) * Hf)
    ix = int((x_img / W_img) * Wf)
    # Asegurarse de que no nos salimos de los límites
    return min(iy, Hf - 1), min(ix, Wf - 1)

def _coords_feat_to_img(iy_feat: int, ix_feat: int, H_img: int, W_img: int,
                        Hf: int = 7, Wf: int = 7) -> Tuple[int, int]:
    """Mapea el centro de una celda (iy, ix) de features de vuelta a píxeles (x,y) en la imagen."""
    y_img = (iy_feat + 0.5) * (H_img / Hf)
    x_img = (ix_feat + 0.5) * (W_img / Wf)
    return int(x_img), int(y_img)


# Datos globales para el callback del ratón (es la forma más simple con cv2)
_click_data = {"x": -1, "y": -1, "clicked": False}

def _on_mouse_click(event, x, y, flags, param):
    """Callback de CV2 para guardar un solo clic."""
    global _click_data
    if event == cv2.EVENT_LBUTTONDOWN:
        _click_data["x"] = x
        _click_data["y"] = y
        _click_data["clicked"] = True

def run_feature_tracking(dff: DeepFeatureFlow, frames: List[np.ndarray], out_video_path: str, video_path_for_fps: str):
    """
    Función principal para la App 2.
    1. Pide al usuario que haga clic en el frame 0.
    2. Extrae el vector de features objetivo de ese clic.
    3. Propaga features y rastrea el punto en todos los frames.
    4. Guarda el resultado en un vídeo.
    """
    global _click_data
    window_name = "Feature Tracker: Haz clic en un punto y presiona 's' para empezar"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, _on_mouse_click)

    frame0 = frames[0]
    H_img, W_img = frame0.shape[:2]
    Hf, Wf = 7, 7 # Asumimos ResNet@224 -> (7,7) mapa de features
    
    print("\n--- Feature Tracker ---")
    print(f"Mostrando Frame 0. Haz clic en el objeto que quieres seguir.")
    print("Presiona 's' para empezar el seguimiento, o 'q' para salir.")
    
    vis = frame0.copy()
    
    # 1. Bucle de selección de punto
    while True:
        # Dibuja un círculo de feedback donde se ha hecho clic
        if _click_data["clicked"]:
            vis = frame0.copy() # Resetea la imagen
            cv2.circle(vis, (_click_data["x"], _click_data["y"]), 5, (0, 255, 0), -1)
            cv2.putText(vis, "Target Locked!", (_click_data["x"] + 10, _click_data["y"] + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow(window_name, vis)
        k = cv2.waitKey(20) & 0xFF
        
        if k == ord('q'): # Salir
            cv2.destroyAllWindows()
            print("Seguimiento cancelado.")
            return
        if k == ord('s') and _click_data["clicked"]: # Empezar
            cv2.destroyAllWindows()
            print(f"¡Empezando seguimiento! Target en ({_click_data['x']}, {_click_data['y']})")
            break
            
    # 2. Extracción del Vector Objetivo
    print("Extrayendo características del KeyFrame...")
    key_feats = dff.extract_backbone(frame0) # (1, C, Hf, Wf)
    
    # Mapear el clic a coords de features
    x_click, y_click = _click_data["x"], _click_data["y"]
    iy_target, ix_target = _coords_img_to_feat(x_click, y_click, H_img, W_img, Hf, Wf)
    
    # Extraer el vector objetivo
    target_vector = key_feats[0, :, iy_target, ix_target].detach() # (C,)
    # --- (NUEVO) Normalizar el vector objetivo para la similitud de coseno ---
    target_vector_norm = F.normalize(target_vector, p=2, dim=0) # (C,)
    print(f"Vector objetivo (dim={target_vector.shape[0]}) extraído de la celda ({iy_target}, {ix_target}).")

    # 3. Bucle de Seguimiento y Propagación (Guardando en un vídeo)
    fps = _probe_fps(video_path_for_fps, default=30) # Reutilizamos la función de `main`
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_video_path, fourcc, fps, (W_img, H_img))

    print(f"Procesando {len(frames)} frames... (esto puede tardar un momento)")
    
    # Frame 0 ya lo tenemos
    current_x, current_y = x_click, y_click
    cv2.circle(frame0, (current_x, current_y), 7, (0, 255, 0), 2)
    vw.write(frame0)
    
    t_start = time.time()
    for i in range(1, len(frames)):
        frame_i = frames[i]
        
        # --- El Corazón de DFF ---
        # A. Calcular Flujo Óptico (Keyframe -> Frame i)
        flow = dff.compute_optical_flow(frame_i, frames[0])
        
        # B. Propagar Features (Deformar key_feats usando el flujo)
        propagated_feats = dff.warp_features(key_feats, flow) # (1, C, Hf, Wf)
        
        # C. Encontrar la mejor coincidencia
        iy_new, ix_new = find_best_match_cosine(target_vector, propagated_feats)
        
        # D. Mapear de vuelta a coords de imagen
        current_x, current_y = _coords_feat_to_img(iy_new, ix_new, H_img, W_img, Hf, Wf)
        
        # Dibujar el resultado
        vis = frame_i.copy()
        cv2.circle(vis, (current_x, current_y), 7, (0, 255, 0), 2) # Círculo de seguimiento
        cv2.putText(vis, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        vw.write(vis)

    t_end = time.time()
    vw.release()
    print(f"\n--- Seguimiento Completo ---")
    print(f"Procesados {len(frames)-1} frames en {t_end - t_start:.2f} segundos.")
    print(f"¡Vídeo de seguimiento guardado en: {out_video_path}!")

# ------------------------------
# Fin de la aplicacion 2 - Feature Tracking
# ------------------------------


# ------------------------------
# Main pipelines
# ------------------------------
def process_video(video_path: str, key_interval: int, adaptive: bool, visualize: bool,
                  max_frames: int, save_viz_dir: Optional[str], diff_thresh: float, flow_thresh: float): ## Añadidos diff_thresh, flow_thresh 
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

    feats, cuts = dff.forward(frames, adaptive=adaptive, diff_thresh=diff_thresh, flow_thresh=flow_thresh)  ## Añadidos diff_thresh, flow_thresh
    
    t1 = time.time()
    n = len(frames); n_key = math.floor((n-1)/key_interval) + 1
    print(f"Inference: {(t1-t0):.3f}s | frames={n} | theo speedup≈{n/max(1,n_key):.2f}x")

    if adaptive and cuts: # Aspecto modificado
        print(f"Adaptive keyframes triggered at {len(cuts)} frames: {cuts}")

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

    # New args for adaptive thresholds
    ap.add_argument("--detect_shots", action="store_true", help="Run standalone shot boundary detection and exit")
    ap.add_argument("--diff_thresh", type=float, default=900.0, help="MSE threshold for shot detection")
    ap.add_argument("--flow_thresh", type=float, default=6.0, help="Flow magnitude threshold for shot detection")

    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--max_frames", type=int, default=300)
    ap.add_argument("--save_viz_dir", type=str, default=None)

    #New args for feature tracking
    ap.add_argument("--track_feature", action="store_true", help="Run feature tracking app")

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
    
    # -----------------------------------------------------------------
    # --- BLOQUE NUEVO: APLICACIÓN 1 (DETECTOR DE CAMBIOS DE ESCENA) ---
    # -----------------------------------------------------------------
    if args.detect_shots:
        print(f"Running Shot Boundary Detection on {args.video_path}...")
        print(f"(Params: MSE_thresh={args.diff_thresh}, Flow_thresh={args.flow_thresh})\n")
        
        frames = load_video_frames(args.video_path, max_frames=args.max_frames)
        
        # Nota: Podríamos optimizar esto para no cargar el backbone de ResNet,
        # pero por simplicidad reutilizamos la clase DFF completa, ya que
        # el backbone no se ejecutará (key_interval es infinito).
        # OJO: Si el vídeo es muy largo, esto es ineficiente.
        # Una implementación 'en producción' solo usaría 'compute_optical_flow'.
        
        # Truco: Ponemos un intervalo de keyframe infinito para que
        # 'forward' solo se guíe por la lógica adaptativa.
        infinite_interval = len(frames) + 1
        
        # No necesitamos la red neuronal para esto, pero sí la clase
        # por el método 'compute_optical_flow'.
        # Pasamos un backbone 'None' y lo manejamos.
        
        # --- Vamos a hacerlo más limpio y sin cargar ResNet ---
        
        # Instanciamos la clase DFF solo para usar 'compute_optical_flow'
        # No cargamos el backbone de verdad.
        dff_dummy = DeepFeatureFlow(backbone=nn.Identity(), key_frame_interval=1)
        
        t0 = time.time()
        detected_cuts = []

        all_mse = [0.0]             # <-- (NUEVO) Guardaremos las métricas aquí
        all_flow_mag = [0.0]        # <-- (NUEVO) (Empezamos con 0 para el frame 0)
        
        prev = frames[0]
        
        for i in range(1, len(frames)):
            frame = frames[i]
            
            # 1. Grayscale MSE
            g1 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY).astype(np.float32)
            g2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            mse = np.mean((g1-g2)**2)
            
            # 2. Flow Magnitude
            flow = dff_dummy.compute_optical_flow(prev, frame)
            mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
            mag_mean = float(np.mean(mag))

            all_mse.append(mse)             # <-- (NUEVO)
            all_flow_mag.append(mag_mean)   # <-- (NUEVO)
            
            if mse > args.diff_thresh or mag_mean > args.flow_thresh:
                detected_cuts.append(i)
                print(f"  -> Cut found at frame {i:04d} (MSE={mse:,.1f}, Flow={mag_mean:.2f})")

            prev = frame
        
        t1 = time.time()
        
        print(f"\n--- Detection Complete (took {t1-t0:.2f}s) ---")
        print(f"Found {len(detected_cuts)} potential shot boundaries at frames:")
        print(detected_cuts)
        
        out_path = "shot_cuts.json"
        with open(out_path, "w") as f:
            json.dump({"video": args.video_path, "cuts": detected_cuts,
                       "params": {"mse_thresh": args.diff_thresh, "flow_thresh": args.flow_thresh}
                      }, f, indent=2)
        print(f"Results saved to {out_path}")

        # --- (NUEVO) INICIO DEL BLOQUE DE GRÁFICO ---
        print("Generating metrics plot...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
        frame_indices = np.arange(len(frames))
        
        # Gráfico 1: MSE
        ax1.plot(frame_indices, all_mse, label="Frame-to-Frame MSE", color="blue")
        ax1.axhline(y=args.diff_thresh, color='r', linestyle='--', label=f"MSE Threshold ({args.diff_thresh})")
        ax1.set_ylabel("MSE")
        ax1.set_title(f"Shot Detection Metrics for {args.video_path}")
        ax1.legend(loc='upper left')
        ax1.grid(True, linestyle=':', alpha=0.6)

        # Gráfico 2: Flujo Óptico
        ax2.plot(frame_indices, all_flow_mag, label="Mean Flow Magnitude", color="green")
        ax2.axhline(y=args.flow_thresh, color='r', linestyle='--', label=f"Flow Threshold ({args.flow_thresh})")
        ax2.set_xlabel("Frame Number")
        ax2.set_ylabel("Flow Magnitude")
        ax2.legend(loc='upper left')
        ax2.grid(True, linestyle=':', alpha=0.6)
        
        # Marcar los cortes detectados en ambos gráficos
        for cut_frame in detected_cuts:
            ax1.axvline(x=cut_frame, color='red', linestyle='-', linewidth=2, alpha=0.7, label="Detected Cut")
            ax2.axvline(x=cut_frame, color='red', linestyle='-', linewidth=2, alpha=0.7)
        
        # Evitar etiquetas duplicadas en la leyenda
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys(), loc='upper left')
        
        plot_path = "shot_detection_plot.png"
        plt.savefig(plot_path)
        print(f"Metrics plot saved to {plot_path}")
        # plt.show() # Descomenta esto si quieres que el gráfico se muestre al ejecutar
        plt.close()
        # --- (NUEVO) FIN DEL BLOQUE DE GRÁFICO ---

        return # Salir del script
    # -------------------------------------------------
    # --- FIN DEL BLOQUE NUEVO ---
    # -------------------------------------------------


    # -----------------------------------------------------------------
    # --- BLOQUE NUEVO: APLICACIÓN 2 (FEATURE TRACKER) ---
    # -----------------------------------------------------------------
    if args.track_feature:
        print("Iniciando modo Feature Tracker...")
        frames = load_video_frames(args.video_path, max_frames=args.max_frames)
        
        # 1. Inicializar DFF (necesitamos el backbone real esta vez)
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        backbone = nn.Sequential(*list(resnet.children())[:-2])
        dff = DeepFeatureFlow(backbone=backbone, key_frame_interval=args.key_interval, flow_method='farneback')
        
        # 2. Definir nombre de salida
        base_name = os.path.splitext(os.path.basename(args.video_path))[0]
        out_video_path = f"{base_name}_tracked.mp4"
        
        # 3. Ejecutar la lógica de seguimiento
        run_feature_tracking(dff, frames, out_video_path, args.video_path)
        return # Salir del script
    # -------------------------------------------------
    # --- FIN DEL BLOQUE NUEVO ---
    # -------------------------------------------------


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
        feats, cuts = dff.forward(frames, adaptive=args.adaptive,
                                  diff_thresh=args.diff_thresh,
                                  flow_thresh=args.flow_thresh)  ## Añadidos diff_thresh, flow_thresh
        
        if adaptive and cuts:
            print(f"Adaptive keyframes triggered at {len(cuts)} frames: {cuts}")

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
    process_video(args.video_path, args.key_interval, args.adaptive, args.visualize, args.max_frames, args.save_viz_dir, diff_thresh=args.diff_thresh, flow_thresh=args.flow_thresh)

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