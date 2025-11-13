# stmV4_demo.py — STM, training-free mask propagation (with annotator)

import os, json, argparse
from typing import List, Optional
import cv2, numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F

# ============== STM modules (keys + values + RAW MASK) ==============

class STMEncoder(nn.Module):
    """Tiny encoders: key from image; value from image+mask."""
    def __init__(self, in_channels=3, key_channels=128, value_channels=256):
        super().__init__()
        self.key_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, key_channels, 3, padding=1),
        )
        self.value_encoder = nn.Sequential(
            nn.Conv2d(in_channels + 1, 64, 7, stride=2, padding=3), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, value_channels, 3, padding=1),
        )

    def forward(self, image: torch.Tensor, mask: Optional[torch.Tensor] = None):
        key = self.key_encoder(image)
        if mask is None:
            return key, None
        if mask.shape[2:] != image.shape[2:]:
            mask = F.interpolate(mask, size=image.shape[2:], mode='bilinear', align_corners=False)
        val = self.value_encoder(torch.cat([image, mask], dim=1))
        return key, val


class MemoryBank:
    def __init__(self, max_size=20):
        self.max_size = max_size
        self.keys: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []   # will include raw mask channel at the end
        self.frame_ids: List[int] = []
    def add(self, key, value, frame_id):
        self.keys.append(key); self.values.append(value); self.frame_ids.append(frame_id)
        if len(self.keys) > self.max_size:
            self.keys.pop(0); self.values.pop(0); self.frame_ids.pop(0)
    def get_all_keys(self):   return None if not self.keys else torch.cat(self.keys,   dim=2)
    def get_all_values(self): return None if not self.values else torch.cat(self.values, dim=2)
    def clear(self): self.keys, self.values, self.frame_ids = [], [], []
    def size(self): return len(self.keys)


# class MemoryReader(nn.Module):
#     """Non-local read with optional top-k and AMP; returns read value tensor."""
#     def __init__(self, key_channels=128, value_channels=257, top_k: Optional[int] = 128):
#         super().__init__()
#         self.Ck, self.Cv, self.top_k = key_channels, value_channels, top_k

#     def forward(self, query_key, memory_keys, memory_values):
#         B, Ck, Hq, Wq = query_key.shape
#         _, Cv, Hm, Wm = memory_values.shape
#         Nq, Nm = Hq*Wq, Hm*Wm
#         Q = query_key.view(B, Ck, Nq); K = memory_keys.view(B, Ck, Nm); V = memory_values.view(B, Cv, Nm)

#         use_amp = query_key.is_cuda and torch.cuda.is_available()
#         dtype_logits = torch.float16 if use_amp else Q.dtype
#         with torch.autocast(device_type='cuda', dtype=dtype_logits, enabled=use_amp):
#             logits = torch.bmm(Q.transpose(1,2), K) / (Ck**0.5)  # (B,Nq,Nm)
#             if self.top_k is not None and 0 < self.top_k < Nm:
#                 scores, idx = torch.topk(logits, k=self.top_k, dim=2)     # (B,Nq,K)
#                 attn = F.softmax(scores, dim=2).to(V.dtype)               # (B,Nq,K)
#                 Vexp = V.unsqueeze(1).expand(B, Nq, Cv, Nm)               # (B,Nq,Cv,Nm)
#                 idx_exp = idx.unsqueeze(2).expand(B, Nq, Cv, self.top_k)  # (B,Nq,Cv,K)
#                 Vtop = torch.gather(Vexp, 3, idx_exp)                     # (B,Nq,Cv,K)
#                 read = (Vtop * attn.unsqueeze(2)).sum(dim=3)              # (B,Nq,Cv)
#                 read = read.transpose(1,2).contiguous().view(B, Cv, Hq, Wq)
#                 return read
#             attn = F.softmax(logits, dim=2).to(V.dtype)                   # (B,Nq,Nm)
#             read = torch.bmm(V, attn.transpose(1,2)).view(B, Cv, Hq, Wq)
#             return read
class MemoryReader(nn.Module):
    """Non-local read with cosine-normalized keys + temperature; top-k per query."""
    def __init__(self, key_channels=128, value_channels=257, top_k: Optional[int] = 32, tau: float = 0.07):
        super().__init__()
        self.Ck, self.Cv, self.top_k, self.tau = key_channels, value_channels, top_k, max(1e-4, float(tau))

    def forward(self, query_key, memory_keys, memory_values):
        B, Ck, Hq, Wq = query_key.shape
        _, Cv, Hm, Wm = memory_values.shape
        Nq, Nm = Hq * Wq, Hm * Wm

        # Flatten
        Q = query_key.view(B, Ck, Nq)             # (B,Ck,Nq)
        K = memory_keys.view(B, Ck, Nm)           # (B,Ck,Nm)
        V = memory_values.view(B, Cv, Nm)         # (B,Cv,Nm)

        # Cosine normalize keys
        Qn = F.normalize(Q, dim=1)
        Kn = F.normalize(K, dim=1)

        # Temperature-scaled cosine similarity
        logits = torch.bmm(Qn.transpose(1, 2), Kn) / self.tau   # (B,Nq,Nm)

        # Top-k per query to avoid diffuse attention
        if self.top_k is not None and 0 < self.top_k < Nm:
            scores, idx = torch.topk(logits, k=self.top_k, dim=2)                    # (B,Nq,K)
            scores = scores - scores.max(dim=2, keepdim=True).values                 # stabilize
            attn = F.softmax(scores, dim=2).to(V.dtype)                               # (B,Nq,K)

            # Gather values
            Vexp = V.unsqueeze(1).expand(B, Nq, Cv, Nm)                               # (B,Nq,Cv,Nm)
            idx_exp = idx.unsqueeze(2).expand(B, Nq, Cv, self.top_k)                  # (B,Nq,Cv,K)
            Vtop = torch.gather(Vexp, 3, idx_exp)                                     # (B,Nq,Cv,K)
            read = (Vtop * attn.unsqueeze(2)).sum(dim=3)                              # (B,Nq,Cv)
            read = read.transpose(1, 2).contiguous().view(B, Cv, Hq, Wq)              # (B,Cv,Hq,Wq)
            return read

        logits = logits - logits.max(dim=2, keepdim=True).values
        attn = F.softmax(logits, dim=2).to(V.dtype)                                   # (B,Nq,Nm)
        read = torch.bmm(V, attn.transpose(1, 2)).view(B, Cv, Hq, Wq)
        return read


class STM(nn.Module):
    """
    Training-free STM:
      - Store key, (value_features + raw_mask_channel_downsampled) in memory.
      - Read value with attention; take the LAST channel as the propagated mask.
      - Optional blur to smooth.
    """
    def __init__(self, key_channels=128, value_feat_channels=256, memory_size=20, top_k=128):
        super().__init__()
        # +1 channel at the end for raw mask
        self.encoder = STMEncoder(3, key_channels, value_feat_channels)
        self.reader  = MemoryReader(key_channels, value_feat_channels + 1, top_k=32, tau=0.07)
        self.memory  = MemoryBank(memory_size)
        self.pool    = nn.AvgPool2d(2,2,ceil_mode=True)   # low-res attention
        self.blur    = nn.Conv2d(1,1,3,padding=1,bias=False)
        with torch.no_grad():
            self.blur.weight[:] = torch.tensor([[[[1,2,1],[2,4,2],[1,2,1]]]], dtype=torch.float32) / 16.0

    @torch.no_grad()
    def _downsample_mask_to(self, mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        # mask: (B,1,H,W), like: (B,C,H',W')
        return F.interpolate(mask, size=like.shape[2:], mode='nearest')

    @torch.no_grad()
    def initialize_memory(self, image: torch.Tensor, mask: torch.Tensor, frame_id: int):
        self.memory.clear()
        k, v = self.encoder(image, mask)                                  # v: (B,Cv,H',W')
        m_ds = self._downsample_mask_to(mask, k)                          # (B,1,H',W')
        v_plus_mask = torch.cat([v, m_ds], dim=1)                         # (B,Cv+1,H',W')
        self.memory.add(k, v_plus_mask, frame_id)

    @torch.no_grad()
    def add_memory(self, image: torch.Tensor, mask: torch.Tensor, frame_id: int):
        k, v = self.encoder(image, mask)
        m_ds = self._downsample_mask_to(mask, k)
        v_plus_mask = torch.cat([v, m_ds], dim=1)
        self.memory.add(k, v_plus_mask, frame_id)

    @torch.no_grad()
    def segment_frame(self, image: torch.Tensor, frame_id: int) -> torch.Tensor:
        qk_full, _ = self.encoder(image, None)                 # (B,Ck,H',W')
        qk = self.pool(qk_full)

        Mk_all = self.memory.get_all_keys()
        Mv_all = self.memory.get_all_values()

        # Pool memory keys
        Mk = self.pool(Mk_all)

        # Avg-pool features, Max-pool mask channel
        feat = Mv_all[:, :-1, :, :]                            # (B,Cv,Htot,W')
        msk  = Mv_all[:, -1:, :, :]                            # (B,1, Htot,W')
        feat_p = self.pool(feat)
        msk_p  = F.max_pool2d(msk, kernel_size=2, stride=2, ceil_mode=True)
        Mv = torch.cat([feat_p, msk_p], dim=1)                 # (B,Cv+1,h,w)

        # Read
        read = self.reader(qk, Mk, Mv)                         # (B,Cv+1,h,w)

        # Last channel is mask; clamp + light blur
        read_mask_lr = read[:, -1:, :, :].clamp_(0, 1)         # (B,1,h,w)
        read_mask_lr = self.blur(read_mask_lr)

        # Min-max normalize per frame to avoid "all ~0.5"
        mmin = read_mask_lr.amin(dim=(2,3), keepdim=True)
        mmax = read_mask_lr.amax(dim=(2,3), keepdim=True)
        read_mask_lr = (read_mask_lr - mmin) / (mmax - mmin + 1e-6)

        mask_hr = F.interpolate(read_mask_lr, size=image.shape[2:], mode='bilinear', align_corners=False)
        return mask_hr



# ================= Utils: video IO, overlay, annotation, boxes =================

def _probe_fps(path, default=30):
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS) or 0; cap.release()
    return fps if fps > 0 else default

def _read_video_frames(path, max_frames=999999, resize_short=360):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): raise RuntimeError(f"Could not open video: {path}")
    frames = []
    while len(frames) < max_frames:
        ok, f = cap.read(); 
        if not ok: break
        H, W = f.shape[:2]
        if resize_short is not None:
            if H < W: newH, newW = resize_short, int(W*(resize_short/H))
            else:     newW, newH = resize_short, int(H*(resize_short/W))
            f = cv2.resize(f, (newW, newH), interpolation=cv2.INTER_LINEAR)
        frames.append(f)
    cap.release()
    return frames

def _bgr_to_tensor(f_bgr):  # (1,3,H,W) RGB float32
    rgb = cv2.cvtColor(f_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).float().permute(2,0,1).unsqueeze(0) / 255.0
    return t

# def _mask_from_boxes_json(json_path, frame_index, H, W):
#     with open(json_path,'r') as f: ann = json.load(f)
#     boxes = ann.get(str(frame_index), [])
#     m = np.zeros((H, W), dtype=np.float32)
#     for (x1,y1,x2,y2) in boxes:
#         x1 = max(0,min(W-1,int(x1))); x2 = max(0,min(W-1,int(x2)))
#         y1 = max(0,min(H-1,int(y1))); y2 = max(0,min(H-1,int(y2)))
#         cv2.rectangle(m,(x1,y1),(x2,y2),1.0,thickness=-1)
#     m = cv2.dilate(m, np.ones((5,5),np.uint8), 1).astype(np.float32)
#     return m


def _mask_from_boxes_json(json_path, frame_index, H, W):
    with open(json_path, "r") as f:
        ann = json.load(f)

    # Allow optional metadata {"meta":{"width":..., "height":...}}
    meta = ann.get("meta", {})
    srcW = meta.get("width", None)
    srcH = meta.get("height", None)

    boxes = ann.get(str(frame_index), [])
    if not boxes:
        return np.zeros((H, W), dtype=np.float32)

    # If no meta dims, try to infer scale if boxes exceed current frame size
    maxx = max(b[2] for b in boxes)
    maxy = max(b[3] for b in boxes)
    if srcW is None or srcH is None:
        sx = (W / maxx) if maxx > W*1.05 else 1.0
        sy = (H / maxy) if maxy > H*1.05 else 1.0
    else:
        sx, sy = W / float(srcW), H / float(srcH)

    m = np.zeros((H, W), dtype=np.float32)
    for (x1,y1,x2,y2) in boxes:
        x1 = int(np.clip(x1 * sx, 0, W-1)); x2 = int(np.clip(x2 * sx, 0, W-1))
        y1 = int(np.clip(y1 * sy, 0, H-1)); y2 = int(np.clip(y2 * sy, 0, H-1))
        if x2 < x1: x1,x2 = x2,x1
        if y2 < y1: y1,y2 = y2,y1
        cv2.rectangle(m, (x1,y1), (x2,y2), 1.0, thickness=-1)
    m = cv2.dilate(m, np.ones((5,5),np.uint8), 1).astype(np.float32)
    return m

# def _overlay_video(frames_bgr, masks, out_path, fps=30, alpha=0.45):
#     H, W = frames_bgr[0].shape[:2]
#     vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
#     for f_bgr, m in zip(frames_bgr, masks):
#         if isinstance(m, torch.Tensor): m = m[0,0].detach().cpu().numpy()
#         if m.ndim == 3: m = m[0]
#         if m.shape != (H,W): m = cv2.resize(m, (W,H), interpolation=cv2.INTER_NEAREST)
#         m_bin = (m >= 0.5).astype(np.uint8)
#         vis = f_bgr.copy()
#         red = np.zeros_like(f_bgr); red[...,2] = 255
#         vis[m_bin>0] = (alpha*red[m_bin>0] + (1-alpha)*vis[m_bin>0]).astype(np.uint8)
#         vw.write(vis)
#     vw.release()
#     print(f"[STM] Saved overlay video to: {out_path}")


def _overlay_video(frames_bgr, masks, out_path, fps=30, alpha=0.45):
    """
    frames_bgr: list of HxWx3 uint8 BGR frames
    masks: list of (H,W) float32 in [0,1] OR None
    """
    if not frames_bgr:
        raise RuntimeError("No frames to write.")
    H, W = frames_bgr[0].shape[:2]
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    for f_bgr, m in zip(frames_bgr, masks):
        if m is None:
            m2d = np.zeros((H, W), dtype=np.float32)
        else:
            m2d = np.asarray(m)
            if m2d.shape != (H, W):
                m2d = cv2.resize(m2d, (W, H), interpolation=cv2.INTER_NEAREST)
            if m2d.max() > 1.0:
                m2d = m2d.astype(np.float32) / 255.0

        m_bin = (m2d >= 0.35).astype(np.uint8)
        vis = f_bgr.copy()
        red = np.zeros_like(f_bgr); red[..., 2] = 255
        vis[m_bin > 0] = (alpha * red[m_bin > 0] + (1 - alpha) * vis[m_bin > 0]).astype(np.uint8)
        vw.write(vis)

    vw.release()
    print(f"[STM] Saved overlay video to: {out_path}")


# --------- quick rectangle annotator for first frame ----------
def annotate_first_frame(frame_bgr):
    clone = frame_bgr.copy()
    boxes = []
    ix, iy, drawing = -1, -1, False

    def mouse(event, x, y, flags, param):
        nonlocal ix, iy, drawing, clone, boxes
        img = clone.copy()
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True; ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.rectangle(img, (ix,iy), (x,y), (0,0,255), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            x1,y1,x2,y2 = ix,iy,x,y
            if x2<x1: x1,x2 = x2,x1
            if y2<y1: y1,y2 = y2,y1
            boxes.append((x1,y1,x2,y2))
        for (a,b,c,d) in boxes: cv2.rectangle(img,(a,b),(c,d),(0,0,255),2)
        cv2.imshow("Annotate (ENTER=done, r=reset)", img)

    cv2.namedWindow("Annotate (ENTER=done, r=reset)")
    cv2.setMouseCallback("Annotate (ENTER=done, r=reset)", mouse)
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 13: break          # ENTER
        if key == ord('r'): boxes = []
        cv2.imshow("Annotate (ENTER=done, r=reset)", clone)
    cv2.destroyAllWindows()
    H,W = frame_bgr.shape[:2]
    m = np.zeros((H,W), dtype=np.float32)
    for (x1,y1,x2,y2) in boxes:
        cv2.rectangle(m,(x1,y1),(x2,y2),1.0, thickness=-1)
    m = cv2.dilate(m, np.ones((5,5),np.uint8),1).astype(np.float32)
    return m

# ================= Adaptive runner =================

def _mean_flow_mag(prev_bgr, curr_bgr):
    g1 = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g1,g2,None,0.5,3,15,3,5,1.2,0)
    mag,_ = cv2.cartToPolar(flow[...,0], flow[...,1])
    return float(mag.mean())

def _iou_tensor(a: torch.Tensor, b: torch.Tensor) -> float:
    a=(a>=0.5).float(); b=(b>=0.5).float()
    inter=(a*b).sum(); union=(a+b-a*b).clamp_min(1e-6).sum()
    return float((inter/union).item())

@torch.no_grad()
def process_video_adaptive(
    video_path: str,
    init_mask_path: Optional[str],
    anno_boxes: Optional[str],
    init_frame: int,
    resize_short: int,
    memory_size: int,
    top_k: Optional[int],
    save_overlay: Optional[str],
    flow_thr=2.5, iou_thr=0.75, conf_thr=0.40,   # conf_thr kept for symmetry (we use mean prob of read mask)
    annotate: bool=False
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    frames_bgr = _read_video_frames(video_path, resize_short=resize_short)
    if not frames_bgr: raise RuntimeError("No frames read.")
    H,W = frames_bgr[0].shape[:2]; fps = _probe_fps(video_path, default=30)

    # Build initial mask
    if annotate:
        print("[STM] Annotation UI: draw one or more rectangles; ENTER to finish, 'r' to reset.")
        m0 = annotate_first_frame(frames_bgr[init_frame])
    elif init_mask_path:
        m0 = cv2.imread(init_mask_path, cv2.IMREAD_GRAYSCALE); 
        if m0 is None: raise RuntimeError(f"Could not read init mask: {init_mask_path}")
        m0 = (cv2.resize(m0,(W,H),interpolation=cv2.INTER_NEAREST)>127).astype(np.float32)
    elif anno_boxes:
        m0 = _mask_from_boxes_json(anno_boxes, init_frame, H, W)
        if m0.sum()==0: raise RuntimeError(f"No boxes for frame {init_frame} in {anno_boxes}")
    else:
        raise RuntimeError("Provide --annotate OR --init_mask OR --anno_boxes.")

    frames_t = [_bgr_to_tensor(f).to(device) for f in frames_bgr]
    init_mask_t = torch.from_numpy(m0).unsqueeze(0).unsqueeze(0).float().to(device)
    print(f"[STM] init mask coverage: {(m0>0.5).mean()*100:.1f}% of pixels")
    
    stm = STM(key_channels=64, value_feat_channels=256, memory_size=memory_size, top_k=top_k).to(device)
    stm.eval()
    stm.initialize_memory(frames_t[init_frame], init_mask_t, init_frame)

    masks = [None]*len(frames_t)
    masks[init_frame] = init_mask_t
    last_mem_mask = init_mask_t.clone()

    # forward
    for i in range(init_frame+1, len(frames_t)):
        m = stm.segment_frame(frames_t[i], i)      # (1,1,H,W), probs 0..1
        mean_conf = float(m.mean().item())
        iou_to_last = _iou_tensor(m, last_mem_mask)
        flow_mean = _mean_flow_mag(frames_bgr[i-1], frames_bgr[i])
        if (iou_to_last < iou_thr) or (mean_conf < conf_thr) or (flow_mean > flow_thr):
            stm.add_memory(frames_t[i], m, i)
            last_mem_mask = m
        masks[i] = m
        if (i - init_frame) % 20 == 0:
            print(f"[STM] frame {i}: mean_conf={mean_conf:.3f}, iou_to_last={iou_to_last:.3f}, flow={flow_mean:.2f}")



    # backward (optional: comment out if not desired)
    for i in range(init_frame-1, -1, -1):
        m = stm.segment_frame(frames_t[i], i)
        mean_conf = float(m.mean().item())
        iou_to_last = _iou_tensor(m, last_mem_mask)
        flow_mean = _mean_flow_mag(frames_bgr[i], frames_bgr[i+1])
        if (iou_to_last < iou_thr) or (mean_conf < conf_thr) or (flow_mean > flow_thr):
            stm.add_memory(frames_t[i], m, i)
            last_mem_mask = m
        masks[i] = m

    if save_overlay:
        # Normalize every mask to shape (H, W) float32 in [0,1]
        H, W = frames_bgr[0].shape[:2]
        masks_np = []
        for mi in masks:
            if mi is None:
                masks_np.append(np.zeros((H, W), dtype=np.float32))
                continue

            # to numpy
            arr = mi.detach().cpu().numpy() if isinstance(mi, torch.Tensor) else np.asarray(mi)

            # squeeze leading singleton dims until 2D
            while arr.ndim > 2 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim == 3 and arr.shape[0] != 1:
                # e.g., (C,H,W) -> take first channel
                arr = arr[0]
            if arr.ndim != 2:
                # fallback to empty mask
                arr = np.zeros((H, W), dtype=np.float32)

            # resize to frame size if needed
            if arr.shape != (H, W):
                arr = cv2.resize(arr, (W, H), interpolation=cv2.INTER_NEAREST)

            # ensure [0,1]
            arr = arr.astype(np.float32)
            if arr.max() > 1.0:
                arr /= 255.0

            masks_np.append(arr)

        _overlay_video(frames_bgr, masks_np, save_overlay, fps=fps, alpha=0.45)
    return masks, frames_bgr


# ================= CLI =================

def main():
    ap = argparse.ArgumentParser("STM VOS — training-free, with raw-mask propagation and annotator")
    ap.add_argument("--video_path", type=str, required=True)
    ap.add_argument("--init_mask",  type=str, default=None, help="PNG of first-frame mask")
    ap.add_argument("--anno_boxes",  type=str, default=None, help="Boxes JSON for init frame")
    ap.add_argument("--init_frame",  type=int, default=0)
    ap.add_argument("--resize_short",type=int, default=320)
    ap.add_argument("--memory_size", type=int, default=6)
    ap.add_argument("--top_k",       type=int, default=128)
    ap.add_argument("--save_overlay",type=str, default="stm_overlay.mp4")
    ap.add_argument("--annotate",    action="store_true", help="Open a simple UI to draw the first mask")
    ap.add_argument("--flow_thr", type=float, default=2.5)
    ap.add_argument("--iou_thr",  type=float, default=0.75)
    ap.add_argument("--conf_thr", type=float, default=0.40)
    args = ap.parse_args()

    top_k = None if args.top_k <= 0 else args.top_k
    _ = process_video_adaptive(
        video_path=args.video_path,
        init_mask_path=args.init_mask,
        anno_boxes=args.anno_boxes,
        init_frame=args.init_frame,
        resize_short=args.resize_short,
        memory_size=args.memory_size,
        top_k=top_k,
        save_overlay=args.save_overlay,
        flow_thr=args.flow_thr, iou_thr=args.iou_thr, conf_thr=args.conf_thr,
        annotate=args.annotate
    )

if __name__ == "__main__":
    main()


"""
python stmV4_demo.py \
  --video_path ../vfencing.mp4 \
  --anno_boxes fencing_boxes.json \
  --init_frame 0 \
  --resize_short 320 \
  --memory_size 8 \
  --top_k 128 \
  --save_overlay vfencing_stm_overlay.mp4
"""