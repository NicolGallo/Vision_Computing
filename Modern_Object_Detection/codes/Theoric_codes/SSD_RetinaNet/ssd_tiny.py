# Save as: ssd_tiny.py
"""
SSD (Simplified) — Tiny, educational implementation for synthetic shapes
- Two feature levels: P3 (stride 16), P4 (stride 32)
- Anchors (aka "default boxes") at each level
- Matching: IoU >= 0.5 positive, < 0.4 negative, in-between ignored
- Loss: Smooth L1 (loc) + CrossEntropy (cls) with Hard-Negative Mining (3:1)
- Decode + NMS + visualization
"""
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np, matplotlib.pyplot as plt, matplotlib.patches as patches
from typing import List, Tuple

# ---------------------------
# Synthetic dataset (boxes list)
# ---------------------------
class ShapesDataset(Dataset):
    def __init__(self, n=1000, image_size=224, num_classes=3, max_objects=4):
        self.n, self.H, self.W = n, image_size, image_size
        self.num_classes, self.max_objects = num_classes, max_objects
        self.class_names = ['circle','square','triangle']
        self.colors = [np.array([0.2,0.4,0.8]), np.array([0.8,0.2,0.2]), np.array([0.2,0.8,0.3])]
    def __len__(self): return self.n
    def _draw(self, img, x,y,s,cid):
        col = self.colors[cid]
        if cid==0: # circle
            yy, xx = np.ogrid[:self.H,:self.W]
            mask = (xx-x)**2+(yy-y)**2 <= (s/2)**2
            img[mask]=col
        elif cid==1: # square
            x1,y1 = max(0,int(x-s/2)), max(0,int(y-s/2))
            x2,y2 = min(self.W,int(x+s/2)), min(self.H,int(y+s/2))
            img[y1:y2,x1:x2]=col
        else: # triangle
            pts = np.array([[x,y-s/2],[x-s/2,y+s/2],[x+s/2,y+s/2]], dtype=np.int32)
            from matplotlib.path import Path
            path = Path(pts)
            Y,X = np.meshgrid(np.arange(self.H), np.arange(self.W), indexing='ij')
            pts2 = np.stack([X.ravel(),Y.ravel()],1)
            mask = path.contains_points(pts2).reshape(self.H,self.W)
            img[mask]=col
    def __getitem__(self, idx):
        img = np.ones((self.H,self.W,3),dtype=np.float32)*0.95
        m = np.random.randint(1,self.max_objects+1)
        boxes, labels = [], []
        for _ in range(m):
            cid = np.random.randint(0,self.num_classes)
            s = np.random.randint(28,80)
            x = np.random.randint(s, self.W-s)
            y = np.random.randint(s, self.H-s)
            self._draw(img, x,y,s,cid)
            # normalized xywh
            w = h = s / self.W
            boxes.append([x/self.W, y/self.H, w, h])  # center-format
            labels.append(cid)
        x_t = torch.from_numpy(img).permute(2,0,1).float()
        return x_t, torch.tensor(boxes).float(), torch.tensor(labels).long()

# ---------------------------
# Tiny backbone (two levels)
# ---------------------------
class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3,64,7,2,3), nn.BatchNorm2d(64), nn.LeakyReLU(0.1,True), nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,1,1), nn.BatchNorm2d(128), nn.LeakyReLU(0.1,True), nn.MaxPool2d(2,2), # stride 8
            nn.Conv2d(128,256,3,1,1), nn.BatchNorm2d(256), nn.LeakyReLU(0.1,True)
        )
        self.p3 = nn.Sequential( # stride 16
            nn.Conv2d(256,256,3,2,1), nn.BatchNorm2d(256), nn.LeakyReLU(0.1,True)
        )
        self.p4 = nn.Sequential( # stride 32
            nn.Conv2d(256,256,3,2,1), nn.BatchNorm2d(256), nn.LeakyReLU(0.1,True)
        )
    def forward(self,x):
        x = self.stem(x)         # ~stride 8
        p3 = self.p3(x)          # stride 16 (H/16,W/16)
        p4 = self.p4(p3)         # stride 32 (H/32,W/32)
        return p3, p4

# ---------------------------
# Anchor (default boxes)
# ---------------------------
def make_anchors(HW_list, strides, ratios=(0.5,1.0,2.0), scales=(1.0, 1.6)):
    # returns per level: [N_i,4] in (cx,cy,w,h) normalized
    all_lvls=[]
    for (Hf,Wf), s in zip(HW_list, strides):
        a = []
        for i in range(Hf):
            for j in range(Wf):
                cx = (j+0.5)*s; cy = (i+0.5)*s
                for r in ratios:
                    for sc in scales:
                        w = sc*s*np.sqrt(1.0/r); h = sc*s*np.sqrt(r)
                        a.append([cx,cy,w,h])
        a = np.array(a, dtype=np.float32)
        # normalize
        a[:,0] /= (Wf*s); a[:,1] /= (Hf*s)
        a[:,2] /= (Wf*s); a[:,3] /= (Hf*s)
        all_lvls.append(torch.from_numpy(a)) # [N,4]
    return all_lvls

def xywh_to_xyxy(xywh): # cx,cy,w,h -> x1y1x2y2
    cx,cy,w,h = xywh.unbind(-1)
    x1 = cx - w/2; y1 = cy - h/2; x2 = cx + w/2; y2 = cy + h/2
    return torch.stack([x1,y1,x2,y2],-1)

def iou(a,b): # a[N,4], b[M,4] (x1y1x2y2) in 0..1
    N,M=a.size(0),b.size(0)
    a_=a[:,None,:].expand(N,M,4); b_=b[None,:,:].expand(N,M,4)
    x1 = torch.max(a_[...,0], b_[...,0]); y1 = torch.max(a_[...,1], b_[...,1])
    x2 = torch.min(a_[...,2], b_[...,2]); y2 = torch.min(a_[...,3], b_[...,3])
    iw = (x2-x1).clamp(min=0); ih=(y2-y1).clamp(min=0); inter=iw*ih
    area_a=(a[...,2]-a[...,0]).clamp(min=0)*(a[...,3]-a[...,1]).clamp(min=0)
    area_b=(b[...,2]-b[...,0]).clamp(min=0)*(b[...,3]-b[...,1]).clamp(min=0)
    union = area_a[:,None]+area_b[None,:]-inter + 1e-6
    return inter/union

# ---------------------------
# SSD Head
# ---------------------------
class SSDTiny(nn.Module):
    def __init__(self, num_classes=3, ratios=(0.5,1.0,2.0), scales=(1.0,1.6)):
        super().__init__()
        self.C=num_classes
        self.backbone=TinyBackbone()
        self.ratios, self.scales = ratios, scales
        self.A = len(ratios)*len(scales)
        # heads per level
        self.cls3 = nn.Conv2d(256, self.A*(self.C+1), 3,1,1) # +1 for background
        self.box3 = nn.Conv2d(256, self.A*4,           3,1,1)
        self.cls4 = nn.Conv2d(256, self.A*(self.C+1), 3,1,1)
        self.box4 = nn.Conv2d(256, self.A*4,           3,1,1)
    def forward(self,x):
        p3,p4 = self.backbone(x)                         # [B,256,H3,W3], [B,256,H4,W4]
        B= x.size(0)
        def head(p, cls, box):
            H,W = p.shape[2], p.shape[3]
            c = cls(p).permute(0,2,3,1).contiguous().view(B, H*W*self.A, self.C+1) # logits
            b = box(p).permute(0,2,3,1).contiguous().view(B, H*W*self.A, 4)        # deltas (absolute offset here)
            return c,b,H,W
        c3,b3,H3,W3 = head(p3,self.cls3,self.box3)
        c4,b4,H4,W4 = head(p4,self.cls4,self.box4)
        cls_logits = torch.cat([c3,c4],1)   # [B, N, C+1]
        box_regs   = torch.cat([b3,b4],1)   # [B, N, 4] (we’ll interpret as absolute cxcywh residuals)
        # anchors
        strides = [16,32]
        anchors = make_anchors([(H3,W3),(H4,W4)], strides, self.ratios, self.scales)
        anchors = torch.cat(anchors,0).to(x.device).unsqueeze(0).expand(B,-1,-1)  # [B,N,4]
        return cls_logits, box_regs, anchors

# ---------------------------
# Matching + Loss (HNM)
# ---------------------------
def match_anchors(anchors_xywh, gt_xywh, gt_labels, pos_iou=0.5, neg_iou=0.4):
    # anchors: [N,4] 0..1; gts: [M,4]
    if gt_xywh.numel()==0:
        return torch.full((anchors_xywh.size(0),), -1, dtype=torch.long), torch.zeros_like(anchors_xywh)
    A = xywh_to_xyxy(anchors_xywh)
    G = xywh_to_xyxy(gt_xywh)
    i = iou(A, G)                          # [N,M]
    max_iou, max_j = i.max(dim=1)          # best GT per anchor
    matches = torch.full((anchors_xywh.size(0),), -1, dtype=torch.long) # -1 ignore
    matches[max_iou>=pos_iou] = max_j[max_iou>=pos_iou]
    matches[max_iou<neg_iou]  = -2   # negative
    # ensure each GT matched at least once (bipartite-ish)
    gt_best_iou, gt_best_anchor = i.max(dim=0)
    matches[gt_best_anchor] = torch.arange(gt_xywh.size(0))
    return matches, gt_xywh[matches.clamp(min=0)]

def smooth_l1(d, beta=1/9):
    absd = d.abs()
    return torch.where(absd<beta, 0.5*absd**2/beta, absd-0.5*beta)

def ssd_loss(cls_logits, box_regs, anchors, gt_xywh_list, gt_labels_list, neg_pos_ratio=3):
    B,N,_ = anchors.shape
    cls_loss, loc_loss = 0.0, 0.0
    for b in range(B):
        gt_xywh, gt_labels = gt_xywh_list[b], gt_labels_list[b]
        m, gt_matched = match_anchors(anchors[b], gt_xywh, gt_labels)
        # labels with background=0, classes 1..C
        targets_cls = torch.zeros(N, dtype=torch.long, device=cls_logits.device) # default bg
        pos_mask = m >= 0
        neg_mask = m == -2
        targets_cls[pos_mask] = gt_labels[m[pos_mask]] + 1  # shift by +1
        # classification loss per anchor
        cls_log = cls_logits[b]  # [N,C+1]
        loss_cls_all = F.cross_entropy(cls_log, targets_cls, reduction='none') # [N]
        # Hard negative mining
        num_pos = pos_mask.sum().item()
        num_neg = min(int(neg_pos_ratio*num_pos), int(neg_mask.sum().item()))
        hard_neg_idx = torch.argsort(loss_cls_all[neg_mask], descending=True)[:num_neg]
        sel = pos_mask.clone()
        sel[neg_mask] = False
        idx = torch.nonzero(neg_mask, as_tuple=False).squeeze(1)[hard_neg_idx]
        sel[idx] = True
        cls_loss += loss_cls_all[sel].sum() / max(1,num_pos)

        # Localization (only positives) — regress absolute cxcywh (simple)
        if num_pos>0:
            loc_pred = box_regs[b][pos_mask]         # [P,4]
            loc_tgt  = gt_matched[pos_mask]          # [P,4]
            loc_loss += smooth_l1(loc_pred - loc_tgt).sum() / num_pos
    return cls_loss/B, loc_loss/B

# ---------------------------
# Decode + NMS + Viz
# ---------------------------
def decode_and_nms(cls_logits, box_regs, anchors, conf_th=0.3, iou_th=0.45):
    # cls probs
    probs = F.softmax(cls_logits, dim=-1) # [B,N,C+1]
    B,N,Cp = probs.shape
    out = []
    for b in range(B):
        p = probs[b]; boxes = box_regs[b]; an = anchors[b]
        # best class (exclude bg=0)
        conf, cls = torch.max(p[:,1:], dim=-1)    # [N]
        keep = conf >= conf_th
        boxes_xyxy = xywh_to_xyxy(boxes[keep].clamp(0,1))
        conf = conf[keep]; cls = cls[keep]
        # NMS per class
        dets=[]
        for c in range(p.shape[1]-1):
            mask = (cls==c)
            if mask.sum()==0: continue
            idx = torchvision_nms(boxes_xyxy[mask], conf[mask], iou_th)  # simple python NMS below
            for k in idx:
                bb = boxes_xyxy[mask][k]
                dets.append((bb, int(c), float(conf[mask][k])))
        out.append(dets)
    return out

def torchvision_nms(boxes, scores, iou_thresh):
    # pure torch IoU NMS (small, CPU ok)
    if boxes.numel()==0: return []
    x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    areas = (x2-x1).clamp(min=0)*(y2-y1).clamp(min=0)
    order = scores.argsort(descending=True)
    keep=[]
    while order.numel()>0:
        i = int(order[0]); keep.append(i)
        if order.numel()==1: break
        rest = order[1:]
        xx1 = torch.maximum(x1[i], x1[rest]); yy1=torch.maximum(y1[i],y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest]); yy2=torch.minimum(y2[i],y2[rest])
        iw=(xx2-xx1).clamp(min=0); ih=(yy2-yy1).clamp(min=0)
        inter=iw*ih
        iou = inter / (areas[i]+areas[rest]-inter + 1e-6)
        order = rest[iou<=iou_thresh]
    return keep

def visualize_sample(model, dataset):
    model.eval()
    x, gt_boxes, gt_labels = dataset[0]
    with torch.no_grad():
        cls_logits, box_regs, anchors = model(x.unsqueeze(0))
    dets = decode_and_nms(cls_logits, box_regs, anchors, 0.3, 0.45)[0]
    img = x.permute(1,2,0).numpy()
    fig,ax = plt.subplots(1,1,figsize=(6,6)); ax.imshow(img); ax.axis('off')
    H=W=dataset.W
    # draw GT dashed
    for (cx,cy,w,h), lab in zip(gt_boxes, gt_labels):
        x1=(cx-w/2)*W; y1=(cy-h/2)*H; wpx=w*W; hpx=h*H
        ax.add_patch(patches.Rectangle((x1,y1),wpx,hpx,ec='y',fc='none',ls='--',lw=2))
        ax.text(x1,y1-4,f'GT:{dataset.class_names[lab]}',color='y',fontsize=8,weight='bold')
    # draw preds
    for bb, c, sc in dets:
        x1,y1,x2,y2 = (bb*W).tolist()
        ax.add_patch(patches.Rectangle((x1,y1),x2-x1,y2-y1,ec='lime',fc='none',lw=2))
        ax.text(x1,y2+12,f'{dataset.class_names[c]} {sc:.2f}',color='lime',fontsize=8,weight='bold')
    plt.tight_layout(); plt.show()

# ---------------------------
# Training loop
# ---------------------------
def train_ssd(epochs=10, bs=16, lr=1e-3, device='cuda' if torch.cuda.is_available() else 'cpu'):
    ds = ShapesDataset(n=800)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=lambda b: list(zip(*b)))
    model = SSDTiny(num_classes=3).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    for e in range(epochs):
        model.train(); Lc=Lr=cnt=0
        for imgs, boxes, labels in dl:
            imgs = torch.stack(imgs).to(device)
            gt_boxes = [b.to(device) for b in boxes]
            gt_labels= [l.to(device) for l in labels]
            cls_logits, box_regs, anchors = model(imgs)
            lc, lr_ = ssd_loss(cls_logits, box_regs, anchors, gt_boxes, gt_labels)
            loss = lc + lr_
            opt.zero_grad(); loss.backward(); opt.step()
            Lc+=lc.item(); Lr+=lr_.item(); cnt+=1
        print(f"Epoch {e+1}: cls {Lc/cnt:.3f} | loc {Lr/cnt:.3f} | total {(Lc+Lr)/cnt:.3f}")
    return model, ds

if __name__=="__main__":
    model, ds = train_ssd(epochs=5)
    visualize_sample(model, ds)
