# Save as: retinanet_tiny.py
"""
RetinaNet (Simplified) — Tiny, educational implementation for synthetic shapes
- FPN over two levels (P3,P4) from a tiny backbone
- Anchors per level, shared regression/cls subnets
- Focal loss (gamma=2, alpha=0.25), sigmoid per class (no background class)
- Smooth L1 for box regression (educational; IoU losses are also common)
"""
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np, matplotlib.pyplot as plt, matplotlib.patches as patches

# ---- reuse the same ShapesDataset as in SSD (paste if running standalone) ----
class ShapesDataset(Dataset):
    def __init__(self, n=1000, image_size=224, num_classes=3, max_objects=4):
        self.n, self.H, self.W = n, image_size, image_size
        self.num_classes, self.max_objects = num_classes, max_objects
        self.class_names = ['circle','square','triangle']
        self.colors = [np.array([0.2,0.4,0.8]), np.array([0.8,0.2,0.2]), np.array([0.2,0.8,0.3])]
    def __len__(self): return self.n
    def _draw(self, img, x,y,s,cid):
        col = self.colors[cid]
        if cid==0:
            yy, xx = np.ogrid[:self.H,:self.W]; mask=(xx-x)**2+(yy-y)**2 <= (s/2)**2; img[mask]=col
        elif cid==1:
            x1,y1=max(0,int(x-s/2)),max(0,int(y-s/2)); x2,y2=min(self.W,int(x+s/2)),min(self.H,int(y+s/2))
            img[y1:y2,x1:x2]=col
        else:
            pts=np.array([[x,y-s/2],[x-s/2,y+s/2],[x+s/2,y+s/2]],dtype=np.int32)
            from matplotlib.path import Path; path=Path(pts)
            Y,X=np.meshgrid(np.arange(self.H),np.arange(self.W),indexing='ij')
            mask=path.contains_points(np.stack([X.ravel(),Y.ravel()],1)).reshape(self.H,self.W)
            img[mask]=col
    def __getitem__(self, idx):
        img=np.ones((self.H,self.W,3),dtype=np.float32)*0.95
        m=np.random.randint(1,self.max_objects+1)
        boxes,labels=[],[]
        for _ in range(m):
            cid=np.random.randint(0,self.num_classes); s=np.random.randint(28,80)
            x=np.random.randint(s,self.W-s); y=np.random.randint(s,self.H-s)
            self._draw(img,x,y,s,cid); w=h=s/self.W
            boxes.append([x/self.W,y/self.H,w,h]); labels.append(cid)
        return torch.from_numpy(img).permute(2,0,1).float(), torch.tensor(boxes).float(), torch.tensor(labels).long()

# ---- tiny backbone + FPN (P3,P4) ----
class TinyBackbone(nn.Module):
    def __init__(self): 
        super().__init__()
        # C3 ~ stride 8, C4 ~ 16, but we'll make P3,P4 strides 16 and 32 for simplicity
        self.stem = nn.Sequential(
            nn.Conv2d(3,64,7,2,3), nn.BatchNorm2d(64), nn.LeakyReLU(0.1,True), nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,1,1), nn.BatchNorm2d(128), nn.LeakyReLU(0.1,True),
            nn.Conv2d(128,256,3,1,1), nn.BatchNorm2d(256), nn.LeakyReLU(0.1,True)
        )
        self.c3_down = nn.Conv2d(256,256,3,2,1) # stride 4 extra -> total ~16
        self.c4_down = nn.Conv2d(256,256,3,2,1) # -> ~32
        self.p3_1x1 = nn.Conv2d(256,256,1)
        self.p4_1x1 = nn.Conv2d(256,256,1)
        self.p3_3x3 = nn.Conv2d(256,256,3,1,1)
        self.p4_3x3 = nn.Conv2d(256,256,3,1,1)
    def forward(self,x):
        c = self.stem(x)
        c3 = self.c3_down(c)  # stride ~16
        c4 = self.c4_down(c3) # stride ~32
        p3 = self.p3_3x3(F.relu(self.p3_1x1(c3)))
        p4 = self.p4_3x3(F.relu(self.p4_1x1(c4)))
        return p3,p4

def make_anchors(HW_list, strides, ratios=(0.5,1.0,2.0), scales=(1.0,1.6)):
    all_lvls=[]
    for (Hf,Wf), s in zip(HW_list, strides):
        a=[]
        for i in range(Hf):
            for j in range(Wf):
                cx=(j+0.5)*s; cy=(i+0.5)*s
                for r in ratios:
                    for sc in scales:
                        w=sc*s*np.sqrt(1.0/r); h=sc*s*np.sqrt(r)
                        a.append([cx,cy,w,h])
        a=np.array(a,np.float32)
        a[:,0]/=(Wf*s); a[:,1]/=(Hf*s); a[:,2]/=(Wf*s); a[:,3]/=(Hf*s)
        all_lvls.append(torch.from_numpy(a))
    return all_lvls

def xywh_to_xyxy(xywh):
    cx,cy,w,h=xywh.unbind(-1)
    return torch.stack([cx-w/2,cy-h/2,cx+w/2,cy+h/2],-1)

def iou(a,b):
    N,M=a.size(0),b.size(0)
    a_=a[:,None,:].expand(N,M,4); b_=b[None,:,:].expand(N,M,4)
    x1=torch.max(a_[...,0],b_[...,0]); y1=torch.max(a_[...,1],b_[...,1])
    x2=torch.min(a_[...,2],b_[...,2]); y2=torch.min(a_[...,3],b_[...,3])
    iw=(x2-x1).clamp(min=0); ih=(y2-y1).clamp(min=0); inter=iw*ih
    aa=(a[...,2]-a[...,0]).clamp(min=0)*(a[...,3]-a[...,1]).clamp(min=0)
    bb=(b[...,2]-b[...,0]).clamp(min=0)*(b[...,3]-b[...,1]).clamp(min=0)
    return inter/(aa[:,None]+bb[None,:]-inter+1e-6)

# ---- RetinaNet head ----
class RetinaTiny(nn.Module):
    def __init__(self, num_classes=3, ratios=(0.5,1.0,2.0), scales=(1.0,1.6)):
        super().__init__()
        self.C=num_classes; self.A=len(ratios)*len(scales)
        self.backbone=TinyBackbone()
        # shared subnets (4 conv layers each)
        def subnet(ch_out):
            return nn.Sequential(
                nn.Conv2d(256,256,3,1,1), nn.ReLU(True),
                nn.Conv2d(256,256,3,1,1), nn.ReLU(True),
                nn.Conv2d(256,256,3,1,1), nn.ReLU(True),
                nn.Conv2d(256,256,3,1,1), nn.ReLU(True),
                nn.Conv2d(256,ch_out,3,1,1)
            )
        self.cls_subnet = subnet(self.A*self.C)
        self.box_subnet = subnet(self.A*4)
        self.ratios, self.scales = ratios, scales
    def forward(self,x):
        p3,p4 = self.backbone(x)
        B=x.size(0)
        def head(p):
            H,W=p.shape[2], p.shape[3]
            cls = self.cls_subnet(p).permute(0,2,3,1).contiguous().view(B, H*W*self.A, self.C)
            box = self.box_subnet(p).permute(0,2,3,1).contiguous().view(B, H*W*self.A, 4)
            return cls, box, H,W
        c3,b3,H3,W3 = head(p3)
        c4,b4,H4,W4 = head(p4)
        cls_logits = torch.cat([c3,c4],1)  # [B,N,C] (logits, sigmoid later)
        box_regs   = torch.cat([b3,b4],1)  # [B,N,4] (absolute cxcywh residuals here for simplicity)
        anchors = make_anchors([(H3,W3),(H4,W4)],[16,32], self.ratios, self.scales)
        anchors = torch.cat(anchors,0).to(x.device).unsqueeze(0).expand(B,-1,-1)
        return cls_logits, box_regs, anchors

# ---- Matching + Focal loss ----
def match_anchors(anchors, gt_xywh, gt_labels, pos_iou=0.5, neg_iou=0.4):
    if gt_xywh.numel()==0:
        return torch.full((anchors.size(0),), -1, dtype=torch.long), torch.zeros_like(anchors)
    A=xywh_to_xyxy(anchors); G=xywh_to_xyxy(gt_xywh)
    I = iou(A,G)
    max_iou, max_j = I.max(dim=1)
    matches = torch.full((anchors.size(0),), -1, dtype=torch.long)
    matches[max_iou>=pos_iou] = max_j[max_iou>=pos_iou]
    matches[max_iou<neg_iou]  = -2
    gt_best_iou, gt_best_anchor = I.max(dim=0)
    matches[gt_best_anchor] = torch.arange(gt_xywh.size(0))
    return matches, gt_xywh[matches.clamp(min=0)], gt_labels[matches.clamp(min=0)]

def focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction='sum'):
    # inputs: [N,C] logits; targets: [N,C] in {0,1}
    p = torch.sigmoid(inputs)
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')  # -y log p - (1-y) log(1-p)
    p_t = p*targets + (1-p)*(1-targets)
    alpha_t = alpha*targets + (1-alpha)*(1-targets)
    fl = alpha_t * (1-p_t).pow(gamma) * ce
    if reduction=='sum': return fl.sum()
    if reduction=='mean': return fl.mean()
    return fl

def retina_loss(cls_logits, box_regs, anchors, gt_xywh_list, gt_labels_list):
    B,N,C = cls_logits.shape
    total_cls=0.0; total_loc=0.0
    for b in range(B):
        gt_xywh, gt_labels = gt_xywh_list[b], gt_labels_list[b]
        m, gt_match, gt_lab_match = match_anchors(anchors[b], gt_xywh, gt_labels)
        pos = m>=0; neg = m==-2
        # classification targets (sigmoid multi-label one-hot)
        y = torch.zeros((N,C), device=cls_logits.device)
        y[pos, gt_lab_match[pos]] = 1.0
        cls = focal_loss(cls_logits[b], y, alpha=0.25, gamma=2.0, reduction='sum')
        num_pos = max(1, pos.sum().item())
        total_cls += cls/num_pos
        # bbox (Smooth L1 on positives)
        if pos.any():
            loc_pred = box_regs[b][pos]
            loc_tgt  = gt_match[pos]
            total_loc += F.smooth_l1_loss(loc_pred, loc_tgt, reduction='sum')/num_pos
    return total_cls/B, total_loc/B

# ---- Decode + NMS + Viz ----
def xywh_to_xyxy_t(x):
    cx,cy,w,h = x.unbind(-1)
    return torch.stack([cx-w/2,cy-h/2,cx+w/2,cy+h/2],-1)

def nms_torch(boxes, scores, iou_th):
    if boxes.numel()==0: return []
    x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    areas=(x2-x1).clamp(min=0)*(y2-y1).clamp(min=0)
    order = scores.argsort(descending=True); keep=[]
    while order.numel()>0:
        i=int(order[0]); keep.append(i)
        if order.numel()==1: break
        rest=order[1:]
        xx1=torch.maximum(x1[i],x1[rest]); yy1=torch.maximum(y1[i],y1[rest])
        xx2=torch.minimum(x2[i],x2[rest]); yy2=torch.minimum(y2[i],y2[rest])
        iw=(xx2-xx1).clamp(min=0); ih=(yy2-yy1).clamp(min=0)
        inter=iw*ih
        iou= inter/(areas[i]+areas[rest]-inter + 1e-6)
        order=rest[iou<=iou_th]
    return keep

def decode_and_nms(cls_logits, box_regs, conf_th=0.3, iou_th=0.45):
    B,N,C = cls_logits.shape
    out=[]
    for b in range(B):
        probs = torch.sigmoid(cls_logits[b])  # [N,C]
        boxes = box_regs[b].clamp(0,1)
        dets=[]
        for c in range(C):
            conf = probs[:,c]
            mask = conf>=conf_th
            if mask.sum()==0: continue
            bb = xywh_to_xyxy_t(boxes[mask])
            sc = conf[mask]
            keep = nms_torch(bb, sc, iou_th)
            for k in keep:
                dets.append((bb[k], c, float(sc[k])))
        out.append(dets)
    return out

def visualize_sample(model, ds):
    model.eval()
    x, gt_boxes, gt_labels = ds[0]
    with torch.no_grad():
        cls_logits, box_regs, anchors = model(x.unsqueeze(0))
    dets = decode_and_nms(cls_logits, box_regs, 0.3, 0.45)[0]
    img = x.permute(1,2,0).numpy()
    fig,ax=plt.subplots(1,1,figsize=(6,6)); ax.imshow(img); ax.axis('off')
    H=W=ds.W
    for (cx,cy,w,h), lab in zip(gt_boxes, gt_labels):
        x1=(cx-w/2)*W; y1=(cy-h/2)*H; ax.add_patch(patches.Rectangle((x1,y1),w*W,h*H,ec='y',fc='none',ls='--',lw=2))
        ax.text(x1,y1-4,f'GT:{ds.class_names[lab]}',color='y',fontsize=8,weight='bold')
    for bb, c, sc in dets:
        x1,y1,x2,y2 = (bb*W).tolist()
        ax.add_patch(patches.Rectangle((x1,y1),x2-x1,y2-y1,ec='lime',fc='none',lw=2))
        ax.text(x1,y2+12,f'{ds.class_names[c]} {sc:.2f}',color='lime',fontsize=8,weight='bold')
    plt.tight_layout(); plt.show()

# ---- Train ----
def train_retina(epochs=10, bs=16, lr=1e-3, device='cuda' if torch.cuda.is_available() else 'cpu'):
    ds = ShapesDataset(n=800)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=lambda b: list(zip(*b)))
    model = RetinaTiny(num_classes=3).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    for e in range(epochs):
        model.train(); Lc=Lr=cnt=0
        for imgs, boxes, labels in dl:
            imgs=torch.stack(imgs).to(device)
            gt_boxes=[b.to(device) for b in boxes]; gt_labels=[l.to(device) for l in labels]
            cls_logits, box_regs, anchors = model(imgs)
            lc, lr_ = retina_loss(cls_logits, box_regs, anchors, gt_boxes, gt_labels)
            loss = lc + lr_
            opt.zero_grad(); loss.backward(); opt.step()
            Lc+=lc.item(); Lr+=lr_.item(); cnt+=1
        print(f"Epoch {e+1}: focal {Lc/cnt:.3f} | loc {Lr/cnt:.3f} | total {(Lc+Lr)/cnt:.3f}")
    return model, ds

if __name__=="__main__":
    model, ds = train_retina(epochs=5)
    visualize_sample(model, ds)
