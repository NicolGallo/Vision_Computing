import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

class MiniSAM(nn.Module):
    """
    Simplified SAM architecture for educational purposes.
    Can be trained from scratch on limited data.
    ~5M parameters vs 636M for full SAM.
    """
    def __init__(self, n_classes=21, embed_dim=256):
        super().__init__()
        
        # 1. Lightweight Image Encoder (~2M params)
        backbone = models.mobilenet_v3_small(pretrained=True)
        self.image_encoder = nn.Sequential(*list(backbone.features))
        
        # Projection to common dimension
        self.img_proj = nn.Conv2d(576, embed_dim, 1)
        
        # 2. Simple Prompt Encoder
        self.point_type_embed = nn.Embedding(2, embed_dim)  # fg/bg
        self.point_pos_embed = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
        
        # Box encoding
        self.box_embed = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(), 
            nn.Linear(128, embed_dim)
        )
        
        # 3. Fusion and Mask Decoder (~1M params)
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim * 2, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # Output heads
        self.mask_head = nn.Conv2d(64, n_classes, 1)
        self.iou_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        
    def encode_image(self, x):
        """Extract image features"""
        B, C, H, W = x.shape
        features = self.image_encoder(x)  # B x 576 x H/8 x W/8
        features = self.img_proj(features)  # B x 256 x H/8 x W/8
        return features
    
    def encode_prompts(self, points=None, point_labels=None, boxes=None, img_size=None):
        """
        Encode point and/or box prompts
        points: (B, N, 2) in [0, 1] normalized coordinates
        point_labels: (B, N) with 0=bg, 1=fg
        boxes: (B, 4) in [x1, y1, x2, y2] normalized
        """
        B = points.shape[0] if points is not None else boxes.shape[0]
        H, W = img_size
        
        prompt_features = []
        
        if points is not None and point_labels is not None:
            # Positional encoding
            pos_enc = self.point_pos_embed(points)  # B x N x 256
            # Type encoding
            type_enc = self.point_type_embed(point_labels)  # B x N x 256
            # Combine
            point_enc = pos_enc + type_enc  # B x N x 256
            # Average pool across points
            point_enc = point_enc.mean(dim=1, keepdim=True)  # B x 1 x 256
            prompt_features.append(point_enc)
        
        if boxes is not None:
            box_enc = self.box_embed(boxes).unsqueeze(1)  # B x 1 x 256
            prompt_features.append(box_enc)
        
        # Combine all prompts
        if len(prompt_features) > 0:
            prompt_enc = torch.cat(prompt_features, dim=1).mean(dim=1)  # B x 256
            # Broadcast to spatial dimensions
            prompt_enc = prompt_enc.view(B, -1, 1, 1).expand(-1, -1, H//8, W//8)
        else:
            # No prompts - use learnable token
            prompt_enc = torch.zeros(B, 256, H//8, W//8, device=points.device)
            
        return prompt_enc
    
    def forward(self, images, points=None, point_labels=None, boxes=None):
        """
        images: B x 3 x H x W
        points: B x N x 2 (normalized [0,1])
        point_labels: B x N (0 or 1)
        boxes: B x 4 (normalized [0,1])
        """
        B, C, H, W = images.shape
        
        # Encode image
        img_features = self.encode_image(images)  # B x 256 x H/8 x W/8
        
        # Encode prompts
        prompt_features = self.encode_prompts(
            points, point_labels, boxes, img_size=(H, W)
        )  # B x 256 x H/8 x W/8
        
        # Fuse features
        fused = torch.cat([img_features, prompt_features], dim=1)  # B x 512 x H/8 x W/8
        
        # Decode mask
        decoded = self.decoder(fused)  # B x 64 x H/8 x W/8
        
        # Output
        mask_logits = self.mask_head(decoded)  # B x n_classes x H/8 x W/8
        mask_logits = self.upsample(mask_logits)  # B x n_classes x H x W
        
        iou_pred = self.iou_head(decoded)  # B x 1
        
        return mask_logits, iou_pred


class MiniSAMTrainer:
    """Training pipeline for Mini-SAM"""
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
    def train_step(self, images, gt_masks, points, point_labels):
        """
        Single training step with simulated prompts
        images: B x 3 x H x W
        gt_masks: B x H x W (class indices)
        points: B x N x 2 (sampled from GT)
        point_labels: B x N
        """
        # Forward pass
        mask_logits, iou_pred = self.model(images, points, point_labels)
        
        # Compute loss
        ce_loss = nn.CrossEntropyLoss()(mask_logits, gt_masks)
        
        # Dice loss
        mask_probs = torch.softmax(mask_logits, dim=1)
        gt_one_hot = nn.functional.one_hot(gt_masks, num_classes=mask_logits.shape[1])
        gt_one_hot = gt_one_hot.permute(0, 3, 1, 2).float()
        
        dice_loss = 1 - (2 * (mask_probs * gt_one_hot).sum() + 1) / \
                    (mask_probs.sum() + gt_one_hot.sum() + 1)
        
        # IoU prediction loss
        pred_masks = mask_logits.argmax(dim=1)
        true_iou = self.compute_iou(pred_masks, gt_masks)
        iou_loss = nn.MSELoss()(iou_pred.squeeze(), true_iou)
        
        # Combined loss
        total_loss = ce_loss + dice_loss + 0.1 * iou_loss
        
        return total_loss, {
            'ce_loss': ce_loss.item(),
            'dice_loss': dice_loss.item(),
            'iou_loss': iou_loss.item()
        }
    
    @staticmethod
    def compute_iou(pred, target):
        """Compute IoU between predictions and targets"""
        intersection = (pred == target).float().sum(dim=[1, 2])
        union = pred.numel() // pred.shape[0] - (pred != target).float().sum(dim=[1, 2])
        return intersection / (union + 1e-6)
    
    @staticmethod
    def sample_points_from_mask(masks, n_points=5):
        """
        Sample points from ground truth masks (simulates user clicks)
        masks: B x H x W
        Returns: points (B x N x 2), labels (B x N)
        """
        B, H, W = masks.shape
        points_list = []
        labels_list = []
        
        for b in range(B):
            mask = masks[b]
            
            # Sample foreground points
            fg_indices = torch.nonzero(mask > 0)
            if len(fg_indices) > 0:
                fg_samples = fg_indices[torch.randint(len(fg_indices), (n_points//2,))]
                fg_points = fg_samples.float() / torch.tensor([H, W]).to(fg_samples.device)
                fg_labels = torch.ones(len(fg_points), dtype=torch.long)
            else:
                fg_points = torch.zeros(0, 2)
                fg_labels = torch.zeros(0, dtype=torch.long)
            
            # Sample background points
            bg_indices = torch.nonzero(mask == 0)
            if len(bg_indices) > 0:
                bg_samples = bg_indices[torch.randint(len(bg_indices), (n_points//2,))]
                bg_points = bg_samples.float() / torch.tensor([H, W]).to(bg_samples.device)
                bg_labels = torch.zeros(len(bg_points), dtype=torch.long)
            else:
                bg_points = torch.zeros(0, 2)
                bg_labels = torch.zeros(0, dtype=torch.long)
            
            # Combine
            points = torch.cat([fg_points, bg_points], dim=0)
            labels = torch.cat([fg_labels, bg_labels], dim=0)
            
            # Pad to n_points
            if len(points) < n_points:
                pad = n_points - len(points)
                points = torch.cat([points, torch.zeros(pad, 2)], dim=0)
                labels = torch.cat([labels, torch.zeros(pad, dtype=torch.long)], dim=0)
            
            points_list.append(points)
            labels_list.append(labels)
        
        return torch.stack(points_list), torch.stack(labels_list)


# Example usage
if __name__ == "__main__":
    # Create model
    model = MiniSAM(n_classes=21)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Example training
    trainer = MiniSAMTrainer(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Dummy data
    images = torch.randn(4, 3, 256, 256)
    gt_masks = torch.randint(0, 21, (4, 256, 256))
    
    # Sample points from masks
    points, point_labels = trainer.sample_points_from_mask(gt_masks, n_points=10)
    
    # Training step
    optimizer.zero_grad()
    loss, metrics = trainer.train_step(images, gt_masks, points, point_labels)
    loss.backward()
    optimizer.step()
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")