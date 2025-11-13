"""
MDNet: Multi-Domain Convolutional Neural Network Tracker
Paper: Learning Multi-Domain Convolutional Neural Networks for Visual Tracking (CVPR 2016)
Author: Hyeonseob Nam, Bohyung Han

Key Ideas:
- Multi-domain learning with shared features + domain-specific layers
- Online adaptation during tracking
- Pre-training on multiple tracking sequences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2


class MDNet(nn.Module):
    """
    MDNet architecture with shared convolutional layers and domain-specific FC layers.
    
    Architecture:
        conv1-3: Shared feature extraction layers (3 conv layers)
        fc4-5: Domain-specific fully connected layers
        fc6: Binary classification (target vs background)
    """
    
    def __init__(self, num_domains=1, init_weights=True):
        super(MDNet, self).__init__()
        self.num_domains = num_domains
        
        # Shared convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=2, alpha=1e-4, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=2, alpha=1e-4, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        
        # Domain-specific fully connected layers
        self.fc4 = nn.ModuleList([nn.Linear(512 * 3 * 3, 512) for _ in range(num_domains)])
        self.fc5 = nn.ModuleList([nn.Linear(512, 512) for _ in range(num_domains)])
        
        # Classification layer (shared)
        self.fc6 = nn.Linear(512, 2)
        
        if init_weights:
            self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, domain_idx=0, return_features=False):
        """
        Forward pass through MDNet
        
        Args:
            x: Input tensor [batch_size, 3, 107, 107]
            domain_idx: Which domain-specific layers to use
            return_features: If True, return intermediate features
        
        Returns:
            scores: Classification scores [batch_size, 2]
        """
        # Shared convolutional layers
        x = self.conv1(x)  # [B, 96, 24, 24]
        x = self.conv2(x)  # [B, 256, 5, 5]
        x = self.conv3(x)  # [B, 512, 3, 3]
        
        # Flatten
        x = x.view(x.size(0), -1)  # [B, 512*3*3]
        
        if return_features:
            conv_features = x.clone()
        
        # Domain-specific FC layers
        x = F.relu(self.fc4[domain_idx](x))
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = F.relu(self.fc5[domain_idx](x))
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Classification
        x = self.fc6(x)
        
        if return_features:
            return x, conv_features
        return x


class MDNetTracker:
    """
    MDNet tracking algorithm with online adaptation
    """
    
    def __init__(self, model_path=None):
        self.model = MDNet(num_domains=1)
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # Tracking parameters
        self.pos_threshold = 0.7
        self.neg_threshold = 0.3
        self.sample_size = 256
        self.trans_f = 0.6  # Translation sampling factor
        self.scale_f = 1.05  # Scale sampling factor
        
        # Online learning parameters
        self.update_interval = 10
        self.lr = 0.0001
        
        # Image preprocessing
        self.img_size = 107
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def init(self, image, bbox):
        """
        Initialize tracker with first frame
        
        Args:
            image: First frame (H, W, 3)
            bbox: Initial bounding box [x, y, w, h]
        """
        self.target_bbox = bbox
        self.image_size = image.shape[:2]
        
        # Sample positive and negative examples
        pos_samples = self._sample_patches(image, bbox, n_samples=50, 
                                          trans_f=0.1, scale_f=1.0)
        neg_samples = self._sample_patches(image, bbox, n_samples=200,
                                          trans_f=2.0, scale_f=1.0)
        
        # Initial training
        self._train_initial(pos_samples, neg_samples)
        
        self.frame_idx = 0
    
    def update(self, image):
        """
        Update tracker with new frame
        
        Args:
            image: Current frame (H, W, 3)
        
        Returns:
            bbox: Predicted bounding box [x, y, w, h]
            score: Confidence score
        """
        self.frame_idx += 1
        
        # Sample candidate patches around previous location
        candidates = self._sample_patches(image, self.target_bbox, 
                                         n_samples=self.sample_size,
                                         trans_f=self.trans_f, 
                                         scale_f=self.scale_f)
        
        # Evaluate candidates
        with torch.no_grad():
            candidate_tensor = torch.stack([self.transform(c) for c in candidates])
            scores = F.softmax(self.model(candidate_tensor), dim=1)[:, 1]
        
        # Select best candidate
        best_idx = torch.argmax(scores)
        best_score = scores[best_idx].item()
        
        # Update target bbox
        self.target_bbox = self._get_bbox_from_sample_idx(best_idx)
        
        # Online update
        if self.frame_idx % self.update_interval == 0 and best_score > 0.5:
            self._update_online(image, self.target_bbox)
        
        return self.target_bbox, best_score
    
    def _sample_patches(self, image, bbox, n_samples, trans_f, scale_f):
        """
        Sample patches around bounding box
        
        Args:
            image: Input image
            bbox: Center bounding box [x, y, w, h]
            n_samples: Number of samples
            trans_f: Translation factor
            scale_f: Scale factor
        
        Returns:
            patches: List of sampled patches
        """
        x, y, w, h = bbox
        patches = []
        
        for _ in range(n_samples):
            # Sample translation
            dx = np.random.uniform(-trans_f * w, trans_f * w)
            dy = np.random.uniform(-trans_f * h, trans_f * h)
            
            # Sample scale
            scale = np.random.uniform(1.0 / scale_f, scale_f)
            
            # Compute new bbox
            new_w = w * scale
            new_h = h * scale
            new_x = x + dx
            new_y = y + dy
            
            # Extract patch
            patch = self._crop_patch(image, [new_x, new_y, new_w, new_h])
            patches.append(patch)
        
        return patches
    
    def _crop_patch(self, image, bbox):
        """Crop patch from image given bbox"""
        x, y, w, h = [int(v) for v in bbox]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(image.shape[1], x + w)
        y2 = min(image.shape[0], y + h)
        
        patch = image[y1:y2, x1:x2]
        if patch.size == 0:
            patch = np.zeros((10, 10, 3), dtype=np.uint8)
        
        return patch
    
    def _train_initial(self, pos_samples, neg_samples):
        """Initial training with positive and negative samples"""
        # Prepare training data
        pos_tensor = torch.stack([self.transform(p) for p in pos_samples])
        neg_tensor = torch.stack([self.transform(n) for n in neg_samples])
        
        X = torch.cat([pos_tensor, neg_tensor])
        y = torch.cat([torch.ones(len(pos_samples)), 
                       torch.zeros(len(neg_samples))]).long()
        
        # Training
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, 
                                   momentum=0.9, weight_decay=0.0005)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(30):
            optimizer.zero_grad()
            outputs = self.model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        
        self.model.eval()
    
    def _update_online(self, image, bbox):
        """Online adaptation during tracking"""
        # Sample new positive examples
        pos_samples = self._sample_patches(image, bbox, n_samples=20,
                                          trans_f=0.1, scale_f=1.0)
        
        # Use high-scoring samples as positive
        pos_tensor = torch.stack([self.transform(p) for p in pos_samples])
        
        self.model.train()
        optimizer = torch.optim.SGD(self.model.fc6.parameters(), 
                                   lr=self.lr * 0.1, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        # Quick update (5 iterations)
        for _ in range(5):
            optimizer.zero_grad()
            outputs = self.model(pos_tensor)
            labels = torch.ones(len(pos_samples)).long()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        self.model.eval()
    
    def _get_bbox_from_sample_idx(self, idx):
        """Get bbox corresponding to sample index"""
        # Simplified - in practice, store sampled bboxes
        return self.target_bbox


def demo_mdnet():
    """
    Demonstration of MDNet tracker
    """
    print("="*60)
    print("MDNet Tracker Demo")
    print("="*60)
    
    # Create model
    model = MDNet(num_domains=1)
    print(f"\n✓ Created MDNet with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    dummy_input = torch.randn(8, 3, 107, 107)
    output = model(dummy_input)
    print(f"✓ Forward pass: input {dummy_input.shape} → output {output.shape}")
    
    # Analyze architecture
    print("\n" + "="*60)
    print("MDNet Architecture")
    print("="*60)
    
    print("\nShared Convolutional Layers:")
    print("  conv1: 3→96 channels, 7×7 kernel, stride 2")
    print("  conv2: 96→256 channels, 5×5 kernel, stride 2")
    print("  conv3: 256→512 channels, 3×3 kernel, stride 1")
    
    print("\nDomain-Specific FC Layers:")
    print("  fc4: 512×3×3 → 512 (per domain)")
    print("  fc5: 512 → 512 (per domain)")
    
    print("\nClassification Layer:")
    print("  fc6: 512 → 2 (target/background)")
    
    # Test tracker
    print("\n" + "="*60)
    print("Tracker Simulation")
    print("="*60)
    
    tracker = MDNetTracker()
    
    # Simulate first frame
    frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    bbox = [100, 100, 50, 50]  # [x, y, w, h]
    
    print(f"\n✓ Initialize tracker with bbox: {bbox}")
    tracker.init(frame1, bbox)
    
    # Simulate tracking
    for i in range(5):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox, score = tracker.update(frame)
        print(f"  Frame {i+1}: bbox={[int(b) for b in bbox]}, score={score:.3f}")
    
    print("\n" + "="*60)
    print("Key Insights:")
    print("="*60)
    print("1. Multi-domain learning: Pre-train on diverse sequences")
    print("2. Online adaptation: Update model during tracking")
    print("3. Binary classification: Target vs background")
    print("4. Hard negative mining: Focus on difficult examples")
    print("5. Dropout regularization: Prevents overfitting during online learning")


if __name__ == "__main__":
    demo_mdnet()