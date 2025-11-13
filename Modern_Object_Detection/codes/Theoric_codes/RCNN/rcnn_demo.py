"""
R-CNN: Regions with CNN features - Complete Implementation
Reference: "Rich feature hierarchies for accurate object detection" (Girshick et al., 2014)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
import cv2
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import pickle
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class SelectiveSearchProposals:
    """Generate region proposals using Selective Search"""
    
    def __init__(self, scale=500, sigma=0.9, min_size=20):
        self.scale = scale
        self.sigma = sigma
        self.min_size = min_size
    
    def generate_proposals(self, image, max_proposals=2000):
        """
        Generate region proposals using OpenCV's Selective Search
        
        Args:
            image: Input image (H, W, 3)
            max_proposals: Maximum number of proposals
        
        Returns:
            proposals: List of [x, y, w, h] bounding boxes
        """
        # Use OpenCV's selective search
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(image)
        ss.switchToSelectiveSearchFast()  # or switchToSelectiveSearchQuality()
        
        rects = ss.process()
        
        # Convert to [x, y, w, h] format and limit number
        proposals = []
        for i, rect in enumerate(rects[:max_proposals]):
            x, y, w, h = rect
            if w > self.min_size and h > self.min_size:
                proposals.append([x, y, w, h])
        
        return proposals


class SimplifiedProposals:
    """Simplified proposal generation (for environments without selective search)"""
    
    def __init__(self, scales=[0.3, 0.5, 0.7], ratios=[0.5, 1.0, 2.0]):
        self.scales = scales
        self.ratios = ratios
    
    def generate_proposals(self, image, max_proposals=2000):
        """Generate proposals using sliding windows at multiple scales"""
        h, w = image.shape[:2]
        proposals = []
        
        for scale in self.scales:
            for ratio in self.ratios:
                box_w = int(w * scale)
                box_h = int(box_w * ratio)
                
                if box_h > h or box_w > w:
                    continue
                
                # Sliding window with stride
                stride = max(box_w // 4, 20)
                for y in range(0, h - box_h, stride):
                    for x in range(0, w - box_w, stride):
                        proposals.append([x, y, box_w, box_h])
        
        # Shuffle and limit
        np.random.shuffle(proposals)
        return proposals[:max_proposals]


class CNNFeatureExtractor(nn.Module):
    """CNN for feature extraction (ResNet-50 backbone)"""
    
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Load pretrained ResNet-50
        if pretrained:
            resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet50(weights=None)
        
        # Remove final classification layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Feature dimension
        self.feature_dim = 2048
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # R-CNN used 227x227, we use 224x224 for ResNet
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_region_features(self, image, bbox):
        """
        Extract features from a region
        
        Args:
            image: Full image (H, W, 3) numpy array
            bbox: [x, y, w, h]
        
        Returns:
            features: Feature vector [feature_dim]
        """
        x, y, w, h = bbox
        x, y, w, h = max(0, x), max(0, y), max(1, w), max(1, h)
        
        # Crop region
        region = image[y:y+h, x:x+w]
        
        if region.size == 0:
            return torch.zeros(self.feature_dim)
        
        # Warp to fixed size and extract features
        region_tensor = self.transform(region).unsqueeze(0)
        
        with torch.no_grad():
            features = self.features(region_tensor)
            features = features.squeeze()
        
        return features
    
    def forward(self, x):
        """Standard forward pass"""
        return self.features(x).squeeze()


class BoundingBoxRegressor:
    """Bounding box regressor for each class"""
    
    def __init__(self, feature_dim=2048):
        self.feature_dim = feature_dim
        self.regressors = {}  # One regressor per class
        self.scalers = {}
    
    def compute_regression_targets(self, proposal_box, gt_box):
        """
        Compute regression targets t = (t_x, t_y, t_w, t_h)
        
        Args:
            proposal_box: [x, y, w, h]
            gt_box: [x, y, w, h]
        
        Returns:
            targets: [t_x, t_y, t_w, t_h]
        """
        px, py, pw, ph = proposal_box
        gx, gy, gw, gh = gt_box
        
        # Compute targets as in R-CNN paper
        tx = (gx - px) / pw
        ty = (gy - py) / ph
        tw = np.log(gw / pw)
        th = np.log(gh / ph)
        
        return np.array([tx, ty, tw, th])
    
    def train_regressor(self, features, proposals, gt_boxes, class_id):
        """Train regressor for a specific class"""
        X = []
        y = []
        
        for feat, proposal, gt in zip(features, proposals, gt_boxes):
            targets = self.compute_regression_targets(proposal, gt)
            X.append(feat.numpy())
            y.append(targets)
        
        X = np.array(X)
        y = np.array(y)
        
        # Train separate regressors for each coordinate
        from sklearn.linear_model import Ridge
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train Ridge regression
        regressor = Ridge(alpha=1000.0)
        regressor.fit(X_scaled, y)
        
        self.regressors[class_id] = regressor
        self.scalers[class_id] = scaler
    
    def predict(self, feature, proposal_box, class_id):
        """Predict refined box"""
        if class_id not in self.regressors:
            return proposal_box
        
        X = feature.numpy().reshape(1, -1)
        X_scaled = self.scalers[class_id].transform(X)
        
        tx, ty, tw, th = self.regressors[class_id].predict(X_scaled)[0]
        
        px, py, pw, ph = proposal_box
        
        # Apply transformations
        gx = tx * pw + px
        gy = ty * ph + py
        gw = pw * np.exp(tw)
        gh = ph * np.exp(th)
        
        return [gx, gy, gw, gh]


class RCNN:
    """
    Complete R-CNN implementation with 3-stage training
    """
    
    def __init__(self, num_classes=20, feature_dim=2048):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        # Components
        self.proposal_generator = SimplifiedProposals()
        self.cnn = CNNFeatureExtractor(pretrained=True)
        self.svms = {}  # One SVM per class
        self.svm_scalers = {}
        self.bbox_regressors = BoundingBoxRegressor(feature_dim)
        
        self.class_names = [f'class_{i}' for i in range(num_classes)]
    
    def compute_iou(self, box1, box2):
        """Compute IoU between two boxes [x, y, w, h]"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Convert to [x1, y1, x2, y2]
        box1_coords = [x1, y1, x1 + w1, y1 + h1]
        box2_coords = [x2, y2, x2 + w2, y2 + h2]
        
        # Intersection
        xi1 = max(box1_coords[0], box2_coords[0])
        yi1 = max(box1_coords[1], box2_coords[1])
        xi2 = min(box1_coords[2], box2_coords[2])
        yi2 = min(box1_coords[3], box2_coords[3])
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def train_stage1_finetune_cnn(self, images, annotations):
        """
        Stage 1: Fine-tune CNN for detection
        
        Args:
            images: List of images
            annotations: List of {'boxes': [[x,y,w,h], ...], 'labels': [class_id, ...]}
        """
        print("Stage 1: Fine-tuning CNN for detection...")
        
        # Prepare training data
        positive_regions = []
        negative_regions = []
        
        for img, annot in zip(images, annotations):
            proposals = self.proposal_generator.generate_proposals(img)
            gt_boxes = annot['boxes']
            
            for proposal in proposals:
                # Compute max IoU with ground truth
                max_iou = 0
                for gt_box in gt_boxes:
                    iou = self.compute_iou(proposal, gt_box)
                    max_iou = max(max_iou, iou)
                
                # Positive if IoU >= 0.5
                if max_iou >= 0.5:
                    positive_regions.append((img, proposal, 1))
                # Negative if IoU < 0.1
                elif max_iou < 0.1:
                    negative_regions.append((img, proposal, 0))
        
        # Balance dataset
        num_pos = len(positive_regions)
        negative_regions = negative_regions[:num_pos]
        
        all_regions = positive_regions + negative_regions
        np.random.shuffle(all_regions)
        
        print(f"  Positive regions: {len(positive_regions)}")
        print(f"  Negative regions: {len(negative_regions)}")
        
        # Fine-tune CNN (simplified: just extract features with pretrained model)
        # In practice, you would fine-tune the CNN with these regions
        print("  Using pretrained CNN features (fine-tuning skipped for simplicity)")
    
    def train_stage2_svm(self, images, annotations):
        """
        Stage 2: Train SVM classifiers
        
        Args:
            images: List of images
            annotations: List of annotations
        """
        print("\nStage 2: Training SVM classifiers...")
        
        # Extract features and labels for each class
        for class_id in range(self.num_classes):
            print(f"  Training SVM for class {class_id}...")
            
            features_list = []
            labels_list = []
            
            for img, annot in zip(images, annotations):
                proposals = self.proposal_generator.generate_proposals(img, max_proposals=500)
                gt_boxes = annot['boxes']
                gt_labels = annot['labels']
                
                for proposal in proposals:
                    # Extract feature
                    feature = self.cnn.extract_region_features(img, proposal)
                    features_list.append(feature.numpy())
                    
                    # Determine label (positive if IoU > 0.3 with GT of this class)
                    is_positive = False
                    for gt_box, gt_label in zip(gt_boxes, gt_labels):
                        if gt_label == class_id:
                            iou = self.compute_iou(proposal, gt_box)
                            if iou > 0.3:
                                is_positive = True
                                break
                    
                    labels_list.append(1 if is_positive else 0)
            
            # Train SVM
            X = np.array(features_list)
            y = np.array(labels_list)
            
            # Normalize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Degenerate case handling
            # Guard: need at least 1 positive and 1 negative
            unique = np.unique(y)
            pos = int(y.sum())
            neg = int(len(y) - pos)
            if len(unique) < 2 or pos == 0 or neg == 0:
                print(f"    Skipping class {class_id}: needs >=1 positive and >=1 negative "
                    f"(positives={pos}, negatives={neg})")
                continue


            # Train linear SVM
            clf = svm.LinearSVC(C=0.01, max_iter=10000, dual='auto')
            clf.fit(X_scaled, y)
            
            self.svms[class_id] = clf
            self.svm_scalers[class_id] = scaler
            
            print(f"    Positive samples: {sum(y)}, Negative samples: {len(y) - sum(y)}")
    
    def train_stage3_bbox_regression(self, images, annotations):
        """
        Stage 3: Train bounding box regressors
        
        Args:
            images: List of images
            annotations: List of annotations
        """
        print("\nStage 3: Training bounding box regressors...")
        
        for class_id in range(self.num_classes):
            print(f"  Training regressor for class {class_id}...")
            
            features_list = []
            proposals_list = []
            gt_boxes_list = []
            
            for img, annot in zip(images, annotations):
                proposals = self.proposal_generator.generate_proposals(img, max_proposals=500)
                gt_boxes = annot['boxes']
                gt_labels = annot['labels']
                
                for proposal in proposals:
                    # Find matching GT box with IoU > 0.6
                    for gt_box, gt_label in zip(gt_boxes, gt_labels):
                        if gt_label == class_id:
                            iou = self.compute_iou(proposal, gt_box)
                            if iou > 0.6:
                                feature = self.cnn.extract_region_features(img, proposal)
                                features_list.append(feature)
                                proposals_list.append(proposal)
                                gt_boxes_list.append(gt_box)
            
            if len(features_list) > 0:
                self.bbox_regressors.train_regressor(
                    features_list, proposals_list, gt_boxes_list, class_id
                )
                print(f"    Trained on {len(features_list)} samples")
            else:
                print(f"    No training samples found")
    
    def train(self, images, annotations):
        """
        Complete 3-stage training pipeline
        
        Args:
            images: List of images (numpy arrays)
            annotations: List of dicts with 'boxes' and 'labels'
        """
        # Stage 1: Fine-tune CNN
        self.train_stage1_finetune_cnn(images, annotations)
        
        # Stage 2: Train SVMs
        self.train_stage2_svm(images, annotations)
        
        # Stage 3: Train bbox regressors
        self.train_stage3_bbox_regression(images, annotations)
        
        print("\nTraining complete!")
    
    def detect(self, image, conf_threshold=0.5, nms_threshold=0.3):
        """
        Run R-CNN detection on an image
        
        Args:
            image: Input image
            conf_threshold: Confidence threshold
            nms_threshold: NMS IoU threshold
        
        Returns:
            detections: List of (box, class_id, score)
        """
        # Generate proposals
        proposals = self.proposal_generator.generate_proposals(image)
        
        # Extract features
        features = []
        for proposal in proposals:
            feat = self.cnn.extract_region_features(image, proposal)
            features.append(feat)
        
        # Classify with SVMs
        detections = []
        
        for proposal, feature in zip(proposals, features):
            for class_id in range(self.num_classes):
                if class_id not in self.svms:
                    continue
                
                # SVM prediction
                X = feature.numpy().reshape(1, -1)
                X_scaled = self.svm_scalers[class_id].transform(X)
                score = self.svms[class_id].decision_function(X_scaled)[0]
                
                if score > conf_threshold:
                    # Refine box
                    refined_box = self.bbox_regressors.predict(feature, proposal, class_id)
                    detections.append((refined_box, class_id, score))
        
        # Apply NMS
        detections = self.apply_nms(detections, nms_threshold)
        
        return detections
    
    def apply_nms(self, detections, threshold):
        """Apply Non-Maximum Suppression"""
        if len(detections) == 0:
            return []
        
        # Group by class
        detections_by_class = {}
        for box, class_id, score in detections:
            if class_id not in detections_by_class:
                detections_by_class[class_id] = []
            detections_by_class[class_id].append((box, score))
        
        # NMS per class
        final_detections = []
        
        for class_id, dets in detections_by_class.items():
            # Sort by score
            dets.sort(key=lambda x: x[1], reverse=True)
            
            keep = []
            while len(dets) > 0:
                best = dets[0]
                keep.append(best)
                dets = dets[1:]
                
                # Remove overlapping boxes
                new_dets = []
                for det in dets:
                    if self.compute_iou(best[0], det[0]) < threshold:
                        new_dets.append(det)
                dets = new_dets
            
            for box, score in keep:
                final_detections.append((box, class_id, score))
        
        return final_detections
    
    def save_model(self, path):
        """Save trained model"""
        model_data = {
            'svms': self.svms,
            'svm_scalers': self.svm_scalers,
            'bbox_regressors': self.bbox_regressors,
            'num_classes': self.num_classes
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load trained model"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        self.svms = model_data['svms']
        self.svm_scalers = model_data['svm_scalers']
        self.bbox_regressors = model_data['bbox_regressors']
        self.num_classes = model_data['num_classes']
        print(f"Model loaded from {path}")


def visualize_detections(image, detections, class_names=None, save_path='rcnn_detection.png'):
    """Visualize R-CNN detections"""
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    colors = plt.cm.Set3(np.linspace(0, 1, 20))
    
    for box, class_id, score in detections:
        x, y, w, h = box
        rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                 edgecolor=colors[class_id % 20],
                                 facecolor='none')
        ax.add_patch(rect)
        
        label = f'{class_names[class_id] if class_names else class_id}: {score:.2f}'
        ax.text(x, y - 5, label, color='white', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[class_id % 20], alpha=0.7))
    
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Detection visualization saved to {save_path}")
    plt.show()


# Example usage and demonstration
if __name__ == "__main__":
    print("="*60)
    print("R-CNN Implementation Demo")
    print("="*60)
    
    # Create synthetic training data
    def create_synthetic_image(size=400, num_objects=3):
        """Create a synthetic image with colored rectangles"""
        img = np.ones((size, size, 3), dtype=np.uint8) * 255
        boxes = []
        labels = []
        
        for i in range(num_objects):
            w = np.random.randint(40, 100)
            h = np.random.randint(40, 100)
            x = np.random.randint(0, size - w)
            y = np.random.randint(0, size - h)
            
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
            
            boxes.append([x, y, w, h])
            labels.append(i % 3)  # 3 classes
        
        return img, {'boxes': boxes, 'labels': labels}
    
    # Generate training data
    print("\nGenerating synthetic training data...")
    num_train_images = 5
    train_images = []
    train_annotations = []
    
    for i in range(num_train_images):
        img, annot = create_synthetic_image(num_objects=3)
        train_images.append(img)
        train_annotations.append(annot)
        print(f"  Image {i+1}: {len(annot['boxes'])} objects")
    
    # Initialize and train R-CNN
    print("\nInitializing R-CNN...")
    rcnn = RCNN(num_classes=3)
    
    print("\nStarting 3-stage training...")
    rcnn.train(train_images, train_annotations)
    
    # Test on new image
    print("\n" + "="*60)
    print("Testing R-CNN Detection")
    print("="*60)
    
    test_img, test_annot = create_synthetic_image(num_objects=3)
    
    print("\nRunning detection...")
    detections = rcnn.detect(test_img, conf_threshold=0.0, nms_threshold=0.3)
    
    print(f"\nDetected {len(detections)} objects:")
    for i, (box, class_id, score) in enumerate(detections):
        print(f"  Detection {i+1}: Class {class_id}, Score {score:.3f}, Box {[int(b) for b in box]}")
    
    # Visualize
    class_names = ['Red Object', 'Green Object', 'Blue Object']
    visualize_detections(test_img, detections, class_names)
    
    # Save model
    rcnn.save_model('rcnn_model.pkl')
    
    print("\n" + "="*60)
    print("R-CNN Demo Complete!")
    print("="*60)