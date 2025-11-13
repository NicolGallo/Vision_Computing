"""
Evaluation metrics for object detection
"""
import torch
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import tempfile
from pathlib import Path


class COCOEvaluator:
    """
    Evaluator for COCO-style metrics
    
    Computes AP, AP50, AP75, APS, APM, APL using COCO API
    """
    
    def __init__(self, coco_gt, iou_type='bbox'):
        """
        Args:
            coco_gt: COCO ground truth object
            iou_type: 'bbox' or 'segm'
        """
        self.coco_gt = coco_gt
        self.iou_type = iou_type
        self.predictions = []
    
    def add_predictions(self, image_ids, detections):
        """
        Add predictions for evaluation
        
        Args:
            image_ids: List of image IDs
            detections: List of detection lists
                       Each detection: (box, score, label)
                       box: [x1, y1, x2, y2] normalized
        """
        for img_id, dets in zip(image_ids, detections):
            for box, score, label in dets:
                # Convert from normalized xyxy to absolute xywh
                x1, y1, x2, y2 = box
                
                # Get image info for denormalization
                img_info = self.coco_gt.loadImgs(img_id)[0]
                img_w, img_h = img_info['width'], img_info['height']
                
                x1_abs = x1 * img_w
                y1_abs = y1 * img_h
                x2_abs = x2 * img_w
                y2_abs = y2 * img_h
                
                w = x2_abs - x1_abs
                h = y2_abs - y1_abs
                
                # COCO format uses 1-indexed category IDs
                category_id = int(label) + 1
                
                self.predictions.append({
                    'image_id': int(img_id),
                    'category_id': category_id,
                    'bbox': [float(x1_abs), float(y1_abs), float(w), float(h)],
                    'score': float(score)
                })
    
    def evaluate(self):
        """
        Run COCO evaluation
        
        Returns:
            results: Dictionary with metrics
        """
        if len(self.predictions) == 0:
            print("No predictions to evaluate!")
            return {
                'mAP': 0.0,
                'AP50': 0.0,
                'AP75': 0.0,
                'APS': 0.0,
                'APM': 0.0,
                'APL': 0.0
            }
        
        # Create temporary file for predictions
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.predictions, f)
            pred_file = f.name
        
        try:
            # Load predictions
            coco_dt = self.coco_gt.loadRes(pred_file)
            
            # Run evaluation
            coco_eval = COCOeval(self.coco_gt, coco_dt, self.iou_type)
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # Extract metrics
            results = {
                'mAP': coco_eval.stats[0],      # AP @ IoU=0.50:0.95
                'AP50': coco_eval.stats[1],     # AP @ IoU=0.50
                'AP75': coco_eval.stats[2],     # AP @ IoU=0.75
                'APS': coco_eval.stats[3],      # AP for small objects
                'APM': coco_eval.stats[4],      # AP for medium objects
                'APL': coco_eval.stats[5],      # AP for large objects
            }
            
        finally:
            # Clean up temp file
            Path(pred_file).unlink(missing_ok=True)
        
        return results
    
    def reset(self):
        """Reset predictions"""
        self.predictions = []


class DetectionMetrics:
    """
    Simple detection metrics calculator (without COCO API)
    
    Useful for quick validation during training
    """
    
    def __init__(self, num_classes, iou_threshold=0.5):
        """
        Args:
            num_classes: Number of classes
            iou_threshold: IoU threshold for TP
        """
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.tp = [[] for _ in range(self.num_classes)]
        self.fp = [[] for _ in range(self.num_classes)]
        self.scores = [[] for _ in range(self.num_classes)]
        self.num_gt = [0 for _ in range(self.num_classes)]
    
    def update(self, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels):
        """
        Update metrics with predictions and ground truth
        
        Args:
            pred_boxes: Tensor [N, 4] predicted boxes (xyxy)
            pred_scores: Tensor [N] prediction scores
            pred_labels: Tensor [N] predicted labels
            gt_boxes: Tensor [M, 4] ground truth boxes (xyxy)
            gt_labels: Tensor [M] ground truth labels
        """
        from utils.nms import compute_iou
        
        # Convert to numpy for easier processing
        if torch.is_tensor(pred_boxes):
            pred_boxes = pred_boxes.cpu().numpy()
            pred_scores = pred_scores.cpu().numpy()
            pred_labels = pred_labels.cpu().numpy()
        if torch.is_tensor(gt_boxes):
            gt_boxes = gt_boxes.cpu().numpy()
            gt_labels = gt_labels.cpu().numpy()
        
        # Count ground truth per class
        for label in gt_labels:
            self.num_gt[int(label)] += 1
        
        if len(pred_boxes) == 0:
            return
        
        # Sort predictions by score (descending)
        sort_idx = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[sort_idx]
        pred_scores = pred_scores[sort_idx]
        pred_labels = pred_labels[sort_idx]
        
        # Track which GT boxes have been matched
        gt_matched = np.zeros(len(gt_boxes), dtype=bool)
        
        # Process each prediction
        for pred_box, pred_score, pred_label in zip(pred_boxes, pred_scores, pred_labels):
            pred_label = int(pred_label)
            
            # Find matching GT boxes of same class
            matching_gt = np.where(gt_labels == pred_label)[0]
            
            if len(matching_gt) == 0:
                # No GT of this class -> FP
                self.fp[pred_label].append(1)
                self.tp[pred_label].append(0)
                self.scores[pred_label].append(pred_score)
                continue
            
            # Compute IoU with all matching GT boxes
            ious = []
            for gt_idx in matching_gt:
                iou = self._compute_iou_np(pred_box, gt_boxes[gt_idx])
                ious.append((iou, gt_idx))
            
            # Find best IoU
            best_iou, best_gt_idx = max(ious, key=lambda x: x[0])
            
            if best_iou >= self.iou_threshold and not gt_matched[best_gt_idx]:
                # TP
                self.tp[pred_label].append(1)
                self.fp[pred_label].append(0)
                gt_matched[best_gt_idx] = True
            else:
                # FP (either low IoU or GT already matched)
                self.tp[pred_label].append(0)
                self.fp[pred_label].append(1)
            
            self.scores[pred_label].append(pred_score)
    
    def _compute_iou_np(self, box1, box2):
        """Compute IoU between two boxes (numpy)"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        inter_w = max(0, inter_xmax - inter_xmin)
        inter_h = max(0, inter_ymax - inter_ymin)
        inter_area = inter_w * inter_h
        
        # Union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / (union_area + 1e-6)
    
    def compute_ap(self, class_idx):
        """
        Compute AP for a single class
        
        Args:
            class_idx: Class index
        
        Returns:
            ap: Average precision
        """
        if self.num_gt[class_idx] == 0:
            return 0.0
        
        if len(self.tp[class_idx]) == 0:
            return 0.0
        
        # Sort by score
        tp = np.array(self.tp[class_idx])
        fp = np.array(self.fp[class_idx])
        scores = np.array(self.scores[class_idx])
        
        sort_idx = np.argsort(-scores)
        tp = tp[sort_idx]
        fp = fp[sort_idx]
        
        # Compute cumulative TP and FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # Compute precision and recall
        recalls = tp_cumsum / self.num_gt[class_idx]
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Compute AP using 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11
        
        return ap
    
    def compute_map(self):
        """
        Compute mAP across all classes
        
        Returns:
            mAP: Mean average precision
        """
        aps = []
        for class_idx in range(self.num_classes):
            ap = self.compute_ap(class_idx)
            if self.num_gt[class_idx] > 0:  # Only count classes that appear in GT
                aps.append(ap)
        
        if len(aps) == 0:
            return 0.0
        
        return np.mean(aps)
    
    def get_results(self):
        """
        Get all results
        
        Returns:
            results: Dictionary with per-class and mean metrics
        """
        results = {
            'mAP': self.compute_map(),
            'per_class_AP': {}
        }
        
        for class_idx in range(self.num_classes):
            if self.num_gt[class_idx] > 0:
                results['per_class_AP'][class_idx] = self.compute_ap(class_idx)
        
        return results


class AverageMeter:
    """
    Computes and stores the average and current value
    
    Useful for tracking training metrics
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_confusion_matrix(pred_labels, gt_labels, num_classes):
    """
    Calculate confusion matrix
    
    Args:
        pred_labels: Predicted labels [N]
        gt_labels: Ground truth labels [N]
        num_classes: Number of classes
    
    Returns:
        confusion_matrix: [num_classes, num_classes] matrix
    """
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    
    for pred, gt in zip(pred_labels, gt_labels):
        confusion_matrix[gt, pred] += 1
    
    return confusion_matrix