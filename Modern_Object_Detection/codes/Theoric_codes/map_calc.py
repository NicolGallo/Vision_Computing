import numpy as np
import matplotlib.pyplot as plt

def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def compute_ap(precision, recall):
    """Compute AP from precision-recall curve"""
    # Add sentinel values
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    
    # Compute envelope (monotone decreasing)
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    # Area under curve
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return ap

def compute_map(detections, ground_truth, iou_thresholds):
    """
    Compute mAP
    
    Args:
        detections: list of (bbox, class, score)
        ground_truth: list of (bbox, class)
        iou_thresholds: list of IoU thresholds [0.5, 0.55, ..., 0.95]
    
    Returns:
        mAP score
    """
    classes = set([gt[1] for gt in ground_truth])
    ap_scores = []
    
    for threshold in iou_thresholds:
        class_aps = []
        
        for c in classes:
            # Filter by class
            dets_c = [(bbox, score) for bbox, cls, score in detections if cls == c]
            gt_c = [bbox for bbox, cls in ground_truth if cls == c]
            
            if len(gt_c) == 0:
                continue
            
            # Sort by confidence descending
            dets_c.sort(key=lambda x: x[1], reverse=True)
            
            # Initialize tracking arrays
            tp = np.zeros(len(dets_c))
            fp = np.zeros(len(dets_c))
            matched_gt = np.zeros(len(gt_c))
            
            # Match detections to GT
            for i, (det_bbox, _) in enumerate(dets_c):
                max_iou = 0
                max_idx = -1
                
                for j, gt_bbox in enumerate(gt_c):
                    if matched_gt[j] == 0:  # Not matched yet
                        iou = compute_iou(det_bbox, gt_bbox)
                        if iou > max_iou:
                            max_iou = iou
                            max_idx = j
                
                if max_iou >= threshold:
                    tp[i] = 1
                    matched_gt[max_idx] = 1
                else:
                    fp[i] = 1
            
            # Compute precision-recall
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            recall = tp_cumsum / len(gt_c)
            precision = tp_cumsum / (tp_cumsum + fp_cumsum)
            
            # Compute AP
            ap = compute_ap(precision, recall)
            class_aps.append(ap)
        
        ap_scores.append(np.mean(class_aps) if class_aps else 0)
    
    return np.mean(ap_scores)

# Example usage and visualization
if __name__ == "__main__":
    # Example detections and ground truth
    detections = [
        ([0.1, 0.1, 0.3, 0.3], 0, 0.95),
        ([0.15, 0.15, 0.35, 0.35], 0, 0.85),
        ([0.5, 0.5, 0.7, 0.7], 1, 0.90),
        ([0.2, 0.6, 0.4, 0.8], 0, 0.80),
        ([0.55, 0.55, 0.75, 0.75], 1, 0.75),
        ([0.3, 0.3, 0.5, 0.5], 0, 0.60),
    ]
    
    ground_truth = [
        ([0.1, 0.1, 0.3, 0.3], 0),
        ([0.5, 0.5, 0.7, 0.7], 1),
        ([0.2, 0.6, 0.4, 0.8], 0),
    ]
    
    # COCO-style IoU thresholds
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    
    # Compute mAP
    map_score = compute_map(detections, ground_truth, iou_thresholds)
    print(f"mAP@[0.5:0.95]: {map_score:.3f}")
    
    # Generate visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Left: PR Curve for class 0 at IoU=0.5
    c = 0
    threshold = 0.5
    dets_c = [(bbox, score) for bbox, cls, score in detections if cls == c]
    gt_c = [bbox for bbox, cls in ground_truth if cls == c]
    dets_c.sort(key=lambda x: x[1], reverse=True)
    
    tp = np.zeros(len(dets_c))
    fp = np.zeros(len(dets_c))
    matched_gt = np.zeros(len(gt_c))
    
    for i, (det_bbox, _) in enumerate(dets_c):
        max_iou = 0
        max_idx = -1
        for j, gt_bbox in enumerate(gt_c):
            if matched_gt[j] == 0:
                iou = compute_iou(det_bbox, gt_bbox)
                if iou > max_iou:
                    max_iou = iou
                    max_idx = j
        if max_iou >= threshold:
            tp[i] = 1
            matched_gt[max_idx] = 1
        else:
            fp[i] = 1
    
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recall = tp_cumsum / len(gt_c)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    ax1.plot(recall, precision, 'b-', linewidth=2.5, marker='o', markersize=6)
    ax1.fill_between(recall, precision, alpha=0.2)
    ax1.set_xlabel('Recall', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Precision', fontsize=13, fontweight='bold')
    ax1.set_title('Precision-Recall Curve\n(Class 0, IoU≥0.5)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.05])
    ap = compute_ap(precision, recall)
    ax1.text(0.5, 0.3, f'AP = {ap:.3f}', fontsize=15, 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
    
    # Right: AP vs IoU threshold
    map_per_threshold = []
    for t in iou_thresholds:
        map_t = compute_map(detections, ground_truth, [t])
        map_per_threshold.append(map_t)
    
    ax2.plot(iou_thresholds, map_per_threshold, 'r-', linewidth=2.5, marker='s', markersize=6)
    ax2.set_xlabel('IoU Threshold', fontsize=13, fontweight='bold')
    ax2.set_ylabel('AP', fontsize=13, fontweight='bold')
    ax2.set_title('AP Sensitivity to IoU Threshold', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=map_score, color='green', linestyle='--', linewidth=2, 
                label=f'mAP = {map_score:.3f}')
    ax2.legend(fontsize=11, loc='upper right')
    ax2.set_xlim([0.5, 0.95])
    
    plt.tight_layout()
    plt.savefig('map_visualization.pdf', bbox_inches='tight', dpi=200)
    plt.savefig('map_visualization.png', bbox_inches='tight', dpi=200)
    print("Visualization saved as 'map_visualization.pdf' and 'map_visualization.png'")
    plt.show()