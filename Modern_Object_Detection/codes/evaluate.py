"""
Evaluation script for object detection models
"""
import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
from tqdm import tqdm
import json

from config import config
from models import YOLODetector, YOLODecoder, FCOSDetector, FCOSDecoder
from datasets import COCODetectionDataset, collate_fn, get_transforms
from utils import COCOEvaluator


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate object detection model')
    
    parser.add_argument('--model', type=str, required=True, choices=['yolo', 'fcos'],
                       help='Model architecture')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--conf-threshold', type=float, default=None,
                       help='Confidence threshold')
    parser.add_argument('--nms-threshold', type=float, default=None,
                       help='NMS IoU threshold')
    parser.add_argument('--save-results', action='store_true',
                       help='Save detection results')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize predictions')
    parser.add_argument('--num-vis', type=int, default=10,
                       help='Number of images to visualize')
    
    args = parser.parse_args()
    return args


def evaluate_model(model, decoder, data_loader, coco_gt, device, save_results=False, visualize=False, num_vis=10):
    """
    Evaluate model on validation set
    
    Args:
        model: Detection model
        decoder: Decoder (YOLODecoder or FCOSDecoder)
        data_loader: Data loader
        coco_gt: COCO ground truth object
        device: Device
        save_results: Whether to save detection results
        visualize: Whether to visualize predictions
        num_vis: Number of images to visualize
    
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    model.eval()
    
    evaluator = COCOEvaluator(coco_gt)
    
    all_detections = []
    vis_count = 0
    
    with torch.no_grad():
        for images, boxes, labels, img_ids in tqdm(data_loader, desc='Evaluating'):
            images = images.to(device)
            
            # Get predictions
            if isinstance(model, YOLODetector):
                predictions = model(images)
                detections = decoder.decode(predictions)
            else:  # FCOS
                cls_logits, reg_preds, centerness = model(images)
                detections = decoder.decode(cls_logits, reg_preds, centerness)
            
            # Add to evaluator
            evaluator.add_predictions(img_ids, detections)
            
            # Save for later
            if save_results or visualize:
                for img_id, dets in zip(img_ids, detections):
                    all_detections.append({
                        'image_id': img_id,
                        'detections': dets
                    })
            
            # Visualize
            if visualize and vis_count < num_vis:
                from utils.visualization import compare_predictions_and_gt
                
                for i in range(min(len(images), num_vis - vis_count)):
                    # Get predictions for this image
                    pred_boxes = []
                    pred_labels = []
                    pred_scores = []
                    
                    for box, score, label in detections[i]:
                        pred_boxes.append(box)
                        pred_labels.append(label)
                        pred_scores.append(score)
                    
                    # Visualize
                    output_path = config.output_dir / f'pred_{img_ids[i]}.png'
                    compare_predictions_and_gt(
                        images[i],
                        pred_boxes, pred_labels, pred_scores,
                        boxes[i], labels[i],
                        save_path=output_path
                    )
                    
                    vis_count += 1
                    if vis_count >= num_vis:
                        break
    
    # Evaluate using COCO API
    print("\nRunning COCO evaluation...")
    metrics = evaluator.evaluate()
    
    # Save results
    if save_results:
        results_file = config.output_dir / 'detection_results.json'
        with open(results_file, 'w') as f:
            json.dump(all_detections, f)
        print(f"\nResults saved to {results_file}")
    
    return metrics


def print_metrics(metrics):
    """Print evaluation metrics"""
    print("\n" + "="*60)
    print("COCO Evaluation Results")
    print("="*60)
    print(f"{'Metric':<20} {'Value':>10}")
    print("-"*60)
    print(f"{'mAP (0.50:0.95)':<20} {metrics['mAP']:>10.3f}")
    print(f"{'AP @ IoU=0.50':<20} {metrics['AP50']:>10.3f}")
    print(f"{'AP @ IoU=0.75':<20} {metrics['AP75']:>10.3f}")
    print("-"*60)
    print(f"{'AP (small)':<20} {metrics['APS']:>10.3f}")
    print(f"{'AP (medium)':<20} {metrics['APM']:>10.3f}")
    print(f"{'AP (large)':<20} {metrics['APL']:>10.3f}")
    print("="*60)


def main():
    """Main evaluation function"""
    args = parse_args()
    
    # Set device
    device = args.device if args.device else config.device
    
    # Set thresholds
    conf_threshold = args.conf_threshold if args.conf_threshold else config.conf_threshold
    nms_threshold = args.nms_threshold if args.nms_threshold else config.nms_threshold
    
    print("="*60)
    print(f"Evaluating {args.model.upper()} detector")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"NMS threshold: {nms_threshold}")
    print("="*60)
    
    # Create dataset
    print("\nLoading validation dataset...")
    val_dataset = COCODetectionDataset(
        config.val_images,
        config.val_ann,
        transform=get_transforms(is_train=False, input_size=config.input_size),
        is_train=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"Validation set: {len(val_dataset)} images")
    
    # Create model
    print(f"\nCreating {args.model.upper()} model...")
    if args.model == 'yolo':
        model = YOLODetector(
            num_classes=config.num_classes,
            grid_size=config.grid_size,
            num_boxes=config.num_boxes,
            backbone=config.yolo_backbone
        )
        decoder = YOLODecoder(
            grid_size=config.grid_size,
            num_boxes=config.num_boxes,
            num_classes=config.num_classes,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold
        )
    else:  # FCOS
        model = FCOSDetector(
            num_classes=config.num_classes,
            backbone=config.fcos_backbone,
            fpn_channels=config.fpn_channels,
            num_convs=config.num_convs
        )
        decoder = FCOSDecoder(
            strides=config.fpn_strides,
            num_classes=config.num_classes,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            input_size=config.input_size
        )
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    if 'epoch' in checkpoint:
        print(f"Checkpoint from epoch {checkpoint['epoch']}")
    if 'best_map' in checkpoint:
        print(f"Best mAP from training: {checkpoint['best_map']:.3f}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Evaluate
    print("\nRunning evaluation...")
    metrics = evaluate_model(
        model, decoder, val_loader, val_dataset.coco, device,
        save_results=args.save_results,
        visualize=args.visualize,
        num_vis=args.num_vis
    )
    
    # Print results
    print_metrics(metrics)
    
    # Save metrics
    metrics_file = config.output_dir / f'{args.model}_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_file}")
    
    if args.visualize:
        print(f"Visualizations saved to {config.output_dir}")


if __name__ == '__main__':
    main()