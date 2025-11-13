"""
Training script for object detection models
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
from pathlib import Path
from tqdm import tqdm
import json

from config import config
from models import YOLODetector, FCOSDetector
from datasets import COCODetectionDataset, collate_fn, get_transforms
from losses import YOLOLoss, FCOSLoss
from utils import AverageMeter


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train object detection model')
    
    parser.add_argument('--model', type=str, default='fcos', choices=['yolo', 'fcos'],
                       help='Model architecture')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (default: from config)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (default: from config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (default: from config)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu, default: from config)')
    parser.add_argument('--num-workers', type=int, default=None,
                       help='Number of data loading workers')
    parser.add_argument('--exp-name', type=str, default=None,
                       help='Experiment name for logging')
    
    args = parser.parse_args()
    return args


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, writer, global_step):
    """
    Train for one epoch
    
    Args:
        model: Detection model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device
        epoch: Current epoch
        writer: TensorBoard writer
        global_step: Global training step
    
    Returns:
        avg_loss: Average loss for epoch
        global_step: Updated global step
    """
    model.train()
    
    loss_meter = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (images, boxes, labels, _) in enumerate(pbar):
        images = images.to(device)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]
        
        # Forward pass
        if isinstance(model, YOLODetector):
            predictions = model(images)
            loss, loss_dict = criterion(predictions, boxes, labels)
        else:  # FCOS
            cls_logits, reg_preds, centerness = model(images)
            loss, loss_dict = criterion(cls_logits, reg_preds, centerness, boxes, labels)
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"Warning: NaN loss detected at step {global_step}")
            continue
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
        
        optimizer.step()
        
        # Update metrics
        loss_meter.update(loss.item(), images.size(0))
        
        # Logging
        if batch_idx % config.log_interval == 0:
            # Log to tensorboard
            writer.add_scalar('train/loss', loss.item(), global_step)
            for key, value in loss_dict.items():
                writer.add_scalar(f'train/{key}', value, global_step)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{loss_meter.avg:.4f}'
            })
        
        global_step += 1
    
    return loss_meter.avg, global_step


def validate(model, val_loader, criterion, device, epoch, writer):
    """
    Validate model
    
    Args:
        model: Detection model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device
        epoch: Current epoch
        writer: TensorBoard writer
    
    Returns:
        avg_loss: Average validation loss
    """
    model.eval()
    
    loss_meter = AverageMeter()
    
    with torch.no_grad():
        for images, boxes, labels, _ in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            
            # Forward pass
            if isinstance(model, YOLODetector):
                predictions = model(images)
                loss, loss_dict = criterion(predictions, boxes, labels)
            else:  # FCOS
                cls_logits, reg_preds, centerness = model(images)
                loss, loss_dict = criterion(cls_logits, reg_preds, centerness, boxes, labels)
            
            loss_meter.update(loss.item(), images.size(0))
    
    # Log validation loss
    writer.add_scalar('val/loss', loss_meter.avg, epoch)
    
    return loss_meter.avg


def save_checkpoint(model, optimizer, epoch, best_map, save_path):
    """Save checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_map': best_map
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_map = checkpoint.get('best_map', 0.0)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, best mAP: {best_map:.4f}")
    return start_epoch, best_map


def main():
    """Main training function"""
    args = parse_args()
    
    # Override config with command line arguments
    batch_size = args.batch_size if args.batch_size else config.batch_size
    num_epochs = args.epochs if args.epochs else config.num_epochs
    learning_rate = args.lr if args.lr else config.learning_rate
    device = args.device if args.device else config.device
    num_workers = args.num_workers if args.num_workers else config.num_workers
    
    # Set up experiment name
    exp_name = args.exp_name if args.exp_name else f'{args.model}_{batch_size}bs'
    
    # Create directories
    checkpoint_dir = config.checkpoint_dir / exp_name
    log_dir = config.log_dir / exp_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir)
    
    print("="*60)
    print(f"Training {args.model.upper()} detector")
    print(f"Experiment: {exp_name}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print("="*60)
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = COCODetectionDataset(
        config.train_images,
        config.train_ann,
        transform=get_transforms(is_train=True, input_size=config.input_size),
        is_train=True
    )
    
    val_dataset = COCODetectionDataset(
        config.val_images,
        config.val_ann,
        transform=get_transforms(is_train=False, input_size=config.input_size),
        is_train=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"Train set: {len(train_dataset)} images")
    print(f"Val set: {len(val_dataset)} images")
    
    # Create model
    print(f"\nCreating {args.model.upper()} model...")
    if args.model == 'yolo':
        model = YOLODetector(
            num_classes=config.num_classes,
            grid_size=config.grid_size,
            num_boxes=config.num_boxes,
            backbone=config.yolo_backbone
        )
        criterion = YOLOLoss(
            num_classes=config.num_classes,
            grid_size=config.grid_size,
            num_boxes=config.num_boxes,
            lambda_coord=config.lambda_coord,
            lambda_noobj=config.lambda_noobj
        )
    else:  # FCOS
        model = FCOSDetector(
            num_classes=config.num_classes,
            backbone=config.fcos_backbone,
            fpn_channels=config.fpn_channels,
            num_convs=config.num_convs
        )
        criterion = FCOSLoss(
            num_classes=config.num_classes,
            strides=config.fpn_strides,
            scale_ranges=config.fpn_scales,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma
        )
    
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs
    )
    
    # Resume from checkpoint
    start_epoch = 0
    best_map = 0.0
    global_step = 0
    
    if args.resume:
        start_epoch, best_map = load_checkpoint(model, optimizer, args.resume)
        global_step = start_epoch * len(train_loader)
    
    # Training loop
    print("\nStarting training...")
    print("="*60)
    
    training_log = []
    
    for epoch in range(start_epoch, num_epochs):
        # Train
        train_loss, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, writer, global_step
        )
        
        print(f'\nEpoch {epoch}/{num_epochs-1}:')
        print(f'  Train Loss: {train_loss:.4f}')
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device, epoch, writer)
        print(f'  Val Loss: {val_loss:.4f}')
        
        # Learning rate step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/lr', current_lr, epoch)
        print(f'  Learning Rate: {current_lr:.6f}')
        
        # Save checkpoint
        if (epoch + 1) % config.save_interval == 0:
            save_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            save_checkpoint(model, optimizer, epoch, best_map, save_path)
        
        # Save best model (based on validation loss for now)
        # In full training, you would evaluate mAP here
        if val_loss < train_loss:  # Simple heuristic
            save_path = checkpoint_dir / 'best.pth'
            save_checkpoint(model, optimizer, epoch, best_map, save_path)
            print(f'  Saved best model!')
        
        # Log
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': current_lr
        }
        training_log.append(log_entry)
        
        # Save training log
        log_file = checkpoint_dir / 'training_log.json'
        with open(log_file, 'w') as f:
            json.dump(training_log, f, indent=2)
        
        print("="*60)
    
    # Save final model
    final_path = checkpoint_dir / 'final.pth'
    save_checkpoint(model, optimizer, num_epochs-1, best_map, final_path)
    
    writer.close()
    
    print("\nTraining complete!")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"Logs saved to: {log_dir}")


if __name__ == '__main__':
    main()