#!/usr/bin/env python3
"""
Convert old checkpoints (with optimizer_state_dict) to new secure format
"""
import torch
import os

# Checkpoints to convert
old_dir = "old_checkpoints"
models = ['fcn32s', 'fcn16s', 'fcn8s', 'deeplabv3plus', 'minisam']

for model_name in models:
    old_path = os.path.join(old_dir, f'best_{model_name}_model.pth')
    new_path = f'best_{model_name}_model.pth'
    
    if os.path.exists(old_path):
        print(f"Converting {model_name}...")
        
        # Load old checkpoint (with weights_only=False)
        old_checkpoint = torch.load(old_path, map_location='cpu', weights_only=False)
        
        # Create new checkpoint with only essential data
        new_checkpoint = {
            'model_state_dict': old_checkpoint['model_state_dict'],
            'epoch': int(old_checkpoint.get('epoch', 0)),
            'best_miou': float(old_checkpoint.get('best_miou', 0.0))
        }
        
        # Save in new format (compatible with weights_only=True)
        torch.save(new_checkpoint, new_path)
        
        print(f"  ✓ Converted: {old_path} -> {new_path}")
        print(f"    Best mIoU: {new_checkpoint['best_miou']:.4f}")
        print(f"    Epoch: {new_checkpoint['epoch']}")
    else:
        print(f"  ✗ Not found: {old_path}")

print("\n✓ Conversion complete!")
print("Old checkpoints are preserved in old_checkpoints/")
