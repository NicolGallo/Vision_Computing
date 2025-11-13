"""
Lab 3: Visualization and Analysis Utilities
VIAR25/26 - Artificial Vision - UVigo
Prof. David Olivieri

Additional utilities for visualization, analysis, and experimentation
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from collections import defaultdict
import cv2

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AttentionVisualizer:
    """Utility class for visualizing attention maps and patterns"""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.attention_maps = {}
        self.hooks = []
        
    def register_attention_hooks(self):
        """Register forward hooks to capture attention maps"""
        def attention_hook(name):
            def hook(module, input, output):
                if hasattr(module, 'attn') and hasattr(module.attn, 'attention_weights'):
                    self.attention_maps[name] = module.attn.attention_weights.detach().cpu()
            return hook
        
        # Register hooks for transformer blocks
        for name, module in self.model.named_modules():
            if 'blocks' in name and 'attn' in name:
                hook = module.register_forward_hook(attention_hook(name))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def visualize_attention_heads(self, image: torch.Tensor, layer_idx: int = -1, 
                                 save_path: Optional[str] = None):
        """Visualize attention patterns for different heads"""
        self.model.eval()
        self.register_attention_hooks()
        
        with torch.no_grad():
            _ = self.model(image.unsqueeze(0).to(self.device))
        
        # Get attention maps from specified layer
        layer_name = f'blocks.{layer_idx}.attn' if layer_idx >= 0 else list(self.attention_maps.keys())[-1]
        
        if layer_name in self.attention_maps:
            attn_map = self.attention_maps[layer_name][0]  # First batch item
            num_heads = attn_map.shape[0]
            
            fig, axes = plt.subplots(2, (num_heads + 1) // 2, figsize=(15, 8))
            axes = axes.flatten() if num_heads > 2 else [axes]
            
            for head in range(num_heads):
                # Extract attention from CLS token to patches
                cls_attention = attn_map[head, 0, 1:].reshape(8, 8)  # Assuming 8x8 patches
                
                im = axes[head].imshow(cls_attention, cmap='viridis')
                axes[head].set_title(f'Head {head + 1}')
                axes[head].set_xticks([])
                axes[head].set_yticks([])
                plt.colorbar(im, ax=axes[head])
            
            plt.suptitle(f'Attention Patterns - Layer {layer_idx}')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
        
        self.remove_hooks()
    
    def visualize_attention_rollout(self, image: torch.Tensor, save_path: Optional[str] = None):
        """Visualize attention rollout across layers"""
        # This is a simplified version - full implementation would require
        # modifying the transformer to return attention weights
        pass


class GradCAMVisualizer:
    """Visualize CNN attention using Grad-CAM"""
    
    def __init__(self, model: nn.Module, target_layer: str, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook = None
        
    def save_gradient(self, grad):
        self.gradients = grad
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def register_hooks(self):
        """Register hooks on target layer"""
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                self.hook = module.register_forward_hook(self.save_activation)
                break
    
    def generate_cam(self, image: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """Generate Grad-CAM heatmap"""
        self.register_hooks()
        
        # Forward pass
        image = image.unsqueeze(0).to(self.device)
        image.requires_grad_()
        
        output = self.model(image)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()
        
        # Generate CAM
        gradients = self.gradients[0]  # Remove batch dimension
        activations = self.activations[0]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=[1, 2])
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU and normalize
        cam = torch.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        # Remove hooks
        if self.hook:
            self.hook.remove()
        
        return cam.cpu().numpy()
    
    def visualize_cam(self, image: torch.Tensor, class_idx: Optional[int] = None,
                     save_path: Optional[str] = None):
        """Visualize Grad-CAM overlay on original image"""
        cam = self.generate_cam(image, class_idx)
        
        # Convert image to numpy
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        
        # Resize CAM to match image size
        cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        
        # Overlay
        overlay = 0.6 * img_np + 0.4 * heatmap
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title('Grad-CAM')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class TrainingVisualizer:
    """Visualize training progress and metrics"""
    
    @staticmethod
    def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
        """Plot training and validation curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss curves
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
        ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_model_comparison(results: Dict[str, Dict[str, Any]], save_path: Optional[str] = None):
        """Create comprehensive model comparison plots"""
        models = list(results.keys())
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Extract metrics
        accuracies = [results[model]['accuracy'] for model in models]
        parameters = [results[model]['parameters'] / 1e6 for model in models]  # In millions
        flops = [results[model]['flops'] / 1e9 for model in models]  # In billions
        latencies = [results[model]['latency'] for model in models]
        memory = [results[model]['memory'] for model in models]
        
        # Accuracy vs Parameters
        axes[0, 0].scatter(parameters, accuracies, s=100, alpha=0.7)
        for i, model in enumerate(models):
            axes[0, 0].annotate(model, (parameters[i], accuracies[i]), 
                               xytext=(5, 5), textcoords='offset points')
        axes[0, 0].set_xlabel('Parameters (Millions)')
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].set_title('Accuracy vs Parameters')
        axes[0, 0].grid(True)
        
        # Accuracy vs FLOPs
        axes[0, 1].scatter(flops, accuracies, s=100, alpha=0.7, color='orange')
        for i, model in enumerate(models):
            axes[0, 1].annotate(model, (flops[i], accuracies[i]), 
                               xytext=(5, 5), textcoords='offset points')
        axes[0, 1].set_xlabel('FLOPs (Billions)')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Accuracy vs FLOPs')
        axes[0, 1].grid(True)
        
        # Accuracy vs Latency
        axes[0, 2].scatter(latencies, accuracies, s=100, alpha=0.7, color='green')
        for i, model in enumerate(models):
            axes[0, 2].annotate(model, (latencies[i], accuracies[i]), 
                               xytext=(5, 5), textcoords='offset points')
        axes[0, 2].set_xlabel('Latency (ms)')
        axes[0, 2].set_ylabel('Accuracy (%)')
        axes[0, 2].set_title('Accuracy vs Latency')
        axes[0, 2].grid(True)
        
        # Bar plots for individual metrics
        x_pos = np.arange(len(models))
        
        axes[1, 0].bar(x_pos, parameters, alpha=0.7)
        axes[1, 0].set_xlabel('Models')
        axes[1, 0].set_ylabel('Parameters (Millions)')
        axes[1, 0].set_title('Model Size Comparison')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(models, rotation=45)
        
        axes[1, 1].bar(x_pos, flops, alpha=0.7, color='orange')
        axes[1, 1].set_xlabel('Models')
        axes[1, 1].set_ylabel('FLOPs (Billions)')
        axes[1, 1].set_title('Computational Complexity')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(models, rotation=45)
        
        axes[1, 2].bar(x_pos, latencies, alpha=0.7, color='green')
        axes[1, 2].set_xlabel('Models')
        axes[1, 2].set_ylabel('Latency (ms)')
        axes[1, 2].set_title('Inference Speed')
        axes[1, 2].set_xticks(x_pos)
        axes[1, 2].set_xticklabels(models, rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class ArchitectureAnalyzer:
    """Analyze and compare different architectural choices"""
    
    @staticmethod
    def analyze_depth_scaling(base_model_fn, depths: List[int], input_size: Tuple[int, ...],
                             device: torch.device) -> Dict[str, List[float]]:
        """Analyze how model performance scales with depth"""
        results = {
            'depths': depths,
            'parameters': [],
            'flops': [],
            'accuracy': [],  # Would need to be filled after training
            'training_time': []
        }
        
        for depth in depths:
            # Create model with specified depth
            model = base_model_fn(depth=depth)
            
            # Count parameters
            params = sum(p.numel() for p in model.parameters())
            results['parameters'].append(params)
            
            # Estimate FLOPs (simplified)
            # In practice, you would use a proper FLOP counter
            flops = params * np.prod(input_size) * 2  # Rough estimate
            results['flops'].append(flops)
        
        return results
    
    @staticmethod
    def analyze_attention_effectiveness(models_with_without_attention: Dict[str, nn.Module],
                                      test_loader, device: torch.device) -> Dict[str, float]:
        """Compare models with and without attention mechanisms"""
        results = {}
        
        for name, model in models_with_without_attention.items():
            model.to(device)
            model.eval()
            
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    outputs = model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            
            accuracy = 100 * correct / total
            results[name] = accuracy
        
        return results
    
    @staticmethod
    def create_efficiency_frontier(models: Dict[str, nn.Module], 
                                  accuracies: Dict[str, float],
                                  input_size: Tuple[int, ...],
                                  device: torch.device,
                                  save_path: Optional[str] = None):
        """Create efficiency frontier plot (accuracy vs computational cost)"""
        model_names = []
        params = []
        accs = []
        
        for name, model in models.items():
            if name in accuracies:
                model_names.append(name)
                params.append(sum(p.numel() for p in model.parameters()) / 1e6)
                accs.append(accuracies[name])
        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(params, accs, s=100, alpha=0.7, c=range(len(model_names)), 
                            cmap='viridis')
        
        for i, name in enumerate(model_names):
            plt.annotate(name, (params[i], accs[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Parameters (Millions)')
        plt.ylabel('Accuracy (%)')
        plt.title('Model Efficiency Frontier')
        plt.grid(True, alpha=0.3)
        
        # Add efficiency frontier line
        sorted_pairs = sorted(zip(params, accs))
        frontier_params, frontier_accs = zip(*sorted_pairs)
        plt.plot(frontier_params, frontier_accs, 'r--', alpha=0.5, label='Efficiency Frontier')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class ExperimentLogger:
    """Log and track experiments"""
    
    def __init__(self):
        self.experiments = []
    
    def log_experiment(self, name: str, config: Dict[str, Any], 
                      results: Dict[str, Any]):
        """Log a single experiment"""
        experiment = {
            'name': name,
            'config': config,
            'results': results,
            'timestamp': pd.Timestamp.now()
        }
        self.experiments.append(experiment)
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Convert experiments to pandas DataFrame for analysis"""
        if not self.experiments:
            return pd.DataFrame()
        
        # Flatten the experiment data
        rows = []
        for exp in self.experiments:
            row = {'name': exp['name'], 'timestamp': exp['timestamp']}
            row.update(exp['config'])
            row.update(exp['results'])
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def plot_experiment_comparison(self, metric: str = 'accuracy', 
                                 save_path: Optional[str] = None):
        """Plot comparison of experiments"""
        df = self.get_results_dataframe()
        
        if df.empty or metric not in df.columns:
            print(f"No data available for metric: {metric}")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Bar plot
        plt.subplot(1, 2, 1)
        plt.bar(range(len(df)), df[metric])
        plt.xlabel('Experiment')
        plt.ylabel(metric.capitalize())
        plt.title(f'{metric.capitalize()} Comparison')
        plt.xticks(range(len(df)), df['name'], rotation=45)
        
        # Timeline plot
        plt.subplot(1, 2, 2)
        plt.plot(df['timestamp'], df[metric], 'o-')
        plt.xlabel('Time')
        plt.ylabel(metric.capitalize())
        plt.title(f'{metric.capitalize()} Over Time')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def export_results(self, filepath: str):
        """Export results to CSV"""
        df = self.get_results_dataframe()
        df.to_csv(filepath, index=False)
        print(f"Results exported to {filepath}")


# Example usage functions
def demonstrate_attention_visualization():
    """Demonstrate attention visualization capabilities"""
    print("Attention Visualization Demo")
    print("This would show attention patterns in trained models")
    # Implementation would require trained models
    pass

def demonstrate_gradcam():
    """Demonstrate Grad-CAM visualization"""
    print("Grad-CAM Visualization Demo")
    print("This would show class activation maps for CNN models")
    # Implementation would require trained models
    pass

def create_sample_comparison_report():
    """Create a sample comparison report with synthetic data"""
    # Generate synthetic results for demonstration
    models = ['ResNet-18', 'ResNet-50', 'SE-ResNet-18', 'ViT-Tiny', 'ViT-Small']
    
    results = {}
    for i, model in enumerate(models):
        results[model] = {
            'accuracy': 85 + np.random.normal(0, 3),  # Random around 85%
            'parameters': (10 + i * 20) * 1e6,  # Increasing parameters
            'flops': (2 + i * 5) * 1e9,  # Increasing FLOPs
            'latency': 5 + i * 10,  # Increasing latency
            'memory': 100 + i * 200  # Increasing memory
        }
    
    # Create visualization
    visualizer = TrainingVisualizer()
    visualizer.plot_model_comparison(results, 'sample_comparison.png')
    
    return results

if __name__ == "__main__":
    print("Lab 3: Visualization and Analysis Utilities")
    print("=" * 50)
    
    # Create sample comparison report
    print("Creating sample comparison report...")
    results = create_sample_comparison_report()
    
    # Display results table
    print("\nSample Results:")
    df = pd.DataFrame(results).T
    print(df.round(2))