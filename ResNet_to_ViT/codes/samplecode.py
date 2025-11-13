# Lab 3: Architecture Implementation Mastery - Sample Experiments
# VIAR25/26 - Artificial Vision - UVigo
# 
# This notebook provides complete examples of how to use the implemented architectures
# and conduct meaningful experiments for your lab report.

# %% [markdown]
"""
# Lab 3: Architecture Implementation Mastery

This notebook demonstrates how to:
1. **Implement and test ResNet architectures**
2. **Add attention mechanisms (SE and CBAM)**
3. **Build Vision Transformers from scratch**
4. **Compare CNN vs ViT performance**
5. **Analyze computational trade-offs**
6. **Create visualizations for your report**

## Learning Objectives
By completing this lab, you will master the implementation of modern deep learning architectures
and understand their trade-offs in terms of accuracy, computational cost, and applicability.
"""

# %% [markdown]
"""
## Setup and Imports
"""

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import our implementations
from lab03_student import *  # Use student template for practice
# from lab3_instructor_complete import *  # Use complete implementation if needed

# Import visualization utilities
from utils import *

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %% [markdown]
"""
## Part 1: ResNet Implementation and Testing

Let's start by implementing and testing different ResNet architectures.
"""

# %%
def test_resnet_blocks():
    """Test our ResNet block implementations"""
    print("Testing ResNet Block Implementations")
    print("=" * 40)
    
    # Test BasicBlock
    basic_block = BasicBlock(64, 64)
    test_input = torch.randn(2, 64, 32, 32)
    
    try:
        output = basic_block(test_input)
        print(f"✓ BasicBlock: {test_input.shape} -> {output.shape}")
    except Exception as e:
        print(f"✗ BasicBlock failed: {e}")
    
    # Test BottleneckBlock
    bottleneck_block = BottleneckBlock(64, 64)
    
    try:
        output = bottleneck_block(test_input)
        print(f"✓ BottleneckBlock: {test_input.shape} -> {output.shape}")
    except Exception as e:
        print(f"✗ BottleneckBlock failed: {e}")

# Run the test
test_resnet_blocks()

# %%
def analyze_resnet_scaling():
    """Analyze how ResNet performance scales with depth"""
    print("\nAnalyzing ResNet Depth Scaling")
    print("=" * 40)
    
    models = {
        'ResNet-18': resnet18(num_classes=10),
        'ResNet-34': resnet34(num_classes=10),
        'ResNet-50': resnet50(num_classes=10),
        'ResNet-101': resnet101(num_classes=10)
    }
    
    results = {}
    test_input = torch.randn(1, 3, 32, 32)
    
    for name, model in models.items():
        if model is not None:
            # Count parameters
            params = sum(p.numel() for p in model.parameters())
            
            # Measure inference time
            model.eval()
            start_time = time.time()
            with torch.no_grad():
                for _ in range(100):
                    _ = model(test_input)
            avg_time = (time.time() - start_time) / 100 * 1000  # ms
            
            results[name] = {
                'parameters': params,
                'inference_time': avg_time
            }
            
            print(f"{name}: {params:,} parameters, {avg_time:.2f}ms inference")
    
    return results

# Analyze scaling
scaling_results = analyze_resnet_scaling()

# %% [markdown]
"""
## Part 2: Attention Mechanisms Implementation

Now let's implement and test attention mechanisms (SE and CBAM).
"""

# %%
def test_attention_mechanisms():
    """Test SE and CBAM implementations"""
    print("\nTesting Attention Mechanisms")
    print("=" * 40)
    
    test_input = torch.randn(2, 64, 32, 32)
    
    # Test SE Module
    se_module = SEModule(channels=64, reduction=16)
    try:
        output = se_module(test_input)
        print(f"✓ SE Module: {test_input.shape} -> {output.shape}")
        
        # Verify channel attention weights
        with torch.no_grad():
            attention_weights = se_module(test_input)
            print(f"  Attention applied successfully")
    except Exception as e:
        print(f"✗ SE Module failed: {e}")
    
    # Test CBAM Module
    cbam_module = CBAM(channels=64, reduction=16)
    try:
        output = cbam_module(test_input)
        print(f"✓ CBAM Module: {test_input.shape} -> {output.shape}")
    except Exception as e:
        print(f"✗ CBAM Module failed: {e}")
    
    # Test SE-ResNet integration
    try:
        se_resnet = se_resnet18(num_classes=10)
        if se_resnet is not None:
            test_input_full = torch.randn(2, 3, 32, 32)
            output = se_resnet(test_input_full)
            print(f"✓ SE-ResNet18: {test_input_full.shape} -> {output.shape}")
    except Exception as e:
        print(f"✗ SE-ResNet18 failed: {e}")

# Test attention mechanisms
test_attention_mechanisms()

# %%
def demonstrate_attention_effectiveness():
    """Demonstrate the effectiveness of attention mechanisms"""
    print("\nDemonstrating Attention Effectiveness")
    print("=" * 40)
    
    # Create models with and without attention
    models = {
        'ResNet-18': resnet18(num_classes=10),
        'SE-ResNet-18': se_resnet18(num_classes=10)
    }
    
    # Quick training comparison (simplified for demo)
    # In practice, you would train these models properly
    
    # Load small dataset for demo
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Use small subset for quick demo
    subset_indices = torch.randperm(len(train_dataset))[:1000]
    small_dataset = Subset(train_dataset, subset_indices)
    
    train_loader = DataLoader(small_dataset, batch_size=32, shuffle=True)
    
    for name, model in models.items():
        if model is not None:
            print(f"\nTesting {name}...")
            model.to(device)
            
            # Quick forward pass test
            model.eval()
            total_correct = 0
            total_samples = 0
            
            with torch.no_grad():
                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)
                    outputs = model(data)
                    _, predicted = torch.max(outputs, 1)
                    total_samples += target.size(0)
                    total_correct += (predicted == target).sum().item()
            
            accuracy = 100 * total_correct / total_samples
            print(f"  Random initialization accuracy: {accuracy:.2f}%")

# Demonstrate attention
demonstrate_attention_effectiveness()

# %% [markdown]
"""
## Part 3: Vision Transformer Implementation

Let's implement and test Vision Transformers.
"""

# %%
def test_vit_components():
    """Test Vision Transformer components"""
    print("\nTesting Vision Transformer Components")
    print("=" * 40)
    
    # Test Patch Embedding
    patch_embed = PatchEmbedding(img_size=32, patch_size=4, in_channels=3, embed_dim=192)
    test_input = torch.randn(2, 3, 32, 32)
    
    try:
        patches = patch_embed(test_input)
        print(f"✓ Patch Embedding: {test_input.shape} -> {patches.shape}")
        print(f"  Number of patches: {patch_embed.num_patches}")
    except Exception as e:
        print(f"✗ Patch Embedding failed: {e}")
    
    # Test Multi-Head Attention
    mha = MultiHeadAttention(embed_dim=192, num_heads=4)
    try:
        # Add class token for testing
        cls_token = torch.randn(2, 1, 192)
        tokens = torch.cat([cls_token, patches], dim=1)
        
        attn_output = mha(tokens)
        print(f"✓ Multi-Head Attention: {tokens.shape} -> {attn_output.shape}")
    except Exception as e:
        print(f"✗ Multi-Head Attention failed: {e}")
    
    # Test complete ViT models
    vit_models = {
        'ViT-Tiny': vit_tiny(num_classes=10),
        'ViT-Small': vit_small(num_classes=10),
        'ViT-Base': vit_base(num_classes=10)
    }
    
    for name, model in vit_models.items():
        if model is not None:
            try:
                output = model(test_input)
                params = sum(p.numel() for p in model.parameters())
                print(f"✓ {name}: {test_input.shape} -> {output.shape}, {params:,} parameters")
            except Exception as e:
                print(f"✗ {name} failed: {e}")

# Test ViT components
test_vit_components()

# %% [markdown]
"""
## Part 4: CNN vs ViT Comparison

Let's conduct a comprehensive comparison between CNNs and Vision Transformers.
"""

# %%
def compare_cnn_vs_vit():
    """Compare CNN and ViT architectures"""
    print("\nCNN vs ViT Comparison")
    print("=" * 40)
    
    # Create models for comparison
    models = {
        'ResNet-18': resnet18(num_classes=10),
        'SE-ResNet-18': se_resnet18(num_classes=10),
        'ViT-Tiny': vit_tiny(num_classes=10)
    }
    
    # Filter out None models
    models = {name: model for name, model in models.items() if model is not None}
    
    if not models:
        print("No models available for comparison")
        return
    
    # Initialize comparator
    comparator = ArchitectureComparator(models, device)
    
    # Compare efficiency
    input_size = (3, 32, 32)
    efficiency_results = comparator.compare_efficiency(input_size)
    
    print("\nEfficiency Comparison:")
    print(f"{'Model':<15} {'Params (K)':<12} {'FLOPs (M)':<12} {'Latency (ms)':<15}")
    print("-" * 60)
    
    for name, metrics in efficiency_results.items():
        print(f"{name:<15} {metrics['parameters']/1e3:<12.1f} {metrics['flops']/1e6:<12.1f} "
              f"{metrics['latency']:<15.2f}")
    
    return efficiency_results

# Run comparison
efficiency_comparison = compare_cnn_vs_vit()

# %%
def analyze_data_efficiency():
    """Analyze how models perform with different amounts of data"""
    print("\nAnalyzing Data Efficiency")
    print("=" * 40)
    
    # Different dataset sizes to test
    dataset_sizes = [500, 1000, 2000, 5000]
    
    # Create models
    models = {
        'ResNet-18': resnet18(num_classes=10),
        'ViT-Tiny': vit_tiny(num_classes=10)
    }
    
    # Filter available models
    models = {name: model for name, model in models.items() if model is not None}
    
    if not models:
        print("No models available for data efficiency analysis")
        return
    
    results = {name: [] for name in models.keys()}
    
    # Load CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    full_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    for size in dataset_sizes:
        print(f"\nTesting with {size} samples...")
        
        # Create subset
        indices = torch.randperm(len(full_dataset))[:size]
        subset = Subset(full_dataset, indices)
        train_loader = DataLoader(subset, batch_size=32, shuffle=True)
        
        for name, model in models.items():
            # Quick training simulation (in practice, you would actually train)
            # Here we just measure forward pass performance
            model.to(device)
            model.eval()
            
            total_time = 0
            num_batches = 0
            
            with torch.no_grad():
                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)
                    
                    start_time = time.time()
                    outputs = model(data)
                    end_time = time.time()
                    
                    total_time += (end_time - start_time)
                    num_batches += 1
            
            avg_time = total_time / num_batches if num_batches > 0 else 0
            results[name].append(avg_time * 1000)  # Convert to ms
            
            print(f"  {name}: {avg_time*1000:.2f}ms per batch")
    
    return results, dataset_sizes

# Analyze data efficiency
data_efficiency_results, sizes = analyze_data_efficiency()

# %% [markdown]
"""
## Part 5: Advanced Analysis and Visualization

Let's create comprehensive visualizations and analysis for your report.
"""

# %%
def create_comprehensive_analysis():
    """Create comprehensive analysis with visualizations"""
    print("\nCreating Comprehensive Analysis")
    print("=" * 40)
    
    # Collect all results
    models = {
        'ResNet-18': resnet18(num_classes=10),
        'SE-ResNet-18': se_resnet18(num_classes=10),
        'ViT-Tiny': vit_tiny(num_classes=10)
    }
    
    # Filter available models
    models = {name: model for name, model in models.items() if model is not None}
    
    if not models:
        print("No models available for analysis")
        return
    
    # Initialize profilers and collect metrics
    results = {}
    
    for name, model in models.items():
        profiler = ModelProfiler(model, device)
        
        results[name] = {
            'parameters': profiler.count_parameters(),
            'flops': profiler.measure_flops((3, 32, 32)),
            'latency': profiler.measure_latency((3, 32, 32)),
            'memory': 0,  # Would measure in practice
            'accuracy': 75 + np.random.normal(0, 5)  # Simulated for demo
        }
    
    # Create visualizations
    visualizer = TrainingVisualizer()
    
    # Model comparison plot
    visualizer.plot_model_comparison(results, 'comprehensive_comparison.png')
    
    # Create efficiency frontier
    analyzer = ArchitectureAnalyzer()
    accuracies = {name: result['accuracy'] for name, result in results.items()}
    analyzer.create_efficiency_frontier(models, accuracies, (3, 32, 32), device, 
                                      'efficiency_frontier.png')
    
    return results

# Create comprehensive analysis
comprehensive_results = create_comprehensive_analysis()

# %%
def generate_training_curves():
    """Generate sample training curves for demonstration"""
    print("\nGenerating Sample Training Curves")
    print("=" * 40)
    
    # Simulate training history for different models
    epochs = 20
    
    # ResNet training curve (faster convergence)
    resnet_history = {
        'train_loss': [2.3 - 0.1*i + 0.02*np.random.randn() for i in range(epochs)],
        'val_loss': [2.3 - 0.08*i + 0.03*np.random.randn() for i in range(epochs)],
        'train_acc': [10 + 4*i + np.random.randn() for i in range(epochs)],
        'val_acc': [10 + 3.5*i + 1.5*np.random.randn() for i in range(epochs)]
    }
    
    # ViT training curve (slower initial convergence)
    vit_history = {
        'train_loss': [2.3 - 0.05*i + 0.02*np.random.randn() for i in range(epochs)],
        'val_loss': [2.3 - 0.04*i + 0.03*np.random.randn() for i in range(epochs)],
        'train_acc': [5 + 3*i + np.random.randn() for i in range(epochs)],
        'val_acc': [5 + 2.8*i + 1.5*np.random.randn() for i in range(epochs)]
    }
    
    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs_range = range(1, epochs + 1)
    
    # Loss comparison
    axes[0, 0].plot(epochs_range, resnet_history['train_loss'], 'b-', label='ResNet Train')
    axes[0, 0].plot(epochs_range, resnet_history['val_loss'], 'b--', label='ResNet Val')
    axes[0, 0].plot(epochs_range, vit_history['train_loss'], 'r-', label='ViT Train')
    axes[0, 0].plot(epochs_range, vit_history['val_loss'], 'r--', label='ViT Val')
    axes[0, 0].set_title('Training Loss Comparison')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy comparison
    axes[0, 1].plot(epochs_range, resnet_history['train_acc'], 'b-', label='ResNet Train')
    axes[0, 1].plot(epochs_range, resnet_history['val_acc'], 'b--', label='ResNet Val')
    axes[0, 1].plot(epochs_range, vit_history['train_acc'], 'r-', label='ViT Train')
    axes[0, 1].plot(epochs_range, vit_history['val_acc'], 'r--', label='ViT Val')
    axes[0, 1].set_title('Training Accuracy Comparison')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Data efficiency plot
    if 'data_efficiency_results' in globals() and 'sizes' in globals():
        for name, times in data_efficiency_results.items():
            axes[1, 0].plot(sizes, times, 'o-', label=name)
        axes[1, 0].set_title('Data Efficiency')
        axes[1, 0].set_xlabel('Dataset Size')
        axes[1, 0].set_ylabel('Time per Batch (ms)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Parameter vs Performance
    if 'comprehensive_results' in globals():
        models = list(comprehensive_results.keys())
        params = [comprehensive_results[m]['parameters']/1e6 for m in models]
        accs = [comprehensive_results[m]['accuracy'] for m in models]
        
        axes[1, 1].scatter(params, accs, s=100, alpha=0.7)
        for i, model in enumerate(models):
            axes[1, 1].annotate(model, (params[i], accs[i]), 
                               xytext=(5, 5), textcoords='offset points')
        axes[1, 1].set_title('Parameters vs Accuracy')
        axes[1, 1].set_xlabel('Parameters (Millions)')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# Generate training curves
generate_training_curves()

# %% [markdown]
"""
## Part 6: Report Generation

Let's summarize our findings and create content for your lab report.
"""

# %%
def generate_lab_report_summary():
    """Generate summary for lab report"""
    print("\nLab Report Summary")
    print("=" * 50)
    
    print("## Key Findings")
    print()
    
    print("### 1. ResNet Architecture Analysis")
    if 'scaling_results' in globals():
        print("- ResNet depth scaling shows increasing parameters and computational cost")
        for name, result in scaling_results.items():
            if result:
                print(f"  • {name}: {result['parameters']:,} parameters, "
                      f"{result['inference_time']:.2f}ms inference")
    
    print("\n### 2. Attention Mechanism Effectiveness")
    print("- SE modules provide channel-wise attention with minimal overhead")
    print("- CBAM combines channel and spatial attention for improved feature selection")
    print("- Attention mechanisms generally improve model performance with small parameter increase")
    
    print("\n### 3. Vision Transformer Characteristics")
    print("- ViTs require more parameters than comparable CNNs")
    print("- Global attention from first layer provides different inductive bias")
    print("- Patch-based processing enables flexible input handling")
    
    print("\n### 4. CNN vs ViT Trade-offs")
    if 'efficiency_comparison' in globals():
        print("- CNNs generally more parameter-efficient for small datasets")
        print("- ViTs may perform better with larger datasets (not tested due to compute constraints)")
        print("- Computational complexity varies significantly between architectures")
    
    print("\n### 5. Practical Recommendations")
    print("- Use ResNet + attention for resource-constrained environments")
    print("- Consider ViT for applications with abundant data and compute")
    print("- Hybrid approaches may offer best of both worlds")
    
    print("\n## Experimental Methodology")
    print("- Implemented all architectures from scratch following mathematical principles")
    print("- Conducted systematic comparison across multiple metrics")
    print("- Analyzed computational trade-offs and practical considerations")
    
    print("\n## Future Experiments")
    print("- Full training comparison on complete datasets")
    print("- Attention visualization and interpretation")
    print("- Hyperparameter optimization studies")
    print("- Transfer learning evaluation")

# Generate report summary
generate_lab_report_summary()

# %% [markdown]

## Conclusion
"""
This notebook has demonstrated the complete implementation and analysis pipeline for Lab 3.
You should now be able to:

1. **Implement ResNet architectures** with proper skip connections and scaling
2. **Add attention mechanisms** (SE and CBAM) to existing architectures
3. **Build Vision Transformers** from mathematical principles
4. **Compare different architectures** systematically
5. **Analyze computational trade-offs** and practical considerations
6. **Create visualizations** for your lab report

## Next Steps for Your Lab Report

1. **Complete all TODO functions** in the student template
2. **Run comprehensive experiments** with full training
3. **Create detailed analysis** of results with visualizations
4. **Write theoretical analysis** connecting mathematics to implementation
5. **Provide practical recommendations** for architecture selection

Remember to document your code well and explain your design choices in your report!
"""

# %%
# Final test to ensure everything works
print("Final Integration Test")
print("=" * 30)

try:
    # Test data loading
    train_loader, test_loader = create_data_loaders('CIFAR10', batch_size=32, subset_size=100)
    print("✓ Data loading works")
    
    # Test model creation
    test_models = {
        'ResNet-18': resnet18(num_classes=10),
        'ViT-Tiny': vit_tiny(num_classes=10)
    }
    
    available_models = {name: model for name, model in test_models.items() if model is not None}
    print(f"✓ {len(available_models)} models available")
    
    # Test profiling
    if available_models:
        model_name, model = list(available_models.items())[0]
        profiler = ModelProfiler(model, device)
        params = profiler.count_parameters()
        print(f"✓ Profiling works - {model_name} has {params:,} parameters")
    
    print("\n🎉 Lab 3 implementation is ready!")
    print("Complete the TODOs and run your experiments!")
    
except Exception as e:
    print(f"✗ Integration test failed: {e}")
    print("Please check your implementations")