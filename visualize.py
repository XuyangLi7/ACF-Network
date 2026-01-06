#!/usr/bin/env python3
"""
ACF Network - Comprehensive Visualization Script

Generates all visualizations for interpretability analysis:
1. Prediction results visualization
2. Grad-CAM class-specific attention
3. t-SNE feature clustering
4. Gradient attribution multi-modal analysis
5. Module effectiveness analysis

Usage:
    python visualize.py --dataset vaihingen --checkpoint checkpoints/acf_vaihingen_best.pth
"""

import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from acf import create_acf_model
from acf.visualization import (
    GradCAM,
    TSNEVisualizer,
    GradientAttributionVisualizer,
    PredictionVisualizer,
    ModuleEffectivenessVisualizer
)


def parse_args():
    parser = argparse.ArgumentParser(description='ACF Network Visualization')
    
    # Basic arguments
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['vaihingen', 'augsburg', 'muufl', 'trento'],
                        help='Dataset name')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory of datasets')
    
    # Visualization options
    parser.add_argument('--vis_types', type=str, nargs='+',
                        default=['all'],
                        choices=['all', 'prediction', 'gradcam', 'tsne', 'attribution', 'module'],
                        help='Types of visualizations to generate')
    parser.add_argument('--save_dir', type=str, default='./visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to visualize')
    
    # Grad-CAM options
    parser.add_argument('--gradcam_layers', type=str, nargs='+',
                        default=['scaf', 'multi_granularity_fusion', 'modality_balancing'],
                        help='Layers to visualize with Grad-CAM')
    
    # t-SNE options
    parser.add_argument('--tsne_perplexity', type=float, default=30.0,
                        help='t-SNE perplexity parameter')
    parser.add_argument('--tsne_samples', type=int, default=1000,
                        help='Number of samples for t-SNE')
    
    # Hardware
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    
    return parser.parse_args()


def visualize_predictions(
    model,
    dataloader,
    class_names,
    save_dir,
    num_samples=10,
    device=None
):
    """Generate prediction visualizations"""
    print("\n" + "="*80)
    print("Generating Prediction Visualizations")
    print("="*80)
    
    save_dir = Path(save_dir) / 'predictions'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get device from model if not specified
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    visualizer = PredictionVisualizer()
    
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(tqdm(dataloader, desc="Visualizing predictions")):
            if idx >= num_samples:
                break
            
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)
            
            # Convert to numpy
            pred_np = predictions[0].cpu().numpy()
            gt_np = labels[0].cpu().numpy()
            
            # Prepare input images for visualization
            images = {}
            for modality_name, tensor in inputs.items():
                img = tensor[0].cpu().numpy()
                if img.shape[0] == 3:  # RGB
                    img = np.transpose(img, (1, 2, 0))
                elif img.shape[0] == 1:  # Single channel
                    img = img[0]
                else:  # Multi-channel (e.g., HSI)
                    # Use first 3 channels as RGB
                    img = np.transpose(img[:3], (1, 2, 0))
                
                # Normalize to [0, 1]
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                images[modality_name] = img
            
            # Visualize
            save_path = save_dir / f'prediction_{idx:03d}.png'
            visualizer.visualize_predictions(
                images, pred_np, gt_np, class_names, save_path=str(save_path)
            )
    
    print(f"✓ Saved {num_samples} prediction visualizations to {save_dir}")


def visualize_gradcam(
    model,
    dataloader,
    class_names,
    target_layers,
    save_dir,
    num_samples=10,
    device=None
):
    """Generate Grad-CAM visualizations"""
    print("\n" + "="*80)
    print("Generating Grad-CAM Visualizations")
    print("="*80)
    
    save_dir = Path(save_dir) / 'gradcam'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get device from model if not specified
    if device is None:
        device = next(model.parameters()).device
    
    gradcam = GradCAM(model, target_layers)
    
    for idx, (inputs, labels) in enumerate(tqdm(dataloader, desc="Generating Grad-CAM")):
        if idx >= num_samples:
            break
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate CAM for each class
        for class_idx, class_name in enumerate(class_names):
            for layer_name in target_layers:
                try:
                    cam = gradcam.generate_cam(inputs, class_idx, layer_name)
                    
                    # Save heatmap
                    plt.figure(figsize=(8, 6))
                    plt.imshow(cam, cmap='jet')
                    plt.colorbar()
                    plt.title(f'Grad-CAM: {class_name} - {layer_name}')
                    plt.axis('off')
                    
                    save_path = save_dir / f'gradcam_sample{idx:03d}_class{class_idx}_{layer_name}.png'
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    print(f"Warning: Failed to generate Grad-CAM for {layer_name}: {e}")
    
    gradcam.remove_hooks()
    print(f"✓ Saved Grad-CAM visualizations to {save_dir}")


def visualize_tsne(
    model,
    dataloader,
    class_names,
    save_dir,
    perplexity=30.0,
    max_samples=1000
):
    """Generate t-SNE visualization"""
    print("\n" + "="*80)
    print("Generating t-SNE Visualization")
    print("="*80)
    
    save_dir = Path(save_dir) / 'tsne'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    tsne_viz = TSNEVisualizer(perplexity=perplexity)
    
    # Extract features
    print("Extracting features...")
    features, labels = tsne_viz.extract_features(model, dataloader, layer_name='scaf')
    
    # Subsample if too many
    if len(features) > max_samples:
        indices = np.random.choice(len(features), max_samples, replace=False)
        features = features[indices]
        labels = labels[indices]
    
    # Visualize
    save_path = save_dir / 'tsne_clustering.png'
    tsne_viz.visualize(features, labels, class_names, save_path=str(save_path))
    
    print(f"✓ Saved t-SNE visualization to {save_dir}")


def visualize_gradient_attribution(
    model,
    dataloader,
    class_names,
    save_dir,
    num_samples=10,
    device=None
):
    """Generate gradient attribution visualizations"""
    print("\n" + "="*80)
    print("Generating Gradient Attribution Visualizations")
    print("="*80)
    
    save_dir = Path(save_dir) / 'attribution'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get device from model if not specified
    if device is None:
        device = next(model.parameters()).device
    
    grad_attr_viz = GradientAttributionVisualizer(model)
    
    for idx, (inputs, labels) in enumerate(tqdm(dataloader, desc="Computing attributions")):
        if idx >= num_samples:
            break
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Compute attribution for each class
        for class_idx, class_name in enumerate(class_names):
            attributions = grad_attr_viz.compute_attribution(inputs, class_idx)
            
            # Visualize attributions for each modality
            n_modalities = len(attributions)
            fig, axes = plt.subplots(1, n_modalities + 1, figsize=(4 * (n_modalities + 1), 4))
            
            for mod_idx, (modality_name, attribution) in enumerate(attributions.items()):
                # Attribution heatmap
                axes[mod_idx].imshow(attribution, cmap='hot')
                axes[mod_idx].set_title(f'{modality_name.upper()} Attribution')
                axes[mod_idx].axis('off')
                
                # Top-k pixels
                top_k_mask = grad_attr_viz.visualize_top_k_pixels(attribution, k=100)
                axes[mod_idx + 1].imshow(top_k_mask, cmap='gray')
                axes[mod_idx + 1].set_title(f'{modality_name.upper()} Top-5%')
                axes[mod_idx + 1].axis('off')
            
            plt.suptitle(f'Gradient Attribution - {class_name}', fontsize=14)
            plt.tight_layout()
            
            save_path = save_dir / f'attribution_sample{idx:03d}_class{class_idx}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
    
    print(f"✓ Saved gradient attribution visualizations to {save_dir}")


def visualize_module_effectiveness(
    save_dir
):
    """Generate module effectiveness visualization"""
    print("\n" + "="*80)
    print("Generating Module Effectiveness Visualization")
    print("="*80)
    
    save_dir = Path(save_dir) / 'module_effectiveness'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Example data (replace with actual ablation results)
    module_names = [
        'Full ACF',
        'w/o AMB',
        'w/o MGCF',
        'w/o SCAF',
        'w/o CMA',
        'Baseline'
    ]
    
    accuracies = [93.11, 91.47, 92.23, 92.68, 90.89, 88.23]
    miou_scores = [84.29, 81.85, 82.91, 83.54, 80.72, 76.45]
    
    save_path = save_dir / 'module_effectiveness.png'
    ModuleEffectivenessVisualizer.visualize_module_contributions(
        module_names, accuracies, miou_scores, save_path=str(save_path)
    )
    
    print(f"✓ Saved module effectiveness visualization to {save_dir}")


def main():
    args = parse_args()
    
    print("="*80)
    print("ACF Network - Comprehensive Visualization")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Save directory: {args.save_dir}")
    print(f"Visualization types: {args.vis_types}")
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print("\nLoading model...")
    model = create_acf_model(dataset=args.dataset, num_classes=6)  # Adjust num_classes
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # 兼容不同的checkpoint格式
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            # 如果checkpoint本身就是字典但没有这些键，尝试直接使用
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # 处理DataParallel前缀
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    print("✓ Model loaded")
    
    # Load data
    print("\nLoading data...")
    from acf.dataset import create_universal_dataloader
    
    # Create dataloader
    dataloader = create_universal_dataloader(
        data_path=os.path.join(args.data_root, args.dataset.capitalize()),
        dataset_name=args.dataset,
        split='test',  # Use test split for visualization
        batch_size=1,
        shuffle=False,
        num_workers=2,
        window_size=(256, 256),
        stride=128
    )
    print(f"✓ Dataloader created: {len(dataloader)} batches")
    
    # Class names (adjust based on dataset)
    class_names = [
        'Impervious surfaces',
        'Building',
        'Low vegetation',
        'Tree',
        'Car',
        'Clutter'
    ]
    
    # Generate visualizations
    vis_types = args.vis_types
    if 'all' in vis_types:
        vis_types = ['prediction', 'gradcam', 'tsne', 'attribution', 'module']
    
    if 'prediction' in vis_types:
        visualize_predictions(
            model, dataloader, class_names, args.save_dir, args.num_samples, device
        )
    
    if 'gradcam' in vis_types:
        visualize_gradcam(
            model, dataloader, class_names, args.gradcam_layers, args.save_dir, args.num_samples, device
        )
    
    if 'tsne' in vis_types:
        visualize_tsne(
            model, dataloader, class_names, args.save_dir, args.tsne_perplexity, args.tsne_samples
        )
    
    if 'attribution' in vis_types:
        visualize_gradient_attribution(
            model, dataloader, class_names, args.save_dir, args.num_samples, device
        )
    
    if 'module' in vis_types:
        visualize_module_effectiveness(args.save_dir)
    
    print("\n" + "="*80)
    print("✓ All visualizations completed!")
    print(f"✓ Results saved to: {args.save_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
