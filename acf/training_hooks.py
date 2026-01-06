"""
Training-time Interpretability Hooks System

This module provides hooks to capture intermediate features, gradients, and activations
during training to enable real-time interpretability analysis and module effectiveness evaluation.

Key Features:
1. Feature extraction hooks for each module
2. Gradient flow monitoring
3. Activation statistics tracking
4. Module contribution analysis
5. Real-time visualization generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import os
import json


class ModuleHook:
    """Base class for module hooks"""
    
    def __init__(self, module: nn.Module, name: str):
        self.module = module
        self.name = name
        self.activations = []
        self.gradients = []
        self.forward_hook = None
        self.backward_hook = None
        
    def register(self):
        """Register forward and backward hooks"""
        self.forward_hook = self.module.register_forward_hook(self._forward_hook)
        self.backward_hook = self.module.register_full_backward_hook(self._backward_hook)
        
    def remove(self):
        """Remove hooks"""
        if self.forward_hook is not None:
            self.forward_hook.remove()
        if self.backward_hook is not None:
            self.backward_hook.remove()
            
    def _forward_hook(self, module, input, output):
        """Capture activations during forward pass"""
        if isinstance(output, torch.Tensor):
            self.activations.append(output.detach().clone())
        elif isinstance(output, tuple):
            self.activations.append(tuple(o.detach().clone() if isinstance(o, torch.Tensor) else o for o in output))
            
    def _backward_hook(self, module, grad_input, grad_output):
        """Capture gradients during backward pass"""
        if grad_output[0] is not None:
            self.gradients.append(grad_output[0].detach().clone())
            
    def clear(self):
        """Clear stored activations and gradients"""
        self.activations = []
        self.gradients = []
        
    def get_latest_activation(self):
        """Get the most recent activation"""
        return self.activations[-1] if self.activations else None
        
    def get_latest_gradient(self):
        """Get the most recent gradient"""
        return self.gradients[-1] if self.gradients else None


class TrainingInterpreter:
    """
    Training-time interpretability system
    
    Monitors module effectiveness, gradient flow, and feature quality during training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_modules: List[str],
        save_dir: str,
        log_interval: int = 100,
        vis_interval: int = 1000
    ):
        """
        Args:
            model: The model to monitor
            target_modules: List of module names to hook
            save_dir: Directory to save analysis results
            log_interval: Steps between logging statistics
            vis_interval: Steps between generating visualizations
        """
        self.model = model
        self.target_modules = target_modules
        self.save_dir = save_dir
        self.log_interval = log_interval
        self.vis_interval = vis_interval
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Create hooks for each target module
        self.hooks = {}
        self._register_hooks()
        
        # Statistics tracking
        self.stats = defaultdict(lambda: defaultdict(list))
        self.global_step = 0
        
    def _register_hooks(self):
        """Register hooks on target modules"""
        for name, module in self.model.named_modules():
            if name in self.target_modules:
                hook = ModuleHook(module, name)
                hook.register()
                self.hooks[name] = hook
                print(f"✓ Registered hook on: {name}")
                
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks.values():
            hook.remove()
        self.hooks.clear()
        
    def step(self, loss: torch.Tensor, epoch: int, batch_idx: int):
        """
        Called after each training step
        
        Args:
            loss: Training loss
            epoch: Current epoch
            batch_idx: Current batch index
        """
        self.global_step += 1
        
        # Compute and log statistics
        if self.global_step % self.log_interval == 0:
            self._compute_statistics(loss, epoch, batch_idx)
            
        # Generate visualizations
        if self.global_step % self.vis_interval == 0:
            self._generate_visualizations(epoch, batch_idx)
            
        # Clear old activations/gradients to save memory
        for hook in self.hooks.values():
            if len(hook.activations) > 2:  # Keep only last 2
                hook.activations = hook.activations[-2:]
            if len(hook.gradients) > 2:
                hook.gradients = hook.gradients[-2:]
                
    def _compute_statistics(self, loss: torch.Tensor, epoch: int, batch_idx: int):
        """Compute and log module statistics"""
        stats_entry = {
            'epoch': epoch,
            'batch': batch_idx,
            'step': self.global_step,
            'loss': float(loss.item()),
            'modules': {}
        }
        
        for name, hook in self.hooks.items():
            activation = hook.get_latest_activation()
            gradient = hook.get_latest_gradient()
            
            module_stats = {}
            
            if activation is not None:
                if isinstance(activation, torch.Tensor):
                    # Activation statistics
                    module_stats['activation_mean'] = float(activation.mean().item())
                    module_stats['activation_std'] = float(activation.std().item())
                    module_stats['activation_max'] = float(activation.max().item())
                    module_stats['activation_min'] = float(activation.min().item())
                    
                    # Sparsity (percentage of near-zero activations)
                    sparsity = float((activation.abs() < 1e-6).float().mean().item())
                    module_stats['sparsity'] = sparsity
                    
            if gradient is not None:
                # Gradient statistics
                module_stats['gradient_mean'] = float(gradient.mean().item())
                module_stats['gradient_std'] = float(gradient.std().item())
                module_stats['gradient_norm'] = float(gradient.norm().item())
                
                # Gradient flow indicator (larger = better gradient flow)
                grad_flow = float(gradient.abs().mean().item())
                module_stats['gradient_flow'] = grad_flow
                
            stats_entry['modules'][name] = module_stats
            
        # Save statistics
        self.stats[epoch][batch_idx] = stats_entry
        
        # Log to console
        print(f"\n[Step {self.global_step}] Module Statistics:")
        for name, mstats in stats_entry['modules'].items():
            if 'gradient_flow' in mstats:
                print(f"  {name}: grad_flow={mstats['gradient_flow']:.6f}, "
                      f"act_mean={mstats.get('activation_mean', 0):.4f}")
                      
    def _generate_visualizations(self, epoch: int, batch_idx: int):
        """Generate interpretability visualizations"""
        vis_dir = os.path.join(self.save_dir, f'epoch_{epoch:03d}')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Save activation heatmaps
        for name, hook in self.hooks.items():
            activation = hook.get_latest_activation()
            if activation is not None and isinstance(activation, torch.Tensor):
                if len(activation.shape) == 4:  # (B, C, H, W)
                    # Average over batch and channels
                    heatmap = activation.mean(dim=(0, 1)).cpu().numpy()
                    
                    # Save as numpy array
                    save_path = os.path.join(vis_dir, f'{name}_step{self.global_step}.npy')
                    np.save(save_path, heatmap)
                    
    def save_statistics(self, filename: str = 'training_interpretability.json'):
        """Save all collected statistics"""
        save_path = os.path.join(self.save_dir, filename)
        
        # Convert defaultdict to regular dict for JSON serialization
        stats_dict = {
            str(epoch): {
                str(batch): entry
                for batch, entry in batches.items()
            }
            for epoch, batches in self.stats.items()
        }
        
        with open(save_path, 'w') as f:
            json.dump(stats_dict, f, indent=2)
            
        print(f"✓ Saved training interpretability statistics to: {save_path}")
        
    def analyze_module_effectiveness(self) -> Dict[str, float]:
        """
        Analyze module effectiveness based on gradient flow
        
        Returns:
            Dictionary mapping module names to effectiveness scores
        """
        effectiveness = {}
        
        for name in self.hooks.keys():
            # Collect all gradient flow values for this module
            grad_flows = []
            for epoch_stats in self.stats.values():
                for batch_stats in epoch_stats.values():
                    if name in batch_stats['modules']:
                        mstats = batch_stats['modules'][name]
                        if 'gradient_flow' in mstats:
                            grad_flows.append(mstats['gradient_flow'])
                            
            if grad_flows:
                # Average gradient flow as effectiveness metric
                effectiveness[name] = float(np.mean(grad_flows))
            else:
                effectiveness[name] = 0.0
                
        return effectiveness


class FeatureExtractor:
    """
    Extract features from specific layers during training
    
    Useful for t-SNE visualization and feature quality analysis.
    """
    
    def __init__(self, model: nn.Module, layer_name: str):
        self.model = model
        self.layer_name = layer_name
        self.features = []
        self.labels = []
        self.hook = None
        
    def register(self):
        """Register forward hook"""
        target_module = dict(self.model.named_modules())[self.layer_name]
        self.hook = target_module.register_forward_hook(self._hook_fn)
        
    def remove(self):
        """Remove hook"""
        if self.hook is not None:
            self.hook.remove()
            
    def _hook_fn(self, module, input, output):
        """Hook function to capture features"""
        if isinstance(output, torch.Tensor):
            # Global average pooling if spatial dimensions exist
            if len(output.shape) == 4:  # (B, C, H, W)
                feat = F.adaptive_avg_pool2d(output, 1).squeeze()
            elif len(output.shape) == 3:  # (B, N, D)
                feat = output.mean(dim=1)
            else:
                feat = output
                
            self.features.append(feat.detach().cpu())
            
    def add_labels(self, labels: torch.Tensor):
        """Add corresponding labels"""
        self.labels.append(labels.detach().cpu())
        
    def get_features_and_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get all collected features and labels"""
        if not self.features:
            return np.array([]), np.array([])
            
        features = torch.cat(self.features, dim=0).numpy()
        labels = torch.cat(self.labels, dim=0).numpy()
        
        return features, labels
        
    def clear(self):
        """Clear stored features and labels"""
        self.features = []
        self.labels = []


class GradientAttributionTracker:
    """
    Track gradient attribution for multi-modal inputs during training
    
    Helps understand which modality contributes more to the prediction.
    """
    
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.attributions = defaultdict(list)
        os.makedirs(save_dir, exist_ok=True)
        
    def compute_attribution(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        num_classes: int
    ):
        """
        Compute gradient attribution for each modality
        
        Args:
            model: The model
            inputs: Dictionary of input tensors
            targets: Ground truth labels
            num_classes: Number of classes
        """
        model.eval()
        
        # Enable gradients for inputs
        for key in inputs:
            inputs[key].requires_grad = True
            
        # Forward pass
        outputs = model(**inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
            
        # Compute loss for each class
        for class_idx in range(num_classes):
            # Create one-hot target for this class
            class_mask = (targets == class_idx).float()
            if class_mask.sum() == 0:
                continue
                
            # Backward pass
            model.zero_grad()
            class_score = (outputs[:, class_idx] * class_mask).sum()
            class_score.backward(retain_graph=True)
            
            # Compute attribution for each modality
            for modality_name, input_tensor in inputs.items():
                if input_tensor.grad is not None:
                    # Gradient magnitude as attribution
                    attribution = input_tensor.grad.abs().mean().item()
                    self.attributions[f'{modality_name}_class{class_idx}'].append(attribution)
                    
        model.train()
        
    def save_attributions(self, filename: str = 'gradient_attributions.json'):
        """Save attribution statistics"""
        save_path = os.path.join(self.save_dir, filename)
        
        # Compute statistics
        stats = {}
        for key, values in self.attributions.items():
            stats[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
            
        with open(save_path, 'w') as f:
            json.dump(stats, f, indent=2)
            
        print(f"✓ Saved gradient attributions to: {save_path}")


if __name__ == '__main__':
    print("Training Interpretability Hooks System")
    print("=" * 80)
    print("\nFeatures:")
    print("1. Real-time module effectiveness monitoring")
    print("2. Gradient flow analysis")
    print("3. Feature extraction for t-SNE")
    print("4. Multi-modal gradient attribution")
    print("\nUsage: Import and integrate into training loop")
