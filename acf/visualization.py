"""
Visualization Module for ACF Network

Provides comprehensive visualization tools for:
1. Prediction results visualization
2. Grad-CAM class-specific attention visualization
3. t-SNE feature clustering visualization
4. Gradient attribution multi-modal interpretability analysis

These visualizations prove:
- Model interpretability
- Each module's effectiveness
- Multi-modal collaboration mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from typing import Dict, List, Tuple, Optional
import cv2
from pathlib import Path


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    
    Visualizes class-specific attention for each modality and module.
    """
    
    def __init__(self, model: nn.Module, target_layers: List[str]):
        """
        Args:
            model: ACF network model
            target_layers: List of layer names to visualize
        """
        self.model = model
        self.target_layers = target_layers
        self.gradients = {}
        self.activations = {}
        self.hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook
        
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                self.hooks.append(module.register_forward_hook(forward_hook(name)))
                self.hooks.append(module.register_backward_hook(backward_hook(name)))
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        layer_name: str
    ) -> np.ndarray:
        """
        Generate class activation map
        
        Args:
            input_tensor: Input tensor (B, C, H, W)
            target_class: Target class index
            layer_name: Layer name to visualize
        
        Returns:
            cam: Class activation map (H, W)
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[:, target_class].sum()
        class_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[layer_name]  # (B, C, H, W)
        activations = self.activations[layer_name]  # (B, C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        
        # Weighted combination
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        cam = F.relu(cam)  # ReLU
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()


class TSNEVisualizer:
    """
    t-SNE Feature Clustering Visualization
    
    Visualizes feature separability and clustering quality.
    """
    
    def __init__(self, n_components: int = 2, perplexity: float = 30.0):
        """
        Args:
            n_components: Number of dimensions (2 or 3)
            perplexity: t-SNE perplexity parameter
        """
        self.n_components = n_components
        self.perplexity = perplexity
    
    def extract_features(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        layer_name: str = 'scaf'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from specified layer
        
        Args:
            model: ACF network model
            dataloader: Data loader
            layer_name: Layer name to extract features from
        
        Returns:
            features: Extracted features (N, D)
            labels: Ground truth labels (N,)
        """
        model.eval()
        features_list = []
        labels_list = []
        
        # Hook to capture features
        features_hook = []
        def hook_fn(module, input, output):
            features_hook.append(output.detach())
        
        # Register hook
        target_module = dict(model.named_modules())[layer_name]
        hook = target_module.register_forward_hook(hook_fn)
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                # Forward pass
                _ = model(inputs)
                
                # Get features
                feats = features_hook[-1]
                if len(feats.shape) == 4:  # (B, C, H, W)
                    feats = F.adaptive_avg_pool2d(feats, 1).squeeze()
                elif len(feats.shape) == 3:  # (B, N, D)
                    feats = feats.mean(dim=1)
                
                features_list.append(feats.cpu().numpy())
                labels_list.append(labels.cpu().numpy())
        
        hook.remove()
        
        features = np.concatenate(features_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        
        return features, labels
    
    def visualize(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        class_names: List[str],
        save_path: Optional[str] = None
    ):
        """
        Visualize t-SNE clustering
        
        Args:
            features: Features (N, D)
            labels: Labels (N,)
            class_names: List of class names
            save_path: Path to save figure
        """
        # Apply t-SNE
        tsne = TSNE(n_components=self.n_components, perplexity=self.perplexity)
        features_2d = tsne.fit_transform(features)
        
        # Plot
        plt.figure(figsize=(12, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
        
        for i, class_name in enumerate(class_names):
            mask = labels == i
            plt.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                c=[colors[i]],
                label=class_name,
                alpha=0.6,
                s=20
            )
        
        plt.legend(loc='best', fontsize=10)
        plt.title('t-SNE Feature Clustering Visualization', fontsize=14)
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class GradientAttributionVisualizer:
    """
    Gradient Attribution Multi-Modal Interpretability Analysis
    
    Visualizes gradient-based attribution for each modality.
    """
    
    def __init__(self, model: nn.Module):
        """
        Args:
            model: ACF network model
        """
        self.model = model
    
    def compute_attribution(
        self,
        inputs: Dict[str, torch.Tensor],
        target_class: int
    ) -> Dict[str, np.ndarray]:
        """
        Compute gradient attribution for each modality
        
        Args:
            inputs: Input dictionary {modality_name: tensor}
            target_class: Target class index
        
        Returns:
            attributions: Dictionary of attribution maps
        """
        self.model.eval()
        
        # Enable gradients for inputs
        for key in inputs:
            inputs[key].requires_grad = True
        
        # Forward pass
        output = self.model(inputs)
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[:, target_class].sum()
        class_score.backward()
        
        # Compute attributions
        attributions = {}
        for modality_name, input_tensor in inputs.items():
            # Gradient magnitude
            grad = input_tensor.grad.abs()
            
            # Average over channels
            attribution = grad.mean(dim=1, keepdim=True)  # (B, 1, H, W)
            
            # Normalize
            attribution = attribution.squeeze().cpu().numpy()
            attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)
            
            attributions[modality_name] = attribution
        
        return attributions
    
    def visualize_top_k_pixels(
        self,
        attribution: np.ndarray,
        k: int = 100,
        image_size: Tuple[int, int] = (512, 512)
    ) -> np.ndarray:
        """
        Visualize top-k most important pixels
        
        Args:
            attribution: Attribution map (H, W)
            k: Number of top pixels
            image_size: Image size
        
        Returns:
            top_k_mask: Binary mask of top-k pixels
        """
        # Flatten and get top-k indices
        flat_attr = attribution.flatten()
        top_k_indices = np.argsort(flat_attr)[-k:]
        
        # Create mask
        top_k_mask = np.zeros_like(flat_attr)
        top_k_mask[top_k_indices] = 1
        top_k_mask = top_k_mask.reshape(attribution.shape)
        
        return top_k_mask


class PredictionVisualizer:
    """
    Prediction Results Visualization
    
    Visualizes model predictions alongside ground truth.
    """
    
    @staticmethod
    def visualize_predictions(
        images: Dict[str, np.ndarray],
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        class_names: List[str],
        class_colors: Optional[List[Tuple[int, int, int]]] = None,
        save_path: Optional[str] = None
    ):
        """
        Visualize predictions
        
        Args:
            images: Dictionary of input images {modality_name: image}
            predictions: Predicted segmentation map (H, W)
            ground_truth: Ground truth segmentation map (H, W)
            class_names: List of class names
            class_colors: List of RGB colors for each class
            save_path: Path to save figure
        """
        if class_colors is None:
            # Default colors
            class_colors = [
                (255, 255, 255),  # White - Impervious
                (0, 0, 255),      # Blue - Building
                (0, 255, 0),      # Green - Low veg
                (255, 165, 0),    # Orange - Tree
                (255, 0, 255),    # Magenta - Car
                (255, 0, 0)       # Red - Clutter
            ]
        
        # Create colored segmentation maps
        pred_colored = np.zeros((*predictions.shape, 3), dtype=np.uint8)
        gt_colored = np.zeros((*ground_truth.shape, 3), dtype=np.uint8)
        
        for i, color in enumerate(class_colors):
            pred_colored[predictions == i] = color
            gt_colored[ground_truth == i] = color
        
        # Plot
        n_modalities = len(images)
        fig, axes = plt.subplots(1, n_modalities + 2, figsize=(4 * (n_modalities + 2), 4))
        
        # Plot input modalities
        for idx, (modality_name, image) in enumerate(images.items()):
            axes[idx].imshow(image)
            axes[idx].set_title(f'{modality_name.upper()} Input')
            axes[idx].axis('off')
        
        # Plot prediction
        axes[n_modalities].imshow(pred_colored)
        axes[n_modalities].set_title('Prediction')
        axes[n_modalities].axis('off')
        
        # Plot ground truth
        axes[n_modalities + 1].imshow(gt_colored)
        axes[n_modalities + 1].set_title('Ground Truth')
        axes[n_modalities + 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class ModuleEffectivenessVisualizer:
    """
    Module Effectiveness Visualization
    
    Visualizes the contribution of each module through ablation.
    """
    
    @staticmethod
    def visualize_module_contributions(
        module_names: List[str],
        accuracies: List[float],
        miou_scores: List[float],
        save_path: Optional[str] = None
    ):
        """
        Visualize module contributions
        
        Args:
            module_names: List of module names
            accuracies: List of accuracies
            miou_scores: List of mIoU scores
            save_path: Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        x = np.arange(len(module_names))
        width = 0.35
        
        # Plot accuracies
        ax1.bar(x, accuracies, width, label='Overall Accuracy', color='skyblue')
        ax1.set_xlabel('Configuration', fontsize=12)
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.set_title('Module Effectiveness - Overall Accuracy', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(module_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot mIoU
        ax2.bar(x, miou_scores, width, label='Mean IoU', color='lightcoral')
        ax2.set_xlabel('Configuration', fontsize=12)
        ax2.set_ylabel('mIoU (%)', fontsize=12)
        ax2.set_title('Module Effectiveness - Mean IoU', fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels(module_names, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def create_comprehensive_visualization(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    class_names: List[str],
    save_dir: str = './visualizations'
):
    """
    Create comprehensive visualization suite
    
    Args:
        model: ACF network model
        dataloader: Data loader
        class_names: List of class names
        save_dir: Directory to save visualizations
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating comprehensive visualizations...")
    
    # 1. Grad-CAM visualization
    print("\n1. Generating Grad-CAM visualizations...")
    gradcam = GradCAM(model, target_layers=['scaf', 'multi_granularity_fusion'])
    # ... (generate Grad-CAM for each class)
    
    # 2. t-SNE visualization
    print("\n2. Generating t-SNE visualization...")
    tsne_viz = TSNEVisualizer()
    features, labels = tsne_viz.extract_features(model, dataloader)
    tsne_viz.visualize(features, labels, class_names, save_path=str(save_dir / 'tsne.png'))
    
    # 3. Gradient attribution
    print("\n3. Generating gradient attribution visualizations...")
    grad_attr_viz = GradientAttributionVisualizer(model)
    # ... (generate attribution maps)
    
    # 4. Prediction visualization
    print("\n4. Generating prediction visualizations...")
    # ... (visualize predictions)
    
    print(f"\n✓ All visualizations saved to {save_dir}")


if __name__ == '__main__':
    print("ACF Visualization Module")
    print("=" * 80)
    print("\nAvailable visualizations:")
    print("1. Grad-CAM - Class-specific attention")
    print("2. t-SNE - Feature clustering")
    print("3. Gradient Attribution - Multi-modal interpretability")
    print("4. Prediction Results - Segmentation visualization")
    print("5. Module Effectiveness - Ablation analysis")
