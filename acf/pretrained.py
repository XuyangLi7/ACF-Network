"""
Pretrained Model Loading for ACF Network

Supports loading pretrained weights from:
- ResNet50 + ViT-B/16 (R50-ViT-B_16)
- ViT-B/16
- ViT-B/32
- ViT-L/16
- ViT-L/32

These pretrained weights can significantly improve performance, especially on small datasets.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
import os
import urllib.request
from tqdm import tqdm


# Pretrained model URLs (from Google's official ViT repository)
PRETRAINED_MODELS = {
    'R50-ViT-B_16': {
        'url': 'https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/R50+ViT-B_16.npz',
        'embed_dim': 768,
        'num_heads': 12,
        'description': 'ResNet50 + ViT-Base, patch size 16x16'
    },
    'ViT-B_16': {
        'url': 'https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-B_16.npz',
        'embed_dim': 768,
        'num_heads': 12,
        'description': 'ViT-Base, patch size 16x16'
    },
    'ViT-B_32': {
        'url': 'https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-B_32.npz',
        'embed_dim': 768,
        'num_heads': 12,
        'description': 'ViT-Base, patch size 32x32'
    },
    'ViT-L_16': {
        'url': 'https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-L_16.npz',
        'embed_dim': 1024,
        'num_heads': 16,
        'description': 'ViT-Large, patch size 16x16'
    },
    'ViT-L_32': {
        'url': 'https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-L_32.npz',
        'embed_dim': 1024,
        'num_heads': 16,
        'description': 'ViT-Large, patch size 32x32'
    },
    'ViT-H_14': {
        'url': 'https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-H_14.npz',
        'embed_dim': 1280,
        'num_heads': 16,
        'description': 'ViT-Huge, patch size 14x14'
    }
}


class PretrainedWeightLoader:
    """
    Utility class for loading pretrained weights into ACF network
    """
    
    def __init__(self, cache_dir: str = './pretrained_weights'):
        """
        Args:
            cache_dir: Directory to cache downloaded weights
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def download_weights(self, model_name: str) -> str:
        """
        Download pretrained weights if not already cached
        
        Args:
            model_name: Name of pretrained model (e.g., 'ViT-B_16')
        
        Returns:
            Path to downloaded weights file
        """
        if model_name not in PRETRAINED_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(PRETRAINED_MODELS.keys())}")
        
        model_info = PRETRAINED_MODELS[model_name]
        url = model_info['url']
        filename = os.path.join(self.cache_dir, f"{model_name}.npz")
        
        # Check if already downloaded
        if os.path.exists(filename):
            print(f"✓ Using cached weights: {filename}")
            return filename
        
        # Download with progress bar
        print(f"Downloading {model_name} from {url}...")
        
        def progress_hook(count, block_size, total_size):
            if total_size > 0:
                percent = count * block_size * 100 / total_size
                print(f"\rDownloading: {percent:.1f}%", end='')
        
        try:
            urllib.request.urlretrieve(url, filename, reporthook=progress_hook)
            print(f"\n✓ Downloaded to {filename}")
            return filename
        except Exception as e:
            print(f"\n✗ Download failed: {e}")
            raise
    
    def load_pretrained_weights(
        self,
        model: nn.Module,
        model_name: str,
        strict: bool = False,
        verbose: bool = True
    ) -> Dict[str, int]:
        """
        Load pretrained weights into ACF model
        
        Args:
            model: ACF network model
            model_name: Name of pretrained model
            strict: Whether to strictly match all keys
            verbose: Print loading information
        
        Returns:
            Dictionary with loading statistics
        """
        import numpy as np
        
        # Download weights
        weights_path = self.download_weights(model_name)
        
        # Load numpy weights
        weights = np.load(weights_path)
        
        if verbose:
            print(f"\nLoading pretrained weights from {model_name}...")
            print(f"Available keys: {len(weights.files)}")
        
        # Convert numpy weights to PyTorch state dict
        state_dict = self._convert_weights(weights, model_name)
        
        # Load into model
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
        
        stats = {
            'loaded': len(state_dict),
            'missing': len(missing_keys),
            'unexpected': len(unexpected_keys)
        }
        
        if verbose:
            print(f"\n✓ Loaded {stats['loaded']} parameters")
            if stats['missing'] > 0:
                print(f"⚠ Missing {stats['missing']} keys (will be randomly initialized)")
            if stats['unexpected'] > 0:
                print(f"⚠ Unexpected {stats['unexpected']} keys (ignored)")
        
        return stats
    
    def _convert_weights(self, weights: Dict, model_name: str) -> Dict[str, torch.Tensor]:
        """
        Convert numpy weights to PyTorch state dict
        
        Args:
            weights: Numpy weights dictionary
            model_name: Name of pretrained model
        
        Returns:
            PyTorch state dict
        """
        state_dict = {}
        
        # Convert each weight
        for key in weights.files:
            value = weights[key]
            
            # Convert to torch tensor
            if value.ndim == 0:
                # Scalar
                tensor = torch.tensor(value.item())
            else:
                tensor = torch.from_numpy(value)
            
            # Map key names from ViT to ACF
            mapped_key = self._map_key_name(key, model_name)
            if mapped_key:
                state_dict[mapped_key] = tensor
        
        return state_dict
    
    def _map_key_name(self, vit_key: str, model_name: str) -> Optional[str]:
        """
        Map ViT weight key names to ACF network key names
        
        Args:
            vit_key: Original key from ViT weights
            model_name: Name of pretrained model
        
        Returns:
            Mapped key name for ACF network, or None if not applicable
        """
        # Key mapping rules
        # ViT uses different naming conventions than ACF
        
        # Patch embedding
        if 'embedding' in vit_key or 'patch_embed' in vit_key:
            # Map to HierarchicalTokenizer
            if 'kernel' in vit_key or 'weight' in vit_key:
                return vit_key.replace('embedding', 'tokenizers.rgb.patch_embed')
            if 'bias' in vit_key:
                return vit_key.replace('embedding', 'tokenizers.rgb.patch_embed')
        
        # Position embedding
        if 'pos_embed' in vit_key or 'position_embedding' in vit_key:
            return vit_key.replace('position_embedding', 'tokenizers.rgb.pos_embed')
        
        # Transformer blocks -> Cross-Modal Attention
        if 'transformer' in vit_key or 'encoder' in vit_key:
            # Map attention layers
            if 'attention' in vit_key:
                if 'query' in vit_key or 'q_proj' in vit_key:
                    return vit_key.replace('query', 'cross_modal_attention.q_proj')
                if 'key' in vit_key or 'k_proj' in vit_key:
                    return vit_key.replace('key', 'cross_modal_attention.k_proj')
                if 'value' in vit_key or 'v_proj' in vit_key:
                    return vit_key.replace('value', 'cross_modal_attention.v_proj')
                if 'out' in vit_key or 'out_proj' in vit_key:
                    return vit_key.replace('out', 'cross_modal_attention.out_proj')
            
            # Map MLP/FFN layers
            if 'mlp' in vit_key or 'ffn' in vit_key:
                # These can map to MGCF or SCAF modules
                return None  # Skip for now, ACF has custom modules
        
        # Layer normalization
        if 'norm' in vit_key or 'ln' in vit_key:
            return vit_key.replace('norm', 'tokenizers.rgb.norm')
        
        # Skip head/classifier weights (ACF has different output)
        if 'head' in vit_key or 'classifier' in vit_key:
            return None
        
        return None


def load_pretrained_acf(
    model: nn.Module,
    pretrained: str = 'ViT-B_16',
    cache_dir: str = './pretrained_weights',
    strict: bool = False,
    verbose: bool = True
) -> nn.Module:
    """
    Convenience function to load pretrained weights into ACF model
    
    Args:
        model: ACF network model
        pretrained: Name of pretrained model or path to weights file
        cache_dir: Directory to cache downloaded weights
        strict: Whether to strictly match all keys
        verbose: Print loading information
    
    Returns:
        Model with loaded pretrained weights
    
    Example:
        >>> from acf import create_acf_model
        >>> from acf.pretrained import load_pretrained_acf
        >>> 
        >>> model = create_acf_model('vaihingen', num_classes=6)
        >>> model = load_pretrained_acf(model, pretrained='ViT-B_16')
    """
    loader = PretrainedWeightLoader(cache_dir=cache_dir)
    
    if pretrained in PRETRAINED_MODELS:
        # Load from official pretrained models
        loader.load_pretrained_weights(model, pretrained, strict=strict, verbose=verbose)
    elif os.path.exists(pretrained):
        # Load from local file
        if verbose:
            print(f"Loading weights from {pretrained}...")
        state_dict = torch.load(pretrained, map_location='cpu')
        model.load_state_dict(state_dict, strict=strict)
        if verbose:
            print("✓ Weights loaded successfully")
    else:
        raise ValueError(f"Unknown pretrained model or file not found: {pretrained}")
    
    return model


def list_pretrained_models():
    """
    List all available pretrained models
    """
    print("\n" + "="*80)
    print("Available Pretrained Models for ACF Network")
    print("="*80)
    
    for name, info in PRETRAINED_MODELS.items():
        print(f"\n{name}")
        print(f"  Description: {info['description']}")
        print(f"  Embed Dim: {info['embed_dim']}")
        print(f"  Num Heads: {info['num_heads']}")
        print(f"  URL: {info['url']}")
    
    print("\n" + "="*80)
    print("\nUsage:")
    print("  from acf import create_acf_model")
    print("  from acf.pretrained import load_pretrained_acf")
    print("  ")
    print("  model = create_acf_model('vaihingen', num_classes=6)")
    print("  model = load_pretrained_acf(model, pretrained='ViT-B_16')")
    print("="*80 + "\n")


def get_pretrained_config(model_name: str) -> Dict:
    """
    Get recommended configuration for pretrained model
    
    Args:
        model_name: Name of pretrained model
    
    Returns:
        Dictionary with recommended hyperparameters
    """
    if model_name not in PRETRAINED_MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    
    info = PRETRAINED_MODELS[model_name]
    
    # Extract patch size from model name
    patch_size = int(model_name.split('_')[-1])
    
    return {
        'embed_dim': info['embed_dim'],
        'num_heads': info['num_heads'],
        'patch_size': patch_size,
        'learning_rate': 1e-4,  # Lower LR for finetuning
        'weight_decay': 0.01,
        'warmup_epochs': 5,
        'freeze_backbone': False,  # Set to True to freeze pretrained layers
    }


if __name__ == '__main__':
    # List available models
    list_pretrained_models()
    
    # Example usage
    print("\nExample: Loading ViT-B/16 pretrained weights")
    print("-" * 80)
    
    from acf import create_acf_model
    
    # Create model
    model = create_acf_model('vaihingen', num_classes=6, embed_dim=768, num_heads=12)
    
    # Load pretrained weights
    model = load_pretrained_acf(model, pretrained='ViT-B_16', verbose=True)
    
    print("\n✓ Model ready for training with pretrained weights!")
