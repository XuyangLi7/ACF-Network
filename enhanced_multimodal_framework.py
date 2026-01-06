#!/usr/bin/env python3
"""
Enhanced Multimodal Framework - Wrapper for ACF Network with pretrained support
"""

import torch
import torch.nn as nn
from acf import ACFNetwork, load_pretrained_acf
import os


class EnhancedMultimodalFramework(nn.Module):
    """
    Enhanced wrapper for ACF Network with pretrained weight loading support
    """
    def __init__(
        self,
        rgb_channels: int = 3,
        dsm_channels: int = 1,
        num_classes: int = 6,
        embed_dim: int = 768,
        enable_remote_sensing_innovations: bool = True,
        pretrained: bool = False,
        pretrained_model: str = 'R50-ViT-B_16',
        pretrained_path: str = None,
        use_multi_scale_aggregator: bool = False,
        use_simple_mode: bool = True,
        dataset: str = 'vaihingen',
        **kwargs
    ):
        """
        Args:
            rgb_channels: Number of RGB channels (for compatibility, not used directly)
            dsm_channels: Number of DSM channels (for compatibility, not used directly)
            num_classes: Number of output classes
            embed_dim: Embedding dimension
            enable_remote_sensing_innovations: Enable remote sensing specific modules
            pretrained: Whether to load pretrained weights
            pretrained_model: Name of pretrained model (e.g., 'R50-ViT-B_16', 'ViT-B_16')
            pretrained_path: Path to local pretrained weights file (.npz or .pth)
            use_multi_scale_aggregator: Use multi-scale context aggregator (not used)
            use_simple_mode: Use simplified mode for better stability (not used)
            dataset: Dataset name ('vaihingen', 'augsburg', 'muufl', 'trento')
        """
        super().__init__()
        
        # Create ACF Network with correct parameters
        self.model = ACFNetwork(
            dataset=dataset,
            num_classes=num_classes,
            embed_dim=embed_dim,
            num_heads=embed_dim // 64,  # Standard ratio
            patch_size=4,
            **kwargs
        )
        
        # Load pretrained weights if requested
        if pretrained:
            self._load_pretrained_weights(
                pretrained_model=pretrained_model,
                pretrained_path=pretrained_path
            )
    
    def _load_pretrained_weights(self, pretrained_model: str, pretrained_path: str = None):
        """Load pretrained weights into the model"""
        try:
            if pretrained_path and os.path.exists(pretrained_path):
                # Load from local file
                print(f"Loading pretrained weights from: {pretrained_path}")
                
                if pretrained_path.endswith('.npz'):
                    # Load numpy weights (ViT format)
                    cache_dir = os.path.dirname(os.path.dirname(pretrained_path))
                    load_pretrained_acf(
                        self.model,
                        pretrained=pretrained_model,
                        cache_dir=cache_dir,
                        strict=False,
                        verbose=True
                    )
                elif pretrained_path.endswith('.pth') or pretrained_path.endswith('.pt'):
                    # Load PyTorch checkpoint
                    checkpoint = torch.load(pretrained_path, map_location='cpu')
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    else:
                        state_dict = checkpoint
                    
                    # Try to load with strict=False to allow partial loading
                    missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                    print(f"✓ Loaded pretrained weights")
                    if missing:
                        print(f"  Missing keys: {len(missing)}")
                    if unexpected:
                        print(f"  Unexpected keys: {len(unexpected)}")
                else:
                    print(f"⚠ Unsupported pretrained weight format: {pretrained_path}")
            else:
                # Load from pretrained model name - auto-detect environment
                print(f"Loading pretrained model: {pretrained_model}")
                
                # Try HPC path first, then local path
                hpc_cache_dir = '/scratch/lixuyang/ACF-Network-Full/pretrained_weights'
                local_cache_dir = './pretrained_weights'
                
                if os.path.exists(hpc_cache_dir):
                    cache_dir = hpc_cache_dir
                else:
                    cache_dir = local_cache_dir
                
                print(f"Using cache directory: {cache_dir}")
                load_pretrained_acf(
                    self.model,
                    pretrained=pretrained_model,
                    cache_dir=cache_dir,
                    strict=False,
                    verbose=True
                )
        except Exception as e:
            print(f"⚠ Failed to load pretrained weights: {e}")
            print("  Continuing with random initialization...")
    
    def forward(self, rgb: torch.Tensor, dsm: torch.Tensor):
        """
        Forward pass
        
        Args:
            rgb: RGB input (B, C, H, W)
            dsm: DSM input (B, C, H, W)
        
        Returns:
            Output logits (B, num_classes, H, W) or tuple with auxiliary outputs
        """
        # ACFNetwork expects a dictionary of inputs
        inputs = {
            'rgb': rgb,
            'dsm': dsm
        }
        return self.model(inputs)
    
    def get_model(self):
        """Get the underlying ACF model"""
        return self.model


if __name__ == '__main__':
    # Test model creation
    print("Testing EnhancedMultimodalFramework...")
    
    # Without pretrained
    model1 = EnhancedMultimodalFramework(
        rgb_channels=3,
        dsm_channels=1,
        num_classes=6,
        embed_dim=768,
        pretrained=False
    )
    print(f"✓ Model created (no pretrained): {sum(p.numel() for p in model1.parameters())/1e6:.2f}M params")
    
    # With pretrained - auto-detect path
    import os
    hpc_path = '/scratch/lixuyang/ACF-Network-Full/pretrained_weights/imagenet21k+imagenet2012/imagenet21k+imagenet2012_R50+ViT-B_16.npz'
    local_path = './pretrained_weights/imagenet21k+imagenet2012/imagenet21k+imagenet2012_R50+ViT-B_16.npz'
    
    pretrained_path = hpc_path if os.path.exists(hpc_path) else (local_path if os.path.exists(local_path) else None)
    
    model2 = EnhancedMultimodalFramework(
        rgb_channels=3,
        dsm_channels=1,
        num_classes=6,
        embed_dim=768,
        pretrained=True,
        pretrained_model='R50-ViT-B_16',
        pretrained_path=pretrained_path
    )
    print(f"✓ Model created (with pretrained): {sum(p.numel() for p in model2.parameters())/1e6:.2f}M params")
    
    # Test forward pass
    rgb = torch.randn(2, 3, 256, 256)
    dsm = torch.randn(2, 1, 256, 256)
    
    with torch.no_grad():
        output = model1(rgb, dsm)
        if isinstance(output, tuple):
            print(f"✓ Forward pass successful: main={output[0].shape}, aux={output[1].shape}")
        else:
            print(f"✓ Forward pass successful: {output.shape}")
    
    print("\n✓ All tests passed!")
