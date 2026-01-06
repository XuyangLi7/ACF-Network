#!/usr/bin/env python3
"""
Test script to verify pretrained configuration and model loading
"""

import torch
from unified_config import UNIFIED_CONFIG
from enhanced_multimodal_framework import EnhancedMultimodalFramework
import json

def test_config():
    """Test configuration"""
    print("="*80)
    print("Testing Unified Configuration")
    print("="*80)
    
    print("\nModel Configuration:")
    print(json.dumps(UNIFIED_CONFIG['model'], indent=2))
    
    print("\n" + "="*80)
    print("Configuration loaded successfully!")
    print("="*80)

def test_model_creation():
    """Test model creation with pretrained weights"""
    print("\n" + "="*80)
    print("Testing Model Creation")
    print("="*80)
    
    cfg = UNIFIED_CONFIG['model']
    
    print(f"\nCreating model with:")
    print(f"  - RGB channels: {cfg['rgb_channels']}")
    print(f"  - DSM channels: {cfg['dsm_channels']}")
    print(f"  - Num classes: {cfg['num_classes']}")
    print(f"  - Embed dim: {cfg['embed_dim']}")
    print(f"  - Pretrained: {cfg.get('pretrained', False)}")
    print(f"  - Pretrained model: {cfg.get('pretrained_model', 'N/A')}")
    print(f"  - Pretrained path: {cfg.get('pretrained_path', 'N/A')}")
    
    try:
        model = EnhancedMultimodalFramework(
            rgb_channels=cfg['rgb_channels'],
            dsm_channels=cfg['dsm_channels'],
            num_classes=cfg['num_classes'],
            embed_dim=cfg['embed_dim'],
            enable_remote_sensing_innovations=cfg['enable_remote_sensing_innovations'],
            pretrained=cfg.get('pretrained', False),
            pretrained_model=cfg.get('pretrained_model', 'R50-ViT-B_16'),
            pretrained_path=cfg.get('pretrained_path', None),
            use_multi_scale_aggregator=cfg.get('use_multi_scale_aggregator', False),
            use_simple_mode=cfg.get('use_simple_mode', False)
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n✓ Model created successfully!")
        print(f"  - Total parameters: {total_params/1e6:.2f}M")
        print(f"  - Trainable parameters: {trainable_params/1e6:.2f}M")
        
        # Test forward pass
        print("\nTesting forward pass...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        rgb = torch.randn(2, cfg['rgb_channels'], 256, 256).to(device)
        dsm = torch.randn(2, cfg['dsm_channels'], 256, 256).to(device)
        
        with torch.no_grad():
            output = model(rgb, dsm)
        
        if isinstance(output, tuple):
            print(f"✓ Forward pass successful!")
            print(f"  - Main output shape: {output[0].shape}")
            print(f"  - Auxiliary output shape: {output[1].shape}")
        else:
            print(f"✓ Forward pass successful!")
            print(f"  - Output shape: {output.shape}")
        
        print("\n" + "="*80)
        print("All tests passed!")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """Main test function"""
    print("\n" + "="*80)
    print("Pretrained Configuration Test Suite")
    print("="*80)
    
    # Test 1: Configuration
    test_config()
    
    # Test 2: Model creation
    success = test_model_creation()
    
    if success:
        print("\n✓ All tests completed successfully!")
        print("\nYou can now run training with:")
        print("  python train.py --dataset_name vaihingen --data_path ./data/Vaihingen")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")

if __name__ == '__main__':
    main()
