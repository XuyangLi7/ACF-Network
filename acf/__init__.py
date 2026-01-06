"""
ACF: Adaptive Collaborative Fusion Network
Complete implementation for multi-modal remote sensing semantic segmentation
"""

from .network import (
    ACFNetwork,
    create_acf_model,
    HierarchicalTokenizer,
    CrossModalAttention,
    AdaptiveModalityBalancing,
    MultiGranularityConsistencyFusion,
    SpatialChannelAdaptiveFactor
)

from .pretrained import (
    load_pretrained_acf,
    list_pretrained_models,
    get_pretrained_config,
    PRETRAINED_MODELS
)

from .visualization import (
    GradCAM,
    TSNEVisualizer,
    GradientAttributionVisualizer,
    PredictionVisualizer,
    ModuleEffectivenessVisualizer,
    create_comprehensive_visualization
)

__version__ = "1.0.0"
__author__ = "Your Name"
__all__ = [
    # Network
    'ACFNetwork',
    'create_acf_model',
    'HierarchicalTokenizer',
    'CrossModalAttention',
    'AdaptiveModalityBalancing',
    'MultiGranularityConsistencyFusion',
    'SpatialChannelAdaptiveFactor',
    # Pretrained
    'load_pretrained_acf',
    'list_pretrained_models',
    'get_pretrained_config',
    'PRETRAINED_MODELS',
    # Visualization
    'GradCAM',
    'TSNEVisualizer',
    'GradientAttributionVisualizer',
    'PredictionVisualizer',
    'ModuleEffectivenessVisualizer',
    'create_comprehensive_visualization'
]
