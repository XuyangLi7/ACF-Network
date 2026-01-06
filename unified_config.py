#!/usr/bin/env python3
"""
Unified configuration module - imports from acf.config
This module provides backward compatibility for imports from unified_config
"""

from acf.config import (
    MODEL_CONFIG,
    DATA_CONFIG,
    TRAIN_CONFIG,
    EVAL_CONFIG,
    PREDICT_CONFIG,
    HPC_CONFIG,
    LOCAL_CONFIG,
    PATH_CONFIG,
    UNIFIED_CONFIG
)

__all__ = [
    'MODEL_CONFIG',
    'DATA_CONFIG',
    'TRAIN_CONFIG',
    'EVAL_CONFIG',
    'PREDICT_CONFIG',
    'HPC_CONFIG',
    'LOCAL_CONFIG',
    'PATH_CONFIG',
    'UNIFIED_CONFIG'
]
