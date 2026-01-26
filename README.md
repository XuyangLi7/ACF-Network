# ACF Network: Adaptive Collaborative Framework for Multi-Modal Remote Sensing Image Segmentation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.10+](https://img.shields.io/badge/pytorch-1.10+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PyTorch implementation of ACF Network for multi-modal remote sensing image segmentation.

## 🔧 Installation

### Requirements
```bash
Python >= 3.8
PyTorch >= 1.10
CUDA >= 11.0 (for GPU)
```

### Setup
```bash
# Clone repository
git clone https://github.com/yourusername/ACF-Network.git
cd ACF-Network

# Create environment
conda create -n acf python=3.8
conda activate acf

# Install dependencies
pip install -r requirements.txt

# Download pretrained weights (optional)
python acf/pretrained.py 
```

## 📊 Supported Datasets

The framework supports multiple remote sensing datasets with different modality combinations:

- **Vaihingen**: RGB + DSM (6 classes)
- **Augsburg**: HSI + PolSAR + DSM (7 classes)
- **MUUFL**: HSI + LiDAR (11 classes)
- **Trento**: HSI + LiDAR (6 classes)

Organize your data as:
```
data/
├── Vaihingen/
│   ├── top/          # RGB images
│   ├── DSM/          # DSM images
│   └── ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE/
├── Augsburg/
├── MUUFL/
└── Trento/
```

## 🏗️ Architecture

ACF Network consists of 5 core modules:

1. **Hierarchical Tokenizer (HT)**: Multi-scale patch embedding
2. **Cross-Modal Attention (CMA)**: Bidirectional modality interaction
3. **Adaptive Modality Balancing (AMB)**: Dynamic modality weighting
4. **Hierarchical Scale Consistency Fusion (HSCF)**: Hierarchical feature fusion
5. **Spatial-Channel Adaptive Fusion (SCAF)**: Dual-branch attention

## 🚀 Usage

### Training
```bash
python train.py \
    --dataset vaihingen \
    --data_path ./data \
    --output_dir ./checkpoints \
    --epochs 180 \
    --batch_size 8 \
    --pretrained 
```

### Evaluation
```bash
python evaluate.py \
    --dataset vaihingen \
    --data_path ./data \
    --model_path checkpoints/best_model.pth \
    --output_dir ./results
```



## 🎯 Pretrained Backbones

Available pretrained models:
- R50-ViT-B_16 (98.6M params) 
- ViT-B_16/32 (86-88M params)
- ViT-L_16/32 (307M params)
- ViT-H_14 (632M params)

Download: `python acf/pretrained.py --download <model_name>`

## 📝 Citation

```bibtex
@article{acf2026,
  title={ACF Network: Adaptive Collaborative Framework for Multi-Modal Remote Sensing Image Segmentation},
  author={Your Name},
  journal={Your Journal},
  year={2026}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.
