# Quick Start Guide

Get started with ACF Network in 5 minutes!

## 📦 Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/ACF-Network.git
cd ACF-Network
```

### 2. Create Environment
```bash
conda create -n acf python=3.8
conda activate acf
pip install -r requirements.txt
```

### 3. Download Data
Download the Vaihingen dataset and organize as:
```
data/Vaihingen/
├── top/                    # RGB images
├── DSM/                    # DSM images
└── ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE/
```

## 🚀 Training

### Basic Training
```bash
python train.py \
    --dataset vaihingen \
    --data_path ./data \
    --output_dir ./checkpoints \
    --epochs 100 \
    --batch_size 8
```

### With Pretrained Backbone
```bash
# Download pretrained weights first
python acf/pretrained.py --download R50-ViT-B_16

# Train with pretrained backbone
python train.py \
    --dataset vaihingen \
    --pretrained R50-ViT-B_16 \
    --epochs 100
```

### With Visualization
```bash
python train.py \
    --dataset vaihingen \
    --enable_vis \
    --vis_interval 10
```

## 📊 Evaluation

```bash
python evaluate.py \
    --dataset vaihingen \
    --data_path ./data \
    --model_path checkpoints/best_model.pth \
    --output_dir ./results
```

## 🎨 Visualization

```bash
python visualize.py \
    --dataset vaihingen \
    --checkpoint checkpoints/best_model.pth \
    --data_root ./data \
    --save_dir ./visualizations \
    --vis_types all
```

## 📈 Monitor Training

Training logs and visualizations are saved to:
- Checkpoints: `{output_dir}/checkpoints/`
- Logs: `{output_dir}/logs/`
- Visualizations: `{output_dir}/visualizations/`

## 🔧 Common Issues

### CUDA Out of Memory
Reduce batch size:
```bash
python train.py --batch_size 4
```

### Slow Training
Enable mixed precision:
```bash
python train.py --amp
```

### Multi-GPU Training
```bash
python train.py --batch_size 16  # Automatically uses all GPUs
```

## 📚 Next Steps

- Read the [full documentation](../README.md)
- Check [training guide](TRAINING.md)
- Explore [visualization options](../VISUALIZATION_GUIDE.md)
- Try [different datasets](DATASETS.md)

## 💡 Tips

1. **Start small**: Use a small dataset or subset for initial testing
2. **Use pretrained weights**: Significantly improves convergence
3. **Monitor metrics**: Check OA and mIoU during training
4. **Visualize results**: Use `--enable_vis` to track progress
5. **Experiment**: Try different hyperparameters and configurations

## 🆘 Need Help?

- Check [FAQ](FAQ.md)
- Open an [issue](https://github.com/yourusername/ACF-Network/issues)
- Read [contributing guide](../CONTRIBUTING.md)

Happy training! 🎉
