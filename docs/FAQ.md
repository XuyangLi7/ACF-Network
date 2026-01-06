# Frequently Asked Questions (FAQ)

## General Questions

### Q: What is ACF Network?
**A:** ACF Network is an Adaptive Collaborative Framework for multi-modal remote sensing image segmentation. It seamlessly integrates multiple data modalities (RGB, DSM, HSI, LiDAR, SAR) for improved segmentation accuracy.

### Q: What datasets are supported?
**A:** Currently supports:
- Vaihingen (RGB + DSM)
- Augsburg (HSI + PolSAR + DSM)
- MUUFL (HSI + LiDAR)
- Trento (HSI + LiDAR)

### Q: Can I use my own dataset?
**A:** Yes! Follow the dataset preparation guide in `docs/DATASETS.md` to add custom datasets.

## Installation & Setup

### Q: What are the system requirements?
**A:** 
- Python >= 3.8
- PyTorch >= 1.10
- CUDA >= 11.0 (for GPU)
- 16GB+ RAM recommended
- GPU with 8GB+ VRAM recommended

### Q: Installation fails with dependency errors
**A:** Try:
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

### Q: Can I run without GPU?
**A:** Yes, but training will be very slow. Evaluation and visualization work fine on CPU.

## Training

### Q: How long does training take?
**A:** 
- Vaihingen: ~2-3 hours on single RTX 3090
- With pretrained backbone: ~1-2 hours
- Multi-GPU: proportionally faster

### Q: CUDA out of memory error
**A:** Solutions:
1. Reduce batch size: `--batch_size 4`
2. Reduce window size: `--window_size 128`
3. Enable gradient checkpointing (if available)
4. Use mixed precision: `--amp`

### Q: Training loss not decreasing
**A:** Check:
1. Learning rate (try `--lr 1e-4`)
2. Data normalization
3. Pretrained weights loading
4. Dataset paths are correct

### Q: How to resume training?
**A:** Use the checkpoint:
```bash
python train.py --resume checkpoints/last_checkpoint.pth
```

### Q: Best hyperparameters?
**A:** Default settings work well. For fine-tuning:
- Learning rate: 1e-4 to 5e-4
- Batch size: 8-16 (depending on GPU)
- Epochs: 100-200
- Use pretrained backbone

## Evaluation

### Q: Evaluation is very slow
**A:** Increase stride:
```bash
python evaluate.py --stride 32  # Faster but less accurate
python evaluate.py --stride 4   # Slower but more accurate
```

### Q: Results don't match paper
**A:** Ensure:
1. Using same dataset split
2. Same preprocessing
3. Same evaluation metrics
4. Correct checkpoint loaded

### Q: How to evaluate on custom images?
**A:** Modify the dataloader to load your images, or use the inference script (coming soon).

## Visualization

### Q: Visualization fails with memory error
**A:** Reduce number of samples:
```bash
python visualize.py --num_samples 5
```

### Q: How to visualize specific layers?
**A:** Use `--gradcam_layers`:
```bash
python visualize.py --vis_types gradcam --gradcam_layers scaf mgcf
```

### Q: Can I visualize during training?
**A:** Yes:
```bash
python train.py --enable_vis --vis_interval 10
```

## Model & Architecture

### Q: Which pretrained backbone is best?
**A:** 
- **R50-ViT-B_16**: Best balance of speed and accuracy (recommended)
- **ViT-L_16**: Higher accuracy, slower
- **ViT-B_16**: Faster, slightly lower accuracy

### Q: Can I modify the architecture?
**A:** Yes! The code is modular. See `acf/network.py` for the main architecture.

### Q: How to perform ablation studies?
**A:** Modify the model creation in `acf/network.py` to disable specific modules.

### Q: Model size and parameters?
**A:** 
- Base model: ~98M parameters
- With R50-ViT-B_16: ~98.6M parameters
- With ViT-L_16: ~307M parameters

## Data & Preprocessing

### Q: What image sizes are supported?
**A:** Any size! The model uses sliding window inference for large images.

### Q: How to handle missing modalities?
**A:** The model automatically adapts. Just provide available modalities in the dataloader.

### Q: Data augmentation options?
**A:** Currently supports:
- Random flip
- Random rotation
- Random crop
- Color jittering (for RGB)

## Performance

### Q: How to speed up training?
**A:** 
1. Use mixed precision: `--amp`
2. Increase batch size
3. Use multiple GPUs
4. Use pretrained backbone
5. Reduce validation frequency

### Q: How to improve accuracy?
**A:** 
1. Use pretrained backbone
2. Train longer (more epochs)
3. Use data augmentation
4. Tune hyperparameters
5. Ensemble multiple models

### Q: Multi-GPU training not working?
**A:** 
- Ensure CUDA is properly installed
- Check GPU visibility: `nvidia-smi`
- Use correct batch size (divisible by num GPUs)

## Errors & Debugging

### Q: "RuntimeError: CUDA error: out of memory"
**A:** See "CUDA out of memory error" above.

### Q: "FileNotFoundError: data not found"
**A:** Check:
1. Data path is correct
2. Dataset structure matches expected format
3. Files have correct names

### Q: "KeyError: 'model_state_dict'"
**A:** Checkpoint format issue. This should be fixed in latest version. Update your code.

### Q: "ImportError: No module named 'acf'"
**A:** Ensure you're in the project root directory and ACF is in Python path.

## Contributing

### Q: How can I contribute?
**A:** See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

### Q: Found a bug, what should I do?
**A:** Open an issue on GitHub with:
- Clear description
- Steps to reproduce
- Error messages
- Environment details

### Q: Can I add a new dataset?
**A:** Yes! We welcome dataset contributions. Follow the dataset template in `universal_dataset.py`.

## Miscellaneous

### Q: License?
**A:** MIT License - free for academic and commercial use.

### Q: How to cite?
**A:** See citation in README.md.

### Q: Where to get help?
**A:** 
1. Check this FAQ
2. Read documentation
3. Search existing issues
4. Open new issue
5. Contact maintainers

### Q: Roadmap for future features?
**A:** See CHANGELOG.md and GitHub issues for planned features.

---

**Still have questions?** Open an issue on [GitHub](https://github.com/yourusername/ACF-Network/issues)!
