#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --account=xdzhang
#SBATCH --clusters=priv
#SBATCH --partition=priv_v100x4

#SBATCH --job-name=acf-vis



mkdir -p logs
mkdir -p visualizations
mkdir -p outputs/interpretability_analysis



python visualize.py \
    --dataset vaihingen \
    --checkpoint outputs/vaihingen_ultimate/best_enhanced_model_epoch_197.pth \
    --data_root ./data \
    --save_dir ./visualizations/vaihingen_final \
    --num_samples 10 \
    --vis_types all


    
python analyze_training_interpretability.py \
    --interp_dir outputs/vaihingen_ultimate/interpretability \
    --output_dir outputs/interpretability_analysis


