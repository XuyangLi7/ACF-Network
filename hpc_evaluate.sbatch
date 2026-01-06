#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --account=xdzhang
#SBATCH --clusters=priv
#SBATCH --partition=priv_v100x4
#SBATCH --job-name=acf-eval


mkdir -p logs
mkdir -p results



python evaluate.py \
    --model_path outputs/vaihingen_ultimate/best_enhanced_model_epoch_197.pth \
    --data_path ./data \
    --output_dir ./results/vaihingen_eval_final \
    --stride 64 \
    --window_size 512


