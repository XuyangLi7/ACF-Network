#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --account=xdzhang
#SBATCH --clusters=priv
#SBATCH --partition=priv_v100x4
#SBATCH --job-name=acf-train

python train.py \
    --dataset vaihingen \
    --output_dir outputs/vaihingen_ultimate \
    --enable_visualization false \
    --enable_interpretability


