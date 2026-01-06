#!/bin/bash
# Quick Demo Script for ACF Network
# This script demonstrates basic usage with minimal setup

set -e  # Exit on error

echo "=================================="
echo "ACF Network - Quick Demo"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda not found. Please install Anaconda or Miniconda.${NC}"
    exit 1
fi

# Create environment
echo -e "${GREEN}Step 1: Creating conda environment...${NC}"
conda create -n acf python=3.8 -y
source activate acf

# Install dependencies
echo -e "${GREEN}Step 2: Installing dependencies...${NC}"
pip install -r requirements.txt

# Download pretrained weights
echo -e "${GREEN}Step 3: Downloading pretrained weights...${NC}"
python acf/pretrained.py --download R50-ViT-B_16

# Check if data exists
if [ ! -d "data/Vaihingen" ]; then
    echo -e "${YELLOW}Warning: Vaihingen dataset not found in data/Vaihingen${NC}"
    echo -e "${YELLOW}Please download and prepare the dataset first.${NC}"
    echo -e "${YELLOW}See docs/DATASETS.md for instructions.${NC}"
    exit 1
fi

# Train for a few epochs (demo)
echo -e "${GREEN}Step 4: Training demo (5 epochs)...${NC}"
python train.py \
    --dataset vaihingen \
    --data_path ./data \
    --output_dir ./demo_output \
    --epochs 5 \
    --batch_size 4 \
    --pretrained R50-ViT-B_16 \
    --enable_vis

# Evaluate
echo -e "${GREEN}Step 5: Evaluating model...${NC}"
python evaluate.py \
    --dataset vaihingen \
    --data_path ./data \
    --model_path demo_output/checkpoints/best_model.pth \
    --output_dir ./demo_results

# Visualize
echo -e "${GREEN}Step 6: Generating visualizations...${NC}"
python visualize.py \
    --dataset vaihingen \
    --checkpoint demo_output/checkpoints/best_model.pth \
    --data_root ./data \
    --save_dir ./demo_visualizations \
    --num_samples 5 \
    --vis_types prediction

echo -e "${GREEN}=================================="
echo -e "Demo completed successfully!"
echo -e "=================================="
echo -e "Results saved to:"
echo -e "  - Checkpoints: demo_output/checkpoints/"
echo -e "  - Evaluation: demo_results/"
echo -e "  - Visualizations: demo_visualizations/"
echo -e "${NC}"
