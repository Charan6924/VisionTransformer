#!/bin/bash
#SBATCH --job-name=vit_train
#SBATCH --account=dlw
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH --constraint=gpu2h100
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

# Create logs directory
mkdir -p logs

# Run training
cd /scratch/pioneer/users/cxv166/VisionTransformer
uv run main.py
echo "End time: $(date)"