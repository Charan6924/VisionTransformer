#!/bin/bash
#SBATCH --job-name=vit_train
#SBATCH --account=dlw
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH --constraint=gpu2h100
#SBATCH --time=13-08:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

mkdir -p logs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /scratch/pioneer/users/cxv166/VisionTransformer
uv run main.py

echo "End time: $(date)"