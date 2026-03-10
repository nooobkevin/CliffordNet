#!/bin/bash
#SBATCH -A itsc
#SBATCH -p admin
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=6
#SBATCH --gpus-per-task=rtx5880:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-gpu=80G
#SBATCH --time 04:00:00
#SBATCH -w gpu40,gpu36

# LR before scaled by batch size and number of nodes
srun uv run python train_imagenet1k.py \
  --data-dir ./imagenet1k \
  --model-size small \
  --batch-size 128 \
  --num-nodes 2 \
  --num-gpus 6 \
  --lr 3e-4 \
  --prefetch-local \
  --output-dir ./outputs \
  --gradient-clip-val 1.0