#!/bin/bash
#SBATCH -A itsc
#SBATCH -p admin
#SBATCH --gpus-per-node=4090d:6
#SBATCH -c 64
#SBATCH --mem 480G
#SBATCH --time 04:00:00
#SBATCH -w gpu41

# uv run python train_imagenet1k.py \
#   --data-dir ./imagenet1k \
#   --model-size nano \
#   --batch-size 48 \
#   --num-gpus 6 \
#   --lr 5e-4 \
#   --gradient-clip-val 1.0

uv run python train_imagenet1k.py   --data-dir ./imagenet1k   --model-size nano   --batch-size 128  --num-gpus 6   --lr 5e-5   --gradient-clip-val 1.0 
#--resume outputs/checkpoints/last-v8.ck