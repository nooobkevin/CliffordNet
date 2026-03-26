# CliffordNet

ImageNet-1k classification using Clifford algebra-based neural networks. Built with PyTorch Lightning.

## Usage

### Install

```bash
uv sync
```

### Train

Single GPU:

```bash
uv run python train_imagenet1k.py \
  --data-dir ./imagenet1k \
  --model-size small \
  --batch-size 64 \
  --lr 1e-3
```

Multi-GPU (single node, 8 GPUs):

```bash
uv run python train_imagenet1k.py \
  --data-dir ./imagenet1k \
  --model-size small \
  --batch-size 64 \
  --num-gpus 8 \
  --lr 1e-3
```

Multi-node (via SLURM):

```bash
sbatch run.sh
```

Set `--batch-size 0` to auto-detect the largest batch size that fits in GPU memory.

### GPU Stress Test

```bash
bash gpu_stress_test.sh
```

Runs single-GPU tests on each card, then a multi-GPU DDP communication test.
