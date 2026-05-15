# H800 Acceptance Runs

These commands exercise both canonical training and W&B sweep paths without SSD prefetch. Set `data.data_dir` to the shared t1 ImageNet-1k Hugging Face cache path on the cluster.

Direct node run:

```bash
uv run cliffordnet-launch \
  --launcher local \
  --config configs/imagenet1k.yaml \
  --config configs/profiles/stable_imagenet1k.yaml \
  --set data.data_dir=/t1/imagenet1k \
  --set runtime.devices=8 \
  --set run.name=h800_direct_acceptance
```

Slurm training run:

```bash
uv run cliffordnet-launch \
  --launcher slurm \
  --submit \
  --config configs/imagenet1k.yaml \
  --config configs/profiles/stress.yaml \
  --config configs/profiles/slurm_h800.yaml \
  --set data.data_dir=/t1/imagenet1k \
  --set slurm.account=<account> \
  --set slurm.partition=<partition>
```

W&B hyperparameter sweep:

```bash
wandb sweep configs/sweeps/hparam.yaml
wandb agent <entity>/<project>/<sweep-id>
```

Recommended acceptance criteria:

- Slurm script contains `--gpus-per-node=h800:8` and `torchrun --nproc_per_node=8`.
- W&B run config records `model.size=hier_can_tiny`, detected resources, effective global batch size, and LR.
- Training creates `run_manifest.yaml` and `checkpoints/last.ckpt` under `run.output_dir/run.name`.
- Sweep runs auto-detect `batch_size` when sweep config sets `batch_size=0`.
