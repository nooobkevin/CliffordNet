#!/bin/bash
set -euo pipefail

# Generate a cluster-specific Slurm script from YAML config, then submit it.
# Override site resources without editing this file, for example:
#   bash run.sh --set slurm.account=itsc --set slurm.partition=admin --set slurm.nodes=2 --set slurm.gpus_per_node=6

uv run python -m cliffordnet.launch \
  --launcher slurm \
  --submit \
  --config configs/imagenet1k.yaml \
  --config configs/profiles/stable_imagenet1k.yaml \
  --config configs/profiles/slurm_t1.yaml \
  "$@"
