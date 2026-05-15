#!/bin/bash
set -euo pipefail

# Local/node-direct launcher. Pass config overrides as repeated --set key=value.
# Example:
#   bash run_training.sh --set data.data_dir=/t1/imagenet1k --set runtime.devices=8

uv run python -m cliffordnet.launch \
  --launcher local \
  --config configs/imagenet1k.yaml \
  --config configs/profiles/stable_imagenet1k.yaml \
  "$@"
