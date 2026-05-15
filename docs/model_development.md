# Model Development

The canonical ImageNet-1k implementation lives in `src/cliffordnet/tasks/imagenet1k.py` and is invoked through `src/cliffordnet/train.py`.

For changes motivated by ParaMind2025/CAN issue #5, keep model and recipe concerns separated:

- Model structure changes: edit `src/cliffordnet/tasks/imagenet1k.py` around `CliffordInteraction`, `CliffordBlock`, `CliffordNet`, `HierarchicalCliffordNet`, and the model builders.
- Training recipe changes: edit YAML profiles such as `configs/profiles/stable_imagenet1k.yaml`.
- Runtime/resource changes: edit `src/cliffordnet/train.py`, `src/cliffordnet/resources.py`, or `src/cliffordnet/launch.py` only when the cluster execution behavior changes.
- Sweeps: keep W&B sweep definitions under `configs/sweeps/` and the legacy sweep runner under `experiments/sweeps/`.

Recommended workflow for stability experiments:

1. Start with `configs/profiles/smoke.yaml` to validate imports and dataloading.
2. Use `configs/profiles/probe.yaml` to test a small model quickly.
3. Use `hier_p4` first; `hier_p2` is available when you want the higher-resolution 112 -> 56 -> 28 -> 14 -> 7 variant.
4. Apply stable recipe edits in `configs/profiles/stable_imagenet1k.yaml`.
5. Move model-structure edits into `src/cliffordnet/tasks/imagenet1k.py` once the recipe-level signal is clear.
6. Use `configs/profiles/stress.yaml` for long HPC workload runs after the small runs are stable.
