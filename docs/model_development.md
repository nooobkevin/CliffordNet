# Model Development

The canonical ImageNet-1k implementation lives in `src/cliffordnet/tasks/imagenet1k.py` and is invoked through `src/cliffordnet/train.py`.

For changes motivated by ParaMind2025/CAN issue #5, keep model and recipe concerns separated:

- Model structure changes:
  - Shift-based CliffordNet / CAN (`arXiv:2601.06793`): `src/cliffordnet/tasks/imagenet1k.py`
  - Subspace Clifford Networks (`arXiv:2610.37226`) & scientific controls: `src/cliffordnet/subspace.py`
  - Fused recomputation Triton kernel (`k in {2, 3, 4}`): `src/cliffordnet/kernels/triton_subspace.py`
- Datasets & evaluation:
  - ImageNet-1k (`data.dataset=imagenet1k`): `src/cliffordnet/tasks/imagenet1k.py`
  - CIFAR-100 & corruption/phase robustness suite (`data.dataset=cifar100`): `src/cliffordnet/data_cifar.py` and `scripts/eval_corruptions.py`
- Training recipe changes: edit YAML profiles such as `configs/cifar100.yaml`, `configs/profiles/stable_imagenet1k.yaml`, or `configs/profiles/imagenet1k_h800_probe.yaml`.
- Runtime/resource changes: edit `src/cliffordnet/train.py`, `src/cliffordnet/resources.py`, or `src/cliffordnet/launch.py` only when the cluster execution behavior changes.
- Sweeps: keep W&B sweep definitions under `configs/sweeps/` and the legacy sweep runner under `experiments/sweeps/`.

Recommended workflow for stability experiments:

1. Start with `configs/profiles/smoke.yaml` to validate imports and dataloading.
2. Use `configs/profiles/probe.yaml` to test a small model quickly.
3. Use `hier_can_tiny` first when you want the closest CAN `model_hier.py` style baseline on ImageNet; `hier_p4` and `hier_p2` remain ImageNet-oriented variants.
4. Apply stable recipe edits in `configs/profiles/stable_imagenet1k.yaml`.
5. Move model-structure edits into `src/cliffordnet/tasks/imagenet1k.py` once the recipe-level signal is clear.
6. Use `configs/profiles/stress.yaml` for long HPC workload runs after the small runs are stable.
