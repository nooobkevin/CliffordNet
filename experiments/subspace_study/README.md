# Subspace Clifford Networks (`arXiv:2610.37226`) and CliffordNet (`arXiv:2601.06793`): experimental design, logs, and reproduction guide

This folder holds the experimental design, reproduction commands, result tables, 22 per-epoch CSV trajectories (`epoch_metrics.csv`), sanitized `run_manifest.yaml` files, console logs, and plots from our empirical study of CliffordNet (`arXiv:2601.06793`) and Subspace Clifford Networks (`arXiv:2610.37226`).

> **Note on AI-assisted execution:** The implementation (`src/cliffordnet/subspace.py`, `src/cliffordnet/kernels/triton_subspace.py`), unit tests, and 8x H800 runs in this study were built and executed using an AI coding agent (GitHub Copilot CLI) under `@nooobkevin`'s direction. All logs, manifests, and test scripts are archived here for direct verification.

- Findings report: [`../../docs/subspace_clifford_research_report.md`](../../docs/subspace_clifford_research_report.md)
- Model and control definitions: [`../../src/cliffordnet/subspace.py`](../../src/cliffordnet/subspace.py)
- Recomputation Triton kernel (`k in {2, 3, 4}`): [`../../src/cliffordnet/kernels/triton_subspace.py`](../../src/cliffordnet/kernels/triton_subspace.py)

## 1. Directory layout

```text
experiments/subspace_study/
├── README.md                          # Experimental design & reproduction guide (this file)
├── figures/
│   ├── imagenet1k_6way_curves.png     # ImageNet-1k 6-way learning curves & per-GPU VRAM
│   ├── cifar100_deconfounding_matrix.png # CIFAR-100 iso-param controls & Fourier phase test
│   └── h800_triton_kernel_benchmark.png  # H800 Eager vs torch.compile vs Triton benchmark
├── results/
│   ├── kernel_benchmark_h800.{json,md}
│   ├── summary_cifar100_wave1.{json,md}
│   ├── summary_cifar100_wave2.{json,md}
│   ├── cifar100_corruptions_and_diagnostics.{json,md}
│   └── summary_imagenet1k_6way_complete.{json,md}
└── logs/
    ├── cifar100/<16_run_folders>/     # epoch_metrics.csv (200 ep) + run_manifest.yaml + console_log.txt
    └── imagenet1k/<6_run_folders>/    # epoch_metrics.csv (10 ep)  + run_manifest.yaml + console_log.txt
```

Each `epoch_metrics.csv` records:
`epoch, global_step, runtime_sec, lr, train_loss, train_acc1, val_loss, val_acc1, val_acc5, diag_cancel_lt1e2, diag_wedge_to_dot_ratio, diag_subspace_cos_sim, diag_lambda_wedge_rel_err, diag_effective_rank`.

## 2. Experimental design

Because `arXiv:2610.37226` has no public code yet and reports single-GPU `fp32` CIFAR-100 numbers, we split the study into five tracks to test each mechanism without confounding parameter count or macro-block changes:

1. **Reconstruction and depth scaling (Track 0).** We solved the parameter-count equations across Tables 2 to 4 of `arXiv:2610.37226` (0-parameter error on all eight architectural deltas) and trained `Uniform k in {2, 4}`, `Hybrid (2,4,4)` (`dpr=0.10` and `0.25`), `24-Layer Hybrid (2,4,4)`, and `SpatialDiffusion-free` alongside Shift-Rolling `Hier-CliffordNet` (`S={1,2,4,8}`).
2. **Equal-parameter and macro-block controls (Track 1).** In Table 4 of `arXiv:2610.37226`, `Uniform k=4` (`1.120M`) has 42% more parameters than `Inner-only` (`0.788M`). We added `inner_wide` (widening SwiGLU-style diagonal products $u_i v_i$ to `1.127M` on CIFAR-100 and `3.942M` on ImageNet-1k), `sym_full` (symmetric $u_i v_j + u_j v_i$ at `1.120M` / `3.930M`), and `bilinear_full` (raw $u_i v_j$ at `1.120M`). We also ran the 2x2 swap `{Subspace GP, Rolling GP} x {Paper 1 GGR+LayerNorm2d, Paper 2 Lie-Trotter+DWConv5x5+BN}`.
3. **ImageNet-1k `bf16` scaling (Track 2).** Six `~3.70M` to `3.95M` pyramidal models (`patch_size=4`) trained for 10 epochs (`12.8M` images seen per model) on 2x H800 DDP (`global_batch_size=1024`, Mixup `0.8`, CutMix `1.0`, EMA `0.999`): `in1k_subspace_femto`, `in1k_subspace_femto_inner_wide`, `in1k_subspace_femto_sym`, `in1k_rolling_femto`, and `timm` `ConvNeXt-Atto` (at both `lr=1e-3` and ConvNeXt's standard `lr=4e-3`).
4. **Mixed-precision numerics and Triton kernel benchmark (Tracks 3 and 4).** We measured how `bf16` subtraction affects $u \wedge (z_2 - \lambda u)$ vs. $u \wedge z_2$ (`naive`, `direct_z2`, and `fp32` FMA), and timed eager PyTorch, `torch.compile` (`default` and `max-autotune`), and our recomputation Triton kernel across NCHW and `channels_last`.
5. **Photometric vs. Fourier-phase perturbations (Track 5).** Checkpoints were evaluated on 12 perturbations (`contrast_0.12`, `contrast_0.25`, `contrast_2.0`, `brightness_±0.30`, `gamma_2.2`, `fog_veil`, `gaussian_noise_0.08`, `highpass_edges`, `lowpass_blur`, and amplitude-equalized `fft_phase_only`) to separate contrast sensitivity from phase selectivity.

## 3. Reproduction commands

### Step 0: environment setup

Requires Python >= 3.11 and CUDA GPUs (tested on 8x NVIDIA H800 `sm_90`, PyTorch `2.10.0+cu128`, Triton `3.6.0`).

```bash
uv sync
```

### Step 1: verify parameter counts, gradients, Triton parity, and `torch.compile(fullgraph=True)`

```bash
uv run python tests/test_triton_subspace.py
uv run python tests/test_subspace_models.py
```

### Step 2: run the H800 operator and memory benchmark

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark_subspace_kernel.py \
  --json-out ./outputs/benchmarks/kernel_benchmark_h800.json \
  --md-out ./outputs/benchmarks/kernel_benchmark_h800.md
```

### Step 3: run the 16 CIFAR-100 200-epoch jobs and corruption suite

Single-model run (`subspace_hier_k4` or `subspace_hier_244_dpr025`):

```bash
CUDA_VISIBLE_DEVICES=0 uv run cliffordnet-train \
  --config configs/cifar100.yaml \
  --set model.size=subspace_hier_244_dpr025 \
  --set data.data_dir=./data/cifar100 \
  --set run.output_dir=./outputs/cifar100
```

8-GPU parallel execution of Wave 1 (reproduction and equal-parameter controls) and Wave 2 (2x2 block swap, 24-layer scaling, and `bf16` numerical runs):

```bash
uv run python scripts/run_parallel_experiments.py \
  --spec configs/experiments/cifar100_wave1_repro_and_deconfound.json \
  --gpus 0,1,2,3,4,5,6,7 \
  --data-dir ./data/cifar100 \
  --output-dir ./outputs/cifar100 \
  --summary-prefix ./outputs/summary_cifar100_wave1

uv run python scripts/run_parallel_experiments.py \
  --spec configs/experiments/cifar100_wave2_blocks_scaling_precision_8gpu.json \
  --gpus 0,1,2,3,4,5,6,7 \
  --data-dir ./data/cifar100 \
  --output-dir ./outputs/cifar100 \
  --summary-prefix ./outputs/summary_cifar100_wave2
```

Evaluate all trained CIFAR-100 checkpoints on the 12 perturbations and diagnostic metrics:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_corruptions.py \
  --runs-root ./outputs/cifar100 \
  --data-dir ./data/cifar100 \
  --output-json ./outputs/cifar100_corruptions_and_diagnostics.json \
  --output-md ./outputs/cifar100_corruptions_and_diagnostics.md
```

### Step 4: run the ImageNet-1k `bf16` DDP scaling and control runs

Point `--data-dir` (or `--set data.data_dir=...`) to your Hugging Face `ILSVRC/imagenet-1k` cache directory:

```bash
# 2-GPU DDP example for Subspace-CAN Full (3.93M, 15.1 GiB/GPU at BS=512/GPU)
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc_per_node=2 -m cliffordnet.train \
  --config configs/imagenet1k.yaml \
  --config configs/profiles/imagenet1k_h800_probe.yaml \
  --set data.data_dir=./imagenet1k \
  --set run.output_dir=./outputs/imagenet1k_scale \
  --set run.name=in1k_subspace_femto_10ep \
  --set model.size=in1k_subspace_femto \
  --set runtime.devices=2 \
  --set training.batch_size=512 \
  --set training.target_global_batch_size=1024 \
  --set training.max_epochs=10 \
  --set recipe.warmup_epochs=2
```

Other ImageNet-1k presets use the same command with `--set model.size=<preset>`:
- `in1k_subspace_femto_inner_wide` (equal-parameter SwiGLU / Inner control, `3.94M`)
- `in1k_subspace_femto_sym` (equal-parameter symmetric off-diagonal control, `3.93M`)
- `in1k_rolling_femto` (shift-based `Hier-CliffordNet` `S={1,2,4}`, `3.95M`; add `--set model.compile=false` on PyTorch 2.10 DDP)
- `baseline_convnext_atto` (`timm` ConvNeXt-Atto 4x FFN baseline, `3.70M`, tested at `optim.base_lr=1.0e-3` and `optim.base_lr=4.0e-3`)

Evaluate all 6 ImageNet-1k checkpoints on full 50k validation, `bf16` diagnostics, and perturbations:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_imagenet1k_6way.py \
  --runs-root ./outputs/imagenet1k_scale \
  --data-dir ./imagenet1k \
  --output-prefix ./outputs/summary_imagenet1k_6way_complete
```
