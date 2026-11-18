# Subspace Clifford Networks (`arXiv:2610.37226`) & CliffordNet (`arXiv:2601.06793`) — Experimental Design, Logs & Reproduction Guide

This directory contains the complete experimental design, step-by-step reproduction guide, structured JSON/Markdown result tables, 22 per-epoch CSV learning/diagnostic trajectories (`epoch_metrics.csv`), sanitized `run_manifest.yaml` configs, cleaned console logs, and summary figures for our empirical study of **CliffordNet** (`arXiv:2601.06793`) and **Subspace Clifford Networks** (`arXiv:2610.37226`).

- **Full Findings & Positive/Negative Evidence Report**: [`../../docs/subspace_clifford_research_report.md`](../../docs/subspace_clifford_research_report.md)
- **Model & Control Implementations**: [`../../src/cliffordnet/subspace.py`](../../src/cliffordnet/subspace.py)
- **Recomputation Fused Triton Kernel (`k in {2, 3, 4}`)**: [`../../src/cliffordnet/kernels/triton_subspace.py`](../../src/cliffordnet/kernels/triton_subspace.py)

---

## 1. Directory Layout

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

---

## 2. Experimental Design (5 Scientific Tracks)

Because `arXiv:2610.37226` had no released code and evaluated exclusively on single-GPU `fp32` CIFAR-100, our experimental matrix was designed to separate architectural mechanisms from confounding variables:

1. **Track 0 — Exact Architectural Reconstruction & Depth Scaling**:
   - Solved the linear system of parameter counts across Tables 2–4 of `arXiv:2610.37226` (achieving **0-parameter error** on all 8 architectural deltas) and verified `Uniform k ∈ {2, 3, 4}`, `Hybrid (2,4,4)` (`dpr=0.10` and `0.25`), `24-Layer Hybrid (2,4,4)`, and `SpatialDiffusion-free` against Shift-Rolling `Hier-CliffordNet` (`S={1,2,4,8}`).
2. **Track 1 — Strict Iso-Parameter Deconfounding Controls (`cli_mode` & $2 \times 2$ Block Factorial)**:
   - *Fixing Table 4's `+42%` parameter confound*: `Uniform k=4` (`1.120M`) has 42% more parameters than `Inner-only` (`0.788M`). We added:
     - `inner_wide` (**Control 1**): Widens internal diagonal gating (`SwiGLU`-style $u_i v_i$) to match `Full k=4` parameter count (`1.127M` on CIFAR-100, `3.942M` on ImageNet-1k).
     - `sym_full` (**Control 2a**): Replaces anti-symmetric Grassmann wedge ($u_i v_j - u_j v_i$) with symmetric off-diagonal terms ($u_i v_j + u_j v_i$) at **identical** parameter count (`1.120M` / `3.930M`).
     - `bilinear_full` (**Control 2b**): Uses raw asymmetric bilinear cross-terms ($u_i v_j$, $i < j$) without skew-symmetrization at **identical** parameter count (`1.120M`).
   - *Fixing the macro-block confound*: Ran the $2 \times 2$ factorial `{Subspace GP, Rolling GP} × {Paper 1 GGR+LayerNorm2d Block, Paper 2 Lie-Trotter+DW5x5+BN Block}`.
3. **Track 2 — Scaling to ImageNet-1k (`bf16-mixed`, 1.28M Train / 50k Val)**:
   - Evaluated 6 matched `~3.70M–3.95M` models under identical 2×H800 DDP (`global_batch_size=1024`, Mixup `0.8`, CutMix `1.0`, EMA `0.999`, 10 full epochs = 12.8M seen images/model): `in1k_subspace_femto` (Full), `in1k_subspace_femto_inner_wide` (Iso-Param SwiGLU), `in1k_subspace_femto_sym` (Iso-Param Symmetric), `in1k_rolling_femto` (Shift-Rolling `S={1,2,4}`), and `timm` `ConvNeXt-Atto` (`4×` MLP FFN, tested at both matched `lr=1e-3` and ConvNeXt's canonical `lr=4e-3`).
4. **Track 3 & Track 4 — Numerical `bf16` $\lambda$-Invariance Diagnostics & Recomputation Triton Kernel Benchmark**:
   - Quantified floating-point violation of $u \wedge (z_2 - \lambda u) = u \wedge z_2$ in `bf16` (`naive` vs `direct_z2` vs `fp32` FMA) and benchmarked `PyTorch Eager` vs `torch.compile` (`default` & `max-autotune`) vs our **Fused Recomputation Triton Kernel** across NCHW and `channels_last` (NHWC).
5. **Track 5 — Falsifiable Photometric vs. Fourier-Phase Robustness Suite**:
   - Evaluated checkpoints across 12 corruptions (`contrast_0.12`, `contrast_0.25`, `contrast_2.0`, `brightness_±0.30`, `gamma_2.2`, `fog_veil`, `gaussian_noise`, `highpass_edges`, `lowpass_blur`, and amplitude-equalized `fft_phase_only`) to test whether bivectors confer contrast invariance vs. pure phase/structural selectivity.

---

## 3. Step-by-Step Reproduction Guide

### Step 0: Environment Setup

Requires Python $\ge 3.11$ and CUDA GPUs (tested on 8× NVIDIA H800 `sm_90`, PyTorch `2.10.0+cu128`, Triton `3.6.0`).

```bash
uv sync
```

### Step 1: Verify Parameter Counts, Gradients, Triton Parity & `torch.compile(fullgraph=True)`

```bash
uv run python tests/test_triton_subspace.py
uv run python tests/test_subspace_models.py
```

### Step 2: Reproduce the H800 Operator & Memory Benchmark (Track 4)

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark_subspace_kernel.py \
  --json-out ./outputs/benchmarks/kernel_benchmark_h800.json \
  --md-out ./outputs/benchmarks/kernel_benchmark_h800.md
```

### Step 3: Reproduce the 16 × 200-Epoch CIFAR-100 Matrix & Corruption Suite (Tracks 0, 1, 3, 5)

Single-model run (e.g., `subspace_hier_k4` or `subspace_hier_244_dpr025`):

```bash
CUDA_VISIBLE_DEVICES=0 uv run cliffordnet-train \
  --config configs/cifar100.yaml \
  --set model.size=subspace_hier_244_dpr025 \
  --set data.data_dir=./data/cifar100 \
  --set run.output_dir=./outputs/cifar100
```

8-GPU parallel execution of Wave 1 (reproduction + iso-param algebraic controls) and Wave 2 ($2 \times 2$ block factorial + 24L scaling + `bf16` numerical runs):

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

Run the 12-corruption + Fourier phase + mechanistic diagnostic evaluator on all trained CIFAR-100 checkpoints:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_corruptions.py \
  --runs-root ./outputs/cifar100 \
  --data-dir ./data/cifar100 \
  --output-json ./outputs/cifar100_corruptions_and_diagnostics.json \
  --output-md ./outputs/cifar100_corruptions_and_diagnostics.md
```

### Step 4: Reproduce the ImageNet-1k `bf16` DDP Scaling & Control Runs (Tracks 2 & 3)

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

Other ImageNet-1k comparison presets use the exact same command with `--set model.size=<preset>`:
- `in1k_subspace_femto_inner_wide` (Iso-Param SwiGLU / Inner control, `3.94M`)
- `in1k_subspace_femto_sym` (Iso-Param Symmetric off-diagonal control, `3.93M`)
- `in1k_rolling_femto` (Shift-based `Hier-CliffordNet` `S={1,2,4}`, `3.95M`, add `--set model.compile=false` on PyTorch 2.10 DDP)
- `baseline_convnext_atto` (`timm` ConvNeXt-Atto 4×FFN baseline, `3.70M`, tested at `optim.base_lr=1.0e-3` and `optim.base_lr=4.0e-3`)

Evaluate all 6 ImageNet-1k checkpoints on full 50k validation + `bf16` diagnostics + corruptions:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_imagenet1k_6way.py \
  --runs-root ./outputs/imagenet1k_scale \
  --data-dir ./imagenet1k \
  --output-prefix ./outputs/summary_imagenet1k_6way_complete
```
