# CliffordNet

ImageNet-1k classification using Clifford algebra-based neural networks. Built with PyTorch Lightning.

## Usage

### Install

```bash
uv sync
```

### Train

The canonical entrypoint is config-driven. Later config files override earlier ones, and `--set key=value` overrides both.

Smoke test:

```bash
uv run cliffordnet-train \
  --config configs/imagenet1k.yaml \
  --config configs/profiles/smoke.yaml \
  --set data.data_dir=./imagenet1k
```

Stable ImageNet-1k workload:

```bash
uv run cliffordnet-train \
  --config configs/imagenet1k.yaml \
  --config configs/profiles/stable_imagenet1k.yaml \
  --set data.data_dir=/t1/imagenet1k
```

Set `training.batch_size=0` to run OOM-aware batch-size probing. The trainer then auto-adjusts `training.accumulate_grad_batches` to reach `training.target_global_batch_size` and scales LR from `optim.base_lr` unless `optim.lr` is set explicitly.

### Local Launcher

```bash
uv run cliffordnet-launch \
  --launcher local \
  --config configs/imagenet1k.yaml \
  --config configs/profiles/stress.yaml \
  --set data.data_dir=/t1/imagenet1k \
  --set runtime.devices=8
```

### Slurm Launcher

Generate and submit a Slurm script:

```bash
uv run cliffordnet-launch \
  --launcher slurm \
  --submit \
  --config configs/imagenet1k.yaml \
  --config configs/profiles/stress.yaml \
  --config configs/profiles/slurm_t1.yaml \
  --set data.data_dir=/t1/imagenet1k \
  --set slurm.account=<account> \
  --set slurm.partition=<partition> \
  --set slurm.nodes=2 \
  --set slurm.gpus_per_node=6
```

`run.sh` is a thin wrapper around the Slurm launcher. `run_training.sh` is a thin wrapper around the local launcher.

### Config Profiles

- `configs/imagenet1k.yaml`: base workload config
- `configs/profiles/smoke.yaml`: tiny offline smoke run
- `configs/profiles/probe.yaml`: short probe model run for quick signal
- `configs/profiles/stress.yaml`: long-running HPC workload profile
- `configs/profiles/stable_imagenet1k.yaml`: conservative recipe for ImageNet-1k loss-spike experiments
- `configs/profiles/slurm_t1.yaml`: site-neutral Slurm defaults

### Resume

By default `checkpoint.resume=auto`. The trainer resumes from `last.ckpt` or `preempted.ckpt` under `run.output_dir/run.name/checkpoints` when present. Set a stable `run.name` to resume a local run; the Slurm launcher freezes `run.name` in the generated script automatically. SIGTERM triggers a `preempted.ckpt` save before exit.

### W&B

W&B is online by default. Each run logs the merged config, detected GPU/CPU/Slurm resources, computed batch/accumulation/LR settings, and a `run_manifest.yaml` artifact. Set `wandb.mode=offline` for offline runs.

### Legacy Entrypoint

```bash
uv run python -m cliffordnet.tasks.imagenet1k --help
```

The canonical ImageNet-1k task lives at `src/cliffordnet/tasks/imagenet1k.py`. New workloads should use `cliffordnet-train` or `cliffordnet-launch`.

### GPU Stress Test

```bash
bash scripts/gpu_stress_test.sh
```

Runs single-GPU tests on each card, then a multi-GPU DDP communication test.

## Repository Layout

- `src/cliffordnet/`: package code and canonical runtime
- `src/cliffordnet/tasks/imagenet1k.py`: ImageNet-1k model/task implementation for CAN issue-driven model changes
- `configs/`: base config, workload profiles, and W&B sweeps
- `experiments/legacy/`: old exploratory scripts kept for reference
- `experiments/sweeps/`: W&B sweep runner
- `scripts/`: operational scripts that call the canonical entrypoints
- `docs/model_development.md`: notes for model-structure and stability experiments
- `docs/h800_acceptance.md`: direct, Slurm, and sweep commands for H800 node acceptance

## Diagnostic Metrics

When `--enable-diagnostics` is on (default), the trainer logs wedge-product health metrics every `--diag-log-interval` steps under the `diag/` prefix. These metrics diagnose numerical stability of the Clifford geometric product.

### Background

The wedge product computes `a*b - c*d` (difference of two products). When `a*b ≈ c*d`, subtraction causes **catastrophic cancellation** — especially severe in bf16 (7-bit mantissa). The diagnostics quantify exactly how much cancellation is occurring and whether the wedge branch is still contributing meaningful signal.

### Cancellation

```
term_a = z_det * z_ctx_rolled          (= a*b)
term_b = z_det_rolled * z_ctx          (= c*d)
rel_diff = |term_a - term_b| / (|term_a| + |term_b| + ε)
```

| Metric | Meaning | Healthy | Concerning |
|--------|---------|---------|------------|
| `cancel/rel_diff_mean` | Average relative difference across all elements. Smaller = more cancellation. | > 0.1 | < 0.01 (wedge is computing near-zero, signal drowned by noise) |
| `cancel/rel_diff_lt1e-2` | Fraction of elements with rel_diff < 1%. Measures how many elements suffer severe cancellation. | < 20% | > 50% (majority of wedge elements are noise) |
| `cancel/rel_diff_lt1e-4` | Fraction with rel_diff < 0.01%. Extreme cancellation where bf16 effective precision is near zero. | < 5% | > 10% |

If `rel_diff_lt1e-2` rises during training, det and ctx branches are learning similar representations and the wedge is degenerating. Increase `--ortho-weight` or switch to `fma` wedge mode.

### Magnitude

| Metric | Meaning | How to read |
|--------|---------|-------------|
| `magnitude/wedge_abs_mean` | Mean |wedge| output | Compare with dot. If wedge << dot (>100x gap), wedge branch contributes almost nothing downstream. |
| `magnitude/wedge_abs_max` | Max |wedge| output | Sudden spikes suggest numerical explosion. Check if accompanied by `inf_count > 0`. |
| `magnitude/dot_abs_mean` | Mean |dot| output (SiLU(a*b)) | Baseline reference for gauging wedge's relative importance. |

The key ratio is `wedge_abs_mean / dot_abs_mean`:
- **0.1 – 1.0** — Healthy. Wedge provides meaningful geometric information.
- **< 0.01** — Wedge degenerate. Likely caused by cancellation or poor initialization.
- **\> 10** — Wedge dominant. Possible numerical instability.

### Health

| Metric | Meaning | Expectation |
|--------|---------|-------------|
| `health/nan_count` | Number of NaN values in wedge output | **Must be 0.** Any nonzero = numerical breakdown. |
| `health/inf_count` | Number of Inf values in wedge output | **Must be 0.** Usually bf16 overflow (values exceeding ±3.39e+38). |

Hard pass/fail indicators. If nonzero, check: (1) whether using `naive` wedge mode, (2) whether learning rate is too high.

### Orthogonality

```
cos_sim = cosine_similarity(z_det, z_ctx, dim=channel)   # per spatial position
ortho/cos_sim_mean = mean(|cos_sim|)
```

| Metric | Meaning | How to read |
|--------|---------|-------------|
| `ortho/cos_sim_mean` | Average channel-wise cosine similarity between det and ctx features | 0 = perfectly orthogonal, 1 = perfectly parallel. Lower is better. |

This metric is causally linked to cancellation: high `cos_sim_mean` → det/ctx learn similar features → `a*b ≈ c*d` → severe cancellation → wedge degenerates. The `--ortho-weight` hyperparameter controls the loss term that pushes this value down.

### Triage Flowchart

```
1. health/nan_count > 0 or inf_count > 0?
   ├─ YES → Stop training. Lower lr or switch wedge_mode.
   └─ NO  → continue ↓

2. cancel/rel_diff_lt1e-2 > 50%?
   ├─ YES → Check ortho/cos_sim_mean
   │        ├─ > 0.5 → Increase --ortho-weight
   │        └─ < 0.3 → Cancellation not from alignment; switch to fma mode
   └─ NO  → continue ↓

3. wedge_abs_mean / dot_abs_mean < 0.01?
   ├─ YES → Wedge branch ineffective (init or gradient issue)
   └─ NO  → Numerically stable, wedge working normally
```
