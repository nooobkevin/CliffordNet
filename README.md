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
