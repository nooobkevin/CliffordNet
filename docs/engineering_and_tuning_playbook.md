# Model development and GPU performance tuning playbook

This playbook distills the practical engineering methods, numerical debugging techniques, and H800 systems optimizations we developed and validated across 58 full training runs (32 CIFAR-100 + 26 ImageNet-1k) during the CliffordNet (`arXiv:2601.06793`) and Subspace Clifford Networks (`arXiv:2610.37226`) study. Every rule below comes from a failure mode or bottleneck we measured directly on 8x NVIDIA H800 GPUs.

---

## 1. Model reconstruction and scientific debugging

### 1.1 Reverse-engineer parameter counts to zero error before training
When reproducing a paper without public code, do not guess bias flags, stem dimensions, or channel-alignment rules. Set up the linear parameter-count deltas across the paper's ablation tables (for example, `k=4` vs. `k=2`, `full` vs. `inner`, with vs. without `SpatialDiffusion`, depth `12 -> 18 -> 24`) and solve for exact integer parity first. In `arXiv:2610.37226`, matching all eight pairwise deltas to 0 integer error immediately pinned down bias-free $W_{\text{in}} / W_{\text{out}} / \text{DWConv}_{3\times 3}$, `SpatialDiffusion` = $28D$ params, and the `GeometricStem(32, p=1) + Conv1x1(32 -> 96)` CIFAR stem. Encode those exact counts in a fast unit test (`tests/test_subspace_models.py`).

### 1.2 Always pair ablations with equal-parameter controls
Dropping a branch (such as removing the wedge product in Table 4) often cuts 30% to 45% of parameters and FLOPs, confounding algebraic structure with capacity. Add matched controls at identical parameter and FLOP budgets:
- `inner_wide`: widen the diagonal branch to match `full` parameter count.
- `sym_full` / `bilinear_full`: replace antisymmetric $u_i v_j - u_j v_i$ with symmetric $u_i v_j + u_j v_i$ or raw $u_i v_j$ at identical channel counts.
- 2x2 macro-block swaps: separate the core interaction operator (`Subspace` vs. `Rolling`) from the surrounding block (`Paper 1 GGR + LN2d` vs. `Paper 2 Lie-Trotter + DWConv5x5 + BN`).

### 1.3 Probe activations and dead-zones before blaming capacity or recipes
When a custom architecture trails a baseline (for example, v1 `Subspace-CAN` losing to `ConvNeXt-Atto` at 25 epochs on ImageNet-1k), run a read-only checkpoint probe over a few validation batches before changing hyperparameters. Log:
- **Branch balance (`|Wedge| / |Dot|`)**: in v1, Stage 3 was `18.27x`, showing the inner branch supplied only `5.4%` of $W_{\text{out}}$'s input.
- **Activation dead-zone fraction**: measuring the fraction of inputs below `-1.28` (where $\text{SiLU}'(p) < 0$ inverts gradients) exposed that unnormalized `DWConv3x3` + `diff` mode ($V = Z_2 - Z_1 \approx -Z_1$) trapped `50.3%` to `55.4%` of Stage 2 and 3 inner channels in saturation. Fixing it (`ctx_bn + ctx_mode="abs" + proj_bias`) dropped dead-zone saturation to `9.8%` to `11.7%` and added `+2.05%` ImageNet-1k Top-1 in isolation.
- **Penultimate SVD effective rank**: ruled out feature collapse (`87.0%` vs. `ConvNeXt-Atto`'s `86.7%`) in under 10 seconds, preventing a wasted detour.

### 1.4 Guard bilinear and quadratic operators in `bf16`
- **Catastrophic cancellation in wedge determinants**: evaluating $u \wedge (z_2 - u)$ naively in `bf16` diverges by `345%` relative error from $u \wedge z_2$ due to 7-bit mantissa cancellation. Always accumulate 2x2 determinants $a b - c d$ in `fp32` registers using Kahan/FMA (`tl.fma` in Triton, `torch.addcmul` in PyTorch).
- **Bounded vs. unbounded cross-products**: replacing $\text{SiLU}(u \odot v)$ with SwiGLU $\text{SiLU}(u) \odot v$ caused `bf16` overflow (`val/loss = NaN`) at `lr=3e-3` because $\text{SiLU}(u) \odot v$ is unbounded on both sides, whereas $\text{SiLU}(u \odot v) \ge -0.2785$ naturally bounds negative products from below.
- **Quadratic operators x `BatchNorm2d` scaling instability**: at >=8M parameters (`Pico` 8.5M, `Nano` 14.7M), `BatchNorm2d` does not bound per-token outliers. Outlier tokens square through $u \wedge v$ to `|wedge| = 3,466 to 18,700`, corrupting `BatchNorm` running variance and producing `val/loss = NaN` on `38%` to `42%` of validation epochs. Replacing all `BatchNorm2d` layers with per-token `LayerNorm2d` (`in1k_v2_atto_ln2d`) clamped peak `|wedge|` 80x to 400x down to `43.8` with `0` NaNs and `0` training loss spikes.

### 1.5 Check optimizer weight-decay exclusions when matching paper baselines
Many reference repos pass `model.parameters()` directly to `AdamW(weight_decay=0.1)` (decaying `BatchNorm` and `LayerScale`), whereas modern frameworks exclude 1D norm/bias parameters by default. On CIFAR-100, matching `CAN/train.py`'s uniform weight decay (`optim.decay_norm_and_bias=true`) rebalanced `|Wedge|/|Dot|` from `3.98x` to `1.05x` and closed the baseline gap (`79.72%` vs. paper `80.25%`).

---

## 2. Custom Triton kernels and `torch.compile` tricks

### 2.1 Fuse elementwise geometric products with recomputation backward
For memory-bound operators like `SubspaceGeometricProduct` (2 inputs `z1, z2` expanding to up to `2.5x` output channels), saving intermediate tensors for autograd wastes VRAM and HBM bandwidth. Writing a fused Triton forward + analytical recomputation backward kernel (`src/cliffordnet/kernels/triton_subspace.py`) that saves only `(z1, z2)` cuts extra saved activations to **0 MiB** and achieves **2,994 GB/s** on H800 (`2.40x` to `9.57x` faster than `torch.compile` on `bf16`).

### 2.2 Register custom Triton kernels via `torch.library.custom_op`
Wrapping a Triton kernel directly in `torch.autograd.Function` causes graph breaks under `torch.compile`. Instead, register forward and backward via `@torch.library.custom_op`, provide `@register_fake` shape/stride meta functions, and bind autograd via `register_autograd`. Verify zero graph breaks with `torch._dynamo.explain(block)(x).graph_break_count == 0` and `torch.compile(block, fullgraph=True)` in unit tests.

### 2.3 Never put `B * G` (or any batch-scaled dimension) in CUDA `gridY` / `gridZ`
CUDA limits `gridY` and `gridZ` to **65,535**, while `gridX` allows up to **$2^{31} - 1$**. Our NCHW and strided Triton kernels initially placed `n_bg = B * G` in `grid_1` (`gridY`). That worked silently on `Atto` (`512 * 110 = 56,320 < 65,535`) and crashed with `Triton Error [CUDA]: invalid argument` only when scaling to `Pico` (`81,920`) and `Nano` (`106,496`). Always place `n_bg` in `grid_0` (`gridX`) and `cdiv(HW, BLOCK_HW)` in `grid_1` (`gridY`), and add a large-`B * G` regression test (`test_large_bg_exceeding_grid_y_limit`).

### 2.4 Watch for `.chunk(2, dim=1)` breaking `channels_last` contiguity
In PyTorch, `z1, z2_prime = conv1x1(h).chunk(2, dim=1)` on a `channels_last` (NHWC) tensor produces non-contiguous strided views (`stride(0) = 2 * C * H * W`). When passed into a custom Triton op that checks `_is_dense_nhwc(z1)`, it silently falls back to a slow uncoalesced strided kernel and forces `aten::copy_` in neighboring ops. Profiling `in1k_v2_pico_k4` revealed this single detail cost **25.8% of total CUDA time** (`11.86%` strided bwd + `3.39%` strided fwd + `13.90%` `aten::copy_`). Replacing it with two separate dense `1x1` convs (`split_in_proj=True`, identical params and FLOPs) plus zero-copy NHWC `F.rms_norm` and 1D `flat_gamma=True` cut `subspace_gp_bwd` by **`7.47x`** (`27.70 ms -> 3.71 ms`), cut `aten::copy_` by **`5.37x`** (`32.45 ms -> 6.05 ms`), and sped up end-to-end `Pico` (`bs=256, compile=True`) training by **`1.42x`** (`80.96 ms -> 57.18 ms / step`).

### 2.5 PyTorch 2.10 Inductor + DDP `(1, C, 1, 1)` LayerScale workaround
Converting `(1, C, 1, 1)` `LayerScale` parameters to `channels_last` under PyTorch 2.10 `torch.compile` + DDP can fail in `CompiledFunctionBackward` with a shape mismatch (`got [C] expected [1, C, 1, 1]`). Storing `gamma` and `global_beta` as **1D `(C,)`** (`flat_gamma=True`) and broadcasting via `.view(1, -1, 1, 1)` permanently fixes the bug and enables full `torch.compile=true` + DDP for `RMSNorm2d` and `LayerNorm2d` models (for legacy 4D checkpoints, force `(1, C, 1, 1)` params back to `.contiguous()`).

---

## 3. H800 profiling and data pipeline optimization

### 3.1 Diagnose bottlenecks in 5 seconds with `nvidia-smi dmon --gpm-metrics`
Run `nvidia-smi dmon --gpm-metrics 5,10,12,13,21,25` to read hardware counters directly:
- **CPU / DataLoader bound**: `sm` oscillates between `0%` and `100%`, `pwr` drops periodically below `200W`.
- **HBM bandwidth / latency bound** (typical for depthwise-conv vision backbones like `Subspace-CAN` and `ConvNeXt` on H800): `sm = 86% to 98%`, `pwr = 450 to 525W`, `pcirx = 1.6 to 2.2 GB/s` (<4% of PCIe Gen5), `dram = 32% to 39%` (~1.2 TB/s sustained), and `mmaact = 3% to 4%` (Tensor Cores finish `1x1 Conv` in milliseconds while depthwise convolutions and norms dominate HBM traffic).

### 3.2 Profile operator time and launch bubbles without Nsight
Use `scripts/profile_bottleneck.py` (`torch.profiler` over 3 active steps) to print:
- Top-20 CUDA ops by `Self CUDA time`.
- Exact launch/sync/idle gap: `Wall step time - Sum(Self CUDA time)`. On `in1k_v2_pico_k4` (`bs=256, compile=True`), the gap was only **`3.13 ms / 80.96 ms` (`3.9%`)**, while `convolution_backward` + `cudnn_convolution` took `37.7%` and strided `SubspaceGP` + `aten::copy_` took `25.8%`.
- **When Hopper Programmatic Dependent Launch (PDL) does and does not help**: PDL hides `3 to 5 us` kernel-to-kernel launch/tail bubbles, which matters for sub-20 us LLM decode kernels. For `bs=256` vision training where kernels run `200 to 1,500 us` and total launch gap is `3.9%`, PDL yields `<2%` gain; eliminating HBM round-trips via kernel fusion or fixing strided fallbacks yields an order of magnitude more speedup.

### 3.3 ImageNet-1k data pipeline progression (measured on 8x H800)

| Pipeline stage | What changed | Measured speed & GPU behavior | Accuracy / methodology trade-off |
| --- | --- | --- | --- |
| 1. Default PIL + `float32` CPU transforms | Standard `ToTensor + ColorJitter + Normalize + RandomErasing` on CPU | `188.8 min` / 25 ep (2x H800); `0% <-> 100%` sawtooth GPU util | Reference baseline |
| 2. Compact `uint8` IPC + GPU-vectorized aug | CPU stops at `PILToTensor()` (`uint8`, 4x smaller IPC/H2D); GPU runs batched `ColorJitter + Normalize + RandomErasing` (`_gpu_preprocess_uint8`) | **`111.2 min` / 25 ep (`1.70x` speedup)** on 2x H800 | **Lossless**: exact same PIL `RandomResizedCrop` + `RandAugment(2, 9)` |
| 3. `/dev/shm` zero-JPEG-decode RAM cache + GPU `grid_sample` | One-time 128-core decode (`2m34s`) of 1.28M train (`234.6 GiB`, `256x256 uint8`) + 50k val (`7.0 GiB`) into `/dev/shm` read-only `np.memmap`; GPU fuses `RandomResizedCrop + HFlip + Rotate` via `F.affine_grid/grid_sample` | **`sm=86% to 98%`, `~500W/GPU`**, `131s/ep` Pico & `162s/ep` Nano (2x 4-GPU parallel); PCIe RX only `1.7 GB/s` | **Caveat**: squashing to square `256x256` + `bilinear` `grid_sample` without `RandAugment` caused a `-0.9% to -1.0%` drop when re-evaluated on aspect-ratio-preserving `BICUBIC` HF validation |
| 4. 8-GPU All-in-VRAM HBM sharding (`CLIFFORDNET_VRAM_CACHE=1`) | Each rank loads its i.i.d. 1/8 shard (`29.3 GiB`) into H800 HBM once (`0` CPU workers, `0` PCIe H2D) | **`~405W/GPU` (~20% lower power/throughput than Stage 3)** when keeping Global BS=1024 (`bs=128/gpu`) | Halving per-GPU batch size (`256 -> 128`) reduced arithmetic intensity and doubled DDP AllReduce ratio; only wins at larger global batch sizes (`>=2048`) |

**Takeaway**: Stage 2 (`uint8` transfer + GPU `ColorJitter/Normalize/RandomErasing`) is the simplest lossless default. When using Stage 3 (`/dev/shm` RAM cache), use our updated aspect-ratio-preserving `320x320` padded canvas (`scripts/build_in1k_ram_cache.py --canvas 320`, `366.5 GiB` + `train_shapes.npy` + `BICUBIC` val) so GPU `F.grid_sample` crops strictly inside each image's true valid `(h_i, w_i)` box: validated in Wave F to give **0.02% parity** with original HuggingFace `BICUBIC` 50k validation (`75.45%` vs. `75.43%`).

---

## 4. Multi-GPU orchestration and experiment hygiene

1. **Strip `SLURM_*` environment variables before local `torchrun`.** Inside an interactive Slurm allocation (`SLURM_NTASKS_PER_NODE=1`), Lightning's SLURM auto-detector rejects `devices=2` or `devices=4`. Unset `SLURM_*` variables (`env -u SLURM_NTASKS ...`) before invoking `torchrun --standalone --nproc_per_node=N` when packing multiple independent DDP jobs onto one 8-GPU node.
2. **Always verify zero orphaned GPU child PIDs after stopping background `torchrun` jobs.** Terminating a parent subshell can leave `torchrun` worker processes alive on GPUs (reparented to PID 1). Always check `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader` and confirm `0 MiB` before relaunching on the same GPUs.
3. **Clean up `/dev/shm` caches when done.** Large `tmpfs` caches (`~242 GiB`) consume host RAM until explicitly deleted (`rm -rf /dev/shm/in1k_raw_cache`), which matters on shared multi-user nodes.
4. **Live tracking + offline resilience.** Running W&B in `WANDB_MODE=offline` while enabling `swanlab.sync_wandb()` gives real-time curve monitoring (which immediately caught the Wave D `val/loss = NaN` spikes on `Pico`/`Nano`) while keeping complete local `.wandb` datastores for post-hoc CSV extraction and spike auditing.
