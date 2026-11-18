# Reproduction and ablation notes on CliffordNet (`arXiv:2601.06793`) and Subspace Clifford Networks (`arXiv:2610.37226`)

> **Methodology note:** This reproduction and ablation study was directed by `@nooobkevin` and executed with an AI coding agent (GitHub Copilot CLI) on an 8x NVIDIA H800 node. The agent reconstructed the un-released `arXiv:2610.37226` architecture from the paper equations, wrote the Triton kernels and unit tests, and ran the 22 training jobs. Every script, unit test, run manifest, and per-epoch CSV log is committed under [`experiments/subspace_study/`](../experiments/subspace_study/) so any number below can be audited or re-run from scratch.

## Overview of findings

`arXiv:2610.37226` (*Subspace Clifford Networks*) replaces the cyclic `torch.roll` shifts of CliffordNet (`arXiv:2601.06793`) with grouped $k$-dimensional subspaces ($k \in \{2, 3, 4\}$) inside an operator-splitting block, and reports CIFAR-100 benchmarks in `fp32` on a single RTX 4090 without public source code. We reconstructed the model family to exact parameter parity across all 13 hierarchical configurations in Tables 2 to 4, wrote an open-source Triton kernel with backward recomputation (`fp32` and `bf16`, NCHW and `channels_last`, compatible with `torch.compile(fullgraph=True)`), and ran 16 CIFAR-100 runs (200 epochs each) plus 6 ImageNet-1k runs (10 epochs each in `bf16` DDP, 1.28M train / 50k val).

The experiments support the main architectural claims of `arXiv:2610.37226` while qualifying two points about mixed-precision numerics and contrast invariance:

| Question tested | Result | Summary of evidence |
| --- | --- | --- |
| Does Subspace-CAN improve over shift-rolling CAN at scale? | Supported | On CIFAR-100, `Hier-Subspace-CAN` (`1.09M` to `1.12M`) matches or beats `Rolling-S4` (`2.54M`) with 56% fewer parameters in 1.51x less time, and the 24-layer model (`2.21M`) reaches `79.78%`. On ImageNet-1k (`~3.94M` params, 10 epochs), Subspace-CAN beats Shift-Rolling by `+5.13%` Top-1 (`52.82%` vs. `47.69%`) and cuts per-GPU training VRAM from `40.4 GiB` to `15.1 GiB`. |
| Is Table 4's gain over `Inner-only` caused by having 42% more parameters? | Refuted (gain holds at equal params) | Widening `Inner-only` from `0.788M` to `1.127M` (`Inner-wide`, SwiGLU-style diagonal gating) barely moves CIFAR-100 accuracy (`76.75%` to `76.91%`), leaving a `1.42%` gap behind `Full k=4` (`78.33%`). On ImageNet-1k, `Full` beats equal-parameter `Inner-wide` by `+2.63%` Top-1 (`52.82%` vs. `50.19%`). |
| Does skew-symmetry ($u_i v_j - u_j v_i$) matter compared to symmetric ($u_i v_j + u_j v_i$) or raw ($u_i v_j$) cross-terms? | Supported vs. symmetric; small edge vs. raw bilinear | At equal parameters, symmetric cross-terms trail by `0.93%` on CIFAR-100 (`77.40%`) and drop `4.96%` on ImageNet-1k (`47.86%`, worse than diagonal `Inner-wide`). On CIFAR-100, raw asymmetric products $u_i v_j$ reach `78.07%`, while skew-symmetric wedge reaches `78.33%`, and `0.898M` `Wedge-only` (`77.84%`) beats `1.127M` `Inner-wide` (`76.91%`). |
| Did the accuracy gain come from Subspace GP or from the new `Lie-Trotter + DWConv5x5 + BN` block? | Both contribute | In a 2x2 block swap, Subspace `k=4` inside Paper 1's `GGR` block (`1.52M`) reaches `77.89%`; switching to `Lie-Trotter + DWConv5x5 + BN` (`1.12M`) trims 26% of the parameters and reaches `78.33%`. Dropping `SpatialDiffusion` (`DWConv5x5`) falls to `77.02%`, and putting `Rolling S=3` inside `Lie-Trotter` drops to `77.19%`. |
| How does No-FFN Subspace-CAN compare against a tuned modern FFN backbone on ImageNet-1k? | Competitive at 10 epochs | At 10 epochs (`global_bs=1024`), `in1k_subspace_femto` (`3.93M`, `52.82%` Top-1) leads `timm` `ConvNeXt-Atto` (`3.70M`, 4x MLP FFN) both at matched `lr=1e-3` (`37.92%`) and at ConvNeXt's standard `lr=4e-3` (`49.71%`, `+3.11%` Top-1). |
| Does a fused kernel still beat `torch.compile` on H800? | Supported | `torch.compile` already speeds up eager PyTorch by `5.4x` to `6.1x`, which narrows raw eager comparisons. Even so, the recomputation Triton kernel is `2.40x` to `4.63x` faster than `torch.compile` in `fp32` and `5.44x` to `9.57x` faster in `bf16` (`2,994 GB/s` HBM3), while saving `0 MiB` of intermediate activations. |
| Does `bf16` preserve the algebraic identity $u \wedge (z_2 - \lambda u) = u \wedge z_2$? | No in naive `bf16`; fixed by `fp32` FMA | Rounding $z_2 - u$ in `bf16` before taking the determinant creates a `345.5%` mean relative difference against $u \wedge z_2$ on ImageNet-1k and costs `0.59%` on CIFAR-100 (`77.74%` vs. `78.33%`). Computing the determinant in `fp32` registers with FMA avoids the loss (`Cancel<1e-2 = 1.88%`). |
| Does the wedge branch give contrast-invariant edge representations? | No for contrast scaling; yes for Fourier phase | Because $\|\alpha u \wedge \alpha v\| = \alpha^2 \|u \wedge v\|$, `Wedge-only` and `Inner-wide` degrade at the same rate under `0.12x` low contrast and fog (`89.5%` vs. `89.3%` relative retention on CIFAR-100). When spatial amplitude is equalized (`FFT phase-only`), `Wedge-only` (`0.898M`) scores highest among all 12-layer CIFAR models (`34.68%`, `+4.99%` over `Rolling-S4`). |

## 1. Architectural reconstruction (`arXiv:2610.37226`)

Solving the parameter-count equations across Tables 2, 3, and 4 pins down the exact layer specification: `GeometricStem(embed_dim=32, patch_size=1)` followed by `stem_proj(32 -> 96)`, `SpatialDiffusion` (`DWConv5x5(bias=False) + BN + SiLU + LayerScale`, `28*D` parameters), and `SubspaceGeometricProduct` (`BN + Conv1x1(D -> 2*D_tilde, bias=False) + DWConv3x3(bias=False) + SubspaceGP + Conv1x1(C_out -> D, bias=False) + LayerScale`). Every hierarchical model matches the paper's reported count within rounding (`<= 0.0006M`), and all eight pairwise deltas match to the exact integer:

| Preset | Exact params | Ours (M) | Paper (M) | Pairwise delta check |
| --- | ---: | ---: | ---: | --- |
| `rolling_hier_s3` | 2,092,500 | 2.093M | 2.093M | Exact |
| `rolling_hier_s4` | 2,534,868 | 2.535M | 2.535M | Exact |
| `subspace_hier_k2` | 898,388 | 0.898M | 0.898M | `u4 - u2 = 221,184` (0 error) |
| `subspace_hier_k3` | 1,017,554 | 1.018M | 1.017M | `u4 - u3 = 102,018` (0 error, $\tilde{D}_2=129, \tilde{D}_3=162$) |
| `subspace_hier_k4` | 1,119,572 | 1.120M | 1.120M | Reference |
| `subspace_hier_234` | 1,061,240 | 1.061M | 1.061M | Exact |
| `subspace_hier_432` | 960,888 | 0.961M | 0.961M | Exact |
| `subspace_hier_244` | 1,091,924 | 1.092M | 1.092M | Reference 12L |
| `subspace_hier_244_18` | 1,650,516 | 1.651M | 1.650M | `18L - 12L = +558,592` (0 error) |
| `subspace_hier_244_24` | 2,209,108 | 2.209M | 2.209M | `24L - 18L = +558,592` (0 error) |
| `subspace_hier_k4_inner` | 787,796 | 0.788M | 0.788M | `u4 - inner = 331,776` (0 error) |
| `subspace_hier_k4_wedge` | 898,388 | 0.898M | 0.898M | `u4 - wedge = 221,184` (0 error) |
| `subspace_hier_k4_spfree` | 1,074,772 | 1.075M | 1.075M | `u4 - spfree = 44,800` (0 error) |

## 2. H800 operator benchmark: eager PyTorch vs. `torch.compile` vs. Triton

Section 4.4 of `arXiv:2610.37226` benchmarks its fused CUDA kernel against eager PyTorch on RTX 4090 in `fp32`. On an NVIDIA H800 (`sm_90`, CUDA 12.8, PyTorch 2.10, Triton 3.6), we added `torch.compile` (`default` and `max-autotune`) across both NCHW and `channels_last` (NHWC) layouts at `(B, D, H, W) = (64, 128, 64, 64)`, `k=4`, `full`, `diff` (forward + backward averaged over 100 timed iterations after 25 warmup iterations):

| Precision and layout | PyTorch eager | `torch.compile` (Inductor) | Triton recompute (ours) | vs. eager | vs. `compile` | Dynamic extra HBM (eager / compile / Triton) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fp32` NCHW (paper setting) | 7.125 ms | 1.184 ms | **0.493 ms** (`2,994 GB/s`) | 14.45x | **2.40x** | `896` / `704` / **320 MiB** |
| `fp32` NHWC (`channels_last`) | 14.377 ms | 2.663 ms | **0.575 ms** (`2,566 GB/s`) | 24.99x | **4.63x** | `1536` / `960` / **320 MiB** |
| `bf16` NCHW | 7.892 ms | 1.374 ms | **0.253 ms** (`2,916 GB/s`) | 31.23x | **5.44x** | `1312` / `928` / **160 MiB** |
| `bf16` NHWC (`channels_last`) | 14.884 ms | 2.430 ms | **0.254 ms** (`2,909 GB/s`) | 58.62x | **9.57x** | `1632` / `928` / **160 MiB** |

Inductor fuses elementwise ops in the forward pass, but AOTAutograd still writes intermediate tensors to HBM for backward instead of recomputing `(z1, z2)` inside registers across the 10 per-group outputs. Recomputing `(z1, z2)` in Triton keeps peak extra allocation equal to the forward output tensor alone (`320 MiB` in `fp32`, `160 MiB` in `bf16`). Full stage-by-stage tables for `k in {2, 3, 4}` and ImageNet-1k feature-map sizes (`1.68x` to `13.49x` faster than `torch.compile`) are in [`../experiments/subspace_study/results/kernel_benchmark_h800.md`](../experiments/subspace_study/results/kernel_benchmark_h800.md).

![H800 Triton Kernel Benchmark](../experiments/subspace_study/figures/h800_triton_kernel_benchmark.png)

## 3. CIFAR-100 200-epoch ablations, controls, and corruption tests (16 runs)

All 16 CIFAR-100 runs trained from scratch for 200 epochs with `batch_size=128`, `AdamW(lr=1e-3, weight_decay=0.1)`, cosine LR decay, `AutoAugment(CIFAR10)`, `RandomErasing(0.25)`, and no Mixup or EMA.

| Run | Configuration | Params | Wall time | Clean Top-1 | Photo mean (ret%) | Contrast 0.12 | High-pass edges | FFT phase | EffRank |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `w1_repro_rolling_hier_s4` | Shift-Rolling `S={1,2,4,8}` (GGR) | 2.535M | 48.69m | 78.26% | 70.35% (89.9%) | 58.15% | 45.87% | 29.69% | 0.800 |
| `w1_repro_subspace_hier_k2` | Subspace uniform `k=2` | 0.898M | 32.14m | 77.33% | 69.10% (89.4%) | 56.50% | 42.59% | 32.10% | 0.561 |
| `w1_repro_subspace_hier_k4` | Subspace uniform `k=4` (Full) | 1.120M | 32.28m | **78.33%** | 70.17% (89.6%) | 57.97% | 45.19% | 32.49% | 0.705 |
| `w1_repro_subspace_hier_244` | Subspace hybrid `(2,4,4)` (`dpr=0.10`) | 1.092M | 31.74m | 77.55% | 69.15% (89.2%) | 54.46% | 45.42% | 32.34% | 0.687 |
| `w2_repro_subspace_244_dpr025` | Subspace hybrid `(2,4,4)` (`dpr=0.25`) | 1.092M | 31.92m | **78.84%** | 69.93% (88.8%) | 54.98% | **49.05%** | 32.60% | 0.702 |
| `w2_scale_subspace_hier_244_24` | 24-layer hybrid `(2,4,4)` (`dpr=0.20`) | 2.209M | 50.79m | **79.78%** | **72.34%** (90.7%) | **60.98%** | 46.25% | **36.49%** | 0.691 |
| `w1_ablate_k4_inner_paper` | Paper Table 4 `Inner-only` | 0.788M | 31.37m | 76.75% | 68.28% (89.0%) | 52.64% | 42.84% | 31.03% | 0.632 |
| `w1_ctrl1_k4_inner_wide_isoparam` | Control 1: equal-param `Inner-wide` (SwiGLU) | 1.127M | 28.28m | 76.91% | 68.69% (89.3%) | 54.24% | 43.35% | 31.94% | 0.686 |
| `w1_ctrl2a_k4_symmetric_isoparam` | Control 2a: equal-param symmetric ($u_i v_j + u_j v_i$) | 1.120M | 33.33m | 77.40% | 69.08% (89.3%) | 55.97% | 45.53% | 32.87% | 0.687 |
| `w1_ctrl2b_k4_bilinear_isoparam` | Control 2b: equal-param bilinear ($u_i v_j$) | 1.120M | 31.65m | 78.07% | 69.98% (89.7%) | 56.66% | 46.07% | 33.31% | 0.692 |
| `w2_ablate_k4_wedge_paper` | `Wedge-only` ($u \wedge v$ only, no diagonal) | 0.898M | 31.59m | 77.84% | 69.63% (89.5%) | 55.31% | 44.19% | **34.68%** | 0.663 |
| `w2_ablate_k4_spfree` | `SpatialDiffusion-free` `k=4` | 1.075M | 27.41m | 77.02% | 67.68% (87.9%) | 49.78% | 43.90% | 30.35% | 0.708 |
| `w2_ctrl3a_subspace_k4_ggr` | Control 3a: Subspace `k=4` in Paper 1 `GGR` | 1.519M | 30.42m | 77.89% | 69.79% (89.6%) | 57.75% | 43.78% | 29.45% | 0.696 |
| `w2_ctrl3b_rolling_s3_lietrotter` | Control 3b: Rolling `S=3` in `Lie-Trotter` | 1.693M | 46.47m | 77.19% | 69.25% (89.7%) | 56.80% | 48.96% | 34.30% | 0.772 |
| `w2_num_k4_bf16_direct_z2` | Numerical: `bf16-mixed` + direct $u \wedge z_2$ | 1.120M | 29.01m | 77.92% | 69.51% (89.3%) | 55.79% | 45.74% | 32.51% | 0.707 |
| `w2_num_k4_bf16_naive` | Numerical: `bf16-mixed` + naive $u \wedge (z_2 - u)$ | 1.120M | 29.69m | 77.74% | 69.99% (90.1%) | 57.55% | 45.72% | 33.01% | 0.706 |

 Note on absolute CIFAR-100 baseline offset: across both shift-rolling and subspace models, our from-scratch 200-epoch numbers sit roughly `1.2%` to `1.4%` below the paper's CIFAR-100 table (`78.26%` vs. `79.53%` on `Rolling-S4`; `78.84%` at `dpr=0.25` vs. `80.25%` on `Hybrid (2,4,4)`), while preserving the paper's relative ordering and deltas (`24L - 12L = +1.52%` vs. `+1.68%` in the paper; `SpatialDiffusion` removal `= -1.31%` vs. `-1.17%`). Raising `drop_path_rate` from `0.10` to `0.25` closed `1.29%` of that gap on `Hybrid (2,4,4)` (`77.55%` to `78.84%`), suggesting regularization strength (`drop_path_rate`, warmup, or `eta_min`) accounts for most of the offset.

Per-epoch CSV trajectories are in [`../experiments/subspace_study/logs/cifar100/`](../experiments/subspace_study/logs/cifar100/) and full 12-corruption tables are in [`../experiments/subspace_study/results/cifar100_corruptions_and_diagnostics.md`](../experiments/subspace_study/results/cifar100_corruptions_and_diagnostics.md).

![CIFAR-100 Deconfounding & Fourier Phase Benchmark](../experiments/subspace_study/figures/cifar100_deconfounding_matrix.png)

## 4. ImageNet-1k 6-way scaling and control runs (`bf16-mixed`, 10 epochs)

All six runs used 2x H800 DDP (`global_batch_size=1024`, `512` per GPU), `Mixup(0.8)`, `CutMix(1.0)`, `label_smoothing=0.1`, `EMA(0.999)`, 2 warmup epochs, cosine LR decay, and `seed=42` on the full 1.28M training set and 50k validation set:

| Model and control role | Params | Peak VRAM / GPU | Val Top-1 | Val Top-5 | Contrast 0.12 | Fog | FFT phase | `bf16` Cancel<1e-2 | `bf16` $\lambda$-err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `in1k_subspace_femto` (Subspace Full, `lr=1e-3`) | 3.930M | **15.1 GiB** | **52.82%** | **77.07%** | 24.09% | **49.58%** | 19.48% | 1.88% | 3.455 |
| `in1k_subspace_femto_inner_wide` (equal-param SwiGLU/Inner, `lr=1e-3`) | 3.942M | 16.1 GiB | 50.19% | 74.98% | 19.79% | 46.85% | 17.27% | 0.00% | 2.350 |
| `baseline_convnext_atto` (`timm` 4x MLP FFN, standard `lr=4e-3`) | 3.696M | 11.1 GiB | 49.71% | 74.62% | **24.78%** | 47.50% | **22.82%** | - | - |
| `in1k_subspace_femto_sym` (equal-param symmetric $u_i v_j + u_j v_i$, `lr=1e-3`) | 3.930M | 18.4 GiB | 47.86% | 72.49% | 18.76% | 44.08% | 13.83% | 0.30% | 2.767 |
| `in1k_rolling_femto` (Shift-Rolling `S={1,2,4}`, GGR, `lr=1e-3`) | 3.946M | 40.4 GiB | 47.69% | 72.80% | 22.25% | 44.41% | 14.17% | 1.99% | 1.425 |
| `baseline_convnext_atto` (`timm` 4x MLP FFN, matched `lr=1e-3`) | 3.696M | 11.1 GiB | 37.92% | 63.20% | 10.00% | 34.18% | 12.29% | - | - |

Per-epoch CSV curves and `bf16` diagnostic logs are in [`../experiments/subspace_study/logs/imagenet1k/`](../experiments/subspace_study/logs/imagenet1k/), with reproduction commands in [`../experiments/subspace_study/README.md`](../experiments/subspace_study/README.md).

![ImageNet-1k 6-Way Trajectory & VRAM](../experiments/subspace_study/figures/imagenet1k_6way_curves.png)

## 5. Practical takeaways for `arXiv:2610.37226` and follow-up work

1. **Equal-parameter controls strengthen Table 4.** Readers looking at Table 4 will notice that `Uniform k=4` (`1.120M`) has 42% more parameters than `Inner-only` (`0.788M`). Adding equal-parameter `Inner-wide` and `Symmetric` baselines removes that objection and shows an even larger `2.63%` to `4.96%` advantage for skew-symmetric wedge interactions on ImageNet-1k.
2. **`bf16` training needs `fp32` FMA determinants.** Computing $v = z_2 - z_1$ in `bf16` before evaluating $u \wedge v$ breaks $u \wedge (z_2 - u) = u \wedge z_2$ by 345% relative error on ImageNet-1k and drops CIFAR-100 accuracy by `0.59%`. Evaluating the determinant in `fp32` registers with FMA inside the `bf16` Triton kernel removes the precision loss at `0.254 ms`.
3. **`BatchNorm2d` trades contrast robustness for clean optimization speed.** Because both $u \odot v$ and $u \wedge v$ scale as $\alpha^2$ when input contrast scales by $\alpha$, `BatchNorm2d` (which uses frozen dataset running stats at eval time) leaves Subspace-CAN slightly more sensitive to severe low contrast (`0.12x`) on ImageNet-1k than per-token `LayerNorm` in `ConvNeXt-Atto` (`24.09%` vs. `24.78%`), despite Subspace-CAN's `+3.11%` higher clean accuracy. Swapping `pre_bn` for `LayerNorm2d` or `RMSNorm2d` is worth testing when domain shift matters.
4. **300-epoch ImageNet-1k run is the natural next check.** Subspace-CAN leads tuned `ConvNeXt-Atto` (`lr=4e-3`) by `+3.11%` at 10 epochs (`12.8M` training images seen). Running that exact pair to 300 epochs will test whether the early sample-efficiency lead holds at full convergence.

