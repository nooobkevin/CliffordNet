# Parallel Experiment Summary

| Run Name | Track | Model | Status | Best Val Acc@1 | Wall Time (min) | Description |
| --- | --- | --- | --- | ---: | ---: | --- |
| `w4_v2_k4_dpr020` | Track9-v2-CIFAR100 | `subspace_hier_k4_v2` | ok | **79.21%** | 34.70 | Subspace-CAN-v2 k=4 (ctx_bn + abs + proj_bias, dpr=0.20, 1.128M) vs Wave 3 v1 anchor (78.71%) |
| `w4_v2_244_allwd` | Track9-v2-CIFAR100 | `subspace_hier_244_v2` | ok | **79.58%** | 34.71 | Subspace-CAN-v2 Hybrid (2,4,4) + CAN all-wd (1.100M) vs Wave 3 (79.72%) & Paper (80.25%) |
| `w4_v2_k6_dpr020` | Track9-v2-CIFAR100 | `subspace_hier_k6_v2` | ok | **79.24%** | 43.26 | Subspace-CAN-v2 iso-param k=6 (15 bivectors, 1.121M, dpr=0.20) vs Wave 3 v1 k=6 (79.05%) |
| `w4_v2_k4_inner_wide` | Track9-v2-Deconfound | `subspace_hier_k4_v2_inner_wide` | ok | **79.55%** | 31.81 | Dead-zone-fixed iso-param Inner-wide control (1.138M, dpr=0.20) testing wedge value after fix |
| `w4_v2_k4_sym` | Track9-v2-Deconfound | `subspace_hier_k4_v2_sym` | ok | **78.85%** | 36.86 | Dead-zone-fixed iso-param Symmetric off-diag control (1.128M, dpr=0.20) |
| `w4_v2_k4_ln2d` | Track9-v2-NormInvariance | `subspace_hier_k4_v2_ln2d` | ok | **79.46%** | 65.15 | 100% LayerNorm2d + v2 dead-zone fix (ctx_ln2d + abs + proj_bias, 1.128M, dpr=0.20) |
| `w4_v2_k4_ctxbn_diff` | Track9-v2-Mechanism | `subspace_hier_k4_v2_ctxbn_diff` | ok | **79.12%** | 34.81 | Isolates ctx_bn + diff (balanced discrete gradient z2-z1, 1.128M) vs ctx_bn + abs |
| `w4_v2_k4_gffn_dw7` | Track9-v2-CIFAR100 | `subspace_hier_k4_v2_gffn_dw7` | ok | **79.04%** | 57.82 | Full Subspace-CAN-v2 (ctx_bn + abs + proj_bias + dw7 + gFFN-G, 1.174M, dpr=0.20) |
