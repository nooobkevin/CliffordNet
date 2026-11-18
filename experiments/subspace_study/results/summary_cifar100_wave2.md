# Parallel Experiment Summary

| Run Name | Track | Model | Status | Best Val Acc@1 | Wall Time (min) | Description |
| --- | --- | --- | --- | ---: | ---: | --- |
| `w2_ablate_k4_wedge_paper` | Track1&5-Wedge | `subspace_hier_k4_wedge` | ok | **77.84%** | 31.59 | Paper 2 Table 4 Wedge-only k=4 (pure anti-symmetric bivector, zero diagonal energy) |
| `w2_ablate_k4_spfree` | Track0-Ablate | `subspace_hier_k4_spfree` | ok | **77.02%** | 27.41 | Paper 2 Table 4 SpatialDiffusion-free k=4 (tests DWConv5x5 contribution) |
| `w2_ctrl3a_subspace_k4_ggr` | Track1-2x2Block | `subspace_hier_k4_ggr` | ok | **77.88%** | 30.42 | Control 3a (2x2 Factorial): Subspace GP k=4 inside Paper 1 GGR+LayerNorm2d block |
| `w2_ctrl3b_rolling_s3_lietrotter` | Track1-2x2Block | `rolling_hier_s3_lietrotter` | ok | **77.19%** | 46.47 | Control 3b (2x2 Factorial): Rolling S={1,2,4} inside Paper 2 Lie-Trotter+DW5x5+BN block |
| `w2_repro_subspace_244_dpr025` | Track0-Reg | `subspace_hier_244_dpr025` | ok | **78.84%** | 31.92 | Hier-Subspace-CAN (2,4,4) with CAN-aligned higher DropPath=0.25 to test CIFAR overfitting gap |
| `w2_scale_subspace_hier_244_24` | Track0-Scale | `subspace_hier_244_24` | ok | **79.78%** | 50.79 | Paper 2 Table 3 24-layer Hier-Subspace-CAN-24 (4,10,10) k=(2,4,4) |
| `w2_num_k4_bf16_naive` | Track3-Numerical | `subspace_hier_k4` | ok | **77.74%** | 29.69 | Track 3 Numerical: Hier-Subspace-CAN k=4 in bf16-mixed with naive subtraction |
| `w2_num_k4_bf16_direct_z2` | Track3-Numerical | `subspace_hier_k4` | ok | **77.92%** | 29.01 | Track 3 Lambda-Invariance: Hier-Subspace-CAN k=4 in bf16-mixed with direct u^z2 wedge |
