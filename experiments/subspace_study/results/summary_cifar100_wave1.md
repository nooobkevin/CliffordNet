# Parallel Experiment Summary

| Run Name | Track | Model | Status | Best Val Acc@1 | Wall Time (min) | Description |
| --- | --- | --- | --- | ---: | ---: | --- |
| `w1_repro_rolling_hier_s4` | Track0-Repro | `rolling_hier_s4` | ok | **78.26%** | 48.69 | Paper 2 Table 2 baseline: Shift-based Hier-CliffordNet S={1,2,4,8} (GGR block) |
| `w1_repro_subspace_hier_k2` | Track0-Repro | `subspace_hier_k2` | ok | **77.33%** | 32.14 | Paper 2 Table 2: Hier-Subspace-CAN uniform k=2 |
| `w1_repro_subspace_hier_k4` | Track0-Repro | `subspace_hier_k4` | ok | **78.33%** | 32.28 | Paper 2 Table 2 & 4: Hier-Subspace-CAN uniform k=4 (Full: Inner + Wedge) |
| `w1_repro_subspace_hier_244` | Track0-Repro | `subspace_hier_244` | ok | **77.54%** | 31.74 | Paper 2 Table 3 & 4: Hier-Subspace-CAN progressive hybrid k=(2,4,4) |
| `w1_ablate_k4_inner_paper` | Track1-Deconfound | `subspace_hier_k4_inner` | ok | **76.75%** | 31.37 | Paper 2 Table 4 unaligned Inner-only (0.788M, -30% fewer params than Full k=4) |
| `w1_ctrl1_k4_inner_wide_isoparam` | Track1-Deconfound | `subspace_hier_k4_inner_wide` | ok | **76.91%** | 28.28 | Control 1: Strict Iso-Param (~1.12M) Inner-only / SwiGLU control (removes +42% param confound) |
| `w1_ctrl2a_k4_symmetric_isoparam` | Track1-Deconfound | `subspace_hier_k4_sym` | ok | **77.40%** | 33.33 | Control 2a: Strict Iso-Param (1.120M) Symmetric Off-Diagonal (u_i v_j + u_j v_i) vs Wedge |
| `w1_ctrl2b_k4_bilinear_isoparam` | Track1-Deconfound | `subspace_hier_k4_bilinear` | ok | **78.07%** | 31.65 | Control 2b: Strict Iso-Param (1.120M) Raw Asymmetric Bilinear (u_i v_j) without skew-symmetry |
