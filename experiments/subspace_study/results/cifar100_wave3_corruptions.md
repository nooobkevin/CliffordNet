# Corruption Robustness & Mechanistic Diagnostics Summary

| Run | Model | Params (M) | Clean | Photo Mean | Photo Ret% | Contrast 0.12 | Fog | HighPass | FFT Phase | Wedge/Dot | Subspace |cos| | EffRank |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `w3_hybrid_k4_gffn_global_dpr020` | `subspace_hier_k4_gffn_global` | 1.128 | **79.08%** | 72.05% | 91.1% | 62.11% | 75.58% | 47.22% | 38.70% | 3.233 | 0.943 | 0.704 |
| `w3_isoparam_k2_dpr020` | `subspace_hier_k2_isoparam` | 1.124 | **78.02%** | 69.18% | 88.7% | 52.73% | 72.23% | 43.83% | 31.52% | 4.952 | 0.879 | 0.552 |
| `w3_isoparam_k4_dpr020` | `subspace_hier_k4_dpr020` | 1.120 | **78.71%** | 70.31% | 89.3% | 56.60% | 73.40% | 45.55% | 34.16% | 3.927 | 0.927 | 0.714 |
| `w3_isoparam_k6_dpr020` | `subspace_hier_k6_isoparam` | 1.114 | **79.05%** | 70.77% | 89.5% | 58.04% | 74.04% | 47.66% | 33.12% | 3.094 | 0.944 | 0.767 |
| `w3_isoparam_k8_dpr020` | `subspace_hier_k8_isoparam` | 1.107 | **78.04%** | 69.67% | 89.3% | 56.40% | 72.45% | 48.31% | 34.32% | 2.664 | 0.950 | 0.792 |
| `w3_norm_ln2d_k4_dpr020` | `subspace_hier_k4_ln2d` | 1.120 | **78.90%** | 71.13% | 90.2% | 60.22% | 73.86% | 46.69% | 32.41% | 3.601 | 0.925 | 0.749 |
| `w3_norm_subspace_rms_k4_dpr020` | `subspace_hier_k4_rms` | 1.120 | **77.72%** | 69.63% | 89.6% | 57.16% | 72.79% | 45.10% | 30.93% | 6.784 | 0.666 | 0.758 |
| `w3_recipe_can_allwd_244_dpr025` | `subspace_hier_244_dpr025` | 1.092 | **79.72%** | 70.78% | 88.8% | 52.92% | 74.40% | 49.64% | 34.37% | 1.051 | 0.850 | 0.718 |
