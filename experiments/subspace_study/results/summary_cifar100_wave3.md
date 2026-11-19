# Wave 3 CIFAR-100 Summary (Higher-k Equal-Param Pareto, Norm Invariance, Global Context & Recipe)

| Run Name | Track | Model | Status | Best Val Acc@1 | Wall Time (min) | Description |
| --- | --- | --- | --- | ---: | ---: | --- |
| `w3_isoparam_k2_dpr020` | Track6-HigherK-Pareto | `subspace_hier_k2_isoparam` | ok | **78.02%** | 33.10 | Equal-param (~1.124M) 2D Subspace k=2 (1 bivector/group, stage_dims=(108,144,180), dpr=0.20) |
| `w3_isoparam_k4_dpr020` | Track6-HigherK-Pareto | `subspace_hier_k4_dpr020` | ok | **78.71%** | 32.38 | Equal-param (1.120M) 4D Subspace k=4 anchor (6 bivectors/group, stage_dims=(96,128,160), dpr=0.20) |
| `w3_isoparam_k6_dpr020` | Track6-HigherK-Pareto | `subspace_hier_k6_isoparam` | ok | **79.05%** | 45.49 | Equal-param (~1.114M) 6D Subspace k=6 (15 bivectors/group, stage_dims=(84,120,144), dpr=0.20) |
| `w3_isoparam_k8_dpr020` | Track6-HigherK-Pareto | `subspace_hier_k8_isoparam` | ok | **78.04%** | 47.89 | Equal-param (~1.107M) 8D Subspace k=8 (28 bivectors/group, stage_dims=(80,104,136), dpr=0.20) |
| `w3_norm_ln2d_k4_dpr020` | Track7-NormInvariance | `subspace_hier_k4_ln2d` | ok | **78.90%** | 45.20 | Per-token LayerNorm2d replacing BatchNorm2d in Subspace-CAN k=4 (1.120M, tests contrast shift fix) |
| `w3_norm_subspace_rms_k4_dpr020` | Track7-NormInvariance | `subspace_hier_k4_rms` | ok | **77.72%** | 30.81 | Per-k-subspace unit-RMS sphere normalization before SubspaceGP (1.120M, pure Grassmann angle sine) |
| `w3_hybrid_k4_gffn_global_dpr020` | Track8-GlobalContext | `subspace_hier_k4_gffn_global` | ok | **79.08%** | 34.50 | Local + Global Mean-Field Superposition (+Subspace-gFFN-G, 1.128M, unifies Paper 1 & Paper 2) |
| `w3_recipe_can_allwd_244_dpr025` | Track0-RecipeGap | `subspace_hier_244_dpr025` | ok | **79.70%** | 31.95 | CAN/train.py exact uniform weight_decay=0.1 on all params including BN/LayerScale on Hybrid (2,4,4) |
