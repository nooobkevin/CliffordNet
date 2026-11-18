# ImageNet-1k (10 Epochs, Global BS=1024, bf16-mixed) 6-Way Complete Benchmark

| Run | Architecture & Config | Params | Val Top-1 | Val Top-5 | Contrast 0.12 | Fog | HighPass | FFT Phase | bf16 Cancel<1e-2 | bf16 LambdaErr | EffRank |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `in1k_subspace_femto_10ep` | Subspace-CAN Full (Inner+Wedge, lr=1e-3) | 3.930M | **52.82%** | 77.07% | 24.09% | 49.58% | 3.84% | 19.48% | 0.0188 | 3.4552 | 0.616 |
| `in1k_subspace_femto_sym_10ep` | Subspace-CAN Symmetric Control (u_i v_j + u_j v_i, lr=1e-3) | 3.930M | **47.86%** | 72.49% | 18.76% | 44.08% | 3.65% | 13.83% | 0.0030 | 2.7670 | 0.602 |
| `in1k_subspace_femto_inner_wide_10ep` | Subspace-CAN Iso-Param Inner/SwiGLU Control (lr=1e-3) | 3.942M | **50.19%** | 74.98% | 19.79% | 46.85% | 4.32% | 17.27% | 0.0000 | 2.3503 | 0.605 |
| `in1k_rolling_femto_10ep_eager` | Shift-Rolling Hier-CliffordNet (S={1,2,4}, GGR, lr=1e-3) | 3.946M | **47.69%** | 72.80% | 22.25% | 44.41% | 4.17% | 14.17% | 0.0199 | 1.4248 | 0.755 |
| `baseline_convnext_atto_10ep` | ConvNeXt-Atto (4x MLP FFN, lr=1e-3) | 3.696M | **37.92%** | 63.20% | 10.00% | 34.18% | 2.34% | 12.29% | 0.0000 | 0.0000 | 0.000 |
| `baseline_convnext_atto_lr4e3_10ep` | ConvNeXt-Atto (4x MLP FFN, canonical lr=4e-3) | 3.696M | **49.71%** | 74.62% | 24.78% | 47.50% | 4.94% | 22.82% | 0.0000 | 0.0000 | 0.000 |
