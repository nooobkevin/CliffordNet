# ImageNet-1k (25 Epochs, Global BS=1024, bf16-mixed) Medium-Horizon & Improvement Study

| Run | Model & Config | Params | Val Top-1 | Val Top-5 | Contrast 0.12 | Fog | HighPass | FFT Phase | Time (2xH800) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `in1k_subspace_femto_25ep` | Subspace-CAN Full (BN, Local, lr=1e-3) | 3.930M | **61.83%** | 83.94% | 38.40% | 58.66% | 7.22% | 23.88% | 188.77m |
| `baseline_convnext_atto_lr4e3_25ep` | ConvNeXt-Atto (4x MLP FFN, tuned lr=4e-3) | 3.696M | **63.27%** | 85.51% | 45.04% | 60.89% | 11.62% | 36.48% | 188.42m |
| `in1k_subspace_femto_gffn_25ep` | Subspace-CAN + Global Context (+gFFN-G, lr=1e-3) | 3.944M | **62.84%** | 84.75% | 39.23% | 60.22% | 7.53% | 25.12% | 188.16m |
| `in1k_subspace_femto_ln2d_25ep` | Subspace-CAN LayerNorm2d (no BN running stats, lr=1e-3) | 3.930M | **61.92%** | 84.37% | 43.98% | 59.96% | 7.67% | 26.41% | 193.26m |
