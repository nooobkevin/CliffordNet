# ImageNet-1k (25 Epochs, Global BS=1024, bf16-mixed) First-Principles v2 Ablation Ladder

| Run | Model & Cumulative Fix | Params | GMACs | EMA Top-1 | RAW Top-1 | Best Top-5 | S3 Dead% | |W|/|D| | Contrast 0.12 | FFT Phase |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `in1k_v2_rung1_recipe_25ep` | Rung 1 (+Fix4 Fair Recipe only: v1 + dpr=0.10, lr=2.5e-3) | 3.944M | 0.737G | **65.43%** | **65.51%** | 86.68% | 44.5% | 8.75x | 44.47% | 29.15% |
| `in1k_v2_rung2_deadzone_25ep` | Rung 2 (+Fix1 Dead-Zone Fix: ctx_bn + abs + proj_bias) | 3.957M | 0.737G | **67.48%** | **67.45%** | 87.97% | 11.7% | 2.64x | 50.50% | 31.08% |
| `in1k_v2_atto_k4_25ep` | Rung 3 Full Subspace-CAN-v2 (dw7 + (48,96,192,440) k=4, 0.491G) | 3.573M | 0.491G | **65.58%** | **65.59%** | 86.66% | 9.8% | 2.70x | 44.65% | 29.66% |
| `in1k_v2_atto_swiglu_k6_25ep` | Rung 4 Subspace-CAN-v2 SwiGLU+Wedge k=(2,4,4,6) (0.495G) | 3.621M | 0.495G | **49.88%** | **50.01%** | 74.38% | 24.1% | 3.13x | 15.03% | 18.13% |
