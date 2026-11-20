# CIFAR-100 Matched 2x2 Control (Wave 5, 3 Seeds)

Protocol: 200-epoch CIFAR-100 (`fp32`, AdamW `lr=1e-3`, `wd=0.1`, AutoAugment + RandomErasing `0.25`, `ortho_weight=0.0`), strictly matched backbone (`dw5`, `proj_bias=True`, `drop_path_rate=0.20`) across seeds `42`, `101`, and `202` (12 runs total; Cell C/D seed `42` reused from Wave 4).

## Cell Summary (`mean +/- std`, n = 3 seeds)

| Context setup | Branch | Params | Clean Top-1 (%) | Contrast 0.12x (%) | Gaussian Noise (%) | FFT Phase-Only (%) | Effective Rank | Stage-3 Dead-Zone (< -1.28) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original (`diff`, no `ctx_bn`) | Full (`Inner + Wedge`) | 1.124M | 78.53 +/- 0.41 | 55.43 +/- 1.45 | 17.33 +/- 1.69 | 33.10 +/- 0.67 | 0.718 +/- 0.001 | 37.8 +/- 0.2% |
| Original (`diff`, no `ctx_bn`) | Inner-wide (`0` bivectors) | 1.132M | 77.65 +/- 0.24 | 53.35 +/- 1.06 | 13.06 +/- 0.80 | 32.40 +/- 1.73 | 0.705 +/- 0.003 | 15.4 +/- 0.3% |
| Fixed (`abs + ctx_bn`) | Full (`Inner + Wedge`) | 1.128M | 78.83 +/- 0.33 | 57.17 +/- 1.45 | 17.57 +/- 1.11 | 35.23 +/- 1.06 | 0.730 +/- 0.005 | 6.4 +/- 0.2% |
| Fixed (`abs + ctx_bn`) | Inner-wide (`0` bivectors) | 1.138M | 79.51 +/- 0.10 | 58.39 +/- 1.11 | 17.29 +/- 1.47 | 33.63 +/- 0.38 | 0.709 +/- 0.003 | 5.6 +/- 0.1% |

## Paired Per-Seed Interaction Effects

| Metric | Original Gap (`A - B`) | Fixed Gap (`C - D`) | Interaction `(A - B) - (C - D)` | Per-seed `(A-B)-(C-D)` (`42, 101, 202`) |
| :--- | ---: | ---: | ---: | :--- |
| Clean Top-1 (%) | +0.88 +/- 0.27 | -0.68 +/- 0.30 | **+1.56 +/- 0.55** | `+1.01, +1.55, +2.11` |
| Contrast 0.12x (%) | +2.07 +/- 2.50 | -1.21 +/- 1.05 | +3.29 +/- 1.63 | `+4.96, +1.70, +3.20` |
| Gaussian Noise (%) | +4.26 +/- 1.03 | +0.28 +/- 2.47 | +3.99 +/- 2.93 | `+0.69, +4.96, +6.31` |
| FFT Phase-Only (%) | +0.69 +/- 2.31 | **+1.61 +/- 1.11** | -0.91 +/- 1.52 | `+0.03, -2.67, -0.10` |
| Effective Rank | +0.013 +/- 0.003 | **+0.021 +/- 0.004** | -0.008 +/- 0.002 | `-0.010, -0.009, -0.007` |
