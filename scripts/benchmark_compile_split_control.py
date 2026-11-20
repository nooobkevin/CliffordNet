#!/usr/bin/env python3
"""Add the missing torch.compile + split_in_proj (WITHOUT Triton) control to the E2E H800 benchmark.

Separates how much of the +33.4% end-to-end speedup comes from eliminating .chunk() NHWC
strided views (layout rewrite) vs. the fused Triton subspace kernel itself.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from cliffordnet.subspace import get_subspace_model_spec


def bench_one(model_size: str, backend: str, use_compile: bool, extra_overrides: dict, bs: int = 256, warmup: int = 8, iters: int = 25) -> dict:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch._dynamo.reset()

    model = get_subspace_model_spec(
        model_size,
        img_size=224,
        num_classes=1000,
        backend=backend,
        **extra_overrides,
    ).build().to(device).to(memory_format=torch.channels_last)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05, fused=True)
    model_run = torch.compile(model) if use_compile else model

    x = torch.randn(bs, 3, 224, 224, device=device).to(memory_format=torch.channels_last)
    y = torch.randint(0, 1000, (bs,), device=device)

    for _ in range(warmup):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model_run(x), y)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model_run(x), y)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    ms = 1000.0 * (time.perf_counter() - t0) / iters
    ips = bs / (ms / 1000.0)
    peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    return {
        "model_size": model_size,
        "backend": backend,
        "params_m": round(n_params / 1e6, 3),
        "use_compile": use_compile,
        "extra_overrides": extra_overrides,
        "ms_per_step": round(ms, 2),
        "img_per_sec": round(ips, 1),
        "peak_vram_gib": round(peak_gb, 2),
    }


def main() -> None:
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    results = {}
    cases = [
        ("pico_chunked_compile_pytorch", "in1k_v2_pico_rmsnorm2d", "pytorch", {"flat_gamma": True}),
        ("pico_split_compile_pytorch", "in1k_v2_pico_rmsnorm2d_fast", "pytorch", {"flat_gamma": True}),
        ("pico_split_compile_triton", "in1k_v2_pico_rmsnorm2d_fast", "triton", {"flat_gamma": True}),
        ("atto_chunked_compile_pytorch", "in1k_v2_atto_k4", "pytorch", {"flat_gamma": True}),
        ("atto_split_compile_pytorch", "in1k_v2_atto_k4", "pytorch", {"split_in_proj": True, "flat_gamma": True}),
        ("atto_split_compile_triton", "in1k_v2_atto_k4", "triton", {"split_in_proj": True, "flat_gamma": True}),
    ]
    for label, msize, backend, extra in cases:
        r = bench_one(msize, backend=backend, use_compile=True, extra_overrides=extra)
        results[label] = r
        print(f"  {label:32s}: {r['ms_per_step']:6.2f} ms | {r['img_per_sec']:7.1f} img/s | {r['peak_vram_gib']:.2f} GiB", flush=True)

    out_p = Path("/project/itscspod/chtaihei/cliffordnet/CliffordNet/experiments/subspace_study/results/e2e_compile_split_control_h800.json")
    out_p.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {out_p}", flush=True)


if __name__ == "__main__":
    main()
