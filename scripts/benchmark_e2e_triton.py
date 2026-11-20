#!/usr/bin/env python3
"""End-to-end training step throughput & peak VRAM benchmark for Subspace-CAN on H800.

Compares PyTorch Eager, torch.compile, Fused Triton (chunked in_proj), and
Fused Triton (split_in_proj) under identical batch size, dtype, and memory layout.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cliffordnet.subspace import get_subspace_model_spec


def _bench_config(
    *,
    label: str,
    model_size: str,
    backend: str,
    compile_model: bool,
    batch_size: int,
    img_size: int,
    dtype: torch.dtype,
    warmup: int = 8,
    iters: int = 25,
    extra_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch._dynamo.reset()

    model = get_subspace_model_spec(
        model_size,
        img_size=img_size,
        num_classes=1000,
        backend=backend,
        **(extra_overrides or {}),
    ).build().cuda()
    model = model.to(memory_format=torch.channels_last)
    model.train()

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    if compile_model:
        model_run = torch.compile(model)
    else:
        model_run = model

    x = torch.randn(batch_size, 3, img_size, img_size, device="cuda").to(
        memory_format=torch.channels_last
    )
    y = torch.randint(0, 1000, (batch_size,), device="cuda")
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)

    use_amp = dtype in (torch.bfloat16, torch.float16)

    def step() -> None:
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=dtype, enabled=use_amp):
            out = model_run(x)
            loss = loss_fn(out, y)
        loss.backward()
        opt.step()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    for _ in range(iters):
        step()
    end_ev.record()
    torch.cuda.synchronize()

    ms_per_step = start_ev.elapsed_time(end_ev) / iters
    imgs_per_sec = batch_size / (ms_per_step / 1000.0)
    peak_vram_gib = torch.cuda.max_memory_allocated() / (1024**3)

    res = {
        "label": label,
        "model_size": model_size,
        "backend": backend,
        "compile": compile_model,
        "dtype": str(dtype).replace("torch.", ""),
        "batch_size": batch_size,
        "params_m": round(params_m, 3),
        "step_ms": round(ms_per_step, 2),
        "imgs_per_sec": round(imgs_per_sec, 1),
        "peak_vram_gib": round(peak_vram_gib, 2),
    }
    print(
        f"{label:42s} | {res['step_ms']:6.2f} ms/step | {res['imgs_per_sec']:7.1f} img/s | "
        f"{res['peak_vram_gib']:5.2f} GiB",
        flush=True,
    )
    del model, model_run, opt, x, y
    torch.cuda.empty_cache()
    return res


def main() -> None:
    device_name = torch.cuda.get_device_name(0)
    print(f"End-to-End Subspace-CAN Benchmark on {device_name} (batch=256, 224x224, bf16-amp, channels_last)\n")

    results: list[dict[str, Any]] = []

    # 1. Subspace-CAN-v2-Pico (RMSNorm2d, 8.49M, 1.32G)
    pico_cases = [
        ("Pico RMSNorm2d | PyTorch Eager", "in1k_v2_pico_rmsnorm2d", "pytorch", False, {"flat_gamma": True}),
        ("Pico RMSNorm2d | torch.compile", "in1k_v2_pico_rmsnorm2d", "pytorch", True, {"flat_gamma": True}),
        ("Pico RMSNorm2d | Triton (chunked)", "in1k_v2_pico_rmsnorm2d", "triton", False, {"flat_gamma": True}),
        ("Pico RMSNorm2d | Triton + compile (chunked)", "in1k_v2_pico_rmsnorm2d", "triton", True, {"flat_gamma": True}),
        ("Pico RMSNorm2d | Triton (split_in_proj)", "in1k_v2_pico_rmsnorm2d_fast", "triton", False, {"flat_gamma": True}),
        ("Pico RMSNorm2d | Triton + compile (split_in_proj)", "in1k_v2_pico_rmsnorm2d_fast", "triton", True, {"flat_gamma": True}),
    ]
    for label, msize, backend, comp, extra in pico_cases:
        results.append(
            _bench_config(
                label=label,
                model_size=msize,
                backend=backend,
                compile_model=comp,
                batch_size=256,
                img_size=224,
                dtype=torch.bfloat16,
                extra_overrides=extra,
            )
        )

    # 2. Subspace-CAN-v2-Atto (3.57M, 0.491G)
    atto_cases = [
        ("Atto v2 | PyTorch Eager", "in1k_v2_atto_k4", "pytorch", False, {"flat_gamma": True}),
        ("Atto v2 | torch.compile", "in1k_v2_atto_k4", "pytorch", True, {"flat_gamma": True}),
        ("Atto v2 | Triton (chunked)", "in1k_v2_atto_k4", "triton", False, {"flat_gamma": True}),
        ("Atto v2 | Triton + compile (chunked)", "in1k_v2_atto_k4", "triton", True, {"flat_gamma": True}),
        ("Atto v2 | Triton (split_in_proj)", "in1k_v2_atto_k4", "triton", False, {"split_in_proj": True, "flat_gamma": True}),
        ("Atto v2 | Triton + compile (split_in_proj)", "in1k_v2_atto_k4", "triton", True, {"split_in_proj": True, "flat_gamma": True}),
    ]
    for label, msize, backend, comp, extra in atto_cases:
        results.append(
            _bench_config(
                label=label,
                model_size=msize,
                backend=backend,
                compile_model=comp,
                batch_size=256,
                img_size=224,
                dtype=torch.bfloat16,
                extra_overrides=extra,
            )
        )

    out_path = REPO_ROOT / "experiments/subspace_study/results/e2e_triton_benchmark_h800.json"
    out_path.write_text(
        json.dumps({"gpu": device_name, "batch_size": 256, "img_size": 224, "dtype": "bfloat16", "results": results}, indent=2)
    )
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
