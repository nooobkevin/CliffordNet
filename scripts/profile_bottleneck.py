#!/usr/bin/env python3
"""
Lightweight PyTorch Profiler script (no Nsight needed) to inspect per-operator CUDA time,
CPU launch overhead, synchronization bubbles, and export a Perfetto/Chrome trace JSON.

Usage (on 1 free GPU):
  CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/profile_bottleneck.py \
      --model in1k_v2_pico_k4 --batch-size 256 --compile --trace /tmp/pico_trace.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.profiler import ProfilerActivity, profile, schedule

from cliffordnet.subspace import SUBSPACE_MODEL_BUILDERS


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile CliffordNet/Subspace-CAN CUDA & launch bottlenecks")
    parser.add_argument("--model", type=str, default="in1k_v2_pico_k4")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--trace", type=str, default="", help="Optional path to export Perfetto/Chrome trace JSON")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    model = SUBSPACE_MODEL_BUILDERS[args.model](num_classes=1000).to(
        device=device, memory_format=torch.channels_last
    )
    if args.compile:
        model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)

    x = torch.randn(args.batch_size, 3, 224, 224, device=device, dtype=torch.bfloat16).contiguous(
        memory_format=torch.channels_last
    )
    y = torch.randint(0, 1000, (args.batch_size,), device=device)

    # Warmup (compile + allocator)
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()

    # Wall-clock step measurement
    t0 = time.perf_counter()
    n_bench = 10
    for _ in range(n_bench):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) * 1000.0 / n_bench

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
        record_shapes=True,
        with_stack=False,
    ) as prof:
        for _ in range(5):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            prof.step()

    if args.trace:
        prof.export_chrome_trace(args.trace)
        print(f"Exported Perfetto/Chrome trace to: {args.trace}")

    avgs = prof.key_averages()
    total_self_cuda_ms = sum(e.self_device_time_total for e in avgs) / 1000.0 / 3.0
    total_self_cpu_ms = sum(e.self_cpu_time_total for e in avgs) / 1000.0 / 3.0
    gap_ms = max(0.0, wall_ms - total_self_cuda_ms)

    print(f"\n=== Bottleneck Summary ({args.model}, bs={args.batch_size}, compile={args.compile}) ===")
    print(f"  Wall time / step       : {wall_ms:7.2f} ms")
    print(f"  Active CUDA kernel sum : {total_self_cuda_ms:7.2f} ms ({100.0 * total_self_cuda_ms / wall_ms:.1f}%)")
    print(f"  Launch/sync/idle gap   : {gap_ms:7.2f} ms ({100.0 * gap_ms / wall_ms:.1f}%)")
    print(f"  CPU time / step        : {total_self_cpu_ms:7.2f} ms")
    print("\n--- Top 20 Ops by Self CUDA Time ---")
    print(avgs.table(sort_by="self_device_time_total", row_limit=20))


if __name__ == "__main__":
    main()
