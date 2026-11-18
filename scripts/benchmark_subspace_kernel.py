"""Fair NVIDIA H800 operator benchmark for Subspace Clifford Networks (arXiv:2610.37226 Sec 3.5 & 4.4).

Compares four implementations across:
1. Paper Sec 4.4 shape (64, 128, 64, 64), k in {2, 3, 4}, full, diff, in {fp32, bf16} x {NCHW, NHWC}.
2. ImageNet-1k hierarchical pyramid stages at B=128: (64, 56, 56), (128, 28, 28), (256, 14, 14),
   (512, 7, 7) in bf16 channels_last (NHWC), k in {2, 3, 4}.

Implementations compared (25 warmup iters, 100 CUDA-event timed iters):
- pytorch_eager
- torch_compile_default
- torch_compile_max_autotune
- triton_recompute
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
from typing import Any, Callable

# Ensure non-/tmp compiler caches and cluster CUDA toolkit PATH if available
_DEFAULT_OUT_DIR = Path("./outputs/benchmarks")
_CACHE_ROOT = Path(os.environ.get("CLIFFORDNET_CACHE_DIR", str(Path.home() / ".cache" / "cliffordnet")))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_CACHE_ROOT / ".torchinductor_cache"))
os.environ.setdefault("TRITON_CACHE_DIR", str(_CACHE_ROOT / ".triton_cache"))
_CLUSTER_CUDA_BIN = "/cm/shared/apps/cuda-latest/toolkit/current/bin"
if os.path.isdir(_CLUSTER_CUDA_BIN) and _CLUSTER_CUDA_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{_CLUSTER_CUDA_BIN}:{os.environ.get('PATH', '')}"

import torch
import triton

from cliffordnet.kernels import (
    _reference_subspace_gp,
    subspace_gp_triton,
    subspace_output_channels,
)


class _GradSink(torch.autograd.Function):
    """Zero-allocation sink injecting pre-allocated grad_out after out can be freed."""

    @staticmethod
    def forward(ctx, out: torch.Tensor, grad_out: torch.Tensor) -> torch.Tensor:
        ctx.grad_out = grad_out
        return out.new_empty(())

    @staticmethod
    def backward(ctx, _grad_scalar: torch.Tensor):
        return ctx.grad_out, None


def _aligned_dim(d: int, k: int) -> int:
    """Aligned subspace width D_tilde = ceil(D / k) * k (arXiv:2610.37226 Sec 3.3.1)."""
    return ((d + k - 1) // k) * k


def _compute_fp64_reference(
    z1: torch.Tensor,
    z2: torch.Tensor,
    grad_out: torch.Tensor,
    k: int,
    cli_mode: str,
    ctx_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z1_64 = z1.detach().to(torch.float64).requires_grad_(True)
    z2_64 = z2.detach().to(torch.float64).requires_grad_(True)
    go_64 = grad_out.detach().to(torch.float64)
    out_64 = _reference_subspace_gp(
        z1_64, z2_64, k=k, cli_mode=cli_mode, ctx_mode=ctx_mode, use_fma=True
    )
    out_64.backward(go_64)
    return out_64.detach(), z1_64.grad.detach(), z2_64.grad.detach()


def _benchmark_single_impl(
    impl_name: str,
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    z1: torch.Tensor,
    z2: torch.Tensor,
    grad_out: torch.Tensor,
    ref_out_64: torch.Tensor,
    ref_dz1_64: torch.Tensor,
    ref_dz2_64: torch.Tensor,
    *,
    uses_cudagraphs: bool,
    warmup: int,
    iters: int,
    min_io_bytes_fwd: int,
    min_io_bytes_bwd: int,
) -> dict[str, Any]:
    z1.grad = None
    z2.grad = None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Baseline memory with ONLY (z1, z2, grad_out, fp64_refs) live before warmup/compilation
    baseline_bytes = torch.cuda.memory_allocated()

    # 1. Numerical accuracy check vs fp64
    if uses_cudagraphs:
        torch.compiler.cudagraph_mark_step_begin()
    out_chk = fn(z1, z2)
    out_chk_clone = out_chk.detach().clone()
    out_chk.backward(grad_out)
    err_fwd = (out_chk_clone.double() - ref_out_64).abs().max().item()
    err_dz1 = (z1.grad.double() - ref_dz1_64).abs().max().item()
    err_dz2 = (z2.grad.double() - ref_dz2_64).abs().max().item()
    err_bwd = max(err_dz1, err_dz2)
    err_max = max(err_fwd, err_bwd)
    del out_chk, out_chk_clone
    z1.grad = None
    z2.grad = None

    # 2. Warmup iterations (iter 0 also measures static buffer peak prior to CUDAGraph capture)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        z1.grad = None
        z2.grad = None
        if uses_cudagraphs:
            torch.compiler.cudagraph_mark_step_begin()
        out = fn(z1, z2)
        out.backward(grad_out)
        del out
    z1.grad = None
    z2.grad = None
    torch.cuda.synchronize()
    cudagraph_pool_peak_bytes = max(0, torch.cuda.max_memory_allocated() - baseline_bytes)

    # 3. Peak extra allocated memory during forward+backward (paper Sec 4.4 protocol)
    torch.cuda.reset_peak_memory_stats()
    if uses_cudagraphs:
        torch.compiler.cudagraph_mark_step_begin()
        out = fn(z1, z2)
        out.backward(grad_out)
        del out
        torch.cuda.synchronize()
        step_peak_bytes = max(0, torch.cuda.max_memory_allocated() - baseline_bytes)
        peak_extra_bytes = (
            step_peak_bytes if step_peak_bytes > 0 else cudagraph_pool_peak_bytes
        )
        saved_extra_bytes = peak_extra_bytes
    else:
        base_step_bytes = torch.cuda.memory_allocated()
        out = fn(z1, z2)
        sink = _GradSink.apply(out, grad_out)
        del out
        saved_extra_bytes = torch.cuda.memory_allocated() - base_step_bytes
        sink.backward()
        del sink
        torch.cuda.synchronize()
        peak_extra_bytes = torch.cuda.max_memory_allocated() - base_step_bytes

    z1.grad = None
    z2.grad = None
    torch.cuda.synchronize()

    # 4. Timed iterations using CUDA Events
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    mid_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        z1.grad = None
        z2.grad = None
        if uses_cudagraphs:
            torch.compiler.cudagraph_mark_step_begin()
        start_events[i].record()
        out = fn(z1, z2)
        mid_events[i].record()
        out.backward(grad_out)
        end_events[i].record()
        del out

    torch.cuda.synchronize()

    fwd_times = [s.elapsed_time(m) for s, m in zip(start_events, mid_events)]
    bwd_times = [m.elapsed_time(e) for m, e in zip(mid_events, end_events)]
    tot_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    fwd_ms = sum(fwd_times) / iters
    bwd_ms = sum(bwd_times) / iters
    total_ms = sum(tot_times) / iters

    fwd_gbps = (min_io_bytes_fwd / 1e9) / (fwd_ms / 1e3) if fwd_ms > 0 else 0.0
    bwd_gbps = (min_io_bytes_bwd / 1e9) / (bwd_ms / 1e3) if bwd_ms > 0 else 0.0
    tot_gbps = (
        ((min_io_bytes_fwd + min_io_bytes_bwd) / 1e9) / (total_ms / 1e3)
        if total_ms > 0
        else 0.0
    )

    return {
        "implementation": impl_name,
        "fwd_ms": round(fwd_ms, 4),
        "bwd_ms": round(bwd_ms, 4),
        "total_ms": round(total_ms, 4),
        "effective_fwd_gbps": round(fwd_gbps, 1),
        "effective_bwd_gbps": round(bwd_gbps, 1),
        "effective_total_gbps": round(tot_gbps, 1),
        "peak_extra_allocated_mib": round(peak_extra_bytes / (1024**2), 2),
        "saved_extra_allocated_mib": round(saved_extra_bytes / (1024**2), 2),
        "max_abs_err_fwd_vs_fp64": err_fwd,
        "max_abs_err_bwd_vs_fp64": err_bwd,
        "max_abs_err_vs_fp64": err_max,
    }


def _run_config_benchmark(
    *,
    suite: str,
    label: str,
    b: int,
    d: int,
    h: int,
    w: int,
    k: int,
    cli_mode: str,
    ctx_mode: str,
    dtype: torch.dtype,
    layout_name: str,
    mem_fmt: torch.memory_format,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    d_tilde = _aligned_dim(d, k)
    d_out = subspace_output_channels(d_tilde, k, cli_mode)
    dtype_str = "fp32" if dtype == torch.float32 else "bf16"

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    z1 = (
        torch.randn(b, d_tilde, h, w, device="cuda", dtype=dtype)
        .contiguous(memory_format=mem_fmt)
        .requires_grad_(True)
    )
    z2 = (
        torch.randn(b, d_tilde, h, w, device="cuda", dtype=dtype)
        .contiguous(memory_format=mem_fmt)
        .requires_grad_(True)
    )
    grad_out = torch.randn(b, d_out, h, w, device="cuda", dtype=dtype).contiguous(
        memory_format=mem_fmt
    )

    ref_out_64, ref_dz1_64, ref_dz2_64 = _compute_fp64_reference(
        z1, z2, grad_out, k=k, cli_mode=cli_mode, ctx_mode=ctx_mode
    )

    elem_bytes = z1.element_size()
    in_elems = z1.numel()
    out_elems = grad_out.numel()
    min_io_bytes_fwd = (2 * in_elems + out_elems) * elem_bytes
    min_io_bytes_bwd = (4 * in_elems + out_elems) * elem_bytes

    def eager_fn(a: torch.Tensor, b_t: torch.Tensor) -> torch.Tensor:
        return _reference_subspace_gp(
            a, b_t, k=k, cli_mode=cli_mode, ctx_mode=ctx_mode, use_fma=True
        )

    def triton_fn(a: torch.Tensor, b_t: torch.Tensor) -> torch.Tensor:
        return subspace_gp_triton(
            a, b_t, k=k, cli_mode=cli_mode, ctx_mode=ctx_mode, use_fma=True
        )

    impl_results: dict[str, dict[str, Any]] = {}

    # 1. pytorch_eager
    impl_results["pytorch_eager"] = _benchmark_single_impl(
        "pytorch_eager",
        eager_fn,
        z1,
        z2,
        grad_out,
        ref_out_64,
        ref_dz1_64,
        ref_dz2_64,
        uses_cudagraphs=False,
        warmup=warmup,
        iters=iters,
        min_io_bytes_fwd=min_io_bytes_fwd,
        min_io_bytes_bwd=min_io_bytes_bwd,
    )

    # 2. torch_compile_default
    torch._dynamo.reset()
    compile_default_fn = torch.compile(eager_fn, mode="default")
    impl_results["torch_compile_default"] = _benchmark_single_impl(
        "torch_compile_default",
        compile_default_fn,
        z1,
        z2,
        grad_out,
        ref_out_64,
        ref_dz1_64,
        ref_dz2_64,
        uses_cudagraphs=False,
        warmup=warmup,
        iters=iters,
        min_io_bytes_fwd=min_io_bytes_fwd,
        min_io_bytes_bwd=min_io_bytes_bwd,
    )
    del compile_default_fn

    # 3. torch_compile_max_autotune
    torch._dynamo.reset()
    compile_max_fn = torch.compile(eager_fn, mode="max-autotune")
    impl_results["torch_compile_max_autotune"] = _benchmark_single_impl(
        "torch_compile_max_autotune",
        compile_max_fn,
        z1,
        z2,
        grad_out,
        ref_out_64,
        ref_dz1_64,
        ref_dz2_64,
        uses_cudagraphs=True,
        warmup=warmup,
        iters=iters,
        min_io_bytes_fwd=min_io_bytes_fwd,
        min_io_bytes_bwd=min_io_bytes_bwd,
    )
    del compile_max_fn
    torch._dynamo.reset()

    # 4. triton_recompute
    impl_results["triton_recompute"] = _benchmark_single_impl(
        "triton_recompute",
        triton_fn,
        z1,
        z2,
        grad_out,
        ref_out_64,
        ref_dz1_64,
        ref_dz2_64,
        uses_cudagraphs=False,
        warmup=warmup,
        iters=iters,
        min_io_bytes_fwd=min_io_bytes_fwd,
        min_io_bytes_bwd=min_io_bytes_bwd,
    )

    # Compute relative speedups on every implementation entry
    eager_tot = impl_results["pytorch_eager"]["total_ms"]
    comp_def_tot = impl_results["torch_compile_default"]["total_ms"]
    comp_max_tot = impl_results["torch_compile_max_autotune"]["total_ms"]
    best_comp_tot = min(comp_def_tot, comp_max_tot)

    for entry in impl_results.values():
        tot = entry["total_ms"]
        entry["speedup_vs_eager"] = round(eager_tot / tot, 2) if tot > 0 else 0.0
        entry["speedup_vs_compile_default"] = round(comp_def_tot / tot, 2) if tot > 0 else 0.0
        entry["speedup_vs_compile_max_autotune"] = (
            round(comp_max_tot / tot, 2) if tot > 0 else 0.0
        )
        entry["speedup_vs_best_compile"] = round(best_comp_tot / tot, 2) if tot > 0 else 0.0

    del z1, z2, grad_out, ref_out_64, ref_dz1_64, ref_dz2_64
    gc.collect()
    torch.cuda.empty_cache()

    tr = impl_results["triton_recompute"]
    print(
        f"[{suite}] {label:32s} | dtype={dtype_str:4s} {layout_name:4s} k={k} "
        f"(D_tilde={d_tilde:3d}->D_out={d_out:4d}) | "
        f"eager={eager_tot:6.3f}ms  comp_def={comp_def_tot:6.3f}ms  comp_max={comp_max_tot:6.3f}ms  "
        f"triton={tr['total_ms']:6.3f}ms ({tr['effective_total_gbps']:5.0f} GB/s, "
        f"{tr['speedup_vs_eager']:5.2f}x eager, {tr['speedup_vs_compile_default']:4.2f}x comp_def) | "
        f"mem={tr['peak_extra_allocated_mib']:6.1f} MiB | err={tr['max_abs_err_vs_fp64']:.2e}",
        flush=True,
    )

    return {
        "suite": suite,
        "label": label,
        "shape_bdhw": [b, d, h, w],
        "aligned_shape_bdhw": [b, d_tilde, h, w],
        "output_shape_bdhw": [b, d_out, h, w],
        "k": k,
        "cli_mode": cli_mode,
        "ctx_mode": ctx_mode,
        "dtype": dtype_str,
        "memory_format": layout_name,
        "output_tensor_mib": round((out_elems * elem_bytes) / (1024**2), 2),
        "results": impl_results,
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    meta = payload["metadata"]
    lines: list[str] = [
        "# Subspace Clifford Geometric Product — NVIDIA H800 Kernel Benchmark",
        "",
        "Reference: **Subspace Clifford Networks** (`arXiv:2610.37226`, Sections 3.5 & 4.4).",
        "",
        "## Hardware & Environment",
        "",
        f"- **GPU**: {meta['gpu_name']} (`sm_{meta['compute_capability'][0]}{meta['compute_capability'][1]}`, "
        f"{meta['total_vram_gib']:.1f} GiB HBM3)",
        f"- **PyTorch**: `{meta['torch_version']}`",
        f"- **Triton**: `{meta['triton_version']}`",
        f"- **Protocol**: {meta['warmup_iters']} warmup iterations, {meta['timed_iters']} CUDA-event timed iterations; "
        "`torch.cuda.reset_peak_memory_stats()` prior to forward+backward; max absolute error measured against `float64` reference.",
        "- **Channel alignment**: For backbone width $D$ and group size $k$, $\\tilde{D} = \\lceil D / k \\rceil \\cdot k$ "
        "(Sec. 3.3.1; $\\tilde{D}=128$ for $k \\in \\{2,4\\}$, $\\tilde{D}=129$ for $D=128, k=3$).",
        "",
    ]

    impl_order = (
        "pytorch_eager",
        "torch_compile_default",
        "torch_compile_max_autotune",
        "triton_recompute",
    )

    for suite_key, suite_title in (
        (
            "paper_sec4_4",
            "## 1. Paper Exact Config (`(B, D, H, W) = (64, 128, 64, 64)`, `full`, `diff`)",
        ),
        (
            "imagenet1k_pyramid",
            "## 2. ImageNet-1k Pyramid Stage Shapes (`B = 128`, `bf16`, `channels_last / NHWC`, `full`, `diff`)",
        ),
    ):
        lines.append(suite_title)
        lines.append("")
        lines.append(
            "| Config | Dtype | Layout | $k$ | $\\tilde{D} \\to D_{\\text{out}}$ | Implementation | "
            "Fwd (ms) | Bwd (ms) | Total (ms) | vs Eager | vs Compile (def) | vs Compile (max) | "
            "Peak Extra (MiB) | Max $|\\Delta|$ vs fp64 |"
        )
        lines.append(
            "| :--- | :---: | :---: | :---: | :---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        suite_entries = [c for c in payload["benchmarks"] if c["suite"] == suite_key]
        for cfg in suite_entries:
            _, d_tilde, _, _ = cfg["aligned_shape_bdhw"]
            d_out = cfg["output_shape_bdhw"][1]
            for impl_name in impl_order:
                r = cfg["results"][impl_name]
                bold = "**" if impl_name == "triton_recompute" else ""
                lines.append(
                    f"| `{cfg['label']}` | `{cfg['dtype']}` | `{cfg['memory_format']}` | {cfg['k']} | "
                    f"`{d_tilde}->{d_out}` | {bold}`{impl_name}`{bold} | "
                    f"{bold}{r['fwd_ms']:.3f}{bold} | {bold}{r['bwd_ms']:.3f}{bold} | {bold}{r['total_ms']:.3f}{bold} | "
                    f"{bold}{r['speedup_vs_eager']:.2f}x{bold} | {bold}{r['speedup_vs_compile_default']:.2f}x{bold} | "
                    f"{bold}{r['speedup_vs_compile_max_autotune']:.2f}x{bold} | "
                    f"{bold}{r['peak_extra_allocated_mib']:.1f}{bold} | `{r['max_abs_err_vs_fp64']:.2e}` |"
                )
        lines.append("")

    # Concise summary table of triton_recompute highlights
    lines.append("## 3. Summary of `triton_recompute` Speedups & HBM3 Bandwidth")
    lines.append("")
    lines.append(
        "| Suite | Shape `(B, D, H, W)` | Dtype | Layout | $k$ | Triton Fwd+Bwd (ms) | "
        "Eff. BW (GB/s) | Speedup vs Eager | Speedup vs Compile (def) | Speedup vs Compile (max) | Peak Extra (MiB) |"
    )
    lines.append("| :--- | :--- | :---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for cfg in payload["benchmarks"]:
        tr = cfg["results"]["triton_recompute"]
        shape_str = str(tuple(cfg["shape_bdhw"]))
        lines.append(
            f"| `{cfg['suite']}` | `{shape_str}` | `{cfg['dtype']}` | `{cfg['memory_format']}` | {cfg['k']} | "
            f"**{tr['total_ms']:.3f}** | {tr['effective_total_gbps']:.0f} | "
            f"**{tr['speedup_vs_eager']:.2f}x** | **{tr['speedup_vs_compile_default']:.2f}x** | "
            f"**{tr['speedup_vs_compile_max_autotune']:.2f}x** | {tr['peak_extra_allocated_mib']:.1f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Subspace Clifford Triton Kernel on H800")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations (default: 25)")
    parser.add_argument("--iters", type=int, default=100, help="Timed iterations (default: 100)")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=_DEFAULT_OUT_DIR / "kernel_benchmark_h800.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=_DEFAULT_OUT_DIR / "kernel_benchmark_h800.md",
        help="Output Markdown path",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required to run benchmark_subspace_kernel.py")

    props = torch.cuda.get_device_properties(0)
    print(
        f"Running Subspace Clifford Kernel Benchmark on {props.name} "
        f"(sm_{props.major}{props.minor}, PyTorch {torch.__version__}, Triton {triton.__version__})",
        flush=True,
    )

    benchmarks: list[dict[str, Any]] = []

    # 1. Paper Sec 4.4 exact config: (64, 128, 64, 64), k in (2, 3, 4), full, diff, fp32 & bf16, NCHW & NHWC
    for dtype in (torch.float32, torch.bfloat16):
        for layout_name, mem_fmt in (
            ("NCHW", torch.contiguous_format),
            ("NHWC", torch.channels_last),
        ):
            for k in (2, 3, 4):
                benchmarks.append(
                    _run_config_benchmark(
                        suite="paper_sec4_4",
                        label="paper_64x128x64x64",
                        b=64,
                        d=128,
                        h=64,
                        w=64,
                        k=k,
                        cli_mode="full",
                        ctx_mode="diff",
                        dtype=dtype,
                        layout_name=layout_name,
                        mem_fmt=mem_fmt,
                        warmup=args.warmup,
                        iters=args.iters,
                    )
                )

    # 2. ImageNet-1k pyramid stage shapes (B=128) in bf16 channels_last
    pyramid_stages = [
        ("stage1_128x64x56x56", 128, 64, 56, 56),
        ("stage2_128x128x28x28", 128, 128, 28, 28),
        ("stage3_128x256x14x14", 128, 256, 14, 14),
        ("stage4_128x512x7x7", 128, 512, 7, 7),
    ]
    for stage_label, b, d, h, w in pyramid_stages:
        for k in (2, 3, 4):
            benchmarks.append(
                _run_config_benchmark(
                    suite="imagenet1k_pyramid",
                    label=stage_label,
                    b=b,
                    d=d,
                    h=h,
                    w=w,
                    k=k,
                    cli_mode="full",
                    ctx_mode="diff",
                    dtype=torch.bfloat16,
                    layout_name="NHWC",
                    mem_fmt=torch.channels_last,
                    warmup=args.warmup,
                    iters=args.iters,
                )
            )

    payload: dict[str, Any] = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "paper_ref": "arXiv:2610.37226 (Sec 3.5 & 4.4)",
            "gpu_name": props.name,
            "compute_capability": [props.major, props.minor],
            "total_vram_gib": round(props.total_memory / (1024**3), 2),
            "torch_version": torch.__version__,
            "triton_version": triton.__version__,
            "warmup_iters": args.warmup,
            "timed_iters": args.iters,
        },
        "benchmarks": benchmarks,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.md_out.write_text(_render_markdown(payload), encoding="utf-8")
    print(f"\nSaved JSON results to: {args.json_out}")
    print(f"Saved Markdown report to: {args.md_out}")


if __name__ == "__main__":
    main()
