"""Comprehensive unit tests for the fused Subspace Clifford Triton kernel."""

from __future__ import annotations

import os
from pathlib import Path
import unittest

# Configure non-/tmp caches and real CUDA toolkit PATH if available on cluster
_CACHE_ROOT = Path(os.environ.get("CLIFFORDNET_CACHE_DIR", str(Path.home() / ".cache" / "cliffordnet")))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_CACHE_ROOT / ".torchinductor_cache"))
os.environ.setdefault("TRITON_CACHE_DIR", str(_CACHE_ROOT / ".triton_cache"))
_CLUSTER_CUDA_BIN = "/cm/shared/apps/cuda-latest/toolkit/current/bin"
if os.path.isdir(_CLUSTER_CUDA_BIN) and _CLUSTER_CUDA_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{_CLUSTER_CUDA_BIN}:{os.environ.get('PATH', '')}"

import torch
import torch.nn as nn

from cliffordnet.kernels import (
    SubspaceGPTritonFn,
    _reference_subspace_gp,
    subspace_gp_triton,
    subspace_output_channels,
)


class _GradSink(torch.autograd.Function):
    """Zero-allocation helper feeding a pre-allocated dense grad_out into backward."""

    @staticmethod
    def forward(ctx, out: torch.Tensor, grad_out: torch.Tensor) -> torch.Tensor:
        ctx.grad_out = grad_out
        return out.new_empty(())

    @staticmethod
    def backward(ctx, _grad_scalar: torch.Tensor):
        return ctx.grad_out, None


def _assert_memory_format(tc: unittest.TestCase, t: torch.Tensor, fmt: torch.memory_format) -> None:
    if fmt == torch.channels_last:
        tc.assertTrue(
            t.is_contiguous(memory_format=torch.channels_last),
            f"Expected channels_last tensor, got shape={t.shape}, strides={t.stride()}",
        )
        if t.shape[1] > 1 and t.shape[2] * t.shape[3] > 1:
            tc.assertEqual(t.stride(1), 1)
    else:
        tc.assertTrue(
            t.is_contiguous(),
            f"Expected contiguous NCHW tensor, got shape={t.shape}, strides={t.stride()}",
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for Triton subspace tests")
class TestTritonSubspaceKernel(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2026)
        torch.cuda.manual_seed_all(2026)

    def test_fp32_vs_fp64_all_combinations(self) -> None:
        """All (k, cli_mode, ctx_mode, memory_format) in fp32 match fp64 within < 5e-6."""
        for k in (2, 3, 4):
            d_tilde = 24  # divisible by 2, 3, and 4
            # Use non-power-of-two spatial dims to exercise tail masking too
            b, h, w = 3, 17, 19
            z1_64 = torch.randn(b, d_tilde, h, w, device="cuda", dtype=torch.float64)
            z2_64 = torch.randn(b, d_tilde, h, w, device="cuda", dtype=torch.float64)

            for cli_mode in ("full", "wedge", "inner"):
                expected_c = subspace_output_channels(d_tilde, k, cli_mode)
                go_64_base = torch.randn(b, expected_c, h, w, device="cuda", dtype=torch.float64)

                for ctx_mode in ("diff", "abs"):
                    z1_ref = z1_64.detach().clone().requires_grad_(True)
                    z2_ref = z2_64.detach().clone().requires_grad_(True)
                    out_ref = _reference_subspace_gp(
                        z1_ref, z2_ref, k=k, cli_mode=cli_mode, ctx_mode=ctx_mode, use_fma=True
                    )
                    self.assertEqual(out_ref.shape, (b, expected_c, h, w))
                    out_ref.backward(go_64_base)

                    for mem_fmt in (torch.contiguous_format, torch.channels_last):
                        with self.subTest(
                            k=k, cli_mode=cli_mode, ctx_mode=ctx_mode, memory_format=mem_fmt
                        ):
                            z1 = (
                                z1_64.to(torch.float32)
                                .contiguous(memory_format=mem_fmt)
                                .requires_grad_(True)
                            )
                            z2 = (
                                z2_64.to(torch.float32)
                                .contiguous(memory_format=mem_fmt)
                                .requires_grad_(True)
                            )
                            go = go_64_base.to(torch.float32).contiguous(memory_format=mem_fmt)

                            out = subspace_gp_triton(
                                z1, z2, k=k, cli_mode=cli_mode, ctx_mode=ctx_mode, use_fma=True
                            )
                            self.assertEqual(out.dtype, torch.float32)
                            self.assertEqual(out.shape, (b, expected_c, h, w))
                            _assert_memory_format(self, out, mem_fmt)

                            out.backward(go)
                            self.assertIsNotNone(z1.grad)
                            self.assertIsNotNone(z2.grad)
                            _assert_memory_format(self, z1.grad, mem_fmt)
                            _assert_memory_format(self, z2.grad, mem_fmt)

                            fwd_diff = (out.double() - out_ref).abs().max().item()
                            dz1_diff = (z1.grad.double() - z1_ref.grad).abs().max().item()
                            dz2_diff = (z2.grad.double() - z2_ref.grad).abs().max().item()

                            self.assertLess(fwd_diff, 5e-6, f"fwd diff={fwd_diff}")
                            self.assertLess(dz1_diff, 5e-6, f"dz1 diff={dz1_diff}")
                            self.assertLess(dz2_diff, 5e-6, f"dz2 diff={dz2_diff}")

    def test_bf16_and_fp16_all_combinations(self) -> None:
        """All (k, cli_mode, ctx_mode, memory_format) in bf16 (and fp16) match FMA PyTorch ref."""
        for dtype, fwd_tol, bwd_tol in (
            (torch.bfloat16, 1.6e-2, 3.2e-2),
            (torch.float16, 2.0e-3, 4.0e-3),
        ):
            for k in (2, 3, 4):
                d_tilde = 24
                b, h, w = 4, 15, 21
                z1_base = torch.randn(b, d_tilde, h, w, device="cuda", dtype=dtype)
                z2_base = torch.randn(b, d_tilde, h, w, device="cuda", dtype=dtype)

                for cli_mode in ("full", "wedge", "inner"):
                    expected_c = subspace_output_channels(d_tilde, k, cli_mode)
                    go_base = torch.randn(b, expected_c, h, w, device="cuda", dtype=dtype)

                    for ctx_mode in ("diff", "abs"):
                        for mem_fmt in (torch.contiguous_format, torch.channels_last):
                            with self.subTest(
                                dtype=dtype,
                                k=k,
                                cli_mode=cli_mode,
                                ctx_mode=ctx_mode,
                                memory_format=mem_fmt,
                            ):
                                z1_ref = (
                                    z1_base.contiguous(memory_format=mem_fmt)
                                    .detach()
                                    .requires_grad_(True)
                                )
                                z2_ref = (
                                    z2_base.contiguous(memory_format=mem_fmt)
                                    .detach()
                                    .requires_grad_(True)
                                )
                                go_ref = go_base.contiguous(memory_format=mem_fmt)
                                out_ref = _reference_subspace_gp(
                                    z1_ref,
                                    z2_ref,
                                    k=k,
                                    cli_mode=cli_mode,
                                    ctx_mode=ctx_mode,
                                    use_fma=True,
                                )
                                out_ref.backward(go_ref)

                                z1 = (
                                    z1_base.contiguous(memory_format=mem_fmt)
                                    .detach()
                                    .requires_grad_(True)
                                )
                                z2 = (
                                    z2_base.contiguous(memory_format=mem_fmt)
                                    .detach()
                                    .requires_grad_(True)
                                )
                                out = subspace_gp_triton(
                                    z1,
                                    z2,
                                    k=k,
                                    cli_mode=cli_mode,
                                    ctx_mode=ctx_mode,
                                    use_fma=True,
                                )
                                self.assertEqual(out.dtype, dtype)
                                _assert_memory_format(self, out, mem_fmt)
                                out.backward(go_ref)
                                _assert_memory_format(self, z1.grad, mem_fmt)
                                _assert_memory_format(self, z2.grad, mem_fmt)

                                fwd_diff = (out.float() - out_ref.float()).abs().max().item()
                                dz1_diff = (
                                    (z1.grad.float() - z1_ref.grad.float()).abs().max().item()
                                )
                                dz2_diff = (
                                    (z2.grad.float() - z2_ref.grad.float()).abs().max().item()
                                )
                                self.assertLess(fwd_diff, fwd_tol)
                                self.assertLess(dz1_diff, bwd_tol)
                                self.assertLess(dz2_diff, bwd_tol)

    def test_wedge_near_cancellation_fma_precision(self) -> None:
        """Verify FMA wedge determinant eliminates catastrophic cancellation vs naive."""
        # Construct nearly collinear vectors u and v where a*b and c*d agree to ~5 digits
        b, d_tilde, h, w = 2, 12, 8, 8
        base = torch.randn(b, d_tilde, h, w, device="cuda", dtype=torch.float32) + 10.0
        eps = 1e-5 * torch.randn(b, d_tilde, h, w, device="cuda", dtype=torch.float32)
        z1_32 = base
        z2_32 = 2.0 * base + eps

        ref_64 = _reference_subspace_gp(
            z1_32.double(), z2_32.double(), k=3, cli_mode="wedge", ctx_mode="diff", use_fma=True
        )
        out_fma = subspace_gp_triton(
            z1_32, z2_32, k=3, cli_mode="wedge", ctx_mode="diff", use_fma=True
        )
        out_no_fma = subspace_gp_triton(
            z1_32, z2_32, k=3, cli_mode="wedge", ctx_mode="diff", use_fma=False
        )
        err_fma = (out_fma.double() - ref_64).abs().max().item()
        err_no_fma = (out_no_fma.double() - ref_64).abs().max().item()
        self.assertLess(err_fma, 1e-6)
        self.assertLess(err_fma, err_no_fma)

    def test_strided_views_and_partial_grad_and_small_spatial(self) -> None:
        """Test non-contiguous strided slices, 1x1 spatial maps, and single-input requires_grad."""
        # 1. Non-contiguous spatial slice + mixed NCHW/NHWC inputs
        z1_full = torch.randn(2, 16, 14, 16, device="cuda", dtype=torch.float32)
        z2_full = torch.randn(2, 16, 14, 16, device="cuda", dtype=torch.float32).contiguous(
            memory_format=torch.channels_last
        )
        z1_sub = z1_full[:, :, ::2, 1::2].detach().requires_grad_(True)
        z2_sub = z2_full[:, :, ::2, 1::2].detach().requires_grad_(False)

        out = subspace_gp_triton(z1_sub, z2_sub, k=4, cli_mode="full", ctx_mode="diff")
        out.sum().backward()  # exercises stride-0 grad_out as well

        z1_ref = z1_sub.detach().double().requires_grad_(True)
        z2_ref = z2_sub.detach().double()
        out_ref = _reference_subspace_gp(z1_ref, z2_ref, k=4, cli_mode="full", ctx_mode="diff")
        out_ref.sum().backward()

        self.assertIsNone(z2_sub.grad)
        self.assertLess((out.double() - out_ref).abs().max().item(), 5e-6)
        self.assertLess((z1_sub.grad.double() - z1_ref.grad).abs().max().item(), 5e-6)

        # 2. 1x1 spatial resolution in both NCHW and channels_last
        for mem_fmt in (torch.contiguous_format, torch.channels_last):
            x1 = (
                torch.randn(4, 12, 1, 1, device="cuda", dtype=torch.float32)
                .contiguous(memory_format=mem_fmt)
                .requires_grad_(True)
            )
            x2 = (
                torch.randn(4, 12, 1, 1, device="cuda", dtype=torch.float32)
                .contiguous(memory_format=mem_fmt)
                .requires_grad_(True)
            )
            y = subspace_gp_triton(x1, x2, k=3, cli_mode="full", ctx_mode="abs")
            _assert_memory_format(self, y, mem_fmt)
            y.sum().backward()
            y_ref = _reference_subspace_gp(
                x1.detach().double(), x2.detach().double(), k=3, cli_mode="full", ctx_mode="abs"
            )
            self.assertLess((y.double() - y_ref).abs().max().item(), 5e-6)

    def test_torch_compile_fullgraph(self) -> None:
        """torch.compile(..., fullgraph=True) traces with zero graph breaks and matches eager."""

        class MiniSubspaceBlock(nn.Module):
            def __init__(self, dim: int = 32, k: int = 4) -> None:
                super().__init__()
                self.k = k
                self.proj_in = nn.Conv2d(dim, 2 * dim, kernel_size=1)
                self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
                out_c = subspace_output_channels(dim, k, "full")
                self.proj_out = nn.Conv2d(out_c, dim, kernel_size=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                z1, z2_0 = self.proj_in(x).chunk(2, dim=1)
                z2 = self.dw(z2_0)
                g = subspace_gp_triton(z1, z2, k=self.k, cli_mode="full", ctx_mode="diff")
                return self.proj_out(g)

        for dtype in (torch.float32, torch.bfloat16):
            for mem_fmt in (torch.contiguous_format, torch.channels_last):
                for k in (2, 3, 4):
                    torch._dynamo.reset()
                    d_tilde = 24
                    fn_compiled = torch.compile(
                        lambda a, b, kk=k: subspace_gp_triton(
                            a, b, k=kk, cli_mode="full", ctx_mode="diff", use_fma=True
                        ),
                        fullgraph=True,
                    )
                    z1_e = (
                        torch.randn(2, d_tilde, 16, 16, device="cuda", dtype=dtype)
                        .contiguous(memory_format=mem_fmt)
                        .requires_grad_(True)
                    )
                    z2_e = (
                        torch.randn(2, d_tilde, 16, 16, device="cuda", dtype=dtype)
                        .contiguous(memory_format=mem_fmt)
                        .requires_grad_(True)
                    )
                    z1_c = z1_e.detach().clone().requires_grad_(True)
                    z2_c = z2_e.detach().clone().requires_grad_(True)

                    out_e = subspace_gp_triton(
                        z1_e, z2_e, k=k, cli_mode="full", ctx_mode="diff", use_fma=True
                    )
                    go = torch.randn_like(out_e)
                    out_e.backward(go)

                    out_c = fn_compiled(z1_c, z2_c)
                    _assert_memory_format(self, out_c, mem_fmt)
                    out_c.backward(go)

                    self.assertTrue(torch.equal(out_c, out_e))
                    self.assertTrue(torch.equal(z1_c.grad, z1_e.grad))
                    self.assertTrue(torch.equal(z2_c.grad, z2_e.grad))

        # Verify end-to-end nn.Module fullgraph compilation with zero graph breaks
        torch._dynamo.reset()
        block = MiniSubspaceBlock(dim=32, k=4).cuda().to(memory_format=torch.channels_last)
        x = (
            torch.randn(2, 32, 16, 16, device="cuda", dtype=torch.bfloat16)
            .contiguous(memory_format=torch.channels_last)
            .requires_grad_(True)
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            explanation = torch._dynamo.explain(block)(x)
        self.assertEqual(explanation.graph_break_count, 0)

        compiled_block = torch.compile(block, fullgraph=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            y = compiled_block(x)
            y.sum().backward()
        self.assertIsNotNone(x.grad)

    def test_large_bg_exceeding_grid_y_limit(self) -> None:
        """Regression test: B*G > 65,535 (CUDA gridY cap) must succeed in NCHW, NHWC, and strided."""
        # Nano Stage 3 shape: B=512, C=832, k=4 -> G=208, B*G=106,496 > 65,535
        for mem_fmt in (torch.contiguous_format, torch.channels_last):
            z1 = (
                torch.randn(512, 832, 7, 7, device="cuda", dtype=torch.bfloat16)
                .contiguous(memory_format=mem_fmt)
                .requires_grad_(True)
            )
            z2 = (
                torch.randn(512, 832, 7, 7, device="cuda", dtype=torch.bfloat16)
                .contiguous(memory_format=mem_fmt)
                .requires_grad_(True)
            )
            y = subspace_gp_triton(z1, z2, k=4, cli_mode="full", ctx_mode="abs", use_fma=True)
            y.sum().backward()
            self.assertTrue(y.isfinite().all().item())
            self.assertIsNotNone(z1.grad)
            self.assertIsNotNone(z2.grad)

        z1_s = torch.randn(512, 832, 7, 7, device="cuda", dtype=torch.bfloat16).transpose(2, 3).requires_grad_(True)
        z2_s = torch.randn(512, 832, 7, 7, device="cuda", dtype=torch.bfloat16).transpose(2, 3).requires_grad_(True)
        y_s = subspace_gp_triton(z1_s, z2_s, k=4, cli_mode="full", ctx_mode="abs", use_fma=True)
        y_s.sum().backward()
        self.assertTrue(y_s.isfinite().all().item())

    def test_peak_extra_memory_recomputation(self) -> None:
        """Peak extra allocated memory at (64, 128, 64, 64) fp32 k=4 full equals ~320 MiB."""
        b, d_tilde, h, w = 64, 128, 64, 64
        z1 = torch.randn(b, d_tilde, h, w, device="cuda", dtype=torch.float32, requires_grad=True)
        z2 = torch.randn(b, d_tilde, h, w, device="cuda", dtype=torch.float32, requires_grad=True)
        grad_out = torch.randn(b, 320, h, w, device="cuda", dtype=torch.float32)
        expected_out_mib = (b * 320 * h * w * 4) / (1024**2)  # 320.0 MiB
        self.assertAlmostEqual(expected_out_mib, 320.0, places=5)

        # Warmup once to ensure Triton kernel compilation is complete
        out_warm = subspace_gp_triton(z1, z2, k=4, cli_mode="full", ctx_mode="diff", use_fma=True)
        out_warm.backward(grad_out)
        del out_warm
        z1.grad = None
        z2.grad = None
        torch.cuda.synchronize()

        base_bytes = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        out = SubspaceGPTritonFn.apply(z1, z2, 4, "full", "diff", True)
        fwd_extra_mib = (torch.cuda.memory_allocated() - base_bytes) / (1024**2)

        # Verify autograd context saved ONLY (z1, z2) and zero intermediate tensors
        sink = _GradSink.apply(out, grad_out)
        del out
        saved_extra_mib = (torch.cuda.memory_allocated() - base_bytes) / (1024**2)

        sink.backward()
        del sink
        torch.cuda.synchronize()
        peak_extra_mib = (torch.cuda.max_memory_allocated() - base_bytes) / (1024**2)

        self.assertAlmostEqual(fwd_extra_mib, 320.0, delta=0.5)
        self.assertAlmostEqual(saved_extra_mib, 0.0, delta=0.01)
        self.assertAlmostEqual(peak_extra_mib, 320.0, delta=0.5)

    def test_input_validation(self) -> None:
        """Invalid k, channel divisibility, modes, or shapes raise ValueError."""
        z1 = torch.randn(2, 10, 4, 4, device="cuda")
        z2 = torch.randn(2, 10, 4, 4, device="cuda")
        with self.assertRaises(ValueError):
            subspace_gp_triton(z1, z2, k=4)  # 10 % 4 != 0
        with self.assertRaises(ValueError):
            subspace_gp_triton(z1, z2, k=5)  # k not in (2, 3, 4)
        z1_ok = torch.randn(2, 12, 4, 4, device="cuda")
        z2_ok = torch.randn(2, 12, 4, 4, device="cuda")
        with self.assertRaises(ValueError):
            subspace_gp_triton(z1_ok, z2_ok, k=4, cli_mode="bad")
        with self.assertRaises(ValueError):
            subspace_gp_triton(z1_ok, z2_ok, k=4, ctx_mode="bad")


if __name__ == "__main__":
    unittest.main(verbosity=2)
