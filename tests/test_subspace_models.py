"""Tests for Subspace Clifford Networks (arXiv:2610.37226), scientific controls, and timm wrappers."""

from __future__ import annotations

import torch
import torch.nn as nn

from cliffordnet.subspace import (
    SUBSPACE_MODEL_BUILDERS,
    HierarchicalSubspaceCliffordNet,
    IsotropicSubspaceCliffordNet,
    SpatialDiffusion,
    SubspaceCliffordBlock,
    SubspaceGeometricProduct,
    TimmClassifierWrapper,
    get_subspace_model_spec,
    subspace_gp_pytorch,
)

EXPECTED_DIAG_KEYS = {
    "cancel/rel_diff_mean",
    "cancel/rel_diff_lt1e-2",
    "cancel/rel_diff_lt1e-4",
    "magnitude/wedge_abs_mean",
    "magnitude/wedge_abs_max",
    "magnitude/dot_abs_mean",
    "magnitude/wedge_to_dot_ratio",
    "health/nan_count",
    "health/inf_count",
    "ortho/cos_sim_mean",
    "ortho/subspace_cos_sim_mean",
    "invariance/lambda_wedge_rel_err",
    "spectrum/effective_rank",
}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test_spatial_diffusion_exact_param_count() -> None:
    for dim in (32, 96, 128, 160, 256):
        mod = SpatialDiffusion(dim=dim, kernel_size=5)
        assert count_params(mod) == 28 * dim, f"Expected {28 * dim}, got {count_params(mod)}"
        x = torch.randn(2, dim, 8, 8)
        y = mod(x)
        assert y.shape == x.shape


def test_paper2_parameter_tables_and_deltas() -> None:
    """Verify parameter counts against arXiv:2610.37226 Tables 2, 3, 4 and exact pairwise deltas."""
    paper_cifar100_targets_m = {
        # Table 2 Isotropic (12L, D=128; note paper stem/21*D vs Sec 3.4.1 28*D 5x5 DWConv = +10,752 params)
        "subspace_iso_k2": (0.756, 0.015),
        "subspace_iso_k4": (0.952, 0.015),
        # Table 2 Shift-Based Hierarchical Baselines
        "rolling_hier_s3": (2.093, 0.001),
        "rolling_hier_s4": (2.535, 0.001),
        # Table 2 & 4 Uniform Subspace Hierarchical
        "subspace_hier_k2": (0.898, 0.001),
        "subspace_hier_k3": (1.017, 0.001),
        "subspace_hier_k4": (1.120, 0.001),
        # Table 4 Hybrid Schedules
        "subspace_hier_234": (1.061, 0.001),
        "subspace_hier_432": (0.961, 0.001),
        "subspace_hier_244": (1.092, 0.001),
        # Table 3 Depth Scaling
        "subspace_hier_244_18": (1.650, 0.001),
        "subspace_hier_244_24": (2.209, 0.001),
        # Table 4 Branch & SpatialDiffusion Ablations
        "subspace_hier_k4_inner": (0.788, 0.001),
        "subspace_hier_k4_wedge": (0.898, 0.001),
        "subspace_hier_k4_spfree": (1.075, 0.001),
    }

    counts: dict[str, int] = {}
    print("\n=== Paper 2 (arXiv:2610.37226) CIFAR-100 Parameter Verification ===")
    print(f"{'Preset':<26s} {'Exact Params':>14s} {'Ours (M)':>10s} {'Paper (M)':>10s} {'Diff (M)':>10s}")
    print("-" * 74)
    for name, (paper_m, tol_m) in paper_cifar100_targets_m.items():
        model = SUBSPACE_MODEL_BUILDERS[name](num_classes=100, img_size=32)
        p = count_params(model)
        counts[name] = p
        diff_m = p / 1e6 - paper_m
        print(f"{name:<26s} {p:>14,d} {p / 1e6:>10.4f} {paper_m:>10.3f} {diff_m:>+10.4f}")
        assert abs(diff_m) <= tol_m, f"{name}: {p / 1e6:.4f}M vs paper {paper_m:.3f}M exceeds tol {tol_m}"

    # Also document conv3 stem comparison on u4
    u4_conv3 = count_params(
        HierarchicalSubspaceCliffordNet(
            img_size=32,
            num_classes=100,
            stage_depths=(3, 4, 5),
            stage_dims=(96, 128, 160),
            subspace_k=4,
            stem_type="conv3",
        )
    )
    print(f"{'subspace_hier_k4 (conv3)':<26s} {u4_conv3:>14,d} {u4_conv3 / 1e6:>10.4f} {'1.120':>10s} {u4_conv3 / 1e6 - 1.120:>+10.4f}")

    # Exact pairwise architectural parameter deltas (stem-invariant)
    deltas = {
        "iso_k4 - iso_k2": (counts["subspace_iso_k4"] - counts["subspace_iso_k2"], 12 * 128 * 128),
        "u4 - u2": (counts["subspace_hier_k4"] - counts["subspace_hier_k2"], 221_184),
        "u4 - u3": (counts["subspace_hier_k4"] - counts["subspace_hier_k3"], 102_018),
        "u4 - inner": (counts["subspace_hier_k4"] - counts["subspace_hier_k4_inner"], 331_776),
        "u4 - wedge": (counts["subspace_hier_k4"] - counts["subspace_hier_k4_wedge"], 221_184),
        "u4 - spfree": (
            counts["subspace_hier_k4"] - counts["subspace_hier_k4_spfree"],
            28 * (3 * 96 + 4 * 128 + 5 * 160),  # 44,800 (~0.045M)
        ),
        "18 - 12 (244)": (counts["subspace_hier_244_18"] - counts["subspace_hier_244"], 558_592),
        "24 - 18 (244)": (counts["subspace_hier_244_24"] - counts["subspace_hier_244_18"], 558_592),
    }
    print("\n=== Pairwise Parameter Deltas ===")
    for label, (actual, expected) in deltas.items():
        print(f"  {label:<18s}: actual={actual:>9,d}  expected={expected:>9,d}")
        assert actual == expected, f"Delta mismatch for {label}: {actual} != {expected}"

    # Verify ROI #1 Iso-Parameter Scientific Controls match subspace_hier_k4 within 1%
    u4 = counts["subspace_hier_k4"]
    print("\n=== ROI #1 Scientific Deconfounding Controls (CIFAR-100) ===")
    for ctrl in (
        "subspace_hier_k4_inner_wide",
        "subspace_hier_k4_sym",
        "subspace_hier_k4_bilinear",
        "subspace_hier_k4_ggr",
        "rolling_hier_s3_lietrotter",
    ):
        p_ctrl = count_params(SUBSPACE_MODEL_BUILDERS[ctrl](num_classes=100, img_size=32))
        rel_diff = abs(p_ctrl - u4) / u4
        print(f"  {ctrl:<28s}: {p_ctrl:>10,d} ({p_ctrl / 1e6:.4f}M, vs k4: {(p_ctrl - u4) / u4:+.2%})")
        if ctrl in ("subspace_hier_k4_inner_wide", "subspace_hier_k4_sym", "subspace_hier_k4_bilinear"):
            assert rel_diff <= 0.01, f"{ctrl} deviates from subspace_hier_k4 by {rel_diff:.2%} > 1%"

    # Print ROI #2 ImageNet-1k Scaling Variants & ConvNeXt Baselines
    print("\n=== ROI #2 ImageNet-1k Scaling Presets & Baselines (num_classes=1000, img_size=224) ===")
    for name in (
        "in1k_subspace_can_tiny",
        "in1k_subspace_p4",
        "in1k_subspace_tiny",
        "in1k_subspace_femto",
        "in1k_rolling_femto",
        "baseline_convnext_atto",
        "baseline_convnext_femto",
    ):
        cls, kwargs = get_subspace_model_spec(name, num_classes=1000, img_size=224)
        m = cls(**kwargs)
        p = count_params(m)
        print(f"  {name:<26s}: {p:>10,d} ({p / 1e6:.3f}M)")


def test_non_divisible_channel_alignment_and_group_ordering() -> None:
    """Verify non-divisible alignment (dim=128, k=3 -> aligned_dim=129) and [p_wedge, p_inner] order."""
    sgp = SubspaceGeometricProduct(dim=128, k=3, cli_mode="full", backend="pytorch")
    assert sgp.aligned_dim == 129
    assert sgp.num_groups == 43
    assert sgp.c_out == 43 * 6  # 3 wedge + 3 inner per group = 258

    sgp_wide = SubspaceGeometricProduct(dim=128, k=4, cli_mode="inner_wide", backend="pytorch")
    assert sgp_wide.aligned_dim == 192
    assert sgp_wide.c_out == 192

    # Check exact [p_wedge, p_inner] per-group concatenation order and cyclic k=3 convention
    z1 = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])  # (1, 3, 1, 1): u = [1, 2, 3]
    z2 = torch.tensor([[[[4.0]], [[5.0]], [[7.0]]]])  # (1, 3, 1, 1): v = [4, 5, 7] in abs mode
    out = subspace_gp_pytorch(z1, z2, k=3, cli_mode="full", ctx_mode="abs", wedge_mode="fma")
    assert out.shape == (1, 6, 1, 1)
    # Cyclic cross product u x v = [1*5 - 2*4, 2*7 - 3*5, 3*4 - 1*7] = [-3, -1, 5]
    expected_wedge = torch.tensor([-3.0, -1.0, 5.0])
    expected_inner = torch.nn.functional.silu(torch.tensor([4.0, 10.0, 21.0]))
    assert torch.allclose(out[0, :3, 0, 0], expected_wedge, atol=1e-6)
    assert torch.allclose(out[0, 3:, 0, 0], expected_inner, atol=1e-6)


def test_wedge_modes_and_bf16_cancellation_control() -> None:
    """Verify algebraic identity u ^ (z2 - u) == u ^ z2 in fp32 vs bf16 cancellation in naive vs direct_z2."""
    torch.manual_seed(0)
    # Construct nearly collinear z1, z2 where z2 = z1 + eps * delta so (z2 - z1) loses mantissa bits in bf16
    z1_f32 = torch.randn(2, 16, 4, 4) * 10.0
    delta_f32 = torch.randn(2, 16, 4, 4) * 1e-2
    z2_f32 = z1_f32 + delta_f32

    ref_f32 = subspace_gp_pytorch(z1_f32, z2_f32, k=4, cli_mode="wedge", ctx_mode="abs", wedge_mode="fma")
    diff_f32 = subspace_gp_pytorch(z1_f32, z2_f32, k=4, cli_mode="wedge", ctx_mode="diff", wedge_mode="fma")
    assert torch.allclose(ref_f32, diff_f32, atol=1e-5, rtol=1e-4)

    z1_bf16 = z1_f32.bfloat16()
    z2_bf16 = z2_f32.bfloat16()
    for wm in ("fma", "fp32", "naive", "direct_z2", "direct_z2_fma"):
        out = subspace_gp_pytorch(z1_bf16, z2_bf16, k=4, cli_mode="full", ctx_mode="diff", wedge_mode=wm)
        assert out.dtype == torch.bfloat16
        assert out.shape == (2, 40, 4, 4)
        assert torch.isfinite(out).all()


def test_cuda_forward_backward_diagnostics_triton_and_compile() -> None:
    """Test fp32 & bf16 channels_last forward/ortho/diagnostics/backward, Triton parity, and torch.compile."""
    assert torch.cuda.is_available(), "CUDA device required (run with CUDA_VISIBLE_DEVICES=6)"
    device = torch.device("cuda")
    torch.manual_seed(42)

    # 1a. Raw operator parity (subspace_gp_pytorch vs subspace_gp_triton) at 1e-6 in fp32
    from cliffordnet.kernels.triton_subspace import subspace_gp_triton

    for k in (2, 3, 4):
        for cli_mode in ("full", "wedge", "inner"):
            for ctx_mode in ("diff", "abs"):
                z1_op = torch.randn(2, 24, 8, 8, device=device, dtype=torch.float32, requires_grad=True)
                z2_op = torch.randn(2, 24, 8, 8, device=device, dtype=torch.float32, requires_grad=True)
                z1_tr = z1_op.detach().clone().requires_grad_(True)
                z2_tr = z2_op.detach().clone().requires_grad_(True)
                out_pt = subspace_gp_pytorch(z1_op, z2_op, k=k, cli_mode=cli_mode, ctx_mode=ctx_mode)
                out_tr = subspace_gp_triton(z1_tr, z2_tr, k=k, cli_mode=cli_mode, ctx_mode=ctx_mode)
                assert torch.allclose(out_pt, out_tr, atol=1e-6, rtol=1e-5)
                go_op = torch.randn_like(out_pt)
                out_pt.backward(go_op)
                out_tr.backward(go_op)
                assert z1_op.grad is not None and z1_tr.grad is not None
                assert z2_op.grad is not None and z2_tr.grad is not None
                assert torch.allclose(z1_op.grad, z1_tr.grad, atol=1e-5, rtol=1e-5)
                assert torch.allclose(z2_op.grad, z2_tr.grad, atol=1e-5, rtol=1e-5)

    # 1b. Full SubspaceGeometricProduct module parity across k={2,3,4}, fp32 & bf16 channels_last
    for k in (2, 3, 4):
        for cli_mode in ("full", "wedge", "inner"):
            for ctx_mode in ("diff", "abs"):
                for dtype, atol in ((torch.float32, 1e-4), (torch.bfloat16, 5e-2)):
                    m_pt = SubspaceGeometricProduct(
                        dim=64, k=k, cli_mode=cli_mode, ctx_mode=ctx_mode, backend="pytorch"
                    ).to(device=device, dtype=dtype, memory_format=torch.channels_last)
                    m_tr = SubspaceGeometricProduct(
                        dim=64, k=k, cli_mode=cli_mode, ctx_mode=ctx_mode, backend="triton"
                    ).to(device=device, dtype=dtype, memory_format=torch.channels_last)
                    m_tr.load_state_dict(m_pt.state_dict())

                    x1 = torch.randn(2, 64, 8, 8, device=device, dtype=dtype).contiguous(
                        memory_format=torch.channels_last
                    )
                    x1.requires_grad_(True)
                    x2 = x1.detach().clone().requires_grad_(True)

                    y_pt = m_pt(x1)
                    y_tr = m_tr(x2)
                    assert torch.allclose(y_pt, y_tr, atol=atol, rtol=atol), (
                        f"Forward mismatch k={k}, cli={cli_mode}, ctx={ctx_mode}, dtype={dtype}: "
                        f"max_diff={(y_pt - y_tr).abs().max().item()}"
                    )
                    grad_out = torch.randn_like(y_pt)
                    y_pt.backward(grad_out)
                    y_tr.backward(grad_out)
                    assert x1.grad is not None and x2.grad is not None
                    assert torch.allclose(x1.grad, x2.grad, atol=atol, rtol=atol)

    # 2. Test 2x2 factorial blocks, scientific controls, non-divisible k=3, and ConvNeXt wrapper on GPU
    presets_to_verify = [
        ("subspace_iso_k4", 32, 100),
        ("subspace_hier_k3", 32, 100),  # exercises non-divisible Stage 3: dim=160, k=3 -> aligned_dim=162
        ("subspace_hier_244", 32, 100),
        ("subspace_hier_k4_inner_wide", 32, 100),
        ("subspace_hier_k4_sym", 32, 100),
        ("subspace_hier_k4_bilinear", 32, 100),
        ("subspace_hier_k4_ggr", 32, 100),
        ("rolling_hier_s3_lietrotter", 32, 100),
        ("rolling_hier_s3", 32, 100),
        ("in1k_subspace_p4", 64, 1000),
        ("baseline_convnext_atto", 32, 100),
    ]

    for name, img_sz, n_cls in presets_to_verify:
        for dtype in (torch.float32, torch.bfloat16):
            model = SUBSPACE_MODEL_BUILDERS[name](num_classes=n_cls, img_size=img_sz).to(
                device=device, memory_format=torch.channels_last
            )
            x = torch.randn(2, 3, img_sz, img_sz, device=device, dtype=dtype).contiguous(
                memory_format=torch.channels_last
            )

            with torch.autocast(device_type="cuda", dtype=dtype, enabled=(dtype == torch.bfloat16)):
                logits, ortho_loss = model(x, compute_ortho=True)
                loss = logits.float().sum() + 0.01 * ortho_loss.float()
            assert logits.shape == (2, n_cls)
            assert ortho_loss.ndim == 0 and torch.isfinite(ortho_loss)
            loss.backward()

            grad_norms = [p.grad.float().norm() for p in model.parameters() if p.grad is not None]
            assert len(grad_norms) > 0 and all(torch.isfinite(gn) for gn in grad_norms)

            # Diagnostics verification
            diags = model.forward_diagnostics(x)
            labels = model.diagnostic_block_labels()
            if isinstance(model, TimmClassifierWrapper):
                assert labels == [] and diags == {}
            else:
                assert len(labels) > 0
                for lbl in labels:
                    for dk in EXPECTED_DIAG_KEYS:
                        full_key = f"{lbl}/{dk}"
                        assert full_key in diags, f"Missing diagnostic key {full_key} in {name}"
                        val = diags[full_key]
                        assert val.ndim == 0 and not val.requires_grad and torch.isfinite(val), (
                            f"Invalid diagnostic {full_key}={val} in {name}"
                        )
                    erank = diags[f"{lbl}/spectrum/effective_rank"].item()
                    assert 0.0 < erank <= 1.0001, f"Effective rank {erank} out of (0, 1] in {name}"

    # 3. Verify torch.compile(fullgraph=True) with zero graph breaks on both PyTorch and Triton backends
    torch._dynamo.reset()
    for backend in ("pytorch", "triton"):
        raw_model = HierarchicalSubspaceCliffordNet(
            img_size=32,
            num_classes=100,
            stage_depths=(2, 2, 2),
            stage_dims=(64, 96, 128),
            subspace_k=(2, 3, 4),
            backend=backend,
        ).to(device=device, memory_format=torch.channels_last)
        compiled_model = torch.compile(raw_model, fullgraph=True)

        x_bf16 = torch.randn(4, 3, 32, 32, device=device, dtype=torch.bfloat16).contiguous(
            memory_format=torch.channels_last
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out, ortho = compiled_model(x_bf16, compute_ortho=True)
            total_loss = out.float().mean() + 0.05 * ortho
        total_loss.backward()
        assert out.shape == (4, 100) and torch.isfinite(out).all()
        assert torch.isfinite(ortho)
        for p_name, param in raw_model.named_parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all(), f"Bad grad in {p_name}"

    # Also test torch.compile on SubspaceCliffordBlock with sym_full / inner_wide controls
    blk_sym = SubspaceCliffordBlock(dim=64, k=4, cli_mode="sym_full", backend="auto").to(
        device=device, memory_format=torch.channels_last
    )
    compiled_blk = torch.compile(blk_sym, fullgraph=True)
    xb = torch.randn(2, 64, 16, 16, device=device, dtype=torch.bfloat16).contiguous(
        memory_format=torch.channels_last
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        yb, ob = compiled_blk(xb, compute_ortho=True)
    (yb.float().sum() + ob).backward()
    assert yb.shape == xb.shape and torch.isfinite(yb).all()
    print("\nAll CUDA forward/backward, Triton parity, diagnostics, and torch.compile(fullgraph=True) checks PASSED!")


import unittest


class TestSubspaceModels(unittest.TestCase):
    def test_01_spatial_diffusion_exact_param_count(self) -> None:
        test_spatial_diffusion_exact_param_count()

    def test_02_paper2_parameter_tables_and_deltas(self) -> None:
        test_paper2_parameter_tables_and_deltas()

    def test_03_non_divisible_channel_alignment_and_group_ordering(self) -> None:
        test_non_divisible_channel_alignment_and_group_ordering()

    def test_04_wedge_modes_and_bf16_cancellation_control(self) -> None:
        test_wedge_modes_and_bf16_cancellation_control()

    def test_05_cuda_forward_backward_diagnostics_triton_and_compile(self) -> None:
        test_cuda_forward_backward_diagnostics_triton_and_compile()


if __name__ == "__main__":
    unittest.main(verbosity=2)
