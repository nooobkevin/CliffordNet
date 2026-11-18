"""Recomputation-based fused Triton kernel for Subspace Clifford Networks.

Implements the subspace geometric product interaction from arXiv:2610.37226
(Sections 3.3.2 & 3.5) with:
- Exact fp32 register upcasting and Kahan/Dekker-style FMA 2x2 determinants,
- Support for k in {2, 3, 4}, cli_mode in {"full", "wedge", "inner"},
  ctx_mode in {"diff", "abs"},
- Support for float32, bfloat16, and float16 in both contiguous NCHW and
  channels_last NHWC (plus arbitrary 4D strided tensors),
- Zero-intermediate-activation recomputation backward pass saving ONLY (z1, z2),
- Full torch.compile (fullgraph=True) compatibility via torch.library.custom_op.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

_SUPPORTED_DTYPES = (torch.float32, torch.bfloat16, torch.float16)
_SUPPORTED_K = (2, 3, 4)
_CLI_MODE_TO_ID = {"full": 0, "wedge": 1, "inner": 2}
_CTX_MODE_TO_ID = {"diff": 0, "abs": 1}


def subspace_output_channels(d_in: int, k: int, cli_mode: str = "full") -> int:
    """Return output channel count D_out for input channel count D_tilde."""
    if k not in _SUPPORTED_K:
        raise ValueError(f"k must be one of {_SUPPORTED_K}, got {k}")
    if d_in <= 0 or d_in % k != 0:
        raise ValueError(f"D_tilde ({d_in}) must be positive and divisible by k ({k})")
    if cli_mode == "full":
        d_k = (k * (k + 1)) // 2
    elif cli_mode == "wedge":
        d_k = (k * (k - 1)) // 2
    elif cli_mode == "inner":
        d_k = k
    else:
        raise ValueError(f"cli_mode must be 'full', 'wedge', or 'inner', got {cli_mode!r}")
    return (d_in // k) * d_k


def _group_out_dim(k: int, cli_mode: str) -> int:
    if cli_mode == "full":
        return (k * (k + 1)) // 2
    if cli_mode == "wedge":
        return (k * (k - 1)) // 2
    if cli_mode == "inner":
        return k
    raise ValueError(f"Unsupported cli_mode: {cli_mode!r}")


def _infer_memory_format(t: torch.Tensor) -> torch.memory_format:
    """Infer whether a 4D tensor uses channels_last or contiguous_format."""
    if t.ndim == 4 and t.is_contiguous(memory_format=torch.channels_last):
        if not t.is_contiguous():
            return torch.channels_last
        # Disambiguate degenerate H == 1, W == 1 when C > 1
        if t.shape[1] > 1 and t.stride(3) == t.shape[1]:
            return torch.channels_last
    return torch.contiguous_format


def _is_dense_nchw(t: torch.Tensor) -> bool:
    """True when 4D tensor has exact packed NCHW strides."""
    b, c, h, w = t.shape
    sb, sc, sh, sw = t.stride()
    return sw == 1 and sh == w and sc == h * w and (b <= 1 or sb == c * h * w)


def _is_dense_nhwc(t: torch.Tensor) -> bool:
    """True when 4D tensor has exact packed NHWC (channels_last) strides."""
    b, c, h, w = t.shape
    sb, sc, sh, sw = t.stride()
    return sc == 1 and (w <= 1 or sw == c) and (h <= 1 or sh == w * c) and (b <= 1 or sb == h * w * c)


def _validate_inputs(
    z1: torch.Tensor,
    z2: torch.Tensor,
    k: int,
    cli_mode: str,
    ctx_mode: str,
    *,
    require_cuda: bool = True,
    allow_fp64: bool = False,
) -> tuple[int, int, int, int, int, int]:
    if z1.ndim != 4 or z2.ndim != 4:
        raise ValueError(f"z1 and z2 must be 4D (B, D_tilde, H, W), got {z1.shape} and {z2.shape}")
    if z1.shape != z2.shape:
        raise ValueError(f"z1 and z2 must have identical shape, got {z1.shape} vs {z2.shape}")
    if z1.device != z2.device:
        raise ValueError(f"z1 and z2 must be on the same device, got {z1.device} vs {z2.device}")
    if require_cuda and not z1.is_cuda:
        raise ValueError(f"subspace_gp_triton requires CUDA tensors, got device={z1.device}")
    if z1.dtype != z2.dtype:
        raise ValueError(f"z1 and z2 must have the same dtype, got {z1.dtype} vs {z2.dtype}")
    valid_dtypes = _SUPPORTED_DTYPES + ((torch.float64,) if allow_fp64 else ())
    if z1.dtype not in valid_dtypes:
        raise ValueError(f"Unsupported dtype {z1.dtype}; expected one of {valid_dtypes}")
    if k not in _SUPPORTED_K:
        raise ValueError(f"k must be in {_SUPPORTED_K}, got {k}")
    if cli_mode not in _CLI_MODE_TO_ID:
        raise ValueError(f"cli_mode must be in {tuple(_CLI_MODE_TO_ID)}, got {cli_mode!r}")
    if ctx_mode not in _CTX_MODE_TO_ID:
        raise ValueError(f"ctx_mode must be in {tuple(_CTX_MODE_TO_ID)}, got {ctx_mode!r}")
    b, d_in, h, w = z1.shape
    if d_in <= 0 or d_in % k != 0:
        raise ValueError(f"Channel dimension D_tilde={d_in} must be positive and divisible by k={k}")
    g = d_in // k
    d_k = _group_out_dim(k, cli_mode)
    return b, g, h, w, d_in, d_k


# ============================================================================
# Pure-PyTorch Reference Implementation (Standalone Testing & Benchmarking)
# ============================================================================


def _det2_torch(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor, use_fma: bool
) -> torch.Tensor:
    if use_fma:
        w = c * d
        f = torch.addcmul(-w, a, b)
        e = torch.addcmul(-w, c, d)
        return f - e
    return a * b - c * d


def _reference_subspace_gp(
    z1: torch.Tensor,
    z2: torch.Tensor,
    k: int,
    cli_mode: str = "full",
    ctx_mode: str = "diff",
    use_fma: bool = True,
) -> torch.Tensor:
    """Pure-PyTorch reference for the Subspace Clifford Geometric Product."""
    b, g, h, w, _, d_k = _validate_inputs(
        z1, z2, k, cli_mode, ctx_mode, require_cuda=False, allow_fp64=True
    )
    mem_fmt = _infer_memory_format(z1)
    if b == 0 or h == 0 or w == 0:
        return torch.empty(
            (b, g * d_k, h, w), device=z1.device, dtype=z1.dtype, memory_format=mem_fmt
        )

    compute_dtype = torch.float64 if z1.dtype == torch.float64 else torch.float32
    u = z1.to(compute_dtype)
    z2_f = z2.to(compute_dtype)
    v = (z2_f - u) if ctx_mode == "diff" else z2_f

    # Reshape into (B, G, k, H, W)
    ug = u.reshape(b, g, k, h, w)
    vg = v.reshape(b, g, k, h, w)

    outputs: list[torch.Tensor] = []

    if cli_mode in ("full", "wedge"):
        if k == 2:
            u0, u1 = ug.unbind(dim=2)
            v0, v1 = vg.unbind(dim=2)
            outputs.append(_det2_torch(u0, v1, u1, v0, use_fma))
        elif k == 3:
            u0, u1, u2 = ug.unbind(dim=2)
            v0, v1, v2 = vg.unbind(dim=2)
            outputs.append(_det2_torch(u0, v1, u1, v0, use_fma))
            outputs.append(_det2_torch(u1, v2, u2, v1, use_fma))
            outputs.append(_det2_torch(u2, v0, u0, v2, use_fma))
        else:  # k == 4
            u0, u1, u2, u3 = ug.unbind(dim=2)
            v0, v1, v2, v3 = vg.unbind(dim=2)
            outputs.append(_det2_torch(u0, v1, u1, v0, use_fma))
            outputs.append(_det2_torch(u0, v2, u2, v0, use_fma))
            outputs.append(_det2_torch(u0, v3, u3, v0, use_fma))
            outputs.append(_det2_torch(u1, v2, u2, v1, use_fma))
            outputs.append(_det2_torch(u1, v3, u3, v1, use_fma))
            outputs.append(_det2_torch(u2, v3, u3, v2, use_fma))

    if cli_mode in ("full", "inner"):
        inner = F.silu(ug * vg)
        if not outputs:
            return inner.reshape(b, g * d_k, h, w).to(
                dtype=z1.dtype, memory_format=mem_fmt
            )
        outputs.extend(inner.unbind(dim=2))

    combined = torch.stack(outputs, dim=2)
    return combined.reshape(b, g * d_k, h, w).to(dtype=z1.dtype, memory_format=mem_fmt)


# ============================================================================
# Triton Device Math Helpers (fp32 registers)
# ============================================================================


@triton.jit
def _det2(a, b, c, d, USE_FMA: tl.constexpr):
    if USE_FMA:
        w = c * d
        f = tl.math.fma(a, b, -w)
        e = tl.math.fma(c, d, -w)
        return f - e
    else:
        return a * b - c * d


@triton.jit
def _silu(p):
    return p * tl.sigmoid(p)


@triton.jit
def _dsilu(p, grad_in):
    sig = tl.sigmoid(p)
    return grad_in * (sig * (1.0 + p * (1.0 - sig)))


# ============================================================================
# Forward Triton Kernels: Specialized NCHW, Specialized NHWC, General Strided
# ============================================================================


@triton.jit
def _subspace_gp_fwd_nchw_kernel(
    z1_ptr,
    z2_ptr,
    out_ptr,
    N_BG,
    HW,
    K: tl.constexpr,
    CLI_MODE: tl.constexpr,  # 0=full, 1=wedge, 2=inner
    CTX_MODE: tl.constexpr,  # 0=diff, 1=abs
    USE_FMA: tl.constexpr,
    D_K: tl.constexpr,
    BLOCK_BG: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    EVEN_HW: tl.constexpr,
    USE_INT64: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_bg = tl.program_id(1)
    if USE_INT64:
        pid_hw = pid_hw.to(tl.int64)
        pid_bg = pid_bg.to(tl.int64)

    if BLOCK_BG == 1:
        bg = pid_bg
        hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
        in_base = bg * (K * HW) + hw
        out_base = bg * (D_K * HW) + hw
        mask = True if EVEN_HW else (hw < HW)
    else:
        bg = pid_bg * BLOCK_BG + tl.arange(0, BLOCK_BG)[:, None]
        hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)[None, :]
        in_base = bg * (K * HW) + hw
        out_base = bg * (D_K * HW) + hw
        mask = (bg < N_BG) & (hw < HW)

    if EVEN_HW and BLOCK_BG == 1:
        u0 = tl.load(z1_ptr + in_base + 0 * HW).to(tl.float32)
        u1 = tl.load(z1_ptr + in_base + 1 * HW).to(tl.float32)
        r0 = tl.load(z2_ptr + in_base + 0 * HW).to(tl.float32)
        r1 = tl.load(z2_ptr + in_base + 1 * HW).to(tl.float32)
    else:
        u0 = tl.load(z1_ptr + in_base + 0 * HW, mask=mask, other=0.0).to(tl.float32)
        u1 = tl.load(z1_ptr + in_base + 1 * HW, mask=mask, other=0.0).to(tl.float32)
        r0 = tl.load(z2_ptr + in_base + 0 * HW, mask=mask, other=0.0).to(tl.float32)
        r1 = tl.load(z2_ptr + in_base + 1 * HW, mask=mask, other=0.0).to(tl.float32)

    v0 = (r0 - u0) if CTX_MODE == 0 else r0
    v1 = (r1 - u1) if CTX_MODE == 0 else r1

    if K >= 3:
        if EVEN_HW and BLOCK_BG == 1:
            u2 = tl.load(z1_ptr + in_base + 2 * HW).to(tl.float32)
            r2 = tl.load(z2_ptr + in_base + 2 * HW).to(tl.float32)
        else:
            u2 = tl.load(z1_ptr + in_base + 2 * HW, mask=mask, other=0.0).to(tl.float32)
            r2 = tl.load(z2_ptr + in_base + 2 * HW, mask=mask, other=0.0).to(tl.float32)
        v2 = (r2 - u2) if CTX_MODE == 0 else r2

    if K == 4:
        if EVEN_HW and BLOCK_BG == 1:
            u3 = tl.load(z1_ptr + in_base + 3 * HW).to(tl.float32)
            r3 = tl.load(z2_ptr + in_base + 3 * HW).to(tl.float32)
        else:
            u3 = tl.load(z1_ptr + in_base + 3 * HW, mask=mask, other=0.0).to(tl.float32)
            r3 = tl.load(z2_ptr + in_base + 3 * HW, mask=mask, other=0.0).to(tl.float32)
        v3 = (r3 - u3) if CTX_MODE == 0 else r3

    # Wedge branch
    if CLI_MODE != 2:
        if K == 2:
            w0 = _det2(u0, v1, u1, v0, USE_FMA)
            if EVEN_HW and BLOCK_BG == 1:
                tl.store(out_ptr + out_base + 0 * HW, w0)
            else:
                tl.store(out_ptr + out_base + 0 * HW, w0, mask=mask)
        elif K == 3:
            w0 = _det2(u0, v1, u1, v0, USE_FMA)
            w1 = _det2(u1, v2, u2, v1, USE_FMA)
            w2 = _det2(u2, v0, u0, v2, USE_FMA)
            if EVEN_HW and BLOCK_BG == 1:
                tl.store(out_ptr + out_base + 0 * HW, w0)
                tl.store(out_ptr + out_base + 1 * HW, w1)
                tl.store(out_ptr + out_base + 2 * HW, w2)
            else:
                tl.store(out_ptr + out_base + 0 * HW, w0, mask=mask)
                tl.store(out_ptr + out_base + 1 * HW, w1, mask=mask)
                tl.store(out_ptr + out_base + 2 * HW, w2, mask=mask)
        else:  # K == 4
            w0 = _det2(u0, v1, u1, v0, USE_FMA)
            w1 = _det2(u0, v2, u2, v0, USE_FMA)
            w2 = _det2(u0, v3, u3, v0, USE_FMA)
            w3 = _det2(u1, v2, u2, v1, USE_FMA)
            w4 = _det2(u1, v3, u3, v1, USE_FMA)
            w5 = _det2(u2, v3, u3, v2, USE_FMA)
            if EVEN_HW and BLOCK_BG == 1:
                tl.store(out_ptr + out_base + 0 * HW, w0)
                tl.store(out_ptr + out_base + 1 * HW, w1)
                tl.store(out_ptr + out_base + 2 * HW, w2)
                tl.store(out_ptr + out_base + 3 * HW, w3)
                tl.store(out_ptr + out_base + 4 * HW, w4)
                tl.store(out_ptr + out_base + 5 * HW, w5)
            else:
                tl.store(out_ptr + out_base + 0 * HW, w0, mask=mask)
                tl.store(out_ptr + out_base + 1 * HW, w1, mask=mask)
                tl.store(out_ptr + out_base + 2 * HW, w2, mask=mask)
                tl.store(out_ptr + out_base + 3 * HW, w3, mask=mask)
                tl.store(out_ptr + out_base + 4 * HW, w4, mask=mask)
                tl.store(out_ptr + out_base + 5 * HW, w5, mask=mask)

    # Inner branch
    if CLI_MODE != 1:
        in_off: tl.constexpr = 0 if CLI_MODE == 2 else ((K * (K - 1)) // 2)
        in0 = _silu(u0 * v0)
        in1 = _silu(u1 * v1)
        if EVEN_HW and BLOCK_BG == 1:
            tl.store(out_ptr + out_base + (in_off + 0) * HW, in0)
            tl.store(out_ptr + out_base + (in_off + 1) * HW, in1)
        else:
            tl.store(out_ptr + out_base + (in_off + 0) * HW, in0, mask=mask)
            tl.store(out_ptr + out_base + (in_off + 1) * HW, in1, mask=mask)
        if K >= 3:
            in2 = _silu(u2 * v2)
            if EVEN_HW and BLOCK_BG == 1:
                tl.store(out_ptr + out_base + (in_off + 2) * HW, in2)
            else:
                tl.store(out_ptr + out_base + (in_off + 2) * HW, in2, mask=mask)
        if K == 4:
            in3 = _silu(u3 * v3)
            if EVEN_HW and BLOCK_BG == 1:
                tl.store(out_ptr + out_base + (in_off + 3) * HW, in3)
            else:
                tl.store(out_ptr + out_base + (in_off + 3) * HW, in3, mask=mask)


@triton.jit
def _subspace_gp_fwd_nhwc_kernel(
    z1_ptr,
    z2_ptr,
    out_ptr,
    M,
    K: tl.constexpr,
    CLI_MODE: tl.constexpr,
    CTX_MODE: tl.constexpr,
    USE_FMA: tl.constexpr,
    D_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    USE_INT64: tl.constexpr,
):
    pid = tl.program_id(0)
    if USE_INT64:
        pid = pid.to(tl.int64)
    m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = True if EVEN_M else (m < M)

    in_base = m * K
    out_base = m * D_K

    if K == 2:
        offs2 = tl.arange(0, 2)[None, :]
        if EVEN_M:
            u_vec = tl.load(z1_ptr + in_base[:, None] + offs2).to(tl.float32)
            r_vec = tl.load(z2_ptr + in_base[:, None] + offs2).to(tl.float32)
        else:
            u_vec = tl.load(
                z1_ptr + in_base[:, None] + offs2, mask=mask[:, None], other=0.0
            ).to(tl.float32)
            r_vec = tl.load(
                z2_ptr + in_base[:, None] + offs2, mask=mask[:, None], other=0.0
            ).to(tl.float32)
        u0, u1 = tl.split(u_vec)
        r0, r1 = tl.split(r_vec)
    elif K == 4:
        offs4 = tl.arange(0, 4)[None, :]
        if EVEN_M:
            u_vec = tl.load(z1_ptr + in_base[:, None] + offs4).to(tl.float32)
            r_vec = tl.load(z2_ptr + in_base[:, None] + offs4).to(tl.float32)
        else:
            u_vec = tl.load(
                z1_ptr + in_base[:, None] + offs4, mask=mask[:, None], other=0.0
            ).to(tl.float32)
            r_vec = tl.load(
                z2_ptr + in_base[:, None] + offs4, mask=mask[:, None], other=0.0
            ).to(tl.float32)
        u_even, u_odd = tl.split(tl.reshape(u_vec, (BLOCK_M, 2, 2)))
        u0, u2 = tl.split(u_even)
        u1, u3 = tl.split(u_odd)
        r_even, r_odd = tl.split(tl.reshape(r_vec, (BLOCK_M, 2, 2)))
        r0, r2 = tl.split(r_even)
        r1, r3 = tl.split(r_odd)
    else:  # K == 3
        if EVEN_M:
            u0 = tl.load(z1_ptr + in_base + 0).to(tl.float32)
            u1 = tl.load(z1_ptr + in_base + 1).to(tl.float32)
            u2 = tl.load(z1_ptr + in_base + 2).to(tl.float32)
            r0 = tl.load(z2_ptr + in_base + 0).to(tl.float32)
            r1 = tl.load(z2_ptr + in_base + 1).to(tl.float32)
            r2 = tl.load(z2_ptr + in_base + 2).to(tl.float32)
        else:
            u0 = tl.load(z1_ptr + in_base + 0, mask=mask, other=0.0).to(tl.float32)
            u1 = tl.load(z1_ptr + in_base + 1, mask=mask, other=0.0).to(tl.float32)
            u2 = tl.load(z1_ptr + in_base + 2, mask=mask, other=0.0).to(tl.float32)
            r0 = tl.load(z2_ptr + in_base + 0, mask=mask, other=0.0).to(tl.float32)
            r1 = tl.load(z2_ptr + in_base + 1, mask=mask, other=0.0).to(tl.float32)
            r2 = tl.load(z2_ptr + in_base + 2, mask=mask, other=0.0).to(tl.float32)

    v0 = (r0 - u0) if CTX_MODE == 0 else r0
    v1 = (r1 - u1) if CTX_MODE == 0 else r1
    if K >= 3:
        v2 = (r2 - u2) if CTX_MODE == 0 else r2
    if K == 4:
        v3 = (r3 - u3) if CTX_MODE == 0 else r3

    offs2_out = tl.arange(0, 2)[None, :]

    if CLI_MODE == 0:  # full
        if K == 2:  # D_K = 3
            w0 = _det2(u0, v1, u1, v0, USE_FMA)
            in0 = _silu(u0 * v0)
            in1 = _silu(u1 * v1)
            if EVEN_M:
                tl.store(out_ptr + out_base + 0, w0)
                tl.store(out_ptr + out_base + 1, in0)
                tl.store(out_ptr + out_base + 2, in1)
            else:
                tl.store(out_ptr + out_base + 0, w0, mask=mask)
                tl.store(out_ptr + out_base + 1, in0, mask=mask)
                tl.store(out_ptr + out_base + 2, in1, mask=mask)
        elif K == 3:  # D_K = 6 (3 x v2 stores)
            w0 = _det2(u0, v1, u1, v0, USE_FMA)
            w1 = _det2(u1, v2, u2, v1, USE_FMA)
            w2 = _det2(u2, v0, u0, v2, USE_FMA)
            in0 = _silu(u0 * v0)
            in1 = _silu(u1 * v1)
            in2 = _silu(u2 * v2)
            p0 = tl.join(w0, w1)
            p1 = tl.join(w2, in0)
            p2 = tl.join(in1, in2)
            if EVEN_M:
                tl.store(out_ptr + out_base[:, None] + 0 + offs2_out, p0)
                tl.store(out_ptr + out_base[:, None] + 2 + offs2_out, p1)
                tl.store(out_ptr + out_base[:, None] + 4 + offs2_out, p2)
            else:
                tl.store(out_ptr + out_base[:, None] + 0 + offs2_out, p0, mask=mask[:, None])
                tl.store(out_ptr + out_base[:, None] + 2 + offs2_out, p1, mask=mask[:, None])
                tl.store(out_ptr + out_base[:, None] + 4 + offs2_out, p2, mask=mask[:, None])
        else:  # K == 4, D_K = 10
            w0 = _det2(u0, v1, u1, v0, USE_FMA)
            w1 = _det2(u0, v2, u2, v0, USE_FMA)
            w2 = _det2(u0, v3, u3, v0, USE_FMA)
            w3 = _det2(u1, v2, u2, v1, USE_FMA)
            w4 = _det2(u1, v3, u3, v1, USE_FMA)
            w5 = _det2(u2, v3, u3, v2, USE_FMA)
            in0 = _silu(u0 * v0)
            in1 = _silu(u1 * v1)
            in2 = _silu(u2 * v2)
            in3 = _silu(u3 * v3)
            if out_ptr.dtype.element_ty == tl.float32:
                # 40B/group exceeds 32B L2 sector: intra-warp shuffle -> stride-1 stores
                z_pad = tl.zeros((BLOCK_M,), dtype=tl.float32)
                v0_4 = tl.reshape(tl.join(tl.join(w0, w2), tl.join(w1, w3)), (BLOCK_M, 4))
                v1_4 = tl.reshape(tl.join(tl.join(w4, in0), tl.join(w5, in1)), (BLOCK_M, 4))
                v2_4 = tl.reshape(tl.join(tl.join(in2, z_pad), tl.join(in3, z_pad)), (BLOCK_M, 4))
                v3_4 = tl.zeros((BLOCK_M, 4), dtype=tl.float32)
                h0 = tl.reshape(tl.permute(tl.join(v0_4, v1_4), (0, 2, 1)), (BLOCK_M, 8))
                h1 = tl.reshape(tl.permute(tl.join(v2_4, v3_4), (0, 2, 1)), (BLOCK_M, 8))
                buf16 = tl.reshape(tl.permute(tl.join(h0, h1), (0, 2, 1)), (BLOCK_M // 32, 32 * 16))

                n_warps_m: tl.constexpr = BLOCK_M // 32
                warp_id = tl.arange(0, n_warps_m)[:, None]
                if USE_INT64:
                    warp_id = warp_id.to(tl.int64)
                warp_out_base = (pid * BLOCK_M + warp_id * 32) * 10
                total_out = M.to(tl.int64) * 10 if USE_INT64 else M * 10

                idx0 = tl.arange(0, 256)[None, :]
                src0 = tl.broadcast_to((idx0 // 10) * 16 + (idx0 % 10), (n_warps_m, 256))
                c0 = tl.gather(buf16, src0, axis=1)
                idx1 = 256 + tl.arange(0, 64)[None, :]
                src1 = tl.broadcast_to((idx1 // 10) * 16 + (idx1 % 10), (n_warps_m, 64))
                c1 = tl.gather(buf16, src1, axis=1)

                if EVEN_M:
                    tl.store(out_ptr + warp_out_base + idx0, c0)
                    tl.store(out_ptr + warp_out_base + idx1, c1)
                else:
                    off0 = warp_out_base + idx0
                    off1 = warp_out_base + idx1
                    tl.store(out_ptr + off0, c0, mask=(off0 < total_out))
                    tl.store(out_ptr + off1, c1, mask=(off1 < total_out))
            else:
                # bf16/fp16: 20B/group <= 32B L2 sector, direct 5 x v2 stores hit ~2.83 TB/s
                p0 = tl.join(w0, w1)
                p1 = tl.join(w2, w3)
                p2 = tl.join(w4, w5)
                p3 = tl.join(in0, in1)
                p4 = tl.join(in2, in3)
                if EVEN_M:
                    tl.store(out_ptr + out_base[:, None] + 0 + offs2_out, p0)
                    tl.store(out_ptr + out_base[:, None] + 2 + offs2_out, p1)
                    tl.store(out_ptr + out_base[:, None] + 4 + offs2_out, p2)
                    tl.store(out_ptr + out_base[:, None] + 6 + offs2_out, p3)
                    tl.store(out_ptr + out_base[:, None] + 8 + offs2_out, p4)
                else:
                    tl.store(out_ptr + out_base[:, None] + 0 + offs2_out, p0, mask=mask[:, None])
                    tl.store(out_ptr + out_base[:, None] + 2 + offs2_out, p1, mask=mask[:, None])
                    tl.store(out_ptr + out_base[:, None] + 4 + offs2_out, p2, mask=mask[:, None])
                    tl.store(out_ptr + out_base[:, None] + 6 + offs2_out, p3, mask=mask[:, None])
                    tl.store(out_ptr + out_base[:, None] + 8 + offs2_out, p4, mask=mask[:, None])
    elif CLI_MODE == 1:  # wedge
        if K == 2:  # D_K = 1
            w0 = _det2(u0, v1, u1, v0, USE_FMA)
            if EVEN_M:
                tl.store(out_ptr + out_base, w0)
            else:
                tl.store(out_ptr + out_base, w0, mask=mask)
        elif K == 3:  # D_K = 3
            w0 = _det2(u0, v1, u1, v0, USE_FMA)
            w1 = _det2(u1, v2, u2, v1, USE_FMA)
            w2 = _det2(u2, v0, u0, v2, USE_FMA)
            if EVEN_M:
                tl.store(out_ptr + out_base + 0, w0)
                tl.store(out_ptr + out_base + 1, w1)
                tl.store(out_ptr + out_base + 2, w2)
            else:
                tl.store(out_ptr + out_base + 0, w0, mask=mask)
                tl.store(out_ptr + out_base + 1, w1, mask=mask)
                tl.store(out_ptr + out_base + 2, w2, mask=mask)
        else:  # K == 4, D_K = 6
            w0 = _det2(u0, v1, u1, v0, USE_FMA)
            w1 = _det2(u0, v2, u2, v0, USE_FMA)
            w2 = _det2(u0, v3, u3, v0, USE_FMA)
            w3 = _det2(u1, v2, u2, v1, USE_FMA)
            w4 = _det2(u1, v3, u3, v1, USE_FMA)
            w5 = _det2(u2, v3, u3, v2, USE_FMA)
            p0 = tl.join(w0, w1)
            p1 = tl.join(w2, w3)
            p2 = tl.join(w4, w5)
            if EVEN_M:
                tl.store(out_ptr + out_base[:, None] + 0 + offs2_out, p0)
                tl.store(out_ptr + out_base[:, None] + 2 + offs2_out, p1)
                tl.store(out_ptr + out_base[:, None] + 4 + offs2_out, p2)
            else:
                tl.store(out_ptr + out_base[:, None] + 0 + offs2_out, p0, mask=mask[:, None])
                tl.store(out_ptr + out_base[:, None] + 2 + offs2_out, p1, mask=mask[:, None])
                tl.store(out_ptr + out_base[:, None] + 4 + offs2_out, p2, mask=mask[:, None])
    else:  # inner (D_K = K)
        in0 = _silu(u0 * v0)
        in1 = _silu(u1 * v1)
        if K == 2:
            vec2 = tl.join(in0, in1)
            if EVEN_M:
                tl.store(out_ptr + out_base[:, None] + offs2_out, vec2)
            else:
                tl.store(out_ptr + out_base[:, None] + offs2_out, vec2, mask=mask[:, None])
        elif K == 3:
            in2 = _silu(u2 * v2)
            if EVEN_M:
                tl.store(out_ptr + out_base + 0, in0)
                tl.store(out_ptr + out_base + 1, in1)
                tl.store(out_ptr + out_base + 2, in2)
            else:
                tl.store(out_ptr + out_base + 0, in0, mask=mask)
                tl.store(out_ptr + out_base + 1, in1, mask=mask)
                tl.store(out_ptr + out_base + 2, in2, mask=mask)
        else:  # K == 4
            in2 = _silu(u2 * v2)
            in3 = _silu(u3 * v3)
            vec4 = tl.reshape(tl.join(tl.join(in0, in2), tl.join(in1, in3)), (BLOCK_M, 4))
            offs4_out = tl.arange(0, 4)[None, :]
            if EVEN_M:
                tl.store(out_ptr + out_base[:, None] + offs4_out, vec4)
            else:
                tl.store(out_ptr + out_base[:, None] + offs4_out, vec4, mask=mask[:, None])


@triton.jit
def _subspace_gp_fwd_strided_kernel(
    z1_ptr,
    z2_ptr,
    out_ptr,
    G,
    H,
    W,
    HW,
    z1_sb,
    z1_sc,
    z1_sh,
    z1_sw,
    z2_sb,
    z2_sc,
    z2_sh,
    z2_sw,
    out_sb,
    out_sc,
    out_sh,
    out_sw,
    K: tl.constexpr,
    CLI_MODE: tl.constexpr,
    CTX_MODE: tl.constexpr,
    USE_FMA: tl.constexpr,
    D_K: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_hw = tl.program_id(0).to(tl.int64)
    bg = tl.program_id(1).to(tl.int64)
    b = bg // G
    g = bg % G

    hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW).to(tl.int64)
    mask = hw < HW
    h_idx = hw // W
    w_idx = hw % W

    c_in = g * K
    c_out = g * D_K

    z1_base = b * z1_sb + c_in * z1_sc + h_idx * z1_sh + w_idx * z1_sw
    z2_base = b * z2_sb + c_in * z2_sc + h_idx * z2_sh + w_idx * z2_sw
    out_base = b * out_sb + c_out * out_sc + h_idx * out_sh + w_idx * out_sw

    u0 = tl.load(z1_ptr + z1_base + 0 * z1_sc, mask=mask, other=0.0).to(tl.float32)
    u1 = tl.load(z1_ptr + z1_base + 1 * z1_sc, mask=mask, other=0.0).to(tl.float32)
    r0 = tl.load(z2_ptr + z2_base + 0 * z2_sc, mask=mask, other=0.0).to(tl.float32)
    r1 = tl.load(z2_ptr + z2_base + 1 * z2_sc, mask=mask, other=0.0).to(tl.float32)
    v0 = (r0 - u0) if CTX_MODE == 0 else r0
    v1 = (r1 - u1) if CTX_MODE == 0 else r1

    if K >= 3:
        u2 = tl.load(z1_ptr + z1_base + 2 * z1_sc, mask=mask, other=0.0).to(tl.float32)
        r2 = tl.load(z2_ptr + z2_base + 2 * z2_sc, mask=mask, other=0.0).to(tl.float32)
        v2 = (r2 - u2) if CTX_MODE == 0 else r2
    if K == 4:
        u3 = tl.load(z1_ptr + z1_base + 3 * z1_sc, mask=mask, other=0.0).to(tl.float32)
        r3 = tl.load(z2_ptr + z2_base + 3 * z2_sc, mask=mask, other=0.0).to(tl.float32)
        v3 = (r3 - u3) if CTX_MODE == 0 else r3

    if CLI_MODE != 2:
        if K == 2:
            tl.store(out_ptr + out_base + 0 * out_sc, _det2(u0, v1, u1, v0, USE_FMA), mask=mask)
        elif K == 3:
            tl.store(out_ptr + out_base + 0 * out_sc, _det2(u0, v1, u1, v0, USE_FMA), mask=mask)
            tl.store(out_ptr + out_base + 1 * out_sc, _det2(u1, v2, u2, v1, USE_FMA), mask=mask)
            tl.store(out_ptr + out_base + 2 * out_sc, _det2(u2, v0, u0, v2, USE_FMA), mask=mask)
        else:
            tl.store(out_ptr + out_base + 0 * out_sc, _det2(u0, v1, u1, v0, USE_FMA), mask=mask)
            tl.store(out_ptr + out_base + 1 * out_sc, _det2(u0, v2, u2, v0, USE_FMA), mask=mask)
            tl.store(out_ptr + out_base + 2 * out_sc, _det2(u0, v3, u3, v0, USE_FMA), mask=mask)
            tl.store(out_ptr + out_base + 3 * out_sc, _det2(u1, v2, u2, v1, USE_FMA), mask=mask)
            tl.store(out_ptr + out_base + 4 * out_sc, _det2(u1, v3, u3, v1, USE_FMA), mask=mask)
            tl.store(out_ptr + out_base + 5 * out_sc, _det2(u2, v3, u3, v2, USE_FMA), mask=mask)

    if CLI_MODE != 1:
        in_off: tl.constexpr = 0 if CLI_MODE == 2 else ((K * (K - 1)) // 2)
        tl.store(out_ptr + out_base + (in_off + 0) * out_sc, _silu(u0 * v0), mask=mask)
        tl.store(out_ptr + out_base + (in_off + 1) * out_sc, _silu(u1 * v1), mask=mask)
        if K >= 3:
            tl.store(out_ptr + out_base + (in_off + 2) * out_sc, _silu(u2 * v2), mask=mask)
        if K == 4:
            tl.store(out_ptr + out_base + (in_off + 3) * out_sc, _silu(u3 * v3), mask=mask)


# ============================================================================
# Recomputation Backward Triton Kernels: NCHW, NHWC, General Strided
# ============================================================================


@triton.jit
def _bwd_k2_math(u0, u1, v0, v1, gw0, gi0, gi1, CLI_MODE: tl.constexpr, CTX_MODE: tl.constexpr):
    if CLI_MODE == 0:  # full
        dp0 = _dsilu(u0 * v0, gi0)
        dp1 = _dsilu(u1 * v1, gi1)
        du0 = dp0 * v0 + gw0 * v1
        du1 = dp1 * v1 - gw0 * v0
        dv0 = dp0 * u0 - gw0 * u1
        dv1 = dp1 * u1 + gw0 * u0
    elif CLI_MODE == 1:  # wedge
        du0 = gw0 * v1
        du1 = -gw0 * v0
        dv0 = -gw0 * u1
        dv1 = gw0 * u0
    else:  # inner
        dp0 = _dsilu(u0 * v0, gi0)
        dp1 = _dsilu(u1 * v1, gi1)
        du0 = dp0 * v0
        du1 = dp1 * v1
        dv0 = dp0 * u0
        dv1 = dp1 * u1

    if CTX_MODE == 0:  # diff: v = z2 - z1 -> dz1 = du - dv, dz2 = dv
        dz1_0 = du0 - dv0
        dz1_1 = du1 - dv1
    else:
        dz1_0 = du0
        dz1_1 = du1
    return dz1_0, dz1_1, dv0, dv1


@triton.jit
def _bwd_k3_math(
    u0,
    u1,
    u2,
    v0,
    v1,
    v2,
    gw0,
    gw1,
    gw2,
    gi0,
    gi1,
    gi2,
    CLI_MODE: tl.constexpr,
    CTX_MODE: tl.constexpr,
    USE_FMA: tl.constexpr,
):
    if CLI_MODE == 0:  # full
        dp0 = _dsilu(u0 * v0, gi0)
        dp1 = _dsilu(u1 * v1, gi1)
        dp2 = _dsilu(u2 * v2, gi2)
        du0 = dp0 * v0 + _det2(gw0, v1, gw2, v2, USE_FMA)
        du1 = dp1 * v1 + _det2(gw1, v2, gw0, v0, USE_FMA)
        du2 = dp2 * v2 + _det2(gw2, v0, gw1, v1, USE_FMA)
        dv0 = dp0 * u0 + _det2(gw2, u2, gw0, u1, USE_FMA)
        dv1 = dp1 * u1 + _det2(gw0, u0, gw1, u2, USE_FMA)
        dv2 = dp2 * u2 + _det2(gw1, u1, gw2, u0, USE_FMA)
    elif CLI_MODE == 1:  # wedge
        du0 = _det2(gw0, v1, gw2, v2, USE_FMA)
        du1 = _det2(gw1, v2, gw0, v0, USE_FMA)
        du2 = _det2(gw2, v0, gw1, v1, USE_FMA)
        dv0 = _det2(gw2, u2, gw0, u1, USE_FMA)
        dv1 = _det2(gw0, u0, gw1, u2, USE_FMA)
        dv2 = _det2(gw1, u1, gw2, u0, USE_FMA)
    else:  # inner
        dp0 = _dsilu(u0 * v0, gi0)
        dp1 = _dsilu(u1 * v1, gi1)
        dp2 = _dsilu(u2 * v2, gi2)
        du0 = dp0 * v0
        du1 = dp1 * v1
        du2 = dp2 * v2
        dv0 = dp0 * u0
        dv1 = dp1 * u1
        dv2 = dp2 * u2

    if CTX_MODE == 0:
        dz1_0 = du0 - dv0
        dz1_1 = du1 - dv1
        dz1_2 = du2 - dv2
    else:
        dz1_0 = du0
        dz1_1 = du1
        dz1_2 = du2
    return dz1_0, dz1_1, dz1_2, dv0, dv1, dv2


@triton.jit
def _bwd_k4_math(
    u0,
    u1,
    u2,
    u3,
    v0,
    v1,
    v2,
    v3,
    gw0,
    gw1,
    gw2,
    gw3,
    gw4,
    gw5,
    gi0,
    gi1,
    gi2,
    gi3,
    CLI_MODE: tl.constexpr,
    CTX_MODE: tl.constexpr,
):
    if CLI_MODE == 0:  # full
        dp0 = _dsilu(u0 * v0, gi0)
        dp1 = _dsilu(u1 * v1, gi1)
        dp2 = _dsilu(u2 * v2, gi2)
        dp3 = _dsilu(u3 * v3, gi3)
        du0 = dp0 * v0 + gw0 * v1 + gw1 * v2 + gw2 * v3
        du1 = dp1 * v1 - gw0 * v0 + gw3 * v2 + gw4 * v3
        du2 = dp2 * v2 - gw1 * v0 - gw3 * v1 + gw5 * v3
        du3 = dp3 * v3 - gw2 * v0 - gw4 * v1 - gw5 * v2
        dv0 = dp0 * u0 - gw0 * u1 - gw1 * u2 - gw2 * u3
        dv1 = dp1 * u1 + gw0 * u0 - gw3 * u2 - gw4 * u3
        dv2 = dp2 * u2 + gw1 * u0 + gw3 * u1 - gw5 * u3
        dv3 = dp3 * u3 + gw2 * u0 + gw4 * u1 + gw5 * u2
    elif CLI_MODE == 1:  # wedge
        du0 = gw0 * v1 + gw1 * v2 + gw2 * v3
        du1 = -gw0 * v0 + gw3 * v2 + gw4 * v3
        du2 = -gw1 * v0 - gw3 * v1 + gw5 * v3
        du3 = -gw2 * v0 - gw4 * v1 - gw5 * v2
        dv0 = -gw0 * u1 - gw1 * u2 - gw2 * u3
        dv1 = gw0 * u0 - gw3 * u2 - gw4 * u3
        dv2 = gw1 * u0 + gw3 * u1 - gw5 * u3
        dv3 = gw2 * u0 + gw4 * u1 + gw5 * u2
    else:  # inner
        dp0 = _dsilu(u0 * v0, gi0)
        dp1 = _dsilu(u1 * v1, gi1)
        dp2 = _dsilu(u2 * v2, gi2)
        dp3 = _dsilu(u3 * v3, gi3)
        du0 = dp0 * v0
        du1 = dp1 * v1
        du2 = dp2 * v2
        du3 = dp3 * v3
        dv0 = dp0 * u0
        dv1 = dp1 * u1
        dv2 = dp2 * u2
        dv3 = dp3 * u3

    if CTX_MODE == 0:
        dz1_0 = du0 - dv0
        dz1_1 = du1 - dv1
        dz1_2 = du2 - dv2
        dz1_3 = du3 - dv3
    else:
        dz1_0 = du0
        dz1_1 = du1
        dz1_2 = du2
        dz1_3 = du3
    return dz1_0, dz1_1, dz1_2, dz1_3, dv0, dv1, dv2, dv3


@triton.jit
def _subspace_gp_bwd_nchw_kernel(
    grad_out_ptr,
    z1_ptr,
    z2_ptr,
    dz1_ptr,
    dz2_ptr,
    N_BG,
    HW,
    K: tl.constexpr,
    CLI_MODE: tl.constexpr,
    CTX_MODE: tl.constexpr,
    USE_FMA: tl.constexpr,
    D_K: tl.constexpr,
    NEED_DZ1: tl.constexpr,
    NEED_DZ2: tl.constexpr,
    BLOCK_BG: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    EVEN_HW: tl.constexpr,
    USE_INT64: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_bg = tl.program_id(1)
    if USE_INT64:
        pid_hw = pid_hw.to(tl.int64)
        pid_bg = pid_bg.to(tl.int64)

    if BLOCK_BG == 1:
        bg = pid_bg
        hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
        in_base = bg * (K * HW) + hw
        out_base = bg * (D_K * HW) + hw
        mask = True if EVEN_HW else (hw < HW)
    else:
        bg = pid_bg * BLOCK_BG + tl.arange(0, BLOCK_BG)[:, None]
        hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)[None, :]
        in_base = bg * (K * HW) + hw
        out_base = bg * (D_K * HW) + hw
        mask = (bg < N_BG) & (hw < HW)

    if EVEN_HW and BLOCK_BG == 1:
        u0 = tl.load(z1_ptr + in_base + 0 * HW).to(tl.float32)
        u1 = tl.load(z1_ptr + in_base + 1 * HW).to(tl.float32)
        r0 = tl.load(z2_ptr + in_base + 0 * HW).to(tl.float32)
        r1 = tl.load(z2_ptr + in_base + 1 * HW).to(tl.float32)
    else:
        u0 = tl.load(z1_ptr + in_base + 0 * HW, mask=mask, other=0.0).to(tl.float32)
        u1 = tl.load(z1_ptr + in_base + 1 * HW, mask=mask, other=0.0).to(tl.float32)
        r0 = tl.load(z2_ptr + in_base + 0 * HW, mask=mask, other=0.0).to(tl.float32)
        r1 = tl.load(z2_ptr + in_base + 1 * HW, mask=mask, other=0.0).to(tl.float32)
    v0 = (r0 - u0) if CTX_MODE == 0 else r0
    v1 = (r1 - u1) if CTX_MODE == 0 else r1

    if K == 2:
        gw0 = 0.0
        gi0 = 0.0
        gi1 = 0.0
        if CLI_MODE != 2:
            gw0 = (
                tl.load(grad_out_ptr + out_base + 0 * HW).to(tl.float32)
                if (EVEN_HW and BLOCK_BG == 1)
                else tl.load(grad_out_ptr + out_base + 0 * HW, mask=mask, other=0.0).to(tl.float32)
            )
        if CLI_MODE != 1:
            in_off: tl.constexpr = 0 if CLI_MODE == 2 else 1
            if EVEN_HW and BLOCK_BG == 1:
                gi0 = tl.load(grad_out_ptr + out_base + (in_off + 0) * HW).to(tl.float32)
                gi1 = tl.load(grad_out_ptr + out_base + (in_off + 1) * HW).to(tl.float32)
            else:
                gi0 = tl.load(
                    grad_out_ptr + out_base + (in_off + 0) * HW, mask=mask, other=0.0
                ).to(tl.float32)
                gi1 = tl.load(
                    grad_out_ptr + out_base + (in_off + 1) * HW, mask=mask, other=0.0
                ).to(tl.float32)
        dz1_0, dz1_1, dz2_0, dz2_1 = _bwd_k2_math(
            u0, u1, v0, v1, gw0, gi0, gi1, CLI_MODE, CTX_MODE
        )
        if NEED_DZ1:
            if EVEN_HW and BLOCK_BG == 1:
                tl.store(dz1_ptr + in_base + 0 * HW, dz1_0)
                tl.store(dz1_ptr + in_base + 1 * HW, dz1_1)
            else:
                tl.store(dz1_ptr + in_base + 0 * HW, dz1_0, mask=mask)
                tl.store(dz1_ptr + in_base + 1 * HW, dz1_1, mask=mask)
        if NEED_DZ2:
            if EVEN_HW and BLOCK_BG == 1:
                tl.store(dz2_ptr + in_base + 0 * HW, dz2_0)
                tl.store(dz2_ptr + in_base + 1 * HW, dz2_1)
            else:
                tl.store(dz2_ptr + in_base + 0 * HW, dz2_0, mask=mask)
                tl.store(dz2_ptr + in_base + 1 * HW, dz2_1, mask=mask)

    elif K == 3:
        if EVEN_HW and BLOCK_BG == 1:
            u2 = tl.load(z1_ptr + in_base + 2 * HW).to(tl.float32)
            r2 = tl.load(z2_ptr + in_base + 2 * HW).to(tl.float32)
        else:
            u2 = tl.load(z1_ptr + in_base + 2 * HW, mask=mask, other=0.0).to(tl.float32)
            r2 = tl.load(z2_ptr + in_base + 2 * HW, mask=mask, other=0.0).to(tl.float32)
        v2 = (r2 - u2) if CTX_MODE == 0 else r2

        gw0 = 0.0
        gw1 = 0.0
        gw2 = 0.0
        gi0 = 0.0
        gi1 = 0.0
        gi2 = 0.0
        if CLI_MODE != 2:
            if EVEN_HW and BLOCK_BG == 1:
                gw0 = tl.load(grad_out_ptr + out_base + 0 * HW).to(tl.float32)
                gw1 = tl.load(grad_out_ptr + out_base + 1 * HW).to(tl.float32)
                gw2 = tl.load(grad_out_ptr + out_base + 2 * HW).to(tl.float32)
            else:
                gw0 = tl.load(grad_out_ptr + out_base + 0 * HW, mask=mask, other=0.0).to(tl.float32)
                gw1 = tl.load(grad_out_ptr + out_base + 1 * HW, mask=mask, other=0.0).to(tl.float32)
                gw2 = tl.load(grad_out_ptr + out_base + 2 * HW, mask=mask, other=0.0).to(tl.float32)
        if CLI_MODE != 1:
            in_off: tl.constexpr = 0 if CLI_MODE == 2 else 3
            if EVEN_HW and BLOCK_BG == 1:
                gi0 = tl.load(grad_out_ptr + out_base + (in_off + 0) * HW).to(tl.float32)
                gi1 = tl.load(grad_out_ptr + out_base + (in_off + 1) * HW).to(tl.float32)
                gi2 = tl.load(grad_out_ptr + out_base + (in_off + 2) * HW).to(tl.float32)
            else:
                gi0 = tl.load(
                    grad_out_ptr + out_base + (in_off + 0) * HW, mask=mask, other=0.0
                ).to(tl.float32)
                gi1 = tl.load(
                    grad_out_ptr + out_base + (in_off + 1) * HW, mask=mask, other=0.0
                ).to(tl.float32)
                gi2 = tl.load(
                    grad_out_ptr + out_base + (in_off + 2) * HW, mask=mask, other=0.0
                ).to(tl.float32)

        dz1_0, dz1_1, dz1_2, dz2_0, dz2_1, dz2_2 = _bwd_k3_math(
            u0, u1, u2, v0, v1, v2, gw0, gw1, gw2, gi0, gi1, gi2, CLI_MODE, CTX_MODE, USE_FMA
        )
        if NEED_DZ1:
            if EVEN_HW and BLOCK_BG == 1:
                tl.store(dz1_ptr + in_base + 0 * HW, dz1_0)
                tl.store(dz1_ptr + in_base + 1 * HW, dz1_1)
                tl.store(dz1_ptr + in_base + 2 * HW, dz1_2)
            else:
                tl.store(dz1_ptr + in_base + 0 * HW, dz1_0, mask=mask)
                tl.store(dz1_ptr + in_base + 1 * HW, dz1_1, mask=mask)
                tl.store(dz1_ptr + in_base + 2 * HW, dz1_2, mask=mask)
        if NEED_DZ2:
            if EVEN_HW and BLOCK_BG == 1:
                tl.store(dz2_ptr + in_base + 0 * HW, dz2_0)
                tl.store(dz2_ptr + in_base + 1 * HW, dz2_1)
                tl.store(dz2_ptr + in_base + 2 * HW, dz2_2)
            else:
                tl.store(dz2_ptr + in_base + 0 * HW, dz2_0, mask=mask)
                tl.store(dz2_ptr + in_base + 1 * HW, dz2_1, mask=mask)
                tl.store(dz2_ptr + in_base + 2 * HW, dz2_2, mask=mask)

    else:  # K == 4
        if EVEN_HW and BLOCK_BG == 1:
            u2 = tl.load(z1_ptr + in_base + 2 * HW).to(tl.float32)
            u3 = tl.load(z1_ptr + in_base + 3 * HW).to(tl.float32)
            r2 = tl.load(z2_ptr + in_base + 2 * HW).to(tl.float32)
            r3 = tl.load(z2_ptr + in_base + 3 * HW).to(tl.float32)
        else:
            u2 = tl.load(z1_ptr + in_base + 2 * HW, mask=mask, other=0.0).to(tl.float32)
            u3 = tl.load(z1_ptr + in_base + 3 * HW, mask=mask, other=0.0).to(tl.float32)
            r2 = tl.load(z2_ptr + in_base + 2 * HW, mask=mask, other=0.0).to(tl.float32)
            r3 = tl.load(z2_ptr + in_base + 3 * HW, mask=mask, other=0.0).to(tl.float32)
        v2 = (r2 - u2) if CTX_MODE == 0 else r2
        v3 = (r3 - u3) if CTX_MODE == 0 else r3

        gw0 = 0.0
        gw1 = 0.0
        gw2 = 0.0
        gw3 = 0.0
        gw4 = 0.0
        gw5 = 0.0
        gi0 = 0.0
        gi1 = 0.0
        gi2 = 0.0
        gi3 = 0.0
        if CLI_MODE != 2:
            if EVEN_HW and BLOCK_BG == 1:
                gw0 = tl.load(grad_out_ptr + out_base + 0 * HW).to(tl.float32)
                gw1 = tl.load(grad_out_ptr + out_base + 1 * HW).to(tl.float32)
                gw2 = tl.load(grad_out_ptr + out_base + 2 * HW).to(tl.float32)
                gw3 = tl.load(grad_out_ptr + out_base + 3 * HW).to(tl.float32)
                gw4 = tl.load(grad_out_ptr + out_base + 4 * HW).to(tl.float32)
                gw5 = tl.load(grad_out_ptr + out_base + 5 * HW).to(tl.float32)
            else:
                gw0 = tl.load(grad_out_ptr + out_base + 0 * HW, mask=mask, other=0.0).to(tl.float32)
                gw1 = tl.load(grad_out_ptr + out_base + 1 * HW, mask=mask, other=0.0).to(tl.float32)
                gw2 = tl.load(grad_out_ptr + out_base + 2 * HW, mask=mask, other=0.0).to(tl.float32)
                gw3 = tl.load(grad_out_ptr + out_base + 3 * HW, mask=mask, other=0.0).to(tl.float32)
                gw4 = tl.load(grad_out_ptr + out_base + 4 * HW, mask=mask, other=0.0).to(tl.float32)
                gw5 = tl.load(grad_out_ptr + out_base + 5 * HW, mask=mask, other=0.0).to(tl.float32)
        if CLI_MODE != 1:
            in_off: tl.constexpr = 0 if CLI_MODE == 2 else 6
            if EVEN_HW and BLOCK_BG == 1:
                gi0 = tl.load(grad_out_ptr + out_base + (in_off + 0) * HW).to(tl.float32)
                gi1 = tl.load(grad_out_ptr + out_base + (in_off + 1) * HW).to(tl.float32)
                gi2 = tl.load(grad_out_ptr + out_base + (in_off + 2) * HW).to(tl.float32)
                gi3 = tl.load(grad_out_ptr + out_base + (in_off + 3) * HW).to(tl.float32)
            else:
                gi0 = tl.load(
                    grad_out_ptr + out_base + (in_off + 0) * HW, mask=mask, other=0.0
                ).to(tl.float32)
                gi1 = tl.load(
                    grad_out_ptr + out_base + (in_off + 1) * HW, mask=mask, other=0.0
                ).to(tl.float32)
                gi2 = tl.load(
                    grad_out_ptr + out_base + (in_off + 2) * HW, mask=mask, other=0.0
                ).to(tl.float32)
                gi3 = tl.load(
                    grad_out_ptr + out_base + (in_off + 3) * HW, mask=mask, other=0.0
                ).to(tl.float32)

        dz1_0, dz1_1, dz1_2, dz1_3, dz2_0, dz2_1, dz2_2, dz2_3 = _bwd_k4_math(
            u0,
            u1,
            u2,
            u3,
            v0,
            v1,
            v2,
            v3,
            gw0,
            gw1,
            gw2,
            gw3,
            gw4,
            gw5,
            gi0,
            gi1,
            gi2,
            gi3,
            CLI_MODE,
            CTX_MODE,
        )
        if NEED_DZ1:
            if EVEN_HW and BLOCK_BG == 1:
                tl.store(dz1_ptr + in_base + 0 * HW, dz1_0)
                tl.store(dz1_ptr + in_base + 1 * HW, dz1_1)
                tl.store(dz1_ptr + in_base + 2 * HW, dz1_2)
                tl.store(dz1_ptr + in_base + 3 * HW, dz1_3)
            else:
                tl.store(dz1_ptr + in_base + 0 * HW, dz1_0, mask=mask)
                tl.store(dz1_ptr + in_base + 1 * HW, dz1_1, mask=mask)
                tl.store(dz1_ptr + in_base + 2 * HW, dz1_2, mask=mask)
                tl.store(dz1_ptr + in_base + 3 * HW, dz1_3, mask=mask)
        if NEED_DZ2:
            if EVEN_HW and BLOCK_BG == 1:
                tl.store(dz2_ptr + in_base + 0 * HW, dz2_0)
                tl.store(dz2_ptr + in_base + 1 * HW, dz2_1)
                tl.store(dz2_ptr + in_base + 2 * HW, dz2_2)
                tl.store(dz2_ptr + in_base + 3 * HW, dz2_3)
            else:
                tl.store(dz2_ptr + in_base + 0 * HW, dz2_0, mask=mask)
                tl.store(dz2_ptr + in_base + 1 * HW, dz2_1, mask=mask)
                tl.store(dz2_ptr + in_base + 2 * HW, dz2_2, mask=mask)
                tl.store(dz2_ptr + in_base + 3 * HW, dz2_3, mask=mask)


@triton.jit
def _subspace_gp_bwd_nhwc_kernel(
    grad_out_ptr,
    z1_ptr,
    z2_ptr,
    dz1_ptr,
    dz2_ptr,
    M,
    K: tl.constexpr,
    CLI_MODE: tl.constexpr,
    CTX_MODE: tl.constexpr,
    USE_FMA: tl.constexpr,
    D_K: tl.constexpr,
    NEED_DZ1: tl.constexpr,
    NEED_DZ2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    USE_INT64: tl.constexpr,
):
    pid = tl.program_id(0)
    if USE_INT64:
        pid = pid.to(tl.int64)
    m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = True if EVEN_M else (m < M)

    in_base = m * K
    out_base = m * D_K
    offs2 = tl.arange(0, 2)[None, :]
    offs4 = tl.arange(0, 4)[None, :]

    if K == 2:
        if EVEN_M:
            u_vec = tl.load(z1_ptr + in_base[:, None] + offs2).to(tl.float32)
            r_vec = tl.load(z2_ptr + in_base[:, None] + offs2).to(tl.float32)
        else:
            u_vec = tl.load(
                z1_ptr + in_base[:, None] + offs2, mask=mask[:, None], other=0.0
            ).to(tl.float32)
            r_vec = tl.load(
                z2_ptr + in_base[:, None] + offs2, mask=mask[:, None], other=0.0
            ).to(tl.float32)
        u0, u1 = tl.split(u_vec)
        r0, r1 = tl.split(r_vec)
        v0 = (r0 - u0) if CTX_MODE == 0 else r0
        v1 = (r1 - u1) if CTX_MODE == 0 else r1

        gw0 = 0.0
        gi0 = 0.0
        gi1 = 0.0
        if CLI_MODE == 0:  # D_K = 3
            if EVEN_M:
                gw0 = tl.load(grad_out_ptr + out_base + 0).to(tl.float32)
                gi0 = tl.load(grad_out_ptr + out_base + 1).to(tl.float32)
                gi1 = tl.load(grad_out_ptr + out_base + 2).to(tl.float32)
            else:
                gw0 = tl.load(grad_out_ptr + out_base + 0, mask=mask, other=0.0).to(tl.float32)
                gi0 = tl.load(grad_out_ptr + out_base + 1, mask=mask, other=0.0).to(tl.float32)
                gi1 = tl.load(grad_out_ptr + out_base + 2, mask=mask, other=0.0).to(tl.float32)
        elif CLI_MODE == 1:  # D_K = 1
            gw0 = (
                tl.load(grad_out_ptr + out_base).to(tl.float32)
                if EVEN_M
                else tl.load(grad_out_ptr + out_base, mask=mask, other=0.0).to(tl.float32)
            )
        else:  # D_K = 2
            g_vec = (
                tl.load(grad_out_ptr + out_base[:, None] + offs2).to(tl.float32)
                if EVEN_M
                else tl.load(
                    grad_out_ptr + out_base[:, None] + offs2, mask=mask[:, None], other=0.0
                ).to(tl.float32)
            )
            gi0, gi1 = tl.split(g_vec)

        dz1_0, dz1_1, dz2_0, dz2_1 = _bwd_k2_math(
            u0, u1, v0, v1, gw0, gi0, gi1, CLI_MODE, CTX_MODE
        )
        if NEED_DZ1:
            dz1_vec = tl.join(dz1_0, dz1_1)
            if EVEN_M:
                tl.store(dz1_ptr + in_base[:, None] + offs2, dz1_vec)
            else:
                tl.store(dz1_ptr + in_base[:, None] + offs2, dz1_vec, mask=mask[:, None])
        if NEED_DZ2:
            dz2_vec = tl.join(dz2_0, dz2_1)
            if EVEN_M:
                tl.store(dz2_ptr + in_base[:, None] + offs2, dz2_vec)
            else:
                tl.store(dz2_ptr + in_base[:, None] + offs2, dz2_vec, mask=mask[:, None])

    elif K == 3:
        if EVEN_M:
            u0 = tl.load(z1_ptr + in_base + 0).to(tl.float32)
            u1 = tl.load(z1_ptr + in_base + 1).to(tl.float32)
            u2 = tl.load(z1_ptr + in_base + 2).to(tl.float32)
            r0 = tl.load(z2_ptr + in_base + 0).to(tl.float32)
            r1 = tl.load(z2_ptr + in_base + 1).to(tl.float32)
            r2 = tl.load(z2_ptr + in_base + 2).to(tl.float32)
        else:
            u0 = tl.load(z1_ptr + in_base + 0, mask=mask, other=0.0).to(tl.float32)
            u1 = tl.load(z1_ptr + in_base + 1, mask=mask, other=0.0).to(tl.float32)
            u2 = tl.load(z1_ptr + in_base + 2, mask=mask, other=0.0).to(tl.float32)
            r0 = tl.load(z2_ptr + in_base + 0, mask=mask, other=0.0).to(tl.float32)
            r1 = tl.load(z2_ptr + in_base + 1, mask=mask, other=0.0).to(tl.float32)
            r2 = tl.load(z2_ptr + in_base + 2, mask=mask, other=0.0).to(tl.float32)
        v0 = (r0 - u0) if CTX_MODE == 0 else r0
        v1 = (r1 - u1) if CTX_MODE == 0 else r1
        v2 = (r2 - u2) if CTX_MODE == 0 else r2

        gw0 = 0.0
        gw1 = 0.0
        gw2 = 0.0
        gi0 = 0.0
        gi1 = 0.0
        gi2 = 0.0
        if CLI_MODE == 0:  # D_K = 6 (3 x v2 loads)
            if EVEN_M:
                g01 = tl.load(grad_out_ptr + out_base[:, None] + 0 + offs2).to(tl.float32)
                g23 = tl.load(grad_out_ptr + out_base[:, None] + 2 + offs2).to(tl.float32)
                g45 = tl.load(grad_out_ptr + out_base[:, None] + 4 + offs2).to(tl.float32)
            else:
                g01 = tl.load(
                    grad_out_ptr + out_base[:, None] + 0 + offs2, mask=mask[:, None], other=0.0
                ).to(tl.float32)
                g23 = tl.load(
                    grad_out_ptr + out_base[:, None] + 2 + offs2, mask=mask[:, None], other=0.0
                ).to(tl.float32)
                g45 = tl.load(
                    grad_out_ptr + out_base[:, None] + 4 + offs2, mask=mask[:, None], other=0.0
                ).to(tl.float32)
            gw0, gw1 = tl.split(g01)
            gw2, gi0 = tl.split(g23)
            gi1, gi2 = tl.split(g45)
        elif CLI_MODE == 1:  # D_K = 3
            if EVEN_M:
                gw0 = tl.load(grad_out_ptr + out_base + 0).to(tl.float32)
                gw1 = tl.load(grad_out_ptr + out_base + 1).to(tl.float32)
                gw2 = tl.load(grad_out_ptr + out_base + 2).to(tl.float32)
            else:
                gw0 = tl.load(grad_out_ptr + out_base + 0, mask=mask, other=0.0).to(tl.float32)
                gw1 = tl.load(grad_out_ptr + out_base + 1, mask=mask, other=0.0).to(tl.float32)
                gw2 = tl.load(grad_out_ptr + out_base + 2, mask=mask, other=0.0).to(tl.float32)
        else:  # D_K = 3
            if EVEN_M:
                gi0 = tl.load(grad_out_ptr + out_base + 0).to(tl.float32)
                gi1 = tl.load(grad_out_ptr + out_base + 1).to(tl.float32)
                gi2 = tl.load(grad_out_ptr + out_base + 2).to(tl.float32)
            else:
                gi0 = tl.load(grad_out_ptr + out_base + 0, mask=mask, other=0.0).to(tl.float32)
                gi1 = tl.load(grad_out_ptr + out_base + 1, mask=mask, other=0.0).to(tl.float32)
                gi2 = tl.load(grad_out_ptr + out_base + 2, mask=mask, other=0.0).to(tl.float32)

        dz1_0, dz1_1, dz1_2, dz2_0, dz2_1, dz2_2 = _bwd_k3_math(
            u0, u1, u2, v0, v1, v2, gw0, gw1, gw2, gi0, gi1, gi2, CLI_MODE, CTX_MODE, USE_FMA
        )
        if NEED_DZ1:
            if EVEN_M:
                tl.store(dz1_ptr + in_base + 0, dz1_0)
                tl.store(dz1_ptr + in_base + 1, dz1_1)
                tl.store(dz1_ptr + in_base + 2, dz1_2)
            else:
                tl.store(dz1_ptr + in_base + 0, dz1_0, mask=mask)
                tl.store(dz1_ptr + in_base + 1, dz1_1, mask=mask)
                tl.store(dz1_ptr + in_base + 2, dz1_2, mask=mask)
        if NEED_DZ2:
            if EVEN_M:
                tl.store(dz2_ptr + in_base + 0, dz2_0)
                tl.store(dz2_ptr + in_base + 1, dz2_1)
                tl.store(dz2_ptr + in_base + 2, dz2_2)
            else:
                tl.store(dz2_ptr + in_base + 0, dz2_0, mask=mask)
                tl.store(dz2_ptr + in_base + 1, dz2_1, mask=mask)
                tl.store(dz2_ptr + in_base + 2, dz2_2, mask=mask)

    else:  # K == 4
        if EVEN_M:
            u_vec = tl.load(z1_ptr + in_base[:, None] + offs4).to(tl.float32)
            r_vec = tl.load(z2_ptr + in_base[:, None] + offs4).to(tl.float32)
        else:
            u_vec = tl.load(
                z1_ptr + in_base[:, None] + offs4, mask=mask[:, None], other=0.0
            ).to(tl.float32)
            r_vec = tl.load(
                z2_ptr + in_base[:, None] + offs4, mask=mask[:, None], other=0.0
            ).to(tl.float32)
        u_even, u_odd = tl.split(tl.reshape(u_vec, (BLOCK_M, 2, 2)))
        u0, u2 = tl.split(u_even)
        u1, u3 = tl.split(u_odd)
        r_even, r_odd = tl.split(tl.reshape(r_vec, (BLOCK_M, 2, 2)))
        r0, r2 = tl.split(r_even)
        r1, r3 = tl.split(r_odd)
        v0 = (r0 - u0) if CTX_MODE == 0 else r0
        v1 = (r1 - u1) if CTX_MODE == 0 else r1
        v2 = (r2 - u2) if CTX_MODE == 0 else r2
        v3 = (r3 - u3) if CTX_MODE == 0 else r3

        gw0 = 0.0
        gw1 = 0.0
        gw2 = 0.0
        gw3 = 0.0
        gw4 = 0.0
        gw5 = 0.0
        gi0 = 0.0
        gi1 = 0.0
        gi2 = 0.0
        gi3 = 0.0
        if CLI_MODE == 0:  # D_K = 10 (5 x v2 loads)
            if EVEN_M:
                g01 = tl.load(grad_out_ptr + out_base[:, None] + 0 + offs2).to(tl.float32)
                g23 = tl.load(grad_out_ptr + out_base[:, None] + 2 + offs2).to(tl.float32)
                g45 = tl.load(grad_out_ptr + out_base[:, None] + 4 + offs2).to(tl.float32)
                g67 = tl.load(grad_out_ptr + out_base[:, None] + 6 + offs2).to(tl.float32)
                g89 = tl.load(grad_out_ptr + out_base[:, None] + 8 + offs2).to(tl.float32)
            else:
                g01 = tl.load(
                    grad_out_ptr + out_base[:, None] + 0 + offs2, mask=mask[:, None], other=0.0
                ).to(tl.float32)
                g23 = tl.load(
                    grad_out_ptr + out_base[:, None] + 2 + offs2, mask=mask[:, None], other=0.0
                ).to(tl.float32)
                g45 = tl.load(
                    grad_out_ptr + out_base[:, None] + 4 + offs2, mask=mask[:, None], other=0.0
                ).to(tl.float32)
                g67 = tl.load(
                    grad_out_ptr + out_base[:, None] + 6 + offs2, mask=mask[:, None], other=0.0
                ).to(tl.float32)
                g89 = tl.load(
                    grad_out_ptr + out_base[:, None] + 8 + offs2, mask=mask[:, None], other=0.0
                ).to(tl.float32)
            gw0, gw1 = tl.split(g01)
            gw2, gw3 = tl.split(g23)
            gw4, gw5 = tl.split(g45)
            gi0, gi1 = tl.split(g67)
            gi2, gi3 = tl.split(g89)
        elif CLI_MODE == 1:  # D_K = 6
            if EVEN_M:
                g01 = tl.load(grad_out_ptr + out_base[:, None] + 0 + offs2).to(tl.float32)
                g23 = tl.load(grad_out_ptr + out_base[:, None] + 2 + offs2).to(tl.float32)
                g45 = tl.load(grad_out_ptr + out_base[:, None] + 4 + offs2).to(tl.float32)
            else:
                g01 = tl.load(
                    grad_out_ptr + out_base[:, None] + 0 + offs2, mask=mask[:, None], other=0.0
                ).to(tl.float32)
                g23 = tl.load(
                    grad_out_ptr + out_base[:, None] + 2 + offs2, mask=mask[:, None], other=0.0
                ).to(tl.float32)
                g45 = tl.load(
                    grad_out_ptr + out_base[:, None] + 4 + offs2, mask=mask[:, None], other=0.0
                ).to(tl.float32)
            gw0, gw1 = tl.split(g01)
            gw2, gw3 = tl.split(g23)
            gw4, gw5 = tl.split(g45)
        else:  # D_K = 4
            g_vec = (
                tl.load(grad_out_ptr + out_base[:, None] + offs4).to(tl.float32)
                if EVEN_M
                else tl.load(
                    grad_out_ptr + out_base[:, None] + offs4, mask=mask[:, None], other=0.0
                ).to(tl.float32)
            )
            g_even, g_odd = tl.split(tl.reshape(g_vec, (BLOCK_M, 2, 2)))
            gi0, gi2 = tl.split(g_even)
            gi1, gi3 = tl.split(g_odd)

        dz1_0, dz1_1, dz1_2, dz1_3, dz2_0, dz2_1, dz2_2, dz2_3 = _bwd_k4_math(
            u0,
            u1,
            u2,
            u3,
            v0,
            v1,
            v2,
            v3,
            gw0,
            gw1,
            gw2,
            gw3,
            gw4,
            gw5,
            gi0,
            gi1,
            gi2,
            gi3,
            CLI_MODE,
            CTX_MODE,
        )
        if NEED_DZ1:
            dz1_vec = tl.reshape(
                tl.join(tl.join(dz1_0, dz1_2), tl.join(dz1_1, dz1_3)), (BLOCK_M, 4)
            )
            if EVEN_M:
                tl.store(dz1_ptr + in_base[:, None] + offs4, dz1_vec)
            else:
                tl.store(dz1_ptr + in_base[:, None] + offs4, dz1_vec, mask=mask[:, None])
        if NEED_DZ2:
            dz2_vec = tl.reshape(
                tl.join(tl.join(dz2_0, dz2_2), tl.join(dz2_1, dz2_3)), (BLOCK_M, 4)
            )
            if EVEN_M:
                tl.store(dz2_ptr + in_base[:, None] + offs4, dz2_vec)
            else:
                tl.store(dz2_ptr + in_base[:, None] + offs4, dz2_vec, mask=mask[:, None])


@triton.jit
def _subspace_gp_bwd_strided_kernel(
    grad_out_ptr,
    z1_ptr,
    z2_ptr,
    dz1_ptr,
    dz2_ptr,
    G,
    H,
    W,
    HW,
    go_sb,
    go_sc,
    go_sh,
    go_sw,
    z1_sb,
    z1_sc,
    z1_sh,
    z1_sw,
    z2_sb,
    z2_sc,
    z2_sh,
    z2_sw,
    dz1_sb,
    dz1_sc,
    dz1_sh,
    dz1_sw,
    dz2_sb,
    dz2_sc,
    dz2_sh,
    dz2_sw,
    K: tl.constexpr,
    CLI_MODE: tl.constexpr,
    CTX_MODE: tl.constexpr,
    USE_FMA: tl.constexpr,
    D_K: tl.constexpr,
    NEED_DZ1: tl.constexpr,
    NEED_DZ2: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_hw = tl.program_id(0).to(tl.int64)
    bg = tl.program_id(1).to(tl.int64)
    b = bg // G
    g = bg % G

    hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW).to(tl.int64)
    mask = hw < HW
    h_idx = hw // W
    w_idx = hw % W

    c_in = g * K
    c_out = g * D_K

    z1_base = b * z1_sb + c_in * z1_sc + h_idx * z1_sh + w_idx * z1_sw
    z2_base = b * z2_sb + c_in * z2_sc + h_idx * z2_sh + w_idx * z2_sw
    go_base = b * go_sb + c_out * go_sc + h_idx * go_sh + w_idx * go_sw
    dz1_base = b * dz1_sb + c_in * dz1_sc + h_idx * dz1_sh + w_idx * dz1_sw
    dz2_base = b * dz2_sb + c_in * dz2_sc + h_idx * dz2_sh + w_idx * dz2_sw

    u0 = tl.load(z1_ptr + z1_base + 0 * z1_sc, mask=mask, other=0.0).to(tl.float32)
    u1 = tl.load(z1_ptr + z1_base + 1 * z1_sc, mask=mask, other=0.0).to(tl.float32)
    r0 = tl.load(z2_ptr + z2_base + 0 * z2_sc, mask=mask, other=0.0).to(tl.float32)
    r1 = tl.load(z2_ptr + z2_base + 1 * z2_sc, mask=mask, other=0.0).to(tl.float32)
    v0 = (r0 - u0) if CTX_MODE == 0 else r0
    v1 = (r1 - u1) if CTX_MODE == 0 else r1

    if K == 2:
        gw0 = (
            tl.load(grad_out_ptr + go_base + 0 * go_sc, mask=mask, other=0.0).to(tl.float32)
            if CLI_MODE != 2
            else 0.0
        )
        gi0 = 0.0
        gi1 = 0.0
        if CLI_MODE != 1:
            in_off: tl.constexpr = 0 if CLI_MODE == 2 else 1
            gi0 = tl.load(grad_out_ptr + go_base + (in_off + 0) * go_sc, mask=mask, other=0.0).to(
                tl.float32
            )
            gi1 = tl.load(grad_out_ptr + go_base + (in_off + 1) * go_sc, mask=mask, other=0.0).to(
                tl.float32
            )
        dz1_0, dz1_1, dz2_0, dz2_1 = _bwd_k2_math(
            u0, u1, v0, v1, gw0, gi0, gi1, CLI_MODE, CTX_MODE
        )
        if NEED_DZ1:
            tl.store(dz1_ptr + dz1_base + 0 * dz1_sc, dz1_0, mask=mask)
            tl.store(dz1_ptr + dz1_base + 1 * dz1_sc, dz1_1, mask=mask)
        if NEED_DZ2:
            tl.store(dz2_ptr + dz2_base + 0 * dz2_sc, dz2_0, mask=mask)
            tl.store(dz2_ptr + dz2_base + 1 * dz2_sc, dz2_1, mask=mask)

    elif K == 3:
        u2 = tl.load(z1_ptr + z1_base + 2 * z1_sc, mask=mask, other=0.0).to(tl.float32)
        r2 = tl.load(z2_ptr + z2_base + 2 * z2_sc, mask=mask, other=0.0).to(tl.float32)
        v2 = (r2 - u2) if CTX_MODE == 0 else r2
        gw0 = 0.0
        gw1 = 0.0
        gw2 = 0.0
        gi0 = 0.0
        gi1 = 0.0
        gi2 = 0.0
        if CLI_MODE != 2:
            gw0 = tl.load(grad_out_ptr + go_base + 0 * go_sc, mask=mask, other=0.0).to(tl.float32)
            gw1 = tl.load(grad_out_ptr + go_base + 1 * go_sc, mask=mask, other=0.0).to(tl.float32)
            gw2 = tl.load(grad_out_ptr + go_base + 2 * go_sc, mask=mask, other=0.0).to(tl.float32)
        if CLI_MODE != 1:
            in_off: tl.constexpr = 0 if CLI_MODE == 2 else 3
            gi0 = tl.load(grad_out_ptr + go_base + (in_off + 0) * go_sc, mask=mask, other=0.0).to(
                tl.float32
            )
            gi1 = tl.load(grad_out_ptr + go_base + (in_off + 1) * go_sc, mask=mask, other=0.0).to(
                tl.float32
            )
            gi2 = tl.load(grad_out_ptr + go_base + (in_off + 2) * go_sc, mask=mask, other=0.0).to(
                tl.float32
            )
        dz1_0, dz1_1, dz1_2, dz2_0, dz2_1, dz2_2 = _bwd_k3_math(
            u0, u1, u2, v0, v1, v2, gw0, gw1, gw2, gi0, gi1, gi2, CLI_MODE, CTX_MODE, USE_FMA
        )
        if NEED_DZ1:
            tl.store(dz1_ptr + dz1_base + 0 * dz1_sc, dz1_0, mask=mask)
            tl.store(dz1_ptr + dz1_base + 1 * dz1_sc, dz1_1, mask=mask)
            tl.store(dz1_ptr + dz1_base + 2 * dz1_sc, dz1_2, mask=mask)
        if NEED_DZ2:
            tl.store(dz2_ptr + dz2_base + 0 * dz2_sc, dz2_0, mask=mask)
            tl.store(dz2_ptr + dz2_base + 1 * dz2_sc, dz2_1, mask=mask)
            tl.store(dz2_ptr + dz2_base + 2 * dz2_sc, dz2_2, mask=mask)

    else:  # K == 4
        u2 = tl.load(z1_ptr + z1_base + 2 * z1_sc, mask=mask, other=0.0).to(tl.float32)
        u3 = tl.load(z1_ptr + z1_base + 3 * z1_sc, mask=mask, other=0.0).to(tl.float32)
        r2 = tl.load(z2_ptr + z2_base + 2 * z2_sc, mask=mask, other=0.0).to(tl.float32)
        r3 = tl.load(z2_ptr + z2_base + 3 * z2_sc, mask=mask, other=0.0).to(tl.float32)
        v2 = (r2 - u2) if CTX_MODE == 0 else r2
        v3 = (r3 - u3) if CTX_MODE == 0 else r3
        gw0 = 0.0
        gw1 = 0.0
        gw2 = 0.0
        gw3 = 0.0
        gw4 = 0.0
        gw5 = 0.0
        gi0 = 0.0
        gi1 = 0.0
        gi2 = 0.0
        gi3 = 0.0
        if CLI_MODE != 2:
            gw0 = tl.load(grad_out_ptr + go_base + 0 * go_sc, mask=mask, other=0.0).to(tl.float32)
            gw1 = tl.load(grad_out_ptr + go_base + 1 * go_sc, mask=mask, other=0.0).to(tl.float32)
            gw2 = tl.load(grad_out_ptr + go_base + 2 * go_sc, mask=mask, other=0.0).to(tl.float32)
            gw3 = tl.load(grad_out_ptr + go_base + 3 * go_sc, mask=mask, other=0.0).to(tl.float32)
            gw4 = tl.load(grad_out_ptr + go_base + 4 * go_sc, mask=mask, other=0.0).to(tl.float32)
            gw5 = tl.load(grad_out_ptr + go_base + 5 * go_sc, mask=mask, other=0.0).to(tl.float32)
        if CLI_MODE != 1:
            in_off: tl.constexpr = 0 if CLI_MODE == 2 else 6
            gi0 = tl.load(grad_out_ptr + go_base + (in_off + 0) * go_sc, mask=mask, other=0.0).to(
                tl.float32
            )
            gi1 = tl.load(grad_out_ptr + go_base + (in_off + 1) * go_sc, mask=mask, other=0.0).to(
                tl.float32
            )
            gi2 = tl.load(grad_out_ptr + go_base + (in_off + 2) * go_sc, mask=mask, other=0.0).to(
                tl.float32
            )
            gi3 = tl.load(grad_out_ptr + go_base + (in_off + 3) * go_sc, mask=mask, other=0.0).to(
                tl.float32
            )
        dz1_0, dz1_1, dz1_2, dz1_3, dz2_0, dz2_1, dz2_2, dz2_3 = _bwd_k4_math(
            u0,
            u1,
            u2,
            u3,
            v0,
            v1,
            v2,
            v3,
            gw0,
            gw1,
            gw2,
            gw3,
            gw4,
            gw5,
            gi0,
            gi1,
            gi2,
            gi3,
            CLI_MODE,
            CTX_MODE,
        )
        if NEED_DZ1:
            tl.store(dz1_ptr + dz1_base + 0 * dz1_sc, dz1_0, mask=mask)
            tl.store(dz1_ptr + dz1_base + 1 * dz1_sc, dz1_1, mask=mask)
            tl.store(dz1_ptr + dz1_base + 2 * dz1_sc, dz1_2, mask=mask)
            tl.store(dz1_ptr + dz1_base + 3 * dz1_sc, dz1_3, mask=mask)
        if NEED_DZ2:
            tl.store(dz2_ptr + dz2_base + 0 * dz2_sc, dz2_0, mask=mask)
            tl.store(dz2_ptr + dz2_base + 1 * dz2_sc, dz2_1, mask=mask)
            tl.store(dz2_ptr + dz2_base + 2 * dz2_sc, dz2_2, mask=mask)
            tl.store(dz2_ptr + dz2_base + 3 * dz2_sc, dz2_3, mask=mask)


# ============================================================================
# Host Launch Heuristics (Tuned for Hopper H800 sm_90)
# ============================================================================


def _select_nchw_tiling(hw: int) -> tuple[int, int, int]:
    """Return (block_bg, block_hw, num_warps) for packed NCHW tensors."""
    if hw >= 512:
        return 1, 512, 4
    if hw >= 128:
        return 1, 256, 4
    block_hw = max(1, triton.next_power_of_2(hw))
    block_bg = max(1, 256 // block_hw)
    return block_bg, block_hw, 4


def _select_nhwc_tiling(m: int, d_k: int) -> tuple[int, int]:
    """Return (block_m, num_warps) with 1 group/thread for coalesced NHWC sectors."""
    return 256, 8


def _launch_fwd(
    z1: torch.Tensor,
    z2: torch.Tensor,
    k: int,
    cli_mode: str,
    ctx_mode: str,
    use_fma: bool,
) -> torch.Tensor:
    b, g, h, w, d_in, d_k = _validate_inputs(z1, z2, k, cli_mode, ctx_mode)
    mem_fmt = _infer_memory_format(z1)
    out = torch.empty(
        (b, g * d_k, h, w), device=z1.device, dtype=z1.dtype, memory_format=mem_fmt
    )
    if b == 0 or h == 0 or w == 0:
        return out

    cli_id = _CLI_MODE_TO_ID[cli_mode]
    ctx_id = _CTX_MODE_TO_ID[ctx_mode]
    hw = h * w
    n_bg = b * g
    max_elems = max(z1.numel(), out.numel())
    use_int64 = max_elems >= (1 << 31)

    if mem_fmt == torch.channels_last and _is_dense_nhwc(z1) and _is_dense_nhwc(z2):
        m = n_bg * hw
        block_m, num_warps = _select_nhwc_tiling(m, d_k)
        even_m = (m % block_m) == 0
        grid = (triton.cdiv(m, block_m),)
        _subspace_gp_fwd_nhwc_kernel[grid](
            z1,
            z2,
            out,
            m,
            K=k,
            CLI_MODE=cli_id,
            CTX_MODE=ctx_id,
            USE_FMA=use_fma,
            D_K=d_k,
            BLOCK_M=block_m,
            EVEN_M=even_m,
            USE_INT64=use_int64,
            num_warps=num_warps,
        )
        return out

    if mem_fmt == torch.contiguous_format and _is_dense_nchw(z1) and _is_dense_nchw(z2):
        block_bg, block_hw, num_warps = _select_nchw_tiling(hw)
        even_hw = (block_bg == 1) and ((hw % block_hw) == 0)
        grid = (triton.cdiv(hw, block_hw), triton.cdiv(n_bg, block_bg))
        _subspace_gp_fwd_nchw_kernel[grid](
            z1,
            z2,
            out,
            n_bg,
            hw,
            K=k,
            CLI_MODE=cli_id,
            CTX_MODE=ctx_id,
            USE_FMA=use_fma,
            D_K=d_k,
            BLOCK_BG=block_bg,
            BLOCK_HW=block_hw,
            EVEN_HW=even_hw,
            USE_INT64=use_int64,
            num_warps=num_warps,
        )
        return out

    block_hw = 256 if hw >= 128 else max(32, triton.next_power_of_2(hw))
    grid = (triton.cdiv(hw, block_hw), n_bg)
    _subspace_gp_fwd_strided_kernel[grid](
        z1,
        z2,
        out,
        g,
        h,
        w,
        hw,
        *z1.stride(),
        *z2.stride(),
        *out.stride(),
        K=k,
        CLI_MODE=cli_id,
        CTX_MODE=ctx_id,
        USE_FMA=use_fma,
        D_K=d_k,
        BLOCK_HW=block_hw,
        num_warps=4,
    )
    return out


def _launch_bwd(
    grad_out: torch.Tensor,
    z1: torch.Tensor,
    z2: torch.Tensor,
    k: int,
    cli_mode: str,
    ctx_mode: str,
    use_fma: bool,
    need_dz1: bool = True,
    need_dz2: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    b, g, h, w, _, d_k = _validate_inputs(z1, z2, k, cli_mode, ctx_mode)
    mem_fmt_1 = _infer_memory_format(z1)
    mem_fmt_2 = _infer_memory_format(z2)

    dz1 = (
        torch.empty(z1.shape, device=z1.device, dtype=z1.dtype, memory_format=mem_fmt_1)
        if need_dz1
        else torch.empty(0, device=z1.device, dtype=z1.dtype)
    )
    dz2 = (
        torch.empty(z2.shape, device=z2.device, dtype=z2.dtype, memory_format=mem_fmt_2)
        if need_dz2
        else torch.empty(0, device=z2.device, dtype=z2.dtype)
    )

    if b == 0 or h == 0 or w == 0 or (not need_dz1 and not need_dz2):
        return dz1, dz2

    cli_id = _CLI_MODE_TO_ID[cli_mode]
    ctx_id = _CTX_MODE_TO_ID[ctx_mode]
    hw = h * w
    n_bg = b * g
    max_elems = max(z1.numel(), grad_out.numel())
    use_int64 = max_elems >= (1 << 31)

    # Fast NHWC path when all active tensors are packed NHWC
    all_nhwc = (
        _is_dense_nhwc(z1)
        and _is_dense_nhwc(z2)
        and _is_dense_nhwc(grad_out)
        and (not need_dz1 or mem_fmt_1 == torch.channels_last)
        and (not need_dz2 or mem_fmt_2 == torch.channels_last)
    )
    if all_nhwc:
        m = n_bg * hw
        block_m, num_warps = _select_nhwc_tiling(m, d_k)
        even_m = (m % block_m) == 0
        grid = (triton.cdiv(m, block_m),)
        _subspace_gp_bwd_nhwc_kernel[grid](
            grad_out,
            z1,
            z2,
            dz1 if need_dz1 else z1,
            dz2 if need_dz2 else z2,
            m,
            K=k,
            CLI_MODE=cli_id,
            CTX_MODE=ctx_id,
            USE_FMA=use_fma,
            D_K=d_k,
            NEED_DZ1=need_dz1,
            NEED_DZ2=need_dz2,
            BLOCK_M=block_m,
            EVEN_M=even_m,
            USE_INT64=use_int64,
            num_warps=num_warps,
        )
        return dz1, dz2

    # Fast NCHW path when all active tensors are packed NCHW
    all_nchw = (
        _is_dense_nchw(z1)
        and _is_dense_nchw(z2)
        and _is_dense_nchw(grad_out)
        and (not need_dz1 or mem_fmt_1 == torch.contiguous_format)
        and (not need_dz2 or mem_fmt_2 == torch.contiguous_format)
    )
    if all_nchw:
        block_bg, block_hw, num_warps = _select_nchw_tiling(hw)
        even_hw = (block_bg == 1) and ((hw % block_hw) == 0)
        grid = (triton.cdiv(hw, block_hw), triton.cdiv(n_bg, block_bg))
        _subspace_gp_bwd_nchw_kernel[grid](
            grad_out,
            z1,
            z2,
            dz1 if need_dz1 else z1,
            dz2 if need_dz2 else z2,
            n_bg,
            hw,
            K=k,
            CLI_MODE=cli_id,
            CTX_MODE=ctx_id,
            USE_FMA=use_fma,
            D_K=d_k,
            NEED_DZ1=need_dz1,
            NEED_DZ2=need_dz2,
            BLOCK_BG=block_bg,
            BLOCK_HW=block_hw,
            EVEN_HW=even_hw,
            USE_INT64=use_int64,
            num_warps=num_warps,
        )
        return dz1, dz2

    # General 4D strided path
    dz1_strides = dz1.stride() if need_dz1 else (0, 0, 0, 0)
    dz2_strides = dz2.stride() if need_dz2 else (0, 0, 0, 0)
    block_hw = 256 if hw >= 128 else max(32, triton.next_power_of_2(hw))
    grid = (triton.cdiv(hw, block_hw), n_bg)
    _subspace_gp_bwd_strided_kernel[grid](
        grad_out,
        z1,
        z2,
        dz1 if need_dz1 else z1,
        dz2 if need_dz2 else z2,
        g,
        h,
        w,
        hw,
        *grad_out.stride(),
        *z1.stride(),
        *z2.stride(),
        *dz1_strides,
        *dz2_strides,
        K=k,
        CLI_MODE=cli_id,
        CTX_MODE=ctx_id,
        USE_FMA=use_fma,
        D_K=d_k,
        NEED_DZ1=need_dz1,
        NEED_DZ2=need_dz2,
        BLOCK_HW=block_hw,
        num_warps=4,
    )
    return dz1, dz2


# ============================================================================
# torch.library.custom_op Registration & Autograd Function
# ============================================================================


@torch.library.custom_op("cliffordnet::subspace_gp_fwd", mutates_args=())
def _subspace_gp_fwd_op(
    z1: torch.Tensor,
    z2: torch.Tensor,
    k: int,
    cli_mode: str,
    ctx_mode: str,
    use_fma: bool,
) -> torch.Tensor:
    return _launch_fwd(z1, z2, k, cli_mode, ctx_mode, use_fma)


@_subspace_gp_fwd_op.register_fake
def _(
    z1: torch.Tensor,
    z2: torch.Tensor,
    k: int,
    cli_mode: str,
    ctx_mode: str,
    use_fma: bool,
) -> torch.Tensor:
    b, g, h, w, _, d_k = _validate_inputs(
        z1, z2, k, cli_mode, ctx_mode, require_cuda=False
    )
    mem_fmt = _infer_memory_format(z1)
    return torch.empty(
        (b, g * d_k, h, w), device=z1.device, dtype=z1.dtype, memory_format=mem_fmt
    )


@torch.library.custom_op("cliffordnet::subspace_gp_bwd", mutates_args=())
def _subspace_gp_bwd_op(
    grad_out: torch.Tensor,
    z1: torch.Tensor,
    z2: torch.Tensor,
    k: int,
    cli_mode: str,
    ctx_mode: str,
    use_fma: bool,
    need_dz1: bool,
    need_dz2: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _launch_bwd(
        grad_out, z1, z2, k, cli_mode, ctx_mode, use_fma, need_dz1, need_dz2
    )


@_subspace_gp_bwd_op.register_fake
def _(
    grad_out: torch.Tensor,
    z1: torch.Tensor,
    z2: torch.Tensor,
    k: int,
    cli_mode: str,
    ctx_mode: str,
    use_fma: bool,
    need_dz1: bool,
    need_dz2: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    mem_fmt_1 = _infer_memory_format(z1)
    mem_fmt_2 = _infer_memory_format(z2)
    dz1 = (
        torch.empty(z1.shape, device=z1.device, dtype=z1.dtype, memory_format=mem_fmt_1)
        if need_dz1
        else torch.empty(0, device=z1.device, dtype=z1.dtype)
    )
    dz2 = (
        torch.empty(z2.shape, device=z2.device, dtype=z2.dtype, memory_format=mem_fmt_2)
        if need_dz2
        else torch.empty(0, device=z2.device, dtype=z2.dtype)
    )
    return dz1, dz2


class SubspaceGPTritonFn(torch.autograd.Function):
    """Recomputation-based autograd Function saving ONLY (z1, z2)."""

    @staticmethod
    def forward(
        ctx,
        z1: torch.Tensor,
        z2: torch.Tensor,
        k: int,
        cli_mode: str = "full",
        ctx_mode: str = "diff",
        use_fma: bool = True,
    ) -> torch.Tensor:
        ctx.save_for_backward(z1, z2)
        ctx.k = k
        ctx.cli_mode = cli_mode
        ctx.ctx_mode = ctx_mode
        ctx.use_fma = use_fma
        if torch.compiler.is_compiling():
            return _subspace_gp_fwd_op(z1, z2, k, cli_mode, ctx_mode, use_fma)
        return _launch_fwd(z1, z2, k, cli_mode, ctx_mode, use_fma)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        z1, z2 = ctx.saved_tensors
        need_dz1 = bool(ctx.needs_input_grad[0])
        need_dz2 = bool(ctx.needs_input_grad[1])
        if torch.compiler.is_compiling():
            dz1, dz2 = _subspace_gp_bwd_op(
                grad_out,
                z1,
                z2,
                ctx.k,
                ctx.cli_mode,
                ctx.ctx_mode,
                ctx.use_fma,
                need_dz1,
                need_dz2,
            )
        else:
            dz1, dz2 = _launch_bwd(
                grad_out,
                z1,
                z2,
                ctx.k,
                ctx.cli_mode,
                ctx.ctx_mode,
                ctx.use_fma,
                need_dz1,
                need_dz2,
            )
        return (
            dz1 if need_dz1 else None,
            dz2 if need_dz2 else None,
            None,
            None,
            None,
            None,
        )


def subspace_gp_triton(
    z1: torch.Tensor,
    z2: torch.Tensor,
    k: int,
    cli_mode: str = "full",
    ctx_mode: str = "diff",
    use_fma: bool = True,
) -> torch.Tensor:
    """Compute the fused Subspace Clifford Geometric Product on CUDA via Triton.

    Args:
        z1: State stream tensor of shape ``(B, D_tilde, H, W)`` on CUDA.
        z2: Context stream tensor of shape ``(B, D_tilde, H, W)`` on CUDA.
        k: Subspace group size in ``{2, 3, 4}`` with ``D_tilde % k == 0``.
        cli_mode: Component selection in ``{"full", "wedge", "inner"}``.
        ctx_mode: Context definition in ``{"diff", "abs"}``.
        use_fma: Use error-free FMA 2x2 determinant evaluation for wedge terms.

    Returns:
        Output tensor of shape ``(B, (D_tilde // k) * d_k, H, W)`` preserving
        ``z1``'s dtype and memory format (``contiguous_format`` or ``channels_last``).
    """
    if not torch.is_grad_enabled() or (not z1.requires_grad and not z2.requires_grad):
        if torch.compiler.is_compiling():
            _validate_inputs(z1, z2, k, cli_mode, ctx_mode)
            return _subspace_gp_fwd_op(z1, z2, k, cli_mode, ctx_mode, use_fma)
        return _launch_fwd(z1, z2, k, cli_mode, ctx_mode, use_fma)
    return SubspaceGPTritonFn.apply(z1, z2, k, cli_mode, ctx_mode, use_fma)
