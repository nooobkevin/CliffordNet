"""Subspace Clifford Networks (arXiv:2610.37226), deconfounding controls, diagnostics, and timm wrappers."""

from __future__ import annotations

import inspect
import math
from typing import Any, Callable, Iterator, NamedTuple, Sequence

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, trunc_normal_

VALID_SUBSPACE_K = (2, 3, 4, 6, 8)
VALID_CLI_MODES = (
    "full",
    "wedge",
    "inner",
    "inner_wide",
    "sym_full",
    "bilinear_full",
    "swiglu_wedge",
)
VALID_CTX_MODES = ("diff", "abs")
VALID_WEDGE_MODES = ("fma", "fp32", "naive", "direct_z2", "direct_z2_fma")
VALID_BACKENDS = ("auto", "pytorch", "triton")
VALID_BLOCK_STYLES = ("lie_trotter", "ggr")
VALID_INTERACTION_TYPES = ("subspace", "rolling")
VALID_NORM_MODES = ("bn", "ln2d", "rmsnorm2d", "subspace_rms", "onesided_sphere", "rmsnorm2d_sphere")

# Off-diagonal index pairs per subspace dimension k (arXiv:2610.37226 Sec. 3.3.2):
#   k=2: 2D signed area [(0, 1)]
#   k=3: cyclic 3D cross-product convention [(0, 1), (1, 2), (2, 0)]
#   k>=4: lexicographic bivector planes [(i, j) for 0 <= i < j < k]
SUBSPACE_OFFDIAG_PAIRS: dict[int, tuple[tuple[int, int], ...]] = {
    2: ((0, 1),),
    3: ((0, 1), (1, 2), (2, 0)),
    4: tuple((i, j) for i in range(4) for j in range(i + 1, 4)),
    6: tuple((i, j) for i in range(6) for j in range(i + 1, 6)),
    8: tuple((i, j) for i in range(8) for j in range(i + 1, 8)),
}

_TRITON_SUBSPACE_FN: Callable[..., torch.Tensor] | None = None
_TRITON_PARAM_NAMES: frozenset[str] = frozenset()


def _resolve_triton_subspace() -> tuple[Callable[..., torch.Tensor] | None, frozenset[str]]:
    """Lazily resolve optional Triton kernel without hard import dependency."""
    global _TRITON_SUBSPACE_FN, _TRITON_PARAM_NAMES
    if _TRITON_SUBSPACE_FN is not None:
        return _TRITON_SUBSPACE_FN, _TRITON_PARAM_NAMES
    try:
        from cliffordnet.kernels.triton_subspace import subspace_gp_triton

        sig = inspect.signature(subspace_gp_triton)
        _TRITON_SUBSPACE_FN = subspace_gp_triton
        _TRITON_PARAM_NAMES = frozenset(sig.parameters.keys())
    except Exception:
        return None, frozenset()
    return _TRITON_SUBSPACE_FN, _TRITON_PARAM_NAMES


# ============================================================================
# Shared Normalization, Stem, and Downsampling Primitives
# ============================================================================


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for 2D (B, C, H, W) feature maps."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class RMSNorm2d(nn.Module):
    """Channel-wise RMSNorm for 2D (B, C, H, W) feature maps using fused F.rms_norm (zero-copy on NHWC)."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # On channels_last (NHWC), permute(0, 2, 3, 1) is a zero-copy contiguous (B, H, W, C) view.
        return F.rms_norm(
            x.permute(0, 2, 3, 1),
            (self.num_channels,),
            self.weight.to(x.dtype),
            self.eps,
        ).permute(0, 3, 1, 2)


class GeometricStem(nn.Module):
    """Convolutional stem matching CAN / CliffordNet reference conventions."""

    def __init__(self, in_chans: int = 3, embed_dim: int = 128, patch_size: int = 4) -> None:
        super().__init__()
        if patch_size == 1:
            self.proj: nn.Module = nn.Sequential(
                nn.Conv2d(
                    in_chans,
                    embed_dim // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(embed_dim // 2),
                nn.SiLU(),
                nn.Conv2d(
                    embed_dim // 2,
                    embed_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
            )
        elif patch_size == 2:
            self.proj = nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=3,
                stride=2,
                padding=1,
            )
        elif patch_size == 4:
            self.proj = nn.Sequential(
                nn.Conv2d(
                    in_chans,
                    embed_dim // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(embed_dim // 2),
                nn.SiLU(),
                nn.Conv2d(
                    embed_dim // 2,
                    embed_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
            )
        else:
            self.proj = nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
            )
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class StageDownsample(nn.Module):
    """Between-stage 2x spatial downsampling and channel projection (Sec. 3.4)."""

    def __init__(self, in_dim: int, out_dim: int, mode: str = "conv") -> None:
        super().__init__()
        if mode == "avgpool":
            self.down: nn.Module = nn.AvgPool2d(kernel_size=2, stride=2)
            self.proj: nn.Module = (
                nn.Conv2d(in_dim, out_dim, kernel_size=1)
                if in_dim != out_dim
                else nn.Identity()
            )
        elif mode == "conv":
            self.down = nn.Sequential(
                nn.Conv2d(
                    in_dim,
                    in_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=in_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(in_dim),
                nn.SiLU(),
            )
            self.proj = (
                nn.Conv2d(in_dim, out_dim, kernel_size=1)
                if in_dim != out_dim
                else nn.Identity()
            )
        elif mode == "patch":
            self.down = nn.Identity()
            self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)
        else:
            raise ValueError(f"Invalid downsample mode: {mode!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.down(x))


# ============================================================================
# 1. SpatialDiffusion (arXiv:2610.37226 Sec. 3.4.1)
# ============================================================================


class SpatialDiffusion(nn.Module):
    """
    Channel-wise spatial diffusion sub-step (Lie-Trotter step 1, Sec. 3.4.1):
        X_sp = X + gamma * SiLU(BN(DWConv_{KxK}(X)))
    With kernel_size=5, parameter count is 25*dim + 2*dim + dim = 28*dim.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 5,
        layer_scale_init_value: float = 1e-5,
        norm_type: str = "bn",
        flat_gamma: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            bias=False,
        )
        if norm_type == "ln2d":
            self.bn: nn.Module = LayerNorm2d(dim)
        elif norm_type == "rmsnorm2d":
            self.bn = RMSNorm2d(dim)
        else:
            self.bn = nn.BatchNorm2d(dim)
        gamma_shape = (dim,) if flat_gamma else (1, dim, 1, 1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(gamma_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gamma.flatten().view(1, -1, 1, 1) * F.silu(self.bn(self.dwconv(x)))


# ============================================================================
# 2. Pure-PyTorch Subspace Geometric Product (torch.compile-safe)
# ============================================================================


def _compute_wedge_diff(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    wedge_mode: str,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Compute a * b - c * d under the specified numerical stability strategy."""
    if wedge_mode in ("fma", "direct_z2_fma"):
        af = a.float()
        bf = b.float()
        cf = c.float()
        df = d.float()
        w = cf * df
        f = torch.addcmul(-w, af, bf)
        e = torch.addcmul(-w, cf, df)
        return (f - e).to(target_dtype)
    if wedge_mode == "fp32":
        return (a.float() * b.float() - c.float() * d.float()).to(target_dtype)
    if wedge_mode in ("naive", "direct_z2"):
        return (a * b - c * d).to(target_dtype)
    raise ValueError(f"Invalid wedge_mode: {wedge_mode!r}")


SUBSPACE_OFFDIAG_IDX: dict[int, tuple[list[int], list[int]]] = {
    k_val: ([i for i, _ in pairs], [j for _, j in pairs])
    for k_val, pairs in SUBSPACE_OFFDIAG_PAIRS.items()
}


def _extract_group_pairs(
    u: torch.Tensor,
    vw: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract (u_i, vw_j, u_j, vw_i) tensors of shape (B, G, C(k, 2), H, W).
    Uses tuple unbinding for small k <= 4 and single-op list indexing for k >= 6
    so torch.compile FX graphs stay compact at k=6 (15 pairs) and k=8 (28 pairs).
    """
    if k == 2:
        return u[:, :, 0:1], vw[:, :, 1:2], u[:, :, 1:2], vw[:, :, 0:1]
    if k >= 6:
        idx_i, idx_j = SUBSPACE_OFFDIAG_IDX[k]
        return u[:, :, idx_i], vw[:, :, idx_j], u[:, :, idx_j], vw[:, :, idx_i]
    pairs = SUBSPACE_OFFDIAG_PAIRS[k]
    u_coords = u.unbind(dim=2)
    vw_coords = vw.unbind(dim=2)
    u_i = torch.stack([u_coords[i] for i, _ in pairs], dim=2)
    vw_j = torch.stack([vw_coords[j] for _, j in pairs], dim=2)
    u_j = torch.stack([u_coords[j] for _, j in pairs], dim=2)
    vw_i = torch.stack([vw_coords[i] for i, _ in pairs], dim=2)
    return u_i, vw_j, u_j, vw_i


def subspace_gp_pytorch(
    z1: torch.Tensor,
    z2: torch.Tensor,
    k: int,
    cli_mode: str = "full",
    ctx_mode: str = "diff",
    wedge_mode: str = "fma",
) -> torch.Tensor:
    """
    Pure PyTorch (torch.compile-safe, zero graph breaks) within-group subspace
    Clifford interaction (arXiv:2610.37226 Sec. 3.3.2) plus scientific controls.

    Args:
        z1: State stream of shape (B, D_tilde, H, W) with D_tilde divisible by k.
        z2: Spatial context stream of shape (B, D_tilde, H, W).
        k: Subspace group dimension in {2, 3, 4}.
        cli_mode:
            - "full"         : [p_wedge, p_inner] per group (antisymmetric + diagonal)
            - "wedge"        : p_wedge only per group
            - "inner"        : p_inner only per group
            - "inner_wide"   : p_inner only per group (widened D_tilde in module)
            - "sym_full"     : [u_i*v_j + u_j*v_i, p_inner] symmetric off-diag control
            - "bilinear_full": [u_i*v_j, p_inner] raw asymmetric bilinear control
        ctx_mode: "diff" (v = z2 - z1) or "abs" (v = z2).
        wedge_mode:
            - "fma"          : fp32 FMA-stabilized a*b - c*d on (u, v)
            - "fp32"         : fp32 upcast a*b - c*d on (u, v)
            - "naive"        : native-dtype a*b - c*d on (u, v)
            - "direct_z2"    : inner uses v, wedge uses v_wedge = z2 directly in native dtype
            - "direct_z2_fma": inner uses v, wedge uses v_wedge = z2 directly with fp32 FMA
    """
    if k not in VALID_SUBSPACE_K:
        raise ValueError(f"k must be in {VALID_SUBSPACE_K}, got {k}")
    if cli_mode not in VALID_CLI_MODES:
        raise ValueError(f"cli_mode must be in {VALID_CLI_MODES}, got {cli_mode!r}")
    if ctx_mode not in VALID_CTX_MODES:
        raise ValueError(f"ctx_mode must be in {VALID_CTX_MODES}, got {ctx_mode!r}")
    if wedge_mode not in VALID_WEDGE_MODES:
        raise ValueError(f"wedge_mode must be in {VALID_WEDGE_MODES}, got {wedge_mode!r}")

    B, d_tilde, H, W = z1.shape
    if d_tilde % k != 0:
        raise ValueError(f"Channel dim {d_tilde} must be divisible by k={k}")

    num_groups = d_tilde // k
    num_pairs = (k * (k - 1)) // 2
    target_dtype = z1.dtype

    z1_g = z1.reshape(B, num_groups, k, H, W)
    z2_g = z2.reshape(B, num_groups, k, H, W)

    u = z1_g
    v = (z2_g - z1_g) if ctx_mode == "diff" else z2_g

    p_offdiag: torch.Tensor | None = None
    if cli_mode in ("full", "wedge", "swiglu_wedge"):
        vw = z2_g if wedge_mode in ("direct_z2", "direct_z2_fma") else v
        u_i, vw_j, u_j, vw_i = _extract_group_pairs(u, vw, k)
        p_offdiag = _compute_wedge_diff(u_i, vw_j, u_j, vw_i, wedge_mode, target_dtype)
    elif cli_mode == "sym_full":
        u_i, v_j, u_j, v_i = _extract_group_pairs(u, v, k)
        p_offdiag = (u_i.float() * v_j.float() + u_j.float() * v_i.float()).to(target_dtype)
    elif cli_mode == "bilinear_full":
        pairs = SUBSPACE_OFFDIAG_PAIRS[k]
        u_coords = u.unbind(dim=2)
        v_coords = v.unbind(dim=2)
        u_i = torch.stack([u_coords[i] for i, _ in pairs], dim=2)
        v_j = torch.stack([v_coords[j] for _, j in pairs], dim=2)
        p_offdiag = (u_i.float() * v_j.float()).to(target_dtype)

    if cli_mode == "wedge":
        assert p_offdiag is not None
        return p_offdiag.reshape(B, num_groups * num_pairs, H, W)

    p_inner = (F.silu(u) * v) if cli_mode == "swiglu_wedge" else F.silu(u * v)
    if cli_mode in ("inner", "inner_wide"):
        return p_inner.reshape(B, num_groups * k, H, W)

    assert p_offdiag is not None
    # CRITICAL per-group layout: [p_wedge / p_offdiag, p_inner] along dim=2 BEFORE flattening G
    combined = torch.cat([p_offdiag, p_inner], dim=2)
    return combined.reshape(B, num_groups * (num_pairs + k), H, W)


def _compute_effective_rank_normalized(g_feat: torch.Tensor, max_tokens: int = 512) -> torch.Tensor:
    """
    Normalized SVD entropy effective rank of g_feat flattened to (B*H*W, D),
    subsampled to max 512 spatial tokens and divided by D so it lies in (0, 1].
    """
    _, dim, _, _ = g_feat.shape
    tokens = g_feat.float().permute(0, 2, 3, 1).reshape(-1, dim)
    num_tokens = tokens.shape[0]
    if num_tokens > max_tokens:
        idx = torch.linspace(0, num_tokens - 1, steps=max_tokens, device=tokens.device).long()
        tokens = tokens.index_select(0, idx)
    if tokens.shape[0] > 1:
        tokens = tokens - tokens.mean(dim=0, keepdim=True)
    svals = torch.linalg.svdvals(tokens)
    s_sum = svals.sum()
    probs = svals / (s_sum + 1e-12)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
    erank = torch.exp(entropy)
    fallback = torch.full((), 1.0 / float(dim), device=g_feat.device, dtype=torch.float32)
    return torch.where(
        s_sum > 1e-12,
        (erank / float(dim)).clamp(min=1e-6, max=1.0),
        fallback,
    )


# ============================================================================
# 3. SubspaceGeometricProduct & RollingGeometricProduct Modules
# ============================================================================


def compute_inner_wide_aligned_dim(dim: int, k: int) -> int:
    """
    Compute widened aligned_dim (multiple of k) so cli_mode="inner_wide" matches
    the reaction parameter count and projection MACs of cli_mode="full" at (dim, k).
    """
    d_tilde_ref = math.ceil(dim / k) * k
    ratio = ((k + 5) * dim / 2.0 + 9.0) / (3.0 * dim + 9.0)
    return max(k, int(round((d_tilde_ref * ratio) / k)) * k)


class SubspaceGeometricProduct(nn.Module):
    """
    Subspace Clifford reaction module (arXiv:2610.37226 Sec. 3.3.1 & 3.4.2):
        [Z_1, Z_2'] = W_in(BN(X_sp))
        Z_2 = DWConv_{3x3}(Z_2')
        F_gp = SubspaceGP_k(Z_1, Z_2)
        G_feat = W_out(F_gp)
    All conv projections (W_in, ctx_dw, W_out) are bias-free per paper specification.
    """

    def __init__(
        self,
        dim: int,
        k: int = 4,
        cli_mode: str = "full",
        ctx_mode: str = "diff",
        wedge_mode: str = "fma",
        backend: str = "auto",
        use_pre_bn: bool = True,
        norm_mode: str = "bn",
        use_global_context: bool = False,
        ctx_bn: bool = False,
        proj_bias: bool = False,
        split_in_proj: bool = False,
    ) -> None:
        super().__init__()
        if k not in VALID_SUBSPACE_K:
            raise ValueError(f"k must be in {VALID_SUBSPACE_K}, got {k}")
        if cli_mode not in VALID_CLI_MODES:
            raise ValueError(f"cli_mode must be in {VALID_CLI_MODES}, got {cli_mode!r}")
        if ctx_mode not in VALID_CTX_MODES:
            raise ValueError(f"ctx_mode must be in {VALID_CTX_MODES}, got {ctx_mode!r}")
        if wedge_mode not in VALID_WEDGE_MODES:
            raise ValueError(f"wedge_mode must be in {VALID_WEDGE_MODES}, got {wedge_mode!r}")
        if backend not in VALID_BACKENDS:
            raise ValueError(f"backend must be in {VALID_BACKENDS}, got {backend!r}")
        if norm_mode not in VALID_NORM_MODES:
            raise ValueError(f"norm_mode must be in {VALID_NORM_MODES}, got {norm_mode!r}")

        self.dim = dim
        self.k = k
        self.cli_mode = cli_mode
        self.ctx_mode = ctx_mode
        self.wedge_mode = wedge_mode
        self.backend = backend
        self.use_pre_bn = use_pre_bn
        self.norm_mode = norm_mode
        self.use_global_context = use_global_context
        self.use_ctx_bn = ctx_bn
        self.proj_bias = proj_bias

        if cli_mode == "inner_wide":
            self.aligned_dim = compute_inner_wide_aligned_dim(dim, k)
            self.c_out = self.aligned_dim
        else:
            self.aligned_dim = math.ceil(dim / k) * k
            num_pairs = (k * (k - 1)) // 2
            num_groups = self.aligned_dim // k
            if cli_mode in ("full", "sym_full", "bilinear_full", "swiglu_wedge"):
                self.c_out = num_groups * (num_pairs + k)
            elif cli_mode == "wedge":
                self.c_out = num_groups * num_pairs
            elif cli_mode == "inner":
                self.c_out = self.aligned_dim
            else:
                raise ValueError(f"Unhandled cli_mode: {cli_mode!r}")

        self.num_groups = self.aligned_dim // k

        if not use_pre_bn:
            self.pre_bn: nn.Module = nn.Identity()
        elif norm_mode == "ln2d":
            self.pre_bn = LayerNorm2d(dim)
        elif norm_mode in ("rmsnorm2d", "rmsnorm2d_sphere"):
            self.pre_bn = RMSNorm2d(dim)
        else:
            self.pre_bn = nn.BatchNorm2d(dim)

        self.split_in_proj = split_in_proj
        if split_in_proj:
            # Separate dense 1x1 convs keep z1 and z2_prime contiguous in channels_last (NHWC),
            # avoiding .chunk(2, dim=1) strided views and hitting the fast coalesced Triton NHWC kernel.
            self.get_state_proj = nn.Conv2d(dim, self.aligned_dim, kernel_size=1, bias=proj_bias)
            self.get_ctx_proj = nn.Conv2d(dim, self.aligned_dim, kernel_size=1, bias=proj_bias)
            self.get_state_ctx = None
        else:
            self.get_state_ctx = nn.Conv2d(dim, 2 * self.aligned_dim, kernel_size=1, bias=proj_bias)
            self.get_state_proj = None
            self.get_ctx_proj = None
        self.ctx_dw = nn.Conv2d(
            self.aligned_dim,
            self.aligned_dim,
            kernel_size=3,
            padding=1,
            groups=self.aligned_dim,
            bias=False,
        )
        if ctx_bn:
            if norm_mode == "ln2d":
                self.ctx_bn_layer: nn.Module = LayerNorm2d(self.aligned_dim)
            elif norm_mode in ("rmsnorm2d", "rmsnorm2d_sphere"):
                self.ctx_bn_layer = RMSNorm2d(self.aligned_dim)
            else:
                self.ctx_bn_layer = nn.BatchNorm2d(self.aligned_dim)
        else:
            self.ctx_bn_layer = nn.Identity()
        if use_global_context:
            # Linear-O(N) global mean-field superposition (Paper 1 Sec. 3.4 / 3.7 adapted to subspaces):
            # mixes each k-subspace of GlobalAvgPool(z2_prime) and superposes into z2 via learned beta.
            self.ctx_global_proj = nn.Conv2d(
                self.aligned_dim,
                self.aligned_dim,
                kernel_size=1,
                groups=self.num_groups,
                bias=False,
            )
            beta_shape = (self.aligned_dim,) if split_in_proj else (1, self.aligned_dim, 1, 1)
            self.global_beta = nn.Parameter(0.1 * torch.ones(beta_shape))
        else:
            self.ctx_global_proj = None
            self.global_beta = None

        self.final_proj = nn.Conv2d(self.c_out, dim, kernel_size=1, bias=proj_bias)

        self._triton_eligible_mode = (
            self.k in (2, 3, 4)
            and self.cli_mode in ("full", "inner", "wedge")
            and self.wedge_mode in ("fma", "fp32", "naive")
        )
        triton_fn, param_names = _resolve_triton_subspace()
        self._triton_fn = triton_fn
        self._triton_takes_wedge = "wedge_mode" in param_names
        self._triton_takes_use_fma = "use_fma" in param_names

    def _should_use_triton(self, x: torch.Tensor) -> bool:
        if not x.is_cuda or not self._triton_eligible_mode:
            return False
        if self.backend not in ("triton", "auto"):
            return False
        # In "auto" mode with reduced precision, "naive" must perform native-dtype
        # subtraction (no fp32 upcast) to faithfully evaluate bf16/fp16 cancellation.
        if (
            self.backend == "auto"
            and self.wedge_mode == "naive"
            and x.dtype != torch.float32
            and not self._triton_takes_wedge
        ):
            return False
        if self._triton_fn is None and not torch.compiler.is_compiling():
            fn, param_names = _resolve_triton_subspace()
            self._triton_fn = fn
            self._triton_takes_wedge = "wedge_mode" in param_names
            self._triton_takes_use_fma = "use_fma" in param_names
        return self._triton_fn is not None

    def _prepare_streams(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.pre_bn(x)
        if self.split_in_proj:
            assert self.get_state_proj is not None and self.get_ctx_proj is not None
            z1 = self.get_state_proj(h)
            z2_prime = self.get_ctx_proj(h)
        else:
            assert self.get_state_ctx is not None
            z1, z2_prime = self.get_state_ctx(h).chunk(2, dim=1)
        z2 = self.ctx_bn_layer(self.ctx_dw(z2_prime)).to(z1.dtype)
        if self.ctx_global_proj is not None and self.global_beta is not None:
            z2_glo = self.ctx_global_proj(z2_prime.mean(dim=(-2, -1), keepdim=True))
            beta = self.global_beta.flatten().view(1, -1, 1, 1).to(z1.dtype)
            z2 = (z2 + beta * z2_glo.to(z1.dtype)).to(z1.dtype)
        if self.norm_mode == "subspace_rms":
            B, _, H, W = z1.shape
            dt = z1.dtype
            z1_g = z1.float().reshape(B, self.num_groups, self.k, H, W)
            z2_g = z2.float().reshape(B, self.num_groups, self.k, H, W)
            z1_rms = torch.rsqrt(z1_g.pow(2).mean(dim=2, keepdim=True) + 1e-6)
            z2_rms = torch.rsqrt(z2_g.pow(2).mean(dim=2, keepdim=True) + 1e-6)
            z1 = (z1_g * z1_rms).to(dt).reshape(B, self.aligned_dim, H, W)
            z2 = (z2_g * z2_rms).to(dt).reshape(B, self.aligned_dim, H, W)
        elif self.norm_mode in ("onesided_sphere", "rmsnorm2d_sphere"):
            # Degree-1 homogeneous Clifford interaction: project only context stream z2 onto
            # each k-dimensional unit-RMS sphere while preserving state magnitude ||z1||.
            B, _, H, W = z2.shape
            dt = z2.dtype
            # Keep channels_last (NHWC) layout intact so Triton hits the coalesced NHWC kernel
            if z2.is_contiguous(memory_format=torch.channels_last):
                z2_nhwc = z2.permute(0, 2, 3, 1).float().reshape(B, H, W, self.num_groups, self.k)
                z2_rms = torch.rsqrt(z2_nhwc.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
                z2 = (z2_nhwc * z2_rms).to(dt).reshape(B, H, W, self.aligned_dim).permute(0, 3, 1, 2)
            else:
                z2_g = z2.float().reshape(B, self.num_groups, self.k, H, W)
                z2_rms = torch.rsqrt(z2_g.pow(2).mean(dim=2, keepdim=True) + 1e-6)
                z2 = (z2_g * z2_rms).to(dt).reshape(B, self.aligned_dim, H, W)
        return z1, z2

    def forward(
        self,
        x: torch.Tensor,
        compute_ortho: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        z1, z2 = self._prepare_streams(x)

        if self._should_use_triton(x):
            assert self._triton_fn is not None
            if self._triton_takes_wedge:
                f_gp = self._triton_fn(
                    z1,
                    z2,
                    self.k,
                    cli_mode=self.cli_mode,
                    ctx_mode=self.ctx_mode,
                    wedge_mode=self.wedge_mode,
                )
            elif self._triton_takes_use_fma:
                f_gp = self._triton_fn(
                    z1,
                    z2,
                    self.k,
                    cli_mode=self.cli_mode,
                    ctx_mode=self.ctx_mode,
                    use_fma=(self.wedge_mode == "fma"),
                )
            else:
                f_gp = self._triton_fn(
                    z1,
                    z2,
                    self.k,
                    cli_mode=self.cli_mode,
                    ctx_mode=self.ctx_mode,
                )
        else:
            f_gp = subspace_gp_pytorch(
                z1,
                z2,
                self.k,
                cli_mode=self.cli_mode,
                ctx_mode=self.ctx_mode,
                wedge_mode=self.wedge_mode,
            )

        g_feat = self.final_proj(f_gp)

        if compute_ortho:
            B, _, H, W = z1.shape
            u_g = z1.reshape(B, self.num_groups, self.k, H * W)
            z2_g = z2.reshape(B, self.num_groups, self.k, H * W)
            ortho_loss = F.cosine_similarity(u_g, z2_g, dim=2).abs().mean()
            return g_feat, ortho_loss

        return g_feat

    def forward_diagnostics(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Diagnostic-only no-grad pass returning detached scalar tensors."""
        with torch.no_grad():
            z1, z2 = self._prepare_streams(x)

            B, _, H, W = z1.shape
            stream_dtype = z1.dtype
            z1_g = z1.reshape(B, self.num_groups, self.k, H, W)
            z2_g = z2.reshape(B, self.num_groups, self.k, H, W)
            u = z1_g
            v = (z2_g - z1_g) if self.ctx_mode == "diff" else z2_g

            f_gp = subspace_gp_pytorch(
                z1,
                z2,
                self.k,
                cli_mode=self.cli_mode,
                ctx_mode=self.ctx_mode,
                wedge_mode=self.wedge_mode,
            )
            g_feat = self.final_proj(f_gp)

            zero = torch.zeros((), device=x.device, dtype=torch.float32)

            # Cancellation & off-diagonal magnitudes
            if self.cli_mode in ("full", "wedge", "swiglu_wedge"):
                vw = z2_g if self.wedge_mode in ("direct_z2", "direct_z2_fma") else v
                u_i, vw_j, u_j, vw_i = _extract_group_pairs(u, vw, self.k)
                p_offdiag = _compute_wedge_diff(u_i, vw_j, u_j, vw_i, self.wedge_mode, stream_dtype)
                term_a = (u_i * vw_j).float()
                term_b = (u_j * vw_i).float()
                rel_diff = (term_a - term_b).abs() / (term_a.abs() + term_b.abs() + 1e-12)
                rel_diff_mean = rel_diff.mean()
                rel_diff_lt1e2 = (rel_diff < 1e-2).float().mean()
                rel_diff_lt1e4 = (rel_diff < 1e-4).float().mean()
                wedge_abs = p_offdiag.float().abs()
                wedge_abs_mean = wedge_abs.mean()
                wedge_abs_max = wedge_abs.max()
            elif self.cli_mode == "sym_full":
                u_i, v_j, u_j, v_i = _extract_group_pairs(u, v, self.k)
                p_offdiag = (u_i.float() * v_j.float() + u_j.float() * v_i.float()).to(stream_dtype)
                term_a = (u_i * v_j).float()
                term_b = (u_j * v_i).float()
                rel_diff = (term_a + term_b).abs() / (term_a.abs() + term_b.abs() + 1e-12)
                rel_diff_mean = rel_diff.mean()
                rel_diff_lt1e2 = (rel_diff < 1e-2).float().mean()
                rel_diff_lt1e4 = (rel_diff < 1e-4).float().mean()
                wedge_abs = p_offdiag.float().abs()
                wedge_abs_mean = wedge_abs.mean()
                wedge_abs_max = wedge_abs.max()
            elif self.cli_mode == "bilinear_full":
                u_i, v_j, u_j, v_i = _extract_group_pairs(u, v, self.k)
                p_offdiag = (u_i.float() * v_j.float()).to(stream_dtype)
                term_a = (u_i * v_j).float()
                term_b = (u_j * v_i).float()
                rel_diff = (term_a - term_b).abs() / (term_a.abs() + term_b.abs() + 1e-12)
                rel_diff_mean = rel_diff.mean()
                rel_diff_lt1e2 = (rel_diff < 1e-2).float().mean()
                rel_diff_lt1e4 = (rel_diff < 1e-4).float().mean()
                wedge_abs = p_offdiag.float().abs()
                wedge_abs_mean = wedge_abs.mean()
                wedge_abs_max = wedge_abs.max()
            else:
                rel_diff_mean = zero
                rel_diff_lt1e2 = zero
                rel_diff_lt1e4 = zero
                wedge_abs_mean = zero
                wedge_abs_max = zero

            # Inner/diagonal magnitude
            if self.cli_mode == "swiglu_wedge":
                p_inner = F.silu(u) * v
                dot_abs_mean = p_inner.float().abs().mean()
            elif self.cli_mode in ("full", "inner", "inner_wide", "sym_full", "bilinear_full"):
                p_inner = F.silu(u * v)
                dot_abs_mean = p_inner.float().abs().mean()
            else:
                dot_abs_mean = zero

            wedge_to_dot_ratio = wedge_abs_mean / (dot_abs_mean + 1e-12)

            # Full-channel and within-k-group cosine similarity
            cos_sim_full = F.cosine_similarity(
                z1.float().flatten(2),
                z2.float().flatten(2),
                dim=1,
            ).abs().mean()
            subspace_cos_sim = F.cosine_similarity(
                u.float(),
                v.float(),
                dim=2,
            ).abs().mean()

            # Lambda-invariance error in stream_dtype: |wedge(z1, z2 - z1) - wedge(z1, z2)| / (|wedge(z1, z2)| + 1e-6)
            v_diff_dt = z2_g - z1_g
            u_i_d, vd_j, u_j_d, vd_i = _extract_group_pairs(u, v_diff_dt, self.k)
            _, z2_j, _, z2_i = _extract_group_pairs(u, z2_g, self.k)
            w_diff = (u_i_d * vd_j - u_j_d * vd_i).to(stream_dtype)
            w_abs = (u_i_d * z2_j - u_j_d * z2_i).to(stream_dtype)
            lambda_wedge_rel_err = (
                (w_diff.float() - w_abs.float()).abs() / (w_abs.float().abs() + 1e-6)
            ).mean()

            effective_rank = _compute_effective_rank_normalized(g_feat)

            return {
                "cancel/rel_diff_mean": rel_diff_mean.detach(),
                "cancel/rel_diff_lt1e-2": rel_diff_lt1e2.detach(),
                "cancel/rel_diff_lt1e-4": rel_diff_lt1e4.detach(),
                "magnitude/wedge_abs_mean": wedge_abs_mean.detach(),
                "magnitude/wedge_abs_max": wedge_abs_max.detach(),
                "magnitude/dot_abs_mean": dot_abs_mean.detach(),
                "magnitude/wedge_to_dot_ratio": wedge_to_dot_ratio.detach(),
                "health/nan_count": (f_gp.isnan().sum() + g_feat.isnan().sum()).float().detach(),
                "health/inf_count": (f_gp.isinf().sum() + g_feat.isinf().sum()).float().detach(),
                "ortho/cos_sim_mean": cos_sim_full.detach(),
                "ortho/subspace_cos_sim_mean": subspace_cos_sim.detach(),
                "invariance/lambda_wedge_rel_err": lambda_wedge_rel_err.detach(),
                "spectrum/effective_rank": effective_rank.detach(),
            }


class RollingGeometricProduct(nn.Module):
    """
    Sparse rolling Clifford interaction (arXiv:2601.06793 / Paper 2 Table 2 baseline),
    supporting both GGR (use_pre_bn=False, preceded by LayerNorm2d) and Lie-Trotter
    (use_pre_bn=True, preceded by SpatialDiffusion) macro-blocks.
    """

    def __init__(
        self,
        dim: int,
        shifts: Sequence[int] = (1, 2, 4),
        cli_mode: str = "full",
        ctx_mode: str = "diff",
        wedge_mode: str = "fma",
        use_pre_bn: bool = False,
    ) -> None:
        super().__init__()
        if cli_mode not in ("full", "wedge", "inner"):
            raise ValueError(f"RollingGeometricProduct cli_mode must be full/wedge/inner, got {cli_mode!r}")
        if ctx_mode not in VALID_CTX_MODES:
            raise ValueError(f"ctx_mode must be in {VALID_CTX_MODES}, got {ctx_mode!r}")
        if wedge_mode not in VALID_WEDGE_MODES:
            raise ValueError(f"wedge_mode must be in {VALID_WEDGE_MODES}, got {wedge_mode!r}")

        valid_shifts = tuple(s for s in shifts if 0 < s < dim)
        if not valid_shifts:
            raise ValueError(f"No valid shifts < dim={dim} in {shifts!r}")

        self.dim = dim
        self.shifts = valid_shifts
        self.cli_mode = cli_mode
        self.ctx_mode = ctx_mode
        self.wedge_mode = wedge_mode
        self.use_pre_bn = use_pre_bn

        self.pre_bn: nn.Module = nn.BatchNorm2d(dim) if use_pre_bn else nn.Identity()
        self.get_context_local = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
        )
        self.get_state = nn.Conv2d(dim, dim, kernel_size=1)

        branch_mult = 2 if cli_mode == "full" else 1
        input_proj_dim = branch_mult * len(self.shifts) * dim
        self.final_proj = nn.Conv2d(input_proj_dim, dim, kernel_size=1)

        base = torch.arange(dim)
        roll_idx = torch.stack([(base - s) % dim for s in self.shifts], dim=0)
        self.register_buffer("_roll_idx", roll_idx, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        compute_ortho: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        h = self.pre_bn(x)
        z_state = self.get_state(h)
        target_dtype = z_state.dtype
        z_ctx = self.get_context_local(h).to(target_dtype)

        c_vec = (z_ctx - z_state) if self.ctx_mode == "diff" else z_ctx
        c_wedge = z_ctx if self.wedge_mode in ("direct_z2", "direct_z2_fma") else c_vec

        B, Ch, H, W = z_state.shape
        num_shifts = len(self.shifts)
        z_state_b = z_state.unsqueeze(1)
        z_state_rolled = z_state[:, self._roll_idx]

        if self.cli_mode == "inner":
            c_rolled = c_vec[:, self._roll_idx]
            dot = F.silu(z_state_b * c_rolled)
            g_raw = dot.reshape(B, num_shifts * Ch, H, W)
        elif self.cli_mode == "wedge":
            cw_rolled = c_wedge[:, self._roll_idx]
            cw_b = c_wedge.unsqueeze(1)
            wedge = _compute_wedge_diff(
                z_state_b, cw_rolled, z_state_rolled, cw_b, self.wedge_mode, target_dtype
            )
            g_raw = wedge.reshape(B, num_shifts * Ch, H, W)
        else:
            c_rolled = c_vec[:, self._roll_idx]
            dot = F.silu(z_state_b * c_rolled)
            cw_rolled = c_wedge[:, self._roll_idx] if c_wedge is not c_vec else c_rolled
            cw_b = c_wedge.unsqueeze(1)
            wedge = _compute_wedge_diff(
                z_state_b, cw_rolled, z_state_rolled, cw_b, self.wedge_mode, target_dtype
            )
            pairs = torch.stack([dot, wedge], dim=2)
            g_raw = pairs.reshape(B, num_shifts * 2 * Ch, H, W)

        g_feat = self.final_proj(g_raw)

        if compute_ortho:
            cos_sim = F.cosine_similarity(z_state.flatten(2), z_ctx.flatten(2), dim=1)
            return g_feat, cos_sim.abs().mean()
        return g_feat

    def forward_diagnostics(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            h = self.pre_bn(x)
            z_state = self.get_state(h)
            z_ctx = self.get_context_local(h).to(z_state.dtype)
            stream_dtype = z_state.dtype

            c_vec = (z_ctx - z_state) if self.ctx_mode == "diff" else z_ctx
            c_wedge = z_ctx if self.wedge_mode in ("direct_z2", "direct_z2_fma") else c_vec

            B, Ch, H, W = z_state.shape
            num_shifts = len(self.shifts)
            z_state_b = z_state.unsqueeze(1)
            z_state_rolled = z_state[:, self._roll_idx]
            c_rolled = c_vec[:, self._roll_idx]
            cw_rolled = c_wedge[:, self._roll_idx]
            cw_b = c_wedge.unsqueeze(1)

            zero = torch.zeros((), device=x.device, dtype=torch.float32)

            if self.cli_mode in ("full", "wedge"):
                wedge = _compute_wedge_diff(
                    z_state_b, cw_rolled, z_state_rolled, cw_b, self.wedge_mode, stream_dtype
                )
                term_a = (z_state_b * cw_rolled).float()
                term_b = (z_state_rolled * cw_b).float()
                rel_diff = (term_a - term_b).abs() / (term_a.abs() + term_b.abs() + 1e-12)
                rel_diff_mean = rel_diff.mean()
                rel_diff_lt1e2 = (rel_diff < 1e-2).float().mean()
                rel_diff_lt1e4 = (rel_diff < 1e-4).float().mean()
                wedge_abs = wedge.float().abs()
                wedge_abs_mean = wedge_abs.mean()
                wedge_abs_max = wedge_abs.max()
            else:
                wedge = None
                rel_diff_mean = zero
                rel_diff_lt1e2 = zero
                rel_diff_lt1e4 = zero
                wedge_abs_mean = zero
                wedge_abs_max = zero

            if self.cli_mode in ("full", "inner"):
                dot = F.silu(z_state_b * c_rolled)
                dot_abs_mean = dot.float().abs().mean()
            else:
                dot = None
                dot_abs_mean = zero

            if self.cli_mode == "inner":
                assert dot is not None
                g_raw = dot.reshape(B, num_shifts * Ch, H, W)
            elif self.cli_mode == "wedge":
                assert wedge is not None
                g_raw = wedge.reshape(B, num_shifts * Ch, H, W)
            else:
                assert dot is not None and wedge is not None
                g_raw = torch.stack([dot, wedge], dim=2).reshape(B, num_shifts * 2 * Ch, H, W)

            g_feat = self.final_proj(g_raw)

            cos_sim = F.cosine_similarity(
                z_state.float().flatten(2),
                z_ctx.float().flatten(2),
                dim=1,
            ).abs().mean()
            pair_u = torch.stack([z_state_b.expand_as(z_state_rolled), z_state_rolled], dim=3)
            pair_v = torch.stack([c_vec.unsqueeze(1).expand_as(c_rolled), c_rolled], dim=3)
            subspace_cos = F.cosine_similarity(pair_u.float(), pair_v.float(), dim=3).abs().mean()

            c_diff_dt = z_ctx - z_state
            cd_rolled = c_diff_dt[:, self._roll_idx]
            cd_b = c_diff_dt.unsqueeze(1)
            zctx_rolled = z_ctx[:, self._roll_idx]
            zctx_b = z_ctx.unsqueeze(1)
            w_diff = (z_state_b * cd_rolled - z_state_rolled * cd_b).to(stream_dtype)
            w_abs = (z_state_b * zctx_rolled - z_state_rolled * zctx_b).to(stream_dtype)
            lambda_wedge_rel_err = (
                (w_diff.float() - w_abs.float()).abs() / (w_abs.float().abs() + 1e-6)
            ).mean()

            effective_rank = _compute_effective_rank_normalized(g_feat)

            return {
                "cancel/rel_diff_mean": rel_diff_mean.detach(),
                "cancel/rel_diff_lt1e-2": rel_diff_lt1e2.detach(),
                "cancel/rel_diff_lt1e-4": rel_diff_lt1e4.detach(),
                "magnitude/wedge_abs_mean": wedge_abs_mean.detach(),
                "magnitude/wedge_abs_max": wedge_abs_max.detach(),
                "magnitude/dot_abs_mean": dot_abs_mean.detach(),
                "magnitude/wedge_to_dot_ratio": (wedge_abs_mean / (dot_abs_mean + 1e-12)).detach(),
                "health/nan_count": (g_raw.isnan().sum() + g_feat.isnan().sum()).float().detach(),
                "health/inf_count": (g_raw.isinf().sum() + g_feat.isinf().sum()).float().detach(),
                "ortho/cos_sim_mean": cos_sim.detach(),
                "ortho/subspace_cos_sim_mean": subspace_cos.detach(),
                "invariance/lambda_wedge_rel_err": lambda_wedge_rel_err.detach(),
                "spectrum/effective_rank": effective_rank.detach(),
            }


# ============================================================================
# 4. SubspaceCliffordBlock (2x2 Factorial: {subspace, rolling} x {lie_trotter, ggr})
# ============================================================================


class SubspaceCliffordBlock(nn.Module):
    """
    Unified Clifford block supporting the full 2x2 factorial deconfounding design:
      - block_style="lie_trotter": Paper 2 operator-splitting (SpatialDiffusion -> Reaction)
      - block_style="ggr"        : Paper 1 Gated Geometric Residual (LayerNorm2d -> Gate + SiLU)
      - interaction_type="subspace": Grouped SubspaceGeometricProduct
      - interaction_type="rolling" : Cyclic shift RollingGeometricProduct
    """

    def __init__(
        self,
        dim: int,
        k: int = 4,
        cli_mode: str = "full",
        ctx_mode: str = "diff",
        wedge_mode: str = "fma",
        use_spatial_diffusion: bool = True,
        block_style: str = "lie_trotter",
        interaction_type: str = "subspace",
        shifts: Sequence[int] = (1, 2, 4),
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-5,
        dw_kernel_size: int = 5,
        backend: str = "auto",
        norm_mode: str = "bn",
        use_global_context: bool = False,
        ctx_bn: bool = False,
        proj_bias: bool = False,
        split_in_proj: bool = False,
    ) -> None:
        super().__init__()
        if block_style not in VALID_BLOCK_STYLES:
            raise ValueError(f"block_style must be in {VALID_BLOCK_STYLES}, got {block_style!r}")
        if interaction_type not in VALID_INTERACTION_TYPES:
            raise ValueError(
                f"interaction_type must be in {VALID_INTERACTION_TYPES}, got {interaction_type!r}"
            )

        self.dim = dim
        self.k = k
        self.block_style = block_style
        self.interaction_type = interaction_type
        self.use_spatial_diffusion = use_spatial_diffusion and (block_style == "lie_trotter")

        if block_style == "lie_trotter":
            self.spatial_diffusion: nn.Module = (
                SpatialDiffusion(
                    dim=dim,
                    kernel_size=dw_kernel_size,
                    layer_scale_init_value=layer_scale_init_value,
                    norm_type=(
                        "ln2d"
                        if norm_mode == "ln2d"
                        else ("rmsnorm2d" if norm_mode in ("rmsnorm2d", "rmsnorm2d_sphere") else "bn")
                    ),
                    flat_gamma=split_in_proj,
                )
                if use_spatial_diffusion
                else nn.Identity()
            )
            use_pre_bn = True
        else:
            self.norm = LayerNorm2d(dim)
            self.gate_linear = nn.Conv2d(dim * 2, dim, kernel_size=1)
            use_pre_bn = False

        if interaction_type == "subspace":
            self.interaction: nn.Module = SubspaceGeometricProduct(
                dim=dim,
                k=k,
                cli_mode=cli_mode,
                ctx_mode=ctx_mode,
                wedge_mode=wedge_mode,
                backend=backend,
                use_pre_bn=use_pre_bn,
                norm_mode=norm_mode,
                use_global_context=use_global_context,
                ctx_bn=ctx_bn,
                proj_bias=proj_bias,
                split_in_proj=split_in_proj,
            )
        else:
            self.interaction = RollingGeometricProduct(
                dim=dim,
                shifts=shifts,
                cli_mode=cli_mode if cli_mode in ("full", "wedge", "inner") else "full",
                ctx_mode=ctx_mode,
                wedge_mode=wedge_mode,
                use_pre_bn=use_pre_bn,
            )

        gamma_shape = (dim,) if split_in_proj else (1, dim, 1, 1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(gamma_shape))
        self.drop_path: nn.Module = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        compute_ortho: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.block_style == "lie_trotter":
            x_sp = self.spatial_diffusion(x)
            gamma = self.gamma.flatten().view(1, -1, 1, 1)
            if compute_ortho:
                g_feat, ortho_loss = self.interaction(x_sp, compute_ortho=True)
                y = x_sp + self.drop_path(gamma * g_feat)
                return y, ortho_loss
            g_feat = self.interaction(x_sp)
            return x_sp + self.drop_path(gamma * g_feat)

        shortcut = x
        x_ln = self.norm(x)
        if compute_ortho:
            g_feat, ortho_loss = self.interaction(x_ln, compute_ortho=True)
        else:
            g_feat = self.interaction(x_ln)
            ortho_loss = None

        alpha = torch.sigmoid(self.gate_linear(torch.cat([x_ln, g_feat], dim=1)))
        h_mix = F.silu(x_ln) + alpha * g_feat
        y = shortcut + self.drop_path(self.gamma * h_mix)
        if compute_ortho:
            assert ortho_loss is not None
            return y, ortho_loss
        return y

    def forward_diagnostics_and_step(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run single-block diagnostics + step outside torch.compile under no_grad."""
        with torch.no_grad():
            gamma = self.gamma.flatten().view(1, -1, 1, 1)
            if self.block_style == "lie_trotter":
                x_sp = self.spatial_diffusion(x)
                diags = self.interaction.forward_diagnostics(x_sp)
                g_feat = self.interaction(x_sp)
                return x_sp + self.drop_path(gamma * g_feat), diags

            x_ln = self.norm(x)
            diags = self.interaction.forward_diagnostics(x_ln)
            g_feat = self.interaction(x_ln)
            alpha = torch.sigmoid(self.gate_linear(torch.cat([x_ln, g_feat], dim=1)))
            h_mix = F.silu(x_ln) + alpha * g_feat
            return x + self.drop_path(gamma * h_mix), diags


# ============================================================================
# 5. Backbones & Timm Baseline Wrapper
# ============================================================================


def _init_subspace_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, LayerNorm2d)):
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias, 0)
        if getattr(m, "weight", None) is not None:
            nn.init.constant_(m.weight, 1.0)


class IsotropicSubspaceCliffordNet(nn.Module):
    """Single-resolution columnar Subspace Clifford Network (Sec. 4.1-4.2)."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int | None = None,
        in_chans: int = 3,
        num_classes: int = 100,
        embed_dim: int = 128,
        depth: int = 12,
        k: int = 4,
        cli_mode: str = "full",
        ctx_mode: str = "diff",
        wedge_mode: str = "fma",
        use_spatial_diffusion: bool = True,
        block_style: str = "lie_trotter",
        interaction_type: str = "subspace",
        shifts: Sequence[int] = (1, 2),
        drop_path_rate: float = 0.1,
        layer_scale_init_value: float = 1e-5,
        dw_kernel_size: int = 5,
        backend: str = "auto",
        norm_mode: str = "bn",
        use_global_context: bool = False,
    ) -> None:
        super().__init__()
        if patch_size is None:
            patch_size = 2 if img_size <= 64 else 4

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.k = k

        self.patch_embed = GeometricStem(
            in_chans=in_chans,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )

        dpr = [v.item() for v in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                SubspaceCliffordBlock(
                    dim=embed_dim,
                    k=k,
                    cli_mode=cli_mode,
                    ctx_mode=ctx_mode,
                    wedge_mode=wedge_mode,
                    use_spatial_diffusion=use_spatial_diffusion,
                    block_style=block_style,
                    interaction_type=interaction_type,
                    shifts=shifts,
                    drop_path=dpr[i],
                    layer_scale_init_value=layer_scale_init_value,
                    dw_kernel_size=dw_kernel_size,
                    backend=backend,
                    norm_mode=norm_mode,
                    use_global_context=use_global_context,
                )
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(_init_subspace_weights)

    def get_interaction_layers(self) -> list[nn.Module]:
        return [block.interaction for block in self.blocks]

    def iter_blocks(self) -> Iterator[tuple[str, SubspaceCliffordBlock]]:
        for idx, block in enumerate(self.blocks):
            yield f"block_{idx}", block

    def diagnostic_block_labels(self) -> list[str]:
        return [label for label, _ in self.iter_blocks()]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        return x

    def forward_diagnostics(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        all_diags: dict[str, torch.Tensor] = {}
        param_dtype = next(self.parameters()).dtype
        need_autocast = (
            x.is_cuda
            and x.dtype in (torch.bfloat16, torch.float16)
            and param_dtype == torch.float32
        )
        with torch.no_grad(), torch.autocast(
            device_type="cuda" if x.is_cuda else "cpu",
            dtype=x.dtype if need_autocast else torch.bfloat16,
            enabled=need_autocast,
        ):
            x = self.patch_embed(x)
            for label, block in self.iter_blocks():
                x, layer_diags = block.forward_diagnostics_and_step(x)
                for key, val in layer_diags.items():
                    all_diags[f"{label}/{key}"] = val
        return all_diags

    def forward(
        self,
        x: torch.Tensor,
        compute_ortho: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if compute_ortho:
            x = self.patch_embed(x)
            ortho_losses: list[torch.Tensor] = []
            for block in self.blocks:
                x, ol = block(x, compute_ortho=True)
                ortho_losses.append(ol)
            x = x.mean(dim=(-2, -1))
            logits = self.head(self.norm(x))
            return logits, torch.stack(ortho_losses).mean()

        x = self.forward_features(x)
        x = x.mean(dim=(-2, -1))
        return self.head(self.norm(x))


class HierarchicalSubspaceCliffordNet(nn.Module):
    """
    Multi-resolution Hierarchical Subspace Clifford Network (arXiv:2610.37226 Sec. 3.4).

    Note on CIFAR-100 Stem & Exact Parameter Verification:
      With `stem_type="geometric"`, `embed_dim=32`, and `patch_size=1` (matching `CAN/model_hier.py`),
      `GeometricStem(3 -> 16 -> 32) + stem_proj(32 -> 96)` preserves spatial resolution
      `32x32 -> 16x16 -> 8x8` across the 3 stages `(96, 128, 160)` and reproduces ALL 13
      reported parameter counts in Tables 2, 3, and 4 of arXiv:2610.37226 within <= 0.0005M:
        - Hier-Subspace-CAN k=2: 0.898M (898,388)   | Rolling S={1,2,4}:   2.093M (2,092,500)
        - Hier-Subspace-CAN k=3: 1.018M (1,017,554) | Rolling S={1,2,4,8}: 2.535M (2,534,868)
        - Hier-Subspace-CAN k=4: 1.120M (1,119,572) | Hybrid (2,3,4):      1.061M (1,061,240)
        - Hybrid (4,3,2):        0.961M (960,888)   | Hybrid (2,4,4)-12:   1.092M (1,091,924)
        - Hybrid (2,4,4)-18:     1.651M (1,650,516) | Hybrid (2,4,4)-24:   2.209M (2,209,108)
        - Inner-only k=4:        0.788M (787,796)   | Wedge-only k=4:      0.898M (898,388)
        - SpatialDiffusion-free: 1.075M (1,074,772)
      `stem_type="conv3"` (`Conv3x3(3, stage_dims[0], bias=False) + BN + SiLU`) is also supported
      and reduces overhead by 5,520 parameters across all configurations.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int | None = None,
        in_chans: int = 3,
        num_classes: int = 100,
        embed_dim: int = 32,
        stage_depths: Sequence[int] = (3, 4, 5),
        stage_dims: Sequence[int] | None = (96, 128, 160),
        dim_policy: str = "double",
        subspace_k: int | Sequence[int] = 4,
        cli_mode: str = "full",
        ctx_mode: str = "diff",
        wedge_mode: str = "fma",
        use_spatial_diffusion: bool = True,
        block_style: str = "lie_trotter",
        interaction_type: str = "subspace",
        shifts: Sequence[int] = (1, 2, 4),
        downsample_mode: str = "conv",
        stem_type: str = "geometric",
        drop_path_rate: float = 0.1,
        layer_scale_init_value: float = 1e-5,
        dw_kernel_size: int = 5,
        backend: str = "auto",
        norm_mode: str = "bn",
        use_global_context: bool = False,
        ctx_bn: bool = False,
        proj_bias: bool = False,
        split_in_proj: bool = False,
    ) -> None:
        super().__init__()
        if len(stage_depths) == 0:
            raise ValueError("stage_depths must not be empty")

        if stage_dims is None:
            if dim_policy == "double":
                resolved_dims = [embed_dim * (2**i) for i in range(len(stage_depths))]
            elif dim_policy == "constant":
                resolved_dims = [embed_dim for _ in stage_depths]
            else:
                raise ValueError(f"Invalid dim_policy: {dim_policy!r}")
        else:
            if len(stage_dims) != len(stage_depths):
                raise ValueError("stage_dims must have the same length as stage_depths")
            resolved_dims = list(stage_dims)

        if isinstance(subspace_k, int):
            resolved_k = [subspace_k for _ in stage_depths]
        else:
            if len(subspace_k) != len(stage_depths):
                raise ValueError("subspace_k sequence must match length of stage_depths")
            resolved_k = list(subspace_k)

        if patch_size is None:
            patch_size = 1 if img_size <= 64 else 4

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.stage_depths = list(stage_depths)
        self.stage_dims = resolved_dims
        self.subspace_k = tuple(resolved_k)
        self.downsample_mode = downsample_mode
        self.stem_type = stem_type

        if stem_type == "geometric":
            self.patch_embed: nn.Module = GeometricStem(
                in_chans=in_chans,
                embed_dim=embed_dim,
                patch_size=patch_size,
            )
            self.stem_proj: nn.Module = (
                nn.Conv2d(embed_dim, resolved_dims[0], kernel_size=1)
                if embed_dim != resolved_dims[0]
                else nn.Identity()
            )
        elif stem_type == "conv3":
            self.patch_embed = nn.Sequential(
                nn.Conv2d(
                    in_chans,
                    resolved_dims[0],
                    kernel_size=3,
                    stride=patch_size,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(resolved_dims[0]),
                nn.SiLU(),
            )
            self.stem_proj = nn.Identity()
        else:
            raise ValueError(f"Invalid stem_type: {stem_type!r}")

        total_blocks = sum(self.stage_depths)
        dpr = [v.item() for v in torch.linspace(0, drop_path_rate, total_blocks)]

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        dp_idx = 0
        for stage_idx, (depth, dim, k_stage) in enumerate(
            zip(self.stage_depths, self.stage_dims, self.subspace_k)
        ):
            blocks = []
            for _ in range(depth):
                blocks.append(
                    SubspaceCliffordBlock(
                        dim=dim,
                        k=k_stage,
                        cli_mode=cli_mode,
                        ctx_mode=ctx_mode,
                        wedge_mode=wedge_mode,
                        use_spatial_diffusion=use_spatial_diffusion,
                        block_style=block_style,
                        interaction_type=interaction_type,
                        shifts=shifts,
                        drop_path=dpr[dp_idx],
                        layer_scale_init_value=layer_scale_init_value,
                        dw_kernel_size=dw_kernel_size,
                        backend=backend,
                        norm_mode=norm_mode,
                        use_global_context=use_global_context,
                        ctx_bn=ctx_bn,
                        proj_bias=proj_bias,
                        split_in_proj=split_in_proj,
                    )
                )
                dp_idx += 1
            self.stages.append(nn.Sequential(*blocks))

            if stage_idx < len(self.stage_depths) - 1:
                self.downsamples.append(
                    StageDownsample(
                        in_dim=self.stage_dims[stage_idx],
                        out_dim=self.stage_dims[stage_idx + 1],
                        mode=downsample_mode,
                    )
                )

        self.norm = nn.LayerNorm(self.stage_dims[-1])
        self.head = nn.Linear(self.stage_dims[-1], num_classes)
        self.apply(_init_subspace_weights)

    def get_interaction_layers(self) -> list[nn.Module]:
        return [block.interaction for _, block in self.iter_blocks()]

    def iter_blocks(self) -> Iterator[tuple[str, SubspaceCliffordBlock]]:
        for stage_idx, stage in enumerate(self.stages):
            for block_idx, block in enumerate(stage):
                assert isinstance(block, SubspaceCliffordBlock)
                yield f"stage_{stage_idx}_block_{block_idx}", block

    def diagnostic_block_labels(self) -> list[str]:
        return [label for label, _ in self.iter_blocks()]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_proj(self.patch_embed(x))
        for stage_idx, stage in enumerate(self.stages):
            x = stage(x)
            if stage_idx < len(self.downsamples):
                x = self.downsamples[stage_idx](x)
        return x

    def forward_diagnostics(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        all_diags: dict[str, torch.Tensor] = {}
        param_dtype = next(self.parameters()).dtype
        need_autocast = (
            x.is_cuda
            and x.dtype in (torch.bfloat16, torch.float16)
            and param_dtype == torch.float32
        )
        with torch.no_grad(), torch.autocast(
            device_type="cuda" if x.is_cuda else "cpu",
            dtype=x.dtype if need_autocast else torch.bfloat16,
            enabled=need_autocast,
        ):
            x = self.stem_proj(self.patch_embed(x))
            for stage_idx, stage in enumerate(self.stages):
                for block_idx, block in enumerate(stage):
                    assert isinstance(block, SubspaceCliffordBlock)
                    label = f"stage_{stage_idx}_block_{block_idx}"
                    x, layer_diags = block.forward_diagnostics_and_step(x)
                    for key, val in layer_diags.items():
                        all_diags[f"{label}/{key}"] = val
                if stage_idx < len(self.downsamples):
                    x = self.downsamples[stage_idx](x)
        return all_diags

    def forward(
        self,
        x: torch.Tensor,
        compute_ortho: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if compute_ortho:
            x = self.stem_proj(self.patch_embed(x))
            ortho_losses: list[torch.Tensor] = []
            for stage_idx, stage in enumerate(self.stages):
                for block in stage:
                    x, ol = block(x, compute_ortho=True)
                    ortho_losses.append(ol)
                if stage_idx < len(self.downsamples):
                    x = self.downsamples[stage_idx](x)
            x = x.mean(dim=(-2, -1))
            logits = self.head(self.norm(x))
            return logits, torch.stack(ortho_losses).mean()

        x = self.forward_features(x)
        x = x.mean(dim=(-2, -1))
        return self.head(self.norm(x))


class TimmClassifierWrapper(nn.Module):
    """
    CliffordNetLightning-compatible wrapper around `timm` backbones (e.g. ConvNeXt-Atto/Femto).
    For CIFAR-scale inputs (`img_size <= 32`), optionally adapts stride-4 patchifying stems
    to a stride-1 3x3 convolution so deeper stages preserve spatial resolution (32->16->8->4).
    """

    def __init__(
        self,
        timm_name: str,
        num_classes: int = 1000,
        in_chans: int = 3,
        img_size: int = 224,
        drop_path_rate: float = 0.1,
        pretrained: bool = False,
        cifar_stem: bool | None = None,
        wedge_mode: str = "fma",
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.timm_name = timm_name
        self.num_classes = num_classes
        self.img_size = img_size
        self.wedge_mode = wedge_mode
        self.backend = backend

        self.backbone = timm.create_model(
            timm_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
        )

        use_cifar_stem = (img_size <= 32) if cifar_stem is None else cifar_stem
        if use_cifar_stem:
            self._adapt_stem_for_cifar()

    def _adapt_stem_for_cifar(self) -> None:
        stem = getattr(self.backbone, "stem", None)
        if (
            isinstance(stem, nn.Sequential)
            and len(stem) > 0
            and isinstance(stem[0], nn.Conv2d)
            and stem[0].stride[0] > 1
        ):
            old_conv = stem[0]
            new_conv = nn.Conv2d(
                old_conv.in_channels,
                old_conv.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=old_conv.bias is not None,
            )
            trunc_normal_(new_conv.weight, std=0.02)
            if new_conv.bias is not None:
                nn.init.constant_(new_conv.bias, 0)
            stem[0] = new_conv

    def get_interaction_layers(self) -> list[nn.Module]:
        return []

    def iter_blocks(self) -> Iterator[tuple[str, nn.Module]]:
        yield from ()

    def diagnostic_block_labels(self) -> list[str]:
        return []

    def forward_diagnostics(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {}

    def forward(
        self,
        x: torch.Tensor,
        compute_ortho: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        logits = self.backbone(x)
        if compute_ortho:
            return logits, x.new_zeros((), dtype=torch.float32)
        return logits


# ============================================================================
# Presets, Specs, and Model Builders Registry
# ============================================================================


class SubspaceModelSpec(NamedTuple):
    """Structured `(model_cls, model_kwargs)` specification supporting tuple unpacking and `.build()`."""

    model_cls: type[nn.Module]
    model_kwargs: dict[str, Any]

    def build(self) -> nn.Module:
        return self.model_cls(**self.model_kwargs)

    def __getitem__(self, key: int | str) -> Any:
        if isinstance(key, str):
            if key in ("model_cls", "cls"):
                return self.model_cls
            if key in ("model_kwargs", "kwargs"):
                return self.model_kwargs
            raise KeyError(key)
        return tuple.__getitem__(self, key)


_CIFAR_PYRAMID_BASE: dict[str, Any] = {
    "embed_dim": 32,
    "stage_depths": (3, 4, 5),
    "stage_dims": (96, 128, 160),
    "downsample_mode": "conv",
    "drop_path_rate": 0.10,
}

_PRESET_REGISTRY: dict[str, tuple[type[nn.Module], dict[str, Any]]] = {
    # ---- Paper 2 Reproduction: Isotropic (12L, D=128) ----
    "subspace_iso_k2": (
        IsotropicSubspaceCliffordNet,
        {"embed_dim": 128, "depth": 12, "k": 2, "drop_path_rate": 0.10},
    ),
    "subspace_iso_k4": (
        IsotropicSubspaceCliffordNet,
        {"embed_dim": 128, "depth": 12, "k": 4, "drop_path_rate": 0.10},
    ),
    # ---- Paper 2 Reproduction: Hierarchical Uniform & Hybrid ((3,4,5), (96,128,160)) ----
    "subspace_hier_k2": (
        HierarchicalSubspaceCliffordNet,
        {**_CIFAR_PYRAMID_BASE, "subspace_k": 2},
    ),
    "subspace_hier_k3": (
        HierarchicalSubspaceCliffordNet,
        {**_CIFAR_PYRAMID_BASE, "subspace_k": 3},
    ),
    "subspace_hier_k4": (
        HierarchicalSubspaceCliffordNet,
        {**_CIFAR_PYRAMID_BASE, "subspace_k": 4},
    ),
    "subspace_hier_234": (
        HierarchicalSubspaceCliffordNet,
        {**_CIFAR_PYRAMID_BASE, "subspace_k": (2, 3, 4)},
    ),
    "subspace_hier_432": (
        HierarchicalSubspaceCliffordNet,
        {**_CIFAR_PYRAMID_BASE, "subspace_k": (4, 3, 2)},
    ),
    "subspace_hier_244": (
        HierarchicalSubspaceCliffordNet,
        {**_CIFAR_PYRAMID_BASE, "subspace_k": (2, 4, 4)},
    ),
    "subspace_hier_244_dpr025": (
        HierarchicalSubspaceCliffordNet,
        {**_CIFAR_PYRAMID_BASE, "subspace_k": (2, 4, 4), "drop_path_rate": 0.25},
    ),
    # ---- Paper 2 Table 3 Depth Scaling ----
    "subspace_hier_244_18": (
        HierarchicalSubspaceCliffordNet,
        {
            **_CIFAR_PYRAMID_BASE,
            "stage_depths": (4, 6, 8),
            "subspace_k": (2, 4, 4),
            "drop_path_rate": 0.15,
        },
    ),
    "subspace_hier_244_24": (
        HierarchicalSubspaceCliffordNet,
        {
            **_CIFAR_PYRAMID_BASE,
            "stage_depths": (4, 10, 10),
            "subspace_k": (2, 4, 4),
            "drop_path_rate": 0.20,
        },
    ),
    # ---- Paper 2 Table 2 Shift-Based Hierarchical Baselines (GGR block) ----
    "rolling_hier_s3": (
        HierarchicalSubspaceCliffordNet,
        {
            **_CIFAR_PYRAMID_BASE,
            "interaction_type": "rolling",
            "block_style": "ggr",
            "shifts": (1, 2, 4),
        },
    ),
    "rolling_hier_s4": (
        HierarchicalSubspaceCliffordNet,
        {
            **_CIFAR_PYRAMID_BASE,
            "interaction_type": "rolling",
            "block_style": "ggr",
            "shifts": (1, 2, 4, 8),
        },
    ),
    # ---- Paper 2 Table 4 Ablations + ROI #1 Scientific Deconfounding Controls ----
    "subspace_hier_k4_inner": (
        HierarchicalSubspaceCliffordNet,
        {**_CIFAR_PYRAMID_BASE, "subspace_k": 4, "cli_mode": "inner"},
    ),
    "subspace_hier_k4_wedge": (
        HierarchicalSubspaceCliffordNet,
        {**_CIFAR_PYRAMID_BASE, "subspace_k": 4, "cli_mode": "wedge"},
    ),
    "subspace_hier_k4_spfree": (
        HierarchicalSubspaceCliffordNet,
        {**_CIFAR_PYRAMID_BASE, "subspace_k": 4, "use_spatial_diffusion": False},
    ),
    "subspace_hier_k4_abs": (
        HierarchicalSubspaceCliffordNet,
        {**_CIFAR_PYRAMID_BASE, "subspace_k": 4, "ctx_mode": "abs"},
    ),
    "subspace_hier_k4_inner_wide": (
        HierarchicalSubspaceCliffordNet,
        {**_CIFAR_PYRAMID_BASE, "subspace_k": 4, "cli_mode": "inner_wide"},
    ),
    "subspace_hier_k4_sym": (
        HierarchicalSubspaceCliffordNet,
        {**_CIFAR_PYRAMID_BASE, "subspace_k": 4, "cli_mode": "sym_full"},
    ),
    "subspace_hier_k4_bilinear": (
        HierarchicalSubspaceCliffordNet,
        {**_CIFAR_PYRAMID_BASE, "subspace_k": 4, "cli_mode": "bilinear_full"},
    ),
    "subspace_hier_k4_ggr": (
        HierarchicalSubspaceCliffordNet,
        {
            **_CIFAR_PYRAMID_BASE,
            "subspace_k": 4,
            "interaction_type": "subspace",
            "block_style": "ggr",
        },
    ),
    "rolling_hier_s3_lietrotter": (
        HierarchicalSubspaceCliffordNet,
        {
            **_CIFAR_PYRAMID_BASE,
            "interaction_type": "rolling",
            "block_style": "lie_trotter",
            "shifts": (1, 2, 4),
        },
    ),
    # ---- Wave 3 Follow-Up Extensions: Higher-k Iso-Param Pareto, Norm Invariance & Global Context ----
    "subspace_hier_k2_isoparam": (
        HierarchicalSubspaceCliffordNet,
        {
            **_CIFAR_PYRAMID_BASE,
            "stage_dims": (108, 144, 180),
            "subspace_k": 2,
            "drop_path_rate": 0.20,
        },
    ),
    "subspace_hier_k4_dpr020": (
        HierarchicalSubspaceCliffordNet,
        {
            **_CIFAR_PYRAMID_BASE,
            "stage_dims": (96, 128, 160),
            "subspace_k": 4,
            "drop_path_rate": 0.20,
        },
    ),
    "subspace_hier_k6_isoparam": (
        HierarchicalSubspaceCliffordNet,
        {
            **_CIFAR_PYRAMID_BASE,
            "stage_dims": (84, 120, 144),
            "subspace_k": 6,
            "drop_path_rate": 0.20,
        },
    ),
    "subspace_hier_k8_isoparam": (
        HierarchicalSubspaceCliffordNet,
        {
            **_CIFAR_PYRAMID_BASE,
            "stage_dims": (80, 104, 136),
            "subspace_k": 8,
            "drop_path_rate": 0.20,
        },
    ),
    "subspace_hier_k4_ln2d": (
        HierarchicalSubspaceCliffordNet,
        {
            **_CIFAR_PYRAMID_BASE,
            "subspace_k": 4,
            "norm_mode": "ln2d",
            "drop_path_rate": 0.20,
        },
    ),
    "subspace_hier_k4_rms": (
        HierarchicalSubspaceCliffordNet,
        {
            **_CIFAR_PYRAMID_BASE,
            "subspace_k": 4,
            "norm_mode": "subspace_rms",
            "drop_path_rate": 0.20,
        },
    ),
    "subspace_hier_k4_gffn_global": (
        HierarchicalSubspaceCliffordNet,
        {
            **_CIFAR_PYRAMID_BASE,
            "subspace_k": 4,
            "use_global_context": True,
            "drop_path_rate": 0.20,
        },
    ),
    "in1k_subspace_femto_ln2d": (
        HierarchicalSubspaceCliffordNet,
        {
            "patch_size": 4,
            "embed_dim": 64,
            "stage_depths": (2, 2, 6, 2),
            "stage_dims": (64, 128, 256, 384),
            "subspace_k": (2, 4, 4, 4),
            "norm_mode": "ln2d",
            "downsample_mode": "conv",
            "drop_path_rate": 0.20,
        },
    ),
    "in1k_subspace_femto_gffn": (
        HierarchicalSubspaceCliffordNet,
        {
            "patch_size": 4,
            "embed_dim": 64,
            "stage_depths": (2, 2, 6, 2),
            "stage_dims": (64, 128, 256, 384),
            "subspace_k": (2, 4, 4, 4),
            "use_global_context": True,
            "downsample_mode": "conv",
            "drop_path_rate": 0.20,
        },
    ),
    "in1k_subspace_femto_rms": (
        HierarchicalSubspaceCliffordNet,
        {
            "patch_size": 4,
            "embed_dim": 64,
            "stage_depths": (2, 2, 6, 2),
            "stage_dims": (64, 128, 256, 384),
            "subspace_k": (2, 4, 4, 4),
            "norm_mode": "subspace_rms",
            "downsample_mode": "conv",
            "drop_path_rate": 0.20,
        },
    ),
    # ---- ROI #2 ImageNet-1k Multi-Stage Scaling Variants (patch_size=4) ----
    "in1k_subspace_can_tiny": (
        HierarchicalSubspaceCliffordNet,
        {
            "patch_size": 4,
            "embed_dim": 32,
            "stage_depths": (3, 4, 5),
            "stage_dims": (32, 64, 128),
            "subspace_k": (2, 4, 4),
            "downsample_mode": "conv",
            "drop_path_rate": 0.10,
        },
    ),
    "in1k_subspace_p4": (
        HierarchicalSubspaceCliffordNet,
        {
            "patch_size": 4,
            "embed_dim": 48,
            "stage_depths": (2, 2, 4, 2),
            "stage_dims": (48, 96, 160, 256),
            "subspace_k": (2, 3, 4, 4),
            "downsample_mode": "conv",
            "drop_path_rate": 0.20,
        },
    ),
    "in1k_subspace_p4_inner_wide": (
        HierarchicalSubspaceCliffordNet,
        {
            "patch_size": 4,
            "embed_dim": 48,
            "stage_depths": (2, 2, 4, 2),
            "stage_dims": (48, 96, 160, 256),
            "subspace_k": (2, 3, 4, 4),
            "cli_mode": "inner_wide",
            "downsample_mode": "conv",
            "drop_path_rate": 0.20,
        },
    ),
    "in1k_subspace_tiny": (
        HierarchicalSubspaceCliffordNet,
        {
            "patch_size": 4,
            "embed_dim": 64,
            "stage_depths": (3, 4, 5),
            "stage_dims": (64, 128, 256),
            "subspace_k": (2, 4, 4),
            "downsample_mode": "conv",
            "drop_path_rate": 0.15,
        },
    ),
    "in1k_subspace_femto": (
        HierarchicalSubspaceCliffordNet,
        {
            "patch_size": 4,
            "embed_dim": 64,
            "stage_depths": (2, 2, 6, 2),
            "stage_dims": (64, 128, 256, 384),
            "subspace_k": (2, 4, 4, 4),
            "downsample_mode": "conv",
            "drop_path_rate": 0.20,
        },
    ),
    "in1k_subspace_femto_inner_wide": (
        HierarchicalSubspaceCliffordNet,
        {
            "patch_size": 4,
            "embed_dim": 64,
            "stage_depths": (2, 2, 6, 2),
            "stage_dims": (64, 128, 256, 384),
            "subspace_k": (2, 4, 4, 4),
            "cli_mode": "inner_wide",
            "downsample_mode": "conv",
            "drop_path_rate": 0.20,
        },
    ),
    "in1k_subspace_femto_sym": (
        HierarchicalSubspaceCliffordNet,
        {
            "patch_size": 4,
            "embed_dim": 64,
            "stage_depths": (2, 2, 6, 2),
            "stage_dims": (64, 128, 256, 384),
            "subspace_k": (2, 4, 4, 4),
            "cli_mode": "sym_full",
            "downsample_mode": "conv",
            "drop_path_rate": 0.20,
        },
    ),
    "in1k_rolling_femto": (
        HierarchicalSubspaceCliffordNet,
        {
            "patch_size": 4,
            "embed_dim": 48,
            "stage_depths": (2, 2, 6, 2),
            "stage_dims": (48, 96, 192, 272),
            "interaction_type": "rolling",
            "block_style": "ggr",
            "shifts": (1, 2, 4),
            "downsample_mode": "conv",
            "drop_path_rate": 0.20,
        },
    ),
    # ---- Modern ConvNeXt Baselines via timm ----
    "baseline_convnext_atto": (
        TimmClassifierWrapper,
        {"timm_name": "convnext_atto", "drop_path_rate": 0.10},
    ),
    "baseline_convnext_femto": (
        TimmClassifierWrapper,
        {"timm_name": "convnext_femto", "drop_path_rate": 0.15},
    ),
    "baseline_convnext_pico": (
        TimmClassifierWrapper,
        {"timm_name": "convnext_pico", "drop_path_rate": 0.15},
    ),
    "baseline_convnext_nano": (
        TimmClassifierWrapper,
        {"timm_name": "convnext_nano", "drop_path_rate": 0.20},
    ),
    # ---- CIFAR-100 v2 Backport & Dead-Zone-Fixed Controls (Wave 4 / Wave A) ----
    "subspace_hier_k4_v2": (
        HierarchicalSubspaceCliffordNet,
        {
            **_CIFAR_PYRAMID_BASE,
            "subspace_k": 4,
            "ctx_mode": "abs",
            "ctx_bn": True,
            "proj_bias": True,
            "drop_path_rate": 0.20,
        },
    ),
    "subspace_hier_244_v2": (
        HierarchicalSubspaceCliffordNet,
        {
            **_CIFAR_PYRAMID_BASE,
            "subspace_k": (2, 4, 4),
            "ctx_mode": "abs",
            "ctx_bn": True,
            "proj_bias": True,
            "drop_path_rate": 0.20,
        },
    ),
    "subspace_hier_k6_v2": (
        HierarchicalSubspaceCliffordNet,
        {
            **_CIFAR_PYRAMID_BASE,
            "stage_dims": (84, 120, 144),
            "subspace_k": 6,
            "ctx_mode": "abs",
            "ctx_bn": True,
            "proj_bias": True,
            "drop_path_rate": 0.20,
        },
    ),
    "subspace_hier_k4_v2_inner_wide": (
        HierarchicalSubspaceCliffordNet,
        {
            **_CIFAR_PYRAMID_BASE,
            "subspace_k": 4,
            "cli_mode": "inner_wide",
            "ctx_mode": "abs",
            "ctx_bn": True,
            "proj_bias": True,
            "drop_path_rate": 0.20,
        },
    ),
    "subspace_hier_k4_v2_sym": (
        HierarchicalSubspaceCliffordNet,
        {
            **_CIFAR_PYRAMID_BASE,
            "subspace_k": 4,
            "cli_mode": "sym_full",
            "ctx_mode": "abs",
            "ctx_bn": True,
            "proj_bias": True,
            "drop_path_rate": 0.20,
        },
    ),
    "subspace_hier_k4_v2_ln2d": (
        HierarchicalSubspaceCliffordNet,
        {
            **_CIFAR_PYRAMID_BASE,
            "subspace_k": 4,
            "norm_mode": "ln2d",
            "ctx_mode": "abs",
            "ctx_bn": True,
            "proj_bias": True,
            "drop_path_rate": 0.20,
        },
    ),
    "subspace_hier_k4_v2_ctxbn_diff": (
        HierarchicalSubspaceCliffordNet,
        {
            **_CIFAR_PYRAMID_BASE,
            "subspace_k": 4,
            "ctx_mode": "diff",
            "ctx_bn": True,
            "proj_bias": True,
            "drop_path_rate": 0.20,
        },
    ),
    "subspace_hier_k4_v2_gffn_dw7": (
        HierarchicalSubspaceCliffordNet,
        {
            **_CIFAR_PYRAMID_BASE,
            "subspace_k": 4,
            "ctx_mode": "abs",
            "ctx_bn": True,
            "proj_bias": True,
            "dw_kernel_size": 7,
            "use_global_context": True,
            "drop_path_rate": 0.20,
        },
    ),
    # ---- First-Principles ImageNet-1k v2 Cumulative Ablation Ladder ----
    "in1k_v2_rung1_recipe": (
        HierarchicalSubspaceCliffordNet,
        {
            "patch_size": 4,
            "embed_dim": 64,
            "stage_depths": (2, 2, 6, 2),
            "stage_dims": (64, 128, 256, 384),
            "subspace_k": (2, 4, 4, 4),
            "use_global_context": True,
            "downsample_mode": "conv",
            "drop_path_rate": 0.10,
        },
    ),
    "in1k_v2_ablate_abs_only": (
        HierarchicalSubspaceCliffordNet,
        {
            "patch_size": 4,
            "embed_dim": 64,
            "stage_depths": (2, 2, 6, 2),
            "stage_dims": (64, 128, 256, 384),
            "subspace_k": (2, 4, 4, 4),
            "ctx_mode": "abs",
            "ctx_bn": False,
            "proj_bias": False,
            "use_global_context": True,
            "downsample_mode": "conv",
            "drop_path_rate": 0.10,
        },
    ),
    "in1k_v2_ablate_ctxbn_diff": (
        HierarchicalSubspaceCliffordNet,
        {
            "patch_size": 4,
            "embed_dim": 64,
            "stage_depths": (2, 2, 6, 2),
            "stage_dims": (64, 128, 256, 384),
            "subspace_k": (2, 4, 4, 4),
            "ctx_mode": "diff",
            "ctx_bn": True,
            "proj_bias": True,
            "use_global_context": True,
            "downsample_mode": "conv",
            "drop_path_rate": 0.10,
        },
    ),
    "in1k_v2_rung2_deadzone": (
        HierarchicalSubspaceCliffordNet,
        {
            "patch_size": 4,
            "embed_dim": 64,
            "stage_depths": (2, 2, 6, 2),
            "stage_dims": (64, 128, 256, 384),
            "subspace_k": (2, 4, 4, 4),
            "ctx_mode": "abs",
            "ctx_bn": True,
            "proj_bias": True,
            "use_global_context": True,
            "downsample_mode": "conv",
            "drop_path_rate": 0.10,
        },
    ),
    "in1k_v2_atto_k4": (
        HierarchicalSubspaceCliffordNet,
        {
            "patch_size": 4,
            "embed_dim": 48,
            "stage_depths": (2, 2, 6, 2),
            "stage_dims": (48, 96, 192, 440),
            "subspace_k": (2, 4, 4, 4),
            "ctx_mode": "abs",
            "ctx_bn": True,
            "proj_bias": True,
            "dw_kernel_size": 7,
            "use_global_context": True,
            "downsample_mode": "conv",
            "drop_path_rate": 0.10,
        },
    ),
    "in1k_v2_atto_ln2d": (
        HierarchicalSubspaceCliffordNet,
        {
            "patch_size": 4,
            "embed_dim": 48,
            "stage_depths": (2, 2, 6, 2),
            "stage_dims": (48, 96, 192, 440),
            "subspace_k": (2, 4, 4, 4),
            "norm_mode": "ln2d",
            "ctx_mode": "abs",
            "ctx_bn": True,
            "proj_bias": True,
            "dw_kernel_size": 7,
            "use_global_context": True,
            "downsample_mode": "conv",
            "drop_path_rate": 0.10,
        },
    ),
    "in1k_v2_atto_inner_wide": (
        HierarchicalSubspaceCliffordNet,
        {
            "patch_size": 4,
            "embed_dim": 48,
            "stage_depths": (2, 2, 6, 2),
            "stage_dims": (48, 96, 192, 440),
            "subspace_k": (2, 4, 4, 4),
            "cli_mode": "inner_wide",
            "ctx_mode": "abs",
            "ctx_bn": True,
            "proj_bias": True,
            "dw_kernel_size": 7,
            "use_global_context": True,
            "downsample_mode": "conv",
            "drop_path_rate": 0.10,
        },
    ),
    "in1k_v2_atto_swiglu_k6": (
        HierarchicalSubspaceCliffordNet,
        {
            "patch_size": 4,
            "embed_dim": 48,
            "stage_depths": (2, 2, 6, 2),
            "stage_dims": (48, 96, 192, 408),
            "subspace_k": (2, 4, 4, 6),
            "cli_mode": "swiglu_wedge",
            "ctx_mode": "abs",
            "ctx_bn": True,
            "proj_bias": True,
            "dw_kernel_size": 7,
            "use_global_context": True,
            "downsample_mode": "conv",
            "drop_path_rate": 0.10,
        },
    ),
    # ---- ImageNet-1k v2 Scaling Presets: Pico (~8.5M, 1.32G) & Nano (~14.7M, 2.39G) ----
    "in1k_v2_pico_k4": (
        HierarchicalSubspaceCliffordNet,
        {
            "patch_size": 4,
            "embed_dim": 76,
            "stage_depths": (2, 2, 8, 2),
            "stage_dims": (76, 152, 304, 640),
            "subspace_k": (2, 4, 4, 4),
            "ctx_mode": "abs",
            "ctx_bn": True,
            "proj_bias": True,
            "dw_kernel_size": 7,
            "use_global_context": True,
            "downsample_mode": "conv",
            "drop_path_rate": 0.15,
        },
    ),
    "in1k_v2_pico_rmsnorm2d": (
        HierarchicalSubspaceCliffordNet,
        {
            "patch_size": 4,
            "embed_dim": 76,
            "stage_depths": (2, 2, 8, 2),
            "stage_dims": (76, 152, 304, 640),
            "subspace_k": (2, 4, 4, 4),
            "norm_mode": "rmsnorm2d",
            "ctx_mode": "abs",
            "ctx_bn": True,
            "proj_bias": True,
            "dw_kernel_size": 7,
            "use_global_context": True,
            "downsample_mode": "conv",
            "drop_path_rate": 0.15,
        },
    ),
    "in1k_v2_pico_rmsnorm2d_fast": (
        HierarchicalSubspaceCliffordNet,
        {
            "patch_size": 4,
            "embed_dim": 76,
            "stage_depths": (2, 2, 8, 2),
            "stage_dims": (76, 152, 304, 640),
            "subspace_k": (2, 4, 4, 4),
            "norm_mode": "rmsnorm2d",
            "ctx_mode": "abs",
            "ctx_bn": True,
            "proj_bias": True,
            "split_in_proj": True,
            "dw_kernel_size": 7,
            "use_global_context": True,
            "downsample_mode": "conv",
            "drop_path_rate": 0.15,
        },
    ),
    "in1k_v2_pico_rmsnorm2d_sphere_fast": (
        HierarchicalSubspaceCliffordNet,
        {
            "patch_size": 4,
            "embed_dim": 76,
            "stage_depths": (2, 2, 8, 2),
            "stage_dims": (76, 152, 304, 640),
            "subspace_k": (2, 4, 4, 4),
            "norm_mode": "rmsnorm2d_sphere",
            "ctx_mode": "abs",
            "ctx_bn": True,
            "proj_bias": True,
            "split_in_proj": True,
            "dw_kernel_size": 7,
            "use_global_context": True,
            "downsample_mode": "conv",
            "drop_path_rate": 0.15,
        },
    ),
    "in1k_v2_pico_onesided_sphere": (
        HierarchicalSubspaceCliffordNet,
        {
            "patch_size": 4,
            "embed_dim": 76,
            "stage_depths": (2, 2, 8, 2),
            "stage_dims": (76, 152, 304, 640),
            "subspace_k": (2, 4, 4, 4),
            "norm_mode": "onesided_sphere",
            "ctx_mode": "abs",
            "ctx_bn": True,
            "proj_bias": True,
            "dw_kernel_size": 7,
            "use_global_context": True,
            "downsample_mode": "conv",
            "drop_path_rate": 0.15,
        },
    ),
    "in1k_v2_nano_k4": (
        HierarchicalSubspaceCliffordNet,
        {
            "patch_size": 4,
            "embed_dim": 104,
            "stage_depths": (2, 2, 8, 2),
            "stage_dims": (104, 208, 416, 832),
            "subspace_k": (2, 4, 4, 4),
            "ctx_mode": "abs",
            "ctx_bn": True,
            "proj_bias": True,
            "dw_kernel_size": 7,
            "use_global_context": True,
            "downsample_mode": "conv",
            "drop_path_rate": 0.20,
        },
    ),
}


def get_subspace_model_spec(
    name: str,
    num_classes: int = 1000,
    img_size: int = 224,
    wedge_mode: str = "fma",
    backend: str = "auto",
    **overrides: Any,
) -> SubspaceModelSpec:
    """Return `(model_cls, model_kwargs)` spec for any registered subspace/control/baseline preset."""
    if name not in _PRESET_REGISTRY:
        valid = ", ".join(sorted(_PRESET_REGISTRY))
        raise KeyError(f"Unknown subspace model preset {name!r}. Valid presets: {valid}")

    model_cls, base_kwargs = _PRESET_REGISTRY[name]
    kwargs: dict[str, Any] = dict(base_kwargs)
    kwargs["num_classes"] = num_classes
    kwargs["img_size"] = img_size
    kwargs["wedge_mode"] = wedge_mode
    kwargs["backend"] = backend
    if model_cls is not TimmClassifierWrapper:
        kwargs.setdefault("in_chans", 3)
        if "patch_size" not in kwargs:
            if model_cls is IsotropicSubspaceCliffordNet:
                kwargs["patch_size"] = 2 if img_size <= 64 else 4
            else:
                kwargs["patch_size"] = 1 if img_size <= 64 else 4
    kwargs.update(overrides)
    return SubspaceModelSpec(model_cls=model_cls, model_kwargs=kwargs)


def _make_preset_builder(preset_name: str) -> Callable[..., nn.Module]:
    def _builder(
        num_classes: int = 1000,
        img_size: int | None = None,
        wedge_mode: str = "fma",
        backend: str = "auto",
        **overrides: Any,
    ) -> nn.Module:
        resolved_img_size = (
            img_size
            if img_size is not None
            else (32 if num_classes <= 200 and not preset_name.startswith("in1k_") else 224)
        )
        return get_subspace_model_spec(
            preset_name,
            num_classes=num_classes,
            img_size=resolved_img_size,
            wedge_mode=wedge_mode,
            backend=backend,
            **overrides,
        ).build()

    _builder.__name__ = f"build_{preset_name}"
    _builder.__qualname__ = f"build_{preset_name}"
    return _builder


SUBSPACE_MODEL_BUILDERS: dict[str, Callable[..., nn.Module]] = {
    preset_name: _make_preset_builder(preset_name) for preset_name in _PRESET_REGISTRY
}

SUBSPACE_MODEL_CHOICES: tuple[str, ...] = tuple(SUBSPACE_MODEL_BUILDERS)

__all__ = [
    "GeometricStem",
    "HierarchicalSubspaceCliffordNet",
    "IsotropicSubspaceCliffordNet",
    "LayerNorm2d",
    "RollingGeometricProduct",
    "SUBSPACE_MODEL_BUILDERS",
    "SUBSPACE_MODEL_CHOICES",
    "SUBSPACE_OFFDIAG_PAIRS",
    "SpatialDiffusion",
    "StageDownsample",
    "SubspaceCliffordBlock",
    "SubspaceGeometricProduct",
    "SubspaceModelSpec",
    "TimmClassifierWrapper",
    "compute_inner_wide_aligned_dim",
    "get_subspace_model_spec",
    "subspace_gp_pytorch",
]
