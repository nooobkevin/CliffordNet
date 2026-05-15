"""ImageNet-1k CliffordNet model, data module, and legacy CLI entrypoint."""

import copy
import wandb
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from datasets import load_dataset
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.mixup import Mixup
from timm.layers import DropPath, trunc_normal_
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
)
import lightning as L
import matplotlib.pyplot as plt
import math
import os
import time as _time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib

matplotlib.use("Agg")


# ============================================================================
# CliffordNet Model Components
# ============================================================================


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for 2D feature maps, matching the reference implementation."""

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class CliffordInteraction(nn.Module):
    """
    Clifford Geometric Product interaction layer with numerical stability options.
    Aligned with the reference implementation: uses get_state (1x1 conv) and
    get_context_local (DWConv->DWConv->BN->SiLU), with diff context mode
    (C = context - state) as default.

    Args:
        dim: Feature dimension.
        shifts: List of cyclic shift offsets for sparse rolling interaction.
        ctx_mode: Context mode — 'diff' (C = ctx - state, paper default) or 'abs' (C = ctx).
        wedge_mode: Numerical strategy for the wedge (exterior) product.
            - 'naive'   : Original subtraction (fast, but prone to bf16 cancellation).
            - 'fp32'    : Upcast operands to fp32 before the subtraction.
            - 'fma'     : Use fused multiply-add for error-free subtraction (most precise).
    """

    def __init__(self, dim, shifts=[1, 2], ctx_mode="diff", wedge_mode="fma"):
        super().__init__()
        self.dim = dim
        self.shifts = shifts
        self.ctx_mode = ctx_mode
        self.wedge_mode = wedge_mode

        # Matches ref: self.get_context_local = DWConv -> DWConv -> BN -> SiLU
        self.get_context_local = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
        )

        # Matches ref: self.get_state = nn.Conv2d(dim, dim, kernel_size=1)
        self.get_state = nn.Conv2d(dim, dim, kernel_size=1)

        input_proj_dim = 2 * len(shifts) * dim
        self.final_proj = nn.Conv2d(input_proj_dim, dim, kernel_size=1)

        base = torch.arange(dim)
        roll_idx = torch.stack([(base - s) % dim for s in shifts], dim=0)
        self.register_buffer("_roll_idx", roll_idx)

    @staticmethod
    def _wedge_naive(prod, z_det_rolled, z_ctx_b, target_dtype):
        """Original: direct subtraction, cast to target dtype."""
        return (prod - z_det_rolled * z_ctx_b).to(target_dtype)

    @staticmethod
    def _wedge_fp32(prod, z_det_rolled, z_ctx_b, target_dtype):
        """Mitigation 1: perform subtraction in fp32 to avoid bf16 cancellation."""
        return (prod.float() - (z_det_rolled * z_ctx_b).float()).to(target_dtype)

    @staticmethod
    def _wedge_fma(z_det_b, z_ctx_rolled, z_det_rolled, z_ctx_b, target_dtype):
        """
        Mitigation 3: FMA-based stable computation of  a*b - c*d.
        Uses the identity:  a*b - c*d = fma(a, b, -w) - fma(c, d, -w)
        where w = c*d, to recover precision lost in direct subtraction.

        torch.addcmul(input, tensor1, tensor2) computes input + tensor1*tensor2
        and maps to hardware FMA on CUDA, providing the same single-rounding
        guarantee as a true fused multiply-add.
        """
        a = z_det_b.float()
        b = z_ctx_rolled.float()
        c = z_det_rolled.float()
        d = z_ctx_b.float()
        w = c * d
        f = torch.addcmul(-w, a, b)  # a*b - w  (fused on CUDA)
        e = torch.addcmul(-w, c, d)  # c*d - w  (≈ rounding error)
        return (f - e).to(target_dtype)

    def forward(self, x, compute_ortho=False):
        """
        Pure forward pass — NO side effects, fully torch.compile-safe.

        Aligned with reference implementation:
        - z_state = get_state(x)
        - z_context_local = get_context_local(x)
        - diff mode: C = z_context_local - z_state

        Args:
            x: (B, C, H, W) input tensor (pre-normed x_ln from the block).
            compute_ortho: If True, also return scalar ortho loss (mean |cos_sim|
                between state and context streams) WITH gradients for backprop.

        Returns:
            g_feat if compute_ortho is False, else (g_feat, ortho_loss).
        """
        z_state = self.get_state(x)
        z_context_local = self.get_context_local(x)

        # Diff mode (paper Eq. 5): C = C_local - H
        if self.ctx_mode == "diff":
            C = z_context_local - z_state
        elif self.ctx_mode == "abs":
            C = z_context_local
        else:
            C = z_context_local - z_state  # default to diff

        B, Ch, H, W = z_state.shape

        # Shifted interactions using pre-computed roll indices
        C_rolled = C[:, self._roll_idx]  # (B, S, C, H, W)
        z_state_rolled = z_state[:, self._roll_idx]  # (B, S, C, H, W)

        z_state_b = z_state.unsqueeze(1)  # (B, 1, C, H, W)
        C_b = C.unsqueeze(1)  # (B, 1, C, H, W)

        # Inner product: SiLU(z_state * C_shifted)
        prod = z_state_b * C_rolled  # a*b term
        dot = F.silu(prod)

        # --- Wedge product: z_state * C_shifted - C * z_state_shifted ---
        target_dtype = x.dtype
        if self.wedge_mode == "fma":
            wedge = self._wedge_fma(
                z_state_b, C_rolled, z_state_rolled, C_b, target_dtype
            )
        elif self.wedge_mode == "fp32":
            wedge = self._wedge_fp32(prod, z_state_rolled, C_b, target_dtype)
        else:  # "naive"
            wedge = self._wedge_naive(prod, z_state_rolled, C_b, target_dtype)

        pairs = torch.stack([dot, wedge], dim=2)
        S = len(self.shifts)
        g_raw = pairs.reshape(B, S * 2 * Ch, H, W)

        g_feat = self.final_proj(g_raw)

        if compute_ortho:
            # Ortho loss between state and context streams
            state_flat = z_state.flatten(2)  # (B, C, H*W)
            ctx_flat = z_context_local.flatten(2)
            cos_sim = F.cosine_similarity(state_flat, ctx_flat, dim=1)
            ortho_loss = cos_sim.abs().mean()
            return g_feat, ortho_loss

        return g_feat

    def forward_diagnostics(self, x):
        """
        Diagnostic-only forward — runs OUTSIDE torch.compile.
        Returns a dict of scalar diagnostic tensors (all detached, no grad).
        """
        with torch.no_grad():
            z_state = self.get_state(x)
            z_context_local = self.get_context_local(x)

            if self.ctx_mode == "diff":
                C = z_context_local - z_state
            elif self.ctx_mode == "abs":
                C = z_context_local
            else:
                C = z_context_local - z_state

            B, Ch, H, W = z_state.shape
            C_rolled = C[:, self._roll_idx]
            z_state_rolled = z_state[:, self._roll_idx]
            z_state_b = z_state.unsqueeze(1)
            C_b = C.unsqueeze(1)

            prod = z_state_b * C_rolled
            dot = F.silu(prod)

            target_dtype = x.dtype
            if self.wedge_mode == "fma":
                wedge = self._wedge_fma(
                    z_state_b, C_rolled, z_state_rolled, C_b, target_dtype
                )
            elif self.wedge_mode == "fp32":
                wedge = self._wedge_fp32(prod, z_state_rolled, C_b, target_dtype)
            else:
                wedge = self._wedge_naive(prod, z_state_rolled, C_b, target_dtype)

            term_a = prod.float()
            term_b = (z_state_rolled * C_b).float()
            abs_sum = term_a.abs() + term_b.abs() + 1e-12
            rel_diff = (term_a - term_b).abs() / abs_sum
            wedge_abs = wedge.float().abs()
            dot_abs = dot.float().abs()

            state_flat = z_state.flatten(2)
            ctx_flat = z_context_local.flatten(2)
            cos_sim = F.cosine_similarity(state_flat, ctx_flat, dim=1)

            return {
                "cancel/rel_diff_mean": rel_diff.mean(),
                "cancel/rel_diff_lt1e-2": (rel_diff < 1e-2).float().mean(),
                "cancel/rel_diff_lt1e-4": (rel_diff < 1e-4).float().mean(),
                "magnitude/wedge_abs_mean": wedge_abs.mean(),
                "magnitude/wedge_abs_max": wedge_abs.max(),
                "magnitude/dot_abs_mean": dot_abs.mean(),
                "health/nan_count": wedge.isnan().sum().float(),
                "health/inf_count": wedge.isinf().sum().float(),
                "ortho/cos_sim_mean": cos_sim.abs().mean(),
            }


class CliffordBlock(nn.Module):
    def __init__(
        self,
        dim,
        shifts,
        drop_path=0.0,
        layer_scale_init_value=1e-5,
        ctx_mode="diff",
        wedge_mode="fma",
    ):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.interaction = CliffordInteraction(
            dim,
            shifts=shifts,
            ctx_mode=ctx_mode,
            wedge_mode=wedge_mode,
        )
        self.gate_linear = nn.Conv2d(dim * 2, dim, kernel_size=1)

        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((1, dim, 1, 1)), requires_grad=True
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, compute_ortho=False):
        shortcut = x
        x_ln = self.norm(x)
        if compute_ortho:
            g_feat, ortho_loss = self.interaction(x_ln, compute_ortho=True)
        else:
            g_feat = self.interaction(x_ln)
            ortho_loss = None

        m = torch.cat([x_ln, g_feat], dim=1)
        alpha = torch.sigmoid(self.gate_linear(m))

        h_mix = F.silu(x_ln) + alpha * g_feat
        x = shortcut + self.drop_path(self.gamma * h_mix)

        if compute_ortho:
            return x, ortho_loss
        return x


def _diagnose_and_step_block(block, x):
    x_ln = block.norm(x)
    layer_diags = block.interaction.forward_diagnostics(x_ln)
    g_feat = block.interaction(x_ln)
    m = torch.cat([x_ln, g_feat], dim=1)
    alpha = torch.sigmoid(block.gate_linear(m))
    h_mix = F.silu(x_ln) + alpha * g_feat
    return x + block.drop_path(block.gamma * h_mix), layer_diags


class CliffordNet(nn.Module):
    def __init__(
        self,
        img_size=224,
        in_chans=3,
        num_classes=1000,
        embed_dim=192,
        depth=12,
        shifts=[1, 2],
        drop_path_rate=0.1,
        ctx_mode="diff",
        wedge_mode="fma",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Stem: match ref impl (patch_size=4 variant) with BN instead of GN
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False
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
            nn.BatchNorm2d(embed_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                CliffordBlock(
                    dim=embed_dim,
                    shifts=shifts,
                    drop_path=dpr[i],
                    ctx_mode=ctx_mode,
                    wedge_mode=wedge_mode,
                )
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, compute_ortho=False):
        """
        Args:
            x: (B, 3, H, W) input images.
            compute_ortho: If True, also return aggregated ortho loss
                (mean |cos_sim|) WITH gradients.  torch.compile traces
                both branches as static — no graph breaks.

        Returns:
            logits if compute_ortho is False, else (logits, ortho_loss).
        """
        x = self.stem(x)
        ortho_losses = []
        for block in self.blocks:
            if compute_ortho:
                x, ol = block(x, compute_ortho=True)
                ortho_losses.append(ol)
            else:
                x = block(x)
        # Ref impl order: global avg pool -> LayerNorm -> head
        x = x.mean(dim=[-2, -1])
        x = self.norm(x)
        logits = self.head(x)

        if compute_ortho:
            ortho_loss = torch.stack(ortho_losses).mean()
            return logits, ortho_loss
        return logits

    def get_interaction_layers(self):
        """Return all CliffordInteraction modules (for diagnostics)."""
        return [block.interaction for block in self.blocks]

    def iter_blocks(self):
        for idx, block in enumerate(self.blocks):
            yield f"block_{idx}", block

    def diagnostic_block_labels(self):
        return [label for label, _block in self.iter_blocks()]

    def forward_diagnostics(self, x):
        all_diags = {}
        with torch.no_grad():
            x = self.stem(x)
            for label, block in self.iter_blocks():
                x, layer_diags = _diagnose_and_step_block(block, x)
                for key, value in layer_diags.items():
                    all_diags[f"{label}/{key}"] = value
        return all_diags


class HierarchicalStem(nn.Module):
    def __init__(self, in_chans, embed_dim, patch_size):
        super().__init__()
        if patch_size == 4:
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
                nn.BatchNorm2d(embed_dim),
            )
        elif patch_size == 2:
            hidden_dim = max(embed_dim // 2, 32)
            self.proj = nn.Sequential(
                nn.Conv2d(
                    in_chans,
                    hidden_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(
                    hidden_dim,
                    embed_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(embed_dim),
            )
        else:
            raise ValueError("Hierarchical ImageNet stem only supports patch_size 2 or 4.")

    def forward(self, x):
        return self.proj(x)


class ConvStageDownsample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.pre_norm = nn.BatchNorm2d(in_dim)
        self.depthwise = nn.Conv2d(
            in_dim,
            in_dim,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=in_dim,
            bias=False,
        )
        self.act = nn.SiLU()
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)
        self.out_norm = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = self.pre_norm(x)
        x = self.depthwise(x)
        x = self.act(x)
        x = self.proj(x)
        x = self.out_norm(x)
        return x


class HierarchicalCliffordNet(nn.Module):
    def __init__(
        self,
        num_classes=1000,
        in_chans=3,
        patch_size=4,
        stage_dims=(48, 96, 160, 256),
        stage_depths=(2, 2, 4, 2),
        stage_shifts=((1,), (1,), (1, 2), (1, 2)),
        drop_path_rate=0.20,
        ctx_mode="diff",
        wedge_mode="fma",
    ):
        super().__init__()
        if len(stage_dims) != len(stage_depths):
            raise ValueError("stage_dims and stage_depths must have the same length.")
        if len(stage_shifts) != len(stage_depths):
            raise ValueError("stage_shifts and stage_depths must have the same length.")

        self.num_classes = num_classes
        self.patch_size = patch_size
        self.stage_dims = tuple(stage_dims)
        self.stage_depths = tuple(stage_depths)
        self.stage_shifts = tuple(tuple(s) for s in stage_shifts)
        self.stem = HierarchicalStem(
            in_chans=in_chans,
            embed_dim=stage_dims[0],
            patch_size=patch_size,
        )

        total_blocks = sum(stage_depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        block_idx = 0
        for stage_idx, (dim, depth, shifts) in enumerate(
            zip(stage_dims, stage_depths, stage_shifts)
        ):
            blocks = nn.ModuleList(
                [
                    CliffordBlock(
                        dim=dim,
                        shifts=list(shifts),
                        drop_path=dpr[block_idx + local_idx],
                        ctx_mode=ctx_mode,
                        wedge_mode=wedge_mode,
                    )
                    for local_idx in range(depth)
                ]
            )
            block_idx += depth
            self.stages.append(blocks)

            if stage_idx < len(stage_depths) - 1:
                self.downsamples.append(
                    ConvStageDownsample(
                        in_dim=stage_dims[stage_idx],
                        out_dim=stage_dims[stage_idx + 1],
                    )
                )

        self.norm = nn.LayerNorm(stage_dims[-1])
        self.head = nn.Linear(stage_dims[-1], num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def iter_blocks(self):
        for stage_idx, stage in enumerate(self.stages):
            for block_idx, block in enumerate(stage):
                yield f"stage_{stage_idx}_block_{block_idx}", block

    def diagnostic_block_labels(self):
        return [label for label, _block in self.iter_blocks()]

    def forward_diagnostics(self, x):
        all_diags = {}
        with torch.no_grad():
            x = self.stem(x)
            for stage_idx, stage in enumerate(self.stages):
                for block_idx, block in enumerate(stage):
                    label = f"stage_{stage_idx}_block_{block_idx}"
                    x, layer_diags = _diagnose_and_step_block(block, x)
                    for key, value in layer_diags.items():
                        all_diags[f"{label}/{key}"] = value
                if stage_idx < len(self.downsamples):
                    x = self.downsamples[stage_idx](x)
        return all_diags

    def forward(self, x, compute_ortho=False):
        x = self.stem(x)
        ortho_losses = []
        for stage_idx, stage in enumerate(self.stages):
            for block in stage:
                if compute_ortho:
                    x, ortho_loss = block(x, compute_ortho=True)
                    ortho_losses.append(ortho_loss)
                else:
                    x = block(x)
            if stage_idx < len(self.downsamples):
                x = self.downsamples[stage_idx](x)

        x = x.mean(dim=[-2, -1])
        x = self.norm(x)
        logits = self.head(x)
        if compute_ortho:
            return logits, torch.stack(ortho_losses).mean()
        return logits


# ============================================================================
# Model Builders
#
# Naming: cliffordnet_{depth}_{num_shifts}
# Following the original author's scaling philosophy:
#   - embed_dim fixed at 128, scale capacity by depth
#   - shifts = [1, 2, 4, ..., 2^(n-1)]  (powers of 2)
#   - drop_path_rate ≈ 0.3 uniformly (0.4 for deepest)
#   - patch_size=4 for ImageNet (224 → 56×56 feature map)
#
# "probe_*" variants are deliberately small for cheap hyperparam sweeps.
# ============================================================================


def _gen_shifts(n):
    """Generate n power-of-2 shifts: [1, 2, 4, ..., 2^(n-1)]."""
    return [1 << i for i in range(n)]


# ---- Probe models (fast hyperparam search) --------------------------------


def cliffordnet_probe_xs(num_classes=1000, wedge_mode="fma"):
    """~0.3M params · depth=4, dim=64, 2 shifts — minutes per epoch on 1 GPU."""
    return CliffordNet(
        img_size=224,
        embed_dim=64,
        depth=4,
        shifts=_gen_shifts(2),
        num_classes=num_classes,
        drop_path_rate=0.1,
        wedge_mode=wedge_mode,
    )


def cliffordnet_probe_s(num_classes=1000, wedge_mode="fma"):
    """~1M params · depth=6, dim=96, 3 shifts — quick sanity / LR sweep."""
    return CliffordNet(
        img_size=224,
        embed_dim=96,
        depth=6,
        shifts=_gen_shifts(3),
        num_classes=num_classes,
        drop_path_rate=0.15,
        wedge_mode=wedge_mode,
    )


# ---- Production models (aligned with author's configs) --------------------


def cliffordnet_12_2(num_classes=1000, wedge_mode="fma"):
    """Author's nano equivalent: depth=12, shifts=[1,2], dim=128."""
    return CliffordNet(
        img_size=224,
        embed_dim=128,
        depth=12,
        shifts=_gen_shifts(2),
        num_classes=num_classes,
        drop_path_rate=0.3,
        wedge_mode=wedge_mode,
    )


def cliffordnet_12_5(num_classes=1000, wedge_mode="fma"):
    """Author's lite equivalent: depth=12, shifts=[1,2,4,8,16], dim=128."""
    return CliffordNet(
        img_size=224,
        embed_dim=128,
        depth=12,
        shifts=_gen_shifts(5),
        num_classes=num_classes,
        drop_path_rate=0.3,
        wedge_mode=wedge_mode,
    )


def cliffordnet_18_5(num_classes=1000, wedge_mode="fma"):
    """depth=18, shifts=[1,2,4,8,16], dim=128."""
    return CliffordNet(
        img_size=224,
        embed_dim=128,
        depth=18,
        shifts=_gen_shifts(5),
        num_classes=num_classes,
        drop_path_rate=0.3,
        wedge_mode=wedge_mode,
    )


def cliffordnet_32_3(num_classes=1000, wedge_mode="fma"):
    """Author's small equivalent: depth=32, shifts=[1,2,4], dim=128."""
    return CliffordNet(
        img_size=224,
        embed_dim=128,
        depth=32,
        shifts=_gen_shifts(3),
        num_classes=num_classes,
        drop_path_rate=0.3,
        wedge_mode=wedge_mode,
    )


def cliffordnet_32_5(num_classes=1000, wedge_mode="fma"):
    """Author's small-wide equivalent: depth=32, shifts=[1,2,4,8,16], dim=128."""
    return CliffordNet(
        img_size=224,
        embed_dim=128,
        depth=32,
        shifts=_gen_shifts(5),
        num_classes=num_classes,
        drop_path_rate=0.3,
        wedge_mode=wedge_mode,
    )


def cliffordnet_64_5(num_classes=1000, wedge_mode="fma"):
    """Author's deep equivalent: depth=64, shifts=[1,2,4,8,16], dim=128."""
    return CliffordNet(
        img_size=224,
        embed_dim=128,
        depth=64,
        shifts=_gen_shifts(5),
        num_classes=num_classes,
        drop_path_rate=0.4,
        wedge_mode=wedge_mode,
    )


def hier_cliffordnet_p4(num_classes=1000, wedge_mode="fma"):
    """Hierarchical 224 -> 56 -> 28 -> 14 -> 7 ImageNet backbone."""
    return HierarchicalCliffordNet(
        num_classes=num_classes,
        patch_size=4,
        stage_dims=(48, 96, 160, 256),
        stage_depths=(2, 2, 4, 2),
        stage_shifts=((1,), (1,), (1, 2), (1, 2)),
        drop_path_rate=0.20,
        wedge_mode=wedge_mode,
    )


def hier_cliffordnet_p2(num_classes=1000, wedge_mode="fma"):
    """Hierarchical 224 -> 112 -> 56 -> 28 -> 14 -> 7 ImageNet backbone."""
    return HierarchicalCliffordNet(
        num_classes=num_classes,
        patch_size=2,
        stage_dims=(32, 64, 96, 160, 256),
        stage_depths=(1, 2, 2, 4, 2),
        stage_shifts=((1,), (1,), (1,), (1, 2), (1, 2)),
        drop_path_rate=0.15,
        wedge_mode=wedge_mode,
    )


MODEL_BUILDERS = {
    # Probe models (hyperparam search)
    "probe_xs": cliffordnet_probe_xs,
    "probe_s": cliffordnet_probe_s,
    # Production single-stage models (author-aligned: depth scaling, dim=128)
    "12_2": cliffordnet_12_2,
    "12_5": cliffordnet_12_5,
    "18_5": cliffordnet_18_5,
    "32_3": cliffordnet_32_3,
    "32_5": cliffordnet_32_5,
    "64_5": cliffordnet_64_5,
    # ImageNet-oriented hierarchical variants suggested in CAN issue #5
    "hier_p4": hier_cliffordnet_p4,
    "hier_p2": hier_cliffordnet_p2,
}


MODEL_SIZE_CHOICES = tuple(MODEL_BUILDERS)


# ============================================================================
# Lightning Module
# ============================================================================


class CliffordNetLightning(L.LightningModule):
    def __init__(
        self,
        model_size="small",
        num_classes=1000,
        learning_rate=3e-4,
        weight_decay=0.05,
        max_epochs=200,
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        mixup_prob=1.0,
        mixup_switch_prob=0.5,
        label_smoothing=0.1,
        ema_decay=0.9999,
        wedge_mode="fma",
        ortho_weight=0.01,
        enable_diagnostics=True,
        diag_log_interval=100,
        warmup_epochs=1,
        eta_min=1e-6,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Build an uncompiled training model for checkpointing/diagnostics, then
        # compile only the training path. EMA validation uses a separate module.
        self._raw_model = MODEL_BUILDERS[model_size](
            num_classes=num_classes,
            wedge_mode=wedge_mode,
        )
        self._raw_model = self._raw_model.to(memory_format=torch.channels_last)

        # Compiled model for training / inference.
        # torch.compile wraps _raw_model; they share the same parameters.
        self.model = torch.compile(self._raw_model)

        # Mixup / CutMix (applied in training_step, not in DataLoader)
        self.mixup_fn = Mixup(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            prob=mixup_prob,
            switch_prob=mixup_switch_prob,
            num_classes=num_classes,
        )
        # With mixup, targets become soft labels → use soft cross-entropy
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        # Validation uses un-smoothed loss for accurate confidence measurement
        self.val_criterion = nn.CrossEntropyLoss()

        # EMA validation model is intentionally separate and uncompiled. Do not
        # swap EMA weights into the training model; state_dict tensors may share
        # storage with compiled model params and cause validation-aligned spikes.
        self.ema_decay = ema_decay
        self._ema_model = copy.deepcopy(self._raw_model)
        self._ema_model.requires_grad_(False)
        self._ema_model.eval()

        self.register_buffer(
            "inv_mean", torch.tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "inv_std", torch.tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1)
        )

        self.val_preds = []
        self.val_labels = []
        self._at_epoch_boundary = False  # avoid confusion matrix on sanity check
        self._train_step_start = None

        # W2: cached parameter groups for gradient norm computation.
        # Populated lazily on first call to _log_grad_norms().
        self._grad_param_groups = None

    def forward(self, x):
        return self.model(x.contiguous(memory_format=torch.channels_last))

    def on_train_batch_start(self, batch, batch_idx):
        self._train_step_start = _time.monotonic()

    def training_step(self, batch, batch_idx):
        images, labels = batch
        # Apply Mixup / CutMix (produces soft labels)
        images, labels = self.mixup_fn(images, labels)

        # C1/C2 fix: when ortho_weight > 0, the compiled model computes
        # ortho loss INSIDE the graph (with gradients) — no side effects.
        ortho_w = self.hparams.ortho_weight
        x_cl = images.contiguous(memory_format=torch.channels_last)
        if ortho_w > 0:
            outputs, ortho_loss = self.model(x_cl, compute_ortho=True)
            loss = self.criterion(outputs, labels) + ortho_w * ortho_loss
            self.log("train/ortho_loss", ortho_loss, prog_bar=False, sync_dist=True)
        else:
            outputs = self.model(x_cl)
            loss = self.criterion(outputs, labels)

        # For logging accuracy, use hard labels (argmax of soft targets)
        hard_labels = labels.argmax(dim=1)
        acc1, acc5 = self._accuracy(outputs, hard_labels, topk=(1, 5))

        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log("train/acc1", acc1, prog_bar=True, sync_dist=True)
        self.log("train/acc5", acc5, sync_dist=True)

        # Log learning rate
        opt = self.optimizers()
        self.log("train/lr", opt.param_groups[0]["lr"], prog_bar=False)

        # Throughput logging
        if self._train_step_start is not None:
            elapsed = _time.monotonic() - self._train_step_start
            if elapsed > 0:
                bs = images.shape[0]
                world_size = self.trainer.world_size
                imgs_per_sec = bs * world_size / elapsed
                self.log("perf/images_per_sec", imgs_per_sec, prog_bar=False)
                self.log("perf/step_time_ms", elapsed * 1000, prog_bar=False)

        # Diagnostic A: wedge statistics (sampled at interval to avoid overhead)
        # C3 fix: run diagnostic forward on the UN-COMPILED model (no graph breaks).
        # This is a no-grad re-forward of the last batch at sampled intervals only.
        if (
            self.hparams.enable_diagnostics
            and batch_idx % self.hparams.diag_log_interval == 0
        ):
            self._log_diagnostics(x_cl)

        return loss

    def _log_diagnostics(self, images):
        """
        Diagnostic A: collect wedge statistics by running a no-grad forward
        through the UN-COMPILED model's interaction layers.
        Called only every diag_log_interval steps.

        Uses a small sub-batch (max 4 samples) to avoid OOM — the full batch
        may already consume nearly all GPU memory during training.
        """
        # Sub-sample to avoid OOM: diagnostics are statistical summaries,
        # so a small sample is sufficient.
        diag_bs = min(4, images.shape[0])
        images = images[:diag_bs]

        all_diags = self._raw_model.forward_diagnostics(images)

        # Log summary across all blocks (mean of per-block values)
        summary_keys = [
            "cancel/rel_diff_mean",
            "cancel/rel_diff_lt1e-2",
            "cancel/rel_diff_lt1e-4",
            "magnitude/wedge_abs_mean",
            "magnitude/wedge_abs_max",
            "magnitude/dot_abs_mean",
            "health/nan_count",
            "health/inf_count",
            "ortho/cos_sim_mean",
        ]
        for sk in summary_keys:
            vals = [v for k, v in all_diags.items() if k.endswith(sk)]
            if vals:
                self.log(
                    f"diag/{sk}",
                    torch.stack(vals).mean(),
                    prog_bar=False,
                    sync_dist=False,
                )

        # Log per-block detail for first, middle, last block
        block_labels = self._raw_model.diagnostic_block_labels()
        sample_indices = sorted(set([0, len(block_labels) // 2, len(block_labels) - 1]))
        for sample_idx in sample_indices:
            label = block_labels[sample_idx]
            for sk in [
                "cancel/rel_diff_mean",
                "magnitude/wedge_abs_mean",
                "ortho/cos_sim_mean",
            ]:
                key = f"{label}/{sk}"
                if key in all_diags:
                    self.log(
                        f"diag/{label}/{sk}",
                        all_diags[key],
                        prog_bar=False,
                        sync_dist=False,
                    )

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update EMA weights after optimizer step + log gradient norms (Diagnostic C)."""
        self._update_ema()

        # Diagnostic C: gradient norm monitoring for wedge vs dot related params
        # Only log at the diagnostic interval to avoid overhead
        if (
            self.hparams.enable_diagnostics
            and batch_idx % self.hparams.diag_log_interval == 0
        ):
            self._log_grad_norms()

    def _init_ema(self):
        """Ensure the separate uncompiled EMA model is on the active device."""
        self._ema_model.eval()
        self._ema_model.to(device=self.device, memory_format=torch.channels_last)

    def _update_ema(self):
        self._init_ema()
        d = self.ema_decay
        with torch.no_grad():
            for ema_param, model_param in zip(
                self._ema_model.parameters(), self._raw_model.parameters()
            ):
                ema_param.mul_(d).add_(model_param.detach(), alpha=1 - d)
            for ema_buffer, model_buffer in zip(
                self._ema_model.buffers(), self._raw_model.buffers()
            ):
                ema_buffer.copy_(model_buffer.detach())

    def _validation_model(self):
        self._init_ema()
        self._ema_model.eval()
        return self._ema_model

    def _build_grad_param_groups(self):
        """W2 fix: cache parameter groups once to avoid re-iterating named_parameters."""
        state_params, ctx_params, proj_params = [], [], []
        for name, param in self._raw_model.named_parameters():
            if ".get_state." in name:
                state_params.append(param)
            elif ".get_context_local." in name:
                ctx_params.append(param)
            elif ".final_proj." in name:
                proj_params.append(param)
        self._grad_param_groups = {
            "state_stream": state_params,
            "ctx_stream": ctx_params,
            "final_proj": proj_params,
        }

    def _log_grad_norms(self):
        """
        Diagnostic C: Log gradient norms for wedge-related vs dot-related parameters.

        W2 fix: uses cached param groups + single torch.norm per group
        to avoid per-parameter .item() calls and CUDA sync points.
        """
        if self._grad_param_groups is None:
            self._build_grad_param_groups()

        norms = {}
        for group_name, params in self._grad_param_groups.items():
            grad_tensors = [
                p.grad.detach().float() for p in params if p.grad is not None
            ]
            if grad_tensors:
                # Single concatenation + norm — one CUDA kernel, no .item() sync
                flat = torch.cat([g.flatten() for g in grad_tensors])
                norms[group_name] = flat.norm()

        for group_name, norm_val in norms.items():
            self.log(
                f"diag/grad_norm/{group_name}",
                norm_val,
                prog_bar=False,
                sync_dist=False,
            )

        # Log ratio: if state/ctx gradient norms diverge, the wedge branch
        # might be experiencing vanishing/exploding gradients relative to dot
        if "state_stream" in norms and "ctx_stream" in norms:
            ratio = norms["state_stream"] / (norms["ctx_stream"] + 1e-12)
            self.log(
                "diag/grad_norm/state_ctx_ratio",
                ratio,
                prog_bar=False,
                sync_dist=False,
            )

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        x_cl = images.contiguous(memory_format=torch.channels_last)
        outputs = self._validation_model()(x_cl)
        loss = self.val_criterion(outputs, labels)

        acc1, acc5 = self._accuracy(outputs, labels, topk=(1, 5))

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/acc1", acc1, prog_bar=True, sync_dist=True)
        self.log("val/acc5", acc5, prog_bar=True, sync_dist=True)

        preds = outputs.argmax(dim=1)
        self.val_preds.append(preds.cpu())
        self.val_labels.append(labels.cpu())

        if batch_idx == 0 and self.trainer.is_global_zero:
            self._log_images(images, labels, outputs)

        return loss

    def on_validation_epoch_end(self):
        # ------------------------------------------------------------------
        # Confusion matrix is expensive for 1000 classes (8MB all_reduce).
        # Only compute at actual epoch boundaries, not every val_check_interval.
        # We check a flag set in on_train_epoch_end to stay safe with DDP
        # (all ranks must agree on whether to all_reduce or not).
        # ------------------------------------------------------------------
        is_epoch_end = self._at_epoch_boundary
        num_cls = self.hparams.num_classes

        if is_epoch_end and len(self.val_preds) > 0:
            all_preds = torch.cat(self.val_preds).numpy()
            all_labels = torch.cat(self.val_labels).numpy()
            cm_local = confusion_matrix(all_labels, all_preds, labels=range(num_cls))

            cm_tensor = torch.tensor(cm_local, dtype=torch.int64, device=self.device)

            if dist.is_initialized() and dist.get_world_size() > 1:
                dist.all_reduce(cm_tensor, op=dist.ReduceOp.SUM)

            if self.trainer.is_global_zero:
                cm = cm_tensor.cpu().numpy()

                if num_cls > 100:
                    self._log_top_confused_pairs(cm, top_k=20)
                else:
                    cm_normalized = cm.astype("float") / (
                        cm.sum(axis=1, keepdims=True) + 1e-8
                    )
                    fig, ax = plt.subplots(figsize=(12, 10))
                    sns.heatmap(cm_normalized, ax=ax, cmap="Blues", cbar=True)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("True")
                    ax.set_title(f"Confusion Matrix — Epoch {self.current_epoch}")
                    fig.tight_layout()

                    self.logger.experiment.log(
                        {"val/confusion_matrix": wandb.Image(fig)},
                    )
                    plt.close(fig)

        self.val_preds.clear()
        self.val_labels.clear()
        self._at_epoch_boundary = False

    def on_train_epoch_end(self):
        # Mark that the next validation is at an epoch boundary
        self._at_epoch_boundary = True

    def _log_top_confused_pairs(self, cm, top_k=20):
        """Log top confused class pairs for large datasets like ImageNet."""
        cm_no_diag = cm.copy()
        np.fill_diagonal(cm_no_diag, 0)

        flat_indices = np.argsort(cm_no_diag.ravel())[-top_k:][::-1]
        top_pairs = [(idx // cm.shape[1], idx % cm.shape[1]) for idx in flat_indices]

        fig, ax = plt.subplots(figsize=(12, 8))
        pair_labels = [f"{true}->{pred}" for true, pred in top_pairs]
        pair_counts = [cm[true, pred] for true, pred in top_pairs]

        bars = ax.barh(range(len(pair_labels)), pair_counts, color="steelblue")
        ax.set_yticks(range(len(pair_labels)))
        ax.set_yticklabels(pair_labels)
        ax.set_xlabel("Count")
        ax.set_title(f"Top {top_k} Confused Class Pairs — Epoch {self.current_epoch}")
        ax.invert_yaxis()

        for bar, count in zip(bars, pair_counts):
            ax.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                str(count),
                va="center",
                fontsize=9,
            )

        fig.tight_layout()

        self.logger.experiment.log(
            {"val/top_confused_pairs": wandb.Image(fig)},
        )
        plt.close(fig)

    def _log_images(self, images, labels, outputs):
        n = min(images.shape[0], 8)
        imgs = images[:n]
        lbls = labels[:n]
        preds = outputs[:n].argmax(dim=1)

        imgs = imgs * self.inv_std + self.inv_mean
        imgs = torch.clamp(imgs, 0, 1)

        ncols = 4
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3.5 * nrows))
        if nrows == 1:
            axes = [axes] if ncols == 1 else list(axes)
        else:
            axes = axes.flatten().tolist()

        for i in range(len(axes)):
            ax = axes[i]
            ax.axis("off")
            if i < n:
                img_np = imgs[i].cpu().permute(1, 2, 0).float().numpy()
                ax.imshow(img_np)
                gt, pd = lbls[i].item(), preds[i].item()
                color = "green" if gt == pd else "red"
                ax.set_title(
                    f"GT:{gt} / P:{pd}",
                    fontsize=11,
                    color=color,
                    fontweight="bold",
                )

        fig.suptitle(f"Epoch {self.current_epoch}", fontsize=14)
        fig.tight_layout()

        self.logger.experiment.log(
            {"val/sample_predictions": wandb.Image(fig)},
        )
        plt.close(fig)

    def _accuracy(self, output, target, topk=(1,)):
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0)
                res.append(correct_k / batch_size)
            return res

    def configure_optimizers(self):
        # Separate params: no weight decay for norm layers, biases, and layer scale
        decay_params = []
        no_decay_params = []
        for name, param in self._raw_model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or "bias" in name or "norm" in name or "gamma" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.hparams.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.hparams.learning_rate,
        )

        # Linear warmup, then cosine decay to eta_min over
        # the remaining epochs.  Both schedulers step per-step (not per-epoch)
        # so the curve is smooth.
        # We estimate steps_per_epoch from the trainer if available.
        total_steps = max(1, self.trainer.estimated_stepping_batches)
        steps_per_epoch = max(
            1, total_steps // max(1, self.hparams.max_epochs)
        )
        warmup_steps = max(1, self.hparams.warmup_epochs * steps_per_epoch)
        if total_steps > 1:
            warmup_steps = min(warmup_steps, total_steps - 1)

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-3,  # LR starts at learning_rate * 1e-3
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_steps - warmup_steps),
            eta_min=self.hparams.eta_min,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


# ============================================================================
# Data Module — NFS + optional local SSD prefetch
# ============================================================================


class HFImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, item["label"]


class ImageNet1kDataModule(L.LightningDataModule):
    def __init__(
        self,
        nfs_data_dir,
        batch_size,
        num_workers,
    ):
        super().__init__()
        self.nfs_data_dir = nfs_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prepare_data_per_node = True

    def prepare_data(self):
        """Ensure the Hugging Face dataset cache exists on shared storage."""
        load_dataset("ILSVRC/imagenet-1k", cache_dir=self.nfs_data_dir)

    def setup(self, stage=None):
        """Load datasets on every rank after Lightning's prepare_data barrier."""

        train_tf = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ColorJitter(0.4, 0.4, 0.4),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                transforms.RandomErasing(p=0.25),
            ]
        )
        val_tf = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )

        ds = load_dataset("ILSVRC/imagenet-1k", cache_dir=self.nfs_data_dir)
        self.train_ds = HFImageNetDataset(ds["train"], transform=train_tf)
        self.val_ds = HFImageNetDataset(ds["validation"], transform=val_tf)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,  # Lightning 在 DDP 下自動替換為 DistributedSampler
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )


# ============================================================================
# Main
# ============================================================================


def auto_find_batch_size(
    model_cls,
    model_kwargs,
    max_batch_size=1024,
    min_batch_size=8,
    input_shape=(3, 224, 224),
    dtype=torch.bfloat16,
    device=None,
):
    """
    Binary search for the largest batch size that fits in GPU memory.
    Simulates the real training loop: forward + backward + optimizer.step()
    with AdamW and bf16 mixed precision, matching what Lightning actually does.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        print("[AutoBS] No CUDA device, returning min_batch_size")
        return min_batch_size

    # Build model exactly as training does: channels_last + compile
    model = model_cls(**model_kwargs).to(
        device=device, memory_format=torch.channels_last
    )
    model = torch.compile(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)

    # Warm-up: run one small forward+backward+step to trigger torch.compile
    # and allocate optimizer states, so they don't inflate later measurements.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        x_warmup = torch.randn(
            min_batch_size, *input_shape, device=device, dtype=dtype
        ).contiguous(memory_format=torch.channels_last)
        y_warmup = torch.randint(
            0,
            model_kwargs.get("num_classes", 1000),
            (min_batch_size,),
            device=device,
        )
        with torch.autocast(device_type="cuda", dtype=dtype):
            out = model(x_warmup)
            loss = criterion(out, y_warmup)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        del x_warmup, y_warmup, out, loss
        torch.cuda.empty_cache()
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        print(f"[AutoBS] Even min_batch_size={min_batch_size} OOM during warmup: {e}")
        del model, criterion, optimizer
        torch.cuda.empty_cache()
        return min_batch_size

    # Binary search: probe with full forward + backward + optimizer.step
    best = min_batch_size
    lo, hi = min_batch_size, max_batch_size

    while lo <= hi:
        mid = (lo + hi) // 2
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        try:
            x = torch.randn(mid, *input_shape, device=device, dtype=dtype).contiguous(
                memory_format=torch.channels_last
            )
            y = torch.randint(
                0,
                model_kwargs.get("num_classes", 1000),
                (mid,),
                device=device,
            )
            with torch.autocast(device_type="cuda", dtype=dtype):
                out = model(x)
                loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            del x, y, out, loss
            torch.cuda.empty_cache()
            best = mid
            lo = mid + 1
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            hi = mid - 1

    del model, criterion, optimizer
    torch.cuda.empty_cache()

    # Round down to nearest multiple of 8 for tensor-core efficiency
    safe_bs = max(min_batch_size, ((int(best * 0.85)) // 8) * 8)
    print(f"[AutoBS] Max fit: {best}, using batch size: {safe_bs}")
    return safe_bs


def main():
    parser = argparse.ArgumentParser(
        description="Train CliffordNet on ImageNet-1k (multi-node DDP)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./imagenet1k",
        help="Shared storage path where HuggingFace caches ImageNet-1k",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="12_2",
        choices=MODEL_SIZE_CHOICES,
        help="Model size variant. "
        "'probe_*' = tiny models for fast hyperparam sweeps. "
        "'{depth}_{shifts}' = single-stage configs. "
        "'hier_*' = pyramidal ImageNet configs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=24,
        help="Batch size per GPU (0 = auto-detect)",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Base learning rate (consider linear scaling with world size)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs per node",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=1,
        help="Number of nodes",
    )
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="DataLoader workers per GPU process",
    )
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument(
        "--val-check-interval",
        type=float,
        default=0.25,
        help="Run validation every N fraction of an epoch (e.g. 0.25 = 4x per epoch). "
        "Values > 1 are treated as number of training steps.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output dir for checkpoints/logs (should be on NFS for multi-node)",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Checkpoint path to resume from"
    )
    # ---- Wandb ----
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="CliffordNet",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Wandb entity (team/user name)",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Wandb run name (default: auto-generated)",
    )
    parser.add_argument(
        "--wandb-offline",
        action="store_true",
        help="Run wandb in offline mode",
    )
    # ---- Numerical stability & diagnostics ----
    parser.add_argument(
        "--wedge-mode",
        type=str,
        default="fma",
        choices=["naive", "fp32", "fma"],
        help="Numerical strategy for wedge product: "
        "'naive' (fast, bf16 cancellation risk), "
        "'fp32' (upcast subtraction), "
        "'fma' (fused multiply-add, most precise, default)",
    )
    parser.add_argument(
        "--ortho-weight",
        type=float,
        default=0.01,
        help="Weight for orthogonality regularization loss between state/ctx streams "
        "(0.0 = disabled, default: 0.01). Recommended range: 0.01–0.1.",
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Disable runtime wedge statistics and gradient norm logging "
        "(Diagnostics A & C). On by default.",
    )
    parser.add_argument(
        "--diag-log-interval",
        type=int,
        default=100,
        help="Log diagnostic metrics every N training steps (default: 100)",
    )

    args = parser.parse_args()

    L.seed_everything(42)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

    # ---- Auto batch size ----
    # In DDP (torchrun), only global rank 0 runs the GPU memory probe.
    # Other ranks wait for the result via a temp file, because dist is not
    # yet initialized at this point (Lightning handles that later).
    if args.batch_size <= 0:
        global_rank = int(os.environ.get("RANK", 0))
        bs_file = os.path.join(args.output_dir, ".auto_batch_size")
        os.makedirs(args.output_dir, exist_ok=True)

        if global_rank == 0:
            # Remove stale file from previous runs before probing
            if os.path.exists(bs_file):
                os.remove(bs_file)

            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            probe_device = torch.device(f"cuda:{local_rank}")
            print(f"[AutoBS] Probing on {probe_device} ...")
            detected_bs = auto_find_batch_size(
                model_cls=CliffordNet,
                model_kwargs=_model_kwargs_for_size(args.model_size),
                device=probe_device,
            )
            # Write result so other ranks can read it
            with open(bs_file, "w") as f:
                f.write(str(detected_bs))
            print(f"[AutoBS] Batch size: {detected_bs}")
        else:
            # Wait for global rank 0 to finish probing (poll up to 10 min)
            import time

            # First, wait for any stale file to be removed by rank 0
            time.sleep(2)
            for _ in range(600):
                if os.path.exists(bs_file):
                    break
                time.sleep(1)
            with open(bs_file, "r") as f:
                detected_bs = int(f.read().strip())
            print(f"[AutoBS] Rank {global_rank} received batch size: {detected_bs}")

        args.batch_size = detected_bs

    data = ImageNet1kDataModule(
        nfs_data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = CliffordNetLightning(
        model_size=args.model_size,
        num_classes=1000,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        wedge_mode=args.wedge_mode,
        ortho_weight=args.ortho_weight,
        enable_diagnostics=not args.no_diagnostics,
        diag_log_interval=args.diag_log_interval,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        monitor="val/acc1",
        mode="max",
        save_last=True,
        filename="{epoch}-{val/acc1:.4f}",
    )

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        save_dir=args.output_dir,
        offline=args.wandb_offline,
        log_model=False,
        config={
            "model_size": args.model_size,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "num_gpus": args.num_gpus,
            "num_nodes": args.num_nodes,
            "gradient_clip_val": args.gradient_clip_val,
            "wedge_mode": args.wedge_mode,
            "ortho_weight": args.ortho_weight,
            "enable_diagnostics": not args.no_diagnostics,
            "diag_log_interval": args.diag_log_interval,
        },
    )

    trainer = L.Trainer(
        accelerator="gpu",
        devices=args.num_gpus,
        num_nodes=args.num_nodes,
        strategy="ddp" if args.num_gpus > 1 or args.num_nodes > 1 else "auto",
        precision="bf16-mixed",
        max_epochs=args.epochs,
        gradient_clip_val=args.gradient_clip_val,
        val_check_interval=args.val_check_interval,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor("step"),
            RichProgressBar(),
        ],
        logger=wandb_logger,
        log_every_n_steps=10,
    )

    trainer.fit(model, data, ckpt_path=args.resume)


def _model_kwargs_for_size(size):
    """Return CliffordNet constructor kwargs for a given model size string."""
    shifts = _gen_shifts
    configs = {
        # Probe models (hyperparam search)
        "probe_xs": dict(embed_dim=64, depth=4, shifts=shifts(2), drop_path_rate=0.1),
        "probe_s": dict(embed_dim=96, depth=6, shifts=shifts(3), drop_path_rate=0.15),
        # Production models (author-aligned)
        "12_2": dict(embed_dim=128, depth=12, shifts=shifts(2), drop_path_rate=0.3),
        "12_5": dict(embed_dim=128, depth=12, shifts=shifts(5), drop_path_rate=0.3),
        "18_5": dict(embed_dim=128, depth=18, shifts=shifts(5), drop_path_rate=0.3),
        "32_3": dict(embed_dim=128, depth=32, shifts=shifts(3), drop_path_rate=0.3),
        "32_5": dict(embed_dim=128, depth=32, shifts=shifts(5), drop_path_rate=0.3),
        "64_5": dict(embed_dim=128, depth=64, shifts=shifts(5), drop_path_rate=0.4),
        # Hierarchical ImageNet variants
        "hier_p4": dict(
            patch_size=4,
            stage_dims=(48, 96, 160, 256),
            stage_depths=(2, 2, 4, 2),
            stage_shifts=((1,), (1,), (1, 2), (1, 2)),
            drop_path_rate=0.20,
        ),
        "hier_p2": dict(
            patch_size=2,
            stage_dims=(32, 64, 96, 160, 256),
            stage_depths=(1, 2, 2, 4, 2),
            stage_shifts=((1,), (1,), (1,), (1, 2), (1, 2)),
            drop_path_rate=0.15,
        ),
    }
    kwargs = configs[size]
    kwargs["num_classes"] = 1000
    kwargs["in_chans"] = 3
    return kwargs


def model_cls_for_size(size):
    return HierarchicalCliffordNet if size.startswith("hier_") else CliffordNet


if __name__ == "__main__":
    main()
