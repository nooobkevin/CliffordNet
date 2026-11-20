#!/usr/bin/env python3
"""Full 2x2x2 CIFAR-100 factorial evaluator + zero-retraining checkpoint intervention & forensics.

Evaluates:
1. All 24 matched CIFAR-100 checkpoints (4 context rows x {Full, Inner-wide} x 3 seeds {42, 101, 202}):
   - Row 1: no-norm + diff  (Wave 5)
   - Row 2: no-norm + abs   (Wave 6)
   - Row 3: ctx_bn  + diff  (Wave 6)
   - Row 4: ctx_bn  + abs   (Wave 4 s42 + Wave 5 s101/s202)
   Metrics per checkpoint:
   - Clean Top-1, 0.12x Contrast, Gaussian Noise, FFT Phase-Only, Effective Rank
   - Exact per-stage RMS(z2)/RMS(u), mean |SiLU'(u*v)|, frac(|SiLU'(u*v)| < 0.1),
     frac(u*v < -1.28), and post-W_out residual RMS separately for inner vs. wedge columns!
2. Zero-retraining checkpoint intervention (g-sweep) on W_out input columns:
   - Scales wedge features entering W_out by g in [1.0, 0.75, 0.5, 0.0] (and separately inner features)
   - Measures Clean Top-1, 0.12x Contrast, and FFT Phase-Only on:
     (a) CIFAR-100 2x2x2 Full models (all 4 context rows x 3 seeds)
     (b) ImageNet-1k 100-epoch v2-Atto Full checkpoint (50k validation set!)
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from cliffordnet.data_cifar import FastCIFAR100Dataset
from cliffordnet.subspace import get_subspace_model_spec, SubspaceCliffordBlock, subspace_gp_pytorch

CIFAR_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR_STD = (0.2675, 0.2565, 0.2761)
IN1K_MEAN = (0.485, 0.456, 0.406)
IN1K_STD = (0.229, 0.224, 0.225)


def _find_best_ckpt(run_dir: Path) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    best_path = None
    best_val = -1.0
    for p in ckpt_dir.rglob("*.ckpt"):
        if p.name in ("last.ckpt", "preempted.ckpt"):
            continue
        stem = p.stem
        if "=" in stem:
            try:
                val = float(stem.split("=")[-1])
                if val > best_val:
                    best_val = val
                    best_path = p
            except ValueError:
                pass
    if best_path is None:
        best_path = ckpt_dir / "last.ckpt"
    return best_path


def _load_model_from_ckpt(model_size: str, ckpt_path: Path, num_classes: int, device: torch.device) -> nn.Module:
    img_size = 32 if num_classes == 100 else 224
    model = get_subspace_model_spec(model_size, img_size=img_size, num_classes=num_classes).build()
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    for prefix in ("_ema_model.", "_raw_model.", "model."):
        sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if sub:
            model.load_state_dict(sub, strict=True)
            return model.to(device).eval()
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


def _apply_corruption(x_norm: torch.Tensor, kind: str, mean_t: torch.Tensor, std_t: torch.Tensor, seed: int = 0) -> torch.Tensor:
    if kind == "clean":
        return x_norm
    x = (x_norm * std_t + mean_t).clamp(0.0, 1.0)
    if kind == "contrast_012":
        m = x.mean(dim=(2, 3), keepdim=True)
        x = ((x - m) * 0.12 + m).clamp(0.0, 1.0)
    elif kind == "gaussian_noise":
        gen = torch.Generator(device=x.device).manual_seed(seed)
        x = (x + torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype) * 0.12).clamp(0.0, 1.0)
    elif kind == "fft_phase":
        freq = torch.fft.rfft2(x.float(), dim=(-2, -1))
        phase = torch.angle(freq)
        rec = torch.fft.irfft2(torch.polar(torch.ones_like(phase), phase), s=x.shape[-2:], dim=(-2, -1))
        mn = rec.amin(dim=(-2, -1), keepdim=True)
        mx = rec.amax(dim=(-2, -1), keepdim=True)
        x = ((rec - mn) / (mx - mn + 1e-6)).to(x_norm.dtype)
    else:
        raise ValueError(kind)
    return (x - mean_t) / std_t


def _silu_prime(x: torch.Tensor) -> torch.Tensor:
    sig = torch.sigmoid(x)
    return sig * (1.0 + x * (1.0 - sig))


@torch.no_grad()
def _collect_mechanistic_forensics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 16,
) -> dict[str, Any]:
    """Measure per-stage RMS(z2)/RMS(u), |SiLU'(u*v)|, deadzone %, and post-W_out inner vs wedge RMS."""
    blocks_by_stage: list[tuple[str, SubspaceCliffordBlock]] = []
    if hasattr(model, "stages"):
        for s_idx, stage in enumerate(model.stages):
            for b in stage:
                if isinstance(b, SubspaceCliffordBlock):
                    blocks_by_stage.append((f"stage{s_idx+1}", b))

    stats_per_stage: dict[str, dict[str, list[float]]] = {}
    handles = []

    def _make_hook(sname: str, blk: SubspaceCliffordBlock):
        def _hook(module, inp, out):
            x_in = inp[0]
            x_sp = blk.spatial_diffusion(x_in) if hasattr(blk, "spatial_diffusion") else blk.norm(x_in)
            gp = blk.interaction
            u, z2 = gp._prepare_streams(x_sp)
            v = (z2 - u) if gp.ctx_mode == "diff" else z2

            u_f, v_f, z2_f = u.float(), v.float(), z2.float()
            rms_u = float(u_f.pow(2).mean().sqrt().item())
            rms_z2 = float(z2_f.pow(2).mean().sqrt().item())
            ratio_z2_u = rms_z2 / max(rms_u, 1e-8)

            uv = u_f * v_f
            sp = _silu_prime(uv).abs()
            dz_frac = float((uv < -1.2784645).float().mean().item() * 100.0)
            sp_mean = float(sp.mean().item())
            sp_low_frac = float((sp < 0.1).float().mean().item() * 100.0)

            # Exact post-W_out (final_proj * gamma) residual RMS for inner vs wedge!
            # In subspace_gp_pytorch: per group of (num_pairs + k), first num_pairs=k*(k-1)//2 are wedge, last k are inner!
            f_gp = subspace_gp_pytorch(
                u_f, z2_f, gp.k, cli_mode=gp.cli_mode, ctx_mode=gp.ctx_mode, wedge_mode="fp32"
            )
            w_out = gp.final_proj.weight.float()  # (C_out, c_in, 1, 1)
            gamma = blk.gamma.float().flatten().view(1, -1, 1, 1)

            if gp.cli_mode == "full":
                K = gp.k
                n_pairs = K * (K - 1) // 2
                G = gp.num_groups
                stride = n_pairs + K
                wedge_mask = torch.zeros(G * stride, device=f_gp.device, dtype=torch.bool)
                for g_i in range(G):
                    wedge_mask[g_i * stride : g_i * stride + n_pairs] = True
                inner_mask = ~wedge_mask

                wedge_res = F.conv2d(f_gp[:, wedge_mask], w_out[:, wedge_mask]) * gamma
                inner_res = F.conv2d(f_gp[:, inner_mask], w_out[:, inner_mask]) * gamma
                inner_post_rms = float(inner_res.pow(2).mean().sqrt().item())
                wedge_post_rms = float(wedge_res.pow(2).mean().sqrt().item())
                post_ratio = wedge_post_rms / max(inner_post_rms, 1e-8)
            else:
                inner_res = F.conv2d(f_gp, w_out) * gamma
                inner_post_rms = float(inner_res.pow(2).mean().sqrt().item())
                wedge_post_rms = 0.0
                post_ratio = 0.0

            d = stats_per_stage.setdefault(sname, {k: [] for k in (
                "rms_z2_over_u", "silu_prime_abs_mean", "silu_prime_low_pct",
                "deadzone_pct", "post_wout_inner_rms", "post_wout_wedge_rms", "post_wout_ratio"
            )})
            d["rms_z2_over_u"].append(ratio_z2_u)
            d["silu_prime_abs_mean"].append(sp_mean)
            d["silu_prime_low_pct"].append(sp_low_frac)
            d["deadzone_pct"].append(dz_frac)
            d["post_wout_inner_rms"].append(inner_post_rms)
            d["post_wout_wedge_rms"].append(wedge_post_rms)
            d["post_wout_ratio"].append(post_ratio)
        return _hook

    for sname, blk in blocks_by_stage:
        handles.append(blk.register_forward_hook(_make_hook(sname, blk)))

    for bi, (imgs, _) in enumerate(loader):
        if bi >= max_batches:
            break
        _ = model(imgs.to(device))

    for h in handles:
        h.remove()

    out = {}
    for sname, d in stats_per_stage.items():
        out[sname] = {k: round(float(np.mean(v)), 4) for k, v in d.items()}
    return out


@torch.no_grad()
def _eval_acc(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    corruptions: list[str],
    compute_erank: bool = False,
    max_batches: int | None = None,
) -> dict[str, float]:
    res: dict[str, float] = {}
    for kind in corruptions:
        correct = 0
        total = 0
        feats_list = []
        for bi, (imgs, targets) in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            x = _apply_corruption(imgs, kind, mean_t, std_t, seed=bi)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                if compute_erank and kind == "clean" and bi < 16 and hasattr(model, "forward_features"):
                    f = model.forward_features(x)
                    if f.ndim == 4:
                        f = f.mean(dim=(2, 3))
                    feats_list.append(f.float().cpu())
                logits = model(x)
            pred = logits.argmax(dim=1)
            correct += int((pred == targets).sum().item())
            total += int(targets.size(0))
        res[kind] = round(100.0 * correct / max(total, 1), 2)
        if compute_erank and kind == "clean" and feats_list:
            F_mat = torch.cat(feats_list, dim=0)
            F_mat = F_mat - F_mat.mean(dim=0, keepdim=True)
            s = torch.linalg.svdvals(F_mat)
            p = (s ** 2) / (s ** 2).sum().clamp(min=1e-12)
            erank = float(torch.exp(-(p * torch.log(p.clamp(min=1e-12))).sum()).item() / F_mat.shape[1])
            res["effective_rank"] = round(erank, 4)
    return res


@torch.no_grad()
def _run_wout_intervention(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    gains: list[float] = [1.0, 0.75, 0.5, 0.0],
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Scale wedge columns (or inner columns) of out_proj.weight by g without retraining."""
    gp_modules = []
    for m in model.modules():
        if isinstance(m, SubspaceCliffordBlock) and m.interaction.cli_mode == "full":
            gp_modules.append(m.interaction)

    # Save exact original final_proj weights and build per-group [wedge(n_pairs), inner(k)] masks
    orig_weights = [gp.final_proj.weight.detach().clone() for gp in gp_modules]
    masks = []
    for gp in gp_modules:
        K = gp.k
        n_pairs = K * (K - 1) // 2
        G = gp.num_groups
        stride = n_pairs + K
        wmask = torch.zeros(G * stride, device=device, dtype=torch.bool)
        for g_i in range(G):
            wmask[g_i * stride : g_i * stride + n_pairs] = True
        masks.append(wmask)

    out: dict[str, Any] = {"wedge_scale": {}, "inner_scale": {}}
    for branch in ("wedge_scale", "inner_scale"):
        for g in gains:
            for gp, orig_w, wmask in zip(gp_modules, orig_weights, masks):
                w = orig_w.clone()
                if branch == "wedge_scale":
                    w[:, wmask] *= g
                else:
                    w[:, ~wmask] *= g
                gp.final_proj.weight.copy_(w)
            accs = _eval_acc(
                model, loader, device, mean_t, std_t,
                corruptions=["clean", "contrast_012", "fft_phase"],
                compute_erank=False,
                max_batches=max_batches,
            )
            out[branch][str(g)] = accs

    # Restore original weights
    for gp, orig_w in zip(gp_modules, orig_weights):
        gp.final_proj.weight.copy_(orig_w)
    return out


def main() -> None:
    t0 = time.monotonic()
    device = torch.device("cuda:0")
    base = Path("/scratch/itscspod/chtaihei/cliffordnet_runs")

    # CIFAR-100 test loader
    tf = T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
    ds = FastCIFAR100Dataset(data_dir="/scratch/itscspod/chtaihei/cifar100", train=False, transform=tf)
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    c_mean = torch.tensor(CIFAR_MEAN, device=device).view(1, 3, 1, 1)
    c_std = torch.tensor(CIFAR_STD, device=device).view(1, 3, 1, 1)

    rows_spec = {
        "row1_diff_nonorm": {
            "label": "diff + no norm (Original)",
            "ctx_mode": "diff", "ctx_bn": False,
            "full": {
                42: ("subspace_hier_k4_ctrl_orig_full", base / "cifar100_wave5_matched_2x2/m2x2_orig_full_s42"),
                101: ("subspace_hier_k4_ctrl_orig_full", base / "cifar100_wave5_matched_2x2/m2x2_orig_full_s101"),
                202: ("subspace_hier_k4_ctrl_orig_full", base / "cifar100_wave5_matched_2x2/m2x2_orig_full_s202"),
            },
            "inner": {
                42: ("subspace_hier_k4_ctrl_orig_inner_wide", base / "cifar100_wave5_matched_2x2/m2x2_orig_inner_s42"),
                101: ("subspace_hier_k4_ctrl_orig_inner_wide", base / "cifar100_wave5_matched_2x2/m2x2_orig_inner_s101"),
                202: ("subspace_hier_k4_ctrl_orig_inner_wide", base / "cifar100_wave5_matched_2x2/m2x2_orig_inner_s202"),
            },
        },
        "row2_abs_nonorm": {
            "label": "abs + no norm (Only remove -u^2)",
            "ctx_mode": "abs", "ctx_bn": False,
            "full": {
                42: ("subspace_hier_k4_ctrl_abs_nonorm_full", base / "cifar100_wave6_factorial_2x2x2/m2x2x2_abs_nonorm_full_s42"),
                101: ("subspace_hier_k4_ctrl_abs_nonorm_full", base / "cifar100_wave6_factorial_2x2x2/m2x2x2_abs_nonorm_full_s101"),
                202: ("subspace_hier_k4_ctrl_abs_nonorm_full", base / "cifar100_wave6_factorial_2x2x2/m2x2x2_abs_nonorm_full_s202"),
            },
            "inner": {
                42: ("subspace_hier_k4_ctrl_abs_nonorm_inner_wide", base / "cifar100_wave6_factorial_2x2x2/m2x2x2_abs_nonorm_inner_s42"),
                101: ("subspace_hier_k4_ctrl_abs_nonorm_inner_wide", base / "cifar100_wave6_factorial_2x2x2/m2x2x2_abs_nonorm_inner_s101"),
                202: ("subspace_hier_k4_ctrl_abs_nonorm_inner_wide", base / "cifar100_wave6_factorial_2x2x2/m2x2x2_abs_nonorm_inner_s202"),
            },
        },
        "row3_diff_ctxbn": {
            "label": "diff + ctx_bn (Only restore context scale)",
            "ctx_mode": "diff", "ctx_bn": True,
            "full": {
                42: ("subspace_hier_k4_v2_ctxbn_diff", base / "cifar100_wave6_factorial_2x2x2/m2x2x2_diff_ctxbn_full_s42"),
                101: ("subspace_hier_k4_v2_ctxbn_diff", base / "cifar100_wave6_factorial_2x2x2/m2x2x2_diff_ctxbn_full_s101"),
                202: ("subspace_hier_k4_v2_ctxbn_diff", base / "cifar100_wave6_factorial_2x2x2/m2x2x2_diff_ctxbn_full_s202"),
            },
            "inner": {
                42: ("subspace_hier_k4_ctrl_diff_ctxbn_inner_wide", base / "cifar100_wave6_factorial_2x2x2/m2x2x2_diff_ctxbn_inner_s42"),
                101: ("subspace_hier_k4_ctrl_diff_ctxbn_inner_wide", base / "cifar100_wave6_factorial_2x2x2/m2x2x2_diff_ctxbn_inner_s101"),
                202: ("subspace_hier_k4_ctrl_diff_ctxbn_inner_wide", base / "cifar100_wave6_factorial_2x2x2/m2x2x2_diff_ctxbn_inner_s202"),
            },
        },
        "row4_abs_ctxbn": {
            "label": "abs + ctx_bn (Full v2 fix)",
            "ctx_mode": "abs", "ctx_bn": True,
            "full": {
                42: ("subspace_hier_k4_v2", base / "cifar100_wave4/w4_v2_k4_dpr020"),
                101: ("subspace_hier_k4_v2", base / "cifar100_wave5_matched_2x2/m2x2_fixed_full_s101"),
                202: ("subspace_hier_k4_v2", base / "cifar100_wave5_matched_2x2/m2x2_fixed_full_s202"),
            },
            "inner": {
                42: ("subspace_hier_k4_v2_inner_wide", base / "cifar100_wave4/w4_v2_k4_inner_wide"),
                101: ("subspace_hier_k4_v2_inner_wide", base / "cifar100_wave5_matched_2x2/m2x2_fixed_inner_s101"),
                202: ("subspace_hier_k4_v2_inner_wide", base / "cifar100_wave5_matched_2x2/m2x2_fixed_inner_s202"),
            },
        },
    }

    factorial_out: dict[str, Any] = {}
    for row_key, rspec in rows_spec.items():
        print(f"\n[eval] === {row_key}: {rspec['label']} ===", flush=True)
        row_res: dict[str, Any] = {"label": rspec["label"], "ctx_mode": rspec["ctx_mode"], "ctx_bn": rspec["ctx_bn"], "full": {}, "inner": {}}
        for branch in ("full", "inner"):
            for seed, (msize, rdir) in rspec[branch].items():
                ckpt = _find_best_ckpt(rdir)
                model = _load_model_from_ckpt(msize, ckpt, num_classes=100, device=device)
                accs = _eval_acc(model, loader, device, c_mean, c_std, ["clean", "contrast_012", "gaussian_noise", "fft_phase"], compute_erank=True)
                forensics = _collect_mechanistic_forensics(model, loader, device)
                entry: dict[str, Any] = {"ckpt": ckpt.name, "metrics": accs, "forensics": forensics}
                if branch == "full":
                    entry["intervention"] = _run_wout_intervention(model, loader, device, c_mean, c_std)
                row_res[branch][str(seed)] = entry
                print(
                    f"  {branch:5s} s{seed}: clean={accs['clean']:.2f}% fft={accs['fft_phase']:.2f}% "
                    f"s3_rms_z2/u={forensics['stage3']['rms_z2_over_u']:.2f} "
                    f"s3_|silu'|={forensics['stage3']['silu_prime_abs_mean']:.3f} "
                    f"s3_postW_ratio={forensics['stage3']['post_wout_ratio']:.2f}",
                    flush=True,
                )

        # Paired Full - Inner summary
        seeds = ["42", "101", "202"]
        f_clean = [row_res["full"][s]["metrics"]["clean"] for s in seeds]
        i_clean = [row_res["inner"][s]["metrics"]["clean"] for s in seeds]
        diffs = [a - b for a, b in zip(f_clean, i_clean)]
        row_res["summary"] = {
            "full_clean_mean": round(float(np.mean(f_clean)), 2),
            "full_clean_std": round(float(np.std(f_clean, ddof=1)), 2),
            "inner_clean_mean": round(float(np.mean(i_clean)), 2),
            "inner_clean_std": round(float(np.std(i_clean, ddof=1)), 2),
            "paired_diff_mean": round(float(np.mean(diffs)), 2),
            "paired_diff_std": round(float(np.std(diffs, ddof=1)), 2),
            "paired_diffs": [round(x, 2) for x in diffs],
            "full_fft_mean": round(float(np.mean([row_res["full"][s]["metrics"]["fft_phase"] for s in seeds])), 2),
            "inner_fft_mean": round(float(np.mean([row_res["inner"][s]["metrics"]["fft_phase"] for s in seeds])), 2),
            "s3_rms_z2_over_u_full": round(float(np.mean([row_res["full"][s]["forensics"]["stage3"]["rms_z2_over_u"] for s in seeds])), 3),
            "s3_silu_prime_full": round(float(np.mean([row_res["full"][s]["forensics"]["stage3"]["silu_prime_abs_mean"] for s in seeds])), 3),
            "s3_post_wout_ratio_full": round(float(np.mean([row_res["full"][s]["forensics"]["stage3"]["post_wout_ratio"] for s in seeds])), 3),
        }
        # Aggregate intervention across seeds
        inv_agg: dict[str, Any] = {"wedge_scale": {}, "inner_scale": {}}
        for bname in ("wedge_scale", "inner_scale"):
            for g in ("1.0", "0.75", "0.5", "0.0"):
                inv_agg[bname][g] = {
                    m: round(float(np.mean([row_res["full"][s]["intervention"][bname][g][m] for s in seeds])), 2)
                    for m in ("clean", "contrast_012", "fft_phase")
                }
        row_res["intervention_mean"] = inv_agg
        factorial_out[row_key] = row_res

    # ImageNet-1k 100-epoch v2-Atto Full checkpoint intervention (on /dev/shm val cache, fast!)
    in1k_out: dict[str, Any] = {}
    in1k_val_img = Path("/dev/shm/in1k_raw_cache_ar320/val_images.npy")
    in1k_val_lbl = Path("/dev/shm/in1k_raw_cache_ar320/val_labels.npy")
    if in1k_val_img.exists():
        print("\n[eval] === ImageNet-1k 100-ep v2-Atto Full Checkpoint Intervention ===", flush=True)
        imgs_mm = np.load(str(in1k_val_img), mmap_mode="r")
        lbls_mm = np.load(str(in1k_val_lbl), mmap_mode="r")

        class _ValDS(torch.utils.data.Dataset):
            def __len__(self): return len(lbls_mm)
            def __getitem__(self, i):
                x = torch.from_numpy(np.array(imgs_mm[i], copy=True)).float().div_(255.0)
                return x, int(lbls_mm[i])

        in1k_loader_raw = DataLoader(_ValDS(), batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
        i_mean = torch.tensor(IN1K_MEAN, device=device).view(1, 3, 1, 1)
        i_std = torch.tensor(IN1K_STD, device=device).view(1, 3, 1, 1)

        class _NormWrap:
            def __init__(self, dl): self.dl = dl
            def __iter__(self):
                for x, y in self.dl:
                    yield (x.to(device) - i_mean) / i_std, y

        in1k_ckpt = _find_best_ckpt(base / "in1k_waveC_100ep/in1k_v2_atto_k4_100ep")
        in1k_model = _load_model_from_ckpt("in1k_v2_atto_k4", in1k_ckpt, num_classes=1000, device=device)
        in1k_forensics = _collect_mechanistic_forensics(in1k_model, _NormWrap(in1k_loader_raw), device, max_batches=16)
        in1k_intervention = _run_wout_intervention(
            in1k_model, _NormWrap(in1k_loader_raw), device, i_mean, i_std,
            gains=[1.0, 0.75, 0.5, 0.0], max_batches=40,  # 10,240 val images (~20% stratified) for fast exact sweep
        )
        in1k_out = {"ckpt": in1k_ckpt.name, "val_samples": 10240, "forensics": in1k_forensics, "intervention": in1k_intervention}
        print("[eval] IN1K v2-Atto intervention:", json.dumps(in1k_intervention), flush=True)

    final = {
        "wall_time_sec": round(time.monotonic() - t0, 2),
        "cifar100_2x2x2": factorial_out,
        "imagenet1k_v2_atto_intervention": in1k_out,
    }
    out_path = Path("/project/itscspod/chtaihei/cliffordnet/CliffordNet/experiments/subspace_study/results/summary_wave6_factorial_2x2x2_and_intervention.json")
    out_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(f"\n[eval] Wrote {out_path} in {time.monotonic() - t0:.1f}s!", flush=True)

    print("\n=== FULL 2x2x2 FACTORIAL SUMMARY (CIFAR-100, 200 ep, 3 paired seeds) ===")
    silu_hdr = "|SiLU'|"
    print(f"{'Context':36s} | {'Full':>11s} | {'Inner-wide':>11s} | {'Full-Inner':>11s} | {'z2/u':>5s} | {silu_hdr:>7s} | {'W/I post':>8s}")
    for rk, rv in factorial_out.items():
        s = rv["summary"]
        print(
            f"{rv['label']:36s} | {s['full_clean_mean']:5.2f}±{s['full_clean_std']:.2f} | "
            f"{s['inner_clean_mean']:5.2f}±{s['inner_clean_std']:.2f} | "
            f"{s['paired_diff_mean']:+5.2f}±{s['paired_diff_std']:.2f} | "
            f"{s['s3_rms_z2_over_u_full']:5.2f} | {s['s3_silu_prime_full']:6.3f} | {s['s3_post_wout_ratio_full']:8.2f}"
        )


if __name__ == "__main__":
    main()
