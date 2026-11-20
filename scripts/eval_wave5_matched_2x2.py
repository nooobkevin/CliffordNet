#!/usr/bin/env python3
"""Evaluate all 12 matched 2x2 CIFAR-100 runs (4 cells x 3 seeds: 42, 101, 202).

Computes:
  - Per-seed & mean+-std Clean Top-1, Low Contrast (0.12x), Gaussian Noise, FFT Phase-Only
  - Feature Effective Rank and empirical SiLU dead-zone fraction (u*v < -1.28)
  - Paired interaction effect (A - B) - (C - D) across matched seeds
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cliffordnet.data_cifar import (
    CIFAR100_MEAN,
    CIFAR100_STD,
    FastCIFAR100Dataset,
    apply_photometric_or_geometric_corruption,
)
from cliffordnet.subspace import SubspaceGeometricProduct
from cliffordnet.tasks.imagenet1k import CliffordNetLightning

WAVE4_ROOT = Path("/scratch/itscspod/chtaihei/cliffordnet_runs/cifar100_wave4")
WAVE5_ROOT = Path("/scratch/itscspod/chtaihei/cliffordnet_runs/cifar100_wave5_matched_2x2")
DATA_DIR = "/scratch/itscspod/chtaihei/cifar100"

CELL_SPECS = {
    "A_orig_full": {
        "label": "Cell A: Original context (diff, no ctx_bn) + Full (Inner + Wedge)",
        "context": "original",
        "branch": "full",
        "runs": {
            42: WAVE5_ROOT / "m2x2_orig_full_s42",
            101: WAVE5_ROOT / "m2x2_orig_full_s101",
            202: WAVE5_ROOT / "m2x2_orig_full_s202",
        },
    },
    "B_orig_inner": {
        "label": "Cell B: Original context (diff, no ctx_bn) + Inner-wide (0 Bivectors)",
        "context": "original",
        "branch": "inner_wide",
        "runs": {
            42: WAVE5_ROOT / "m2x2_orig_inner_s42",
            101: WAVE5_ROOT / "m2x2_orig_inner_s101",
            202: WAVE5_ROOT / "m2x2_orig_inner_s202",
        },
    },
    "C_fixed_full": {
        "label": "Cell C: Fixed context (abs + ctx_bn) + Full (Inner + Wedge)",
        "context": "fixed",
        "branch": "full",
        "runs": {
            42: WAVE4_ROOT / "w4_v2_k4_dpr020",
            101: WAVE5_ROOT / "m2x2_fixed_full_s101",
            202: WAVE5_ROOT / "m2x2_fixed_full_s202",
        },
    },
    "D_fixed_inner": {
        "label": "Cell D: Fixed context (abs + ctx_bn) + Inner-wide (0 Bivectors)",
        "context": "fixed",
        "branch": "inner_wide",
        "runs": {
            42: WAVE4_ROOT / "w4_v2_k4_inner_wide",
            101: WAVE5_ROOT / "m2x2_fixed_inner_s101",
            202: WAVE5_ROOT / "m2x2_fixed_inner_s202",
        },
    },
}

METRICS = ["clean", "contrast_0.12", "gaussian_noise_0.08", "fft_phase_only"]


def _find_best_ckpt(run_dir: Path) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    candidates = [p for p in ckpt_dir.rglob("*.ckpt") if p.name not in ("last.ckpt", "preempted.ckpt")]
    if candidates:
        def _score(p: Path) -> float:
            stem = p.stem.split("=")[-1]
            try:
                return float(stem.split("-")[-1])
            except ValueError:
                return p.stat().st_mtime
        return max(candidates, key=_score)
    return ckpt_dir / "last.ckpt"


@torch.no_grad()
def _measure_deadzone_stats(model: torch.nn.Module, loader: DataLoader, mean: torch.Tensor, std: torch.Tensor, device: torch.device) -> dict[str, Any]:
    """Hook SubspaceGeometricProduct inputs to measure exact u*v < -1.28 fraction per stage."""
    stage_stats: list[dict[str, float]] = []
    # Run 4 batches and inspect every SubspaceGeometricProduct module directly
    modules: list[tuple[int, SubspaceGeometricProduct]] = []
    for s_idx, stage in enumerate(model.stages):  # type: ignore[attr-defined]
        for blk in stage:
            if isinstance(blk.interaction, SubspaceGeometricProduct):
                modules.append((s_idx, blk.interaction))

    accum: dict[int, dict[str, list[float]]] = {0: {"dz": [], "mean_uv": [], "w2d": []}, 1: {"dz": [], "mean_uv": [], "w2d": []}, 2: {"dz": [], "mean_uv": [], "w2d": []}}
    hooks = []

    def _make_hook(stage_idx: int, mod: SubspaceGeometricProduct):
        def _hook(_m: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            x = inputs[0]
            z1, z2 = mod._prepare_streams(x)
            v = (z2 - z1) if mod.ctx_mode == "diff" else z2
            uv = (z1 * v).float()
            dz_frac = float((uv < -1.28).float().mean().item()) * 100.0
            mean_uv = float(uv.mean().item())
            dot_mag = float(torch.nn.functional.silu(uv).abs().mean().item())
            # Wedge mag if full
            if mod.cli_mode == "full" and mod.k == 4:
                B, C, H, W = z1.shape
                u4 = z1.float().view(B, C // 4, 4, H, W)
                v4 = v.float().view(B, C // 4, 4, H, W)
                w_list = [
                    u4[:, :, i] * v4[:, :, j] - u4[:, :, j] * v4[:, :, i]
                    for i in range(4) for j in range(i + 1, 4)
                ]
                w_mag = float(torch.stack(w_list, dim=2).abs().mean().item())
                w2d = w_mag / max(dot_mag, 1e-8)
            else:
                w2d = 0.0
            accum[stage_idx]["dz"].append(dz_frac)
            accum[stage_idx]["mean_uv"].append(mean_uv)
            accum[stage_idx]["w2d"].append(w2d)
        return _hook

    for s_idx, mod in modules:
        hooks.append(mod.register_forward_pre_hook(_make_hook(s_idx, mod)))

    try:
        for b_idx, (imgs_01, _) in enumerate(loader):
            if b_idx >= 4:
                break
            imgs_01 = imgs_01[:64].to(device)
            normed = ((imgs_01 - mean) / std).contiguous(memory_format=torch.channels_last)
            _ = model(normed)
    finally:
        for h in hooks:
            h.remove()

    per_stage = {}
    all_dz = []
    for s_idx in sorted(accum):
        dz_mean = sum(accum[s_idx]["dz"]) / max(1, len(accum[s_idx]["dz"]))
        all_dz.extend(accum[s_idx]["dz"])
        per_stage[f"stage{s_idx+1}"] = {
            "deadzone_pct": round(dz_mean, 2),
            "mean_uv": round(sum(accum[s_idx]["mean_uv"]) / max(1, len(accum[s_idx]["mean_uv"])), 3),
            "wedge_to_dot": round(sum(accum[s_idx]["w2d"]) / max(1, len(accum[s_idx]["w2d"])), 2),
        }
    return {
        "overall_deadzone_pct": round(sum(all_dz) / max(1, len(all_dz)), 2),
        "stage3_deadzone_pct": per_stage["stage3"]["deadzone_pct"],
        "per_stage": per_stage,
    }


@torch.no_grad()
def _eval_one(ckpt_path: Path, loader: DataLoader, mean: torch.Tensor, std: torch.Tensor, device: torch.device) -> dict[str, Any]:
    lit = CliffordNetLightning.load_from_checkpoint(str(ckpt_path), map_location=device)
    lit.eval()
    model = lit._raw_model.to(device).eval()
    params_m = round(sum(p.numel() for p in model.parameters()) / 1e6, 3)

    accs: dict[str, float] = {}
    for corr in METRICS:
        gen = torch.Generator(device=device)
        gen.manual_seed(2026)
        correct = 0
        total = 0
        for imgs_01, labels in loader:
            imgs_01 = imgs_01.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            corrupted = apply_photometric_or_geometric_corruption(imgs_01, corr, generator=gen)
            normed = ((corrupted - mean) / std).contiguous(memory_format=torch.channels_last)
            preds = model(normed).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
        accs[corr] = round(100.0 * correct / max(1, total), 2)

    # Effective rank from forward_diagnostics
    eff_ranks = []
    for b_idx, (imgs_01, _) in enumerate(loader):
        if b_idx >= 4:
            break
        imgs_01 = imgs_01[:64].to(device)
        normed = ((imgs_01 - mean) / std).contiguous(memory_format=torch.channels_last)
        d = model.forward_diagnostics(normed.float())
        for k, v in d.items():
            if k.endswith("effective_rank"):
                eff_ranks.append(float(v.detach().cpu()))
    eff_rank = round(sum(eff_ranks) / max(1, len(eff_ranks)), 4)

    dz = _measure_deadzone_stats(model, loader, mean, std, device)

    del lit, model
    torch.cuda.empty_cache()
    return {
        "ckpt": ckpt_path.name,
        "params_m": params_m,
        "clean_top1": accs["clean"],
        "contrast_012": accs["contrast_0.12"],
        "gaussian_noise": accs["gaussian_noise_0.08"],
        "fft_phase": accs["fft_phase_only"],
        "effective_rank": eff_rank,
        "deadzone": dz,
    }


def _agg(vals: list[float]) -> dict[str, float]:
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return {"mean": round(mean, 2), "std": round(std, 2), "values": [round(v, 2) for v in vals]}


def _agg4(vals: list[float]) -> dict[str, float]:
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return {"mean": round(mean, 3), "std": round(std, 3), "values": [round(v, 3) for v in vals]}


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    raw_val_ds = FastCIFAR100Dataset(data_dir=DATA_DIR, train=False, transform=transforms.ToTensor())
    loader = DataLoader(raw_val_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    mean = torch.tensor(CIFAR100_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR100_STD, device=device).view(1, 3, 1, 1)

    cells_out: dict[str, Any] = {}
    for cell_key, spec in CELL_SPECS.items():
        per_seed: dict[str, Any] = {}
        for seed, run_dir in spec["runs"].items():
            ckpt = _find_best_ckpt(run_dir)
            print(f"Evaluating {cell_key} seed={seed} ({ckpt})...", flush=True)
            per_seed[str(seed)] = _eval_one(ckpt, loader, mean, std, device)

        seeds = [str(s) for s in (42, 101, 202)]
        cells_out[cell_key] = {
            "label": spec["label"],
            "context": spec["context"],
            "branch": spec["branch"],
            "params_m": per_seed["42"]["params_m"],
            "per_seed": per_seed,
            "summary": {
                "clean_top1": _agg([per_seed[s]["clean_top1"] for s in seeds]),
                "contrast_012": _agg([per_seed[s]["contrast_012"] for s in seeds]),
                "gaussian_noise": _agg([per_seed[s]["gaussian_noise"] for s in seeds]),
                "fft_phase": _agg([per_seed[s]["fft_phase"] for s in seeds]),
                "effective_rank": _agg4([per_seed[s]["effective_rank"] for s in seeds]),
                "overall_deadzone_pct": _agg([per_seed[s]["deadzone"]["overall_deadzone_pct"] for s in seeds]),
                "stage3_deadzone_pct": _agg([per_seed[s]["deadzone"]["stage3_deadzone_pct"] for s in seeds]),
            },
        }

    # Compute paired per-seed interaction effects: (A - B) - (C - D)
    interactions: dict[str, Any] = {}
    for metric in ("clean_top1", "contrast_012", "gaussian_noise", "fft_phase", "effective_rank"):
        ab = []
        cd = []
        inter = []
        for s in ("42", "101", "202"):
            a = cells_out["A_orig_full"]["per_seed"][s][metric]
            b = cells_out["B_orig_inner"]["per_seed"][s][metric]
            c = cells_out["C_fixed_full"]["per_seed"][s][metric]
            d = cells_out["D_fixed_inner"]["per_seed"][s][metric]
            ab.append(a - b)
            cd.append(c - d)
            inter.append((a - b) - (c - d))
        fn = _agg4 if metric == "effective_rank" else _agg
        interactions[metric] = {
            "orig_gap_A_minus_B": fn(ab),
            "fixed_gap_C_minus_D": fn(cd),
            "interaction_AB_minus_CD": fn(inter),
        }

    result = {
        "status": "complete",
        "protocol": "CIFAR-100 200-epoch matched 2x2 control (seeds 42, 101, 202; identical dw5 backbone, proj_bias=True, dpr=0.20, AdamW lr=1e-3, wd=0.1)",
        "cells": cells_out,
        "interactions": interactions,
    }
    out_path = REPO_ROOT / "experiments/subspace_study/results/summary_cifar100_wave5_matched_2x2.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved matched 2x2 summary to {out_path}")
    print(json.dumps(interactions, indent=2))


if __name__ == "__main__":
    main()
