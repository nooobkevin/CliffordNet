#!/usr/bin/env python3
"""
Evaluate trained CliffordNet / Subspace-CAN checkpoints on clean + structured
photometric/geometric corruptions (Track 5 / ROI #5) AND collect final-checkpoint
mechanistic diagnostics (Track 3 / ROI #3: wedge/dot ratio, collinearity, effective rank,
floating-point lambda-invariance gap).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from cliffordnet.data_cifar import (
    CIFAR100_MEAN,
    CIFAR100_STD,
    CORRUPTION_SUITE,
    FastCIFAR100Dataset,
    apply_photometric_or_geometric_corruption,
)
from cliffordnet.tasks.imagenet1k import CliffordNetLightning


def _find_best_checkpoint(run_dir: Path) -> Path | None:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    candidates = [p for p in ckpt_dir.rglob("*.ckpt") if p.name not in ("last.ckpt", "preempted.ckpt")]
    if candidates:
        # Sort by best accuracy encoded in "{epoch}-{val/acc1:.4f}.ckpt" or mtime fallback
        def _score(path: Path) -> float:
            stem = path.stem
            if "=" in stem:
                stem = stem.split("=")[-1]
            parts = stem.split("-")
            try:
                return float(parts[-1])
            except ValueError:
                return path.stat().st_mtime

        return max(candidates, key=_score)
    last_ckpt = ckpt_dir / "last.ckpt"
    return last_ckpt if last_ckpt.exists() else None


@torch.no_grad()
def evaluate_checkpoint_on_cifar100_corruptions(
    ckpt_path: Path,
    data_dir: str,
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 4,
) -> dict[str, Any]:
    lit = CliffordNetLightning.load_from_checkpoint(str(ckpt_path), map_location=device)
    lit.eval()
    eval_model = lit._validation_model() if getattr(lit, "ema_decay", 0.0) > 0 else lit._raw_model
    eval_model.to(device)
    eval_model.eval()

    n_params = sum(p.numel() for p in lit._raw_model.parameters() if p.requires_grad)

    raw_val_ds = FastCIFAR100Dataset(
        data_dir=data_dir, train=False, transform=transforms.ToTensor()
    )
    loader = DataLoader(
        raw_val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    mean = torch.tensor(CIFAR100_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR100_STD, device=device).view(1, 3, 1, 1)

    corruption_accs: dict[str, float] = {}
    for corr in CORRUPTION_SUITE:
        gen = torch.Generator(device=device)
        gen.manual_seed(2026)
        correct = 0
        total = 0
        for imgs_01, labels in loader:
            imgs_01 = imgs_01.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            corrupted = apply_photometric_or_geometric_corruption(imgs_01, corr, generator=gen)
            normed = ((corrupted - mean) / std).contiguous(memory_format=torch.channels_last)
            logits = eval_model(normed)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
        corruption_accs[corr] = round(100.0 * correct / max(1, total), 2)

    # Collect mechanistic diagnostics across 4 batches (fp32 and bf16)
    diag_fp32_accum: dict[str, list[float]] = {}
    diag_bf16_accum: dict[str, list[float]] = {}
    if hasattr(eval_model, "forward_diagnostics"):
        for b_idx, (imgs_01, _labels) in enumerate(loader):
            if b_idx >= 4:
                break
            imgs_01 = imgs_01[:32].to(device)
            normed = ((imgs_01 - mean) / std).contiguous(memory_format=torch.channels_last)
            d32 = eval_model.forward_diagnostics(normed.float())
            for k, v in d32.items():
                diag_fp32_accum.setdefault(k, []).append(float(v.detach().cpu()))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                d16 = eval_model.forward_diagnostics(normed.to(torch.bfloat16))
            for k, v in d16.items():
                diag_bf16_accum.setdefault(k, []).append(float(v.detach().cpu()))

    def _summarize_diags(accum: dict[str, list[float]]) -> dict[str, float]:
        if not accum:
            return {}
        mean_per_key = {k: sum(vs) / len(vs) for k, vs in accum.items()}
        suffixes = sorted({"/".join(k.split("/")[1:]) for k in mean_per_key if "/" in k})
        out: dict[str, float] = {}
        for sfx in suffixes:
            vals = [v for k, v in mean_per_key.items() if k.endswith(sfx)]
            if vals:
                out[sfx] = round(sum(vals) / len(vals), 6)
        # Also record per-stage wedge_to_dot_ratio and effective_rank if hierarchical
        for stage_idx in range(5):
            prefix = f"stage_{stage_idx}_"
            for metric in ("magnitude/wedge_to_dot_ratio", "ortho/subspace_cos_sim_mean", "spectrum/effective_rank"):
                vals = [v for k, v in mean_per_key.items() if k.startswith(prefix) and k.endswith(metric)]
                if vals:
                    out[f"stage_{stage_idx}/{metric}"] = round(sum(vals) / len(vals), 6)
        return out

    clean_acc = corruption_accs["clean"]
    photometric_keys = [
        "contrast_0.25",
        "contrast_0.12",
        "contrast_2.0",
        "brightness_+0.30",
        "brightness_-0.30",
        "gamma_2.2",
        "fog_veil",
    ]
    structural_keys = ["highpass_edges", "fft_phase_only"]

    photo_mean = round(sum(corruption_accs[k] for k in photometric_keys) / len(photometric_keys), 2)
    struct_mean = round(sum(corruption_accs[k] for k in structural_keys) / len(structural_keys), 2)

    return {
        "checkpoint": str(ckpt_path),
        "model_size": lit.hparams.get("model_size", "unknown"),
        "params": n_params,
        "params_m": round(n_params / 1e6, 3),
        "clean_acc1": clean_acc,
        "photometric_mean_acc1": photo_mean,
        "photometric_retention_pct": round(100.0 * photo_mean / max(1e-6, clean_acc), 2),
        "structural_edge_phase_mean_acc1": struct_mean,
        "corruptions": corruption_accs,
        "diagnostics_fp32": _summarize_diags(diag_fp32_accum),
        "diagnostics_bf16": _summarize_diags(diag_bf16_accum),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on corruption suite & diagnostics")
    parser.add_argument("--runs-root", type=str, required=True, help="Directory containing run folders")
    parser.add_argument("--data-dir", type=str, default="./data/cifar100")
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--output-md", type=str, required=True)
    parser.add_argument("--filter", type=str, default="", help="Optional substring filter on run folder names")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs_root = Path(args.runs_root)
    run_dirs = sorted([p for p in runs_root.iterdir() if p.is_dir() and args.filter in p.name])

    results: dict[str, Any] = {}
    for run_dir in run_dirs:
        ckpt = _find_best_checkpoint(run_dir)
        if ckpt is None:
            print(f"[skip] no checkpoint in {run_dir}")
            continue
        print(f"[eval] {run_dir.name} -> {ckpt.name}")
        results[run_dir.name] = evaluate_checkpoint_on_cifar100_corruptions(
            ckpt_path=ckpt,
            data_dir=args.data_dir,
            device=device,
        )

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [
        "# Corruption Robustness & Mechanistic Diagnostics Summary",
        "",
        "| Run | Model | Params (M) | Clean | Photo Mean | Photo Ret% | Contrast 0.12 | Fog | HighPass | FFT Phase | Wedge/Dot | Subspace |cos| | EffRank |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, r in results.items():
        c = r["corruptions"]
        d = r["diagnostics_fp32"]
        w_ratio = d.get("magnitude/wedge_to_dot_ratio", 0.0)
        scos = d.get("ortho/subspace_cos_sim_mean", d.get("ortho/cos_sim_mean", 0.0))
        erank = d.get("spectrum/effective_rank", 0.0)
        lines.append(
            f"| `{name}` | `{r['model_size']}` | {r['params_m']:.3f} | "
            f"**{r['clean_acc1']:.2f}%** | {r['photometric_mean_acc1']:.2f}% | "
            f"{r['photometric_retention_pct']:.1f}% | {c['contrast_0.12']:.2f}% | "
            f"{c['fog_veil']:.2f}% | {c['highpass_edges']:.2f}% | {c['fft_phase_only']:.2f}% | "
            f"{w_ratio:.3f} | {scos:.3f} | {erank:.3f} |"
        )
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
