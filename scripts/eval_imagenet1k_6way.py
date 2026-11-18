#!/usr/bin/env python3
"""
Unified ImageNet-1k 50k validation, bf16/fp32 geometric diagnostics, and
photometric/edge/FFT-phase corruption evaluator across trained 10-epoch runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torchvision.transforms as transforms
from datasets import load_dataset
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from cliffordnet.data_cifar import apply_photometric_or_geometric_corruption
from cliffordnet.tasks.imagenet1k import CliffordNetLightning, HFImageNetDataset
from scripts.eval_corruptions import _find_best_checkpoint


RUNS = [
    ("in1k_subspace_femto_10ep", "Subspace-CAN Full (Inner+Wedge, lr=1e-3)"),
    ("in1k_subspace_femto_sym_10ep", "Subspace-CAN Symmetric Control (u_i v_j + u_j v_i, lr=1e-3)"),
    ("in1k_subspace_femto_inner_wide_10ep", "Subspace-CAN Iso-Param Inner/SwiGLU Control (lr=1e-3)"),
    ("in1k_rolling_femto_10ep_eager", "Shift-Rolling Hier-CliffordNet (S={1,2,4}, GGR, lr=1e-3)"),
    ("baseline_convnext_atto_10ep", "ConvNeXt-Atto (4x MLP FFN, lr=1e-3)"),
    ("baseline_convnext_atto_lr4e3_10ep", "ConvNeXt-Atto (4x MLP FFN, canonical lr=4e-3)"),
]


def _agg(d_acc: dict[str, list[float]]) -> dict[str, float]:
    if not d_acc:
        return {}
    mp = {k: sum(vs) / len(vs) for k, vs in d_acc.items()}
    sfxs = sorted({"/".join(k.split("/")[1:]) for k in mp if "/" in k})
    out: dict[str, float] = {}
    for s in sfxs:
        vals = [v for k, v in mp.items() if k.endswith(s)]
        if vals:
            out[s] = round(sum(vals) / len(vals), 6)
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate 6-way ImageNet-1k checkpoints on 50k val & corruptions")
    parser.add_argument("--runs-root", type=Path, default=Path("./outputs/imagenet1k_scale"))
    parser.add_argument("--data-dir", type=str, default="./imagenet1k")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("./outputs/summary_imagenet1k_6way_complete"),
    )
    args = parser.parse_args()

    root = args.runs_root
    device = torch.device("cuda:0")
    ds = load_dataset("ILSVRC/imagenet-1k", cache_dir=args.data_dir)
    raw_tf = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
    )
    val_raw = HFImageNetDataset(ds["validation"], transform=raw_tf)
    loader = torch.utils.data.DataLoader(
        val_raw, batch_size=256, shuffle=False, num_workers=12, pin_memory=True
    )

    mean = torch.tensor(IMAGENET_DEFAULT_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD, device=device).view(1, 3, 1, 1)

    all_results: list[dict[str, Any]] = []
    for folder, label in RUNS:
        run_dir = root / folder
        ckpt = _find_best_checkpoint(run_dir)
        assert ckpt is not None, f"Missing ckpt for {folder}"
        lit = CliffordNetLightning.load_from_checkpoint(str(ckpt), map_location=device)
        lit.eval()
        model = lit._validation_model().to(device=device, memory_format=torch.channels_last)
        model.eval()
        n_params = sum(p.numel() for p in lit._raw_model.parameters() if p.requires_grad)

        c1 = c5 = tot = 0
        d16_acc: dict[str, list[float]] = {}
        d32_acc: dict[str, list[float]] = {}
        with torch.no_grad():
            for b_idx, (imgs01, y) in enumerate(loader):
                imgs01 = imgs01.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                x = ((imgs01 - mean) / std).contiguous(memory_format=torch.channels_last)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(x)
                _, pred = logits.topk(5, dim=1)
                eq = pred.eq(y.view(-1, 1).expand_as(pred))
                c1 += eq[:, :1].sum().item()
                c5 += eq[:, :5].sum().item()
                tot += y.numel()
                if b_idx < 4 and hasattr(model, "forward_diagnostics"):
                    for k, v in model.forward_diagnostics(x[:16].to(torch.bfloat16)).items():
                        d16_acc.setdefault(k, []).append(float(v.detach().cpu()))
                    for k, v in model.forward_diagnostics(x[:16].float()).items():
                        d32_acc.setdefault(k, []).append(float(v.detach().cpu()))

        corrs: dict[str, float] = {}
        with torch.no_grad():
            for cname in ("contrast_0.12", "fog_veil", "highpass_edges", "fft_phase_only"):
                cc1 = ctot = 0
                for b_idx, (imgs01, y) in enumerate(loader):
                    if b_idx >= 40:
                        break
                    imgs01 = imgs01.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    xc = apply_photometric_or_geometric_corruption(imgs01, cname)
                    xn = ((xc - mean) / std).contiguous(memory_format=torch.channels_last)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = model(xn)
                    cc1 += (logits.argmax(dim=1) == y).sum().item()
                    ctot += y.numel()
                corrs[cname] = round(100.0 * cc1 / ctot, 2)

        params_m = round(n_params / 1e6, 3)
        acc1 = round(100.0 * c1 / tot, 2)
        acc5 = round(100.0 * c5 / tot, 2)
        rec = {
            "run": folder,
            "label": label,
            "model_size": lit.hparams.get("model_size"),
            "params": n_params,
            "params_m": params_m,
            "val_acc1": acc1,
            "val_acc5": acc5,
            "corruptions_10k": corrs,
            "diag_bf16": _agg(d16_acc),
            "diag_fp32": _agg(d32_acc),
        }
        all_results.append(rec)
        print(
            f"{folder:38s} | {params_m:.3f}M | Top1={acc1:5.2f}% | Top5={acc5:5.2f}% | corrs={corrs}",
            flush=True,
        )

    out_json = Path(f"{args.output_prefix}.json")
    out_md = Path(f"{args.output_prefix}.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    lines = [
        "# ImageNet-1k (10 Epochs, Global BS=1024, bf16-mixed) 6-Way Complete Benchmark",
        "",
        "| Run | Architecture & Config | Params | Val Top-1 | Val Top-5 | Contrast 0.12 | Fog | HighPass | FFT Phase | bf16 Cancel<1e-2 | bf16 LambdaErr | EffRank |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in all_results:
        c = r["corruptions_10k"]
        d = r["diag_bf16"]
        run_id = r["run"]
        lbl = r["label"]
        pm = r["params_m"]
        a1 = r["val_acc1"]
        a5 = r["val_acc5"]
        c_con = c["contrast_0.12"]
        c_fog = c["fog_veil"]
        c_hp = c["highpass_edges"]
        c_ph = c["fft_phase_only"]
        can = d.get("cancel/rel_diff_lt1e-2", 0.0)
        lam = d.get("invariance/lambda_wedge_rel_err", 0.0)
        erk = d.get("spectrum/effective_rank", 0.0)
        lines.append(
            f"| `{run_id}` | {lbl} | {pm:.3f}M | **{a1:.2f}%** | {a5:.2f}% | "
            f"{c_con:.2f}% | {c_fog:.2f}% | {c_hp:.2f}% | {c_ph:.2f}% | "
            f"{can:.4f} | {lam:.4f} | {erk:.3f} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
