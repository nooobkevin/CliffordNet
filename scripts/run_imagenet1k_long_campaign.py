#!/usr/bin/env python3
"""
Multi-wave ImageNet-1k 8x H800 campaign runner (4 simultaneous 2xH800 DDP jobs per wave,
streaming live to SwanLab, followed by automatic clean + 12-corruption validation):

  Wave B (25 ep): Mechanistic decomposition of v2 dead-zone fix + LN2d + v2 Inner-wide control
    - in1k_v2_ablate_abs_only_25ep    (3.944M, 0.737G)
    - in1k_v2_ablate_ctxbn_diff_25ep  (3.957M, 0.737G)
    - in1k_v2_atto_ln2d_25ep          (3.573M, 0.491G, 100% LayerNorm2d)
    - in1k_v2_atto_inner_wide_25ep    (3.594M, 0.493G, dead-zone-fixed Inner-wide)

  Wave C (100 ep): Full-convergence Atto showdown (testing whether v2 beats ConvNeXt & Inner-wide)
    - in1k_v2_atto_k4_100ep           (3.573M, 0.491G)
    - in1k_convnext_atto_100ep        (3.696M, 0.547G)
    - in1k_v2_rung2_deadzone_100ep    (3.957M, 0.737G)
    - in1k_v2_atto_inner_wide_100ep   (3.594M, 0.493G)

  Wave D (100 ep or --epochs): Scaling-up study at Pico (~9M) & Nano (~15.6M)
    - in1k_v2_pico_k4_100ep           (8.499M, 1.319G)
    - in1k_convnext_pico_100ep        (9.046M, 1.367G)
    - in1k_v2_nano_k4_100ep           (14.689M, 2.392G)
    - in1k_convnext_nano_100ep        (15.594M, 2.446G)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets import load_dataset
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from cliffordnet.data_cifar import apply_photometric_or_geometric_corruption
from cliffordnet.tasks.imagenet1k import CliffordNetLightning, HFImageNetDataset
from scripts.eval_corruptions import _find_best_checkpoint


WAVES: dict[str, dict[str, Any]] = {
    "B": {
        "default_epochs": 25,
        "warmup_epochs": 3,
        "ema_decay": "0.999",
        "swanlab_group": "in1k_waveB_v2_mechanism_25ep",
        "jobs": [
            {
                "run_name": "in1k_v2_ablate_abs_only_25ep",
                "model_size": "in1k_v2_ablate_abs_only",
                "base_lr": "2.5e-3",
                "compile": "true",
                "gmacs": 0.737,
                "gpus": "0,1",
                "port": 29541,
                "stagger_sec": 0,
                "label": "Wave B1: abs-only (no ctx_bn, no proj_bias)",
            },
            {
                "run_name": "in1k_v2_ablate_ctxbn_diff_25ep",
                "model_size": "in1k_v2_ablate_ctxbn_diff",
                "base_lr": "2.5e-3",
                "compile": "true",
                "gmacs": 0.737,
                "gpus": "2,3",
                "port": 29542,
                "stagger_sec": 6,
                "label": "Wave B2: ctx_bn + diff + proj_bias (balanced discrete grad)",
            },
            {
                "run_name": "in1k_v2_atto_ln2d_25ep",
                "model_size": "in1k_v2_atto_ln2d",
                "base_lr": "3.0e-3",
                "compile": "false",
                "gmacs": 0.491,
                "gpus": "4,5",
                "port": 29543,
                "stagger_sec": 12,
                "label": "Wave B3: Subspace-CAN-v2-Atto 100% LayerNorm2d (0.491G)",
            },
            {
                "run_name": "in1k_v2_atto_inner_wide_25ep",
                "model_size": "in1k_v2_atto_inner_wide",
                "base_lr": "3.0e-3",
                "compile": "true",
                "gmacs": 0.493,
                "gpus": "6,7",
                "port": 29544,
                "stagger_sec": 18,
                "label": "Wave B4: Subspace-CAN-v2-Atto Inner-wide control (0.493G)",
            },
        ],
    },
    "C": {
        "default_epochs": 100,
        "warmup_epochs": 5,
        "ema_decay": "0.9995",
        "swanlab_group": "in1k_waveC_100ep_convergence",
        "jobs": [
            {
                "run_name": "in1k_v2_atto_k4_100ep",
                "model_size": "in1k_v2_atto_k4",
                "base_lr": "3.0e-3",
                "compile": "true",
                "gmacs": 0.491,
                "gpus": "0,1",
                "port": 29551,
                "stagger_sec": 0,
                "label": "Wave C1: Subspace-CAN-v2-Atto (3.57M, 0.491G, 100ep)",
            },
            {
                "run_name": "in1k_convnext_atto_100ep",
                "model_size": "baseline_convnext_atto",
                "base_lr": "4.0e-3",
                "compile": "true",
                "gmacs": 0.547,
                "gpus": "2,3",
                "port": 29552,
                "stagger_sec": 6,
                "label": "Wave C2: ConvNeXt-Atto baseline (3.70M, 0.547G, 100ep)",
            },
            {
                "run_name": "in1k_v2_rung2_deadzone_100ep",
                "model_size": "in1k_v2_rung2_deadzone",
                "base_lr": "2.5e-3",
                "compile": "true",
                "gmacs": 0.737,
                "gpus": "4,5",
                "port": 29553,
                "stagger_sec": 12,
                "label": "Wave C3: Subspace-CAN-v2-Rung2 (3.96M, 0.737G, 100ep)",
            },
            {
                "run_name": "in1k_v2_atto_inner_wide_100ep",
                "model_size": "in1k_v2_atto_inner_wide",
                "base_lr": "3.0e-3",
                "compile": "true",
                "gmacs": 0.493,
                "gpus": "6,7",
                "port": 29554,
                "stagger_sec": 18,
                "label": "Wave C4: Subspace-CAN-v2-Atto Inner-wide (3.59M, 0.493G, 100ep)",
            },
        ],
    },
    "D": {
        "default_epochs": 100,
        "warmup_epochs": 5,
        "ema_decay": "0.9995",
        "swanlab_group": "in1k_waveD_100ep_scaling_pico_nano",
        "jobs": [
            {
                "run_name": "in1k_v2_pico_k4_100ep",
                "model_size": "in1k_v2_pico_k4",
                "base_lr": "2.5e-3",
                "compile": "true",
                "gmacs": 1.319,
                "gpus": "0,1",
                "port": 29561,
                "stagger_sec": 0,
                "label": "Wave D1: Subspace-CAN-v2-Pico (8.50M, 1.319G)",
            },
            {
                "run_name": "in1k_convnext_pico_100ep",
                "model_size": "baseline_convnext_pico",
                "base_lr": "4.0e-3",
                "compile": "true",
                "gmacs": 1.367,
                "gpus": "2,3",
                "port": 29562,
                "stagger_sec": 6,
                "label": "Wave D2: ConvNeXt-Pico baseline (9.05M, 1.367G)",
            },
            {
                "run_name": "in1k_v2_nano_k4_100ep",
                "model_size": "in1k_v2_nano_k4",
                "base_lr": "2.0e-3",
                "compile": "true",
                "gmacs": 2.392,
                "gpus": "4,5",
                "port": 29563,
                "stagger_sec": 12,
                "label": "Wave D3: Subspace-CAN-v2-Nano (14.69M, 2.392G)",
            },
            {
                "run_name": "in1k_convnext_nano_100ep",
                "model_size": "baseline_convnext_nano",
                "base_lr": "4.0e-3",
                "compile": "true",
                "gmacs": 2.446,
                "gpus": "6,7",
                "port": 29564,
                "stagger_sec": 18,
                "label": "Wave D4: ConvNeXt-Nano baseline (15.59M, 2.446G)",
            },
        ],
    },
}


def _run_ddp_job(
    job: dict[str, Any],
    wave_cfg: dict[str, Any],
    repo_root: Path,
    output_dir: Path,
    data_dir: str,
    epochs: int,
    batch_size_per_gpu: int,
) -> dict[str, Any]:
    if job["stagger_sec"] > 0:
        time.sleep(job["stagger_sec"])

    name = job["run_name"]
    if epochs != wave_cfg["default_epochs"] and name.endswith(f"_{wave_cfg['default_epochs']}ep"):
        name = name[: -len(f"_{wave_cfg['default_epochs']}ep")] + f"_{epochs}ep"

    run_dir = output_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "stdout.log"

    global_bs = batch_size_per_gpu * 2
    cmd = [
        "uv",
        "run",
        "--no-sync",
        "torchrun",
        "--standalone",
        "--nproc_per_node=2",
        f"--master_port={job['port']}",
        "-m",
        "cliffordnet.train",
        "--config",
        "configs/imagenet1k.yaml",
        "--config",
        "configs/profiles/imagenet1k_h800_probe.yaml",
        "--set",
        f"run.name={name}",
        "--set",
        f"run.output_dir={output_dir}",
        "--set",
        f"data.data_dir={data_dir}",
        "--set",
        "data.num_workers=22",
        "--set",
        f"model.size={job['model_size']}",
        "--set",
        f"model.compile={job['compile']}",
        "--set",
        "runtime.devices=2",
        "--set",
        "runtime.num_nodes=1",
        "--set",
        "runtime.strategy=ddp",
        "--set",
        "runtime.precision=bf16-mixed",
        "--set",
        f"training.batch_size={batch_size_per_gpu}",
        "--set",
        f"training.target_global_batch_size={global_bs}",
        "--set",
        f"training.max_epochs={epochs}",
        "--set",
        "training.val_check_interval=1.0",
        "--set",
        f"recipe.warmup_epochs={wave_cfg['warmup_epochs']}",
        "--set",
        f"recipe.ema_decay={wave_cfg['ema_decay']}",
        "--set",
        f"optim.base_lr={job['base_lr']}",
        "--set",
        "optim.base_global_batch_size=1024",
        "--set",
        "model.ortho_weight=0.0",
        "--set",
        "model.diag_log_interval=500",
        "--set",
        "wandb.swanlab=true",
        "--set",
        f"wandb.group={wave_cfg['swanlab_group']}",
    ]

    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env["CUDA_VISIBLE_DEVICES"] = job["gpus"]
    env["WANDB_MODE"] = "offline"
    env["OMP_NUM_THREADS"] = "4"

    t0 = time.monotonic()
    print(f"[GPUs {job['gpus']}] START {name} ({epochs} ep)", flush=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed_sec = time.monotonic() - t0
    best_ckpt = _find_best_checkpoint(run_dir)
    last_ckpt = run_dir / "checkpoints" / "last.ckpt"
    status = "ok" if proc.returncode == 0 else f"failed({proc.returncode})"
    print(
        f"[GPUs {job['gpus']}] DONE  {name}: status={status}, time={elapsed_sec/60.0:.2f} min",
        flush=True,
    )
    return {
        "run": name,
        "label": job["label"],
        "model_size": job["model_size"],
        "gmacs": job["gmacs"],
        "gpus": job["gpus"],
        "epochs": epochs,
        "status": status,
        "wall_time_min": round(elapsed_sec / 60.0, 2),
        "best_ckpt": str(best_ckpt) if best_ckpt else None,
        "last_ckpt": str(last_ckpt) if last_ckpt.exists() else None,
    }


@torch.no_grad()
def _eval_model(
    ckpt_path: Path,
    clean_loader: torch.utils.data.DataLoader,
    corr_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    lit = CliffordNetLightning.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    lit.eval()
    net = lit.model_ema.module if getattr(lit, "model_ema", None) is not None else lit.model
    net = net.to(device).eval()

    mean = torch.tensor(IMAGENET_DEFAULT_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD, device=device).view(1, 3, 1, 1)

    correct1 = 0
    correct5 = 0
    total = 0
    diag_accum: dict[str, list[float]] = {}
    for bidx, (imgs, targets) in enumerate(clean_loader):
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        norm_imgs = (imgs - mean) / std
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = net(norm_imgs)
        if bidx < 4 and hasattr(net, "forward_diagnostics"):
            diags = net.forward_diagnostics(norm_imgs.to(torch.bfloat16))
            for k, v in diags.items():
                diag_accum.setdefault(k, []).append(float(v.item()))
        top5 = logits.topk(5, dim=1).indices
        correct1 += int((top5[:, 0] == targets).sum().item())
        correct5 += int((top5 == targets.unsqueeze(1)).any(dim=1).sum().item())
        total += int(targets.numel())

    clean_top1 = round(100.0 * correct1 / total, 2)
    clean_top5 = round(100.0 * correct5 / total, 2)

    corruptions = [
        "contrast_0.12",
        "contrast_0.25",
        "contrast_2.0",
        "brightness_+0.30",
        "brightness_-0.30",
        "gamma_2.2",
        "fog_veil",
        "gaussian_noise_0.08",
        "highpass_edges",
        "lowpass_blur",
        "fft_phase_only",
    ]
    corr_res: dict[str, float] = {}
    for cname in corruptions:
        c1 = 0
        tot = 0
        for imgs, targets in corr_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            x_corr = apply_photometric_or_geometric_corruption(imgs, cname)
            norm_imgs = (x_corr - mean) / std
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = net(norm_imgs)
            c1 += int((logits.argmax(dim=1) == targets).sum().item())
            tot += int(targets.numel())
        corr_res[cname] = round(100.0 * c1 / tot, 2)

    mean_diag: dict[str, float] = {}
    if diag_accum:
        by_suffix: dict[str, list[float]] = {}
        for k, vals in diag_accum.items():
            suffix = "/".join(k.split("/")[1:])
            by_suffix.setdefault(suffix, []).append(sum(vals) / len(vals))
        mean_diag = {k: round(sum(v) / len(v), 5) for k, v in by_suffix.items()}

    params_m = round(sum(p.numel() for p in net.parameters()) / 1e6, 3)
    return {
        "params_m": params_m,
        "clean_top1": clean_top1,
        "clean_top5": clean_top5,
        "corruptions": corr_res,
        "diagnostics": mean_diag,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ImageNet-1k multi-wave 8x H800 campaign")
    parser.add_argument("--wave", type=str, required=True, choices=list(WAVES.keys()))
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="./imagenet1k")
    parser.add_argument("--epochs", type=int, default=0, help="Override default epochs for wave")
    parser.add_argument("--batch-size-per-gpu", type=int, default=512)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    wave_cfg = WAVES[args.wave]
    epochs = args.epochs if args.epochs > 0 else wave_cfg["default_epochs"]

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_results: list[dict[str, Any]] = []
    if not args.eval_only:
        with ThreadPoolExecutor(max_workers=len(wave_cfg["jobs"])) as pool:
            futs = [
                pool.submit(
                    _run_ddp_job,
                    job,
                    wave_cfg,
                    repo_root,
                    output_dir,
                    args.data_dir,
                    epochs,
                    args.batch_size_per_gpu,
                )
                for job in wave_cfg["jobs"]
            ]
            for fut in as_completed(futs):
                run_results.append(fut.result())
    else:
        for job in wave_cfg["jobs"]:
            name = job["run_name"]
            if epochs != wave_cfg["default_epochs"] and name.endswith(
                f"_{wave_cfg['default_epochs']}ep"
            ):
                name = name[: -len(f"_{wave_cfg['default_epochs']}ep")] + f"_{epochs}ep"
            run_dir = output_dir / name
            best_ckpt = _find_best_checkpoint(run_dir)
            last_ckpt = run_dir / "checkpoints" / "last.ckpt"
            run_results.append(
                {
                    "run": name,
                    "label": job["label"],
                    "model_size": job["model_size"],
                    "gmacs": job["gmacs"],
                    "gpus": job["gpus"],
                    "epochs": epochs,
                    "status": "ok" if (best_ckpt or last_ckpt.exists()) else "missing",
                    "wall_time_min": 0.0,
                    "best_ckpt": str(best_ckpt) if best_ckpt else None,
                    "last_ckpt": str(last_ckpt) if last_ckpt.exists() else None,
                }
            )

    order = {j["model_size"]: i for i, j in enumerate(wave_cfg["jobs"])}
    run_results.sort(key=lambda r: order.get(r["model_size"], 99))

    if not args.skip_eval:
        print(f"\n=== Wave {args.wave}: Evaluating Clean 50k + 11 Corruptions (10k subset) ===", flush=True)
        device = torch.device("cuda:0")
        val_tf = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )
        hf_val = load_dataset("ILSVRC/imagenet-1k", split="validation", cache_dir=args.data_dir)
        clean_ds = HFImageNetDataset(hf_val, transform=val_tf)
        corr_ds = HFImageNetDataset(hf_val.select(range(0, 50000, 5)), transform=val_tf)
        clean_loader = torch.utils.data.DataLoader(
            clean_ds, batch_size=256, shuffle=False, num_workers=12, pin_memory=True
        )
        corr_loader = torch.utils.data.DataLoader(
            corr_ds, batch_size=256, shuffle=False, num_workers=12, pin_memory=True
        )

        for r in run_results:
            ckpt = r.get("best_ckpt") or r.get("last_ckpt")
            if not ckpt:
                continue
            print(f"Evaluating {r['run']} from {Path(ckpt).name} ...", flush=True)
            r["eval"] = _eval_model(Path(ckpt), clean_loader, corr_loader, device)
            print(
                f"  -> {r['run']}: Clean Top-1={r['eval']['clean_top1']:.2f}%, "
                f"Top-5={r['eval']['clean_top5']:.2f}%, "
                f"Contrast0.12={r['eval']['corruptions']['contrast_0.12']:.2f}%, "
                f"FFT-Phase={r['eval']['corruptions']['fft_phase_only']:.2f}%",
                flush=True,
            )

    out_json = output_dir / f"summary_wave{args.wave}.json"
    out_md = output_dir / f"summary_wave{args.wave}.md"
    out_json.write_text(json.dumps(run_results, indent=2), encoding="utf-8")

    lines = [
        f"# ImageNet-1k Wave {args.wave} ({epochs} Epochs, Global BS={args.batch_size_per_gpu * 2}, bf16-mixed)",
        "",
        "| Run | Params | GMACs | Clean Top-1 | Top-5 | Contrast 0.12 | Fog | Gauss | FFT-Phase | |W|/|D| | Cancel <1e-2 | Eff. Rank | Time (min) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in run_results:
        ev = r.get("eval")
        if not ev:
            lines.append(f"| `{r['run']}` | - | {r['gmacs']:.3f} | {r['status']} | - | - | - | - | - | - | - | - | {r['wall_time_min']:.1f} |")
            continue
        c = ev["corruptions"]
        d = ev["diagnostics"]
        w2d = f"{d.get('magnitude/wedge_to_dot_ratio', 0.0):.3f}" if d else "-"
        c1e2 = f"{100.0 * d.get('cancel/rel_diff_lt1e-2', 0.0):.2f}%" if d else "-"
        erank = f"{d.get('spectrum/effective_rank', 0.0):.3f}" if d else "-"
        lines.append(
            f"| `{r['run']}` | {ev['params_m']:.3f}M | {r['gmacs']:.3f} | "
            f"**{ev['clean_top1']:.2f}%** | {ev['clean_top5']:.2f}% | "
            f"{c['contrast_0.12']:.2f}% | {c['fog_veil']:.2f}% | {c['gaussian_noise_0.08']:.2f}% | "
            f"{c['fft_phase_only']:.2f}% | {w2d} | {c1e2} | {erank} | {r['wall_time_min']:.1f} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
