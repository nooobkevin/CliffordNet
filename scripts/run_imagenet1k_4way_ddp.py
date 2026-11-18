#!/usr/bin/env python3
"""
Launch 4 simultaneous 2xH800 DDP ImageNet-1k (1.28M train / 50k val, bf16-mixed,
global_batch_size=1024, 15 full epochs) head-to-head scaling & stability runs across GPUs 0..7:
  1. GPUs 0,1: in1k_rolling_femto            (3.946M - Shift-based Hier-CliffordNet S={1,2,4}, GGR)
  2. GPUs 2,3: in1k_subspace_femto           (3.930M - Subspace Hier-CAN k=(2,4,4,4), Lie-Trotter + Triton)
  3. GPUs 4,5: in1k_subspace_femto_inner_wide(3.95M  - Strict Iso-Param Inner-only / SwiGLU control)
  4. GPUs 6,7: baseline_convnext_atto        (3.696M - timm ConvNeXt-Atto with 4x MLP FFN baseline)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as transforms
from datasets import load_dataset
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from cliffordnet.tasks.imagenet1k import CliffordNetLightning, HFImageNetDataset


JOBS: list[dict[str, Any]] = [
    {
        "name": "in1k_rolling_femto_15ep",
        "model_size": "in1k_rolling_femto",
        "gpus": "0,1",
        "port": 29511,
        "stagger_sec": 0,
        "description": "Shift-based 4-stage Hier-CliffordNet (S={1,2,4}, GGR block, 3.95M)",
    },
    {
        "name": "in1k_subspace_femto_15ep",
        "model_size": "in1k_subspace_femto",
        "gpus": "2,3",
        "port": 29512,
        "stagger_sec": 6,
        "description": "Subspace 4-stage Hier-CAN k=(2,4,4,4) (Lie-Trotter + Triton, 3.93M)",
    },
    {
        "name": "in1k_subspace_femto_inner_wide_15ep",
        "model_size": "in1k_subspace_femto_inner_wide",
        "gpus": "4,5",
        "port": 29513,
        "stagger_sec": 12,
        "description": "Strict Iso-Param Inner-only / SwiGLU Control (no wedge, 3.95M)",
    },
    {
        "name": "in1k_baseline_convnext_atto_15ep",
        "model_size": "baseline_convnext_atto",
        "gpus": "6,7",
        "port": 29514,
        "stagger_sec": 18,
        "description": "Modern timm ConvNeXt-Atto baseline (with 4x MLP FFN, 3.70M)",
    },
]


def _find_best_ckpt_and_acc(run_dir: Path) -> tuple[Path | None, float | None]:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None, None
    best_path: Path | None = None
    best_val = -1.0
    for ckpt in ckpt_dir.rglob("*.ckpt"):
        if ckpt.name in ("last.ckpt", "preempted.ckpt"):
            continue
        stem = ckpt.stem
        if "=" in stem:
            stem = stem.split("=")[-1]
        try:
            v = float(stem.split("-")[-1])
        except ValueError:
            continue
        if v > best_val:
            best_val = v
            best_path = ckpt
    if best_path is None:
        last = ckpt_dir / "last.ckpt"
        return (last if last.exists() else None), None
    acc_pct = round(best_val * 100.0 if best_val <= 1.0 else best_val, 2)
    return best_path, acc_pct


def _run_ddp_job(
    job: dict[str, Any],
    repo_root: Path,
    output_dir: Path,
    data_dir: str,
    epochs: int,
    batch_size_per_gpu: int,
) -> dict[str, Any]:
    if job["stagger_sec"] > 0:
        time.sleep(job["stagger_sec"])

    name = f"{job['model_size']}_{epochs}ep"
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
        "data.num_workers=16",
        "--set",
        f"model.size={job['model_size']}",
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
        "recipe.warmup_epochs=2",
        "--set",
        "recipe.ema_decay=0.999",
        "--set",
        "optim.base_lr=1.0e-3",
        "--set",
        "optim.base_global_batch_size=1024",
        "--set",
        "model.ortho_weight=0.0",
        "--set",
        "model.diag_log_interval=100",
    ]

    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env["CUDA_VISIBLE_DEVICES"] = job["gpus"]
    env["WANDB_MODE"] = "offline"
    env["OMP_NUM_THREADS"] = "4"

    t0 = time.monotonic()
    print(f"[GPUs {job['gpus']}] START {name} (port {job['port']})", flush=True)
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
    best_ckpt, best_acc1 = _find_best_ckpt_and_acc(run_dir)
    status = "ok" if proc.returncode == 0 else f"failed({proc.returncode})"
    print(
        f"[GPUs {job['gpus']}] DONE  {name}: status={status}, best_acc1={best_acc1}%, "
        f"time={elapsed_sec/60.0:.2f} min",
        flush=True,
    )
    return {
        "name": name,
        "model_size": job["model_size"],
        "description": job["description"],
        "gpus": job["gpus"],
        "returncode": proc.returncode,
        "status": status,
        "wall_time_min": round(elapsed_sec / 60.0, 2),
        "best_val_acc1_pct": best_acc1,
        "best_ckpt": str(best_ckpt) if best_ckpt else None,
        "log_path": str(log_path),
    }


@torch.no_grad()
def _eval_val_and_diagnostics(
    ckpt_path: str,
    data_dir: str,
    device: torch.device,
) -> dict[str, Any]:
    lit = CliffordNetLightning.load_from_checkpoint(ckpt_path, map_location=device)
    lit.eval()
    eval_model = lit._validation_model()
    eval_model.to(device=device, memory_format=torch.channels_last)
    eval_model.eval()

    n_params = sum(p.numel() for p in lit._raw_model.parameters() if p.requires_grad)

    val_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )
    ds = load_dataset("ILSVRC/imagenet-1k", cache_dir=data_dir)
    val_ds = HFImageNetDataset(ds["validation"], transform=val_tf)
    loader = torch.utils.data.DataLoader(
        val_ds, batch_size=256, shuffle=False, num_workers=8, pin_memory=True
    )

    correct1 = 0
    correct5 = 0
    total = 0
    diag_bf16_accum: dict[str, list[float]] = {}

    for b_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = eval_model(images)
        _, pred = logits.topk(5, dim=1, largest=True, sorted=True)
        eq = pred.eq(labels.view(-1, 1).expand_as(pred))
        correct1 += eq[:, :1].sum().item()
        correct5 += eq[:, :5].sum().item()
        total += labels.numel()

        if b_idx < 4 and hasattr(eval_model, "forward_diagnostics"):
            d16 = eval_model.forward_diagnostics(images[:16].to(torch.bfloat16))
            for k, v in d16.items():
                diag_bf16_accum.setdefault(k, []).append(float(v.detach().cpu()))

    diag_summary: dict[str, float] = {}
    if diag_bf16_accum:
        mean_per_key = {k: sum(vs) / len(vs) for k, vs in diag_bf16_accum.items()}
        suffixes = sorted({"/".join(k.split("/")[1:]) for k in mean_per_key if "/" in k})
        for sfx in suffixes:
            vals = [v for k, v in mean_per_key.items() if k.endswith(sfx)]
            if vals:
                diag_summary[sfx] = round(sum(vals) / len(vals), 6)

    return {
        "params": n_params,
        "params_m": round(n_params / 1e6, 3),
        "verified_val_acc1_pct": round(100.0 * correct1 / max(1, total), 2),
        "verified_val_acc5_pct": round(100.0 * correct5 / max(1, total), 2),
        "diagnostics_bf16": diag_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 4-way 2xH800 DDP ImageNet-1k scaling benchmark")
    parser.add_argument("--data-dir", type=str, default="./imagenet1k")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/imagenet1k_scale",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size-per-gpu", type=int, default=512)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-warm HF dataset cache metadata once in parent process
    print(f"[prep] Verifying HF ImageNet-1k cache at {args.data_dir} ...", flush=True)
    _ = load_dataset("ILSVRC/imagenet-1k", cache_dir=args.data_dir)
    print("[prep] Cache ready. Launching 4x (2xH800 DDP) runs ...", flush=True)

    results_by_model: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = {
            pool.submit(
                _run_ddp_job,
                job,
                repo_root,
                output_dir,
                args.data_dir,
                args.epochs,
                args.batch_size_per_gpu,
            ): job["model_size"]
            for job in JOBS
        }
        for fut in as_completed(futs):
            r = fut.result()
            results_by_model[r["model_size"]] = r

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ordered: list[dict[str, Any]] = []
    for job in JOBS:
        r = results_by_model[job["model_size"]]
        if r["status"] == "ok" and r["best_ckpt"]:
            print(f"[verify] Evaluating {r['name']} on full 50k ImageNet-1k val set ...", flush=True)
            extra = _eval_val_and_diagnostics(r["best_ckpt"], args.data_dir, device)
            r.update(extra)
        ordered.append(r)

    json_out = output_dir.parent / "summary_imagenet1k_4way.json"
    md_out = output_dir.parent / "summary_imagenet1k_4way.md"
    json_out.write_text(json.dumps(ordered, indent=2), encoding="utf-8")

    lines = [
        f"# ImageNet-1k ({args.epochs} Epochs, Global BS={args.batch_size_per_gpu*2}, bf16-mixed) 4-Way Head-to-Head Summary",
        "",
        "| Model | Params (M) | Status | Val Top-1 | Val Top-5 | Wall Time (2xH800) | Wedge/Dot | Subspace |cos| | Cancel <1e-2 | EffRank |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in ordered:
        d = r.get("diagnostics_bf16", {})
        lines.append(
            f"| `{r['model_size']}` | {r.get('params_m', 0.0):.3f} | {r['status']} | "
            f"**{r.get('verified_val_acc1_pct', r.get('best_val_acc1_pct', 0.0))}%** | "
            f"{r.get('verified_val_acc5_pct', 0.0)}% | {r['wall_time_min']:.2f} min | "
            f"{d.get('magnitude/wedge_to_dot_ratio', 0.0):.3f} | "
            f"{d.get('ortho/subspace_cos_sim_mean', 0.0):.3f} | "
            f"{d.get('cancel/rel_diff_lt1e-2', 0.0):.3f} | "
            f"{d.get('spectrum/effective_rank', 0.0):.3f} |"
        )
    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {json_out} and {md_out}", flush=True)


if __name__ == "__main__":
    main()
