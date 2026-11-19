#!/usr/bin/env python3
"""
Launch 4 simultaneous 2xH800 DDP 25-epoch ImageNet-1k (bf16-mixed, global_bs=1024)
medium-horizon & architectural improvement runs with live SwanLab streaming:
  1. GPUs 0,1: in1k_subspace_femto_25ep       (3.930M - Subspace-CAN baseline, lr=1e-3)
  2. GPUs 2,3: baseline_convnext_atto_lr4e3_25ep (3.696M - Tuned ConvNeXt-Atto 4xFFN, lr=4e-3)
  3. GPUs 4,5: in1k_subspace_femto_gffn_25ep  (3.943M - Subspace-CAN + Global Mean-Field gFFN-G, lr=1e-3)
  4. GPUs 6,7: in1k_subspace_femto_ln2d_25ep  (3.930M - Subspace-CAN with per-token LayerNorm2d, lr=1e-3)
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
import torchvision.transforms as transforms
from datasets import load_dataset
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from cliffordnet.data_cifar import apply_photometric_or_geometric_corruption
from cliffordnet.tasks.imagenet1k import CliffordNetLightning, HFImageNetDataset
from scripts.eval_corruptions import _find_best_checkpoint


JOBS: list[dict[str, Any]] = [
    {
        "run_name": "in1k_subspace_femto_25ep",
        "model_size": "in1k_subspace_femto",
        "base_lr": "1.0e-3",
        "compile": "true",
        "gpus": "0,1",
        "port": 29521,
        "stagger_sec": 0,
        "label": "Subspace-CAN Full (BN, Local, lr=1e-3)",
    },
    {
        "run_name": "baseline_convnext_atto_lr4e3_25ep",
        "model_size": "baseline_convnext_atto",
        "base_lr": "4.0e-3",
        "compile": "true",
        "gpus": "2,3",
        "port": 29522,
        "stagger_sec": 6,
        "label": "ConvNeXt-Atto (4x MLP FFN, tuned lr=4e-3)",
    },
    {
        "run_name": "in1k_subspace_femto_gffn_25ep",
        "model_size": "in1k_subspace_femto_gffn",
        "base_lr": "1.0e-3",
        "compile": "true",
        "gpus": "4,5",
        "port": 29523,
        "stagger_sec": 12,
        "label": "Subspace-CAN + Global Context (+gFFN-G, lr=1e-3)",
    },
    {
        "run_name": "in1k_subspace_femto_ln2d_25ep",
        "model_size": "in1k_subspace_femto_ln2d",
        "base_lr": "1.0e-3",
        "compile": "false",
        "gpus": "6,7",
        "port": 29524,
        "stagger_sec": 18,
        "label": "Subspace-CAN LayerNorm2d (no BN running stats, lr=1e-3)",
    },
]


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

    name = job["run_name"]
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
        "data.num_workers=14",
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
        "recipe.warmup_epochs=3",
        "--set",
        "recipe.ema_decay=0.999",
        "--set",
        f"optim.base_lr={job['base_lr']}",
        "--set",
        "optim.base_global_batch_size=1024",
        "--set",
        "model.ortho_weight=0.0",
        "--set",
        "model.diag_log_interval=100",
        "--set",
        "wandb.swanlab=true",
        "--set",
        "wandb.group=in1k_25ep_horizon",
    ]

    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env["CUDA_VISIBLE_DEVICES"] = job["gpus"]
    env["WANDB_MODE"] = "offline"
    env["OMP_NUM_THREADS"] = "4"

    t0 = time.monotonic()
    print(f"[GPUs {job['gpus']}] START {name}", flush=True)
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
    status = "ok" if proc.returncode == 0 else f"failed({proc.returncode})"
    print(f"[GPUs {job['gpus']}] DONE  {name}: status={status}, time={elapsed_sec/60.0:.2f} min", flush=True)
    return {
        "run": name,
        "label": job["label"],
        "model_size": job["model_size"],
        "gpus": job["gpus"],
        "status": status,
        "wall_time_min": round(elapsed_sec / 60.0, 2),
        "best_ckpt": str(best_ckpt) if best_ckpt else None,
    }


def _agg(d_acc: dict[str, list[float]]) -> dict[str, float]:
    if not d_acc:
        return {}
    mp = {k: sum(vs) / len(vs) for k, vs in d_acc.items()}
    sfxs = sorted({"/".join(k.split("/")[1:]) for k in mp if "/" in k})
    return {
        s: round(
            sum(v for k, v in mp.items() if k.endswith(s))
            / len([1 for k in mp if k.endswith(s)]),
            6,
        )
        for s in sfxs
    }


@torch.no_grad()
def _eval_50k_and_corruptions(ckpt_path: str, data_dir: str, device: torch.device) -> dict[str, Any]:
    lit = CliffordNetLightning.load_from_checkpoint(ckpt_path, map_location=device)
    lit.eval()
    model = lit._validation_model().to(device=device, memory_format=torch.channels_last)
    model.eval()
    n_params = sum(p.numel() for p in lit._raw_model.parameters() if p.requires_grad)

    ds = load_dataset("ILSVRC/imagenet-1k", cache_dir=data_dir)
    raw_tf = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
    )
    val_raw = HFImageNetDataset(ds["validation"], transform=raw_tf)
    loader = torch.utils.data.DataLoader(
        val_raw, batch_size=256, shuffle=False, num_workers=12, pin_memory=True
    )

    mean = torch.tensor(IMAGENET_DEFAULT_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD, device=device).view(1, 3, 1, 1)

    c1 = c5 = tot = 0
    d16_acc: dict[str, list[float]] = {}
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

    corrs: dict[str, float] = {}
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

    return {
        "params": n_params,
        "params_m": round(n_params / 1e6, 3),
        "val_acc1": round(100.0 * c1 / tot, 2),
        "val_acc5": round(100.0 * c5 / tot, 2),
        "corruptions_10k": corrs,
        "diag_bf16": _agg(d16_acc),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 25-epoch ImageNet-1k horizon & norm study")
    parser.add_argument("--data-dir", type=str, default="./imagenet1k")
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/imagenet1k_25ep"))
    parser.add_argument("--summary-prefix", type=Path, default=Path("./outputs/summary_imagenet1k_25ep"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size-per-gpu", type=int, default=512)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _ = load_dataset("ILSVRC/imagenet-1k", cache_dir=args.data_dir)

    by_run: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = {
            pool.submit(
                _run_ddp_job,
                job,
                repo_root,
                args.output_dir,
                args.data_dir,
                args.epochs,
                args.batch_size_per_gpu,
            ): job["run_name"]
            for job in JOBS
        }
        for fut in as_completed(futs):
            r = fut.result()
            by_run[r["run"]] = r

    device = torch.device("cuda:0")
    ordered: list[dict[str, Any]] = []
    for job in JOBS:
        r = by_run[job["run_name"]]
        if r["status"] == "ok" and r["best_ckpt"]:
            print(f"[eval] {r['run']} -> {r['best_ckpt']}", flush=True)
            r.update(_eval_50k_and_corruptions(r["best_ckpt"], args.data_dir, device))
        ordered.append(r)

    json_out = Path(f"{args.summary_prefix}.json")
    md_out = Path(f"{args.summary_prefix}.md")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(ordered, indent=2), encoding="utf-8")

    lines = [
        f"# ImageNet-1k ({args.epochs} Epochs, Global BS=1024, bf16-mixed) Medium-Horizon & Improvement Study",
        "",
        "| Run | Model & Config | Params | Val Top-1 | Val Top-5 | Contrast 0.12 | Fog | HighPass | FFT Phase | Time (2xH800) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in ordered:
        c = r.get("corruptions_10k", {})
        lines.append(
            f"| `{r['run']}` | {r['label']} | {r.get('params_m', 0.0):.3f}M | "
            f"**{r.get('val_acc1', 0.0):.2f}%** | {r.get('val_acc5', 0.0):.2f}% | "
            f"{c.get('contrast_0.12', 0.0):.2f}% | {c.get('fog_veil', 0.0):.2f}% | "
            f"{c.get('highpass_edges', 0.0):.2f}% | {c.get('fft_phase_only', 0.0):.2f}% | "
            f"{r['wall_time_min']:.2f}m |"
        )
    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
