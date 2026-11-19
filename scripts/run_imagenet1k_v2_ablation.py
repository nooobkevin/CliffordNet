#!/usr/bin/env python3
"""
First-Principles ImageNet-1k v2 Cumulative Ablation Ladder (25 Epochs, Global BS=1024, bf16-mixed)
across 8x H800 GPUs (4 simultaneous 2xH800 DDP jobs, streaming live to SwanLab):

  1. GPUs 0,1 (`in1k_v2_rung1_recipe_25ep`):
     Isolates Fix #4 ONLY (optimization symmetry: dpr=0.10, lr=2.5e-3) on v1 `in1k_subspace_femto_gffn`
     (3.944M, 0.737 GMACs, unnormalized ctx_dw + diff).
  2. GPUs 2,3 (`in1k_v2_rung2_deadzone_25ep`):
     Adds Fix #1 (`ctx_bn=True + ctx_mode="abs" + proj_bias=True`) on the exact same backbone & recipe
     (3.957M, 0.737 GMACs) to isolate eliminating the SiLU <-1.28 negative dead-zone trap.
  3. GPUs 4,5 (`in1k_v2_atto_k4_25ep`):
     Adds Fix #2 (7x7 SpatialDiffusion) + Fix #3 (rebalanced pyramid `(48,96,192,440)`, `k=(2,4,4,4)`):
     3.573M params (<3.696M ConvNeXt-Atto) & 0.491 GMACs (10.2% fewer FLOPs than 0.547 GMACs ConvNeXt-Atto),
     100% fused Triton kernel, lr=3.0e-3, dpr=0.10.
  4. GPUs 6,7 (`in1k_v2_atto_swiglu_k6_25ep`):
     Tests first-order hyperplane inner gate `SiLU(u)*v` + second-order Grassmann bivectors `u^v`
     with deep `k=(2,4,4,6)` (`(48,96,192,408)`, 3.621M params < 3.696M, 0.495 GMACs < 0.547 GMACs),
     lr=3.0e-3, dpr=0.10.
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


JOBS: list[dict[str, Any]] = [
    {
        "run_name": "in1k_v2_rung1_recipe_25ep",
        "model_size": "in1k_v2_rung1_recipe",
        "base_lr": "2.5e-3",
        "gmacs": 0.737,
        "gpus": "0,1",
        "port": 29531,
        "stagger_sec": 0,
        "label": "Rung 1 (+Fix4 Fair Recipe only: v1 + dpr=0.10, lr=2.5e-3)",
    },
    {
        "run_name": "in1k_v2_rung2_deadzone_25ep",
        "model_size": "in1k_v2_rung2_deadzone",
        "base_lr": "2.5e-3",
        "gmacs": 0.737,
        "gpus": "2,3",
        "port": 29532,
        "stagger_sec": 6,
        "label": "Rung 2 (+Fix1 Dead-Zone Fix: ctx_bn + abs + proj_bias)",
    },
    {
        "run_name": "in1k_v2_atto_k4_25ep",
        "model_size": "in1k_v2_atto_k4",
        "base_lr": "3.0e-3",
        "gmacs": 0.491,
        "gpus": "4,5",
        "port": 29533,
        "stagger_sec": 12,
        "label": "Rung 3 Full Subspace-CAN-v2 (dw7 + (48,96,192,440) k=4, 0.491G)",
    },
    {
        "run_name": "in1k_v2_atto_swiglu_k6_25ep",
        "model_size": "in1k_v2_atto_swiglu_k6",
        "base_lr": "3.0e-3",
        "gmacs": 0.495,
        "gpus": "6,7",
        "port": 29534,
        "stagger_sec": 18,
        "label": "Rung 4 Subspace-CAN-v2 SwiGLU+Wedge k=(2,4,4,6) (0.495G)",
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
        "data.num_workers=22",
        "--set",
        f"model.size={job['model_size']}",
        "--set",
        "model.compile=true",
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
        "model.diag_log_interval=500",
        "--set",
        "wandb.swanlab=true",
        "--set",
        "wandb.group=in1k_v2_first_principles",
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
    last_ckpt = run_dir / "checkpoints" / "last.ckpt"
    status = "ok" if proc.returncode == 0 else f"failed({proc.returncode})"
    print(f"[GPUs {job['gpus']}] DONE  {name}: status={status}, time={elapsed_sec/60.0:.2f} min", flush=True)
    return {
        "run": name,
        "label": job["label"],
        "model_size": job["model_size"],
        "gmacs": job["gmacs"],
        "gpus": job["gpus"],
        "status": status,
        "wall_time_min": round(elapsed_sec / 60.0, 2),
        "best_ckpt": str(best_ckpt) if best_ckpt else None,
        "last_ckpt": str(last_ckpt) if last_ckpt.exists() else None,
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
def _eval_50k_corruptions_and_deadzone(
    ckpt_path: str,
    last_ckpt_path: str | None,
    data_dir: str,
    device: torch.device,
) -> dict[str, Any]:
    lit = CliffordNetLightning.load_from_checkpoint(ckpt_path, map_location=device)
    lit.eval()
    ema_model = lit._validation_model().to(device=device, memory_format=torch.channels_last).eval()
    n_params = sum(p.numel() for p in lit._raw_model.parameters() if p.requires_grad)

    raw_model = None
    if last_ckpt_path:
        lit_last = CliffordNetLightning.load_from_checkpoint(last_ckpt_path, map_location=device)
        raw_model = lit_last._raw_model.to(device=device, memory_format=torch.channels_last).eval()

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

    c1_ema = c5_ema = c1_raw = c5_raw = tot = 0
    d16_acc: dict[str, list[float]] = {}
    stage3_dead_pcts: list[float] = []

    for b_idx, (imgs01, y) in enumerate(loader):
        imgs01 = imgs01.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x = ((imgs01 - mean) / std).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits_ema = ema_model(x)
            if raw_model is not None:
                logits_raw = raw_model(x)
        _, pred_e = logits_ema.topk(5, dim=1)
        eq_e = pred_e.eq(y.view(-1, 1).expand_as(pred_e))
        c1_ema += eq_e[:, :1].sum().item()
        c5_ema += eq_e[:, :5].sum().item()
        if raw_model is not None:
            _, pred_r = logits_raw.topk(5, dim=1)
            eq_r = pred_r.eq(y.view(-1, 1).expand_as(pred_r))
            c1_raw += eq_r[:, :1].sum().item()
            c5_raw += eq_r[:, :5].sum().item()
        tot += y.numel()

        if b_idx < 4 and hasattr(ema_model, "forward_diagnostics"):
            for k, v in ema_model.forward_diagnostics(x[:16].to(torch.bfloat16)).items():
                d16_acc.setdefault(k, []).append(float(v.detach().cpu()))
            # Measure Stage 3 SiLU negative dead-zone percentage directly
            if hasattr(ema_model, "stages"):
                h = ema_model.stem_proj(ema_model.patch_embed(x[:32].float()))
                for s_i, st in enumerate(ema_model.stages):
                    for blk in st:
                        if s_i == len(ema_model.stages) - 1:
                            x_sp = blk.spatial_diffusion(h)
                            inter = blk.interaction
                            z1, z2 = inter._prepare_streams(x_sp)
                            v = (z2 - z1) if inter.ctx_mode == "diff" else z2
                            gate_arg = z1 if inter.cli_mode == "swiglu_wedge" else (z1 * v)
                            stage3_dead_pcts.append(float((gate_arg < -1.28).float().mean().cpu()) * 100.0)
                        h = blk(h)
                    if s_i < len(ema_model.downsamples):
                        h = ema_model.downsamples[s_i](h)

    eval_net = ema_model
    if raw_model is not None and c1_raw > c1_ema:
        eval_net = raw_model

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
                logits = eval_net(xn)
            cc1 += (logits.argmax(dim=1) == y).sum().item()
            ctot += y.numel()
        corrs[cname] = round(100.0 * cc1 / ctot, 2)

    ema_acc1 = round(100.0 * c1_ema / tot, 2)
    ema_acc5 = round(100.0 * c5_ema / tot, 2)
    raw_acc1 = round(100.0 * c1_raw / tot, 2) if raw_model is not None else ema_acc1
    raw_acc5 = round(100.0 * c5_raw / tot, 2) if raw_model is not None else ema_acc5

    return {
        "params": n_params,
        "params_m": round(n_params / 1e6, 3),
        "val_acc1_ema": ema_acc1,
        "val_acc5_ema": ema_acc5,
        "val_acc1_raw": raw_acc1,
        "val_acc5_raw": raw_acc5,
        "val_acc1": max(ema_acc1, raw_acc1),
        "val_acc5": max(ema_acc5, raw_acc5),
        "stage3_silu_deadzone_pct": round(sum(stage3_dead_pcts) / max(1, len(stage3_dead_pcts)), 2),
        "corruptions_10k": corrs,
        "diag_bf16": _agg(d16_acc),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 25-epoch ImageNet-1k v2 first-principles study")
    parser.add_argument("--data-dir", type=str, default="./imagenet1k")
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/imagenet1k_v2"))
    parser.add_argument("--summary-prefix", type=Path, default=Path("./outputs/summary_imagenet1k_v2"))
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
            r.update(
                _eval_50k_corruptions_and_deadzone(
                    r["best_ckpt"], r["last_ckpt"], args.data_dir, device
                )
            )
        ordered.append(r)

    json_out = Path(f"{args.summary_prefix}.json")
    md_out = Path(f"{args.summary_prefix}.md")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(ordered, indent=2), encoding="utf-8")

    lines = [
        f"# ImageNet-1k ({args.epochs} Epochs, Global BS=1024, bf16-mixed) First-Principles v2 Ablation Ladder",
        "",
        "| Run | Model & Cumulative Fix | Params | GMACs | EMA Top-1 | RAW Top-1 | Best Top-5 | S3 Dead% | |W|/|D| | Contrast 0.12 | FFT Phase |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in ordered:
        c = r.get("corruptions_10k", {})
        d = r.get("diag_bf16", {})
        lines.append(
            f"| `{r['run']}` | {r['label']} | {r.get('params_m', 0.0):.3f}M | {r['gmacs']:.3f}G | "
            f"**{r.get('val_acc1_ema', 0.0):.2f}%** | **{r.get('val_acc1_raw', 0.0):.2f}%** | "
            f"{r.get('val_acc5', 0.0):.2f}% | {r.get('stage3_silu_deadzone_pct', 0.0):.1f}% | "
            f"{d.get('magnitude/wedge_to_dot_ratio', 0.0):.2f}x | "
            f"{c.get('contrast_0.12', 0.0):.2f}% | {c.get('fft_phase_only', 0.0):.2f}% |"
        )
    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
