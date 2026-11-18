#!/usr/bin/env python3
"""
Parallel multi-GPU experiment scheduler for 8x H800 single-GPU runs.
Dispatches a JSON/Python list of `cliffordnet-train` jobs across available GPUs,
tracks wall-clock training duration, parses best validation accuracy, and writes
a structured JSON + Markdown summary report.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from typing import Any

import yaml


def _best_acc_from_checkpoints(run_dir: Path) -> float | None:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    vals: list[float] = []
    for ckpt in ckpt_dir.rglob("*.ckpt"):
        if ckpt.name in ("last.ckpt", "preempted.ckpt"):
            continue
        stem = ckpt.stem
        if "=" in stem:
            stem = stem.split("=")[-1]
        try:
            vals.append(float(stem.split("-")[-1]))
        except ValueError:
            continue
    if not vals:
        return None
    best = max(vals)
    # Lightning logs acc1 as fraction in [0, 1] -> convert to percentage if <= 1.0
    return round(best * 100.0 if best <= 1.0 else best, 2)


def _run_one_job(
    job: dict[str, Any],
    gpu_queue: Queue[int],
    repo_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    gpu_id = gpu_queue.get()
    name = job["name"]
    run_dir = output_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "stdout.log"

    cmd = ["uv", "run", "--no-sync", "cliffordnet-train"]
    for cfg in job.get("configs", ["configs/cifar100.yaml"]):
        cmd.extend(["--config", cfg])
    cmd.extend(["--set", f"run.name={name}", "--set", f"run.output_dir={output_dir}"])
    for ov in job.get("overrides", []):
        cmd.extend(["--set", ov])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["WANDB_MODE"] = "offline"

    t0 = time.monotonic()
    print(f"[GPU {gpu_id}] START {name}: {' '.join(cmd)}", flush=True)
    try:
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
        best_acc = _best_acc_from_checkpoints(run_dir)
        manifest_path = run_dir / "run_manifest.yaml"
        manifest = yaml.safe_load(manifest_path.read_text()) if manifest_path.exists() else {}
        status = "ok" if proc.returncode == 0 else f"failed({proc.returncode})"
        print(
            f"[GPU {gpu_id}] DONE  {name}: status={status}, best_acc={best_acc}%, "
            f"time={elapsed_sec/60.0:.2f} min",
            flush=True,
        )
        return {
            "name": name,
            "description": job.get("description", ""),
            "track": job.get("track", ""),
            "gpu_id": gpu_id,
            "returncode": proc.returncode,
            "status": status,
            "wall_time_sec": round(elapsed_sec, 2),
            "wall_time_min": round(elapsed_sec / 60.0, 2),
            "best_val_acc1_pct": best_acc,
            "overrides": job.get("overrides", []),
            "model_size": manifest.get("config", {}).get("model", {}).get("size"),
            "log_path": str(log_path),
        }
    finally:
        gpu_queue.put(gpu_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run parallel single-GPU CliffordNet jobs across H800 GPUs")
    parser.add_argument("--spec", type=str, required=True, help="JSON file defining list of experiment jobs")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7", help="Comma-separated GPU IDs")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/cifar100",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Optional override for data.data_dir across all jobs",
    )
    parser.add_argument("--summary-prefix", type=str, required=True, help="Output prefix for .json and .md summary")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[dict[str, Any]] = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    if args.data_dir:
        for job in jobs:
            job.setdefault("overrides", []).append(f"data.data_dir={args.data_dir}")
    gpus = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]

    gpu_queue: Queue[int] = Queue()
    for g in gpus:
        gpu_queue.put(g)

    results_by_name: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futs = {
            pool.submit(_run_one_job, job, gpu_queue, repo_root, output_dir): job["name"]
            for job in jobs
        }
        for fut in as_completed(futs):
            res = fut.result()
            results_by_name[res["name"]] = res

    ordered = [results_by_name[j["name"]] for j in jobs]
    json_out = Path(f"{args.summary_prefix}.json")
    md_out = Path(f"{args.summary_prefix}.md")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(ordered, indent=2), encoding="utf-8")

    lines = [
        "# Parallel Experiment Summary",
        "",
        "| Run Name | Track | Model | Status | Best Val Acc@1 | Wall Time (min) | Description |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for r in ordered:
        acc_str = f"{r['best_val_acc1_pct']:.2f}%" if r["best_val_acc1_pct"] is not None else "N/A"
        lines.append(
            f"| `{r['name']}` | {r['track']} | `{r['model_size']}` | {r['status']} | "
            f"**{acc_str}** | {r['wall_time_min']:.2f} | {r['description']} |"
        )
    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote summary to {json_out} and {md_out}")


if __name__ == "__main__":
    main()
