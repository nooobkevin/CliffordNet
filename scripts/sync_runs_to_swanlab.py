#!/usr/bin/env python3
"""Convert and upload local offline W&B run directories into SwanLab Cloud."""

from __future__ import annotations

import argparse
from pathlib import Path

from swanlab.converter import WandbLocalConverter


def sync_root(runs_root: Path, project: str, tags: list[str]) -> int:
    count = 0
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        wb_root = run_dir / "wandb"
        if not wb_root.is_dir():
            continue
        offline_dirs = sorted(
            [p for p in wb_root.iterdir() if p.is_dir() and p.name.startswith("offline-run-")],
            key=lambda p: p.stat().st_mtime,
        )
        if not offline_dirs:
            continue
        latest_wb = offline_dirs[-1]
        marker = run_dir / ".swanlab_synced"
        if marker.exists():
            print(f"[skip] already synced {run_dir.name}", flush=True)
            continue
        print(f"[sync] {run_dir.name} ({latest_wb.name}) -> SwanLab project={project}", flush=True)
        conv = WandbLocalConverter(
            project=project,
            mode="online",
            tags=tags + [run_dir.name],
            log_dir=str(run_dir / "swanlog_converted"),
        )
        conv.run(root_wandb_dir=str(wb_root), wandb_run_dir=latest_wb.name)
        marker.write_text(latest_wb.name + "\n", encoding="utf-8")
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill offline W&B runs into SwanLab")
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--tag", action="append", default=[])
    args = parser.parse_args()

    n = sync_root(args.runs_root, args.project, args.tag)
    print(f"Synced {n} runs from {args.runs_root} to SwanLab ({args.project}).", flush=True)


if __name__ == "__main__":
    main()
