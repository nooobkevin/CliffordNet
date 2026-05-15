from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from cliffordnet.config import Config, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch CliffordNet locally or through Slurm")
    parser.add_argument("--launcher", choices=["local", "slurm"], default="local")
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--submit", action="store_true", help="Submit generated Slurm script with sbatch")
    parser.add_argument("--dry-run", action="store_true", help="Print command or script path without running")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config_paths = args.config or ["configs/imagenet1k.yaml"]
    config = load_config(config_paths, args.set)
    if args.launcher == "local":
        _launch_local(config_paths, args.set, args.dry_run)
    else:
        script = _write_slurm_script(config, config_paths, args.set)
        print(f"[slurm] wrote {script}")
        if args.submit and not args.dry_run:
            subprocess.run(["sbatch", str(script)], check=True)


def _launch_local(config_paths: list[str], overrides: list[str], dry_run: bool) -> None:
    cmd = [sys.executable, "-m", "cliffordnet.train"]
    for path in config_paths:
        cmd.extend(["--config", path])
    for override in overrides:
        cmd.extend(["--set", override])
    print("[local] " + " ".join(shlex.quote(part) for part in cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def _write_slurm_script(config: Config, config_paths: list[str], overrides: list[str]) -> Path:
    script_dir = Path(config.slurm.script_dir).expanduser()
    script_dir.mkdir(parents=True, exist_ok=True)
    run_name = config.run.name or f"{config.run.profile}_{time.strftime('%Y%m%d_%H%M%S')}"
    script = script_dir / f"{run_name}.sbatch"

    train_args = []
    for path in config_paths:
        train_args.extend(["--config", str(Path(path).resolve())])
    for override in overrides:
        train_args.extend(["--set", override])
    train_args.extend(["--set", f"run.name={run_name}"])

    if config.slurm.use_torchrun:
        srun_line = _torchrun_srun_line(config, train_args)
    else:
        command = [*config.slurm.command_prefix.split(), "-m", "cliffordnet.train", *train_args]
        srun = ["srun", *config.slurm.srun_args, *command]
        srun_line = " ".join(shlex.quote(part) for part in srun)

    lines = ["#!/bin/bash", "set -euo pipefail"]
    lines.extend(_sbatch_directives(config, run_name))
    lines.append("")
    lines.append("cd " + shlex.quote(os.getcwd()))
    for key, value in config.slurm.env.items():
        lines.append(f"export {key}={shlex.quote(str(value))}")
    lines.append(srun_line)
    lines.append("")
    script.write_text("\n".join(lines), encoding="utf-8")
    return script


def _sbatch_directives(config: Config, run_name: str) -> list[str]:
    slurm = config.slurm
    job_name = slurm.job_name or run_name
    output = slurm.output or str(Path(config.run.output_dir) / "slurm" / "%x-%j.out")
    error = slurm.error or str(Path(config.run.output_dir) / "slurm" / "%x-%j.err")
    cpus_per_worker = _auto_or_value(slurm.cpus_per_task, default=8)
    gpus_per_node = _auto_or_value(slurm.gpus_per_node, default=1)
    slurm_cpus_per_task = cpus_per_worker * gpus_per_node if slurm.use_torchrun else cpus_per_worker
    gpus_per_task = slurm.gpus_per_task
    gpu_per_node_value = f"{slurm.gpu_type}:{gpus_per_node}" if slurm.gpu_type else gpus_per_node
    if slurm.gpu_type and gpus_per_task in {None, "1", 1}:
        gpus_per_task = f"{slurm.gpu_type}:1"
    ntasks_per_node = 1 if slurm.use_torchrun else gpus_per_node

    pairs = [
        ("--job-name", job_name),
        ("--nodes", slurm.nodes),
        ("--ntasks-per-node", ntasks_per_node),
        ("--cpus-per-task", slurm_cpus_per_task),
        ("--time", slurm.time),
        ("--output", output),
        ("--error", error),
        ("--account", slurm.account),
        ("--partition", slurm.partition),
        ("--qos", slurm.qos),
        ("--mem-per-gpu", slurm.mem_per_gpu),
        ("--gpus-per-node", gpu_per_node_value if slurm.use_torchrun else None),
        ("--gpus-per-task", gpus_per_task if not slurm.use_torchrun else None),
        ("--nodelist", slurm.nodelist),
        ("--exclude", slurm.exclude),
        ("--constraint", slurm.constraint),
    ]
    directives = [f"#SBATCH {key}={value}" for key, value in pairs if value not in {None, ""}]
    directives.extend(f"#SBATCH {line}" for line in slurm.extra_directives)
    return directives


def _torchrun_srun_line(config: Config, train_args: list[str]) -> str:
    gpus_per_node = _auto_or_value(config.slurm.gpus_per_node, default=1)
    cpus_per_worker = _auto_or_value(config.slurm.cpus_per_task, default=8)
    cpus_per_node = cpus_per_worker * gpus_per_node
    runtime_overrides = [
        "--set",
        f"runtime.devices={gpus_per_node}",
        "--set",
        "runtime.num_nodes=${SLURM_NNODES}",
    ]
    command = [
        *config.slurm.torchrun_prefix.split(),
        f"--nnodes=${{SLURM_NNODES}}",
        f"--nproc_per_node={gpus_per_node}",
        "--rdzv_backend=c10d",
        "--rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}",
        "--rdzv_id=${SLURM_JOB_ID}",
        "-m",
        "cliffordnet.train",
        *train_args,
        *runtime_overrides,
    ]
    parts = ["srun", *config.slurm.srun_args, *command]
    prefix = [
        f"export CLIFFORDNET_CPUS_PER_NODE=${{SLURM_CPUS_PER_TASK:-{cpus_per_node}}}",
        f"export CLIFFORDNET_PROCS_PER_NODE={gpus_per_node}",
        "MASTER_ADDR=$(scontrol show hostnames \"${SLURM_JOB_NODELIST}\" | sed -n '1p')",
        f"MASTER_PORT=${{MASTER_PORT:-{config.slurm.master_port}}}",
    ]
    return "\n".join(prefix + [" ".join(_quote_dynamic(part) for part in parts)])


def _quote_dynamic(part: str) -> str:
    if "${" in part or "$(" in part:
        return part
    return shlex.quote(part)


def _auto_or_value(value, default: int) -> int:
    if isinstance(value, str) and value.lower() == "auto":
        return default
    return int(value)


if __name__ == "__main__":
    main()
