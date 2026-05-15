from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Any

import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import WandbLogger

from cliffordnet.checkpointing import PreemptionCheckpoint, resolve_resume_checkpoint
from cliffordnet.config import Config, config_to_dict, dump_config, load_config, write_config
from cliffordnet.resources import (
    detect_resources,
    global_rank,
    local_rank,
    resolve_devices,
    resolve_num_nodes,
    resolve_num_workers,
    world_size,
)
from cliffordnet.tasks.imagenet1k import (
    CliffordNet,
    CliffordNetLightning,
    ImageNet1kDataModule,
    _model_kwargs_for_size,
    auto_find_batch_size,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Canonical CliffordNet training entrypoint")
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="YAML config file. May be passed multiple times; later files override earlier files.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override a config value, e.g. --set training.batch_size=64",
    )
    parser.add_argument("--print-config", action="store_true", help="Print merged config and exit")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config_paths = args.config or ["configs/imagenet1k.yaml"]
    config = load_config(config_paths, args.set)
    if args.print_config:
        print(dump_config(config))
        return
    train(config)


def train(config: Config) -> None:
    resources = detect_resources()
    devices = resolve_devices(config.runtime.devices)
    num_nodes = resolve_num_nodes(config.runtime.num_nodes)
    workers = resolve_num_workers(config.data.num_workers, devices)

    run_dir = _prepare_run_dir(config)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    L.seed_everything(config.run.seed)
    torch.set_float32_matmul_precision(config.runtime.matmul_precision)
    torch.backends.cudnn.benchmark = config.runtime.benchmark
    torch.backends.cudnn.deterministic = config.runtime.deterministic
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

    _resolve_auto_batch(config, run_dir)

    attempt = 0
    while True:
        computed = _compute_runtime_settings(config, devices, num_nodes)
        manifest = _build_manifest(config, resources.to_dict(), computed)
        if global_rank() == 0:
            write_config(run_dir / "run_manifest.yaml", manifest)

        resume_path = resolve_resume_checkpoint(
            config.checkpoint.resume,
            run_dir=run_dir,
            output_dir=Path(config.run.output_dir),
        )
        if resume_path:
            print(f"[resume] {resume_path}")

        try:
            _fit_once(config, run_dir, checkpoint_dir, resources.to_dict(), computed, resume_path)
            return
        except BaseException as exc:
            if not _is_oom(exc) or attempt >= config.runtime.oom_retries:
                raise
            old_batch = config.training.batch_size
            new_batch = max(config.runtime.auto_batch.min_batch_size, old_batch // 2)
            if new_batch >= old_batch:
                raise
            attempt += 1
            config.training.batch_size = new_batch
            config.checkpoint.resume = "auto"
            print(f"[oom-retry] reducing batch_size {old_batch} -> {new_batch} (attempt {attempt})")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(5)


def _fit_once(
    config: Config,
    run_dir: Path,
    checkpoint_dir: Path,
    resources: dict[str, Any],
    computed: dict[str, Any],
    resume_path: str | None,
) -> None:
    data = ImageNet1kDataModule(
        nfs_data_dir=config.data.data_dir,
        batch_size=config.training.batch_size,
        num_workers=computed["num_workers"],
    )

    model = CliffordNetLightning(
        model_size=config.model.size,
        num_classes=config.data.num_classes,
        learning_rate=computed["learning_rate"],
        weight_decay=config.optim.weight_decay,
        max_epochs=config.training.max_epochs,
        mixup_alpha=config.recipe.mixup_alpha,
        cutmix_alpha=config.recipe.cutmix_alpha,
        mixup_prob=config.recipe.mixup_prob,
        mixup_switch_prob=config.recipe.mixup_switch_prob,
        ema_decay=config.recipe.ema_decay,
        wedge_mode=config.model.wedge_mode,
        ortho_weight=config.model.ortho_weight,
        enable_diagnostics=config.model.enable_diagnostics,
        diag_log_interval=config.model.diag_log_interval,
        label_smoothing=config.recipe.label_smoothing,
        warmup_epochs=config.recipe.warmup_epochs,
        eta_min=config.recipe.eta_min,
    )

    callbacks = [
        _checkpoint_callback(config, checkpoint_dir),
        LearningRateMonitor("step"),
        RichProgressBar(),
    ]
    if config.runtime.preemption_checkpoint:
        callbacks.append(PreemptionCheckpoint(checkpoint_dir))

    logger = _build_logger(config, run_dir, resources, computed)
    accelerator = _resolve_accelerator(config)
    devices = computed["devices"] if accelerator == "gpu" else "auto"
    strategy = _resolve_strategy(config, computed["devices"], computed["num_nodes"])

    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        num_nodes=computed["num_nodes"],
        strategy=strategy,
        precision=config.runtime.precision,
        max_epochs=config.training.max_epochs,
        accumulate_grad_batches=computed["accumulate_grad_batches"],
        gradient_clip_val=config.training.gradient_clip_val,
        val_check_interval=config.training.val_check_interval,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config.training.log_every_n_steps,
        deterministic=config.runtime.deterministic,
        benchmark=config.runtime.benchmark,
        limit_train_batches=config.training.limit_train_batches,
        limit_val_batches=config.training.limit_val_batches,
    )

    trainer.fit(model, data, ckpt_path=resume_path)
    if trainer.is_global_zero:
        _log_manifest_artifact(config, run_dir, logger)


def _prepare_run_dir(config: Config) -> Path:
    if not config.run.name:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        config.run.name = f"{config.run.profile}_{config.model.size}_{timestamp}"
    run_dir = Path(config.run.output_dir).expanduser() / config.run.name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _resolve_auto_batch(config: Config, run_dir: Path) -> None:
    if config.training.batch_size > 0 or not config.runtime.auto_batch.enabled:
        return

    sync_file = run_dir / ".auto_batch_size"
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if env_world_size > 1:
        rank = global_rank()
        candidate_file = run_dir / f".auto_batch_size.rank{rank}"
        device = torch.device(
            f"cuda:{local_rank()}" if torch.cuda.is_available() else "cpu"
        )
        detected = auto_find_batch_size(
            model_cls=CliffordNet,
            model_kwargs=_model_kwargs_for_size(config.model.size),
            max_batch_size=config.runtime.auto_batch.max_batch_size,
            min_batch_size=config.runtime.auto_batch.min_batch_size,
            device=device,
        )
        candidate_file.write_text(str(detected), encoding="utf-8")
        print(f"[autotune] rank {rank} candidate batch_size={detected}")

        if rank == 0:
            candidates = []
            for candidate_rank in range(env_world_size):
                path = run_dir / f".auto_batch_size.rank{candidate_rank}"
                for _ in range(900):
                    if path.exists():
                        break
                    time.sleep(1)
                if not path.exists():
                    raise TimeoutError(f"Timed out waiting for {path}")
                candidates.append(int(path.read_text(encoding="utf-8").strip()))
            detected = min(candidates)
            sync_file.write_text(str(detected), encoding="utf-8")
            print(f"[autotune] selected min batch_size={detected} from {candidates}")
        else:
            for _ in range(900):
                if sync_file.exists():
                    break
                time.sleep(1)
            if not sync_file.exists():
                raise TimeoutError(f"Timed out waiting for {sync_file}")
            detected = int(sync_file.read_text(encoding="utf-8").strip())
            print(f"[autotune] rank {rank} received batch_size={detected}")
    elif global_rank() == 0:
        if sync_file.exists():
            sync_file.unlink()
        device = torch.device(
            f"cuda:{local_rank()}" if torch.cuda.is_available() else "cpu"
        )
        detected = auto_find_batch_size(
            model_cls=CliffordNet,
            model_kwargs=_model_kwargs_for_size(config.model.size),
            max_batch_size=config.runtime.auto_batch.max_batch_size,
            min_batch_size=config.runtime.auto_batch.min_batch_size,
            device=device,
        )
        sync_file.write_text(str(detected), encoding="utf-8")
        print(f"[autotune] batch_size={detected}")
    else:
        for _ in range(900):
            if sync_file.exists():
                break
            time.sleep(1)
        if not sync_file.exists():
            raise TimeoutError(f"Timed out waiting for {sync_file}")
        detected = int(sync_file.read_text(encoding="utf-8").strip())
        print(f"[autotune] rank {global_rank()} received batch_size={detected}")
    config.training.batch_size = int(detected)


def _compute_runtime_settings(config: Config, devices: int, num_nodes: int) -> dict[str, Any]:
    workers = resolve_num_workers(config.data.num_workers, devices)
    size = world_size(devices, num_nodes)
    micro_global = max(1, config.training.batch_size * size)
    accumulate = max(1, config.training.accumulate_grad_batches)
    if config.training.target_global_batch_size:
        accumulate = max(
            accumulate,
            math.ceil(config.training.target_global_batch_size / micro_global),
        )
    effective_global = micro_global * accumulate
    if config.optim.lr is not None:
        learning_rate = config.optim.lr
    elif config.optim.scale_lr:
        learning_rate = config.optim.base_lr * (
            effective_global / config.optim.base_global_batch_size
        )
    else:
        learning_rate = config.optim.base_lr

    return {
        "devices": devices,
        "num_nodes": num_nodes,
        "world_size": size,
        "num_workers": workers,
        "micro_global_batch_size": micro_global,
        "accumulate_grad_batches": accumulate,
        "effective_global_batch_size": effective_global,
        "learning_rate": learning_rate,
    }


def _checkpoint_callback(config: Config, checkpoint_dir: Path) -> ModelCheckpoint:
    kwargs = {
        "dirpath": checkpoint_dir,
        "monitor": config.checkpoint.monitor,
        "mode": config.checkpoint.mode,
        "save_top_k": config.checkpoint.save_top_k,
        "save_last": config.checkpoint.save_last,
        "filename": config.checkpoint.filename,
    }
    if config.checkpoint.every_n_train_steps is not None:
        kwargs["every_n_train_steps"] = config.checkpoint.every_n_train_steps
    return ModelCheckpoint(**kwargs)


def _build_logger(
    config: Config,
    run_dir: Path,
    resources: dict[str, Any],
    computed: dict[str, Any],
):
    if not config.wandb.enabled or config.wandb.mode == "disabled":
        return False
    tags = sorted(set(config.run.tags + config.wandb.tags + [config.run.profile]))
    return WandbLogger(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=config.run.name,
        group=config.wandb.group,
        save_dir=str(run_dir),
        offline=config.wandb.mode == "offline",
        log_model=config.wandb.log_model,
        tags=tags,
        config={
            "config": config_to_dict(config),
            "resources": resources,
            "computed": computed,
        },
    )


def _log_manifest_artifact(config: Config, run_dir: Path, logger) -> None:
    if not config.wandb.enabled or not config.wandb.save_manifest_artifact or logger is False:
        return
    try:
        artifact = wandb.Artifact(f"{config.run.name}-manifest", type="run-manifest")
        artifact.add_file(str(run_dir / "run_manifest.yaml"))
        logger.experiment.log_artifact(artifact)
    except Exception as exc:  # W&B artifact upload must not fail training.
        print(f"[wandb] failed to log manifest artifact: {exc}")


def _resolve_accelerator(config: Config) -> str:
    if config.runtime.accelerator != "auto":
        return config.runtime.accelerator
    return "gpu" if torch.cuda.is_available() else "cpu"


def _resolve_strategy(config: Config, devices: int, num_nodes: int) -> str:
    if config.runtime.strategy != "auto":
        return config.runtime.strategy
    return "ddp" if devices > 1 or num_nodes > 1 else "auto"


def _build_manifest(
    config: Config, resources: dict[str, Any], computed: dict[str, Any]
) -> dict[str, Any]:
    return {
        "config": config_to_dict(config),
        "resources": resources,
        "computed": computed,
        "env": {
            "git_commit": _git_value(["rev-parse", "HEAD"]),
            "git_dirty": bool(_git_value(["status", "--porcelain"])),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "wandb_mode": config.wandb.mode,
        },
    }


def _git_value(args: list[str]) -> str | None:
    import subprocess

    try:
        result = subprocess.run(
            ["git", *args],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return None
    value = result.stdout.strip()
    return value or None


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text or "cublas_status_alloc_failed" in text


if __name__ == "__main__":
    main()
