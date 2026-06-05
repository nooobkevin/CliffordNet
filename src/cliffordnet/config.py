from __future__ import annotations

import copy
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RunConfig:
    profile: str = "imagenet1k"
    name: str | None = None
    output_dir: str = "./outputs"
    seed: int = 42
    tags: list[str] = field(default_factory=list)
    notes: str | None = None


@dataclass
class DataConfig:
    dataset: str = "imagenet1k"
    dataset_id: str = "ILSVRC/imagenet-1k"
    data_dir: str = "./imagenet1k"
    num_workers: int | str = "auto"
    image_size: int = 224
    num_classes: int = 1000


@dataclass
class ModelConfig:
    size: str = "12_2"
    wedge_mode: str = "fma"
    ortho_weight: float = 0.01
    enable_diagnostics: bool = True
    diag_log_interval: int = 100


@dataclass
class RecipeConfig:
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 1.0
    mixup_prob: float = 1.0
    mixup_switch_prob: float = 0.5
    label_smoothing: float = 0.1
    ema_decay: float = 0.9999
    warmup_epochs: int = 5
    eta_min: float = 1e-6


@dataclass
class OptimConfig:
    lr: float | None = None
    base_lr: float = 5e-4
    base_global_batch_size: int = 1024
    scale_lr: bool = True
    weight_decay: float = 0.05


@dataclass
class TrainingConfig:
    batch_size: int = 0
    target_global_batch_size: int | None = 1024
    accumulate_grad_batches: int = 1
    max_epochs: int = 300
    val_check_interval: float = 0.25
    gradient_clip_val: float = 1.0
    limit_train_batches: float | int | None = None
    limit_val_batches: float | int | None = None
    log_every_n_steps: int = 10


@dataclass
class AutoBatchConfig:
    enabled: bool = True
    min_batch_size: int = 8
    max_batch_size: int = 1024
    safety_factor: float = 0.92


@dataclass
class RuntimeConfig:
    accelerator: str = "auto"
    devices: int | str = "auto"
    num_nodes: int | str = "auto"
    strategy: str = "auto"
    precision: str = "bf16-mixed"
    deterministic: bool = False
    benchmark: bool = True
    matmul_precision: str = "high"
    auto_batch: AutoBatchConfig = field(default_factory=AutoBatchConfig)
    oom_retries: int = 1
    preemption_checkpoint: bool = True


@dataclass
class CheckpointConfig:
    resume: str | None = "auto"
    monitor: str = "val/acc1"
    mode: str = "max"
    save_top_k: int = 2
    save_last: bool = True
    filename: str = "{epoch}-{val/acc1:.4f}"
    every_n_train_steps: int | None = None


@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "CliffordNet"
    entity: str | None = None
    mode: str = "online"
    group: str | None = None
    log_model: bool | str = False
    save_manifest_artifact: bool = True
    tags: list[str] = field(default_factory=list)


@dataclass
class SlurmConfig:
    account: str | None = None
    partition: str | None = None
    qos: str | None = None
    nodes: int = 1
    gpus_per_node: int | str = 1
    gpus_per_task: str | None = "1"
    gpu_type: str | None = None
    cpus_per_task: int | str = "auto"
    mem_per_gpu: str | None = None
    time: str = "04:00:00"
    job_name: str | None = None
    nodelist: str | None = None
    exclude: str | None = None
    constraint: str | None = None
    output: str | None = None
    error: str | None = None
    script_dir: str = "./outputs/slurm"
    command_prefix: str = "uv run python"
    torchrun_prefix: str = "uv run torchrun"
    use_torchrun: bool = True
    master_port: int = 29500
    srun_args: list[str] = field(default_factory=list)
    extra_directives: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    recipe: RecipeConfig = field(default_factory=RecipeConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    slurm: SlurmConfig = field(default_factory=SlurmConfig)


def config_to_dict(config: Config) -> dict[str, Any]:
    return asdict(config)


def load_config(paths: list[str] | tuple[str, ...], overrides: list[str] | None = None) -> Config:
    data = config_to_dict(Config())
    for path in paths:
        loaded = _load_yaml(path)
        _deep_update(data, loaded)
    for override in overrides or []:
        _apply_override(data, override)
    data = _expand_env(data)
    return _config_from_dict(data)


def dump_config(config: Config) -> str:
    return yaml.safe_dump(config_to_dict(config), sort_keys=False)


def write_config(path: str | Path, config: Config | dict[str, Any]) -> None:
    payload = config_to_dict(config) if isinstance(config, Config) else config
    Path(path).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return loaded


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _apply_override(data: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Override must use key=value syntax: {override}")
    key, raw_value = override.split("=", 1)
    value = raw_value if key == "model.size" else yaml.safe_load(raw_value)
    cursor = data
    parts = key.split(".")
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    return value


def _config_from_dict(data: dict[str, Any]) -> Config:
    runtime_data = data.get("runtime", {})
    runtime_data["auto_batch"] = AutoBatchConfig(**runtime_data.get("auto_batch", {}))
    return Config(
        run=RunConfig(**data.get("run", {})),
        data=DataConfig(**data.get("data", {})),
        model=ModelConfig(**data.get("model", {})),
        recipe=RecipeConfig(**data.get("recipe", {})),
        optim=OptimConfig(**data.get("optim", {})),
        training=TrainingConfig(**data.get("training", {})),
        runtime=RuntimeConfig(**runtime_data),
        checkpoint=CheckpointConfig(**data.get("checkpoint", {})),
        wandb=WandbConfig(**data.get("wandb", {})),
        slurm=SlurmConfig(**data.get("slurm", {})),
    )
