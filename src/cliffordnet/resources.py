from __future__ import annotations

import os
import socket
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class GPUInfo:
    index: int
    name: str
    total_memory_gb: float


@dataclass
class ResourceInfo:
    hostname: str
    cpu_count: int
    cuda_available: bool
    visible_devices: int
    gpus: list[GPUInfo] = field(default_factory=list)
    slurm: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hostname": self.hostname,
            "cpu_count": self.cpu_count,
            "cuda_available": self.cuda_available,
            "visible_devices": self.visible_devices,
            "gpus": [gpu.__dict__ for gpu in self.gpus],
            "slurm": self.slurm,
        }


def detect_resources() -> ResourceInfo:
    gpus = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            gpus.append(
                GPUInfo(
                    index=idx,
                    name=props.name,
                    total_memory_gb=round(props.total_memory / (1024**3), 2),
                )
            )

    slurm = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("SLURM_")
        and key
        in {
            "SLURM_JOB_ID",
            "SLURM_JOB_NAME",
            "SLURM_NODELIST",
            "SLURM_NNODES",
            "SLURM_NTASKS",
            "SLURM_NTASKS_PER_NODE",
            "SLURM_CPUS_PER_TASK",
            "SLURM_GPUS_ON_NODE",
            "SLURM_PROCID",
            "SLURM_LOCALID",
        }
    }

    return ResourceInfo(
        hostname=socket.gethostname(),
        cpu_count=os.cpu_count() or 1,
        cuda_available=torch.cuda.is_available(),
        visible_devices=_visible_cuda_count(),
        gpus=gpus,
        slurm=slurm,
    )


def resolve_devices(value: int | str) -> int:
    if isinstance(value, int):
        return max(1, value)
    if str(value).lower() != "auto":
        return max(1, int(value))
    if torch.cuda.is_available():
        return max(1, torch.cuda.device_count())
    return 1


def resolve_num_nodes(value: int | str) -> int:
    if isinstance(value, int):
        return max(1, value)
    if str(value).lower() != "auto":
        return max(1, int(value))
    return max(1, int(os.environ.get("SLURM_NNODES", "1")))


def resolve_num_workers(value: int | str, devices: int) -> int:
    if isinstance(value, int):
        return max(0, value)
    if str(value).lower() != "auto":
        return max(0, int(value))

    if "CLIFFORDNET_CPUS_PER_NODE" in os.environ and "CLIFFORDNET_PROCS_PER_NODE" in os.environ:
        cpus_per_process = max(
            1,
            int(os.environ["CLIFFORDNET_CPUS_PER_NODE"])
            // max(1, int(os.environ["CLIFFORDNET_PROCS_PER_NODE"])),
        )
    elif "SLURM_CPUS_PER_TASK" in os.environ:
        cpus_per_process = int(os.environ["SLURM_CPUS_PER_TASK"])
    else:
        cpus_per_process = max(1, (os.cpu_count() or 1) // max(1, devices))
    return max(2, min(16, cpus_per_process - 1))


def global_rank() -> int:
    return int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))


def world_size(devices: int, num_nodes: int) -> int:
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    if "SLURM_NTASKS" in os.environ:
        return int(os.environ["SLURM_NTASKS"])
    return max(1, devices * num_nodes)


def _visible_cuda_count() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or visible.strip() == "":
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    if visible.strip() in {"-1", "NoDevFiles"}:
        return 0
    return len([part for part in visible.split(",") if part.strip()])
