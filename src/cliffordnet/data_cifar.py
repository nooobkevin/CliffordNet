"""CIFAR-100 DataModule and structured geometric/photometric corruption transforms."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def ensure_cifar100_cache(data_dir: str) -> Path:
    """
    Ensure a fast in-memory-ready uint8 cache (`cifar100_uint8.pt`) exists in `data_dir`,
    built from HuggingFace `uoft-cs/cifar100` (identical 50k train / 10k test images & fine_labels
    to torchvision CIFAR-100, avoiding slow cs.toronto.edu downloads and Arrow IPC overhead).
    """
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    cache_file = root / "cifar100_uint8.pt"
    if cache_file.exists():
        return cache_file

    ds = load_dataset("uoft-cs/cifar100", cache_dir=str(root))
    train_imgs = np.stack([np.asarray(img.convert("RGB"), dtype=np.uint8) for img in ds["train"]["img"]], axis=0)
    train_labels = np.asarray(ds["train"]["fine_label"], dtype=np.int64)
    test_imgs = np.stack([np.asarray(img.convert("RGB"), dtype=np.uint8) for img in ds["test"]["img"]], axis=0)
    test_labels = np.asarray(ds["test"]["fine_label"], dtype=np.int64)

    tmp_file = root / f".cifar100_uint8.{torch.randint(0, 10**9, (1,)).item()}.tmp"
    torch.save(
        {
            "train_data": torch.from_numpy(train_imgs),
            "train_labels": torch.from_numpy(train_labels),
            "test_data": torch.from_numpy(test_imgs),
            "test_labels": torch.from_numpy(test_labels),
        },
        tmp_file,
    )
    tmp_file.replace(cache_file)
    return cache_file


class FastCIFAR100Dataset(Dataset):
    """In-memory uint8 CIFAR-100 dataset identical in behavior to torchvision.datasets.CIFAR100."""

    def __init__(self, data_dir: str, train: bool = True, transform: Callable | None = None):
        cache_file = ensure_cifar100_cache(data_dir)
        payload = torch.load(cache_file, map_location="cpu", weights_only=True)
        self.data = (payload["train_data"] if train else payload["test_data"]).numpy()
        self.targets = (payload["train_labels"] if train else payload["test_labels"]).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        img = Image.fromarray(self.data[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[idx]


class CIFAR100DataModule(L.LightningDataModule):
    """
    CIFAR-100 data pipeline aligned with arXiv:2601.06793 and arXiv:2610.37226:
      - RandomCrop(32, padding=4)
      - RandomHorizontalFlip()
      - AutoAugment(CIFAR10)
      - Normalize(CIFAR100_MEAN, CIFAR100_STD)
      - RandomErasing(p=0.25)
    Supports optional resizing (e.g. image_size=224 if testing high-res stems).
    """

    def __init__(
        self,
        data_dir: str = "./data/cifar100",
        batch_size: int = 128,
        num_workers: int = 4,
        image_size: int = 32,
        random_erasing_prob: float = 0.25,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.random_erasing_prob = random_erasing_prob
        self.prepare_data_per_node = True

    def prepare_data(self) -> None:
        ensure_cifar100_cache(self.data_dir)

    def setup(self, stage: str | None = None) -> None:
        train_ops: list[Callable] = []
        val_ops: list[Callable] = []

        if self.image_size != 32:
            train_ops.append(transforms.Resize((self.image_size, self.image_size)))
            val_ops.append(transforms.Resize((self.image_size, self.image_size)))
            pad = max(4, self.image_size // 8)
            train_ops.append(transforms.RandomCrop(self.image_size, padding=pad))
        else:
            train_ops.append(transforms.RandomCrop(32, padding=4))

        train_ops.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
            ]
        )
        if self.random_erasing_prob > 0.0:
            train_ops.append(transforms.RandomErasing(p=self.random_erasing_prob))

        val_ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
            ]
        )

        train_tf = transforms.Compose(train_ops)
        val_tf = transforms.Compose(val_ops)

        self.train_ds = FastCIFAR100Dataset(self.data_dir, train=True, transform=train_tf)
        self.val_ds = FastCIFAR100Dataset(self.data_dir, train=False, transform=val_tf)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )


# ============================================================================
# Structured Photometric & Geometric Corruptions (Track 5 / ROI #5)
# ============================================================================


def _gaussian_kernel2d(kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2.0 * sigma**2))
    g = g / g.sum()
    k2d = torch.outer(g, g)
    return k2d


def _blur2d(x: torch.Tensor, kernel_size: int = 5, sigma: float = 1.2) -> torch.Tensor:
    c = x.shape[1]
    k2d = _gaussian_kernel2d(kernel_size, sigma, x.device, x.dtype)
    weight = k2d.view(1, 1, kernel_size, kernel_size).expand(c, 1, kernel_size, kernel_size)
    pad = kernel_size // 2
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(x_pad, weight, groups=c)


def apply_photometric_or_geometric_corruption(
    images_01: torch.Tensor,
    corruption: str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Apply deterministic or seeded corruption to unnormalized images in [0, 1] of shape (B, C, H, W).
    Designed to falsify/corroborate the CliffordNet hypothesis that:
      - Inner (u * v) encodes intensity/energy
      - Wedge (u ^ v) encodes contrast-invariant geometric edges/structure
    """
    x = images_01.float()
    if corruption == "clean":
        return x

    if corruption == "contrast_0.25":
        mean = x.mean(dim=(-2, -1), keepdim=True)
        return (mean + 0.25 * (x - mean)).clamp(0.0, 1.0)

    if corruption == "contrast_0.12":
        mean = x.mean(dim=(-2, -1), keepdim=True)
        return (mean + 0.12 * (x - mean)).clamp(0.0, 1.0)

    if corruption == "contrast_2.0":
        mean = x.mean(dim=(-2, -1), keepdim=True)
        return (mean + 2.0 * (x - mean)).clamp(0.0, 1.0)

    if corruption == "brightness_+0.30":
        return (x + 0.30).clamp(0.0, 1.0)

    if corruption == "brightness_-0.30":
        return (x - 0.30).clamp(0.0, 1.0)

    if corruption == "gamma_2.2":
        return x.clamp(1e-6, 1.0).pow(2.2).clamp(0.0, 1.0)

    if corruption == "fog_veil":
        # Smooth low-frequency atmospheric veil reducing local contrast
        B, C, H, W = x.shape
        yy = torch.linspace(-1.0, 1.0, H, device=x.device).view(1, 1, H, 1)
        xx = torch.linspace(-1.0, 1.0, W, device=x.device).view(1, 1, 1, W)
        veil = 0.55 + 0.20 * torch.cos(math.pi * 0.5 * (xx + yy))
        return (0.45 * x + 0.55 * veil).clamp(0.0, 1.0)

    if corruption == "gaussian_noise_0.08":
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator) * 0.08
        return (x + noise).clamp(0.0, 1.0)

    if corruption == "highpass_edges":
        # Remove low-frequency intensity/color fill, retaining structural edges re-centered at 0.5
        low = _blur2d(x, kernel_size=5, sigma=1.5)
        high = x - low
        # Normalize high-pass energy to mid-gray [0, 1]
        return (0.5 + 2.5 * high).clamp(0.0, 1.0)

    if corruption == "lowpass_blur":
        return _blur2d(x, kernel_size=5, sigma=1.5).clamp(0.0, 1.0)

    if corruption == "fft_phase_only":
        # Keep exact 2D Fourier phase (edges/geometry), flatten per-image spatial amplitude variation
        # using radially-averaged smooth 1/f-like batch mean spectrum so intensity contrast is equalized
        spec = torch.fft.rfft2(x - x.mean(dim=(-2, -1), keepdim=True))
        phase = torch.angle(spec)
        mean_amp = spec.abs().mean(dim=(0, 1), keepdim=True)
        recon = torch.fft.irfft2(mean_amp * torch.exp(1j * phase), s=x.shape[-2:])
        std = recon.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        recon = 0.5 + 0.18 * (recon / std)
        return recon.clamp(0.0, 1.0)

    raise ValueError(f"Unknown corruption: {corruption}")


CORRUPTION_SUITE = (
    "clean",
    "contrast_0.25",
    "contrast_0.12",
    "contrast_2.0",
    "brightness_+0.30",
    "brightness_-0.30",
    "gamma_2.2",
    "fog_veil",
    "gaussian_noise_0.08",
    "highpass_edges",
    "lowpass_blur",
    "fft_phase_only",
)
