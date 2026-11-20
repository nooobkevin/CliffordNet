#!/usr/bin/env python3
"""
One-time parallel builder for a zero-JPEG-decode RAM cache of ImageNet-1k in /dev/shm.

Writes:
  - <out_dir>/train_images.npy : (N_train, 3, 256, 256) uint8 (~252 GB) — Resize((256, 256), BILINEAR)
  - <out_dir>/train_labels.npy : (N_train,) int64
  - <out_dir>/val_images.npy   : (N_val, 3, 224, 224) uint8 (~7.5 GB)  — exact Resize(256, BICUBIC) + CenterCrop(224)
  - <out_dir>/val_labels.npy   : (N_val,) int64

All DDP ranks map these files read-only via np.memmap, sharing the exact same physical RAM pages
with zero duplication and zero per-epoch CPU JPEG decoding.
"""

from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from datasets import load_dataset
from PIL import Image


_G_DATA_DIR: str = ""
_G_SPLIT: str = ""
_G_IMG_PATH: str = ""
_G_LBL_PATH: str = ""
_G_IS_TRAIN: bool = True
_G_CANVAS: int = 320
_G_DS = None
_G_IMGS = None
_G_LBLS = None
_G_SHAPES = None


def _init_worker(
    data_dir: str,
    split: str,
    img_path: str,
    lbl_path: str,
    shp_path: str,
    is_train: bool,
    canvas: int,
) -> None:
    global _G_DATA_DIR, _G_SPLIT, _G_IMG_PATH, _G_LBL_PATH, _G_IS_TRAIN, _G_CANVAS
    global _G_DS, _G_IMGS, _G_LBLS, _G_SHAPES
    _G_DATA_DIR = data_dir
    _G_SPLIT = split
    _G_IMG_PATH = img_path
    _G_LBL_PATH = lbl_path
    _G_IS_TRAIN = is_train
    _G_CANVAS = canvas
    _G_DS = load_dataset("ILSVRC/imagenet-1k", split=split, cache_dir=data_dir)
    _G_IMGS = np.load(img_path, mmap_mode="r+")
    _G_LBLS = np.load(lbl_path, mmap_mode="r+")
    _G_SHAPES = np.load(shp_path, mmap_mode="r+") if shp_path else None


def _process_chunk(bounds: tuple[int, int]) -> int:
    start, end = bounds
    ds = _G_DS
    imgs_mm = _G_IMGS
    lbls_mm = _G_LBLS
    shps_mm = _G_SHAPES
    is_train = _G_IS_TRAIN
    canvas = _G_CANVAS

    for i in range(start, end):
        item = ds[i]
        im = item["image"]
        if im.mode != "RGB":
            im = im.convert("RGB")
        w, h = im.size
        if is_train:
            # Aspect-ratio-preserving resize so max(h, w) == canvas, placed at top-left (0, 0)
            scale = canvas / float(max(w, h))
            nw = max(1, min(canvas, int(round(w * scale))))
            nh = max(1, min(canvas, int(round(h * scale))))
            im = im.resize((nw, nh), Image.Resampling.BILINEAR)
            arr = np.asarray(im, dtype=np.uint8)
            imgs_mm[i, :, :nh, :nw] = np.transpose(arr, (2, 0, 1))
            if shps_mm is not None:
                shps_mm[i, 0] = nh
                shps_mm[i, 1] = nw
        else:
            # Exact torchvision Resize(256, BICUBIC) + CenterCrop(224)
            if w < h:
                nw = 256
                nh = int(round(h * 256.0 / w))
            else:
                nh = 256
                nw = int(round(w * 256.0 / h))
            im = im.resize((nw, nh), Image.Resampling.BICUBIC)
            left = (nw - 224) // 2
            top = (nh - 224) // 2
            im = im.crop((left, top, left + 224, top + 224))
            arr = np.asarray(im, dtype=np.uint8)
            imgs_mm[i] = np.transpose(arr, (2, 0, 1))

        lbls_mm[i] = int(item["label"])
    return end - start


def _build_split(
    data_dir: str,
    out_dir: Path,
    split: str,
    prefix: str,
    size: int,
    is_train: bool,
    num_workers: int,
) -> None:
    ds = load_dataset("ILSVRC/imagenet-1k", split=split, cache_dir=data_dir)
    n = len(ds)
    del ds

    img_path = out_dir / f"{prefix}_images.npy"
    lbl_path = out_dir / f"{prefix}_labels.npy"
    shp_path = out_dir / f"{prefix}_shapes.npy" if is_train else None
    done_path = out_dir / f"{prefix}.done"

    if done_path.exists() and img_path.exists() and lbl_path.exists():
        print(f"[cache] {prefix} already built at {img_path} ({n} images), skipping.", flush=True)
        return

    print(
        f"[cache] Building {prefix}: {n} images ({size}x{size}x3 uint8, "
        f"{n * 3 * size * size / (1024**3):.1f} GiB) with {num_workers} workers ...",
        flush=True,
    )
    t0 = time.monotonic()

    mm_img = np.lib.format.open_memmap(str(img_path), mode="w+", dtype=np.uint8, shape=(n, 3, size, size))
    mm_lbl = np.lib.format.open_memmap(str(lbl_path), mode="w+", dtype=np.int64, shape=(n,))
    del mm_img, mm_lbl
    if shp_path is not None:
        mm_shp = np.lib.format.open_memmap(str(shp_path), mode="w+", dtype=np.int16, shape=(n, 2))
        del mm_shp

    chunk_size = 2048
    chunks = [(s, min(s + chunk_size, n)) for s in range(0, n, chunk_size)]
    done = 0
    with Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(data_dir, split, str(img_path), str(lbl_path), str(shp_path or ""), is_train, size),
    ) as pool:
        for cnt in pool.imap_unordered(_process_chunk, chunks):
            done += cnt
            if done % 131072 < chunk_size or done == n:
                dt = time.monotonic() - t0
                print(
                    f"  [{prefix}] {done:,}/{n:,} ({100.0*done/n:5.1f}%) | "
                    f"{done/max(dt, 1e-3):,.0f} img/s | elapsed {dt:.1f}s",
                    flush=True,
                )

    done_path.write_text(f"ok {n} canvas={size}\n", encoding="utf-8")
    print(f"[cache] DONE {prefix} in {time.monotonic() - t0:.1f}s!", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build /dev/shm aspect-ratio-preserving ImageNet-1k uint8 RAM cache")
    parser.add_argument("--data-dir", type=str, default="./imagenet1k")
    parser.add_argument("--out-dir", type=str, default="/dev/shm/in1k_raw_cache_ar320")
    parser.add_argument("--canvas", type=int, default=320)
    parser.add_argument("--workers", type=int, default=128)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _build_split(args.data_dir, out_dir, "validation", "val", 224, is_train=False, num_workers=min(64, args.workers))
    _build_split(args.data_dir, out_dir, "train", "train", args.canvas, is_train=True, num_workers=args.workers)


if __name__ == "__main__":
    main()
