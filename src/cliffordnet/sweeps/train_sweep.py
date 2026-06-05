from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cliffordnet.tasks.imagenet1k import (
    CliffordNetLightning,
    ImageNet1kDataModule,
    MODEL_SIZE_CHOICES,
    _model_kwargs_for_size,
    auto_find_batch_size,
    model_cls_for_size,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CliffordNet W&B Sweep Entry Point",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", type=str, default="./imagenet1k")
    p.add_argument("--output-dir", type=str, default="./outputs/sweep")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument(
        "--model-size",
        type=str,
        default="probe_xs",
        choices=MODEL_SIZE_CHOICES,
    )
    p.add_argument("--wedge-mode", type=str, default="fma", choices=["naive", "fp32", "fma"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--auto-batch-max", type=int, default=2048)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--gradient-clip-val", type=float, default=1.0)
    p.add_argument("--val-check-interval", type=float, default=0.5)
    p.add_argument("--ortho-weight", type=float, default=0.01)
    p.add_argument("--drop-path-rate", type=float, default=None)
    p.add_argument("--mixup-alpha", type=float, default=0.8)
    p.add_argument("--cutmix-alpha", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--eta-min", type=float, default=1e-6)
    p.add_argument("--no-diagnostics", action="store_true")
    p.add_argument("--diag-log-interval", type=int, default=200)
    p.add_argument("--wandb-project", type=str, default="CliffordNet-sweep")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-offline", action="store_true")
    return p


def merge_wandb_config(args: argparse.Namespace) -> argparse.Namespace:
    cfg = dict(wandb.config)
    key_map = {
        "lr": "lr",
        "weight_decay": "weight_decay",
        "ortho_weight": "ortho_weight",
        "drop_path_rate": "drop_path_rate",
        "mixup_alpha": "mixup_alpha",
        "cutmix_alpha": "cutmix_alpha",
        "label_smoothing": "label_smoothing",
        "warmup_epochs": "warmup_epochs",
        "eta_min": "eta_min",
        "model_size": "model_size",
        "epochs": "epochs",
        "batch_size": "batch_size",
        "auto_batch_max": "auto_batch_max",
        "num_workers": "num_workers",
        "wedge_mode": "wedge_mode",
    }
    for wkey, akey in key_map.items():
        if wkey in cfg:
            setattr(args, akey, cfg[wkey])
            print(f"[sweep] {akey} = {cfg[wkey]}  (from wandb.config)")
    return args


class CliffordNetSweep(CliffordNetLightning):
    def __init__(self, drop_path_rate_override=None, **kwargs):
        super().__init__(**kwargs)
        if drop_path_rate_override is not None and drop_path_rate_override >= 0:
            self._patch_drop_path(drop_path_rate_override)

    def _patch_drop_path(self, rate: float):
        from timm.layers import DropPath
        import torch.nn as nn

        block_refs = list(self._raw_model.iter_blocks())
        n_blocks = len(block_refs)
        dpr = torch.linspace(0, rate, n_blocks).tolist()
        for i, (_label, block) in enumerate(block_refs):
            block.drop_path = DropPath(dpr[i]) if dpr[i] > 0.0 else nn.Identity()
        print(f"[sweep] drop_path_rate patched to {rate:.3f} ({n_blocks} blocks)")


def train(args: argparse.Namespace):
    L.seed_everything(42)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    os.makedirs(args.output_dir, exist_ok=True)
    run_id = wandb.run.id if wandb.run else f"manual_{int(time.time())}"
    run_dir = os.path.join(args.output_dir, run_id)

    if args.batch_size <= 0:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        args.batch_size = auto_find_batch_size(
            model_cls=model_cls_for_size(args.model_size),
            model_kwargs=_model_kwargs_for_size(args.model_size),
            max_batch_size=args.auto_batch_max,
            device=device,
        )
        wandb.config.update({"detected_batch_size": args.batch_size}, allow_val_change=True)
        print(f"[sweep] auto batch_size = {args.batch_size}")

    data = ImageNet1kDataModule(
        nfs_data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = CliffordNetSweep(
        drop_path_rate_override=args.drop_path_rate,
        model_size=args.model_size,
        num_classes=1000,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        label_smoothing=args.label_smoothing,
        warmup_epochs=args.warmup_epochs,
        eta_min=args.eta_min,
        wedge_mode=args.wedge_mode,
        ortho_weight=args.ortho_weight,
        enable_diagnostics=not args.no_diagnostics,
        diag_log_interval=args.diag_log_interval,
    )

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        id=wandb.run.id,
        resume="allow",
        save_dir=run_dir,
        offline=args.wandb_offline,
        log_model=False,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(run_dir, "checkpoints"),
        monitor="val/acc1",
        mode="max",
        save_last=False,
        save_top_k=1,
        filename="{epoch}-{val/acc1:.4f}",
    )

    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        strategy="auto",
        precision="bf16-mixed",
        max_epochs=args.epochs,
        gradient_clip_val=args.gradient_clip_val,
        val_check_interval=args.val_check_interval,
        callbacks=[checkpoint_cb, LearningRateMonitor("step"), RichProgressBar()],
        logger=wandb_logger,
        log_every_n_steps=20,
        limit_val_batches=0.2,
    )
    trainer.fit(model, data)


def main():
    parser = build_parser()
    args, _unknown = parser.parse_known_args()
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args),
        mode="offline" if args.wandb_offline else "online",
    )
    args = merge_wandb_config(args)
    try:
        train(args)
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
