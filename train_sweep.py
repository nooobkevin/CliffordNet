"""
CliffordNet W&B Sweep Entry Point
===================================
This script is the `program` target for sweep_hparam.yaml and sweep_arch.yaml.

How it works:
  1. wandb.init() is called first so the sweep agent can inject hyperparams
     into wandb.config before any training code runs.
  2. wandb.config values override argparse defaults, so the same CLI flags
     still work for manual runs (just without sweep injection).
  3. All actual training logic lives in train_imagenet1k.py — this file is
     intentionally thin.

Usage (manual, for testing):
  uv run python train_sweep.py --data-dir ./imagenet1k --batch-size 256

Usage (via wandb agent):
  wandb sweep sweep_hparam.yaml          # prints sweep ID
  wandb agent <entity>/<project>/<id>    # launch one agent per GPU

Multi-GPU note:
  W&B sweep agents are single-process.  For multi-GPU sweeps, wrap with
  torchrun and use --sweep-mode (see train_imagenet1k.py --help).
  For single-GPU probe sweeps this script is sufficient.
"""

import argparse
import os
import time

import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger

# Re-use all model / data definitions from the main training script.
from train_imagenet1k import (
    CliffordNetLightning,
    ImageNet1kDataModule,
)


# ---------------------------------------------------------------------------
# Argument parsing — mirrors train_imagenet1k.py but only the sweep-relevant
# subset.  wandb.config will override these defaults at runtime.
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    # NOTE: W&B sweep agents pass hyperparams as --key=value with underscores
    # (e.g. --batch_size=256).  We use parse_known_args in main() to silently
    # ignore those — all sweep values are read from wandb.config instead.
    p = argparse.ArgumentParser(
        description="CliffordNet W&B Sweep Entry Point",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--data-dir", type=str, default="./imagenet1k")
    p.add_argument("--output-dir", type=str, default="./outputs/sweep")
    p.add_argument("--prefetch-local", action="store_true")
    p.add_argument("--local-cache-dir", type=str, default=None)
    p.add_argument("--num-workers", type=int, default=8)

    # Model
    p.add_argument(
        "--model-size",
        type=str,
        default="probe_xs",
        choices=["probe_xs", "probe_s", "12_2", "12_5", "18_5", "32_3", "32_5", "64_5"],
    )
    p.add_argument(
        "--wedge-mode", type=str, default="fma", choices=["naive", "fp32", "fma"]
    )

    # Training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--gradient-clip-val", type=float, default=1.0)
    p.add_argument(
        "--val-check-interval",
        type=float,
        default=0.5,
        help="Validate every N fraction of epoch (0.5 = twice per epoch)",
    )

    # Regularisation
    p.add_argument("--ortho-weight", type=float, default=0.01)
    p.add_argument(
        "--drop-path-rate",
        type=float,
        default=None,
        help="Override model default drop_path_rate (None = use model default)",
    )
    p.add_argument("--mixup-alpha", type=float, default=0.8)
    p.add_argument("--cutmix-alpha", type=float, default=1.0)

    # Diagnostics
    p.add_argument("--no-diagnostics", action="store_true")
    p.add_argument("--diag-log-interval", type=int, default=200)

    # W&B
    p.add_argument("--wandb-project", type=str, default="CliffordNet-sweep")
    p.add_argument("--wandb-entity", type=str, default="nooobkevin")
    p.add_argument("--wandb-offline", action="store_true")

    return p


# ---------------------------------------------------------------------------
# Merge wandb.config (sweep injected) into parsed args.
# Keys in wandb.config use underscores; argparse uses hyphens → normalise.
# ---------------------------------------------------------------------------


def merge_wandb_config(args: argparse.Namespace) -> argparse.Namespace:
    """Overwrite args with any values present in wandb.config."""
    cfg = dict(wandb.config)
    if not cfg:
        return args

    # Map wandb key → argparse attribute name (underscores, hyphens → _)
    key_map = {
        "lr": "lr",
        "weight_decay": "weight_decay",
        "ortho_weight": "ortho_weight",
        "drop_path_rate": "drop_path_rate",
        "mixup_alpha": "mixup_alpha",
        "cutmix_alpha": "cutmix_alpha",
        "model_size": "model_size",
        "epochs": "epochs",
        "batch_size": "batch_size",
        "num_workers": "num_workers",
        "wedge_mode": "wedge_mode",
    }
    for wkey, akey in key_map.items():
        if wkey in cfg:
            setattr(args, akey, cfg[wkey])
            print(f"[sweep] {akey} = {cfg[wkey]}  (from wandb.config)")

    return args


# ---------------------------------------------------------------------------
# Build a modified CliffordNetLightning that accepts drop_path_rate override.
# We need this because the production model builders hard-code drop_path_rate
# but sweeps want to vary it.
# ---------------------------------------------------------------------------


class CliffordNetSweep(CliffordNetLightning):
    """
    Thin subclass that patches the raw model's drop_path_rate after init,
    so sweeps can control it without changing the model builder signatures.

    NOTE: This replaces DropPath modules in-place *before* torch.compile,
    so no recompilation is triggered.
    """

    def __init__(self, drop_path_rate_override=None, **kwargs):
        super().__init__(**kwargs)
        if drop_path_rate_override is not None and drop_path_rate_override >= 0:
            self._patch_drop_path(drop_path_rate_override)

    def _patch_drop_path(self, rate: float):
        from timm.layers import DropPath
        import torch.nn as nn

        n_blocks = len(self._raw_model.blocks)
        # Linear schedule from 0 → rate (same as the original linspace)
        import torch

        dpr = torch.linspace(0, rate, n_blocks).tolist()
        for i, block in enumerate(self._raw_model.blocks):
            new_dp = DropPath(dpr[i]) if dpr[i] > 0.0 else nn.Identity()
            block.drop_path = new_dp
        print(f"[sweep] drop_path_rate patched to {rate:.3f} ({n_blocks} blocks)")


# ---------------------------------------------------------------------------
# Main training function called by the sweep agent
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace):
    L.seed_everything(42)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    os.makedirs(args.output_dir, exist_ok=True)

    # Each sweep run gets a unique sub-directory so checkpoints don't collide.
    run_id = wandb.run.id if wandb.run else f"manual_{int(time.time())}"
    run_dir = os.path.join(args.output_dir, run_id)

    data = ImageNet1kDataModule(
        nfs_data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_local=args.prefetch_local,
        local_cache_dir=args.local_cache_dir,
    )

    model = CliffordNetSweep(
        # sweep-controlled override
        drop_path_rate_override=args.drop_path_rate,
        # standard kwargs
        model_size=args.model_size,
        num_classes=1000,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        wedge_mode=args.wedge_mode,
        ortho_weight=args.ortho_weight,
        enable_diagnostics=not args.no_diagnostics,
        diag_log_interval=args.diag_log_interval,
    )

    # Re-use the already-initialised wandb run (sweep agent owns it).
    # Passing id + resume="allow" tells WandbLogger to attach to the existing
    # run instead of creating a second one — otherwise Lightning logs metrics
    # to an orphan run that never appears in the sweep dashboard.
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
        save_last=False,  # saves disk space during sweeps
        save_top_k=1,
        filename="{epoch}-{val/acc1:.4f}",
    )

    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,  # sweeps run one GPU per agent
        strategy="auto",
        precision="bf16-mixed",
        max_epochs=args.epochs,
        gradient_clip_val=args.gradient_clip_val,
        val_check_interval=args.val_check_interval,
        callbacks=[
            checkpoint_cb,
            LearningRateMonitor("step"),
            RichProgressBar(),
        ],
        logger=wandb_logger,
        log_every_n_steps=20,
        # Limit validation batches during sweeps to keep runs fast.
        # Full val set (50k) is slow; 20% gives stable enough estimates.
        limit_val_batches=0.2,
    )

    trainer.fit(model, data)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = build_parser()
    # parse_known_args silently drops the --key=value (underscore) flags that
    # the W&B agent injects — those values come from wandb.config instead.
    args, _unknown = parser.parse_known_args()

    # Initialise wandb.  If called by a sweep agent, wandb.config is already
    # populated; otherwise this starts a regular run.
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args),  # seed with CLI defaults; sweep will override
        mode="offline" if args.wandb_offline else "online",
    )

    # Overwrite args with sweep-injected hyperparams (if any).
    args = merge_wandb_config(args)

    try:
        train(args)
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
