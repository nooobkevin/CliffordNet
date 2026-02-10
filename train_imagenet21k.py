"""
CliffordNet Training Script for ImageNet-21k with PyTorch Lightning + FSDP
Optimized for 6x 4090D GPUs on a single node
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
)
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.loggers import TensorBoardLogger
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from timm.layers import DropPath, trunc_normal_
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import functools
from pathlib import Path
import argparse


# ============================================================================
# CliffordNet Model Components
# ============================================================================

class CliffordInteraction(nn.Module):

    def __init__(self, dim, shifts=[1, 2]):
        super().__init__()
        self.dim = dim
        self.shifts = shifts

        self.ctx_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3,
                      padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, kernel_size=3,
                      padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU()
        )

        self.det_proj = nn.Conv2d(dim, dim, kernel_size=1)

        self.norm_ctx = nn.GroupNorm(1, dim, eps=1e-6)
        self.norm_det = nn.GroupNorm(1, dim, eps=1e-6)

        input_proj_dim = 2 * len(shifts) * dim
        self.final_proj = nn.Conv2d(input_proj_dim, dim, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape

        z_ctx = self.ctx_conv(x)
        z_det = self.det_proj(x)

        z_ctx = self.norm_ctx(z_ctx)
        z_det = self.norm_det(z_det)
        f_list = []

        for s in self.shifts:
            z_det_s = torch.roll(z_det, shifts=s, dims=1)
            z_ctx_s = torch.roll(z_ctx, shifts=s, dims=1)

            dot_s = F.silu(z_det * z_ctx_s)
            wedge_s = z_det * z_ctx_s - z_det_s * z_ctx

            f_list.append(dot_s)
            f_list.append(wedge_s)

        g_raw = torch.cat(f_list, dim=1)
        g_feat = self.final_proj(g_raw)

        return g_feat


class CliffordBlock(nn.Module):

    def __init__(self, dim, shifts, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()

        self.norm = nn.GroupNorm(1, dim)
        self.interaction = CliffordInteraction(dim, shifts=shifts)
        self.gate_linear = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((1, dim, 1, 1)), requires_grad=True)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x_ln = self.norm(x)
        g_feat = self.interaction(x_ln)
        m = torch.cat([x_ln, g_feat], dim=1)
        alpha = torch.sigmoid(self.gate_linear(m))
        h_mix = F.silu(x_ln) + alpha * g_feat
        x = shortcut + self.drop_path(self.gamma * h_mix)
        return x


class CliffordNet(nn.Module):
    def __init__(self,
                 img_size=224,
                 in_chans=3,
                 num_classes=10450,  # ImageNet-21k-P classes
                 embed_dim=192,
                 depth=12,
                 shifts=[1, 2],
                 drop_path_rate=0.1,
                 patch_size=4):  # Downsample for 224x224 images
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Stem: Patch Embedding with downsampling for larger images
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(embed_dim // 2),
            nn.SiLU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU()
        )

        # Backbone: Stack of Clifford Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            CliffordBlock(
                dim=embed_dim,
                shifts=shifts,
                drop_path=dpr[i]
            ) for i in range(depth)
        ])

        # Head
        self.norm = nn.GroupNorm(1, embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=[-2, -1])  # Global Average Pooling
        x = self.head(x)
        return x


# ============================================================================
# Model Configurations
# ============================================================================

def cliffordnet_small_21k(num_classes=10450):
    """Small model for ImageNet-21k (~5M params)"""
    return CliffordNet(
        img_size=224,
        embed_dim=192,
        depth=12,
        shifts=[1, 2, 4],
        num_classes=num_classes,
        drop_path_rate=0.1
    )


def cliffordnet_base_21k(num_classes=10450):
    """Base model for ImageNet-21k (~15M params)"""
    return CliffordNet(
        img_size=224,
        embed_dim=384,
        depth=16,
        shifts=[1, 2, 4, 8],
        num_classes=num_classes,
        drop_path_rate=0.2
    )


def cliffordnet_large_21k(num_classes=10450):
    """Large model for ImageNet-21k (~40M params)"""
    return CliffordNet(
        img_size=224,
        embed_dim=512,
        depth=24,
        shifts=[1, 2, 4, 8, 16],
        num_classes=num_classes,
        drop_path_rate=0.3
    )


# ============================================================================
# Lightning Module
# ============================================================================

class CliffordNetLightning(L.LightningModule):
    def __init__(
        self,
        model_size: str = "small",
        num_classes: int = 10450,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.05,
        warmup_epochs: int = 5,
        max_epochs: int = 90,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Build model
        model_builders = {
            "small": cliffordnet_small_21k,
            "base": cliffordnet_base_21k,
            "large": cliffordnet_large_21k,
        }
        self.model = model_builders[model_size](num_classes=num_classes)

        # Loss with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Metrics tracking
        self.train_acc = 0.0
        self.val_acc = 0.0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        # Calculate accuracy
        _, predicted = outputs.max(1)
        acc = (predicted == labels).float().mean()

        # Log metrics
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log("train/acc", acc, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        _, predicted = outputs.max(1)
        acc = (predicted == labels).float().mean()

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/acc", acc, prog_bar=True, sync_dist=True)

        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        # AdamW optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
        )

        # Warmup + Cosine Annealing
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.001,
            end_factor=1.0,
            total_iters=self.hparams.warmup_epochs,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs - self.hparams.warmup_epochs,
            eta_min=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.hparams.warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


# ============================================================================
# Data Module
# ============================================================================

class ImageNet21kDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/data/imagenet21k",
        batch_size: int = 64,
        num_workers: int = 8,
        img_size: int = 224,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size

    def setup(self, stage=None):
        # Training transforms with strong augmentation
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                self.img_size,
                scale=(0.08, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            transforms.RandomErasing(p=0.25),
        ])

        # Validation transforms
        self.val_transform = transforms.Compose([
            transforms.Resize(
                int(self.img_size * 256 / 224),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])

        if stage == "fit" or stage is None:
            train_dir = self.data_dir / "train"
            val_dir = self.data_dir / "val"

            if not train_dir.exists():
                # Fallback: single directory structure
                train_dir = self.data_dir
                val_dir = self.data_dir

            self.train_dataset = ImageFolder(
                train_dir, transform=self.train_transform)
            self.val_dataset = ImageFolder(
                val_dir, transform=self.val_transform)

            print(f"Train dataset size: {len(self.train_dataset)}")
            print(f"Val dataset size: {len(self.val_dataset)}")
            print(f"Number of classes: {len(self.train_dataset.classes)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )


# ============================================================================
# FSDP Configuration
# ============================================================================

def get_fsdp_strategy():
    """Configure FSDP strategy optimized for 6x 4090D GPUs"""

    # Auto-wrap policy based on module size (wrap modules with >1M params)
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy,
        min_num_params=1_000_000,
    )

    strategy = FSDPStrategy(
        auto_wrap_policy=auto_wrap_policy,
        # Checkpoint blocks to save memory
        activation_checkpointing_policy={CliffordBlock},
        sharding_strategy="FULL_SHARD",  # Most memory efficient
        state_dict_type="sharded",  # Efficient checkpointing
        limit_all_gathers=True,  # Reduce memory spikes
        cpu_offload=False,  # Keep on GPU for 4090D
    )

    return strategy


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train CliffordNet on ImageNet-21k")
    parser.add_argument("--data-dir", type=str, default="/data/imagenet21k",
                        help="Path to ImageNet-21k dataset")
    parser.add_argument("--model-size", type=str, default="small",
                        choices=["small", "base", "large"],
                        help="Model size variant")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=90,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.05,
                        help="Weight decay")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="Number of warmup epochs")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of data loading workers per GPU")
    parser.add_argument("--num-classes", type=int, default=10450,
                        help="Number of classes (10450 for ImageNet-21k-P)")
    parser.add_argument("--accumulate-grad-batches", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--precision", type=str, default="bf16-mixed",
                        choices=["32", "16-mixed", "bf16-mixed"],
                        help="Training precision")
    parser.add_argument("--gradient-clip-val", type=float, default=1.0,
                        help="Gradient clipping value")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    args = parser.parse_args()

    # Print configuration
    print("=" * 60)
    print("CliffordNet Training on ImageNet-21k with FSDP")
    print("=" * 60)
    print(f"Model size: {args.model_size}")
    print(f"Batch size per GPU: {args.batch_size}")
    print(f"Total batch size: {args.batch_size * 6}")  # 6 GPUs
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Precision: {args.precision}")
    print(f"Data directory: {args.data_dir}")
    print("=" * 60)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize data module
    data_module = ImageNet21kDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Initialize model
    model = CliffordNetLightning(
        model_size=args.model_size,
        num_classes=args.num_classes,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
    )

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 60)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, "checkpoints"),
            filename="cliffordnet-{epoch:02d}-{val/acc:.4f}",
            monitor="val/acc",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(),
    ]

    # Logger
    logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name="cliffordnet_imagenet21k",
    )

    # FSDP Strategy
    strategy = get_fsdp_strategy()

    # Trainer
    trainer = L.Trainer(
        accelerator="gpu",
        devices=6,  # 6x 4090D GPUs
        num_nodes=1,
        strategy=strategy,
        precision=args.precision,
        max_epochs=args.epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
        val_check_interval=0.5,  # Validate twice per epoch
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False,  # For speed
        benchmark=True,  # Optimize cudnn for consistent input sizes
    )

    # Train
    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=args.resume,
    )

    print("Training completed!")


if __name__ == "__main__":
    main()
