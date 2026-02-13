"""
CliffordNet Training Script for ImageNet-1k with PyTorch Lightning
Optimized for Stability on 6x 4090D GPUs
"""

from datasets import load_dataset
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath, trunc_normal_
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
)
import lightning as L
import matplotlib.pyplot as plt
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')

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
            nn.GroupNorm(1, dim, eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(dim, dim, kernel_size=3,
                      padding=1, groups=dim, bias=False),
            nn.GroupNorm(1, dim, eps=1e-6),
            nn.SiLU(),
        )

        self.det_proj = nn.Conv2d(dim, dim, kernel_size=1)

        self.norm_ctx = nn.GroupNorm(1, dim, eps=1e-6)
        self.norm_det = nn.GroupNorm(1, dim, eps=1e-6)

        input_proj_dim = 2 * len(shifts) * dim
        self.final_proj = nn.Conv2d(input_proj_dim, dim, kernel_size=1)

        # 預算所有 shift 的 channel 索引: (S, C)
        base = torch.arange(dim)
        roll_idx = torch.stack([(base - s) % dim for s in shifts], dim=0)
        self.register_buffer('_roll_idx', roll_idx)

    def forward(self, x):
        z_ctx = self.ctx_conv(x)
        z_det = self.det_proj(x)

        z_ctx = self.norm_ctx(z_ctx)
        z_det = self.norm_det(z_det)

        B, C, H, W = z_det.shape
        S = len(self.shifts)

        # 用 advanced indexing 一次完成所有 shift 的 roll
        # (B, C, H, W) -> (B, S, C, H, W)
        z_det_rolled = z_det[:, self._roll_idx]
        z_ctx_rolled = z_ctx[:, self._roll_idx]

        # 廣播原始張量: (B, 1, C, H, W)
        z_det_b = z_det.unsqueeze(1)
        z_ctx_b = z_ctx.unsqueeze(1)

        # 所有 shift 並行計算
        prod = z_det_b * z_ctx_rolled                       # (B, S, C, H, W)
        dot = F.silu(prod)                                # (B, S, C, H, W)
        wedge = (prod - z_det_rolled * z_ctx_b).to(x.dtype)  # (B, S, C, H, W)

        # 交錯排列 [dot_s0, wedge_s0, dot_s1, wedge_s1, ...]
        pairs = torch.stack([dot, wedge], dim=2)  # (B, S, 2, C, H, W)
        g_raw = pairs.reshape(B, S * 2 * C, H, W)

        g_feat = self.final_proj(g_raw)
        return g_feat


class CliffordBlock(nn.Module):
    def __init__(self, dim, shifts, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.interaction = CliffordInteraction(dim, shifts=shifts)
        self.gate_linear = nn.Conv2d(dim * 2, dim, kernel_size=1)

        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((1, dim, 1, 1)), requires_grad=True
        )
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()

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
    def __init__(
        self,
        img_size=224,
        in_chans=3,
        num_classes=1000,
        embed_dim=192,
        depth=12,
        shifts=[1, 2],
        drop_path_rate=0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.GroupNorm(1, embed_dim // 2),
            nn.SiLU(),
            nn.Conv2d(
                embed_dim // 2,
                embed_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, embed_dim),
            nn.SiLU(),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                CliffordBlock(dim=embed_dim, shifts=shifts, drop_path=dpr[i])
                for i in range(depth)
            ]
        )

        self.norm = nn.GroupNorm(1, embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=[-2, -1])
        x = self.head(x)
        return x


# Model Builders

def cliffordnet_nano(num_classes=1000):
    return CliffordNet(
        img_size=224,
        embed_dim=128,
        depth=8,
        shifts=[1, 2, 4, 8],
        num_classes=num_classes,
        drop_path_rate=0.05
    )


def cliffordnet_small(num_classes=1000):
    return CliffordNet(
        embed_dim=192,
        depth=12,
        shifts=[1, 2, 4],
        num_classes=num_classes,
        drop_path_rate=0.1,
    )


def cliffordnet_base(num_classes=1000):
    return CliffordNet(
        img_size=224,
        embed_dim=384,
        depth=16,
        shifts=[1, 2, 4, 8],
        num_classes=num_classes,
        drop_path_rate=0.2
    )


def cliffordnet_large(num_classes=1000):
    return CliffordNet(
        img_size=224,
        embed_dim=512,
        depth=24,
        shifts=[1, 2, 4, 8, 16],
        num_classes=num_classes,
        drop_path_rate=0.3
    )

# ============================================================================
# Lightning Module (TensorBoard logging fixed)
# ============================================================================


class CliffordNetLightning(L.LightningModule):
    def __init__(
        self,
        model_size="small",
        num_classes=1000,
        learning_rate=5e-4,
        weight_decay=0.05,
        warmup_epochs=1,
        max_epochs=200,
    ):
        super().__init__()
        self.save_hyperparameters()

        model_builders = {
            "nano": cliffordnet_nano,
            "small": cliffordnet_small,
            "base": cliffordnet_base,
            "large": cliffordnet_large,
        }
        self.model = model_builders[model_size](num_classes=num_classes)
        self.model = self.model.to(memory_format=torch.channels_last)
        self.model = torch.compile(
            self.model)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.register_buffer(
            "inv_mean", torch.tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "inv_std", torch.tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1)
        )

    def forward(self, x):
        return self.model(x.contiguous(memory_format=torch.channels_last))
        # return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        acc1, acc5 = self._accuracy(outputs, labels, topk=(1, 5))

        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log("train/acc1", acc1, prog_bar=True, sync_dist=True)
        self.log("train/acc5", acc5, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        acc1, acc5 = self._accuracy(outputs, labels, topk=(1, 5))

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/acc1", acc1, prog_bar=True, sync_dist=True)
        self.log("val/acc5", acc5, prog_bar=True, sync_dist=True)

        # Image/text visualization: only rank 0, only first batch
        if batch_idx == 0 and self.trainer.is_global_zero:
            self._log_images(images, labels, outputs)

        return loss

    def _log_images(self, images, labels, outputs):
        n = min(images.shape[0], 8)
        imgs = images[:n]
        lbls = labels[:n]
        preds = outputs[:n].argmax(dim=1)

        # Denormalize
        imgs = imgs * self.inv_std + self.inv_mean
        imgs = torch.clamp(imgs, 0, 1)

        ncols = 4
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(3 * ncols, 3.5 * nrows))
        if nrows == 1:
            axes = [axes] if ncols == 1 else list(axes)
        else:
            axes = axes.flatten().tolist()

        for i in range(len(axes)):
            ax = axes[i]
            ax.axis("off")
            if i < n:
                img_np = imgs[i].cpu().permute(1, 2, 0).float().numpy()
                ax.imshow(img_np)
                gt, pd = lbls[i].item(), preds[i].item()
                color = "green" if gt == pd else "red"
                ax.set_title(f"GT:{gt} / P:{pd}", fontsize=11,
                             color=color, fontweight="bold")

        fig.suptitle(f"Epoch {self.current_epoch}", fontsize=14)
        fig.tight_layout()

        tb = self.logger.experiment
        tb.add_figure("val/sample_predictions", fig, self.current_epoch)
        tb.flush()
        plt.close(fig)

    def _accuracy(self, output, target, topk=(1,)):
        """
        Computes top-k accuracy as a fraction in [0, 1].
        FIX: returns 0-D scalar tensors instead of shape-[1] tensors,
        which is what self.log() expects.
        """
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0)
                res.append(correct_k / batch_size)
            return res

    def configure_optimizers(self):
        optimizer = torch.optim.Muon(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.hparams.warmup_epochs / self.hparams.max_epochs,
            div_factor=25,
            final_div_factor=1e4,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


# ============================================================================
# Data Module
# ============================================================================


class HFImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, item["label"]


class ImageNet1kDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        train_tf = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN,
                                     IMAGENET_DEFAULT_STD),
            ]
        )
        val_tf = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN,
                                     IMAGENET_DEFAULT_STD),
            ]
        )

        ds = load_dataset(
            "ILSVRC/imagenet-1k", cache_dir=self.data_dir
        )
        self.train_ds = HFImageNetDataset(ds["train"], transform=train_tf)
        self.val_ds = HFImageNetDataset(ds["validation"], transform=val_tf)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train CliffordNet on ImageNet-1k")
    parser.add_argument("--data-dir", type=str, default="./imagenet1k",
                        help="Path to cache ImageNet-1k dataset")
    parser.add_argument("--model-size", type=str, default="small",
                        choices=["nano", "small", "base", "large"],
                        help="Model size variant")
    parser.add_argument("--batch-size", type=int, default=24,
                        help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=90,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate (default: 5e-4 for stability)")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Number of GPUs to use")
    parser.add_argument("--weight-decay", type=float, default=0.05,
                        help="Weight decay")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="Number of warmup epochs")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of data loading workers per GPU")
    parser.add_argument("--gradient-clip-val", type=float, default=0.5,
                        help="Gradient clipping value (default: 0.5 for stability)")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    args = parser.parse_args()

    # Scale learning rate based on effective batch size
    base_batch_size = 80  # Reference batch size for base LR
    effective_batch_size = args.batch_size * args.num_gpus
    lr_scale = effective_batch_size / base_batch_size
    scaled_lr = args.lr * lr_scale

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Starting training with Base LR={args.lr}, Scaled LR={scaled_lr:.6f}")
        print(
            f"Effective Batch Size: {effective_batch_size} (BS={args.batch_size} × GPUs={args.num_gpus})")
        print(f"LR Scale Factor: {lr_scale:.3f}")
        print(f"Model Size: {args.model_size}")

    L.seed_everything(42)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
    data = ImageNet1kDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    model = CliffordNetLightning(
        model_size=args.model_size,
        num_classes=1000,
        learning_rate=scaled_lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        monitor="val/acc1",
        mode="max",
        save_last=True,
        filename="{epoch}-{val/acc1:.4f}",
    )

    trainer = L.Trainer(
        accelerator="gpu",
        devices=args.num_gpus,
        strategy="ddp",
        precision="bf16-mixed",
        max_epochs=args.epochs,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=[checkpoint_callback, LearningRateMonitor(
            "step"), RichProgressBar()],
        logger=TensorBoardLogger(
            args.output_dir, name="cliffordnet_imagenet1k"),
        log_every_n_steps=10,
    )
    # tuner = L.pytorch.tuner.Tuner(trainer)
    # tuner.scale_batch_size(model, datamodule=data, mode="power",
    #                        init_val=args.batch_size, max_trials=10,
    #                        steps_per_trial=3)
    # print(f"Optimal batch size: {data.batch_size}")
    trainer.fit(model, data, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
