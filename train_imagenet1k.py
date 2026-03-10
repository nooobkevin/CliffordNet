"""
CliffordNet Multi-Node DDP Training Script for ImageNet-1k
PyTorch Lightning — NFS data with optional local SSD prefetch (/tmp/$UID)

Launch examples:

  # ── torchrun (2 nodes × 6 GPUs each) ──────────────────────────────
  # On node 0 (master):
  torchrun --nnodes=2 --nproc_per_node=6 --node_rank=0 \
           --master_addr=NODE0_IP --master_port=29500 \
           train.py --num-nodes 2 --num-gpus 6 --prefetch-local \
                    --data-dir /nfs/imagenet1k --output-dir /nfs/outputs

  # On node 1:
  torchrun --nnodes=2 --nproc_per_node=6 --node_rank=1 \
           --master_addr=NODE0_IP --master_port=29500 \
           train.py --num-nodes 2 --num-gpus 6 --prefetch-local \
                    --data-dir /nfs/imagenet1k --output-dir /nfs/outputs

  # ── SLURM ─────────────────────────────────────────────────────────
  #SBATCH --nodes=2
  #SBATCH --ntasks-per-node=6
  #SBATCH --gpus-per-node=6
  srun python train.py --num-nodes 2 --num-gpus 6 --prefetch-local \
       --data-dir /nfs/imagenet1k --output-dir /nfs/outputs
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
import subprocess
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns


# ============================================================================
# NFS → Local SSD Prefetch
# ============================================================================

def prefetch_nfs_to_local(nfs_dir, local_dir):
    """
    Copy HuggingFace dataset cache from NFS to node-local SSD.
    Marker file prevents redundant copies on restart.
    Called only by local-rank-0 on each node (via prepare_data).
    """
    marker = os.path.join(local_dir, ".prefetch_complete")
    if os.path.exists(marker):
        print(f"[Prefetch] Local cache already present at {local_dir}")
        return

    os.makedirs(local_dir, exist_ok=True)
    print(f"[Prefetch] Copying {nfs_dir} → {local_dir}  (may take a while) ...")

    ret = subprocess.run(
        ["rsync", "-a", "--info=progress2", f"{nfs_dir}/", f"{local_dir}/"]
    )
    if ret.returncode != 0:
        print("[Prefetch] rsync unavailable or failed, falling back to cp -a")
        subprocess.run(
            ["cp", "-a", f"{nfs_dir}/.", f"{local_dir}/"], check=True
        )

    with open(marker, "w") as f:
        f.write("done\n")
    print("[Prefetch] Complete.")


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

        base = torch.arange(dim)
        roll_idx = torch.stack([(base - s) % dim for s in shifts], dim=0)
        self.register_buffer('_roll_idx', roll_idx)

    def forward(self, x):
        z_ctx = self.ctx_conv(x)
        z_det = self.det_proj(x)

        z_ctx = self.norm_ctx(z_ctx)
        z_det = self.norm_det(z_det)

        B, C, H, W = z_det.shape

        z_det_rolled = z_det[:, self._roll_idx]
        z_ctx_rolled = z_ctx[:, self._roll_idx]

        z_det_b = z_det.unsqueeze(1)
        z_ctx_b = z_ctx.unsqueeze(1)

        prod = z_det_b * z_ctx_rolled
        dot = F.silu(prod)
        wedge = (prod - z_det_rolled * z_ctx_b).to(x.dtype)

        pairs = torch.stack([dot, wedge], dim=2)
        S = len(self.shifts)
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
        img_size=224,
        embed_dim=192,
        depth=12,
        shifts=[1, 2, 4, 8],
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
# Lightning Module
# ============================================================================


class CliffordNetLightning(L.LightningModule):
    def __init__(
        self,
        model_size="small",
        num_classes=1000,
        learning_rate=3e-4,
        weight_decay=0.05,
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
        self.model = torch.compile(self.model)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.register_buffer(
            "inv_mean", torch.tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "inv_std", torch.tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1)
        )

        self.val_preds = []
        self.val_labels = []

    def forward(self, x):
        return self.model(x.contiguous(memory_format=torch.channels_last))

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

        preds = outputs.argmax(dim=1)
        self.val_preds.append(preds.cpu())
        self.val_labels.append(labels.cpu())

        if batch_idx == 0 and self.trainer.is_global_zero:
            self._log_images(images, labels, outputs)

        return loss

    def on_validation_epoch_end(self):
        # ------------------------------------------------------------------
        # 每個 rank 各自只看到 DistributedSampler 分配的子集，
        # 先算局部混淆矩陣，再 all_reduce 加總得到全局結果。
        # 注意：所有 rank 都必須參與 all_reduce，否則會 deadlock。
        # ------------------------------------------------------------------
        num_cls = self.hparams.num_classes

        if len(self.val_preds) > 0:
            all_preds = torch.cat(self.val_preds).numpy()
            all_labels = torch.cat(self.val_labels).numpy()
            cm_local = confusion_matrix(
                all_labels, all_preds, labels=range(num_cls)
            )
        else:
            cm_local = np.zeros((num_cls, num_cls), dtype=np.int64)

        cm_tensor = torch.tensor(cm_local, dtype=torch.int64, device=self.device)

        # 跨所有 rank 加總
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(cm_tensor, op=dist.ReduceOp.SUM)

        # 只有 global rank 0 負責繪圖 / 記錄
        if self.trainer.is_global_zero:
            cm = cm_tensor.cpu().numpy()
            cm_normalized = cm.astype("float") / (
                cm.sum(axis=1, keepdims=True) + 1e-8
            )

            if num_cls > 100:
                self._log_top_confused_pairs(cm, top_k=20)
            else:
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(cm_normalized, ax=ax, cmap="Blues", cbar=True)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                ax.set_title(
                    f"Confusion Matrix — Epoch {self.current_epoch}"
                )
                fig.tight_layout()

                tb = self.logger.experiment
                tb.add_figure(
                    "val/confusion_matrix", fig, self.global_step
                )
                tb.flush()
                plt.close(fig)

        self.val_preds.clear()
        self.val_labels.clear()

    def _log_top_confused_pairs(self, cm, top_k=20):
        """Log top confused class pairs for large datasets like ImageNet."""
        cm_no_diag = cm.copy()
        np.fill_diagonal(cm_no_diag, 0)

        flat_indices = np.argsort(cm_no_diag.ravel())[-top_k:][::-1]
        top_pairs = [
            (idx // cm.shape[1], idx % cm.shape[1]) for idx in flat_indices
        ]

        fig, ax = plt.subplots(figsize=(12, 8))
        pair_labels = [f"{true}→{pred}" for true, pred in top_pairs]
        pair_counts = [cm[true, pred] for true, pred in top_pairs]

        bars = ax.barh(
            range(len(pair_labels)), pair_counts, color="steelblue"
        )
        ax.set_yticks(range(len(pair_labels)))
        ax.set_yticklabels(pair_labels)
        ax.set_xlabel("Count")
        ax.set_title(
            f"Top {top_k} Confused Class Pairs — Epoch {self.current_epoch}"
        )
        ax.invert_yaxis()

        for bar, count in zip(bars, pair_counts):
            ax.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                str(count),
                va="center",
                fontsize=9,
            )

        fig.tight_layout()

        tb = self.logger.experiment
        tb.add_figure("val/top_confused_pairs", fig, self.global_step)
        tb.flush()
        plt.close(fig)

    def _log_images(self, images, labels, outputs):
        n = min(images.shape[0], 8)
        imgs = images[:n]
        lbls = labels[:n]
        preds = outputs[:n].argmax(dim=1)

        imgs = imgs * self.inv_std + self.inv_mean
        imgs = torch.clamp(imgs, 0, 1)

        ncols = 4
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(3 * ncols, 3.5 * nrows)
        )
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
                ax.set_title(
                    f"GT:{gt} / P:{pd}",
                    fontsize=11,
                    color=color,
                    fontweight="bold",
                )

        fig.suptitle(f"Epoch {self.current_epoch}", fontsize=14)
        fig.tight_layout()

        tb = self.logger.experiment
        tb.add_figure("val/sample_predictions", fig, self.global_step)
        tb.flush()
        plt.close(fig)

    def _accuracy(self, output, target, topk=(1,)):
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
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=4,
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


# ============================================================================
# Data Module — NFS + optional local SSD prefetch
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
    def __init__(
        self,
        nfs_data_dir,
        batch_size,
        num_workers,
        prefetch_local=False,
        local_cache_dir=None,
    ):
        super().__init__()
        self.nfs_data_dir = nfs_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_local = prefetch_local
        self.local_cache_dir = local_cache_dir or os.path.join(
            "/tmp", str(os.getuid()), "imagenet1k_cache"
        )
        # 讓 prepare_data 在每個節點的 local rank 0 各跑一次
        self.prepare_data_per_node = True

    def prepare_data(self):
        """
        在每個節點的 local rank 0 上執行。
        1) 確認 NFS 上已有 HF dataset cache（首次會下載）
        2) 若 --prefetch-local，將 NFS cache 複製到 /tmp/$UID
        """
        load_dataset(
            "ILSVRC/imagenet-1k", cache_dir=self.nfs_data_dir
        )
        if self.prefetch_local:
            prefetch_nfs_to_local(self.nfs_data_dir, self.local_cache_dir)

    def setup(self, stage=None):
        """
        在所有 rank 上執行（Lightning 會在 prepare_data 完成後加 barrier）。
        """
        effective_dir = (
            self.local_cache_dir if self.prefetch_local else self.nfs_data_dir
        )

        train_tf = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4),
                transforms.ToTensor(),
                transforms.Normalize(
                    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
                ),
            ]
        )
        val_tf = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
                ),
            ]
        )

        ds = load_dataset(
            "ILSVRC/imagenet-1k", cache_dir=effective_dir
        )
        self.train_ds = HFImageNetDataset(ds["train"], transform=train_tf)
        self.val_ds = HFImageNetDataset(ds["validation"], transform=val_tf)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,  # Lightning 在 DDP 下自動替換為 DistributedSampler
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
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
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train CliffordNet on ImageNet-1k (multi-node DDP)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./imagenet1k",
        help="NFS path where HuggingFace caches ImageNet-1k",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=["nano", "small", "base", "large"],
        help="Model size variant",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=24,
        help="Batch size per GPU",
    )
    parser.add_argument("--epochs", type=int, default=999)
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Base learning rate (consider linear scaling with world size)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs per node",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=1,
        help="Number of nodes",
    )
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="DataLoader workers per GPU process",
    )
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output dir for checkpoints/logs (should be on NFS for multi-node)",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Checkpoint path to resume from"
    )
    # ---- Prefetch ----
    parser.add_argument(
        "--prefetch-local",
        action="store_true",
        help="Copy HF cache from NFS to /tmp/$UID before training",
    )
    parser.add_argument(
        "--local-cache-dir",
        type=str,
        default=None,
        help="Local SSD path for prefetch (default: /tmp/$UID/imagenet_cache)",
    )

    args = parser.parse_args()

    L.seed_everything(42)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

    data = ImageNet1kDataModule(
        nfs_data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_local=args.prefetch_local,
        local_cache_dir=args.local_cache_dir,
    )

    model = CliffordNetLightning(
        model_size=args.model_size,
        num_classes=1000,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
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
        num_nodes=args.num_nodes,
        strategy="ddp",
        precision="bf16-mixed",
        max_epochs=args.epochs,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor("step"),
            RichProgressBar(),
        ],
        logger=TensorBoardLogger(
            args.output_dir, name="cliffordnet_imagenet1k"
        ),
        log_every_n_steps=10,
    )

    trainer.fit(model, data, ckpt_path=args.resume)


if __name__ == "__main__":
    main()