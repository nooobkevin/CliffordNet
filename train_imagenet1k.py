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

import wandb
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from datasets import load_dataset
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.mixup import Mixup
from timm.layers import DropPath, trunc_normal_
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
)
import lightning as L
import matplotlib.pyplot as plt
import os
import time as _time
import subprocess
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib

matplotlib.use("Agg")


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
    print(
        f"[Prefetch] Copying {nfs_dir} → {local_dir}  (may take a while) ...")

    ret = subprocess.run(
        ["rsync", "-a", "--info=progress2", f"{nfs_dir}/", f"{local_dir}/"]
    )
    if ret.returncode != 0:
        print("[Prefetch] rsync unavailable or failed, falling back to cp -a")
        subprocess.run(
            ["cp", "-a", f"{nfs_dir}/.", f"{local_dir}/"], check=True)

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
        self.register_buffer("_roll_idx", roll_idx)

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
        drop_path_rate=0.05,
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
        drop_path_rate=0.2,
    )


def cliffordnet_large(num_classes=1000):
    return CliffordNet(
        img_size=224,
        embed_dim=512,
        depth=24,
        shifts=[1, 2, 4, 8, 16],
        num_classes=num_classes,
        drop_path_rate=0.3,
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
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        mixup_prob=1.0,
        mixup_switch_prob=0.5,
        ema_decay=0.9999,
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

        # Mixup / CutMix (applied in training_step, not in DataLoader)
        self.mixup_fn = Mixup(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            prob=mixup_prob,
            switch_prob=mixup_switch_prob,
            num_classes=num_classes,
        )
        # With mixup, targets become soft labels → use soft cross-entropy
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # EMA model (updated manually in on_before_zero_grad)
        self.ema_decay = ema_decay
        self._ema_model = None  # lazily initialized on first step

        self.register_buffer(
            "inv_mean", torch.tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "inv_std", torch.tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1)
        )

        self.val_preds = []
        self.val_labels = []
        self._train_step_start = None

    def forward(self, x):
        return self.model(x.contiguous(memory_format=torch.channels_last))

    def on_train_batch_start(self, batch, batch_idx):
        self._train_step_start = _time.monotonic()

    def training_step(self, batch, batch_idx):
        images, labels = batch
        # Apply Mixup / CutMix (produces soft labels)
        images, labels = self.mixup_fn(images, labels)
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        # For logging accuracy, use hard labels (argmax of soft targets)
        hard_labels = labels.argmax(dim=1)
        acc1, acc5 = self._accuracy(outputs, hard_labels, topk=(1, 5))

        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log("train/acc1", acc1, prog_bar=True, sync_dist=True)
        self.log("train/acc5", acc5, sync_dist=True)

        # Log learning rate
        opt = self.optimizers()
        self.log("train/lr", opt.param_groups[0]["lr"], prog_bar=False)

        # Throughput logging
        if self._train_step_start is not None:
            elapsed = _time.monotonic() - self._train_step_start
            if elapsed > 0:
                bs = images.shape[0]
                world_size = self.trainer.world_size
                imgs_per_sec = bs * world_size / elapsed
                self.log("perf/images_per_sec", imgs_per_sec, prog_bar=False)
                self.log("perf/step_time_ms", elapsed * 1000, prog_bar=False)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update EMA weights after optimizer step completes (safe with torch.compile)."""
        self._update_ema()

    def _init_ema(self):
        """Lazily create EMA state dicts (avoids doubling memory at init)."""
        if self._ema_model is not None:
            return
        # Store EMA as a flat dict of tensors (no extra nn.Module overhead)
        self._ema_model = {
            k: v.clone().detach() for k, v in self.model.state_dict().items()
        }

    def _update_ema(self):
        self._init_ema()
        d = self.ema_decay
        model_sd = self.model.state_dict()
        for k in self._ema_model:
            v = model_sd[k].detach()
            if v.is_floating_point():
                self._ema_model[k].lerp_(v, 1 - d)
            else:
                # Non-float buffers (e.g. num_batches_tracked): just copy
                self._ema_model[k].copy_(v)

    def _swap_ema(self):
        """Swap model weights with EMA weights (call before/after val)."""
        if self._ema_model is None:
            return
        model_sd = self.model.state_dict()
        for k in self._ema_model:
            model_sd[k], self._ema_model[k] = (
                self._ema_model[k],
                model_sd[k],
            )
        self.model.load_state_dict(model_sd)

    def on_validation_start(self):
        self._swap_ema()  # use EMA weights for validation

    def on_validation_end(self):
        self._swap_ema()  # swap back to training weights

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
        # Confusion matrix is expensive for 1000 classes (8MB all_reduce).
        # Only compute at actual epoch boundaries, not every val_check_interval.
        # We check a flag set in on_train_epoch_end to stay safe with DDP
        # (all ranks must agree on whether to all_reduce or not).
        # ------------------------------------------------------------------
        is_epoch_end = getattr(self, "_at_epoch_boundary", True)
        num_cls = self.hparams.num_classes

        if is_epoch_end and len(self.val_preds) > 0:
            all_preds = torch.cat(self.val_preds).numpy()
            all_labels = torch.cat(self.val_labels).numpy()
            cm_local = confusion_matrix(
                all_labels, all_preds, labels=range(num_cls))

            cm_tensor = torch.tensor(
                cm_local, dtype=torch.int64, device=self.device)

            if dist.is_initialized() and dist.get_world_size() > 1:
                dist.all_reduce(cm_tensor, op=dist.ReduceOp.SUM)

            if self.trainer.is_global_zero:
                cm = cm_tensor.cpu().numpy()

                if num_cls > 100:
                    self._log_top_confused_pairs(cm, top_k=20)
                else:
                    cm_normalized = cm.astype("float") / (
                        cm.sum(axis=1, keepdims=True) + 1e-8
                    )
                    fig, ax = plt.subplots(figsize=(12, 10))
                    sns.heatmap(cm_normalized, ax=ax, cmap="Blues", cbar=True)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("True")
                    ax.set_title(
                        f"Confusion Matrix — Epoch {self.current_epoch}")
                    fig.tight_layout()

                    self.logger.experiment.log(
                        {"val/confusion_matrix": wandb.Image(fig)},
                    )
                    plt.close(fig)

        self.val_preds.clear()
        self.val_labels.clear()
        self._at_epoch_boundary = False

    def on_train_epoch_end(self):
        # Mark that the next validation is at an epoch boundary
        self._at_epoch_boundary = True

    def _log_top_confused_pairs(self, cm, top_k=20):
        """Log top confused class pairs for large datasets like ImageNet."""
        cm_no_diag = cm.copy()
        np.fill_diagonal(cm_no_diag, 0)

        flat_indices = np.argsort(cm_no_diag.ravel())[-top_k:][::-1]
        top_pairs = [(idx // cm.shape[1], idx % cm.shape[1])
                     for idx in flat_indices]

        fig, ax = plt.subplots(figsize=(12, 8))
        pair_labels = [f"{true}->{pred}" for true, pred in top_pairs]
        pair_counts = [cm[true, pred] for true, pred in top_pairs]

        bars = ax.barh(range(len(pair_labels)), pair_counts, color="steelblue")
        ax.set_yticks(range(len(pair_labels)))
        ax.set_yticklabels(pair_labels)
        ax.set_xlabel("Count")
        ax.set_title(
            f"Top {top_k} Confused Class Pairs — Epoch {self.current_epoch}")
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

        self.logger.experiment.log(
            {"val/top_confused_pairs": wandb.Image(fig)},
        )
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
                ax.set_title(
                    f"GT:{gt} / P:{pd}",
                    fontsize=11,
                    color=color,
                    fontweight="bold",
                )

        fig.suptitle(f"Epoch {self.current_epoch}", fontsize=14)
        fig.tight_layout()

        self.logger.experiment.log(
            {"val/sample_predictions": wandb.Image(fig)},
        )
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
        # Separate params: no weight decay for norm layers, biases, and layer scale
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or "bias" in name or "norm" in name or "gamma" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.hparams.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.hparams.learning_rate,
        )

        # Linear warmup for 1 epoch, then cosine decay to eta_min over
        # the remaining epochs.  Both schedulers step per-step (not per-epoch)
        # so the curve is smooth.
        # We estimate steps_per_epoch from the trainer if available.
        steps_per_epoch = (
            self.trainer.estimated_stepping_batches // self.hparams.max_epochs
        )
        warmup_steps = 1 * steps_per_epoch
        total_steps = self.trainer.estimated_stepping_batches

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-3,  # LR starts at learning_rate * 1e-3
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
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
        load_dataset("ILSVRC/imagenet-1k", cache_dir=self.nfs_data_dir)
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
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ColorJitter(0.4, 0.4, 0.4),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN,
                                     IMAGENET_DEFAULT_STD),
                transforms.RandomErasing(p=0.25),
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

        ds = load_dataset("ILSVRC/imagenet-1k", cache_dir=effective_dir)
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


def auto_find_batch_size(
    model_cls,
    model_kwargs,
    max_batch_size=1024,
    min_batch_size=8,
    input_shape=(3, 224, 224),
    dtype=torch.bfloat16,
    device=None,
):
    """
    Binary search for the largest batch size that fits in GPU memory.
    Simulates the real training loop: forward + backward + optimizer.step()
    with AdamW and bf16 mixed precision, matching what Lightning actually does.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        print("[AutoBS] No CUDA device, returning min_batch_size")
        return min_batch_size

    # Build model exactly as training does: channels_last + compile
    model = model_cls(**model_kwargs).to(
        device=device, memory_format=torch.channels_last
    )
    model = torch.compile(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=0.05)

    # Warm-up: run one small forward+backward+step to trigger torch.compile
    # and allocate optimizer states, so they don't inflate later measurements.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        x_warmup = torch.randn(
            min_batch_size, *input_shape, device=device, dtype=dtype
        ).contiguous(memory_format=torch.channels_last)
        y_warmup = torch.randint(
            0,
            model_kwargs.get("num_classes", 1000),
            (min_batch_size,),
            device=device,
        )
        with torch.autocast(device_type="cuda", dtype=dtype):
            out = model(x_warmup)
            loss = criterion(out, y_warmup)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        del x_warmup, y_warmup, out, loss
        torch.cuda.empty_cache()
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        print(
            f"[AutoBS] Even min_batch_size={min_batch_size} OOM during warmup: {e}")
        del model, criterion, optimizer
        torch.cuda.empty_cache()
        return min_batch_size

    # Binary search: probe with full forward + backward + optimizer.step
    best = min_batch_size
    lo, hi = min_batch_size, max_batch_size

    while lo <= hi:
        mid = (lo + hi) // 2
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        try:
            x = torch.randn(mid, *input_shape, device=device, dtype=dtype).contiguous(
                memory_format=torch.channels_last
            )
            y = torch.randint(
                0,
                model_kwargs.get("num_classes", 1000),
                (mid,),
                device=device,
            )
            with torch.autocast(device_type="cuda", dtype=dtype):
                out = model(x)
                loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            del x, y, out, loss
            torch.cuda.empty_cache()
            best = mid
            lo = mid + 1
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            hi = mid - 1

    del model, criterion, optimizer
    torch.cuda.empty_cache()

    # Round down to nearest multiple of 8 for tensor-core efficiency
    safe_bs = max(min_batch_size, ((int(best * 0.85)) // 8) * 8)
    print(f"[AutoBS] Max fit: {best}, using batch size: {safe_bs}")
    return safe_bs


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
        help="Batch size per GPU (0 = auto-detect)",
    )
    parser.add_argument("--epochs", type=int, default=300)
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
        "--val-check-interval",
        type=float,
        default=0.25,
        help="Run validation every N fraction of an epoch (e.g. 0.25 = 4x per epoch). "
        "Values > 1 are treated as number of training steps.",
    )
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
    # ---- Wandb ----
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="CliffordNet",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="nooobkevin",
        help="Wandb entity (team/user name)",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Wandb run name (default: auto-generated)",
    )
    parser.add_argument(
        "--wandb-offline",
        action="store_true",
        help="Run wandb in offline mode",
    )

    args = parser.parse_args()

    L.seed_everything(42)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

    # ---- Auto batch size ----
    # In DDP (torchrun), only local rank 0 runs the GPU memory probe.
    # Other ranks wait for the result via a temp file, because dist is not
    # yet initialized at this point (Lightning handles that later).
    if args.batch_size <= 0:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        bs_file = os.path.join(args.output_dir, ".auto_batch_size")
        os.makedirs(args.output_dir, exist_ok=True)

        if local_rank == 0:
            # Remove stale file from previous runs before probing
            if os.path.exists(bs_file):
                os.remove(bs_file)

            probe_device = torch.device("cuda:0")
            print(f"[AutoBS] Probing on {probe_device} ...")
            detected_bs = auto_find_batch_size(
                model_cls=CliffordNet,
                model_kwargs=_model_kwargs_for_size(args.model_size),
                device=probe_device,
            )
            # Write result so other ranks can read it
            with open(bs_file, "w") as f:
                f.write(str(detected_bs))
            print(f"[AutoBS] Batch size: {detected_bs}")
        else:
            # Wait for rank 0 to finish probing (poll up to 10 min)
            import time

            # First, wait for any stale file to be removed by rank 0
            time.sleep(2)
            for _ in range(600):
                if os.path.exists(bs_file):
                    break
                time.sleep(1)
            with open(bs_file, "r") as f:
                detected_bs = int(f.read().strip())
            print(
                f"[AutoBS] Rank {local_rank} received batch size: {detected_bs}")

        args.batch_size = detected_bs

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

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        save_dir=args.output_dir,
        offline=args.wandb_offline,
        log_model=False,
        config={
            "model_size": args.model_size,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "num_gpus": args.num_gpus,
            "num_nodes": args.num_nodes,
            "gradient_clip_val": args.gradient_clip_val,
        },
    )

    trainer = L.Trainer(
        accelerator="gpu",
        devices=args.num_gpus,
        num_nodes=args.num_nodes,
        strategy="ddp" if args.num_gpus > 1 or args.num_nodes > 1 else "auto",
        precision="bf16-mixed",
        max_epochs=args.epochs,
        gradient_clip_val=args.gradient_clip_val,
        val_check_interval=args.val_check_interval,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor("step"),
            RichProgressBar(),
        ],
        logger=wandb_logger,
        log_every_n_steps=10,
    )

    trainer.fit(model, data, ckpt_path=args.resume)


def _model_kwargs_for_size(size):
    """Return CliffordNet constructor kwargs for a given model size string."""
    configs = {
        "nano": dict(embed_dim=128, depth=8, shifts=[1, 2, 4, 8], drop_path_rate=0.05),
        "small": dict(embed_dim=192, depth=12, shifts=[1, 2, 4, 8], drop_path_rate=0.1),
        "base": dict(embed_dim=384, depth=16, shifts=[1, 2, 4, 8], drop_path_rate=0.2),
        "large": dict(
            embed_dim=512, depth=24, shifts=[1, 2, 4, 8, 16], drop_path_rate=0.3
        ),
    }
    kwargs = configs[size]
    kwargs["num_classes"] = 1000
    kwargs["img_size"] = 224
    kwargs["in_chans"] = 3
    return kwargs


if __name__ == "__main__":
    main()
