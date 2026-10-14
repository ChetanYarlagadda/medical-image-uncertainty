"""ResNet-18 classifier with a dropout head, GPU-side augmentation, training with early stopping, inference.

Images stay on the GPU as uint8 tensors (n, h, w, 3); every batch is augmented, scaled and normalised there, so
the loader needs no worker processes. Training uses bfloat16 autocast and keeps the checkpoint with the lowest
validation negative log-likelihood.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class TrainConfig:
    """Settings of one ensemble member."""

    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 128
    max_epochs: int = 12
    patience: int = 3
    dropout: float = 0.5
    class_weight_power: float = 0.0  # 0 = plain cross-entropy; 0.5 = weights proportional to n_c ** -0.5
    pretrained: bool = True


class Classifier(nn.Module):
    """ImageNet ResNet-18 backbone, then dropout and a linear layer (the dropout is reused for MC dropout)."""

    def __init__(self, n_classes: int, dropout: float = 0.5, pretrained: bool = True) -> None:
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        net.fc = nn.Identity()
        self.backbone = net
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, n_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate 512-d features (global-average-pooled layer4)."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(self.features(x)))


def to_float(x: torch.Tensor) -> torch.Tensor:
    """uint8 (n, h, w, 3) -> float (n, 3, h, w) in [0, 1]."""
    return x.permute(0, 3, 1, 2).float().div_(255.0)


def normalize(x: torch.Tensor) -> torch.Tensor:
    """ImageNet normalisation of a float (n, 3, h, w) batch in [0, 1]."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def augment(x: torch.Tensor, gen: torch.Generator, pad: int = 4) -> torch.Tensor:
    """Geometric augmentation of a float batch: random flips, 90-degree rotations and shifts of up to ``pad`` pixels.

    Photometric changes (brightness, contrast, blur, noise) are deliberately left out, so the corruption tests
    measure robustness to shifts the model never saw in training.
    """
    n, _, h, w = x.shape
    dev = x.device
    flip_h = torch.rand(n, device=dev, generator=gen) < 0.5
    flip_v = torch.rand(n, device=dev, generator=gen) < 0.5
    x = torch.where(flip_h.view(n, 1, 1, 1), x.flip(3), x)
    x = torch.where(flip_v.view(n, 1, 1, 1), x.flip(2), x)
    k = torch.randint(0, 4, (n,), device=dev, generator=gen)
    x = torch.where((k % 2 == 1).view(n, 1, 1, 1), x.transpose(2, 3), x)  # transpose + flips = all 8 symmetries
    padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    dx = torch.randint(0, 2 * pad + 1, (n,), device=dev, generator=gen)
    dy = torch.randint(0, 2 * pad + 1, (n,), device=dev, generator=gen)
    rows = (dy.view(n, 1) + torch.arange(h, device=dev).view(1, h))  # (n, h)
    cols = (dx.view(n, 1) + torch.arange(w, device=dev).view(1, w))  # (n, w)
    idx_r = rows.view(n, 1, h, 1).expand(n, 3, h, w + 2 * pad)
    x = padded.gather(2, idx_r)
    idx_c = cols.view(n, 1, 1, w).expand(n, 3, h, w)
    return x.gather(3, idx_c)


def class_weights(y: np.ndarray, n_classes: int, power: float) -> torch.Tensor | None:
    """Loss weights proportional to n_c ** -power, scaled to a mean of 1 over training images (None for power 0)."""
    if power == 0:
        return None
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = counts ** -power
    w = w / (w[y].mean())
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def predict(model: Classifier, x: torch.Tensor, batch_size: int = 1024,
            transform: Callable[[torch.Tensor, int], torch.Tensor] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Logits and penultimate features of uint8 images ``x`` (on the model's device), dropout off.

    ``transform(batch, start)`` is applied to each float [0, 1] batch before normalisation (used for corruptions;
    ``start`` is the batch offset so random corruptions are reproducible per image).
    """
    model.eval()
    logits, feats = [], []
    for i in range(0, len(x), batch_size):
        xb = to_float(x[i:i + batch_size])
        if transform is not None:
            xb = transform(xb, i).clamp_(0, 1)
        xb = normalize(xb).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=xb.is_cuda):
            f = model.features(xb)
        f = f.float()
        feats.append(f.cpu())
        logits.append(model.fc(f).cpu())
    return torch.cat(logits).numpy(), torch.cat(feats).numpy()


def evaluate_nll(model: Classifier, x: torch.Tensor, y: torch.Tensor,
                 weight: torch.Tensor | None = None) -> tuple[float, float]:
    """Mean (class-weighted, if ``weight`` is given) negative log-likelihood and accuracy on a held-out set."""
    logits, _ = predict(model, x)
    lg = torch.from_numpy(logits)
    yc = y.cpu()
    w = weight.cpu() if weight is not None else None
    return float(F.cross_entropy(lg, yc, weight=w)), float((lg.argmax(1) == yc).float().mean())


def train(model: Classifier, x_tr: torch.Tensor, y_tr: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor,
          cfg: TrainConfig, seed: int, log: Callable[[str], None] = print) -> tuple[dict, pd.DataFrame]:
    """Fine-tune ``model`` with AdamW, one-cycle cosine schedule and early stopping on validation NLL.

    The stopping loss is the training objective on the validation images: plain NLL, or the class-weighted NLL
    when ``cfg.class_weight_power`` > 0.

    Returns the best state dict (on the CPU) and the per-epoch history.
    """
    dev = x_tr.device
    torch.manual_seed(seed)
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    model.to(dev).to(memory_format=torch.channels_last)
    n = len(y_tr)
    steps_per_epoch = (n + cfg.batch_size - 1) // cfg.batch_size
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.max_epochs * steps_per_epoch,
                                                pct_start=0.1, anneal_strategy="cos")
    n_classes = model.fc.out_features
    w = class_weights(y_tr.cpu().numpy(), n_classes, cfg.class_weight_power)
    w = w.to(dev) if w is not None else None
    best, best_nll, wait, hist = None, float("inf"), 0, []
    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.perf_counter()
        model.train()
        perm = torch.randperm(n, device=dev, generator=gen)
        total = 0.0
        for i in range(steps_per_epoch):
            idx = perm[i * cfg.batch_size:(i + 1) * cfg.batch_size]
            xb = normalize(augment(to_float(x_tr[idx]), gen)).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=xb.is_cuda):
                loss = F.cross_entropy(model(xb).float(), y_tr[idx], weight=w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.detach()) * len(idx)
        val_nll, val_acc = evaluate_nll(model, x_val, y_val, w)
        secs = time.perf_counter() - t0
        hist.append({"epoch": epoch, "train_loss": total / n, "val_nll": val_nll, "val_acc": val_acc, "seconds": secs})
        log(f"  epoch {epoch:2d}: train loss {total / n:.4f}, val NLL {val_nll:.4f}, val acc {val_acc:.4f} ({secs:.1f} s)")
        if val_nll < best_nll - 1e-4:
            best_nll, wait = val_nll, 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    model.load_state_dict(best)
    return best, pd.DataFrame(hist)
