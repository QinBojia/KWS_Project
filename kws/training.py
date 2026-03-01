from __future__ import annotations

from typing import Dict, Optional
import time
import math

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from tqdm import tqdm
from torch.amp import autocast, GradScaler


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str, non_blocking: bool = False) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss()

    for x, y in loader:
        x = x.to(device, non_blocking=non_blocking)
        y = y.to(device, non_blocking=non_blocking)
        logits = model(x)
        loss = ce(logits, y)
        loss_sum += float(loss.item()) * x.size(0)

        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += x.size(0)

    return {
        "acc": correct / max(total, 1),
        "loss": loss_sum / max(total, 1),
    }


def _spec_augment(x: torch.Tensor, time_masks: int = 2, time_width: int = 5,
                   freq_masks: int = 1, freq_width: int = 2) -> torch.Tensor:
    """Apply SpecAugment on MFCC features.

    Args:
        x: (B, 1, T, F) MFCC tensor
        time_masks: number of time masks
        time_width: max width of each time mask
        freq_masks: number of frequency masks
        freq_width: max width of each frequency mask
    """
    B, C, T, F = x.shape
    x = x.clone()
    for _ in range(time_masks):
        t = torch.randint(0, max(time_width, 1), (1,)).item()
        t0 = torch.randint(0, max(T - t, 1), (1,)).item()
        x[:, :, t0:t0 + t, :] = 0.0
    for _ in range(freq_masks):
        f = torch.randint(0, max(freq_width, 1), (1,)).item()
        f0 = torch.randint(0, max(F - f, 1), (1,)).item()
        x[:, :, :, f0:f0 + f] = 0.0
    return x


def _mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """Mixup data augmentation. Returns mixed x, y_a, y_b, lam."""
    lam = torch.distributions.Beta(alpha, alpha).sample().item() if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def train_one_experiment(model: nn.Module, train_loader, val_loader, cfg, device: str) -> Dict[str, float]:
    """Train model and return best validation metrics.

    Supports optional training enhancements via cfg attributes:
        - scheduler: "none" | "cosine" (CosineAnnealingLR + linear warmup)
        - warmup_epochs: int, warmup epochs for cosine scheduler
        - label_smoothing: float, label smoothing factor
        - mixup_alpha: float, mixup alpha (0 = disabled)
        - spec_augment: bool, enable SpecAugment
        - spec_time_masks/spec_time_width/spec_freq_masks/spec_freq_width: SpecAugment params
    """
    model.to(device)

    # Loss function with optional label smoothing
    label_smoothing = getattr(cfg, "label_smoothing", 0.0) or 0.0
    ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # LR scheduler
    scheduler_name = getattr(cfg, "scheduler", "none") or "none"
    scheduler = None
    if scheduler_name == "cosine":
        warmup_epochs = getattr(cfg, "warmup_epochs", 5) or 5
        warmup_sched = LambdaLR(opt, lr_lambda=lambda ep: min(1.0, (ep + 1) / warmup_epochs))
        cosine_sched = CosineAnnealingLR(opt, T_max=cfg.epochs - warmup_epochs, eta_min=cfg.lr / 100)
        scheduler = SequentialLR(opt, schedulers=[warmup_sched, cosine_sched],
                                 milestones=[warmup_epochs])

    try:
        scaler = GradScaler(device_type="cuda", enabled=cfg.use_amp and device.startswith("cuda"))
    except TypeError:
        scaler = GradScaler(enabled=cfg.use_amp and device.startswith("cuda"))

    # Optional augmentation settings
    mixup_alpha = getattr(cfg, "mixup_alpha", 0.0) or 0.0
    use_spec_aug = getattr(cfg, "spec_augment", False)
    spec_cfg = {
        "time_masks": getattr(cfg, "spec_time_masks", 2),
        "time_width": getattr(cfg, "spec_time_width", 5),
        "freq_masks": getattr(cfg, "spec_freq_masks", 1),
        "freq_width": getattr(cfg, "spec_freq_width", 2),
    } if use_spec_aug else None

    best_val_acc = 0.0
    best_val_loss = 1e9
    best_epoch = -1
    best_state = None
    wait = 0
    patience = getattr(cfg, "early_stop_patience", 0) or 0
    min_delta = getattr(cfg, "early_stop_min_delta", 0.0) or 0.0

    total_epochs = cfg.epochs

    for epoch in range(total_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{total_epochs}", leave=False)
        for x, y in pbar:
            x = x.to(device, non_blocking=cfg.non_blocking)
            y = y.to(device, non_blocking=cfg.non_blocking)

            # SpecAugment
            if spec_cfg is not None:
                x = _spec_augment(x, **spec_cfg)

            # Mixup
            use_mixup = mixup_alpha > 0
            if use_mixup:
                x, y_a, y_b, lam = _mixup(x, y, alpha=mixup_alpha)

            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=scaler.is_enabled()):
                logits = model(x)
                if use_mixup:
                    loss = lam * ce(logits, y_a) + (1 - lam) * ce(logits, y_b)
                else:
                    loss = ce(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            pbar.set_postfix(loss=float(loss.item()))

        if scheduler is not None:
            scheduler.step()

        if ((epoch + 1) % cfg.val_every) != 0 and (epoch + 1) != total_epochs:
            continue

        val_metrics = evaluate(model, val_loader, device, non_blocking=cfg.non_blocking)

        improved = False
        if val_metrics["acc"] > best_val_acc + min_delta:
            improved = True
        elif abs(val_metrics["acc"] - best_val_acc) <= min_delta and val_metrics["loss"] < best_val_loss - min_delta:
            improved = True

        if improved:
            best_val_acc = val_metrics["acc"]
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if patience > 0 and wait >= patience:
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    epochs_ran = best_epoch if best_epoch != -1 else epoch + 1
    return {
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "epochs_ran": epochs_ran,
        "early_stopped": patience > 0 and wait >= patience,
    }


@torch.no_grad()
def benchmark_inference_ms(model: nn.Module, example_input: torch.Tensor, device: str, iters: int = 200) -> float:
    """Benchmark average inference latency in milliseconds."""
    model.eval()
    model.to(device)
    x = example_input.to(device)

    for _ in range(20):
        _ = model(x)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) * 1000.0 / iters
