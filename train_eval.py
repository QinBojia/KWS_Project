from __future__ import annotations

from typing import Dict
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
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


def train_one_experiment(model: nn.Module, train_loader, val_loader, cfg, device: str) -> Dict[str, float]:
    """
    训练 + 返回最好的 val acc
    """
    model.to(device)
    ce = nn.CrossEntropyLoss()
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    try:
        scaler = GradScaler(device_type="cuda", enabled=cfg.use_amp and device.startswith("cuda"))
    except TypeError:
        scaler = GradScaler(enabled=cfg.use_amp and device.startswith("cuda"))

    best_val_acc = 0.0
    best_val_loss = 1e9
    best_epoch = -1
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

            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=scaler.is_enabled()):
                logits = model(x)
                loss = ce(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            pbar.set_postfix(loss=float(loss.item()))

        # skip validation on some epochs to save time
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
            wait = 0
        else:
            wait += 1
            if patience > 0 and wait >= patience:
                break

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
    """
    简单基准：测平均推理时间（ms）
    - 对比不同模型大小、不同量化时非常有用
    """
    model.eval()
    model.to(device)
    x = example_input.to(device)

    # warmup
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
