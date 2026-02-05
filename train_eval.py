from __future__ import annotations

from typing import Dict
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
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

    best_val_acc = 0.0
    best_val_loss = 1e9

    for epoch in range(cfg.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg.epochs}", leave=False)
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()

            pbar.set_postfix(loss=float(loss.item()))

        val_metrics = evaluate(model, val_loader, device)
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_val_loss = val_metrics["loss"]

    return {"best_val_acc": best_val_acc, "best_val_loss": best_val_loss}


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
