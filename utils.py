import os
import random
import json
from typing import Dict, Any

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_bytes(model: torch.nn.Module) -> int:
    """
    估算模型权重占用（只算参数张量的字节数，不算优化器等）
    """
    total = 0
    for _, v in model.state_dict().items():
        if torch.is_tensor(v):
            total += v.numel() * v.element_size()
    return int(total)

def count_macc(model: torch.nn.Module, input_shape: tuple) -> int:
    """
    计算模型的 MACC (multiply-accumulate operations)。
    通过 hook 遍历所有 Conv2d / Linear 层，累加 MACC。
    input_shape: (B, C, H, W)，例如 (1, 1, 62, 13)
    """
    total_macc = 0
    hooks = []

    def _hook_conv2d(module, inp, out):
        nonlocal total_macc
        # MACC = Cout × Hout × Wout × (Cin/groups × Kh × Kw)
        batch, c_out, h_out, w_out = out.shape
        c_in = module.in_channels
        kh, kw = module.kernel_size
        groups = module.groups
        macc = c_out * h_out * w_out * (c_in // groups) * kh * kw
        total_macc += macc

    def _hook_linear(module, inp, out):
        nonlocal total_macc
        # MACC = in_features × out_features
        total_macc += module.in_features * module.out_features

    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(_hook_conv2d))
        elif isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(_hook_linear))

    device = next(model.parameters()).device
    x = torch.zeros(input_shape, device=device)
    with torch.no_grad():
        model.eval()
        model(x)

    for h in hooks:
        h.remove()

    return total_macc


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
