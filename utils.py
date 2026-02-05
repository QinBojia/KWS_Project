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
    for p in model.parameters():
        total += p.numel() * p.element_size()
    return total


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
