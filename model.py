from __future__ import annotations

import math
import torch
import torch.nn as nn


def _make_divisible(v: float, divisor: int = 8) -> int:
    """
    MobileNet 常用：把通道数对齐到 8 的倍数，方便硬件/内存对齐
    """
    return int(math.ceil(v / divisor) * divisor)


class DSConvBlock(nn.Module):
    """
    Depthwise Separable Conv block:
      depthwise 3x3 + BN + ReLU
      pointwise 1x1 + BN + ReLU
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.pw = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class MobileNetStyleKWS(nn.Module):
    def __init__(self, num_classes: int, width_mult: float = 1.0, dropout: float = 0.1, depth_mult: float = 1.0):
        super().__init__()

        # 通道按 width_mult 缩放
        c32 = _make_divisible(32 * width_mult)
        c64 = _make_divisible(64 * width_mult)
        c128 = _make_divisible(128 * width_mult)
        c256 = _make_divisible(256 * width_mult)
        c512 = _make_divisible(512 * width_mult)

        self.stem = nn.Sequential(
            nn.Conv2d(1, c32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c32),
            nn.ReLU(inplace=True),
        )

        # 13 个 DS blocks（常见 MobileNet 节奏：少数层 stride=2 下采样）
        base_cfg = [
            (c32, 1),
            (c64, 1),
            (c128, 2),
            (c128, 1),
            (c256, 2),
            (c256, 1),
            (c512, 2),
            (c512, 1),
            # 后面这些是重复段（可缩短）
            (c512, 1),
            (c512, 1),
            (c512, 1),
            (c512, 1),
            (c512, 1),
        ]

        # 让 depth_mult 只影响“重复段”的长度
        fixed_part = base_cfg[:8]  # 到第 8 个为止（含第一个 c512 stride=1）
        repeat_part = base_cfg[8:]  # 其余重复的 c512 stride=1

        # 计算重复段要保留多少层
        base_repeat = len(repeat_part)  # 原本 5 层
        k = max(1, int(round(base_repeat * depth_mult)))
        repeat_part = repeat_part[:k]

        cfg = fixed_part + repeat_part

        blocks = []
        in_ch = c32
        for out_ch, stride in cfg:
            blocks.append(DSConvBlock(in_ch, out_ch, stride=stride))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T, n_mels)
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.drop(x)
        return self.fc(x)


class CustomDSCNN(nn.Module):
    """
    Flexible DS-CNN with explicit per-block channel/stride config.
    Allows arbitrary architectures to target specific MACC budgets.

    Args:
        stem_ch: Number of output channels for the stem conv.
        stem_stride: Stride for the stem conv.
        block_cfg: List of (out_channels, stride) for each DS block.
        num_classes: Number of output classes.
        dropout: Dropout probability before FC.
    """

    def __init__(self, stem_ch: int, stem_stride: int,
                 block_cfg: list, num_classes: int = 8,
                 dropout: float = 0.1):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, stem_ch, 3, stride=stem_stride, padding=1, bias=False),
            nn.BatchNorm2d(stem_ch),
            nn.ReLU(inplace=True),
        )

        blocks = []
        in_ch = stem_ch
        for out_ch, stride in block_cfg:
            blocks.append(DSConvBlock(in_ch, out_ch, stride=stride))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.drop(x)
        return self.fc(x)
