from __future__ import annotations

import math
import torch
import torch.nn as nn


def _make_divisible(v: float, divisor: int = 8) -> int:
    """Round channel count to nearest multiple of divisor for hardware alignment."""
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

        base_cfg = [
            (c32, 1),
            (c64, 1),
            (c128, 2),
            (c128, 1),
            (c256, 2),
            (c256, 1),
            (c512, 2),
            (c512, 1),
            (c512, 1),
            (c512, 1),
            (c512, 1),
            (c512, 1),
            (c512, 1),
        ]

        fixed_part = base_cfg[:8]
        repeat_part = base_cfg[8:]

        base_repeat = len(repeat_part)
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


# ═══════════════════════════════════════════════════════════════════════
# TENet: Temporal Efficient Network (Li et al., Interspeech 2020)
# Paper: arXiv:2010.09960
# GitHub: https://github.com/Interlagos/TENet-kws (TensorFlow)
#
# Key idea: 1D temporal convolutions with MobileNetV2-style inverted
# bottleneck blocks. Treats MFCC frequency bins as input channels.
# Block: PW-expand → DW-temporal → PW-project(linear) + residual
# ═══════════════════════════════════════════════════════════════════════


class InvertedBottleneck1D(nn.Module):
    """
    MobileNetV2-style inverted bottleneck for 1D temporal signals.

    Structure: PW-expand → DW-temporal(k) → PW-project (linear, no ReLU)
    Residual:  identity if stride=1 and in_ch==out_ch, else 1x1 projection.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 expand_ratio: int = 3, kernel_size: int = 9):
        super().__init__()
        hidden_ch = in_ch * expand_ratio
        self.use_res = (stride == 1 and in_ch == out_ch)

        layers = []
        # Pointwise expansion
        if expand_ratio != 1:
            layers += [
                nn.Conv1d(in_ch, hidden_ch, 1, bias=False),
                nn.BatchNorm1d(hidden_ch),
                nn.ReLU(inplace=True),
            ]
        # Depthwise temporal convolution
        layers += [
            nn.Conv1d(hidden_ch, hidden_ch, kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=hidden_ch, bias=False),
            nn.BatchNorm1d(hidden_ch),
            nn.ReLU(inplace=True),
        ]
        # Pointwise projection (linear — no activation)
        layers += [
            nn.Conv1d(hidden_ch, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
        ]
        self.conv = nn.Sequential(*layers)

        # Projection shortcut for dimension mismatch
        self.shortcut = None
        if not self.use_res:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.use_res:
            out = out + x
        elif self.shortcut is not None:
            out = out + self.shortcut(x)
        return out


class TENet(nn.Module):
    """
    Temporal Efficient Network for keyword spotting.

    Args:
        n_channels: list of [stem_ch, block0_ch, block1_ch, ...]
        n_strides:  list of strides for each block (applied on first IBB)
        n_ratios:   list of expansion ratios for each block
        n_layers:   list of number of IBBs per block
        num_classes: number of output classes
        in_channels: number of input channels (MFCC coefficients)
        kernel_size: depthwise temporal kernel size
        dropout: dropout probability
    """

    def __init__(self, n_channels: list, n_strides: list, n_ratios: list,
                 n_layers: list, num_classes: int = 8, in_channels: int = 13,
                 kernel_size: int = 9, dropout: float = 0.1):
        super().__init__()
        assert len(n_strides) == len(n_ratios) == len(n_layers)
        n_blocks = len(n_strides)
        assert len(n_channels) == n_blocks + 1

        # Stem: project input channels to first channel dim
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, n_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm1d(n_channels[0]),
            nn.ReLU(inplace=True),
        )

        # Build blocks
        blocks = []
        in_ch = n_channels[0]
        for i in range(n_blocks):
            out_ch = n_channels[i + 1]
            for j in range(n_layers[i]):
                stride = n_strides[i] if j == 0 else 1
                blocks.append(InvertedBottleneck1D(
                    in_ch, out_ch, stride=stride,
                    expand_ratio=n_ratios[i], kernel_size=kernel_size,
                ))
                in_ch = out_ch
        self.features = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T, F) where T=62 frames, F=13 MFCC
        # Reshape to (B, F, T) — treat freq bins as channels
        x = x.squeeze(1)        # (B, T, F)
        x = x.permute(0, 2, 1)  # (B, F, T)
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        x = self.drop(x)
        return self.fc(x)


# ═══════════════════════════════════════════════════════════════════════
# LiCoNet: Linearized Convolution Network (Yang et al., Meta 2022)
# Paper: arXiv:2211.04635
#
# Key idea: Bottleneck blocks with DW-spatial conv first, then PW
# expand/project. At inference, streaming convolutions can be
# linearized to FC layers for DSP efficiency.
# Block: DW-spatial(k) → PW-expand → PW-project(linear) + residual
# ═══════════════════════════════════════════════════════════════════════


class LiCoBlock(nn.Module):
    """
    LiCoNet block: DW-spatial → PW-expand → PW-project (linear).

    Unlike MobileNetV2 (expand-first), LiCoNet applies the spatial
    depthwise convolution first, then expands and projects.
    """

    def __init__(self, channels: int, kernel_size: int = 5,
                 expansion: int = 6, stride: int = 1):
        super().__init__()
        expanded = channels * expansion
        self.use_res = (stride == 1)

        # Layer 1: Depthwise spatial convolution
        self.spatial = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=channels, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU6(inplace=True),
        )
        # Layer 2: Pointwise expansion
        self.expand = nn.Sequential(
            nn.Conv1d(channels, expanded, 1, bias=False),
            nn.BatchNorm1d(expanded),
            nn.ReLU6(inplace=True),
        )
        # Layer 3: Pointwise projection (linear — no activation)
        self.project = nn.Sequential(
            nn.Conv1d(expanded, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.spatial(x)
        out = self.expand(out)
        out = self.project(out)
        if self.use_res:
            out = out + x
        return out


class LiCoNet(nn.Module):
    """
    LiCoNet: Linearized Convolution Network.

    Args:
        width: bottleneck channel width
        n_blocks: number of LiCoBlocks
        kernel_size: depthwise temporal kernel size
        expansion: channel expansion ratio
        num_classes: output classes
        in_channels: input MFCC channels
        dropout: dropout probability
        strides: list of strides per block (default: all stride-1)
    """

    def __init__(self, width: int = 72, n_blocks: int = 5,
                 kernel_size: int = 5, expansion: int = 6,
                 num_classes: int = 8, in_channels: int = 13,
                 dropout: float = 0.1, strides: list = None):
        super().__init__()
        if strides is None:
            strides = [1] * n_blocks
        assert len(strides) == n_blocks

        # Stem: project input to bottleneck width
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, width, 1, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU6(inplace=True),
        )

        blocks = []
        for i in range(n_blocks):
            blocks.append(LiCoBlock(width, kernel_size, expansion, stride=strides[i]))
        self.blocks = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T, F) → (B, F, T)
        x = x.squeeze(1)        # (B, T, F)
        x = x.permute(0, 2, 1)  # (B, F, T)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        x = self.drop(x)
        return self.fc(x)


# ═══════════════════════════════════════════════════════════════════════
# BC-ResNet: Broadcasted Residual Network (Kim et al., Interspeech 2021)
# Paper: arXiv:2106.04140
# GitHub: https://github.com/Qualcomm-AI-research/bcresnet
#
# Key idea: Most computation uses 1D temporal convolutions, but a
# "broadcasted residual" expands 1D features back to 2D for the skip
# connection, achieving 2D representation at 1D cost.
# Uses SubSpectralNorm for frequency-aware normalization.
# ═══════════════════════════════════════════════════════════════════════


class SubSpectralNorm(nn.Module):
    """
    Sub-Spectral Normalization: applies batch norm independently to
    sub-bands of the frequency dimension.
    """

    def __init__(self, channels: int, sub_bands: int):
        super().__init__()
        self.sub_bands = sub_bands
        self.bn = nn.BatchNorm2d(channels * sub_bands)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, T)
        B, C, F_orig, T = x.shape
        S = self.sub_bands
        # Always pad frequency dim to next multiple of S (no-op if already divisible)
        pad = (S - F_orig % S) % S
        x = nn.functional.pad(x, (0, 0, 0, pad))
        F_padded = F_orig + pad
        x = x.reshape(B, C * S, F_padded // S, T)
        x = self.bn(x)
        x = x.reshape(B, C, F_padded, T)
        return x[:, :, :F_orig, :]


class BCResBlock(nn.Module):
    """
    BC-ResNet block with broadcasted residual connection.

    Uses 2D depthwise conv along frequency, then 1D temporal depthwise
    conv (after averaging frequency). Temporal output is broadcast
    back to 2D for residual addition.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 dw_kernel: int = 3, sub_bands: int = 5):
        super().__init__()
        self.match_dim = (in_ch != out_ch)

        # Frequency-wise 2D depthwise conv
        self.freq_dw = nn.Conv2d(in_ch, in_ch, (dw_kernel, 1),
                                  padding=(dw_kernel // 2, 0),
                                  groups=in_ch, bias=False)
        self.freq_ssn = SubSpectralNorm(in_ch, sub_bands)

        # Temporal 1D depthwise conv (after frequency averaging)
        self.temp_dw = nn.Conv1d(in_ch, in_ch, dw_kernel, stride=stride,
                                  padding=dw_kernel // 2, groups=in_ch, bias=False)
        self.temp_bn = nn.BatchNorm1d(in_ch)

        # Pointwise conv to change channels
        self.pw = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

        # Channel matching for residual
        self.skip = None
        if self.match_dim:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )

        self.stride = stride
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, T)
        residual = x

        # Frequency-wise processing (2D)
        out = self.freq_dw(x)
        out = self.freq_ssn(out)
        out = self.act(out)

        # Average over frequency → temporal processing (1D)
        out_1d = out.mean(dim=2)    # (B, C, T)
        out_1d = self.temp_dw(out_1d)
        out_1d = self.temp_bn(out_1d)
        out_1d = self.act(out_1d)

        # Broadcast 1D temporal back to 2D: (B, C, 1, T') + (B, C, F, T')
        if self.stride > 1:
            # Downsample 2D branch to match strided temporal output
            out = nn.functional.adaptive_avg_pool2d(
                out, (out.shape[2], out_1d.shape[-1]))
        out = out_1d.unsqueeze(2) + out

        # Pointwise channel mixing
        out = self.pw(out)

        # Residual
        if self.skip is not None:
            residual = self.skip(residual)
        if self.stride > 1:
            residual = nn.functional.adaptive_avg_pool2d(
                residual, (residual.shape[2], out.shape[3]))
        return self.act(out + residual)


class BCResNet(nn.Module):
    """
    Broadcasted Residual Network for keyword spotting.

    Args:
        channels_list: channels for each stage, e.g. [8, 12, 16, 20]
        layers_list: number of BCResBlocks per stage, e.g. [2, 2, 4, 4]
        strides_list: temporal stride per stage, e.g. [1, 1, 2, 2]
        num_classes: output classes
        in_channels: input channels (1 for single-channel MFCC)
        sub_bands: number of sub-bands for SubSpectralNorm
        dropout: dropout probability
    """

    def __init__(self, channels_list: list, layers_list: list,
                 strides_list: list = None, num_classes: int = 8,
                 in_channels: int = 1, sub_bands: int = 5,
                 dropout: float = 0.1):
        super().__init__()
        n_stages = len(channels_list)
        assert len(layers_list) == n_stages
        if strides_list is None:
            strides_list = [1] * n_stages
        assert len(strides_list) == n_stages

        # Stem: initial 2D convolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels_list[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(channels_list[0]),
            nn.ReLU(inplace=True),
        )

        # Build stages
        stages = []
        in_ch = channels_list[0]
        for i in range(n_stages):
            out_ch = channels_list[i]
            for j in range(layers_list[i]):
                stride = strides_list[i] if j == 0 else 1
                stages.append(BCResBlock(in_ch, out_ch, stride=stride,
                                          sub_bands=min(sub_bands, 13)))
                in_ch = out_ch
        self.features = nn.Sequential(*stages)

        # Head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T, F) → (B, 1, F, T) for 2D convolutions
        x = x.squeeze(1)         # (B, T, F)
        x = x.permute(0, 2, 1)   # (B, F, T)
        x = x.unsqueeze(1)       # (B, 1, F, T)
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)
