from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import yaml


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    win_length: int = 512
    hop_length: int = 256
    n_fft: int = 512
    n_mels: int = 16
    n_mfcc: int = 13
    f_min: float = 20.0
    f_max: Optional[float] = None
    fixed_num_samples: int = 16000 + 256  # 16256 -> 62 frames

    @property
    def n_frames(self) -> int:
        return (self.fixed_num_samples - self.win_length) // self.hop_length + 1

    @property
    def input_shape(self) -> Tuple[int, int, int, int]:
        """Returns (B, C, T, F) input shape for the model."""
        return (1, 1, self.n_frames, self.n_mfcc)


@dataclass
class TrainConfig:
    batch_size: int = 512
    epochs: int = 15
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 123
    num_workers: int = 8

    train_device: str = "cuda"
    infer_device: str = "cpu"

    val_every: int = 1
    use_amp: bool = True
    cudnn_benchmark: bool = True
    pin_memory: bool = True
    prefetch_factor: int = 4
    persistent_workers: bool = True
    non_blocking: bool = True

    early_stop_patience: int = 5
    early_stop_min_delta: float = 0.0


@dataclass
class ArchConfig:
    """Architecture definition, loadable from YAML."""
    name: str = "unnamed"
    model_type: str = "custom_dscnn"  # "custom_dscnn" or "mobilenet"

    # CustomDSCNN parameters
    stem_ch: int = 16
    stem_stride: int = 2
    block_cfg: List[Tuple[int, int]] = field(default_factory=list)

    # MobileNetStyleKWS parameters
    width_mult: float = 1.0
    depth_mult: float = 1.0

    # Common
    num_classes: int = 8
    dropout: float = 0.1


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    name: str = "unnamed"
    arch: ArchConfig = field(default_factory=ArchConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_config(yaml_path: str) -> ExperimentConfig:
    """Load experiment config from a YAML file."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    name = raw.get("name", "unnamed")

    # Parse audio config
    audio_dict = raw.get("audio", {})
    audio = AudioConfig(**{k: v for k, v in audio_dict.items() if k in AudioConfig.__dataclass_fields__})

    # Parse train config
    train_dict = raw.get("train", {})
    train = TrainConfig(**{k: v for k, v in train_dict.items() if k in TrainConfig.__dataclass_fields__})

    # Parse architecture config
    model_dict = raw.get("model", {})
    model_type = model_dict.get("type", "custom_dscnn")
    block_cfg_raw = model_dict.get("block_cfg", [])
    block_cfg = [tuple(b) for b in block_cfg_raw] if block_cfg_raw else []

    arch = ArchConfig(
        name=name,
        model_type=model_type,
        stem_ch=model_dict.get("stem_ch", 16),
        stem_stride=model_dict.get("stem_stride", 2),
        block_cfg=block_cfg,
        width_mult=model_dict.get("width_mult", 1.0),
        depth_mult=model_dict.get("depth_mult", 1.0),
        num_classes=model_dict.get("num_classes", 8),
        dropout=model_dict.get("dropout", 0.1),
    )

    return ExperimentConfig(name=name, arch=arch, audio=audio, train=train)


def build_model(arch: ArchConfig):
    """Instantiate a model from an ArchConfig."""
    from kws.models import CustomDSCNN, MobileNetStyleKWS

    if arch.model_type == "custom_dscnn":
        return CustomDSCNN(
            stem_ch=arch.stem_ch,
            stem_stride=arch.stem_stride,
            block_cfg=arch.block_cfg,
            num_classes=arch.num_classes,
            dropout=arch.dropout,
        )
    elif arch.model_type == "mobilenet":
        return MobileNetStyleKWS(
            num_classes=arch.num_classes,
            width_mult=arch.width_mult,
            dropout=arch.dropout,
            depth_mult=arch.depth_mult,
        )
    else:
        raise ValueError(f"Unknown model_type: {arch.model_type}")
