from dataclasses import dataclass
from typing import Optional


@dataclass
class AudioConfig:
    # Paper: 16 kHz
    sample_rate: int = 16000

    # Paper: frame length = 32 ms => 512 samples @ 16k
    win_length: int = 512

    # Paper: stride = 256 samples (16 ms)
    hop_length: int = 256

    # Typically same as win_length
    n_fft: int = 512

    # Paper: 16 mel filterbank energies -> log -> DCT
    n_mels: int = 16

    # Paper: keep 13 MFCC coefficients
    n_mfcc: int = 13

    f_min: float = 20.0
    f_max: Optional[float] = None

    # Force output to be 62 frames:
    # 1s wave is 16000 samples, but to include the last partial frame,
    # pad an extra hop (256) -> 16256
    # This matches the paper statement about padding to get 62 frames.
    fixed_num_samples: int = 16000 + 256  # 16256


@dataclass
class ModelConfig:
    # 12-class KWS is the common setup: 10 keywords + unknown + silence
    num_classes: int = 8
    width_mult: float = 1.0
    dropout: float = 0.1
    depth_mult: float = 1.0


@dataclass
class TrainConfig:
    batch_size: int = 512
    epochs: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 123
    num_workers: int = 8

    train_device: str = "cuda"   # 训练用
    infer_device: str = "cpu"    # 推理/评估/计时用



@dataclass
class ExperimentConfig:
    name: str
    audio: AudioConfig
    model: ModelConfig
    train: TrainConfig
