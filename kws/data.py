from __future__ import annotations

from typing import List, Tuple, Dict, Optional
import os
import re

import torch
from torch.utils.data import DataLoader, Dataset

try:
    import torchaudio
    from torchaudio.datasets import SPEECHCOMMANDS
except Exception:
    torchaudio = None
    SPEECHCOMMANDS = None

from kws.config import AudioConfig


KEYWORDS_6 = ["go", "stop", "left", "right", "up", "down"]
CLASS_NAMES = ["go", "stop", "left", "right", "up", "down", "unknown", "silence"]
NUM_CLASSES = len(CLASS_NAMES)


def collate_kws(batch: List[Tuple[torch.Tensor, int]]):
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return xs, ys


def _safe_filename(s: str) -> str:
    s = s.replace("\\", "/")
    s = re.sub(r"[^a-zA-Z0-9_\-./]", "_", s)
    s = s.replace("/", "__")
    return s


class SpeechCommandsMFCC12(Dataset):
    """
    SpeechCommands v2 -> MFCC(62 x n_mfcc)
    Caches MFCC features per sample with config-dependent filenames.
    """

    def __init__(
        self,
        subset: str,
        audio_cfg: AudioConfig,
        silence_ratio: float = 0.10,
        cache_dir: Optional[str] = "./cache_mfcc",
        use_cache: bool = True,
        cache_dtype: torch.dtype = torch.float16,
    ):
        if torchaudio is None or SPEECHCOMMANDS is None:
            raise RuntimeError("torchaudio not found. Install: pip install torchaudio")

        self.ds = SPEECHCOMMANDS(root="./data", download=True, subset=subset)
        self.subset = subset
        self.audio_cfg = audio_cfg

        self.use_cache = bool(use_cache and cache_dir is not None)
        self.cache_dir = cache_dir
        self.cache_dtype = cache_dtype

        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.label2idx: Dict[str, int] = {k: i for i, k in enumerate(KEYWORDS_6)}
        self.unknown_idx = 6
        self.silence_idx = 7

        self.base_len = len(self.ds)
        self.silence_len = int(self.base_len * silence_ratio)

        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=audio_cfg.sample_rate,
            n_mfcc=audio_cfg.n_mfcc,
            melkwargs=dict(
                n_fft=audio_cfg.n_fft,
                win_length=audio_cfg.win_length,
                hop_length=audio_cfg.hop_length,
                n_mels=audio_cfg.n_mels,
                f_min=audio_cfg.f_min,
                f_max=audio_cfg.f_max,
                power=2.0,
                normalized=False,
                center=False,
            ),
        )

        fmax = "None" if audio_cfg.f_max is None else str(audio_cfg.f_max)
        self.cfg_sig = (
            f"sr{audio_cfg.sample_rate}_"
            f"fft{audio_cfg.n_fft}_win{audio_cfg.win_length}_hop{audio_cfg.hop_length}_"
            f"mels{audio_cfg.n_mels}_mfcc{audio_cfg.n_mfcc}_"
            f"fmin{audio_cfg.f_min}_fmax{fmax}_"
            f"len{audio_cfg.fixed_num_samples}"
        )

    def __len__(self) -> int:
        return self.base_len + self.silence_len

    def _to_target_sr(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        if sr == self.audio_cfg.sample_rate:
            return wav
        return torchaudio.functional.resample(wav, sr, self.audio_cfg.sample_rate)

    def _to_fixed_length(self, wav: torch.Tensor) -> torch.Tensor:
        target_len = self.audio_cfg.fixed_num_samples
        cur = wav.shape[-1]
        if cur < target_len:
            wav = torch.nn.functional.pad(wav, (0, target_len - cur))
        elif cur > target_len:
            wav = wav[..., :target_len]
        return wav

    def _label_to_idx(self, label: str) -> int:
        return self.label2idx.get(label, self.unknown_idx)

    def _wav_to_mfcc(self, wav: torch.Tensor) -> torch.Tensor:
        m = self.mfcc(wav)      # (1, n_mfcc, frames)
        m = m.transpose(1, 2)   # (1, frames, n_mfcc)
        return m

    def _cache_path(self, sample_id: str) -> str:
        fn = _safe_filename(f"{self.subset}__{sample_id}__{self.cfg_sig}.pt")
        return os.path.join(self.cache_dir, fn)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        if i >= self.base_len:
            wav = torch.zeros(1, self.audio_cfg.fixed_num_samples, dtype=torch.float32)
            x = self._wav_to_mfcc(wav)
            y = self.silence_idx
            return x, y

        sample_id = self.ds._walker[i] if hasattr(self.ds, "_walker") else str(i)
        cache_fp = self._cache_path(sample_id) if self.use_cache else None

        if self.use_cache and cache_fp is not None and os.path.exists(cache_fp):
            x = torch.load(cache_fp, map_location="cpu")
        else:
            wav, sr, label, *_ = self.ds[i]
            wav = self._to_target_sr(wav, sr)
            wav = self._to_fixed_length(wav)
            x = self._wav_to_mfcc(wav)

            if self.use_cache and cache_fp is not None:
                try:
                    torch.save(x.to(self.cache_dtype), cache_fp)
                except Exception:
                    pass

        x = x.to(torch.float32)

        _, _, label, *_ = self.ds[i]
        y = self._label_to_idx(label)
        return x, y


def make_loaders(audio_cfg: AudioConfig, batch_size: int, num_workers: int,
                 pin_memory: bool = True, prefetch_factor: int = 4, persistent_workers: bool = True,
                 train_device: str = "cuda"):
    cache_dir = "./cache_mfcc"

    train_ds = SpeechCommandsMFCC12("training", audio_cfg, silence_ratio=0.10, cache_dir=cache_dir, use_cache=True)
    val_ds = SpeechCommandsMFCC12("validation", audio_cfg, silence_ratio=0.10, cache_dir=cache_dir, use_cache=True)
    test_ds = SpeechCommandsMFCC12("testing", audio_cfg, silence_ratio=0.10, cache_dir=cache_dir, use_cache=True)

    pin = bool(pin_memory and train_device.startswith("cuda"))
    worker_kwargs = {}
    if num_workers > 0:
        if prefetch_factor is not None and prefetch_factor > 0:
            worker_kwargs["prefetch_factor"] = prefetch_factor
        if persistent_workers:
            worker_kwargs["persistent_workers"] = True

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=pin, collate_fn=collate_kws,
                          **worker_kwargs)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin, collate_fn=collate_kws,
                        **worker_kwargs)
    test_ld = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=pin, collate_fn=collate_kws,
                         **worker_kwargs)

    return train_ld, val_ld, test_ld
