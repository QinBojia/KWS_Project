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

from config import AudioConfig


def collate_kws(batch: List[Tuple[torch.Tensor, int]]):
    xs = torch.stack([b[0] for b in batch], dim=0)  # (B, 1, 62, n_mfcc)
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return xs, ys


def _safe_filename(s: str) -> str:
    """把路径/特殊字符变成安全文件名（Windows 兼容）"""
    s = s.replace("\\", "/")
    s = re.sub(r"[^a-zA-Z0-9_\-./]", "_", s)
    s = s.replace("/", "__")
    return s


class SpeechCommandsMFCC12(Dataset):
    """
    SpeechCommands v2 -> MFCC(62 x n_mfcc)

    缓存策略：
      - 每条样本的 MFCC 特征保存为 .pt 文件
      - cache key 会包含：subset + sample_id + MFCC/Mel/STFT 的关键参数
      - 这样你扫 (n_mels,n_mfcc,win,hop,fft) 时不会互相污染
    """

    KEYWORDS_6 = ["go", "stop", "left", "right", "up", "down"]

    def __init__(
        self,
        subset: str,
        audio_cfg: AudioConfig,
        silence_ratio: float = 0.10,
        cache_dir: Optional[str] = "./cache_mfcc",
        use_cache: bool = True,
        cache_dtype: torch.dtype = torch.float16,
    ):
        """
        cache_dir:
          MFCC 缓存目录。建议放在项目目录下（默认 ./cache_mfcc）
        use_cache:
          是否启用缓存。调试时可关掉。
        cache_dtype:
          缓存保存的 dtype。float16 能把缓存体积减半，IO 更快。
          训练时 DataLoader 会把它读出来，再在训练里 .to(device)。
          （注意：模型训练一般用 float32 输入也没问题；你可以在 __getitem__ 里再转回 float32）
        """
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

        # label mapping
        self.label2idx: Dict[str, int] = {k: i for i, k in enumerate(self.KEYWORDS_6)}
        self.unknown_idx = 6
        self.silence_idx = 7

        # Add synthetic silence samples
        self.base_len = len(self.ds)
        self.silence_len = int(self.base_len * silence_ratio)

        # MFCC transform
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
                center=False,  # avoid implicit padding
            ),
        )

        # 用于构造 cache key 的“参数签名”，确保不同配置不会读错缓存
        # 你扫参时，这一串不同就会落到不同文件名
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
        """
        wav: (1, N) -> mfcc: (1, n_mfcc, frames) -> (1, frames, n_mfcc)
        """
        m = self.mfcc(wav)      # (1, n_mfcc, frames)
        m = m.transpose(1, 2)   # (1, frames, n_mfcc)
        return m

    def _cache_path(self, sample_id: str) -> str:
        """
        sample_id: 一个稳定的标识（来自 SPEECHCOMMANDS 的 _walker 里的相对路径）
        """
        fn = _safe_filename(f"{self.subset}__{sample_id}__{self.cfg_sig}.pt")
        return os.path.join(self.cache_dir, fn)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        # Synthetic silence part（不缓存也行，生成很快）
        if i >= self.base_len:
            wav = torch.zeros(1, self.audio_cfg.fixed_num_samples, dtype=torch.float32)
            x = self._wav_to_mfcc(wav)
            y = self.silence_idx
            # 这里也可以选择缓存 silence，但收益不大
            return x, y

        # SPEECHCOMMANDS 的内部 walker 通常是相对路径列表（稳定）
        # 用它作为 cache key 的 sample_id
        sample_id = self.ds._walker[i] if hasattr(self.ds, "_walker") else str(i)
        cache_fp = self._cache_path(sample_id) if self.use_cache else None

        if self.use_cache and cache_fp is not None and os.path.exists(cache_fp):
            x = torch.load(cache_fp, map_location="cpu")
        else:
            wav, sr, label, *_ = self.ds[i]
            wav = self._to_target_sr(wav, sr)
            wav = self._to_fixed_length(wav)
            x = self._wav_to_mfcc(wav)

            # 保存缓存：用 float16 减少磁盘/IO（读出来训练时可再转 float32）
            if self.use_cache and cache_fp is not None:
                try:
                    torch.save(x.to(self.cache_dtype), cache_fp)
                except Exception:
                    # 缓存失败也不影响训练
                    pass

        # 如果你希望训练输入永远是 float32（更稳），这里转回去
        x = x.to(torch.float32)

        # label
        # 注意：我们需要 label，若缓存命中则 label 还需要从 ds 取一次
        # 但取 label 比算 MFCC 便宜太多
        _, _, label, *_ = self.ds[i]
        y = self._label_to_idx(label)
        return x, y


def make_loaders(audio_cfg: AudioConfig, batch_size: int, num_workers: int,
                 pin_memory: bool = True, prefetch_factor: int = 4, persistent_workers: bool = True,
                 train_device: str = "cuda"):
    # 建议：cache_dir 按配置区分（可选）
    # 但我们已经在文件名里包含 cfg_sig 了，所以同一个 cache_dir 也安全
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
