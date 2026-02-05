from __future__ import annotations

from typing import List, Tuple, Dict

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

class SpeechCommandsMFCC12(Dataset):
    """
    SpeechCommands v2 -> MFCC(62 x 13) like the paper.

    Labels:
      0..9  = 10 keywords
      10    = unknown
      11    = silence (synthetic)

    Input feature:
      - MFCC computed from 16 mel filterbanks (n_mels=16)
      - keep 13 coefficients (n_mfcc=13)
      - frame length 512, hop 256
      - waveform padded to 16256 samples to get 62 frames
    """

    # Common 10-keyword set used in many KWS baselines
    KEYWORDS_6= ["go", "stop", "left", "right", "up", "down"]

    def __init__(self, subset: str, audio_cfg: AudioConfig, silence_ratio: float = 0.10):
        """
        silence_ratio:
          how many synthetic silence samples to add relative to dataset length.
          Example: 0.10 means add 10% extra samples as "silence".
        """
        if torchaudio is None or SPEECHCOMMANDS is None:
            raise RuntimeError("torchaudio not found. Install: pip install torchaudio")

        self.ds = SPEECHCOMMANDS(root="./data", download=True, subset=subset)
        self.audio_cfg = audio_cfg

        # label mapping
        self.label2idx: Dict[str, int] = {k: i for i, k in enumerate(self.KEYWORDS_6)}
        self.unknown_idx = 6
        self.silence_idx = 7

        # Add synthetic silence samples
        self.base_len = len(self.ds)
        self.silence_len = int(self.base_len * silence_ratio)

        # MFCC transform matching the paper
        # torchaudio MFCC returns shape: (channel=1, n_mfcc, frames)
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
                center=False,  # important: avoid implicit padding that changes frame count
            ),
        )

    def __len__(self) -> int:
        # include synthetic silence samples
        return self.base_len + self.silence_len

    def _to_target_sr(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        if sr == self.audio_cfg.sample_rate:
            return wav
        return torchaudio.functional.resample(wav, sr, self.audio_cfg.sample_rate)

    def _to_fixed_length(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Paper wants 62 frames with (win=512, hop=256).
        Easiest way: pad waveform to 16256 samples (16000 + 256).
        """
        target_len = self.audio_cfg.fixed_num_samples  # 16256
        cur = wav.shape[-1]
        if cur < target_len:
            wav = torch.nn.functional.pad(wav, (0, target_len - cur))
        elif cur > target_len:
            wav = wav[..., :target_len]
        return wav

    def _label_to_idx(self, label: str) -> int:
        if label in self.label2idx:
            return self.label2idx[label]
        return self.unknown_idx

    def _wav_to_mfcc_62x13(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: (1, N) -> mfcc: (1, n_mfcc=13, frames=62) -> return (1, 62, 13)
        We treat it as a single-channel "image": (C=1, H=62, W=13)
        """
        m = self.mfcc(wav)                 # (1, 13, frames)
        m = m.transpose(1, 2)              # (1, frames, 13)
        # Safety assert (can comment out after you verify once)
        # assert m.shape[1] == 62 and m.shape[2] == 13, f"MFCC shape is {m.shape}, expected (1,62,13)"
        return m

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        # Synthetic silence part
        if i >= self.base_len:
            # silence waveform: all zeros
            wav = torch.zeros(1, self.audio_cfg.fixed_num_samples, dtype=torch.float32)
            x = self._wav_to_mfcc_62x13(wav)
            y = self.silence_idx
            return x, y

        wav, sr, label, *_ = self.ds[i]

        wav = self._to_target_sr(wav, sr)
        wav = self._to_fixed_length(wav)
        x = self._wav_to_mfcc_62x13(wav)
        y = self._label_to_idx(label)
        return x, y


def make_loaders(audio_cfg: AudioConfig, batch_size: int, num_workers: int):
    train_ds = SpeechCommandsMFCC12("training", audio_cfg, silence_ratio=0.10)
    val_ds = SpeechCommandsMFCC12("validation", audio_cfg, silence_ratio=0.10)
    test_ds = SpeechCommandsMFCC12("testing", audio_cfg, silence_ratio=0.10)

    # def _collate(batch: List[Tuple[torch.Tensor, int]]):
    #     xs = torch.stack([b[0] for b in batch], dim=0)  # (B, 1, 62, 13)
    #     ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    #     return xs, ys

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=False, collate_fn=collate_kws)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=False, collate_fn=collate_kws)
    test_ld = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=False, collate_fn=collate_kws)

    return train_ld, val_ld, test_ld
