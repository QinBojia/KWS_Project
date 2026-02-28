"""Unified ONNX export for X-CUBE-AI deployment."""
from __future__ import annotations

from pathlib import Path

import torch

from kws.config import AudioConfig


def export_onnx(
    model: torch.nn.Module,
    audio_cfg: AudioConfig,
    output_path: str | Path,
    opset: int = 13,
) -> None:
    """
    Export model to ONNX format compatible with X-CUBE-AI.

    Args:
        model: Trained PyTorch model (float32).
        audio_cfg: Audio config to compute input shape.
        output_path: Path to save the .onnx file.
        opset: ONNX opset version (13 recommended for X-CUBE-AI).
    """
    n_frames = audio_cfg.n_frames
    dummy = torch.randn(1, 1, n_frames, audio_cfg.n_mfcc, dtype=torch.float32)

    model_cpu = model.to("cpu").eval()
    torch.onnx.export(
        model_cpu,
        dummy,
        str(output_path),
        input_names=["mfcc"],
        output_names=["logits"],
        dynamic_axes=None,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"Saved ONNX model to {output_path} (opset {opset})")
