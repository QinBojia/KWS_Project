"""Convert an existing ONNX model to int8 TFLite using onnx2tf + TensorFlow Lite.
- Expects an ONNX export produced by export_cubeai.py (input shape: [1, 1, 62, n_mfcc]).
- Uses SpeechCommands MFCC validation samples as representative dataset for post-training int8 quantization.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Iterable
import sys

import tensorflow as tf
import torch

from config import AudioConfig
from data import SpeechCommandsMFCC12


def run_onnx2tf(onnx_path: Path, saved_model_dir: Path) -> None:
    saved_model_dir.mkdir(parents=True, exist_ok=True)
    # onnx2tf CLI expects -i (input ONNX) and -o (output folder)
    cmd = [
        sys.executable,
        "-m",
        "onnx2tf",
        "-i",
        str(onnx_path),
        "-o",
        str(saved_model_dir),
        "--non_verbose",
        "-kt",
        "input",
    ]
    subprocess.run(cmd, check=True)


def representative_dataset(calib_samples: int, audio_cfg: AudioConfig) -> Iterable[list[tf.Tensor]]:
    ds = SpeechCommandsMFCC12(
        subset="validation",
        audio_cfg=audio_cfg,
        silence_ratio=0.0,
        cache_dir="./cache_mfcc",
        use_cache=True,
    )
    limit = min(calib_samples, len(ds))
    for i in range(limit):
        x, _ = ds[i]  # x: (1, frames, n_mfcc)
        x = x.unsqueeze(0)  # (1, 1, frames, n_mfcc)
        yield [tf.convert_to_tensor(x.numpy(), dtype=tf.float32)]


def convert_saved_model_to_tflite(saved_model_dir: Path, tflite_path: Path, calib_samples: int, audio_cfg: AudioConfig) -> None:
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(calib_samples, audio_cfg)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    tflite_path.write_bytes(tflite_model)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert ONNX -> TFLite int8")
    p.add_argument("--onnx-path", type=Path, required=True, help="Path to model_float.onnx")
    p.add_argument("--out-dir", type=Path, default=Path("./outputs_deploy/tflite_int8"))
    p.add_argument("--calib-samples", type=int, default=200, help="Number of validation samples for calibration")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    onnx_path: Path = args.onnx_path
    out_dir: Path = args.out_dir
    saved_model_dir = out_dir / "tf_saved_model"
    tflite_path = out_dir / "model_int8.tflite"

    out_dir.mkdir(parents=True, exist_ok=True)

    audio_cfg = AudioConfig()

    print(f"[1/2] Converting ONNX -> TF SavedModel at {saved_model_dir}")
    run_onnx2tf(onnx_path, saved_model_dir)

    print(f"[2/2] Converting SavedModel -> int8 TFLite at {tflite_path}")
    convert_saved_model_to_tflite(saved_model_dir, tflite_path, args.calib_samples, audio_cfg)

    print(f"Done. int8 TFLite saved to {tflite_path}")


if __name__ == "__main__":
    main()

