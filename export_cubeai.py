"""Train the w=0.25/d=0.25 model and export artifacts suitable for STM32Cube.AI.
- Produces: PyTorch state, TorchScript, ONNX (float). Optional: TFLite (float/int8) if onnx2tf+TF are installed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Optional

import torch

from config import AudioConfig, ModelConfig, TrainConfig, ExperimentConfig
from data import make_loaders
from model import MobileNetStyleKWS
from quantization import fuse_for_quant, ptq_int8_static
from train_eval import train_one_experiment, evaluate, benchmark_inference_ms
from utils import ensure_dir, set_seed, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export w=0.25/d=0.25 KWS model for STM32Cube.AI")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--calib-batches", type=int, default=30, help="PTQ calibration batches")
    p.add_argument("--out-dir", type=Path, default=Path("./outputs_deploy/w025_cubeai"))
    p.add_argument("--export-tflite", action="store_true", help="Attempt TFLite export (requires onnx2tf + tensorflow)")
    p.add_argument("--onnx-opset", type=int, default=18, help="ONNX opset to export (use 18 to avoid version_converter failures)")
    return p.parse_args()


def build_experiment(args: argparse.Namespace) -> ExperimentConfig:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_workers=args.num_workers,
        train_device=device,
        infer_device="cpu",
    )
    audio_cfg = AudioConfig()
    model_cfg = ModelConfig(num_classes=8, width_mult=0.25, depth_mult=0.25)
    return ExperimentConfig(
        name="mfcc_mel16_mfcc13_w0.25_d0.25",
        audio=audio_cfg,
        model=model_cfg,
        train=train_cfg,
    )


def export_onnx(model: torch.nn.Module, audio_cfg: AudioConfig, path: Path, opset: int) -> None:
    n_frames = (audio_cfg.fixed_num_samples - audio_cfg.win_length) // audio_cfg.hop_length + 1
    dummy = torch.randn(1, 1, n_frames, audio_cfg.n_mfcc)
    model.eval()
    try:
        torch.onnx.export(
            model,
            dummy,
            path,
            input_names=["mfcc"],
            output_names=["logits"],
            dynamic_axes=None,
            opset_version=opset,
            do_constant_folding=True,
        )
        return
    except Exception as exc:
        print(f"[warn] torch.onnx.export failed (opset {opset}), trying legacy _export: {exc}")
        from torch.onnx import _export
        _export(
            model,
            dummy,
            path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL,
        )


def try_export_tflite(onnx_path: Path, out_dir: Path) -> Optional[Path]:
    """Optional ONNX->TFLite path via onnx2tf + TF Lite. Returns tflite path if successful."""
    saved_model_dir = out_dir / "onnx2tf_saved_model"
    tflite_path = out_dir / "model_float.tflite"

    try:
        # 1) ONNX -> TF SavedModel
        subprocess.run(
            ["python", "-m", "onnx2tf", "--onnx_path", str(onnx_path), "--output_path", str(saved_model_dir)],
            check=True,
        )
        # 2) SavedModel -> TFLite
        import tensorflow as tf  # noqa: WPS433

        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        converter.optimizations = []
        tflite_model = converter.convert()
        tflite_path.write_bytes(tflite_model)
        return tflite_path
    except Exception as exc:  # noqa: W0703
        print(f"[warn] TFLite export skipped: {exc}")
        return None


def main() -> None:
    args = parse_args()
    exp = build_experiment(args)

    ensure_dir(args.out_dir)
    set_seed(exp.train.seed)

    train_device = exp.train.train_device
    infer_device = exp.train.infer_device

    if torch.cuda.is_available() and train_device.startswith("cuda"):
        torch.backends.cudnn.benchmark = bool(exp.train.cudnn_benchmark)

    train_ld, val_ld, test_ld = make_loaders(
        audio_cfg=exp.audio,
        batch_size=exp.train.batch_size,
        num_workers=exp.train.num_workers,
        pin_memory=exp.train.pin_memory,
        prefetch_factor=exp.train.prefetch_factor,
        persistent_workers=exp.train.persistent_workers,
        train_device=train_device,
    )

    model = MobileNetStyleKWS(
        num_classes=exp.model.num_classes,
        width_mult=exp.model.width_mult,
        dropout=exp.model.dropout,
        depth_mult=exp.model.depth_mult,
    ).to(train_device)

    train_stats = train_one_experiment(model, train_ld, val_ld, exp.train, device=train_device)

    model = model.to(infer_device).eval()
    float_metrics = evaluate(model, test_ld, device=infer_device)
    example_x, _ = next(iter(val_ld))
    example_x = example_x[:1].to(infer_device)
    float_ms = benchmark_inference_ms(model, example_x, device=infer_device)

    float_state_path = args.out_dir / "model_float.pth"
    torch.save(model.state_dict(), float_state_path)

    float_script_path = args.out_dir / "model_float_script.pt"
    torch.jit.script(model).save(float_script_path)

    # ONNX (float) for STM32Cube.AI import
    onnx_path = args.out_dir / "model_float.onnx"
    export_onnx(model, exp.audio, onnx_path, opset=args.onnx_opset)

    # Optional: PyTorch int8 PTQ export (for reference/testing)
    quant_model = MobileNetStyleKWS(
        num_classes=exp.model.num_classes,
        width_mult=exp.model.width_mult,
        dropout=exp.model.dropout,
        depth_mult=exp.model.depth_mult,
    )
    quant_model.load_state_dict(model.state_dict())
    quant_model.eval()
    quant_model = fuse_for_quant(quant_model)
    int8_model = ptq_int8_static(
        quant_model,
        calibration_loader=train_ld,
        device="cpu",
        calibration_batches=args.calib_batches,
    )
    int8_model.eval()
    quant_metrics = evaluate(int8_model, test_ld, device="cpu")
    int8_ms = benchmark_inference_ms(int8_model, example_x.to("cpu"), device="cpu")
    int8_state_path = args.out_dir / "model_int8.pth"
    torch.save(int8_model.state_dict(), int8_state_path)

    # Optional TFLite export (float) if requested
    tflite_path = None
    if args.export_tflite:
        tflite_path = try_export_tflite(onnx_path, args.out_dir)

    result = {
        "experiment": exp.name,
        "train": train_stats,
        "float": {
            "test": float_metrics,
            "infer_ms": float_ms,
        },
        "int8_ptq": {
            "test": quant_metrics,
            "infer_ms": int8_ms,
        },
        "paths": {
            "float_state": str(float_state_path),
            "float_script": str(float_script_path),
            "float_onnx": str(onnx_path),
            "int8_state": str(int8_state_path),
            "tflite_float_optional": str(tflite_path) if tflite_path else None,
        },
    }

    save_json(args.out_dir / "result.json", result)
    print(json.dumps(result, indent=2))
    print("Export complete. Import `model_float.onnx` into STM32Cube.AI.")



