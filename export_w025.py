"""Train the w=0.25,d=0.25 model, quantize it, and export both variants for deployment."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import AudioConfig, ModelConfig, TrainConfig, ExperimentConfig
from data import make_loaders
from model import MobileNetStyleKWS
from quantization import fuse_for_quant, ptq_int8_static
from train_eval import train_one_experiment, evaluate, benchmark_inference_ms
from utils import ensure_dir, set_seed, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and export w=0.25/0.25 KWS model")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs for the float model")
    parser.add_argument("--batch-size", type=int, default=1024, help="Training batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader worker count")
    parser.add_argument("--calib-batches", type=int, default=30, help="Calibration batches for PTQ")
    parser.add_argument("--out-dir", type=Path, default=Path("./outputs_deploy/w025"), help="Export directory")
    parser.add_argument("--dry-run", action="store_true", help="Print config without running training/quantization")
    parser.add_argument("--export-onnx", action="store_true", help="Export float ONNX (opset=13) for STM32Cube.AI")
    parser.add_argument("--onnx-opset", type=int, default=13, help="ONNX opset for Cube.AI (recommend 13)")

    return parser.parse_args()


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


def script_model(model: torch.nn.Module, path: Path) -> None:
    try:
        script = torch.jit.script(model)
        script.save(path)
        print(f"Saved scripted model to {path}")
    except Exception as e:
        print(f"Warning: Failed to script model: {e}")


def trace_model(model: torch.nn.Module, example_input: torch.Tensor, path: Path) -> None:
    """Use tracing instead of scripting for quantized models to avoid type errors."""
    try:
        # Tracing requires the model to be on CPU usually for quantized models
        model = model.cpu()
        example_input = example_input.cpu()
        traced = torch.jit.trace(model, example_input)
        traced.save(path)
        print(f"Saved traced model to {path}")
    except Exception as e:
        print(f"Warning: Failed to trace model: {e}")


def export_onnx_float_cubeai(model: torch.nn.Module, audio_cfg: AudioConfig, path: Path, opset: int = 13) -> None:
    # 计算帧数：你当前配置应当得到 62 帧
    n_frames = (audio_cfg.fixed_num_samples - audio_cfg.win_length) // audio_cfg.hop_length + 1
    dummy = torch.randn(1, 1, n_frames, audio_cfg.n_mfcc, dtype=torch.float32)

    model_cpu = model.to("cpu").eval()
    try:
        # PyTorch 2.10+ defaults to the new dynamo-based exporter which
        # produces opset>=18 and then tries (often fails) to downconvert.
        # Force the legacy TorchScript-based exporter via dynamo=False
        # so we can directly emit the requested opset (e.g. 13).
        torch.onnx.export(
            model_cpu,
            dummy,
            str(path),
            input_names=["mfcc"],
            output_names=["logits"],
            dynamic_axes=None,          # 固定形状，Cube.AI 更稳
            opset_version=opset,        # 关键：用 13 避免 Reshape.allowzero
            do_constant_folding=True,
            dynamo=False,               # 强制用旧版 TorchScript exporter
        )
        print(f"Saved ONNX model to {path} (opset {opset})")
    except Exception as e:
        print(f"Warning: Failed to export ONNX: {e}")


def export(exp: ExperimentConfig, out_dir: Path, calibration_batches: int, export_onnx: bool, onnx_opset: int) -> dict:

    ensure_dir(out_dir)
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
    example_x_infer = example_x[:1].to(infer_device)
    float_ms = benchmark_inference_ms(model, example_x_infer, device=infer_device)

    float_state_path = out_dir / "model_float.pth"
    torch.save(model.state_dict(), float_state_path)

    float_script_path = out_dir / "model_float_script.pt"
    # For float model, scripting is usually fine
    script_model(model, float_script_path)

    if export_onnx:
        onnx_path = out_dir / "model_float_opset13_cubeai.onnx"
        export_onnx_float_cubeai(model, exp.audio, onnx_path, opset=onnx_opset)

    quant_model = MobileNetStyleKWS(
        num_classes=exp.model.num_classes,
        width_mult=exp.model.width_mult,
        dropout=exp.model.dropout,
        depth_mult=exp.model.depth_mult,
    )
    quant_model.load_state_dict(model.state_dict())
    quant_model = fuse_for_quant(quant_model)

    int8_model = ptq_int8_static(
        quant_model,
        calibration_loader=train_ld,
        device="cpu",
        calibration_batches=calibration_batches,
    )

    int8_model.eval()
    quant_metrics = evaluate(int8_model, test_ld, device="cpu")
    # For benchmark and saving, ensure input is on CPU
    example_x_cpu = example_x[:1].to("cpu")
    int8_ms = benchmark_inference_ms(int8_model, example_x_cpu, device="cpu")

    int8_state_path = out_dir / "model_int8.pth"
    torch.save(int8_model.state_dict(), int8_state_path)

    int8_script_path = out_dir / "model_int8_script.pt"
    # USE TRACE FOR QUANTIZED MODEL
    trace_model(int8_model, example_x_cpu, int8_script_path)

    result = {
        "experiment": exp.name,
        "train": train_stats,
        "float": {
            "test": float_metrics,
            "infer_ms": float_ms,
        },
        "int8": {
            "test": quant_metrics,
            "infer_ms": int8_ms,
        },
        "paths": {
            "float_state": str(float_state_path),
            "float_script": str(float_script_path),
            "int8_state": str(int8_state_path),
            "int8_script": str(int8_script_path),
        },
    }

    save_json(out_dir / "result.json", result)
    return result


def main() -> None:
    args = parse_args()
    exp = build_experiment(args)
    print(f"Configs: epochs={exp.train.epochs}, batch={exp.train.batch_size}, device={exp.train.train_device}")
    print(f"Export target: {args.out_dir}")
    if args.dry_run:
        return

    result = export(
        exp,
        args.out_dir,
        calibration_batches=args.calib_batches,
        export_onnx=args.export_onnx,
        onnx_opset=args.onnx_opset,
    )

    print("Export complete:")
    print(f"  float accuracy: {result['float']['test']['acc']:.4f}, infer_ms={result['float']['infer_ms']:.2f}")
    print(f"  int8 accuracy: {result['int8']['test']['acc']:.4f}, infer_ms={result['int8']['infer_ms']:.2f}")
    print(f"  artifacts saved to {args.out_dir}")


if __name__ == "__main__":
    main()
