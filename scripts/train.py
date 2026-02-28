"""
Unified training script for KWS models.

Usage:
    python scripts/train.py --config configs/arch_b.yaml
    python scripts/train.py --config configs/arch_b.yaml --output-dir experiments/arch_b
    python scripts/train.py --config configs/arch_b.yaml --skip-train --export-onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from kws.config import load_config, build_model, ExperimentConfig
from kws.data import make_loaders
from kws.training import train_one_experiment, evaluate, benchmark_inference_ms
from kws.quantization import fuse_for_quant, ptq_int8_static
from kws.export import export_onnx
from kws.utils import ensure_dir, set_seed, save_json, count_macc, count_params, model_size_bytes


def parse_args():
    parser = argparse.ArgumentParser(description="Train a KWS model from YAML config")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: experiments/<config_name>)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, load existing weights from output-dir/model_float.pth")
    parser.add_argument("--export-onnx", action="store_true", help="Export ONNX after training")
    parser.add_argument("--onnx-opset", type=int, default=13, help="ONNX opset version")
    parser.add_argument("--calib-batches", type=int, default=30, help="PTQ calibration batches")
    parser.add_argument("--no-quantize", action="store_true", help="Skip INT8 quantization")

    # Override training params via CLI
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def apply_overrides(exp: ExperimentConfig, args) -> ExperimentConfig:
    """Apply CLI overrides to the loaded config."""
    if args.epochs is not None:
        exp.train.epochs = args.epochs
    if args.batch_size is not None:
        exp.train.batch_size = args.batch_size
    if args.lr is not None:
        exp.train.lr = args.lr
    if args.num_workers is not None:
        exp.train.num_workers = args.num_workers
    if args.seed is not None:
        exp.train.seed = args.seed
    return exp


def run(exp: ExperimentConfig, out_dir: Path, args):
    ensure_dir(out_dir)
    set_seed(exp.train.seed)

    device = exp.train.train_device
    if not torch.cuda.is_available():
        device = "cpu"
        exp.train.train_device = "cpu"

    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.backends.cudnn.benchmark = bool(exp.train.cudnn_benchmark)

    # Build model
    model = build_model(exp.arch)
    macc = count_macc(model, exp.audio.input_shape)
    params = count_params(model)
    size_kb = model_size_bytes(model) / 1024

    print(f"Model: {exp.name}")
    print(f"  Type: {exp.arch.model_type}")
    print(f"  MACC: {macc:,}")
    print(f"  Params: {params:,}")
    print(f"  Float size: {size_kb:.1f} KB")

    # Load data
    print("Loading data...")
    train_ld, val_ld, test_ld = make_loaders(
        audio_cfg=exp.audio,
        batch_size=exp.train.batch_size,
        num_workers=exp.train.num_workers,
        pin_memory=exp.train.pin_memory,
        prefetch_factor=exp.train.prefetch_factor,
        persistent_workers=exp.train.persistent_workers,
        train_device=device,
        preload=True,
        num_classes=exp.arch.num_classes,
    )

    float_state_path = out_dir / "model_float.pth"
    train_stats = {}

    if args.skip_train:
        if not float_state_path.exists():
            raise FileNotFoundError(f"--skip-train but {float_state_path} not found")
        model.load_state_dict(torch.load(float_state_path, map_location="cpu"))
        print(f"Loaded existing weights from {float_state_path}")
    else:
        print(f"Training for {exp.train.epochs} epochs on {device}...")
        model = model.to(device)
        train_stats = train_one_experiment(model, train_ld, val_ld, exp.train, device=device)
        model = model.to("cpu")
        torch.save(model.state_dict(), float_state_path)
        print(f"  Best val acc: {train_stats['best_val_acc']:.4f} (epoch {train_stats['best_epoch']})")

    # Float evaluation — run on GPU if data is there, benchmark on CPU for deployment
    model.eval()
    model = model.to(device)
    float_metrics = evaluate(model, test_ld, device=device)
    model = model.to("cpu")
    example_x = next(iter(val_ld))[0][:1].to("cpu")
    float_ms = benchmark_inference_ms(model, example_x, device="cpu")
    print(f"  Float test acc: {float_metrics['acc']:.4f}  latency: {float_ms:.2f}ms")

    result = {
        "config": exp.name,
        "arch": {
            "model_type": exp.arch.model_type,
            "stem_ch": exp.arch.stem_ch,
            "stem_stride": exp.arch.stem_stride,
            "block_cfg": [list(b) for b in exp.arch.block_cfg] if exp.arch.block_cfg else None,
            "width_mult": exp.arch.width_mult,
            "depth_mult": exp.arch.depth_mult,
        },
        "macc": macc,
        "params": params,
        "float_size_kb": size_kb,
        "train": train_stats,
        "float": {"acc": float_metrics["acc"], "loss": float_metrics["loss"], "infer_ms": float_ms},
    }

    # INT8 quantization
    if not args.no_quantize:
        print("Quantizing to INT8...")
        quant_model = build_model(exp.arch)
        quant_model.load_state_dict(model.state_dict())
        quant_model = fuse_for_quant(quant_model)
        int8_model = ptq_int8_static(quant_model, train_ld, device="cpu",
                                      calibration_batches=args.calib_batches)
        int8_model.eval()
        int8_metrics = evaluate(int8_model, test_ld, device="cpu")
        int8_ms = benchmark_inference_ms(int8_model, example_x, device="cpu")
        int8_size = model_size_bytes(int8_model) / 1024

        print(f"  INT8 test acc: {int8_metrics['acc']:.4f}  latency: {int8_ms:.2f}ms  size: {int8_size:.1f}KB")

        torch.save(int8_model.state_dict(), out_dir / "model_int8.pth")

        result["int8"] = {
            "acc": int8_metrics["acc"],
            "loss": int8_metrics["loss"],
            "infer_ms": int8_ms,
            "size_kb": int8_size,
        }

    # ONNX export
    if args.export_onnx:
        onnx_path = out_dir / f"model_opset{args.onnx_opset}.onnx"
        export_onnx(model, exp.audio, onnx_path, opset=args.onnx_opset)
        result["onnx_path"] = str(onnx_path)

    save_json(out_dir / "result.json", result)
    print(f"\nResults saved to {out_dir / 'result.json'}")
    return result


def main():
    args = parse_args()
    exp = load_config(args.config)
    exp = apply_overrides(exp, args)

    out_dir = Path(args.output_dir) if args.output_dir else Path("experiments") / exp.name
    run(exp, out_dir, args)


if __name__ == "__main__":
    main()
