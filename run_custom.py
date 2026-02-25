"""
Train custom DS-CNN architectures targeting ~287K MACC.
Trains multiple candidates and exports the best to ONNX.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from config import AudioConfig, TrainConfig
from data import make_loaders
from model import CustomDSCNN
from quantization import fuse_for_quant, ptq_int8_static
from train_eval import train_one_experiment, evaluate, benchmark_inference_ms
from utils import ensure_dir, set_seed, save_json, count_macc, count_params, model_size_bytes


# ─── Candidate architectures ────────────────────────────────────────────

CANDIDATES = {
    # 287,552 MACC, 6,352 params - near-exact MACC match, 8 blocks, thin start
    "arch_a": {
        "stem_ch": 4, "stem_stride": 1,
        "block_cfg": [(8,1),(8,2),(16,1),(16,2),(32,1),(32,2),(32,1),(32,1)],
    },
    # 286,240 MACC, 5,976 params - compact 5 blocks, stem stride-2
    "arch_b": {
        "stem_ch": 16, "stem_stride": 2,
        "block_cfg": [(16,1),(32,2),(32,1),(32,2),(32,1)],
    },
    # 288,712 MACC, 3,656 params - wider channels (8->16->24->32), 5 blocks
    "arch_c": {
        "stem_ch": 8, "stem_stride": 1,
        "block_cfg": [(16,2),(16,1),(24,2),(24,1),(32,2)],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train custom DS-CNN models")
    parser.add_argument("--configs", nargs="+", default=list(CANDIDATES.keys()),
                        choices=list(CANDIDATES.keys()),
                        help="Which configs to train")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--calib-batches", type=int, default=30)
    parser.add_argument("--out-dir", type=Path, default=Path("./outputs_custom"))
    parser.add_argument("--export-onnx", action="store_true",
                        help="Export ONNX for best model")
    parser.add_argument("--onnx-opset", type=int, default=13)
    return parser.parse_args()


def train_and_eval(cfg_name: str, cfg: dict, audio_cfg: AudioConfig,
                   train_cfg: TrainConfig, train_ld, val_ld, test_ld,
                   out_dir: Path, calib_batches: int) -> dict:
    """Train a single config, quantize, evaluate."""
    print(f"\n{'='*60}")
    print(f"Training: {cfg_name}")
    print(f"{'='*60}")

    sub_dir = out_dir / cfg_name
    ensure_dir(sub_dir)
    set_seed(train_cfg.seed)

    device = train_cfg.train_device

    model = CustomDSCNN(
        stem_ch=cfg["stem_ch"],
        stem_stride=cfg["stem_stride"],
        block_cfg=cfg["block_cfg"],
        num_classes=8,
        dropout=0.1,
    )

    macc = count_macc(model, (1, 1, 62, 13))
    params = count_params(model)
    size_kb = model_size_bytes(model) / 1024
    print(f"  MACC={macc:,}  Params={params:,}  Size={size_kb:.1f}KB")
    print(f"  Stem: 1->{cfg['stem_ch']} s{cfg['stem_stride']}")
    for i, (o, s) in enumerate(cfg["block_cfg"]):
        print(f"  DS{i}: ->{o} s{s}")

    # Train
    model = model.to(device)
    train_stats = train_one_experiment(model, train_ld, val_ld, train_cfg, device=device)
    model = model.to("cpu")
    torch.save(model.state_dict(), sub_dir / "model_float.pth")

    # Float eval
    model.eval()
    float_metrics = evaluate(model, test_ld, device="cpu")
    example_x = next(iter(val_ld))[0][:1].to("cpu")
    float_ms = benchmark_inference_ms(model, example_x, device="cpu")

    print(f"  Float: acc={float_metrics['acc']:.4f}  infer={float_ms:.2f}ms")

    # INT8 quantization
    quant_model = CustomDSCNN(
        stem_ch=cfg["stem_ch"],
        stem_stride=cfg["stem_stride"],
        block_cfg=cfg["block_cfg"],
        num_classes=8,
        dropout=0.1,
    )
    quant_model.load_state_dict(model.state_dict())
    quant_model = fuse_for_quant(quant_model)
    int8_model = ptq_int8_static(quant_model, train_ld, device="cpu",
                                  calibration_batches=calib_batches)
    int8_model.eval()
    int8_metrics = evaluate(int8_model, test_ld, device="cpu")
    int8_ms = benchmark_inference_ms(int8_model, example_x, device="cpu")
    int8_size = model_size_bytes(int8_model) / 1024

    print(f"  INT8:  acc={int8_metrics['acc']:.4f}  infer={int8_ms:.2f}ms  size={int8_size:.1f}KB")

    torch.save(int8_model.state_dict(), sub_dir / "model_int8.pth")

    result = {
        "config": cfg_name,
        "arch": cfg,
        "macc": macc,
        "params": params,
        "float_size_kb": size_kb,
        "train": train_stats,
        "float": {"acc": float_metrics["acc"], "loss": float_metrics["loss"], "infer_ms": float_ms},
        "int8": {"acc": int8_metrics["acc"], "loss": int8_metrics["loss"],
                 "infer_ms": int8_ms, "size_kb": int8_size},
    }
    save_json(sub_dir / "result.json", result)
    return result


def export_onnx(model, audio_cfg, path, opset=13):
    n_frames = (audio_cfg.fixed_num_samples - audio_cfg.win_length) // audio_cfg.hop_length + 1
    dummy = torch.randn(1, 1, n_frames, audio_cfg.n_mfcc, dtype=torch.float32)
    model_cpu = model.to("cpu").eval()
    torch.onnx.export(
        model_cpu, dummy, str(path),
        input_names=["mfcc"], output_names=["logits"],
        dynamic_axes=None, opset_version=opset,
        do_constant_folding=True, dynamo=False,
    )
    print(f"Saved ONNX to {path}")


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_cfg = AudioConfig()
    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_workers=args.num_workers,
        train_device=device,
        infer_device="cpu",
    )

    # Load data once
    train_ld, val_ld, test_ld = make_loaders(
        audio_cfg=audio_cfg,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory,
        prefetch_factor=train_cfg.prefetch_factor,
        persistent_workers=train_cfg.persistent_workers,
        train_device=device,
    )

    results = []
    for cfg_name in args.configs:
        cfg = CANDIDATES[cfg_name]
        r = train_and_eval(cfg_name, cfg, audio_cfg, train_cfg,
                           train_ld, val_ld, test_ld,
                           args.out_dir, args.calib_batches)
        results.append(r)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<20} {'MACC':>10} {'Float%':>8} {'INT8%':>8} {'INT8 KB':>8} {'Target':>8}")
    print("-" * 60)
    for r in results:
        marker = "OK" if r["int8"]["acc"] >= 0.91 else "LOW"
        print(f"{r['config']:<20} {r['macc']:>10,} {r['float']['acc']:>7.2%} {r['int8']['acc']:>7.2%} "
              f"{r['int8']['size_kb']:>7.1f} {marker:>8}")
    print(f"\nBenchmark target: 287,673 MACC, >=91% accuracy")

    save_json(args.out_dir / "summary.json", results)

    # Export ONNX for the best model that meets accuracy threshold
    if args.export_onnx:
        valid = [r for r in results if r["int8"]["acc"] >= 0.91]
        if valid:
            best = min(valid, key=lambda r: abs(r["macc"] - 287673))
            cfg = CANDIDATES[best["config"]]
            model = CustomDSCNN(
                stem_ch=cfg["stem_ch"], stem_stride=cfg["stem_stride"],
                block_cfg=cfg["block_cfg"], num_classes=8,
            )
            model.load_state_dict(torch.load(args.out_dir / best["config"] / "model_float.pth",
                                             map_location="cpu"))
            onnx_path = args.out_dir / best["config"] / "model_opset13.onnx"
            export_onnx(model, audio_cfg, onnx_path, args.onnx_opset)
            print(f"\nBest model: {best['config']} -> {onnx_path}")
        else:
            print("\nNo model met 91% accuracy threshold for ONNX export.")


if __name__ == "__main__":
    main()
