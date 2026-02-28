"""
Unified ONNX export script for X-CUBE-AI deployment.

Usage:
    python scripts/export_onnx.py --config configs/arch_b.yaml --checkpoint experiments/arch_b/model_float.pth
    python scripts/export_onnx.py --config configs/arch_b.yaml --checkpoint experiments/arch_b/model_float.pth --opset 13
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from kws.config import load_config, build_model
from kws.export import export_onnx


def parse_args():
    parser = argparse.ArgumentParser(description="Export KWS model to ONNX")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model_float.pth")
    parser.add_argument("--output", type=str, default=None,
                        help="Output ONNX path (default: alongside checkpoint)")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version (default: 13)")
    return parser.parse_args()


def main():
    args = parse_args()
    exp = load_config(args.config)

    # Load model
    model = build_model(exp.arch)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)

    # Determine output path
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(args.checkpoint).parent / f"model_opset{args.opset}.onnx"

    export_onnx(model, exp.audio, out_path, opset=args.opset)

    # Print file size
    size_kb = out_path.stat().st_size / 1024
    print(f"ONNX file size: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
