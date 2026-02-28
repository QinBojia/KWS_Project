"""
Unified evaluation script for KWS models.

Usage:
    python scripts/evaluate.py --config configs/arch_b.yaml --checkpoint experiments/arch_b/model_float.pth
    python scripts/evaluate.py --config configs/arch_b.yaml --checkpoint experiments/arch_b/model_float.pth --quantize
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, accuracy_score
)

from kws.config import load_config, build_model
from kws.data import make_loaders, CLASS_NAMES
from kws.quantization import fuse_for_quant, ptq_int8_static
from kws.utils import save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a KWS model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model_float.pth")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path (default: alongside checkpoint)")
    parser.add_argument("--quantize", action="store_true", help="Also evaluate INT8 quantized model")
    parser.add_argument("--calib-batches", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers (default: 2 for eval)")
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def collect_predictions(model, loader, device="cpu"):
    """Run model on loader, return all predictions and labels."""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def compute_metrics(y_true, y_pred, class_names=CLASS_NAMES):
    """Compute comprehensive metrics."""
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    prec_macro = precision_score(y_true, y_pred, average="macro")
    rec_macro = recall_score(y_true, y_pred, average="macro")

    report_dict = classification_report(y_true, y_pred, target_names=class_names,
                                         digits=4, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "per_class": report_dict,
        "confusion_matrix": cm.tolist(),
    }


def print_report(title, metrics, class_names=CLASS_NAMES):
    """Print human-readable evaluation report."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']:.2%})")
    print(f"  F1 (macro):         {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted):      {metrics['f1_weighted']:.4f}")
    print(f"  Precision (macro):  {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):     {metrics['recall_macro']:.4f}")

    print(f"\n  Per-class results:")
    print(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'-'*52}")
    for name in class_names:
        if name in metrics["per_class"]:
            c = metrics["per_class"][name]
            print(f"  {name:<12} {c['precision']:>10.4f} {c['recall']:>10.4f} "
                  f"{c['f1-score']:>10.4f} {int(c['support']):>10}")

    cm = metrics["confusion_matrix"]
    print(f"\n  Confusion Matrix:")
    header = "          " + " ".join(f"{n:>8s}" for n in class_names)
    print(f"  {header}")
    for i, row in enumerate(cm):
        row_str = " ".join(f"{v:>8d}" for v in row)
        print(f"  {class_names[i]:>8s} {row_str}")


def main():
    args = parse_args()
    exp = load_config(args.config)

    # Load data
    print("Loading data...")
    train_ld, _, test_ld = make_loaders(
        audio_cfg=exp.audio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        prefetch_factor=2,
        persistent_workers=False,
        train_device="cpu",
    )

    # Load float model
    print(f"Loading model from {args.checkpoint}...")
    model = build_model(exp.arch)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)

    # Float evaluation
    print("Evaluating float model...")
    y_pred_float, y_true = collect_predictions(model, test_ld, device="cpu")
    float_metrics = compute_metrics(y_true, y_pred_float)
    print_report(f"{exp.name} - Float32 Model", float_metrics)

    output = {
        "model": exp.name,
        "checkpoint": args.checkpoint,
        "float": float_metrics,
    }

    # INT8 evaluation
    if args.quantize:
        print("\nQuantizing to INT8...")
        quant_model = build_model(exp.arch)
        quant_model.load_state_dict(state)
        quant_model = fuse_for_quant(quant_model)
        int8_model = ptq_int8_static(quant_model, train_ld, device="cpu",
                                      calibration_batches=args.calib_batches)

        print("Evaluating INT8 model...")
        y_pred_int8, y_true2 = collect_predictions(int8_model, test_ld, device="cpu")
        int8_metrics = compute_metrics(y_true2, y_pred_int8)
        print_report(f"{exp.name} - INT8 Quantized Model", int8_metrics)

        output["int8"] = int8_metrics

    # Save results
    if args.output:
        out_path = args.output
    else:
        out_path = str(Path(args.checkpoint).parent / "eval_f1.json")

    save_json(out_path, output)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
