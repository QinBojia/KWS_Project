"""
Evaluate arch_b (best CustomDSCNN) on the test set.
Computes per-class and macro/weighted F1-score, precision, recall,
confusion matrix, and classification report.
"""
from __future__ import annotations

import json
import torch
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, accuracy_score
)

from config import AudioConfig
from data import make_loaders
from model import CustomDSCNN
from quantization import fuse_for_quant, ptq_int8_static


CLASS_NAMES = ["go", "stop", "left", "right", "up", "down", "unknown", "silence"]

# arch_b config
ARCH_B = {
    "stem_ch": 16, "stem_stride": 2,
    "block_cfg": [(16, 1), (32, 2), (32, 1), (32, 2), (32, 1)],
}


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


def print_results(title, y_true, y_pred):
    """Print classification report and confusion matrix."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    prec_macro = precision_score(y_true, y_pred, average="macro")
    rec_macro = recall_score(y_true, y_pred, average="macro")

    print(f"\n  Accuracy:           {acc:.4f} ({acc:.2%})")
    print(f"  F1 (macro):         {f1_macro:.4f}")
    print(f"  F1 (weighted):      {f1_weighted:.4f}")
    print(f"  Precision (macro):  {prec_macro:.4f}")
    print(f"  Recall (macro):     {rec_macro:.4f}")

    print(f"\n  Per-class classification report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print(f"  Confusion Matrix:")
    # Header
    header = "          " + " ".join(f"{n:>8s}" for n in CLASS_NAMES)
    print(header)
    for i, row in enumerate(cm):
        row_str = " ".join(f"{v:>8d}" for v in row)
        print(f"  {CLASS_NAMES[i]:>8s} {row_str}")

    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "per_class": json.loads(
            json.dumps(
                classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                      digits=4, output_dict=True)
            )
        ),
        "confusion_matrix": cm.tolist(),
    }


def main():
    audio_cfg = AudioConfig()
    print("Loading data (num_workers=2 for eval-only)...")
    _, _, test_ld = make_loaders(
        audio_cfg=audio_cfg,
        batch_size=512,
        num_workers=2,
        pin_memory=False,
        prefetch_factor=2,
        persistent_workers=False,
        train_device="cpu",
    )

    # Load arch_b float model
    print("Loading arch_b float model...")
    model = CustomDSCNN(
        stem_ch=ARCH_B["stem_ch"],
        stem_stride=ARCH_B["stem_stride"],
        block_cfg=ARCH_B["block_cfg"],
        num_classes=8,
        dropout=0.0,  # no dropout at eval
    )
    state = torch.load("outputs_custom/arch_b/model_float.pth", map_location="cpu")
    model.load_state_dict(state)

    # Float evaluation
    print("Evaluating float model...")
    y_pred_float, y_true = collect_predictions(model, test_ld, device="cpu")
    float_results = print_results("arch_b - Float32 Model", y_true, y_pred_float)

    # INT8 quantization
    print("\nQuantizing to INT8...")
    # Need train loader for calibration
    train_ld, _, _ = make_loaders(
        audio_cfg=audio_cfg,
        batch_size=512,
        num_workers=2,
        pin_memory=False,
        prefetch_factor=2,
        persistent_workers=False,
        train_device="cpu",
    )

    quant_model = CustomDSCNN(
        stem_ch=ARCH_B["stem_ch"],
        stem_stride=ARCH_B["stem_stride"],
        block_cfg=ARCH_B["block_cfg"],
        num_classes=8,
        dropout=0.0,
    )
    quant_model.load_state_dict(state)
    quant_model = fuse_for_quant(quant_model)
    int8_model = ptq_int8_static(quant_model, train_ld, device="cpu", calibration_batches=30)

    # INT8 evaluation
    print("Evaluating INT8 model...")
    y_pred_int8, y_true2 = collect_predictions(int8_model, test_ld, device="cpu")
    int8_results = print_results("arch_b - INT8 Quantized Model", y_true2, y_pred_int8)

    # Save results
    output = {
        "model": "arch_b",
        "float": float_results,
        "int8": int8_results,
    }
    with open("outputs_custom/arch_b/eval_f1.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to outputs_custom/arch_b/eval_f1.json")


if __name__ == "__main__":
    main()
