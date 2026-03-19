"""Generate all 5 figures for the paper."""

import json
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

OUT = Path("C:/Users/m1339/PycharmProjects/KWS_Project/figures")
OUT.mkdir(exist_ok=True)
DATA = Path("C:/Users/m1339/PycharmProjects/KWS_Project/experiments")


# =========================================================================
# Fig 1: System Pipeline
# =========================================================================
def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(7.5, 2.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.4)
    ax.axis("off")

    # Top row: inference pipeline
    boxes_top = [
        (0.3, 1.5, "Audio Input\n(16 kHz, 1s)", "#E8F0FE"),
        (2.5, 1.5, "MFCC\nExtraction\n(62×13)", "#D4E6F1"),
        (4.7, 1.5, "TENet\nInference\n(276K MACC)", "#D5F5E3"),
        (6.9, 1.5, "Keyword\nPrediction\n(12 classes)", "#FADBD8"),
    ]

    # Bottom row: deployment pipeline
    boxes_bot = [
        (0.3, 0.2, "PyTorch\nTraining", "#FEF9E7"),
        (2.1, 0.2, "ONNX\nExport", "#FEF9E7"),
        (3.9, 0.2, "INT8\nQuantization", "#FEF9E7"),
        (5.7, 0.2, "X-CUBE-AI\nConversion", "#FEF9E7"),
        (7.5, 0.2, "STM32F767\nDeployment", "#F5EEF8"),
    ]

    bw, bh = 1.6, 0.75
    bw2, bh2 = 1.4, 0.65

    for x, y, txt, color in boxes_top:
        rect = mpatches.FancyBboxPatch(
            (x, y), bw, bh, boxstyle="round,pad=0.08",
            facecolor=color, edgecolor="#555", linewidth=1.2
        )
        ax.add_patch(rect)
        ax.text(x + bw / 2, y + bh / 2, txt, ha="center", va="center",
                fontsize=8, fontweight="bold")

    for x, y, txt, color in boxes_bot:
        rect = mpatches.FancyBboxPatch(
            (x, y), bw2, bh2, boxstyle="round,pad=0.06",
            facecolor=color, edgecolor="#888", linewidth=0.8
        )
        ax.add_patch(rect)
        ax.text(x + bw2 / 2, y + bh2 / 2, txt, ha="center", va="center",
                fontsize=7)

    # Arrows top row
    arrow_kw = dict(arrowstyle="->,head_width=0.15,head_length=0.1",
                    color="#333", lw=1.5)
    for i in range(3):
        x1 = boxes_top[i][0] + bw
        x2 = boxes_top[i + 1][0]
        y_mid = boxes_top[i][1] + bh / 2
        ax.annotate("", xy=(x2, y_mid), xytext=(x1, y_mid),
                     arrowprops=arrow_kw)

    # Arrows bottom row
    arrow_kw2 = dict(arrowstyle="->,head_width=0.1,head_length=0.08",
                     color="#888", lw=1.0)
    for i in range(4):
        x1 = boxes_bot[i][0] + bw2
        x2 = boxes_bot[i + 1][0]
        y_mid = boxes_bot[i][1] + bh2 / 2
        ax.annotate("", xy=(x2, y_mid), xytext=(x1, y_mid),
                     arrowprops=arrow_kw2)

    # Labels
    ax.text(5, 2.35, "Inference Pipeline", ha="center", fontsize=10,
            fontweight="bold", color="#333")
    ax.text(4.8, 0.92, "Training & Deployment Pipeline", ha="center",
            fontsize=9, fontstyle="italic", color="#666")

    fig.savefig(OUT / "fig1_pipeline.png")
    fig.savefig(OUT / "fig1_pipeline.pdf")
    plt.close(fig)
    print("Fig 1 saved.")


# =========================================================================
# Fig 2: TENet Architecture
# =========================================================================
def fig2_architecture():
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.8)
    ax.axis("off")

    # Main blocks
    blocks = [
        (0.1, 2.5, 1.1, 0.7, "Input\n(13, 62)", "#E8F0FE"),
        (1.6, 2.5, 1.1, 0.7, "Stem\nConv1d\n13→14, k=3", "#D4E6F1"),
        (3.1, 2.5, 1.1, 0.7, "Block 0\n×1, s=2\n(14, 31)", "#D5F5E3"),
        (4.6, 2.5, 1.1, 0.7, "Block 1\n×2, s=2\n(14, 16)", "#D5F5E3"),
        (6.1, 2.5, 1.1, 0.7, "Block 2\n×1, s=1\n(14, 16)", "#D5F5E3"),
        (7.6, 2.5, 1.1, 0.7, "Block 3\n×1, s=1\n(14, 16)", "#D5F5E3"),
        (9.0, 2.5, 0.9, 0.7, "GAP\n+FC\n→12", "#FADBD8"),
    ]

    for x, y, w, h, txt, color in blocks:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.06",
            facecolor=color, edgecolor="#555", linewidth=1.0
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
                fontsize=7, fontweight="bold")

    # Arrows
    arrow_kw = dict(arrowstyle="->,head_width=0.12,head_length=0.08",
                    color="#333", lw=1.2)
    positions = [(b[0] + b[2], b[1] + b[3] / 2) for b in blocks]
    for i in range(len(blocks) - 1):
        ax.annotate("", xy=(blocks[i + 1][0], positions[i][1]),
                     xytext=positions[i], arrowprops=arrow_kw)

    # Inverted Bottleneck detail
    ax.text(5.0, 1.9, "Inverted Bottleneck Block Detail (expansion ratio = 4):",
            fontsize=9, fontweight="bold", color="#333")

    ib_blocks = [
        (0.4, 0.5, 1.5, 0.7, "PW Expand\nConv1d 1×1\n14 → 56", "#FFF3CD"),
        (2.3, 0.5, 1.5, 0.7, "BN + ReLU", "#FDEBD0"),
        (4.2, 0.5, 1.5, 0.7, "DW Temporal\nConv1d k=9\n56ch, groups=56", "#D5F5E3"),
        (6.1, 0.5, 1.5, 0.7, "BN + ReLU", "#FDEBD0"),
        (8.0, 0.5, 1.5, 0.7, "PW Project\nConv1d 1×1\n56 → 14\n(linear)", "#E8DAEF"),
    ]

    for x, y, w, h, txt, color in ib_blocks:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="#777", linewidth=0.8
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
                fontsize=6.5)

    for i in range(4):
        x1 = ib_blocks[i][0] + ib_blocks[i][2]
        x2 = ib_blocks[i + 1][0]
        y_mid = ib_blocks[i][1] + ib_blocks[i][3] / 2
        ax.annotate("", xy=(x2, y_mid), xytext=(x1, y_mid),
                     arrowprops=dict(arrowstyle="->", color="#777", lw=0.8))

    # Residual connection
    ax.annotate("", xy=(9.5, 0.5), xytext=(0.4, 0.5),
                arrowprops=dict(arrowstyle="->", color="#C0392B",
                                lw=1.0, ls="--",
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(4.8, 0.1, "Residual Connection (identity or 1×1 proj)",
            ha="center", fontsize=7, color="#C0392B", fontstyle="italic")

    fig.savefig(OUT / "fig2_architecture.png")
    fig.savefig(OUT / "fig2_architecture.pdf")
    plt.close(fig)
    print("Fig 2 saved.")


# =========================================================================
# Fig 3: Grid Search Scatter Plot
# =========================================================================
def fig3_grid_search():
    # Load all rounds
    all_points = []
    colors_map = {
        "round1": ("#AED6F1", 0.4, 8, "Round 1 (15ep, n=1280)"),
        "round2a": ("#85C1E9", 0.5, 12, "Round 2a (15ep, n=548)"),
        "round2b": ("#F9E79F", 0.7, 16, "Round 2b (200ep, n=464)"),
        "round2c": ("#E74C3C", 1.0, 60, "Round 2c (1000ep, n=10)"),
    }

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for rnd, (color, alpha, sz, label) in colors_map.items():
        fpath = DATA / "grid_search" / rnd / "summary.json"
        if not fpath.exists():
            continue
        with open(fpath) as f:
            data = json.load(f)
        maccs = [d["macc"] for d in data]
        accs = [d["val_acc"] * 100 for d in data]
        ax.scatter(maccs, accs, s=sz, c=color, alpha=alpha, edgecolors="none",
                   label=label, zorder=2 if rnd != "round2c" else 5)

    # Highlight winner
    ax.scatter([276136], [97.40], s=200, c="#E74C3C", edgecolors="black",
               linewidths=1.5, zorder=10, marker="*")
    ax.annotate("Winner\n(276K, 97.40%)", xy=(276136, 97.40),
                xytext=(250000, 96.0),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
                fontsize=8, fontweight="bold", ha="center")

    # MACC budget line
    ax.axvline(x=287673, color="#888", ls="--", lw=1.0, zorder=1)
    ax.text(287673 + 500, ax.get_ylim()[0] + 0.3, "MACC budget\n(287,673)",
            fontsize=7, color="#888", va="bottom")

    ax.set_xlabel("MACCs")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Grid Search: Accuracy vs. Computational Cost")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.savefig(OUT / "fig3_grid_search.png")
    fig.savefig(OUT / "fig3_grid_search.pdf")
    plt.close(fig)
    print("Fig 3 saved.")


# =========================================================================
# Fig 4: Per-Class F1 Bar Chart
# =========================================================================
def fig4_per_class_f1():
    with open(DATA / "final_model" / "eval_f1.json") as f:
        ev = json.load(f)

    classes = ["yes", "no", "up", "down", "left", "right",
               "on", "off", "stop", "go", "unknown", "silence"]

    f1_float = [ev["float"]["per_class"][c]["f1-score"] for c in classes]
    f1_int8 = [ev["int8"]["per_class"][c]["f1-score"] for c in classes]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(classes))
    w = 0.35

    bars1 = ax.bar(x - w / 2, f1_float, w, label="Float32", color="#5DADE2",
                   edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + w / 2, f1_int8, w, label="INT8", color="#E74C3C",
                   edgecolor="white", linewidth=0.5)

    ax.set_ylabel("F1-Score")
    ax.set_title("Per-Class F1-Score: Float32 vs INT8 (12-class)")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylim(0.88, 1.01)
    ax.legend(loc="lower left")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on top of bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                f"{h:.3f}", ha="center", va="bottom", fontsize=6, color="#2874A6")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                f"{h:.3f}", ha="center", va="bottom", fontsize=6, color="#C0392B")

    fig.savefig(OUT / "fig4_per_class_f1.png")
    fig.savefig(OUT / "fig4_per_class_f1.pdf")
    plt.close(fig)
    print("Fig 4 saved.")


# =========================================================================
# Fig 5: Training Curve
# =========================================================================
def fig5_training_curve():
    # Parse training log from swa.log
    log_path = DATA / "final_model" / "swa.log"
    epochs = []
    val_accs = []
    losses = []
    lrs = []

    pattern = re.compile(
        r"ep\s+(\d+):\s+loss=([\d.]+)\s+val_acc=([\d.]+)\s+best=[\d.]+@\d+\s+lr=([\d.]+)"
    )

    with open(log_path, "rb") as f:
        raw = f.read().decode("utf-8", errors="replace")
    for seg in raw.split("\r"):
        m = pattern.search(seg)
        if m:
            ep = int(m.group(1))
            loss = float(m.group(2))
            val_acc = float(m.group(3))
            lr = float(m.group(4))
            epochs.append(ep)
            val_accs.append(val_acc * 100)
            losses.append(loss)
            lrs.append(lr)

    if not epochs:
        print("WARNING: Could not parse training curve data from swa.log")
        return

    print(f"  Parsed {len(epochs)} epochs from swa.log")

    fig, ax1 = plt.subplots(figsize=(7, 3.8))

    # Val accuracy — plot with markers since we have sparse samples
    color_acc = "#2E86C1"
    ax1.plot(epochs, val_accs, color=color_acc, lw=1.2, alpha=0.8,
             marker="o", markersize=3, label="Validation Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Accuracy (%)", color=color_acc)
    ax1.tick_params(axis="y", labelcolor=color_acc)

    # Best epoch marker — use the highest val_acc point from log
    best_ep = 273  # actual best epoch
    best_val = 97.40  # actual best val_acc
    # Find closest logged epoch
    if best_ep in epochs:
        best_idx = epochs.index(best_ep)
        best_val = val_accs[best_idx]
    else:
        # Use highest logged val_acc as proxy
        best_idx = int(np.argmax(val_accs))
        best_val = val_accs[best_idx]
        best_ep = epochs[best_idx]

    ax1.axvline(x=best_ep, color="#E74C3C", ls="--", lw=1.0, alpha=0.7)
    ax1.scatter([best_ep], [best_val], s=80, c="#E74C3C",
                zorder=5, marker="*")
    ax1.annotate(f"Best: {best_val:.2f}%\n(epoch {best_ep})",
                 xy=(best_ep, best_val),
                 xytext=(best_ep - 80, best_val - 3),
                 arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=0.8),
                 fontsize=8, color="#E74C3C")

    # Learning rate on secondary axis
    ax2 = ax1.twinx()
    color_lr = "#E67E22"
    ax2.plot(epochs, lrs, color=color_lr, lw=0.8, alpha=0.5, ls="--",
             label="Learning Rate")
    ax2.set_ylabel("Learning Rate", color=color_lr)
    ax2.tick_params(axis="y", labelcolor=color_lr)
    ax2.set_yscale("log")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right",
               framealpha=0.9)

    ax1.set_title("Training Curve with Cosine Annealing Schedule")
    ax1.grid(True, alpha=0.2)

    fig.savefig(OUT / "fig5_training_curve.png")
    fig.savefig(OUT / "fig5_training_curve.pdf")
    plt.close(fig)
    print("Fig 5 saved.")


# =========================================================================
# Main
# =========================================================================
if __name__ == "__main__":
    print("Generating figures...")
    fig1_pipeline()
    fig2_architecture()
    fig3_grid_search()
    fig4_per_class_f1()
    fig5_training_curve()
    print(f"\nAll figures saved to: {OUT}")
