"""
Stage 2c: Retrain top unique architectures from 2b with 1000 epochs.
Deduplicates by (stem, block, n_blocks, strides, ratio, kernel) — ignores
layers variants that produce identical MACC/accuracy.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from kws.config import AudioConfig, TrainConfig, ArchConfig, build_model
from kws.data import make_loaders
from kws.training import train_one_experiment, evaluate
from kws.utils import count_macc, count_params, set_seed, save_json, ensure_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-n", type=int, default=10,
                   help="Take top-N unique architectures from 2b")
    p.add_argument("--prev", type=str,
                   default="experiments/grid_search/round2b/summary.json")
    args = p.parse_args()

    out_dir = Path("experiments/grid_search/round2c")
    ensure_dir(out_dir)

    # Load 2b results
    with open(args.prev) as f:
        r2b = json.load(f)
    print(f"Loaded {len(r2b)} results from {args.prev}")

    # Deduplicate: keep first (best) entry per unique (stem, block, strides, ratio, kernel)
    seen = set()
    unique = []
    for r in r2b:
        key = (r["stem_ch"], r["block_ch"], tuple(r["strides"]),
               r["ratio"], r["kernel"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    print(f"Unique architectures: {len(unique)} (from {len(r2b)})")

    # Take top-N
    candidates = unique[:args.top_n]
    print(f"\nTop-{len(candidates)} for 1000-epoch finals:")
    for i, r in enumerate(candidates):
        print(f"  {i+1}. {r['name']}  val={r['val_acc']:.4f}  "
              f"test={r.get('test_acc',0):.4f}  MACC={r['macc']:,}")

    # Load data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_cfg = AudioConfig()
    print("\nLoading data to GPU...")
    train_ld, val_ld, test_ld = make_loaders(
        audio_cfg=audio_cfg, batch_size=1024, num_workers=0,
        train_device=device, preload=True, num_classes=12,
    )
    print("Data loaded.\n")

    train_cfg = TrainConfig(
        batch_size=1024, epochs=1000, lr=1e-3, weight_decay=1e-4,
        seed=args.seed, early_stop_patience=30,
        use_amp=True, val_every=1, non_blocking=True,
        scheduler="cosine", warmup_epochs=5,
        label_smoothing=0.1, mixup_alpha=0.2,
        spec_augment=True, spec_time_masks=2, spec_time_width=5,
        spec_freq_masks=1, spec_freq_width=2,
    )

    results = []
    n = len(candidates)
    for i, entry in enumerate(candidates):
        # Reconstruct ArchConfig
        n_blocks = entry["n_blocks"]
        layers = entry.get("n_layers", [1] * n_blocks)
        arch = ArchConfig(
            name=entry["name"],
            model_type="tenet",
            n_channels=[entry["stem_ch"]] + [entry["block_ch"]] * n_blocks,
            n_strides=entry["strides"],
            n_ratios=[entry["ratio"]] * n_blocks,
            n_layers=layers,
            kernel_size=entry["kernel"],
            in_channels=13,
            num_classes=12,
            dropout=0.1,
        )
        macc = entry["macc"]
        params = entry["params"]

        print(f"\n[{i+1}/{n}] {arch.name}  MACC={macc:,}  params={params:,}")

        set_seed(train_cfg.seed)
        model = build_model(arch).to(device)

        t0 = time.perf_counter()
        stats = train_one_experiment(model, train_ld, val_ld, train_cfg, device=device)
        elapsed = time.perf_counter() - t0

        # Test eval
        model.eval()
        test_m = evaluate(model, test_ld, device=device)

        result = {
            "name": arch.name,
            "stem_ch": entry["stem_ch"],
            "block_ch": entry["block_ch"],
            "n_blocks": n_blocks,
            "strides": entry["strides"],
            "n_layers": layers,
            "ratio": entry["ratio"],
            "kernel": entry["kernel"],
            "macc": macc,
            "params": params,
            "val_acc": stats["best_val_acc"],
            "val_loss": stats["best_val_loss"],
            "best_epoch": stats["best_epoch"],
            "test_acc": test_m["acc"],
            "test_loss": test_m["loss"],
            "train_time_s": round(elapsed, 1),
        }
        results.append(result)
        print(f"  val={stats['best_val_acc']:.4f}  test={test_m['acc']:.4f}  "
              f"epoch={stats['best_epoch']}  time={elapsed:.1f}s")

        # Save checkpoint of best model
        ckpt_path = out_dir / f"{arch.name}_best.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"  saved: {ckpt_path}")

        del model
        torch.cuda.empty_cache()

    results.sort(key=lambda r: r["val_acc"], reverse=True)

    # Print final ranking
    print(f"\n{'='*110}")
    print(f"FINAL TOP {len(results)} (1000 epochs + optimizations)")
    print(f"{'='*110}")
    print(f"{'#':<3} {'Name':<55} {'MACC':>8} {'ValAcc':>7} {'TestAcc':>7} {'Epoch':>5}")
    print("-" * 90)
    for i, r in enumerate(results):
        print(f"{i+1:<3} {r['name']:<55} {r['macc']:>8,} "
              f"{r['val_acc']:>6.2%} {r['test_acc']:>6.2%} {r['best_epoch']:>5}")

    save_json(out_dir / "summary.json", results)
    print(f"\nResults saved to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
