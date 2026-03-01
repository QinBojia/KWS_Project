"""
TENet hyperparameter grid search — multi-stage progressive refinement.

Stage 1 (Round 1, done): Coarse Cartesian, large step, 15 epochs.
Stage 2a: Fine Cartesian (large step + layers), 15 epochs, no optimizations → top 100.
Stage 2b: Top 100, 200 epochs + optimizations + early stop → top 10.
Stage 2c: Top 10, 1000 epochs + optimizations + early stop → final ranking.

Usage:
    python scripts/grid_search_tenet.py --round 1   --output-dir experiments/grid_search/round1
    python scripts/grid_search_tenet.py --round 2a   --output-dir experiments/grid_search/round2a
    python scripts/grid_search_tenet.py --round 2b   --output-dir experiments/grid_search/round2b
    python scripts/grid_search_tenet.py --round 2c   --output-dir experiments/grid_search/round2c
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from kws.config import AudioConfig, TrainConfig, ArchConfig, build_model
from kws.data import make_loaders
from kws.training import train_one_experiment, evaluate
from kws.utils import count_macc, count_params, set_seed, save_json, ensure_dir

MACC_BUDGET = 287_673


# ──────────────────────────────────────────────────────────────
# Grid definitions
# ──────────────────────────────────────────────────────────────

STRIDE_PATTERNS = {
    2: [[2, 1], [1, 2]],
    3: [[2, 1, 1], [1, 2, 1], [2, 2, 1]],
    4: [[2, 1, 1, 1], [2, 2, 1, 1], [1, 2, 1, 1]],
}

ROUND1_GRID = {
    "stem_ch":    [8, 12, 16, 20, 24],
    "block_ch":   [16, 20, 24, 28, 32],
    "n_blocks":   [2, 3, 4],
    "ratio":      [1, 2, 3, 4],
    "kernel":     [3, 5, 7, 9, 11, 15],
}

# Stage 2a: coarse step (step=4) + layers, centered on Round 1 winners
ROUND2A_GRID = {
    "stem_ch":    [12, 16, 20],
    "block_ch":   [16, 20, 24],
    "n_blocks":   [3, 4],
    "ratio":      [2, 3, 4],
    "kernel":     [7, 9, 11, 15],
}

# Stage 2b/2c: fine step (step=2), derived from 2a top results at runtime
# (grid generated from top-N ±2 in each dimension)

# Layers patterns: all-1s baseline + one block bumped to 2
LAYERS_PATTERNS = {
    3: [[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2]],
    4: [[1, 1, 1, 1], [2, 1, 1, 1], [1, 2, 1, 1], [1, 1, 2, 1], [1, 1, 1, 2]],
}


def _make_arch_config(stem_ch, block_ch, n_blocks, strides, ratio, kernel,
                      layers=None, num_classes=12, dropout=0.1) -> ArchConfig:
    """Build ArchConfig for a TENet candidate."""
    n_channels = [stem_ch] + [block_ch] * n_blocks
    n_ratios = [ratio] * n_blocks
    n_layers = layers if layers is not None else [1] * n_blocks
    layers_tag = ''.join(map(str, n_layers))
    return ArchConfig(
        name=f"t_s{stem_ch}_b{block_ch}_n{n_blocks}_st{''.join(map(str, strides))}_r{ratio}_k{kernel}_L{layers_tag}",
        model_type="tenet",
        n_channels=n_channels,
        n_strides=strides,
        n_ratios=n_ratios,
        n_layers=n_layers,
        kernel_size=kernel,
        in_channels=13,
        num_classes=num_classes,
        dropout=dropout,
    )


def _generate_cartesian(grid: Dict, macc_lo_pct: float = 0.5) -> List[ArchConfig]:
    """Generate Cartesian product candidates with layers variations, MACC filtered."""
    candidates = []
    total = 0
    filtered = 0
    audio = AudioConfig()
    input_shape = audio.input_shape
    seen = set()

    for stem_ch, block_ch, n_blocks, ratio, kernel in itertools.product(
        grid["stem_ch"], grid["block_ch"], grid["n_blocks"],
        grid["ratio"], grid["kernel"],
    ):
        for strides in STRIDE_PATTERNS[n_blocks]:
            for layers in LAYERS_PATTERNS[n_blocks]:
                total += 1
                key = (stem_ch, block_ch, n_blocks, tuple(strides),
                       ratio, kernel, tuple(layers))
                if key in seen:
                    continue
                seen.add(key)

                arch = _make_arch_config(stem_ch, block_ch, n_blocks, strides,
                                         ratio, kernel, layers=layers)
                try:
                    model = build_model(arch)
                    macc = count_macc(model, input_shape)
                    if MACC_BUDGET * macc_lo_pct <= macc <= MACC_BUDGET:
                        arch._macc = macc
                        arch._params = count_params(model)
                        candidates.append(arch)
                    else:
                        filtered += 1
                except Exception:
                    filtered += 1
                del model

    print(f"Grid: {total} total, {filtered} filtered by MACC, "
          f"{len(candidates)} candidates to train")
    return candidates


def generate_round1_candidates() -> List[ArchConfig]:
    """Round 1: coarse search, layers all-1s only."""
    candidates = []
    total = 0
    filtered = 0
    audio = AudioConfig()
    input_shape = audio.input_shape

    for stem_ch, block_ch, n_blocks, ratio, kernel in itertools.product(
        ROUND1_GRID["stem_ch"], ROUND1_GRID["block_ch"],
        ROUND1_GRID["n_blocks"], ROUND1_GRID["ratio"], ROUND1_GRID["kernel"],
    ):
        for strides in STRIDE_PATTERNS[n_blocks]:
            total += 1
            arch = _make_arch_config(stem_ch, block_ch, n_blocks, strides, ratio, kernel)
            try:
                model = build_model(arch)
                macc = count_macc(model, input_shape)
                if MACC_BUDGET * 0.5 <= macc <= MACC_BUDGET:
                    arch._macc = macc
                    arch._params = count_params(model)
                    candidates.append(arch)
                else:
                    filtered += 1
            except Exception:
                filtered += 1
            del model

    print(f"Grid: {total} total, {filtered} filtered by MACC, "
          f"{len(candidates)} candidates to train")
    return candidates


def generate_refinement_candidates(prev_results: List[Dict], top_n: int) -> List[ArchConfig]:
    """Generate fine-step (±2) candidates around top-N from previous stage."""
    candidates = []
    audio = AudioConfig()
    input_shape = audio.input_shape
    seen = set()

    for entry in prev_results[:top_n]:
        base_stem = entry["stem_ch"]
        base_block = entry["block_ch"]
        base_n = entry["n_blocks"]
        base_strides = entry["strides"]
        base_layers = entry.get("n_layers", [1] * base_n)
        base_ratio = entry["ratio"]
        base_kernel = entry["kernel"]

        for d_stem in [-2, 0, 2]:
            for d_block in [-2, 0, 2]:
                for d_kernel in [-2, 0, 2]:
                    for d_ratio in [-1, 0, 1]:
                        stem = base_stem + d_stem
                        block = base_block + d_block
                        kernel = base_kernel + d_kernel
                        ratio = base_ratio + d_ratio
                        if stem < 4 or block < 4 or kernel < 3 or ratio < 1:
                            continue

                        # Try base layers + all layer variants
                        layers_list = [base_layers]
                        for lp in LAYERS_PATTERNS.get(base_n, []):
                            if lp != base_layers:
                                layers_list.append(lp)

                        for layers in layers_list:
                            key = (stem, block, base_n, tuple(base_strides),
                                   ratio, kernel, tuple(layers))
                            if key in seen:
                                continue
                            seen.add(key)

                            arch = _make_arch_config(stem, block, base_n,
                                                     base_strides, ratio, kernel,
                                                     layers=layers)
                            try:
                                model = build_model(arch)
                                macc = count_macc(model, input_shape)
                                if MACC_BUDGET * 0.5 <= macc <= MACC_BUDGET:
                                    arch._macc = macc
                                    arch._params = count_params(model)
                                    candidates.append(arch)
                            except Exception:
                                pass
                            del model

    print(f"Refinement: {len(candidates)} candidates (from top-{top_n})")
    return candidates


# ──────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────

def run_search(candidates: List[ArchConfig], train_ld, val_ld, test_ld,
               train_cfg: TrainConfig, device: str, out_dir: Path,
               do_test_eval: bool = True) -> List[Dict]:
    """Train all candidates and collect results."""
    results = []
    n = len(candidates)

    for i, arch in enumerate(candidates):
        name = arch.name
        macc = getattr(arch, '_macc', 0)
        params = getattr(arch, '_params', 0)

        print(f"\n[{i + 1}/{n}] {name}  MACC={macc:,}  params={params:,}")

        set_seed(train_cfg.seed)
        model = build_model(arch).to(device)

        t0 = time.perf_counter()
        train_stats = train_one_experiment(model, train_ld, val_ld, train_cfg, device=device)
        elapsed = time.perf_counter() - t0

        result = {
            "name": name,
            "stem_ch": arch.n_channels[0],
            "block_ch": arch.n_channels[1] if len(arch.n_channels) > 1 else 0,
            "n_blocks": len(arch.n_strides),
            "strides": arch.n_strides,
            "n_layers": arch.n_layers,
            "ratio": arch.n_ratios[0] if arch.n_ratios else 0,
            "kernel": arch.kernel_size,
            "macc": macc,
            "params": params,
            "val_acc": train_stats["best_val_acc"],
            "val_loss": train_stats["best_val_loss"],
            "best_epoch": train_stats["best_epoch"],
            "train_time_s": round(elapsed, 1),
        }

        if do_test_eval:
            model.eval()
            test_metrics = evaluate(model, test_ld, device=device)
            result["test_acc"] = test_metrics["acc"]
            result["test_loss"] = test_metrics["loss"]

        results.append(result)
        print(f"  val_acc={train_stats['best_val_acc']:.4f}  "
              f"epoch={train_stats['best_epoch']}  time={elapsed:.1f}s")

        del model
        torch.cuda.empty_cache()

    results.sort(key=lambda r: r["val_acc"], reverse=True)
    return results


def print_top_results(results: List[Dict], top_n: int = 20):
    """Print top-N results table."""
    print(f"\n{'=' * 110}")
    print(f"TOP {min(top_n, len(results))} RESULTS")
    print(f"{'=' * 110}")
    header = (f"{'Rank':>4} {'Name':<60} {'MACC':>8} {'Params':>7} "
              f"{'ValAcc':>7} {'TestAcc':>8} {'Epoch':>5} {'Time':>6}")
    print(header)
    print("-" * 110)

    for i, r in enumerate(results[:top_n]):
        test_acc = r.get("test_acc", 0)
        print(f"{i + 1:>4} {r['name']:<60} {r['macc']:>8,} {r['params']:>7,} "
              f"{r['val_acc']:>6.2%} {test_acc:>7.2%} {r['best_epoch']:>5} "
              f"{r['train_time_s']:>5.1f}s")


def _load_prev_results(path_str: str, auto_dir: str) -> List[Dict]:
    """Load previous stage results, with auto-detection fallback."""
    if path_str:
        p = Path(path_str)
    else:
        p = Path(auto_dir) / "summary.json"
    if not p.exists():
        print(f"ERROR: {p} not found")
        sys.exit(1)
    with open(p) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} results from {p}")
    return data


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="TENet grid search (multi-stage)")
    p.add_argument("--round", type=str, required=True, choices=["1", "2a", "2b", "2c"])
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--prev-results", type=str, default=None,
                   help="Path to previous stage summary.json (auto-detected if omitted)")
    p.add_argument("--top-n", type=int, default=100,
                   help="Top-N from previous stage to refine (default: 100 for 2b, 10 for 2c)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    base_dir = Path("experiments/grid_search")

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = base_dir / f"round{args.round}"
    ensure_dir(out_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_cfg = AudioConfig()

    # Load data ONCE
    print("Loading data to GPU (shared across all candidates)...")
    train_ld, val_ld, test_ld = make_loaders(
        audio_cfg=audio_cfg, batch_size=1024, num_workers=0,
        train_device=device, preload=True, num_classes=12,
    )
    print("Data loaded.\n")

    if args.round == "1":
        # ── Round 1: Coarse Cartesian, all-1 layers, 15 epochs ──
        candidates = generate_round1_candidates()
        train_cfg = TrainConfig(
            batch_size=1024, epochs=15, lr=1e-3, weight_decay=1e-4,
            seed=args.seed, early_stop_patience=0,
            use_amp=True, val_every=1, non_blocking=True,
        )
        results = run_search(candidates, train_ld, val_ld, test_ld,
                             train_cfg, device, out_dir, do_test_eval=True)

    elif args.round == "2a":
        # ── Stage 2a: Coarse step + layers, 15 epochs, no optimizations ──
        candidates = _generate_cartesian(ROUND2A_GRID, macc_lo_pct=0.5)
        train_cfg = TrainConfig(
            batch_size=1024, epochs=15, lr=1e-3, weight_decay=1e-4,
            seed=args.seed, early_stop_patience=0,
            use_amp=True, val_every=1, non_blocking=True,
        )
        results = run_search(candidates, train_ld, val_ld, test_ld,
                             train_cfg, device, out_dir, do_test_eval=True)

    elif args.round == "2b":
        # ── Stage 2b: Refine top-100 from 2a, 200 epochs + optimizations ──
        prev = _load_prev_results(args.prev_results, str(base_dir / "round2a"))
        top_n = args.top_n if args.top_n != 100 else 100  # default 100
        candidates = generate_refinement_candidates(prev, top_n=top_n)
        train_cfg = TrainConfig(
            batch_size=1024, epochs=200, lr=1e-3, weight_decay=1e-4,
            seed=args.seed, early_stop_patience=20,
            use_amp=True, val_every=1, non_blocking=True,
            scheduler="cosine", warmup_epochs=5,
            label_smoothing=0.1, mixup_alpha=0.2,
            spec_augment=True, spec_time_masks=2, spec_time_width=5,
            spec_freq_masks=1, spec_freq_width=2,
        )
        results = run_search(candidates, train_ld, val_ld, test_ld,
                             train_cfg, device, out_dir, do_test_eval=True)

    elif args.round == "2c":
        # ── Stage 2c: Final top-10 from 2b, 1000 epochs + optimizations ──
        prev = _load_prev_results(args.prev_results, str(base_dir / "round2b"))
        top_n = args.top_n if args.top_n != 100 else 10  # default 10
        candidates = generate_refinement_candidates(prev, top_n=top_n)
        train_cfg = TrainConfig(
            batch_size=1024, epochs=1000, lr=1e-3, weight_decay=1e-4,
            seed=args.seed, early_stop_patience=30,
            use_amp=True, val_every=1, non_blocking=True,
            scheduler="cosine", warmup_epochs=5,
            label_smoothing=0.1, mixup_alpha=0.2,
            spec_augment=True, spec_time_masks=2, spec_time_width=5,
            spec_freq_masks=1, spec_freq_width=2,
        )
        results = run_search(candidates, train_ld, val_ld, test_ld,
                             train_cfg, device, out_dir, do_test_eval=True)

    print_top_results(results, top_n=20)
    save_json(out_dir / "summary.json", results)
    print(f"\nResults saved to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
