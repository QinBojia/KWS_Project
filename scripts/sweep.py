"""
Sweep runner: train multiple configs and compare results.

Usage:
    python scripts/sweep.py --configs configs/arch_a.yaml configs/arch_b.yaml configs/arch_c.yaml
    python scripts/sweep.py --configs-dir configs/ --output-dir experiments/sweep_001
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kws.config import load_config
from kws.utils import save_json, ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Run training sweep across multiple configs")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--configs", nargs="+", help="List of YAML config files")
    group.add_argument("--configs-dir", type=str, help="Directory containing YAML configs")
    parser.add_argument("--output-dir", type=str, default="experiments",
                        help="Base output directory (default: experiments/)")
    parser.add_argument("--export-onnx", action="store_true", help="Export ONNX for each model")
    parser.add_argument("--onnx-opset", type=int, default=13)
    parser.add_argument("--calib-batches", type=int, default=30)
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, load existing weights")

    # Override training params
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    # Collect config files
    if args.configs:
        config_files = [Path(c) for c in args.configs]
    else:
        config_dir = Path(args.configs_dir)
        config_files = sorted(config_dir.glob("*.yaml"))

    if not config_files:
        print("No config files found.")
        return

    print(f"Found {len(config_files)} configs to sweep:")
    for cf in config_files:
        print(f"  - {cf}")

    # Import train.py's run function
    from scripts.train import run, apply_overrides

    base_out = Path(args.output_dir)
    all_results = []

    for cf in config_files:
        exp = load_config(str(cf))
        exp = apply_overrides(exp, args)
        out_dir = base_out / exp.name
        ensure_dir(out_dir)

        print(f"\n{'='*60}")
        print(f"  SWEEP: {exp.name}")
        print(f"{'='*60}")

        try:
            result = run(exp, out_dir, args)
            all_results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results.append({"config": exp.name, "error": str(e)})

    # Summary
    print(f"\n{'='*80}")
    print("SWEEP SUMMARY")
    print(f"{'='*80}")
    print(f"{'Config':<25} {'MACC':>10} {'Float%':>8} {'INT8%':>8} {'INT8 KB':>8} {'Benchmark':>10}")
    print("-" * 80)

    for r in all_results:
        if "error" in r:
            print(f"{r['config']:<25} ERROR: {r['error']}")
            continue
        int8_acc = r.get("int8", {}).get("acc", 0)
        int8_kb = r.get("int8", {}).get("size_kb", 0)
        marker = "PASS" if int8_acc >= 0.91 else "BELOW"
        print(f"{r['config']:<25} {r['macc']:>10,} {r['float']['acc']:>7.2%} "
              f"{int8_acc:>7.2%} {int8_kb:>7.1f} {marker:>10}")

    print(f"\nBenchmark: 287,673 MACC, >=91% accuracy")

    save_json(base_out / "sweep_summary.json", all_results)
    print(f"Summary saved to {base_out / 'sweep_summary.json'}")


if __name__ == "__main__":
    main()
