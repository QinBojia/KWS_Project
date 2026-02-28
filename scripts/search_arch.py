"""
Architecture search tool. Computes MACC and parameter counts analytically
without training to find architectures matching a target MACC budget.

Usage:
    python scripts/search_arch.py --target-macc 287673 --tolerance 5000
    python scripts/search_arch.py --target-macc 287673 --tolerance 5000 --mode fine
    python scripts/search_arch.py --target-macc 287673 --mode coarse --top 20
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kws.utils import save_json


def calc_arch(stem_ch, stem_stride, blocks, num_classes=8, H=62, W=13):
    """
    Compute MACC and params analytically for a DS-CNN architecture.
    blocks: list of (out_ch, stride)
    Returns total MACC, total params, and per-layer details.
    """
    total_macc = 0
    total_params = 0
    details = []

    # Stem: Conv2d(1, stem_ch, 3, stride, padding=1) + BN + ReLU
    Ho = math.ceil(H / stem_stride)
    Wo = math.ceil(W / stem_stride)
    macc = stem_ch * Ho * Wo * 1 * 9
    params = 1 * stem_ch * 9 + stem_ch * 2
    total_macc += macc
    total_params += params
    details.append(f"  Stem 1->{stem_ch} s{stem_stride}: {H}x{W}->{Ho}x{Wo}  MACC={macc:>8,}  params={params:>6,}")
    H, W = Ho, Wo
    in_ch = stem_ch

    for i, (out_ch, stride) in enumerate(blocks):
        Ho = math.ceil(H / stride)
        Wo = math.ceil(W / stride)
        dw_macc = in_ch * Ho * Wo * 9
        dw_params = in_ch * 9 + in_ch * 2
        pw_macc = out_ch * Ho * Wo * in_ch
        pw_params = in_ch * out_ch + out_ch * 2
        block_macc = dw_macc + pw_macc
        block_params = dw_params + pw_params
        total_macc += block_macc
        total_params += block_params
        details.append(f"  DS{i:>2} {in_ch:>3}->{out_ch:>3} s{stride}: {H:>2}x{W:>2}->{Ho:>2}x{Wo:>2}  MACC={block_macc:>8,}  params={block_params:>6,}")
        H, W = Ho, Wo
        in_ch = out_ch

    fc_macc = in_ch * num_classes
    fc_params = in_ch * num_classes + num_classes
    total_macc += fc_macc
    total_params += fc_params
    details.append(f"  FC {in_ch}->{num_classes}  MACC={fc_macc:>8,}  params={fc_params:>6,}")

    return total_macc, total_params, details


def _gen_stages(ch_options, n_blocks):
    """Generate non-decreasing channel sequences using 2-3 stage values."""
    results = []
    for c1 in ch_options:
        for c2 in [c for c in ch_options if c >= c1]:
            mid = n_blocks // 2
            channels_2 = [c1] * mid + [c2] * (n_blocks - mid)
            results.append(channels_2)

            for c3 in [c for c in ch_options if c >= c2]:
                t1 = max(1, n_blocks // 3)
                t2 = max(1, (n_blocks - t1) // 2)
                t3 = n_blocks - t1 - t2
                channels_3 = [c1] * t1 + [c2] * t2 + [c3] * t3
                results.append(channels_3)
    return results


def search_neighborhood(target, tolerance):
    """Search around arch_b: vary channels in each position."""
    results = []
    stride_pattern = [1, 2, 1, 2, 1]
    ch_options = list(range(8, 65, 2))

    for stem_ch in range(8, 33, 2):
        for c1 in ch_options:
            for c2 in [c for c in ch_options if c >= c1]:
                for c3 in [c for c in ch_options if c >= c2]:
                    channels = [c1, c2, c2, c3, c3]
                    blocks = list(zip(channels, stride_pattern))
                    macc, params, details = calc_arch(stem_ch, 2, blocks)
                    diff = abs(macc - target)
                    if diff <= tolerance:
                        results.append({
                            "name": f"nb_s{stem_ch}s2_{c1}_{c2}_{c3}",
                            "stem_ch": stem_ch, "stem_stride": 2,
                            "block_cfg": blocks,
                            "macc": macc, "params": params,
                            "diff": macc - target,
                        })
    return results


def search_varied_depth(target, tolerance):
    """Try 4-7 block architectures with stem stride-2."""
    results = []
    stride_patterns = {
        4: [[1,2,2,1], [2,1,2,1], [1,2,1,2], [2,2,1,1]],
        5: [[1,2,1,2,1], [2,1,2,1,1], [1,1,2,1,2], [1,2,2,1,1], [2,1,1,2,1]],
        6: [[1,2,1,2,1,1], [1,1,2,1,2,1], [1,2,1,1,2,1], [2,1,2,1,1,1], [1,1,2,1,1,2]],
        7: [[1,2,1,1,2,1,1], [1,1,2,1,1,2,1], [1,2,1,2,1,1,1]],
    }

    ch_options = list(range(8, 65, 4))

    for n_blocks, patterns in stride_patterns.items():
        for strides in patterns:
            for stem_ch in range(8, 33, 4):
                for channels in _gen_stages(ch_options, n_blocks):
                    blocks = list(zip(channels, strides))
                    macc, params, details = calc_arch(stem_ch, 2, blocks)
                    diff = abs(macc - target)
                    if diff <= tolerance:
                        ch_str = "-".join(str(c) for c in channels)
                        s_str = "".join(str(s) for s in strides)
                        results.append({
                            "name": f"d{n_blocks}_s{stem_ch}s2_{ch_str}_p{s_str}",
                            "stem_ch": stem_ch, "stem_stride": 2,
                            "block_cfg": blocks,
                            "macc": macc, "params": params,
                            "diff": macc - target,
                        })
    return results


def search_coarse(target, tolerance):
    """Coarse search with hand-crafted and parametric configs."""
    results = []

    for stem_ch in range(4, 33, 2):
        for stem_s in [1, 2]:
            for c1 in range(stem_ch, 65, 2):
                for c2 in range(c1, 97, 2):
                    for c3 in range(c2, 129, 2):
                        blocks = [(c1, 2), (c1, 1), (c2, 2), (c2, 1), (c3, 2), (c3, 1)]
                        macc, params, details = calc_arch(stem_ch, stem_s, blocks)
                        diff = abs(macc - target)
                        if diff <= tolerance:
                            ch_str = f"{c1}-{c2}-{c3}"
                            results.append({
                                "name": f"coarse_s{stem_ch}s{stem_s}_{ch_str}",
                                "stem_ch": stem_ch, "stem_stride": stem_s,
                                "block_cfg": blocks,
                                "macc": macc, "params": params,
                                "diff": macc - target,
                            })
    return results


def parse_cli():
    parser = argparse.ArgumentParser(description="Architecture search for KWS models")
    parser.add_argument("--target-macc", type=int, default=287673, help="Target MACC budget")
    parser.add_argument("--tolerance", type=int, default=5000, help="MACC tolerance (+/-)")
    parser.add_argument("--mode", choices=["coarse", "fine", "all"], default="fine",
                        help="Search mode: coarse (6-block parametric), fine (neighborhood + varied depth), all")
    parser.add_argument("--top", type=int, default=30, help="Number of top results to display")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")
    return parser.parse_args()


def main():
    args = parse_cli()
    target = args.target_macc
    tolerance = args.tolerance

    print(f"Target MACC: {target:,} (tolerance: +/-{tolerance:,})")

    all_results = []

    if args.mode in ("fine", "all"):
        print("Searching arch_b neighborhood...")
        r1 = search_neighborhood(target, tolerance)
        print(f"  Found {len(r1)} configs")
        all_results.extend(r1)

        print("Searching varied depths (4-7 blocks)...")
        r2 = search_varied_depth(target, tolerance)
        print(f"  Found {len(r2)} configs")
        all_results.extend(r2)

    if args.mode in ("coarse", "all"):
        print("Searching coarse parametric (6-block)...")
        r3 = search_coarse(target, tolerance)
        print(f"  Found {len(r3)} configs")
        all_results.extend(r3)

    # Deduplicate
    seen = set()
    unique = []
    for r in all_results:
        key = (r["stem_ch"], r["stem_stride"], tuple(tuple(b) for b in r["block_cfg"]))
        if key not in seen:
            seen.add(key)
            unique.append(r)

    print(f"\nTotal unique: {len(unique)} configs")

    unique.sort(key=lambda r: (abs(r["diff"]), r["params"]))

    # Print results
    print(f"\n{'='*90}")
    print(f"TOP {args.top} CANDIDATES (sorted by MACC closeness, then params)")
    print(f"{'='*90}")
    print(f"{'Name':<50} {'MACC':>10} {'Params':>8} {'KB':>6} {'Diff':>7} {'Blks':>4}")
    print("-" * 90)

    for r in unique[:args.top]:
        n_blk = len(r["block_cfg"])
        print(f"{r['name']:<50} {r['macc']:>10,} {r['params']:>8,} "
              f"{r['params']/1024:>5.1f} {r['diff']:>+7,} {n_blk:>4}")

    if args.output:
        save_json(args.output, unique[:args.top])
        print(f"\nSaved top {args.top} results to {args.output}")


if __name__ == "__main__":
    main()
