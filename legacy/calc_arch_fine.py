"""
Fine-grained architecture search around arch_b's structure.
Explores variations in channel widths, block counts, and stride patterns
while targeting ~287K MACC.

arch_b reference: stem=16 s2 -> [(16,1),(32,2),(32,1),(32,2),(32,1)]
  MACC=286,240  Params=5,976  INT8~6.3KB  Acc=94.94%
"""
import math
import itertools
from calc_arch import calc_arch

TARGET = 287673
TOLERANCE = 5000  # within 5K MACC of target


def search_neighborhood():
    """Search around arch_b: vary channels in each position."""
    results = []

    # arch_b topology: stem s2 -> 5 blocks with strides [1, 2, 1, 2, 1]
    stride_pattern = [1, 2, 1, 2, 1]

    for stem_ch in range(8, 33, 2):
        for stem_s in [2]:  # stem stride-2 is key to arch_b's success
            # Each block can have different channel width
            # Use ranges around arch_b's 16->32 pattern
            ch_options = list(range(8, 65, 2))  # 8,10,12,...,64

            # For 5 blocks, enumerate channel progressions (monotonically non-decreasing)
            # To keep search tractable, use 3 stages: c1 (blocks 0), c2 (blocks 1-2), c3 (blocks 3-4)
            for c1 in ch_options:
                for c2 in [c for c in ch_options if c >= c1]:
                    for c3 in [c for c in ch_options if c >= c2]:
                        channels = [c1, c2, c2, c3, c3]
                        blocks = list(zip(channels, stride_pattern))
                        macc, params, details = calc_arch(stem_ch, stem_s, blocks)
                        diff = abs(macc - TARGET)
                        if diff <= TOLERANCE:
                            results.append({
                                "name": f"nb_s{stem_ch}s2_{c1}_{c2}_{c3}",
                                "stem_ch": stem_ch, "stem_stride": stem_s,
                                "block_cfg": blocks,
                                "macc": macc, "params": params,
                                "diff": macc - TARGET,
                                "details": details,
                            })
    return results


def search_varied_depth():
    """Try 4, 5, 6, and 7 block architectures with stem stride-2."""
    results = []

    # Generate stride patterns for different depths
    # Key: must have 2-3 stride-2 blocks for sufficient downsampling
    stride_patterns = {
        4: [
            [1, 2, 2, 1],
            [2, 1, 2, 1],
            [1, 2, 1, 2],
            [2, 2, 1, 1],
        ],
        5: [
            [1, 2, 1, 2, 1],  # arch_b pattern
            [2, 1, 2, 1, 1],
            [1, 1, 2, 1, 2],
            [1, 2, 2, 1, 1],
            [2, 1, 1, 2, 1],
        ],
        6: [
            [1, 2, 1, 2, 1, 1],
            [1, 1, 2, 1, 2, 1],
            [1, 2, 1, 1, 2, 1],
            [2, 1, 2, 1, 1, 1],
            [1, 1, 2, 1, 1, 2],
        ],
        7: [
            [1, 2, 1, 1, 2, 1, 1],
            [1, 1, 2, 1, 1, 2, 1],
            [1, 2, 1, 2, 1, 1, 1],
        ],
    }

    ch_options = list(range(8, 65, 4))  # coarser for speed: 8,12,16,...,64

    for n_blocks, patterns in stride_patterns.items():
        for strides in patterns:
            for stem_ch in range(8, 33, 4):
                # Generate channel progressions (non-decreasing, 2-3 stages)
                if n_blocks == 4:
                    stage_configs = _gen_stages(ch_options, 4)
                elif n_blocks == 5:
                    stage_configs = _gen_stages(ch_options, 5)
                elif n_blocks == 6:
                    stage_configs = _gen_stages(ch_options, 6)
                else:
                    stage_configs = _gen_stages(ch_options, 7)

                for channels in stage_configs:
                    blocks = list(zip(channels, strides))
                    macc, params, details = calc_arch(stem_ch, 2, blocks)
                    diff = abs(macc - TARGET)
                    if diff <= TOLERANCE:
                        ch_str = "-".join(str(c) for c in channels)
                        s_str = "".join(str(s) for s in strides)
                        results.append({
                            "name": f"d{n_blocks}_s{stem_ch}s2_{ch_str}_p{s_str}",
                            "stem_ch": stem_ch, "stem_stride": 2,
                            "block_cfg": blocks,
                            "macc": macc, "params": params,
                            "diff": macc - TARGET,
                            "details": details,
                        })
    return results


def _gen_stages(ch_options, n_blocks):
    """Generate non-decreasing channel sequences of length n_blocks using 2-3 stage values."""
    results = []
    for c1 in ch_options:
        for c2 in [c for c in ch_options if c >= c1]:
            # 2-stage: first half c1, second half c2
            mid = n_blocks // 2
            channels_2 = [c1] * mid + [c2] * (n_blocks - mid)
            results.append(channels_2)

            # 3-stage: split into thirds
            for c3 in [c for c in ch_options if c >= c2]:
                t1 = max(1, n_blocks // 3)
                t2 = max(1, (n_blocks - t1) // 2)
                t3 = n_blocks - t1 - t2
                channels_3 = [c1] * t1 + [c2] * t2 + [c3] * t3
                results.append(channels_3)
    return results


def search_stem_stride1():
    """Also try stem stride-1 with more aggressive block strides."""
    results = []

    stride_patterns = [
        [2, 1, 2, 1, 2, 1],  # 3 stride-2 blocks
        [2, 1, 2, 1, 2],     # 5 blocks, 3 stride-2
        [2, 2, 1, 2, 1],     # 5 blocks, 3 stride-2
        [1, 2, 1, 2, 2, 1],  # 6 blocks, 3 stride-2
        [2, 1, 2, 2, 1],     # 5 blocks, 3 stride-2
    ]

    ch_options = list(range(4, 49, 2))

    for strides in stride_patterns:
        n_blocks = len(strides)
        for stem_ch in range(4, 17, 2):
            for channels in _gen_stages(ch_options, n_blocks):
                blocks = list(zip(channels, strides))
                macc, params, details = calc_arch(stem_ch, 1, blocks)
                diff = abs(macc - TARGET)
                if diff <= TOLERANCE:
                    ch_str = "-".join(str(c) for c in channels)
                    s_str = "".join(str(s) for s in strides)
                    results.append({
                        "name": f"s1_d{n_blocks}_s{stem_ch}_{ch_str}_p{s_str}",
                        "stem_ch": stem_ch, "stem_stride": 1,
                        "block_cfg": blocks,
                        "macc": macc, "params": params,
                        "diff": macc - TARGET,
                        "details": details,
                    })
    return results


def main():
    print(f"Target MACC: {TARGET:,} (tolerance: +/-{TOLERANCE:,})")
    print(f"arch_b reference: MACC=286,240  Params=5,976  Acc=94.94%\n")

    # Run all searches
    print("Searching arch_b neighborhood...")
    r1 = search_neighborhood()
    print(f"  Found {len(r1)} configs")

    print("Searching varied depths (4-7 blocks)...")
    r2 = search_varied_depth()
    print(f"  Found {len(r2)} configs")

    print("Searching stem stride-1 variants...")
    r3 = search_stem_stride1()
    print(f"  Found {len(r3)} configs")

    all_results = r1 + r2 + r3
    print(f"\nTotal: {len(all_results)} configs within tolerance")

    # Deduplicate by (stem_ch, stem_stride, block_cfg)
    seen = set()
    unique = []
    for r in all_results:
        key = (r["stem_ch"], r["stem_stride"], tuple(tuple(b) for b in r["block_cfg"]))
        if key not in seen:
            seen.add(key)
            unique.append(r)

    print(f"Unique: {len(unique)} configs")

    # Sort by closeness to target, then by fewer params
    unique.sort(key=lambda r: (abs(r["diff"]), r["params"]))

    # Print top 30
    print(f"\n{'='*90}")
    print(f"TOP 30 CANDIDATES (sorted by MACC closeness, then params)")
    print(f"{'='*90}")
    print(f"{'Name':<50} {'MACC':>10} {'Params':>8} {'KB':>6} {'Diff':>7} {'Blks':>4}")
    print("-" * 90)

    for r in unique[:30]:
        n_blk = len(r["block_cfg"])
        print(f"{r['name']:<50} {r['macc']:>10,} {r['params']:>8,} "
              f"{r['params']/1024:>5.1f} {r['diff']:>+7,} {n_blk:>4}")

    # Show configs that have FEWER params than arch_b (5,976) but close MACC
    print(f"\n{'='*90}")
    print(f"SMALLER THAN arch_b (params < 5,976)")
    print(f"{'='*90}")
    smaller = [r for r in unique if r["params"] < 5976]
    smaller.sort(key=lambda r: -r["params"])  # largest first among small ones
    print(f"Found {len(smaller)} configs")
    print(f"{'Name':<50} {'MACC':>10} {'Params':>8} {'KB':>6} {'Diff':>7}")
    print("-" * 90)
    for r in smaller[:20]:
        print(f"{r['name']:<50} {r['macc']:>10,} {r['params']:>8,} "
              f"{r['params']/1024:>5.1f} {r['diff']:>+7,}")

    # Show configs with MORE params (potentially higher accuracy)
    print(f"\n{'='*90}")
    print(f"LARGER THAN arch_b (params 6,000-15,000, potentially higher accuracy)")
    print(f"{'='*90}")
    larger = [r for r in unique if 6000 <= r["params"] <= 15000]
    larger.sort(key=lambda r: -r["params"])  # largest first
    print(f"Found {len(larger)} configs")
    print(f"{'Name':<50} {'MACC':>10} {'Params':>8} {'KB':>6} {'Diff':>7}")
    print("-" * 90)
    for r in larger[:20]:
        print(f"{r['name']:<50} {r['macc']:>10,} {r['params']:>8,} "
              f"{r['params']/1024:>5.1f} {r['diff']:>+7,}")

    # Show detailed breakdown for top 5 overall
    print(f"\n{'='*90}")
    print(f"DETAILED BREAKDOWN: Top 5")
    print(f"{'='*90}")
    for r in unique[:5]:
        print(f"\n--- {r['name']} ---")
        print(f"MACC={r['macc']:,} ({r['diff']:+,})  Params={r['params']:,}  INT8~{r['params']/1024:.1f}KB")
        print(f"stem_ch={r['stem_ch']}  stem_stride={r['stem_stride']}")
        print(f"block_cfg={r['block_cfg']}")
        for d in r["details"]:
            print(d)


if __name__ == "__main__":
    main()
