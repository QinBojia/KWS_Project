"""
Search for thin DS-CNN architectures targeting ~287K MACC.
Our model already achieves 93.88% accuracy (float) with w=0.25 (~2M MACC).
We have accuracy headroom vs the paper's 91%, so we can aggressively shrink.

Input: 62x13x1 MFCC, 8 classes.
"""
import math


def calc_arch(stem_ch, stem_stride, blocks, num_classes=8, H=62, W=13):
    """
    blocks: list of (out_ch, stride)
    Returns total MACC, total params (trainable), and per-layer details.
    All convs bias=False, BN affine=True. Only counting trainable params.
    """
    total_macc = 0
    total_params = 0
    details = []

    # Stem: Conv2d(1, stem_ch, 3, stride, padding=1) + BN + ReLU
    Ho = math.ceil(H / stem_stride)
    Wo = math.ceil(W / stem_stride)
    macc = stem_ch * Ho * Wo * 1 * 9
    params = 1 * stem_ch * 9 + stem_ch * 2  # conv + BN(gamma,beta)
    total_macc += macc
    total_params += params
    details.append(f"  Stem 1->{stem_ch} s{stem_stride}: {H}x{W}->{Ho}x{Wo}  MACC={macc:>8,}  params={params:>6,}")
    H, W = Ho, Wo
    in_ch = stem_ch

    for i, (out_ch, stride) in enumerate(blocks):
        Ho = math.ceil(H / stride)
        Wo = math.ceil(W / stride)
        # DW
        dw_macc = in_ch * Ho * Wo * 9
        dw_params = in_ch * 9 + in_ch * 2
        # PW
        pw_macc = out_ch * Ho * Wo * in_ch
        pw_params = in_ch * out_ch + out_ch * 2
        block_macc = dw_macc + pw_macc
        block_params = dw_params + pw_params
        total_macc += block_macc
        total_params += block_params
        details.append(f"  DS{i:>2} {in_ch:>3}->{out_ch:>3} s{stride}: {H:>2}x{W:>2}->{Ho:>2}x{Wo:>2}  MACC={block_macc:>8,}  params={block_params:>6,}")
        H, W = Ho, Wo
        in_ch = out_ch

    # FC
    fc_macc = in_ch * num_classes
    fc_params = in_ch * num_classes + num_classes
    total_macc += fc_macc
    total_params += fc_params
    details.append(f"  FC {in_ch}->{num_classes}  MACC={fc_macc:>8,}  params={fc_params:>6,}")

    return total_macc, total_params, details


TARGET = 287673

# ─── Define candidate architectures ─────────────────────────────────────
configs = []

# Strategy 1: Very thin channels (4-32), moderate depth, 3 stride-2
for stem in [4, 8]:
    for ss in [1, 2]:
        for mid in [8, 16]:
            for final in [32, 64]:
                configs.append((
                    f"Thin s{ss} {stem}-{mid}-{final}",
                    stem, ss,
                    [(stem, 1), (mid, 2), (mid, 1), (final, 2), (final, 1), (final, 2), (final, 1)]
                ))
                # Deeper variant
                configs.append((
                    f"ThinDeep s{ss} {stem}-{mid}-{final}",
                    stem, ss,
                    [(stem, 1), (mid, 2), (mid, 1), (mid, 1), (final, 2), (final, 1), (final, 1), (final, 2), (final, 1)]
                ))

# Strategy 2: Gradual channel growth with aggressive stride
for stem in [4, 6, 8]:
    for ss in [1, 2]:
        configs.append((
            f"Grad s{ss} {stem}->{stem*2}->{stem*4}->{stem*8}",
            stem, ss,
            [(stem, 1), (stem*2, 2), (stem*2, 1), (stem*4, 2), (stem*4, 1), (stem*8, 2), (stem*8, 1)]
        ))
        # With extra blocks
        configs.append((
            f"GradX s{ss} {stem}->{stem*2}->{stem*4}->{stem*8}",
            stem, ss,
            [(stem, 1), (stem*2, 2), (stem*2, 1), (stem*2, 1), (stem*4, 2), (stem*4, 1), (stem*4, 1), (stem*8, 2), (stem*8, 1)]
        ))

# Strategy 3: Wide stem with fast downsample
for stem in [16, 32]:
    configs.append((
        f"Wide s2 {stem}->{stem*2}->{stem*4}",
        stem, 2,
        [(stem, 1), (stem*2, 2), (stem*2, 1), (stem*4, 2), (stem*4, 1)]
    ))
    configs.append((
        f"WideDeep s2 {stem}->{stem*2}->{stem*4}",
        stem, 2,
        [(stem, 1), (stem, 1), (stem*2, 2), (stem*2, 1), (stem*4, 2), (stem*4, 1), (stem*4, 1)]
    ))

# Strategy 4: Custom hand-tuned for ~287K
configs.extend([
    ("Custom-A", 8, 1, [(8,1),(16,2),(16,1),(32,2),(32,1),(32,2),(32,1)]),
    ("Custom-B", 8, 1, [(8,1),(8,1),(16,2),(16,1),(32,2),(32,1),(32,2)]),
    ("Custom-C", 4, 1, [(8,1),(8,2),(16,1),(16,2),(32,1),(32,2),(32,1),(32,1)]),
    ("Custom-D", 8, 2, [(8,1),(16,2),(16,1),(32,2),(32,1),(32,1)]),
    ("Custom-E", 8, 2, [(16,1),(16,2),(32,1),(32,2),(32,1)]),
    ("Custom-F", 16, 2, [(16,1),(32,2),(32,1),(32,2),(32,1)]),
    ("Custom-G", 16, 2, [(16,1),(32,2),(32,1),(64,2),(64,1)]),
    ("Custom-H", 8, 1, [(16,2),(16,1),(32,2),(32,1),(64,2),(64,1)]),
    ("Custom-I", 8, 1, [(16,2),(16,1),(16,1),(32,2),(32,1),(64,2)]),
    ("Custom-J", 8, 1, [(8,1),(16,2),(16,1),(24,2),(24,1),(32,2),(32,1)]),
    ("Custom-K", 8, 1, [(8,1),(16,2),(16,1),(24,2),(32,1),(32,2)]),
    ("Custom-L", 4, 1, [(4,1),(8,2),(8,1),(16,2),(16,1),(32,2),(32,1),(32,1),(32,1)]),
    ("Custom-M", 4, 1, [(4,1),(8,2),(16,1),(16,2),(32,1),(32,2),(32,1)]),
    ("Custom-N", 8, 1, [(8,1),(16,2),(16,1),(32,2),(32,1),(48,2)]),
    ("Custom-O", 8, 1, [(8,1),(16,2),(32,1),(32,2),(32,1),(32,2)]),
    ("Custom-P", 8, 2, [(16,1),(16,1),(32,2),(32,1),(64,2)]),
    ("Custom-Q", 8, 2, [(8,1),(16,2),(32,1),(32,2),(64,1)]),
])

# ─── Evaluate all ────────────────────────────────────────────────────────
results = []
for name, stem_ch, stem_s, blocks in configs:
    macc, params, details = calc_arch(stem_ch, stem_s, blocks)
    diff_pct = (macc - TARGET) / TARGET * 100
    int8_kb = params / 1024  # rough: trainable params ≈ int8 bytes
    results.append((name, macc, params, int8_kb, len(blocks), diff_pct, stem_ch, stem_s, blocks, details))

# Sort by closeness to target
results.sort(key=lambda x: abs(x[5]))

print(f"{'Name':<45} {'MACC':>10} {'Params':>8} {'INT8KB':>7} {'#Blk':>4} {'Diff%':>8}")
print("-" * 90)
for name, macc, params, int8_kb, n_blk, diff_pct, *_ in results:
    marker = " <<<" if abs(diff_pct) < 10 else (" **" if abs(diff_pct) < 20 else "")
    print(f"{name:<45} {macc:>10,} {params:>8,} {int8_kb:>6.1f} {n_blk:>4} {diff_pct:>+7.1f}%{marker}")

# ─── Show detailed breakdown for top 10 closest ─────────────────────────
print(f"\n{'='*70}")
print(f"DETAILED BREAKDOWN: Top 10 closest to {TARGET:,} MACC")
print(f"{'='*70}")

for name, macc, params, int8_kb, n_blk, diff_pct, stem_ch, stem_s, blocks, details in results[:10]:
    print(f"\n--- {name} ---")
    print(f"MACC={macc:,} ({diff_pct:+.1f}%)  Params={params:,}  INT8≈{int8_kb:.1f}KB  Blocks={n_blk}")
    for d in details:
        print(d)

# ─── Parametric sweep to nail exactly ~287K ──────────────────────────────
print(f"\n{'='*70}")
print(f"PARAMETRIC SWEEP: Fine-tuning to hit ~287K")
print(f"{'='*70}")

best = []
for stem_ch in range(4, 33, 2):
    for stem_s in [1, 2]:
        for c1 in range(stem_ch, 65, 2):
            for c2 in range(c1, 97, 2):
                for c3 in range(c2, 129, 2):
                    # 3-stride pattern: s2 at transitions
                    blocks = [(c1, 2), (c1, 1), (c2, 2), (c2, 1), (c3, 2), (c3, 1)]
                    macc, params, details = calc_arch(stem_ch, stem_s, blocks)
                    diff = abs(macc - TARGET)
                    if diff < 5000:
                        best.append((diff, macc, params, stem_ch, stem_s, blocks, details))

best.sort()
print(f"\nFound {len(best)} configs within 5K of target")
print(f"{'MACC':>10} {'Params':>8} {'INT8KB':>7} {'Diff':>6} Config")
print("-" * 70)
for diff, macc, params, stem_ch, stem_s, blocks, details in best[:20]:
    ch_str = f"stem={stem_ch} s{stem_s} -> " + " -> ".join(f"{o}(s{s})" for o, s in blocks)
    print(f"{macc:>10,} {params:>8,} {params/1024:>6.1f} {macc-TARGET:>+6,} {ch_str}")

# Show full detail for top 3
for i, (diff, macc, params, stem_ch, stem_s, blocks, details) in enumerate(best[:3]):
    print(f"\n  === Top {i+1}: MACC={macc:,} ===")
    for d in details:
        print(d)
