# SOTA Keyword Spotting Model Comparison Report

## Overview

This report compares 5 state-of-the-art lightweight neural network architectures for on-device keyword spotting (KWS), all evaluated under identical conditions:

- **Task**: 8-class keyword spotting (go/stop/left/right/up/down/unknown/silence)
- **Dataset**: Google Speech Commands v2
- **Features**: MFCC 62x13 (n_mels=16, n_mfcc=13, win=512, hop=256)
- **Target MACC budget**: ~287,000 multiply-accumulate operations
- **Target platform**: STM32F769NI (Cortex-M7) via X-CUBE-AI

---

## 1. CustomDSCNN (Depthwise Separable CNN)

### Design Philosophy
The classic DS-CNN approach for KWS, popularized by Zhang et al. (2017). Processes MFCC features as 2D images using depthwise separable convolutions (depthwise 3x3 + pointwise 1x1). Each DS block reduces spatial dimensions through striding while increasing channel width.

### Architecture
```
Input (1, 62, 13) → Stem Conv2d(1→16, 3x3, s2) → BN → ReLU
→ DS Block (16→16, s1) → DS Block (16→32, s2) → DS Block (32→32, s1)
→ DS Block (32→32, s2) → DS Block (32→32, s1)
→ AdaptiveAvgPool2d → Dropout → FC(32→8)
```

### Key Properties
| Property | Value |
|----------|-------|
| Config | `configs/arch_b.yaml` |
| MACC | 286,240 |
| Parameters | 5,976 |
| Model size (float32) | 23.3 KB |
| Conv type | 2D (Conv2d) |
| Input format | (B, 1, T=62, F=13) |

### Analysis
- Simplest architecture; strong baseline with well-understood behavior
- Stem stride-2 halves spatial resolution immediately, concentrating compute in later blocks
- 2D convolutions capture both time and frequency jointly
- No inverted bottleneck or expansion — direct channel mapping
- Easily quantizable to INT8 with minimal accuracy loss

### References
- Y. Zhang, N. Suda, L. Lai, V. Chandra, "Hello Edge: Keyword Spotting on Microcontrollers," arXiv:1711.07128, 2017. https://arxiv.org/abs/1711.07128
- Our implementation: `kws/models.py` → `CustomDSCNN`

---

## 2. MobileNetStyleKWS (MobileNet with Width/Depth Scaling)

### Design Philosophy
Applies MobileNetV1 architecture (Howard et al., 2017) to KWS with width and depth multipliers for scaling. The full MobileNet has 13 depthwise separable blocks with channels scaling from 32 to 512. Width multiplier uniformly scales all channel counts; depth multiplier controls the number of repeated blocks.

### Architecture (minimum viable: width_mult=0.1, depth_mult=0.2)
```
Input (1, 62, 13) → Stem Conv2d(1→8, 3x3, s1) → BN → ReLU
→ 9 DS Blocks (channels: 8→8→8→8→8→8→8→8→8)
→ AdaptiveAvgPool2d → Dropout → FC(8→8)
```

### Key Properties
| Property | Value |
|----------|-------|
| Config | `configs/mobilenet_min.yaml` |
| MACC | 683,624 (minimum achievable) |
| Parameters | 13,768 |
| Model size (float32) | 53.8 KB |
| Conv type | 2D (Conv2d) |
| Input format | (B, 1, T=62, F=13) |

### Analysis
- **Cannot reach 287K MACC target** — the fixed deep topology (13 blocks min) results in a floor of ~684K MACC
- Width/depth multipliers provide smooth scaling but the base architecture is designed for ImageNet-scale tasks
- At extreme downscaling (0.1x width), channels collapse to 8 throughout, losing representational capacity
- Included for comparison to demonstrate that generic vision architectures are suboptimal for ultra-low MACC KWS

### References
- A. Howard et al., "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications," arXiv:1704.04861, 2017. https://arxiv.org/abs/1704.04861
- Our implementation: `kws/models.py` → `MobileNetStyleKWS`

---

## 3. TENet (Temporal Efficient Network)

### Design Philosophy
TENet (Li et al., Interspeech 2020) treats MFCC frequency bins as input channels and applies purely 1D temporal convolutions. Uses MobileNetV2-style inverted bottleneck blocks: pointwise expand → depthwise temporal → pointwise project (linear, no activation on projection). Residual connections when dimensions match.

### Architecture (287K config)
```
Input (1, 62, 13) → reshape to (13, 62) [freq as channels, time as sequence]
→ Stem Conv1d(13→16, k=3) → BN → ReLU
→ Block 0: IBB(16→24, s2, expand=2, k=9) × 1
→ Block 1: IBB(24→24, s1, expand=2, k=9) × 1
→ Block 2: IBB(24→24, s1, expand=2, k=9) × 1
→ AdaptiveAvgPool1d → Dropout → FC(24→8)
```

**Inverted Bottleneck Block (IBB)**:
```
x → PW Conv1d(C→C*r) → BN → ReLU
  → DW Conv1d(C*r, k=9, groups=C*r) → BN → ReLU
  → PW Conv1d(C*r→C') → BN [linear, no ReLU]
  → + residual (identity or 1x1 projection)
```

### Key Properties
| Property | Value |
|----------|-------|
| Config | `configs/tenet_287k.yaml` |
| MACC | 284,896 |
| Parameters | 8,984 |
| Model size (float32) | 35.1 KB |
| Conv type | 1D (Conv1d) |
| Input format | (B, F=13, T=62) |

### Analysis
- 1D temporal processing avoids redundant frequency-domain convolutions
- Inverted bottleneck (expand ratio 2-4) provides good capacity in a narrow channel pipeline
- Large kernel size (k=9) captures ~150ms temporal context per layer
- Linear projection (no ReLU after final PW) preserves information flow, following MobileNetV2 theory
- Residual connections on all dimension-matched blocks aid gradient flow
- Original paper reports strong accuracy on Speech Commands with compact models (TENet6Narrow: ~13K params)

### References
- S. Li, Y. Gao, "Temporal Efficient Neural Network for Keyword Spotting," Interspeech 2020. arXiv:2010.09960. https://arxiv.org/abs/2010.09960
- Original code (TensorFlow): https://github.com/Interlagos/TENet-kws
- Our implementation: `kws/models.py` → `TENet`, `InvertedBottleneck1D`

---

## 4. LiCoNet (Linearized Convolution Network)

### Design Philosophy
LiCoNet (Yang et al., Meta 2022) reverses the MobileNetV2 block order: applies depthwise spatial convolution first, then pointwise expansion and projection. At inference time, streaming temporal convolutions can be "linearized" (converted to FC layers) for DSP-efficient deployment. Uses ReLU6 activation.

### Architecture (287K config)
```
Input (1, 62, 13) → reshape to (13, 62) [freq as channels, time as sequence]
→ Stem Conv1d(13→20, k=1) → BN → ReLU6
→ LiCoBlock(20, k=7, exp=2, s1) × 1
→ LiCoBlock(20, k=7, exp=2, s2) × 1
→ LiCoBlock(20, k=7, exp=2, s1) × 1
→ LiCoBlock(20, k=7, exp=2, s1) × 1
→ AdaptiveAvgPool1d → Dropout → FC(20→8)
```

**LiCoBlock**:
```
x → DW Conv1d(C, k=7, groups=C) → BN → ReLU6    [spatial first]
  → PW Conv1d(C→C*exp) → BN → ReLU6              [expand]
  → PW Conv1d(C*exp→C) → BN                       [project, linear]
  → + residual (identity if stride=1)
```

### Key Properties
| Property | Value |
|----------|-------|
| Config | `configs/liconet_287k.yaml` |
| MACC | 285,980 |
| Parameters | 8,068 |
| Model size (float32) | 31.5 KB |
| Conv type | 1D (Conv1d) |
| Input format | (B, F=13, T=62) |

### Analysis
- "Spatial-first" block order: DW conv processes the input at full channel width before expansion
- At the same expansion ratio, this produces fewer parameters than expand-first (MobileNetV2/TENet) because the DW conv operates on narrow channels
- ReLU6 (capped at 6) improves quantization robustness by bounding activations
- Linearization potential: streaming 1D convolutions can be converted to matrix multiplications for DSP chips
- Fixed bottleneck width (no channel growth between stages) — simpler but less flexible

### References
- Z. Yang et al., "LiCoNet: Linearized Convolution Network for Efficient Keyword Spotting," Meta, 2022. arXiv:2211.04635. https://arxiv.org/abs/2211.04635
- No public code available (proprietary Meta implementation)
- Our implementation: `kws/models.py` → `LiCoNet`, `LiCoBlock`

---

## 5. BC-ResNet (Broadcasted Residual Network)

### Design Philosophy
BC-ResNet (Kim et al., Qualcomm, Interspeech 2021) achieves 2D representational power at near-1D computational cost. Most operations use cheap 1D temporal convolutions, but a "broadcasted residual" connection expands the 1D output back to 2D for the skip addition. Sub-Spectral Normalization (SSN) applies batch normalization independently to frequency sub-bands, enabling frequency-aware feature learning.

### Architecture (287K config)
```
Input (1, 62, 13) → reshape to (1, F=13, T=62)
→ Stem Conv2d(1→4, 3x3) → BN → ReLU
→ Stage 0: BCResBlock(4→4, s1) × 1
→ Stage 1: BCResBlock(4→8, s1) × 1
→ Stage 2: BCResBlock(8→12, s1) × 1
→ AdaptiveAvgPool2d → Dropout → FC(12→8)
```

**BCResBlock**:
```
x (B,C,F,T) → DW Conv2d(C, 3×1, freq-only) → SubSpectralNorm → ReLU    [2D frequency]
           → mean over F → (B,C,T)                                        [collapse freq]
           → DW Conv1d(C, k=3, stride) → BN → ReLU                        [1D temporal]
           → unsqueeze(2) → broadcast add with 2D branch                   [broadcast residual]
           → PW Conv2d(C→C', 1×1) → BN                                    [channel mixing]
           → + skip connection (with 1x1 projection if C≠C')
           → ReLU
```

### Key Properties
| Property | Value |
|----------|-------|
| Config | `configs/bcresnet_287k.yaml` |
| MACC | 290,008 |
| Parameters | 796 |
| Model size (float32) | 3.1 KB |
| Conv type | 2D + 1D hybrid |
| Input format | (B, 1, F=13, T=62) |

### Analysis
- Extremely parameter-efficient: only 796 params at ~290K MACC
- Hybrid 2D/1D processing: frequency-wise 2D DW conv preserves spectral structure, while temporal processing is 1D
- Sub-Spectral Normalization: divides frequency axis into sub-bands, applying independent BN — captures different normalization statistics per frequency region
- Broadcast residual: the key innovation — 1D temporal features are "broadcast" back to the full 2D frequency-time tensor, achieving 2D representation at 1D computational cost
- Original paper achieves 96.9% accuracy on Speech Commands v2 (12-class) with only 9.2K params
- May need deeper configs for competitive accuracy at this tiny scale

### References
- B. Kim et al., "Broadcasted Residual Learning for Efficient Keyword Spotting," Interspeech 2021. arXiv:2106.04140. https://arxiv.org/abs/2106.04140
- Original code: https://github.com/Qualcomm-AI-research/bcresnet
- Our implementation: `kws/models.py` → `BCResNet`, `BCResBlock`, `SubSpectralNorm`

---

## Architecture Comparison Summary

| Model | Type | MACC | Params | KB (f32) | Block Design | Input |
|-------|------|------|--------|----------|-------------|-------|
| CustomDSCNN | 2D DS-CNN | 286,240 | 5,976 | 23.3 | DW3x3 + PW1x1 | (1,T,F) |
| MobileNet | 2D DS-CNN | 683,624* | 13,768 | 53.8 | DW3x3 + PW1x1 + scaling | (1,T,F) |
| TENet | 1D InvBottleneck | 284,896 | 8,984 | 35.1 | PW→DW(k9)→PW + res | (F,T) |
| LiCoNet | 1D Linearized | 285,980 | 8,068 | 31.5 | DW(k7)→PW→PW + res | (F,T) |
| BC-ResNet | 2D/1D Hybrid | 290,008 | 796 | 3.1 | FreqDW+SSN→TempDW→broadcast | (1,F,T) |

*MobileNet cannot reach 287K MACC target; shown at minimum viable scaling.

## Design Trade-offs

### 2D vs 1D Processing
- **2D (CustomDSCNN, MobileNet)**: Captures time-frequency correlations jointly. Higher per-layer MACC.
- **1D (TENet, LiCoNet)**: Treats frequency as channels, only convolves temporally. More MACC-efficient per parameter.
- **Hybrid (BC-ResNet)**: Best of both — 2D frequency awareness with 1D temporal efficiency via broadcasting.

### Block Order: Expand-First vs Spatial-First
- **Expand-first (TENet/MobileNetV2)**: PW expand → DW → PW project. The DW conv operates on expanded channels, capturing richer features but requiring more MACC.
- **Spatial-first (LiCoNet)**: DW → PW expand → PW project. DW conv on narrow channels is cheaper; expansion happens in the pointwise domain.

### Residual Connection Strategies
- **Identity shortcut** (TENet, LiCoNet): Simple skip when dimensions match.
- **Broadcasted residual** (BC-ResNet): 1D temporal output is broadcast to 2D, adding frequency-awareness at near-zero cost.
- **No residual** (CustomDSCNN): Pure feedforward. Simpler but harder to train deep.

### Quantization Friendliness
- **ReLU6** (LiCoNet): Bounded activations improve INT8 quantization range estimation.
- **Linear projection** (TENet, LiCoNet): No activation on bottleneck output preserves signal fidelity.
- **SubSpectralNorm** (BC-ResNet): Per-sub-band normalization may require careful calibration for INT8.

---

## Additional SOTA Models (Not Implemented)

### MatchboxNet (NVIDIA, 2020)
- 1D time-channel separable convolutions with sub-word encoding
- arXiv:2004.08531. Code: https://github.com/NVIDIA/NeMo

### MicroNets (ARM, 2021)
- Neural Architecture Search optimized for MCU deployment
- arXiv:2010.11267.

### Keyword Transformer (KWT, 2021)
- Vision Transformer adapted for audio spectrograms
- arXiv:2104.00769. Code: https://github.com/ID56/Keyword-MLP
- Too compute-heavy for 287K MACC target

### EfficientWord-Net (2021)
- Few-shot custom keyword spotting
- arXiv:2111.00379.

### TKWS (MobileNetV2 Inverted Bottleneck, 2025)
- MobileNetV2 architecture applied directly to KWS
- arXiv:2509.07051
- Very similar to TENet; our TENet implementation covers this design space

---

## Experimental Setup

All models trained with identical hyperparameters:
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4)
- **Epochs**: 50 (early stopping, patience=5)
- **Batch size**: 1024
- **Seed**: 123
- **AMP**: fp16 training on CUDA
- **Evaluation**: Float32 accuracy + INT8 PTQ accuracy + F1 scores
