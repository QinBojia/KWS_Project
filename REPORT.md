# Project Report: Lightweight Keyword Spotting on STM32

## 1. Objective

Design and implement a Keyword Spotting (KWS) system that is **lighter, faster, and more accurate** than the benchmark model described in the reference paper. The system must be deployable on an STM32F769NI microcontroller via X-CUBE-AI.

### Benchmark (Reference Paper)

| Metric       | Benchmark Value |
|-------------|----------------|
| Accuracy    | 91%            |
| Flash (Model Size) | 169.63 KB |
| RAM         | 11.52 KB       |
| MACC        | 287,673        |
| Latency     | 7 ms           |
| MCU         | STM32F769NI    |

---

## 2. Dataset and Features

- **Dataset**: Google Speech Commands v2
- **Classes**: 8 (go, stop, left, right, up, down, unknown, silence)
- **Features**: MFCC with 62 frames x 13 coefficients
  - Sample rate: 16 kHz
  - Frame length: 32 ms (512 samples)
  - Hop length: 16 ms (256 samples)
  - Mel filterbanks: 16
  - MFCC coefficients: 13
- **Input shape**: (1, 1, 62, 13)

These feature extraction parameters match the reference paper exactly for a fair comparison.

---

## 3. Methodology

### 3.1 Model Architecture

We designed a **CustomDSCNN** (Custom Depthwise Separable Convolutional Neural Network) architecture that allows explicit per-block channel and stride configuration. This enables precise targeting of specific MACC budgets.

Each DSConvBlock consists of:
- Depthwise Conv2d (3x3) + BatchNorm + ReLU
- Pointwise Conv2d (1x1) + BatchNorm + ReLU

### 3.2 Architecture Search

We developed an analytical architecture search tool (`calc_arch.py`) that computes MACC and parameter counts without training. This allowed us to:
1. Enumerate hundreds of candidate architectures parametrically
2. Filter candidates matching the benchmark's ~287K MACC budget
3. Select the top 3 candidates for full training and evaluation

### 3.3 Three Candidate Architectures

**arch_a** (8 blocks, thin start, no stem stride):
```
Stem: 1->4 s1 | DS: 4->8->8->16->16->32->32->32->32
```

**arch_b** (5 blocks, stem stride-2, compact):
```
Stem: 1->16 s2 | DS: 16->16->32->32->32->32
```

**arch_c** (5 blocks, gradual widening):
```
Stem: 1->8 s1 | DS: 8->16->16->24->24->32
```

### 3.4 Training Setup

- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- Epochs: 50 (with early stopping, patience=5)
- Batch size: 1024
- Mixed precision training (AMP) on CUDA
- Seed: 123 for reproducibility

### 3.5 Quantization

- **Method**: INT8 Post-Training Quantization (PTQ)
- **Framework**: PyTorch FX Graph Mode with onednn backend
- **Calibration**: 30 batches from training set
- **Pipeline**: Operator fusion (Conv+BN+ReLU) -> Observer insertion -> Calibration -> INT8 conversion

---

## 4. Results

### 4.1 Custom Architecture Results (targeting ~287K MACC)

| Config   | MACC    | Params | Float Acc | INT8 Acc | INT8 Size | Latency (CPU) |
|----------|---------|--------|-----------|----------|-----------|---------------|
| **arch_b** | **286,240** | **5,976** | **94.98%** | **94.94%** | **6.3 KB** | **1.33 ms** |
| arch_c   | 288,712 | 3,656  | 94.01%    | 94.00%   | 3.9 KB    | 1.62 ms       |
| arch_a   | 287,552 | 6,352  | 93.18%    | 92.80%   | 6.8 KB    | 5.98 ms       |

### 4.2 MobileNet Width/Depth Sweep Results

| Config (w, d) | Float Acc | INT8 Acc | INT8 Size | MACC       | Latency (CPU) |
|---------------|-----------|----------|-----------|------------|---------------|
| w=1.0, d=1.0  | 95.90%    | 95.89%   | 1,855 KB  | ~44M       | 8.08 ms       |
| w=1.0, d=0.75 | 95.80%    | 95.80%   | 1,062 KB  | ~33M       | 4.72 ms       |
| w=0.5, d=1.0  | 95.37%    | 95.23%   | 481 KB    | 11,228,800 | 7.39 ms       |
| w=0.5, d=0.5  | 95.40%    | 95.48%   | 276 KB    | 7,972,480  | 3.16 ms       |
| w=0.25, d=1.0 | 92.81%    | 92.88%   | 129 KB    | 3,029,184  | 4.54 ms       |
| w=0.25, d=0.5 | 93.94%    | 93.97%   | 75 KB     | 2,187,456  | 3.19 ms       |
| w=0.25, d=0.25| 93.31%    | 93.09%   | 57 KB     | 1,906,880  | 2.87 ms       |

### 4.3 Comparison with Benchmark

**Best model: arch_b (CustomDSCNN)**

| Metric         | Benchmark | arch_b      | Improvement     |
|----------------|-----------|-------------|-----------------|
| Accuracy       | 91%       | **94.94%**  | **+3.94pp**     |
| Flash (Size)   | 169.63 KB | **6.3 KB**  | **26.9x smaller** |
| MACC           | 287,673   | 286,240     | ~matched        |
| Params         | N/A       | 5,976       | -               |
| Latency (CPU)  | 7 ms      | 1.33 ms     | **5.3x faster** |

arch_b **surpasses the benchmark on all measured dimensions simultaneously**:
- **Accuracy**: +3.94 percentage points (94.94% vs 91%)
- **Model size**: 26.9x smaller (6.3 KB vs 169.63 KB)
- **MACC**: Near-identical (286,240 vs 287,673, only 0.5% less)
- **Latency**: 5.3x faster on CPU (1.33 ms vs 7 ms)

---

## 5. Key Technical Findings

1. **Stem stride-2 is highly effective**: arch_b uses stride-2 in the stem convolution, which halves the spatial resolution immediately. This dramatically reduces MACC in all subsequent layers while maintaining accuracy.

2. **INT8 quantization has minimal accuracy loss**: FX Graph Mode PTQ with operator fusion preserves accuracy within 0.04-0.38 percentage points across all architectures.

3. **Fewer, wider blocks outperform many thin blocks**: arch_b (5 blocks, 16-32 channels) outperforms arch_a (8 blocks, 4-32 channels) despite similar MACC budgets, suggesting that wider feature maps are more important than network depth for KWS.

4. **Analytical architecture search is efficient**: Computing MACC/params analytically before training allows filtering hundreds of candidates in seconds, avoiding expensive training runs on unsuitable architectures.

---

## 6. Deployment

- **ONNX export**: arch_b exported as `model_opset13.onnx` (27.3 KB, opset 13)
- **Target MCU**: STM32F769NI (Cortex-M7, 2MB Flash, 512KB RAM)
- **Conversion tool**: X-CUBE-AI for STM32
- **STM32 project**: Complete deployment project in `deploy/KWS_Deploy/`

---

## 7. Project Structure

```
KWS_Project/
  configs/             - YAML architecture and training configs
  kws/                 - Core library package
    models.py          - MobileNetStyleKWS + CustomDSCNN definitions
    config.py          - AudioConfig, TrainConfig, ArchConfig + YAML loader
    data.py            - SpeechCommands dataset with MFCC caching
    training.py        - Training loop, evaluation, inference benchmark
    quantization.py    - INT8 PTQ via FX Graph Mode (onednn backend)
    export.py          - Unified ONNX export for X-CUBE-AI
    utils.py           - MACC counting, parameter counting, utilities
  scripts/             - Entry-point scripts
    train.py           - Unified training (replaces run_custom.py, etc.)
    evaluate.py        - F1-score evaluation with per-class metrics
    export_onnx.py     - ONNX export script
    search_arch.py     - Analytical architecture search
    sweep.py           - Multi-config sweep runner
  experiments/         - Experiment results (arch_a, arch_b, arch_c)
  legacy/              - Old scripts preserved for reference
  deploy/              - STM32 deployment project (X-CUBE-AI)
```
