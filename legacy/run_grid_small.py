from __future__ import annotations
import os

# ffmpeg DLL
ffmpeg_bin = r"C:\ffmpeg\bin"
os.add_dll_directory(ffmpeg_bin)

import torchcodec
import torchaudio

from typing import Dict, Any, List

import torch

from config import AudioConfig, ModelConfig, TrainConfig, ExperimentConfig
from utils import ensure_dir, save_json
from run_experiments import run_one


def main():
    # ====== 小模型网格：探索更小的 width_mult ======
    feat_list = [(16, 13)]  # 固定特征配置

    # 关键：更小的 width_mult，目标是接近或低于 Benchmark 的 169 KB Flash
    width_list = [0.5, 0.25, 0.15, 0.1]

    # depth: 只测 1.0 和 0.5
    depth_list = [1.0, 0.5]

    num_classes = 8

    train_cfg = TrainConfig(
        batch_size=512,
        epochs=15,          # 小模型需要更多 epoch 来收敛
        lr=1e-3,
        weight_decay=1e-4,
        seed=123,
        num_workers=8,
        train_device="cuda" if torch.cuda.is_available() else "cpu",
        infer_device="cpu",
    )

    out_root = "./outputs_small"
    ensure_dir(out_root)

    rows: List[Dict[str, Any]] = []

    for n_mels, n_mfcc in feat_list:
        for w in width_list:
            for d in depth_list:
                exp_name = f"mfcc_mel{n_mels}_mfcc{n_mfcc}_w{w}_d{d}"
                out_dir = os.path.join(out_root, exp_name)
                ensure_dir(out_dir)

                exp = ExperimentConfig(
                    name=exp_name,
                    audio=AudioConfig(
                        sample_rate=16000,
                        win_length=512,
                        hop_length=256,
                        n_fft=512,
                        n_mels=n_mels,
                        n_mfcc=n_mfcc,
                        fixed_num_samples=16000 + 256
                    ),
                    model=ModelConfig(
                        num_classes=num_classes,
                        width_mult=w,
                        dropout=0.1,
                        depth_mult=d,
                    ),
                    train=train_cfg
                )

                print(f"\n=== Running {exp_name} ===")
                res = run_one(exp, out_dir)

                row = {
                    "exp": exp_name,
                    "n_mels": n_mels,
                    "n_mfcc": n_mfcc,
                    "width": w,
                    "depth": d,
                    "classes": num_classes,
                    "params": res["float"]["params"],
                    "float_acc": res["float"]["test"]["acc"],
                    "float_ms": res["float"]["infer_ms"],
                    "float_kb": res["float"]["weights_bytes"] / 1024.0,
                    "macc": res["float"]["macc"],
                    "int8_acc": res["int8_ptq"]["test"]["acc"],
                    "int8_ms_cpu": res["int8_ptq"]["infer_ms"],
                    "int8_kb": res["int8_ptq"]["weights_bytes"] / 1024.0,
                }

                rows.append(row)
                save_json(os.path.join(out_root, "summary_partial.json"), rows)

                # 实时打印结果
                print(f"  params={row['params']:,}  float_acc={row['float_acc']:.4f}  "
                      f"int8_acc={row['int8_acc']:.4f}  int8_kb={row['int8_kb']:.1f}  "
                      f"MACC={row['macc']:,}  float_ms={row['float_ms']:.2f}")

    # ====== 最终结果 ======
    print("\n\n===== FINAL COMPARISON =====")
    print(f"{'exp':<40} {'params':>8} {'f_acc':>7} {'i8_acc':>7} "
          f"{'f_kb':>8} {'i8_kb':>8} {'MACC':>12} {'f_ms':>7} {'i8_ms':>7}")
    print("=" * 120)
    for r in sorted(rows, key=lambda x: x["int8_kb"]):
        print(f"{r['exp']:<40} {r['params']:>8,} {r['float_acc']:>7.4f} {r['int8_acc']:>7.4f} "
              f"{r['float_kb']:>8.1f} {r['int8_kb']:>8.1f} {r['macc']:>12,} "
              f"{r['float_ms']:>7.2f} {r['int8_ms_cpu']:>7.2f}")

    print(f"\n--- Benchmark: Flash=169.63 KB, RAM=11.52 KB, MACC=287,673, Latency=7 ms, Acc=91% ---")

    save_json(os.path.join(out_root, "summary.json"), rows)
    print(f"\nSaved: {out_root}/summary.json")


if __name__ == "__main__":
    main()
