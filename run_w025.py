"""只跑 w=0.25 的两个配置，目标是 Flash < 169 KB 同时 acc > 91%"""
from __future__ import annotations
import os

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
    configs = [
        (0.25, 1.0),
        (0.25, 0.5),
        (0.25, 0.25),
    ]

    num_classes = 8

    train_cfg = TrainConfig(
        batch_size=512,
        epochs=15,
        lr=1e-3,
        weight_decay=1e-4,
        seed=123,
        num_workers=8,
        train_device="cuda" if torch.cuda.is_available() else "cpu",
        infer_device="cpu",
    )

    out_root = "./outputs_small"
    ensure_dir(out_root)

    for w, d in configs:
        exp_name = f"mfcc_mel16_mfcc13_w{w}_d{d}"
        out_dir = os.path.join(out_root, exp_name)
        ensure_dir(out_dir)

        exp = ExperimentConfig(
            name=exp_name,
            audio=AudioConfig(
                sample_rate=16000, win_length=512, hop_length=256,
                n_fft=512, n_mels=16, n_mfcc=13,
                fixed_num_samples=16000 + 256
            ),
            model=ModelConfig(
                num_classes=num_classes, width_mult=w,
                dropout=0.1, depth_mult=d,
            ),
            train=train_cfg
        )

        print(f"\n=== Running {exp_name} ===")
        res = run_one(exp, out_dir)

        row = {
            "exp": exp_name,
            "width": w, "depth": d,
            "params": res["float"]["params"],
            "float_acc": res["float"]["test"]["acc"],
            "float_ms": res["float"]["infer_ms"],
            "float_kb": res["float"]["weights_bytes"] / 1024.0,
            "macc": res["float"]["macc"],
            "int8_acc": res["int8_ptq"]["test"]["acc"],
            "int8_ms_cpu": res["int8_ptq"]["infer_ms"],
            "int8_kb": res["int8_ptq"]["weights_bytes"] / 1024.0,
        }

        print(f"\n*** RESULT: {exp_name} ***")
        print(f"  Params: {row['params']:,}")
        print(f"  Float acc: {row['float_acc']:.4f}  |  INT8 acc: {row['int8_acc']:.4f}")
        print(f"  Float KB: {row['float_kb']:.1f}  |  INT8 KB: {row['int8_kb']:.1f}")
        print(f"  MACC: {row['macc']:,}")
        print(f"  Float ms: {row['float_ms']:.2f}  |  INT8 ms: {row['int8_ms_cpu']:.2f}")
        print(f"  Benchmark: Flash=169.63 KB, MACC=287,673, Acc=91%, Latency=7ms")

    print("\nDone!")


if __name__ == "__main__":
    main()
