from __future__ import annotations
import os

# 1) 如果你是 conda 的 ffmpeg：
# ffmpeg_bin = r"C:\Miniconda3\Library\bin"

# 2) 如果你是手动解压的 ffmpeg：
ffmpeg_bin = r"C:\ffmpeg\bin"

os.add_dll_directory(ffmpeg_bin)

# 现在再 import torchaudio / torchcodec
import torchcodec
import torchaudio




import os

from typing import Dict, Any, List, Tuple

import torch

from config import AudioConfig, ModelConfig, TrainConfig, ExperimentConfig
from utils import ensure_dir, save_json
from run_experiments import run_one  # 你已有：训练+float/fp16/int8的评估


def _fmt(x, nd=4):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def print_table(rows: List[Dict[str, Any]], sort_key: str = "float_acc", descending: bool = True) -> None:
    """
    把实验结果以表格形式打印到终端（不依赖额外库）
    """
    if not rows:
        print("No rows.")
        return

    rows = sorted(rows, key=lambda r: r.get(sort_key, -1), reverse=descending)

    headers = [
        "exp",
        "n_mels",
        "n_mfcc",
        "width",
        "classes",
        "float_acc",
        "float_ms",
        "float_kb",
        "int8_acc",
        "int8_ms_cpu",
        "int8_kb",
    ]

    # calculate column widths
    col_w = {h: max(len(h), max(len(_fmt(r.get(h))) for r in rows)) for h in headers}

    def line(ch="-"):
        print(ch * (sum(col_w.values()) + 3 * (len(headers) - 1)))

    # header
    line("=")
    print(" | ".join(h.ljust(col_w[h]) for h in headers))
    line("=")

    # body
    for r in rows:
        print(" | ".join(_fmt(r.get(h)).ljust(col_w[h]) for h in headers))

    line("=")


def main():
    # ====== 你在这里设置“实验网格” ======
    # (n_mels, n_mfcc)
    feat_list = [
        (16, 13),  # 论文 baseline
        (13, 13),  # 只减少 mel bins，但 MFCC 仍 13（可行，因为 n_mels>=n_mfcc）
        (12, 12),
        (10, 10),
        (8, 8),
        (6, 6),
    ]

    width_list = [1.0, 0.75, 0.5]    # 实验3：减少模型尺寸
    num_classes = 8                # 10 keywords + unknown + silence（若你是6+unknown+silence改成8）

    # 训练设置（先少 epochs 验证流程，稳定后再加）
    train_cfg = TrainConfig(
        batch_size=512,
        epochs=6,
        lr=1e-3,
        weight_decay=1e-4,
        seed=123,
        num_workers=8,
        train_device="cuda" if torch.cuda.is_available() else "cpu",
        infer_device="cpu",
    )

    out_root = "./outputs_grid"
    ensure_dir(out_root)

    # ====== 跑所有组合 ======
    rows: List[Dict[str, Any]] = []
    full_dump: Dict[str, Any] = {}

    for n_mels, n_mfcc in feat_list:
        for w in width_list:
            exp_name = f"mfcc_mel{n_mels}_mfcc{n_mfcc}_w{w}"
            out_dir = os.path.join(out_root, exp_name)
            ensure_dir(out_dir)

            exp = ExperimentConfig(
                name=exp_name,
                audio=AudioConfig(
                    sample_rate=16000,
                    win_length=512,
                    hop_length=256,
                    n_fft=512,
                    n_mels=n_mels,      # sweep
                    n_mfcc=n_mfcc,          # paper
                    fixed_num_samples=16000 + 256
                ),
                model=ModelConfig(
                    num_classes=num_classes,
                    width_mult=w,       # sweep
                    dropout=0.1
                ),
                train=train_cfg
            )

            print(f"\n=== Running {exp_name} ===")
            res = run_one(exp, out_dir)  # 你已有的函数：float + int8 + (optional fp16)

            # 抽取对比所需关键指标
            row = {
                "exp": exp_name,
                "n_mels": n_mels,
                "n_mfcc": n_mfcc,
                "width": w,
                "classes": num_classes,

                "float_acc": res["float"]["test"]["acc"],
                "float_ms": res["float"]["infer_ms"],
                "float_kb": res["float"]["weights_bytes"] / 1024.0,

                "int8_acc": res["int8_ptq"]["test"]["acc"],
                "int8_ms_cpu": res["int8_ptq"]["infer_ms"],
                "int8_kb": res["int8_ptq"]["weights_bytes"] / 1024.0,
            }

            rows.append(row)
            full_dump[exp_name] = res

            # 每跑完一个就保存一次（防止中途断电/崩溃丢结果）
            save_json(os.path.join(out_root, "summary_partial.json"), rows)

    # ====== 打印最终对比表（一次运行就能看到） ======
    print("\n\n===== FINAL COMPARISON (sorted by float_acc) =====")
    print_table(rows, sort_key="float_acc", descending=True)

    # 保存最终结果
    save_json(os.path.join(out_root, "summary.json"), rows)
    save_json(os.path.join(out_root, "full_dump.json"), full_dump)
    print(f"\nSaved: {out_root}/summary.json and {out_root}/full_dump.json")


if __name__ == "__main__":
    main()

