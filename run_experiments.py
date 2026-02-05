from __future__ import annotations
import os
from dataclasses import asdict
from typing import List, Dict, Any

# 1) 如果你是 conda 的 ffmpeg：
# ffmpeg_bin = r"C:\Miniconda3\Library\bin"

# 2) 如果你是手动解压的 ffmpeg：
ffmpeg_bin = r"C:\ffmpeg\bin"
os.add_dll_directory(ffmpeg_bin)

import torch

from config import AudioConfig, ModelConfig, TrainConfig, ExperimentConfig
from data import make_loaders
from model import MobileNetStyleKWS
from train_eval import train_one_experiment, evaluate, benchmark_inference_ms
from quantization import ptq_int8_static, fp16_cast
from utils import set_seed, count_params, model_size_bytes, ensure_dir, save_json


from dataclasses import asdict

def run_one(exp: ExperimentConfig, out_dir: str) -> dict:
    ensure_dir(out_dir)
    set_seed(exp.train.seed)

    train_device = exp.train.train_device
    infer_device = exp.train.infer_device  # 你想要 "cpu"

    # ====== loaders ======
    train_ld, val_ld, test_ld = make_loaders(
        audio_cfg=exp.audio,
        batch_size=exp.train.batch_size,
        num_workers=exp.train.num_workers,
    )

    # ====== build model ======
    model = MobileNetStyleKWS(
        num_classes=exp.model.num_classes,
        width_mult=exp.model.width_mult,
        dropout=exp.model.dropout,
    )

    # ====== TRAIN on train_device (CUDA) ======
    model = model.to(train_device)
    train_stats = train_one_experiment(model, train_ld, val_ld, exp.train, device=train_device)

    # ====== MOVE to infer_device (CPU) for evaluation/benchmark ======
    # 关键：推理前把模型搬到 CPU
    model = model.to(infer_device).eval()
    assert next(model.parameters()).device.type == torch.device(infer_device).type, "Model not on infer_device!"

    # float eval + benchmark (CPU)
    float_test = evaluate(model, test_ld, device=infer_device)
    example_x, _ = next(iter(val_ld))
    example_x = example_x[:1].to(infer_device)
    float_ms = benchmark_inference_ms(model, example_x, device=infer_device, iters=200)

    result = {
        "experiment": asdict(exp),
        "train": train_stats,
        "float": {
            "test": float_test,
            "params": count_params(model),
            "weights_bytes": model_size_bytes(model),
            "infer_ms": float_ms,
            "infer_device": infer_device,
        }
    }

    # ====== FP16 on CPU?（可选：一般没意义，且可能不支持） ======
    # 你如果只关心 CPU 对比，可以直接跳过 FP16
    # result["fp16_cpu"] = {"skipped": "FP16 on CPU usually not helpful"}

    # ====== INT8 PTQ on CPU ======
    # PTQ 必须在 CPU 上做（你也想 CPU 推理）
    int8_model = MobileNetStyleKWS(
        num_classes=exp.model.num_classes,
        width_mult=exp.model.width_mult,
        dropout=exp.model.dropout,
    )
    # 注意：model 现在在 CPU(infer_device)，可以直接拿 state_dict
    int8_model.load_state_dict(model.state_dict())

    int8_q = ptq_int8_static(
        int8_model,
        calibration_loader=train_ld,   # 校准用 train loader（CPU 跑）
        device="cpu",
        calibration_batches=30
    )

    int8_test = evaluate(int8_q, test_ld, device="cpu")
    int8_ms = benchmark_inference_ms(int8_q, example_x.to("cpu"), device="cpu", iters=200)

    result["int8_ptq"] = {
        "test": int8_test,
        "params": count_params(int8_q),
        "weights_bytes": model_size_bytes(int8_q),
        "infer_ms": int8_ms,
        "infer_device": "cpu",
    }

    save_json(f"{out_dir}/result.json", result)
    return result




def main():
    train_cfg = TrainConfig(
        batch_size=512,
        epochs=8,
        lr=1e-3,
        weight_decay=1e-4,
        seed=123,
        num_workers=8,
        train_device="cuda" if torch.cuda.is_available() else "cpu",
        infer_device="cpu",
    )

    # Paper baseline: n_mels=16, n_mfcc=13, win=512, hop=256, fixed_len=16256
    # Your experiment 2: reduce mel channels (must be >= 13)
    mel_list = [16, 14, 13]

    # Your experiment 3: reduce model size
    width_list = [1.0, 0.75, 0.5]

    # If you use 12 classes (10 keywords + unknown + silence)
    num_classes = 8

    exps = []
    for n_mels in mel_list:
        for w in width_list:
            name = f"mfcc_mel{n_mels}_w{w}"
            exps.append(
                ExperimentConfig(
                    name=name,
                    audio=AudioConfig(
                        n_mels=n_mels,
                        n_mfcc=13,
                        sample_rate=16000,
                        win_length=512,
                        hop_length=256,
                        n_fft=512,
                        fixed_num_samples=16000 + 256
                    ),
                    model=ModelConfig(num_classes=num_classes, width_mult=w, dropout=0.1),
                    train=train_cfg
                )
            )

    out_root = "./outputs"
    ensure_dir(out_root)

    summary = {}
    for exp in exps:
        out_dir = f"{out_root}/{exp.name}"
        print(f"\n=== Running: {exp.name} | train={exp.train.train_device} | infer={exp.train.infer_device} ===")
        res = run_one(exp, out_dir)
        summary[exp.name] = {
            "float_test_acc": res["float"]["test"]["acc"],
            "float_infer_ms": res["float"]["infer_ms"],
            "float_weights_kb": res["float"]["weights_bytes"] / 1024.0,
            "int8_test_acc": res["int8_ptq"]["test"]["acc"],
            "int8_infer_ms_cpu": res["int8_ptq"]["infer_ms"],
            "int8_weights_kb": res["int8_ptq"]["weights_bytes"] / 1024.0,
        }

        if "fp16" in res:
            summary[exp.name].update({
                "fp16_test_acc": res["fp16"]["test"]["acc"],
                "fp16_infer_ms": res["fp16"]["infer_ms"],
                "fp16_weights_kb": res["fp16"]["weights_bytes"] / 1024.0,
            })

    save_json(f"{out_root}/summary.json", summary)
    print("\nSaved summary to outputs/summary.json")



if __name__ == "__main__":
    main()
