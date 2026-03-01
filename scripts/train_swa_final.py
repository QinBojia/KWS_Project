"""
Final model training with SWA (Stochastic Weight Averaging).

Strategy:
  1. Train from scratch for `swa_start` epochs with CosineAnnealing + all optimizations
  2. Switch to SWA with constant LR for `swa_epochs` more epochs
  3. Update BatchNorm statistics with the averaged model
  4. Evaluate float + INT8, export ONNX

Usage:
    python -u scripts/train_swa_final.py > experiments/final_model/swa.log 2>&1
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from kws.config import AudioConfig, TrainConfig, ArchConfig, build_model
from kws.data import make_loaders
from kws.training import evaluate, _spec_augment, _mixup
from kws.quantization import ptq_int8_static
from kws.export import export_onnx
from kws.utils import count_macc, count_params, model_size_bytes, set_seed, save_json, ensure_dir


# ── Winner architecture from grid search 2c ──
WINNER_ARCH = ArchConfig(
    name="tenet_grid_winner",
    model_type="tenet",
    n_channels=[14, 14, 14, 14, 14],   # stem=14, block=14, 4 blocks
    n_strides=[2, 2, 1, 1],
    n_ratios=[4, 4, 4, 4],
    n_layers=[1, 2, 1, 1],
    kernel_size=9,
    in_channels=13,
    num_classes=12,
    dropout=0.1,
)


def train_phase1(model, train_ld, val_ld, device, epochs, lr, warmup, seed):
    """Phase 1: CosineAnnealing training with all optimizations."""
    print(f"\n=== Phase 1: CosineAnnealing training, {epochs} epochs ===")
    set_seed(seed)
    model.train()

    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup_sched = LambdaLR(opt, lr_lambda=lambda ep: min(1.0, (ep + 1) / warmup))
    cosine_sched = CosineAnnealingLR(opt, T_max=epochs - warmup, eta_min=lr / 100)
    scheduler = SequentialLR(opt, [warmup_sched, cosine_sched], milestones=[warmup])

    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler("cuda")

    best_val_acc = 0.0
    best_state = None
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_samples = 0

        for x, y in tqdm(train_ld, desc=f"epoch {epoch}/{epochs}", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # SpecAugment
            x = _spec_augment(x, time_masks=2, time_width=5, freq_masks=1, freq_width=2)

            # Mixup
            mixed_x, y_a, y_b, lam = _mixup(x, y, alpha=0.2)

            opt.zero_grad(set_to_none=True)
            with autocast("cuda"):
                logits = model(mixed_x)
                loss = lam * ce(logits, y_a) + (1 - lam) * ce(logits, y_b)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running_loss += loss.item() * x.size(0)
            n_samples += x.size(0)

        scheduler.step()

        # Validate
        val_m = evaluate(model, val_ld, device=device)
        lr_now = opt.param_groups[0]["lr"]

        if val_m["acc"] > best_val_acc:
            best_val_acc = val_m["acc"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        if epoch % 10 == 0 or epoch <= 5:
            print(f"  ep {epoch:>4}: loss={running_loss/n_samples:.4f}  "
                  f"val_acc={val_m['acc']:.4f}  best={best_val_acc:.4f}@{best_epoch}  "
                  f"lr={lr_now:.6f}")

    # Restore best weights before SWA
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Phase 1 done: best_val_acc={best_val_acc:.4f} @ epoch {best_epoch}")
    return best_val_acc, best_epoch


def train_phase2_swa(model, train_ld, val_ld, device, swa_epochs, swa_lr):
    """Phase 2: SWA with constant LR."""
    print(f"\n=== Phase 2: SWA, {swa_epochs} epochs, lr={swa_lr} ===")

    swa_model = AveragedModel(model)
    opt = AdamW(model.parameters(), lr=swa_lr, weight_decay=1e-4)
    swa_scheduler = SWALR(opt, swa_lr=swa_lr)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler("cuda")

    for epoch in range(1, swa_epochs + 1):
        model.train()
        running_loss = 0.0
        n_samples = 0

        for x, y in tqdm(train_ld, desc=f"SWA {epoch}/{swa_epochs}", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            x = _spec_augment(x, time_masks=2, time_width=5, freq_masks=1, freq_width=2)
            mixed_x, y_a, y_b, lam = _mixup(x, y, alpha=0.2)

            opt.zero_grad(set_to_none=True)
            with autocast("cuda"):
                logits = model(mixed_x)
                loss = lam * ce(logits, y_a) + (1 - lam) * ce(logits, y_b)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running_loss += loss.item() * x.size(0)
            n_samples += x.size(0)

        swa_scheduler.step()
        swa_model.update_parameters(model)

        if epoch % 10 == 0 or epoch == 1:
            val_m = evaluate(model, val_ld, device=device)
            print(f"  SWA ep {epoch:>3}: loss={running_loss/n_samples:.4f}  "
                  f"val_acc(base)={val_m['acc']:.4f}")

    # Update BN stats for SWA model
    print("Updating BN statistics for SWA model...")
    # SWA model wraps the original — need to run data through it
    torch.optim.swa_utils.update_bn(train_ld, swa_model, device=device)

    # Evaluate SWA model
    swa_val = evaluate(swa_model, val_ld, device=device)
    print(f"SWA model val_acc={swa_val['acc']:.4f}")

    return swa_model, swa_val


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--swa-start", type=int, default=273,
                   help="Epochs for phase 1 (CosineAnnealing)")
    p.add_argument("--swa-epochs", type=int, default=50,
                   help="Epochs for SWA phase")
    p.add_argument("--swa-lr", type=float, default=1e-4,
                   help="SWA constant learning rate")
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    out_dir = Path("experiments/final_model")
    ensure_dir(out_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    # Build model
    arch = WINNER_ARCH
    model = build_model(arch).to(device)
    audio_cfg = AudioConfig()
    input_shape = audio_cfg.input_shape
    macc = count_macc(model, input_shape)
    params = count_params(model)
    print(f"Architecture: {arch.name}")
    print(f"  MACC: {macc:,}  Params: {params:,}  Size: {model_size_bytes(model)/1024:.1f} KB")

    # Load data
    print("\nLoading data to GPU...")
    train_ld, val_ld, test_ld = make_loaders(
        audio_cfg=audio_cfg, batch_size=1024, num_workers=0,
        train_device=device, preload=True, num_classes=12,
    )
    print("Data loaded.\n")

    # Phase 1: CosineAnnealing
    t0 = time.perf_counter()
    best_val, best_ep = train_phase1(
        model, train_ld, val_ld, device,
        epochs=args.swa_start, lr=args.lr, warmup=5, seed=args.seed)

    # Phase 2: SWA
    swa_model, swa_val = train_phase2_swa(
        model, train_ld, val_ld, device,
        swa_epochs=args.swa_epochs, swa_lr=args.swa_lr)
    total_time = time.perf_counter() - t0

    # ── Evaluation ──
    print(f"\n{'='*60}")
    print("FINAL EVALUATION")
    print(f"{'='*60}")

    # Extract the inner model from AveragedModel for saving/export
    swa_inner = swa_model.module

    # Float eval
    test_float = evaluate(swa_inner, test_ld, device=device)
    print(f"Float:  test_acc={test_float['acc']:.4f}  test_loss={test_float['loss']:.4f}")

    # INT8 PTQ
    print("Running INT8 PTQ...")
    model_int8 = ptq_int8_static(swa_inner, val_ld)
    test_int8 = evaluate(model_int8, test_ld, device="cpu")
    int8_size = model_size_bytes(model_int8)
    print(f"INT8:   test_acc={test_int8['acc']:.4f}  test_loss={test_int8['loss']:.4f}  "
          f"size={int8_size/1024:.1f} KB")

    # Save checkpoints
    torch.save(swa_inner.state_dict(), out_dir / "model_float.pth")
    print(f"\nSaved: {out_dir / 'model_float.pth'}")

    # ONNX export
    onnx_path = out_dir / "model_opset13.onnx"
    export_onnx(swa_inner, input_shape, str(onnx_path), opset=13)
    print(f"Saved: {onnx_path}")

    # Summary
    summary = {
        "arch": arch.name,
        "stem_ch": arch.n_channels[0],
        "block_ch": arch.n_channels[1],
        "n_blocks": len(arch.n_strides),
        "strides": arch.n_strides,
        "n_layers": arch.n_layers,
        "ratio": arch.n_ratios[0],
        "kernel": arch.kernel_size,
        "macc": macc,
        "params": params,
        "phase1_best_epoch": best_ep,
        "phase1_best_val_acc": best_val,
        "swa_epochs": args.swa_epochs,
        "swa_val_acc": swa_val["acc"],
        "test_acc_float": test_float["acc"],
        "test_acc_int8": test_int8["acc"],
        "int8_size_kb": round(int8_size / 1024, 1),
        "total_time_s": round(total_time, 1),
    }
    save_json(out_dir / "summary.json", summary)

    print(f"\n{'='*60}")
    print(f"FINAL MODEL SUMMARY")
    print(f"{'='*60}")
    print(f"  Architecture: {arch.name}")
    print(f"  MACC: {macc:,} / {287_673:,} budget")
    print(f"  Float test acc: {test_float['acc']*100:.2f}%")
    print(f"  INT8  test acc: {test_int8['acc']*100:.2f}%")
    print(f"  INT8  size:     {int8_size/1024:.1f} KB")
    print(f"  Total time:     {total_time/60:.1f} min")
    print(f"  Files: {out_dir}/")


if __name__ == "__main__":
    main()
