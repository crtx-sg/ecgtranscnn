#!/usr/bin/env python3
"""Train ECG-TransCovNet on synthetic ECG data.

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 150 --batch-size 64 --leads all
    python scripts/train.py --num-train 8000 --num-val 1600 --noise-level high
"""

from __future__ import annotations

import argparse
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ecg_transcovnet import (
    ECGTransCovNet,
    FocalLoss,
    NUM_CLASSES,
    CLASS_NAMES,
    SIGNAL_LENGTH,
    ALL_LEADS,
)
from ecg_transcovnet.data import load_or_generate_data, evaluate_hdf5_test, AugmentedECGDataset
from ecg_transcovnet.training import train_one_epoch, validate, evaluate_detailed
from ecg_transcovnet.visualization import save_training_curves, save_confusion_matrix


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train ECG-TransCovNet on synthetic ECG data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    g = p.add_argument_group("data")
    g.add_argument("--num-train", type=int, default=16000)
    g.add_argument("--num-val", type=int, default=3200)
    g.add_argument(
        "--leads", type=str, default="all",
        help="Comma-separated lead names, or 'all' for all 7 leads",
    )
    g.add_argument("--noise-level", type=str, default="clean",
                   choices=["clean", "low", "medium", "high", "mixed"])
    g.add_argument("--distribution", type=str, default="balanced",
                   choices=["balanced", "mit_bih"],
                   help="Training data distribution (balanced or mit_bih)")
    g.add_argument("--cache-dir", type=str, default="data/training_cache")
    g.add_argument("--test-dir", type=str, default=None,
                   help="Directory with HDF5 test files for post-training evaluation")

    # Model
    g = p.add_argument_group("model")
    g.add_argument("--embed-dim", type=int, default=128)
    g.add_argument("--nhead", type=int, default=8)
    g.add_argument("--num-encoder-layers", type=int, default=3)
    g.add_argument("--num-decoder-layers", type=int, default=3)
    g.add_argument("--dim-feedforward", type=int, default=512)
    g.add_argument("--dropout", type=float, default=0.1)

    # Training
    g = p.add_argument_group("training")
    g.add_argument("--epochs", type=int, default=100)
    g.add_argument("--batch-size", type=int, default=64)
    g.add_argument("--lr", type=float, default=5e-4)
    g.add_argument("--weight-decay", type=float, default=1e-4)
    g.add_argument("--warmup-epochs", type=int, default=5)
    g.add_argument("--patience", type=int, default=20)
    g.add_argument("--seed", type=int, default=42)

    # Output
    g = p.add_argument_group("output")
    g.add_argument("--output-dir", type=str, default="models")
    return p


def main():
    args = build_parser().parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve leads
    if args.leads.lower() == "all":
        leads = ALL_LEADS
    else:
        leads = [l.strip() for l in args.leads.split(",")]
    in_channels = len(leads)
    print(f"Using {in_channels} lead(s): {leads}")

    # ── Data ──────────────────────────────────────────────────────────────
    print("\n=== Data Generation ===")
    t0 = time.time()
    train_X, train_y, val_X, val_y = load_or_generate_data(
        args.cache_dir, args.num_train, args.num_val, leads, args.noise_level, args.seed,
        distribution=args.distribution,
    )
    print(f"Data ready in {time.time() - t0:.1f}s")
    print(f"Train: {train_X.shape}  Val: {val_X.shape}")
    print(f"Train dist: {dict(sorted(Counter(train_y.tolist()).items()))}")
    print(f"Val   dist: {dict(sorted(Counter(val_y.tolist()).items()))}")

    train_ds_raw = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y))
    train_ds = AugmentedECGDataset(train_ds_raw)
    val_ds = TensorDataset(torch.from_numpy(val_X), torch.from_numpy(val_y))
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    print("\n=== Model ===")
    model = ECGTransCovNet(
        num_classes=NUM_CLASSES,
        in_channels=in_channels,
        signal_length=SIGNAL_LENGTH,
        embed_dim=args.embed_dim,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    print(f"CNN output sequence length: {model.seq_len}")

    # Quick sanity check
    with torch.no_grad():
        dummy = torch.randn(2, in_channels, SIGNAL_LENGTH, device=device)
        out = model(dummy)
        assert out.shape == (2, NUM_CLASSES), f"Unexpected shape {out.shape}"
    print("Forward-pass sanity check passed.")

    # ── Training setup ────────────────────────────────────────────────────
    # Compute per-class weights: inverse frequency, normalised to mean=1
    class_counts = np.bincount(train_y, minlength=NUM_CLASSES).astype(np.float32)
    class_counts = np.maximum(class_counts, 1.0)  # avoid div-by-zero
    class_weights = (1.0 / class_counts)
    class_weights = class_weights / class_weights.mean()  # normalise so mean weight = 1
    alpha_tensor = torch.from_numpy(class_weights).to(device)
    print(f"Class weights (min={class_weights.min():.2f}, max={class_weights.max():.2f})")
    loss_fn = FocalLoss(alpha=alpha_tensor, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warmup = args.warmup_epochs
    total = args.epochs

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"\n=== Training ({args.epochs} epochs, patience={args.patience}) ===")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    patience_ctr = 0

    for epoch in range(args.epochs):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device, scaler)
        vl_loss, vl_acc = validate(model, val_loader, loss_fn, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        lr_now = scheduler.get_last_lr()[0]
        dt = time.time() - t0

        marker = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            patience_ctr = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": vl_acc,
                    "val_loss": vl_loss,
                    "args": vars(args),
                    "leads": leads,
                    "class_names": CLASS_NAMES,
                },
                output_dir / "best_model.pt",
            )
            marker = "  *best*"
        else:
            patience_ctr += 1

        print(
            f"Epoch {epoch + 1:3d}/{args.epochs} | "
            f"Train {tr_loss:.4f} / {tr_acc:.4f} | "
            f"Val {vl_loss:.4f} / {vl_acc:.4f} | "
            f"LR {lr_now:.2e} | {dt:.1f}s{marker}"
        )

        if patience_ctr >= args.patience:
            print(f"\nEarly stopping (no improvement for {args.patience} epochs)")
            break

    # ── Final evaluation ──────────────────────────────────────────────────
    print(f"\n=== Final Evaluation (best val acc: {best_val_acc:.4f}) ===")
    ckpt = torch.load(output_dir / "best_model.pt", weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    macro, per_class, cm = evaluate_detailed(model, val_loader, device)

    print(f"\nMacro-averaged Metrics:")
    for k in ("accuracy", "precision", "recall", "specificity", "f1"):
        print(f"  {k:<14s}: {macro[k]:.4f}")

    print(f"\nPer-class Metrics:")
    print(f"  {'Condition':<28s} {'Prec':>6s} {'Rec':>6s} {'Spec':>6s} {'F1':>6s} {'N':>5s}")
    print("  " + "-" * 57)
    for name in CLASS_NAMES:
        m = per_class[name]
        print(
            f"  {name:<28s} {m['precision']:6.3f} {m['recall']:6.3f} "
            f"{m['specificity']:6.3f} {m['f1']:6.3f} {m['support']:5d}"
        )

    # ── Save outputs ──────────────────────────────────────────────────────
    save_training_curves(history, str(output_dir / "training_curves.png"))
    save_confusion_matrix(cm, CLASS_NAMES, str(output_dir / "confusion_matrix.png"))

    # Final model with all metadata
    torch.save(
        {
            "epoch": ckpt["epoch"],
            "model_state_dict": model.state_dict(),
            "val_acc": best_val_acc,
            "metrics": macro,
            "per_class_metrics": per_class,
            "confusion_matrix": cm,
            "class_names": CLASS_NAMES,
            "leads": leads,
            "args": vars(args),
        },
        output_dir / "final_model.pt",
    )

    # ── HDF5 test evaluation ─────────────────────────────────────────────
    if args.test_dir:
        evaluate_hdf5_test(model, args.test_dir, leads, device)

    print(f"\nAll outputs saved to {output_dir}/")
    print(f"  best_model.pt       - checkpoint with best val accuracy")
    print(f"  final_model.pt      - checkpoint with full evaluation metadata")
    print(f"  training_curves.png - loss and accuracy plots")
    print(f"  confusion_matrix.png - per-class confusion matrix")
    print("Done!")


if __name__ == "__main__":
    main()
