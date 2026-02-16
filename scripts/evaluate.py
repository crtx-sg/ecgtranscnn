#!/usr/bin/env python3
"""Evaluate a trained ECG-TransCovNet checkpoint.

Usage:
    python scripts/evaluate.py --checkpoint models/best_model.pt
    python scripts/evaluate.py --checkpoint models/best_model.pt --test-dir data/test_clean
    python scripts/evaluate.py --checkpoint models/best_model.pt --num-samples 500 --noise-level medium
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ecg_transcovnet import (
    ECGTransCovNet,
    FocalLoss,
    NUM_CLASSES,
    CLASS_NAMES,
    SIGNAL_LENGTH,
    ALL_LEADS,
    FILTER_PRESETS,
)
from ecg_transcovnet.data import generate_dataset, evaluate_hdf5_test
from ecg_transcovnet.training import evaluate_detailed
from ecg_transcovnet.visualization import save_confusion_matrix


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate a trained ECG-TransCovNet model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, required=True,
                    help="Path to model checkpoint (.pt)")
    p.add_argument("--test-dir", type=str, default=None,
                    help="Directory with HDF5 test files")
    p.add_argument("--num-samples", type=int, default=1000,
                    help="Number of synthetic samples to evaluate on (if no test-dir)")
    p.add_argument("--noise-level", type=str, default="clean",
                    choices=["clean", "low", "medium", "high", "mixed"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=99)
    p.add_argument("--output-dir", type=str, default=None,
                    help="Directory for confusion matrix output")
    p.add_argument("--filter-preset", type=str, default="none",
                    choices=list(FILTER_PRESETS.keys()),
                    help="Preprocessing filter preset")
    return p


def main():
    args = build_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found at {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    saved_args = ckpt.get("args", {})
    leads = ckpt.get("leads", ALL_LEADS)
    in_channels = len(leads)

    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    print(f"  Val accuracy at save: {ckpt.get('val_acc', '?')}")
    print(f"  Leads: {leads}")

    # Build model
    model = ECGTransCovNet(
        num_classes=NUM_CLASSES,
        in_channels=in_channels,
        signal_length=SIGNAL_LENGTH,
        embed_dim=saved_args.get("embed_dim", 128),
        nhead=saved_args.get("nhead", 8),
        num_encoder_layers=saved_args.get("num_encoder_layers", 3),
        num_decoder_layers=saved_args.get("num_decoder_layers", 3),
        dim_feedforward=saved_args.get("dim_feedforward", 512),
        dropout=saved_args.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    filter_config = FILTER_PRESETS[args.filter_preset]
    print(f"Filter preset: {args.filter_preset}")

    # HDF5 test evaluation
    if args.test_dir:
        evaluate_hdf5_test(model, args.test_dir, leads, device, filter_config=filter_config)
        return

    # Synthetic data evaluation
    print(f"\nGenerating {args.num_samples} evaluation samples (noise={args.noise_level})...")
    from ecg_transcovnet.simulator.conditions import Condition
    balanced = {c: 1.0 / NUM_CLASSES for c in Condition}
    test_X, test_y = generate_dataset(
        args.num_samples, leads, args.noise_level, balanced, args.seed,
        filter_config=filter_config,
    )

    test_ds = TensorDataset(torch.from_numpy(test_X), torch.from_numpy(test_y))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    macro, per_class, cm = evaluate_detailed(model, test_loader, device)

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

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        cm_path = str(out / "eval_confusion_matrix.png")
        save_confusion_matrix(cm, CLASS_NAMES, cm_path)
        print(f"\nConfusion matrix saved to {cm_path}")


if __name__ == "__main__":
    main()
