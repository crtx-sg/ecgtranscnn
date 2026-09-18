#!/usr/bin/env python3
"""Compute per-class and macro AUC for trained ECG-TransCovNet checkpoints.

Usage:
    python scripts/compute_auc.py
    python scripts/compute_auc.py --num-samples 3200 --noise-level mixed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

from ecg_transcovnet import (
    ECGTransCovNet,
    NUM_CLASSES,
    CLASS_NAMES,
    SIGNAL_LENGTH,
    ALL_LEADS,
    FILTER_PRESETS,
)
from ecg_transcovnet.checkpoint import load_model as load_checkpoint_model
from ecg_transcovnet.classes import ClassSpec
from ecg_transcovnet.data import generate_dataset
from ecg_transcovnet.simulator.conditions import Condition


CHECKPOINTS = [
    ("Baseline", "models/best_model.pt"),
    ("Noise-Robust", "models/noise_robust/best_model.pt"),
    ("AV Block Fix", "models/avblock_fix/best_model.pt"),
]


def load_model(ckpt_path: str, device: torch.device) -> ECGTransCovNet | None:
    path = Path(ckpt_path)
    if not path.exists():
        print(f"  Checkpoint not found: {path}")
        return None
    loaded = load_checkpoint_model(path, device)
    if loaded.class_spec != ClassSpec.default():
        print(f"  Skipping {path}: simulator AUC needs a 16-class simulator checkpoint")
        return None
    return loaded.model


def collect_probabilities(
    model: ECGTransCovNet, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and return (all_probs, all_labels)."""
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            logits = model(X.to(device))
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def compute_auc(probs: np.ndarray, labels: np.ndarray) -> dict:
    """Compute per-class OVR AUC and macro AUC."""
    n_classes = probs.shape[1]
    # One-hot encode labels for per-class AUC
    y_onehot = np.eye(n_classes)[labels.astype(int)]

    per_class = {}
    valid_aucs = []
    for i, name in enumerate(CLASS_NAMES):
        # Need at least one positive and one negative sample
        if y_onehot[:, i].sum() == 0 or y_onehot[:, i].sum() == len(labels):
            per_class[name] = None
        else:
            auc = roc_auc_score(y_onehot[:, i], probs[:, i])
            per_class[name] = auc
            valid_aucs.append(auc)

    macro_auc = np.mean(valid_aucs) if valid_aucs else 0.0

    # Also compute weighted macro using sklearn
    try:
        macro_ovr = roc_auc_score(
            y_onehot, probs, multi_class="ovr", average="macro"
        )
    except ValueError:
        macro_ovr = macro_auc

    return {"per_class": per_class, "macro_auc": macro_auc, "macro_ovr": macro_ovr}


def main():
    p = argparse.ArgumentParser(description="Compute AUC for ECG-TransCovNet models")
    p.add_argument("--num-samples", type=int, default=3200)
    p.add_argument("--noise-level", type=str, default="clean",
                    choices=["clean", "low", "medium", "high", "mixed"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=99)
    p.add_argument("--filter-preset", type=str, default="none",
                    choices=list(FILTER_PRESETS.keys()))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Generate balanced evaluation data
    print(f"\nGenerating {args.num_samples} balanced samples (noise={args.noise_level})...")
    distribution = {c: 1.0 / NUM_CLASSES for c in Condition}
    filter_config = FILTER_PRESETS[args.filter_preset]
    X, y = generate_dataset(
        args.num_samples, ALL_LEADS, args.noise_level, distribution, args.seed,
        filter_config=filter_config,
    )
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    print(f"Samples: {len(y)}  Classes: {len(np.unique(y))}")

    # Evaluate each model
    results = {}
    for name, ckpt_path in CHECKPOINTS:
        print(f"\n{'=' * 60}")
        print(f"Model: {name}  ({ckpt_path})")
        print("=" * 60)

        model = load_model(ckpt_path, device)
        if model is None:
            continue

        probs, labels = collect_probabilities(model, loader, device)
        auc = compute_auc(probs, labels)
        results[name] = auc

        print(f"\n  Macro AUC (OVR): {auc['macro_ovr']:.4f}")
        print(f"\n  {'Condition':<28s} {'AUC':>8s}")
        print(f"  {'-' * 38}")
        for cname in CLASS_NAMES:
            val = auc["per_class"][cname]
            val_str = f"{val:.4f}" if val is not None else "N/A"
            print(f"  {cname:<28s} {val_str:>8s}")

    # Summary table
    if results:
        print(f"\n{'=' * 60}")
        print("SUMMARY — Macro AUC (One-vs-Rest)")
        print("=" * 60)
        print(f"  {'Model':<20s} {'Macro AUC':>10s}")
        print(f"  {'-' * 32}")
        for name in results:
            print(f"  {name:<20s} {results[name]['macro_ovr']:>10.4f}")


if __name__ == "__main__":
    main()
