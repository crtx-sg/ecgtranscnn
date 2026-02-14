#!/usr/bin/env python3
"""Visualization CLI for ECG signals, predictions, and attention maps.

Usage:
    python scripts/visualize.py signal --condition ATRIAL_FIBRILLATION
    python scripts/visualize.py predict --checkpoint models/best_model.pt --condition PVC
    python scripts/visualize.py attention --checkpoint models/best_model.pt --condition NORMAL_SINUS
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ecg_transcovnet import (
    ECGTransCovNet,
    NUM_CLASSES,
    CLASS_NAMES,
    SIGNAL_LENGTH,
    ALL_LEADS,
)
from ecg_transcovnet.simulator import ECGSimulator
from ecg_transcovnet.simulator.conditions import Condition, CONDITION_REGISTRY
from ecg_transcovnet.constants import CONDITION_TO_IDX
from ecg_transcovnet.visualization import (
    plot_ecg_waveform,
    plot_predictions,
    plot_attention_map,
)


def cmd_signal(args):
    """Generate and plot an ECG signal for a condition."""
    condition = Condition[args.condition]
    sim = ECGSimulator(seed=args.seed)
    ecg = sim.generate_ecg(condition, noise_level=args.noise_level)

    leads = ALL_LEADS if args.leads == "all" else [l.strip() for l in args.leads.split(",")]
    signal = np.stack([ecg[lead] for lead in leads], axis=0)

    output = args.output or f"ecg_{args.condition.lower()}.png"
    cfg = CONDITION_REGISTRY[condition]
    title = f"{condition.name} (HR range: {cfg.hr_range[0]:.0f}-{cfg.hr_range[1]:.0f} BPM)"

    plot_ecg_waveform(signal, lead_names=leads, title=title, path=output)
    print(f"Saved ECG plot to {output}")


def _load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    saved_args = ckpt.get("args", {})
    leads = ckpt.get("leads", ALL_LEADS)
    in_channels = len(leads)

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
    return model, leads


def cmd_predict(args):
    """Generate a signal, run prediction, and plot probabilities."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, leads = _load_model(args.checkpoint, device)

    condition = Condition[args.condition]
    sim = ECGSimulator(seed=args.seed)
    ecg = sim.generate_ecg(condition, noise_level=args.noise_level)

    signal = np.stack([ecg[lead] for lead in leads], axis=0)
    for ch in range(signal.shape[0]):
        mu, std = signal[ch].mean(), signal[ch].std()
        if std > 1e-6:
            signal[ch] = (signal[ch] - mu) / std

    x = torch.from_numpy(signal).unsqueeze(0).float().to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

    true_idx = CONDITION_TO_IDX[condition]
    pred_idx = probs.argmax()

    print(f"True: {CLASS_NAMES[true_idx]}")
    print(f"Predicted: {CLASS_NAMES[pred_idx]} (confidence: {probs[pred_idx]:.3f})")

    output = args.output or f"predict_{args.condition.lower()}.png"
    plot_predictions(probs, true_idx=true_idx, path=output)
    print(f"Saved prediction plot to {output}")


def cmd_attention(args):
    """Generate a signal, run inference, and plot attention map."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, leads = _load_model(args.checkpoint, device)

    condition = Condition[args.condition]
    sim = ECGSimulator(seed=args.seed)
    ecg = sim.generate_ecg(condition, noise_level=args.noise_level)

    signal = np.stack([ecg[lead] for lead in leads], axis=0)
    for ch in range(signal.shape[0]):
        mu, std = signal[ch].mean(), signal[ch].std()
        if std > 1e-6:
            signal[ch] = (signal[ch] - mu) / std

    x = torch.from_numpy(signal).unsqueeze(0).float().to(device)
    with torch.no_grad():
        logits, attn = model.forward_with_attention(x)

    if attn is None:
        print("Error: no attention weights captured")
        return

    attention = attn.squeeze(0).cpu().numpy()  # (num_queries, seq_len)
    output = args.output or f"attention_{args.condition.lower()}.png"
    plot_attention_map(attention, title=f"Attention — {condition.name}", path=output)
    print(f"Saved attention map to {output}")


def main():
    parser = argparse.ArgumentParser(
        description="ECG-TransCovNet visualization tools",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # signal subcommand
    p_sig = sub.add_parser("signal", help="Plot raw ECG signal")
    p_sig.add_argument("--condition", type=str, required=True,
                       help="Condition name (e.g. ATRIAL_FIBRILLATION)")
    p_sig.add_argument("--leads", type=str, default="all")
    p_sig.add_argument("--noise-level", type=str, default="clean",
                       choices=["clean", "low", "medium", "high"])
    p_sig.add_argument("--seed", type=int, default=42)
    p_sig.add_argument("--output", type=str, default=None)

    # predict subcommand
    p_pred = sub.add_parser("predict", help="Run prediction and plot probabilities")
    p_pred.add_argument("--checkpoint", type=str, required=True)
    p_pred.add_argument("--condition", type=str, required=True)
    p_pred.add_argument("--noise-level", type=str, default="clean",
                        choices=["clean", "low", "medium", "high"])
    p_pred.add_argument("--seed", type=int, default=42)
    p_pred.add_argument("--output", type=str, default=None)

    # attention subcommand
    p_attn = sub.add_parser("attention", help="Plot decoder cross-attention")
    p_attn.add_argument("--checkpoint", type=str, required=True)
    p_attn.add_argument("--condition", type=str, required=True)
    p_attn.add_argument("--noise-level", type=str, default="clean",
                        choices=["clean", "low", "medium", "high"])
    p_attn.add_argument("--seed", type=int, default=42)
    p_attn.add_argument("--output", type=str, default=None)

    args = parser.parse_args()
    if args.command == "signal":
        cmd_signal(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "attention":
        cmd_attention(args)


if __name__ == "__main__":
    main()
