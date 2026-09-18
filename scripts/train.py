#!/usr/bin/env python3
"""Train ECG-TransCovNet on synthetic ECG data or on an ecg_sigma ``ecgpkg`` package.

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 150 --batch-size 64 --leads all
    python scripts/train.py --num-train 8000 --num-val 1600 --noise-level high

    # Real data (class head from package.json). Stops cleanly after the time
    # budget; rerun the same command with --resume to continue.
    python scripts/train.py --data-source package \\
        --package ../ecg_sigma/packages/ecg_pkg_v1 --output-dir models/real_v1 \\
        --time-budget-min 9 --resume
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from ecg_transcovnet import (
    ECGTransCovNet,
    FocalLoss,
    NUM_CLASSES,
    CLASS_NAMES,
    SIGNAL_LENGTH,
    ALL_LEADS,
    FILTER_PRESETS,
)
from ecg_transcovnet.checkpoint import build_model, load_checkpoint, warm_start
from ecg_transcovnet.data import load_or_generate_data, evaluate_hdf5_test, AugmentedECGDataset
from ecg_transcovnet.evaluation import (excluded_from_primary, macro_f1,
                                        save_confusion_png, write_reports)
from ecg_transcovnet.package import (PackageDataset, balanced_sample_weights,
                                     cv_datasets, load_package)
from ecg_transcovnet.package_eval import EvalModel, evaluate_split
from ecg_transcovnet.training import train_one_epoch, validate, evaluate_detailed
from ecg_transcovnet.visualization import save_training_curves, save_confusion_matrix


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train ECG-TransCovNet on synthetic ECG data or an ecgpkg package",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    g = p.add_argument_group("data")
    g.add_argument("--data-source", type=str, default="sim", choices=["sim", "package"],
                   help="Simulator data or an ecg_sigma ecgpkg training package")
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
    g.add_argument("--filter-preset", type=str, default=None,
                   choices=list(FILTER_PRESETS.keys()),
                   help="Preprocessing filter preset (default: none for sim, default for package)")

    # Package data
    g = p.add_argument_group("package data (--data-source package)")
    g.add_argument("--package", type=str, default=None, help="Path to an ecgpkg package")
    g.add_argument("--crop-len", type=int, default=2000,
                   help="Training/validation crop length in samples")
    g.add_argument("--noise-aug-prob", type=float, default=0.5,
                   help="Probability of simulator-artefact injection per training item")
    g.add_argument("--fabricated-leads", type=str, default="keep", choices=["keep", "zero"],
                   help="Zero leads that are not measured (real_lead_mask bit 0)")
    g.add_argument("--lead-fab-aug-prob", type=float, default=0.0,
                   help="Probability of rebuilding non-ECG2 leads from ECG2 (ecg_sigma rules)")
    g.add_argument("--sampler", type=str, default="shuffle", choices=["shuffle", "balanced"],
                   help="Uniform shuffling or class-and-subject balanced sampling")
    g.add_argument("--balance-beta", type=float, default=0.5,
                   help="Subject damping exponent for --sampler balanced")
    g.add_argument("--class-weights", type=str, default="auto", choices=["auto", "inverse", "none"],
                   help="Focal-loss class weights from train counts (auto: inverse unless balanced sampler)")
    g.add_argument("--select-metric", type=str, default=None, choices=["macro_f1", "accuracy"],
                   help="Checkpoint selection metric (default: accuracy for sim, macro_f1 for package)")
    g.add_argument("--init-checkpoint", type=str, default=None,
                   help="Warm-start weights from this checkpoint")
    g.add_argument("--init-queries", type=str, default="by_name", choices=["by_name", "reinit"],
                   help="Object queries on warm start: copy shared classes or re-initialise")
    g.add_argument("--workers", type=int, default=12, help="DataLoader workers")
    g.add_argument("--amp", type=str, default="off", choices=["on", "off"],
                   help="Mixed precision for package training (FP32 measured ~2x faster for this model)")
    g.add_argument("--eval-batch-size", type=int, default=128)
    g.add_argument("--cv-fold", type=int, default=None, metavar="N",
                   help="Train on cross-validation fold N from splits.json (fit on the other "
                        "folds, select on N). Uses train+val subjects; test is never touched.")
    g.add_argument("--resume", action="store_true", help="Continue from <output-dir>/last.pt")
    g.add_argument("--time-budget-min", type=float, default=None,
                   help="Stop between epochs before this many minutes; rerun with --resume")

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
    if args.filter_preset is None:
        args.filter_preset = "default" if args.data_source == "package" else "none"
    if args.select_metric is None:
        args.select_metric = "macro_f1" if args.data_source == "package" else "accuracy"
    if args.data_source == "package":
        train_package(args)
    else:
        train_sim(args)


def train_sim(args):
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
    filter_config = FILTER_PRESETS[args.filter_preset]
    print(f"Filter preset: {args.filter_preset}")
    t0 = time.time()
    train_X, train_y, val_X, val_y = load_or_generate_data(
        args.cache_dir, args.num_train, args.num_val, leads, args.noise_level, args.seed,
        distribution=args.distribution,
        filter_config=filter_config,
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
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda(args))
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
        evaluate_hdf5_test(model, args.test_dir, leads, device, filter_config=filter_config)

    print(f"\nAll outputs saved to {output_dir}/")
    print(f"  best_model.pt       - checkpoint with best val accuracy")
    print(f"  final_model.pt      - checkpoint with full evaluation metadata")
    print(f"  training_curves.png - loss and accuracy plots")
    print(f"  confusion_matrix.png - per-class confusion matrix")
    print("Done!")


def _lr_lambda(args):
    warmup = args.warmup_epochs
    total = args.epochs

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


# ═══════════════════════════════════════════════════════════════════════════
# Package training
# ═══════════════════════════════════════════════════════════════════════════

def _print_split(name: str, ds: PackageDataset, class_names: list[str]) -> None:
    counts = np.bincount(ds.labels, minlength=len(class_names))
    subjects: dict[int, set] = defaultdict(set)
    for label, subject in zip(ds.labels.tolist(), ds.subjects):
        subjects[label].add(subject)
    cells = ", ".join(f"{n} {counts[i]}({len(subjects[i])})" for i, n in enumerate(class_names))
    print(f"  {name:<5s} {len(ds):6d} events — {cells}")


def train_package(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)  # leave CPU cores to the DataLoader workers
    if not args.package:
        raise SystemExit("--data-source package requires --package PATH")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pkg = load_package(args.package)
    class_names = list(pkg.spec.names)
    leads = pkg.leads if args.leads.lower() == "all" else [l.strip() for l in args.leads.split(",")]
    filter_config = FILTER_PRESETS[args.filter_preset]
    print(f"Package {pkg.version} ({pkg.root}) — {len(class_names)} classes: {class_names}")
    print(f"Leads {leads} · filter preset {args.filter_preset} · crop {args.crop_len} · "
          f"noise aug {args.noise_aug_prob} · lead-fab aug {args.lead_fab_aug_prob} · "
          f"fabricated leads {args.fabricated_leads} · sampler {args.sampler}")

    # Caches are built here, before CUDA is initialised.
    common = dict(leads=leads, filter_config=filter_config, fabricated_leads=args.fabricated_leads,
                  cache_dir=args.cache_dir, seed=args.seed)
    if args.cv_fold is not None:
        train_ds, val_ds = cv_datasets(
            pkg, args.cv_fold, crop_len=args.crop_len, train=True,
            noise_aug_prob=args.noise_aug_prob,
            lead_fab_aug_prob=args.lead_fab_aug_prob, **common,
        )
        cv = pkg.cv
        print(f"Cross-validation: fold {args.cv_fold} of {cv['k']} held out "
              f"({cv.get('scope', 'train+val')}); test split untouched")
    else:
        train_ds = PackageDataset(pkg, "train", crop_len=args.crop_len, train=True,
                                  noise_aug_prob=args.noise_aug_prob,
                                  lead_fab_aug_prob=args.lead_fab_aug_prob, **common)
        val_ds = PackageDataset(pkg, "val", crop_len=args.crop_len, train=False, **common)
    print("Events (subjects) per class:")
    _print_split("fit" if args.cv_fold is not None else "train", train_ds, class_names)
    _print_split("select" if args.cv_fold is not None else "val", val_ds, class_names)
    if args.cv_fold is not None:
        overlap = set(train_ds.subjects) & set(val_ds.subjects)
        assert not overlap, f"fold {args.cv_fold} leaks {len(overlap)} subjects into selection"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    model = build_model(pkg.spec, len(leads), vars(args)).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    resuming = args.resume and (output_dir / "last.pt").exists()
    if args.init_checkpoint and not resuming:
        src = load_checkpoint(args.init_checkpoint, "cpu")
        if list(src.get("leads") or ALL_LEADS) != leads:
            raise SystemExit(f"--init-checkpoint leads {src.get('leads')} differ from {leads}")
        info = warm_start(model, src, pkg.spec, args.init_queries)
        print(f"Warm start from {args.init_checkpoint}: {info['loaded_tensors']} tensors, "
              f"queries copied for {len(info['queries_copied'])} classes, skipped {info['skipped']}")

    counts = np.bincount(train_ds.labels, minlength=len(class_names)).astype(np.float64)
    use_weights = args.class_weights == "inverse" or (
        args.class_weights == "auto" and args.sampler == "shuffle")
    if use_weights:
        w = 1.0 / np.maximum(counts, 1.0)
        alpha = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
        print(f"Focal-loss class weights: min {alpha.min():.2f} max {alpha.max():.2f}")
    else:
        alpha = torch.ones(len(class_names), dtype=torch.float32, device=device)
    loss_fn = FocalLoss(alpha=alpha, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda(args))
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" and args.amp == "on" else None

    sample_weights = None
    if args.sampler == "balanced":
        sample_weights = torch.as_tensor(
            balanced_sample_weights(train_ds.labels, train_ds.subjects, args.balance_beta),
            dtype=torch.double)

    loader_kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda")
    # Few, non-persistent validation workers: persistent workers for both loaders
    # exhausted the 15 GB of RAM and stalled an epoch for 20+ minutes.
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False,
                            num_workers=min(4, args.workers), pin_memory=device.type == "cuda")

    history = defaultdict(list)
    start_epoch, best_metric, patience_ctr, completed = 0, -math.inf, 0, False
    last_path, best_path = output_dir / "last.pt", output_dir / "best_model.pt"
    if resuming:
        state = torch.load(last_path, weights_only=False, map_location=device)
        if state["package_manifest_sha256"] != pkg.manifest_sha256:
            raise SystemExit("last.pt was trained on a different package manifest")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        if scaler is not None and state.get("scaler_state_dict"):
            scaler.load_state_dict(state["scaler_state_dict"])
        start_epoch, best_metric = state["epoch"], state["best_metric"]
        patience_ctr, completed = state["patience_ctr"], state.get("completed", False)
        history = defaultdict(list, state["history"])
        print(f"Resumed from {last_path} after epoch {start_epoch} (best {args.select_metric} {best_metric:.4f})")

    def checkpoint_payload(epoch: int) -> dict:
        return {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "leads": leads,
            "class_names": class_names,
            "data_source": "package",
            "package_version": pkg.version,
            "package_manifest_sha256": pkg.manifest_sha256,
            "crop_len": args.crop_len,
            "filter_preset": args.filter_preset,
            "noise_aug_prob": args.noise_aug_prob,
            "fabricated_leads": args.fabricated_leads,
            "lead_fab_aug_prob": args.lead_fab_aug_prob,
            "select_metric": args.select_metric,
        }

    # Selection metric drops classes the package flags as uninterpretable (v2: VF has no
    # seven-measured-lead events anywhere, so its score cannot reflect deployment).
    select_excluded = excluded_from_primary(
        class_names, pkg.eval_flags if args.data_source == "package" else None,
    )
    select_classes = [i for i, n in enumerate(class_names) if n not in select_excluded] or None
    if select_excluded:
        print(f"Selection macro-F1 excludes {', '.join(select_excluded)} "
              f"({len(select_classes)} of {len(class_names)} classes)")

    print(f"\n=== Training (epochs {start_epoch + 1}..{args.epochs}, patience {args.patience}, "
          f"select on val {args.select_metric}) ===")
    t_start = time.time()
    epochs_run = 0
    for epoch in range(start_epoch, args.epochs):
        if completed:
            break
        t0 = time.time()
        train_ds.set_epoch(epoch)
        generator = torch.Generator().manual_seed(args.seed * 1000 + epoch)
        if sample_weights is not None:
            sampler = WeightedRandomSampler(sample_weights, len(train_ds), replacement=True,
                                            generator=generator)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                      drop_last=True, **loader_kw)
        else:
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                      generator=generator, drop_last=True, **loader_kw)
        tr_loss, tr_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device, scaler)
        vl_loss, vl_acc, y_val, p_val = validate(model, val_loader, loss_fn, device, return_predictions=True)
        vl_f1 = macro_f1(y_val, p_val, len(class_names), select_classes)
        scheduler.step()

        dt = time.time() - t0
        for key, value in (("train_loss", tr_loss), ("train_acc", tr_acc), ("val_loss", vl_loss),
                           ("val_acc", vl_acc), ("val_macro_f1", vl_f1),
                           ("lr", scheduler.get_last_lr()[0]), ("epoch_seconds", dt)):
            history[key].append(value)

        metric = vl_f1 if args.select_metric == "macro_f1" else vl_acc
        marker = ""
        if metric > best_metric:
            best_metric, patience_ctr, marker = metric, 0, "  *best*"
            torch.save({**checkpoint_payload(epoch + 1), "val_acc": vl_acc, "val_loss": vl_loss,
                        "val_macro_f1": vl_f1, "best_metric": best_metric}, best_path)
        else:
            patience_ctr += 1
        if patience_ctr >= args.patience:
            completed = True

        torch.save({
            **checkpoint_payload(epoch + 1),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "history": dict(history),
            "best_metric": best_metric,
            "patience_ctr": patience_ctr,
            "completed": completed,
        }, last_path)

        print(f"Epoch {epoch + 1:3d}/{args.epochs} | Train {tr_loss:.4f} / {tr_acc:.4f} | "
              f"Val {vl_loss:.4f} / acc {vl_acc:.4f} / mF1 {vl_f1:.4f} | "
              f"LR {scheduler.get_last_lr()[0]:.2e} | {dt:.1f}s{marker}", flush=True)
        epochs_run += 1

        if completed:
            print(f"\nEarly stopping (no improvement for {args.patience} epochs)")
            break
        if args.time_budget_min is not None and epoch + 1 < args.epochs:
            elapsed = time.time() - t_start
            if elapsed + elapsed / epochs_run > args.time_budget_min * 60:
                print(f"\nTime budget of {args.time_budget_min} min reached after epoch {epoch + 1}; "
                      f"rerun with --resume to continue.")
                return

    (output_dir / "history.json").write_text(json.dumps(dict(history), indent=2))
    if history["train_loss"]:
        save_training_curves(history, str(output_dir / "training_curves.png"))

    # ── Final evaluation on the test split ───────────────────────────────
    best = torch.load(best_path, weights_only=False, map_location=device)
    model.load_state_dict(best["model_state_dict"])

    if args.cv_fold is not None:
        # A cross-validation run must not see the test split: the fold score is the
        # selection signal, and test is spent once, on the artifact we commit to.
        torch.save(best, output_dir / "final_model.pt")
        print(f"\n=== Fold {args.cv_fold} complete (best epoch {best['epoch']}, "
              f"held-out {args.select_metric} {best['best_metric']:.4f}) ===")
        print("Test split not evaluated: cross-validation run.")
        print(f"\nOutputs in {output_dir}/: best_model.pt, last.pt, final_model.pt, "
              f"history.json, training_curves.png")
        return

    print(f"\n=== Test evaluation (best epoch {best['epoch']}, val {args.select_metric} "
          f"{best['best_metric']:.4f}) ===")
    models = [EvalModel(model, class_names, str(best_path))]
    sections = {}
    for label, crop in ((f"crop{args.crop_len}", args.crop_len), ("full", None)):
        ds = PackageDataset(pkg, "test", crop_len=crop, **common)
        sections[label] = evaluate_split(models, ds, device, args.eval_batch_size, args.workers)["report"]
        ov = sections[label]["overall"]
        lo, hi = ov["macro_f1_ci95"]
        print(f"  {label:<10s} n={ov['n']} accuracy {ov['accuracy']:.4f} macro-F1 {ov['macro']['f1']:.4f} "
              f"[{lo:.3f}, {hi:.3f}] recall {ov['macro']['recall']:.4f} spec {ov['macro']['specificity']:.4f}")
    crop_key = f"crop{args.crop_len}"
    per_class = sections[crop_key]["overall"]["per_class"]
    print(f"\n  {'Class':<26s} {'Prec':>6s} {'Rec':>6s} {'Spec':>6s} {'F1':>6s} {'N':>5s} {'Subj':>5s}")
    for name in class_names:
        m = per_class[name]
        print(f"  {name:<26s} {m['precision']:6.3f} {m['recall']:6.3f} {m['specificity']:6.3f} "
              f"{m['f1']:6.3f} {m['support']:5d} {m['subjects']:5d}")

    paths = write_reports(output_dir / "reports", "test", f"{pkg.version} test — {output_dir.name}",
                          sections, class_names, extra={"checkpoint": str(best_path)})
    save_confusion_png(np.asarray(sections[crop_key]["overall"]["confusion_matrix"]), class_names,
                       output_dir / "confusion_matrix.png", f"{pkg.version} test ({crop_key})")
    torch.save({**best, "test_reports": sections}, output_dir / "final_model.pt")
    print(f"\nOutputs in {output_dir}/: best_model.pt, last.pt, final_model.pt, history.json, "
          f"training_curves.png, confusion_matrix.png, {paths['md'].relative_to(output_dir)}")


if __name__ == "__main__":
    main()
