#!/usr/bin/env python3
"""Evaluate a trained ECG-TransCovNet checkpoint.

Usage:
    python scripts/evaluate.py --checkpoint models/best_model.pt
    python scripts/evaluate.py --checkpoint models/best_model.pt --test-dir data/test_clean
    python scripts/evaluate.py --checkpoint models/best_model.pt --num-samples 500 --noise-level medium

    # Real-data package split (baseline mode is automatic for a different head)
    python scripts/evaluate.py --checkpoint models/real_v1/best_model.pt \\
        --package ../ecg_sigma/packages/ecg_pkg_v1 --split test --output-dir reports/real_v1

    # Ensemble with val-fitted logit bias
    python scripts/evaluate.py --checkpoint run1/best_model.pt run2/best_model.pt \\
        --package ../ecg_sigma/packages/ecg_pkg_v1 --calibrate-on val --output-dir reports/ens
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ecg_transcovnet import FILTER_PRESETS
from ecg_transcovnet.augment import FABRICATION_PATTERNS
from ecg_transcovnet.checkpoint import load_model
from ecg_transcovnet.classes import ClassSpec
from ecg_transcovnet.data import generate_dataset, evaluate_hdf5_test
from ecg_transcovnet.evaluation import fit_logit_bias, summary_metrics, write_reports
from ecg_transcovnet.package import PackageDataset, load_package
from ecg_transcovnet.package_eval import EvalModel, evaluate_split
from ecg_transcovnet.training import evaluate_detailed
from ecg_transcovnet.visualization import save_confusion_matrix

# Classes that in ecgpkg v1 come mostly from sources with only ECG2 measured.
SINGLE_LEAD_SOURCE_CLASSES = (
    "ATRIAL_FIBRILLATION", "ATRIAL_FLUTTER", "VENTRICULAR_TACHYCARDIA", "VENTRICULAR_FIBRILLATION",
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate a trained ECG-TransCovNet model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, nargs="+", required=True,
                    help="Path(s) to model checkpoint(s) (.pt); several = softmax-averaged ensemble")
    p.add_argument("--test-dir", type=str, default=None,
                    help="Directory with HDF5 test files")
    p.add_argument("--num-samples", type=int, default=1000,
                    help="Number of synthetic samples to evaluate on (if no test-dir)")
    p.add_argument("--noise-level", type=str, default="clean",
                    choices=["clean", "low", "medium", "high", "mixed"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=99)
    p.add_argument("--output-dir", type=str, default=None,
                    help="Directory for confusion matrix / report output")
    p.add_argument("--filter-preset", type=str, default=None,
                    choices=list(FILTER_PRESETS.keys()),
                    help="Preprocessing filter preset (default: the checkpoint's)")

    g = p.add_argument_group("package evaluation")
    g.add_argument("--package", type=str, default=None, help="Evaluate on an ecgpkg package split")
    g.add_argument("--split", type=str, default="test", choices=["val", "test"])
    g.add_argument("--crop-len", type=int, default=2000, help="Centre-crop length")
    g.add_argument("--no-full-length", action="store_true", help="Skip the full-length section")
    g.add_argument("--lead-conversion", type=str, default="both",
                   choices=["none", "both", *FABRICATION_PATTERNS],
                   help="Counterfactual: rebuild non-ECG2 leads of all-real events and re-predict")
    g.add_argument("--calibrate-on", type=str, default="none", choices=["none", "val"],
                   help="Fit per-class logit bias on val macro-F1 and apply it")
    g.add_argument("--fabricated-leads", type=str, default=None, choices=["keep", "zero"],
                   help="Default: the checkpoint's setting")
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--cache-dir", type=str, default="data/training_cache")
    g.add_argument("--tag", type=str, default=None, help="Report file stem (default: split name)")
    return p


def main():
    args = build_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    try:
        loaded = [load_model(path, device) for path in args.checkpoint]
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return
    for path, lm in zip(args.checkpoint, loaded):
        ck = lm.checkpoint
        print(f"Loaded {path}: epoch {ck.get('epoch', '?')}, {len(lm.class_spec)} classes, "
              f"leads {lm.leads}, preset {lm.filter_preset}")

    if args.package:
        evaluate_package(args, loaded, device)
        return

    lm = loaded[0]
    model, leads, spec = lm.model, lm.leads, lm.class_spec
    filter_config = FILTER_PRESETS[args.filter_preset or lm.filter_preset]
    print(f"Filter preset: {args.filter_preset or lm.filter_preset}")

    # HDF5 test evaluation
    if args.test_dir:
        evaluate_hdf5_test(model, args.test_dir, leads, device, filter_config=filter_config,
                           class_names=list(spec.names))
        return

    if spec != ClassSpec.default():
        print("Error: simulator evaluation needs a 16-class simulator checkpoint; use --package")
        return

    # Synthetic data evaluation
    print(f"\nGenerating {args.num_samples} evaluation samples (noise={args.noise_level})...")
    from ecg_transcovnet.simulator.conditions import Condition
    balanced = {c: 1.0 / len(spec) for c in Condition}
    test_X, test_y = generate_dataset(
        args.num_samples, leads, args.noise_level, balanced, args.seed,
        filter_config=filter_config,
    )

    test_ds = TensorDataset(torch.from_numpy(test_X), torch.from_numpy(test_y))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    macro, per_class, cm = evaluate_detailed(model, test_loader, device, spec.names)

    print(f"\nMacro-averaged Metrics:")
    for k in ("accuracy", "precision", "recall", "specificity", "f1"):
        print(f"  {k:<14s}: {macro[k]:.4f}")

    print(f"\nPer-class Metrics:")
    print(f"  {'Condition':<28s} {'Prec':>6s} {'Rec':>6s} {'Spec':>6s} {'F1':>6s} {'N':>5s}")
    print("  " + "-" * 57)
    for name in spec.names:
        m = per_class[name]
        print(
            f"  {name:<28s} {m['precision']:6.3f} {m['recall']:6.3f} "
            f"{m['specificity']:6.3f} {m['f1']:6.3f} {m['support']:5d}"
        )

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        cm_path = str(out / "eval_confusion_matrix.png")
        save_confusion_matrix(cm, list(spec.names), cm_path)
        print(f"\nConfusion matrix saved to {cm_path}")


def evaluate_package(args, loaded, device) -> None:
    if not args.output_dir:
        raise SystemExit("--package evaluation requires --output-dir")
    pkg = load_package(args.package)
    head = list(pkg.spec.names)
    ref = loaded[0]
    if any(lm.leads != ref.leads for lm in loaded):
        raise SystemExit("all checkpoints must use the same leads")
    preset = args.filter_preset or ref.filter_preset
    fabricated = args.fabricated_leads or ref.checkpoint.get("fabricated_leads", "keep")
    models = [EvalModel(lm.model, list(lm.class_spec.names), path)
              for lm, path in zip(loaded, args.checkpoint)]
    baseline = any(m.class_names != head for m in models)
    if baseline:
        print(f"Baseline mode: checkpoint head differs from the package head {head}; "
              f"predictions are mapped by name, predictions outside the head count as wrong")
    common = dict(leads=ref.leads, filter_config=FILTER_PRESETS[preset], fabricated_leads=fabricated,
                  cache_dir=args.cache_dir)
    bs, workers = args.batch_size, args.workers

    bias = None
    if args.calibrate_on == "val":
        if baseline:
            print("Skipping calibration in baseline mode")
        else:
            val = evaluate_split(models, PackageDataset(pkg, "val", crop_len=args.crop_len, **common),
                                 device, bs, workers)
            print("WARNING: --calibrate-on val fits a per-class logit bias on the "
                  "validation split. Measured at -0.05 macro-F1 on v1 test and "
                  "ecg_sigma advises against it (RESPONSE_change_requests.md, Q1). "
                  "Use for research only; do not ship a calibrated checkpoint.")
            bias = fit_logit_bias(np.log(val["probs"] + 1e-9), val["labels"])
            print("Val-fitted logit bias: " + ", ".join(f"{n} {b:+.2f}" for n, b in zip(head, bias)))

    sections: dict[str, dict] = {}
    crop_key = f"crop{args.crop_len}"
    crop = evaluate_split(models, PackageDataset(pkg, args.split, crop_len=args.crop_len, **common),
                          device, bs, workers, bias)
    sections[crop_key] = crop["report"]
    if not args.no_full_length:
        full_ds = PackageDataset(pkg, args.split, crop_len=None, **common)
        sections["full"] = evaluate_split(models, full_ds, device, bs, workers, bias)["report"]

    patterns = {"none": (), "both": FABRICATION_PATTERNS}.get(args.lead_conversion, (args.lead_conversion,))
    conversion: dict[str, dict] = {}
    labels = crop["labels"]
    for pattern in patterns:
        ds = PackageDataset(pkg, args.split, crop_len=args.crop_len, force_pattern=pattern, **common)
        target = np.array([ds.is_counterfactual_target(i) for i in range(len(ds))])
        if not target.any():
            continue
        cf = evaluate_split(models, ds, device, bs, workers, bias, rows_mask=target)
        sections[f"leadconv_{pattern}"] = cf["report"]
        native, converted = crop["pred"][target], cf["pred"][target]
        single = [head.index(c) for c in SINGLE_LEAD_SOURCE_CLASSES if c in head]
        flips = {}
        for i, name in enumerate(head):
            sel = labels[target] == i
            if sel.any():
                flips[name] = float((native[sel] != converted[sel]).mean())
        conversion[pattern] = {
            "events": int(target.sum()),
            "flip_rate": float((native != converted).mean()),
            "native": summary_metrics(labels[target], native, head),
            "converted": summary_metrics(labels[target], converted, head),
            "pred_single_lead_source_classes_native": float(np.isin(native, single).mean()),
            "pred_single_lead_source_classes_converted": float(np.isin(converted, single).mean()),
            "flip_rate_by_true_class": flips,
        }

    stem = args.tag or args.split
    title = f"{pkg.version} {args.split} — " + ", ".join(Path(c).parent.name or c for c in args.checkpoint)
    paths = write_reports(args.output_dir, stem, title, sections, head, extra={
        "checkpoints": args.checkpoint, "baseline_mode": baseline, "filter_preset": preset,
        "fabricated_leads": fabricated, "logit_bias": None if bias is None else bias.tolist(),
        "lead_conversion": conversion,
    })

    print(f"\n{'Section':<20s} {'N':>6s} {'Acc':>7s} {'MacroF1':>8s} {'95% CI':>15s} {'Recall':>7s} {'Spec':>7s}")
    for name, rep in sections.items():
        ov = rep["overall"]
        lo, hi = ov["macro_f1_ci95"]
        print(f"{name:<20s} {ov['n']:6d} {ov['accuracy']:7.4f} {ov['macro']['f1']:8.4f} "
              f"   [{lo:.3f},{hi:.3f}] {ov['macro']['recall']:7.4f} {ov['macro']['specificity']:7.4f}")
    per_class = sections[crop_key]["overall"]["per_class"]
    print(f"\n{'Class':<26s} {'Prec':>6s} {'Rec':>6s} {'Spec':>6s} {'F1':>6s} {'N':>5s} {'Subj':>5s}")
    for name in head:
        m = per_class[name]
        print(f"{name:<26s} {m['precision']:6.3f} {m['recall']:6.3f} {m['specificity']:6.3f} "
              f"{m['f1']:6.3f} {m['support']:5d} {m['subjects']:5d}")
    for pattern, c in conversion.items():
        print(f"\nLead conversion to {pattern} on {c['events']} events with more real leads: "
              f"flip rate {c['flip_rate']:.3f}, "
              f"macro-F1 {c['native']['macro_f1']:.3f} → {c['converted']['macro_f1']:.3f}, "
              f"predicted AF/AFL/VT/VF {c['pred_single_lead_source_classes_native']:.3f} → "
              f"{c['pred_single_lead_source_classes_converted']:.3f}")
    print(f"\nReports: {paths['md']}, {paths['json']}")


if __name__ == "__main__":
    main()
