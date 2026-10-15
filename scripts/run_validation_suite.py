#!/usr/bin/env python3
"""Run the validation suite against a trained model and produce a comprehensive report.

Processes all HDF5 files in a validation suite directory, computes per-file and
aggregate metrics, and generates visual reports. Works with legacy 16-class and
package-trained checkpoints, and with simulator and ecg_sigma HDF5 files.

Usage:
    python scripts/run_validation_suite.py \\
        --suite-dir data/validation_suite \\
        --checkpoint models/avblock_fix/best_model.pt

    # Save plots for each file
    python scripts/run_validation_suite.py \\
        --suite-dir data/validation_suite \\
        --checkpoint models/avblock_fix/best_model.pt \\
        --save-plots
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ecg_transcovnet import FILTER_PRESETS, PreprocessingPipeline
from ecg_transcovnet.checkpoint import load_model as load_checkpoint_model
from ecg_transcovnet.classes import ClassSpec
from ecg_transcovnet.evaluation import compute_metrics as _compute_metrics, save_confusion_png
from ecg_transcovnet.hdf5_io import event_keys as list_event_keys, read_ecg_leads

# Conditions are resolved by enum value (simulator, e.g. "V") or name
# (ecg_sigma, e.g. "PVC"); labels outside the head are skipped with one warning.
_NOT_IN_HEAD_SEEN: set[str] = set()


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device):
    try:
        loaded = load_checkpoint_model(checkpoint_path, device)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    ckpt = loaded.checkpoint
    epoch = ckpt.get("epoch", "?")
    val_acc = ckpt.get("val_acc", "?")
    return loaded.model, loaded.leads, epoch, val_acc, loaded.class_spec, loaded.filter_preset


# ── Per-file inference ────────────────────────────────────────────────────────

def process_file(
    filepath: Path,
    model: torch.nn.Module,
    leads: list[str],
    device: torch.device,
    class_spec: ClassSpec | None = None,
    pipeline: PreprocessingPipeline | None = None,
) -> list[dict]:
    """Run inference on all events in an HDF5 file. Returns list of result dicts."""
    spec = class_spec or ClassSpec.default()
    pipeline = pipeline or PreprocessingPipeline(FILTER_PRESETS["none"])
    results = []
    try:
        hf = h5py.File(filepath, "r")
    except Exception as e:
        print(f"    [!] Could not open {filepath.name}: {e}")
        return results

    with hf:
        for event_key in list_event_keys(hf):
            grp = hf[event_key]
            event_id = event_key.replace("event_", "")

            # Ground truth
            gt_val = grp.attrs.get("condition", None)
            if gt_val is None:
                continue
            gt_name, gt_idx = spec.resolve(gt_val)
            if gt_idx is None:
                if gt_name not in _NOT_IN_HEAD_SEEN:
                    _NOT_IN_HEAD_SEEN.add(gt_name)
                    print(f"    [!] Ground truth '{gt_name}' is not in the model head — event(s) skipped")
                continue

            # ECG leads
            if "ecg" not in grp:
                continue
            try:
                signal = read_ecg_leads(grp["ecg"], leads)
            except KeyError:
                continue

            # Preprocessing (preset "none" = per-lead z-score only)
            signal = pipeline(signal)

            # Inference
            with torch.no_grad():
                x = torch.from_numpy(signal.astype(np.float32)).unsqueeze(0).to(device)
                logits = model(x)
                probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])

            results.append({
                "event_id": event_id,
                "gt_idx": gt_idx,
                "gt_name": spec.names[gt_idx],
                "pred_idx": pred_idx,
                "pred_name": spec.names[pred_idx],
                "confidence": confidence,
                "correct": pred_idx == gt_idx,
            })

    return results


# ── Metrics computation ───────────────────────────────────────────────────────

def compute_metrics(y_true: list[int], y_pred: list[int], class_names: list[str]) -> dict:
    """:func:`evaluation.compute_metrics` plus the totals this report prints."""
    m = _compute_metrics(y_true, y_pred, class_names)
    return {**m, "macro_f1": m["macro"]["f1"], "total": m["n"],
            "correct": int(np.sum(np.asarray(y_true) == np.asarray(y_pred)))}


def save_noise_comparison_chart(noise_metrics: dict[str, dict], path: str):
    """Bar chart comparing accuracy and macro F1 across noise levels."""
    noise_levels = sorted(noise_metrics.keys(),
                          key=lambda x: ["clean", "low", "medium", "high"].index(x)
                          if x in ["clean", "low", "medium", "high"] else 99)

    accuracies = [noise_metrics[n]["accuracy"] * 100 for n in noise_levels]
    f1_scores = [noise_metrics[n]["macro_f1"] * 100 for n in noise_levels]

    x = np.arange(len(noise_levels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, accuracies, width, label="Accuracy", color="#2196F3")
    bars2 = ax.bar(x + width / 2, f1_scores, width, label="Macro F1", color="#4CAF50")

    ax.set_xlabel("Noise Level")
    ax.set_ylabel("Score (%)")
    ax.set_title("Model Performance vs Noise Level")
    ax.set_xticks(x)
    ax.set_xticklabels(noise_levels)
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_per_class_f1_chart(per_class: dict, title: str, path: str, class_names: list[str]):
    """Horizontal bar chart of per-class F1 scores."""
    names = []
    f1s = []
    for name in class_names:
        if per_class[name]["support"] > 0:
            names.append(name)
            f1s.append(per_class[name]["f1"])

    colors = ["#F44336" if f < 0.8 else "#FF9800" if f < 0.9 else "#4CAF50" for f in f1s]

    fig, ax = plt.subplots(figsize=(12, 8))
    y = np.arange(len(names))
    ax.barh(y, f1s, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("F1 Score")
    ax.set_title(title)
    ax.set_xlim(0, 1.05)
    ax.axvline(0.9, color="gray", linestyle="--", alpha=0.5, label="0.9 threshold")
    ax.axvline(0.8, color="gray", linestyle=":", alpha=0.5, label="0.8 threshold")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    # Value labels
    for i, v in enumerate(f1s):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=7)

    fig.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ── Report generation ─────────────────────────────────────────────────────────

def print_section(title: str, char: str = "="):
    width = 70
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def generate_report(
    file_results: list[dict],
    output_dir: Path,
    checkpoint_info: dict,
    save_plots: bool,
):
    """Generate comprehensive text + visual report from all file results."""
    class_names = list(checkpoint_info.get("class_names") or ClassSpec.default().names)
    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    report_lines: list[str] = []

    def rprint(line: str = ""):
        print(line)
        report_lines.append(line)

    # ── Header ──
    print_section("ECG-TransCovNet Validation Suite Report")
    rprint(f"  Checkpoint:  {checkpoint_info['path']}")
    rprint(f"  Epoch:       {checkpoint_info['epoch']}")
    rprint(f"  Val accuracy: {checkpoint_info['val_acc']}")
    rprint(f"  Device:      {checkpoint_info['device']}")
    rprint(f"  Head:        {len(class_names)} classes · filter preset {checkpoint_info.get('filter_preset', 'none')}")

    # ── Per-file results ──
    all_true, all_pred = [], []
    noise_buckets: dict[str, dict] = defaultdict(lambda: {"true": [], "pred": []})

    for fr in file_results:
        noise = fr["noise_level"]
        for r in fr["results"]:
            all_true.append(r["gt_idx"])
            all_pred.append(r["pred_idx"])
            noise_buckets[noise]["true"].append(r["gt_idx"])
            noise_buckets[noise]["pred"].append(r["pred_idx"])

    # ── Per-file summary table ──
    print_section("Per-File Results", "-")
    rprint(f"  {'File':<48s} {'Noise':<8s} {'Events':>6s} {'Correct':>7s} {'Acc':>7s}")
    rprint("  " + "-" * 78)
    for fr in file_results:
        results = fr["results"]
        correct = sum(1 for r in results if r["correct"])
        total = len(results)
        acc = correct / total * 100 if total > 0 else 0
        rprint(
            f"  {fr['filename']:<48s} {fr['noise_level']:<8s} "
            f"{total:>6d} {correct:>7d} {acc:>6.1f}%"
        )

    # ── Aggregate metrics ──
    print_section("Aggregate Metrics")
    agg = compute_metrics(all_true, all_pred, class_names)
    rprint(f"  Total events:  {agg['total']}")
    rprint(f"  Correct:       {agg['correct']}")
    rprint(f"  Accuracy:      {agg['accuracy']:.4f} ({agg['accuracy'] * 100:.1f}%)")
    rprint(f"  Macro F1:      {agg['macro_f1']:.4f}")

    # ── Per-class metrics ──
    print_section("Per-Class Metrics (Aggregate)", "-")
    rprint(f"  {'Condition':<28s} {'Prec':>6s} {'Rec':>6s} {'Spec':>6s} {'F1':>6s} {'N':>5s}")
    rprint("  " + "-" * 57)
    for name in class_names:
        m = agg["per_class"][name]
        if m["support"] > 0:
            rprint(
                f"  {name:<28s} {m['precision']:6.3f} {m['recall']:6.3f} "
                f"{m['specificity']:6.3f} {m['f1']:6.3f} {m['support']:5d}"
            )

    # ── Noise level comparison ──
    print_section("Performance by Noise Level", "-")
    noise_metrics = {}
    rprint(f"  {'Noise Level':<12s} {'Events':>7s} {'Accuracy':>10s} {'Macro F1':>10s}")
    rprint("  " + "-" * 41)
    for noise in ["clean", "low", "medium", "high"]:
        if noise not in noise_buckets:
            continue
        bucket = noise_buckets[noise]
        nm = compute_metrics(bucket["true"], bucket["pred"], class_names)
        noise_metrics[noise] = nm
        rprint(
            f"  {noise:<12s} {nm['total']:>7d} "
            f"{nm['accuracy'] * 100:>9.1f}% {nm['macro_f1']:>10.4f}"
        )

    # ── AV Block focus (if data available) ──
    av_conditions = {"AV_BLOCK_1", "AV_BLOCK_2_TYPE1", "AV_BLOCK_2_TYPE2", "NORMAL_SINUS"}
    av_true, av_pred = [], []
    for t, p in zip(all_true, all_pred):
        if class_names[t] in av_conditions:
            av_true.append(t)
            av_pred.append(p)

    if av_true:
        print_section("AV Block Discrimination (Key Validation)", "-")
        av_metrics = compute_metrics(av_true, av_pred, class_names)
        rprint(f"  Events (N/1AVB/2AVB1/2AVB2 only): {len(av_true)}")
        rprint(f"  Accuracy: {av_metrics['accuracy']:.4f} ({av_metrics['accuracy'] * 100:.1f}%)")
        rprint()
        # Per-class rows come from all events: false positives from other true classes
        # must count against precision.
        for name in ["NORMAL_SINUS", "AV_BLOCK_1", "AV_BLOCK_2_TYPE1", "AV_BLOCK_2_TYPE2"]:
            m = agg["per_class"].get(name)
            if m and m["support"] > 0:
                rprint(
                    f"  {name:<28s} Prec={m['precision']:.3f}  "
                    f"Rec={m['recall']:.3f}  F1={m['f1']:.3f}  N={m['support']}"
                )

    # ── Misclassifications detail ──
    misses = []
    for fr in file_results:
        for r in fr["results"]:
            if not r["correct"]:
                misses.append({**r, "file": fr["filename"], "noise": fr["noise_level"]})

    if misses:
        print_section("Misclassifications Detail", "-")
        rprint(f"  Total misclassifications: {len(misses)}/{agg['total']}")
        rprint()
        rprint(f"  {'File':<35s} {'Event':>5s} {'True':<24s} {'Predicted':<24s} {'Conf':>5s} {'Noise':<6s}")
        rprint("  " + "-" * 101)
        for m in misses[:50]:  # cap at 50
            rprint(
                f"  {m['file']:<35s} {m['event_id']:>5s} "
                f"{m['gt_name']:<24s} {m['pred_name']:<24s} "
                f"{m['confidence']:5.2f} {m['noise']:<6s}"
            )
        if len(misses) > 50:
            rprint(f"  ... and {len(misses) - 50} more")
    else:
        print_section("Misclassifications Detail", "-")
        rprint("  No misclassifications! Perfect accuracy.")

    # ── Save plots ──
    if save_plots:
        print_section("Saving Visual Reports", "-")

        cm_path = str(report_dir / "confusion_matrix_aggregate.png")
        save_confusion_png(agg["confusion_matrix"], class_names, cm_path, "Aggregate Confusion Matrix")
        rprint(f"  Saved: {cm_path}")

        f1_path = str(report_dir / "per_class_f1_aggregate.png")
        save_per_class_f1_chart(agg["per_class"], "Per-Class F1 (Aggregate)", f1_path, class_names)
        rprint(f"  Saved: {f1_path}")

        if len(noise_metrics) > 1:
            noise_path = str(report_dir / "noise_comparison.png")
            save_noise_comparison_chart(noise_metrics, noise_path)
            rprint(f"  Saved: {noise_path}")

        # Per-noise confusion matrices
        for noise, bucket in noise_buckets.items():
            nm = compute_metrics(bucket["true"], bucket["pred"], class_names)
            if nm["total"] > 0:
                path = str(report_dir / f"confusion_matrix_{noise}.png")
                save_confusion_png(nm["confusion_matrix"], class_names, path,
                                   f"Confusion Matrix — {noise} noise (acc={nm['accuracy']:.1%})")
                rprint(f"  Saved: {path}")

    # ── Save text report ──
    report_path = report_dir / "validation_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\n  Text report saved to: {report_path}")

    # ── Save JSON metrics ──
    json_metrics = {
        "checkpoint": checkpoint_info["path"],
        "class_names": class_names,
        "aggregate": {
            "accuracy": agg["accuracy"],
            "macro_f1": agg["macro_f1"],
            "total_events": agg["total"],
            "correct": agg["correct"],
        },
        "per_noise_level": {
            noise: {"accuracy": nm["accuracy"], "macro_f1": nm["macro_f1"], "total": nm["total"]}
            for noise, nm in noise_metrics.items()
        },
        "per_class": {
            name: {k: v for k, v in m.items()}
            for name, m in agg["per_class"].items()
        },
        "misclassifications": len(misses),
    }
    json_path = report_dir / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(json_metrics, f, indent=2)
    print(f"  JSON metrics saved to: {json_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run validation suite and generate comprehensive report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--suite-dir", type=str, required=True,
                    help="Directory containing validation suite HDF5 files.")
    p.add_argument("--checkpoint", type=str, required=True,
                    help="Path to model checkpoint (.pt).")
    p.add_argument("--filter-preset", type=str, default=None,
                    choices=list(FILTER_PRESETS.keys()),
                    help="Preprocessing filter preset (default: the checkpoint's).")
    p.add_argument("--save-plots", action="store_true",
                    help="Save confusion matrices and charts to report/ subdirectory.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    suite_dir = Path(args.suite_dir)
    if not suite_dir.exists():
        print(f"Error: suite directory not found: {suite_dir}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, leads, epoch, val_acc, class_spec, checkpoint_preset = load_model(args.checkpoint, device)
    filter_preset = args.filter_preset or checkpoint_preset
    pipeline = PreprocessingPipeline(FILTER_PRESETS[filter_preset])

    checkpoint_info = {
        "path": args.checkpoint,
        "epoch": epoch,
        "val_acc": val_acc,
        "device": str(device),
        "class_names": list(class_spec.names),
        "filter_preset": filter_preset,
    }

    # Load manifest if available (for noise level info)
    manifest_path = suite_dir / "manifest.json"
    file_noise_map = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        for fi in manifest.get("files", []):
            levels = list(fi.get("noise_levels", {}))
            noise = fi.get("noise_level") or (levels[0] if len(levels) == 1 else "mixed")
            file_noise_map[fi.get("filename") or fi["file"]] = noise

    # Find and process all HDF5 files
    h5_files = sorted(suite_dir.glob("*.h5"))
    if not h5_files:
        print(f"No .h5 files found in {suite_dir}")
        sys.exit(1)

    print(f"Processing {len(h5_files)} files from {suite_dir}/...")
    t0 = time.time()

    file_results = []
    for filepath in h5_files:
        noise = file_noise_map.get(filepath.name, "unknown")
        results = process_file(filepath, model, leads, device, class_spec, pipeline)
        correct = sum(1 for r in results if r["correct"])
        total = len(results)
        acc_str = f"{correct}/{total}" if total > 0 else "0/0"
        print(f"  {filepath.name:<48s} {acc_str:>6s} ({noise})")
        file_results.append({
            "filename": filepath.name,
            "noise_level": noise,
            "results": results,
        })

    elapsed = time.time() - t0
    print(f"\nInference complete in {elapsed:.1f}s")

    generate_report(file_results, suite_dir, checkpoint_info, args.save_plots)


if __name__ == "__main__":
    main()
