#!/usr/bin/env python3
"""Slide-ready PNG figures from real-data (ecgpkg) evaluation reports.

Reads the JSON written by ``scripts/evaluate.py --package`` (or the
``reports/test.json`` written at the end of package training) and the
baseline-mode reports of legacy checkpoints.

Usage:
    python scripts/plot_real_data_results.py \\
        --report "Real data: run b=models/experiments/v1_b/eval/test.json" \\
        --baseline "Simulator: best_model=models/experiments/a_baseline_best_model/test.json" \\
        --baseline "Simulator: noise_robust=models/experiments/a_baseline_noise_robust/test.json" \\
        --baseline "Simulator: avblock_fix=models/experiments/a_baseline_avblock_fix/test.json" \\
        --output-dir docs/figures/real_data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, MultipleLocator

# Chart chrome and palette (light surface)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]           # categorical slots 1-3
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

LABELS = {
    "NORMAL_SINUS": "Normal sinus", "SINUS_BRADYCARDIA": "Sinus brady", "SINUS_TACHYCARDIA": "Sinus tachy",
    "ATRIAL_FIBRILLATION": "AF", "ATRIAL_FLUTTER": "Atrial flutter", "PAC": "PAC", "PVC": "PVC",
    "VENTRICULAR_TACHYCARDIA": "VT", "VENTRICULAR_FIBRILLATION": "VF", "LBBB": "LBBB", "RBBB": "RBBB",
    "AV_BLOCK_1": "AV block 1",
}
DATASETS = {"ptbxl": "PTB-XL", "incart": "INCART", "mitbih": "MIT-BIH", "afdb": "AFDB",
            "vfdb": "VFDB", "cudb": "CUDB"}
SIZE = (13.33, 7.5)   # 16:9 slide
DPI = 150

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 12, "text.color": INK,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
})
PCT = FuncFormatter(lambda v, _: f"{v:.0%}")


def _titles(fig, title: str, subtitle: str) -> None:
    fig.text(0.03, 0.955, title, fontsize=20, fontweight="semibold", color=INK, va="top")
    fig.text(0.03, 0.895, subtitle, fontsize=12.5, color=INK_2, va="top")


def _hairline_grid(ax, axis: str = "y") -> None:
    ax.grid(axis=axis, color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)


def _load(spec: str) -> tuple[str, dict]:
    label, _, path = spec.partition("=")
    return label, json.loads(Path(path).read_text())


def _crop_section(report: dict) -> dict:
    key = next(k for k in report["sections"] if k.startswith("crop"))
    return report["sections"][key]


# ---------------------------------------------------------------------------

def headline(models: list[tuple[str, dict]], path: Path) -> None:
    metrics = [("accuracy", "Accuracy"), ("f1", "Macro-F1"), ("recall", "Macro sensitivity")]
    fig, ax = plt.subplots(figsize=SIZE)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.74, bottom=0.14)
    x = np.arange(len(models))
    width = 0.2
    for k, (key, label) in enumerate(metrics):
        values = []
        for _, rep in models:
            ov = _crop_section(rep)["overall"]
            values.append(ov["accuracy"] if key == "accuracy" else ov["macro"][key])
        pos = x + (k - 1) * (width + 0.03)
        ax.bar(pos, values, width, color=SERIES[k], label=label, zorder=3)
        for p, v in zip(pos, values):
            ax.text(p, v + 0.015, f"{v:.0%}", ha="center", va="bottom", fontsize=11, color=INK_2)
    ax.set_xticks(x, [m[0] for m in models], fontsize=12.5, color=INK)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PCT)
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    _hairline_grid(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, frameon=False, fontsize=12)
    real = _crop_section(models[-1][1])["overall"]
    lo, hi = real["macro_f1_ci95"]
    _titles(fig, "Real-data training lifts accuracy from ≤26 % to 75 %",
            f"ecg_sigma package v1 test split · {real['n']:,} events from unseen subjects · 12 classes · "
            f"real-data macro-F1 95 % CI {lo:.2f}–{hi:.2f} (subject bootstrap)")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def confusion(report: dict, title: str, subtitle: str, path: Path, mode: str) -> None:
    ov = _crop_section(report)["overall"]
    names = report["class_names"]
    cm = np.asarray(ov["confusion_matrix"])
    n = len(names)
    x_names = [LABELS.get(c, c) for c in names] + (["Outside\nhead"] if cm.shape[1] > n else [])
    row_totals = cm.sum(axis=1)
    frac = cm / np.maximum(row_totals[:, None], 1)

    fig, ax = plt.subplots(figsize=SIZE)
    fig.subplots_adjust(left=0.17, right=0.93, top=0.83, bottom=0.14)
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    im = ax.imshow(frac, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(cm.shape[1]), x_names, rotation=35, ha="right", fontsize=11, color=INK)
    ax.set_yticks(np.arange(n), [f"{LABELS.get(c, c)}  ({row_totals[i]:,})" for i, c in enumerate(names)],
                  fontsize=11, color=INK)
    ax.set_xticks(np.arange(-0.5, cm.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    for i in range(n):
        for j in range(cm.shape[1]):
            if cm[i, j] == 0:
                continue
            if mode == "percent" and frac[i, j] < 0.005:
                continue
            text = f"{cm[i, j]:,}" if mode == "counts" else f"{frac[i, j]:.0%}"
            ax.text(j, i, text, ha="center", va="center", fontsize=10.5,
                    color="white" if frac[i, j] > 0.45 else INK,
                    fontweight="semibold" if i == j else "normal")
    ax.set_xlabel("Predicted class", fontsize=12, labelpad=8)
    ax.set_ylabel("True class  (test events)", fontsize=12, labelpad=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, format=PCT)
    cbar.outline.set_visible(False)
    cbar.set_label("Share of the true class", color=INK_2)
    cbar.ax.tick_params(colors=INK_2, length=0)
    _titles(fig, title, subtitle)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def per_class(report: dict, path: Path) -> None:
    ov = _crop_section(report)["overall"]
    names = report["class_names"]
    metrics = [("recall", "Sensitivity"), ("specificity", "Specificity"), ("precision", "Precision (PPV)")]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=SIZE)
    fig.subplots_adjust(left=0.2, right=0.86, top=0.8, bottom=0.1)
    for k, (key, label) in enumerate(metrics):
        values = [ov["per_class"][c][key] for c in names]
        ax.scatter(values, y + (k - 1) * 0.24, s=70, color=SERIES[k], edgecolors=SURFACE,
                   linewidths=2, label=label, zorder=3)
    labels = []
    for c in names:
        s = ov["per_class"][c]["subjects"]
        labels.append(f"{LABELS.get(c, c)}  ({s} subj.{' ⚠' if s < 5 else ''})")
    ax.set_yticks(y, labels, fontsize=11.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_formatter(PCT)
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    _hairline_grid(ax, "x")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    for i, c in enumerate(names):
        ax.text(1.02, i, f"F1 {ov['per_class'][c]['f1']:.2f}", transform=ax.get_yaxis_transform(),
                va="center", fontsize=11, color=INK_2)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=12)
    m = ov["macro"]
    _titles(fig, "Per-class sensitivity, specificity and precision",
            f"Real-data model · v1 test · macro sensitivity {m['recall']:.2f}, specificity {m['specificity']:.2f}, "
            f"precision {m['precision']:.2f} · ⚠ fewer than 5 test subjects (indicative)")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def by_source(report: dict, path: Path) -> None:
    groups = _crop_section(report)["by_dataset"]
    keys = [k for k in DATASETS if k in groups]
    fig, ax = plt.subplots(figsize=SIZE)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.74, bottom=0.14)
    x = np.arange(len(keys))
    width = 0.3
    for k, (field, label) in enumerate((("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"))):
        values = [groups[g][field] for g in keys]
        pos = x + (k - 0.5) * (width + 0.04)
        ax.bar(pos, values, width, color=SERIES[k], label=label, zorder=3)
        for p, v in zip(pos, values):
            ax.text(p, v + 0.015, f"{v:.0%}", ha="center", va="bottom", fontsize=11, color=INK_2)
    ax.set_xticks(x, [f"{DATASETS[g]}\n{groups[g]['n']:,} events" for g in keys], fontsize=12, color=INK)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PCT)
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    _hairline_grid(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False, fontsize=12)
    masks = _crop_section(report)["by_real_lead_mask"]
    mask_txt = " · ".join(f"{lbl} {masks[m]['accuracy']:.0%}" for m, lbl in
                          (("1111111", "7 real leads"), ("0100001", "ECG2+V1 real"), ("0100000", "ECG2 only"))
                          if m in masks)
    _titles(fig, "Performance by source database",
            f"Real-data model · v1 test · accuracy by measured leads: {mask_txt}")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def lead_realism(report: dict, path: Path) -> None:
    conv = report.get("lead_conversion") or {}
    patterns = [(p, lbl) for p, lbl in (("0100001", "Limb leads rebuilt\nfrom ECG2 (V1 kept)"),
                                        ("0100000", "All leads except ECG2\nrebuilt"))
                if p in conv]
    if not patterns:
        return
    fig, axes = plt.subplots(1, 2, figsize=SIZE, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.72, bottom=0.16, wspace=0.12)
    panels = [
        ("Macro-F1 on the converted events", lambda c, s: c[s]["macro_f1"]),
        ("Share predicted AF / flutter / VT / VF", lambda c, s: c[f"pred_single_lead_source_classes_{s}"]),
    ]
    x = np.arange(len(patterns))
    width = 0.3
    for ax, (panel_title, getter) in zip(axes, panels):
        for k, (state, label) in enumerate((("native", "Measured leads"), ("converted", "After conversion"))):
            values = [getter(conv[p], state) for p, _ in patterns]
            pos = x + (k - 0.5) * (width + 0.04)
            ax.bar(pos, values, width, color=SERIES[k], label=label, zorder=3)
            for p, v in zip(pos, values):
                ax.text(p, v + 0.015, f"{v:.0%}", ha="center", va="bottom", fontsize=11, color=INK_2)
        ax.set_xticks(x, [f"{lbl}\n{conv[p]['events']:,} events · {conv[p]['flip_rate']:.0%} flipped"
                          for p, lbl in patterns], fontsize=11, color=INK)
        ax.set_title(panel_title, fontsize=13, color=INK, loc="left")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PCT)
        ax.yaxis.set_major_locator(MultipleLocator(0.25))
        _hairline_grid(ax)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2, frameon=False, fontsize=12)
    _titles(fig, "Lead-realism check: does the model rely on fabricated leads?",
            "Leads rebuilt from ECG2 as in single-lead sources (VFDB, CUDB, AFDB) · "
            "a rhythm-based model should barely change")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--report", required=True, help="'Label=path/to/test.json' of the real-data model")
    p.add_argument("--baseline", action="append", default=[], help="'Label=path/to/test.json' (repeatable)")
    p.add_argument("--output-dir", default="docs/figures/real_data")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    real_label, real = _load(args.report)
    baselines = [_load(b) for b in args.baseline]
    ov = _crop_section(real)["overall"]

    headline(baselines + [(real_label, real)], out / "01_headline_comparison.png")
    confusion(real, f"Confusion matrix — {real_label} (counts)",
              f"v1 test · {ov['n']:,} events · accuracy {ov['accuracy']:.1%} · macro-F1 {ov['macro']['f1']:.2f} · "
              "colour = share of the true class", out / "02_confusion_real_counts.png", "counts")
    confusion(real, f"Confusion matrix — {real_label} (row %)",
              "Each row sums to 100 %: the diagonal is the sensitivity of that class",
              out / "03_confusion_real_percent.png", "percent")
    per_class(real, out / "04_per_class_metrics.png")
    by_source(real, out / "05_by_source.png")
    lead_realism(real, out / "06_lead_realism_check.png")
    for i, (label, rep) in enumerate(baselines):
        bov = _crop_section(rep)["overall"]
        slug = label.split(":")[-1].strip().replace(" ", "_")
        confusion(rep, f"Confusion matrix — {label} (row %)",
                  f"Simulator-trained checkpoint on real v1 test · accuracy {bov['accuracy']:.1%} · "
                  f"macro-F1 {bov['macro']['f1']:.2f} · predictions of classes outside the 12-class head in the last column",
                  out / f"{7 + i:02d}_confusion_baseline_{slug}.png", "percent")
    for f in sorted(out.glob("*.png")):
        print(f)


if __name__ == "__main__":
    main()
