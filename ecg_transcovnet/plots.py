"""Plot generation for ECG inference reports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .constants import ALL_LEADS
from .mews import compute_mews_history
from .report import FileResult
from .visualization import plot_ecg_waveform


# ---------------------------------------------------------------------------
# Vitals subplot config shared by file-level and per-event plots
# ---------------------------------------------------------------------------

_VITALS_CONFIG = [
    ("HR", "Heart Rate (bpm)", "tab:red"),
    ("SpO2", "SpO2 (%)", "tab:blue"),
    ("Systolic", "Blood Pressure (mmHg)", "tab:green"),
    ("RespRate", "Resp Rate (br/min)", "tab:orange"),
    ("Temp", "Temperature (\u00b0F)", "tab:purple"),
]

_DEFAULT_THRESH = {
    "HR": (60, 100), "SpO2": (94, 100), "Systolic": (90, 140),
    "RespRate": (12, 20), "Temp": (96.8, 100.4),
}


# ---------------------------------------------------------------------------
# Per-event plot functions
# ---------------------------------------------------------------------------

def plot_event_ecg(
    signal: np.ndarray,
    event_id: str,
    output_path: Path,
    fs: float = 200.0,
    pacer_offset: int | None = None,
) -> None:
    """Plot all-lead ECG waveform for a single event.

    Args:
        signal: (num_leads, signal_length) array.
        event_id: event identifier for the title.
        output_path: where to save the plot.
        fs: sampling frequency in Hz.
        pacer_offset: pacer offset in samples; drawn as vertical line if > 0.
    """
    pacer_time = pacer_offset / fs if pacer_offset and pacer_offset > 0 else None
    lead_names = list(ALL_LEADS[: signal.shape[0]])
    plot_ecg_waveform(
        signal,
        lead_names=lead_names,
        title=f"ECG — Event {event_id}",
        path=str(output_path),
        fs=fs,
        pacer_time=pacer_time,
    )


def plot_event_vitals(event_dict: dict, output_path: Path) -> None:
    """Plot vitals history subplots for a single event.

    Args:
        event_dict: dict with ``vitals``, ``vitals_history``, ``vitals_thresholds``.
        output_path: where to save the plot.
    """
    history = event_dict.get("vitals_history", {})
    vitals = event_dict.get("vitals", {})
    thresholds = event_dict.get("vitals_thresholds", {})

    fig, axes = plt.subplots(len(_VITALS_CONFIG), 1, figsize=(10, 3 * len(_VITALS_CONFIG)))
    if len(_VITALS_CONFIG) == 1:
        axes = [axes]

    for ax, (key, label, color) in zip(axes, _VITALS_CONFIG):
        lo, hi = _DEFAULT_THRESH.get(key, (0, 0))
        thresh = thresholds.get(key)
        if thresh:
            lo, hi = thresh["lower"], thresh["upper"]

        hist = history.get(key, [])
        if len(hist) >= 2:
            t0 = hist[0]["timestamp"]
            x_min = [(s["timestamp"] - t0) / 60.0 for s in hist]
            y_val = [s["value"] for s in hist]
            ax.plot(x_min, y_val, color=color, linewidth=1.2)
            ax.plot(x_min[-1], y_val[-1], "o", color=color, markersize=7, zorder=5)

            # Diastolic overlay on BP subplot
            if key == "Systolic":
                dia_hist = history.get("Diastolic", [])
                if dia_hist:
                    dx = [(s["timestamp"] - t0) / 60.0 for s in dia_hist]
                    dy = [s["value"] for s in dia_hist]
                    ax.plot(dx, dy, color="tab:olive", linewidth=1.0, alpha=0.6, linestyle="--", label="Diastolic")
                    ax.legend(fontsize=7)

            ax.set_xlabel("Time (min)")
        else:
            # Single dot fallback
            val = vitals.get(key)
            if val is not None:
                ax.plot(0, val, "o", color=color, markersize=8)
                ax.set_xlabel("Event")
            else:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")

        ax.axhline(lo, color="gray", linestyle="--", alpha=0.5, label=f"Low ({lo})")
        ax.axhline(hi, color="gray", linestyle="--", alpha=0.5, label=f"High ({hi})")
        ax.set_ylabel("BP (mmHg)" if key == "Systolic" else label)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Vitals — Event {event_dict.get('event_id', '?')}", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()


def plot_event_mews_history(
    mews_history: list[dict],
    event_id: str,
    output_path: Path,
) -> None:
    """Line plot of MEWS total score over time with risk-band shading.

    Args:
        mews_history: output of ``compute_mews_history()``.
        event_id: event identifier for the title.
        output_path: where to save the plot.
    """
    if not mews_history:
        return

    t0 = mews_history[0]["timestamp"]
    x_min = [(h["timestamp"] - t0) / 60.0 for h in mews_history]
    y_score = [h["mews"].total_score for h in mews_history]

    fig, ax = plt.subplots(figsize=(8, 4))

    # Risk-band background shading
    y_max = max(max(y_score) + 2, 8)
    ax.axhspan(0, 2, color="green", alpha=0.08)
    ax.axhspan(2, 4, color="gold", alpha=0.08)
    ax.axhspan(4, 6, color="orange", alpha=0.08)
    ax.axhspan(6, y_max, color="red", alpha=0.08)

    ax.plot(x_min, y_score, "o-", color="black", linewidth=1.5, markersize=5)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("MEWS Score")
    ax.set_title(f"MEWS History — Event {event_id}")
    ax.set_ylim(0, y_max)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# File-level plot functions (kept for backward compat, no longer called from
# generate_plots)
# ---------------------------------------------------------------------------

def plot_vital_trends(events: list[dict], output_path: Path) -> None:
    """Plot vital sign trends across events as subplots.

    When events include ``vitals_history`` and ``vitals_thresholds``, the plot
    renders a rich time-series with per-event coloured segments, threshold
    lines from the data, and vertical separators.  Falls back to one-dot-per-
    event when no history is available.

    Args:
        events: list of EventResult-like dicts with vitals, event_id, and
            optionally vitals_history / vitals_thresholds.
        output_path: where to save the plot.
    """
    vitals_config = _VITALS_CONFIG

    # Detect whether we have history data
    has_history = any(e.get("vitals_history") for e in events)

    event_ids = [e["event_id"] for e in events]
    event_colors = plt.cm.tab10(np.linspace(0, 1, max(len(events), 1)))

    fig, axes = plt.subplots(len(vitals_config), 1, figsize=(10, 3 * len(vitals_config)), sharex=has_history)
    if len(vitals_config) == 1:
        axes = [axes]

    for ax, (key, label, base_color) in zip(axes, vitals_config):
        # Resolve thresholds: prefer data-provided, fall back to defaults
        lo, hi = _DEFAULT_THRESH.get(key, (0, 0))
        for e in events:
            thresh = e.get("vitals_thresholds", {}).get(key)
            if thresh:
                lo, hi = thresh["lower"], thresh["upper"]
                break

        if has_history:
            # --- Rich time-series mode ---
            # Collect global first timestamp for relative minutes
            all_ts: list[float] = []
            for e in events:
                for s in e.get("vitals_history", {}).get(key, []):
                    all_ts.append(s["timestamp"])
            if not all_ts:
                # No history for this vital — fall back to dots
                _plot_dots(ax, events, key, base_color, lo, hi, event_ids)
            else:
                t0 = min(all_ts)
                boundary_minutes: list[float] = []

                for idx, e in enumerate(events):
                    hist = e.get("vitals_history", {}).get(key, [])
                    if not hist:
                        continue
                    x_min = [(s["timestamp"] - t0) / 60.0 for s in hist]
                    y_val = [s["value"] for s in hist]
                    c = event_colors[idx % len(event_colors)]
                    ax.plot(x_min, y_val, color=c, linewidth=1.2, alpha=0.8)

                    # Mark current value with a larger dot
                    cur_val = e["vitals"].get(key)
                    if cur_val is not None and x_min:
                        ax.plot(x_min[-1], cur_val, "o", color=c, markersize=7, zorder=5)

                    # Track boundary for vertical separator
                    if x_min:
                        boundary_minutes.append(x_min[-1])

                # Diastolic overlay on BP subplot
                if key == "Systolic":
                    for idx, e in enumerate(events):
                        dia_hist = e.get("vitals_history", {}).get("Diastolic", [])
                        if not dia_hist:
                            continue
                        dx = [(s["timestamp"] - t0) / 60.0 for s in dia_hist]
                        dy = [s["value"] for s in dia_hist]
                        c = event_colors[idx % len(event_colors)]
                        ax.plot(dx, dy, color=c, linewidth=1.0, alpha=0.5, linestyle="--")

                # Vertical separators between events
                for bm in boundary_minutes[:-1]:
                    ax.axvline(bm, color="gray", linestyle=":", alpha=0.4)

                ax.set_xlabel("Time (min)")

                # Threshold lines
                ax.axhline(lo, color="gray", linestyle="--", alpha=0.5, label=f"Low ({lo})")
                ax.axhline(hi, color="gray", linestyle="--", alpha=0.5, label=f"High ({hi})")
        else:
            # --- Simple one-dot-per-event mode ---
            _plot_dots(ax, events, key, base_color, lo, hi, event_ids)

        ax.set_ylabel("BP (mmHg)" if key == "Systolic" else label)
        ax.grid(True, alpha=0.3)

    if not has_history:
        x = range(len(events))
        axes[-1].set_xlabel("Event")
        axes[-1].set_xticks(list(x))
        axes[-1].set_xticklabels(event_ids, rotation=45, ha="right", fontsize=7)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()


def _plot_dots(
    ax, events: list[dict], key: str, color: str,
    lo: float, hi: float, event_ids: list[str],
) -> None:
    """Fallback: plot one dot per event for a vital."""
    x = range(len(events))
    values = [e["vitals"].get(key) for e in events]
    valid = [(i, v) for i, v in zip(x, values) if v is not None]
    if valid:
        xi, yi = zip(*valid)
        ax.plot(xi, yi, marker="o", color=color, linewidth=1.5)
        ax.axhline(lo, color="gray", linestyle="--", alpha=0.5, label=f"Low ({lo})")
        ax.axhline(hi, color="gray", linestyle="--", alpha=0.5, label=f"High ({hi})")

        if key == "Systolic":
            dia_values = [e["vitals"].get("Diastolic") for e in events]
            dia_valid = [(i, v) for i, v in zip(x, dia_values) if v is not None]
            if dia_valid:
                dxi, dyi = zip(*dia_valid)
                ax.plot(dxi, dyi, marker="s", color="tab:olive", linewidth=1.5, label="Diastolic")
    else:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")


def plot_mews_trend(
    mews_scores: list[int],
    event_ids: list[str],
    output_path: Path,
) -> None:
    """Bar chart of MEWS total scores per event, color-coded by risk level."""
    colors = []
    for score in mews_scores:
        if score <= 2:
            colors.append("green")
        elif score <= 4:
            colors.append("gold")
        elif score <= 6:
            colors.append("orange")
        else:
            colors.append("red")

    fig, ax = plt.subplots(figsize=(max(6, len(mews_scores) * 0.8), 4))
    x = range(len(mews_scores))
    ax.bar(x, mews_scores, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(event_ids, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("MEWS Score")
    ax.set_title("MEWS Trend")
    ax.set_ylim(0, max(mews_scores) + 2 if mews_scores else 5)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main entry point — per-event plots
# ---------------------------------------------------------------------------

def generate_plots(file_result: FileResult, plot_dir: Path) -> dict[str, dict[str, Path]]:
    """Generate per-event plots for a file result.

    Args:
        file_result: completed FileResult with events and clinical_summary.
        plot_dir: directory to write plot images.

    Returns:
        Map of event_id -> {"ecg": Path, "vitals": Path, "mews": Path}.
        Only keys whose plots were actually generated are included.
    """
    plot_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{file_result.patient_id}-{file_result.alarm_id}"
    event_plots: dict[str, dict[str, Path]] = {}

    for ev in file_result.events:
        eid = ev.event_id
        plots: dict[str, Path] = {}

        # ECG — all leads
        if ev.ecg_signal is not None:
            p = plot_dir / f"{prefix}_ecg_{eid}.png"
            plot_event_ecg(
                ev.ecg_signal, eid, p,
                pacer_offset=ev.pacer_offset if ev.pacer_type else None,
            )
            plots["ecg"] = p

        # Vitals history
        ev_dict = {
            "event_id": eid,
            "vitals": ev.vitals,
            "vitals_history": ev.vitals_history,
            "vitals_thresholds": ev.vitals_thresholds,
        }
        p = plot_dir / f"{prefix}_vitals_{eid}.png"
        plot_event_vitals(ev_dict, p)
        plots["vitals"] = p

        # MEWS history
        if ev.vitals_history:
            mews_hist = compute_mews_history(ev.vitals_history)
            if mews_hist:
                p = plot_dir / f"{prefix}_mews_{eid}.png"
                plot_event_mews_history(mews_hist, eid, p)
                plots["mews"] = p

        if plots:
            event_plots[eid] = plots

    return event_plots
