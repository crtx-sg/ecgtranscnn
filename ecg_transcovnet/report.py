"""Markdown report generation for ECG inference results."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

from .mews import MEWSResult, ClinicalSummary, compute_mews_history, assess_mews_trend


@dataclass
class EventResult:
    """Inference result for a single event."""
    event_id: str
    gt_name: str
    pred_name: str
    pred_prob: float
    match: bool
    vitals: dict
    mews: MEWSResult | None = None
    clinical_notes: list[str] = field(default_factory=list)
    ecg_signal: np.ndarray | None = None
    vitals_history: dict = field(default_factory=dict)
    vitals_thresholds: dict = field(default_factory=dict)


@dataclass
class FileResult:
    """Inference results for an entire HDF5 file."""
    filepath: Path
    patient_id: str
    alarm_id: str
    events: list[EventResult] = field(default_factory=list)
    clinical_summary: ClinicalSummary | None = None


def extract_ids(filepath: Path, hf: h5py.File) -> tuple[str, str]:
    """Extract patient_id from metadata and alarm_id from filename.

    Returns:
        (patient_id, alarm_id)
    """
    patient_id = "unknown"
    if "metadata" in hf and "patient_id" in hf["metadata"]:
        pid = hf["metadata"]["patient_id"][()]
        if isinstance(pid, bytes):
            pid = pid.decode("utf-8")
        patient_id = str(pid)

    # alarm_id from filename: <patient>_<alarm>.h5
    stem = filepath.stem
    m = re.match(r"^(.+?)_(.+)$", stem)
    alarm_id = m.group(2) if m else stem

    return patient_id, alarm_id


def generate_report(
    file_result: FileResult,
    plot_dir: Path | None = None,
    event_plots: dict[str, dict[str, Path]] | None = None,
) -> str:
    """Build a markdown report string from inference results."""
    fr = file_result
    lines: list[str] = []

    # Header
    lines.append(f"# ECG Inference Report: {fr.patient_id} / {fr.alarm_id}")
    lines.append("")

    # Metadata table
    lines.append("## Metadata")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| File | `{fr.filepath.name}` |")
    lines.append(f"| Patient ID | {fr.patient_id} |")
    lines.append(f"| Alarm ID | {fr.alarm_id} |")
    lines.append(f"| Events | {len(fr.events)} |")
    lines.append(f"| Generated | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |")
    lines.append("")

    # Summary
    correct = sum(1 for e in fr.events if e.match)
    total = len(fr.events)
    accuracy = correct / total * 100 if total > 0 else 0.0
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Accuracy**: {correct}/{total} ({accuracy:.1f}%)")

    if fr.clinical_summary:
        cs = fr.clinical_summary
        lines.append(f"- **Overall MEWS Trend**: {cs.overall_trend}")
        if cs.ecg_vital_correlations:
            lines.append(f"- **Clinical Alerts**: {len(cs.ecg_vital_correlations)}")
    lines.append("")

    # Per-event table
    lines.append("## Event Results")
    lines.append("")
    header = "| Event | Ground Truth | Predicted | Prob | Match | HR | SpO2 | BP | RR | Temp | MEWS |"
    sep = "|-------|-------------|-----------|------|-------|----|------|----|----|----|------|"
    lines.append(header)
    lines.append(sep)

    for ev in fr.events:
        v = ev.vitals
        hr = str(int(v["HR"])) if "HR" in v else "—"
        spo2 = f"{int(v['SpO2'])}%" if "SpO2" in v else "—"
        bp = f"{int(v['Systolic'])}/{int(v['Diastolic'])}" if "Systolic" in v and "Diastolic" in v else "—"
        rr = str(int(v["RespRate"])) if "RespRate" in v else "—"
        temp = f"{v['Temp']:.1f}" if "Temp" in v else "—"
        mews_str = str(ev.mews.total_score) if ev.mews else "—"
        match_str = "Yes" if ev.match else "No"

        lines.append(
            f"| {ev.event_id} | {ev.gt_name} | {ev.pred_name} | {ev.pred_prob:.3f} "
            f"| {match_str} | {hr} | {spo2} | {bp} | {rr} | {temp} | {mews_str} |"
        )
    lines.append("")

    # Per-event analysis
    if fr.clinical_summary:
        lines.append("## Clinical Analysis")
        lines.append("")

        for i, ev in enumerate(fr.events):
            if ev.mews is None and not ev.clinical_notes:
                continue
            lines.append(f"### Event {ev.event_id}")
            lines.append("")

            if ev.mews:
                lines.append(f"**MEWS Total: {ev.mews.total_score} ({ev.mews.risk_level})**")
                lines.append("")
                lines.append("| Component | Value | Score |")
                lines.append("|-----------|-------|-------|")
                for comp in ev.mews.components:
                    lines.append(f"| {comp.name} | {comp.value} | {comp.score} |")
                lines.append("")

                # Per-event MEWS trend via Mann-Kendall
                if ev.vitals_history:
                    mh = compute_mews_history(ev.vitals_history)
                    mk = assess_mews_trend(mh)
                    if mk is not None:
                        if mk.p_value < 0.05:
                            lines.append(f"**MEWS Trend**: {mk.trend} (p={mk.p_value:.3f}, slope={mk.slope:+.1f})")
                        else:
                            lines.append(f"**MEWS Trend**: no significant trend (p={mk.p_value:.2f})")
                        lines.append("")

            # Threshold status
            if ev.vitals_thresholds:
                lines.append("**Threshold Status**")
                lines.append("")
                for vname, thresh in ev.vitals_thresholds.items():
                    val = ev.vitals.get(vname)
                    if val is None:
                        continue
                    lo, hi = thresh["lower"], thresh["upper"]
                    units = {"HR": "bpm", "Pulse": "bpm", "SpO2": "%",
                             "Systolic": "mmHg", "Diastolic": "mmHg",
                             "RespRate": "br/min", "Temp": "\u00b0F"}.get(vname, "")
                    if val > hi:
                        status = f"\u26a0 ABOVE"
                    elif val < lo:
                        status = f"\u26a0 BELOW"
                    else:
                        status = "\u2713 normal"
                    lines.append(f"- {vname}: {val} {units} [{lo}-{hi}] {status}")
                lines.append("")

            # History summary
            if ev.vitals_history:
                lines.append("**History Summary**")
                lines.append("")
                for vname, hist in ev.vitals_history.items():
                    n = len(hist)
                    if n < 2:
                        continue
                    span_sec = hist[-1]["timestamp"] - hist[0]["timestamp"]
                    span_min = int(span_sec / 60)
                    first_val = hist[0]["value"]
                    last_val = hist[-1]["value"]
                    if last_val > first_val:
                        arrow = "\u2191"
                    elif last_val < first_val:
                        arrow = "\u2193"
                    else:
                        arrow = "\u2192"
                    lines.append(f"- {vname}: {n} samples over {span_min} min, trending {arrow}")
                lines.append("")

            if ev.clinical_notes:
                for note in ev.clinical_notes:
                    lines.append(f"- {note}")
                lines.append("")

            # Per-event plots
            if event_plots and ev.event_id in event_plots:
                ep = event_plots[ev.event_id]
                lines.append("#### Plots")
                lines.append("")
                if "ecg" in ep:
                    lines.append(f"![ECG — Event {ev.event_id}]({ep['ecg']})")
                if "vitals" in ep:
                    lines.append(f"![Vitals — Event {ev.event_id}]({ep['vitals']})")
                if "mews" in ep:
                    lines.append(f"![MEWS History — Event {ev.event_id}]({ep['mews']})")
                lines.append("")

        # Trends
        if fr.clinical_summary.trends:
            lines.append("## Vital Sign Trends")
            lines.append("")
            lines.append("| Vital | Direction | Slope | p-value | Sig |")
            lines.append("|-------|-----------|-------|---------|-----|")
            for t in fr.clinical_summary.trends:
                p_str = f"{t.p_value:.3f}" if t.p_value is not None else "—"
                sig_str = "*" if t.significant else "—"
                lines.append(f"| {t.vital_name} | {t.direction} | {t.slope:+.2f} | {p_str} | {sig_str} |")
            lines.append("")

    # Footer
    lines.append("---")
    lines.append(f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by Vios using ECG-TransCovNet DL & ML models*")
    lines.append("")

    return "\n".join(lines)


def write_report(
    file_result: FileResult,
    plot_dir: Path | None = None,
    event_plots: dict[str, dict[str, Path]] | None = None,
) -> Path:
    """Write markdown report alongside the HDF5 file.

    Returns:
        Path to the written report file.
    """
    report_name = f"report-{file_result.patient_id}-{file_result.alarm_id}.md"
    report_path = file_result.filepath.parent / report_name
    content = generate_report(file_result, plot_dir=plot_dir, event_plots=event_plots)
    report_path.write_text(content)
    return report_path
