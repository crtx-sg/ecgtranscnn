"""Markdown report generation for ECG inference results."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

from .mews import MEWSResult, TrendAssessment, ClinicalSummary, compute_mews_history, assess_mews_trend


@dataclass
class EventResult:
    """Inference result for a single event."""
    event_id: str
    gt_name: str
    pred_name: str
    pred_prob: float
    match: bool
    vitals: dict
    pacer_type: int = 0    # 0=None,1=Single,2=Dual,3=Biventricular
    pacer_rate: int = 0
    pacer_offset: int = 0
    mews: MEWSResult | None = None
    clinical_notes: list[str] = field(default_factory=list)
    ecg_signal: np.ndarray | None = None
    vitals_history: dict = field(default_factory=dict)
    vitals_thresholds: dict = field(default_factory=dict)
    vitals_trends: list[TrendAssessment] = field(default_factory=list)


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

    # Per-event analysis
    if fr.clinical_summary:
        lines.append("## Clinical Analysis")
        lines.append("")

        for i, ev in enumerate(fr.events):
            if ev.mews is None and not ev.clinical_notes:
                continue
            lines.append(f"### Event {ev.event_id}")
            lines.append("")

            # --- ECG Plots ---
            ep = event_plots.get(ev.event_id, {}) if event_plots else {}
            lines.append("#### ECG Plots")
            lines.append("")
            if ev.pacer_type > 0:
                _pacer_names = {0: "None", 1: "Single", 2: "Dual", 3: "Biventricular"}
                pname = _pacer_names.get(ev.pacer_type, f"Type {ev.pacer_type}")
                ptime = ev.pacer_offset / 200.0
                lines.append(f"**Pacer**: {pname} chamber @ {ev.pacer_rate} bpm (offset {ptime:.1f}s)")
                lines.append("")
            if "ecg" in ep:
                lines.append(f"![ECG 7-Lead — Event {ev.event_id}]({ep['ecg']})")
                lines.append("")

            # --- ECG Classification ---
            match_str = "Yes" if ev.match else "No"
            lines.append("| Event | Ground Truth | Predicted | Prob | Match |")
            lines.append("|-------|-------------|-----------|------|-------|")
            lines.append(
                f"| {ev.event_id} | {ev.gt_name} | {ev.pred_name} "
                f"| {ev.pred_prob:.3f} | {match_str} |"
            )
            lines.append("")

            # --- Vitals at Event ---
            if ev.mews:
                lines.append(f"#### Vitals at Event — MEWS {ev.mews.total_score} ({ev.mews.risk_level})")
                lines.append("")
                lines.append("| Component | Value | Score |")
                lines.append("|-----------|-------|-------|")
                for comp in ev.mews.components:
                    lines.append(f"| {comp.name} | {comp.value} | {comp.score} |")
                lines.append("")

            # --- Threshold Status ---
            if ev.vitals_thresholds:
                lines.append("#### Threshold Status")
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
                        status = "\u26a0 ABOVE"
                    elif val < lo:
                        status = "\u26a0 BELOW"
                    else:
                        status = "\u2713 normal"
                    lines.append(f"- {vname}: {val} {units} [{lo}-{hi}] {status}")
                lines.append("")

            # --- Vitals Trend Plots ---
            if "vitals" in ep or "mews" in ep:
                lines.append("#### Vitals Trend Plots")
                lines.append("")
                if "vitals" in ep:
                    lines.append(f"![Vitals — Event {ev.event_id}]({ep['vitals']})")
                if "mews" in ep:
                    lines.append(f"![MEWS History — Event {ev.event_id}]({ep['mews']})")
                lines.append("")

            # --- Vital Sign Trends ---
            if ev.vitals_trends:
                lines.append("#### Vital Sign Trends")
                lines.append("")
                lines.append("| Vital | Direction | Slope | p-value | Sig |")
                lines.append("|-------|-----------|-------|---------|-----|")
                for t in ev.vitals_trends:
                    p_str = f"{t.p_value:.3f}" if t.p_value is not None else "—"
                    sig_str = "*" if t.significant else "—"
                    lines.append(f"| {t.vital_name} | {t.direction} | {t.slope:+.2f} | {p_str} | {sig_str} |")
                lines.append("")

            # --- Care Guidance ---
            if ev.clinical_notes:
                lines.append("#### Care Guidance")
                lines.append("")
                for note in ev.clinical_notes:
                    lines.append(f"- {note}")
                lines.append("")

    # Footer
    lines.append("---")
    lines.append(f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by Vios using custom ECG-TransCovNet DL & ML models*")
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
