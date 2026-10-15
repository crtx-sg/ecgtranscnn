#!/usr/bin/env python3
"""ECG-TransCovNet Inference Processor — watches a directory for new HDF5 files,
runs inference, and prints per-event results with aggregate classification metrics.

Works with simulator and ecg_sigma HDF5 files, and with legacy 16-class and
package-trained checkpoints (the class head and filter preset come from the
checkpoint).

Usage:
    python scripts/processor.py --watch-dir data/inference --checkpoint models/noise_robust/best_model.pt
    python scripts/processor.py --watch-dir data/inference --checkpoint models/noise_robust/best_model.pt --process-existing
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from ecg_transcovnet import FILTER_PRESETS, PreprocessingPipeline
from ecg_transcovnet.checkpoint import load_models as load_checkpoint_models
from ecg_transcovnet.classes import NOT_IN_HEAD, ClassSpec
from ecg_transcovnet.constants import SAMPLE_RATE_HZ
from ecg_transcovnet.evaluation import compute_metrics
from ecg_transcovnet.hdf5_io import (event_keys as list_event_keys, missing_leads,
                                     read_ecg_leads, read_metadata_field)
from ecg_transcovnet.mews import analyze_file, correlate_ecg_vitals, assess_event_trends
from ecg_transcovnet.report import EventResult, FileResult, extract_ids, write_report
from ecg_transcovnet.plots import generate_plots

# Ground truths are resolved by enum value (simulator files, e.g. "V") or enum
# name (ecg_sigma files, e.g. "PVC") and mapped by name into the model head.
_NOT_IN_HEAD_SEEN: set[str] = set()


def _resolve_ground_truth(spec: ClassSpec, raw) -> tuple[str, int | None]:
    name, idx = spec.resolve(raw)
    if idx is None:
        if name not in _NOT_IN_HEAD_SEEN:
            _NOT_IN_HEAD_SEEN.add(name)
            print(f"  [!] Ground truth '{name}' is not in the model head — shown as "
                  f"{NOT_IN_HEAD} and excluded from metrics")
        return NOT_IN_HEAD, None
    return name, idx


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Watch a directory for HDF5 files and run ECG inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--watch-dir", type=str, required=True,
                    help="Directory to watch for new .h5 files.")
    p.add_argument("--checkpoint", type=str, nargs="+",
                    default=["models/noise_robust/best_model.pt"],
                    help="Model checkpoint(s) (.pt). Several average their softmax "
                         "outputs as an ensemble; they must share head, leads and preset.")
    p.add_argument("--process-existing", action="store_true",
                    help="Process .h5 files already present in watch-dir on startup.")
    p.add_argument("--filter-preset", type=str, default=None,
                    choices=list(FILTER_PRESETS.keys()),
                    help="Preprocessing filter preset (default: the checkpoint's).")
    p.add_argument("--plot-dir", type=str, default=None,
                    help="Directory for generated plots. If omitted, no plots are created.")
    return p


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_paths: str | list[str], device: torch.device):
    """Return ``(model, leads, class_spec, filter_preset)`` for one or more checkpoints."""
    if isinstance(checkpoint_paths, str):
        checkpoint_paths = [checkpoint_paths]
    try:
        loaded = load_checkpoint_models(checkpoint_paths, device)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    return loaded.model, loaded.leads, loaded.class_spec, loaded.filter_preset


# ---------------------------------------------------------------------------
# HDF5 parsing + inference for a single file
# ---------------------------------------------------------------------------

@torch.no_grad()
def process_file(
    filepath: Path,
    model: torch.nn.Module,
    leads: list[str],
    device: torch.device,
    tracker: MetricsTracker,
    pipeline: PreprocessingPipeline | None = None,
    keep_signals: bool = False,
    class_spec: ClassSpec | None = None,
) -> FileResult | None:
    """Parse one HDF5 file, run inference per event, and print results.

    Returns a FileResult for report/plot generation, or None on failure.
    """
    spec = class_spec or ClassSpec.default()
    max_retries = 5
    hf = None
    for attempt in range(max_retries):
        try:
            hf = h5py.File(filepath, "r", locking=False)
            break
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))
            else:
                print(f"  [!] Could not open {filepath.name} after {max_retries} attempts: {e}")
                return None

    with hf:
        fs = float(read_metadata_field(hf, "sampling_rate_ecg", SAMPLE_RATE_HZ))
        if fs != SAMPLE_RATE_HZ:
            print(f"  [!] Skipping {filepath.name}: ECG sampled at {fs:g} Hz, "
                  f"model expects {SAMPLE_RATE_HZ:g} Hz")
            return None
        event_keys = list_event_keys(hf)
        if not event_keys:
            print(f"  [!] No events found in {filepath.name}")
            return None

        patient_id, alarm_id = extract_ids(filepath, hf)
        file_result = FileResult(filepath=filepath, patient_id=patient_id, alarm_id=alarm_id)

        print(f"\n── {filepath.name} ({len(event_keys)} events) " + "─" * max(0, 60 - len(filepath.name)))

        header = (
            f"  {'Event':<8s}"
            f"{'Ground Truth':<28s}"
            f"{'Predicted':<28s}"
            f"{'Match':>5s}"
            f"  {'HR':>4s}"
            f"  {'SpO2':>5s}"
            f"  {'BP':>10s}"
            f"  {'RR':>3s}"
        )
        print(header)

        file_correct = 0
        file_total = 0

        for event_key in event_keys:
            grp = hf[event_key]
            event_id = event_key.replace("event_", "")

            # --- Read ground truth ---
            gt_val = grp.attrs.get("condition", None)
            if gt_val is None:
                continue
            gt_name, gt_idx = _resolve_ground_truth(spec, gt_val)

            # --- Read ECG leads ---
            if "ecg" not in grp:
                continue
            ecg_grp = grp["ecg"]
            # --- Read pacer info from ECG extras ---
            pacer_type = pacer_rate = pacer_offset = 0
            if "extras" in ecg_grp:
                ecg_ex = json.loads(ecg_grp["extras"][()].decode("utf-8"))
                pi = ecg_ex.get("pacer_info", 0)
                pacer_type = pi & 0xFF
                pacer_rate = (pi >> 8) & 0xFF
                pacer_offset = ecg_ex.get("pacer_offset", 0)

            try:
                signal = read_ecg_leads(ecg_grp, leads)  # (num_leads, T)
            except KeyError as exc:
                print(f"  [!] {event_key}: {exc}")
                continue
            absent = missing_leads(ecg_grp, leads)
            if absent:
                print(f"  [!] {event_key}: leads {', '.join(absent)} missing — zero-filled, "
                      f"prediction unreliable")

            # Preprocessing (filtering + normalization)
            if pipeline is not None:
                signal = pipeline(signal)

            # --- Inference ---
            x = torch.from_numpy(signal.astype(np.float32)).unsqueeze(0).to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=-1)[0]
            pred_idx = probs.argmax().item()
            pred_prob = probs[pred_idx].item()
            pred_name = spec.names[pred_idx]
            match = gt_idx is not None and pred_idx == gt_idx

            if gt_idx is not None:
                if match:
                    file_correct += 1
                file_total += 1
                tracker.record(gt_idx, pred_idx)

            # --- Read vitals ---
            vitals: dict[str, float] = {}
            hr_str = sp_str = bp_str = rr_str = "—"
            if "vitals" in grp:
                vg = grp["vitals"]
                if "HR" in vg:
                    vitals["HR"] = float(vg["HR/value"][()])
                    hr_str = str(int(vitals["HR"]))
                if "SpO2" in vg:
                    vitals["SpO2"] = float(vg["SpO2/value"][()])
                    sp_str = f"{int(vitals['SpO2'])}%"
                if "Systolic" in vg and "Diastolic" in vg:
                    vitals["Systolic"] = float(vg["Systolic/value"][()])
                    vitals["Diastolic"] = float(vg["Diastolic/value"][()])
                    bp_str = f"{int(vitals['Systolic'])}/{int(vitals['Diastolic'])}"
                if "RespRate" in vg:
                    vitals["RespRate"] = float(vg["RespRate/value"][()])
                    rr_str = str(int(vitals["RespRate"]))
                if "Temp" in vg:
                    vitals["Temp"] = float(vg["Temp/value"][()])

            # Parse extras for history and thresholds
            vitals_history: dict = {}
            vitals_thresholds: dict = {}
            if "vitals" in grp:
                vg = grp["vitals"]
                for vname in vg:
                    if "extras" in vg[vname]:
                        ex = json.loads(vg[vname]["extras"][()].decode("utf-8"))
                        if "history" in ex:
                            vitals_history[vname] = ex["history"]
                        upper = ex.get("upper_threshold")
                        lower = ex.get("lower_threshold")
                        if upper is not None and lower is not None:
                            vitals_thresholds[vname] = {"upper": upper, "lower": lower}

            match_char = "—" if gt_idx is None else ("T" if match else "F")
            print(
                f"  {event_id:<8s}"
                f"{gt_name:<28s}"
                f"{pred_name:<28s}"
                f"{match_char:>5s}"
                f"  {hr_str:>4s}"
                f"  {sp_str:>5s}"
                f"  {bp_str:>10s}"
                f"  {rr_str:>3s}"
            )

            # --- Build EventResult ---
            ev_result = EventResult(
                event_id=event_id,
                gt_name=gt_name,
                pred_name=pred_name,
                pred_prob=pred_prob,
                match=match,
                vitals=vitals,
                pacer_type=pacer_type,
                pacer_rate=pacer_rate,
                pacer_offset=pacer_offset,
                ecg_signal=signal.copy() if keep_signals else None,
                vitals_history=vitals_history,
                vitals_thresholds=vitals_thresholds,
                missing_leads=absent,
            )
            file_result.events.append(ev_result)

        if file_total > 0:
            pct = 100.0 * file_correct / file_total
            print(f"  File accuracy: {file_correct}/{file_total} ({pct:.1f}%)")

    return file_result


# ---------------------------------------------------------------------------
# Metrics tracker
# ---------------------------------------------------------------------------

class MetricsTracker:
    """Accumulates ground truth / prediction pairs for aggregate reporting."""

    def __init__(self, class_names: list[str] | tuple[str, ...] | None = None) -> None:
        self.class_names = list(class_names) if class_names is not None else list(ClassSpec.default().names)
        self.y_true: list[int] = []
        self.y_pred: list[int] = []

    def record(self, true_idx: int, pred_idx: int) -> None:
        self.y_true.append(true_idx)
        self.y_pred.append(pred_idx)

    @property
    def total(self) -> int:
        return len(self.y_true)

    def print_report(self) -> None:
        if not self.y_true:
            print("\nNo events processed.")
            return

        m = compute_metrics(self.y_true, self.y_pred, self.class_names)

        print()
        print("═" * 2 + " Aggregate Classification Report " + "═" * 30)
        correct = sum(t == p for t, p in zip(self.y_true, self.y_pred))
        print(f"  Accuracy: {m['accuracy']:.3f}  ({correct}/{m['n']})")
        print()
        print(f"  {'Condition':<28s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'N':>5s}")
        print("  " + "─" * 53)
        for name in m["present_classes"]:
            c = m["per_class"][name]
            print(f"  {name:<28s} {c['precision']:6.3f} {c['recall']:6.3f} {c['f1']:6.3f} {c['support']:5d}")

        print()
        print(f"  Macro F1: {m['macro']['f1']:.3f}")
        print()


# ---------------------------------------------------------------------------
# Directory watcher (polling, stdlib only)
# ---------------------------------------------------------------------------

def poll_ready_files(watch_dir: Path, seen: set[Path], pending: dict[Path, tuple]) -> list[Path]:
    """New ``.h5`` files whose size and mtime did not change since the previous poll.

    shortcut: a writer that pauses longer than one poll interval mid-file can be read
    early; write to a temp name and rename into watch-dir if that ever happens.
    """
    ready = []
    for path in sorted(watch_dir.glob("*.h5")):
        if path in seen:
            continue
        try:
            st = path.stat()
        except FileNotFoundError:
            continue
        stamp = (st.st_size, st.st_mtime_ns)
        if pending.get(path) == stamp:
            pending.pop(path)
            seen.add(path)
            ready.append(path)
        else:
            pending[path] = stamp
    return ready


def handle_file(filepath: Path, model, leads, device, tracker, pipeline, class_spec,
                plot_dir: Path | None) -> None:
    """Classify one file, then attach clinical analysis, plots and the report."""
    file_result = process_file(
        filepath, model, leads, device, tracker, pipeline,
        keep_signals=plot_dir is not None, class_spec=class_spec,
    )
    if not (file_result and file_result.events):
        return

    # Clinical analysis (MEWS, trends, correlations)
    event_dicts = [
        {"condition": e.pred_name, "vitals": e.vitals, "vitals_history": e.vitals_history}
        for e in file_result.events
    ]
    file_result.clinical_summary = analyze_file(event_dicts)
    for ev, mews in zip(file_result.events, file_result.clinical_summary.mews_scores):
        ev.mews = mews
        ev.vitals_trends = assess_event_trends(ev.vitals_history)
        ev.clinical_notes = correlate_ecg_vitals(ev.pred_name, ev.vitals, mews)

    event_plots = None
    if plot_dir is not None:
        event_plots = generate_plots(file_result, plot_dir)
        if event_plots:
            n = sum(len(v) for v in event_plots.values())
            print(f"  Plots: {n} saved to {plot_dir}/")

    report_path = write_report(file_result, plot_dir=plot_dir, event_plots=event_plots)
    print(f"  Report: {report_path}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

POLL_SECONDS = 0.5


def main() -> None:
    args = build_parser().parse_args()

    watch_dir = Path(args.watch_dir)
    watch_dir.mkdir(parents=True, exist_ok=True)

    plot_dir = Path(args.plot_dir) if args.plot_dir else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, leads, class_spec, checkpoint_preset = load_model(args.checkpoint, device)

    # Preprocessing pipeline (checkpoint's preset unless overridden)
    filter_preset = args.filter_preset or checkpoint_preset
    pipeline = PreprocessingPipeline(FILTER_PRESETS[filter_preset])

    # Banner
    ckpt_label = (args.checkpoint[0] if len(args.checkpoint) == 1
                  else f"{len(args.checkpoint)}-model ensemble ({args.checkpoint[0]}, ...)")
    ckpt_short = ckpt_label if len(ckpt_label) < 45 else "..." + ckpt_label[-42:]
    print("╔" + "═" * 66 + "╗")
    print(f"║  ECG-TransCovNet Inference Processor{' ' * 29}║")
    print(f"║  Watching: {str(watch_dir):<20s}  Model: {ckpt_short:<24s}║")
    print("╚" + "═" * 66 + "╝")
    print(f"  Device: {device}")
    print(f"  Head: {len(class_spec)} classes · leads {len(leads)} · filter preset: {filter_preset}")

    tracker = MetricsTracker(class_spec.names)
    # Files already present are skipped unless --process-existing.
    seen: set[Path] = set() if args.process_existing else set(watch_dir.glob("*.h5"))
    pending: dict[Path, tuple] = {}

    shutdown = False

    def on_sigint(signum, frame):
        nonlocal shutdown
        shutdown = True

    signal.signal(signal.SIGINT, on_sigint)

    print(f"\n  Waiting for .h5 files in {watch_dir}/ ... (Ctrl+C to stop)\n")

    try:
        while not shutdown:
            for filepath in poll_ready_files(watch_dir, seen, pending):
                if shutdown:
                    break
                try:
                    handle_file(filepath, model, leads, device, tracker, pipeline,
                                class_spec, plot_dir)
                except Exception as exc:  # one bad file must not stop the watcher
                    print(f"  [!] Failed on {filepath.name}: {type(exc).__name__}: {exc}")
            time.sleep(POLL_SECONDS)
    finally:
        tracker.print_report()


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Known Limitations / Future Improvements
# ---------------------------------------------------------------------------
#
# - No GPU batching: processes one event at a time. Could batch all events in
#   a file for better throughput on GPU.
# - A file that fails is logged and skipped; there is no re-queue or
#   dead-letter handling. A file rewritten in place under the same name is
#   not processed again.
