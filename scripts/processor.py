#!/usr/bin/env python3
"""ECG-TransCovNet Inference Processor — watches a directory for new HDF5 files,
runs inference, and prints per-event results with aggregate classification metrics.

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
from collections import defaultdict
from pathlib import Path
from queue import Queue, Empty

import h5py
import numpy as np
import pyinotify
import torch
import torch.nn.functional as F

from ecg_transcovnet import (
    ECGTransCovNet,
    NUM_CLASSES,
    CLASS_NAMES,
    CONDITION_TO_IDX,
    SIGNAL_LENGTH,
    ALL_LEADS,
    FILTER_PRESETS,
    PreprocessingPipeline,
)
from ecg_transcovnet.mews import analyze_file, calculate_mews, correlate_ecg_vitals
from ecg_transcovnet.report import EventResult, FileResult, extract_ids, write_report
from ecg_transcovnet.plots import generate_plots
from ecg_transcovnet.simulator.conditions import Condition

# Reverse mapping: condition value → class index
_CONDITION_VAL_TO_IDX = {c.value: CONDITION_TO_IDX[c] for c in Condition}


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
    p.add_argument("--checkpoint", type=str,
                    default="models/noise_robust/best_model.pt",
                    help="Path to model checkpoint (.pt).")
    p.add_argument("--process-existing", action="store_true",
                    help="Process .h5 files already present in watch-dir on startup.")
    p.add_argument("--filter-preset", type=str, default="none",
                    choices=list(FILTER_PRESETS.keys()),
                    help="Preprocessing filter preset.")
    p.add_argument("--plot-dir", type=str, default=None,
                    help="Directory for generated plots. If omitted, no plots are created.")
    return p


# ---------------------------------------------------------------------------
# Model loading (matches evaluate.py pattern)
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device):
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found at {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
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
) -> FileResult | None:
    """Parse one HDF5 file, run inference per event, and print results.

    Returns a FileResult for report/plot generation, or None on failure.
    """
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
        event_keys = sorted(k for k in hf.keys() if k.startswith("event_"))
        if not event_keys:
            print(f"  [!] No events found in {filepath.name}")
            return None

        patient_id, alarm_id = extract_ids(filepath, hf)
        file_result = FileResult(filepath=filepath, patient_id=patient_id, alarm_id=alarm_id)

        print(f"\n\u2500\u2500 {filepath.name} ({len(event_keys)} events) " + "\u2500" * max(0, 60 - len(filepath.name)))

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
            if isinstance(gt_val, bytes):
                gt_val = gt_val.decode("utf-8")
            gt_idx = _CONDITION_VAL_TO_IDX.get(gt_val)
            if gt_idx is None:
                continue
            gt_name = CLASS_NAMES[gt_idx]

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

            lead_arrays = []
            for lead in leads:
                if lead in ecg_grp:
                    lead_arrays.append(ecg_grp[lead][:])
                else:
                    lead_arrays.append(np.zeros(SIGNAL_LENGTH, dtype=np.float32))
            signal = np.stack(lead_arrays, axis=0)  # (num_leads, 2400)

            # Preprocessing (filtering + normalization)
            if pipeline is not None:
                signal = pipeline(signal)

            # --- Inference ---
            x = torch.from_numpy(signal.astype(np.float32)).unsqueeze(0).to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=-1)[0]
            pred_idx = probs.argmax().item()
            pred_prob = probs[pred_idx].item()
            pred_name = CLASS_NAMES[pred_idx]
            match = pred_idx == gt_idx

            if match:
                file_correct += 1
            file_total += 1

            tracker.record(gt_idx, pred_idx)

            # --- Read vitals ---
            vitals: dict[str, float] = {}
            hr_str = sp_str = bp_str = rr_str = "\u2014"
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

            match_char = "T" if match else "F"
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

    def __init__(self) -> None:
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

        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)

        accuracy = (y_true == y_pred).mean()

        print()
        print("\u2550" * 2 + " Aggregate Classification Report " + "\u2550" * 30)
        print(f"  Accuracy: {accuracy:.3f}  ({(y_true == y_pred).sum()}/{len(y_true)})")
        print()
        print(f"  {'Condition':<28s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'N':>5s}")
        print("  " + "\u2500" * 53)

        f1_scores = []
        for idx, name in enumerate(CLASS_NAMES):
            tp = int(((y_true == idx) & (y_pred == idx)).sum())
            fp = int(((y_true != idx) & (y_pred == idx)).sum())
            fn = int(((y_true == idx) & (y_pred != idx)).sum())
            support = int((y_true == idx).sum())

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            if support > 0:
                f1_scores.append(f1)
                print(f"  {name:<28s} {prec:6.3f} {rec:6.3f} {f1:6.3f} {support:5d}")

        macro_f1 = np.mean(f1_scores) if f1_scores else 0.0
        print()
        print(f"  Macro F1: {macro_f1:.3f}")
        print()


# ---------------------------------------------------------------------------
# pyinotify watcher
# ---------------------------------------------------------------------------

class HDF5EventHandler(pyinotify.ProcessEvent):
    """Enqueue new .h5 files when they finish being written."""

    def __init__(self, queue: Queue) -> None:
        super().__init__()
        self._queue = queue

    def process_IN_CLOSE_WRITE(self, event: pyinotify.Event) -> None:
        if event.pathname.endswith(".h5"):
            self._queue.put(Path(event.pathname))

    def process_IN_MOVED_TO(self, event: pyinotify.Event) -> None:
        """Catch files that arrive via atomic rename (e.g. os.replace)."""
        if event.pathname.endswith(".h5"):
            self._queue.put(Path(event.pathname))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()

    watch_dir = Path(args.watch_dir)
    watch_dir.mkdir(parents=True, exist_ok=True)

    plot_dir = Path(args.plot_dir) if args.plot_dir else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, leads = load_model(args.checkpoint, device)

    # Preprocessing pipeline
    pipeline = PreprocessingPipeline(FILTER_PRESETS[args.filter_preset])

    # Banner
    ckpt_short = args.checkpoint if len(args.checkpoint) < 45 else "..." + args.checkpoint[-42:]
    print("\u2554" + "\u2550" * 66 + "\u2557")
    print(f"\u2551  ECG-TransCovNet Inference Processor{' ' * 29}\u2551")
    print(f"\u2551  Watching: {str(watch_dir):<20s}  Model: {ckpt_short:<24s}\u2551")
    print("\u255a" + "\u2550" * 66 + "\u255d")
    print(f"  Device: {device}")

    tracker = MetricsTracker()
    file_queue: Queue[Path] = Queue()

    # Process existing files if requested
    if args.process_existing:
        for p in sorted(watch_dir.glob("*.h5")):
            file_queue.put(p)

    # Set up inotify watcher
    wm = pyinotify.WatchManager()
    handler = HDF5EventHandler(file_queue)
    notifier = pyinotify.Notifier(wm, handler, timeout=500)
    wm.add_watch(str(watch_dir), pyinotify.IN_CLOSE_WRITE | pyinotify.IN_MOVED_TO)

    # Graceful shutdown on Ctrl+C
    shutdown = False

    def on_sigint(signum, frame):
        nonlocal shutdown
        shutdown = True

    signal.signal(signal.SIGINT, on_sigint)

    print(f"\n  Waiting for .h5 files in {watch_dir}/ ... (Ctrl+C to stop)\n")

    try:
        while not shutdown:
            # Check for inotify events (non-blocking, 500ms timeout)
            if notifier.check_events(timeout=500):
                notifier.read_events()
                notifier.process_events()

            # Drain the queue
            while not file_queue.empty():
                try:
                    filepath = file_queue.get_nowait()
                    file_result = process_file(
                        filepath, model, leads, device, tracker, pipeline,
                        keep_signals=plot_dir is not None,
                    )

                    # Generate report and plots
                    if file_result and file_result.events:
                        # Clinical analysis (MEWS, trends, correlations)
                        event_dicts = [
                            {"condition": e.pred_name, "vitals": e.vitals,
                             "vitals_history": e.vitals_history}
                            for e in file_result.events
                        ]
                        file_result.clinical_summary = analyze_file(event_dicts)

                        # Attach per-event MEWS and clinical notes
                        for ev, mews in zip(
                            file_result.events,
                            file_result.clinical_summary.mews_scores,
                        ):
                            ev.mews = mews
                            ev.clinical_notes = correlate_ecg_vitals(
                                ev.pred_name, ev.vitals, mews,
                            )

                        # Generate plots if requested
                        event_plots = None
                        if plot_dir is not None:
                            event_plots = generate_plots(file_result, plot_dir)
                            if event_plots:
                                n = sum(len(v) for v in event_plots.values())
                                print(f"  Plots: {n} saved to {plot_dir}/")

                        # Write markdown report
                        report_path = write_report(
                            file_result, plot_dir=plot_dir, event_plots=event_plots,
                        )
                        print(f"  Report: {report_path}")

                except Empty:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        notifier.stop()
        tracker.print_report()


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Known Limitations / Future Improvements
# ---------------------------------------------------------------------------
#
# - No GPU batching: processes one event at a time. Could batch all events in
#   a file for better throughput on GPU.
# - pyinotify is Linux-only. For macOS support, could fall back to polling
#   (e.g. watchdog library or a simple os.listdir loop).
# - No retry on corrupt HDF5 files. A try/except per file is added but no
#   re-queue or dead-letter handling.
