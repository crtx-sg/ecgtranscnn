#!/usr/bin/env python3
"""Generate a comprehensive demo test package for ECG-TransCovNet.

Creates HDF5 files covering all 16 conditions across all noise levels
(clean, low, medium, high) with boundary heart-rate cases. Designed to
be dropped into a directory watched by the inference processor.

Usage
-----
Terminal 1 – start the processor:
    python scripts/processor.py --watch-dir data/demo --checkpoint models/avblock_fix/best_model.pt

Terminal 2 – generate the demo package:
    python scripts/generate_demo.py --output-dir data/demo --num-files 20 --max-events 10

Press Ctrl+C in Terminal 1 to see the aggregate classification report.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import os
import tempfile

# ── project imports ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecg_transcovnet.simulator.conditions import (
    CONDITION_REGISTRY,
    Condition,
)
from ecg_transcovnet.simulator.ecg_simulator import ECGSimulator
from ecg_transcovnet.simulator.hdf5_writer import HDF5EventWriter

# ── constants ──────────────────────────────────────────────────────
ALL_CONDITIONS: list[Condition] = list(Condition)
NOISE_LEVELS: list[str] = ["clean", "low", "medium", "high"]

CONDITION_GROUPS: dict[str, list[Condition]] = {
    "normal": [
        Condition.NORMAL_SINUS,
        Condition.SINUS_BRADYCARDIA,
        Condition.SINUS_TACHYCARDIA,
    ],
    "supraventricular": [
        Condition.ATRIAL_FIBRILLATION,
        Condition.ATRIAL_FLUTTER,
        Condition.PAC,
        Condition.SVT,
    ],
    "ventricular": [
        Condition.PVC,
        Condition.VENTRICULAR_TACHYCARDIA,
        Condition.VENTRICULAR_FIBRILLATION,
    ],
    "conduction": [
        Condition.LBBB,
        Condition.RBBB,
        Condition.AV_BLOCK_1,
        Condition.AV_BLOCK_2_TYPE1,
        Condition.AV_BLOCK_2_TYPE2,
    ],
    "ischemic": [
        Condition.ST_ELEVATION,
    ],
}


# ── plan dataclass ─────────────────────────────────────────────────
@dataclass
class FilePlan:
    """Describes one HDF5 file to be generated."""

    name: str
    description: str
    events: list[tuple[Condition, float | None, str]]  # (condition, hr|None, noise)

    # filled during generation
    path: str = ""
    actual_events: int = 0


# ── plan builders ──────────────────────────────────────────────────

def _hr_boundary(cond: Condition) -> tuple[float, float]:
    """Return (lo, hi) heart-rate for a condition."""
    cfg = CONDITION_REGISTRY[cond]
    return float(cfg.hr_range[0]), float(cfg.hr_range[1])


def _build_noise_sweep(max_events: int, noise_mode: str = "all") -> list[FilePlan]:
    """One file per noise level, cycling through all 16 conditions."""
    plans: list[FilePlan] = []
    levels = NOISE_LEVELS if noise_mode == "all" else [noise_mode]
    for noise in levels:
        events = [(c, None, noise) for c in ALL_CONDITIONS]
        # repeat to fill max_events
        full = (events * math.ceil(max_events / len(events)))[:max_events]
        plans.append(FilePlan(
            name=f"noise_sweep_{noise}",
            description=f"All 16 conditions at {noise} noise",
            events=full,
        ))
    return plans


def _build_boundary_hr(max_events: int, noise_mode: str = "all") -> list[FilePlan]:
    """Two files: one with min-HR per condition, one with max-HR."""
    noise = "clean" if noise_mode == "all" else noise_mode
    for label, hr_idx in [("min_hr", 0), ("max_hr", 1)]:
        events: list[tuple[Condition, float | None, str]] = []
        for cond in ALL_CONDITIONS:
            lo, hi = _hr_boundary(cond)
            hr = lo if hr_idx == 0 else hi
            events.append((cond, hr, noise))
        full = (events * math.ceil(max_events / len(events)))[:max_events]
        yield FilePlan(
            name=f"boundary_{label}",
            description=f"All conditions at {label.replace('_', ' ')} ({noise} noise)",
            events=full,
        )


def _build_group_files(max_events: int, noise_mode: str = "all") -> list[FilePlan]:
    """One file per condition group at the selected noise level."""
    plans: list[FilePlan] = []
    noise = "medium" if noise_mode == "all" else noise_mode
    for group_name, conditions in CONDITION_GROUPS.items():
        events: list[tuple[Condition, float | None, str]] = []
        for cond in conditions:
            events.append((cond, None, noise))
        full = (events * math.ceil(max_events / len(events)))[:max_events]
        plans.append(FilePlan(
            name=f"group_{group_name}",
            description=f"{group_name.title()} conditions at {noise} noise",
            events=full,
        ))
    return plans


def _build_av_block_stress(max_events: int, noise_mode: str = "all") -> list[FilePlan]:
    """Focused AV-block vs normal-sinus discrimination tests."""
    noise = "clean" if noise_mode == "all" else noise_mode
    events: list[tuple[Condition, float | None, str]] = []
    # Interleave normal sinus and AV blocks at similar HRs
    test_hrs = [55.0, 60.0, 65.0, 72.0, 80.0, 90.0]
    targets = [
        Condition.NORMAL_SINUS,
        Condition.AV_BLOCK_1,
        Condition.AV_BLOCK_2_TYPE1,
        Condition.AV_BLOCK_2_TYPE2,
    ]
    for hr in test_hrs:
        for cond in targets:
            lo, hi = _hr_boundary(cond)
            if lo <= hr <= hi:
                events.append((cond, hr, noise))
    full = (events * math.ceil(max_events / len(events)))[:max_events]
    return [FilePlan(
        name="av_block_stress",
        description=f"AV-block vs normal-sinus at overlapping HRs ({noise} noise)",
        events=full,
    )]


def _build_mixed_realistic(max_events: int, noise_mode: str = "all") -> list[FilePlan]:
    """Simulate realistic monitoring: proportional conditions, mixed noise."""
    rng = np.random.default_rng(999)
    events: list[tuple[Condition, float | None, str]] = []
    for _ in range(max_events):
        cond = rng.choice(ALL_CONDITIONS)
        noise = rng.choice(NOISE_LEVELS) if noise_mode == "all" else noise_mode
        events.append((cond, None, noise))
    label = "mixed noise" if noise_mode == "all" else f"{noise_mode} noise"
    return [FilePlan(
        name="mixed_realistic",
        description=f"Random conditions ({label})",
        events=events,
    )]


# ── master plan ────────────────────────────────────────────────────

def build_plan(num_files: int, max_events: int,
               noise_mode: str = "all") -> list[FilePlan]:
    """Assemble the demo file plan, trimming or padding to num_files."""
    pool: list[FilePlan] = []
    pool.extend(_build_noise_sweep(max_events, noise_mode))       # 4 files (1 if fixed)
    pool.extend(_build_boundary_hr(max_events, noise_mode))       # 2 files
    pool.extend(_build_group_files(max_events, noise_mode))       # 5 files
    pool.extend(_build_av_block_stress(max_events, noise_mode))   # 1 file
    pool.extend(_build_mixed_realistic(max_events, noise_mode))   # 1 file

    if len(pool) >= num_files:
        return pool[:num_files]

    # pad with additional all-condition files
    extra_idx = 0
    while len(pool) < num_files:
        noise = NOISE_LEVELS[extra_idx % len(NOISE_LEVELS)] if noise_mode == "all" else noise_mode
        events = [(c, None, noise) for c in ALL_CONDITIONS]
        full = (events * math.ceil(max_events / len(events)))[:max_events]
        pool.append(FilePlan(
            name=f"extra_{noise}_{extra_idx}",
            description=f"Extra coverage – all conditions at {noise} noise",
            events=full,
        ))
        extra_idx += 1

    return pool[:num_files]


# ── generation ─────────────────────────────────────────────────────

def generate(plan: list[FilePlan], output_dir: Path,
             seed: int, delay: float) -> dict:
    """Generate HDF5 files according to the plan. Returns manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = HDF5EventWriter()

    manifest_files: list[dict] = []
    total_events = 0

    for idx, fp in enumerate(plan):
        sim = ECGSimulator(seed=seed + idx)
        events = []
        for cond, hr, noise in fp.events:
            ev = sim.generate_event(
                condition=cond,
                hr=hr,
                noise_level=noise,
            )
            events.append(ev)

        patient_id = f"DM{idx + 1:04d}"
        filename = f"{patient_id}_{fp.name}.h5"
        filepath = output_dir / filename

        # Write to a temp file then atomically rename so the processor
        # never sees a partially-written or locked file.
        fd, tmp_path = tempfile.mkstemp(suffix=".h5.tmp", dir=str(output_dir))
        os.close(fd)
        writer.write_file(tmp_path, events, patient_id=patient_id)
        os.replace(tmp_path, str(filepath))

        fp.path = str(filepath)
        fp.actual_events = len(events)
        total_events += len(events)

        # bookkeeping for manifest
        condition_counts: dict[str, int] = {}
        noise_counts: dict[str, int] = {}
        for cond, _hr, noise in fp.events:
            condition_counts[cond.value] = condition_counts.get(cond.value, 0) + 1
            noise_counts[noise] = noise_counts.get(noise, 0) + 1

        manifest_files.append({
            "file": filename,
            "description": fp.description,
            "events": fp.actual_events,
            "conditions": condition_counts,
            "noise_levels": noise_counts,
        })

        print(f"  [{idx + 1:>3}/{len(plan)}] {filename:<40s} "
              f"{fp.actual_events:>3} events  –  {fp.description}")

        if delay > 0 and idx < len(plan) - 1:
            time.sleep(delay)

    manifest = {
        "suite": "ecg-transcovnet-demo",
        "seed": seed,
        "total_files": len(plan),
        "total_events": total_events,
        "files": manifest_files,
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


# ── plotting ───────────────────────────────────────────────────────

def save_plots(plan: list[FilePlan], output_dir: Path) -> int:
    """Render PNG visualizations for every event in every generated file.

    Uses the plot_event function from visualize_hdf5.py.
    Returns the total number of PNGs saved.
    """
    matplotlib.use("Agg")

    # import here to avoid circular / heavy imports when not plotting
    from visualize_hdf5 import plot_event  # noqa: E402

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for fp in plan:
        filepath = fp.path
        if not filepath:
            continue
        # Copy to a temp file for plotting so we don't hold a lock on
        # the original that would block the inference processor.
        import shutil
        fd, tmp_path = tempfile.mkstemp(suffix=".h5", dir=str(plots_dir))
        os.close(fd)
        shutil.copy2(filepath, tmp_path)
        try:
            with h5py.File(tmp_path, "r") as hf:
                event_keys = sorted(k for k in hf.keys() if k.startswith("event_"))
                stem = Path(filepath).stem
                for ek in event_keys:
                    png_path = str(plots_dir / f"{stem}_{ek}.png")
                    plot_event(hf, ek, filepath, save_path=png_path, show_ppg_resp=True)
                    count += 1
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return count


# ── CLI ────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a comprehensive demo test package for ECG-TransCovNet.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--output-dir", type=str, default="data/demo",
                   help="Directory for generated HDF5 files (default: data/demo)")
    p.add_argument("--num-files", type=int, default=13,
                   help="Number of HDF5 files to generate (default: 13)")
    p.add_argument("--max-events", type=int, default=16,
                   help="Maximum events per HDF5 file (default: 16)")
    p.add_argument("--delay", type=float, default=1.0,
                   help="Seconds between file drops for real-time sim (default: 1.0)")
    p.add_argument("--seed", type=int, default=42,
                   help="Base random seed (default: 42)")
    p.add_argument("--noise", type=str, default="all",
                   choices=["clean", "low", "medium", "high", "all"],
                   help="Noise level: clean, low, medium, high, or all (default: all)")
    p.add_argument("--save-plots", action="store_true",
                   help="Save PNG visualizations for every event in a plots/ subdirectory")
    p.add_argument("--list", action="store_true",
                   help="List the plan without generating files")
    return p


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)

    noise_mode = args.noise
    plan = build_plan(args.num_files, args.max_events, noise_mode=noise_mode)

    # ── list mode ──────────────────────────────────────────────────
    if args.list:
        total_events = 0
        print(f"\nDemo plan: {len(plan)} files, up to {args.max_events} events each\n")
        print(f"  {'#':>3}  {'File':<40s} {'Events':>6}  Description")
        print(f"  {'─' * 3}  {'─' * 40} {'─' * 6}  {'─' * 40}")
        for i, fp in enumerate(plan):
            n = len(fp.events)
            total_events += n
            print(f"  {i + 1:>3}  {fp.name + '.h5':<40s} {n:>6}  {fp.description}")
        print(f"\n  Total events: {total_events}")

        # condition coverage
        cond_set: set[str] = set()
        noise_set: set[str] = set()
        for fp in plan:
            for cond, _, noise in fp.events:
                cond_set.add(cond.value)
                noise_set.add(noise)
        print(f"  Conditions covered: {len(cond_set)}/16")
        print(f"  Noise levels covered: {', '.join(sorted(noise_set))}")
        return

    # ── generate ───────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"  ECG-TransCovNet Demo Package Generator")
    print(f"{'═' * 70}")
    print(f"  Output dir  : {output_dir}")
    print(f"  Files       : {len(plan)}")
    print(f"  Max events  : {args.max_events}")
    print(f"  Delay       : {args.delay}s between files")
    print(f"  Noise       : {noise_mode}")
    print(f"  Save plots  : {args.save_plots}")
    print(f"  Seed        : {args.seed}")
    print(f"{'─' * 70}\n")

    manifest = generate(plan, output_dir, args.seed, args.delay)

    # ── optional plots ────────────────────────────────────────────
    n_plots = 0
    if args.save_plots:
        print(f"\n  Rendering PNG plots ...")
        n_plots = save_plots(plan, output_dir)

    # ── summary ────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  Generated {manifest['total_files']} files, "
          f"{manifest['total_events']} total events")
    if n_plots:
        print(f"  Plots    : {n_plots} PNGs in {output_dir / 'plots'}")
    print(f"  Manifest : {output_dir / 'manifest.json'}")
    print(f"{'─' * 70}")

    # aggregate condition/noise coverage
    cond_total: dict[str, int] = {}
    noise_total: dict[str, int] = {}
    for finfo in manifest["files"]:
        for c, n in finfo["conditions"].items():
            cond_total[c] = cond_total.get(c, 0) + n
        for nl, n in finfo["noise_levels"].items():
            noise_total[nl] = noise_total.get(nl, 0) + n

    print(f"\n  Condition coverage ({len(cond_total)} conditions):")
    for c in sorted(cond_total, key=lambda x: -cond_total[x]):
        print(f"    {c:<35s} {cond_total[c]:>4} events")

    print(f"\n  Noise level distribution:")
    for nl in NOISE_LEVELS:
        count = noise_total.get(nl, 0)
        print(f"    {nl:<10s} {count:>4} events")

    print(f"\n  To run the demo:\n")
    print(f"    # Terminal 1 – start processor:")
    print(f"    python scripts/processor.py \\")
    print(f"        --watch-dir {output_dir} \\")
    print(f"        --checkpoint models/avblock_fix/best_model.pt \\")
    print(f"        --process-existing\n")
    print(f"    # Terminal 2 – generate data (if using --delay):")
    print(f"    python scripts/generate_demo.py \\")
    print(f"        --output-dir {output_dir} --delay 1.0\n")
    print(f"    # Press Ctrl+C in Terminal 1 for aggregate report.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Interrupted – partial output may be available in the output directory.")
        sys.exit(130)
