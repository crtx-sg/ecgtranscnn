#!/usr/bin/env python3
"""Generate a comprehensive validation test suite for ECG-TransCovNet.

Creates a structured set of HDF5 test files covering all 16 conditions across
multiple noise levels, boundary heart rates, and mixed-condition scenarios.
Designed to validate the AV-block-fixed model and demonstrate performance.

Usage:
    # Default: 10 files, 10 events each
    python scripts/generate_validation_suite.py --output-dir data/validation_suite

    # Custom size
    python scripts/generate_validation_suite.py --num-files 20 --events-per-file 15

    # Quick smoke test
    python scripts/generate_validation_suite.py --num-files 5 --events-per-file 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

from ecg_transcovnet.simulator import ECGSimulator, HDF5EventWriter
from ecg_transcovnet.simulator.conditions import Condition, CONDITION_REGISTRY


# ── Test suite definition ────────────────────────────────────────────────────

@dataclass
class TestCase:
    """Definition for a single test file."""
    name: str
    description: str
    conditions: list[tuple[Condition, float | None]]  # (condition, optional HR)
    noise_level: str
    events_per_file: int = 0  # 0 = use global default


# Conditions grouped by clinical category
_NORMAL_RHYTHMS = [
    Condition.NORMAL_SINUS,
    Condition.SINUS_BRADYCARDIA,
    Condition.SINUS_TACHYCARDIA,
]
_SUPRAVENTRICULAR = [
    Condition.ATRIAL_FIBRILLATION,
    Condition.ATRIAL_FLUTTER,
    Condition.PAC,
    Condition.SVT,
]
_VENTRICULAR = [
    Condition.PVC,
    Condition.VENTRICULAR_TACHYCARDIA,
    Condition.VENTRICULAR_FIBRILLATION,
]
_BBB = [Condition.LBBB, Condition.RBBB]
_AV_BLOCKS = [
    Condition.AV_BLOCK_1,
    Condition.AV_BLOCK_2_TYPE1,
    Condition.AV_BLOCK_2_TYPE2,
]


def _all_conditions_uniform(noise: str) -> list[tuple[Condition, float | None]]:
    """All 16 conditions at default HR."""
    return [(c, None) for c in Condition]


def _boundary_hr_cases() -> list[tuple[Condition, float | None]]:
    """Conditions at boundary heart rates (min/max of their HR range)."""
    cases = []
    for cond in Condition:
        cfg = CONDITION_REGISTRY[cond]
        hr_lo, hr_hi = cfg.hr_range
        cases.append((cond, hr_lo))       # lowest valid HR
        cases.append((cond, hr_hi))       # highest valid HR
    return cases


def _av_block_focus() -> list[tuple[Condition, float | None]]:
    """AV blocks + normal sinus for confusion testing."""
    cases = []
    # Normal sinus at various HRs for comparison
    for hr in [60, 72, 85, 95]:
        cases.append((Condition.NORMAL_SINUS, float(hr)))
    # AV Block 1 at various HRs
    for hr in [55, 65, 75, 90]:
        cases.append((Condition.AV_BLOCK_1, float(hr)))
    # AV Block 2 types
    for hr in [50, 60, 70]:
        cases.append((Condition.AV_BLOCK_2_TYPE1, float(hr)))
    for hr in [45, 55, 65]:
        cases.append((Condition.AV_BLOCK_2_TYPE2, float(hr)))
    return cases


def build_test_suite(num_files: int, events_per_file: int) -> list[TestCase]:
    """Build the test suite definition based on requested size.

    Generates a structured mix of test categories:
      - Per-noise-level sweeps (clean, low, medium, high)
      - Boundary HR stress tests
      - AV block discrimination focus
      - Mixed-condition realistic scenarios
      - Per-category group tests
    """
    suite: list[TestCase] = []

    # ── Category 1: Noise sweep — all 16 conditions at each noise level ──
    for noise in ["clean", "low", "medium", "high"]:
        suite.append(TestCase(
            name=f"all_conditions_{noise}",
            description=f"All 16 conditions with {noise} noise",
            conditions=_all_conditions_uniform(noise),
            noise_level=noise,
        ))

    # ── Category 2: AV block discrimination (key validation for the fix) ──
    suite.append(TestCase(
        name="av_block_discrimination_clean",
        description="AV blocks vs Normal Sinus — clean (validates PR prolongation fix)",
        conditions=_av_block_focus(),
        noise_level="clean",
    ))
    suite.append(TestCase(
        name="av_block_discrimination_medium",
        description="AV blocks vs Normal Sinus — medium noise",
        conditions=_av_block_focus(),
        noise_level="medium",
    ))

    # ── Category 3: Boundary HR stress test ──
    suite.append(TestCase(
        name="boundary_hr_clean",
        description="All conditions at min/max HR boundaries — clean",
        conditions=_boundary_hr_cases(),
        noise_level="clean",
    ))
    suite.append(TestCase(
        name="boundary_hr_medium",
        description="All conditions at min/max HR boundaries — medium noise",
        conditions=_boundary_hr_cases(),
        noise_level="medium",
    ))

    # ── Category 4: Per-group focused tests ──
    suite.append(TestCase(
        name="supraventricular_clean",
        description="Supraventricular arrhythmias (AFib, AFL, PAC, SVT) — clean",
        conditions=[(c, None) for c in _SUPRAVENTRICULAR for _ in range(3)],
        noise_level="clean",
    ))
    suite.append(TestCase(
        name="ventricular_high_noise",
        description="Ventricular arrhythmias (PVC, VT, VF) — high noise stress",
        conditions=[(c, None) for c in _VENTRICULAR for _ in range(4)],
        noise_level="high",
    ))
    suite.append(TestCase(
        name="bundle_branch_blocks_medium",
        description="LBBB vs RBBB discrimination — medium noise",
        conditions=[(c, None) for c in _BBB for _ in range(6)],
        noise_level="medium",
    ))

    # ── Category 5: Mixed realistic scenarios ──
    suite.append(TestCase(
        name="mixed_realistic_low",
        description="Random mix of all conditions — low noise (realistic ward monitor)",
        conditions=[(c, None) for c in Condition],
        noise_level="low",
    ))
    suite.append(TestCase(
        name="mixed_realistic_high",
        description="Random mix of all conditions — high noise (ambulance/transport)",
        conditions=[(c, None) for c in Condition],
        noise_level="high",
    ))

    # Trim or extend to match requested num_files
    if len(suite) > num_files:
        suite = suite[:num_files]
    else:
        # Add more mixed-condition files to reach the target
        extra_idx = 0
        noise_cycle = ["clean", "low", "medium", "high"]
        while len(suite) < num_files:
            noise = noise_cycle[extra_idx % len(noise_cycle)]
            suite.append(TestCase(
                name=f"mixed_extra_{extra_idx + 1}_{noise}",
                description=f"Additional mixed-condition test — {noise} noise",
                conditions=[(c, None) for c in Condition],
                noise_level=noise,
            ))
            extra_idx += 1

    # Apply default events_per_file to cases that don't override
    for tc in suite:
        if tc.events_per_file == 0:
            # Use the specified count, but cap conditions list
            tc.events_per_file = events_per_file

    return suite


# ── Generation ────────────────────────────────────────────────────────────────

def generate_suite(
    suite: list[TestCase],
    output_dir: Path,
    seed: int,
) -> dict:
    """Generate HDF5 files for the test suite. Returns manifest dict."""
    writer = HDF5EventWriter()
    manifest = {
        "suite_name": "ECG-TransCovNet Validation Suite",
        "seed": seed,
        "total_files": len(suite),
        "total_events": 0,
        "files": [],
    }

    for idx, tc in enumerate(suite):
        sim = ECGSimulator(seed=seed + idx)

        # Generate events: cycle through the condition list to fill events_per_file
        events = []
        for ev_idx in range(tc.events_per_file):
            cond, hr = tc.conditions[ev_idx % len(tc.conditions)]
            event = sim.generate_event(
                condition=cond,
                hr=hr,
                noise_level=tc.noise_level,
            )
            events.append(event)

        # Write HDF5
        patient_id = f"VS{idx + 1:04d}"
        filename = f"{tc.name}.h5"
        filepath = output_dir / filename
        writer.write_file(str(filepath), events, patient_id=patient_id)

        # Build manifest entry
        conditions_summary = {}
        for ev in events:
            name = ev.condition.name
            conditions_summary[name] = conditions_summary.get(name, 0) + 1

        file_info = {
            "filename": filename,
            "description": tc.description,
            "noise_level": tc.noise_level,
            "num_events": len(events),
            "conditions": conditions_summary,
        }
        manifest["files"].append(file_info)
        manifest["total_events"] += len(events)

        # Print progress
        cond_str = ", ".join(f"{k}:{v}" for k, v in sorted(conditions_summary.items()))
        print(
            f"  [{idx + 1:2d}/{len(suite)}] {filename:<45s} "
            f"noise={tc.noise_level:<6s} events={len(events):2d}  [{cond_str}]"
        )

    return manifest


# ── Main ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a comprehensive validation test suite for ECG-TransCovNet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output-dir", type=str, default="data/validation_suite",
        help="Output directory for HDF5 files and manifest.",
    )
    p.add_argument(
        "--num-files", type=int, default=10,
        help="Number of HDF5 test files to generate.",
    )
    p.add_argument(
        "--events-per-file", type=int, default=10,
        help="Default number of events per file.",
    )
    p.add_argument("--seed", type=int, default=12345, help="Base random seed.")
    p.add_argument(
        "--list", action="store_true",
        help="List what would be generated without writing files.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)

    suite = build_test_suite(args.num_files, args.events_per_file)

    if args.list:
        print(f"Validation suite: {len(suite)} files, {args.events_per_file} events/file default\n")
        for i, tc in enumerate(suite):
            n_conds = len(set(c for c, _ in tc.conditions))
            print(f"  {i + 1:2d}. {tc.name:<45s} noise={tc.noise_level:<6s} "
                  f"conditions={n_conds:2d}  | {tc.description}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating validation suite: {len(suite)} files → {output_dir}/")
    print(f"  Events per file: {args.events_per_file}")
    print(f"  Seed: {args.seed}")
    print()

    manifest = generate_suite(suite, output_dir, args.seed)

    # Write manifest JSON
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSuite complete:")
    print(f"  Files:       {manifest['total_files']}")
    print(f"  Total events: {manifest['total_events']}")
    print(f"  Manifest:    {manifest_path}")
    print(f"\nTo evaluate:")
    print(f"  python scripts/run_validation_suite.py --suite-dir {output_dir} "
          f"--checkpoint models/avblock_fix/best_model.pt")


if __name__ == "__main__":
    main()
