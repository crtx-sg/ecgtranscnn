#!/usr/bin/env python3
"""Generate synthetic ECG HDF5 files and drop them into a watched directory.

This script wraps the existing ECGSimulator + HDF5EventWriter to produce files
that the inference processor can pick up.

Usage:
    # Basic generation
    python scripts/generate_inference_data.py --num-files 3 --events-per-file 5 --output-dir data/inference

    # Specific conditions (uniform weight)
    python scripts/generate_inference_data.py --conditions ATRIAL_FIBRILLATION,NORMAL_SINUS

    # Weighted conditions (3:1 ratio of AFib to Normal)
    python scripts/generate_inference_data.py --conditions ATRIAL_FIBRILLATION:3,NORMAL_SINUS:1

    # Single condition (all events are AFib)
    python scripts/generate_inference_data.py --conditions ATRIAL_FIBRILLATION

    # Noise presets
    python scripts/generate_inference_data.py --noise-level high
    python scripts/generate_inference_data.py --noise-level mixed   # random per event

    # Custom noise parameters (override preset)
    python scripts/generate_inference_data.py --noise-level medium --gaussian-std 0.3 --emg-prob 0.8

    # Simulate real-time arrival
    python scripts/generate_inference_data.py --num-files 5 --delay 2
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

from ecg_transcovnet.simulator import ECGSimulator, HDF5EventWriter
from ecg_transcovnet.simulator.conditions import Condition
from ecg_transcovnet.simulator.noise import NoiseConfig, NOISE_PRESETS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic ECG HDF5 files for inference testing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--num-files", type=int, default=3,
        help="Number of HDF5 files to generate (default 3).",
    )
    parser.add_argument(
        "--events-per-file", type=int, default=5,
        help="Number of alarm events per file (default 5).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/inference",
        help="Directory to write HDF5 files into (default data/inference).",
    )
    parser.add_argument(
        "--delay", type=float, default=0.0,
        help="Seconds to wait between file drops (default 0, no delay).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    # --- Condition selection ---
    cond_group = parser.add_argument_group("condition selection")
    cond_group.add_argument(
        "--conditions", type=str, default="random",
        help=(
            "Condition specification. Accepts:\n"
            "  'random'   — uniform over all 16 conditions (default)\n"
            "  'balanced' — equal weight to all 16 conditions\n"
            "  Comma-separated names — uniform among listed conditions:\n"
            "      ATRIAL_FIBRILLATION,NORMAL_SINUS\n"
            "  Weighted — name:weight pairs:\n"
            "      ATRIAL_FIBRILLATION:3,NORMAL_SINUS:1,PVC:2"
        ),
    )
    cond_group.add_argument(
        "--list-conditions", action="store_true",
        help="Print all valid condition names and exit.",
    )

    # --- Noise control ---
    noise_group = parser.add_argument_group("noise control")
    noise_group.add_argument(
        "--noise-level", type=str, default="medium",
        choices=["clean", "low", "medium", "high", "mixed"],
        help=(
            "Noise level preset (default medium). "
            "'mixed' randomises per event from clean/low/medium/high."
        ),
    )
    noise_group.add_argument(
        "--baseline-wander", type=float, default=None, metavar="AMP",
        help="Override baseline wander amplitude (mV). E.g. 0.15",
    )
    noise_group.add_argument(
        "--gaussian-std", type=float, default=None, metavar="STD",
        help="Override additive Gaussian noise std. E.g. 0.20",
    )
    noise_group.add_argument(
        "--emg-prob", type=float, default=None, metavar="P",
        help="Override EMG artifact probability [0-1]. E.g. 0.50",
    )
    noise_group.add_argument(
        "--motion-prob", type=float, default=None, metavar="P",
        help="Override motion artifact probability [0-1]. E.g. 0.30",
    )
    noise_group.add_argument(
        "--powerline-prob", type=float, default=None, metavar="P",
        help="Override powerline interference probability [0-1]. E.g. 0.30",
    )
    noise_group.add_argument(
        "--electrode-prob", type=float, default=None, metavar="P",
        help="Override poor electrode contact probability [0-1]. E.g. 0.20",
    )

    return parser


def resolve_proportions(conditions_str: str) -> dict[Condition, float] | None:
    """Parse the --conditions flag into a proportions dict.

    Supports:
      - "random" → None (uniform)
      - "balanced" → equal weight to all conditions
      - "AFIB,NORMAL_SINUS" → uniform among listed
      - "AFIB:3,NORMAL_SINUS:1" → weighted
    """
    if conditions_str == "random":
        return None
    if conditions_str == "balanced":
        return {c: 1.0 / len(Condition) for c in Condition}

    parts = [p.strip() for p in conditions_str.split(",") if p.strip()]
    proportions: dict[Condition, float] = {}
    for part in parts:
        if ":" in part:
            name, weight_str = part.rsplit(":", 1)
            name = name.strip()
            try:
                weight = float(weight_str)
            except ValueError:
                raise SystemExit(f"Invalid weight '{weight_str}' in '{part}'")
        else:
            name = part
            weight = 1.0

        try:
            cond = Condition[name]
        except KeyError:
            valid = ", ".join(c.name for c in Condition)
            raise SystemExit(f"Unknown condition '{name}'. Valid:\n  {valid}")
        proportions[cond] = weight

    return proportions


_MIXED_PRESETS = ["clean", "low", "medium", "high"]


def resolve_noise(args: argparse.Namespace, rng) -> tuple[str, NoiseConfig | None]:
    """Resolve noise settings from CLI args.

    Returns (noise_level_str, custom_config_or_None).
    When custom overrides are specified, returns a NoiseConfig.
    When --noise-level=mixed, returns ("mixed", None) — caller handles per-event.
    """
    has_overrides = any(
        getattr(args, attr) is not None
        for attr in [
            "baseline_wander", "gaussian_std", "emg_prob",
            "motion_prob", "powerline_prob", "electrode_prob",
        ]
    )

    if not has_overrides:
        return args.noise_level, None

    if args.noise_level == "mixed":
        raise SystemExit(
            "Cannot combine --noise-level mixed with custom noise overrides. "
            "Use a fixed preset (clean/low/medium/high) as the base instead."
        )

    # Start from the preset, then override individual fields
    base = NOISE_PRESETS[args.noise_level]
    config = NoiseConfig(
        baseline_wander_amp=(
            args.baseline_wander if args.baseline_wander is not None
            else base.baseline_wander_amp
        ),
        gaussian_std=(
            args.gaussian_std if args.gaussian_std is not None
            else base.gaussian_std
        ),
        emg_probability=(
            args.emg_prob if args.emg_prob is not None
            else base.emg_probability
        ),
        motion_probability=(
            args.motion_prob if args.motion_prob is not None
            else base.motion_probability
        ),
        powerline_probability=(
            args.powerline_prob if args.powerline_prob is not None
            else base.powerline_probability
        ),
        electrode_probability=(
            args.electrode_prob if args.electrode_prob is not None
            else base.electrode_probability
        ),
    )
    return args.noise_level, config


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_conditions:
        print("Valid condition names:")
        for c in Condition:
            print(f"  {c.name:<28s} ({c.value})")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sim = ECGSimulator(seed=args.seed)
    writer = HDF5EventWriter()
    proportions = resolve_proportions(args.conditions)
    noise_level, custom_noise = resolve_noise(args, sim._rng)

    # Print summary
    if proportions:
        total_w = sum(proportions.values())
        cond_desc = ", ".join(
            f"{c.name}({w / total_w:.0%})" for c, w in proportions.items()
        )
    else:
        cond_desc = "random (uniform over all 16)"

    print(f"Generating {args.num_files} files × {args.events_per_file} events → {output_dir}/")
    print(f"  Conditions: {cond_desc}")
    if custom_noise:
        print(f"  Noise: {noise_level} base + overrides "
              f"(gauss={custom_noise.gaussian_std}, bw={custom_noise.baseline_wander_amp}, "
              f"emg={custom_noise.emg_probability}, motion={custom_noise.motion_probability}, "
              f"powerline={custom_noise.powerline_probability}, electrode={custom_noise.electrode_probability})")
    else:
        print(f"  Noise: {noise_level}")
    if args.delay > 0:
        print(f"  Delay between files: {args.delay}s")

    for file_idx in range(args.num_files):
        events = []
        for i in range(args.events_per_file):
            if noise_level == "mixed" and custom_noise is None:
                # Pick a random preset per event
                ev_noise = sim._rng.choice(_MIXED_PRESETS)
                event = sim.generate_event(
                    noise_level=ev_noise,
                    condition_proportions=proportions,
                )
            elif custom_noise is not None:
                event = sim.generate_event(
                    noise_level=noise_level,
                    noise_config=custom_noise,
                    condition_proportions=proportions,
                )
            else:
                event = sim.generate_event(
                    noise_level=noise_level,
                    condition_proportions=proportions,
                )
            events.append(event)

        # Build filename: PatientID_YYYY-MM.h5
        patient_id = f"PT{sim._rng.integers(1000, 9999)}"
        date_str = datetime.now().strftime("%Y-%m")
        filename = f"{patient_id}_{date_str}.h5"
        filepath = output_dir / filename

        writer.write_file(str(filepath), events, patient_id=patient_id)

        conditions_summary = ", ".join(e.condition.name for e in events)
        print(f"  [{file_idx + 1}/{args.num_files}] {filename}  [{conditions_summary}]")

        if args.delay > 0 and file_idx < args.num_files - 1:
            time.sleep(args.delay)

    print("Done.")


if __name__ == "__main__":
    main()
