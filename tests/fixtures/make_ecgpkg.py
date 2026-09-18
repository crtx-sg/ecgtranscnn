"""Build a small contract-conformant ``ecgpkg`` v1 package for tests.

Mimics ecg_sigma output: attribute ``/metadata`` with UTF-8 bytes strings,
per-lead ``source``/``method``/``notes``/``units`` attributes, JSON extras,
``uuid`` datasets, 2400-sample multi-event records (all-real INCART-like and
MIT-BIH-like ``0100001`` records whose limb leads are fabricated from ECG2)
and 2000-sample single-event PTB-XL-like records, several subjects per split,
and excluded rows.  Signals come from the simulator, band-passed like the
package.

    python -m tests.fixtures.make_ecgpkg /tmp/ecg_pkg_fixture
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Callable

import h5py
import numpy as np

from ecg_transcovnet.augment import fabricate_from_ecg2, package_prefilter
from ecg_transcovnet.simulator.conditions import Condition
from ecg_transcovnet.simulator.ecg_simulator import ECGSimulator

LEADS = ["ECG1", "ECG2", "ECG3", "aVR", "aVL", "aVF", "vVX"]
CLASSES = ["NORMAL_SINUS", "ATRIAL_FIBRILLATION", "PVC", "LBBB"]
EXCLUDED_CONDITIONS = ["OTHER", "SVT"]
MANIFEST_COLUMNS = (
    "event_uid", "split", "exclude_reason", "label", "label_idx", "condition", "label_method",
    "label_purity", "dataset", "subject_id", "record_id", "h5_relpath", "event_key", "n_samples",
    "fs", "hr_bpm", "quality", "real_lead_mask", "vvx_lead", "source_sample",
)
THRESHOLDS = {"train": {"subjects": 2, "events": 4},
              "val": {"subjects": 1, "events": 2},
              "test": {"subjects": 1, "events": 2}}
_UUID_NS = uuid.UUID("5d0b6c1e-5a52-4c4e-9f3a-6a1f2f0c7e11")

# Multi-event 2400-sample records per split: (dataset, real_lead_mask)
_MULTI_RECORDS = {
    "train": [("incart", "1111111"), ("mitbih", "0100001"), ("incart", "1111111"), ("mitbih", "0100001")],
    "val": [("incart", "1111111"), ("mitbih", "0100001")],
    "test": [("incart", "1111111"), ("mitbih", "0100001")],
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _signal(sim: ECGSimulator, condition: str, n_samples: int, mask: str) -> np.ndarray:
    enum = Condition[condition] if condition in Condition.__members__ else Condition.NORMAL_SINUS
    ecg = sim.generate_ecg(enum, noise_level="clean")
    x = package_prefilter(np.stack([ecg[l] for l in LEADS]).astype(np.float64)[:, :n_samples], 200.0)
    if mask != "1111111":
        x = fabricate_from_ecg2(x, LEADS, mask)
    return x.astype(np.float32)


def _write_record(root: Path, rows: list[dict], sim: ECGSimulator, rng: np.random.Generator,
                  dataset: str, record: str, subject: str, split: str, n_samples: int,
                  mask: str, conditions: list[str]) -> None:
    rel = f"data/{dataset}/{record}_2025-01.h5"
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    vvx_lead = "V1" if mask[6] == "1" else "synthetic"
    with h5py.File(path, "w") as hf:
        md = hf.create_group("metadata")
        for key, value in {"patient_id": subject.split(":", 1)[1], "record_id": f"{dataset}:{record}",
                           "subject_id": subject, "source_dataset": dataset, "vvx_lead": vvx_lead,
                           "device_info": "RMSAI-SimDevice-v2.0"}.items():
            md.attrs[key] = np.bytes_(value)
        md.attrs["sampling_rate_ecg"] = 200.0
        md.attrs["data_quality_score"] = 0.9
        md.attrs["seconds_before_event"] = n_samples / 400.0
        md.attrs["seconds_after_event"] = n_samples / 400.0

        for j, condition in enumerate(conditions):
            key = f"event_{1001 + j}"
            uid = str(uuid.uuid5(_UUID_NS, f"{rel}/{key}"))
            x = _signal(sim, condition, n_samples, mask)
            hr = float(rng.uniform(55, 110))
            method = "record_level" if dataset == "ptbxl" else "beat_morphology"

            grp = hf.create_group(key)
            ecg = grp.create_group("ecg")
            for i, lead in enumerate(LEADS):
                ds = ecg.create_dataset(lead, data=x[i])
                real = mask[i] == "1"
                ds.attrs["source"] = np.bytes_("real" if real else "synthetic")
                ds.attrs["method"] = np.bytes_(
                    "direct" if real else ("rule_based" if lead in ("ECG1", "vVX") else "einthoven"))
                ds.attrs["notes"] = np.bytes_("fixture")
                ds.attrs["units"] = np.bytes_("mV")
            ecg.create_dataset("extras", data=json.dumps({"pacer_info": 0, "pacer_offset": 0}).encode())
            grp.create_dataset("uuid", data=np.bytes_(uid))
            grp.create_dataset("timestamp", data=1.7e9 + 60.0 * j)
            grp.attrs["condition"] = np.bytes_(condition)
            grp.attrs["heart_rate"] = hr
            grp.attrs["event_timestamp"] = 1.7e9 + 60.0 * j
            grp.attrs["label_method"] = np.bytes_(method)
            grp.attrs["label_purity"] = 1.0
            grp.attrs["source_sample"] = 1000 * j

            included = condition in CLASSES
            rows.append({
                "event_uid": uid,
                "split": split if included else "excluded",
                "exclude_reason": "" if included else "class_not_in_head",
                "label": condition if included else "",
                "label_idx": CLASSES.index(condition) if included else -1,
                "condition": condition,
                "label_method": method,
                "label_purity": "1.0000",
                "dataset": dataset,
                "subject_id": subject,
                "record_id": f"{dataset}:{record}",
                "h5_relpath": rel,
                "event_key": key,
                "n_samples": n_samples,
                "fs": "200",
                "hr_bpm": f"{hr:.3f}",
                "quality": "0.9000",
                "real_lead_mask": mask,
                "vvx_lead": vvx_lead,
                "source_sample": 1000 * j,
            })


def write_metadata(root: Path, rows: list[dict], version: str = "vtest", seed: int = 0) -> None:
    """(Re)write manifest.csv, splits.json, package.json, DATACARD.md and SHA256SUMS."""
    root = Path(root)
    with open(root / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    splits = {s: sorted({r["subject_id"] for r in rows if r["split"] == s}) for s in ("train", "val", "test")}
    (root / "splits.json").write_text(json.dumps(
        {"seed": seed, "method": "subject_grouped_stratified", "notes": "fixture", "subjects": splits},
        indent=2))
    counts = Counter(r["split"] for r in rows)
    test_subjects = {c: {r["subject_id"] for r in rows if r["label"] == c and r["split"] == "test"}
                     for c in CLASSES}
    meta = {
        "format": "ecgpkg",
        "format_version": 1,
        "package_version": version,
        "created_utc": "2026-09-15T00:00:00Z",
        "classes": CLASSES,
        "low_confidence_eval": [c for c in CLASSES if len(test_subjects[c]) < 3],
        "inclusion_thresholds": THRESHOLDS,
        "leads": LEADS,
        "vvx_lead": "V1",
        "sampling_rate_hz": 200.0,
        "window_samples": [2400, 2000],
        "signal_units": "mV",
        "prefilter": {"bandpass_hz": [0.5, 40.0], "notch_hz": 50.0, "zero_phase": True},
        "split_counts": {s: counts.get(s, 0) for s in ("train", "val", "test", "excluded")},
        "seed": seed,
        "provenance": {"ecg_sigma_git_sha": "fixture", "pipeline_config_sha256": "",
                       "package_config": {}, "numpy": np.__version__, "scipy": "", "h5py": h5py.__version__},
        "manifest_sha256": _sha256(root / "manifest.csv"),
    }
    (root / "package.json").write_text(json.dumps(meta, indent=2))
    (root / "DATACARD.md").write_text(f"# Training package `{version}` — test fixture\n")

    files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                   if p.is_file() and p.name != "SHA256SUMS")
    (root / "SHA256SUMS").write_text("".join(f"{_sha256(root / rel)}  {rel}\n" for rel in files))


def make_package(root: str | Path, events_per_class: int = 2, seed: int = 0,
                 version: str = "vtest") -> Path:
    """Write the fixture package to *root* and return its path."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    sim = ECGSimulator(seed=seed)
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for split, records in _MULTI_RECORDS.items():
        for k, (dataset, mask) in enumerate(records):
            conditions = [c for c in CLASSES for _ in range(events_per_class)] + EXCLUDED_CONDITIONS
            _write_record(root, rows, sim, rng, dataset, f"{split}{k}", f"{dataset}:{split}{k}",
                          split, 2400, mask, conditions)
        for c, condition in enumerate(CLASSES):
            _write_record(root, rows, sim, rng, "ptbxl", f"{split}_{c:02d}", f"ptbxl:{split}_{c:02d}",
                          split, 2000, "1111111", [condition])
    write_metadata(root, rows, version, seed)
    return root


def rewrite_manifest(root: str | Path, mutate: Callable[[list[dict]], None]) -> None:
    """Apply *mutate* to the manifest rows and refresh all derived metadata."""
    root = Path(root)
    with open(root / "manifest.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    mutate(rows)
    meta = json.loads((root / "package.json").read_text())
    write_metadata(root, rows, meta["package_version"], meta["seed"])


if __name__ == "__main__":
    out = make_package(sys.argv[1] if len(sys.argv) > 1 else "ecg_pkg_fixture")
    print(out)
