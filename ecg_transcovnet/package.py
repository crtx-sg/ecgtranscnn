"""Loader and PyTorch dataset for ecg_sigma ``ecgpkg`` v1 training packages.

See ``CONTRACT_ecgpkg_v1.md``.  Metadata is read with ``csv``/``json`` only.
Signals are cached per split in a memory-mapped ``.npy`` file so that no
split is ever held in RAM.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Collection, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from . import augment
from .classes import ClassSpec
from .preprocessing import FILTER_PRESETS, FilterConfig, PreprocessingPipeline

FORMAT = "ecgpkg"
FORMAT_VERSION = 2
SUPPORTED_FORMAT_VERSIONS = (1, 2)
# Fabrication rules our augment.py port reproduces. A package declaring a different
# version means ecg_sigma changed the ECG2 -> other-lead reconstruction and the port
# is stale (see docs/RESPONSE_change_requests.md §5).
FABRICATION_RULES_VERSION = "1"
# v2.1 replaces paced_beats (WFDB beat symbols only, blind to record-level sources
# like PTB-XL) with paced_record.  Prefer the latter where present.
PACED_COLUMNS = ("paced_record", "paced_beats")
SPLITS = ("train", "val", "test")
EXCLUDED = "excluded"
MAX_SAMPLES = 2400
_INT_COLUMNS = ("label_idx", "n_samples", "source_sample")
# v2 appends these; absent in v1.
_OPTIONAL_INT_COLUMNS = ("paced_beats",)
_BOOL_COLUMNS = ("paced_record",)


class PackageFormatError(ValueError):
    """The package does not satisfy the ``ecgpkg`` contract."""


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------------

@dataclass
class Package:
    root: Path
    meta: dict
    spec: ClassSpec
    rows: list[dict]
    splits: dict | None

    @property
    def version(self) -> str:
        return self.meta["package_version"]

    @property
    def manifest_sha256(self) -> str:
        return self.meta["manifest_sha256"]

    @property
    def leads(self) -> list[str]:
        return list(self.meta["leads"])

    @property
    def fs(self) -> float:
        return float(self.meta.get("sampling_rate_hz", 200.0))

    @property
    def format_version(self) -> int:
        return int(self.meta.get("format_version", 1))

    @property
    def low_confidence_eval(self) -> list[str]:
        """Classes ecg_sigma flags as low-confidence (v1: empty)."""
        return list(self.meta.get("low_confidence_eval") or ())

    @property
    def eval_flags(self) -> dict[str, list[str]]:
        """Per-class evaluation flags, e.g. ``no_seven_real_lead_events``."""
        return {k: list(v) for k, v in (self.meta.get("eval_flags") or {}).items()}

    @property
    def class_counts(self) -> dict:
        """Per class x split counts shipped by the package (v2+)."""
        return self.meta.get("class_counts") or {}

    @property
    def label_definitions(self) -> dict:
        return self.meta.get("label_definitions") or {}

    @property
    def reporting(self) -> dict:
        """Metric contract shipped by the package (v2.1+): primary classes and CI rule."""
        return self.meta.get("reporting") or {}

    def classes_with_flag(self, flag: str) -> list[str]:
        """Head classes carrying *flag*, in head order."""
        flags = self.eval_flags
        return [c for c in self.spec.names if flag in flags.get(c, ())]

    @property
    def cv(self) -> dict | None:
        """Grouped cross-validation folds from ``splits.json`` (v2+)."""
        return (self.splits or {}).get("cv")

    def cv_subjects(self, held_out: int) -> tuple[set[str], set[str]]:
        """``(fit_subjects, select_subjects)`` for fold *held_out*."""
        cv = self.cv
        if not cv:
            raise PackageFormatError(f"{self.root}: splits.json has no cv folds")
        folds = cv["folds"]
        k = int(cv["k"])
        if not 0 <= held_out < k:
            raise ValueError(f"held_out must be in [0, {k}), got {held_out}")
        select = set(folds[str(held_out)])
        fit = {s for i in range(k) if i != held_out for s in folds[str(i)]}
        if fit & select:
            raise PackageFormatError(f"{self.root}: cv folds overlap for fold {held_out}")
        return fit, select

    def split_rows(self, split: str) -> list[dict]:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        return [r for r in self.rows if r["split"] == split]


def load_package(root: str | Path, verify_checksums: bool = False) -> Package:
    """Load and check an ``ecgpkg`` v1 or v2 package.

    Raises :class:`PackageFormatError` on an unsupported format/version, a
    manifest hash mismatch, labels inconsistent with ``classes``, subjects
    shared between splits, or (optionally) a ``SHA256SUMS`` mismatch.
    """
    root = Path(root)
    pkg_json = root / "package.json"
    if not pkg_json.exists():
        raise PackageFormatError(f"{root}: package.json not found")
    meta = json.loads(pkg_json.read_text(encoding="utf-8"))
    if meta.get("format") != FORMAT or meta.get("format_version") not in SUPPORTED_FORMAT_VERSIONS:
        raise PackageFormatError(
            f"{root}: unsupported package format {meta.get('format')!r} version "
            f"{meta.get('format_version')!r}; this code reads {FORMAT!r} versions "
            f"{', '.join(str(v) for v in SUPPORTED_FORMAT_VERSIONS)}"
        )
    declared_rules = meta.get("fabrication_rules_version")
    if declared_rules is not None and str(declared_rules) != FABRICATION_RULES_VERSION:
        raise PackageFormatError(
            f"{root}: package declares fabrication_rules_version {declared_rules!r} but "
            f"ecg_transcovnet.augment reproduces version {FABRICATION_RULES_VERSION!r}; "
            f"the lead-fabrication port is stale - re-verify it against ecg_sigma's "
            f"LeadMapper before training"
        )

    manifest = root / "manifest.csv"
    if _sha256(manifest) != meta.get("manifest_sha256"):
        raise PackageFormatError(f"{root}: manifest.csv does not match manifest_sha256")

    try:
        spec = ClassSpec.from_names(meta["classes"])
    except ValueError as exc:
        raise PackageFormatError(f"{root}: invalid classes: {exc}") from exc

    with open(manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    subject_splits: dict[str, set[str]] = defaultdict(set)
    for n, row in enumerate(rows, start=2):
        for col in _INT_COLUMNS:
            row[col] = int(row[col])
        for col in _OPTIONAL_INT_COLUMNS:
            if row.get(col) not in (None, ""):
                row[col] = int(row[col])
        for col in _BOOL_COLUMNS:
            if row.get(col) not in (None, ""):
                row[col] = str(row[col]).strip().lower() in ("true", "1", "yes")
        split = row["split"]
        if split == EXCLUDED:
            continue
        if split not in SPLITS:
            raise PackageFormatError(f"manifest line {n}: unknown split {split!r}")
        label = row["label"]
        if label not in spec or row["label_idx"] != spec.index(label):
            raise PackageFormatError(
                f"manifest line {n}: label {label!r} / label_idx {row['label_idx']} "
                f"inconsistent with classes"
            )
        subject_splits[row["subject_id"]].add(split)
    leaked = sorted(s for s, sp in subject_splits.items() if len(sp) > 1)
    if leaked:
        raise PackageFormatError(
            f"{root}: {len(leaked)} subject(s) appear in more than one split, e.g. {leaked[:5]}"
        )

    splits_path = root / "splits.json"
    splits = json.loads(splits_path.read_text(encoding="utf-8")) if splits_path.exists() else None

    if verify_checksums:
        _verify_sha256sums(root)

    return Package(root=root, meta=meta, spec=spec, rows=rows, splits=splits)


def _verify_sha256sums(root: Path) -> None:
    sums = root / "SHA256SUMS"
    if not sums.exists():
        raise PackageFormatError(f"{root}: SHA256SUMS not found")
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(maxsplit=1)
        path = root / rel.strip()
        if not path.exists():
            raise PackageFormatError(f"{rel}: listed in SHA256SUMS but missing")
        if _sha256(path) != digest:
            raise PackageFormatError(f"{rel}: sha256 mismatch")


# ---------------------------------------------------------------------------
# Memory-mapped cache
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CachePaths:
    data: Path
    lengths: Path
    labels: Path
    meta: Path


def lead_tag(leads: Sequence[str], package_leads: Sequence[str]) -> str:
    return "all" if list(leads) == list(package_leads) else "-".join(leads)


def cache_paths(cache_dir: str | Path, pkg: Package, split: str, leads: Sequence[str]) -> CachePaths:
    stem = f"{pkg.version}_{split}_{lead_tag(leads, pkg.leads)}"
    d = Path(cache_dir)
    return CachePaths(d / f"{stem}.npy", d / f"{stem}.lengths.npy",
                      d / f"{stem}.labels.npy", d / f"{stem}.meta.json")


def _rows_digest(rows: Sequence[dict]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(r["event_uid"].encode())
        h.update(b"\n")
    return h.hexdigest()


def _write_file_group(args) -> int:
    """Worker: copy the events of a few HDF5 files into their memmap rows."""
    import h5py

    data_path, shape, root, leads, groups = args
    mm = np.lib.format.open_memmap(data_path, mode="r+")
    if mm.shape != tuple(shape):
        raise RuntimeError(f"cache shape {mm.shape} != expected {shape}")
    written = 0
    for relpath, events in groups:
        with h5py.File(Path(root) / relpath, "r") as hf:
            for row_idx, event_key, n_samples in events:
                ecg = hf[event_key]["ecg"]
                for j, lead in enumerate(leads):
                    sig = ecg[lead][:]
                    if len(sig) != n_samples:
                        raise PackageFormatError(
                            f"{relpath}/{event_key}/ecg/{lead}: length {len(sig)} != n_samples {n_samples}"
                        )
                    mm[row_idx, j, :n_samples] = sig
                written += 1
    mm.flush()
    del mm
    return written


def build_cache(
    pkg: Package,
    split: str,
    leads: Sequence[str] | None = None,
    cache_dir: str | Path = "data/training_cache",
    workers: int | None = None,
    verbose: bool = True,
) -> CachePaths:
    """Create (or reuse) the memmap cache of a split; returns its paths."""
    leads = list(leads or pkg.leads)
    rows = pkg.split_rows(split)
    paths = cache_paths(cache_dir, pkg, split, leads)
    expected = {
        "package_version": pkg.version,
        "manifest_sha256": pkg.manifest_sha256,
        "split": split,
        "leads": leads,
        "n": len(rows),
        "max_samples": MAX_SAMPLES,
        "rows_sha256": _rows_digest(rows),
    }
    if paths.meta.exists() and paths.data.exists():
        if json.loads(paths.meta.read_text()) == expected:
            return paths

    paths.data.parent.mkdir(parents=True, exist_ok=True)
    shape = (len(rows), len(leads), MAX_SAMPLES)
    partial = paths.data.with_name(paths.data.stem + ".partial.npy")
    mm = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=shape)
    del mm

    by_file: dict[str, list[tuple[int, str, int]]] = defaultdict(list)
    for i, r in enumerate(rows):
        if r["n_samples"] > MAX_SAMPLES:
            raise PackageFormatError(f"{r['event_uid']}: n_samples {r['n_samples']} > {MAX_SAMPLES}")
        by_file[r["h5_relpath"]].append((i, r["event_key"], r["n_samples"]))
    files = sorted(by_file.items())

    if workers is None:
        workers = max(1, min(16, (os.cpu_count() or 2) - 2))
    if verbose:
        print(f"Building cache {paths.data.name}: {len(rows)} events from {len(files)} files "
              f"({workers} worker(s))")
    if workers <= 1:
        written = sum(_write_file_group((str(partial), shape, str(pkg.root), leads, [g])) for g in files)
    else:
        per_task = max(1, math.ceil(len(files) / (workers * 8)))
        tasks = [(str(partial), shape, str(pkg.root), leads, files[k : k + per_task])
                 for k in range(0, len(files), per_task)]
        method = "spawn" if torch.cuda.is_initialized() else "fork"
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context(method)) as ex:
            written = sum(ex.map(_write_file_group, tasks))
    if written != len(rows):
        raise RuntimeError(f"cache build wrote {written} of {len(rows)} events")

    os.replace(partial, paths.data)
    np.save(paths.lengths, np.array([r["n_samples"] for r in rows], dtype=np.int32))
    np.save(paths.labels, np.array([r["label_idx"] for r in rows], dtype=np.int64))
    paths.meta.write_text(json.dumps(expected, indent=2))
    return paths


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PackageDataset(Dataset):
    """One split of an ``ecgpkg`` package as ``(signal, label)`` items.

    Order per item: crop → [train] lead-fabrication aug → [train] artefact
    injection → optional zeroing of fabricated leads → preprocessing →
    [train] scale / noise / channel dropout.

    Parameters
    ----------
    crop_len : samples per item; ``None`` returns the full valid length
        (evaluation only; batch with :class:`LengthBucketSampler`).
    fabricated_leads : ``"keep"`` or ``"zero"`` (zero leads whose
        ``real_lead_mask`` bit is 0, after any fabrication).
    force_pattern : evaluation counterfactual; fabricate this pattern
        (``"0100001"``/``"0100000"``) for every event that has more real leads.
    """

    def __init__(
        self,
        package: Package | str | Path,
        split: str,
        leads: Sequence[str] | None = None,
        subjects: Collection[str] | None = None,
        filter_config: FilterConfig | None = None,
        crop_len: int | None = 2000,
        train: bool = False,
        noise_aug_prob: float = 0.0,
        fabricated_leads: str = "keep",
        lead_fab_aug_prob: float = 0.0,
        force_pattern: str | None = None,
        cache_dir: str | Path = "data/training_cache",
        seed: int = 42,
        workers: int | None = None,
        scale_range: tuple[float, float] = (0.8, 1.2),
        noise_std_range: tuple[float, float] = (0.01, 0.05),
        channel_drop_prob: float = 0.1,
    ):
        self.package = package if isinstance(package, Package) else load_package(package)
        pkg = self.package
        self.split = split
        self.rows = pkg.split_rows(split)
        self.leads = list(leads or pkg.leads)
        missing = [l for l in self.leads if l not in pkg.leads]
        if missing:
            raise ValueError(f"leads {missing} not in package leads {pkg.leads}")
        if fabricated_leads not in ("keep", "zero"):
            raise ValueError("fabricated_leads must be 'keep' or 'zero'")
        if force_pattern is not None and force_pattern not in augment.FABRICATION_PATTERNS:
            raise ValueError(f"force_pattern must be one of {augment.FABRICATION_PATTERNS}")
        if "ECG2" not in self.leads and (lead_fab_aug_prob > 0 or force_pattern):
            raise ValueError("lead fabrication needs ECG2 among the leads")

        self.paths = build_cache(pkg, split, self.leads, cache_dir, workers)
        self.lengths = np.load(self.paths.lengths)
        self.labels = np.load(self.paths.labels)
        # A subject filter keeps the split's cache intact and indexes a subset of it,
        # which is how cross-validation folds reuse the train/val caches unchanged.
        self._index: np.ndarray | None = None
        if subjects is not None:
            wanted = set(subjects)
            keep = np.array([i for i, r in enumerate(self.rows) if r["subject_id"] in wanted],
                            dtype=np.int64)
            self.rows = [self.rows[i] for i in keep]
            self.lengths = self.lengths[keep]
            self.labels = self.labels[keep]
            self._index = keep
        if crop_len is not None and len(self.lengths) and crop_len > int(self.lengths.min()):
            raise ValueError(f"crop_len {crop_len} exceeds the shortest event ({self.lengths.min()})")

        self.crop_len = crop_len
        self.train = train
        self.noise_aug_prob = noise_aug_prob
        self.fabricated_leads = fabricated_leads
        self.lead_fab_aug_prob = lead_fab_aug_prob
        self.force_pattern = force_pattern
        self.seed = seed
        self.epoch = 0
        self.scale_range = scale_range
        self.noise_std_range = noise_std_range
        self.channel_drop_prob = channel_drop_prob
        self.filter_config = filter_config or FILTER_PRESETS["none"]
        self.pipeline = PreprocessingPipeline(self.filter_config)
        self.fs = pkg.fs
        self._lead_pos = [pkg.leads.index(l) for l in self.leads]
        self._data: np.ndarray | None = None

    # -- metadata ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.rows)

    def column(self, name: str) -> list[str]:
        return [r[name] for r in self.rows]

    @property
    def subjects(self) -> list[str]:
        return self.column("subject_id")

    def effective_length(self, idx: int) -> int:
        n = int(self.lengths[idx])
        return n if self.crop_len is None else self.crop_len

    def is_counterfactual_target(self, idx: int) -> bool:
        mask = self.rows[idx]["real_lead_mask"]
        return self.force_pattern is not None and augment.can_fabricate(mask, self.force_pattern)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    # -- item access -------------------------------------------------------

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_data"] = None  # reopen the memmap in each worker
        return state

    def _array(self) -> np.ndarray:
        if self._data is None:
            self._data = np.load(self.paths.data, mmap_mode="r")
        return self._data

    def crop_bounds(self, idx: int, rng: np.random.Generator | None = None) -> tuple[int, int]:
        n = int(self.lengths[idx])
        if self.crop_len is None or self.crop_len >= n:
            return 0, n
        if self.train:
            if rng is None:
                rng = self._rng(idx)
            start = int(rng.integers(0, n - self.crop_len + 1))
        else:
            start = (n - self.crop_len) // 2
        return start, start + self.crop_len

    def _rng(self, idx: int) -> np.random.Generator:
        return np.random.default_rng((self.seed, self.epoch, idx))

    def _mask_for_leads(self, mask: str) -> list[bool]:
        return [mask[p] == "1" for p in self._lead_pos]

    def __getitem__(self, idx: int):
        rng = self._rng(idx)
        start, stop = self.crop_bounds(idx, rng)
        row = idx if self._index is None else int(self._index[idx])
        x = np.array(self._array()[row, :, start:stop], dtype=np.float32)
        mask = self.rows[idx]["real_lead_mask"]

        pattern = None
        if self.force_pattern is not None and augment.can_fabricate(mask, self.force_pattern):
            pattern = self.force_pattern
        elif self.train and self.lead_fab_aug_prob > 0 and rng.random() < self.lead_fab_aug_prob:
            options = [p for p in augment.FABRICATION_PATTERNS if augment.can_fabricate(mask, p)]
            if options:
                pattern = options[int(rng.integers(len(options)))]
        if pattern is not None:
            x = augment.fabricate_from_ecg2(x, self.leads, pattern, self.fs)
            mask = pattern

        if self.train and self.noise_aug_prob > 0 and rng.random() < self.noise_aug_prob:
            x = augment.inject_artefacts(x, rng, self.fs)

        if self.fabricated_leads == "zero":
            for j, real in enumerate(self._mask_for_leads(mask)):
                if not real:
                    x[j] = 0.0

        x = self.pipeline(x)
        if self.train:
            x = augment.post_augment(x, rng, self.scale_range, self.noise_std_range,
                                     self.channel_drop_prob)
        return torch.from_numpy(x), torch.tensor(int(self.labels[idx]), dtype=torch.long)


class ConcatPackageDataset(Dataset):
    """Several :class:`PackageDataset` parts addressed as one dataset.

    Cross-validation folds span the train and val splits, whose caches are
    separate files, so a fold is expressed as the concatenation of two
    subject-filtered parts.  Exposes the surface the samplers and training
    loop use (``labels``, ``rows``, ``effective_length``, ``set_epoch``).
    """

    def __init__(self, parts: Sequence[PackageDataset]):
        parts = [p for p in parts if len(p)]
        if not parts:
            raise ValueError("ConcatPackageDataset needs at least one non-empty part")
        self.parts = list(parts)
        self.package = parts[0].package
        self.leads = parts[0].leads
        self.crop_len = parts[0].crop_len
        self.train = parts[0].train
        self._starts = np.cumsum([0] + [len(p) for p in self.parts])
        self.rows = [r for p in self.parts for r in p.rows]
        self.labels = np.concatenate([p.labels for p in self.parts])
        self.lengths = np.concatenate([p.lengths for p in self.parts])

    def __len__(self) -> int:
        return int(self._starts[-1])

    def _locate(self, idx: int) -> tuple[PackageDataset, int]:
        part = int(np.searchsorted(self._starts, idx, side="right") - 1)
        return self.parts[part], idx - int(self._starts[part])

    def __getitem__(self, idx: int):
        part, local = self._locate(idx)
        return part[local]

    def column(self, name: str) -> list[str]:
        return [r[name] for r in self.rows]

    @property
    def subjects(self) -> list[str]:
        return self.column("subject_id")

    def effective_length(self, idx: int) -> int:
        part, local = self._locate(idx)
        return part.effective_length(local)

    def is_counterfactual_target(self, idx: int) -> bool:
        part, local = self._locate(idx)
        return part.is_counterfactual_target(local)

    def set_epoch(self, epoch: int) -> None:
        for p in self.parts:
            p.set_epoch(epoch)


def cv_datasets(
    package: Package | str | Path, held_out: int, **kwargs,
) -> tuple[ConcatPackageDataset, ConcatPackageDataset]:
    """``(fit, select)`` datasets for cross-validation fold *held_out*.

    Folds partition the train+val subjects; the test split is never touched.
    Augmentation keyword arguments apply to the fit part only, so the
    selection part is evaluated deterministically.
    """
    pkg = package if isinstance(package, Package) else load_package(package)
    fit_subjects, select_subjects = pkg.cv_subjects(held_out)
    eval_kwargs = dict(kwargs)
    eval_kwargs.update(train=False, noise_aug_prob=0.0, lead_fab_aug_prob=0.0)
    fit = ConcatPackageDataset(
        [PackageDataset(pkg, s, subjects=fit_subjects, **kwargs) for s in ("train", "val")]
    )
    select = ConcatPackageDataset(
        [PackageDataset(pkg, s, subjects=select_subjects, **eval_kwargs) for s in ("train", "val")]
    )
    return fit, select


class LengthBucketSampler(Sampler[list[int]]):
    """Sequential batches of indices that share the same effective length."""

    def __init__(self, dataset: PackageDataset, batch_size: int):
        self.batch_size = batch_size
        buckets: dict[int, list[int]] = defaultdict(list)
        for i in range(len(dataset)):
            buckets[dataset.effective_length(i)].append(i)
        self.batches = [
            idx[k : k + batch_size]
            for _, idx in sorted(buckets.items())
            for k in range(0, len(idx), batch_size)
        ]

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def balanced_sample_weights(
    labels: Sequence[int], subjects: Sequence[str], beta: float = 0.5,
) -> np.ndarray:
    """Weights giving every class equal mass and damping prolific subjects.

    Within a class, an event of a subject contributing ``k`` events of that
    class gets weight ``k**-beta`` before per-class normalisation.
    """
    labels = np.asarray(labels)
    counts: dict[tuple[int, str], int] = defaultdict(int)
    for c, s in zip(labels.tolist(), subjects):
        counts[(c, s)] += 1
    w = np.array([counts[(c, s)] ** -beta for c, s in zip(labels.tolist(), subjects)], dtype=np.float64)
    classes = np.unique(labels)
    for c in classes:
        sel = labels == c
        w[sel] /= w[sel].sum() * len(classes)
    return w
