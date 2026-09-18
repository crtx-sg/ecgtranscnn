"""Tests for the ecgpkg loader, memmap cache, PackageDataset and augmentation."""

from __future__ import annotations

import csv
import json
import shutil
import tracemalloc
from pathlib import Path

import h5py
import numpy as np
import pytest

from ecg_transcovnet import augment
from ecg_transcovnet.package import (
    LengthBucketSampler,
    PackageDataset,
    PackageFormatError,
    balanced_sample_weights,
    build_cache,
    load_package,
)
from ecg_transcovnet.preprocessing import FILTER_PRESETS, PreprocessingPipeline
from tests.fixtures.make_ecgpkg import CLASSES, LEADS, make_package, rewrite_manifest

V1 = Path("/home/sganesh/aiwork/vios/repo/ecg_sigma/packages/ecg_pkg_v1")


@pytest.fixture(scope="module")
def pkg_root(tmp_path_factory) -> Path:
    return make_package(tmp_path_factory.mktemp("ecgpkg") / "ecg_pkg_vtest")


@pytest.fixture
def cache_dir(tmp_path) -> Path:
    return tmp_path / "cache"


def _copy(pkg_root: Path, tmp_path: Path) -> Path:
    dst = tmp_path / "copy"
    shutil.copytree(pkg_root, dst)
    return dst


# ---------------------------------------------------------------------------
# Contract invariants of the fixture and loader checks
# ---------------------------------------------------------------------------

def test_fixture_meets_contract_invariants(pkg_root):
    pkg = load_package(pkg_root, verify_checksums=True)
    assert list(pkg.spec.names) == CLASSES
    included = [r for r in pkg.rows if r["split"] != "excluded"]
    assert {r["n_samples"] for r in included} == {2000, 2400}
    assert any(r["split"] == "excluded" for r in pkg.rows)

    # 1. no subject in more than one split
    splits_of: dict[str, set] = {}
    for r in included:
        splits_of.setdefault(r["subject_id"], set()).add(r["split"])
    assert all(len(s) == 1 for s in splits_of.values())
    # 2. label/label_idx consistency; 3./5. HDF5 layout and attribute metadata
    for r in included:
        assert r["label_idx"] == CLASSES.index(r["label"])
        with h5py.File(pkg_root / r["h5_relpath"], "r") as f:
            assert isinstance(f["metadata"].attrs["subject_id"], bytes)
            grp = f[r["event_key"]]
            assert grp["uuid"][()].decode() == r["event_uid"]
            mask = ""
            for lead in LEADS:
                ds = grp["ecg"][lead]
                assert ds.dtype == np.float32 and ds.shape == (r["n_samples"],)
                mask += "1" if ds.attrs["source"] == b"real" else "0"
            assert mask == r["real_lead_mask"]
    # 6. thresholds in every split
    thr = pkg.meta["inclusion_thresholds"]
    for c in CLASSES:
        for split in ("train", "val", "test"):
            rows = [r for r in included if r["label"] == c and r["split"] == split]
            assert len(rows) >= thr[split]["events"]
            assert len({r["subject_id"] for r in rows}) >= thr[split]["subjects"]


def test_bad_format_version_rejected(pkg_root, tmp_path):
    root = _copy(pkg_root, tmp_path)
    meta = json.loads((root / "package.json").read_text())
    meta["format_version"] = 99          # v1 and v2 are both supported
    (root / "package.json").write_text(json.dumps(meta))
    with pytest.raises(PackageFormatError, match="version"):
        load_package(root)


def test_format_version_2_accepted(pkg_root, tmp_path):
    root = _copy(pkg_root, tmp_path)
    meta = json.loads((root / "package.json").read_text())
    meta["format_version"] = 2
    (root / "package.json").write_text(json.dumps(meta))
    assert load_package(root).format_version == 2


def test_stale_fabrication_rules_rejected(pkg_root, tmp_path):
    """A rules bump must fail the load, not silently invalidate augment.py."""
    root = _copy(pkg_root, tmp_path)
    meta = json.loads((root / "package.json").read_text())
    meta["fabrication_rules_version"] = "2"
    (root / "package.json").write_text(json.dumps(meta))
    with pytest.raises(PackageFormatError, match="fabrication_rules_version"):
        load_package(root)


def test_matching_fabrication_rules_accepted(pkg_root, tmp_path):
    root = _copy(pkg_root, tmp_path)
    meta = json.loads((root / "package.json").read_text())
    meta["fabrication_rules_version"] = "1"
    (root / "package.json").write_text(json.dumps(meta))
    assert load_package(root).version


def test_eval_flag_accessors_default_empty(pkg_root):
    """v1 packages carry no flags; accessors must not require them."""
    pkg = load_package(pkg_root)
    assert pkg.low_confidence_eval == []
    assert pkg.eval_flags == {}
    assert pkg.classes_with_flag("no_seven_real_lead_events") == []
    assert pkg.cv is None


def test_cv_folds_partition_subjects(pkg_root, tmp_path):
    """cv_subjects returns disjoint fit/select sets covering every fold subject."""
    root = _copy(pkg_root, tmp_path)
    splits = json.loads((root / "splits.json").read_text())
    subjects = sorted({r["subject_id"] for r in load_package(root).rows})
    folds = {str(i): subjects[i::3] for i in range(3)}
    splits["cv"] = {"k": 3, "scope": "train+val", "folds": folds}
    (root / "splits.json").write_text(json.dumps(splits))
    pkg = load_package(root)
    assert pkg.cv["k"] == 3
    seen = set()
    for held_out in range(3):
        fit, select = pkg.cv_subjects(held_out)
        assert not (fit & select)
        assert select == set(folds[str(held_out)])
        seen |= select
    assert seen == set(subjects)
    with pytest.raises(ValueError):
        pkg.cv_subjects(3)


def test_manifest_hash_mismatch_rejected(pkg_root, tmp_path):
    root = _copy(pkg_root, tmp_path)
    with open(root / "manifest.csv", "a", encoding="utf-8") as f:
        f.write("\n")
    with pytest.raises(PackageFormatError, match="manifest_sha256"):
        load_package(root)


def test_subject_in_two_splits_rejected(pkg_root, tmp_path):
    root = _copy(pkg_root, tmp_path)

    def leak(rows):
        train_subject = next(r["subject_id"] for r in rows if r["split"] == "train")
        next(r for r in rows if r["split"] == "test")["subject_id"] = train_subject

    rewrite_manifest(root, leak)
    with pytest.raises(PackageFormatError, match="more than one split"):
        load_package(root)


def test_label_idx_mismatch_rejected(pkg_root, tmp_path):
    root = _copy(pkg_root, tmp_path)

    def corrupt(rows):
        row = next(r for r in rows if r["split"] == "train")
        row["label_idx"] = str((int(row["label_idx"]) + 1) % len(CLASSES))

    rewrite_manifest(root, corrupt)
    with pytest.raises(PackageFormatError, match="label_idx"):
        load_package(root)


def test_checksum_verification_detects_tampering(pkg_root, tmp_path):
    root = _copy(pkg_root, tmp_path)
    (root / "DATACARD.md").write_text("tampered\n")
    load_package(root)  # optional check is off by default
    with pytest.raises(PackageFormatError, match="sha256"):
        load_package(root, verify_checksums=True)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def test_cache_matches_hdf5_and_pads(pkg_root, cache_dir):
    pkg = load_package(pkg_root)
    paths = build_cache(pkg, "train", cache_dir=cache_dir, workers=2, verbose=False)
    data = np.load(paths.data, mmap_mode="r")
    rows = pkg.split_rows("train")
    assert data.shape == (len(rows), 7, 2400)
    assert np.load(paths.lengths).tolist() == [r["n_samples"] for r in rows]
    assert np.load(paths.labels).tolist() == [r["label_idx"] for r in rows]
    for i, r in enumerate(rows):
        with h5py.File(pkg_root / r["h5_relpath"], "r") as f:
            expected = np.stack([f[r["event_key"]]["ecg"][l][:] for l in LEADS])
        np.testing.assert_array_equal(data[i, :, : r["n_samples"]], expected)
        assert not data[i, :, r["n_samples"]:].any()


def test_cache_is_reused(pkg_root, cache_dir, monkeypatch):
    pkg = load_package(pkg_root)
    build_cache(pkg, "val", cache_dir=cache_dir, workers=1, verbose=False)
    import ecg_transcovnet.package as package_module

    def fail(*_a, **_k):
        raise AssertionError("cache rebuilt")

    monkeypatch.setattr(package_module, "_write_file_group", fail)
    build_cache(pkg, "val", cache_dir=cache_dir, workers=1, verbose=False)


def test_cache_build_memory_is_bounded(tmp_path_factory, cache_dir):
    root = make_package(tmp_path_factory.mktemp("bigger") / "pkg", events_per_class=6)
    pkg = load_package(root)
    n = len(pkg.split_rows("train"))
    dataset_bytes = n * 7 * 2400 * 4
    tracemalloc.start()
    build_cache(pkg, "train", cache_dir=cache_dir, workers=1, verbose=False)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < dataset_bytes / 4, f"peak {peak} bytes vs dataset {dataset_bytes} bytes"


# ---------------------------------------------------------------------------
# Dataset behaviour
# ---------------------------------------------------------------------------

def test_train_crops_stay_inside_valid_region(pkg_root, cache_dir):
    ds = PackageDataset(pkg_root, "train", crop_len=2000, train=True, cache_dir=cache_dir, workers=1)
    for epoch in range(5):
        ds.set_epoch(epoch)
        for i in range(len(ds)):
            start, stop = ds.crop_bounds(i)
            assert 0 <= start and stop <= ds.lengths[i] and stop - start == 2000
    x, y = ds[0]
    assert tuple(x.shape) == (7, 2000) and x.dtype.is_floating_point


def test_eval_crops_are_centred_and_deterministic(pkg_root, cache_dir):
    ds = PackageDataset(pkg_root, "test", crop_len=2000, train=False, noise_aug_prob=1.0,
                        lead_fab_aug_prob=1.0, cache_dir=cache_dir, workers=1)
    for i in range(len(ds)):
        start, _ = ds.crop_bounds(i)
        assert start == (ds.lengths[i] - 2000) // 2
    a, _ = ds[1]
    ds.set_epoch(3)
    b, _ = ds[1]
    assert np.array_equal(a.numpy(), b.numpy())
    # train=False ignores augmentation: equals preprocessing of the raw centre crop
    raw = np.load(ds.paths.data, mmap_mode="r")
    start, stop = ds.crop_bounds(1)
    expected = PreprocessingPipeline(FILTER_PRESETS["none"])(np.array(raw[1, :, start:stop]))
    np.testing.assert_array_equal(a.numpy(), expected)


def test_full_length_items_and_length_buckets(pkg_root, cache_dir):
    ds = PackageDataset(pkg_root, "test", crop_len=None, cache_dir=cache_dir, workers=1)
    lengths = {int(ds[i][0].shape[-1]) for i in range(len(ds))}
    assert lengths == {2000, 2400}
    sampler = LengthBucketSampler(ds, batch_size=3)
    assert sorted(i for b in sampler for i in b) == list(range(len(ds)))
    for batch in sampler:
        assert len({ds.effective_length(i) for i in batch}) == 1


def test_crop_longer_than_shortest_event_rejected(pkg_root, cache_dir):
    with pytest.raises(ValueError, match="crop_len"):
        PackageDataset(pkg_root, "test", crop_len=2400, cache_dir=cache_dir, workers=1)


def test_train_augmentation_is_seeded(pkg_root, cache_dir):
    kw = dict(crop_len=2000, train=True, noise_aug_prob=0.5, lead_fab_aug_prob=0.5,
              cache_dir=cache_dir, workers=1, seed=7)
    a = PackageDataset(pkg_root, "train", **kw)
    b = PackageDataset(pkg_root, "train", **kw)
    for i in range(len(a)):
        np.testing.assert_array_equal(a[i][0].numpy(), b[i][0].numpy())
    b.set_epoch(1)
    assert any(not np.array_equal(a[i][0].numpy(), b[i][0].numpy()) for i in range(len(a)))


def test_fabricated_leads_zero(pkg_root, cache_dir):
    ds = PackageDataset(pkg_root, "train", crop_len=2000, fabricated_leads="zero",
                        cache_dir=cache_dir, workers=1)
    for i, row in enumerate(ds.rows):
        x = ds[i][0].numpy()
        for j, bit in enumerate(row["real_lead_mask"]):
            assert (not x[j].any()) == (bit == "0")


@pytest.mark.parametrize("pattern, converted_masks", [
    ("0100001", {"1111111"}),              # keeps real vVX: only all-real events change
    ("0100000", {"1111111", "0100001"}),   # ECG2 only: every event with more real leads changes
])
def test_counterfactual_converts_events_with_more_real_leads(pkg_root, cache_dir, pattern, converted_masks):
    native = PackageDataset(pkg_root, "test", crop_len=2000, cache_dir=cache_dir, workers=1)
    cf = PackageDataset(pkg_root, "test", crop_len=2000, force_pattern=pattern,
                        cache_dir=cache_dir, workers=1)
    for i, row in enumerate(cf.rows):
        same = np.array_equal(native[i][0].numpy(), cf[i][0].numpy())
        assert cf.is_counterfactual_target(i) == (row["real_lead_mask"] in converted_masks)
        assert same == (not cf.is_counterfactual_target(i))
        # ECG2 is untouched by fabrication
        np.testing.assert_array_equal(native[i][0][1].numpy(), cf[i][0][1].numpy())


def test_metadata_columns_exposed(pkg_root, cache_dir):
    ds = PackageDataset(pkg_root, "val", crop_len=2000, cache_dir=cache_dir, workers=1)
    assert len(ds.subjects) == len(ds)
    assert set(ds.column("real_lead_mask")) == {"1111111", "0100001"}
    assert set(ds.column("dataset")) == {"incart", "mitbih", "ptbxl"}
    assert set(ds.column("label_method")) <= {"beat_morphology", "record_level"}


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def test_inject_artefacts_scales_with_amplitude():
    rng = np.random.default_rng(0)
    t = np.arange(2000) / 200.0
    base = np.sin(2 * np.pi * 1.2 * t)
    x = np.stack([0.05 * base, 1.0 * base]).astype(np.float32)
    rel = []
    for seed in range(20):
        y = augment.inject_artefacts(x, np.random.default_rng(seed), presets=("medium",))
        rel.append((y - x).std(axis=1) / x.std(axis=1))
    rel = np.mean(rel, axis=0)
    assert rel[0] < 2.0 and rel[1] < 2.0          # noise stays proportionate
    assert abs(np.log(rel[0] / rel[1])) < np.log(3.0)
    flat = np.zeros((2, 2000), dtype=np.float32)
    assert not augment.inject_artefacts(flat, rng).any()


def test_can_fabricate_rules():
    assert augment.can_fabricate("1111111", "0100001")
    assert augment.can_fabricate("1111111", "0100000")
    assert augment.can_fabricate("0100001", "0100000")
    assert not augment.can_fabricate("0100001", "0100001")
    assert not augment.can_fabricate("0100000", "0100000")
    assert not augment.can_fabricate("1111110", "0100001")


def test_balanced_sample_weights():
    labels = [0, 0, 0, 0, 1, 1]
    subjects = ["a", "a", "a", "b", "c", "d"]
    w = balanced_sample_weights(labels, subjects, beta=1.0)
    assert w[:4].sum() == pytest.approx(0.5) and w[4:].sum() == pytest.approx(0.5)
    assert w[0] == pytest.approx(w[3] / 3)   # subject a's events are damped by its count


@pytest.mark.skipif(not V1.exists(), reason="ecg_pkg_v1 not available")
def test_fabrication_reproduces_ecg_sigma_leads():
    with open(V1 / "manifest.csv", encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r["split"] != "excluded" and r["real_lead_mask"] in ("0100000", "0100001")]
    picks = {}
    for r in rows:
        picks.setdefault(r["dataset"], r)
    for r in picks.values():
        with h5py.File(V1 / r["h5_relpath"], "r") as f:
            x = np.stack([f[r["event_key"]]["ecg"][l][:] for l in LEADS])
        y = augment.fabricate_from_ecg2(x, LEADS, r["real_lead_mask"])
        assert np.corrcoef(x[0], y[0])[0, 1] >= 0.9, r["dataset"]
