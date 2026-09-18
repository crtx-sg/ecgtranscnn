"""Tests for the data-driven class head, checkpoint loading and warm start."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ecg_transcovnet.checkpoint import build_model, checkpoint_filter_preset, load_model, warm_start
from ecg_transcovnet.classes import ClassSpec, canonical_condition_name
from ecg_transcovnet.simulator.conditions import Condition

V1_CLASSES = [
    "NORMAL_SINUS", "SINUS_BRADYCARDIA", "SINUS_TACHYCARDIA", "ATRIAL_FIBRILLATION",
    "ATRIAL_FLUTTER", "PAC", "PVC", "VENTRICULAR_TACHYCARDIA", "VENTRICULAR_FIBRILLATION",
    "LBBB", "RBBB", "AV_BLOCK_1",
]
LEGACY_CKPT = Path("models/avblock_fix/best_model.pt")


def test_default_spec_is_enum_order():
    spec = ClassSpec.default()
    assert list(spec.names) == [c.name for c in Condition]
    assert len(spec) == 16


@pytest.mark.parametrize("raw, expected", [
    ("V", "PVC"), ("PVC", "PVC"), (b"PVC", "PVC"), (np.bytes_(b"AFIB"), "ATRIAL_FIBRILLATION"),
    ("SVTA", "SVT"), ("ST", "SINUS_TACHYCARDIA"), ("ST_ELEVATION", "ST_ELEVATION"),
    ("OTHER", None), (b"PACED", None),
])
def test_condition_resolution(raw, expected):
    assert canonical_condition_name(raw) == expected


def test_resolve_into_head():
    spec = ClassSpec.from_names(V1_CLASSES)
    assert spec.resolve("V") == ("PVC", 6)
    assert spec.resolve(b"SVT") == ("SVT", None)       # enum member, not in head
    assert spec.resolve(b"OTHER") == ("OTHER", None)   # not an enum member


def test_spec_rejects_invalid_names():
    with pytest.raises(ValueError):
        ClassSpec.from_names(["NORMAL_SINUS", "AFIB"])
    with pytest.raises(ValueError):
        ClassSpec.from_names(["PVC", "PVC"])


def test_spec_from_package_json(tmp_path):
    (tmp_path / "package.json").write_text(json.dumps({"classes": V1_CLASSES}))
    assert list(ClassSpec.from_package(tmp_path).names) == V1_CLASSES


def test_spec_from_checkpoint():
    model = build_model(ClassSpec.default(), 7)
    assert ClassSpec.from_checkpoint({"model_state_dict": model.state_dict()}) == ClassSpec.default()
    small = build_model(ClassSpec.from_names(V1_CLASSES), 7)
    with pytest.raises(ValueError):
        ClassSpec.from_checkpoint({"model_state_dict": small.state_dict()})
    assert len(ClassSpec.from_checkpoint({"class_names": V1_CLASSES})) == 12


def test_checkpoint_filter_preset():
    assert checkpoint_filter_preset({}) == "none"
    assert checkpoint_filter_preset({"args": {"filter_preset": None}}) == "none"
    assert checkpoint_filter_preset({"args": {"filter_preset": "default"}}) == "default"
    assert checkpoint_filter_preset({"filter_preset": "conservative", "args": {}}) == "conservative"


def test_twelve_class_model_both_window_lengths():
    model = build_model(ClassSpec.from_names(V1_CLASSES), 7).eval()
    with torch.no_grad():
        for length in (2000, 2400):
            out = model(torch.randn(2, 7, length))
            assert out.shape == (2, 12)
            assert torch.isfinite(out).all()


@pytest.mark.skipif(not LEGACY_CKPT.exists(), reason="legacy checkpoint not available")
def test_legacy_checkpoint_loads_unchanged():
    loaded = load_model(LEGACY_CKPT, "cpu")
    assert len(loaded.class_spec) == 16
    assert loaded.filter_preset == "none"
    state = torch.load(LEGACY_CKPT, weights_only=False, map_location="cpu")["model_state_dict"]
    for key, value in loaded.model.state_dict().items():
        assert torch.equal(value, state[key])


def _source_checkpoint():
    torch.manual_seed(0)
    model = build_model(ClassSpec.default(), 7)
    return {"model_state_dict": model.state_dict(), "class_names": list(ClassSpec.default().names)}


def test_warm_start_copies_queries_by_name():
    src = _source_checkpoint()
    spec = ClassSpec.from_names(V1_CLASSES)
    model = build_model(spec, 7)
    info = warm_start(model, src, spec, "by_name")
    assert info["queries_copied"] == V1_CLASSES
    assert info["missing"] == []
    q_src = src["model_state_dict"]["object_queries"][0]
    q_dst = model.state_dict()["object_queries"][0]
    for i, name in enumerate(V1_CLASSES):
        assert torch.equal(q_dst[i], q_src[ClassSpec.default().index(name)])
    assert torch.equal(model.state_dict()["ffn_head.0.weight"], src["model_state_dict"]["ffn_head.0.weight"])


def test_warm_start_reinit_skips_queries_and_head():
    src = _source_checkpoint()
    spec = ClassSpec.from_names(V1_CLASSES)
    torch.manual_seed(1)
    model = build_model(spec, 7)
    fresh_queries = model.state_dict()["object_queries"].clone()
    info = warm_start(model, src, spec, "reinit")
    assert info["queries_copied"] == []
    assert all(k.startswith("ffn_head.") for k in info["skipped"])
    assert torch.equal(model.state_dict()["object_queries"], fresh_queries)
    assert torch.equal(model.state_dict()["cnn_backbone.stage1.conv.0.weight"],
                       src["model_state_dict"]["cnn_backbone.stage1.conv.0.weight"])
