"""Smoke test: package training (resume), checkpoint metadata, evaluation and
processor round trip on the fixture package (CPU, tiny model)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from ecg_transcovnet import FILTER_PRESETS, PreprocessingPipeline
from ecg_transcovnet.checkpoint import load_model
from tests.fixtures.make_ecgpkg import CLASSES, make_package


def _run(args: list[str]) -> subprocess.CompletedProcess:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run([sys.executable, *args], capture_output=True, text=True, env=env, timeout=900)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    return result


def test_package_training_resume_evaluate_and_processor(tmp_path):
    root = make_package(tmp_path / "pkg")
    out = tmp_path / "run"
    cache = tmp_path / "cache"
    base = [
        "scripts/train.py", "--data-source", "package", "--package", str(root),
        "--output-dir", str(out), "--cache-dir", str(cache), "--workers", "0",
        "--batch-size", "8", "--eval-batch-size", "16", "--warmup-epochs", "1",
        "--embed-dim", "32", "--nhead", "4", "--num-encoder-layers", "1",
        "--num-decoder-layers", "1", "--dim-feedforward", "64",
        "--sampler", "balanced", "--lead-fab-aug-prob", "0.5",
    ]
    _run(base + ["--epochs", "1"])
    assert torch.load(out / "last.pt", weights_only=False)["epoch"] == 1

    resumed = _run(base + ["--epochs", "2", "--resume"])
    assert "Resumed from" in resumed.stdout
    assert torch.load(out / "last.pt", weights_only=False)["epoch"] == 2

    best = torch.load(out / "best_model.pt", weights_only=False)
    meta = json.loads((root / "package.json").read_text())
    assert best["class_names"] == CLASSES
    assert best["data_source"] == "package"
    assert best["package_version"] == meta["package_version"]
    assert best["package_manifest_sha256"] == meta["manifest_sha256"]
    assert best["filter_preset"] == "default"
    assert best["crop_len"] == 2000
    assert best["fabricated_leads"] == "keep"
    assert best["lead_fab_aug_prob"] == 0.5
    report = json.loads((out / "reports" / "test.json").read_text())
    assert set(report["sections"]) == {"crop2000", "full"}
    assert (out / "confusion_matrix.png").exists() and (out / "training_curves.png").exists()

    # Warm start applies even with --resume when the output directory has no last.pt
    warm = _run([arg if arg != str(out) else str(tmp_path / "warm") for arg in base]
                + ["--epochs", "1", "--resume", "--init-checkpoint", str(out / "best_model.pt")])
    assert "Warm start from" in warm.stdout and "queries copied for 4 classes" in warm.stdout

    # Checkpoint round trip through the processor
    loaded = load_model(out / "best_model.pt", "cpu")
    assert list(loaded.class_spec.names) == CLASSES and loaded.filter_preset == "default"
    spec = importlib.util.spec_from_file_location("script_processor", "scripts/processor.py")
    processor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(processor)
    tracker = processor.MetricsTracker(loaded.class_spec.names)
    result = processor.process_file(
        root / "data" / "ptbxl" / "test_00_2025-01.h5", loaded.model, loaded.leads,
        torch.device("cpu"), tracker, PreprocessingPipeline(FILTER_PRESETS[loaded.filter_preset]),
        class_spec=loaded.class_spec,
    )
    assert result.events[0].gt_name == "NORMAL_SINUS"
    assert result.events[0].pred_name in CLASSES

    # Package evaluation with counterfactual and calibration
    _run(["scripts/evaluate.py", "--checkpoint", str(out / "best_model.pt"), "--package", str(root),
          "--split", "test", "--output-dir", str(tmp_path / "eval"), "--cache-dir", str(cache),
          "--workers", "0", "--calibrate-on", "val"])
    evaluation = json.loads((tmp_path / "eval" / "test.json").read_text())
    assert set(evaluation["lead_conversion"]) == {"0100001", "0100000"}
    assert len(evaluation["logit_bias"]) == len(CLASSES)
    assert not evaluation["baseline_mode"]
