"""processor.py / run_validation_suite.py with ecg_sigma and simulator files,
legacy 16-class and package-sized heads."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import pytest
import torch

from ecg_transcovnet import FILTER_PRESETS, PreprocessingPipeline
from ecg_transcovnet.checkpoint import build_model
from ecg_transcovnet.classes import NOT_IN_HEAD, ClassSpec
from ecg_transcovnet.report import extract_ids
from ecg_transcovnet.simulator import ECGSimulator, HDF5EventWriter
from ecg_transcovnet.simulator.conditions import Condition
from tests.fixtures.make_ecgpkg import CLASSES, LEADS, make_package

LEGACY_CKPT = Path("models/avblock_fix/best_model.pt")
CPU = torch.device("cpu")


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


processor = _load_script("processor")
validation_suite = _load_script("run_validation_suite")


@pytest.fixture(scope="module")
def sigma_file(tmp_path_factory) -> Path:
    root = make_package(tmp_path_factory.mktemp("pkg") / "pkg")
    return root / "data" / "incart" / "train0_2025-01.h5"


@pytest.fixture(scope="module")
def sim_file(tmp_path_factory) -> Path:
    sim = ECGSimulator(seed=3)
    events = [sim.generate_event(condition=c, noise_level="clean")
              for c in (Condition.PVC, Condition.ST_ELEVATION, Condition.NORMAL_SINUS)]
    path = tmp_path_factory.mktemp("sim") / "PT1234_2026-09.h5"
    HDF5EventWriter().write_file(str(path), events, patient_id="PT1234")
    return path


def _model(spec: ClassSpec):
    torch.manual_seed(0)
    return build_model(spec, len(LEADS)).eval()


def test_extract_ids_attribute_and_dataset_metadata(sigma_file, sim_file):
    with h5py.File(sigma_file, "r") as hf:
        assert extract_ids(sigma_file, hf)[0] == "train0"
    with h5py.File(sim_file, "r") as hf:
        assert extract_ids(sim_file, hf)[0] == "PT1234"


def test_processor_ecg_sigma_file_with_package_head(sigma_file, capsys):
    spec = ClassSpec.from_names(CLASSES)
    tracker = processor.MetricsTracker(spec.names)
    result = processor.process_file(sigma_file, _model(spec), LEADS, CPU, tracker,
                                    PreprocessingPipeline(FILTER_PRESETS["default"]), class_spec=spec)
    assert result.patient_id == "train0"
    gts = [e.gt_name for e in result.events]
    assert set(gts) == set(CLASSES) | {NOT_IN_HEAD}
    assert tracker.total == sum(g != NOT_IN_HEAD for g in gts)
    assert all(e.pred_name in CLASSES for e in result.events)
    assert not any(e.match for e in result.events if e.gt_name == NOT_IN_HEAD)
    out = capsys.readouterr().out
    assert out.count("'OTHER' is not in the model head") <= 1
    tracker.print_report()


def test_processor_simulator_file_with_legacy_head(sim_file):
    spec = ClassSpec.default()
    tracker = processor.MetricsTracker()
    result = processor.process_file(sim_file, _model(spec), LEADS, CPU, tracker,
                                    PreprocessingPipeline(FILTER_PRESETS["none"]))
    assert result.patient_id == "PT1234"
    assert [e.gt_name for e in result.events] == ["PVC", "ST_ELEVATION", "NORMAL_SINUS"]
    assert tracker.total == 3


def test_processor_simulator_file_with_package_head(sim_file):
    spec = ClassSpec.from_names(CLASSES)
    tracker = processor.MetricsTracker(spec.names)
    result = processor.process_file(sim_file, _model(spec), LEADS, CPU, tracker, class_spec=spec)
    assert [e.gt_name for e in result.events] == ["PVC", NOT_IN_HEAD, "NORMAL_SINUS"]
    assert tracker.total == 2


def test_validation_suite_process_file(sigma_file, sim_file):
    spec = ClassSpec.from_names(CLASSES)
    results = validation_suite.process_file(sigma_file, _model(spec), LEADS, CPU, spec)
    assert results and all(r["gt_name"] in CLASSES for r in results)
    legacy = validation_suite.process_file(sim_file, _model(ClassSpec.default()), LEADS, CPU)
    assert [r["gt_name"] for r in legacy] == ["PVC", "ST_ELEVATION", "NORMAL_SINUS"]
    metrics = validation_suite.compute_metrics([r["gt_idx"] for r in results],
                                               [r["pred_idx"] for r in results], CLASSES)
    assert metrics["confusion_matrix"].shape == (4, 4)


@pytest.mark.skipif(not LEGACY_CKPT.exists(), reason="legacy checkpoint not available")
def test_legacy_checkpoint_on_ecg_sigma_file(sigma_file):
    model, leads, spec, preset = processor.load_model(str(LEGACY_CKPT), CPU)
    assert len(spec) == 16 and preset == "none"
    tracker = processor.MetricsTracker(spec.names)
    result = processor.process_file(sigma_file, model, leads, CPU, tracker,
                                    PreprocessingPipeline(FILTER_PRESETS[preset]), class_spec=spec)
    assert tracker.total == len([e for e in result.events if e.gt_name != NOT_IN_HEAD]) > 0
