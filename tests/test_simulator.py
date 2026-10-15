"""Tests for the ECG simulator."""

import numpy as np
import pytest

from ecg_transcovnet.simulator import ECGSimulator, Condition, CONDITION_REGISTRY
from ecg_transcovnet.simulator.conditions import ConditionConfig
from ecg_transcovnet.simulator.morphology import generate_patient_params, generate_beat_times
from ecg_transcovnet.simulator.noise import NoiseConfig, NOISE_PRESETS, apply_noise_pipeline


class TestConditions:
    def test_all_conditions_registered(self):
        for condition in Condition:
            assert condition in CONDITION_REGISTRY

    def test_condition_count(self):
        assert len(Condition) == 16

    def test_registry_types(self):
        for condition, config in CONDITION_REGISTRY.items():
            assert isinstance(config, ConditionConfig)
            assert config.hr_range[0] < config.hr_range[1]


class TestNoise:
    def test_presets_exist(self):
        for level in ("clean", "low", "medium", "high"):
            assert level in NOISE_PRESETS

    def test_clean_no_noise(self):
        config = NOISE_PRESETS["clean"]
        assert config.baseline_wander_amp == 0.0
        assert config.gaussian_std == 0.0

    def test_noise_pipeline(self):
        rng = np.random.default_rng(42)
        signal = np.zeros(2400)
        time = np.linspace(0, 12, 2400, endpoint=False)
        config = NOISE_PRESETS["medium"]
        noisy = apply_noise_pipeline(signal, time, 200.0, rng, config)
        assert noisy.shape == (2400,)


class TestMorphology:
    def test_patient_params(self):
        rng = np.random.default_rng(42)
        params = generate_patient_params(rng)
        assert 0.12 <= params.pr_interval <= 0.20
        assert 0.08 <= params.qrs_duration <= 0.12

    def test_beat_times(self):
        rng = np.random.default_rng(42)
        beats = generate_beat_times(12.0, 75.0, 0.03, rng)
        assert len(beats) > 0
        # All beat times should be within signal duration
        assert beats[0] >= 0
        assert beats[-1] < 12.0


class TestECGSimulator:
    def test_generate_ecg(self):
        sim = ECGSimulator(seed=42)
        ecg = sim.generate_ecg(Condition.NORMAL_SINUS)
        assert len(ecg) == 7  # 7 leads
        for lead_name, signal in ecg.items():
            assert signal.shape == (2400,)
            assert signal.dtype == np.float32

    def test_generate_ecg_all_conditions(self):
        sim = ECGSimulator(seed=42)
        for condition in Condition:
            ecg = sim.generate_ecg(condition, noise_level="clean")
            assert len(ecg) == 7

    def test_deterministic(self):
        ecg1 = ECGSimulator(seed=42).generate_ecg(Condition.NORMAL_SINUS, noise_level="clean")
        ecg2 = ECGSimulator(seed=42).generate_ecg(Condition.NORMAL_SINUS, noise_level="clean")
        for lead in ecg1:
            np.testing.assert_array_equal(ecg1[lead], ecg2[lead])

    def test_generate_event(self):
        sim = ECGSimulator(seed=42)
        event = sim.generate_event(Condition.ATRIAL_FIBRILLATION)
        assert event.condition == Condition.ATRIAL_FIBRILLATION
        assert len(event.ecg_signals) == 7
        assert event.ppg_signal is not None
        assert event.resp_signal is not None


def _beats(monkeypatch, condition, seed=0):
    """(p_times, qrs_times) that generate_single_lead draws for *condition*."""
    from ecg_transcovnet.simulator import morphology

    p_times, qrs_times = [], []
    real_p, real_qrs = morphology.create_p_wave, morphology.create_qrs_complex
    monkeypatch.setattr(morphology, "create_p_wave",
                        lambda t, c, *a, **k: p_times.append(c) or real_p(t, c, *a, **k))
    monkeypatch.setattr(morphology, "create_qrs_complex",
                        lambda t, c, *a, **k: qrs_times.append(c) or real_qrs(t, c, *a, **k))
    rng = np.random.default_rng(seed)
    params = morphology.generate_patient_params(rng)
    time = np.arange(2400) / 200.0
    morphology.generate_single_lead(time, CONDITION_REGISTRY[condition], params, 60.0, 1.0, rng)
    return np.array(p_times), np.array(qrs_times)


@pytest.mark.parametrize("condition", [Condition.AV_BLOCK_2_TYPE1, Condition.AV_BLOCK_2_TYPE2])
def test_second_degree_block_drops_beats(monkeypatch, condition):
    p, qrs = _beats(monkeypatch, condition)
    assert len(qrs) < len(p)  # some P waves are not conducted
    # PR of each conducted beat: the QRS follows its own P wave
    pr = np.array([q - p[p <= q].max() for q in qrs])
    groups = np.split(pr, np.flatnonzero(np.diff(qrs) > 1.5 * np.median(np.diff(qrs))) + 1)
    assert len(groups) >= 3
    for g in groups[:-1]:
        if condition is Condition.AV_BLOCK_2_TYPE1:
            assert np.all(np.diff(g) > 0)  # Wenckebach: PR lengthens until the drop
        else:
            assert np.allclose(g, g[0])     # Mobitz II: constant PR


def test_wide_qrs_conditions_use_their_qrs_range():
    for condition in (Condition.LBBB, Condition.NORMAL_SINUS):
        sim = ECGSimulator(seed=0)
        captured = {}
        orig = sim._generate_lead
        sim._generate_lead = lambda c, cfg, patient, hr, scale: (
            captured.setdefault("qrs", patient.qrs_duration), orig(c, cfg, patient, hr, scale))[1]
        sim.generate_ecg(condition, noise_level="clean")
        lo, hi = CONDITION_REGISTRY[condition].qrs_duration_range
        assert lo <= captured["qrs"] <= hi
