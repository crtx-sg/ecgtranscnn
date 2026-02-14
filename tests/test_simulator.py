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
