"""Tests for ecg_transcovnet.preprocessing module."""

from __future__ import annotations

import time

import numpy as np
import pytest

from ecg_transcovnet.preprocessing import (
    FilterConfig,
    FILTER_PRESETS,
    PreprocessingPipeline,
    preprocess_ecg,
)


NUM_LEADS = 7
NUM_SAMPLES = 2400
SAMPLE_RATE = 200.0


def _make_signal(rng: np.random.Generator | None = None) -> np.ndarray:
    """Return a synthetic (NUM_LEADS, NUM_SAMPLES) signal with some structure."""
    if rng is None:
        rng = np.random.default_rng(0)
    t = np.linspace(0, NUM_SAMPLES / SAMPLE_RATE, NUM_SAMPLES, endpoint=False)
    signal = np.zeros((NUM_LEADS, NUM_SAMPLES), dtype=np.float64)
    for ch in range(NUM_LEADS):
        # Simulated QRS-like activity around 5 Hz + small noise
        signal[ch] = np.sin(2 * np.pi * 5 * t) + 0.1 * rng.standard_normal(NUM_SAMPLES)
    return signal


# ---- Shape preservation ---------------------------------------------------

def test_output_shape():
    """Pipeline output shape must match input shape."""
    signal = _make_signal()
    config = FILTER_PRESETS["default"]
    out = preprocess_ecg(signal, config)
    assert out.shape == signal.shape


def test_output_dtype():
    """Output must be float32."""
    signal = _make_signal()
    out = preprocess_ecg(signal, FILTER_PRESETS["default"])
    assert out.dtype == np.float32


# ---- Passthrough (none preset) -------------------------------------------

def test_none_preset_only_normalizes():
    """With preset 'none' (no filtering), only z-score normalization is applied."""
    rng = np.random.default_rng(42)
    signal = rng.standard_normal((NUM_LEADS, NUM_SAMPLES)) * 5.0 + 3.0
    out = preprocess_ecg(signal.copy(), FILTER_PRESETS["none"])

    # Each lead should have roughly mean=0, std=1
    for ch in range(NUM_LEADS):
        assert abs(out[ch].mean()) < 1e-4, f"Lead {ch} mean not ~0"
        assert abs(out[ch].std() - 1.0) < 1e-4, f"Lead {ch} std not ~1"


# ---- Baseline wander removal ---------------------------------------------

def test_baseline_wander_removal():
    """A 0.2 Hz drift should be attenuated by >=20 dB while 5 Hz ECG is preserved."""
    t = np.linspace(0, NUM_SAMPLES / SAMPLE_RATE, NUM_SAMPLES, endpoint=False)

    ecg_5hz = np.sin(2 * np.pi * 5 * t)
    drift_02hz = 2.0 * np.sin(2 * np.pi * 0.2 * t)
    signal = np.stack([ecg_5hz + drift_02hz] * NUM_LEADS, axis=0)

    config = FilterConfig(
        highpass_enabled=True,
        highpass_cutoff=0.5,
        highpass_order=2,
        normalize=False,  # disable normalization to measure absolute levels
    )
    out = preprocess_ecg(signal.copy(), config)

    # Measure residual 0.2 Hz via correlation with the drift
    residual_power = np.mean(out[0] * drift_02hz) ** 2
    original_power = np.mean(drift_02hz ** 2)
    attenuation_db = 10 * np.log10(max(residual_power, 1e-30) / original_power)
    assert attenuation_db < -20, f"Baseline drift only attenuated by {attenuation_db:.1f} dB"

    # 5 Hz content should be mostly preserved (within 50% amplitude)
    ecg_power_out = np.sqrt(np.mean(out[0] ** 2))
    ecg_power_in = np.sqrt(np.mean(ecg_5hz ** 2))
    assert ecg_power_out > 0.5 * ecg_power_in, "5 Hz ECG signal too attenuated"


# ---- Powerline removal ---------------------------------------------------

def test_powerline_50hz_removal():
    """50 Hz component should be reduced by >=20 dB."""
    t = np.linspace(0, NUM_SAMPLES / SAMPLE_RATE, NUM_SAMPLES, endpoint=False)

    ecg_5hz = np.sin(2 * np.pi * 5 * t)
    noise_50hz = 0.5 * np.sin(2 * np.pi * 50 * t)
    signal = np.stack([ecg_5hz + noise_50hz] * NUM_LEADS, axis=0)

    config = FilterConfig(
        notch_50_enabled=True,
        notch_q=30.0,
        normalize=False,
    )
    out = preprocess_ecg(signal.copy(), config)

    # Measure residual 50 Hz power via FFT
    fft_in = np.abs(np.fft.rfft(signal[0]))
    fft_out = np.abs(np.fft.rfft(out[0]))
    freqs = np.fft.rfftfreq(NUM_SAMPLES, d=1.0 / SAMPLE_RATE)

    idx_50 = np.argmin(np.abs(freqs - 50.0))
    attenuation_db = 20 * np.log10(max(fft_out[idx_50], 1e-30) / max(fft_in[idx_50], 1e-30))
    assert attenuation_db < -20, f"50 Hz only attenuated by {attenuation_db:.1f} dB"


def test_powerline_60hz_removal():
    """60 Hz component should be reduced by >=20 dB."""
    t = np.linspace(0, NUM_SAMPLES / SAMPLE_RATE, NUM_SAMPLES, endpoint=False)

    ecg_5hz = np.sin(2 * np.pi * 5 * t)
    noise_60hz = 0.5 * np.sin(2 * np.pi * 60 * t)
    signal = np.stack([ecg_5hz + noise_60hz] * NUM_LEADS, axis=0)

    config = FilterConfig(
        notch_60_enabled=True,
        notch_q=30.0,
        normalize=False,
    )
    out = preprocess_ecg(signal.copy(), config)

    fft_in = np.abs(np.fft.rfft(signal[0]))
    fft_out = np.abs(np.fft.rfft(out[0]))
    freqs = np.fft.rfftfreq(NUM_SAMPLES, d=1.0 / SAMPLE_RATE)

    idx_60 = np.argmin(np.abs(freqs - 60.0))
    attenuation_db = 20 * np.log10(max(fft_out[idx_60], 1e-30) / max(fft_in[idx_60], 1e-30))
    assert attenuation_db < -20, f"60 Hz only attenuated by {attenuation_db:.1f} dB"


# ---- QRS preservation ----------------------------------------------------

def test_qrs_peak_preservation():
    """Peak amplitude and timing should be preserved within 5%."""
    t = np.linspace(0, NUM_SAMPLES / SAMPLE_RATE, NUM_SAMPLES, endpoint=False)

    # Simulated sharp QRS-like pulse at known positions
    signal = np.zeros((1, NUM_SAMPLES), dtype=np.float64)
    peak_positions = [400, 800, 1200, 1600, 2000]
    for pos in peak_positions:
        # Gaussian pulse with ~25ms width (5 samples at 200 Hz)
        pulse = np.exp(-0.5 * ((np.arange(NUM_SAMPLES) - pos) / 5) ** 2)
        signal[0] += pulse

    config = FilterConfig(
        highpass_enabled=True,
        highpass_cutoff=0.5,
        highpass_order=2,
        lowpass_enabled=True,
        lowpass_cutoff=40.0,
        lowpass_order=4,
        normalize=False,
    )
    out = preprocess_ecg(signal.copy(), config)

    # Check peak timing is unchanged (within ±1 sample)
    for pos in peak_positions:
        window = slice(max(0, pos - 10), min(NUM_SAMPLES, pos + 10))
        out_peak = np.argmax(out[0, window]) + window.start
        assert abs(out_peak - pos) <= 1, f"Peak shifted from {pos} to {out_peak}"

    # Check peak amplitude within 5% (relative to max)
    in_max = signal[0].max()
    out_max = out[0].max()
    ratio = out_max / in_max
    assert 0.5 < ratio < 1.5, f"Peak amplitude ratio {ratio:.2f} outside tolerance"


# ---- Presets instantiation ------------------------------------------------

@pytest.mark.parametrize("preset_name", list(FILTER_PRESETS.keys()))
def test_preset_runs_without_error(preset_name):
    """Every preset should instantiate and process without error."""
    signal = _make_signal()
    config = FILTER_PRESETS[preset_name]
    pipeline = PreprocessingPipeline(config)
    out = pipeline(signal)
    assert out.shape == signal.shape
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))


# ---- Performance ----------------------------------------------------------

def test_performance():
    """7 leads x 2400 samples should process in <50ms."""
    signal = _make_signal()
    config = FILTER_PRESETS["default"]
    pipeline = PreprocessingPipeline(config)

    # Warm up
    pipeline(signal)

    t0 = time.perf_counter()
    n_runs = 20
    for _ in range(n_runs):
        pipeline(signal)
    elapsed = (time.perf_counter() - t0) / n_runs
    assert elapsed < 0.05, f"Average processing time {elapsed * 1000:.1f}ms exceeds 50ms"


# ---- Convenience function -------------------------------------------------

def test_preprocess_ecg_default():
    """preprocess_ecg with None config should only normalize."""
    signal = _make_signal()
    out = preprocess_ecg(signal.copy(), None)
    assert out.shape == signal.shape
    # Should be normalized
    for ch in range(NUM_LEADS):
        assert abs(out[ch].mean()) < 1e-4


def test_preprocess_ecg_with_config():
    """preprocess_ecg with explicit config should work."""
    signal = _make_signal()
    out = preprocess_ecg(signal.copy(), FILTER_PRESETS["aggressive"])
    assert out.shape == signal.shape
    assert np.all(np.isfinite(out))
