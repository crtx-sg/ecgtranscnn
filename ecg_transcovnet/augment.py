"""Signal augmentation for real ECG windows.

- :func:`inject_artefacts` adds simulator artefacts scaled to each lead's
  amplitude (the simulator presets are absolute mV tuned for ~0.2 mV leads).
- :func:`fabricate_from_ecg2` rebuilds non-real leads from ECG2 with the
  same rules ecg_sigma uses for single-limb-lead records, so the
  "fabricated leads" pattern can be spread over every class (augmentation)
  or imposed on all-real events (counterfactual evaluation).
- :func:`post_augment` is the post-preprocessing augmentation of
  :class:`~ecg_transcovnet.data.AugmentedECGDataset` without the circular shift.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np

from .simulator.noise import NOISE_PRESETS, apply_noise_pipeline

SIM_REFERENCE_STD_MV = 0.2
ALL_REAL = "1111111"
PATTERN_ECG2_V1 = "0100001"    # MIT-BIH style: ECG2 and vVX measured
PATTERN_ECG2_ONLY = "0100000"  # VFDB/CUDB/AFDB style: only ECG2 measured
FABRICATION_PATTERNS = (PATTERN_ECG2_V1, PATTERN_ECG2_ONLY)
MASK_LEADS = ("ECG1", "ECG2", "ECG3", "aVR", "aVL", "aVF", "vVX")


# ---------------------------------------------------------------------------
# Artefact injection
# ---------------------------------------------------------------------------

def inject_artefacts(
    signal: np.ndarray,
    rng: np.random.Generator,
    fs: float = 200.0,
    presets: Sequence[str] = ("low", "medium"),
    ref_std: float = SIM_REFERENCE_STD_MV,
    scale_clip: tuple[float, float] = (0.25, 4.0),
) -> np.ndarray:
    """Add simulator artefacts to a raw ``(leads, T)`` crop.

    Each lead is brought to simulator amplitude, passed through
    :func:`apply_noise_pipeline` with a random preset, and scaled back.
    Flat (e.g. zeroed) leads are left untouched.
    """
    out = np.array(signal, dtype=np.float64, copy=True)
    time = np.arange(out.shape[-1]) / fs
    config = NOISE_PRESETS[presets[int(rng.integers(len(presets)))]]
    for ch in range(out.shape[0]):
        std = out[ch].std()
        if std < 1e-6:
            continue
        scale = float(np.clip(std / ref_std, *scale_clip))
        out[ch] = apply_noise_pipeline(out[ch] / scale, time, fs, rng, config) * scale
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Lead fabrication (port of ecg_sigma.signals.lead_mapper)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _sos(order: int, cutoff: tuple[float, ...] | float, btype: str, fs: float) -> np.ndarray:
    from scipy.signal import butter

    return butter(order, cutoff, btype=btype, fs=fs, output="sos")


@lru_cache(maxsize=None)
def _notch(freq: float, q: float, fs: float) -> tuple[np.ndarray, np.ndarray]:
    from scipy.signal import iirnotch

    return iirnotch(freq, q, fs=fs)


def synth_partner_lead(x: np.ndarray, fs: float) -> np.ndarray:
    """Lead I from Lead II: −0.6 × low-passed (≤20 Hz) signal lagged 3 ms."""
    from scipy.signal import sosfiltfilt

    cutoff = min(20.0, 0.45 * 0.5 * fs)
    smoothed = sosfiltfilt(_sos(4, cutoff, "lowpass", fs), x, axis=-1)
    lag = max(1, int(round(0.003 * fs)))
    partner = -0.6 * np.roll(smoothed, lag, axis=-1)
    partner[..., :lag] = partner[..., lag : lag + 1]
    return partner


def synth_v(x: np.ndarray, fs: float) -> np.ndarray:
    """V-lead-like signal from Lead II: 1 Hz high-pass + QRS-band emphasis."""
    from scipy.signal import sosfiltfilt

    hp = sosfiltfilt(_sos(2, 1.0, "highpass", fs), x, axis=-1)
    qrs = sosfiltfilt(_sos(4, (5.0, 20.0), "bandpass", fs), x, axis=-1)
    return hp + np.sign(qrs) * (np.abs(qrs) ** 1.2) * 0.4


def package_prefilter(x: np.ndarray, fs: float) -> np.ndarray:
    """0.5–40 Hz zero-phase band-pass + 50 Hz notch, as applied by ecg_sigma."""
    from scipy.signal import filtfilt, sosfiltfilt

    y = sosfiltfilt(_sos(4, (0.5, 40.0), "bandpass", fs), x, axis=-1)
    b, a = _notch(50.0, 30.0, fs)
    return filtfilt(b, a, y, axis=-1)


def can_fabricate(mask: str, pattern: str) -> bool:
    """True when *pattern* has strictly fewer real leads than *mask* and ECG2 is real."""
    if mask[1] != "1" or pattern not in FABRICATION_PATTERNS:
        return False
    if pattern == PATTERN_ECG2_V1 and mask[6] != "1":
        return False
    return mask.count("1") > pattern.count("1")


def fabricate_from_ecg2(
    signal: np.ndarray,
    leads: Sequence[str],
    pattern: str,
    fs: float = 200.0,
    prefilter: bool = True,
) -> np.ndarray:
    """Replace every lead that is not real in *pattern* by its ECG2-derived version.

    Parameters
    ----------
    signal : (len(leads), T) raw mV crop.
    leads : lead names of the rows of *signal*; must include ``ECG2``.
    pattern : ``"0100001"`` (keep vVX) or ``"0100000"`` (synthesise vVX).
    """
    if pattern not in FABRICATION_PATTERNS:
        raise ValueError(f"unsupported fabrication pattern {pattern!r}")
    index = {name: i for i, name in enumerate(leads)}
    if "ECG2" not in index:
        raise ValueError("lead fabrication needs ECG2")
    ii = np.asarray(signal[index["ECG2"]], dtype=np.float64)
    i_lead = synth_partner_lead(ii, fs)
    iii = ii - i_lead
    derived = {
        "ECG1": i_lead,
        "ECG3": iii,
        "aVR": -(i_lead + ii) / 2.0,
        "aVL": (i_lead - iii) / 2.0,
        "aVF": (ii + iii) / 2.0,
    }
    if pattern == PATTERN_ECG2_ONLY:
        derived["vVX"] = synth_v(ii, fs)

    out = np.array(signal, dtype=np.float64, copy=True)
    rows = [index[name] for name in derived if name in index]
    if rows:
        stacked = np.stack([derived[leads[r]] for r in rows])
        if prefilter:
            stacked = package_prefilter(stacked, fs)
        out[rows] = stacked
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Post-preprocessing augmentation
# ---------------------------------------------------------------------------

def post_augment(
    signal: np.ndarray,
    rng: np.random.Generator,
    scale_range: tuple[float, float] = (0.8, 1.2),
    noise_std_range: tuple[float, float] = (0.01, 0.05),
    channel_drop_prob: float = 0.1,
) -> np.ndarray:
    """Per-lead scaling, additive Gaussian noise and single-lead dropout."""
    out = signal * rng.uniform(*scale_range, size=(signal.shape[0], 1))
    out = out + rng.standard_normal(signal.shape) * rng.uniform(*noise_std_range)
    if signal.shape[0] > 1 and rng.random() < channel_drop_prob:
        out[int(rng.integers(signal.shape[0]))] = 0.0
    return out.astype(np.float32)
