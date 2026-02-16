"""ECG preprocessing: noise filtering + normalization pipeline.

Applies per-lead filtering (baseline wander removal, powerline notch,
high-frequency cutoff) followed by z-score normalization.  Filter
coefficients are precomputed once and reused across signals.

Scipy is imported lazily so the package loads without it when filtering
is disabled (preset ``"none"``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FilterConfig:
    """Per-filter enable/disable flags and parameters.

    Parameters
    ----------
    sample_rate : float
        Sampling rate in Hz.
    highpass_enabled : bool
        Enable Butterworth high-pass (baseline wander removal).
    highpass_cutoff : float
        High-pass cutoff frequency in Hz.
    highpass_order : int
        Butterworth filter order.
    notch_50_enabled : bool
        Enable 50 Hz notch filter.
    notch_60_enabled : bool
        Enable 60 Hz notch filter.
    notch_q : float
        Quality factor for notch filters.
    lowpass_enabled : bool
        Enable Butterworth low-pass (high-frequency noise removal).
    lowpass_cutoff : float
        Low-pass cutoff frequency in Hz.
    lowpass_order : int
        Butterworth filter order.
    median_enabled : bool
        Enable median filter for spike/motion artifact removal.
    median_kernel : int
        Median filter kernel size (must be odd).
    normalize : bool
        Apply per-lead z-score normalization as the final step.
    """

    sample_rate: float = 200.0
    highpass_enabled: bool = False
    highpass_cutoff: float = 0.5
    highpass_order: int = 2
    notch_50_enabled: bool = False
    notch_60_enabled: bool = False
    notch_q: float = 30.0
    lowpass_enabled: bool = False
    lowpass_cutoff: float = 40.0
    lowpass_order: int = 4
    median_enabled: bool = False
    median_kernel: int = 5
    normalize: bool = True

    @property
    def filtering_enabled(self) -> bool:
        """True if any filter (excluding normalization) is turned on."""
        return (
            self.highpass_enabled
            or self.notch_50_enabled
            or self.notch_60_enabled
            or self.lowpass_enabled
            or self.median_enabled
        )


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

FILTER_PRESETS: dict[str, FilterConfig] = {
    "none": FilterConfig(normalize=True),
    "default": FilterConfig(
        highpass_enabled=True,
        highpass_cutoff=0.5,
        highpass_order=2,
        notch_50_enabled=True,
        notch_60_enabled=True,
        notch_q=30.0,
        lowpass_enabled=True,
        lowpass_cutoff=40.0,
        lowpass_order=4,
        normalize=True,
    ),
    "conservative": FilterConfig(
        highpass_enabled=True,
        highpass_cutoff=0.3,
        highpass_order=2,
        notch_50_enabled=True,
        notch_60_enabled=True,
        notch_q=50.0,
        lowpass_enabled=True,
        lowpass_cutoff=45.0,
        lowpass_order=4,
        normalize=True,
    ),
    "aggressive": FilterConfig(
        highpass_enabled=True,
        highpass_cutoff=0.67,
        highpass_order=2,
        notch_50_enabled=True,
        notch_60_enabled=True,
        notch_q=30.0,
        lowpass_enabled=True,
        lowpass_cutoff=35.0,
        lowpass_order=4,
        median_enabled=True,
        median_kernel=5,
        normalize=True,
    ),
}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class PreprocessingPipeline:
    """Precomputes IIR filter coefficients and applies them per-lead.

    Parameters
    ----------
    config : FilterConfig
        Filter configuration.  Use ``FILTER_PRESETS["default"]`` for a
        sensible starting point.
    """

    def __init__(self, config: FilterConfig) -> None:
        self.config = config
        self._hp_sos: Optional[np.ndarray] = None
        self._notch50_ba: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._notch60_ba: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._lp_sos: Optional[np.ndarray] = None

        if config.filtering_enabled:
            self._precompute()

    # -- coefficient precomputation ----------------------------------------

    def _precompute(self) -> None:
        from scipy.signal import butter, iirnotch

        fs = self.config.sample_rate

        if self.config.highpass_enabled:
            self._hp_sos = butter(
                self.config.highpass_order,
                self.config.highpass_cutoff,
                btype="highpass",
                fs=fs,
                output="sos",
            )

        if self.config.notch_50_enabled:
            b, a = iirnotch(50.0, self.config.notch_q, fs=fs)
            self._notch50_ba = (b, a)

        if self.config.notch_60_enabled:
            b, a = iirnotch(60.0, self.config.notch_q, fs=fs)
            self._notch60_ba = (b, a)

        if self.config.lowpass_enabled:
            self._lp_sos = butter(
                self.config.lowpass_order,
                self.config.lowpass_cutoff,
                btype="lowpass",
                fs=fs,
                output="sos",
            )

    # -- per-lead filter helpers -------------------------------------------

    @staticmethod
    def _apply_median(lead: np.ndarray, kernel: int) -> np.ndarray:
        from scipy.signal import medfilt

        return medfilt(lead, kernel_size=kernel).astype(lead.dtype)

    @staticmethod
    def _apply_sos(lead: np.ndarray, sos: np.ndarray) -> np.ndarray:
        from scipy.signal import sosfiltfilt

        return sosfiltfilt(sos, lead).astype(lead.dtype)

    @staticmethod
    def _apply_ba(lead: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
        from scipy.signal import filtfilt

        return filtfilt(b, a, lead).astype(lead.dtype)

    @staticmethod
    def _normalize_lead(lead: np.ndarray) -> np.ndarray:
        mu = lead.mean()
        std = lead.std()
        if std > 1e-6:
            return (lead - mu) / std
        return lead - mu

    # -- public interface --------------------------------------------------

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Apply the full preprocessing pipeline.

        Parameters
        ----------
        signal : np.ndarray
            ECG signal of shape ``(num_leads, num_samples)``.

        Returns
        -------
        np.ndarray
            Preprocessed signal, same shape, dtype ``float32``.
        """
        signal = signal.astype(np.float64, copy=True)

        for ch in range(signal.shape[0]):
            lead = signal[ch]

            # 1. Spike removal (median filter)
            if self.config.median_enabled:
                lead = self._apply_median(lead, self.config.median_kernel)

            # 2. Baseline wander removal (high-pass)
            if self._hp_sos is not None:
                lead = self._apply_sos(lead, self._hp_sos)

            # 3. Powerline 50 Hz notch
            if self._notch50_ba is not None:
                b, a = self._notch50_ba
                lead = self._apply_ba(lead, b, a)

            # 4. Powerline 60 Hz notch
            if self._notch60_ba is not None:
                b, a = self._notch60_ba
                lead = self._apply_ba(lead, b, a)

            # 5. High-frequency noise removal (low-pass)
            if self._lp_sos is not None:
                lead = self._apply_sos(lead, self._lp_sos)

            # 6. Per-lead z-score normalization
            if self.config.normalize:
                lead = self._normalize_lead(lead)

            signal[ch] = lead

        return signal.astype(np.float32)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def preprocess_ecg(
    signal: np.ndarray,
    config: Optional[FilterConfig] = None,
) -> np.ndarray:
    """One-shot preprocessing: build pipeline, apply, return.

    For batch processing prefer constructing a :class:`PreprocessingPipeline`
    once and calling it repeatedly.

    Parameters
    ----------
    signal : np.ndarray
        Shape ``(num_leads, num_samples)``.
    config : FilterConfig or None
        If *None*, only z-score normalization is applied (backward compatible).
    """
    if config is None:
        config = FILTER_PRESETS["none"]
    pipeline = PreprocessingPipeline(config)
    return pipeline(signal)
