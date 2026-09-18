"""The vectorised PreprocessingPipeline must equal lead-by-lead filtering."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import filtfilt, medfilt, sosfiltfilt

from ecg_transcovnet.preprocessing import FILTER_PRESETS, PreprocessingPipeline


def _per_lead_reference(pipeline: PreprocessingPipeline, signal: np.ndarray) -> np.ndarray:
    cfg = pipeline.config
    out = signal.astype(np.float64, copy=True)
    for ch in range(out.shape[0]):
        lead = out[ch]
        if cfg.median_enabled:
            lead = medfilt(lead, kernel_size=cfg.median_kernel)
        if pipeline._hp_sos is not None:
            lead = sosfiltfilt(pipeline._hp_sos, lead)
        if pipeline._notch50_ba is not None:
            lead = filtfilt(*pipeline._notch50_ba, lead)
        if pipeline._notch60_ba is not None:
            lead = filtfilt(*pipeline._notch60_ba, lead)
        if pipeline._lp_sos is not None:
            lead = sosfiltfilt(pipeline._lp_sos, lead)
        if cfg.normalize:
            mu, std = lead.mean(), lead.std()
            lead = (lead - mu) / std if std > 1e-6 else lead - mu
        out[ch] = lead
    return out.astype(np.float32)


@pytest.mark.parametrize("preset", list(FILTER_PRESETS))
@pytest.mark.parametrize("length", [2000, 2400])
def test_vectorised_equals_per_lead(preset, length):
    rng = np.random.default_rng(7)
    signal = (rng.standard_normal((7, length)) * 0.3 + 0.1).astype(np.float32)
    signal[3] = 0.0  # a flat (e.g. zeroed) lead
    pipeline = PreprocessingPipeline(FILTER_PRESETS[preset])
    np.testing.assert_array_equal(pipeline(signal), _per_lead_reference(pipeline, signal))
