## Context
The ECG-TransCovNet model receives 7-lead ECG signals (7 x 2400 samples, 12s at 200 Hz). The only preprocessing was per-lead z-score normalization, leaving noise artifacts intact in the signal fed to the CNN backbone. The simulator generates six types of noise: baseline wander, Gaussian, EMG bursts, motion artifacts, powerline interference (50/60 Hz), and electrode contact degradation.

## Goals / Non-Goals
- **Goals**: Remove noise artifacts before model input; centralize normalization into a single pipeline; provide configurable presets; maintain backward compatibility; precompute filter coefficients for batch processing efficiency
- **Non-Goals**: Adaptive filtering (signal-dependent parameter tuning); real-time streaming filter (full 12s window available); FIR filter support; wavelet denoising

## Decisions

### Filter type: IIR Butterworth (not FIR)
At 200 Hz sampling rate, a 0.5 Hz FIR high-pass would require ~1600 taps (prohibitive latency and computation). IIR Butterworth filters achieve the same cutoff with order 2-4 (a few coefficients). Trade-off: IIR filters have nonlinear phase, but this is mitigated by using `filtfilt`.

### Zero-phase filtering via `filtfilt`
Forward-backward filtering (`sosfiltfilt` / `filtfilt`) eliminates phase distortion entirely, preserving QRS complex timing and morphology. This requires the full signal to be available (not streaming), which is satisfied since we always have the complete 12-second window.

### SOS format for Butterworth, BA format for notch
Butterworth filters use second-order sections (SOS) for numerical stability at higher orders. Notch filters from `iirnotch` return transfer function (b, a) coefficients directly; since they are inherently 2nd-order, numerical stability is not a concern.

### Precomputed coefficients
`PreprocessingPipeline.__init__()` computes all filter coefficients once. The `__call__()` method only applies the precomputed filters per-lead. This avoids redundant coefficient computation when processing thousands of signals during training data generation.

### Lazy scipy imports
Scipy is imported inside filter methods (`_precompute`, `_apply_sos`, `_apply_ba`, `_apply_median`) rather than at module level. This allows the package to load without scipy when filtering is disabled (preset `none`), maintaining the original dependency footprint for users who don't need filtering.

### Four presets
| Preset | Rationale |
|--------|-----------|
| `none` | Z-score only -- identical to previous behavior, backward compatible default |
| `default` | Standard clinical ECG filtering: 0.5 Hz HP, 50/60 Hz notch, 40 Hz LP |
| `conservative` | Wider passband (0.3-45 Hz) and narrower notch (Q=50) to minimize signal alteration |
| `aggressive` | Median spike removal + tighter passband (0.67-35 Hz) for heavily corrupted signals |

### Both 50 Hz and 60 Hz notch in all presets
Power grid frequency varies by country (50 Hz in Europe/Asia/Africa, 60 Hz in Americas/parts of Asia). Applying both notch filters costs negligible computation and avoids requiring users to know their local grid frequency.

## Risks / Trade-offs
- **Over-filtering risk**: The low-pass at 40 Hz removes some high-frequency ECG content (e.g., sharp QRS peaks). Mitigated by the `conservative` preset (45 Hz cutoff) and unit tests verifying QRS peak preservation within tolerance.
- **Scipy dependency**: Added as a required dependency in pyproject.toml. Mitigated by lazy imports -- the package functions without scipy when filtering is disabled.
- **Cache invalidation**: Filter preset is included in the data cache key, so changing presets triggers data regeneration. This is intentional -- cached filtered data must match the requested configuration.

## Open Questions
- None -- implementation complete and tested.
