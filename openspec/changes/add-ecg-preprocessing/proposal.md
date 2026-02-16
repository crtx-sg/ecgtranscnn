# Change: Add ECG preprocessing module (noise filtering + normalization)

## Why
The model degrades on noisy signals because the only preprocessing is per-lead z-score normalization -- no actual noise removal. Simulated noise includes baseline wander (0.1-0.5 Hz), powerline interference (50/60 Hz), EMG bursts (20-90 Hz), motion artifact spikes, Gaussian noise, and electrode contact issues. A preprocessing module that filters these artifacts before normalization improves robustness on noisy data.

## What Changes
- **ADDED** `ecg_transcovnet/preprocessing.py`: `FilterConfig` dataclass, `FILTER_PRESETS` dict (none/default/conservative/aggressive), `PreprocessingPipeline` class with precomputed IIR filter coefficients, `preprocess_ecg()` convenience function
- **ADDED** `tests/test_preprocessing.py`: 14 unit tests covering shape preservation, baseline wander removal (>=20 dB), powerline 50/60 Hz removal (>=20 dB), QRS peak preservation, all presets, performance (<50 ms)
- **MODIFIED** `ecg_transcovnet/data.py`: `generate_dataset()` and `load_hdf5_test_samples()` accept `filter_config` parameter; inline z-score normalization replaced with `PreprocessingPipeline`; cache key includes filter preset tag
- **MODIFIED** `scripts/processor.py`: Added `--filter-preset` CLI arg; creates pipeline once, passes to `process_file()`; replaces inline z-score
- **MODIFIED** `scripts/train.py`: Added `--filter-preset` CLI arg; passes `FilterConfig` through to data generation
- **MODIFIED** `scripts/evaluate.py`: Added `--filter-preset` CLI arg; passes through to dataset generation and HDF5 evaluation
- **MODIFIED** `ecg_transcovnet/__init__.py`: Exports `FilterConfig`, `FILTER_PRESETS`, `PreprocessingPipeline`, `preprocess_ecg`
- **MODIFIED** `pyproject.toml`: Added `scipy>=1.10` to dependencies

## Impact
- Affected specs: `data-pipeline` (new filter_config parameter, preprocessing replaces inline normalization), `inference-pipeline` (new --filter-preset CLI flag, pipeline-based preprocessing)
- Affected code: `ecg_transcovnet/preprocessing.py` (new), `ecg_transcovnet/data.py`, `ecg_transcovnet/__init__.py`, `scripts/processor.py`, `scripts/train.py`, `scripts/evaluate.py`, `pyproject.toml`, `tests/test_preprocessing.py` (new)
- Risk: Low -- backward compatible. Default `--filter-preset none` applies only z-score normalization (identical to previous behavior). Scipy is lazy-imported so the package loads without it when filtering is disabled.
