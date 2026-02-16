## ADDED Requirements

### Requirement: ECG Preprocessing Pipeline
The system SHALL provide a configurable preprocessing pipeline (`PreprocessingPipeline`) that applies per-lead IIR noise filters followed by z-score normalization. Filter coefficients SHALL be precomputed once at pipeline construction and reused across signals. The pipeline SHALL accept a `FilterConfig` dataclass specifying which filters are enabled and their parameters.

#### Scenario: Default preset filtering
- **WHEN** a raw 7-lead ECG signal (7 x 2400 samples) is processed with the `default` preset
- **THEN** the pipeline SHALL apply in order: 2nd-order Butterworth high-pass at 0.5 Hz, IIR notch at 50 Hz (Q=30), IIR notch at 60 Hz (Q=30), 4th-order Butterworth low-pass at 40 Hz, and per-lead z-score normalization
- **AND** the output SHALL have shape (7, 2400) with dtype float32

#### Scenario: None preset (backward compatibility)
- **WHEN** a signal is processed with the `none` preset or with `filter_config=None`
- **THEN** only per-lead z-score normalization SHALL be applied (no filtering)
- **AND** the output SHALL be identical to the previous inline normalization behavior

#### Scenario: Aggressive preset with median filter
- **WHEN** a signal is processed with the `aggressive` preset
- **THEN** the pipeline SHALL additionally apply a median filter (kernel=5) before the Butterworth high-pass for spike/motion artifact removal

### Requirement: Filter Presets
The system SHALL provide four named filter presets accessible via `FILTER_PRESETS` dict: `none` (z-score only), `default` (HP 0.5 Hz + notch 50/60 Hz + LP 40 Hz + z-score), `conservative` (HP 0.3 Hz + notch 50/60 Hz Q=50 + LP 45 Hz + z-score), `aggressive` (median + HP 0.67 Hz + notch 50/60 Hz + LP 35 Hz + z-score).

#### Scenario: Preset selection via CLI
- **WHEN** a user specifies `--filter-preset default` on any CLI script (train.py, evaluate.py, processor.py)
- **THEN** the corresponding `FilterConfig` from `FILTER_PRESETS` SHALL be used for all signal preprocessing in that run

#### Scenario: All presets produce valid output
- **WHEN** any preset is applied to a valid ECG signal
- **THEN** the output SHALL be finite (no NaN or Inf values), have the same shape as the input, and have dtype float32

### Requirement: Zero-Phase Filtering
The system SHALL use forward-backward filtering (`filtfilt` / `sosfiltfilt`) for all IIR filters to achieve zero phase distortion, preserving QRS complex timing and morphology.

#### Scenario: QRS timing preservation
- **WHEN** a signal containing sharp QRS-like peaks is filtered with the default preset
- **THEN** peak positions SHALL remain within +-1 sample of their original locations

### Requirement: Lazy Scipy Import
The system SHALL import scipy lazily (inside filter methods, not at module level) so that the package loads without scipy installed when filtering is disabled (preset `none`).

#### Scenario: Package import without scipy
- **WHEN** scipy is not installed and `FILTER_PRESETS["none"]` is used
- **THEN** the preprocessing module SHALL load and function correctly using only numpy

## MODIFIED Requirements

### Requirement: Dataset Generation with Condition Sampling
The system SHALL generate training and evaluation datasets by sampling cardiac conditions according to configurable distributions: MIT-BIH prevalence proportions (default) or uniform distribution. Each sample SHALL consist of a 7-lead ECG signal with the selected noise level applied. The `generate_dataset()` function SHALL accept an optional `filter_config` parameter (`FilterConfig` or `None`); when provided, the preprocessing pipeline SHALL be applied to each signal instead of inline z-score normalization. When `filter_config` is `None`, only z-score normalization SHALL be applied (backward compatible).

#### Scenario: Proportional condition sampling
- **WHEN** generating a dataset with MIT-BIH proportions
- **THEN** conditions SHALL be sampled according to their clinical prevalence (e.g., Normal Sinus most frequent, VFib least frequent)

#### Scenario: Uniform condition sampling
- **WHEN** generating a dataset with uniform distribution
- **THEN** each of the 16 conditions SHALL have approximately equal representation

#### Scenario: Dataset generation with filter preset
- **WHEN** `generate_dataset()` is called with `filter_config=FILTER_PRESETS["default"]`
- **THEN** each generated signal SHALL be processed through the full preprocessing pipeline (high-pass, notch, low-pass, z-score) before being included in the dataset

### Requirement: Per-Lead Z-Score Normalization
The system SHALL normalize each ECG lead independently using z-score normalization (subtract mean, divide by standard deviation) before feeding signals to the model. This SHALL be applied consistently in both training and inference. Normalization is now the final stage of the `PreprocessingPipeline` and SHALL NOT be applied as separate inline code.

#### Scenario: Normalization of a 7-lead signal
- **WHEN** a raw 7-lead ECG signal is processed for model input
- **THEN** each lead SHALL have approximately zero mean and unit variance after normalization

### Requirement: Dataset Caching
The system SHALL cache generated datasets as `.npz` files in `data/training_cache/` to avoid redundant regeneration. The cache key SHALL incorporate the number of samples, noise level, distribution type, and filter preset name. `load_or_generate_data()` SHALL check for a cached dataset before generating a new one. Changing the filter preset SHALL result in a cache miss and trigger regeneration.

#### Scenario: Cache hit
- **WHEN** a dataset with matching parameters (including filter preset) exists in cache
- **THEN** the cached dataset SHALL be loaded directly without regeneration

#### Scenario: Cache miss due to different filter preset
- **WHEN** cached data exists for `filter_config=none` but `filter_config=default` is requested
- **THEN** a new dataset SHALL be generated with the default preprocessing pipeline, saved to cache, and returned
