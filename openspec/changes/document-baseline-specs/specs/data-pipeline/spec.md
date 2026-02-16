## ADDED Requirements

### Requirement: Dataset Generation with Condition Sampling
The system SHALL generate training and evaluation datasets by sampling cardiac conditions according to configurable distributions: MIT-BIH prevalence proportions (default) or uniform distribution. Each sample SHALL consist of a 7-lead ECG signal with the selected noise level applied.

#### Scenario: Proportional condition sampling
- **WHEN** generating a dataset with MIT-BIH proportions
- **THEN** conditions SHALL be sampled according to their clinical prevalence (e.g., Normal Sinus most frequent, VFib least frequent)

#### Scenario: Uniform condition sampling
- **WHEN** generating a dataset with uniform distribution
- **THEN** each of the 16 conditions SHALL have approximately equal representation

### Requirement: Per-Lead Z-Score Normalization
The system SHALL normalize each ECG lead independently using z-score normalization (subtract mean, divide by standard deviation) before feeding signals to the model. This SHALL be applied consistently in both training and inference.

#### Scenario: Normalization of a 7-lead signal
- **WHEN** a raw 7-lead ECG signal is processed for model input
- **THEN** each lead SHALL have approximately zero mean and unit variance after normalization

### Requirement: Dataset Caching
The system SHALL cache generated datasets as `.pt` files in `data/training_cache/` to avoid redundant regeneration. The cache key SHALL incorporate the number of samples, noise level, and distribution type. `load_or_generate_data()` SHALL check for a cached dataset before generating a new one.

#### Scenario: Cache hit
- **WHEN** a dataset with matching parameters exists in cache
- **THEN** the cached dataset SHALL be loaded directly without regeneration

#### Scenario: Cache miss
- **WHEN** no matching cached dataset exists
- **THEN** a new dataset SHALL be generated, saved to cache, and returned

### Requirement: PyTorch Dataset Compatibility
The system SHALL provide an `AugmentedECGDataset` class compatible with PyTorch `DataLoader`, returning `(signal_tensor, label_tensor)` pairs where signal shape is `(7, 2400)` and label shape is `(16,)` as a multi-hot binary vector.

#### Scenario: DataLoader iteration
- **WHEN** the dataset is wrapped in a PyTorch DataLoader with batch_size=64
- **THEN** each batch SHALL yield tensors of shape `(64, 7, 2400)` for signals and `(64, 16)` for labels

### Requirement: Evaluation Metrics
The system SHALL compute per-class precision, recall, specificity, and F1-score, plus macro-averaged aggregates, overall accuracy, and a 16x16 confusion matrix for model evaluation.

#### Scenario: Formal evaluation run
- **WHEN** `evaluate.py` is run with a checkpoint and evaluation dataset
- **THEN** it SHALL report per-class metrics for all 16 conditions, macro averages, overall accuracy, and optionally save a confusion matrix visualization
