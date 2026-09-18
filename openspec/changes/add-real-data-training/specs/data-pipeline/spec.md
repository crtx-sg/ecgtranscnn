## ADDED Requirements

### Requirement: Data-Driven Class Specification
The system SHALL represent a model head as a `ClassSpec` (ordered `Condition` enum names with name↔index lookup) loaded from a package's `package.json` `classes` or from a checkpoint's `class_names`. When a checkpoint has no `class_names` and 16 object queries, the spec SHALL default to the `Condition` enum order. No training, evaluation or inference code path SHALL assume 16 classes.

#### Scenario: Head from package
- **WHEN** a `ClassSpec` is loaded from `ecg_pkg_v1/package.json`
- **THEN** it SHALL contain the 12 package classes in package order and models built from it SHALL output 12 logits

#### Scenario: Legacy checkpoint head
- **WHEN** a simulator checkpoint without package metadata is loaded
- **THEN** the spec SHALL be the 16-class enum order and the weights SHALL load unchanged

#### Scenario: Condition resolution
- **WHEN** a stored condition is an enum value (`"V"`) or an enum name (`"PVC"`)
- **THEN** it SHALL resolve to the enum name `PVC` and to that class's head index, or to no index when the class is not in the head

### Requirement: ecgpkg Package Loading
The system SHALL load an `ecgpkg` training package with `load_package(root)`, using only the standard library `csv`/`json` modules for metadata. It SHALL reject packages whose `format` is not `ecgpkg` or whose `format_version` is not 1, whose `manifest.csv` sha256 differs from `manifest_sha256`, whose non-excluded rows have a label outside `classes` or a `label_idx` that disagrees with the label, or in which a `subject_id` appears in more than one of train/val/test. Verification of `SHA256SUMS` SHALL be optional.

#### Scenario: Unsupported version
- **WHEN** `package.json` declares `format_version: 2`
- **THEN** loading SHALL raise an error naming the supported format and version

#### Scenario: Subject leakage
- **WHEN** a manifest assigns the same `subject_id` to train and test rows
- **THEN** loading SHALL raise an error listing the offending subject

### Requirement: Memory-Mapped Package Cache
The system SHALL cache a package split's signals as a float32 memory-mapped array of shape `(N, leads, 2400)` at `data/training_cache/<package_version>_<split>_<leads>.npy`, right-padding 2000-sample events, with sidecar lengths, labels and metadata (package version, manifest sha256, row digest, leads). Signals SHALL be stored as read (not normalised). The cache SHALL be written streaming, one HDF5 file per write step, SHALL be reused only when its metadata matches the package, and building it SHALL NOT allocate the whole split in memory.

#### Scenario: Cache reuse
- **WHEN** a dataset for a split whose cache metadata matches the package is created
- **THEN** the existing cache SHALL be opened without reading HDF5 files

#### Scenario: Bounded memory during build
- **WHEN** the cache for a split is built
- **THEN** peak in-process array allocations SHALL stay far below the split's total signal size

### Requirement: Package Dataset Cropping and Augmentation
`PackageDataset` SHALL return `(signal, label)` items by cropping inside the event's valid length (training: uniformly random start of `crop_len`; evaluation: centre crop of `crop_len`, or the full valid length when `crop_len` is None), then, for training only, optionally applying lead-fabrication augmentation and amplitude-relative simulator-artefact injection to the raw crop, then optionally zeroing fabricated leads (`fabricated_leads="zero"`), then applying `PreprocessingPipeline`, then, for training only, per-lead amplitude scaling, additive Gaussian noise and channel dropout. Padding SHALL never enter a crop. Training randomness SHALL be determined by (seed, epoch, index). Each item's `dataset`, `subject_id`, `label_method` and `real_lead_mask` SHALL be available for grouped evaluation.

#### Scenario: Deterministic evaluation crops
- **WHEN** the same evaluation item is read twice
- **THEN** both reads SHALL return identical arrays

#### Scenario: Train-only augmentation
- **WHEN** `train=False`
- **THEN** no augmentation SHALL be applied, regardless of augmentation probabilities

### Requirement: Lead-Fabrication Augmentation and Counterfactual
The system SHALL provide a transform that rebuilds non-real leads from ECG2 with ecg_sigma's rules (Lead I as −0.6 × 20 Hz low-passed ECG2 lagged 3 ms; III, aVR, aVL, aVF by Einthoven/Goldberger; synthetic vVX from ECG2 when the target pattern has no real precordial lead), band-passed like the package. Training SHALL apply it with probability `lead_fab_aug_prob` to events with more real leads than the target pattern. Evaluation SHALL be able to apply it to all-real-lead events to measure prediction changes.

#### Scenario: Fabrication matches the source pipeline
- **WHEN** ECG1 is fabricated from the ECG2 of a real VFDB or MIT-BIH event
- **THEN** it SHALL correlate strongly (r ≥ 0.9) with the ECG1 stored by ecg_sigma

### Requirement: Grouped Evaluation Metrics
The system SHALL report, for a head of any size: a confusion matrix sized to the head, per-class precision, recall, specificity, F1, support and subject count, one-vs-rest AUROC, macro averages over classes with support > 0 (listing absent classes), accuracy, a subject-bootstrap 95 % interval for macro-F1, and the same summary grouped by `dataset`, `label_method` and `real_lead_mask`.

#### Scenario: Absent classes
- **WHEN** a split has no events for a head class
- **THEN** that class SHALL be excluded from macro averages and listed as absent

## MODIFIED Requirements

### Requirement: Evaluation Metrics
The system SHALL compute per-class precision, recall, specificity, and F1-score, plus macro-averaged aggregates over classes present in the evaluated data, overall accuracy, and a confusion matrix sized to the model head (16×16 for simulator checkpoints, `len(classes)`² for package checkpoints).

#### Scenario: Formal evaluation run
- **WHEN** `evaluate.py` is run with a checkpoint and evaluation dataset
- **THEN** it SHALL report per-class metrics for every head class, macro averages over present classes, overall accuracy, and optionally save a confusion matrix visualization

#### Scenario: Package evaluation run
- **WHEN** `evaluate.py --package PATH --split test` is run
- **THEN** it SHALL write JSON, markdown and confusion-matrix PNG reports with per-class, per-dataset, per-label-method and per-real-lead-mask results for centre-crop 2000 and full-length inputs
