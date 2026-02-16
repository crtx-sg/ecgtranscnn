## MODIFIED Requirements

### Requirement: HDF5 Event Processing
The system SHALL parse each HDF5 file according to the Phase-0 schema, extract 7-lead ECG signals for each event, apply the configured preprocessing pipeline (filtering + normalization via `PreprocessingPipeline`), run the model forward pass, and produce predictions. The preprocessing pipeline SHALL be created once at processor startup based on the `--filter-preset` CLI argument and reused for all events.

#### Scenario: Multi-event file processing with filtering
- **WHEN** an HDF5 file contains multiple events and `--filter-preset default` is specified
- **THEN** each event's ECG signal SHALL be processed through the full preprocessing pipeline (high-pass, notch 50/60 Hz, low-pass, z-score) before model inference
- **AND** predictions SHALL be displayed per-event with condition names and confidence scores

#### Scenario: Processing without filtering (backward compatible)
- **WHEN** `--filter-preset none` is specified (the default)
- **THEN** only per-lead z-score normalization SHALL be applied, matching the previous behavior

## ADDED Requirements

### Requirement: Filter Preset CLI Argument for Processor
The inference processor SHALL accept a `--filter-preset` CLI argument with choices `none`, `default`, `conservative`, `aggressive` (default: `none`). The selected preset SHALL determine the preprocessing pipeline applied to all ECG signals during inference.

#### Scenario: Specifying filter preset
- **WHEN** the processor is started with `--filter-preset default`
- **THEN** a `PreprocessingPipeline` with the `default` `FilterConfig` SHALL be created once and applied to every ECG signal before model inference

#### Scenario: Default behavior without flag
- **WHEN** the processor is started without `--filter-preset`
- **THEN** the `none` preset SHALL be used, applying only z-score normalization (backward compatible)
