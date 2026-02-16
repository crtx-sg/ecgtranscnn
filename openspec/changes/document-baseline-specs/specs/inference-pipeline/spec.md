## ADDED Requirements

### Requirement: Directory-Watching Inference Processor
The system SHALL monitor a specified directory for new HDF5 files using Linux inotify (`pyinotify`) and automatically process each file as it arrives. The processor SHALL handle `IN_CLOSE_WRITE` events on `*.h5` files.

#### Scenario: New file detection
- **WHEN** a new `.h5` file is written to the watched directory
- **THEN** the processor SHALL detect it via inotify and begin inference processing

#### Scenario: Process existing files on startup
- **WHEN** the processor is started with `--process-existing` flag
- **THEN** it SHALL process all existing `.h5` files in the directory before entering watch mode

### Requirement: HDF5 Event Processing
The system SHALL parse each HDF5 file according to the Phase-0 schema, extract 7-lead ECG signals for each event, apply per-lead z-score normalization, run the model forward pass, and threshold sigmoid outputs at 0.5 to produce binary predictions.

#### Scenario: Multi-event file processing
- **WHEN** an HDF5 file contains multiple events under `events/`
- **THEN** each event SHALL be processed independently and predictions SHALL be displayed per-event with condition names and confidence scores

### Requirement: Model Checkpoint Loading
The system SHALL load a model checkpoint and reconstruct the model architecture from stored hyperparameters (`embed_dim`, `nhead`, `num_encoder_layers`, `num_decoder_layers`), ensuring the inference model matches the trained configuration.

#### Scenario: Checkpoint with hyperparameters
- **WHEN** a checkpoint containing architecture hyperparameters is loaded
- **THEN** the model SHALL be constructed with those exact hyperparameters before loading the state dict

### Requirement: Aggregate Classification Report
The system SHALL accumulate predictions and ground truth labels across all processed events and, on graceful shutdown (SIGINT/Ctrl+C), display an aggregate classification report with per-condition precision, recall, F1-score, and overall accuracy.

#### Scenario: Graceful shutdown with report
- **WHEN** the user sends SIGINT (Ctrl+C) to the processor
- **THEN** the processor SHALL stop watching, compute aggregate metrics over all processed events, and display the classification report before exiting
