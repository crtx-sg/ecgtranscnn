## MODIFIED Requirements

### Requirement: HDF5 Event Processing
The system SHALL parse each HDF5 file from the simulator or from ecg_sigma, read `/metadata` fields from HDF5 attributes or datasets (decoding bytes), extract the checkpoint's ECG leads for each event, apply the preprocessing pipeline, run the model forward pass, and report the softmax arg-max class of the checkpoint's head. Stored conditions SHALL be resolved by enum value or enum name and mapped by name into the checkpoint's head.

#### Scenario: Multi-event file processing
- **WHEN** an HDF5 file contains multiple `event_*` groups
- **THEN** each event SHALL be processed independently and predictions SHALL be displayed per event with condition names and confidence scores

#### Scenario: ecg_sigma file with attribute metadata
- **WHEN** a file stores `/metadata/patient_id` as a bytes attribute and conditions as enum names
- **THEN** the patient id SHALL be displayed correctly and ground truths SHALL resolve to their class names

#### Scenario: Ground truth outside the head
- **WHEN** an event's condition is not in the checkpoint head (e.g. `OTHER`, or `SVT` for a 12-class head)
- **THEN** the ground truth SHALL be shown as `n/a (not in head)`, the prediction SHALL still be printed, the event SHALL be excluded from metrics, and a warning SHALL be printed once per unknown label

### Requirement: Model Checkpoint Loading
The system SHALL load a model checkpoint and reconstruct the model from stored hyperparameters (`embed_dim`, `nhead`, `num_encoder_layers`, `num_decoder_layers`, `dim_feedforward`, `dropout`), the stored leads and the stored head (`class_names`, defaulting to the 16-class enum for legacy checkpoints), ensuring the inference model matches the trained configuration.

#### Scenario: Checkpoint with hyperparameters
- **WHEN** a checkpoint containing architecture hyperparameters is loaded
- **THEN** the model SHALL be constructed with those exact hyperparameters before loading the state dict

#### Scenario: Package checkpoint
- **WHEN** a checkpoint with 12 `class_names` is loaded
- **THEN** the model SHALL output 12 logits and all reports SHALL use those names

### Requirement: Filter Preset CLI Argument for Processor
The inference processor SHALL accept a `--filter-preset` CLI argument with choices `none`, `default`, `conservative`, `aggressive`. When the argument is omitted, the preset recorded in the checkpoint SHALL be used (`none` for checkpoints that record no preset).

#### Scenario: Specifying filter preset
- **WHEN** the processor is started with `--filter-preset default`
- **THEN** a `PreprocessingPipeline` with the `default` `FilterConfig` SHALL be created once and applied to every ECG signal before model inference

#### Scenario: Default behavior without flag
- **WHEN** the processor is started without `--filter-preset` and the checkpoint records `filter_preset: default`
- **THEN** the `default` preset SHALL be used; for a legacy checkpoint without a recorded preset the `none` preset SHALL be used
