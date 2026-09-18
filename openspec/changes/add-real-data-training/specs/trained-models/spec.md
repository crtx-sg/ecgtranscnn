## ADDED Requirements

### Requirement: Real-Data Checkpoint Format
Checkpoints trained on an `ecgpkg` package SHALL record, in addition to `model_state_dict`, `epoch`, `args` and `leads`: `class_names`, `data_source` (`package`), `package_version`, `package_manifest_sha256`, `crop_len`, `filter_preset`, `noise_aug_prob`, `fabricated_leads`, `lead_fab_aug_prob`, the selection metric and its best value. Training SHALL also save a resumable `last.pt` each epoch containing optimiser, scheduler, scaler, history and early-stopping state.

#### Scenario: Round trip through the processor
- **WHEN** a package-trained checkpoint is passed to `processor.py`
- **THEN** the processor SHALL rebuild the 12-class model, use the recorded filter preset and leads, and label predictions with `class_names`

#### Scenario: Resume after interruption
- **WHEN** training is restarted with `--resume` in the same output directory
- **THEN** it SHALL continue from the epoch after the last saved one with the saved optimiser and early-stopping state

### Requirement: Real-Data Model Evaluation Record
The best real-data model SHALL be saved to `models/real_v1/` with training curves, a confusion matrix and v1 test-split reports, and `docs/real-data-training.md` SHALL record every experiment's test macro-F1, per-class F1 with test subject counts, per-dataset, per-label-method and per-real-lead-mask results, the lead-conversion counterfactual, the baseline-mode numbers of the legacy checkpoints, and classes with fewer than 5 test subjects.

#### Scenario: Comparison with baseline
- **WHEN** the real-data model is evaluated on the v1 test split
- **THEN** its macro-F1 and accuracy SHALL be reported next to the legacy checkpoints' baseline-mode results on the same split
