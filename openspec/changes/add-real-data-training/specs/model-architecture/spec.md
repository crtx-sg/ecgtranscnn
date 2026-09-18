## REMOVED Requirements

### Requirement: Multi-Label Classification Head
**Reason**: The code has always trained and served a single-label softmax head (cross-entropy based focal loss, arg-max prediction); the sigmoid multi-label description was inaccurate, and the `ecgpkg` contract fixes single-label output.
**Migration**: Replaced by "Single-Label Softmax Classification Head".

## ADDED Requirements

### Requirement: Single-Label Softmax Classification Head
The system SHALL classify each ECG window into exactly one class of its head using one learnable object query per class, a shared per-query FFN (128→64→1) producing one logit per class, softmax over the head, and arg-max prediction. The number of queries SHALL equal the head size (16 for simulator checkpoints, `len(classes)` for package checkpoints).

#### Scenario: Classification output
- **WHEN** the decoder outputs `C` query embeddings of shape `(B, C, 128)`
- **THEN** the head SHALL produce logits of shape `(B, C)` and the predicted class SHALL be the softmax arg-max

#### Scenario: Variable input length
- **WHEN** the same weights receive 2000-sample and 2400-sample inputs
- **THEN** both SHALL produce finite logits of shape `(B, C)`

## MODIFIED Requirements

### Requirement: Training Pipeline
The system SHALL train using AdamW optimizer (lr=5e-4, weight_decay=1e-4) with mixed precision (AMP), gradient clipping (max_norm=1.0), linear warmup (5 epochs), cosine decay and early stopping on a selection metric (simulator: validation accuracy; package: validation macro-F1 by default). Training data SHALL come from the simulator (`--data-source sim`, default) or from an `ecgpkg` package (`--data-source package`). Package training SHALL support warm start from a checkpoint with a different head, class weights from train-split counts or a class-and-subject balanced sampler, and resumable per-epoch checkpoints. Checkpoints SHALL save the best model and the final model, including hyperparameters and the head for reconstruction.

#### Scenario: Training with early stopping
- **WHEN** the selection metric does not improve for `--patience` consecutive epochs
- **THEN** training SHALL stop early and the best checkpoint SHALL be preserved

#### Scenario: Checkpoint contents
- **WHEN** a model checkpoint is saved
- **THEN** it SHALL contain `model_state_dict`, `epoch`, `args` with architecture hyperparameters (`embed_dim`, `nhead`, `num_encoder_layers`, `num_decoder_layers`, `dim_feedforward`, `dropout`), `leads` and `class_names`

#### Scenario: Warm start across heads
- **WHEN** a 12-class package model is initialised from a 16-class simulator checkpoint with `--init-queries by_name`
- **THEN** all shape-compatible backbone, encoder, decoder and FFN weights SHALL be loaded and the object queries of the 12 shared class names SHALL be copied
