## ADDED Requirements

### Requirement: Model Checkpoint Inventory
The system SHALL maintain pre-trained model checkpoints in the `models/` directory, each stored as a `.pt` file containing model weights, optimizer state, training epoch, best validation loss, and architecture hyperparameters.

#### Scenario: Available checkpoints
- **WHEN** the models directory is inspected
- **THEN** the following checkpoints SHALL be available:
  - `models/best_model.pt` — Improved model (90.0% accuracy, 16K clean samples, 83 epochs)
  - `models/noise_robust/best_model.pt` — Noise-robust model (87.4% accuracy, 16K mixed-noise samples)
  - `models/avblock_fix/best_model.pt` — AV-block-focused model with enhanced PR interval morphology

### Requirement: Model Selection Guidance
The system SHALL provide clear guidance for model selection based on deployment context: the improved model for clean signal environments (highest accuracy), the noise-robust model for real-world deployment with noisy signals, and the AV-block-fix model for scenarios requiring accurate AV block discrimination.

#### Scenario: Clean environment deployment
- **WHEN** deploying in a controlled environment with clean ECG signals
- **THEN** the improved model (`models/best_model.pt`) SHALL be recommended for maximum accuracy (90.0%)

#### Scenario: Noisy environment deployment
- **WHEN** deploying in a real-world environment with variable noise levels
- **THEN** the noise-robust model (`models/noise_robust/best_model.pt`) SHALL be recommended for consistent performance across noise conditions

### Requirement: Known Classification Weaknesses
The system SHALL document known classification weaknesses for transparency: Normal Sinus vs AV Block 1st Degree confusion (subtle PR interval differences), LBBB/RBBB cross-confusion, accuracy degradation at high noise levels, and class imbalance effects on rare conditions.

#### Scenario: AV block confusion awareness
- **WHEN** evaluating model predictions on AV Block 1st Degree cases
- **THEN** operators SHALL be aware that F1-score for this condition is approximately 0.650 due to subtle morphological differences from Normal Sinus Rhythm
