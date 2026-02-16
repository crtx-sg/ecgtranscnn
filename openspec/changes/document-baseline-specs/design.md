## Context
ECG-TransCovNet is a hybrid CNN-Transformer model for 16-class ECG arrhythmia classification. The system includes synthetic data generation, model training, evaluation, and real-time inference. All components are implemented and operational but lack formal specifications.

This change establishes baseline specs by decomposing the system into five capabilities that align with the existing module boundaries.

## Goals / Non-Goals
- **Goals**: Capture current system behavior as formal specs; provide a foundation for spec-driven changes
- **Non-Goals**: Proposing any code changes; redesigning the architecture; specifying future features

## Decisions
- **Capability decomposition**: Split into 5 capabilities matching the natural module boundaries:
  - `model-architecture` — the neural network (model.py)
  - `ecg-simulator` — synthetic signal generation (simulator/)
  - `data-pipeline` — dataset generation and loading (data.py)
  - `inference-pipeline` — real-time processing (processor.py)
  - `trained-models` — checkpoint inventory (models/)
- **Alternatives considered**: Single monolithic spec (rejected — too large, hard to maintain), per-file specs (rejected — too granular, couples to implementation)

## Risks / Trade-offs
- Specs describe synthetic data behavior only — real patient data integration would require spec updates
- Model hyperparameters are documented as current values; changes require spec modifications

## Open Questions
- None — this captures existing behavior only
