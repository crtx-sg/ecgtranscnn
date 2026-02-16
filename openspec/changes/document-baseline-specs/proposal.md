# Change: Document baseline specifications for ECG-TransCovNet

## Why
The project has a working implementation spanning model architecture, ECG simulation, data pipelines, inference, and trained models — but no formal specifications exist. Establishing baseline specs captures the current system as the foundation for all future changes, enabling spec-driven development going forward.

## What Changes
- **ADDED** `model-architecture` spec: CNN-Transformer hybrid model with DETR-style object queries
- **ADDED** `ecg-simulator` spec: Synthetic ECG generation for 16 cardiac conditions with noise pipeline
- **ADDED** `data-pipeline` spec: Dataset generation, caching, normalization, and augmentation
- **ADDED** `inference-pipeline` spec: Real-time directory-watching inference processor
- **ADDED** `trained-models` spec: Pre-trained checkpoint inventory and performance baselines

## Impact
- Affected specs: All new (no existing specs)
- Affected code: None — documentation-only change capturing current behavior
- Risk: None — no code modifications
