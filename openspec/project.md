# Project Context

## Purpose
ECG-TransCovNet is a hybrid CNN-Transformer deep learning model for ECG arrhythmia classification across 16 cardiac conditions. Based on the paper "ECG-TransCovNet: A hybrid transformer model for accurate arrhythmia detection using Electrocardiogram signals" (Shah et al., IET CIT 2024).

**Primary goals:**
- Classify 16 cardiac conditions from 7-lead ECG signals (12 seconds @ 200 Hz)
- Provide realistic synthetic ECG data generation with configurable noise
- Support end-to-end inference pipeline with real-time directory watching
- Deliver comprehensive evaluation and visualization tools

## Tech Stack
- Python 3.10+
- PyTorch 2.0+ (model, training, inference)
- NumPy 1.24+ (signal processing, data generation)
- SciPy 1.10+ (IIR filter design and application for ECG preprocessing; lazy-imported)
- h5py 3.8+ (HDF5 dataset I/O)
- matplotlib 3.7+ (visualization)
- pyinotify 0.9.6 (Linux inotify-based file watching)
- pytest 7.0+ (testing, dev dependency)

## Project Conventions

### Code Style
- Python package structure under `ecg_transcovnet/`
- CLI scripts under `scripts/` with argparse-based interfaces
- Constants centralized in `ecg_transcovnet/constants.py`
- Type hints used throughout
- Enum-based condition definitions

### Architecture Patterns
- **Facade pattern**: `ECGSimulator` wraps morphology, noise, and conditions
- **Separation of concerns**: Model, data, training, visualization are independent modules
- **Cached data generation**: Training data cached to disk to avoid regeneration
- **HDF5 Phase-0 schema**: Standardized file format with metadata, signals, and ground truth

### Testing Strategy
- Unit tests for model components (SKConv, ResidualBlock, CNN backbone, full model)
- Unit tests for simulator (conditions, morphology, noise)
- Noise robustness tests across clean/low/medium/high noise levels
- End-to-end tests for training and inference pipelines
- Validation suite for comprehensive per-condition accuracy testing

### Git Workflow
- Single `main` branch for primary development
- Commit messages: descriptive summaries of changes

## Domain Context

### 16 Cardiac Conditions
| # | Condition | MIT-BIH Code | Category |
|---|-----------|-------------|----------|
| 0 | Normal Sinus Rhythm | N | Normal |
| 1 | Sinus Bradycardia | N | Normal variant |
| 2 | Sinus Tachycardia | N | Normal variant |
| 3 | Atrial Fibrillation | AFIB | Supraventricular |
| 4 | Atrial Flutter | AFL | Supraventricular |
| 5 | Premature Atrial Complex | A | Supraventricular |
| 6 | Supraventricular Tachycardia | SVTA | Supraventricular |
| 7 | Premature Ventricular Complex | V | Ventricular |
| 8 | Ventricular Tachycardia | VT | Ventricular |
| 9 | Ventricular Fibrillation | VF | Ventricular |
| 10 | Left Bundle Branch Block | LBBB | Conduction |
| 11 | Right Bundle Branch Block | RBBB | Conduction |
| 12 | AV Block 1st Degree | - | Conduction |
| 13 | AV Block 2nd Degree Type I | - | Conduction |
| 14 | AV Block 2nd Degree Type II | - | Conduction |
| 15 | ST Elevation | STE | Ischemic |

### Signal Specifications
- **Sampling rate**: 200 Hz
- **Duration**: 12 seconds per event
- **Signal length**: 2,400 samples
- **Leads**: ECG1, ECG2, ECG3, aVR, aVL, aVF, vVX (7 leads)
- **Additional signals**: PPG (50 Hz), Respiratory (25 Hz)
- **Vitals**: HR, SpO2, Blood Pressure, Respiratory Rate, Temperature, Posture

### Noise Model
Six-stage artifact pipeline:
1. Baseline wander (0.1-0.5 Hz sinusoidal drift)
2. Gaussian noise (additive white noise)
3. EMG bursts (high-frequency muscular interference)
4. Motion artifacts (sudden amplitude shifts)
5. Powerline interference (50/60 Hz)
6. Electrode contact degradation (intermittent signal dropout)

Four noise presets: clean, low, medium, high

## Important Constraints
- Synthetic data only — no real patient data in repository
- Model designed for 7-lead ECG input (not standard 12-lead)
- All signals preprocessed via `PreprocessingPipeline` (configurable noise filtering + per-lead z-score normalization)
- Multi-label classification (sigmoid outputs, not softmax)
- HDF5 files must conform to Phase-0 schema for pipeline compatibility

## External Dependencies
- No external APIs or cloud services required
- Self-contained data generation via simulator
- Pre-trained model checkpoints stored in `models/` directory
- pyinotify requires Linux (inotify kernel feature)
