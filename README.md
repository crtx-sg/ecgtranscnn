# ECG-TransCovNet

Hybrid CNN-Transformer for ECG arrhythmia classification across 16 cardiac conditions.

Based on: Shah et al., *"ECG-TransCovNet: A hybrid transformer model for accurate arrhythmia detection using Electrocardiogram signals"*, IET CIT 2024.

---

## Table of Contents

- [Architecture](#architecture)
- [Preprocessing](#preprocessing)
- [Cardiac Conditions](#cardiac-conditions)
- [HDF5 Dataset Schema](#hdf5-dataset-schema)
- [Model Performance](#model-performance)
- [Setup](#setup)
- [Training](#training)
- [Inference Pipeline](#inference-pipeline)
- [Data Generation](#data-generation)
- [Visualization](#visualization)
- [Evaluation](#evaluation)
- [Test & Evaluation Sequence](#test--evaluation-sequence)
- [Package Structure](#package-structure)
- [Requirements](#requirements)

---

## Architecture

ECG-TransCovNet is a hybrid model that combines the local feature extraction strength of CNNs with the global context modelling of Transformers.

```
Input: 7-lead ECG signal (7 × 2400 samples, 12s at 200 Hz)
  │
  ▼
┌─────────────────────────────────────────┐
│  CNN Backbone                           │
│  ├─ ResidualBlock (7→32, stride 2+pool) │    2400 → 600
│  ├─ Selective Kernel Conv (32→64)       │     600 → 600
│  ├─ ResidualBlock (64→128, stride 2+pool)│    600 → 150
│  ├─ ResidualBlock (128→256, stride 2+pool)│   150 → 38
│  └─ 1×1 Conv bottleneck (256→128)       │     38 → 38
└─────────────────────────────────────────┘
  │  Feature map: (B, 128, 38)
  ▼
┌─────────────────────────────────────────┐
│  Sinusoidal Positional Encoding         │
│  Transformer Encoder (3 layers, 8 heads)│
└─────────────────────────────────────────┘
  │  Memory: (B, 38, 128)
  ▼
┌─────────────────────────────────────────┐
│  Transformer Decoder (3 layers, 8 heads)│
│  + 16 Learnable Object Queries (DETR)   │
└─────────────────────────────────────────┘
  │  Decoded queries: (B, 16, 128)
  ▼
┌─────────────────────────────────────────┐
│  FFN Classification Head (per query)    │
│  128 → 64 → 1  (× 16 queries)          │
└─────────────────────────────────────────┘
  │
  ▼
Output: 16-class logits
```

**Key components:**

- **Selective Kernel (SK) Convolution**: Uses multiple parallel convolution branches with different kernel sizes (3, 5) and a channel-wise attention mechanism to dynamically weight them, adapting receptive field per input.
- **Residual Blocks**: Each CNN stage uses skip connections with 1×1 projection for channel alignment.
- **DETR-style Object Queries**: 16 learnable query embeddings (one per class) attend to the encoded feature sequence via cross-attention in the decoder.
- **Focal Loss**: Addresses class imbalance with per-class alpha weights and a focusing parameter (gamma=2.0) that down-weights easy examples.

**Default hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Embedding dimension | 128 |
| Attention heads | 8 |
| Encoder layers | 3 |
| Decoder layers | 3 |
| Feed-forward dimension | 512 |
| Dropout | 0.1 |
| Signal length | 2400 (12s × 200 Hz) |
| Input channels | 7 (all leads) |

---

## Preprocessing

Raw ECG signals are noisy — baseline wander, powerline interference, EMG bursts, and motion artifacts all degrade classification accuracy. The preprocessing module (`ecg_transcovnet/preprocessing.py`) applies a per-lead IIR filter pipeline followed by z-score normalization, replacing the previous normalization-only approach.

### Filter Pipeline

Filters are applied per-lead in this order using zero-phase `filtfilt` (forward-backward filtering) to preserve QRS timing:

| Stage | Algorithm | Default Parameters | Target Artifact |
|-------|-----------|-------------------|-----------------|
| 1. Spike removal | Median filter (optional) | kernel=5 | Motion artifact spikes |
| 2. Baseline wander | 2nd-order Butterworth high-pass | cutoff=0.5 Hz | 0.1–0.5 Hz drift |
| 3. Powerline 50 Hz | IIR notch filter | Q=30 | 50 Hz interference (Europe/Asia) |
| 4. Powerline 60 Hz | IIR notch filter | Q=30 | 60 Hz interference (Americas/Japan) |
| 5. High-freq noise | 4th-order Butterworth low-pass | cutoff=40 Hz | EMG, Gaussian noise |
| 6. Normalization | Per-lead z-score | mean=0, std=1 | Amplitude/offset variation |

### Filter Presets

Four named presets are available via the `--filter-preset` CLI flag on all scripts:

| Preset | Filters Applied | Use Case |
|--------|----------------|----------|
| `none` | Z-score normalization only | Backward compatibility (default) |
| `default` | HP 0.5 Hz + notch 50/60 Hz + LP 40 Hz + z-score | Recommended for noisy data |
| `conservative` | HP 0.3 Hz + notch 50/60 Hz (Q=50) + LP 45 Hz + z-score | Minimal signal alteration |
| `aggressive` | Median + HP 0.67 Hz + notch 50/60 Hz + LP 35 Hz + z-score | Heavy noise environments |

### Usage

```python
from ecg_transcovnet import PreprocessingPipeline, FILTER_PRESETS, FilterConfig

# Use a named preset
pipeline = PreprocessingPipeline(FILTER_PRESETS["default"])
clean_signal = pipeline(raw_signal)  # (7, 2400) → (7, 2400) float32

# Custom configuration
config = FilterConfig(
    highpass_enabled=True, highpass_cutoff=0.5,
    notch_50_enabled=True, notch_60_enabled=True,
    lowpass_enabled=True, lowpass_cutoff=40.0,
    normalize=True,
)
pipeline = PreprocessingPipeline(config)
clean_signal = pipeline(raw_signal)
```

### CLI Usage

All scripts (`train.py`, `evaluate.py`, `processor.py`) accept `--filter-preset`:

```bash
# Training with preprocessing
python scripts/train.py --noise-level mixed --filter-preset default

# Inference with preprocessing
python scripts/processor.py \
    --watch-dir data/inference \
    --checkpoint models/noise_robust/best_model.pt \
    --process-existing \
    --filter-preset default

# Evaluation with preprocessing
python scripts/evaluate.py \
    --checkpoint models/noise_robust/best_model.pt \
    --noise-level high \
    --filter-preset default
```

### Design Notes

- **IIR Butterworth** filters (not FIR) — at 200 Hz sampling rate, a 0.5 Hz FIR high-pass would need ~1600 taps
- **`filtfilt`** (forward-backward) for zero phase distortion — preserves QRS morphology and timing
- **Precomputed coefficients** — `PreprocessingPipeline` computes `butter`/`iirnotch` coefficients once at construction, reuses per signal
- **Lazy scipy imports** — scipy is only imported when filtering is enabled, so the package loads without it when using preset `none`

---

## Cardiac Conditions

The model classifies 16 cardiac conditions based on MIT-BIH annotation codes:

| # | Condition | Code | Category |
|---|-----------|------|----------|
| 1 | Normal Sinus Rhythm | N | Normal |
| 2 | Sinus Bradycardia | SB | Normal |
| 3 | Sinus Tachycardia | ST | Normal |
| 4 | Atrial Fibrillation | AFIB | Supraventricular |
| 5 | Atrial Flutter | AFL | Supraventricular |
| 6 | Premature Atrial Complex | A | Supraventricular |
| 7 | Supraventricular Tachycardia | SVTA | Supraventricular |
| 8 | Premature Ventricular Complex | V | Ventricular |
| 9 | Ventricular Tachycardia | VT | Ventricular |
| 10 | Ventricular Fibrillation | VF | Ventricular |
| 11 | Left Bundle Branch Block | L | Bundle Branch |
| 12 | Right Bundle Branch Block | R | Bundle Branch |
| 13 | AV Block 1st Degree | 1AVB | AV Block |
| 14 | AV Block 2nd Degree Type 1 | 2AVB1 | AV Block |
| 15 | AV Block 2nd Degree Type 2 | 2AVB2 | AV Block |
| 16 | ST Elevation | STE | Other |

---

## HDF5 Dataset Schema

Each generated file follows the naming convention `PatientID_YYYY-MM.h5` and contains a global metadata group plus one or more event groups:

```
PatientID_YYYY-MM.h5
├── metadata/                      # Global file metadata
│   ├── patient_id                 # "PT1234"
│   ├── sampling_rate_ecg          # 200.0 Hz
│   ├── sampling_rate_ppg          # 75.0 Hz
│   ├── sampling_rate_resp         # 33.33 Hz
│   ├── alarm_time_epoch           # Epoch timestamp
│   ├── alarm_offset_seconds       # 6.0 (center position)
│   ├── seconds_before_event       # 6.0 seconds
│   ├── seconds_after_event        # 6.0 seconds
│   ├── data_quality_score         # 0.85–0.98
│   ├── device_info                # "RMSAI-SimDevice-v2.0"
│   └── max_vital_history          # 30
│
├── event_1001/                    # First alarm event
│   ├── ecg/                       # ECG signal group (200 Hz)
│   │   ├── ECG1                   # Lead I      [2400 float32, gzip]
│   │   ├── ECG2                   # Lead II     [2400 float32, gzip]
│   │   ├── ECG3                   # Lead III    [2400 float32, gzip]
│   │   ├── aVR                    # Augmented R [2400 float32, gzip]
│   │   ├── aVL                    # Augmented L [2400 float32, gzip]
│   │   ├── aVF                    # Augmented F [2400 float32, gzip]
│   │   ├── vVX                    # Chest lead  [2400 float32, gzip]
│   │   └── extras                 # JSON: {"pacer_info": int, "pacer_offset": int}
│   │
│   ├── ppg/                       # PPG signal group (75 Hz)
│   │   ├── PPG                    # Photoplethysmogram [900 float32, gzip]
│   │   └── extras                 # JSON: {}
│   │
│   ├── resp/                      # Respiratory signal group (33.33 Hz)
│   │   ├── RESP                   # Respiratory waveform [400 float32, gzip]
│   │   └── extras                 # JSON: {}
│   │
│   ├── vitals/                    # Vital sign measurements
│   │   ├── HR/                    # Heart rate
│   │   │   ├── value              #   int (bpm)
│   │   │   ├── units              #   "bpm"
│   │   │   ├── timestamp          #   epoch float
│   │   │   └── extras             #   JSON: {"upper_threshold", "lower_threshold"}
│   │   ├── Pulse/                 # Pulse rate
│   │   │   ├── value, units, timestamp, extras
│   │   ├── SpO2/                  # Oxygen saturation
│   │   │   ├── value, units, timestamp, extras
│   │   ├── Systolic/              # Systolic blood pressure
│   │   │   ├── value, units, timestamp, extras
│   │   ├── Diastolic/             # Diastolic blood pressure
│   │   │   ├── value, units, timestamp, extras
│   │   ├── RespRate/              # Respiratory rate
│   │   │   ├── value, units, timestamp, extras
│   │   ├── Temp/                  # Temperature
│   │   │   ├── value, units, timestamp, extras
│   │   └── XL_Posture/            # Posture/accelerometer
│   │       ├── value              #   int (degrees)
│   │       ├── units              #   "degrees"
│   │       ├── timestamp          #   epoch float
│   │       └── extras             #   JSON: {"step_count", "time_since_posture_change"}
│   │
│   ├── timestamp                  # Event epoch timestamp (float)
│   ├── uuid                       # Unique event identifier (string)
│   │
│   └── [attributes]               # HDF5 group attributes
│       ├── condition              #   Condition code string (e.g. "AFIB", "N", "VT")
│       ├── heart_rate             #   Heart rate float (bpm)
│       └── event_timestamp        #   Epoch timestamp float
│
├── event_1002/                    # Second alarm event (same structure)
└── event_100N/                    # ...
```

### Schema Details

| Group | Dataset | Type | Shape / Value | Notes |
|-------|---------|------|---------------|-------|
| `metadata/` | `patient_id` | bytes | e.g. `"PT1234"` | Unique patient identifier |
| | `sampling_rate_ecg` | float | `200.0` | ECG sampling frequency (Hz) |
| | `sampling_rate_ppg` | float | `75.0` | PPG sampling frequency (Hz) |
| | `sampling_rate_resp` | float | `33.33` | Respiratory sampling frequency (Hz) |
| | `alarm_time_epoch` | float | epoch | Timestamp of first alarm |
| | `alarm_offset_seconds` | float | `6.0` | Center offset within the 12s window |
| | `seconds_before_event` | float | `6.0` | Pre-event signal duration |
| | `seconds_after_event` | float | `6.0` | Post-event signal duration |
| | `data_quality_score` | float | 0.85–0.98 | Simulated data quality metric |
| | `device_info` | bytes | `"RMSAI-SimDevice-v2.0"` | Source device identifier |
| | `max_vital_history` | int | `30` | Max historical vital samples |
| `event_XXXX/ecg/` | `ECG1`–`vVX` | float32 | `(2400,)` | 7 leads, 12s at 200 Hz, gzip |
| | `extras` | bytes | JSON string | Pacer info and offset |
| `event_XXXX/ppg/` | `PPG` | float32 | `(900,)` | 12s at 75 Hz, gzip |
| `event_XXXX/resp/` | `RESP` | float32 | `(~400,)` | 12s at 33.33 Hz, gzip |
| `event_XXXX/vitals/*/` | `value` | int/float | scalar | Vital sign measurement |
| | `units` | bytes | e.g. `"bpm"` | Unit string |
| | `timestamp` | float | epoch | Measurement time |
| | `extras` | bytes | JSON string | Thresholds or posture metadata |
| `event_XXXX/` | `timestamp` | float | epoch | Event timestamp |
| | `uuid` | string | UUID4 | Unique event ID |
| *(attrs)* | `condition` | string | e.g. `"AFIB"` | Ground truth condition code |
| *(attrs)* | `heart_rate` | float | bpm | Heart rate at event time |
| *(attrs)* | `event_timestamp` | float | epoch | Event timestamp (attribute) |

### Signal Dimensions

| Signal | Sampling Rate | Duration | Samples |
|--------|--------------|----------|---------|
| ECG (7 leads) | 200 Hz | 12s | 2400 |
| PPG | 75 Hz | 12s | 900 |
| Respiratory | 33.33 Hz | 12s | ~400 |

### Vital Signs

| Vital | Units | Typical Range | Extras |
|-------|-------|---------------|--------|
| HR | bpm | 40–180 | upper/lower threshold |
| Pulse | bpm | 40–180 | upper/lower threshold |
| SpO2 | % | 88–100 | upper/lower threshold |
| Systolic | mmHg | 100–180 | upper/lower threshold |
| Diastolic | mmHg | 60–110 | upper/lower threshold |
| RespRate | breaths/min | 12–30 | upper/lower threshold |
| Temp | °F | 96–101 | upper/lower threshold |
| XL_Posture | degrees | -10–45 | step_count, time_since_posture_change |

---

## Model Performance

Three model checkpoints are provided, each trained with different strategies:

### Improved Model (Best) — `models/improved/best_model.pt`

Trained on 16,000 clean samples over 83 epochs (early stopping, patience=20).

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | **90.0%** |
| Macro Precision | 0.896 |
| Macro Recall | 0.895 |
| Macro F1 | 0.893 |
| Macro Specificity | 0.993 |

**Per-condition breakdown:**

| Condition | Precision | Recall | F1 | Support |
|-----------|-----------|--------|----|---------|
| NORMAL_SINUS | 0.593 | 0.762 | 0.667 | 63 |
| SINUS_BRADYCARDIA | 0.943 | 0.943 | 0.943 | 53 |
| SINUS_TACHYCARDIA | 0.983 | 1.000 | 0.992 | 59 |
| ATRIAL_FIBRILLATION | 1.000 | 1.000 | 1.000 | 63 |
| ATRIAL_FLUTTER | 1.000 | 1.000 | 1.000 | 61 |
| PAC | 0.824 | 0.808 | 0.816 | 52 |
| SVT | 1.000 | 1.000 | 1.000 | 50 |
| PVC | 1.000 | 1.000 | 1.000 | 62 |
| VENTRICULAR_TACHYCARDIA | 1.000 | 1.000 | 1.000 | 73 |
| VENTRICULAR_FIBRILLATION | 1.000 | 1.000 | 1.000 | 57 |
| LBBB | 0.800 | 0.667 | 0.727 | 60 |
| RBBB | 0.750 | 0.750 | 0.750 | 56 |
| AV_BLOCK_1 | 0.684 | 0.619 | 0.650 | 63 |
| AV_BLOCK_2_TYPE1 | 0.790 | 0.831 | 0.810 | 77 |
| AV_BLOCK_2_TYPE2 | 0.776 | 0.731 | 0.752 | 52 |
| ST_ELEVATION | 1.000 | 1.000 | 1.000 | 70 |

**Key observations:**
- 9/16 conditions achieve perfect F1 (1.000): AFib, AFlutter, SVT, PVC, VTach, VFib, Sinus Tachy, ST Elevation
- Most challenging conditions: Normal Sinus (F1=0.667), AV Block 1st (F1=0.650), LBBB (F1=0.727)
- Ventricular and supraventricular arrhythmias are classified with near-perfect accuracy

### Baseline Model — `models/baseline/best_model.pt`

Trained on 4,800 clean samples over 24 epochs.

| Metric | Value |
|--------|-------|
| Validation Accuracy | 87.2% |
| Macro Precision | 0.875 |
| Macro Recall | 0.870 |
| Macro F1 | 0.869 |

### Noise-Robust Model — `models/noise_robust/best_model.pt`

Trained on 16,000 mixed-noise samples (clean/low/medium randomised per sample) over 22 epochs.

| Metric | Value |
|--------|-------|
| Validation Accuracy | 87.4% |
| Macro Precision | 0.866 |
| Macro Recall | 0.876 |
| Macro F1 | 0.860 |

This model is designed for deployment on noisy real-world data where clean signals are not guaranteed.

### Performance by Noise Level

Using the noise-robust model against synthetic data at each noise level:

| Noise Level | Description | Expected Accuracy |
|-------------|-------------|-------------------|
| clean | No noise, pure synthetic waveforms | ~90% |
| low | Mild baseline wander, slight Gaussian | ~88% |
| medium | Moderate wander, EMG bursts, motion artifacts | ~85% |
| high | Heavy noise, frequent artifacts, electrode issues | ~78-82% |
| mixed | Random per-event from clean/low/medium/high | ~85% |

---

## Setup

```bash
# Clone and install
git clone <repository-url>
cd ecgtranscnn
pip install -e ".[dev]"

# Or install dependencies directly
pip install -r requirements.txt
```

**Requirements:**
- Python >= 3.10
- PyTorch >= 2.0
- NumPy >= 1.24
- SciPy >= 1.10 (signal filtering; lazy-loaded, only needed when filter preset is not `none`)
- h5py >= 3.8
- matplotlib >= 3.7
- pyinotify >= 0.9.6 (Linux; for inference processor directory watching)

---

## Training

### Quick test run

```bash
python scripts/train.py --num-train 256 --num-val 64 --epochs 5 --batch-size 32
```

### Full training (recommended)

```bash
python scripts/train.py \
    --num-train 16000 \
    --num-val 3200 \
    --epochs 100 \
    --batch-size 64 \
    --leads all \
    --noise-level clean \
    --distribution balanced
```

### Noise-robust training

```bash
python scripts/train.py \
    --num-train 16000 \
    --num-val 3200 \
    --epochs 100 \
    --batch-size 64 \
    --leads all \
    --noise-level mixed \
    --distribution balanced \
    --output-dir models/noise_robust
```

### Training with preprocessing filters

```bash
python scripts/train.py \
    --num-train 16000 \
    --num-val 3200 \
    --epochs 100 \
    --batch-size 64 \
    --noise-level high \
    --filter-preset default \
    --output-dir models/filtered
```

Checkpoints, training curves, and confusion matrices are saved to the output directory.

### Training Options

| Flag | Default | Description |
|------|---------|-------------|
| `--num-train` | 16000 | Number of training samples |
| `--num-val` | 3200 | Number of validation samples |
| `--leads` | `all` | Comma-separated lead names, or `all` for all 7 leads |
| `--noise-level` | `clean` | Noise preset: clean, low, medium, high, mixed |
| `--filter-preset` | `none` | Preprocessing filter preset: none, default, conservative, aggressive |
| `--distribution` | `balanced` | Training data distribution: balanced or mit_bih |
| `--cache-dir` | `data/training_cache` | Cache directory for generated datasets |
| `--test-dir` | — | Directory with HDF5 test files for post-training evaluation |
| `--epochs` | 100 | Maximum training epochs |
| `--batch-size` | 64 | Batch size |
| `--lr` | 5e-4 | Learning rate |
| `--patience` | 20 | Early stopping patience |
| `--output-dir` | `models` | Output directory for checkpoints and plots |

---

## Inference Pipeline

The inference pipeline has two components: a **data generator** that drops HDF5 files into a watched directory, and a **processor** that picks up new files, runs the model, and prints results.

### Start the Processor

```bash
# Watch for new files (runs until Ctrl+C)
python scripts/processor.py \
    --watch-dir data/inference \
    --checkpoint models/noise_robust/best_model.pt

# Also process files already in the directory
python scripts/processor.py \
    --watch-dir data/inference \
    --checkpoint models/noise_robust/best_model.pt \
    --process-existing
```

### Drop Files in Another Terminal

```bash
python scripts/generate_inference_data.py \
    --num-files 5 \
    --events-per-file 5 \
    --output-dir data/inference \
    --delay 2
```

### Processor Output

```
╔══════════════════════════════════════════════════════════════════╗
║  ECG-TransCovNet Inference Processor                           ║
║  Watching: data/inference    Model: models/noise_robust/...    ║
╚══════════════════════════════════════════════════════════════════╝
  Device: cuda

── PT1234_2026-02.h5 (5 events) ──────────────────────────────────
  Event   Ground Truth                Predicted                   Match    HR   SpO2          BP   RR
  1001    ATRIAL_FIBRILLATION         ATRIAL_FIBRILLATION             T   142    95%      138/88   22
  1002    NORMAL_SINUS                NORMAL_SINUS                    T    72    98%      120/78   16
  ...
  File accuracy: 4/5 (80.0%)

[Ctrl+C]

══ Aggregate Classification Report ══════════════════════════════
  Accuracy: 0.875  (7/8)

  Condition                      Prec    Rec     F1     N
  ─────────────────────────────────────────────────────
  ATRIAL_FIBRILLATION           1.000  1.000  1.000     3
  ...
  Macro F1: 0.860
```

### Processor Options

| Flag | Default | Description |
|------|---------|-------------|
| `--watch-dir` | *(required)* | Directory to monitor for new `.h5` files |
| `--checkpoint` | `models/noise_robust/best_model.pt` | Model checkpoint path |
| `--process-existing` | off | Process files already present on startup |
| `--filter-preset` | `none` | Preprocessing filter preset: none, default, conservative, aggressive |

---

## Data Generation

### `scripts/generate_inference_data.py`

Generates synthetic ECG HDF5 files using the built-in simulator with full control over conditions and noise.

### Condition Selection

```bash
# List all 16 valid condition names
python scripts/generate_inference_data.py --list-conditions

# Random (uniform over all 16) — default
python scripts/generate_inference_data.py --conditions random

# Balanced (equal weight to all 16)
python scripts/generate_inference_data.py --conditions balanced

# Single condition (all events are AFib)
python scripts/generate_inference_data.py --conditions ATRIAL_FIBRILLATION

# Subset (uniform among listed)
python scripts/generate_inference_data.py --conditions ATRIAL_FIBRILLATION,NORMAL_SINUS,PVC

# Weighted proportions (3:1 ratio)
python scripts/generate_inference_data.py --conditions ATRIAL_FIBRILLATION:3,NORMAL_SINUS:1

# Complex weighted mix
python scripts/generate_inference_data.py --conditions ATRIAL_FIBRILLATION:3,NORMAL_SINUS:1,PVC:2,SVT:1
```

### Noise Control

The noise pipeline applies these artifact types in sequence: baseline wander, Gaussian noise, EMG burst, motion artifact, powerline interference, electrode contact degradation.

**Presets:**

| Preset | Baseline Wander | Gaussian Std | EMG Prob | Motion Prob | Powerline Prob | Electrode Prob |
|--------|----------------|--------------|----------|-------------|----------------|----------------|
| `clean` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `low` | 0.05 | 0.05 | 0.10 | 0.05 | 0.10 | 0.05 |
| `medium` | 0.10 | 0.10 | 0.30 | 0.15 | 0.20 | 0.10 |
| `high` | 0.15 | 0.20 | 0.50 | 0.30 | 0.30 | 0.20 |
| `mixed` | *random preset per event from clean/low/medium/high* |||||

```bash
# Noise presets
python scripts/generate_inference_data.py --noise-level clean
python scripts/generate_inference_data.py --noise-level low
python scripts/generate_inference_data.py --noise-level medium
python scripts/generate_inference_data.py --noise-level high
python scripts/generate_inference_data.py --noise-level mixed
```

**Custom noise overrides** (fine-tune individual parameters on top of a preset):

```bash
# Start from medium, crank up Gaussian noise and EMG artifacts
python scripts/generate_inference_data.py --noise-level medium --gaussian-std 0.3 --emg-prob 0.8

# Start from low, add heavy motion artifacts
python scripts/generate_inference_data.py --noise-level low --motion-prob 0.6

# Clean base + only powerline interference
python scripts/generate_inference_data.py --noise-level clean --powerline-prob 0.9

# Full custom noise profile
python scripts/generate_inference_data.py \
    --noise-level high \
    --baseline-wander 0.25 \
    --gaussian-std 0.4 \
    --emg-prob 0.9 \
    --motion-prob 0.6 \
    --powerline-prob 0.5 \
    --electrode-prob 0.4
```

### All Options

| Flag | Default | Description |
|------|---------|-------------|
| `--num-files` | 3 | Number of HDF5 files to generate |
| `--events-per-file` | 5 | Alarm events per file |
| `--output-dir` | `data/inference` | Output directory |
| `--conditions` | `random` | Condition specification (see above) |
| `--noise-level` | `medium` | Noise preset: clean, low, medium, high, mixed |
| `--baseline-wander` | — | Override baseline wander amplitude (mV) |
| `--gaussian-std` | — | Override Gaussian noise std |
| `--emg-prob` | — | Override EMG artifact probability [0-1] |
| `--motion-prob` | — | Override motion artifact probability [0-1] |
| `--powerline-prob` | — | Override powerline interference probability [0-1] |
| `--electrode-prob` | — | Override electrode contact degradation probability [0-1] |
| `--delay` | 0 | Seconds between file drops (simulates real-time) |
| `--seed` | None | Random seed for reproducibility |
| `--list-conditions` | — | Print all valid condition names and exit |

### General HDF5 Generation

For general-purpose HDF5 file creation (single file, more presets):

```bash
python scripts/generate_hdf5.py 10 --condition ATRIAL_FIBRILLATION --noise-level high
python scripts/generate_hdf5.py 20 --balanced --output-dir data/train
python scripts/generate_hdf5.py 30 --mit-bih --seed 42
```

---

## Visualization

### `scripts/visualize_hdf5.py`

Inspect and plot events from HDF5 files — shows 7-lead ECG, PPG, respiratory signal, vitals, and condition label.

### List Events

```bash
python scripts/visualize_hdf5.py data/inference/PT1234_2026-02.h5 --list
```

Output:

```
File: data/inference/PT1234_2026-02.h5
Patient: PT1234
Events: 5

  Event        Condition                       HR   SpO2         BP   RR
  --------------------------------------------------------------------
  1001         ATRIAL_FIBRILLATION            144    97%     149/87   19
  1002         NORMAL_SINUS                    78    99%     125/85   13
  1003         PVC                             83    97%     122/73   20
  ...
```

### Plot Events

```bash
# Interactive display — all events
python scripts/visualize_hdf5.py data/inference/PT1234_2026-02.h5

# Single event
python scripts/visualize_hdf5.py data/inference/PT1234_2026-02.h5 --event 1001

# Multiple specific events
python scripts/visualize_hdf5.py data/inference/PT1234_2026-02.h5 --event 1001 1003 1005

# Save as PNG files
python scripts/visualize_hdf5.py data/inference/PT1234_2026-02.h5 --save-dir plots/

# Save specific event
python scripts/visualize_hdf5.py data/inference/PT1234_2026-02.h5 --event 1002 --save-dir plots/

# ECG leads only (skip PPG and respiratory)
python scripts/visualize_hdf5.py data/inference/PT1234_2026-02.h5 --no-ppg-resp

# ECG only, saved to file
python scripts/visualize_hdf5.py data/inference/PT1234_2026-02.h5 --no-ppg-resp --save-dir plots/
```

### Visualization Options

| Flag | Default | Description |
|------|---------|-------------|
| `file` | *(required)* | Path to HDF5 file |
| `--event` | all | Event ID(s) to plot (e.g. `1001 1003`) |
| `--save-dir` | — | Save PNG files here instead of interactive display |
| `--list` | — | List events and exit (no plotting) |
| `--no-ppg-resp` | off | Only plot ECG leads, skip PPG and respiratory |

---

## Evaluation

### Formal Evaluation with `scripts/evaluate.py`

```bash
# Evaluate on synthetic clean data
python scripts/evaluate.py \
    --checkpoint models/noise_robust/best_model.pt \
    --num-samples 1000 \
    --noise-level clean

# Evaluate on synthetic noisy data
python scripts/evaluate.py \
    --checkpoint models/noise_robust/best_model.pt \
    --num-samples 1000 \
    --noise-level medium

# Evaluate on HDF5 test files
python scripts/evaluate.py \
    --checkpoint models/noise_robust/best_model.pt \
    --test-dir data/test_clean

# Save confusion matrix
python scripts/evaluate.py \
    --checkpoint models/noise_robust/best_model.pt \
    --num-samples 1000 \
    --noise-level clean \
    --output-dir results/
```

### Evaluation Options

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | *(required)* | Model checkpoint path |
| `--test-dir` | — | Directory with HDF5 test files |
| `--num-samples` | 1000 | Synthetic samples to evaluate (if no test-dir) |
| `--noise-level` | `clean` | Noise: clean, low, medium, high, mixed |
| `--filter-preset` | `none` | Preprocessing filter preset: none, default, conservative, aggressive |
| `--batch-size` | 64 | Evaluation batch size |
| `--seed` | 99 | Random seed |
| `--output-dir` | — | Directory for confusion matrix PNG |

---

## Test & Evaluation Sequence

A recommended sequence to systematically evaluate model performance.

### Step 1: Baseline — Clean Data, Balanced Conditions

```bash
python scripts/generate_inference_data.py \
    --num-files 3 --events-per-file 10 \
    --output-dir data/eval_clean \
    --noise-level clean --conditions balanced --seed 42

python scripts/visualize_hdf5.py data/eval_clean/*.h5 --list

python scripts/processor.py \
    --watch-dir data/eval_clean \
    --checkpoint models/noise_robust/best_model.pt \
    --process-existing
# Press Ctrl+C after all files are processed to see aggregate report
```

### Step 2: Noise Robustness — Increasing Noise Levels

```bash
for NOISE in clean low medium high; do
    python scripts/generate_inference_data.py \
        --num-files 2 --events-per-file 15 \
        --output-dir data/eval_${NOISE} \
        --noise-level ${NOISE} --conditions balanced --seed 42

    echo "=== Noise: ${NOISE} (no filter) ==="
    python scripts/processor.py \
        --watch-dir data/eval_${NOISE} \
        --checkpoint models/noise_robust/best_model.pt \
        --process-existing --filter-preset none
    # Ctrl+C after processing completes

    echo "=== Noise: ${NOISE} (default filter) ==="
    python scripts/processor.py \
        --watch-dir data/eval_${NOISE} \
        --checkpoint models/noise_robust/best_model.pt \
        --process-existing --filter-preset default
    # Ctrl+C — compare accuracy vs unfiltered
done
```

### Step 3: Per-Condition Deep Dive

```bash
for COND in ATRIAL_FIBRILLATION VENTRICULAR_TACHYCARDIA NORMAL_SINUS PVC LBBB RBBB; do
    python scripts/generate_inference_data.py \
        --num-files 1 --events-per-file 20 \
        --output-dir data/eval_${COND} \
        --conditions ${COND} --noise-level medium --seed 42

    echo "=== Condition: ${COND} ==="
    python scripts/processor.py \
        --watch-dir data/eval_${COND} \
        --checkpoint models/noise_robust/best_model.pt \
        --process-existing
    # Ctrl+C after processing
done
```

### Step 4: Custom Noise Stress Test

```bash
python scripts/generate_inference_data.py \
    --num-files 2 --events-per-file 10 \
    --output-dir data/eval_custom_noise \
    --noise-level medium --gaussian-std 0.35 --emg-prob 0.8 --motion-prob 0.5 \
    --conditions balanced --seed 42

# Compare clean vs noisy signals visually
python scripts/visualize_hdf5.py data/eval_clean/*.h5 --event 1001 --save-dir results/plots_clean
python scripts/visualize_hdf5.py data/eval_custom_noise/*.h5 --event 1001 --save-dir results/plots_noisy

python scripts/processor.py \
    --watch-dir data/eval_custom_noise \
    --checkpoint models/noise_robust/best_model.pt \
    --process-existing
```

### Step 5: Mixed Noise (Realistic Scenario)

```bash
python scripts/generate_inference_data.py \
    --num-files 5 --events-per-file 10 \
    --output-dir data/eval_mixed \
    --noise-level mixed --conditions random --seed 42

python scripts/processor.py \
    --watch-dir data/eval_mixed \
    --checkpoint models/noise_robust/best_model.pt \
    --process-existing
```

### Step 6: Live Watch Test (Two Terminals)

```bash
# Terminal 1 — start processor
python scripts/processor.py \
    --watch-dir data/eval_live \
    --checkpoint models/noise_robust/best_model.pt

# Terminal 2 — drop files with delay
python scripts/generate_inference_data.py \
    --num-files 5 --events-per-file 5 \
    --output-dir data/eval_live \
    --delay 3 --noise-level medium --conditions balanced
```

### Step 7: Model Comparison

```bash
python scripts/generate_inference_data.py \
    --num-files 3 --events-per-file 15 \
    --output-dir data/eval_compare \
    --noise-level medium --conditions balanced --seed 42

for MODEL in models/baseline/best_model.pt models/improved/best_model.pt models/noise_robust/best_model.pt; do
    echo "=== Model: ${MODEL} ==="
    python scripts/processor.py \
        --watch-dir data/eval_compare \
        --checkpoint ${MODEL} \
        --process-existing
    # Ctrl+C after processing
done
```

### Step 8: Formal Evaluation (Large-Scale)

```bash
python scripts/evaluate.py --checkpoint models/noise_robust/best_model.pt --num-samples 1000 --noise-level clean
python scripts/evaluate.py --checkpoint models/noise_robust/best_model.pt --num-samples 1000 --noise-level medium
python scripts/evaluate.py --checkpoint models/noise_robust/best_model.pt --num-samples 1000 --noise-level high
python scripts/evaluate.py --checkpoint models/improved/best_model.pt --num-samples 1000 --noise-level clean --output-dir results/

# With preprocessing filters — compare filtered vs unfiltered on noisy data
python scripts/evaluate.py --checkpoint models/noise_robust/best_model.pt --num-samples 1000 --noise-level high --filter-preset none
python scripts/evaluate.py --checkpoint models/noise_robust/best_model.pt --num-samples 1000 --noise-level high --filter-preset default
python scripts/evaluate.py --checkpoint models/noise_robust/best_model.pt --num-samples 1000 --noise-level high --filter-preset aggressive
```

### What Each Step Tests

| Step | Purpose |
|------|---------|
| 1 | Baseline accuracy on clean, balanced data |
| 2 | How accuracy degrades across noise levels; filter vs no-filter comparison |
| 3 | Per-condition precision/recall to find weak spots |
| 4 | Robustness to specific artifact types (EMG, motion) |
| 5 | Realistic mixed-noise scenario |
| 6 | End-to-end live pipeline with inotify directory watching |
| 7 | Compare baseline vs improved vs noise-robust models on same data |
| 8 | Large-scale formal evaluation; filtered vs unfiltered on noisy data |

---

## Package Structure

```
ecg_transcovnet/                # Python package
  __init__.py                   # Public API exports
  model.py                      # ECGTransCovNet, SKConv, CNNBackbone, FocalLoss
  preprocessing.py              # FilterConfig, PreprocessingPipeline, preprocess_ecg
  constants.py                  # NUM_CLASSES, CLASS_NAMES, SIGNAL_LENGTH, leads
  data.py                       # Dataset generation, loading, augmentation
  training.py                   # train_one_epoch, validate, evaluate_detailed
  visualization.py              # Plotting utilities (waveforms, confusion matrix, attention)
  simulator/                    # Synthetic ECG signal simulator
    ecg_simulator.py            #   ECGSimulator facade (7-lead ECG + PPG + RESP + vitals)
    hdf5_writer.py              #   HDF5EventWriter (Phase-0 compatible output)
    conditions.py               #   16 cardiac condition definitions
    morphology.py               #   Beat morphology generation (P-QRS-T)
    noise.py                    #   Composable noise pipeline (6 artifact types)

scripts/                        # CLI tools
  train.py                      # Training pipeline (AdamW, cosine LR, focal loss)
  evaluate.py                   # Formal model evaluation with metrics
  generate_hdf5.py              # General HDF5 file generation
  generate_inference_data.py    # Inference data generator (conditions, noise, delay)
  generate_test_data.py         # Per-condition test set generation
  processor.py                  # Inference processor (inotify watcher + model)
  visualize.py                  # Signal/prediction/attention visualization
  visualize_hdf5.py             # HDF5 event inspection and plotting

models/                         # Saved checkpoints
  baseline/best_model.pt        # 87.2% accuracy (4,800 clean samples)
  improved/best_model.pt        # 90.0% accuracy (16,000 clean samples)
  noise_robust/best_model.pt    # 87.4% accuracy (16,000 mixed-noise samples)

tests/                          # Test suite (pytest)
  test_preprocessing.py         # Preprocessing pipeline tests (14 tests)
notebooks/                      # Educational Colab notebook
```

---

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- NumPy >= 1.24
- SciPy >= 1.10
- h5py >= 3.8
- matplotlib >= 3.7
- pyinotify >= 0.9.6 (Linux only; for inference processor)

```bash
pip install -r requirements.txt
```

## Tests

```bash
pytest tests/ -v
```

## License

MIT
