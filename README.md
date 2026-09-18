# ECG-TransCovNet

Hybrid CNN-Transformer for ECG arrhythmia classification from 7-lead signals. The class head is
data-driven: the real-ECG model (`models/real_v2`) predicts **13** rhythm classes, the synthetic
simulator models predict **16**.

Based on: Shah et al., *"ECG-TransCovNet: A hybrid transformer model for accurate arrhythmia detection using Electrocardiogram signals"*, IET CIT 2024.

---

## Table of Contents

- [Architecture](#architecture)
- [Preprocessing](#preprocessing)
- [Cardiac Conditions](#cardiac-conditions)
- [HDF5 Dataset Schema](#hdf5-dataset-schema)
- [Clinical Analysis](#clinical-analysis)
- [Report Generation](#report-generation)
- [Model Performance](#model-performance)
- [Setup](#setup)
- [Training](#training)
- [Inference Pipeline](#inference-pipeline)
- [Data Generation](#data-generation)
- [Visualization](#visualization)
- [Evaluation](#evaluation)
- [Test & Evaluation Sequence](#test--evaluation-sequence)
- [Using This Model From Another Project](#using-this-model-from-another-project)
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

**The class head is data-driven, not fixed.** Each checkpoint stores its own ordered label list in
`class_names`, and every script sizes the model from it — nothing assumes 16 classes. Two heads
exist today:

| Head | Classes | Used by |
|---|---:|---|
| Simulator (`Condition` enum) | 16 | `models/best_model.pt`, `noise_robust/`, `avblock_fix/` |
| Real ECG (`ecgpkg` v2 `package.json`) | 13 | `models/real_v2/` — current |
| Real ECG (`ecgpkg` v1) | 12 | `models/real_v1/` — superseded |

### Real-ECG head — `models/real_v2` (13 classes)

Output index order, exactly as the softmax returns it. F1 is the 5-fold ensemble on the `ecgpkg` v2
test split (3,513 events); see [Model Performance](#model-performance).

| Idx | Label | MIT-BIH code | Category | Test F1 | Package flags |
|---:|---|---|---|---:|---|
| 0 | `NORMAL_SINUS` | N | Normal | 0.873 | — |
| 1 | `SINUS_BRADYCARDIA` | SB | Normal | 0.743 | — |
| 2 | `SINUS_TACHYCARDIA` | ST | Normal | 0.837 | — |
| 3 | `ATRIAL_FIBRILLATION` | AFIB | Supraventricular | 0.753 | — |
| 4 | `ATRIAL_FLUTTER` | AFL | Supraventricular | 0.098 ⚠ | few subjects, one record dominates |
| 5 | `PAC` | A | Supraventricular | 0.551 | — |
| 6 | `SVT` | SVTA | Supraventricular | 0.089 ⚠ | few subjects |
| 7 | `PVC` | V | Ventricular | 0.833 | — |
| 8 | `VENTRICULAR_TACHYCARDIA` | VT | Ventricular | 0.421 ⚠ | few subjects, one record dominates |
| 9 | `VENTRICULAR_FIBRILLATION` | VF | Ventricular | *0.840* ⚠ | **no seven-real-lead events** |
| 10 | `LBBB` | L | Bundle branch | 0.531 | lead-realism skew |
| 11 | `RBBB` | R | Bundle branch | 0.920 | lead-realism skew |
| 12 | `AV_BLOCK_1` | 1AVB | AV block | 0.394 | — |

**The primary metric excludes VF**, not the other flagged classes: macro-F1 over 12 of 13 classes.
VF's 0.840 is measured entirely on **fabricated-lead data** — no VF event in the package has seven
measured leads, in any split — so it cannot be read as deployment performance. The other flags mean
the estimate is noisy, not invalid, so those classes stay in the average with a subject-level CI.

⚠ **Not usable to rule a rhythm out.** ATRIAL_FLUTTER recall 0.062 (6 test subjects, one recording
supplying most events), SVT recall 0.080 with **AUROC 0.632 — near-random ranking**, VT recall 0.462.
A missing AFL, SVT or VT prediction carries no information.

**Three `Condition` members are absent from this head**, for two different reasons:

- `AV_BLOCK_2_TYPE1` and `AV_BLOCK_2_TYPE2` are **permanently undetectable from these sources**:
  MIT-BIH's `(BII` note and PTB-XL's `2AVB` code both mean *Mobitz type unknown*, and separating
  Wenckebach from Mobitz II needs beat-to-beat PR-interval measurement the annotations do not
  carry. More subjects cannot fix this.
- `ST_ELEVATION` is a **data shortage**: 26 eligible events against the 140 the thresholds need.

The model can never predict those three. Ground truth carrying one is reported as
`n/a (not in head)`, still receives a prediction, and is excluded from accuracy
(`ecg_transcovnet.classes.NOT_IN_HEAD`).

**Label provenance.** Accuracy tracks how a label was derived (`label_method`), and the two VT
routes are not the same clinical statement: `beat_run` is a ≥ 3-beat run at > 100 bpm from audited
beat annotations, possibly non-sustained; `rhythm_annotation` is an adjudicated episode, typically
sustained. The v2 ensemble scores VT F1 **0.828 on `beat_run` against 0.425 on
`rhythm_annotation`** — the sustained episodes are the hard ones.

A `NORMAL_SINUS` label from `record_level` means "this patient's ECG was reported as normal", while
one from `beat_morphology` means "the beats in this window are normal" — not interchangeable claims.

### Simulator head (16 classes)

The synthetic simulator generates all 16 `Condition` members (MIT-BIH annotation codes):

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

Simulator-trained checkpoints score 10–26 % accuracy on real ECG and must not be used on real
recordings — see [Model Performance](#model-performance).

**Single-label, not multi-label.** The head is a softmax over mutually exclusive classes: every
window gets exactly one prediction, and probabilities sum to 1. A recording that is both AF and
RBBB can only be reported as one of them.

---

## HDF5 Dataset Schema

Each generated file follows the naming convention `PatientID_YYYY-MM.h5` and contains a global metadata group plus one or more event groups. Every vital sign carries a **history array** of time-stamped samples that record the trend leading up to the current value. Paced events include **pacer metadata** in the ECG extras, and each alarm-capable vital includes an **`alarm_enabled`** flag.

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
│   │   └── extras                 # JSON (see ECG Extras below)
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
│   │   │   └── extras             #   JSON (see Vitals Extras below)
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
│   │       └── extras             #   JSON (see Vitals Extras below)
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
| | `max_vital_history` | int | `30` | Max historical vital samples per vital |
| `event_XXXX/ecg/` | `ECG1`–`vVX` | float32 | `(2400,)` | 7 leads, 12s at 200 Hz, gzip |
| | `extras` | bytes | JSON string | Pacer info and offset (see ECG Extras below) |
| `event_XXXX/ppg/` | `PPG` | float32 | `(900,)` | 12s at 75 Hz, gzip |
| `event_XXXX/resp/` | `RESP` | float32 | `(~400,)` | 12s at 33.33 Hz, gzip |
| `event_XXXX/vitals/*/` | `value` | int/float | scalar | Current vital sign measurement |
| | `units` | bytes | e.g. `"bpm"` | Unit string |
| | `timestamp` | float | epoch | Measurement time |
| | `extras` | bytes | JSON string | Thresholds, alarm flag, history (see Vitals Extras below) |
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

### ECG Extras JSON

The `ecg/extras` dataset is a JSON string containing pacer metadata:

```json
{
  "pacer_info": 23042,
  "pacer_offset": 302
}
```

| Field | Type | Description |
|-------|------|-------------|
| `pacer_info` | int | Bit-packed pacer descriptor (0 = no pacer). See decoding below. |
| `pacer_offset` | int | Sample index within the 2400-sample ECG where the pacer fires. Convert to seconds: `pacer_offset / 200.0`. |

**Decoding `pacer_info`**:

The integer packs four bytes: `type | rate<<8 | amplitude<<16 | flags<<24`.

```python
pacer_type = pacer_info & 0xFF          # 0=None, 1=Single, 2=Dual, 3=Biventricular
pacer_rate = (pacer_info >> 8) & 0xFF   # pacing rate in bpm (60–100)
pacer_amp  = (pacer_info >> 16) & 0xFF  # amplitude (1–10)
pacer_flags = (pacer_info >> 24) & 0xFF # reserved flags (0–15)
```

| Type code | Pacer type |
|-----------|------------|
| 0 | None (no pacer) |
| 1 | Single chamber |
| 2 | Dual chamber |
| 3 | Biventricular |

**Condition-specific pacer offset**: VT/VF and Bradycardia events use bimodal offset placement — early (10–25%) or late (75–90%) in the signal window with 50/50 probability. All other conditions use a uniform 20–80% range.

**Condition-specific pacer probability**: VT/VF events have ~40% chance of a pacer being present; Bradycardia has ~80% chance; all other conditions have ~5% chance.

### Vital Signs

| Vital | Units | Typical Range | History Interval | MEWS Scored |
|-------|-------|---------------|-----------------|-------------|
| HR | bpm | 40–180 | 60–300s | Yes |
| Pulse | bpm | 40–180 | 60–300s | No |
| SpO2 | % | 88–100 | 30–180s | Yes |
| Systolic | mmHg | 100–180 | 120–1800s | Yes |
| Diastolic | mmHg | 60–110 | 120–1800s | No (plotted) |
| RespRate | breaths/min | 12–30 | 60–600s | Yes |
| Temp | °F | 96–101 | 300–3600s | Yes |
| XL_Posture | degrees | -10–45 | 10–60s | No |

### Vitals Extras JSON

Each vital's `extras` dataset is a JSON string with structure varying by vital type.

**Standard vitals** (HR, Pulse, SpO2, Systolic, Diastolic, RespRate, Temp):

```json
{
  "upper_threshold": 100,
  "lower_threshold": 60,
  "alarm_enabled": true,
  "history": [
    {"value": 75.2, "timestamp": 1741816800.0},
    {"value": 74.8, "timestamp": 1741816860.0}
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `upper_threshold` | number | Upper alarm threshold for this vital. |
| `lower_threshold` | number | Lower alarm threshold for this vital. |
| `alarm_enabled` | bool | Whether alarms are active for this vital (always `true` for standard vitals). |
| `history` | array | Time-ordered historical samples (see Vital History below). |

**XL_Posture** (no alarm thresholds):

```json
{
  "step_count": 142,
  "time_since_posture_change": 1200,
  "history": [
    {"value": 14, "timestamp": 1741816800.0},
    {"value": 7, "timestamp": 1741816810.0}
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `step_count` | int | Pedometer count since last reset. |
| `time_since_posture_change` | int | Seconds since last posture change. |
| `history` | array | Time-ordered historical samples. |

### Vital History

Each vital carries up to `max_vital_history` (default 30) historical samples in its `extras.history` array. Samples are ordered ascending by timestamp and represent the trend leading up to the current `value`.

- **Timestamps** are epoch floats; intervals vary by vital type (e.g. HR samples every 1–5 min, Temp samples every 5–60 min)
- **Values** interpolate from a condition-dependent baseline toward the current value with jitter, simulating realistic monitor trends
- History is used by the MEWS history scorer (`compute_mews_history`) and per-event vitals plots
- The `--verify-history` flag on `generate_hdf5.py` validates history integrity (sort order, sample count, range bounds)

### Reading Pacer Data (Example)

```python
import h5py, json

hf = h5py.File("PT1234_2026-03.h5", "r")
ecg_extras = json.loads(hf["event_1001/ecg/extras"][()].decode("utf-8"))

pi = ecg_extras.get("pacer_info", 0)
pacer_type   = pi & 0xFF            # 0=None, 1=Single, 2=Dual, 3=Biventricular
pacer_rate   = (pi >> 8) & 0xFF     # bpm
pacer_offset = ecg_extras.get("pacer_offset", 0)
pacer_time_s = pacer_offset / 200.0  # seconds into the 12s window

if pacer_type > 0:
    names = {1: "Single", 2: "Dual", 3: "Biventricular"}
    print(f"Pacer: {names[pacer_type]} chamber @ {pacer_rate} bpm (offset {pacer_time_s:.1f}s)")
```

---

## Clinical Analysis

The inference pipeline includes automated clinical analysis for each event, implemented across three modules:

### MEWS Scoring (`ecg_transcovnet/mews.py`)

Modified Early Warning Score with SpO2 replacing AVPU. Five components are scored 0–3 each:

| Component | Score 0 | Score 1 | Score 2 | Score 3 |
|-----------|---------|---------|---------|---------|
| Heart Rate | 51–100 | 101–110 | 41–50 or 111–130 | <40 or >130 |
| Systolic BP | 101–200 | 81–100 | 71–80 or >200 | <70 |
| Resp Rate | 9–14 | 15–20 | <9 or 21–29 | >=30 |
| Temperature | 35.0–38.4°C | 38.5–39.0°C | <35.0 or >39.0°C | — |
| SpO2 | >=94% | 90–93% | 85–89% | <85% |

**Risk levels**: Low (0–2), Medium (3–4), High (5–6), Critical (>6)

#### History-Based MEWS (`compute_mews_history`)

In addition to single-point MEWS per event, `compute_mews_history()` computes MEWS at every aligned timestamp from vitals history:

1. Collects all unique timestamps from the 5 scored vitals (HR, Systolic, RespRate, Temp, SpO2)
2. Sorts timestamps ascending
3. Forward-fills each vital (at any query time, uses the most recent sample <= that time)
4. At each timestamp where all 5 vitals have at least one prior sample, calls `calculate_mews()`
5. Returns `list[dict]` of `{"timestamp": float, "mews": MEWSResult}` ordered by time

This produces a MEWS trend over time for each event, enabling early deterioration detection.

### Trend Analysis

`assess_event_trends()` computes per-vital Mann-Kendall trends from each event's own history samples. Trends are classified as "improving", "deteriorating", or "stable" based on statistical significance (p < 0.05).

### ECG-Vital Correlations

`correlate_ecg_vitals()` generates rule-based clinical notes:

- VT with hypoxemia (SpO2 < 90%) — immediate intervention
- VF detected — initiate ACLS protocol
- Bradycardia with hypotension (HR < 50, SBP < 90)
- Tachycardia with desaturation (HR > 130, SpO2 < 92%)
- AFib with rapid ventricular response (HR > 120)
- High MEWS (>= 5) — escalate care

---

## Report Generation

### Markdown Reports (`ecg_transcovnet/report.py`)

Each processed HDF5 file produces a markdown report (`report-{patient_id}-{alarm_id}.md`) with:

1. **Metadata table** — file, patient ID, alarm ID, event count, generation timestamp
2. **Per-event clinical analysis**, each containing:
   - **ECG Plots** — 7-lead ECG waveform (with pacer marker if paced)
   - **ECG table** — ground truth, prediction, probability, match
   - **Vitals at Event** — MEWS component breakdown (Component, Value, Score)
   - **Threshold Status** — each vital vs. alarm thresholds (normal / above / below)
   - **Vitals Trend Plots** — vitals history and MEWS history plots
   - **Vital Sign Trends** — per-event Mann-Kendall slope, direction, p-value
   - **Care Guidance** — clinical action items (when critical patterns detected)

### Per-Event Plots (`ecg_transcovnet/plots.py`)

Each event generates three plot types:

| Plot | Description | Filename |
|------|-------------|----------|
| **ECG** | All 7 leads (ECG1, ECG2, ECG3, aVR, aVL, aVF, vVX) as subplots | `{patient}-{alarm}_ecg_{event_id}.png` |
| **Vitals** | 5 subplots (HR, SpO2, BP with Diastolic overlay, RespRate, Temp) from history | `{patient}-{alarm}_vitals_{event_id}.png` |
| **MEWS History** | MEWS score over time with risk-band shading (green/gold/orange/red) | `{patient}-{alarm}_mews_{event_id}.png` |

Plots are embedded in the markdown report as image references under each event's `#### Plots` section.

```bash
# Generate data with vitals history and run full pipeline
python scripts/generate_hdf5.py 5 --seed 42 --output-dir data/inference --verify-history

timeout 20 python scripts/processor.py \
    --watch-dir data/inference \
    --checkpoint models/noise_robust/best_model.pt \
    --process-existing \
    --plot-dir data/inference/plots
```

Output:
```
  Plots: 15 saved to data/inference/plots/
  Report: data/inference/report-PT4210-2026-03.md
```

---

## Model Performance

### Real ECG — current model (`ecgpkg` v2 test split)

`models/real_v2`, a 5-fold cross-validation ensemble on the v2 test split (3,513 events from unseen
subjects, 13-class head). **Primary metric: macro-F1 over 12 of 13 classes**, excluding
VENTRICULAR_FIBRILLATION, whose score is measured only on fabricated-lead data. Details and the
cross-validation protocol: `docs/real-data-training.md` → "Package v2".

| Model | Accuracy | Primary macro-F1, 12 cls (95 % CI) | All-class macro-F1 | Macro recall | Macro AUROC |
|---|---:|---|---:|---:|---:|
| Simulator checkpoints (baseline mode, v1 test) | 0.10–0.26 | — | 0.09–0.18 | — | — |
| **`models/real_v2` 5-fold ensemble** | **0.782** | **0.587 (0.506–0.684)** | 0.606 | 0.653 | 0.942 |
| `models/real_v2` ensemble, full length | 0.790 | 0.593 | — | — | — |

Cross-validation (the selection signal, over train+val subjects): mean **0.617 ± 0.069** across the
five folds, range 0.521–0.705. Test sits 0.030 below the CV mean, inside half a fold-sd — expected,
since the folds and the test split are disjoint subject sets.

**Per real-lead mask — the deployment configuration is now the strongest subset:**

| Real-lead mask | Events | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| `1111111` (7 measured — what the monitor sends) | 2,441 | **0.826** | **0.715** |
| `0100001` (ECG2 + V1) | 728 | 0.659 | 0.484 |
| `0100000` (ECG2 only) | 344 | 0.727 | 0.464 |

Lead-conversion counterfactual flip rate 0.070 / 0.167, so the lead-realism shortcut stays
suppressed by `--lead-fab-aug-prob 0.5`.

**What the v2 package fixed, in one number.** On the v1 package, 74 of 111 ventricular-tachycardia
test events were predicted as PVC — VT was effectively unavailable. On v2 that is **6 of 39**. The
change came from ecg_sigma's split repair, not from the model: v1 trained VT on 12 %-measured-lead
VFDB episodes and tested it on 78 %-measured-lead INCART beat-runs, and v2 stratified by
`(condition, dataset, label_method)` so both sides see the same mixture. VT is still the weakest
ventricular class (recall 0.462), but its errors now go to AF and VF rather than collapsing into PVC.

**Caveats that travel with these numbers.** Fold-to-fold sd is 0.069, so differences below ~0.07
between configurations are not resolvable on this data. Accuracy on the 86 paced test events is
0.602 against 0.786 unpaced, across 3 patients. The degradation is specific to paced atrial
fibrillation (recall 0.200 on 35 events) — paced PVC is fine at 0.902, above its overall 0.804.
SVT has AUROC 0.632 — its ranking is near-random. Report files: `models/real_v2/reports/test.{md,json}`, `models/real_v2/cv_summary.json`.

### Real ECG — superseded v1 results (`ecgpkg` v1 test split)

Kept for provenance; **not comparable** to the v2 table above (12-class head, different splits,
different primary metric). `models/real_v1` 3-seed ensemble: accuracy 0.814, macro-F1 over all 12
classes 0.668 (0.610–0.777); best single seed 0.798 / 0.667. Full per-class tables, the a–h
experiment series and the lead-realism analysis: `docs/real-data-training.md`.

### Simulator checkpoints (synthetic validation data)

Three model checkpoints are provided, each trained with different strategies:

### Improved Model (Best) — `models/best_model.pt`

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

### Baseline Model — *(not retained in this checkout)*

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

### Training on real ECG (`ecgpkg` packages)

Real recordings come from ecg_sigma as an `ecgpkg` v1 training package (contract:
`CONTRACT_ecgpkg_v1.md`): a manifest with subject-grouped train/val/test splits, the
unchanged ecg_sigma HDF5 files, and `package.json` with the class head.

```bash
python scripts/train.py --data-source package \
    --package ../ecg_sigma/packages/ecg_pkg_v2.2 \
    --output-dir models/experiments/my_run \
    --epochs 40 --patience 10 --time-budget-min 9 --resume
# Rerun the same command until it prints the test report: each call stops cleanly
# between epochs before the time budget, and --resume continues from last.pt.
```

The recipe that produced `models/real_v2` — one run per cross-validation fold, 40 epochs with
patience 10, ~1–3 h each on an RTX 4050. `--cv-fold N` fits on the other folds, selects on N, and
never touches the test split:

```bash
python scripts/train.py --data-source package \
    --package ../ecg_sigma/packages/ecg_pkg_v2.2 \
    --cv-fold 0 --output-dir models/experiments/v2_cv_fold0 \
    --init-checkpoint models/avblock_fix/best_model.pt --init-queries reinit \
    --crop-len 2000 --filter-preset default \
    --lead-fab-aug-prob 0.5 --noise-aug-prob 0 \
    --sampler shuffle --class-weights auto \
    --epochs 40 --warmup-epochs 3 --patience 10 --workers 14 --seed 42
```

Repeat for folds 1–4, then evaluate the five checkpoints together as one ensemble. Select on the
**mean** held-out score: the fold-to-fold sd is 0.069, so a single fold says little.

- **Data-driven head.** The model predicts exactly the classes in `package.json`
  (12 in v1) in that order; the list is stored in the checkpoint as `class_names`.
  Nothing assumes 16 classes: processor, validation suite, evaluation and visualisation
  size the model from the checkpoint.
- **Memory.** On first use each split is cached as a memory-mapped array under
  `data/training_cache/<package_version>_<split>_<leads>.npy` (v1: 1.7 GB on disk; building
  it takes ~1 min with 16 processes and never loads a split into RAM).
- **Windows.** 12 s sources (2400 samples) and PTB-XL (2000 samples) are mixed by training on
  random 2000-sample crops; evaluation reports centre-crop 2000 and full length. Inference on
  12 s device windows uses the full 2400 samples.
- **Filter preset.** Package runs default to `--filter-preset default` (package signals are
  already band-passed; re-filtering is harmless and matches unfiltered device input). The
  preset is saved in the checkpoint and used by `processor.py` unless overridden.
- **Real-lead caveat.** Only INCART and PTB-XL have seven measured leads. MIT-BIH measures ECG2
  and V1; VFDB, CUDB and AFDB measure only ECG2 and every other lead is fabricated. VF,
  most VT and most atrial-flutter windows come from those single-lead sources, so a model can
  learn "fabricated leads ⇒ ventricular arrhythmia". Evaluate with `scripts/evaluate.py`
  (per real-lead-mask tables and the lead-conversion counterfactual) and consider
  `--lead-fab-aug-prob`, which rebuilds the non-ECG2 leads of all-real training windows
  with ecg_sigma's own rules so that pattern stops predicting the class. **Use it**:
  `--lead-fab-aug-prob 0.5` cut the lead-conversion flip rate from 0.41 to 0.09 and raised test
  accuracy from 0.699 to 0.799 — the largest single effect measured (`docs/real-data-training.md`).
  Training with `--fabricated-leads zero` instead is not a substitute: that model then cannot use
  the fabricated leads it will be given at inference (0.573 macro-F1).
- **Selection.** Validation macro-F1 is a noisy selector on this package — val has 3
  atrial-flutter and 6 VF subjects — and three seeds of one configuration span 0.034 macro-F1 on
  test. Run 3 seeds before believing a difference, and skip `--calibrate-on val` (it cost
  0.04–0.05 macro-F1 in every test here).
- **Speed.** Mixed precision is off for package runs (FP32 is ~2× faster for this model on the
  RTX 4050); DataLoader workers do cropping, augmentation and filtering (~4–8 ms per item).

| Flag (package runs) | Default | Description |
|------|---------|-------------|
| `--data-source` | `sim` | `sim` (simulator) or `package` |
| `--package` | — | Path to the `ecgpkg` directory |
| `--crop-len` | 2000 | Training/validation crop length (samples) |
| `--noise-aug-prob` | 0.5 | Probability of simulator artefact injection, scaled to each lead's amplitude |
| `--lead-fab-aug-prob` | 0.0 | Probability of rebuilding non-ECG2 leads from ECG2 (lead-realism augmentation) |
| `--fabricated-leads` | `keep` | `zero` blanks every lead that is not measured |
| `--sampler` | `shuffle` | `balanced`: equal class mass, prolific subjects damped (`--balance-beta`) |
| `--class-weights` | `auto` | Focal-loss weights from train counts (`auto`: inverse unless balanced sampler) |
| `--select-metric` | `macro_f1` | Checkpoint selection on validation macro-F1 or accuracy |
| `--init-checkpoint` / `--init-queries` | — / `by_name` | Warm start; copy object queries of shared classes or re-initialise |
| `--resume` / `--time-budget-min` | off / — | Resumable training in time-boxed chunks |
| `--workers` / `--amp` | 12 / `off` | DataLoader workers / mixed precision |

### Training Options

| Flag | Default | Description |
|------|---------|-------------|
| `--num-train` | 16000 | Number of training samples |
| `--num-val` | 3200 | Number of validation samples |
| `--leads` | `all` | Comma-separated lead names, or `all` for all 7 leads |
| `--noise-level` | `clean` | Noise preset: clean, low, medium, high, mixed |
| `--filter-preset` | `none` (sim) / `default` (package) | Preprocessing filter preset: none, default, conservative, aggressive |
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

# With per-event plots and reports
python scripts/processor.py \
    --watch-dir data/inference \
    --checkpoint models/noise_robust/best_model.pt \
    --process-existing \
    --plot-dir data/inference/plots

# Real ECG: the recommended real_v2 ensemble (softmax averaged over five checkpoints)
python scripts/processor.py \
    --watch-dir data/inference \
    --checkpoint models/real_v2/fold0.pt models/real_v2/fold1.pt models/real_v2/fold2.pt \
                 models/real_v2/fold3.pt models/real_v2/fold4.pt \
    --process-existing
```

### Ensemble Inference (recommended for real ECG)

`--checkpoint` accepts several paths. Their softmax outputs are averaged, which is the
combination rule `scripts/evaluate.py` uses, so live predictions match the offline report. The
`models/real_v2` ensemble scores 0.782 accuracy / 0.587 primary macro-F1 on the v2 test split — see [Model Performance](#model-performance).

```bash
# Ensemble + per-event plots and markdown reports
python scripts/processor.py \
    --watch-dir data/inference \
    --checkpoint models/real_v2/fold0.pt \
                 models/real_v2/fold1.pt \
                 models/real_v2/fold2.pt \
                 models/real_v2/fold3.pt \
                 models/real_v2/fold4.pt \
    --process-existing \
    --plot-dir data/inference/plots
```

The banner names the ensemble instead of a single path, and the head, leads and filter preset
come from its members:

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  ECG-TransCovNet Inference Processor                                         ║
║  Watching: data/inference   Model: ...ensemble (models/real_v2/fold0.pt, ...) ║
╚══════════════════════════════════════════════════════════════════════════════╝
  Device: cuda
  Head: 13 classes · leads 7 · filter preset: default

── PT2901_2026-02.h5 (15 events) ───────────────────────────────────────────
  Event   Ground Truth                Predicted                   Match    HR   SpO2          BP   RR
  1002    PAC                         ATRIAL_FIBRILLATION             F    72    96%      114/82   14
```

Everything downstream is unchanged — prediction, confidence, MEWS, plots and reports all read the
averaged probabilities. Requirements and costs:

- **Members must agree** on class head, leads and filter preset; `load_models` raises a clear
  error otherwise (mixing a 16-class simulator checkpoint with a 12-class real-data one fails).
- **One forward pass per member**, so a 3-model ensemble is ~3× the inference cost. Per event that
  is still milliseconds; use `models/real_v2/best_model.pt` if you need single-model latency.
- **Do not add `--calibrate-on val`** when evaluating these checkpoints — the val-fitted logit bias
  costs 0.04–0.05 macro-F1 (`docs/real-data-training.md`, run h).

The same flag works for offline evaluation:

```bash
python scripts/evaluate.py \
    --checkpoint models/real_v2/fold0.pt \
                 models/real_v2/fold1.pt \
                 models/real_v2/fold2.pt \
                 models/real_v2/fold3.pt \
                 models/real_v2/fold4.pt \
    --package ../ecg_sigma/packages/ecg_pkg_v2 --split test \
    --output-dir reports/real_v2_ensemble
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

── PT4210_2026-03.h5 (5 events) ──────────────────────────────────
  Event   Ground Truth                Predicted                   Match    HR   SpO2          BP   RR
  1001    AV_BLOCK_1                  NORMAL_SINUS                    F    75    96%      110/73   17
  1002    AV_BLOCK_2_TYPE2            AV_BLOCK_2_TYPE2                T    66    96%      125/80   15
  1003    ATRIAL_FIBRILLATION         ATRIAL_FIBRILLATION             T    89    97%      120/95   19
  ...
  File accuracy: 4/5 (80.0%)
  Plots: 15 saved to data/inference/plots/
  Report: data/inference/report-PT4210-2026-03.md

[Ctrl+C]

══ Aggregate Classification Report ══════════════════════════════
  Accuracy: 0.800  (4/5)

  Condition                      Prec    Rec     F1     N
  ─────────────────────────────────────────────────────
  ATRIAL_FIBRILLATION           1.000  1.000  1.000     1
  ...
  Macro F1: 0.860
```

When `--plot-dir` is specified, the processor generates 3 plots per event (ECG all leads, vitals history, MEWS history) and a markdown report with embedded plot references.

### Processor Options

| Flag | Default | Description |
|------|---------|-------------|
| `--watch-dir` | *(required)* | Directory to monitor for new `.h5` files |
| `--checkpoint` | `models/noise_robust/best_model.pt` | Model checkpoint path. Several paths average their softmax outputs as an ensemble; members must share head, leads and filter preset |
| `--process-existing` | off | Process files already present on startup |
| `--filter-preset` | checkpoint's preset | Preprocessing filter preset: none, default, conservative, aggressive (legacy checkpoints record none) |
| `--plot-dir` | — | Directory for per-event plots (ECG, vitals, MEWS). If omitted, no plots are created |

Ensembling costs one forward pass per member and changes nothing else: the averaged
probabilities drive the same prediction, confidence, report and plot paths, and the rule matches
`evaluate.py`, so offline and live numbers agree.

The processor reads simulator files (enum-value conditions, `/metadata` datasets) and
ecg_sigma files (enum-name conditions, `/metadata` attributes). The class head comes from the
checkpoint: a ground truth outside the head (e.g. `SVT` for a 12-class real-data model, or
`OTHER`) is printed as `n/a (not in head)`, still gets a prediction, is excluded from metrics,
and triggers one warning per label.

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

# Generate with vitals history verification
python scripts/generate_hdf5.py 5 --seed 42 --output-dir data/inference --verify-history
```

The `--verify-history` flag validates that all vitals history arrays are correctly sorted by timestamp, within expected ranges, and have the expected sample count.

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
| `--checkpoint` | *(required)* | Model checkpoint path(s); several paths = softmax-averaged ensemble |
| `--test-dir` | — | Directory with HDF5 test files |
| `--num-samples` | 1000 | Synthetic samples to evaluate (if no test-dir) |
| `--noise-level` | `clean` | Noise: clean, low, medium, high, mixed |
| `--filter-preset` | checkpoint's preset | Preprocessing filter preset: none, default, conservative, aggressive |
| `--batch-size` | 64 | Evaluation batch size |
| `--seed` | 99 | Random seed |
| `--output-dir` | — | Directory for confusion matrix PNG / package reports |

### Evaluating on a real-data package

```bash
# Real-data checkpoint on the current test split
python scripts/evaluate.py --checkpoint models/real_v2/best_model.pt \
    --package ../ecg_sigma/packages/ecg_pkg_v2.2 --split test --output-dir reports/real_v2

# Legacy 16-class checkpoint: baseline mode is automatic (names mapped into the package
# head; predictions outside the head count as wrong)
python scripts/evaluate.py --checkpoint models/noise_robust/best_model.pt \
    --package ../ecg_sigma/packages/ecg_pkg_v2.2 --output-dir reports/baseline_noise_robust
```

Writes `<split>.json`, `<split>.md` and one confusion-matrix PNG per section:
centre-crop 2000 and full length, plus the **lead-conversion counterfactual** — events with
more measured leads are re-predicted after rebuilding their non-ECG2 leads from ECG2 the way
ecg_sigma fabricates them (`0100001` keeps V1, `0100000` synthesises it). The flip rate and the
change in predicted AF/AFL/VT/VF share show how much the model relies on lead realism rather
than rhythm. Every per-class metric carries event and subject counts; macro-F1 has a
subject-bootstrap 95 % interval; tables are broken down by dataset, label method and real-lead mask.

| Flag (package mode) | Default | Description |
|------|---------|-------------|
| `--package` / `--split` | — / `test` | Package directory and split (val or test) |
| `--crop-len` / `--no-full-length` | 2000 / off | Centre-crop length; skip the full-length section |
| `--lead-conversion` | `both` | `none`, `0100001`, `0100000` or `both` |
| `--calibrate-on` | `none` | `val`: fit a per-class logit bias on val macro-F1 and apply it |
| `--fabricated-leads` | checkpoint's | `keep` or `zero` |
| `--workers` / `--cache-dir` / `--tag` | 8 / `data/training_cache` / split | Loader workers, cache location, report stem |

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

for MODEL in models/best_model.pt models/noise_robust/best_model.pt models/avblock_fix/best_model.pt; do
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
python scripts/evaluate.py --checkpoint models/best_model.pt --num-samples 1000 --noise-level clean --output-dir results/

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

## Using This Model From Another Project

`models/real_v2` is consumable as a library. Everything a caller needs — class head, lead order,
filter preset — travels inside the checkpoint, so the integration contract is small.

### Input contract

| Property | Value |
|---|---|
| Leads | `["ECG1", "ECG2", "ECG3", "aVR", "aVL", "aVF", "vVX"]`, in this order (`vVX` = V1) |
| Sampling rate | 200 Hz |
| Window length | 2000–2400 samples (10–12 s). Trained on 2000-sample crops, validated at both |
| Units | mV |
| Tensor | `float32`, shape `(batch, 7, samples)` |
| Preprocessing | the checkpoint's preset (`default` for `real_v2`), applied by `PreprocessingPipeline` |

The `default` preset is a 0.5 Hz high-pass, 50 and 60 Hz notches, a 40 Hz low-pass, then a
**per-lead z-score**. Feed raw mV in and let the pipeline normalise — do not pre-normalise
yourself. Re-filtering already-filtered signals is harmless and is what the training data saw.

### Minimal integration

```python
import numpy as np, torch
from ecg_transcovnet import FILTER_PRESETS
from ecg_transcovnet.checkpoint import load_models
from ecg_transcovnet.preprocessing import PreprocessingPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# One checkpoint, or several to ensemble them (recommended)
loaded = load_models([
    "models/real_v2/fold0.pt",
    "models/real_v2/fold1.pt",
    "models/real_v2/fold2.pt",
    "models/real_v2/fold3.pt",
    "models/real_v2/fold4.pt",
], device)

labels = list(loaded.class_spec.names)              # 13 labels, output order
pipeline = PreprocessingPipeline(FILTER_PRESETS[loaded.filter_preset])

signal = np.zeros((7, 2400), dtype=np.float32)      # your leads, in loaded.leads order, mV
x = torch.from_numpy(pipeline(signal)).unsqueeze(0).to(device)

with torch.no_grad():
    probs = torch.softmax(loaded.model(x), dim=-1)[0]

idx = int(probs.argmax())
print(labels[idx], float(probs[idx]))               # e.g. ATRIAL_FIBRILLATION 0.93
```

`load_models` returns a `LoadedModel` with `.model`, `.class_spec`, `.leads`, `.filter_preset` and
the raw `.checkpoint`. Read the head from `.class_spec.names` rather than hard-coding thirteen
labels — a future package version may change it.

### Output contract

A softmax over `class_spec.names`, single-label: one prediction per window, probabilities sum to 1.
There is no "unknown" or "other" class and no abstention — an unrecognisable window still produces
a confident-looking distribution. Feeding the ensemble an all-zeros window returns `PVC` at 0.36,
not a shrug. If the caller needs a reject option, threshold on the max probability and calibrate
that threshold on their own data.

### What to tell the other project

Paste this brief into the consuming project's spec or agent prompt:

```text
Use the ECG-TransCovNet real-ECG model at models/real_v2 (from the ecgtranscnn repo).

Input: 7 leads in the order ECG1, ECG2, ECG3, aVR, aVL, aVF, vVX (vVX = V1), 200 Hz, mV,
2000-2400 samples per window, float32, shape (batch, 7, samples).

Load with ecg_transcovnet.checkpoint.load_models([...]) and read the label list from the
returned class_spec.names — do not hard-code labels. Apply PreprocessingPipeline with the
preset the checkpoint reports (filter_preset), feeding raw mV; it z-scores per lead.

Output: softmax over 13 mutually exclusive rhythm classes, one label per window.

Use the 5-fold ensemble (fold0.pt ... fold4.pt); best_model.pt is a single-model fallback.
Cost is 5 forward passes.

Trust these classes: NORMAL_SINUS, RBBB, SINUS_TACHYCARDIA, PVC, ATRIAL_FIBRILLATION,
SINUS_BRADYCARDIA (F1 0.74-0.92). Treat as low-confidence: PAC (0.55), LBBB (0.53, recall
0.37), AV_BLOCK_1 (0.39, precision 0.25 — it over-triggers on normal sinus).

Do NOT use the model to rule out atrial flutter (recall 0.06), SVT (recall 0.08, AUROC 0.63 —
its ranking is near-random) or ventricular tachycardia (recall 0.46). A missing AFL, SVT or VT
prediction means nothing.

VENTRICULAR_FIBRILLATION scores 0.84 F1, but every VF event it was trained and tested on has
only 1-2 genuinely measured leads; the rest are reconstructed. VF is NOT validated for a
7-measured-lead monitor. Do not rely on it as an alarm source.

The model cannot predict AV_BLOCK_2_TYPE1, AV_BLOCK_2_TYPE2 (the source annotations cannot
express Mobitz type — permanently undetectable) or ST_ELEVATION (too little data).

Accuracy by how many leads are genuinely measured: 0.826 with all 7 real (the deployment case),
0.659 with ECG2+V1, 0.727 with ECG2 only.

Accuracy drops to 0.60 on paced patients (from 0.79). This is specific to paced atrial
fibrillation, where recall falls to 0.20; paced PVC is unaffected. The model never saw a paced
beat in training.

Do not apply val-fitted logit bias calibration; it costs 0.04-0.05 macro-F1.

This is a research model trained on one package of public datasets, not a medical device. Do
not use it for diagnosis or unsupervised alarms.
```

### Operational notes

- **Fewer real leads cost accuracy.** The model was trained with lead-fabrication augmentation so
  it does not *depend* on lead realism, but genuinely measured leads still carry information:
  0.826 accuracy with 7 real leads, 0.659 with ECG2 + V1, 0.727 with ECG2 only.
- **Paced patients degrade, but only for one class.** 0.602 accuracy on the 88 paced test events
  (3 patients) vs 0.786 unpaced. It is paced *atrial fibrillation* that fails — recall 0.200 on 35
  events against 0.818 overall — while paced PVC scores 0.902, above its own overall 0.804. No paced
  beat appears anywhere in the training split, so this is untrained territory, not a tuning problem.
- **Windows shorter than 2000 or longer than 2400 samples** are untested. Crop or segment upstream.
- **Batch for throughput**: preprocessing is ~3.5 ms per item and dominates single-item latency.
- **Version pinning**: checkpoints record `package_version` and `package_manifest_sha256`, so a
  consumer can assert which training package a model came from.

---

## Package Structure

```
ecg_transcovnet/                # Python package
  __init__.py                   # Public API exports
  model.py                      # ECGTransCovNet, SKConv, CNNBackbone, FocalLoss
  preprocessing.py              # FilterConfig, PreprocessingPipeline (vectorised), preprocess_ecg
  constants.py                  # NUM_CLASSES, CLASS_NAMES, SIGNAL_LENGTH, ALL_LEADS
  classes.py                    # ClassSpec — the data-driven class head, condition name resolution
  checkpoint.py                 # load_model/load_models, EnsembleModel, build_model, warm_start
  package.py                    # ecgpkg loader, PackageDataset, memmap cache, length-bucket sampler
  package_eval.py               # Ensemble prediction, softmax averaging, baseline head mapping
  augment.py                    # Artefact injection, lead fabrication from ECG2
  evaluation.py                 # Metrics, bootstrap CI, grouped breakdowns, report writing
  hdf5_io.py                    # Metadata reading for simulator and ecg_sigma files
  data.py                       # Dataset generation, loading, augmentation
  training.py                   # train_one_epoch, validate, evaluate_detailed
  visualization.py              # Plotting utilities (waveforms, confusion matrix, attention)
  mews.py                       # MEWS scoring, compute_mews_history, trend analysis, correlations
  plots.py                      # Per-event plot generation (ECG, vitals, MEWS history)
  report.py                     # Markdown report generation (EventResult, FileResult, write_report)
  simulator/                    # Synthetic ECG signal simulator
    ecg_simulator.py            #   ECGSimulator facade (7-lead ECG + PPG + RESP + vitals + history)
    hdf5_writer.py              #   HDF5EventWriter (Phase-0 compatible output)
    conditions.py               #   16 cardiac condition definitions
    morphology.py               #   Beat morphology generation (P-QRS-T)
    noise.py                    #   Composable noise pipeline (6 artifact types)

scripts/                        # CLI tools
  train.py                      # Training (simulator or ecgpkg package; resumable)
  evaluate.py                   # Evaluation: package splits, baseline mode, counterfactuals, ensembles
  compute_auc.py                # AUROC computation
  processor.py                  # Inference processor (inotify watcher + model + plots + reports)
  run_validation_suite.py       # Per-condition validation suite
  generate_hdf5.py              # General HDF5 file generation (--verify-history)
  generate_inference_data.py    # Inference data generator (conditions, noise, delay)
  generate_test_data.py         # Per-condition test set generation
  generate_validation_suite.py  # Validation suite data generation
  generate_demo.py              # Demo data generation
  visualize.py                  # Signal/prediction/attention visualization
  visualize_hdf5.py             # HDF5 event inspection and plotting
  plot_real_data_results.py     # Figures for docs/real-data-training.md

models/                         # Saved checkpoints (not tracked by git)
  best_model.pt                 # Simulator, clean — 16-class head
  noise_robust/best_model.pt    # Simulator, mixed noise — 16-class head
  avblock_fix/best_model.pt     # Simulator, AV-block morphology fix — 16-class head
  real_v2/                      # Real ECG (ecgpkg v2) — 13-class head, recommended
    fold0.pt ... fold4.pt           #   5-fold CV ensemble members (the artifact)
    best_model.pt                   #   single-model fallback (copy of fold0.pt)
    cv_summary.json                 #   per-fold scores and the protocol
    reports/ confusion_matrix.png   #   the single ensemble test evaluation
  real_v1/                      # Real ECG (ecgpkg v1) — 12-class head, superseded
  experiments/                  # Per-run outputs for docs/real-data-training.md

tests/                          # Test suite (pytest) — 118 passed, 1 skipped
  test_preprocessing.py         # Filter pipeline
  test_preprocessing_vectorised.py  # Vectorised pipeline equals the per-lead reference
  test_classes.py               # ClassSpec, condition-name resolution
  test_package.py               # ecgpkg contract invariants, loader errors, cache
  test_train_package.py         # Package training smoke run with resume
  test_evaluation.py            # Metrics, grouped breakdowns, bootstrap CI
  test_processor_compat.py      # Processor with legacy and package checkpoints
  test_model.py test_e2e.py test_simulator.py test_noise_robustness.py
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
