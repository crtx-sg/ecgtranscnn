# ECG-TransCovNet: Real-Time Arrhythmia Detection System

Hybrid CNN-Transformer for 16-class ECG classification with clinical decision support

*Vios Medical — March 2026*

---

## Problem

**Challenge**: ICU monitors generate thousands of cardiac alarms daily. 85–99% are false or non-actionable, leading to alarm fatigue and missed critical events.

**What's needed**:
- Accurate multi-class arrhythmia detection from raw ECG signals
- Clinical context — vitals, trends, severity scoring — alongside the classification
- Near-real-time processing as new patient data arrives
- Robustness to noisy, artifact-laden signals typical of bedside monitors

**Scope**: 16 cardiac conditions across 5 categories — Normal rhythms, Supraventricular, Ventricular, Bundle Branch Blocks, AV Blocks, ST changes

---

## Cardiac Conditions (16 Classes)

| Category | Conditions |
|----------|-----------|
| Normal | Normal Sinus, Sinus Bradycardia, Sinus Tachycardia |
| Supraventricular | AFib, Atrial Flutter, PAC, SVT |
| Ventricular | PVC, Ventricular Tachycardia, Ventricular Fibrillation |
| Bundle Branch | LBBB, RBBB |
| AV Block | 1st Degree, 2nd Degree Type 1, 2nd Degree Type 2 |
| Other | ST Elevation |

---

## Architecture

```
Input: 7-lead ECG (7 × 2400 samples, 12s at 200 Hz)
         │
         ▼
┌──────────────────────────────────────────┐
│  CNN Backbone                            │
│  ResBlock (7→32) → SK Conv (32→64)       │  Local feature extraction
│  → ResBlock (64→128) → ResBlock (128→256)│  2400 → 38 feature positions
│  → 1×1 bottleneck (256→128)              │
└──────────────────────────────────────────┘
         │  (B, 128, 38)
         ▼
┌──────────────────────────────────────────┐
│  Transformer Encoder (3 layers, 8 heads) │  Global context modelling
│  + Sinusoidal Positional Encoding        │
└──────────────────────────────────────────┘
         │  (B, 38, 128)
         ▼
┌──────────────────────────────────────────┐
│  Transformer Decoder (3 layers, 8 heads) │  DETR-style classification
│  + 16 Learnable Object Queries           │  One query per condition
└──────────────────────────────────────────┘
         │
         ▼
   FFN Head (128→64→1) × 16 → 16-class logits
```

**Key innovations**: Selective Kernel convolutions (multi-scale), DETR-style object queries, Focal Loss for class imbalance

---

## Dataset & Simulator

**Synthetic data generator** — physiologically realistic ECG waveforms with:
- 7 derived leads (Einthoven + augmented + precordial)
- Condition-specific morphology (P/QRS/T wave parameters, intervals)
- PPG and respiratory signals
- 8 vital signs with 10–30 sample history trends
- Pacer simulation (Single/Dual/Biventricular) with condition-dependent probability
- Configurable noise pipeline: baseline wander, Gaussian, EMG, motion artifacts, powerline, electrode pop

| Dataset | Samples | Noise | Use |
|---------|---------|-------|-----|
| Baseline training | 4,800 | Clean | Quick validation |
| Full training | 16,000 | Clean | Best accuracy |
| Noise-robust training | 16,000 | Mixed (clean/low/medium) | Production |

---

## HDF5 Data Schema

Each file follows `PatientID_YYYY-MM.h5` — a metadata group + N event groups:

```
PatientID_YYYY-MM.h5
├── metadata/
│   ├── patient_id              "PT1234"
│   ├── sampling_rate_ecg       200.0 Hz
│   ├── sampling_rate_ppg       75.0 Hz
│   ├── sampling_rate_resp      33.33 Hz
│   ├── alarm_time_epoch        epoch float
│   ├── data_quality_score      0.85–0.98
│   └── max_vital_history       30
│
└── event_1001/
    ├── ecg/                    7 leads × 2400 samples (12s @ 200 Hz, gzip)
    │   ├── ECG1..ECG3, aVR, aVL, aVF, vVX
    │   └── extras              JSON: pacer_info + pacer_offset
    ├── ppg/PPG                 900 samples (12s @ 75 Hz)
    ├── resp/RESP               ~400 samples (12s @ 33.33 Hz)
    ├── vitals/
    │   ├── HR, Pulse, SpO2, Systolic, Diastolic, RespRate, Temp
    │   │   └── value, units, timestamp, extras (thresholds + alarm_enabled + history[])
    │   └── XL_Posture
    │       └── value, units, timestamp, extras (step_count + posture_change + history[])
    ├── timestamp, uuid
    └── [attrs] condition, heart_rate, event_timestamp
```

**ECG extras** — pacer metadata (bit-packed): `type | rate<<8 | amp<<16 | flags<<24`

| Type | 0=None | 1=Single | 2=Dual | 3=Biventricular |
|------|--------|----------|--------|------------------|
| Probability | ~95% normal | varies | higher in Brady/VT/VF | rare |

**Vitals extras** — per-vital JSON with `upper_threshold`, `lower_threshold`, `alarm_enabled: true`, and `history[]` (10–30 time-stamped samples trending toward current value)

---

## Training

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW, LR 5e-4 |
| Loss | Focal Loss (gamma=2.0, per-class alpha) |
| Epochs | 100 max, early stopping patience=20 |
| Batch size | 64 |
| Validation split | 16,000 train / 3,200 val |
| Augmentation | Per-sample noise randomization (mixed mode) |

```bash
# Full training (best accuracy)
python scripts/train.py --num-train 16000 --epochs 100 --output-dir models/improved

# Noise-robust (production)
python scripts/train.py --num-train 16000 --noise-level mixed --output-dir models/noise_robust
```

---

## Model Performance

| Model | Accuracy | Macro F1 | Macro Precision | Macro Recall |
|-------|----------|----------|-----------------|-------------|
| **Improved (best)** | **90.0%** | **0.893** | 0.896 | 0.895 |
| Noise-robust | 87.4% | 0.860 | 0.866 | 0.876 |
| Baseline | 87.2% | 0.869 | 0.875 | 0.870 |

**Per-condition highlights (Improved model)**:
- **Perfect F1 (1.000)**: AFib, AFlutter, SVT, PVC, VTach, VFib, Sinus Tachy, ST Elevation — 9 of 16 conditions
- **Hardest conditions**: Normal Sinus (0.667), AV Block 1st (0.650), LBBB (0.727) — morphologically subtle differences

---

## Noise Robustness

| Noise Level | Description | Accuracy |
|-------------|-------------|----------|
| Clean | No noise | ~90% |
| Low | Mild baseline wander, slight Gaussian | ~88% |
| Medium | Moderate wander, EMG bursts, motion artifacts | ~85% |
| High | Heavy noise, frequent artifacts, electrode issues | ~78–82% |
| Mixed | Random per-event from all levels | ~85% |

Noise-robust model degrades < 5% vs. clean-trained model on medium noise.

---

## Inference Pipeline

**Architecture**: File watcher (inotify) → HDF5 parser → preprocessing → model inference → clinical analysis → report + plots

```bash
# Terminal 1: Start processor
python scripts/processor.py --watch-dir data/inference \
  --checkpoint models/noise_robust/best_model.pt \
  --process-existing --plot-dir data/inference/plots

# Terminal 2: Drop files
python scripts/generate_inference_data.py --num-files 2 --events-per-file 5 \
  --output-dir data/inference --conditions VENTRICULAR_FIBRILLATION,SINUS_BRADYCARDIA
```

**Console output snapshot**:

```
╔══════════════════════════════════════════════════════════════════╗
║  ECG-TransCovNet Inference Processor                             ║
║  Watching: data/inference        Model: models/noise_robust/best_model.pt║
╚══════════════════════════════════════════════════════════════════╝
  Device: cuda

── PT4305_2026-03.h5 (3 events) ───────────────────────────────────────────
  Event   Ground Truth                Predicted                   Match    HR   SpO2          BP   RR
  1001    VENTRICULAR_FIBRILLATION    VENTRICULAR_FIBRILLATION        T   203    90%     173/107   22
  1002    VENTRICULAR_TACHYCARDIA     VENTRICULAR_TACHYCARDIA         T   120    93%      180/92   28
  1003    PVC                         PVC                             T    87    98%      132/79   17
  File accuracy: 3/3 (100.0%)
  Plots: 9 saved to data/inference/plots/
  Report: data/inference/report-PT4305-2026-03.md

══ Aggregate Classification Report ══════════════════════════════
  Accuracy: 0.833  (5/6)

  Condition                    Prec    Rec     F1      N
  ─────────────────────────────────────────────────────────
  VENTRICULAR_FIBRILLATION    1.000  1.000  1.000      1
  VENTRICULAR_TACHYCARDIA     1.000  1.000  1.000      1
  PVC                         1.000  1.000  1.000      1
  ...
  Macro F1: 0.889
```

---

## Clinical Analysis & MEWS

**Modified Early Warning Score** — 5 components scored 0–3:

| Component | Score 0 | Score 1 | Score 2 | Score 3 |
|-----------|---------|---------|---------|---------|
| Heart Rate | 51–100 | 101–110 | 41–50 or 111–130 | <40 or >130 |
| Systolic BP | 101–200 | 81–100 | 71–80 or >200 | <70 |
| Resp Rate | 9–14 | 15–20 | <9 or 21–29 | >=30 |
| Temperature | 35–38.4°C | 38.5–39°C | <35 or >39°C | — |
| SpO2 | >=94% | 90–93% | 85–89% | <85% |

**Risk levels**: Low (0–2) · Medium (3–4) · High (5–6) · Critical (>6)

**Automated care guidance** (rule-based):
- VF detected → initiate ACLS protocol
- VT with SpO2 < 90% → immediate intervention
- Bradycardia + hypotension → consider atropine/pacing
- AFib with HR > 120 → rate control
- MEWS >= 5 → escalate care

---

## Report & Plot Output

**Per-event markdown report** includes:

| Section | Content |
|---------|---------|
| ECG Plots | 7-lead ECG waveform (with pacer marker if paced) |
| ECG table | Ground truth, prediction, probability, match |
| Vitals at Event | MEWS component breakdown (Component, Value, Score) |
| Threshold Status | Each vital vs. alarm thresholds (normal / above / below) |
| Vitals Trend Plots | Vitals history and MEWS history plots |
| Vital Sign Trends | Per-event Mann-Kendall slope, direction, p-value |
| Care Guidance | Clinical action items (when critical patterns detected) |

**Report excerpt**:

```markdown
#### ECG Plots
**Pacer**: Dual chamber @ 92 bpm (offset 1.5s)
![ECG 7-Lead — Event 1002](plots/ecg_1002.png)

| Event | Ground Truth | Predicted | Prob | Match |
|-------|-------------|-----------|------|-------|
| 1002  | VT          | VT        | 1.000| Yes   |

#### Vitals at Event — MEWS 6 (High)
| Component | Value | Score |
|-----------|-------|-------|
| HR        | 203   | 3     |
| Systolic  | 173   | 0     |
| RespRate  | 22    | 2     |
| Temp      | 99.1  | 0     |
| SpO2      | 90    | 1     |

#### Vital Sign Trends
| Vital | Direction | Slope | p-value | Sig |
|-------|-----------|-------|---------|-----|
| HR    | deteriorating | +16.31 | 0.000 | * |
| SpO2  | deteriorating | -0.37  | 0.000 | * |

#### Care Guidance
- Critical: VF detected — initiate ACLS protocol
- MEWS 6 (High) — escalate care
```

**3 plot types per event**: ECG waveform (7 leads + pacer marker), Vitals history (5 subplots with thresholds), MEWS score timeline (risk-band shading)

---

## Output Data Snapshot

**ECG Plot with pacer marker** (red dashed line at offset):

```
  ECG1  ╌╌╌╌╌╌|╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌  ← 7 leads stacked
  ECG2  ╌╌╌╌╌╌|╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌     with pacer line
  ECG3  ╌╌╌╌╌╌|╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌     at 1.5s
         0   1.5  3    6    9    12s
              ↑ red dashed pacer line
```

**Generated outputs per HDF5 file**:
- `report-PT4305-2026-03.md` — full clinical report
- `PT4305-2026-03_ecg_1001.png` — ECG waveform
- `PT4305-2026-03_vitals_1001.png` — Vitals history
- `PT4305-2026-03_mews_1001.png` — MEWS timeline

---

## Next Steps

**Model improvements**
- Train on real clinical ECG data (MIT-BIH, PTB-XL) to validate synthetic-to-real transfer
- Add attention visualization for model interpretability / explainability
- Improve weakest conditions (Normal Sinus, AV Block 1st, LBBB) with targeted augmentation

**Pipeline enhancements**
- GPU batching — process all events in a file as a single batch
- Cross-platform file watching (macOS/Windows support via watchdog)
- Streaming mode — continuous ECG strip analysis vs. 12s windows
- FHIR/HL7 integration for EMR interoperability

**Clinical validation**
- Prospective study with bedside monitor data
- Alarm reduction metrics — false alarm suppression rate
- Clinician feedback loop on care guidance accuracy

**Deployment**
- Edge inference on bedside compute (ONNX/TensorRT export)
- REST API wrapper for integration with central monitoring systems
- Dashboard for multi-patient real-time monitoring
