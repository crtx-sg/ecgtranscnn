## ADDED Requirements

### Requirement: 16 Cardiac Condition Definitions
The system SHALL define 16 cardiac conditions as a Python Enum, each with a `ConditionConfig` specifying: heart rate range, QRS duration range, P wave presence probability, T wave inversion probability, RR irregularity, wide QRS flag, ST elevation flag, and optional minimum PR interval (for AV blocks).

#### Scenario: Condition configuration completeness
- **WHEN** the condition system is initialized
- **THEN** all 16 conditions SHALL have a `ConditionConfig` with physiologically appropriate parameter ranges:
  - Normal variants: NSR (60-100 BPM), Bradycardia (<60), Tachycardia (>100)
  - Supraventricular: AFib (irregular, no P waves), Flutter, PAC, SVT
  - Ventricular: PVC (wide QRS), VTach (fast, wide), VFib (chaotic)
  - Conduction: LBBB/RBBB (wide QRS), AV blocks (min_pr_interval≥0.22s)
  - Ischemic: ST Elevation

#### Scenario: AV block PR interval enforcement
- **WHEN** generating an AV block condition (1st, 2nd Type I, 2nd Type II)
- **THEN** the condition config SHALL specify `min_pr_interval=0.22` to ensure prolonged PR intervals distinguishable from normal rhythm

### Requirement: Beat Morphology Generation
The system SHALL generate realistic ECG beat morphology for each lead including P wave (Gaussian-shaped), QRS complex (triangular with lead-specific morphology), and T wave (Gaussian-shaped, possible inversion). Patient-specific parameters SHALL vary PR interval, QRS duration, and wave amplitudes within physiologically plausible ranges.

#### Scenario: Normal sinus beat generation
- **WHEN** generating a normal sinus beat
- **THEN** the morphology SHALL include P wave, QRS complex, and T wave with PR interval 0.12-0.20s and QRS duration 0.06-0.10s

#### Scenario: Condition-specific morphology
- **WHEN** generating beats for AFib
- **THEN** P waves SHALL be replaced with fibrillatory baseline waves
- **WHEN** generating beats for Atrial Flutter
- **THEN** sawtooth flutter waves at ~300 BPM SHALL be added
- **WHEN** generating beats for VFib
- **THEN** a completely chaotic signal SHALL be produced
- **WHEN** generating beats for LBBB/RBBB
- **THEN** QRS complexes SHALL be widened and notched

### Requirement: 7-Lead ECG Signal System
The system SHALL generate 7 ECG leads (ECG1, ECG2, ECG3, aVR, aVL, aVF, vVX) with inter-lead correlations reflecting the Einthoven triangle relationship for limb leads, augmented limb leads, and a precordial-like lead.

#### Scenario: Lead generation
- **WHEN** a complete ECG event is generated
- **THEN** all 7 leads SHALL be produced at 200 Hz for 12 seconds (2,400 samples each) with physiologically consistent inter-lead relationships

### Requirement: Six-Stage Noise Pipeline
The system SHALL apply noise artifacts sequentially through six stages: baseline wander (0.1-0.5 Hz drift), Gaussian noise, EMG bursts, motion artifacts, powerline interference (50/60 Hz), and electrode contact degradation. Four noise presets SHALL be available: clean (no artifacts), low, medium, and high.

#### Scenario: Clean noise preset
- **WHEN** noise_level is set to "clean"
- **THEN** all noise parameters SHALL be zero and the output signal SHALL be identical to the clean morphology

#### Scenario: Mixed noise mode for training
- **WHEN** noise_level is set to "mixed"
- **THEN** each sample SHALL randomly receive one of the four noise presets, building robustness across all noise levels

### Requirement: Supplementary Signal Generation
The system SHALL generate supplementary physiological signals alongside ECG: PPG signal at 50 Hz (600 samples for 12s) and respiratory signal at 25 Hz (300 samples for 12s), plus vital signs (HR, SpO2, blood pressure, respiratory rate, temperature, posture).

#### Scenario: Complete event generation
- **WHEN** `generate_event(condition)` is called
- **THEN** a `SimulatedEvent` SHALL be returned containing ECG (7 leads), PPG, respiratory signal, vitals dictionary, condition label, heart rate, and noise level

### Requirement: HDF5 Phase-0 Schema Output
The system SHALL write simulated events to HDF5 files conforming to the Phase-0 schema with groups: `metadata/` (patient_id, sampling rates, alarm_time, data_quality), `events/{event_id}/ecg/` (7 lead datasets), `events/{event_id}/ppg`, `events/{event_id}/respiratory`, `events/{event_id}/vitals/`, and `events/{event_id}/ground_truth/` (condition, heart_rate, timestamp).

#### Scenario: HDF5 file structure
- **WHEN** an HDF5 file is written with multiple events
- **THEN** each event SHALL be stored under `events/{event_id}/` with all required groups and datasets, and metadata SHALL be at the file root level

#### Scenario: ECG signal data types
- **WHEN** ECG lead data is stored in HDF5
- **THEN** each lead SHALL be stored as `float64` array of length 2400
