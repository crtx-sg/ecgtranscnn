## Shared interface contract — `ecgpkg` format v1

> This section is identical in the ecg_sigma and ecgtranscnn prompts. It is the only coupling
> between the two workstreams. Change it in both places or not at all.

### Decisions already made (do not re-open)
| Topic | Decision |
|---|---|
| Label vocabulary | ecgtranscnn `Condition` enum **names** are the baseline (e.g. `ATRIAL_FIBRILLATION`, not `AFIB` / `"AFIB"` value). |
| Model head | Only the baseline classes that real data supports (inclusion thresholds below). The class list is **data-driven** and shipped in `package.json`. Consumers must never hard-code 16. |
| Output type | Single-label softmax; one label per event. |
| `vVX` | **V1** wherever the source has a real precordial lead (MIT-BIH 40 of 46 convertible records, INCART, PTB-XL). A different real V lead standing in for V1 is excluded (`vvx_not_v1`). Sources with no precordial lead at all (VFDB, CUDB, AFDB) carry a synthesised vVX: `vvx_lead = "synthetic"`, last `real_lead_mask` bit `0`. |
| Fabricated leads | Keep (MIT-BIH limb leads stay in the data; provenance is tracked). |
| 10 s datasets | Allowed. PTB-XL events are 2000 samples; 12 s sources are 2400 samples. |
| Licences | No redistribution restriction for this package; still recorded in DATACARD. |

### Candidate classes (canonical order = ecgtranscnn `Condition` enum order)
```
NORMAL_SINUS, SINUS_BRADYCARDIA, SINUS_TACHYCARDIA, ATRIAL_FIBRILLATION, ATRIAL_FLUTTER,
PAC, SVT, PVC, VENTRICULAR_TACHYCARDIA, VENTRICULAR_FIBRILLATION, LBBB, RBBB,
AV_BLOCK_1, AV_BLOCK_2_TYPE1, AV_BLOCK_2_TYPE2, ST_ELEVATION
```
The head is the ordered subset that passes the inclusion thresholds **after** the subject-grouped split.
Defaults (configurable, recorded in `package.json`):
- train ≥ 2 subjects and ≥ 100 events
- val ≥ 1 subject and ≥ 20 events
- test ≥ 1 subject and ≥ 20 events

Classes with < 3 test subjects are flagged `low_confidence_eval` in `package.json`.
Package v1 has 12 classes; `SVT`, `ST_ELEVATION`, `AV_BLOCK_2_TYPE1` and `AV_BLOCK_2_TYPE2` fail the thresholds.

### Directory layout
```
ecg_pkg_<version>/
  package.json
  manifest.csv          # one row per candidate event, including excluded ones
  splits.json
  DATACARD.md
  SHA256SUMS            # sha256 of every file in the package except itself
  data/<dataset>/*.h5   # unchanged ecg_sigma HDF5 files (hardlink or copy)
```

### `package.json`
```json
{
  "format": "ecgpkg",
  "format_version": 1,
  "package_version": "v0",
  "created_utc": "2026-09-15T00:00:00Z",
  "classes": ["NORMAL_SINUS", "SINUS_BRADYCARDIA", "..."],
  "low_confidence_eval": ["..."],
  "inclusion_thresholds": {"train": {"subjects": 2, "events": 100},
                           "val":   {"subjects": 1, "events": 20},
                           "test":  {"subjects": 1, "events": 20}},
  "leads": ["ECG1", "ECG2", "ECG3", "aVR", "aVL", "aVF", "vVX"],
  "vvx_lead": "V1",
  "sampling_rate_hz": 200,
  "window_samples": [2400, 2000],
  "signal_units": "mV",
  "prefilter": {"bandpass_hz": [0.5, 40.0], "notch_hz": 50.0, "zero_phase": true},
  "split_counts": {"train": 0, "val": 0, "test": 0, "excluded": 0},
  "seed": 42,
  "provenance": {"ecg_sigma_git_sha": "...", "pipeline_config_sha256": "...",
                 "package_config": {}, "numpy": "...", "scipy": "...", "h5py": "..."},
  "manifest_sha256": "..."
}
```

### `manifest.csv` (UTF-8, header row, comma-separated, no index column)
| column | type | meaning |
|---|---|---|
| `event_uid` | str | the event's `/uuid` dataset value (stable uuid5) |
| `split` | str | `train` \| `val` \| `test` \| `excluded` |
| `exclude_reason` | str | empty unless `split == excluded` (e.g. `class_not_in_head`, `mixed_rhythm`, `vvx_not_v1`, `multi_head_class_conflict`, `low_quality`) |
| `label` | str | head class name; empty if excluded |
| `label_idx` | int | index into `package.json.classes`; `-1` if excluded |
| `condition` | str | event `condition` attribute as written by ecg_sigma |
| `label_method` | str | `rhythm_annotation` \| `beat_run` \| `beat_morphology` \| `rate_derived` \| `record_level` \| `mixed_rhythm` |
| `label_purity` | float | 0–1, see ecg_sigma definition |
| `dataset` | str | `mitbih` \| `incart` \| `ptbxl` \| `vfdb` \| `cudb` \| `afdb` \| … |
| `subject_id` | str | globally unique, dataset-prefixed (e.g. `incart:p14`, `mitbih:201_202`) |
| `record_id` | str | dataset-prefixed record id |
| `h5_relpath` | str | path relative to package root, e.g. `data/incart/I30_2025-01.h5` |
| `event_key` | str | e.g. `event_1001` |
| `n_samples` | int | 2400 or 2000 (must equal the stored lead length) |
| `fs` | float | 200.0 |
| `hr_bpm` | float | event `heart_rate` attribute |
| `quality` | float | per-event data-quality score (use file-level `/metadata.data_quality_score` if no per-event value exists) |
| `real_lead_mask` | str | 7 chars of `0`/`1` in `leads` order; `1` = dataset attr `source == "real"` |
| `vvx_lead` | str | `V1`, or `synthetic` when the source has no precordial lead |
| `source_sample` | int | onset index at source fs |

### `splits.json`
```json
{"seed": 42, "method": "subject_grouped_stratified",
 "notes": "PTB-XL uses strat_fold 1-8 train, 9 val, 10 test",
 "subjects": {"train": ["..."], "val": ["..."], "test": ["..."]}}
```

### Invariants a consumer may rely on (the package validator enforces them)
1. No `subject_id` appears in more than one of train/val/test.
2. Every non-excluded row's `label` is in `classes`, and `label_idx == classes.index(label)`.
3. `data/…/<event_key>/ecg/<lead>` exists for all 7 leads, float32, length `n_samples`.
4. Signals are stored exactly as ecg_sigma writes them (mV, band-passed 0.5–40 Hz, 50 Hz notch, **not** z-scored). Consumers apply their own normalisation.
5. `/metadata` fields are HDF5 **attributes** (not datasets); string attributes are UTF-8 bytes.
6. Every class in `classes` meets the inclusion thresholds in every split.
