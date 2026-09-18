# Prompt — train ECG-TransCovNet on real ECG data from ecg_sigma (`ecgpkg` v1)

You are working in the `ecgtranscnn` repository (`/home/sganesh/aiwork/vios/repo/ecgtranscnn`, package `ecg_transcovnet`). Follow the repo's `AGENTS.md` / openspec workflow: open a change `add-real-data-training` (proposal, design, tasks, spec deltas) before coding. State a short plan with a verification step per item, and verify before you claim something works. Do not commit unless asked.

**Environment**
- Python: `~/miniconda3/envs/ecgtranscnn/bin/python` (torch 2.10 + CUDA). GPU: RTX 4050, 6 GB. RAM: 15 GB. CPU: 22 cores.
- Loading ~17k × 7 × 2400 events into RAM has already caused an out-of-memory kill on this machine: use memory-mapped caches, never whole-dataset in-RAM arrays.
- Long background jobs have been killed by the session's memory monitor; prefer foreground runs under 10 minutes, resumable where they are longer (e.g. training with checkpoints per epoch).
- The working tree has **uncommitted edits** in `scripts/processor.py`, `scripts/run_validation_suite.py`, `scripts/visualize_hdf5.py` and `slides.md`. Review them and fold them into your work; do not discard them.

## Goal

Train, evaluate and run inference with ECG-TransCovNet on the **real-ECG training package produced by ecg_sigma**, whose class head is decided by the data (12 classes in v1), while keeping the existing simulator training path working.

## The package (already built and validated)

| | |
|---|---|
| Path | `/home/sganesh/aiwork/vios/repo/ecg_sigma/packages/ecg_pkg_v1` (read-only: its HDF5 files are hardlinks of ecg_sigma's `out_v1/`) |
| Smaller package for fast iteration | `/home/sganesh/aiwork/vios/repo/ecg_sigma/packages/ecg_pkg_v0` — MIT-BIH + INCART only, 9 classes, 2400-sample windows |
| Validate | `cd /home/sganesh/aiwork/vios/repo/ecg_sigma && .venv/bin/python scripts/validate_package.py packages/ecg_pkg_v1` |
| Format | `ecgpkg` v1 — contract at the end of this prompt |
| Sources | MIT-BIH, INCART, PTB-XL, VFDB, CUDB, AFDB (PhysioNet) |
| Windows | 2400 samples (12 s) for five sources, **2000 samples (10 s) for PTB-XL**; 200 Hz; mV; already band-passed 0.5–40 Hz with a 50 Hz notch; not z-scored |
| Manifest | 37,587 rows = 25,605 included + 11,982 excluded (keep the excluded rows out of training) |
| Human-readable summary | `DATACARD.md` inside the package |

**Head (`package.json` `classes`, in this order).** Cells are events (subjects):

| idx | class | train | val | test |
|---:|---|---:|---:|---:|
| 0 | NORMAL_SINUS | 8,694 (6,424) | 1,361 (824) | 1,619 (800) |
| 1 | SINUS_BRADYCARDIA | 1,168 (417) | 186 (46) | 396 (59) |
| 2 | SINUS_TACHYCARDIA | 759 (368) | 163 (50) | 192 (50) |
| 3 | ATRIAL_FIBRILLATION | 1,960 (780) | 418 (103) | 422 (103) |
| 4 | ATRIAL_FLUTTER | 222 (36) | 55 (3) | 53 (4) |
| 5 | PAC | 808 (202) | 262 (28) | 142 (34) |
| 6 | PVC | 2,516 (377) | 616 (58) | 838 (61) |
| 7 | VENTRICULAR_TACHYCARDIA | 519 (23) | 111 (11) | 111 (7) |
| 8 | VENTRICULAR_FIBRILLATION | 422 (18) | 90 (6) | 91 (6) |
| 9 | LBBB | 340 (209) | 78 (26) | 81 (26) |
| 10 | RBBB | 308 (153) | 71 (19) | 71 (20) |
| 11 | AV_BLOCK_1 | 407 (237) | 25 (23) | 30 (28) |

Not in v1's head: SVT, ST_ELEVATION, AV_BLOCK_2_TYPE1, AV_BLOCK_2_TYPE2. Class names are the `Condition` enum **names** of `ecg_transcovnet/simulator/conditions.py`.

## Facts that shape the work (measured; do not re-derive)

- **Baseline.** The current checkpoints (`models/best_model.pt`, `models/noise_robust/best_model.pt`, `models/avblock_fix/best_model.pt`; simulator-only, filter preset `none`) score **4.8–8.3 % accuracy, macro-F1 0.04–0.08** on 3,802 real ecg_sigma events — below chance. Predictions collapse to AV_BLOCK_2_TYPE2, VENTRICULAR_FIBRILLATION or ST_ELEVATION; the `default` filter preset does not help.
- **No real-data path.** `data.py` only generates simulator data; `load_hdf5_test_samples` labels by filename prefix; `NUM_CLASSES`/`CLASS_NAMES` come from the enum; checkpoints are chosen on val accuracy; `evaluate_detailed` averages over all 16 classes including absent ones.
- **Reader mismatches.** ecg_sigma writes `/metadata` fields as HDF5 **attributes** (`report.extract_ids` and `visualize_hdf5` read datasets, so patient ids show as "unknown"); strings are bytes; `condition` uses enum names, while simulator files use enum values.
- **Lead realism varies by source** (`real_lead_mask`, order ECG1, ECG2, ECG3, aVR, aVL, aVF, vVX):
  - `1111111` — INCART, PTB-XL: all seven leads measured.
  - `0100001` — MIT-BIH: only ECG2 (MLII) and vVX (V1) measured; the other limb leads are fabricated from MLII (Lead I vs II correlation ≈ −0.96).
  - `0100000` — VFDB, CUDB, AFDB: one unnamed channel used as ECG2; everything else fabricated or synthesised.
  - Consequence: **no VF event has seven real leads** (582 `0100000`, 21 `0100001`); 471 of 741 VT events are `0100000`; 215 of 330 AFL events are `0100000`. A model can learn "few real leads ⇒ ventricular arrhythmia". Fabricated leads stay in the data by decision; your evaluation must expose this.
- **Label methods** (`label_method`): `rhythm_annotation`, `beat_run` (≥ 3 fast ectopic beats), `beat_morphology`, `rate_derived` (sinus windows, rate from annotated RR), `record_level` (PTB-XL). Report metrics per method.
- **Weak evaluation for some classes:** ATRIAL_FLUTTER has 3 val / 4 test subjects; VENTRICULAR_TACHYCARDIA 7 test subjects; VENTRICULAR_FIBRILLATION 6. Put subject counts next to every per-class metric.
- `heart_rate` is unreliable in VFDB/CUDB VT/VF windows. Vitals, PPG, RESP and pacer data are synthetic: do not use them as features.
- Memory: the included events as a `(N, 7, 2400)` float32 memmap are ~1.7 GB; the train split ~1.2 GB.

## Work items

**1. Data-driven class specification** (`ecg_transcovnet/constants.py` or a new `classes.py`)
- A `ClassSpec` (ordered names, name → idx) loaded from `package.json` or a checkpoint's `class_names`. The simulator path keeps the enum defaults.
- Every model construction uses `num_classes=len(spec)`: `train.py`, `evaluate.py`, `processor.py`, `run_validation_suite.py`, `visualize.py`.
- Verify: a 12-class model builds from v1's `package.json`; a legacy 16-class checkpoint still loads.

**2. HDF5 reader compatibility** (`report.py`, `scripts/processor.py`, `scripts/run_validation_suite.py`, `scripts/visualize_hdf5.py`, `data.py`)
- Read `/metadata` from attributes or datasets; decode bytes.
- Resolve `condition` by enum value or enum name, then map by name into the checkpoint's `ClassSpec`. Ground truth outside the head prints as `n/a (not in head)`, is excluded from metrics, and still shows a prediction; warn once per unknown label.
- `processor.py` defaults `--filter-preset` to the checkpoint's saved preset.
- Verify: `processor.py --process-existing` on a directory holding one ecg_sigma file (e.g. `ecg_sigma/out_v1/incart/I30_2025-01.h5`) and one simulator file prints correct patient ids and ground truths.

**3. Package loader** (new `ecg_transcovnet/package.py`; `csv`/`json` only, no pandas)
- `load_package(root)`: checks `format == "ecgpkg"` and `format_version == 1` (clear error otherwise) and `manifest_sha256`; optional `SHA256SUMS` check; returns the spec and manifest rows.
- `PackageDataset(root, split, leads, filter_config, crop_len, train, noise_aug_prob, fabricated_leads="keep")`:
  - First use builds a **memmap cache** `data/training_cache/<package_version>_<split>_<leads>.npy`, shape `(N, 7, 2400)` float32, right-padded for 2000-sample events, plus lengths and labels; written streaming, one HDF5 file at a time. The cache holds signals **as read, unnormalised**.
  - `__getitem__`: crop (train: random `crop_len` inside the valid region; val/test: centre crop, or the full valid length when `crop_len` is None) → train-only augmentation → `PreprocessingPipeline` on the crop. Padding never enters a crop.
  - `fabricated_leads="zero"` zeroes every lead whose `real_lead_mask` bit is `0` (for the ablation in item 10).
  - Exposes per-row `dataset`, `subject_id`, `label_method`, `real_lead_mask` for grouped metrics.
  - If per-item filtering exceeds ~5 ms (measure it), add a preprocessed cache for val/test only.
- Verify with tests: shapes, crops never touch padding, deterministic eval crops, cache reuse, bad `format_version` rejected, peak RSS during cache build far below the dataset size.

**4. Variable window length**
- Confirm `ECGTransCovNet` returns valid logits for 2000- and 2400-sample inputs with the same weights (`seq_len` only sizes the positional encoding, `max_len ≥ 512`); fix if not.
- Default `--crop-len 2000` for package training so both window lengths mix. Report both centre-crop 2000 and full 2400 results for 12 s events; deployment inference stays at 2400.

**5. Augmentation** (package-aware version of `AugmentedECGDataset`)
- Random crop replaces the circular shift. Keep per-lead scaling, Gaussian noise, channel dropout (after preprocessing, as today).
- Optional simulator-artefact injection via `simulator/noise.py` `apply_noise_pipeline` (random low/medium preset) with probability `--noise-aug-prob` (default 0.5), applied to the raw crop **before** preprocessing.
- Verify: train-only, deterministic under a fixed seed.

**6. `scripts/train.py`**
- Flags: `--data-source {sim,package}` (default `sim`), `--package PATH`, `--crop-len`, `--noise-aug-prob`, `--fabricated-leads {keep,zero}` (default `keep`), `--select-metric {macro_f1,accuracy}` (default `macro_f1` for package), `--init-checkpoint PATH` (warm start: load compatible backbone/encoder/decoder weights; re-initialise `object_queries` and the FFN head when the class count differs).
- Package runs use `--filter-preset default`. Class weights from **train-split** counts. Validate on the package val split; evaluate the test split at the end. Save a checkpoint every epoch so a killed run resumes (`--resume`).
- Checkpoints add `class_names`, `data_source`, `package_version`, `package_manifest_sha256`, `crop_len`, `filter_preset`, `leads`, `noise_aug_prob`, `fabricated_leads`, full args.
- Verify: 1–2 epoch smoke run on `ecg_pkg_v0`; the checkpoint round-trips through `processor.py`.

**7. Metrics** (`ecg_transcovnet/training.py`)
- `evaluate_detailed(model, loader, device, class_names)`: confusion matrix sized to the head; macro metrics over classes with support > 0, listing classes absent from the split; per-class subject counts.
- Grouped breakdowns by `dataset`, `label_method`, and `real_lead_mask` (`1111111` / `0100001` / `0100000`).

**8. `scripts/evaluate.py`**
- `--package PATH --split {val,test}`: per-class precision / recall / specificity / F1 with support and subjects, macro, and the grouped breakdowns; crop 2000 and full length; JSON + markdown + confusion-matrix PNG to `--output-dir`.
- **Baseline mode** for legacy 16-class checkpoints: map predicted names into the head; a prediction outside the head counts as wrong.

**9. Fixture package and tests** (`tests/fixtures/make_ecgpkg.py`, `tests/test_package.py`, `tests/test_processor_compat.py`)
- The fixture generator writes a contract-conformant package with ecg_sigma-style HDF5 (attribute `/metadata`, bytes attrs, per-lead `source`/`method` attrs, extras JSON), mixing 2400- and 2000-sample events, 3–4 classes, several subjects, some excluded rows.
- Tests cover every contract invariant, loader behaviour, a `train.py` smoke run, processor compatibility, legacy checkpoints, and a loader error on a manifest with a subject in two splits. Existing tests stay green.

**10. Experiments** (record in `docs/real-data-training.md`; iterate on v0, report on v1)

| Run | Setup |
|---|---|
| a | Baseline: current checkpoints in baseline mode on v1 test |
| b | Package-only, from scratch |
| c | Package-only, warm start from `models/avblock_fix/best_model.pt` |
| d | Best of b/c with `--noise-aug-prob 0` vs `0.5` |
| e | Best run with `--fabricated-leads zero` — checks whether the model relies on lead realism instead of rhythm |

For each: test macro-F1, per-class F1 with subject counts, per-dataset, per-`label_method` and per-`real_lead_mask` results. Flag classes with fewer than 5 test subjects. Save the best model to `models/real_v1/` with curves and confusion matrix.

**11. Docs and specs**
- README: training on a package, data-driven head, checkpoint format, filter-preset guidance, the real-lead caveat.
- `models/README.md`; openspec deltas for data-pipeline, inference-pipeline, trained-models.
- `openspec/project.md` says sigmoid multi-label; the code and this change are single-label softmax — correct it.

## Acceptance criteria
- All existing and new tests pass; simulator training behaves as before.
- `train.py --data-source package` trains end-to-end on v0 and v1 within 6 GB GPU / 15 GB RAM.
- `processor.py` and `run_validation_suite.py` work with legacy and package checkpoints, and with simulator and ecg_sigma HDF5 files.
- The v1 test report beats baseline mode (run a), with per-class, per-dataset and per-lead-mask tables.
- Final report: what changed, test results, the experiment table, whether run e shows a lead-realism shortcut, known limitations, and anything you could not verify.

## Out of scope
Changing ecg_sigma or the package; changing the contract; multi-label outputs; mixing simulator data into package training (possible follow-up); vitals/MEWS logic beyond the reader fixes.

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
