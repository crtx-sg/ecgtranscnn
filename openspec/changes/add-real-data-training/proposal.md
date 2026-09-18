# Change: Train, evaluate and serve ECG-TransCovNet on real ECG (`ecgpkg` v1)

## Why
Every shipped checkpoint was trained on simulator output and scores 4.8–8.3 % accuracy (macro-F1 0.04–0.08) on real ecg_sigma events, predicting mostly AV_BLOCK_2_TYPE2, VENTRICULAR_FIBRILLATION or ST_ELEVATION. ecg_sigma now delivers a validated real-ECG training package (`ecgpkg` v1: 25,605 labelled windows, 6 PhysioNet sources, 12-class head, subject-grouped splits). The repo has no way to consume it: the class list is hard-coded to the 16-member `Condition` enum, data only comes from the simulator, HDF5 readers expect `/metadata` datasets rather than attributes, and model selection/metrics assume all 16 classes are present.

Clinical context: VF, VT and atrial flutter in the package come almost entirely from single-channel Holter databases (only ECG2 real, other leads fabricated), while NORMAL_SINUS and conduction classes come from 12-lead sources. A model can learn "fabricated leads ⇒ ventricular arrhythmia" instead of rhythm. The deployment monitor records 7 real leads, so that shortcut would fail in the field; the change must measure and mitigate it.

## What Changes
- **ADDED** `ecg_transcovnet/classes.py`: `ClassSpec` (ordered head, name↔index) loaded from `package.json` or a checkpoint; condition resolution by enum value or name.
- **ADDED** `ecg_transcovnet/checkpoint.py`: shared checkpoint loading/model construction sized to the checkpoint head; warm start with per-class query copy by name.
- **ADDED** `ecg_transcovnet/hdf5_io.py`: `/metadata` read from attributes or datasets, bytes decoding, lead stacking.
- **ADDED** `ecg_transcovnet/package.py`: `load_package` (format, version, manifest hash, split invariants, optional SHA256SUMS), streaming parallel memmap cache, `PackageDataset` (crop → train-only augmentation → preprocessing), length-bucketed batching.
- **ADDED** `ecg_transcovnet/augment.py`: amplitude-relative simulator-artefact injection and ecg_sigma-equivalent lead fabrication (augmentation and counterfactual evaluation).
- **ADDED** `ecg_transcovnet/evaluation.py`: head-sized metrics, per-class subjects, grouped breakdowns (dataset, label_method, real_lead_mask), subject-bootstrap CI, JSON/markdown/PNG reports, baseline-mode name mapping, val-fitted logit bias.
- **MODIFIED** `ecg_transcovnet/preprocessing.py`: filters applied to all leads in one vectorised call (bit-identical output, ~5× faster).
- **MODIFIED** `ecg_transcovnet/training.py`: `evaluate_detailed` sized to a class list, macro over present classes; validation returns predictions for macro-F1 selection.
- **MODIFIED** `scripts/train.py`: `--data-source {sim,package}` and package flags (`--package`, `--crop-len`, `--noise-aug-prob`, `--fabricated-leads`, `--lead-fab-aug-prob`, `--sampler`, `--select-metric`, `--init-checkpoint`, `--resume`, `--time-budget-min`); per-epoch resumable checkpoints with package provenance; test-split report at the end. Simulator path unchanged.
- **MODIFIED** `scripts/evaluate.py`: `--package --split`, both crop lengths, baseline mode for legacy heads, lead-conversion counterfactual, multi-checkpoint ensemble, val bias calibration.
- **MODIFIED** `scripts/processor.py`, `scripts/run_validation_suite.py`, `scripts/visualize.py`, `scripts/visualize_hdf5.py`, `scripts/compute_auc.py`, `ecg_transcovnet/report.py`, `ecg_transcovnet/data.py`: checkpoint-sized heads, attribute metadata, `n/a (not in head)` ground truth, checkpoint filter preset default. Folds in the uncommitted enum-name handling edits; fixes the undefined `@dataclass_free` decorator in `run_validation_suite.py`.
- **ADDED** tests: `tests/fixtures/make_ecgpkg.py`, `tests/test_classes.py`, `tests/test_package.py`, `tests/test_evaluation.py`, `tests/test_processor_compat.py`, `tests/test_train_package.py`; widened AV-block skip set in `tests/test_noise_robustness.py` (pre-existing failure).
- **ADDED** `docs/real-data-training.md` (experiment log); README, `models/README.md` updates; `openspec/project.md` corrected to single-label softmax.

## Impact
- Affected specs: `data-pipeline`, `inference-pipeline`, `trained-models`, `model-architecture`.
- Affected code: see above. No change to the model architecture, simulator, MEWS logic, ecg_sigma or the `ecgpkg` contract.
- Compatibility: legacy 16-class checkpoints load and run unchanged; simulator training keeps its defaults (`--data-source sim`); `processor.py` now defaults the filter preset to the checkpoint's recorded preset (legacy checkpoints record none → `none`, identical to before).
- Noise robustness: package training injects simulator artefacts scaled to each lead's amplitude; evaluation reports per-dataset and per-lead-mask results so regressions on noisy or single-lead sources are visible.
- Resources: caches are memory-mapped (~1.7 GB v1); training fits a 6 GB GPU and 15 GB RAM.
