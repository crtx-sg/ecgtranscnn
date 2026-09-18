## 1. Core compatibility
- [x] 1.1 `classes.py` `ClassSpec` + condition resolution — verify: 12-class spec from v1 `package.json`; value/name resolution unit tests
- [x] 1.2 `checkpoint.py` shared loader, `build_model`, warm start — verify: legacy 16-class checkpoint loads; 12-class model builds; by-name query copy test
- [x] 1.3 `hdf5_io.py` metadata attrs/datasets; `report.extract_ids` uses it — verify: ids read from ecg_sigma and simulator files
- [x] 1.4 Vectorised `PreprocessingPipeline` — verify: bit-identical to per-lead reference (all presets, 2000/2400); existing preprocessing tests green; 19–24 ms → 3.5 ms per item
- [x] 1.5 Forward at 2000 and 2400 samples with the same weights — verify: unit test

## 2. Package data
- [x] 2.1 `tests/fixtures/make_ecgpkg.py` contract-conformant fixture — verify: ecg_sigma `validate_package.py` prints OK
- [x] 2.2 `package.load_package` checks — verify: bad format/version, manifest hash mismatch, subject overlap, label_idx mismatch, SHA256SUMS tampering raise
- [x] 2.3 Parallel streaming memmap cache — verify: contents equal HDF5; reuse; tracemalloc peak < ¼ dataset size; v1 build 52 s, peak RSS 780 MB (torch baseline 610 MB) vs 1.72 GB cache
- [x] 2.4 `PackageDataset` crops/eval determinism/metadata; length-bucketed batch sampler — verify: crops never include padding; eval crops deterministic
- [x] 2.5 `augment.py` artefact injection (amplitude-relative) and lead fabrication — verify: train-only; seed-deterministic; fabricated leads vs ecg_sigma-stored leads median r ≥ 0.995 (afdb, cudb, mitbih, vfdb)

## 3. Training and evaluation
- [x] 3.1 `evaluation.py` metrics, grouped breakdowns, bootstrap CI, AUROC, reports, baseline mapping, bias fit — verify: hand-built prediction tests
- [x] 3.2 `training.py` `evaluate_detailed(class_names)`, macro over present classes, validate with predictions — verify: existing e2e and noise tests
- [x] 3.3 `scripts/train.py` package mode, sampler, warm start, resume, time budget, checkpoint provenance, FP32 default — verify: fixture smoke test with resume; v0 run learns (val acc 0.79 / macro-F1 0.70 at epoch 13)
- [x] 3.4 `scripts/evaluate.py` package/baseline/counterfactual/ensemble/calibration — verify: fixture run; baseline mode on v1 test (10–26 % accuracy)

## 4. Inference compatibility
- [x] 4.1 `processor.py` spec-sized head, not-in-head GT, checkpoint preset default (fold in uncommitted edits) — verify: INCART I30 + simulator file with legacy and package checkpoints
- [x] 4.2 `run_validation_suite.py` (fix `@dataclass_free`), `visualize.py`, `visualize_hdf5.py`, `compute_auc.py`, `data.py` — verify: compat tests; CLI runs
- [x] 4.3 Widen AV-block skip set in `tests/test_noise_robustness.py` — verify: full suite 118 passed, 1 skipped

## 5. Experiments (iterate on v0, report on v1)
- [x] 5.1 a: baseline mode, three legacy checkpoints, v1 test
- [x] 5.2 b: package-only from scratch — `v1_b`, test macro-F1 0.607 / acc 0.751
- [x] 5.3 c: warm start from `models/avblock_fix/best_model.pt` (queries by name vs reinit) — `v1_c_byname` 0.652, `v1_c_reinit` 0.646; both beat b
- [x] 5.4 d: best of b/c with `--noise-aug-prob 0` vs `0.5` — `v1_d_noise0` 0.614 vs 0.646 (within seed noise); noise aug **dropped** for the final model: combined with lead-fab aug it scored 0.606 (`v1_base_n05_lf05`)
- [x] 5.5 e: best with `--fabricated-leads zero` — `v1_e_fabzero` 0.629 matched, 0.573 with leads kept; not a deployment option
- [x] 5.6 f: lead-fabrication augmentation 0 / 0.3 / 0.5 with lead-conversion counterfactual — flip rate 0.41 → 0.16 → 0.09, test acc 0.699 → 0.795 → 0.799; shortcut removed
- [x] 5.7 g: balanced sampler vs class-weighted loss — `v1_g_balanced` 0.620 vs 0.649 test macro-F1; keep class-weighted loss (sampler wins val, loses test)
- [x] 5.8 h: val bias calibration and 3-seed ensemble — ensemble 0.814 acc / 0.668 macro-F1 (best single seed 0.798 / 0.667); calibration harmful (−0.05 macro-F1); seed spread 0.034
- [x] 5.9 Save best to `models/real_v1/` with curves, confusion matrix, reports — 3-seed ensemble (0.814 acc / 0.668 macro-F1), `best_model.pt` = seed 43 fallback; `processor.py` gained multi-checkpoint ensemble support

## 6. Documentation
- [x] 6.1 `docs/real-data-training.md` experiment table, shortcut findings, limitations
- [x] 6.2 README and `models/README.md`
- [x] 6.3 `openspec/project.md` single-label softmax correction
- [x] 6.4 `openspec validate add-real-data-training --strict` — valid
