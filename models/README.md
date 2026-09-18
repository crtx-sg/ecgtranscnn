# Pre-trained Models

Place trained model checkpoints here. These files are not tracked by git.

## Checkpoints

| Path | Data | Head | Filter preset |
|---|---|---|---|
| `best_model.pt` | simulator, clean | 16 `Condition` classes | none |
| `noise_robust/best_model.pt` | simulator, mixed noise | 16 | none |
| `avblock_fix/best_model.pt` | simulator, clean, AV-block morphology fix | 16 | none |
| `real_v2/fold{0..4}.pt` | ecg_sigma `ecgpkg` v2 (real ECG) — **the recommended ensemble** | 13 classes from `package.json` | default |
| `real_v2/best_model.pt` | as above, single-model fallback (copy of `fold0.pt`) | 13 | default |
| `real_v1/*` | ecg_sigma `ecgpkg` v1 — superseded, kept for provenance | 12 | default |

The simulator checkpoints score 10–26 % accuracy on real ECG (v1 test split, baseline
mode); use a real-data checkpoint for real recordings. See `docs/real-data-training.md`.

### `real_v2` — the current real-ECG release

Trained on `ecg_pkg_v2` with 5-fold grouped cross-validation, a warm start from `avblock_fix`,
`--lead-fab-aug-prob 0.5` (lead-fabrication augmentation, which removes the lead-realism shortcut),
`--noise-aug-prob 0`, class-weighted focal loss and `default` filtering.

| Model | v2 test accuracy | Primary macro-F1 (12 of 13 cls) | Full length |
|---|---:|---:|---:|
| **5-fold ensemble** (`fold0.pt` … `fold4.pt`) | **0.782** | **0.587** (CI 0.506–0.684) | 0.790 / 0.593 |

Cross-validation mean 0.617 ± 0.069 over the five folds (`cv_summary.json`). The ensemble was
pre-committed as the artifact before the test split was evaluated, so there is no single-model test
number to compare against — `best_model.pt` is a copy of `fold0.pt` for one-checkpoint tooling only.

The primary metric excludes VENTRICULAR_FIBRILLATION: no VF event in the package has seven measured
leads, so its 0.840 F1 is fabricated-lead only and not deployment-validated.

```bash
# Ensemble inference — processor.py and evaluate.py both accept several checkpoints
python scripts/processor.py --watch-dir data/inference \
    --checkpoint models/real_v2/fold0.pt models/real_v2/fold1.pt models/real_v2/fold2.pt \
                 models/real_v2/fold3.pt models/real_v2/fold4.pt

# Single model
python scripts/processor.py --watch-dir data/inference --checkpoint models/real_v2/best_model.pt
```

Members must share head, leads and filter preset; `ecg_transcovnet.checkpoint.load_models`
raises otherwise. The members' softmax outputs are averaged, the same rule
`package_eval.combine_predictions` uses, so `evaluate.py` and `processor.py` agree.

Do **not** apply `--calibrate-on val` to these checkpoints: the val-fitted logit bias costs
0.04–0.05 macro-F1 on test (see `docs/real-data-training.md`, run h).

**Head (12 classes, softmax output order).** Read it from the checkpoint
(`ClassSpec.from_checkpoint(ckpt).names`) rather than hard-coding it:

```
0 NORMAL_SINUS          4 ATRIAL_FLUTTER ⚠      8 VENTRICULAR_FIBRILLATION ⚠  12 AV_BLOCK_1
1 SINUS_BRADYCARDIA     5 PAC                   9 LBBB
2 SINUS_TACHYCARDIA     6 SVT ⚠                10 RBBB
3 ATRIAL_FIBRILLATION   7 PVC                  11 VENTRICULAR_TACHYCARDIA ⚠
```

⚠ ATRIAL_FLUTTER (recall 0.06), SVT (recall 0.08, AUROC 0.63) and VENTRICULAR_TACHYCARDIA
(recall 0.46) cannot be used to rule those rhythms out. VENTRICULAR_FIBRILLATION is not validated
for a 7-measured-lead monitor. `AV_BLOCK_2_TYPE1` and `AV_BLOCK_2_TYPE2` are **permanently
undetectable** (the source annotations cannot express Mobitz type) and `ST_ELEVATION` lacks data;
none of the three is in this head. Class definitions, per-class scores and label provenance:
README [Cardiac Conditions](../README.md#cardiac-conditions). Integrating from another codebase:
README [Using This Model From Another Project](../README.md#using-this-model-from-another-project).

## Training a model

```bash
# Simulator data
python scripts/train.py --num-train 16000 --num-val 3200 --epochs 100

# Real data from an ecgpkg package (resumable; rerun with --resume after the time budget)
python scripts/train.py --data-source package \
    --package ../ecg_sigma/packages/ecg_pkg_v1 --output-dir models/real_v1 \
    --time-budget-min 9 --resume
```

## Output files

- `best_model.pt` — checkpoint with the best validation selection metric
  (simulator: accuracy; package: macro-F1)
- `final_model.pt` — best weights plus evaluation metadata
- `last.pt` — package runs only: resumable state (optimiser, scheduler, history, early stopping)
- `training_curves.png`, `confusion_matrix.png`
- `history.json`, `reports/test.{json,md}` — package runs only: per-epoch history and the
  test-split report (per class with subject counts, per dataset, per label method, per real-lead mask)
- `eval/` — `scripts/evaluate.py` output: the same tables plus lead-conversion counterfactuals

## Checkpoint format

Every `.pt` file contains:
- `model_state_dict` — model weights
- `epoch` — training epoch
- `args` — training arguments (architecture hyper-parameters are rebuilt from these)
- `leads` — lead configuration used
- `class_names` — the model head, in output order (16 for simulator checkpoints)

Package-trained checkpoints also contain `data_source` (`package`), `package_version`,
`package_manifest_sha256`, `crop_len`, `filter_preset`, `noise_aug_prob`,
`fabricated_leads`, `lead_fab_aug_prob`, `select_metric`, `best_metric`, `val_acc`,
`val_macro_f1`.

Scripts never assume 16 classes: `ecg_transcovnet.checkpoint.load_model` builds the model
from `class_names` (legacy checkpoints without it get the 16-class enum head) and returns the
recorded filter preset, which `processor.py`, `run_validation_suite.py` and `evaluate.py` use
unless `--filter-preset` is given.
