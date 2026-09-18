# Plan — migrating ecgtranscnn to `ecgpkg` v2

Response reviewed: `docs/RESPONSE_change_requests.md`. Package verified present at
`ecg_sigma/packages/ecg_pkg_v2` and their headline claims reproduce against its manifest:

| Claim | Verified |
|---|---|
| VT measured-lead share 12/17/78 % → 35/25/22 % | yes (train 362 / val 93 / test 78 events) |
| VT label-method skew removed | yes — test is now 17 `beat_run` / 61 `rhythm_annotation` |
| AFL val/test subjects 3/4 → 5/6 | yes |
| VF has no seven-lead events | yes — 0 of 464, all splits |
| SVT back in head at exactly 100 train events | yes — head is 13 classes |
| `low_confidence_eval` populated | yes — 6 classes flagged |
| 5 grouped CV folds over train+val | yes — ~1,825 subjects each, test in none |

**Retraining is required.** Not because our recipe is wrong but because the subject→split mapping
changed for every class and the head gained SVT (12 → 13). A v1 checkpoint cannot be scored on v2:
v1 train subjects now sit in v2 test. `models/real_v1` stays as the v1-era artefact; v2 gets a new
release directory.

One clarification on their guidance *"do not warm-start from a v1 checkpoint and report test
numbers"*: our recipe warm-starts from `models/avblock_fix/best_model.pt`, which is
**simulator-trained and has never seen real ECG**, so it carries no v1 subject leakage. We keep it.

---

## Phase 1 — Loader and contract (blocks everything else)

| # | Change | File |
|---|---|---|
| 1.1 | Accept format version 2; keep 1 readable for the archived runs (`SUPPORTED_FORMAT_VERSIONS = {1, 2}`) | `ecg_transcovnet/package.py:31` |
| 1.2 | Assert `fabrication_rules_version == "1"` when the package declares one; fail loudly on mismatch so `augment.py` cannot drift | `package.py`, `augment.py` |
| 1.3 | Carry the new manifest columns (`paced_beats`, `source_beat_condition`, `source_rhythm_condition`) into row metadata | `package.py` |
| 1.4 | Expose `low_confidence_eval`, `eval_flags`, `class_counts`, `label_definitions` on the `Package` object | `package.py` |
| 1.5 | Read CV folds from `splits.json → cv`; build train/val subject sets per fold | `package.py` |

Nothing changes in the HDF5 read path, the cache (keyed on `package_version`, so it rebuilds
itself), the preprocessing pipeline or the model. Rows are read by column name, so the three
appended columns are additive.

## Phase 2 — Metric semantics (changes every reported number) — **done**

My first draft had this wrong: I proposed averaging only the 7 unflagged classes. ecg_sigma's
answer (Q1) is that the six flags are three different problems and only one justifies exclusion:

| Kind | Classes | Treatment |
|---|---|---|
| **Validity** — score does not measure what deployment needs | VF | **exclude from the headline** |
| **Confound** — a lead-realism shortcut exists | LBBB, RBBB | keep in, pair with a 7-real-lead number |
| **Precision** — unbiased but noisy | AFL, SVT, VT | keep in, report a CI |

> "Dropping the precision-flagged classes is the thing that flatters."

**Agreed definition, now implemented: macro-F1 over 12 of 13 classes (all but VF), with a
subject-level bootstrap CI**, never reported alone.

- `evaluation.VALIDITY_FLAGS` drives the exclusion from `eval_flags`, so it is **data-driven, not a
  hardcoded VF** — a future package that fixes VF automatically brings it back into the headline.
- The markdown report leads with the primary metric *and its exclusion reason in the same block*;
  the number cannot appear without the caveat.
- The per-class table gained a **Package flags** column.
- The bootstrap already resampled subjects, not events, which is what their point (d) requires.
- Checkpoint selection during training uses the same exclusion (`train.py`).
- `--calibrate-on val` now prints a warning citing the −0.05 measurement; kept for research only.

**Open:** they offered to ship a `reporting` block in `package.json` so both teams compute the
identical number. Worth accepting — say yes on the next build.

## Phase 3 — Retraining with cross-validation

They ask for fold-based selection rather than the single val split. That fits our ensemble need
exactly: **the 5 fold models become the deployed ensemble**, replacing the 3-seed ensemble.

- Fit on 4 folds, select on the held-out fold, repeat ×5 → 5 checkpoints + an averaged held-out
  score with a real spread (this replaces "3 seeds to estimate noise").
- **The artifact must be chosen before test is touched** (their Q2): either deploy the 5-fold
  ensemble and report the ensemble's test number, or refit once on all of train+val with the epoch
  count fixed to the median best epoch across folds and report that. Computing both and reporting
  the better one is selecting on test. *Decision pending — see below.*
- Recipe unchanged, since it was settled on v1 and nothing in v2 invalidates it: warm start from
  `avblock_fix` (`--init-queries reinit`), `--lead-fab-aug-prob 0.5`, `--noise-aug-prob 0`,
  class-weighted focal loss, `--crop-len 2000 --filter-preset default`, 40 epochs, patience 10.
- Test split is touched **once**, at the end, for the final number.

Cost: 5 × ~2 h ≈ **10 h** on the RTX 4050. A 1-fold smoke run comes first (~2 h) to confirm the
pipeline end to end before committing the rest.

## Phase 4 — Analyses v2 makes possible

| # | Analysis | Why |
|---|---|---|
| 4.1 | VT metrics split by `label_method` | Their suggestion: if recall recovers on `rhythm_annotation` but not `beat_run`, the residue is the 3-beat-run boundary, not the model |
| 4.2 | VT→PVC confusion against `source_beat_condition` / `source_rhythm_condition` | Measures how much of our 74/111 confusion is source ambiguity rather than model error |
| 4.3 | Metrics grouped by `paced_beats > 0` (**implemented** in `build_report`) | All 136 paced events are in val/test, none in train (two MIT-BIH patients), so training on them is not an option. Check PVC recall on the 50 paced val events vs unpaced — a real finding for paced patients if it diverges |
| 4.4 | Metrics by per-event `quality` quartile. **No global floor at all** — their Q7 shows a 0.20 floor removes 65 % of VT and 47 % of PVC: "a ventricular-arrhythmia filter wearing a quality costume". If used, only as a within-class percentile | v1's `quality` was the record mean, so this is newly meaningful |

`grouped_metrics` already takes an arbitrary column, so 4.1–4.4 are report additions, not new
machinery.

## Phase 5 — Release and documentation

- `models/real_v2/` — 5 fold checkpoints, `best_model.pt` fallback, ensemble reports, curves.
- README: head 12 → **13 classes** (SVT added); the "4 excluded classes" statement becomes
  **3**, and AV_BLOCK_2_TYPE1/TYPE2 change from "too little data" to **"permanently undetectable —
  the source annotations cannot express Mobitz type"**, which is a stronger and more useful claim.
- Integration brief: new label list, new macro definition, `low_confidence_eval` guidance.
- `models/README.md`, `docs/real-data-training.md` (v2 section), and close out
  `docs/ecg_sigma-change-requests.md` with the response status.

---

## Sequencing and effort

| Phase | Effort | Blocking |
|---|---|---|
| 1 Loader | ~2 h | yes — nothing runs without it |
| 2 Metrics | ~2 h | yes — must land before training so selection uses the right metric |
| 3 Smoke fold | ~2 h | gate before the full run |
| 3 Full CV | ~8 h | unattended |
| 4 Analyses | ~1 h | no |
| 5 Release + docs | ~2 h | no |

Roughly **half a day of engineering plus ~10 h of GPU**, most of it unattended.

---

## ecg_sigma's answers — resolved

| # | Question | Their answer | Effect here |
|---|---|---|---|
| 1 | Primary metric | Exclude **only VF** (validity); keep the noisy classes with CIs | Phase 2 rewritten and implemented |
| 2 | Final-model protocol | CV selects; commit to the artifact *before* touching test | **Decision needed** — ensemble vs single refit |
| 3 | `LeadMapper` | Keep our port + version assert this run; import at the next bump. Their `__init__` is now lazy (numpy + scipy only) | Assert implemented; no new dependency |
| 4 | SVT | Train the 13-class head now; real margin is 14 events, not 1 | Proceed with 13 classes |
| 5 | Per-record cap | Keep 50 — cap 20 costs 28 % of the corpus **and the SVT class**, and still does not clear AFL | No request |
| 6 | VF phase 3 | Expect low yield for a structural reason: VF is a resuscitation rhythm, 12-lead databases are resting outpatient ECGs. May be permanently unfixable from public data | Report VF as not deployment-validated; no plan change |
| 7 | `quality` floor | Don't. Global floors are ventricular-arrhythmia filters | Phase 4.4 changed to within-class percentile only |
| 8 | Paced events | None are in train; keep and report as a subgroup | Phase 4.3 implemented |

Two offers worth taking up: a **`reporting` block in `package.json`** so both teams compute the
same headline number, and their standing question on whether to lower
`inclusion_thresholds.train.events` from 100 to ~60 to give SVT margin — they will not do it
unilaterally, and I would decline: moving a threshold to admit a class is the move they warned
against.

## Artifact pre-commitment (recorded 2026-09-17, before test was touched)

**The deployed artifact is the 5-fold cross-validation ensemble**, and the v2 test number will be
the ensemble's. This is fixed now, in writing, because ecg_sigma's Q2 is explicit that computing
both an ensemble and a refit and reporting the better one is selecting on test.

- Selection signal: the mean held-out fold score. Test is spent **once**, at the end, on the
  ensemble.
- Enforced in code, not just by discipline: `train.py --cv-fold N` returns before the test
  evaluation block and prints "Test split not evaluated: cross-validation run."
- Rationale: v1 measured the ensemble beating the mean single seed by +0.019 accuracy and winning
  on val; `processor.py` already runs multi-checkpoint ensembles; and it avoids fixing the epoch
  count blind, which the single-refit route requires.
- If the ensemble disappoints on test, that is the reported result. Switching to the refit
  afterwards would invalidate the number.

### Fold validation (before training)

All 5 folds check out: no fit/select subject overlap, **no test subject in any fold**, selection
sets cover the train+val subjects exactly once, and all 13 classes appear in every selection fold.

| Fold | Fit events / subjects | Select events / subjects |
|---:|---:|---:|
| 0 | 16,316 / 7,300 | 4,094 / 1,831 |
| 1 | 15,818 / 7,308 | 4,592 / 1,823 |
| 2 | 16,387 / 7,305 | 4,023 / 1,826 |
| 3 | 16,573 / 7,306 | 3,837 / 1,825 |
| 4 | 16,546 / 7,305 | 3,864 / 1,826 |

## Superseded — decision required before Phase 3

**Which artifact do we commit to?**

| | 5-fold ensemble | Single refit on train+val |
|---|---|---|
| Test number comes from | the ensemble | the refit |
| Evidence | v1: ensemble beat the mean seed by +0.019 accuracy and won on val | trains on ~20 % more data; one model |
| Inference cost | 5 forward passes | 1 |
| Early stopping | per fold, honest | none left — epochs fixed to the fold median |

Recommendation: **the 5-fold ensemble**, because we have direct v1 evidence that ensembling helps
here, `processor.py` already supports it, and it avoids fixing the epoch count blind. Must be
declared before the test split is touched.
