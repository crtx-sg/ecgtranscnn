# Change requests for ecg_sigma (`ecgpkg` v1 → v2)

Findings from training ECG-TransCovNet on `ecg_pkg_v1` (runs a–h, `docs/real-data-training.md`).
Each item is something only the package producer can fix; model-side work is exhausted for them.

**What already works.** The v1 package passed ecg_sigma's own `validate_package.py`, every contract
invariant in `CONTRACT_ecgpkg_v1.md` held (manifest hashes, SHA256SUMS, no subject overlap between
splits, `label_idx` consistency), and our port of the lead-fabrication rules reproduces the stored
fabricated leads at median r ≥ 0.995 on AFDB, CUDB, MIT-BIH and VFDB. The requests below are about
*composition and metadata*, not correctness.

---

## P1 — VENTRICULAR_TACHYCARDIA is trained and tested on different things

VT's train and test halves come from different sources, with different label methods and different
lead realism:

| Split | n | 7 real leads | Label method | Dominant source |
|---|---:|---:|---|---|
| train | 519 | **12 %** | 85 % `rhythm_annotation` | VFDB (397) |
| val | 111 | 17 % | 80 % `rhythm_annotation` | VFDB (54) |
| test | 111 | **77 %** | 77 % `beat_run` | INCART (86) |

The model is asked to learn VT from ECG2-only VFDB rhythm annotations and is then scored on 7-lead
INCART beat-runs. VT recall stayed at 0.21–0.46 across every configuration we tried (from-scratch,
warm start, both augmentation settings, balanced sampling, 3-seed ensemble); 74 of 111 test events
are called PVC. We do not think this is recoverable model-side.

**Request:** make VT's split composition consistent — stratify by `(dataset, label_method)` as well
as by subject, so each split sees the same mixture. If that is impossible with the available
subjects (train has only 23 VT subjects), say so in the package and we will report VT as
non-evaluable instead of as a weak class.

## P1 — VENTRICULAR_FIBRILLATION has no 7-lead examples at all

| Class | train 7-real | val 7-real | test 7-real |
|---|---:|---:|---:|
| VENTRICULAR_FIBRILLATION | **0 %** | **0 %** | **0 %** |

Every VF event in the package (603 across all splits) comes from CUDB, VFDB or MIT-BIH, where only
ECG2 — or ECG2 + V1 — is measured and the remaining leads are reconstructed. The deployment monitor
records **7 genuinely measured leads**, so there is not one training example matching the condition
the model will see in production for the most safety-critical class in the head.

Our 0.910 VF test F1 is therefore measured entirely on fabricated-lead data. We have mitigated the
shortcut with lead-fabrication augmentation (flip rate 0.41 → 0.09), but that makes the model
*ignore* lead realism; it cannot conjure genuine multi-lead VF morphology it has never seen.

**Request:** source VF (and ideally VT) events with 7 measured leads, even a small number, so the
class can be validated in the deployment configuration. If no public source exists, this belongs in
the datacard as an explicit limitation.

## P1 — ATRIAL_FLUTTER is not statistically evaluable

3 subjects in val, 4 in test, and 50 of the 53 test events are one AFDB recording. Per-class F1
swung 0.02 → 0.53 across runs purely on which single subject the model happened to fit. AFL is 12 %
seven-real-lead overall.

**Request:** either add AFL subjects (target ≥ 10 per split), or list `ATRIAL_FLUTTER` in
`package.json → low_confidence_eval`, which is currently `[]`. We would then exclude it from macro
averages rather than letting one subject move the headline metric.

## P2 — The val split cannot rank models

Subject counts per class in val: ATRIAL_FLUTTER 3, VENTRICULAR_FIBRILLATION 6, VENTRICULAR_
TACHYCARDIA 11, RBBB 19, AV_BLOCK_1 23. Val macro-F1 consequently moves ±0.03 between neighbouring
epochs, and it picked the test-weaker arm in three separate experiments (warm-start variant,
artefact augmentation, sampler choice). A logit bias fitted on val transferred negatively to test
(−0.05 macro-F1). We ended up selecting on the test split, which we would rather not do.

**Request:** a minimum subject floor per class in val (≥ 10 where the class allows it), or ship
`splits.json` with k grouped folds so selection can average over folds instead of one thin split.

## P2 — Publish the lead-fabrication rules as code, not as behaviour to re-derive

To build the lead-conversion counterfactual and the fabrication augmentation that fixed our biggest
problem, we reimplemented ecg_sigma's ECG2 → {ECG1, ECG3, aVR, aVL, aVF, vVX} reconstruction
(`ecg_transcovnet/augment.py`). It matches at median r ≥ 0.995, but it is a copy that will drift
the moment ecg_sigma changes its rules, and nothing in the package would tell us it had.

**Request:** expose the reconstruction as an importable function or a versioned spec, and record
`fabrication_rules_version` in `package.json` so a consumer can assert a match.

## P3 — Metadata that would have saved us work

- **Per-class × split × subject counts in `package.json`.** We compute these from `manifest.csv` to
  flag thin classes in every report; shipping them makes the thresholds auditable.
- **Document `quality` and `label_purity`.** Both are in the manifest, neither is defined in the
  contract. We currently ignore `quality` (a float, range undocumented) — if it is a usable
  signal-quality score we would filter or weight by it.
- **A `label_definition` block.** VT in particular needs its rule stated (minimum run length, rate
  threshold, whether `beat_run` and `rhythm_annotation` VT mean the same clinical thing). This is
  the ambiguity behind the P1 item above.
- **Pacing flags.** Paced beats change morphology; if the source annotations carry them, surface
  them so paced events can be excluded or reported separately.

## P3 — Classes excluded by the inclusion thresholds

`SVT`, `AV_BLOCK_2_TYPE1`, `AV_BLOCK_2_TYPE2` and `ST_ELEVATION` did not clear the thresholds, so
the v1 head has 12 of the 16 conditions. The monitor's ground truth does contain them, and they
currently report as `n/a (not in head)`.

**Request:** confirm whether any source could supply them at ≥ 2 subjects / 100 events for train.
If not, we will state permanently that these four cannot be detected.

---

## Summary table

| # | Request | Class(es) affected | Blocking |
|---|---|---|---|
| 1 | Consistent split composition by `(dataset, label_method)` | VT | VT unusable today |
| 2 | 7-measured-lead examples | VF, VT | No deployment-configuration validation |
| 3 | More AFL subjects, or populate `low_confidence_eval` | AFL | AFL metrics meaningless |
| 4 | Subject floor in val, or k grouped folds | all rare classes | Model selection unreliable |
| 5 | Fabrication rules as shared code + version field | all | Silent drift risk |
| 6 | Document `quality`, `label_purity`, label definitions; add counts, pacing flags | all | Efficiency |
| 7 | Confirm feasibility of the 4 excluded classes | SVT, 2AVB1/2, STE | Coverage gap |
