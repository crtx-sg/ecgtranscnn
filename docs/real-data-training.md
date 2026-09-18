# Training ECG-TransCovNet on real ECG (`ecgpkg` v1)

Status: v1 runs a–h complete (`models/real_v1`); **v2 retrained on `ecg_pkg_v2` with 5-fold CV (`models/real_v2`)** — see "Package v2" at the end.

## Data

Package `ecg_sigma/packages/ecg_pkg_v1` (validated with ecg_sigma `validate_package.py`):
25,605 included events (train 18,123 / val 3,436 / test 4,046), 12 classes, subject-grouped
splits, 2400-sample windows (MIT-BIH, INCART, VFDB, CUDB, AFDB) and 2000-sample PTB-XL windows.

Test split, events (subjects):

| Class | Test events | Test subjects | Main test sources |
|---|---:|---:|---|
| NORMAL_SINUS | 1,619 | 800 | PTB-XL 861, INCART 458, MIT-BIH 300 |
| SINUS_BRADYCARDIA | 396 | 59 | INCART 197, MIT-BIH 145, PTB-XL 54 |
| SINUS_TACHYCARDIA | 192 | 50 | INCART 136, PTB-XL 46 |
| ATRIAL_FIBRILLATION | 422 | 103 | AFDB 210, PTB-XL 112, MIT-BIH 100 |
| ATRIAL_FLUTTER ⚠ | 53 | 4 | AFDB 50 |
| PAC | 142 | 34 | MIT-BIH 67, INCART 50, PTB-XL 25 |
| PVC | 838 | 61 | INCART 585, MIT-BIH 203, PTB-XL 50 |
| VENTRICULAR_TACHYCARDIA | 111 | 7 | INCART 86, VFDB 13, MIT-BIH 12 |
| VENTRICULAR_FIBRILLATION | 91 | 6 | VFDB 50, CUDB 41 |
| LBBB | 81 | 26 | MIT-BIH 50, PTB-XL 31 |
| RBBB | 71 | 20 | MIT-BIH 50, PTB-XL 21 |
| AV_BLOCK_1 | 30 | 28 | PTB-XL 30 |

⚠ fewer than 5 test subjects: ATRIAL_FLUTTER (4). VT (7) and VF (6) are also thin; treat
their per-class numbers as indicative.

### Lead-realism confound

`real_lead_mask` is `1111111` for INCART and PTB-XL, `0100001` for MIT-BIH (ECG2 + V1 measured)
and `0100000` for VFDB, CUDB and AFDB (only ECG2 measured). All VF test events and 50/53
atrial-flutter test events are single-lead sources, so plain test metrics reward a model that
uses "fabricated leads" as a class cue. The deployment monitor records **7 real leads**, where
that cue does not exist. Two measurements expose it:

- per-real-lead-mask tables in every report;
- the **lead-conversion counterfactual** (`scripts/evaluate.py --lead-conversion both`): every
  event with more measured leads is re-predicted after its non-ECG2 leads are rebuilt from ECG2
  with ecg_sigma's own rules (our port reproduces the stored fabricated leads with median
  r ≥ 0.995 on AFDB, CUDB, MIT-BIH and VFDB). A rhythm-based model should rarely change its
  prediction.

## Pipeline measurements

| Item | Result |
|---|---|
| Vectorised preprocessing (`default`, 7 × 2000) | 19–24 ms → 3.5 ms per item, bit-identical output |
| v1 cache build (16 processes) | 52 s, 1.72 GB on disk, peak RSS 780 MB (610 MB is the torch import) |
| Item cost (eval / + artefacts / + lead fabrication) | 3.5 / 5.8 / 8.0 ms |
| Training step, batch 64, RTX 4050 | FP32 40–70 ms, AMP 137 ms → package runs use FP32 |
| Epoch on v1 (14 workers) | 35–48 s |

Host note: the WSL VM showed soft lockups and clock jumps while other GPU containers (vLLM) were
restarting; affected epochs report inflated wall times. `vllm` was stopped during experiments.

## Experiments

Common settings: `--epochs 40 --warmup-epochs 3 --patience 10 --crop-len 2000
--filter-preset default --workers 14`, AdamW 5e-4, focal loss γ = 2, checkpoint selected on
val macro-F1. Test metrics below are centre-crop 2000 unless stated.

### a — Baseline: simulator checkpoints in baseline mode (v1 test)

| Checkpoint | Accuracy | Macro-F1 (95 % CI) | Macro recall | Macro specificity | Full-length acc / F1 | Flip rate `0100001` / `0100000` |
|---|---:|---|---:|---:|---|---|
| `models/best_model.pt` | 0.124 | 0.098 (0.063–0.125) | 0.191 | 0.956 | 0.125 / 0.100 | 0.60 / 0.57 |
| `models/noise_robust/best_model.pt` | 0.255 | 0.176 (0.133–0.211) | 0.256 | 0.960 | 0.263 / 0.181 | 0.47 / 0.40 |
| `models/avblock_fix/best_model.pt` | 0.101 | 0.090 (0.060–0.121) | 0.144 | 0.945 | 0.106 / 0.096 | 0.74 / 0.66 |

These are higher than the 4.8–8.3 % measured by ecg_sigma on its earlier 3,802-event sample
(different events, labels and class mix) but confirm the collapse: PVC recall ≤ 0.07, PAC,
VT, LBBB and RBBB near 0, and large shares of predictions on VF/AF/AFL. Reports:
`models/experiments/a_baseline_*/test.md`.

### v0 sanity run (MIT-BIH + INCART, 9 classes, 15 epochs)

Val accuracy 0.804, macro-F1 0.706; test accuracy 0.636 (full length 0.664), macro-F1 0.429.
AF, LBBB and RBBB have one test subject each in v0 and score 0 — v0 is only used to confirm
the pipeline learns.

### b — Package-only, from scratch

Best val macro-F1 0.633 at epoch 29, early stop at epoch 39.

| Section | Accuracy | Macro-F1 (95 % CI) | Macro recall | Macro specificity | Macro AUROC |
|---|---:|---|---:|---:|---:|
| crop 2000 | 0.751 | 0.607 (0.530–0.724) | 0.675 | 0.975 | 0.952 |
| full length | 0.766 | 0.617 (0.543–0.729) | 0.681 | 0.977 | 0.956 |

Per class, confusion matrix and per-source tables: README "Model Performance" and
`models/experiments/v1_b/reports/test.md`. Weakest classes: ATRIAL_FLUTTER (F1 0.14, 4 test
subjects, 46/53 called AF), PAC (0.28), AV_BLOCK_1 (0.41, precision 0.26), VT (0.52, half called
PVC), RBBB (0.54, precision 0.38), LBBB (0.61, recall 0.46, called PVC).
Per source: MIT-BIH is hardest (acc 0.555, macro-F1 0.536); PTB-XL easiest (0.877 / 0.797).

### c — Package-only, warm start from `models/avblock_fix/best_model.pt`

Everything except the decoder queries is reused; the two arms differ in how the 12 queries are
built — `--init-queries by_name` copies the query of every legacy class whose name survives into
the v1 head (9 of 12) and reinitialises the rest, `--init-queries reinit` starts all 12 fresh.

| Arm | Best val macro-F1 (epoch) | Test acc / macro-F1 (crop 2000) | Test acc / macro-F1 (full) |
|---|---|---:|---:|
| `by_name` | 0.660 (12) | 0.735 / 0.652 | 0.752 / 0.662 |
| `reinit` | **0.690** (15) | 0.755 / 0.646 | 0.767 / 0.646 |

Warm starting is worth it: both arms beat run b on test macro-F1 (0.607) and reach their best
epoch in 12–15 epochs instead of 29. The two arms are level on test (0.646 vs 0.652) even though
val separates them by 0.03 — with 3 atrial-flutter and 6 VF subjects in val, val macro-F1 moves
by that much on its own, so the ordering is not a real difference. Per class they trade places:
`reinit` is better on ATRIAL_FLUTTER (0.53 vs 0.25) and LBBB (0.80 vs 0.50), `by_name` on PAC
(0.33 vs 0.17) and AV_BLOCK_1 (0.60 vs 0.37).

Runs d–f take the val-selected winner (`reinit`) as their base, which is the rule fixed before
the runs; see the caveat under d.

### d — Noise augmentation 0 vs 0.5

The 0.5 arm is run c `reinit` itself; `v1_d_noise0` repeats it with `--noise-aug-prob 0`.

| Arm | Best val macro-F1 | Test acc / macro-F1 (crop 2000) | Flip rate `0100000` |
|---|---:|---:|---:|
| `--noise-aug-prob 0.5` (c reinit) | 0.690 | **0.755 / 0.646** | 0.391 |
| `--noise-aug-prob 0` | **0.699** | 0.699 / 0.614 | 0.408 |

Turning artefact injection off wins on val and loses on test: ATRIAL_FLUTTER 0.53 → 0.13,
VENTRICULAR_FIBRILLATION 0.82 → 0.74, SINUS_BRADYCARDIA 0.75 → 0.61. On its own,
`--noise-aug-prob 0.5` is therefore the better setting and the val gain is selection noise on the
thin classes — **but this does not survive being combined with lead-fabrication augmentation**;
see the confirmation run below.

Caveat on the experiment chain: runs e and f were launched from the val winner (noise aug 0)
before these test numbers existed. Every e/f comparison below therefore shares that base and
differs from its own control in exactly one flag, so the conclusions hold, but the absolute
numbers in e and f are measured on a base that is ~0.03 macro-F1 below the best c arm.

### e — `--fabricated-leads zero`

Trains and evaluates with every lead that ecg_sigma fabricated from ECG2 set to zero, so the
model can only use the measured leads.

| Evaluation | Acc / macro-F1 (crop 2000) | Acc / macro-F1 (full) |
|---|---:|---:|
| fabricated leads zeroed (matched, `eval/test.md`) | 0.719 / 0.629 | 0.733 / 0.634 |
| fabricated leads kept (`eval/test_keep.md`) | 0.718 / 0.573 | 0.733 / 0.591 |
| control: run d, same base, leads kept | 0.699 / 0.614 | 0.716 / 0.633 |

Zeroing costs nothing overall (0.629 vs 0.614 for the control) and redistributes accuracy:
LBBB 0.74 → 0.93, RBBB 0.83 → 0.92, ATRIAL_FIBRILLATION 0.61 → 0.74, while ATRIAL_FLUTTER
collapses to 0.02 (all 50 AFDB test events are single-lead sources whose only real signal is
ECG2) and VT drops 0.52 → 0.48. Handed real fabricated leads at test time the same model falls
to 0.573 — it never learned to use them — so zeroing is not a deployment option for a monitor
that sends 7 real leads.

The useful reading of e is that the *content* of the fabricated leads carries little class
information, while *which* leads are fabricated carries a lot: the keep-trained control still
loses 0.589 → 0.371 macro-F1 under the ECG2-only counterfactual. Run f attacks that directly.

Note on the counterfactual sections for this run: `PackageDataset.__getitem__`
(`ecg_transcovnet/package.py:415-433`) fabricates first and zeroes afterwards, so for a
`fabricated_leads=zero` model `leadconv_*` means "fewer measured leads, the rest zeroed" —
information removal, not a realism probe. Compare e against d, not against its own leadconv rows.

### Confirmation run — noise 0.5 **and** lead-fab 0.5 (`v1_base_n05_lf05`)

d says keep artefact injection, f says add lead fabrication, so the two were combined on the same
warm-started base. The combination is worse than fabrication alone:

| Run | noise aug | lead-fab aug | Best val macro-F1 (epoch) | Epochs run | Test acc / macro-F1 |
|---|---:|---:|---|---:|---:|
| f p = 0.5 | 0.0 | 0.5 | 0.709 (34) | 40 | 0.799 / 0.649 |
| confirmation | 0.5 | 0.5 | 0.677 (5) | 15 (early stop) | 0.743 / 0.606 |

VENTRICULAR_TACHYCARDIA collapses (F1 0.115, recall 0.063) and ATRIAL_FLUTTER falls to 0.10,
while LBBB (0.93) and VF (0.94) improve. Read this result with care: the run peaked at epoch 5 and
`--patience 10` stopped it at epoch 15, whereas f p = 0.5 did not reach its best until epoch 34, so
part of the gap is early stopping on a lucky early peak rather than the configuration itself. A
fair rematch needs a longer patience or a minimum-epoch floor. Until then, **the deployed
configuration is lead-fabrication augmentation without artefact injection**, and runs g and h are
based on f p = 0.5.

### f — Lead-fabrication augmentation 0 / 0.3 / 0.5

During training, an event whose leads are all measured is rebuilt from its own ECG2 with
ecg_sigma's fabrication rules with probability p, so lead realism stops predicting the class.
The p = 0 arm is run d.

| p | Best val macro-F1 | Test acc / macro-F1 (crop 2000) | Test acc / macro-F1 (full) | Flip `0100001` / `0100000` | Converted macro-F1 `0100000` |
|---:|---:|---:|---:|---:|---:|
| 0.0 (d) | 0.699 | 0.699 / 0.614 | 0.716 / 0.633 | 0.311 / 0.408 | 0.371 |
| 0.3 | 0.697 | 0.795 / **0.653** | 0.813 / **0.676** | 0.081 / 0.156 | 0.540 |
| 0.5 | **0.709** | **0.799** / 0.649 | 0.806 / 0.658 | **0.068 / 0.094** | **0.622** |

This is the single largest effect in the change, and it moves both axes at once. The
lead-conversion flip rate falls from 0.41 to 0.09 and the counterfactual macro-F1 gap closes
from 0.589 → 0.371 (a 0.218 drop) to 0.698 → 0.622 (0.076), while plain test accuracy rises ten
points. Per real-lead mask, on the deployment-like 7-measured-lead subset:

| Run | `1111111` acc / macro-F1 | `0100001` acc / macro-F1 | `0100000` acc / macro-F1 |
|---|---:|---:|---:|
| d (p = 0) | 0.784 / 0.702 | 0.451 / 0.467 | 0.698 / 0.541 |
| f p = 0.3 | 0.821 / 0.710 | 0.755 / 0.721 | 0.706 / 0.545 |
| f p = 0.5 | 0.826 / 0.712 | 0.758 / 0.721 | 0.703 / 0.543 |

MIT-BIH (mask `0100001`), the hardest source for every earlier run, gains 30 accuracy points and
stops being an outlier. The remaining weak classes are ATRIAL_FLUTTER (0.11 / 0.04 — 50 of its 53
test events are one AFDB subject, and the cue that used to carry it was lead realism),
VENTRICULAR_TACHYCARDIA (0.51 / 0.40, mostly called PVC), RBBB precision (0.41 / 0.39, over-called
on PTB-XL) and AV_BLOCK_1 precision (0.28 / 0.25).

p = 0.3 and p = 0.5 are within noise of each other on test macro-F1; p = 0.5 is better on val, on
accuracy and on every shortcut measure, p = 0.3 on full-length macro-F1.

### g — Balanced sampler vs class-weighted loss

Both arms use the f p = 0.5 configuration and differ in one mechanism only. The control
(`v1_f_leadfab05`) draws batches with plain shuffling and applies inverse-frequency weights to the
focal loss; `v1_g_balanced` uses `--sampler balanced` (class weight ÷ subject count^0.5, sampling
with replacement), which `--class-weights auto` automatically leaves unweighted
(`scripts/train.py:406`), so no class is compensated twice.

| Arm | Best val macro-F1 (epoch) | Test acc / macro-F1 (crop 2000) | Test acc / macro-F1 (full) |
|---|---|---:|---:|
| class-weighted loss (f p = 0.5) | 0.709 (34) | **0.799 / 0.649** | 0.806 / 0.658 |
| balanced sampler | **0.720** (34) | 0.773 / 0.620 | 0.787 / 0.624 |

The sampler wins on val by 0.011 and loses on test by 0.029, and the per-class pattern is a real
trade rather than noise. Oversampling helps the rare record-level classes — LBBB 0.81 → 0.92,
AV_BLOCK_1 0.39 → 0.47, ATRIAL_FIBRILLATION 0.85 → 0.87 — and hurts the rate-derived and
single-lead-source ones: SINUS_BRADYCARDIA 0.72 → 0.44 (recall 0.32), VENTRICULAR_FIBRILLATION
0.90 → 0.77, VENTRICULAR_TACHYCARDIA 0.40 → 0.35. Repeating a rare subject's events many times per
epoch buys generalisation only where the class has many subjects to draw from; VF has 18 train
subjects and AFL 36, so the sampler mostly re-shows the same recordings.

**Class-weighted loss is kept.** This is the third time val macro-F1 has preferred the arm that
loses on test (after c and d), so the choice was made on the test report instead — a deliberate
departure from the pre-registered select-on-val rule, recorded here because it means the test split
has been used for model selection three times (base for g/h, the g arm, and the h configuration).
The remaining test numbers should be read as mildly optimistic for that reason; a clean estimate
needs a split this change has not touched.

### h — Val-fitted logit bias and 3-seed ensemble

Three seeds (42, 43, 44) of the f p = 0.5 configuration, then a per-class logit bias fitted on the
val split (`scripts/evaluate.py --calibrate-on val`) and a logit-averaged ensemble of the three.

| Model | Best val macro-F1 | Test acc / macro-F1 (crop 2000) | Test acc / macro-F1 (full) | Flip `0100001` / `0100000` |
|---|---:|---:|---:|---:|
| seed 42 (`v1_f_leadfab05`) | 0.709 | 0.799 / 0.649 | 0.806 / 0.658 | 0.068 / 0.094 |
| seed 43 | 0.723 | 0.798 / 0.667 | 0.818 / **0.685** | 0.060 / 0.088 |
| seed 44 | 0.699 | 0.775 / 0.633 | 0.785 / 0.635 | 0.071 / 0.101 |
| **3-seed ensemble** | — | **0.814** / 0.668 | **0.829** / 0.678 | **0.043 / 0.076** |
| seed 42 + val logit bias | — | 0.769 / 0.599 | 0.781 / 0.606 | 0.029 / 0.052 |
| ensemble + val logit bias | — | 0.790 / 0.628 | 0.808 / 0.644 | — |

**Seed spread is the headline.** Three runs that differ only in seed span 0.633–0.667 test macro-F1
(mean 0.650, spread 0.034). That is larger than the gap that decided g (0.029), larger than the two
c arms (0.006) and comparable to d (0.032). Every single-run comparison in this document below
about 0.03 macro-F1 is therefore **not resolvable** — c, d and g should be read as "no measurable
difference", and only f's effect (+0.04 macro-F1, +10 points accuracy, flip rate 0.41 → 0.09) is
clearly outside seed noise. Ranking future configurations needs 3 seeds per arm.

**The ensemble helps accuracy, not macro-F1.** It gains 1.5 points of accuracy over the best single
seed and matches it on macro-F1 (0.668 vs 0.667), while on full-length inputs it is marginally
behind (0.678 vs 0.685). It does reduce the counterfactual flip rate further (0.094 → 0.076). Per
class it is the most balanced model produced: AF 0.89, PVC 0.88, VF 0.91, LBBB 0.92, NS 0.86, and
it holds SINUS_BRADYCARDIA at 0.76 where the balanced sampler lost it. The two classes it cannot
fix are VT (recall 0.21 — the ensemble is more conservative than any single seed, precision 0.74)
and ATRIAL_FLUTTER (recall 0.057 at precision 1.00).

**Val bias calibration is harmful** and should not be used: −0.050 macro-F1 on the single model,
−0.040 on the ensemble. The fitted bias is `NORMAL_SINUS +1.0, PVC +1.2, PAC +0.8, SINUS_TACHYCARDIA
−0.6, LBBB −0.6`, i.e. it corrects under-prediction that exists on a val split with 3 AFL and 6 VF
subjects and does not exist on test. Same root cause as the val-vs-test disagreements in c, d and g.

Recommended deployment model: the **3-seed ensemble**, uncalibrated.

On test the ensemble and seed 43 look tied on macro-F1 (0.668 vs 0.667), but seed 43 is the
*maximum of three* picked on the test split, which with mean 0.650 and spread 0.034 is worth about
+0.017 of pure optimism. Two comparisons that do not have that problem both favour the ensemble:

| Model | val acc | val macro-F1 | test acc | test macro-F1 |
|---|---:|---:|---:|---:|
| seed 42 | 0.777 | 0.709 | 0.799 | 0.649 |
| seed 43 | 0.779 | 0.723 | 0.798 | 0.667 |
| seed 44 | 0.772 | 0.699 | 0.775 | 0.633 |
| **ensemble** | **0.797** | **0.733** | **0.814** | **0.668** |

- On **val** — the split each member's checkpoint was selected to maximise, so biased *towards*
  the single models — the ensemble still wins both metrics, beating even seed 43 by 0.018 accuracy
  and 0.010 macro-F1.
- Against the **average single seed**, the fair comparator for "train one model with this recipe",
  the ensemble gains +0.019 accuracy and +0.018 macro-F1 on test.

It also has the lowest counterfactual flip rate (0.076 vs 0.088–0.101), which matters most for the
7-real-lead monitor. Cost is three forward passes; `processor.py --checkpoint a.pt b.pt c.pt`
averages their softmax outputs with the same rule as `evaluate.py`, so live and offline numbers
agree.

### Release — `models/real_v1/`

| File | Contents |
|---|---|
| `seed42.pt`, `seed43.pt`, `seed44.pt` | the three ensemble members |
| `best_model.pt` | copy of `seed43.pt` — single-model fallback for one-checkpoint tooling |
| `confusion_matrix.png`, `reports/` | ensemble test report, val report, lead-conversion matrices |
| `training_curves.png`, `history.json` | seed 43's training run |

## Known limitations

- **Single-run comparisons below ~0.03 macro-F1 are not resolvable.** Three seeds of one
  configuration span 0.633–0.667 (run h), so the c, d and g conclusions are within noise and are
  reported as "no measurable difference". Only f's effect is clearly outside it. Future arms need
  3 seeds each.
- **Class-level selection is noisy.** val has 3 ATRIAL_FLUTTER and 6 VF subjects, so val
  macro-F1 moves ±0.03 between neighbouring epochs. It picked the test-weaker arm in c, d and g,
  and the val-fitted logit bias in h transferred negatively (−0.05 macro-F1).
- **The test split was used for model selection** three times (the g/h base, the g arm, the h
  configuration), on the user's explicit decision after val proved unreliable. Test numbers here
  are therefore mildly optimistic as an estimate of unseen-data performance.
- **ATRIAL_FLUTTER is not measurable here.** 50 of 53 test events come from a single AFDB
  subject; once lead realism is removed (run f) its F1 falls to 0.04–0.11. Any AFL claim from
  this package is about one subject.
- **VT, VF, AV_BLOCK_1 are thin** (7, 6 and 28 test subjects) and VT is systematically confused
  with PVC — the two differ by run length, and the labels come from different methods
  (`beat_run` vs `beat_morphology`).
- **Artefact injection is off in the deployed configuration.** The confirmation run that combined
  it with lead fabrication scored 0.606, but it early-stopped at epoch 15 from a lucky epoch-5 peak,
  so that config has not had a fair 40-epoch run.
- **The host is not a stable timing environment.** Epoch times ranged 22–140 s for identical work
  while other GPU containers were restarting; wall-clock numbers here are indicative only.
- **VT and ATRIAL_FLUTTER remain unsolved.** The ensemble reaches VT recall 0.21 (precision 0.74)
  and AFL recall 0.057 (precision 1.00); both are conservative rather than wrong, but neither is
  usable as an alarm source from this package.
- **Some limits are the package's, not the model's.** VT is trained on 12 %-seven-lead VFDB rhythm
  annotations and tested on 77 %-seven-lead INCART beat-runs; VF has no seven-measured-lead events
  in any split. Requests raised with ecg_sigma in `docs/ecg_sigma-change-requests.md`.

---

# Package v2 (`ecg_pkg_v2`) — 5-fold CV ensemble

ecg_sigma delivered v2 (now v2.2) in response to `docs/ecg_sigma-change-requests.md` (their reply and the
follow-up addendum are in `docs/RESPONSE_change_requests.md`). Migration plan and the verification
of their claims: `docs/v2-migration-plan.md`.

`models/real_v2` was trained on `ecg_pkg_v2`. **v2, v2.1 and v2.2 are the same data** —
`manifest.csv` and `splits.json` are byte-identical across all three, verified here, and the later
versions only add `package.json` metadata (`reporting`, `corpus_diagnostics`) and the `paced_record`
column. The checkpoints and every number below therefore stand unretrained; new work should point at
**v2.2**.

**v1 and v2 numbers are not comparable.** The head gained SVT (12 → 13 classes), every subject→split
assignment changed, and the primary metric changed. Nothing below should be read against the v1
tables above.

## Protocol

Pre-committed before the test split was touched: **the artifact is the 5-fold ensemble** and the
test number is the ensemble's. Enforced in code — `train.py --cv-fold N` returns before the test
evaluation block.

- 5 grouped folds over train+val subjects, stratified by `(condition, dataset, label_method)`;
  test appears in no fold. Verified: no fit/select overlap, no test subject in any fold, selection
  sets cover train+val exactly once, 13/13 classes present in every selection set.
- Recipe carried over from v1 unchanged: warm start from `models/avblock_fix/best_model.pt`
  (`--init-queries reinit`, simulator-trained so no v1 subject leakage), `--lead-fab-aug-prob 0.5`,
  `--noise-aug-prob 0`, class-weighted focal loss, `--crop-len 2000 --filter-preset default`,
  40 epochs, patience 10.
- Primary metric: **macro-F1 over 12 of 13 classes**, excluding VENTRICULAR_FIBRILLATION, whose
  `no_seven_real_lead_events` flag makes its score uninterpretable for deployment. The precision-
  flagged classes (AFL, SVT, VT) stay in with a subject-level CI — dropping them is what flatters.

## Cross-validation

| Fold | Held-out macro-F1 (12 cls) | Best epoch | Epochs run |
|---:|---:|---:|---:|
| 0 | 0.705 | 10 | 20 |
| 1 | 0.655 | 23 | 33 |
| 2 | 0.591 | 37 | 40 |
| 3 | 0.521 | 12 | 22 |
| 4 | 0.614 | 8 | 18 |
| | **mean 0.617, sd 0.069** | median 12 | |

**The fold sd of 0.069 is the most important number in this document.** It is the variance from
*which subjects* land in the held-out set, and it is twice the 0.034 seed spread measured on v1,
which varied initialisation against one fixed split. Every v1 configuration comparison had a margin
of 0.006–0.032 — all inside this band. Read runs c, d, g and the p = 0.3 vs 0.5 choice as
"no measurable difference", and require the 5-fold mean before believing any future config change.

## Test — 5-model ensemble, one evaluation

| Section | Primary macro-F1 (12 cls) | All-class macro-F1 | Accuracy |
|---|---:|---:|---:|
| crop 2000 | **0.587** (95 % CI 0.506–0.684) | 0.606 | 0.782 |
| full length | 0.593 | — | 0.790 |

Test primary (0.587) sits inside the CV range (0.521–0.705), 0.030 below the CV mean — within half
a fold-sd, and expected: folds are over train+val subjects, test is a disjoint subject set.

| Class | F1 | Recall | Precision | Test subjects | Package flags |
|---|---:|---:|---:|---:|---|
| NORMAL_SINUS | 0.873 | 0.827 | 0.924 | 653 | — |
| RBBB | 0.920 | 0.972 | 0.873 | 20 | real_lead_skew, test_record_dominates |
| SINUS_TACHYCARDIA | 0.837 | 0.917 | 0.771 | 48 | — |
| PVC | 0.833 | 0.804 | 0.865 | 60 | — |
| ATRIAL_FIBRILLATION | 0.753 | 0.818 | 0.699 | 104 | — |
| SINUS_BRADYCARDIA | 0.743 | 0.824 | 0.676 | 58 | — |
| PAC | 0.551 | 0.594 | 0.514 | 31 | — |
| LBBB | 0.531 | 0.370 | 0.938 | 26 | real_lead_skew, label_method_skew, test_record_dominates |
| VENTRICULAR_TACHYCARDIA | 0.421 | 0.462 | 0.387 | 9 | few_val/test_subjects, test_record_dominates |
| AV_BLOCK_1 | 0.394 | 0.900 | 0.252 | 28 | — |
| ATRIAL_FLUTTER | 0.098 | 0.062 | 0.231 | 6 | few_val/test_subjects, test_record_dominates |
| SVT | 0.089 | 0.080 | 0.100 | 7 | few_val/test_subjects |
| *VENTRICULAR_FIBRILLATION* | *0.840* | *0.855* | *0.826* | *10* | *no_seven_real_lead_events* — **fabricated-lead only, not deployment-validated** |

### Deployment configuration is now the best subset

| Real-lead mask | Events | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| `1111111` (7 measured — deployment) | 2,441 | **0.826** | **0.715** |
| `0100001` (ECG2 + V1) | 728 | 0.659 | 0.484 |
| `0100000` (ECG2 only) | 344 | 0.727 | 0.464 |

Lead-conversion counterfactual: flip rate 0.070 (`0100001`) and 0.167 (`0100000`), macro-F1
0.715 → 0.676 and 0.589 → 0.490. The shortcut remains suppressed by lead-fabrication augmentation.

## What the v2 split fixed — the VT→PVC collapse

The single clearest measure that the v2 stratification worked, and the justification for the whole
migration:

| Package | VT test events predicted as PVC | VT recall |
|---|---:|---:|
| v1 | **74 of 111** | 0.207 (3-seed ensemble) |
| v2 | **6 of 39** | 0.462 (5-fold ensemble) |

On v1, VT was trained on VFDB episodes (12 % seven-measured-lead, `rhythm_annotation`) and tested on
INCART beat-runs (78 %, `beat_run`) — a train/test shift inside the class. v2 stratifies subjects by
`(condition, dataset, label_method)`, so both sides see the same mixture. Nothing on the model side
changed: same architecture, same recipe, same augmentation.

VT is still the weakest ventricular class, but the failure mode is different: its errors now go to
ATRIAL_FIBRILLATION (17) and VENTRICULAR_FIBRILLATION (12), with PVC down to 6. Neither team was
looking for this number; it came out of the `source_beat_condition` analysis ecg_sigma requested.

## Three findings for ecg_sigma

**1. Their VT hypothesis is refuted — the split is the other way round.** They predicted VT recall
would stay low on `beat_run` and recover on `rhythm_annotation`, pointing at the 3-beat-run
boundary. Measured:

| VT `label_method` | F1 | Recall | Test events / subjects |
|---|---:|---:|---|
| `beat_run` | **0.828** | 0.706 | 17 / 2 |
| `rhythm_annotation` | **0.425** | 0.393 | 61 / 7 |

VT is detected *well* on the audited beat-runs and poorly on the adjudicated episodes — the reverse
of the prediction. Caveat: `beat_run` rests on 2 subjects. If it holds, the difficulty is in the
sustained VFDB/CUDB episodes, not the run-length boundary.

**2. SVT is in the head but not learnable at this volume.** F1 0.089, recall 0.080, and **AUROC
0.632** — barely above chance, the only class in the head whose ranking is near-random. It entered
at exactly 100 train events (30 subjects, 3 VFDB subjects holding 90 of 154 eligible events). We
followed their advice to train the 13-class head and flag it; the flag is doing real work.

**3. Paced events degrade, as they suspected.** Accuracy 0.616 on the 86 paced test events vs 0.786
unpaced. It concentrates in ATRIAL_FIBRILLATION: F1 0.32 on 35 paced events (1 subject) against
0.753 overall. PVC holds up (0.78 vs 0.833). One patient, so indicative — but it is the failure mode
they flagged, and the model never sees a paced beat in training.

## Release — `models/real_v2/`

| File | Contents |
|---|---|
| `fold0.pt` … `fold4.pt` | the 5 ensemble members — **this is the artifact** |
| `best_model.pt` | copy of `fold0.pt`, single-model fallback for one-checkpoint tooling |
| `cv_summary.json` | per-fold scores, protocol, primary-metric definition |
| `reports/test.{md,json}`, `confusion_matrix.png` | the single ensemble test evaluation |

```bash
python scripts/processor.py --watch-dir data/inference \
    --checkpoint models/real_v2/fold0.pt models/real_v2/fold1.pt models/real_v2/fold2.pt \
                 models/real_v2/fold3.pt models/real_v2/fold4.pt
```
