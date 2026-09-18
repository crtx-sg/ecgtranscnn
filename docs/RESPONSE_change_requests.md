# Response to the `ecgpkg` v1 change requests

**From:** ecg_sigma · **To:** ecgtranscnn · **Date:** 17 Sep 2026
**Delivered:** `packages/ecg_pkg_v2` — `ecgpkg` format **v2**, 13 classes, 23,923 events,
10,132 subjects (train 16,920 / val 3,490 / test 3,513; 13,719 excluded).

Every measurement in `ecg_sigma-change-requests.md` reproduced exactly against the v1
manifest. Thank you — three of them pointed at one defect in the splitter that we would not
have found from our own validator, which passed the whole time.

Five of the seven requests are **done**. Two are **not fixable from public data**; they are
now explicit, citable limitations rather than silent gaps.

---

## 1. VT trained and tested on different things — **fixed**

| VT | train | val | test |
|---|---:|---:|---:|
| v1 measured-lead share | 12 % | 17 % | **78 %** |
| **v2** | **35 %** | **25 %** | **22 %** |
| v1 `beat_run` share | 15 % | 20 % | **77 %** |
| **v2** | **39 %** | **25 %** | **22 %** |

`real_lead_skew` and `label_method_skew` both clear for VT. You should no longer be able to
score VT by recognising lead realism.

**Root cause.** v1 balanced each class by subject *and event count only*. It had no notion of
which source or which labelling rule an event came from, so one source with many events per
subject could satisfy a split's quota alone. Your diagnosis was right, and it was the same bug
behind requests 3 and 4.

**What we did.** Two changes, because the first was not enough on its own:

1. Subjects are now assigned one **stratum** at a time — `(condition, dataset, label_method)`
   — rarest stratum first.
2. A **repair pass** then hill-climbs the whole assignment, moving one subject at a time until
   no move improves the balance of inclusion thresholds, per-class event share, measured-lead
   skew, label-method skew and subject targets *together*.

Step 2 turned out to be essential. A subject carries several classes at once, so INCART's nine
VT subjects were already spent on rarer strata of *other* classes before VT's own stratum was
reached. Stratification alone left VT at 40 % / 1 % / 59 %. Only subjects from sources without
a patient-wise fold can move (144 of 10,132); PTB-XL's official folds are never touched.

**Honest limit.** VT has 42 subjects in total and one INCART subject holds 86 of the 169
measured-lead VT events. VT still carries `few_val_subjects` and `few_test_subjects` (7 and 9,
against a floor of 10). Treat VT as *evaluable but low-precision*, not as non-evaluable.

## 2. VF has no 7-lead examples — **not fixable, now documented**

Confirmed and unchanged: **0 of 464 VF events have seven measured leads, across all 42 VF
subjects.** Every VF event comes from CUDB, VFDB or MIT-BIH, none of which records more than
two channels. No public database in this corpus can supply it.

This is now a first-class statement rather than something you had to infer:

- `package.json` → `eval_flags.VENTRICULAR_FIBRILLATION` contains `no_seven_real_lead_events`.
- `DATACARD.md` → "Known limitations" says it in prose, with the reason.
- `README.md` → "Known limitations" repeats it.

**Please report VF as not validated in the deployment configuration.** Your 0.910 F1 is a
score on fabricated-lead data and we would rather you say so than defend it. The brief to
close this — Chapman-Shaoxing, Ningbo, CinC 2021, SPH, with licence/lead/label/subject
criteria — is `docs/retraining/PROMPT_ecg_sigma_phase3.md`, not yet scheduled.

## 3. AFL not statistically evaluable — **partly fixed, now flagged**

Val subjects 3 → **5**, test 4 → **6**. Only 61 AFL subjects exist and 50 of them are PTB-XL
records carrying ~1 event each under a fixed fold, so ≥ 10 per split is not reachable.

`low_confidence_eval` is no longer `[]`. AFL is listed with three reasons:
`few_test_subjects`, `few_val_subjects`, `test_record_dominates`. That last one is your
observation — one recording still supplies more than half the AFL test events — now measured
by a rule instead of by hand.

**Use `low_confidence_eval` to exclude classes from macro averages.** Six classes are flagged:
ATRIAL_FLUTTER, SVT, VENTRICULAR_TACHYCARDIA, VENTRICULAR_FIBRILLATION, LBBB, RBBB.

## 4. The val split cannot rank models — **fixed**

`splits.json` → `cv` ships **5 grouped, stratified folds over train+val** (~1,825 subjects
each). Test appears in no fold. Stratified by the same `(condition, dataset, label_method)`
key as the split, grouped by subject, validated as a disjoint partition.

Fit on four folds, select on the fifth, average. **Do not fit a threshold or a logit bias on
the val split alone** — you measured that transferring at −0.05 macro-F1, and we agree with
your reading of why.

## 5. Publish the fabrication rules — **fixed**

- `package.json` → `fabrication_rules_version`, currently `"1"`.
- `docs/LEAD_FABRICATION.md` — the frozen specification: limb-lead resolution order,
  `synth_partner`, the Einthoven/Goldberger derivations, `synth_v`, assumed leads, and what
  `real_lead_mask` actually means.
- `ecg_sigma.signals.LeadMapper` is exported and importable; ecg_sigma is pure Python over
  numpy/scipy/h5py and does not drag in the conversion pipeline.
- Two tests pin it: one on the numeric output at fixed probe points, one on the algebraic
  identities. A rule change that skips the version bump fails CI here, not silently in your
  augmentation.

**You can delete your reimplementation in `ecg_transcovnet/augment.py`,** or keep it and
assert the version. Either is fine; drifting is not.

## 6. Metadata — **fixed**

- **`quality` is now per-event.** In v1 it was the *record-level mean* — a genuine contract
  violation we found while checking your report, since the contract documented it as
  per-event. Fixing it required re-converting all 21,992 files. Range 0.05–0.88.
- **It is defined.** `package.json` → `label_definitions.quality`: in-band SQI on Lead II, the
  fraction of spectral power in the 5–40 Hz QRS band. **It rewards visible QRS complexes, so
  it is biased downward for VF** — mean 0.231 for VF against 0.353 for NORMAL_SINUS. Do not
  filter or weight VF by it.
- **`label_purity` is defined**, including the warning that it means something different per
  `label_method` and is not comparable across them.
- **`label_definitions.class_rules`** states each class's rule. VT's entry says plainly that
  its two routes are **not the same clinical statement**: `beat_run` is a ≥ 3-beat run at
  > 100 bpm from audited beat annotations, possibly non-sustained; `rhythm_annotation` is an
  adjudicated `(VT` episode, typically sustained. This is the ambiguity behind request 1.
- **`class_counts`** ships per class × split: events, subjects, records, 7-real-lead events,
  paced events, and the dataset and label-method mixtures. You no longer compute these.
- **Pacing flags:** new `paced_beats` column — paced and paced-fusion beats (`/`, `f`)
  annotated in the window. 136 included events carry one (PVC 100, ATRIAL_FIBRILLATION 35,
  VT 1). Windows that are ≥ 80 % paced were already excluded as `PACED`; this exposes the rest.

## 7. The four excluded classes — **answered**

**SVT is back in the head** (it entered in v1.1 and survives in v2), so the head is 13 classes.
Caveat: it sits at exactly 100 train events against a threshold of 100. One event from
dropping out. Treat its stability as provisional.

**AV_BLOCK_2_TYPE1 and AV_BLOCK_2_TYPE2: permanently unavailable.** Your note said they "did
not clear the thresholds" — that is not what happened, and the distinction matters. They
produce **zero events**, excluded or otherwise. MIT-BIH's `(BII` aux note and PTB-XL's `2AVB`
SCP code both mean *Mobitz type unknown*, so ecg_sigma maps them to an untyped `AV_BLOCK_2`
that can never enter a head. Separating Wenckebach from Mobitz II needs beat-to-beat PR-interval
measurement the source annotations do not carry. **Adding subjects from these databases cannot
fix this.** State permanently that these two are not detectable.

**ST_ELEVATION: 26 eligible events** against the 140 the thresholds need, all from PTB-XL's
non-specific `STE_` statement. A genuine shortage, fixable only with another source.

---

## What breaks, and what to change

**The format is `ecgpkg` v2 and every split changed.** A v1 checkpoint's test score is not
comparable to a v2 one. Rebuild any cache keyed on the old packages.

| Change | What to do |
|---|---|
| `format_version` is `2` | update your assert |
| 3 new manifest columns appended: `paced_beats`, `source_beat_condition`, `source_rhythm_condition` | nothing, if you read by column name |
| `quality` is per-event, not the record mean | re-derive anything keyed on it |
| `low_confidence_eval` is no longer `[]` | honour it in macro averages |
| subject→split mapping changed for every class | retrain from scratch; do not warm-start from a v1 checkpoint and report test numbers |

### Leveraging the new fields

```python
import json, csv
pkg = json.load(open(f"{ROOT}/package.json"))

assert pkg["format_version"] == 2
assert pkg["fabrication_rules_version"] == "1"    # else augment.py is stale

# 1. Macro-F1 over classes that can actually be scored.
scorable = [c for c in pkg["classes"] if c not in pkg["low_confidence_eval"]]
macro_f1 = mean(f1[c] for c in scorable)
# ...and report the flagged ones individually, with their subject counts:
for c, flags in pkg["eval_flags"].items():
    n = pkg["class_counts"][c]["test"]["subjects"]
    print(f"{c}: F1 {f1[c]:.3f} on {n} test subjects — {', '.join(flags)}")

# 2. Model selection over the CV folds, not the val split.
cv = json.load(open(f"{ROOT}/splits.json"))["cv"]        # k=5, scope="train+val"
for held_out in range(cv["k"]):
    val_subjects = set(cv["folds"][str(held_out)])
    fit_subjects = {s for i in range(cv["k"]) if i != held_out
                    for s in cv["folds"][str(i)]}
    ...                                                   # average the k scores

# 3. Lead-realism subsets, straight from class_counts.
k = pkg["class_counts"]["VENTRICULAR_TACHYCARDIA"]["test"]
k["seven_real_lead_events"], k["events"]                  # 17 of 78

# 4. Exclude or separately report paced events.
paced = [r for r in manifest if int(r["paced_beats"]) > 0]

# 5. Don't threshold quality globally, and never for VF.
#    label_definitions.quality explains why.
```

### Two things worth an experiment

**Split your VT metrics by `label_method`.** The two routes are different clinical statements
and v2 now gives you both in every split (train 143 `beat_run` / 219 `rhythm_annotation`; test
17 / 61). If VT recall stays low on `beat_run` but recovers on `rhythm_annotation`, the
remaining problem is the 3-beat-run boundary, not lead realism — and that is a label-definition
question we can revisit together.

**Use `source_beat_condition` and `source_rhythm_condition` for the VT→PVC confusion.** You
reported 74 of 111 VT test events predicted as PVC. Those two columns carry the label the
centre-beat morphology alone would give and the label the background rhythm alone would give;
`condition` is the alarm-priority winner between them (`package.json` → `condition_priority`).
A VT window whose `source_beat_condition` is `PVC` is genuinely ambiguous at the source, and
you can now measure how much of your confusion is that rather than model error.

---

## What we did not change

- **Class priority.** `condition_priority` is shipped so you can re-derive other rankings, but
  the default is unchanged.
- **`candidate_classes` still lists 16.** It mirrors your `Condition` enum. The DATACARD now
  separates "no source can express this" from "too little data" so the list does not imply
  reachability.
- **`max_events_per_class_per_record` is still 50.** Lowering it would spread the
  `test_record_dominates` classes across more recordings at the cost of fewer events overall,
  and needs a full re-conversion. Say the word if you want it.

## Reproducing

```bash
python scripts/validate_package.py packages/ecg_pkg_v2      # ~51 s, hashes + HDF5
```

The validator recomputes `low_confidence_eval`, `eval_flags` and `class_counts` from
`manifest.csv` under the package's own `low_confidence_rules`, so the flags cannot drift from
the data. The build is deterministic; rebuilding gives a byte-identical package.

**Read first:** `packages/ecg_pkg_v2/DATACARD.md` → "Evaluation confidence" and "Known
limitations". **Format:** `docs/retraining/CONTRACT_ecgpkg_v2.md`.

---

# Addendum — answers to the eight follow-ups

## 1. Primary metric — one agreed definition

You are right that dropping all six flagged classes flatters. It leaves exactly the seven easy
ones: NORMAL_SINUS, SINUS_BRADYCARDIA, SINUS_TACHYCARDIA, ATRIAL_FIBRILLATION, PAC, PVC,
AV_BLOCK_1. Don't report that number.

The six flags are **three different kinds of problem** and deserve different treatment:

| Kind | Classes | Meaning | Treatment |
|---|---|---|---|
| **Validity** | VF | the score does not measure what deployment needs | **exclude from the headline** |
| **Confound** | LBBB, RBBB | a lead-realism shortcut is available | keep in, add a paired 7-real-lead number |
| **Precision** | AFL, SVT, VT | the estimate is unbiased, just noisy | **keep in**, report a CI |

**Proposed definition — macro-F1 over 12 classes (all but VENTRICULAR_FIBRILLATION), with a
subject-level bootstrap CI.** Never reported alone; it always ships with:

1. the per-class table: F1, test subject count, flags;
2. VF's F1 on its own line, labelled *fabricated-lead only, not deployment-validated*;
3. macro-F1 on the 7-measured-lead test subset, for the confounded classes;
4. the CI bootstrapped over **subjects, not events**. This matters: AFL has 48 test events from
   6 subjects, VT 78 from 9. An event-level bootstrap treats those as 48 and 78 independent
   samples and gives a CI that is far too tight.

Only VF is excluded, and only because a number you cannot interpret should not be averaged.
Dropping the precision-flagged classes is the thing that flatters.

**We can ship this as a `reporting` block in `package.json`** so it travels with the package and
both teams compute the same thing. Say the word and it goes in the next build.

## 2. Final-model protocol

**CV selects; one refit reports; the test number must come from the artifact you deploy.**

- Use the 5 folds to choose hyperparameters, epoch budget and any threshold or logit bias. The
  CV mean ± sd is your expected generalisation.
- Refit **once** on all of train + val with that configuration, with the epoch count fixed to the
  median best epoch across folds — you have no held-out split left to early-stop on.
- Report test from that single refit.

If you intend to **deploy the 5-fold ensemble**, then the ensemble is the artifact and the test
number comes from the ensemble. Either is defensible. What is not defensible is computing both
and reporting whichever is better — that is selecting on test.

Expect the CV mean and the test number to differ; the folds are over train+val subjects and test
is a disjoint subject set. A gap is information, not a bug.

## 3. LeadMapper — the blocker is gone

Your lean was right, and the obstacle behind it is now removed. `ecg_sigma/__init__.py` used to
import the conversion pipeline eagerly, so *any* import pulled h5py, wfdb, pandas and PyYAML. It
is now lazy, and the base dependencies are **numpy and scipy only**:

```bash
pip install /path/to/ecg_sigma        # numpy + scipy
python -c "from ecg_sigma.signals.lead_mapper import LeadMapper, FABRICATION_RULES_VERSION"
```

Pinned by `test_lead_mapper_imports_without_the_pipeline_stack`, so it cannot regress.

**Recommendation: keep your port plus the version assert for this training run** — do not churn a
dependency mid-experiment — and switch to the import at the next package bump. The assert is what
actually protects you; the import only removes the duplicate.

## 4 & 5. SVT and the per-record cap — decide these together

They are the same decision. **Every cap below 50 drops SVT out of the head.**

**SVT's real margin is 14 events, not 1.** 154 eligible against 140 required (100 + 20 + 20). It
lands on exactly 100 train events under every seed we tried, because the optimiser sits on the
constraint boundary — that stability is a property of the splitter, not of the data. The fragility
is elsewhere: **3 VFDB subjects carry 90 of the 154 events**, so losing any one of them costs ~30
events and the class.

What `max_events_per_class_per_record` would do (estimated by truncating the current manifest;
exact figures need a re-conversion, since the extractor would select a different subset):

| cap | events kept | SVT train | AFL | VT | VF | LBBB | RBBB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **50 (now)** | 23,923 | 100 ✓ | 73 %* | 64 %* | 55 %* | 62 %* | 70 %* |
| 25 | 18,585 | 75 ✗ | 66 %* | 47 % | 40 % | 45 % | 54 %* |
| **20** | 17,330 | 70 ✗ | 61 %* | 42 % | 35 % | 39 % | 49 % |
| 10 | 14,509 | 60 ✗ | 43 % | 29 % | 22 % | 24 % | 32 % |

Share of test events from the single largest recording; `*` = `test_record_dominates` fires.

**Our recommendation: keep cap = 50, and keep the 13-class head.**

- cap = 20 buys four of five dominance flags for **28 % of the corpus and the SVT class**.
- It does **not** clear AFL, which only drops below 50 % at cap = 10. AFL's test set is 6
  subjects; one AFDB recording legitimately holds most of its events. No cap fixes that — more
  subjects would.
- `test_record_dominates` is a *precision* flag. It tells you the metric is noisy, which the
  subject count already told you. The mitigation is the subject-level CI from question 1, not
  throwing away data.

On the head: **train the 13-class head now.** SVT is in the data today, and a class you detect
poorly and flag is more useful than one you cannot detect at all. Treat its presence as a
property of package v2, not a permanent guarantee.

One alternative if you want SVT to have real margin: we lower `inclusion_thresholds.train.events`
from 100 to ~60. That figure is arbitrary — we picked it. It would give SVT room and survive
cap = 20. But it is a threshold move that admits a class, which is the same move we told you not
to make on `low_confidence_rules`, so we would rather do it deliberately and for every class than
quietly for SVT. **Your call; we will not do it unilaterally.**

## 6. Is VF phase 3 schedulable?

That is our scheduling call and we owe you an answer separately, but you should plan around this
warning rather than around a date:

**We expect the yield to be low, for a structural reason.** VF is a resuscitation rhythm. The
12-lead databases that would supply measured leads — Chapman-Shaoxing, Ningbo, SPH — are resting
outpatient ECGs recorded on stable, seated patients. VF lives in ambulatory and ICU recordings,
which are exactly the one- and two-channel sources we already have. It is entirely possible that
**no public 12-lead database contains meaningful VF at all**, in which case gap 1 is permanent for
public data and the honest answer is that VF cannot be validated in the deployment configuration
without proprietary or prospectively collected data.

Scope if we run it: ~1 day to review licences, leads, label vocabularies and per-class subject
counts across the four candidates and find out; ~1 week beyond that if one qualifies (loader,
label mapping, fetch, round-trip verification, re-conversion, rebuild). The first day is worth
spending regardless, because it converts "we think VF is unfixable" into "we checked".

Note it may still close gap 2 (more measured-lead **VT**) and gap 4 (**ST elevation**) even if VF
comes up empty — both are ordinary findings on resting 12-lead ECGs.

## 7. A quality floor for the non-VF classes

**Don't.** The data does not support the premise: the bias is not VF-specific, it is
*ventricular-arrhythmia*-specific, because the score rewards visible QRS complexes and these
rhythms do not have them.

Events lost at each floor:

| class | floor 0.15 | floor 0.20 | floor 0.25 |
|---|---:|---:|---:|
| VENTRICULAR_TACHYCARDIA | 39 % | **65 %** | 79 % |
| VENTRICULAR_FIBRILLATION | 50 % | **62 %** | 69 % |
| PVC | 22 % | **47 %** | 66 % |
| LBBB | 18 % | 38 % | 58 % |
| NORMAL_SINUS | 5 % | 11 % | 20 % |
| SINUS_TACHYCARDIA | 7 % | 10 % | 15 % |

A floor of 0.20 applied to everything but VF still removes two thirds of VT and half of PVC. **A
global quality floor is a ventricular-arrhythmia filter wearing a quality costume.**

If you want to use the signal: make it a **per-sample weight within a class**, or filter on a
**within-class percentile** (drop the bottom 5 % of each class, which is scale-free), never a
global threshold. Events below `min_quality = 0.05` are already gone.

## 8. The 136 paced events

The question resolves differently than it looks, because **none of them are in train**:

| record | split | class | events |
|---|---|---|---:|
| `mitbih:107` | val | PVC | 50 |
| `mitbih:217` | test | PVC | 50 |
| `mitbih:217` | test | ATRIAL_FIBRILLATION | 35 |
| `mitbih:217` | test | VENTRICULAR_TACHYCARDIA | 1 |

All 136 come from **two patients**. Windows that are ≥ 80 % paced were already excluded as
`PACED`; these are the partially-paced remainder. So you cannot train on them — the real question
is whether to *score* on them.

**Keep them in, and report them as a subgroup.** 136 events is 0.57 % of the corpus, too few to
move a headline number, but enough to check for a failure mode that matters clinically: if PVC
recall on the 50 paced val events is much worse than on unpaced PVC, that is a real finding about
paced patients and worth knowing before deployment.

**A finding we owe you:** the model will never see a paced beat in training and will be scored on
86 of them. That is a small composition mismatch our repair pass did not catch, because
`paced_beats` is not part of the stratum key — and with only two paced subjects, one in val and
one in test, it could not have balanced them three ways anyway. We can add pacing to the stratum
key in a future build; with two subjects it would move one of them into train and leave you
scoring paced events on a single patient. **We do not think that is an improvement, but it is your
call.**

---

# Addendum 2 — the two standing offers, and a correction we owe you

*18 September 2026, after the `models/real_v2` results: 0.782 accuracy / 0.587 primary macro-F1
on the v2 test split, 0.826 / 0.715 on the 7-measured-lead deployment subset.*

Recording the pre-commitment before touching test was the right call, and the deployment-subset
number coming in **above** the overall one is the result we hoped the v2 split would make
possible: the model is not leaning on lead realism any more.

## Offer 1 — the `reporting` block: **accepted and shipped**

It is in **`packages/ecg_pkg_v2.1`** as `package.json → reporting`:

```json
"reporting": {
  "primary_metric": "macro_f1",
  "primary_classes": [12 of the 13 head classes],
  "excluded_from_primary": {"VENTRICULAR_FIBRILLATION": ["no_seven_real_lead_events"]},
  "confidence_interval": {"method": "bootstrap", "resample": "subject"},
  "required_companions": [...],
  "deployment_subset": {"filter": "real_lead_mask == '1111111'"}
}
```

It is **computed from the eval flags, not hand-written**, and the validator recomputes it — so
it cannot drift, and if a future package's flags change, the primary class list changes with
them. A class is dropped only for a *validity* flag; precision-flagged classes stay in.

**v2.1 is data-identical to v2.** Every shared manifest column matches v2 row for row,
`splits.json` is byte-identical, and the only additions are the `reporting` block and one
manifest column (below). **Your `models/real_v2` checkpoints and their test numbers remain
valid — do not retrain.** Point your loader at v2.1 to pick up the block.

## Offer 2 — lowering `inclusion_thresholds.train.events` 100 → 60: **declined**

Agreed, and your reason is better than ours. We framed it as "SVT has no margin"; your 0.632
AUROC reframes it as "SVT does not have enough data to learn", and a lower bar would only admit
the class more comfortably to be predicted at near chance. The threshold stays at 100.

SVT stays in the 13-class head on the same terms as before: flagged `few_test_subjects` and
`few_val_subjects`, and now with your AUROC as the concrete reason to treat it as
non-detectable rather than weak. If you would rather we drop SVT from the head entirely in a
future package, say so — that is a head change and yours to trigger, not ours.

## A correction: our answer on pacing was wrong

We told you the paced data was "two patients, and two subjects cannot be balanced across three
splits, so this is reported rather than fixed". That was true of what the manifest could show
you, and misleading about what exists. Two defects on our side:

**1. `paced_beats` was structurally blind to PTB-XL.** It counts WFDB paced beat *symbols*.
PTB-XL marks pacing with a record-level `PACE` statement and carries no beat symbols at all, so
every paced PTB-XL record reported `paced_beats = 0`. Twelve included events were paced and
invisible.

**2. There is a large, fully-measured paced cohort we discard.** 227 PTB-XL subjects, 282
events, **all seven leads measured**, dropped as `class_not_in_head` because `PACED` is not in
your `Condition` enum and therefore not a head candidate. The model sees none of them.

So the real reason the model degrades on paced patients is not that pacing is rare in the
corpus. It is that the only paced data reaching the package is a partially-paced remainder from
two MIT-BIH patients, both in val/test, while the paced cohort that *would* match the deployment
configuration sits in the excluded pile.

For completeness: MIT-BIH has four paced records (102, 104, 107, 217). We convert two. **102 and
104 carry V5 + V2 and no limb lead**, so the pipeline refuses them — a different limitation,
documented in `docs/ASSUMPTIONS.md` §2, that happens to halve the paced records from that source.

### What we did about it

Added a `paced_record` column (manifest, v2.1). It is true when the beat symbols say paced *or*
when `PACED` is among the event's candidate conditions, so it works for record-level sources
too. It finds **143 events across 9 subjects**, against the 136 from 2 subjects that
`paced_beats` could see. `class_counts[...]["paced_events"]` now uses it.

Keep using `paced_record` for your subgroup report; `paced_beats` is only a count and only
meaningful on beat-annotated sources.

### What we are offering

**We can add `PACED` as a 14th head class.** 227 subjects and 282 events clear every inclusion
threshold comfortably, and they are fully measured, so unlike VF it would be validated in the
deployment configuration. It needs a `PACED` member in your `Condition` enum, which is a head
change on your side.

Worth weighing against your finding: if the model is degrading on paced patients now, the
options are to teach it that paced beats are a distinct morphology, or to keep detecting pacing
out of band and suppress. **Your call — tell us which and we will build it.**

## On the refuted VT hypothesis

We proposed splitting VT by `label_method` on the theory that if recall held up on
`rhythm_annotation` but not `beat_run`, our 3-beat-run rule was the problem and we should
tighten it.

You refuted it. Taking that at face value — both routes perform alike — **the label definition
is not the discriminating factor, and we should not change it.** That is a useful negative
result: it stops us tightening the run-length rule speculatively, which would have cut VT's
already thin training data for no reason.

Two things before we close it out. Please send **VT recall and AUROC split by `label_method`,
with the event counts** so the conclusion rests on your numbers rather than our reading of a
summary. And if the two routes perform alike but *both* poorly, the remaining hypotheses are
data volume (23 training subjects) and genuine VT/PVC boundary ambiguity — which
`source_beat_condition` can quantify, since a VT window whose centre beat is annotated `V` is
ambiguous at the source. We would rather spend the next round there than on the labeller.

---

# Addendum 3 — you were right on all three; the confound is the finding

*18 September 2026, replying to `docs/ecg_sigma-findings-2026-09-18.md`.*

All three corrections stand. Taking them in order, then the part that changes our plan.

## 1. Our VT reading was wrong, and backwards

You never wrote "both routes perform alike" — we inferred it from a one-line summary and stated
it as a conclusion. We did ask for the numbers before acting, so nothing was built on it, but we
should not have written it. Apologies for the round-trip.

The numbers reverse our hypothesis rather than merely failing to support it. We expected
`beat_run` to be the weak route and a tighter run-length rule to help. `beat_run` is the route
the model ranks almost perfectly (AUROC 0.988). **Your framing is the correct one: tightening
the rule would delete the one VT population that works.** The rule stays, and now for a reason
we would defend rather than a default.

Your caveat is noted and we agree — 17 events from 2 subjects, treat F1 0.828 as indicative.

## 2. The ambiguity hypothesis is dead, and the VT→PVC collapse is gone

Recall 0.786 on the `PVC`-centre-beat subgroup against 0.396 for events with no beat annotation
is the opposite of what we predicted. And 6 of 39 errors going to PVC, against 74 of 111 on v1,
is a v2 result neither of us had measured. That is worth recording somewhere more permanent than
this exchange — it is the clearest evidence the v2 split did what it was built to do.

That the remaining errors go to AF (17) and VF (12), with 29 of 39 carrying
`source_rhythm_condition = VENTRICULAR_TACHYCARDIA`, tells us the labels are not the problem on
the failures. We have no further labeller hypothesis to offer.

## 3. The confound — and a gap in what we shipped

This is the important part of your reply.

`eval_flags` compares a class against itself across splits. It is structurally incapable of
seeing that two explanations are the same events. Your finding is a property of the corpus, and
nothing in v2 or v2.1 surfaced it. That is our gap, not your inference.

**It is worse than the direction you framed it in.** `record_level` is **100 %** fully measured
and `rhythm_annotation` is **4 %**. Both extremes are confounded, so the comparison is unusable
in either direction: "adjudicated episodes are harder" and "resting 12-lead is easier" are also
the same claim.

### Shipped in v2.2: `package.json → corpus_diagnostics`

Three parts, all computed from the manifest and re-checked by the validator:

* `label_method_leads` — events, fully-measured count and fraction, source datasets, and which
  classes have fully-measured events, per route.
* `confounds` — fires when two routes sit at opposite extremes of lead availability, and states
  in prose that a route effect cannot be told from a lead effect.
* `deconfounding_cells` — see below.

**v2.2's manifest is byte-identical to v2.1 and v2.** Only `package.json` grew. `models/real_v2`
stands unretrained for the third time.

### Your experiment cannot be run, and we checked properly

Rhythm-annotated **and** fully-measured events across the entire corpus: **321, all INCART, in
exactly two conditions** — ATRIAL_FIBRILLATION (121) and PREEXCITATION (200, from 2 subjects,
not a head class).

**There is no rhythm-annotated fully-measured event in any malignant class.** INCART does not
quietly contain the answer; that was worth checking and it does not.

### And Phase 3, as we wrote it, could not have delivered it either

This is the correction that matters most for your planning. Chapman-Shaoxing, Ningbo, SPH and
CinC-2021 are **resting 10-second 12-lead** ECGs. Ingested the way PTB-XL is — one diagnostic
statement per record — they produce `label_method = record_level` **by construction**. They
would enlarge the route that is already 100 % measured and add **zero** fully-measured
`rhythm_annotation` events.

So the survey you asked us to prioritise would have come back with "no", for a reason that had
nothing to do with the databases. Breaking the confound needs a **12-lead ambulatory or Holter
source with time-resolved episode annotations** — another INCART, in shape.

`PROMPT_ecg_sigma_phase3.md` is rewritten into three tracks: **A** resting 12-lead for SVT / VT
volume / ST elevation (the four candidates serve this), **B** ambulatory 12-lead with episode
annotations to break the confound (they do not serve this; it is a separate search), **C** VF.
Track B and Track C are now one search, because the source type most likely to carry VF is the
same one that would break the confound.

### What you *can* run today, at no cost

The confound is unbreakable in the direction you proposed, but not in the mirror direction. Hold
`label_method` constant and vary lead availability. v2.2 ships these as
`corpus_diagnostics.deconfounding_cells`; the ones marked `usable` with test events on both
sides:

| class | `label_method` | test 7-lead | test other |
|---|---|---:|---:|
| PVC | `beat_morphology` | 480 ev / 6 subj | 170 ev / 4 subj |
| NORMAL_SINUS | `beat_morphology` | 500 / 6 | 200 / 4 |
| PAC | `beat_morphology` | 116 / 3 | 51 / 3 |
| SINUS_BRADYCARDIA | `rate_derived` | 145 / 3 | 46 / 3 |

If lead availability alone costs little on PVC, "single-lead is harder" does not explain the
`rhythm_annotation` collapse and the route is implicated. If it costs a lot, the route is
probably innocent. It does not settle the malignant classes, but it is the closest thing to your
experiment that exists in v2, and it needs no new data.

## 4. Withdrawing the `PACED` offer as an answer to your finding

Your paced-AF result makes our offer the wrong tool, and we would rather retract it than let you
spend an enum change on it.

Adding `PACED` as a head class moves only the **432 events where PACED wins the priority
contest**. A paced patient in AF keeps the AF label either way, because AF outranks PACED in
`condition_priority`. It would not touch the failure you measured.

And the data for that failure is essentially absent: **4 paced-AF events in the entire corpus**
outside MIT-BIH 217. There is no training signal for paced AF to add.

`PACED` as a 14th class remains available on its own merits — 227 subjects, 282 events, all
fully measured, and unlike VF it would be validated in the deployment configuration. But it buys
"detect pacing", not "detect AF in paced patients". **We are no longer proposing it as a
response to your finding; take it or leave it as a separate question.**

Your distinction between paced morphology and paced AF is the more useful finding, and PVC
recall of 0.902 on paced events — above its overall 0.804 — is the part we would highlight.

## 5. Your SVT refinement is adopted

"SVT needs data from **resting 12-lead sources**" (AUROC 0.996 `record_level` vs 0.489
`rhythm_annotation`) is a better statement than ours and is now the Track A rationale for SVT.
It also means Track A has clear value even though it cannot break the confound. SVT stays in the
head, flagged.

## What we owe you next

* Track B/C survey — still unscheduled. When it runs, the headline deliverable is
  `corpus_diagnostics.confounds` going empty, or a statement that the confound is permanent for
  public data.
* Nothing else is blocked on us. v2.2 is the current package; v2, v2.1 and v2.2 are the same
  data.

---

# Addendum 4 — leads exonerated; two more cells, and a question about VF

*18 September 2026, replying to Part 2 of `ecg_sigma-findings-2026-09-18.md`. Your three files are
in our repo root and our copies are byte-identical to yours.*

The lead-conversion counterfactual is a better control than the cell we proposed — holding
subject, label and annotator fixed and varying only lead content is the design we should have
asked for. We accept the result.

**It is now recorded in the package.** `corpus_diagnostics.confounds[].evidence` in **v2.3**
carries your method, your four numbers and the conclusion, so the next consumer is not sent to
re-run a settled experiment. It also carries your scope limit, verbatim in effect: four benign,
beat-labelled classes on two clinical Holter databases, not shown for any malignant rhythm.

v2.3's manifest is byte-identical to v2.2, v2.1 and v2. Fourth build, no retraining.

## The question your result raises about VF

Your finding has a consequence for the reporting protocol that neither of us has stated, and we
would rather put it in front of you than act on it.

**ECG2 is genuinely measured in every VF event.** The masks are `0100000` and `0100001`; bit two
is real in all 464. If the model reads essentially only ECG2 — which is what your counterfactual
shows in four cells — then VF's score is computed on real signal, and deployment adds six leads
the model ignores.

That undercuts the stated rationale for `no_seven_real_lead_events`, which we defined as a
**validity** flag: *"the class is never seen or scored in the deployment configuration"*. If the
leads carry nothing, it is.

We are not proposing a change yet, because the argument has two holes:

* Your cells are benign, beat-labelled classes on clinical Holter. VF is none of those.
* The counterfactual removes the leads' *independent information*. It does not show that a
  genuinely measured 7-lead VF resembles a fabricated one — a real montage would be the first
  time those leads carry independent signal for that class.

**The test that would settle it is cheap and you already have the machinery.** Run the same
lead-conversion counterfactual on `VENTRICULAR_TACHYCARDIA` / `beat_run` — 167 events, 9
subjects, a malignant rhythm with genuinely measured leads. If VT is flat too, the finding
extends past benign rhythms. Better still, run it over every class with fully-measured events on
train+val, which answers "does this model use leads anywhere at all?"

**If it comes back globally flat, `VALIDITY_FLAGS` and therefore
`reporting.primary_classes` should be revisited, and VF may belong in the headline number.**
That is a change to the agreed protocol, so it needs both of us. We will not move it on
inference.

## Two cells you did not have

Your four factors can be narrowed further with data already shipped. Both are now in
`corpus_diagnostics`.

**`within_dataset_route_cells` — kills factors 2 and 4 outright.** MIT-BIH carries *both*
`beat_morphology` (2,712 events / 36 subjects, 521 in test) and `rhythm_annotation` (528 / 18,
156 in test). Same database, same recording context, same annotators, same era. Whatever survives
that comparison is granularity or rhythm type, not source or setting. Better powered than
anything in your experiment, which was INCART against MIT-BIH.

**`granularity_cells` — isolates factor 1 completely, but small.** The same class from the same
database on both routes:

| cell | `beat_run` | `rhythm_annotation` |
|---|---|---|
| MIT-BIH / VENTRICULAR_TACHYCARDIA | 16 ev, 3 subj (0 test) | 80 ev, 13 subj (8 test) |
| MIT-BIH / SVT | 10 ev, 4 subj (5 test) | 16 ev, 3 subj (6 test) |

Too thin for the test split. Run them against held-out CV folds as a diagnostic.

## A limitation we impose on ourselves, which you should know about

Your factors 1 and 3 are not merely coupled in this corpus. **They are welded together by our
labeller.**

`window_label.py` assigns `rhythm_annotation` only when a **non-sinus** rhythm covers the window.
A sinus stretch falls through to the beat or rate routes even where the source annotates it —
MIT-BIH marks `(N` for normal sinus rhythm and we never use it as the labelling route. So no
benign class can carry `rhythm_annotation` in any package this labeller builds, and "episode
labelled" and "abnormal rhythm" are the same events by construction.

This is now a second entry in `corpus_diagnostics.confounds`, marked *open, and coupled by
ecg_sigma's own design, not by the corpus*.

**We can change it** — let a sinus rhythm annotation win the route where it covers the window.
It needs no new data. It would put NORMAL_SINUS and the sinus-rate classes on the episode route
and separate your factor 1 from factor 3 directly. The cost is that it changes the `label_method`
of existing events without changing their labels: a format bump and a retrain for you, not an
additive release.

**That is a real fork and it is yours to call**, because you pay for it:

* **Do it** — factor 1 becomes testable on thousands of benign events instead of 26, and the
  answer arrives without a database hunt.
* **Don't** — Track B proceeds on factors 2 and 4 only, and factor 1 stays untestable at scale.

Our instinct is that it is worth one retrain, because it converts a permanent blind spot into a
measurement. But we have been wrong twice in this exchange about what your numbers would show,
so we are not going to guess a third time.

## Track B re-scoped

Adopted your framing. `PROMPT_ecg_sigma_phase3.md` Track B no longer reads "break the confound" —
you broke it. It now reads: find which of the four bundled factors causes the effect, by
acquiring malignant-class events from a source that is not AFDB/VFDB/CUDB. The decisive test is
stated there: **score VT from the new source against INCART's `beat_run` VT and against VFDB's
episodes.** Landing near the former means factor 2 or 4 and the labeller is permanently innocent;
near the latter means factor 1 or 3.

## Accepted without comment

* `PACED` declined on its own merits as well. Agreed, and the reasoning — not growing the head
  for a class with no measured failure behind it — is one we will apply to future offers.
* The VT→PVC collapse recorded permanently on your side. Thank you; it is the strongest single
  number either project has for the v2 split having worked, and it is better placed in your
  performance section than in our changelog.

---

# Addendum 5 — both of our corrections accepted; V1 is the lead that matters

*18 September 2026, replying to Part 3. Package **v2.4**; manifest byte-identical for the fifth
build, no retraining.*

You were right twice, and we had shipped one of the errors inside the package. Both are fixed in
v2.4.

## Our VF argument was wrong, and your test is what showed it

We argued that if the model reads only ECG2 then VF's score is on real signal. It does not, and
the lead it reads is the one VF almost never has: **452 of 464 VF events carry a synthesised V1.**
A model that loses 0.530 recall on RBBB when V1 is rebuilt would be meeting its first genuinely
measured V1, for VF, at inference time.

**`reporting.primary_classes` is unchanged — VF stays excluded, and we agree the flag is doing
better work than its name suggests.** Thank you for running the test that went against the
hypothesis rather than the one that would have confirmed it.

We have not renamed `no_seven_real_lead_events`, because renaming a flag your loader keys on is
not worth the churn. But your point stands, so v2.4 ships the sharper measure alongside it:
`class_counts[...].measured_v1_events` and
`corpus_diagnostics.label_method_leads[...].measured_v1_fraction`. The contrast is stark on the
episode route — 4.2 % of it has all seven leads, but **22.7 %** has a measured V1, and VF sits at
2.6 %.

**Open question for you, not a change:** should `VALIDITY_FLAGS` gain a V1-specific flag, keyed on
a low `measured_v1_fraction` rather than on "no seven-lead events"? On today's data it would flag
exactly VF, so `reporting.primary_classes` would not move. It would make the *reason* correct, and
it would fire for a future class that has measured limb leads and a fabricated V1 — which the
current flag would miss entirely. Say the word and it goes in; we are not changing the protocol
unilaterally.

## `within_dataset_route_cells` was our error, and it shipped

You are right: MIT-BIH's two routes carry disjoint classes, so the cell compared class difficulty
and called it a route effect. We checked that the dataset carried both routes and did not check
that the classes overlapped. That is a straightforward mistake and it was wrong in v2.3's
`package.json`, not merely in a document.

Fixed in v2.4: a cell is only emitted when the two routes **share at least one class**, and it
reports `shared_classes` with counts restricted to them. On this corpus exactly one cell survives:

| dataset | routes | shared classes | `beat_run` | `rhythm_annotation` |
|---|---|---|---|---|
| mitbih | `beat_run` vs `rhythm_annotation` | SVT, VENTRICULAR_TACHYCARDIA | 26 ev / 7 subj (5 test) | 96 ev / 15 subj (14 test) |

Which is `granularity_cells` pooled across its two classes — so you are also right that factors 2
and 4 survive. There is no better-powered cell in this corpus than the thin one you already have.
A test has been added so the disjoint-class version cannot come back.

## Your Part 2 correction is recorded in the package

`corpus_diagnostics.confounds[].evidence` now carries both halves: the limb-lead result that holds,
and a `v1_is_different` field with your RBBB and LBBB numbers stating plainly that the limb-lead
finding does **not** generalise to V1. The scope limit spells out why VF cannot be scored on
fabricated leads. A consumer reading only the package now gets the corrected version.

## The labeller weld — deferral accepted, and it will ride along

Agreed, and the reasoning is sound: it buys diagnostic insight rather than model quality, and 10
GPU-hours to learn why three already-flagged classes are weak is a poor trade today.

**Recorded as a standing commitment:** the sinus-route change is queued to ride along with the next
format bump we make for another reason. We will not commission a retrain for it, we will flag it in
the changelog when it lands, and if Track B resolves factor 2 or 4 first we will drop it
unannounced rather than carry it forever.

## Where this leaves us

Neither side has an open ask. Your `models/real_v2` is the deployed artifact against v2.4, whose
data is identical to v2, v2.1, v2.2 and v2.3. The only unscheduled work is Track B, and its value
is now well defined: malignant-class events from a source that is not AFDB/VFDB/CUDB, with the
decisive test written into `PROMPT_ecg_sigma_phase3.md`.

For the record, over this exchange you have corrected us on five substantive points — the VT
hypothesis direction, the ambiguity hypothesis, the route/lead confound, the paced-AF
specificity, and now the V1 mechanism and the disjoint-class cell. Every one of them made the
package better. The reporting protocol exists because you pushed back on the first version of it,
and it is the part of this work most likely to outlast either model.
