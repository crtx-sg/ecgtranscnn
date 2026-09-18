# Findings for ecg_sigma — VT, pacing, and a confound worth more than either

Reply to Addendum 2 in `docs/RESPONSE_change_requests.md`. All numbers are the `models/real_v2`
5-fold ensemble on the v2 test split (3,513 events), centre-crop 2000 — the same single evaluation
we pre-committed to, re-grouped. v2.1 changes no signal, so nothing was re-scored.

## First, a correction to your reading of our VT result

You wrote: *"Taking that at face value — both routes perform alike — the label definition is not the
discriminating factor."* **The two routes do not perform alike.** They differ by a wide margin, in
the direction opposite to your hypothesis:

### Ask (a) — VT by `label_method`, with AUROC and counts

| `label_method` | Test events | Subjects | Recall | F1 | AUROC |
|---|---:|---:|---:|---:|---:|
| `beat_run` | 17 | 2 | **0.706** | **0.828** | **0.988** |
| `rhythm_annotation` | 61 | 7 | 0.393 | 0.425 | 0.805 |

Your hypothesis was that `beat_run` would be the weak route and that a tighter run-length rule would
help. The opposite holds: the audited ≥ 3-beat runs are the VT the model ranks almost perfectly
(AUROC 0.988), and the adjudicated sustained episodes are where it fails.

**Your decision not to touch the rule stands, but the reason is stronger than "no difference":**
tightening the run-length rule would remove the one VT population the model actually detects.

Caveat we want on the record: `beat_run` is 17 events from 2 subjects. Treat 0.828 as indicative;
the AUROC of 0.988 is the more robust half of that row.

## Ask (b) — `source_beat_condition` does not support the ambiguity hypothesis either

You proposed that VT windows whose centre beat is annotated `V` are ambiguous at source, and that
this drives the VT→PVC confusion. Measured on the 78 VT test events:

| `source_beat_condition` | VT events | Subjects | Predicted VT | Predicted PVC | Recall |
|---|---:|---:|---:|---:|---:|
| `PVC` | 14 | 5 | 11 | 2 | **0.786** |
| `NORMAL_SINUS` | 9 | 3 | 4 | 3 | 0.444 |
| *(empty — no beat annotation)* | 53 | 3 | 21 | 0 | 0.396 |
| `RBBB` | 2 | 1 | 0 | 1 | 0.000 |

**The events that are ambiguous at source are the ones the model gets right.** VT with a `PVC`
centre beat scores recall 0.786 — the best subgroup — while the events with no beat annotation at
all score 0.396.

And the VT→PVC framing itself no longer holds on v2. Where the 39 VT errors actually go:

| Predicted instead of VT | Count |
|---|---:|
| ATRIAL_FIBRILLATION | 17 |
| VENTRICULAR_FIBRILLATION | 12 |
| PVC | **6** |
| SVT | 4 |
| ATRIAL_FLUTTER | 3 |

On v1 we reported 74 of 111 VT events called PVC; on v2 that is 6 of 39. The confusion moved to AF
and VF. Also, 29 of the 39 errors carry `source_rhythm_condition = VENTRICULAR_TACHYCARDIA`, so the
source annotation agrees it is VT — the label is not ambiguous on the failures.

Both of your candidate explanations for VT therefore look wrong: it is not the labeller, and it is
not the PVC boundary.

## What the data does point at — and a confound you should know about

The same pattern appears in all three weak classes, not just VT:

| Class | `record_level` | `beat_run` | `rhythm_annotation` |
|---|---|---|---|
| VENTRICULAR_TACHYCARDIA | — | AUROC **0.988**, recall 0.706 | AUROC 0.805, recall 0.393 |
| ATRIAL_FLUTTER | AUROC **0.993**, recall 0.667 (3 ev) | — | AUROC 0.816, recall **0.022** (45 ev) |
| SVT | AUROC **0.996**, recall 0.500 (4 ev) | AUROC 0.871, recall 0.0 (5 ev) | AUROC **0.489**, recall 0.0 (16 ev) |

`rhythm_annotation` is the weak route every time, and for SVT it is at chance (0.489).

**But route and lead availability are collinear in this package, so this is not attributable yet.**
Across all splits:

| `label_method` | Events | 7 measured leads |
|---|---:|---:|
| `record_level` | 10,921 | **100 %** |
| `beat_run` | 211 | 88 % |
| `rate_derived` | 1,877 | 75 % |
| `beat_morphology` | 8,052 | 66 % |
| `rhythm_annotation` | 2,862 | **4 %** |

Only 121 of 2,862 `rhythm_annotation` events have seven measured leads, and **all 121 are
ATRIAL_FIBRILLATION from INCART**. So "the adjudicated-episode route is harder" and "single-lead
sources are harder" cannot be separated with v2 data — they are the same events.

This is what we would spend the next round on, rather than data volume or the PVC boundary:
**rhythm-annotated events with seven measured leads, in any class other than AF.** If those score
like `record_level`, the route is innocent and lead availability is the whole story. If they score
like the rest of `rhythm_annotation`, adjudicated episodes are genuinely a different problem. INCART
has rhythm annotations and full leads, so the question may be answerable without a new database.

## Pacing — `paced_record` confirms the finding and sharpens it

Switched our subgroup report to `paced_record` as instructed (`paced_beats` is now only a count).

| Column | Test events | Subjects | Accuracy | Unpaced accuracy |
|---|---:|---:|---:|---:|
| `paced_record` (v2.1) | 88 | 3 | 0.602 | 0.786 |
| `paced_beats` (v2) | 86 | 1 | 0.616 | 0.786 |

The extra column changes the subject count from 1 to 3, which matters — the finding is no longer one
patient. Per class on the paced events:

| True class | Paced events | Recall |
|---|---:|---:|
| PVC | 51 | **0.902** |
| ATRIAL_FIBRILLATION | 35 | **0.200** |
| SINUS_BRADYCARDIA | 1 | 0.0 |
| VENTRICULAR_TACHYCARDIA | 1 | 0.0 |

**The degradation is not a general failure on paced morphology — it is specific to paced AF.** PVC
recall on paced events (0.902) is *above* its overall 0.804. Paced AF collapses to 0.200 against
0.818 overall. That is a narrower and more actionable finding than "accuracy drops on paced
patients", and it is worth weighing when you decide what a `PACED` class would buy.

Thank you for the correction on `paced_beats` being blind to PTB-XL. It explains why our v1-era
analysis found nothing worth reporting on pacing.

## Your three items

1. **`reporting` block — in use.** We verified v2.1 independently before adopting it: all 37,642
   manifest rows match v2 on every shared column, `splits.json` is byte-identical, `paced_record` is
   the only addition. `models/real_v2` was not retrained. Your `reporting.primary_classes` matches
   our own flag-derived list exactly; our loader now honours the block and **raises** if the block
   and the flags ever disagree, rather than picking one.
2. **Threshold 100 → 60 — agreed, declined.** One refinement from the numbers above: SVT is not
   uniformly undetectable. It reaches AUROC 0.996 on the `record_level` route and 0.489 on
   `rhythm_annotation`. "SVT needs data" is better stated as **"SVT needs data from resting 12-lead
   sources"**. We are not asking you to drop it from the head; flagged is the right treatment.
3. **`PACED` as a 14th class — we owe you an answer and will send it separately.** The paced-AF
   result above is the relevant input: the model is not failing on paced morphology generally, it is
   failing on one class within it.

## Still open on your side

The VF 12-lead investigation. Given the structural argument in Addendum 1 we are not counting on it,
and our reports now state VF as not deployment-validated. If the one-day survey happens anyway, the
finding we would most value is whether any of the four databases carries **rhythm-annotated,
fully-measured** events of any malignant class — that would break the confound above, which is worth
more to us than VF alone.

---

# Part 2 — we ran the deconfounding cells, and the leads are exonerated

Reply to Addendum 3. v2.2 verified before adopting: `manifest.csv` and `splits.json` are
byte-identical to v2.1 and v2, `corpus_diagnostics` is the only addition to `package.json`.
`models/real_v2` unretrained for the third time. Your 321-event figure reproduces exactly — all
INCART, ATRIAL_FIBRILLATION 121 and PREEXCITATION 200, 3 subjects.

## Your decision rule is met, in the direction that implicates the route

We ran all four `usable` cells, and added a control you did not propose. Your cell holds
`label_method` constant and varies lead availability — but in this corpus lead availability **is
dataset identity**: the 7-lead side is always INCART and the other side always MIT-BIH. So the cell
measures dataset *and* leads together, and a difference could be annotator, patient mix or recording
era.

The **lead-conversion counterfactual** separates them: re-predict the *same* INCART events with
their non-ECG2 leads rebuilt from ECG2, so subjects, labels and annotator are held fixed and only
lead content changes.

| Cell | 7-lead native | Fewer-lead native (MIT-BIH) | Same events, leads rebuilt | Between-dataset Δ | **Within-subject lead Δ** |
|---|---:|---:|---:|---:|---:|
| PVC / `beat_morphology` | 0.808 | 0.759 | 0.806 | +0.049 | **+0.002** |
| NORMAL_SINUS / `beat_morphology` | 0.738 | 0.825 | 0.686 | −0.087 | **+0.052** |
| PAC / `beat_morphology` | 0.612 | 0.451 | 0.664 | +0.161 | **−0.052** |
| SINUS_BRADYCARDIA / `rate_derived` | 0.834 | 0.848 | 0.890 | −0.014 | **−0.056** |

(recall; 480/170, 500/200, 116/51 and 145/46 test events respectively. Full numbers with F1, AUROC,
subject and dataset counts: `ecg_sigma-findings-2026-09-18.deconfounding.json`.)

**The within-subject lead effect is indistinguishable from zero** — four cells spanning −0.056 to
+0.052, no consistent sign. And the between-dataset differences are small and change direction
(PAC favours INCART by 0.161, NORMAL_SINUS favours MIT-BIH by 0.087), which is what dataset-level
noise across 3–6 subjects looks like, not a lead effect.

By your own criterion — *"if lead availability alone costs little on PVC, 'single-lead is harder'
does not explain the `rhythm_annotation` collapse and the route is implicated"* — **the route is
implicated and lead availability is exonerated.**

Note this is a stronger statement than "the model ignores lead realism". Rebuilding six leads from
ECG2 makes them deterministic functions of it, so the counterfactual removes their *independent
information*, not just their realism. Recall does not move, so in these cells the model was getting
almost nothing from the other six leads to begin with.

## What "the route" actually names, and what it does not

We would not read this as "episode-level annotation is intrinsically harder". `rhythm_annotation`
is a bundle of four things that vary together and that these cells do not isolate:

1. **Annotation granularity** — an episode spanning the window vs a labelled centre beat.
2. **Source databases** — AFDB, VFDB, CUDB. Neither arm of our experiment contains them; both arms
   are INCART vs MIT-BIH.
3. **Rhythm type** — the route carries the malignant and atrial rhythms almost exclusively.
4. **Recording context** — ambulatory and resuscitation, against INCART's clinical Holter.

The cells test benign, beat-labelled classes on two clinical Holter databases. They establish that
leads are not the mechanism; they cannot tell us which of the remaining four it is. We would rank
(3) and (4) above (1), because the model's VT errors go to AF and VF — other malignant rhythms —
rather than to anything the annotation route would predict.

**The practical consequence for Track B:** its value is no longer "prove the route effect is real".
It is to supply malignant-class events from a source that is *not* AFDB/VFDB/CUDB. If VT from a new
ambulatory 12-lead database scores like INCART's `beat_run` VT rather than like VFDB's episodes,
the answer is (2) or (4), and the labeller stays innocent permanently.

## On `PACED` — withdrawal accepted, and thank you for retracting it

The priority-contest argument settles it: a paced patient in AF keeps the AF label, so a `PACED`
class cannot touch the failure we measured, and 4 paced-AF events outside MIT-BIH 217 is no signal
to learn from. We are not spending an enum change on it. We are also not taking it on its own
merits right now — "detect pacing" is not a gap we have evidence for, and we would rather not grow
the head for a class we cannot motivate from a measured failure.

## Recording the VT→PVC collapse permanently

Agreed, and done. The v1 → v2 change — **74 of 111 VT events predicted as PVC, down to 6 of 39** —
is now in `docs/real-data-training.md` and the README's Model Performance section, attributed to the
v2 stratification rather than to anything on our side. It is the clearest single number showing the
split fix worked, and neither of us was looking for it.

## Status

Nothing is blocked on you. We will re-run the deconfounding cells if Track B lands, since
`corpus_diagnostics.confounds` going empty is exactly the trigger that would make the malignant
classes testable.

---

# Part 3 — the model does use leads, VF stays excluded, and a correction to Part 2

Reply to Addendum 4. v2.3 adopted; manifest byte-identical for the fourth build, no retraining.

## We ran your test. The answer is no — the model is not globally flat.

Both passes you asked for. `force_pattern="0100000"` rebuilds all six non-ECG2 leads from ECG2,
which is exactly the configuration VF is stored in.

**Out-of-sample over train+val** (each fold model on its own held-out subjects — the power you
wanted; 7-measured-lead events only):

| Class | Events / subjects | Recall, real leads | Leads rebuilt from ECG2 | Δ |
|---|---:|---:|---:|---:|
| RBBB | 277 / 169 | 0.884 | 0.354 | **−0.530** |
| LBBB | 268 / 232 | 0.937 | 0.690 | **−0.246** |
| AV_BLOCK_1 | 284 / 257 | 0.820 | 0.746 | −0.074 |
| PVC | 2,460 / 414 | 0.825 | 0.780 | −0.045 |
| NORMAL_SINUS | 8,177 / 6,201 | 0.816 | 0.775 | −0.041 |
| SINUS_BRADYCARDIA | 1,181 / 456 | 0.781 | 0.755 | −0.026 |
| ATRIAL_FIBRILLATION | 1,126 / 856 | 0.904 | 0.885 | −0.019 |
| SINUS_TACHYCARDIA | 861 / 414 | 0.805 | 0.794 | −0.010 |
| PAC | 676 / 216 | 0.658 | 0.683 | +0.025 |
| VENTRICULAR_TACHYCARDIA | 145 / 5 | 0.524 | 0.531 | +0.007 |
| ATRIAL_FLUTTER | 37 / 29 | 0.622 | 0.622 | 0.000 |
| SVT | 40 / 30 | 0.250 | 0.250 | 0.000 |

On the test split the same test gives RBBB −0.381, PAC −0.404, VT −0.353, PVC −0.130,
NORMAL_SINUS −0.105 (full table in `ecg_sigma-findings-2026-09-18.leadsensitivity.json`).

**RBBB loses more than half its recall when its leads are rebuilt. The premise fails.**

## The mechanism, which we did not expect: it is V1, not the limb leads

Part 2's cells used `force_pattern="0100001"` — which keeps **V1 genuinely measured** and rebuilds
only the five limb leads. Those came back flat. This test rebuilds V1 as well, and the same classes
move sharply. The difference between the two runs is exactly one lead.

So the model uses `vVX` (V1) and is close to indifferent to the fabricated limb leads. Clinically
that is the expected place to look for bundle-branch morphology, which is why RBBB and LBBB dominate
the table. It also explains the aggregate flip rates in our reports: 0.070 converting to `0100001`
against 0.167 converting to `0100000`.

### A correction we owe you

In Part 2 we wrote that the counterfactual removes the other six leads' *independent information*
and that "the model was getting almost nothing from the other six leads to begin with". **That
overstated the scope.** It was true of the five fabricated limb leads and false of V1. The Part 2
cells and their conclusion — lead availability does not explain the `rhythm_annotation` collapse —
still hold, because the classes in those cells are not V1-dependent; but the sentence generalised
past its evidence and should be read as "the five limb leads", not "the other six leads".

## VF: keep it excluded, and the flag is doing better work than its name suggests

Your argument was that if the model reads only ECG2, VF's score is on real signal. It does not read
only ECG2 — and the lead it does read is the one VF almost never has. 452 of 464 VF events are
`0100000`, so V1 is fabricated for all but 12 of them.

The deployment monitor supplies a genuinely measured V1. A model that demonstrably uses V1 for other
classes would be seeing, for VF, its first real V1 at inference time. That is precisely the
untested transfer `no_seven_real_lead_events` was meant to flag.

**We are not asking to change `reporting.primary_classes`.** Keep VF excluded, keep the flag. If
anything this strengthens it: the flag would be sharper if it were about *V1* rather than about all
seven leads, since that is the lead that carries information for this model.

**VT is the one honestly ambiguous row.** Out-of-sample it is flat (+0.007 on 145 events, but only
5 subjects); on test it drops 0.353 (17 events, 2 subjects). Two thin estimates that disagree, so we
would not build on either. It does not change the VF conclusion, which rests on RBBB and LBBB.

## Your two new cell types

`granularity_cells` is the right design and we agree it is too thin — 0 test events for MIT-BIH
`beat_run` VT.

**`within_dataset_route_cells` does not do what it claims, though.** Within MIT-BIH the two routes
carry disjoint classes: `beat_morphology` has NORMAL_SINUS, PAC, PVC, LBBB, RBBB; `rhythm_annotation`
has AF, AFL, VF, VT, SVT. Comparing 521 test events of one set against 156 of the other compares
*classes*, not routes — and a route effect cannot be told from a class-difficulty effect. Only SVT
and VT appear on both routes inside MIT-BIH, which is your `granularity_cells`, with its
sample-size problem. We think factors 2 and 4 survive.

## The labeller weld — deferring, and why

Thank you for disclosing it; a design coupling is worth more to us than a corpus coupling, because
it is the kind we would never have found from the manifest.

**We are deferring the change, not declining it.** Our reasoning, as the side that pays:

* It buys diagnostic insight, not model quality. Same events, same labels, only `label_method`
  changes — our deployment numbers would not move.
* We have just spent roughly 15 GPU-hours on v2. Another 10 to learn *why* three already-flagged
  classes are weak is a poor trade against the alternative uses of that time right now.
* The cost is near zero if it rides along: **please fold the sinus-route change into whichever
  format bump you next make for other reasons, and we will pick it up on that retrain.** We would
  rather wait for a free ride than commission one.

If Track B lands first and points at factor 2 or 4, the question may not need answering at all.

## Status from our side

Nothing is blocked on you, and we have no open asks. `models/real_v2` is unchanged and remains the
deployed artifact; v2.3 is what our loader points at. We will re-run the deconfounding cells and
this lead-sensitivity test if a future package changes the corpus.

