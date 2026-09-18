## Context
ecg_sigma ships `ecgpkg` v1 (`ecg_sigma/packages/ecg_pkg_v1`, contract in `CONTRACT_ecgpkg_v1.md`): 37,587 manifest rows (25,605 included), 12 head classes, 2400-sample (12 s) windows from MIT-BIH/INCART/VFDB/CUDB/AFDB and 2000-sample (10 s) PTB-XL windows, 200 Hz, mV, already band-passed 0.5–40 Hz + 50 Hz notch. Machine: RTX 4050 6 GB, 15 GB RAM (whole-dataset arrays have OOM'd), long background jobs get killed.

Measured facts that shape the design:
- `default` preprocessing costs 19–24 ms per (7, 2000) item when filtering lead by lead; the same filters vectorised over leads cost 3.5 ms with bit-identical output.
- PTB-XL stores one event per file (21,792 files, ~25 ms open+read each); cache building must be parallel.
- Class × source confounding: VF events are 100 % `0100000`/`0100001`; AFL test events are 50/53 AFDB (`0100000`); AV_BLOCK_1 test events are 100 % PTB-XL. Real-lead mask alone separates {AF, AFL, VT, VF, AV_BLOCK_1} from the rest.
- ecg_sigma's fabrication is deterministic: ECG1 = −0.6·lowpass20(II) shifted 3 ms; ECG3/aVR/aVL/aVF by Einthoven/Goldberger; synthetic vVX = highpass1(II) + 0.4·sign(q)|q|^1.2 with q = bandpass5–20(II).
- Events per subject: PTB-XL 1; INCART median 170; MIT-BIH 95.

## Goals / Non-Goals
- Goals: train/evaluate/infer on `ecgpkg` with a data-driven head; keep the simulator path; expose and reduce the lead-realism shortcut; resumable training within machine limits; significantly beat the baseline on v1 test.
- Non-Goals: changing ecg_sigma, the package or the contract; multi-label output; mixing simulator events into package training; model-architecture changes; vitals/MEWS beyond reader fixes.

## Decisions
- **ClassSpec is the only source of class order.** Loaded from `package.json` for training and from `checkpoint["class_names"]` for inference (falls back to the `Condition` enum when absent and the state dict has 16 queries). All scripts build models through `checkpoint.load_model`, so no script touches `NUM_CLASSES`.
- **Condition resolution.** Stored `condition` may be an enum value (`"V"`, simulator) or name (`"PVC"`, ecg_sigma). Resolve to the enum name, then map into the head; names outside the head (or outside the enum, e.g. `OTHER`) show `n/a (not in head)`, are excluded from metrics, and warn once.
- **Cache.** `data/training_cache/<package_version>_<split>_<leadtag>.npy` created with `np.lib.format.open_memmap`, shape `(N, L, 2400)` float32, right-padded; sidecars `.lengths.npy`, `.labels.npy`, `.meta.json` (manifest sha, row uids digest, leads). Rows grouped by HDF5 file; a process pool writes each group into its own row range through its own `r+` memmap, so peak RSS is bounded by one file's events. Built into a `.partial` name and renamed on success; reused only when the sidecar matches. Signals stored unnormalised.
- **Item pipeline.** `crop (train: uniform start in valid region; eval: centre crop or full valid length) → [train] lead-fabrication aug → [train] artefact injection → fabricated_leads="zero" masking → PreprocessingPipeline → [train] per-lead scale, Gaussian noise, channel dropout`. Per-item RNG is `np.random.default_rng((seed, epoch, index))`, so results do not depend on worker count; the loop calls `set_epoch` and uses non-persistent workers.
- **Vectorised preprocessing.** `sosfiltfilt`/`filtfilt` with `axis=-1` over the (L, T) array; the median filter stays per lead (kernel semantics). Output equality is asserted by a test against the per-lead reference.
- **Artefact injection scaled to amplitude.** Simulator noise presets are absolute mV calibrated for ~0.2 mV simulator leads, while real lead std ranges 0.04–1.0 mV. Each lead is divided by `std/0.2` (clipped 0.25–4), passed through `apply_noise_pipeline` with a random low/medium preset, and rescaled.
- **Lead-fabrication augmentation (extra).** With probability `--lead-fab-aug-prob`, a train item whose mask has more real leads is converted, from its own ECG2, to the `0100001` (keep real vVX) or `0100000` pattern using the ecg_sigma formulas, then band-passed like the package. The same transform applied to all `1111111` test events gives the **lead-conversion counterfactual**: prediction flip rate and metrics after conversion quantify the shortcut more directly than zeroing.
- **Sampler (extra).** `--sampler balanced`: weight_i ∝ class_share(c_i) / (events of subject_i in class c_i)^β, normalised so each class receives equal mass; β=0.5 default. When used, loss class weights are off by default to avoid double correction.
- **Warm start.** Load every shape-compatible tensor from `--init-checkpoint`. `ffn_head` is per-query (output 1), so it is class-count independent. `--init-queries by_name` (default) copies object-query rows for classes present in both heads; `reinit` follows the brief literally (re-initialise queries and FFN head when counts differ). Run c compares both.
- **Selection and metrics.** Select on val macro-F1 over present classes. Reports: per-class P/R/specificity/F1 with support and subject counts, macro, accuracy, confusion matrix sized to the head, grouped by dataset / label_method / real_lead_mask, subject-bootstrap 95 % CI for macro-F1, per-class one-vs-rest AUROC; for 12 s events both centre-crop 2000 and full 2400.
- **Baseline mode.** A checkpoint whose head differs from the package head is mapped by name; predictions outside the package head count as wrong.
- **Bias calibration and ensemble (extra).** `evaluate.py --calibrate-on val` fits a per-class additive logit bias by coordinate ascent on val macro-F1 and applies it to test; multiple `--checkpoint` arguments average softmax outputs.
- **Resumable training.** `last.pt` each epoch (model, optimiser, scheduler, scaler, history, best metric, patience, epoch); `--resume` continues; `--time-budget-min` stops cleanly between epochs so each foreground call stays under 10 minutes.

## Risks / Trade-offs
- Few evaluation subjects (AFL 4 test, VF 6, VT 7) → wide intervals; mitigated by reporting subject counts and bootstrap CIs, not fixed.
- Fabrication augmentation may slightly reduce use of real limb-lead morphology (LBBB/RBBB/AV_BLOCK_1); probability is tuned (0 / 0.3 / 0.5) and per-class effects reported.
- Vectorised filtering changes the code path of the simulator pipeline; output equality is tested, so cached simulator datasets stay valid.
- Val-fitted bias can overfit val (3 AFL subjects); reported separately from uncalibrated results.
- `processor.py` default preset now follows the checkpoint; users who relied on passing nothing with a checkpoint trained on `default` get filtering they previously lacked (intended).

## Migration Plan
No data migration. Legacy checkpoints keep working. New checkpoints carry `class_names`, `data_source`, package provenance and preprocessing settings. Rollback = use a legacy checkpoint.

## Open Questions
- Deployment chest lead is V1 (confirmed by user: 7 real leads incl. V1).
- Questions relayed to ecg_sigma (non-blocking): fabrication order vs band-pass/resampling; VFDB AV_BLOCK_1 annotation mapping; PTB-XL NORMAL_SINUS inclusion rule; per-dataset ADC gain.
