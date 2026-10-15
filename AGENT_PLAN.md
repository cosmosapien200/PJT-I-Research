# Agent Plan: Pathology Bias in PCG Quality Filters

**Read this whole file before writing any code.**

## 0. What you are building and why

Heart sound (PCG) research pipelines use automatic *quality filters* that reject recordings or beats judged too noisy. We suspect these filters also reject *pathological* heartbeats (murmurs, irregular rhythm) more often than normal ones, because pathology can look or sound like noise.

Your job is to implement several published quality filters, train them on one dataset, and measure whether they reject clean pathological beats more often than clean normal beats, both as recorded and under identical added noise.

**The one question every experiment serves:** at the same noise level, do quality filters reject pathological beats more often than normal ones?

## 1. Ground rules (important)

- **Aim for a correct, respectable implementation, not a perfect one.** Use defaults, standard libraries and simple choices. Do not tune hyperparameters beyond what is written here.
- **Don't get stuck.** If a step hits a blocker, take the stated fallback, log it, and move on.
- **Log every deviation** from this plan or from an original paper in `DEVIATIONS.md` (one line each: what, why). This file is more valuable than polish.
- **Never use test datasets (Section 6.2) for training or tuning**, except the calibration split defined in Phase C.
- **Split by patient** wherever a patient ID exists. Never put beats of one patient on both sides of a split.
- **Save results as CSV files**, not only notebook output. Every number in the final summary must come from a file in `results/`.
- If something here is ambiguous, pick the simplest reasonable option, log it, and continue. Do not stop to ask unless a dataset cannot be obtained.
- Fixed random seed `42` everywhere. One seed per model.

## 2. Speed rules

1. **Start dataset downloads first, in the background,** and write code while they run.
2. **Extract features once and cache them** (log-mels and hand-crafted features per window, saved as `.npz` in `data/processed/`). Every model and experiment reads the cache; never recompute.
3. **Quiet output.** No progress bars, `verbose=0`. Long runs go to the background and write to log files; check only the last few lines. Never print or open whole datasets, CSVs or logs; use `head` and counts.
4. **Deep models (CNN, CRNN, E5 classifier): one train/validation split, one seed.** Use 5-fold CV only for the cheap classic models.
5. **Commit once per phase,** not after every change.

## 3. Progress and training logs

- **`PROGRESS.md`** (keep it short; overwrite the relevant line, don't append essays). Update it whenever a milestone completes:

  ```
  # Progress   (last updated: <time>)
  ## Datasets
  - [x] PhysioNet 2016: 3,153 records, 2 kHz
  - [ ] CirCor: downloading
  ## Models (validation AUROC / balanced accuracy)
  - [x] B4 CNN: 0.91 / 0.84
  - [ ] B1 Springer SQI
  ## Experiments (one key number each)
  - [x] E0 label check: SQI-0 rate normal 4.1% vs abnormal 9.8%
  - [ ] E2 equal noise
  ## Blockers
  - ZCHSound download link not found; skipped
  ```

- **Training curves (deep models):** one CSV per model at `logs/train/<model>.csv`, one row per epoch: `epoch, train_loss, val_loss, val_auroc, val_balanced_acc, lr`. Write each row as the epoch finishes.
- **Classic models:** one CSV per model at `logs/train/<model>_cv.csv`, one row per fold: `fold, auroc, balanced_acc, sensitivity, specificity`.
- At the end, one script plots all training curves to `figures/training_curves.png`.

## 4. Data splits and leakage rules (must follow)

1. **Split first, window later.** Assign every recording (and patient, where IDs exist) to a split, save it in `manifests/`, and only then cut windows or beats. Windows from one recording never cross splits.
2. **Patients stay together.** All recordings and beats of a patient go to the same split (PhysioNet subset b, CirCor, ZCHSound, BMD-HS where IDs exist). For PhysioNet subsets without IDs, split by recording and log it.
3. **Training vs test datasets.** Quality models train only on PhysioNet 2016. CirCor, ZCHSound, BMD-HS and PASCAL B are never used for training or model selection.
4. **Calibration patients.** In each test dataset, 20% of normal patients are reserved for setting thresholds and the clean-beat cut-off (Phase C, Phase D). They are excluded from every evaluation, including E5.
5. **Test noise must be unheard.** Bank 2 (E2 test noise) uses only sources no model was trained with: **LibriSpeech `test-clean`** (speech) and the **DEMAND** environmental noise database. Bank 1 (training augmentation) uses ESC-50, Mini LibriSpeech `dev-clean-2` and PhysioNet `(N ... N)` stretches.
6. **Pre-trained models.** PCGKolect (B6) was trained on PhysioNet 2016 + CirCor: report its CirCor results with that caveat. CardioPHON's as-published model was trained on PASCAL: do not report its PASCAL results.
7. **E5 split.** CirCor non-calibration patients are split 60/40 by patient; the 40% test patients are never used to train the murmur classifier.
8. **Automatic check.** Write one function `check_leakage()` that asserts: no test-dataset record or patient ID is in any training manifest; no patient appears in two splits; no calibration patient is in any evaluation table; no Bank 2 file is in Bank 1. Run it at the end of Phase A and before Phase E; record "leakage check passed" in `PROGRESS.md`.

## 5. Environment

- Python 3.12, PyTorch, NumPy, SciPy, pandas, scikit-learn, LightGBM (only if needed), statsmodels, librosa, soundfile, matplotlib.
- Optional helper: `pyPCG` (Python PCG toolbox) for standard signal-quality indices and HSMM segmentation.
- Hardware: one RTX 3060 (12 GB), 32 GB RAM. Every model here trains in minutes to a few hours.
- Existing code to reuse: [PCGKolect_model](https://github.com/cosmosapien200/PCGKolect_model) (U-Net + HMM model, noise bank scripts, PhysioNet loaders, trained checkpoints).

### Repository layout

```
configs/            one YAML per model (preprocessing, features, hyperparameters, seed)
data/raw/           downloaded datasets (git-ignored)
data/processed/     resampled audio, beat tables, window tables
manifests/          CSV of record/patient IDs per split (committed)
src/                code
results/            CSV outputs per experiment
logs/train/         per-epoch / per-fold training metrics
PROGRESS.md         short live status (see Section 3)
figures/            plots
DEVIATIONS.md       every deviation, one line each
ANALYSIS_PLAN.md    thresholds, metrics and tests, written in Phase C before testing
RESULTS.md          final summary (tables + 1-2 sentences each)
```

## 6. Datasets

### 6.1 Training data (quality filters are trained ONLY on this)

| Dataset | What to use | Link |
|---|---|---|
| PhysioNet/CinC Challenge 2016 | `training.zip` + `annotations.zip`. Recording quality labels in `annotations/updated/training-*/REFERENCE_withSQI.csv` (SQI 1 = usable, 0 = unusable). Noise stretches marked `(N ... N)` in `annotations/hand_corrected/`. 2 kHz. | https://physionet.org/content/challenge-2016/1.0.0/ (fast mirror: `https://physionet-open.s3.amazonaws.com/challenge-2016/1.0.0/`) |
| EPHNOGRAM (optional extra negatives) | Recordings flagged low quality in `ECGPCGSpreadsheet.csv`; cut PCG into 5 s windows labelled unusable. Resample to 2 kHz. | https://physionet.org/content/ephnogram/1.0.0/ |

### 6.2 Test data (never trained on)

| Dataset | Use | Labels to extract | Link |
|---|---|---|---|
| CirCor DigiScope v1.0.3 | **Main test** | Per patient: murmur Present/Absent/Unknown, murmur locations, systolic/diastolic timing, grade, shape; outcome Normal/Abnormal. Per recording: `.tsv` segmentation (1=S1, 2=systole, 3=S2, 4=diastole, 0=unannotated) | https://physionet.org/content/circor-heart-sound/1.0.3/ |
| ZCHSound | Replication (children, other hospital) | Control, ASD, VSD, PFO, PDA | Paper: https://doi.org/10.1109/tbme.2023.3348800 (find the download link from the paper; if unobtainable, skip and log) |
| BMD-HS | Replication (adults, valve diseases) | Multi-label AS, AR, MR, MS, multi-disease, normal | https://github.com/mHealthBuet/BMD-HS-Dataset |
| PASCAL Classifying Heart Sounds, dataset B | Irregular rhythm test | Normal, murmur, extrasystole | Search "PASCAL Classifying Heart Sounds Challenge"; a Kaggle mirror ("heartbeat sounds") exists |

Known caveats (handle as written, do not over-engineer):
- **CirCor:** a murmur label applies only to the locations listed as audible. Beats from a murmur patient's *other* locations go into a separate group `murmur_patient_other_location`, never into "normal". Patients with murmur `Unknown` form their own group.
- **BMD-HS:** healthy subjects were recorded in a lab with hospital noise played back. Report its E1 numbers but flag them as confounded; rely on E2 for BMD-HS.
- **ZCHSound, BMD-HS, PASCAL** have no beat annotations: segment them with one fixed segmenter (Phase D).

### 6.3 Noise banks

- **Bank 1 (training augmentation only):** ESC-50, Mini LibriSpeech `dev-clean-2`, PhysioNet 2016 `(N ... N)` stretches (as built by `PCGKolect_model/seg/prep_noise.py`).
- **Bank 2 (E2 test noise only):** LibriSpeech `test-clean` (https://www.openslr.org/12) and DEMAND (https://zenodo.org/record/1227121). No model has seen these.

Save the file lists to `manifests/noise_banks.csv`.

## 7. Phases

### Phase A: Data setup

1. Download all datasets. Write `manifests/*.csv` listing every record with: dataset, record ID, patient ID (if any), split, labels.
2. Resample everything to **2 kHz** mono float32 (PhysioNet is already 2 kHz; never upsample). Models needing 1 kHz (CardioPHON) resample again from the 2 kHz copy.
3. PhysioNet split: 5 folds, by patient for subset b (has subject IDs), by recording elsewhere. Log this limitation.
4. **Training windows:** 5 s windows, 2.5 s hop, from PhysioNet.
   - Label **unusable** if the recording has SQI 0, or at least 50% of the window lies inside a `(N ... N)` stretch.
   - Label **usable** if the recording has SQI 1 and the window has no overlap with any noise stretch.
   - Drop windows that fit neither.
   - Augmentation: for 30% of usable windows, add a Bank 1 noise clip at a random SNR in [-5, 5] dB and label it unusable.
5. **E0, label check (do this first):** for PhysioNet 2016, compute the SQI 0 rate for normal vs abnormal recordings, overall and per subset (a to f), with counts. Save `results/e0_label_check.csv`.

### Phase B: Build and validate models

Train every model on the Phase A training windows (the **common corpus**). Classic models: 5-fold CV on PhysioNet, then a final model on all PhysioNet data. Deep models: one train/validation split (80/20, by patient where possible); use that model for testing. Report, per model, on held-out folds: AUROC, balanced accuracy, sensitivity, specificity, EER (`results/b_quality_performance.csv`).

**Score convention:** every model outputs a score in [0, 1] where **higher = more likely unusable**. Window-level models score a 5 s window.

Optional **as-published arm:** if a paper's original trained model is downloadable (check CardioPHON's repo), also score the test data with it. Do not spend time reproducing papers whose data is private.

#### B1. Springer-style SQI + logistic regression / SVM

- **Source:** Springer et al., "Signal quality classification of mobile phone-recorded phonocardiogram signals", ICASSP 2014 (SVM, 10 SQIs); Springer et al., "Automated signal quality assessment of mobile phone-recorded heart sound signals", J. Med. Eng. Technol. 2016 (logistic regression, 9 SQIs). Full feature definitions: Springer DPhil thesis, Oxford 2015: https://ora.ox.ac.uk/objects/uuid:5ec8c818-dafb-4571-8198-97607f8d0451
- **Implementation (pragmatic):** compute these SQIs per window: kurtosis of the signal; standard deviation of the Hilbert envelope; maximum envelope autocorrelation in the 0.5 to 2 s lag range; sample entropy of the envelope; degree of periodicity (pyPCG has all five). Add any further SQIs from the thesis only if quick to implement.
- **Preprocessing:** 25 to 400 Hz Butterworth band-pass (order 4), as commonly used in Springer's work (log as an assumption if not confirmed from the thesis).
- **Classifiers:** logistic regression (default, class-balanced) and SVM (RBF, class-balanced). Report both; use the SVM in the bias experiments.

#### B2. Das et al. autocorrelation + spectral features + random forest

- **Source:** Das et al., "Novel features from autocorrelation and spectrum to classify phonocardiogram quality", Computing in Cardiology 2017 (TCS). Detailed method in the same team's patent US11373757B2: https://patents.google.com/patent/US11373757
- **Preprocessing:** Hilbert envelope; 2nd-order Butterworth low-pass about 20 Hz; autocorrelation of the envelope; 2nd-order Butterworth low-pass 10 Hz with zero-phase (`filtfilt`).
- **Features (17; use those that are well defined, log the rest):** sample entropy of the processed autocorrelation (patent lists 4 variants: implement 1 and log); kurtosis; minimum ratio of 2nd to 1st singular value over sliding windows; signal power; variance; correlation with a fitted sum of 1, 2 and 3 rectified sinusoids; power ratio 0 to 240 Hz vs 240 to 1000 Hz; mean and variance of the ratio of 0 to 500 Hz power to the rest, over 2 s frames with 0.5 s overlap. Skip the "decaying rectified sinusoid" features if under-specified (log it).
- **Classifier:** random forest, 120 trees, `max_features="sqrt"`, class-balanced. (The original trained 16 balanced models and averaged them; class weighting is an acceptable simplification; log it.)
- **Original result for reference:** sensitivity 88.28%, specificity 66.67% on a private smartphone test set.

#### B3. CardioPHON quality model

- **Source:** Despotovic, Pocta, Zgank, "CardioPHON: Quality assessment and self-supervised pretraining for screening of cardiac function based on phonocardiogram recordings", Biomedical Signal Processing and Control 2025/26; arXiv: https://arxiv.org/abs/2511.04533 ; code and labels: https://git.lih.lu/vdespotovic/cardiophon
- **Method:** 416 features (following Grooby et al., neonatal heart/lung sound quality assessment) → mutual-information ranking, keep top 20% → soft-voting ensemble of SVM, random forest and gradient boosting. Input resampled to 1 kHz; recordings shorter than 6 s extended by concatenating copies.
- **Original result for reference:** accuracy 0.933, F1 0.955, AUROC 0.983 (random 80/20 split, their 8,359-recording corpus).
- **Plan:** clone the repo and use its feature extraction code. If their trained model is included, also run it as the as-published arm.
- **Fallback (if the repo lacks feature code):** skip this model and log it. Do not reimplement 416 features.

#### B4. CNN (study baseline)

- No published PCG quality CNN with a full specification was found, so this is our generic baseline; describe it as such.
- **Input:** 5 s window at 2 kHz, 20 Hz high-pass, per-window z-score → log-mel spectrogram, 64 mel bands (20 to 1000 Hz), 25 ms window, 10 ms hop.
- **Architecture:** 4 × [Conv 3×3 → BatchNorm → ReLU → MaxPool 2×2] with 32, 64, 128, 128 channels → global average pool → dropout 0.3 → linear → sigmoid.
- **Training:** Adam, lr 1e-3, batch 64, binary cross-entropy with class weights, up to 50 epochs, early stopping on validation loss (patience 5).

#### B5. CRNN (study baseline)

- Same input and first two conv blocks as B4; last two blocks pool over frequency only (2×1), keeping time resolution → reshape to a time sequence → 2-layer bidirectional LSTM, 64 units → mean over time → linear → sigmoid. Same training settings.

#### B6. PCGKolect segmentation model (1D U-Net + duration-constrained HMM)

- **Source:** this project, https://github.com/cosmosapien200/PCGKolect_model (see its README and `seg/`).
- **Do not retrain** for the main experiments; it was trained for segmentation, not quality labels. Use the released checkpoint (`release/seg_final.pt`) with the Python reference pipeline (`seg/stream.py`, `seg/decode.py`).
- **Pipeline (as in repo):** 2 kHz, 20 Hz high-pass; 128-point Hann STFT, hop 40 (20 ms); 32 mel bands 25 to 900 Hz; log; minus mean; divide by 4 → U-Net → 5-state frame probabilities → duration-constrained Viterbi → beats with confidence.
- **Two variants:**
  - **B6a:** unusable score = 1 − beat confidence.
  - **B6b:** as B6a, and also unusable if beat length is outside 0.3 to 2.0 s or differs by more than 30% from the neighbouring beats (the deployed rhythm check).
- **Beat matching:** a reference beat takes the score of the decoded beat overlapping it most (at least 50% overlap). No match → unusable.
- **Published threshold:** confidence 0.80 (device default). The repo README says 0.70; use 0.80 and log it.
- **Caveat to log:** the checkpoint was trained on PhysioNet 2016 + CirCor segmentation labels, so CirCor is not fully unseen for B6. Note this in RESULTS.md; the comparison on ZCHSound, BMD-HS and PASCAL is clean.

#### B7. SNR reference line (not ML)

- Score = −(local SNR in dB), SNR = power of S1+S2 segments over power of the quietest diastolic segment in the window. Based on Giordano et al. 2021, "Automated Assessment of the Quality of Phonocardographic Recordings through Signal-to-Noise Ratio for Home Monitoring Applications" (published threshold SNR ≥ 14 dB = usable).

#### Not implemented (log in DEVIATIONS.md)

- **PCSCN** (Jiang et al., Computer Methods and Programs in Biomedicine 2025, 271:108977): no public code, paywalled. Skip unless the user supplies code or the PDF.
- **Zabihi et al. 2016 neural-network ensemble** (Computing in Cardiology): optional; implement only if everything else is done early.

### Phase C: Thresholds and analysis plan

Before scoring any test set, write `ANALYSIS_PLAN.md` with the thresholds, metrics and tests below, and do not change it after seeing test results (if you must, log why).

Two thresholds per model:

1. **Published / default threshold:** the paper's value if given; otherwise 0.5 on the score.
2. **Matched threshold (main analysis):** for each test dataset separately, randomly reserve **20% of its normal patients** as a calibration set. Set the threshold so the model rejects **10% of clean normal beats** in that calibration set. Exclude calibration patients from all evaluation.

### Phase D: Prepare test beats

1. **Beat cutting:**
   - CirCor: use its `.tsv` annotations; a beat is one S1 onset to the next S1 onset, fully annotated (labels 1→2→3→4 with no 0).
   - ZCHSound, BMD-HS, PASCAL B: segment with the Springer LR-HSMM (pyPCG or another Python port of Springer's algorithm). Use it for every model. Record the segmentation failure rate per class.
2. **Window per beat:** a 5 s window centred on the beat (pad by reflection at recording edges). Window-level models score this window; B6 and B7 score the beat directly.
3. **Class labels per beat:**
   - CirCor: `normal` (murmur Absent and outcome Normal), `murmur_systolic`, `murmur_diastolic` (at audible locations only), `murmur_patient_other_location`, `unknown`. Keep murmur timing, grade and shape as extra columns.
   - ZCHSound: control / ASD / VSD / PFO / PDA. BMD-HS: normal / AS / AR / MR / MS / multi. PASCAL B: normal / murmur / extrasystole.
4. **Clean-beat rule (simple, objective):**
   - Estimate beat SNR = power of S1+S2 over power of the quietest phase (systole or diastole) **that is not the murmur phase**. For non-murmur beats, use diastole.
   - A beat is **clean** if this SNR ≥ the 30th percentile of SNR among the dataset's **calibration** normal beats (so about 70% of normal beats pass).
   - Report the pass rate per class in `results/d_clean_pass_rates.csv`. A lower pass rate for pathological beats is itself a finding; do not try to "fix" it.
5. **Semi-synthetic irregular rhythm (for the rhythm test):** from clean normal CirCor beats, build 20 s sequences where 20% of beats are shifted early by 25 to 40% of the RR interval (mimicking premature beats), and a second set with RR intervals drawn uniformly from ±30% (mimicking AF-like irregularity). Label `synthetic_premature` and `synthetic_irregular`.

### Phase E: Experiments

All rejection rates are reported per model × dataset × class at both thresholds.

| Exp | What | Output |
|---|---|---|
| **E1, raw rejection** | Score all test beats as recorded. Rejection rate per class. | `results/e1_raw.csv` |
| **E2, equal noise (decisive)** | Take clean beats only. For each beat, pick a Bank 2 noise clip using a seed derived from the beat ID (so paired normal and pathological beats get comparable noise), and add it to the 5 s window at SNR 0, 5, 10, 20 dB, plus no noise. SNR is computed against the clean window's power. Rejection rate per class per SNR level. | `results/e2_equal_noise.csv` |
| **E3, breakdown** | From E1/E2: CirCor by murmur timing, grade, shape; ZCHSound and BMD-HS by disease; PASCAL extrasystole and the synthetic rhythm sets. | `results/e3_breakdown.csv` |
| **E4, by source** | Every result is already split by dataset (each is a different device and population). Just produce a summary table comparing the normal-vs-pathological gap across datasets. | `results/e4_by_dataset.csv` |
| **E5, downstream effect** | Murmur classifier = the B4 CNN architecture with target murmur present/absent. Split CirCor non-calibration patients 60/40 by patient. Train once on the unfiltered 60%, and once per quality model on the 60% filtered at that model's matched threshold. Test all on the unfiltered 40%. Report patient-level murmur sensitivity, specificity, AUROC (patient score = max over beats). 1 seed is fine. | `results/e5_downstream.csv` |
| **E6, mitigation** | (a) B6a vs B6b shows the effect of the rhythm check. (b) Retrain the CNN (B4) with the usable class balanced 50/50 between normal and abnormal PhysioNet recordings; rerun E2 for it. | `results/e6_mitigation.csv` |

### Phase F: Statistics

Keep it simple:
1. **Confidence intervals:** 95% bootstrap over **patients** (1,000 resamples) for every rejection rate and for the difference "pathological minus normal".
2. **Main test (E2):** logistic regression `rejected ~ class + snr_level` with standard errors clustered by patient (statsmodels GEE, exchangeable correlation), per model and dataset. Report the odds ratio for each pathology class vs normal.
3. **Multiple models:** Holm correction across models for the main test.

## 8. Final deliverables

- `RESULTS.md`: for each experiment, one table and one or two plain sentences. Lead with E2: the rejection gap between pathological and normal beats at each SNR, per model, with CIs.
- Figures:
  - E2: rejection rate vs SNR, one line per class, one panel per model.
  - E3: rejection rate by murmur grade/timing.
  - E5: murmur sensitivity with vs without filtering, per model.
- `DEVIATIONS.md`, `ANALYSIS_PLAN.md`, `PROGRESS.md`, `manifests/`, `configs/`, all `results/*.csv`, `logs/train/*.csv`.
- `figures/training_curves.png`.
- A short `REPRODUCE.md`: the commands to rerun everything from raw data.

## 9. Order of work (summary)

1. E0 label check.
2. Phase A data setup, then `check_leakage()`.
3. B4 CNN and B7 SNR first (fastest), then run Phase C to E2 end-to-end on CirCor with just these two. **This validates the whole pipeline early.**
4. Add B6, B1, B2, B3, B5.
5. Full E1 to E6 on all datasets.
6. Statistics and RESULTS.md.

If time is short, the minimum publishable core is: **B1, B2, B4, B6a, B6b + E1, E2, E3 on CirCor and BMD-HS.**
