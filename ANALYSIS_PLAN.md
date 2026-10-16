# Analysis plan (Phase C)

Written 2026-10-16, after training/validating the quality models on PhysioNet 2016 and **before any
test-set scoring**. Not to be changed after seeing test results; any change is logged at the bottom with the reason.

## Question
At the same noise level, do quality filters reject pathological beats more often than normal beats?

## Test data and units
- Datasets (never trained on): CirCor 1.0.3 (main), ZCHSound clean subset, BMD-HS, PASCAL B, plus the
  semi-synthetic CirCor rhythm sets (`synthetic_premature`, `synthetic_irregular`).
- Unit of scoring: one beat (S1 onset to next S1 onset). Window-level models score a 5 s window centred on
  the beat; B6 scores the decoded beat matching the reference beat; B7 scores the window around the beat.
- At most 8 beats per recording (random, seed from record ID).
- Unit of inference: patient (bootstrap and GEE clusters).
- Classes
  - CirCor: `normal` (murmur Absent and outcome Normal), `murmur_systolic`, `murmur_diastolic` (audible
    locations only; any diastolic murmur), `murmur_patient_other_location`, `unknown`, and
    `no_murmur_abnormal_outcome` (murmur Absent, outcome Abnormal; reported, not a main contrast).
  - ZCHSound: control / ASD / VSD / PFO / PDA. BMD-HS: normal / AS / AR / MR / MS / multi.
    PASCAL B: normal / murmur / extrasystole.
- Reference class for every contrast: the dataset's normal class (`normal`, or `control` for ZCHSound).
  Synthetic rhythm sets are compared with CirCor `normal`.

## Calibration split
- For each test dataset, 20 % of normal patients (seeded) are calibration patients. They are used only for
  (a) the clean-beat SNR cut-off and (b) the matched thresholds, and are excluded from all evaluation.
- PASCAL: a patient is normal only if all their recordings are labelled normal.

## Clean-beat rule
- Beat SNR = power of S1+S2 over power of the quietest phase that is not the murmur phase
  (25-400 Hz band-pass). Non-murmur beats: diastole. Systolic murmur: diastole. Diastolic murmur: systole.
  Murmur in both phases or unknown phase (PASCAL murmur, ZCHSound PFO/PDA, BMD-HS multi with mixed phases): the quieter phase.
- Clean if SNR >= 30th percentile of SNR among the dataset's calibration normal beats.
- Pass rate per class is reported (`results/d_clean_pass_rates.csv`) and not corrected.

## Models and scores (higher = more likely unusable)
| Model | Score | Published / default threshold |
|---|---|---|
| B1 LR, B1 SVM (Springer SQIs) | P(unusable) | 0.5 |
| B2 RF (Das features) | P(unusable) | 0.5 |
| B4 CNN, B5 CRNN, E6 balanced CNN | P(unusable) | 0.5 |
| B6a PCGKolect | 1 − beat confidence (no match → 1) | confidence 0.80 → score > 0.20 |
| B6b PCGKolect + rhythm check | as B6a, and 1 if duration/rhythm check fails | score > 0.20 |
| B7 SNR | −SNR dB | SNR < 14 dB → score > −14 |
B3 (CardioPHON) was added after the first full analysis, at the user's request (re-implemented features; see DEVIATIONS.md); its threshold, metrics and tests are exactly those of the other models.
The SVM is the B1 model used in the bias experiments; LR is reported alongside.

## Thresholds
1. Published/default threshold as in the table (reject if score > threshold).
2. Matched threshold (main analysis): per model × dataset, the 90th percentile of scores of the clean
   normal calibration beats as recorded (rejects 10 % of them). Reject if score > threshold.
   The same matched threshold is used at every E2 noise level. Synthetic rhythm sets use CirCor's thresholds.

## Experiments
- E1: all evaluation beats as recorded; rejection rate per model × dataset × class × threshold.
- E2 (decisive): clean evaluation beats only. Noise from Bank 2 (LibriSpeech test-clean or DEMAND; source chosen
  50/50, then a clip and offset, all from a seed derived from the beat ID; the same noise segment at every level)
  added to the 5 s window at SNR 20, 10, 5, 0 dB relative to the clean window power, plus no noise.
- E3: E1/E2 broken down by CirCor murmur timing, grade, shape; ZCHSound and BMD-HS disease; PASCAL
  extrasystole and synthetic rhythm sets.
- E4: per dataset, the normal-vs-pathological gap (all pathological classes pooled) for E1 and E2 at the matched threshold.
- E5: murmur classifier (B4 architecture) on CirCor non-calibration patients split 60/40 by patient
  (murmur Unknown excluded). Train once unfiltered and once per quality model with training beats filtered at
  that model's matched threshold; test all on the unfiltered 40 %. Patient score = max over beats.
  Report patient-level sensitivity and specificity at 0.5, and AUROC. Validation for early stopping: 15 % of
  training patients.
- E6: (a) B6a vs B6b; (b) E2 rerun for the balanced CNN vs B4.

## Statistics
- 95 % bootstrap CIs over patients (1,000 resamples, seed 42) for every rejection rate and for
  "pathological minus normal" (patients resampled within each class group, independently per class).
- Main test (E2), per model × dataset: GEE logistic regression `rejected ~ C(class, Treatment(normal)) + C(snr_level)`,
  binomial family, exchangeable working correlation, clusters = patients, at the matched threshold.
  Report the odds ratio (95 % CI, p) for each pathology class vs normal.
- Holm correction across models for the main test, applied per dataset × class contrast.
- Primary contrast: CirCor `murmur_systolic` vs `normal`, matched threshold. All others are secondary.

## Changes after seeing results
(none)
- 2026-10-16, after the first analysis run (seen: calibration thresholds and E1/E2 tables). (1) Deep-model scores
  were saturating at exactly 1.0 (float32 sigmoid), and B6 gives exactly 1.0 to every unmatched beat, so for some
  model × dataset pairs more than 10 % of calibration normal beats were tied at the maximum and "reject if score >
  90th percentile" rejected far fewer than 10 %. Fix: deep scores recomputed in float64; the matched threshold is the
  90th percentile (an observed value) and beats exactly at it are rejected with the probability that makes the
  calibration rejection exactly 10 % (uniform number from a hash of the beat ID, identical across noise levels).
  The plan's intent (reject 10 % of clean calibration normal beats) is unchanged; only tie handling was unspecified.
- 2026-10-16: B3 CardioPHON added to every table after the other results were seen (user request). It uses the same thresholds/tests; Holm correction now spans 10 models, which slightly changes adjusted p-values of the other models.
- 2026-10-16 (user decision, option A): when more than 5 % of the calibration beats sit exactly on the matched threshold,
  every tied beat is rejected instead of randomised tie-breaking (affects B6 on ZCHSound and the B6pn variants only;
  calibration rejection 13.6-19.3 % instead of 10 %). Reason: B6 scores every unmatched beat 1.0, and randomised ties capped
  its rejection under noise. Applied to the pooled threshold in the same way.

---

# Pre-registration: B8 "PCGKolect-v3" (proposed filter), written 2026-10-16 BEFORE B8 is trained or scored on any test set

## Motivation (from results already obtained)
* Recording-level expert quality labels carry pathology bias (E0: SQI-0 rate 14.0 % abnormal vs 7.5 % normal), and B1-B5/E6
  learn from them.
* B6 rejects beats its HMM cannot decode confidently, so murmurs, premature/irregular beats and fast infant heart rates
  look "unusable" (E2, fig 18).

## Design (fixed now)
1. Backbone: the PCGKolect 1-D U-Net and front-end (unchanged architecture, 5 frame classes S1/systole/S2/diastole/noise).
2. Training data: PhysioNet 2016 only, same patient-grouped folds as B4/B5: train on folds 1-4, fold 0 held out for validation.
   Bank-1 noise only (ESC-50, Mini LibriSpeech, PhysioNet noise stretches), noise clips from fold-0 records excluded from training.
3. **No recording-level SQI labels are used.** "Noise" is learned only from frame-level (N...N) annotations and from synthetic
   corruption (the repo's corruption recipe: partial / full overlays at -20...25 dB, pure-noise windows).
4. **Rhythm/tempo augmentation** (label-aware, no pitch change): per training crop, diastole durations are rescaled by a
   recording factor f ~ log-uniform(0.35, 1.4) (heart rates up to ~2.5x faster) times a per-beat jitter U(0.7, 1.3), and with
   probability 0.15 a beat's diastole is shortened by a further 25-40 % (premature-like). 10 ms cross-fade at cuts.
5. **Score = mean posterior of the noise class over the frames of the scored beat** (window-level: over the whole window).
   No HMM decoding, no beat matching, no duration or rhythm rule enters the score.
6. Training recipe: repo defaults (AdamW 2e-3, OneCycle, 12,000 steps, batch 64, label smoothing 0.05, spec-augment), seed 42.

## Allowed development (PhysioNet only) and freeze
* Exactly two trainings: B8 (as above) and the ablation B8-noaug (same without step 4). Only PhysioNet fold 0 and synthetic
  rhythm sequences built from PhysioNet fold-0 clean normal records may be inspected. No hyper-parameter search.
* The B8 configuration above is the one that goes to testing regardless of the development results; B8-noaug is reported as an ablation.

## Evaluation (identical to the other filters)
* Matched threshold per dataset (10 % of clean normal calibration beats; option-A tie rule), E1, E2, E3, E4, E5, GEE with and
  without heart-rate/age adjustment. Holm correction across the 11-model family (B1 LR, B1 SVM, B2, B3, B4, B5, E6, B6a, B6b, B7, B8);
  any verdict of the previous 10 models that changes because the family grew is reported.

## Pre-specified success criteria
* C1 (noise detection, PhysioNet): window AUROC on fold 0 >= 0.963 (B2, the fairest existing filter).
* C2 (noise detection, test sets): AUROC clean vs 0 dB of the same beats >= 0.90 on each of the 4 test datasets.
* C3 (fairness, primary): number of heartbeat conditions significantly over-rejected in the adjusted verdict matrix (15 conditions).
* Claims: "B8 removes the diagnosed flaws" if C3 = 0 and C1, C2 hold. "B8 is the best filter overall" only if additionally its C3
  is <= every other filter's and its C1 exceeds every filter with an equal or lower C3. Whatever the outcome, B8's results are
  reported in full; a failure of any criterion is reported as such.
