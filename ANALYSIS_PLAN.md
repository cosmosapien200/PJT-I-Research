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
