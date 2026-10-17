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

## B8 outcome and second iteration (disclosed; written before v4 is trained or tested)
* B8 result: C3 = 1 over-rejected condition (CirCor systolic murmur), but C1 failed (PhysioNet fold-0 AUROC 0.773) and C2
  failed (clean vs 0 dB AUROC 0.69 ZCHSound, 0.21 BMD-HS). B8 therefore does not qualify. The C1 failure was visible on
  PhysioNet before testing. Ablation B8-noaug: C1 0.966 but biased on 9 conditions.
* Iteration v4 (motivated by the PhysioNet C1 failure: aggressive diastole compression makes heart sound dense and
  noise-like): two candidates with gentler augmentation, otherwise identical to B8:
  v4a recording factor log-U(0.60, 1.25), jitter U(0.85, 1.15), premature p = 0.10, extra shortening 25-40 %;
  v4b recording factor log-U(0.50, 1.30), jitter U(0.80, 1.20), premature p = 0.15.
* Selection uses PhysioNet fold 0 only: candidates must have C1 >= 0.963; among those, the one with the smallest
  rhythm-shift AUROC (|AUROC - 0.5| of the noise score between clean fold-0 normal crops and the same crops after strong
  premature/irregular augmentation) is chosen; ties -> v4a. The chosen one is called B8v4 and tested once with the B8 criteria
  (C1-C3). This iteration is post hoc (B8's test results were seen) and is reported as such.

## B8v4 outcome and third iteration B8c (disclosed; written before B8c is computed on any test set)
* B8v4 candidates: neither passed C1 (fold-0 AUROC 0.809 v4a, 0.792 v4b); v4a selected by the fallback rule and reported.
* B8c = combination of the noise scores of B8-noaug (a: noise-sensitive, rhythm-biased) and B8 (b: rhythm-robust, weak on
  noise), no retraining. Candidates: geometric mean sqrt(a*b); arithmetic mean (a+b)/2; minimum min(a, b).
* Selection on PhysioNet fold 0 only, same rule as v4: C1 >= 0.963 required; among passing candidates the smallest
  |rhythm-shift AUROC - 0.5| (same 288 crops, same seed); ties -> geometric mean. If none passes C1, the candidate with the
  highest fold-0 AUROC is reported as failing C1. The chosen B8c is tested once with C1-C3 and reported as post hoc.

---

# Post-hoc analyses (2026-10-17): seed robustness and hybrid filters

Both analyses were requested after every result above had been seen; they are **post hoc** and reported as such.
Written before any new test-set score (new seeds, S1+S2-referenced E2, synthetic_regular, hybrids) was computed.

## Shared additions (apply to every model in both analyses)
* **E2s, S1+S2-referenced SNR variant of E2.** Identical to E2 (same clean evaluation beats, same Bank-2 clip and offset per
  beat, same levels 20/10/5/0 dB, same matched thresholds), except the noise is scaled against the power of the clean window
  inside its S1 and S2 segments instead of the whole-window power: P(S1+S2 samples) / P(noise) = SNR. Segments = the
  reference segmentation used by B7 (annotations or the fixed Springer segmenter), every S1/S2 lying fully inside the 5 s
  window; if none, the scored beat's own S1 and S2. Reason: window power includes murmur energy, so at the same window SNR
  a murmur beat receives louder noise relative to its heart sounds than a normal beat. Both variants are reported; E2 stays
  the pre-registered primary.
* **synthetic_regular.** A construction control for the synthetic rhythm sets: the same 300 source recordings and the same
  splicing code, every beat kept at its own length (no premature shift, no RR change); up to 4 inner beats per sequence are
  scored. Built after the two existing kinds with the same generator, so their beats are unchanged (checked bit-for-bit).
  Extra contrasts: synthetic_premature vs synthetic_regular and synthetic_irregular vs synthetic_regular (GEE as for the
  other contrasts). The pre-registered 15 conditions (vs normal) are unchanged.

## Task 1: seed robustness
* Retrain B4 CNN, B5 CRNN and E6 balanced CNN with seeds 43 and 44 (seed 42 = existing models); same data, folds,
  hyper-parameters and early stopping. Refit B2 RF and B3 CardioPHON with seeds 43 and 44: the random_state of their
  random forests / gradient boosting / SVM Platt CV changes; data, MI feature selection (random_state 42) and SVM subsample
  are unchanged. B1, B6, B7 and B8 variants are deterministic or not reseeded and are identical in every seed.
* For each seed s, a full replicate: Phase B metrics, matched thresholds (same calibration patients), E1, E2, E2s, E3, E4,
  and the GEE (unadjusted, + HR + HR², + age) with Holm across the main model family, where each seeded model enters with
  its seed-s fit.
* Report mean ± SD over the 3 seeds for every metric and rejection gap (`results/*_seeds.csv`).
* **Seed-robust verdict.** A cell is "biased against" (or "keeps more") only if (i) it is Holm-significant (p_holm < 0.05)
  with OR > 1 (< 1) in at least 2 of 3 seeds AND (ii) the pooled GEE over the three seeds (same formula + C(seed), clusters =
  patients, observations of all seeds in the patient's cluster; Holm across the same family) is significant in the same
  direction. Otherwise "no significant difference". For deterministic models the pooled fit is their single fit.
* Figures: seed means; intervals are seed-averaged 95 % bootstrap CIs; markers show seed-robust significance. "fair" is
  renamed "no significant difference" everywhere.

## Task 2: hybrid hypothesis, periodicity (Das, B2) + deep model (pre-specified)
Hypothesis: combining B2's periodicity features (noise-sensitive, pathology-blind) with B4/B5 (very noise-sensitive,
pathology-biased) keeps the noise sensitivity of B4/B5 without their bias. Hybrids are built for **both** B4 and B5; none is
chosen using test results. Training uses PhysioNet 2016 only, with the same 5 patient-grouped folds.
* **Percentile scores.** For a test dataset, a model's score is converted to its percentile = mid-rank empirical CDF among
  the dataset's clean normal calibration beats (as recorded; synthetic sets use CirCor's). On PhysioNet (Phase B), the
  reference is the out-of-fold scores of usable windows of normal recordings in the four other folds.
* **H-AND** (B4, B5): score = min(Das percentile, DL percentile); rejects only when both look noisy.
* **H-CASCADE** (B4, B5): score = Das percentile if it is < 0.4 or > 0.9, otherwise the DL percentile (both on the same
  percentile scale, so one matched threshold applies). Cut-offs 0.4 / 0.9 fixed now, not tuned.
* **H-FUSE** (B4, B5): features = the 11 Das features + the deep model's penultimate embedding (128-d: B4 global average
  pool, B5 mean BiLSTM output). Out-of-fold embeddings come from fold-specific deep models (train on 4 folds, early stopping
  on the held-out fold; fold 0 = the main model). Classifier = random forest, 120 trees, max_features='sqrt',
  class_weight='balanced', seed 42, trained on the out-of-fold features of all PhysioNet windows (validation: nested 5-fold,
  RF trained on the other folds' out-of-fold features). Test embeddings come from the main deep model of the same seed.
* Hybrid seed s uses the seed-s deep model(s) and, for H-AND / H-CASCADE, the seed-s B2; the H-FUSE forest keeps seed 42.
* Every hybrid goes through the full benchmark like the other models: Phase B, matched thresholds (same rule, option-A
  ties), E1, E2, E2s, E3, E4, synthetic contrasts vs synthetic_regular, covariate GEE, E5 (filter seeds 42-44 × E5 seeds
  42-44), all with 3 seeds and the seed-robust verdict. Holm: each hybrid is added to the main 11-model family on its own
  (as for B8v4 / B8c), so the existing models' verdicts do not depend on the hybrids.
* **Read-out (pre-specified).** Noise sensitivity = mean rejection of noisy normal evaluation beats at 10 and 5 dB, CirCor +
  ZCHSound, matched threshold (E2), mean over seeds. Bias = number of seed-robust "biased against" cells among the 15
  conditions (adjusted verdict matrix) and mean |log OR| over those conditions (adjusted, pooled over seeds). The hypothesis
  holds for a hybrid if it has 0 seed-robust biased cells and noise sensitivity at least that of B4/B5 minus 5 pp
  (≥ min(B4, B5) − 0.05). If no hybrid meets this, that is reported plainly.

## Change after seeing results (2026-10-17, seed robustness)
* The pooled fit over seeds was first run with an exchangeable working correlation (as every other GEE here). Its pooled odds
  ratios fell outside the range of the three per-seed odds ratios (e.g. B3, ZCHSound PFO, HR + age: per seed 3.44-3.78, pooled
  6.15): stacking three near-identical copies of every beat makes the within-patient correlation ~1, so the exchangeable
  estimator re-weights towards patients and estimates a different quantity. The pooled fit now uses independence working
  correlation (patient-clustered robust SEs; the estimand of the per-seed fits, averaged over seeds). The exchangeable pooled
  OR and p are kept as sensitivity columns (pooled_exch_odds_ratio, pooled_exch_p). Per-seed fits are unchanged (exchangeable,
  as pre-registered). Changed after seeing the first pooled verdicts.
