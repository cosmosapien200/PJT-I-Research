# Results: pathology bias in PCG quality filters

Every number here comes from a CSV in `results/` (file named under each table). Rates are per beat; CIs are 95 %
bootstrap CIs over patients (1,000 resamples). "Matched" = threshold that rejects exactly 10 % of clean normal
calibration beats in that dataset (main analysis, see `ANALYSIS_PLAN.md`). Differences are pathological minus
normal, in percentage points (pp). Models: B1 Springer SQIs (LR, SVM), B2 Das RF, B3 CardioPHON (re-implemented 416-feature set + MI selection + soft-voting ensemble), B4 CNN, B5 CRNN, E6 balanced CNN,
B6a/B6b PCGKolect U-Net+HMM (without/with rhythm check), B7 SNR rule. B3's features had to be re-implemented from the
feature names because its repository has no extraction code, so B3 approximates rather than reproduces CardioPHON.

**Caveats to read first.** B6 was trained on PhysioNet + CirCor *segmentation* labels, so CirCor is not fully unseen
for B6 (ZCHSound, BMD-HS, PASCAL are clean for B6). BMD-HS healthy subjects were recorded in a lab with hospital
noise played back, so BMD-HS E1 numbers are confounded; rely on E2 for BMD-HS. ZCHSound, BMD-HS and PASCAL beats come
from the Springer LR-HSMM segmenter, not from annotations. ZCHSound uses only its curated "clean" subset. B7 uses the
clean-recording segmentation positions under added noise (oracle positions), and B7 resembles the clean-beat rule itself.

## Seed robustness

Retraining B4, B5 and E6 and refitting B2 and B3 with seeds 43 and 44 left PhysioNet AUROC essentially unchanged (B4 0.986 ± 0.002, B5 0.988 ± 0.001, E6 0.980 ± 0.003, B2 0.964 ± 0.000, B3 0.972 ± 0.000). The deep models' class gaps, however, move by several points between seeds: ZCHSound PFO, E2 clean, is +36.7 ± 6.7 pp for B4 and +38.9 ± 7.5 pp for B5. Of the 31 heart-rate/age-adjusted Holm-significant cells of the seed-42 analysis (main models, E2), 24 are seed-robust (significant in ≥ 2 of 3 seeds and in the pooled fit); B5's CirCor systolic murmur (OR 1.31, significant in 1/3 seeds), ZCHSound VSD and BMD-HS AR, E6's ZCHSound ASD, and the "keeps more premature beats" cells of B4, E6 and B8 are not, so B5 is left with one seed-robust bias (BMD-HS MR) and B4, E6 and B2 with none. The seed-robust biases are B6 (6 conditions: diastolic murmur, MR, MS, multi-valve, premature, AF-like), B3 (ASD, PFO, premature, AF-like), B1 (premature, AF-like), B5 (MR), B7 (VSD) and B8 (CirCor systolic murmur); the S1+S2-referenced E2s variant gives the same pattern (21 of 25 seed-42 cells robust; no "biased against" cell differs between E2 and E2s in direction). Files: `results/*_seeds.csv` (mean, SD), `results/*_per_seed.csv`, `results/gee_covariates_seeds.csv` (per-seed and pooled fits, verdict; pooled fit with independence working correlation, see ANALYSIS_PLAN.md).

## Hybrid hypothesis

Six pre-specified hybrids of Das periodicity (B2) with B4 or B5 reached PhysioNet AUROC 0.970-0.977 (5-fold out-of-fold). H-FUSE-B4 (random forest on the 11 Das features + B4's penultimate embedding) matches B4's noise sensitivity (65.4 ± 2.2 % vs 65.5 ± 4.2 % of noisy normal beats rejected at 10 and 5 dB, CirCor + ZCHSound) with no seed-robust "biased against" cell and smaller adjusted class effects (mean |log OR| 0.24 vs 0.35 for B4), so it meets the pre-specified criterion; H-AND-B4 and H-CASCADE-B4 also have no biased cell but lose about 8 pp of noise sensitivity (57 %, still inside the pre-specified B4/B5 − 5 pp margin), and H-FUSE-B5 inherits two of B5's biases (MR, PFO). This is not a strict improvement over B4, which itself has no seed-robust biased cell: the hybrid's gain is smaller adjusted odds ratios, while its raw clean-beat gaps stay at B4's level (RMS 11.4 vs 11.6 pp; H-AND-B4 6.9 pp). Downstream (E5), every hybrid stays within 0.015 AUROC of the unfiltered murmur detector (0.895-0.920 vs 0.909); H-FUSE-B5 keeps only about 70 % of training beats in both classes but still reaches 0.910. Files: `results/noise_vs_bias.csv`, `figures/paper/fig22_noise_vs_bias.png`, verdict matrix `fig12a`.

## Hybrid

Debiased hybrids of Das (B2) and B4, pre-registered and trained on PhysioNet only (branch `hybrid-debias`): HD1 mean and HD2 min of the two percentile scores, HD3 a logistic regression on the Das features + B4 embedding, and HD4 a B4 trunk + Das features network trained with rhythm augmentation and a gradient-reversal adversary against the normal/abnormal label (λ 0.1, 0.3, 1.0), 3 seeds each. No candidate met the PhysioNet-only selection rule (fold-0 pathology and rhythm gap CIs including 0 in ≥ 2 of 3 seeds), so the fallback rule chose **HD4 λ0.1** (smallest summed gap among candidates within 0.01 AUROC of B4; AUROC 0.986 ± 0.001 vs B4 0.986, PhysioNet abnormal-vs-normal gap +11.0 vs +17.1 pp). On the test sets **it meets the pre-specified read-out: B4-level noise sensitivity (66.8 ± 3.9 % vs 65.5 ± 4.2 %, bar ≥ 60.5 %) and 0 seed-robust "biased against" cells among the 15 conditions (E2)**. But it is not less biased than B4, which also has 0: its adjusted class effects are as large (mean |log OR| 0.35, same as B4), its ZCHSound PFO odds ratio is higher (2.21 vs 1.84, Holm-significant in 2 of 3 seeds but not in the pooled fit; seed-robustly biased under E2s, OR 2.59), and unlike B4 it rejects AF-like and premature synthetic beats more often than the matched regular synthetic beats (OR 1.41 and 1.43, seed-robust), so the adversary and rhythm augmentation did not transfer from PhysioNet to the test sets. The other candidates trade noise sensitivity for smaller gaps (HD2 54 %, RMS clean gap 6.7 vs 11.6 pp for B4) or gain nothing; E5 AUROC stays 0.899-0.915 (unfiltered 0.909). Files: `results/hd_selection.csv`, `results/accuracy_vs_bias.csv`, `results/noise_vs_bias.csv`, `figures/paper/fig23_accuracy_vs_bias.png`.

## E2 (decisive): equal added noise, clean beats only

**Main contrast, CirCor systolic murmur minus normal** (`results/e2_equal_noise.csv`, matched threshold):

| model | clean | 20 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|---|
| B1 LR | -3.8 (-5.8, -1.7) | -4.7 (-6.7, -2.4) | -7.5 (-10.4, -4.6) | -9.0 (-12.6, -5.1) | -2.0 (-5.1, +1.5) |
| B1 SVM | excluded on CirCor (fix 1) | | | | |
| B2 RF | -0.2 (-2.3, +2.1) | -0.4 (-2.6, +1.9) | +0.4 (-2.3, +3.4) | +2.7 (-0.8, +6.5) | +2.3 (-0.5, +5.1) |
| B3 CardioPHON | -2.9 (-5.0, -0.6) | -3.0 (-5.2, -0.5) | -5.1 (-7.8, -2.3) | -5.9 (-9.6, -2.2) | -3.2 (-6.7, +0.4) |
| B4 CNN | -2.7 (-5.2, -0.4) | -2.4 (-5.1, +0.2) | +0.9 (-2.9, +4.7) | +0.6 (-2.1, +3.4) | +1.2 (-0.8, +3.3) |
| B5 CRNN | +6.2 (+2.4, +10.4) | +7.1 (+3.3, +11.2) | +9.2 (+4.7, +13.7) | +7.1 (+3.6, +10.7) | +4.6 (+2.3, +7.0) |
| E6 bal. CNN | -1.8 (-4.4, +0.8) | -1.6 (-4.2, +0.9) | -2.4 (-5.7, +1.2) | +0.9 (-1.8, +3.7) | -0.1 (-2.3, +2.1) |
| B6a | +6.8 (+3.4, +10.9) | +5.7 (+2.4, +9.7) | +6.0 (+1.8, +10.7) | +4.3 (+0.7, +8.0) | +0.9 (-0.7, +2.6) |
| B6b | +6.8 (+3.4, +10.8) | +5.3 (+2.0, +9.6) | +5.5 (+1.3, +10.2) | +4.4 (+0.9, +8.1) | +0.9 (-0.8, +2.6) |
| B7 SNR | -2.6 (-5.3, +0.2) | -3.3 (-6.4, -0.2) | -2.9 (-6.5, +1.1) | -2.3 (-5.5, +1.0) | -2.2 (-4.7, +0.4) |

**GEE odds ratios, pathological vs normal, all noise levels pooled** (`results/f_gee_e2.csv`; `rejected ~ class + SNR level`,
exchangeable correlation within patient; * = Holm-adjusted p < 0.05 across the 10 models):

| dataset / class | B1 LR | B1 SVM | B2 RF | B3 CardioPHON | B4 CNN | B5 CRNN | E6 bal. CNN | B6a | B6b | B7 |
|---|---|---|---|---|---|---|---|---|---|---|
| CirCor systolic murmur | 0.75* | excl. | 1.17 | 0.90 | 1.08 | 1.28* | 1.03 | 1.28 | 1.27 | 0.87 |
| CirCor diastolic murmur (5 pts) | 1.21 | excl. | 1.15 | 0.69 | 1.07 | 1.09 | 0.77 | 14.2* | 12.4* | 3.76 |
| CirCor murmur unknown | 3.09* | excl. | 2.85* | 1.20 | 5.94* | 5.08* | 7.17* | 6.88* | 6.67* | 2.47* |
| CirCor synthetic premature | 1.63* | excl. | 0.98 | 1.27* | 0.75* | 0.74* | 0.72* | 14.2* | 21.1* | 0.59* |
| CirCor synthetic irregular | 1.80* | excl. | 1.01 | 1.27* | 0.76* | 0.78* | 0.72* | 3.46* | 3.63* | 0.58* |
| ZCHSound VSD | 1.33 | 1.22 | 0.95 | 1.45 | 1.40 | 2.32* | 1.43 | 1.03 | 1.02 | 1.69* |
| ZCHSound ASD | 0.86 | 0.99 | 1.03 | 2.55* | 1.73* | 2.05* | 2.57* | 1.15 | 1.14 | 0.93 |
| ZCHSound PFO | 1.57 | 2.44* | 1.18 | 4.25* | 3.33* | 6.71* | 5.62* | 2.74* | 2.80* | 1.41 |
| ZCHSound PDA | 1.80 | 2.26 | 1.63 | 2.90 | 1.52 | 2.21* | 2.28 | 1.84 | 1.98 | 2.67 |
| BMD-HS multi-disease | 1.67 | 1.75 | 1.18 | 1.13 | 1.03 | 1.21 | 0.87 | 2.43* | 2.44* | 2.10* |
| BMD-HS MR | 2.08 | 2.57 | 1.02 | 0.92 | 1.49 | 2.94* | 1.18 | 3.58 | 3.58 | 2.21 |
| BMD-HS AR | 0.26* | 0.26* | 0.80 | 0.73 | 0.47 | 0.30* | 0.70 | 1.29 | 1.29 | 1.01 |
| PASCAL murmur | 0.82 | 0.79 | 1.88 | 1.21 | 1.44 | 1.98 | 1.13 | 0.80 | 0.78 | 1.30 |
| PASCAL extrasystole | 0.89 | 0.94 | 0.96 | 1.07 | 0.82 | 0.93 | 0.86 | 0.74 | 0.77 | 1.14 |

CIs and the remaining classes (AS, MS, other-location, no-murmur-abnormal-outcome) are in `results/f_gee_e2.csv`.
Noise levels at which every beat in a contrast was rejected (or none was) were dropped from that fit (no class information).
When the exchangeable fit failed to converge, independence working correlation was used (column `cov_struct`). Every row also
reports the independence-correlation fit (`or_indep`) as a sensitivity check; see Fixes. ZCHSound rows reflect the corrected
(paediatric heart-rate) segmentation, and heart-rate/age-adjusted versions are in `results/fix4_gee_e2_covariates.csv`.

At equal noise, the answer depends on the filter and the dataset; there is no single direction. On CirCor, the
segmentation-based filter (B6) and the CRNN reject clean systolic-murmur beats about 6-7 pp more often than normal beats,
while B1 LR and B3 CardioPHON reject them *less* often (B3: -3 to -6 pp) and B2/B4/E6/B7 show no clear difference. On the
paediatric congenital set (ZCHSound), the deep models and B3 reject CHD beats more often, most strongly for PFO (OR 3.3-6.7
for B3/B4/B5/E6, unadjusted). Part of this is a heart-rate/age effect (PFO patients are mostly infants); after adjustment
B3 and B5 remain significant (see Fixes). B6 and B7 over-reject BMD-HS multi-disease beats (OR ≈ 2.1-2.4). The gaps shrink at
0 dB only because nearly all beats are rejected. Two patterns hold across models: CirCor "murmur unknown" beats are
rejected 2-7× more often at equal noise, and B6 rejects irregular-rhythm beats heavily (synthetic premature: +61 pp
clean B6a, +69 pp B6b), which the window-level models do not.
Figures: `figures/e2_circor.png`, `e2_zch.png`, `e2_bmdhs.png`, `e2_pascal.png`.

## Fixes (follow-up review)

1. **B1 SVM on CirCor.** No pipeline bug: the scaler is fitted on training data inside the sklearn Pipeline and the
   4→2 kHz resampling is correct. The SVM's Platt probability saturates near 0.75, and CirCor's matched threshold (0.72)
   sits at that ceiling, so noise cannot raise its rejection (18 % at 0 dB vs 81-93 % on the other datasets;
   `figures/fix1_b1svm_score_vs_snr_circor.png`). B1 SVM is excluded from all CirCor results. The B1 conclusions on
   CirCor now rest on B1 LR, which shows the same under-rejection of murmur beats (OR 0.75).
2. **Recalibration for BMD-HS and PASCAL** (`results/fix2_*.csv`). The matched thresholds rest on only 4 (BMD-HS) and
   18 (PASCAL) calibration patients, so evaluation clean-normal rejection is 7-32 % rather than 10 %. A threshold pooled over all
   calibration patients is dominated by the domain-shifted datasets and rejects only 0-3 % of BMD-HS/PASCAL clean normal
   beats, so most gaps collapse to ~0 through a floor effect. Under published thresholds the BMD-HS multi-disease
   over-rejection by B1, B6 and B7 persists (+14 to +19 pp, E2 clean), and so do the PASCAL murmur gaps for B2/B3/B4/B5
   (+12 to +21 pp). Treat these two datasets as supportive evidence only.
3. **Heart rate and age.** The audit found a segmentation error: the Springer segmenter capped heart rate at 120 bpm
   and halved infant heart rates in ZCHSound. ZCHSound was re-segmented with a paediatric range (neonates now 136 bpm,
   infants 130, children 103) and all its results were regenerated. PFO patients are mostly infants (73 %, median HR 130 bpm)
   versus mostly children among controls (85 %, 103 bpm) (`results/fix3_hr_age_by_class.csv`). Adjusting for heart rate and age
   (`results/fix4_gee_e2_covariates.csv`) shrinks the ZCHSound PFO odds ratios (B4 3.33 → 1.11, B1 SVM 2.44 → 1.29,
   E6 5.62 → 2.09, all no longer significant). B3 (4.25 → 3.78) and B5 (6.71 → 2.50) remain Holm-significant, and within
   heart-rate bands the PFO gap stays positive for B3/B4/B5/E6 in most bands (`results/fix3_hr_band_gaps.csv`). PDA is no
   longer significant for any model after adjustment. ZCHSound VSD remains over-rejected by B5 (OR 1.66) and B7 (1.65), and
   ASD by B3 (2.23) and E6 (1.63). On CirCor the B5 systolic-murmur effect (1.31) and the B1 LR under-rejection (0.78) survive
   adjustment. Conclusion: the paediatric CHD bias is partly an age/heart-rate effect but does not disappear for B3 and B5.
4. **Statistics.** The covariate GEE (`rejected ~ class + SNR + HR + HR² [+ age]`, exchangeable, patient clusters, Holm
   across the 10 main models) is in `results/fix4_gee_e2_covariates.csv`, and counts per group in `results/fix4_counts.csv`.
   BMD-HS age adjustment is not identifiable because all healthy subjects are 20-29. An independence-correlation fit is
   reported for every row: it agrees in direction except in 30/209 near-null fits, and for the PhysioNet-only B6 on CirCor
   murmurs (exchangeable 0.86, independence 1.33, p = 0.009). The CirCor late-systolic group is a single patient
   (cc85132, 16 murmur beats at PV/TV, grade I/VI; `results/fix4_late_systolic.csv`), so its E3 rows are not interpretable.
5. **E5 redone** (mean over beats, threshold from training patients, 3 seeds): every filter stays within ±0.02 AUROC of
   unfiltered (0.909 ± 0.008). The earlier single-seed B7 drop (0.86) did not replicate (0.914 ± 0.019). No filter clearly
   changes murmur detection.
6. **B6 tie handling (option A, user decision after the follow-up).** B6 gives every unmatched beat the maximum score
   (1.0); on ZCHSound 17.5 % of clean control calibration beats are unmatched, so randomised tie-breaking at the matched
   threshold capped B6's rejection. Now every tied beat is rejected wherever > 5 % of calibration beats tie (B6 on
   ZCHSound, B6pn on CirCor/ZCHSound only; calibration rejection 17.5 % instead of 10 %). B6 then over-rejects ZCHSound
   PFO (OR 2.74, Holm-significant; E2 clean +19 pp) and PDA (OR 1.84, not significant after Holm). After heart-rate and
   age adjustment the PFO effect drops to OR 1.58 (not significant; heart rate alone OR 1.72, significant), so B6's PFO
   bias is mainly a heart-rate effect, consistent with its adult beat-duration limits. The adjusted verdict matrix is unchanged.
7. **B6 retrained on PhysioNet only** (B6pn, `e*_*_b6pn*`). Without CirCor training it matches only 77 % of CirCor
   reference beats (vs 99 %). Its CirCor systolic-murmur over-rejection is essentially unchanged (+6.6 vs +6.8 pp at clean,
   E2; independence OR 1.33). Its premature-beat over-rejection is smaller (+32 vs +61 pp) but still large. So B6's murmur bias
   is not an artefact of having seen CirCor.

## E0: label check (PhysioNet 2016)

| subset | normal SQI-0 rate (n) | abnormal SQI-0 rate (n) |
|---|---|---|
| all | 7.5 % (186/2,488) | 14.0 % (93/665) |
| b | 23.6 % (91/386) | 29.8 % (31/104) |
| e | 4.9 % (91/1,871) | 20.2 % (37/183) |
| a | 0.9 % (1/117) | 5.5 % (16/292) |

`results/e0_label_check.csv`. Experts labelled abnormal recordings as unusable almost twice as often as normal ones, so
the training labels themselves carry a pathology-quality association.

## Phase B: quality models on PhysioNet

| model | eval set | AUROC | bal. acc | sens | spec | EER |
|---|---|---|---|---|---|---|
| B1 LR | 5-fold OOF | 0.847 | 0.767 | 0.761 | 0.773 | 0.233 |
| B1 SVM (not used on CirCor, fix 1) | 5-fold OOF | 0.848 | 0.764 | 0.654 | 0.874 | 0.224 |
| B2 RF | 5-fold OOF | 0.963 | 0.892 | 0.838 | 0.945 | 0.102 |
| B3 CardioPHON (re-impl.) | 5-fold OOF | 0.972 | 0.899 | 0.833 | 0.966 | 0.091 |
| B4 CNN | fold 0 | 0.987 | 0.936 | 0.934 | 0.938 | 0.064 |
| B5 CRNN | fold 0 | 0.987 | 0.935 | 0.912 | 0.957 | 0.061 |
| E6 balanced CNN | fold 0 | 0.982 | 0.915 | 0.863 | 0.968 | 0.075 |
| B6a (not trained on quality; saw PhysioNet) | all windows | 0.924 | 0.839 | 0.727 | 0.951 | 0.141 |
| B6b | all windows | 0.921 | 0.834 | 0.728 | 0.940 | 0.144 |
| B7 (annotation-based) | all windows | 0.805 | 0.730 | 0.661 | 0.800 | n/a |

For reference, CardioPHON reported AUROC 0.983 on its own 8,359-recording corpus. B3's 83 selected features are
listed in `results/b3_feature_mutual_information.csv`; the top-ranked ones are spectral spread/bandwidth, spectral
slope/intercept, LPC/LSF, PSD noise ratio and MFCCs.
`results/b_quality_performance.csv`, per-fold/per-epoch logs in `logs/train/`, curves in `figures/training_curves.png`.
Positive class = unusable; 30,173 windows (21,694 usable, 2,012 unusable, 6,467 augmented unusable).
Published thresholds transfer badly. On clean normal CirCor calibration beats, the 0.5 threshold rejects 80 % (B4),
85 % (B5), 76 % (B3) and 73 % (B2), and B6 at confidence 0.80 rejects 27 % on ZCHSound (`results/c_thresholds.csv`, column
`cal_reject_rate_published`). Matched thresholds are therefore the main analysis.

## Phase D: clean-beat pass rates

| dataset | class | pass rate | minus normal (95 % CI) |
|---|---|---|---|
| CirCor | normal | 69.7 % | — |
| CirCor | systolic murmur | 75.4 % | +5.7 |
| CirCor | diastolic murmur | 55.2 % | -14.5 |
| CirCor | unknown | 49.2 % | -20.5 |
| ZCHSound | control | 70.8 % | — |
| ZCHSound | VSD / ASD / PFO / PDA | 47.6 / 60.9 / 64.0 / 63.5 % | -23.2 / -9.9 / -6.8 / -7.3 |
| BMD-HS | normal | 56.1 % | — |
| BMD-HS | AS / AR / MR / MS / multi | 32.4 / 44.0 / 21.4 / 51.6 / 41.0 % | -23.6 / -12.1 / -34.6 / -4.5 / -15.0 |
| PASCAL | normal / murmur / extrasystole | 46.5 / 36.2 / 54.9 % | — / -10.3 / +8.4 |

`results/d_clean_pass_rates.csv` (with CIs). The objective clean-beat rule itself passes fewer pathological beats in
ZCHSound, BMD-HS and PASCAL murmurs. So any filter correlated with beat SNR will remove pathology there before
model-specific bias even enters. Springer-segmenter failure rates (`results/d_segmentation_failures_*.csv`) were
0 % of recordings for BMD-HS; for PASCAL 3.4 % normal, 1.1 % murmur and 6.5 % extrasystole; for ZCHSound (paediatric
heart-rate range, fix 3) 4.7 % control, 1.7 % ASD, 1.4 % PFO, 7.5 % VSD and 12.5 % PDA.

## E1: raw rejection (as recorded, all evaluation beats)

Pathological minus normal, pp (95 % CI), matched threshold (`results/e1_raw.csv`; published-threshold rows in the same file):

| dataset / class | B1 SVM | B2 RF | B3 CardioPHON | B4 CNN | B5 CRNN | B6a | B6b | B7 |
|---|---|---|---|---|---|---|---|---|
| CirCor systolic murmur | excl. | -0.8 (-3.2, +1.7) | -3.9 (-5.9, -1.7) | -4.8 (-8.0, -1.8) | +3.5 (-0.3, +7.5) | +4.7 (+0.7, +8.8) | +4.8 (+0.8, +8.7) | -6.7 (-11.2, -1.9) |
| CirCor unknown | excl. | +18.7 (+12.3, +25.3) | +11.3 (+7.1, +15.4) | +28.5 (+20.8, +35.4) | +25.4 (+18.3, +32.2) | +28.6 (+21.2, +36.9) | +28.6 (+21.6, +36.8) | +17.3 (+9.0, +25.0) |
| ZCHSound VSD | +9.6 (+4.2, +15.0) | +3.6 (-1.0, +8.0) | +20.8 (+13.7, +27.3) | +11.7 (+5.3, +18.5) | +16.5 (+10.2, +23.3) | +1.8 (-3.3, +7.4) | +1.4 (-3.8, +6.9) | +25.3 (+17.9, +32.5) |
| ZCHSound PFO | +22.8 (+13.5, +32.9) | +9.6 (+2.0, +17.3) | +32.5 (+22.6, +42.9) | +35.5 (+24.7, +46.0) | +48.6 (+38.6, +58.9) | +28.3 (+17.8, +38.7) | +29.0 (+18.6, +39.5) | +14.3 (+3.9, +24.8) |
| BMD-HS multi (confounded) | +21.8 (+10.7, +33.8) | +2.9 (-4.3, +9.3) | +5.1 (-1.4, +11.7) | +7.3 (-5.0, +19.0) | +14.6 (+4.4, +25.2) | +33.5 (+19.1, +47.2) | +33.6 (+19.1, +47.4) | +23.7 (+12.5, +34.6) |
| PASCAL murmur | -6.4 (-20.5, +7.7) | +12.5 (-3.2, +25.9) | -2.9 (-15.8, +10.3) | +11.7 (-4.5, +28.5) | +19.5 (+4.0, +34.8) | +13.2 (-1.0, +27.7) | +13.0 (-1.0, +27.7) | +21.4 (+4.8, +36.6) |
| PASCAL extrasystole | -30.0 (-39.9, -19.0) | -22.1 (-34.0, -8.3) | -22.1 (-33.4, -9.2) | -25.1 (-38.0, -9.7) | -28.2 (-39.5, -14.8) | -16.7 (-26.5, -6.3) | -15.7 (-25.7, -5.3) | -13.6 (-27.7, +2.4) |

As recorded, gaps are larger than under equal noise (e.g. ZCHSound PFO +33 to +49 pp for B3 and the CNNs), because
pathological recordings are also noisier (Phase D). The E1-E2 difference is the part explained by recording
quality rather than by the filter's response to pathology. PASCAL extrasystole recordings are rejected *less* than
normal ones in every model.

## E3: breakdowns

CirCor systolic-murmur beats minus normal by grade, E1, matched threshold (`results/e3_breakdown.csv`, `figures/e3_circor_grade_timing.png`):

| grade (patients) | B1 LR | B4 CNN | B5 CRNN | B6a | B7 |
|---|---|---|---|---|---|
| I/VI (102) | -8.6 (-12, -4) | -2.2 (-7, +2) | +3.6 (-2, +9) | +3.4 (-1, +9) | -8.9 (-15, -3) |
| II/VI (27) | -5.3 (-13, +3) | -6.7 (-12, -0) | -3.0 (-8, +3) | -4.4 (-8, -0) | -6.8 (-14, +1) |
| III/VI (45) | -9.3 (-13, -5) | -6.6 (-11, -2) | +7.2 (+1, +14) | +11.6 (+5, +20) | -4.3 (-12, +4) |

By timing, the over-rejection by B5 and B6 is concentrated in holosystolic murmurs (B5 +7.0, B6a +8.3 pp); early-systolic
murmurs are rejected less than normal by most models. The late-systolic group is a single patient (fix 4) and is not interpreted. Under equal noise (E2), grade III/VI beats are over-rejected by
B5 (+9.6 to +15.7 pp, clean to 5 dB) and B6a (+7.9 to +12.1 pp, clean to 10 dB), while grade I/VI and II/VI gaps are
smaller (B6a ≤ +9.8 pp, B5 ≤ +6.9 pp). For B5/B6 the bias is largest for the loudest murmurs.
B3 behaves like B1 here: it rejects grade III/VI systolic-murmur beats less than normal (-7.1 pp, CI -9.3 to -4.5, E1).
BMD-HS by disease present (E2, clean): B6 +24 to +32 pp, B1 +11 to +25 pp, B7 +15 to +17 pp for AS, AR, MR and MS alike, while
B3 (-3 to +5 pp) and B4/E6 (-13 to +8 pp) show little. PASCAL extrasystole: no over-rejection by any model; synthetic premature and irregular
rhythms are strongly over-rejected only by B6; B1 and B3 over-reject them mildly (B3 OR 1.27), see E2.

## E4: normal-vs-pathological gap by dataset

Pooled pathological classes minus normal, matched threshold, pp (95 % CI) (`results/e4_by_dataset.csv`):

| dataset | condition | B1 SVM | B2 RF | B3 CardioPHON | B4 CNN | B5 CRNN | E6 bal. CNN | B6a | B7 |
|---|---|---|---|---|---|---|---|---|---|
| CirCor | E1 raw | excl. | -1.0 (-3.2, +1.4) | -4.0 (-6.1, -1.7) | -5.0 (-7.8, -1.6) | +3.4 (-0.6, +7.8) | -3.4 (-6.1, -0.4) | +5.4 (+1.4, +9.8) | -5.8 (-10.0, -1.0) |
| CirCor | E2 clean | excl. | -0.4 (-2.5, +2.0) | -2.9 (-4.9, -0.5) | -2.7 (-5.1, -0.1) | +6.2 (+2.6, +10.1) | -1.9 (-4.2, +0.8) | +7.3 (+3.6, +11.8) | -2.1 (-4.7, +0.8) |
| CirCor | E2 10 dB | excl. | +0.4 (-2.4, +3.3) | -5.1 (-7.5, -2.0) | +1.0 (-2.6, +4.9) | +9.1 (+4.7, +13.8) | -2.4 (-5.6, +0.9) | +6.4 (+1.6, +11.7) | -2.5 (-6.6, +1.8) |
| ZCHSound | E1 raw | +12.3 (+8.2, +16.5) | +4.4 (+0.8, +7.8) | +24.1 (+19.0, +29.2) | +18.7 (+13.6, +24.4) | +23.9 (+18.7, +29.4) | +20.7 (+16.2, +25.8) | +9.2 (+4.6, +13.7) | +18.7 (+13.3, +24.3) |
| ZCHSound | E2 clean | +7.1 (+4.1, +10.1) | +3.3 (-0.2, +7.0) | +19.6 (+13.7, +25.3) | +15.5 (+10.2, +21.4) | +19.0 (+14.1, +24.2) | +18.9 (+14.0, +24.0) | +5.2 (+1.2, +9.2) | +8.4 (+4.6, +12.3) |
| ZCHSound | E2 10 dB | +5.6 (+1.2, +10.0) | +0.1 (-3.7, +3.6) | +6.3 (+3.4, +9.0) | +10.4 (+6.9, +14.3) | +8.6 (+6.1, +11.4) | +12.6 (+9.0, +16.6) | +4.6 (+1.0, +8.7) | +9.8 (+4.8, +14.6) |
| BMD-HS | E1 raw (confounded) | +19.6 (+8.9, +30.1) | +3.1 (-4.0, +9.4) | +4.2 (-2.1, +9.8) | +9.1 (-3.3, +20.3) | +14.4 (+3.9, +23.8) | +3.7 (-8.7, +13.4) | +32.5 (+18.3, +45.6) | +20.7 (+9.5, +30.8) |
| BMD-HS | E2 clean | +15.7 (+8.5, +23.1) | +2.6 (-6.2, +9.9) | +1.8 (-3.6, +6.5) | +0.8 (-11.0, +11.5) | +4.5 (-3.0, +11.4) | -4.8 (-15.0, +3.8) | +26.8 (+13.4, +39.5) | +13.6 (+7.1, +19.4) |
| BMD-HS | E2 10 dB | +3.0 (-9.4, +15.4) | -1.4 (-7.1, +4.7) | +2.1 (-1.8, +6.0) | -1.7 (-4.6, +1.4) | -2.1 (-4.9, +0.6) | +0.2 (-2.7, +3.2) | +2.4 (-2.2, +7.2) | +9.4 (+0.5, +17.9) |
| PASCAL | E1 raw | -13.6 (-24.9, -2.5) | +1.9 (-10.1, +13.5) | -8.8 (-19.1, +1.5) | +0.4 (-12.7, +13.0) | +4.9 (-8.1, +17.0) | -4.2 (-16.2, +7.7) | +4.0 (-8.4, +15.1) | +10.7 (-2.9, +23.3) |
| PASCAL | E2 clean | -4.1 (-12.4, +5.1) | +9.7 (-2.5, +22.2) | +3.7 (-7.0, +14.5) | +5.5 (-6.3, +17.1) | +4.2 (-6.8, +14.4) | -1.5 (-12.9, +10.3) | -5.1 (-10.7, +0.0) | +5.3 (-2.2, +13.4) |

The bias is strongly dataset-dependent. It is largest and most consistent in paediatric CHD (ZCHSound), present for
B6/B7/B1 in adult valve disease (BMD-HS), small and model-specific in CirCor, and absent in PASCAL.

## E5: downstream murmur detection (CirCor)

B4-architecture murmur classifier, trained on 60 % of CirCor non-calibration patients (unfiltered, or with beats
rejected by each filter at its matched threshold removed) and tested on the other 40 % (316 patients, 72 with murmur).
Patient score = mean over beats; decision threshold = Youden's J on held-out training patients; mean ± SD over 3 seeds
(`results/e5_downstream.csv`, per seed `results/e5_downstream_seeds.csv`, `figures/e5_downstream.png`). B1 SVM is not used on CirCor (fix 1).

| training filter | murmur beats kept | absent beats kept | AUROC | sensitivity | specificity |
|---|---|---|---|---|---|
| none | 100 % | 100 % | 0.909 ± 0.008 | 0.815 ± 0.042 | 0.888 ± 0.030 |
| B1 LR | 87.5 % | 82.0 % | 0.895 ± 0.029 | 0.815 ± 0.053 | 0.842 ± 0.052 |
| B2 RF | 89.2 % | 86.7 % | 0.924 ± 0.008 | 0.866 ± 0.066 | 0.829 ± 0.089 |
| B3 CardioPHON | 93.3 % | 88.3 % | 0.912 ± 0.011 | 0.815 ± 0.035 | 0.891 ± 0.010 |
| B4 CNN | 87.5 % | 84.1 % | 0.924 ± 0.007 | 0.870 ± 0.042 | 0.828 ± 0.035 |
| B5 CRNN | 80.9 % | 84.1 % | 0.913 ± 0.020 | 0.819 ± 0.064 | 0.855 ± 0.063 |
| E6 bal. CNN | 90.6 % | 85.9 % | 0.887 ± 0.038 | 0.847 ± 0.014 | 0.796 ± 0.111 |
| B6a | 78.8 % | 84.8 % | 0.887 ± 0.024 | 0.755 ± 0.081 | 0.891 ± 0.024 |
| B6b | 78.8 % | 84.9 % | 0.905 ± 0.011 | 0.787 ± 0.063 | 0.863 ± 0.049 |
| B7 | 79.0 % | 76.7 % | 0.914 ± 0.019 | 0.838 ± 0.029 | 0.857 ± 0.011 |
| H-AND-B4 | 88.6 % | 85.7 % | 0.911 ± 0.011 | 0.819 ± 0.070 | 0.855 ± 0.063 |
| H-AND-B5 | 86.7 % | 86.5 % | 0.909 ± 0.029 | 0.826 ± 0.052 | 0.849 ± 0.045 |
| H-CASCADE-B4 | 89.5 % | 86.7 % | 0.920 ± 0.015 | 0.809 ± 0.065 | 0.881 ± 0.057 |
| H-CASCADE-B5 | 86.9 % | 86.8 % | 0.895 ± 0.029 | 0.816 ± 0.056 | 0.831 ± 0.080 |
| H-FUSE-B4 | 88.2 % | 83.4 % | 0.914 ± 0.009 | 0.843 ± 0.054 | 0.847 ± 0.057 |
| H-FUSE-B5 | 69.8 % | 69.9 % | 0.910 ± 0.011 | 0.826 ± 0.096 | 0.832 ± 0.081 |
| HD1 avg | 89.1 % | 86.7 % | 0.906 ± 0.016 | 0.833 ± 0.056 | 0.814 ± 0.105 |
| HD2 min | 88.8 % | 86.3 % | 0.907 ± 0.024 | 0.824 ± 0.069 | 0.872 ± 0.042 |
| HD3 LR | 85.8 % | 84.7 % | 0.899 ± 0.027 | 0.841 ± 0.072 | 0.786 ± 0.126 |
| HD4 λ0.1 (chosen) | 88.0 % | 83.3 % | 0.915 ± 0.014 | 0.785 ± 0.064 | 0.906 ± 0.064 |
| HD4 λ0.3 | 87.9 % | 83.6 % | 0.911 ± 0.016 | 0.806 ± 0.050 | 0.891 ± 0.045 |
| HD4 λ1.0 | 88.2 % | 83.2 % | 0.911 ± 0.023 | 0.810 ± 0.077 | 0.876 ± 0.063 |

B5 and B6 discard murmur training beats at a higher rate than murmur-absent beats, which matches their E1/E2 bias.
Hybrid and HD rows are mean ± SD over 3 filter seeds × 3 E5 seeds. Downstream AUROC stays within about ±0.02 of the unfiltered model for every filter. B6a gives the lowest murmur
sensitivity (0.755 vs 0.815), a difference of about one seed-SD, so filtering did not clearly change murmur detection here.

## E6: mitigation

(a) Rhythm check (`results/e6_mitigation.csv`, E2 tables): B6b and B6a differ by < 1 pp on natural beats in every dataset,
because the check rarely fires inside a 5 s window. On synthetic rhythms it adds rejection: premature beats +69 pp
vs normal for B6b vs +61 pp for B6a (clean), irregular +28 vs +26 pp (E1). The confidence term alone already causes most
of B6's rhythm bias. (b) Balancing usable training windows 50/50 between normal and abnormal PhysioNet recordings (E6 vs B4)
did not remove the gap. In E2 clean it was +18.9 vs +15.5 pp on ZCHSound, -1.9 vs -2.7 pp on CirCor, -4.8 vs +0.8 pp
on BMD-HS and -1.5 vs +5.5 pp on PASCAL. It moved the gap in different directions per dataset with overlapping CIs, at a
small cost in PhysioNet AUROC (0.982 vs 0.987).

## Not done / limitations
- B1 SVM is excluded on CirCor (score ceiling, fix 1); BMD-HS and PASCAL calibration sets are small (4 and 18 normal patients).
- B3 CardioPHON's 416 features are a re-implementation from feature names (the repository has no extraction code), so
  B3 approximates rather than reproduces the published model; there is no as-published arm. PCSCN (no code) and the
  Zabihi ensemble (optional) were not implemented.
- EPHNOGRAM extra negatives not used; ZCHSound noisy subset not used.
- One train/validation split per deep model; seeds 42-44 for B2-B5, E6 and the hybrids (section "Seed robustness"); B1, B6, B7, B8 not reseeded.
- Full list of deviations: `DEVIATIONS.md`.
