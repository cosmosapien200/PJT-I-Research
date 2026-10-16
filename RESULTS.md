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

## E2 (decisive): equal added noise, clean beats only

**Main contrast, CirCor systolic murmur minus normal** (`results/e2_equal_noise.csv`, matched threshold):

| model | clean | 20 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|---|
| B1 LR | -3.8 (-5.8, -1.7) | -4.7 (-6.7, -2.4) | -7.5 (-10.4, -4.6) | -9.0 (-12.6, -5.1) | -2.0 (-5.1, +1.5) |
| B1 SVM | -3.5 (-5.5, -1.3) | -3.6 (-5.5, -1.5) | -5.6 (-7.8, -3.2) | -7.0 (-9.2, -4.5) | -3.5 (-5.5, -1.6) |
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
| CirCor systolic murmur | 0.75* | 0.67* | 1.17 | 0.90 | 1.08 | 1.28* | 1.03 | 1.28 | 1.27 | 0.87 |
| CirCor diastolic murmur (5 pts) | 1.21 | 0.67 | 1.15 | 0.69 | 1.07 | 1.09 | 0.77 | 14.2* | 12.4* | 3.76 |
| CirCor murmur unknown | 3.09* | 2.18* | 2.85* | 1.20 | 5.94* | 5.08* | 7.17* | 6.88* | 6.67* | 2.47* |
| CirCor synthetic premature | 1.63* | 1.62* | 0.98 | 1.27* | 0.75* | 0.74* | 0.72* | 14.2* | 21.1* | 0.59* |
| CirCor synthetic irregular | 1.80* | 1.63* | 1.01 | 1.27* | 0.76* | 0.78* | 0.72* | 3.46* | 3.63* | 0.58* |
| ZCHSound VSD | 1.43 | 1.23 | 1.12 | 1.51* | 1.52* | 2.45* | 1.42 | 1.11 | 1.07 | 1.98* |
| ZCHSound ASD | 0.56* | 0.72 | 1.03 | 2.91* | 1.53 | 1.60* | 2.44* | 0.95 | 0.96 | 1.02 |
| ZCHSound PFO | 2.57* | 1.90* | 1.38 | 5.31* | 3.61* | 7.05* | 7.06* | 1.45 | 1.45 | 2.89* |
| ZCHSound PDA | 1.58 | 1.52 | 2.19 | 4.42* | 1.57 | 1.81 | 2.50 | 1.45 | 1.15 | 3.87* |
| BMD-HS multi-disease | 1.67 | 1.75 | 1.18 | 1.13 | 1.03 | 1.21 | 0.87 | 2.43* | 2.44* | 2.10* |
| BMD-HS MR | 2.08 | 2.57 | 1.02 | 0.92 | 1.49 | 2.94* | 1.18 | 3.58 | 3.58 | 2.21 |
| BMD-HS AR | 0.26* | 0.26* | 0.80 | 0.73 | 0.47 | 0.30* | 0.70 | 1.29 | 1.29 | 1.01 |
| PASCAL murmur | 0.82 | 0.79 | 1.88 | 1.21 | 1.44 | 1.98 | 1.13 | 0.80 | 0.78 | 1.30 |
| PASCAL extrasystole | 0.89 | 0.94 | 0.96 | 1.07 | 0.82 | 0.93 | 0.86 | 0.74 | 0.77 | 1.14 |

CIs and the remaining classes (AS, MS, other-location, no-murmur-abnormal-outcome) are in `results/f_gee_e2.csv`.
Noise levels at which every beat in a contrast was rejected (or none was) were dropped from that fit (no class information).
18 of 180 fits fell back to independence working correlation after the exchangeable fit failed to converge (column `cov_struct`).

At equal noise, the answer depends on the filter and the dataset; there is no single direction. On CirCor, the
segmentation-based filter (B6) and the CRNN reject clean systolic-murmur beats about 6-7 pp more often than normal beats,
while the Springer SQI models and B3 CardioPHON reject them *less* often (B3: -3 to -6 pp) and B2/B4/E6/B7 show no clear difference. On the
paediatric congenital set (ZCHSound), most models, and especially the deep ones and B3, reject CHD beats more often. B3 is
Holm-significant for all four CHD classes (OR 1.5-5.3), and PFO reaches odds ratios of 3.6-7 for B3/B4/B5/E6. B6 and B7 over-reject BMD-HS multi-disease beats (OR ≈ 2.1-2.4). The gaps shrink at
0 dB only because nearly all beats are rejected. Two patterns hold across models: CirCor "murmur unknown" beats are
rejected 2-7× more often at equal noise, and B6 rejects irregular-rhythm beats heavily (synthetic premature: +61 pp
clean B6a, +69 pp B6b), which the window-level models do not.
Figures: `figures/e2_circor.png`, `e2_zch.png`, `e2_bmdhs.png`, `e2_pascal.png`.

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
| B1 SVM | 5-fold OOF | 0.848 | 0.764 | 0.654 | 0.874 | 0.224 |
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
| ZCHSound | control | 69.2 % | — |
| ZCHSound | VSD / ASD / PFO / PDA | 44.0 / 47.6 / 50.9 / 45.3 % | -25.2 / -21.6 / -18.3 / -23.9 |
| BMD-HS | normal | 56.1 % | — |
| BMD-HS | AS / AR / MR / MS / multi | 32.4 / 44.0 / 21.4 / 51.6 / 41.0 % | -23.6 / -12.1 / -34.6 / -4.5 / -15.0 |
| PASCAL | normal / murmur / extrasystole | 46.5 / 36.2 / 54.9 % | — / -10.3 / +8.4 |

`results/d_clean_pass_rates.csv` (with CIs). The objective clean-beat rule itself passes fewer pathological beats in
ZCHSound, BMD-HS and PASCAL murmurs. So any filter correlated with beat SNR will remove pathology there before
model-specific bias even enters. Springer-segmenter failure rates (`results/d_segmentation_failures_*.csv`) were
0 % of recordings for ZCHSound and BMD-HS; for PASCAL they were 3.4 % normal, 1.1 % murmur and 6.5 % extrasystole.

## E1: raw rejection (as recorded, all evaluation beats)

Pathological minus normal, pp (95 % CI), matched threshold (`results/e1_raw.csv`; published-threshold rows in the same file):

| dataset / class | B1 SVM | B2 RF | B3 CardioPHON | B4 CNN | B5 CRNN | B6a | B6b | B7 |
|---|---|---|---|---|---|---|---|---|
| CirCor systolic murmur | -3.9 (-6.1, -1.6) | -0.8 (-3.2, +1.7) | -3.9 (-5.9, -1.7) | -4.8 (-8.0, -1.8) | +3.5 (-0.3, +7.5) | +4.7 (+0.7, +8.8) | +4.8 (+0.8, +8.7) | -6.7 (-11.2, -1.9) |
| CirCor unknown | +19.8 (+14.1, +25.8) | +18.7 (+12.3, +25.3) | +11.3 (+7.1, +15.4) | +28.5 (+20.8, +35.4) | +25.4 (+18.3, +32.2) | +28.6 (+21.2, +36.9) | +28.6 (+21.6, +36.8) | +17.3 (+9.0, +25.0) |
| ZCHSound VSD | +9.6 (+4.8, +14.5) | +5.8 (+1.6, +10.3) | +22.7 (+16.7, +29.1) | +14.5 (+8.5, +20.6) | +21.7 (+14.8, +27.9) | +8.4 (+4.1, +12.4) | +8.0 (+3.9, +11.7) | +28.7 (+21.6, +36.0) |
| ZCHSound PFO | +16.9 (+9.4, +24.6) | +10.2 (+3.0, +18.3) | +33.8 (+23.9, +44.2) | +37.0 (+27.0, +47.2) | +50.2 (+39.9, +59.6) | +18.6 (+11.6, +25.2) | +18.6 (+11.8, +25.2) | +38.8 (+28.2, +49.9) |
| BMD-HS multi (confounded) | +21.8 (+10.7, +33.8) | +2.9 (-4.3, +9.3) | +5.1 (-1.4, +11.7) | +7.3 (-5.0, +19.0) | +14.6 (+4.4, +25.2) | +33.5 (+19.1, +47.2) | +33.6 (+19.1, +47.4) | +23.7 (+12.5, +34.6) |
| PASCAL murmur | -6.4 (-20.5, +7.7) | +12.5 (-3.2, +25.9) | -2.9 (-15.8, +10.3) | +11.7 (-4.5, +28.5) | +19.5 (+4.0, +34.8) | +13.2 (-1.0, +27.7) | +13.0 (-1.0, +27.7) | +21.4 (+4.8, +36.6) |
| PASCAL extrasystole | -30.0 (-39.9, -19.0) | -22.1 (-34.0, -8.3) | -22.1 (-33.4, -9.2) | -25.1 (-38.0, -9.7) | -28.2 (-39.5, -14.8) | -16.7 (-26.5, -6.3) | -15.7 (-25.7, -5.3) | -13.6 (-27.7, +2.4) |

As recorded, gaps are larger than under equal noise (e.g. ZCHSound PFO +34 to +50 pp for B3 and the CNNs), because
pathological recordings are also noisier (Phase D). The E1-E2 difference is the part explained by recording
quality rather than by the filter's response to pathology. PASCAL extrasystole recordings are rejected *less* than
normal ones in every model.

## E3: breakdowns

CirCor systolic-murmur beats minus normal by grade, E1, matched threshold (`results/e3_breakdown.csv`, `figures/e3_circor_grade_timing.png`):

| grade (patients) | B1 SVM | B4 CNN | B5 CRNN | B6a | B7 |
|---|---|---|---|---|---|
| I/VI (102) | -1.6 (-5, +2) | -2.2 (-7, +2) | +3.6 (-2, +9) | +3.4 (-1, +9) | -8.9 (-15, -3) |
| II/VI (27) | -2.4 (-6, +2) | -6.7 (-12, -0) | -3.0 (-8, +3) | -4.4 (-8, -0) | -6.8 (-14, +1) |
| III/VI (45) | -7.4 (-9, -5) | -6.6 (-11, -2) | +7.2 (+1, +14) | +11.6 (+5, +20) | -4.3 (-12, +4) |

By timing, the over-rejection by B5 and B6 is concentrated in holosystolic murmurs (B5 +7.0, B6a +8.3 pp); early-systolic
murmurs are rejected less than normal by most models. Under equal noise (E2), grade III/VI beats are over-rejected by
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
| CirCor | E1 raw | -4.0 (-6.1, -1.7) | -1.0 (-3.2, +1.4) | -4.0 (-6.1, -1.7) | -5.0 (-7.8, -1.6) | +3.4 (-0.6, +7.8) | -3.4 (-6.1, -0.4) | +5.4 (+1.4, +9.8) | -5.8 (-10.0, -1.0) |
| CirCor | E2 clean | -3.5 (-5.4, -1.5) | -0.4 (-2.5, +2.0) | -2.9 (-4.9, -0.5) | -2.7 (-5.1, -0.1) | +6.2 (+2.6, +10.1) | -1.9 (-4.2, +0.8) | +7.3 (+3.6, +11.8) | -2.1 (-4.7, +0.8) |
| CirCor | E2 10 dB | -5.6 (-7.8, -3.4) | +0.4 (-2.4, +3.3) | -5.1 (-7.5, -2.0) | +1.0 (-2.6, +4.9) | +9.1 (+4.7, +13.8) | -2.4 (-5.6, +0.9) | +6.4 (+1.6, +11.7) | -2.5 (-6.6, +1.8) |
| ZCHSound | E1 raw | +11.2 (+7.9, +15.2) | +6.4 (+3.0, +10.1) | +25.9 (+21.7, +30.7) | +20.7 (+15.9, +25.9) | +27.4 (+22.7, +32.0) | +27.3 (+22.3, +32.7) | +11.1 (+8.0, +14.5) | +30.7 (+25.2, +36.5) |
| ZCHSound | E2 clean | +5.0 (+2.2, +7.7) | +4.9 (+1.0, +8.7) | +21.0 (+15.3, +26.8) | +10.7 (+5.8, +15.5) | +17.7 (+12.8, +22.3) | +18.1 (+12.6, +23.6) | +2.9 (+0.6, +5.5) | +12.0 (+7.8, +16.3) |
| ZCHSound | E2 10 dB | +3.3 (-0.6, +7.4) | -0.0 (-3.7, +3.9) | +6.5 (+3.6, +9.2) | +10.5 (+7.5, +13.9) | +8.9 (+6.3, +11.4) | +12.1 (+9.0, +15.3) | +2.6 (-0.3, +5.4) | +13.8 (+8.5, +18.8) |
| BMD-HS | E1 raw (confounded) | +19.6 (+8.9, +30.1) | +3.1 (-4.0, +9.4) | +4.2 (-2.1, +9.8) | +9.1 (-3.3, +20.3) | +14.4 (+3.9, +23.8) | +3.7 (-8.7, +13.4) | +32.5 (+18.3, +45.6) | +20.7 (+9.5, +30.8) |
| BMD-HS | E2 clean | +15.7 (+8.5, +23.1) | +2.6 (-6.2, +9.9) | +1.8 (-3.6, +6.5) | +0.8 (-11.0, +11.5) | +4.5 (-3.0, +11.4) | -4.8 (-15.0, +3.8) | +26.8 (+13.4, +39.5) | +13.6 (+7.1, +19.4) |
| BMD-HS | E2 10 dB | +3.0 (-9.4, +15.4) | -1.4 (-7.1, +4.7) | +2.1 (-1.8, +6.0) | -1.7 (-4.6, +1.4) | -2.1 (-4.9, +0.6) | +0.2 (-2.7, +3.2) | +2.4 (-2.2, +7.2) | +9.4 (+0.5, +17.9) |
| PASCAL | E1 raw | -13.6 (-24.9, -2.5) | +1.9 (-10.1, +13.5) | -8.8 (-19.1, +1.5) | +0.4 (-12.7, +13.0) | +4.9 (-8.1, +17.0) | -4.2 (-16.2, +7.7) | +4.0 (-8.4, +15.1) | +10.7 (-2.9, +23.3) |
| PASCAL | E2 clean | -4.1 (-12.4, +5.1) | +9.7 (-2.5, +22.2) | +3.7 (-7.0, +14.5) | +5.5 (-6.3, +17.1) | +4.2 (-6.8, +14.4) | -1.5 (-12.9, +10.3) | -5.1 (-10.7, +0.0) | +5.3 (-2.2, +13.4) |

The bias is strongly dataset-dependent. It is largest and most consistent in paediatric CHD (ZCHSound), present for
B6/B7/B1 in adult valve disease (BMD-HS), small and model-specific in CirCor, and absent in PASCAL.

## E5: downstream murmur detection (CirCor)

B4-architecture murmur classifier, trained on 60 % of CirCor non-calibration patients (unfiltered, or with beats
rejected by each filter at its matched threshold removed) and tested on the other 40 % (316 patients, 72 with murmur);
patient score = max over beats (`results/e5_downstream.csv`, `figures/e5_downstream.png`):

| training filter | train beats kept | murmur beats kept | absent beats kept | sens | spec | AUROC |
|---|---|---|---|---|---|---|
| none | 100 % | 100 % | 100 % | 0.986 | 0.230 | 0.917 |
| B1 LR | 83.2 % | 87.5 % | 82.0 % | 0.986 | 0.119 | 0.902 |
| B1 SVM | 88.7 % | 91.8 % | 87.8 % | 0.972 | 0.311 | 0.921 |
| B2 RF | 87.3 % | 89.2 % | 86.7 % | 0.986 | 0.107 | 0.923 |
| B3 CardioPHON | 89.4 % | 93.3 % | 88.3 % | 0.958 | 0.311 | 0.907 |
| B4 CNN | 84.9 % | 87.5 % | 84.1 % | 0.917 | 0.689 | 0.914 |
| B5 CRNN | 83.3 % | 80.9 % | 84.1 % | 0.903 | 0.549 | 0.897 |
| E6 bal. CNN | 86.9 % | 90.6 % | 85.9 % | 0.931 | 0.426 | 0.908 |
| B6a | 83.5 % | 78.8 % | 84.8 % | 0.944 | 0.439 | 0.911 |
| B6b | 83.5 % | 78.8 % | 84.9 % | 0.958 | 0.213 | 0.906 |
| B7 | 77.2 % | 79.0 % | 76.7 % | 0.944 | 0.090 | 0.860 |

B5 and B6 discard murmur training beats at a higher rate than murmur-absent beats (79-81 % vs 84-85 % kept), the
reverse of the other filters, which matches their E1/E2 bias. Downstream AUROC stayed within 0.90-0.92 for every
filter except B7 (0.86), so with one seed and 72 test murmur patients no filter clearly helped or hurt detection.
Sensitivity and specificity at 0.5 vary a lot between runs because the max-over-beats patient score is poorly
calibrated; AUROC is the more stable comparison.

## E6: mitigation

(a) Rhythm check (`results/e6_mitigation.csv`, E2 tables): B6b and B6a differ by < 1 pp on natural beats in every dataset,
because the check rarely fires inside a 5 s window. On synthetic rhythms it adds rejection: premature beats +69 pp
vs normal for B6b vs +61 pp for B6a (clean), irregular +28 vs +26 pp (E1). The confidence term alone already causes most
of B6's rhythm bias. (b) Balancing usable training windows 50/50 between normal and abnormal PhysioNet recordings (E6 vs B4)
did not remove the gap. In E2 clean it was +18.1 vs +10.7 pp on ZCHSound, -1.9 vs -2.7 pp on CirCor, -4.8 vs +0.8 pp
on BMD-HS and -1.5 vs +5.5 pp on PASCAL. It moved the gap in different directions per dataset with overlapping CIs, at a
small cost in PhysioNet AUROC (0.982 vs 0.987).

## Not done / limitations
- B3 CardioPHON's 416 features are a re-implementation from feature names (the repository has no extraction code), so
  B3 approximates rather than reproduces the published model; there is no as-published arm. PCSCN (no code) and the
  Zabihi ensemble (optional) were not implemented.
- EPHNOGRAM extra negatives not used; ZCHSound noisy subset not used.
- One seed per deep model; one train/validation split.
- Full list of deviations: `DEVIATIONS.md`.
