# Progress   (last updated: 2026-10-17 21:28)
## Datasets
- [x] PhysioNet 2016: 3,153 records, 19.7 h, 2 kHz, 2,769 patient groups, 428 noise stretches
- [x] CirCor 1.0.3: 3,163 recordings, 942 patients, 24,049 beats (<=8/recording), clean cut-off 10.1 dB
- [x] ZCHSound (clean subset, user-supplied mirror): 941 recordings, 7,493 beats (Springer HSMM)
- [x] BMD-HS: 872 recordings, 109 patients, 6,931 beats (Springer HSMM)
- [x] PASCAL B (Kaggle mirror): 461 labelled recordings, 165 patients, 2,425 beats (Springer HSMM)
- [x] Synthetic rhythm sets: 1,051 premature + 1,096 irregular beats from 300 CirCor recordings
- [x] Noise banks: Bank1 3,507 clips (ESC-50, MiniLibri, PN noise); Bank2 2,790 clips (LibriSpeech test-clean, DEMAND)
## Models (validation AUROC / balanced accuracy)
- [x] B1 LR (5-fold): 0.848 / 0.767
- [x] B1 SVM (5-fold): 0.849 / 0.765
- [x] B2 RF (5-fold): 0.964 / 0.892
- [x] B3 CardioPHON (re-implemented 416 features, 5-fold): 0.972 / 0.899
- [x] B4 CNN (fold 0): 0.987 / 0.936
- [x] B5 CRNN (fold 0): 0.987 / 0.935
- [x] B6a / B6b PCGKolect (all PN windows, seen in training): 0.924 / 0.839, 0.921 / 0.834
- [x] B7 SNR (all PN windows, annotation-based): 0.805 / 0.730
- [x] E6 balanced CNN (fold 0): 0.982 / 0.915
- [ ] B8 PCGKolect-v3 (proposed): pre-registered in ANALYSIS_PLAN.md, training
- [x] Seeds 42-44 (fold 0 / OOF AUROC mean ± SD): B4 0.986 ± 0.002, B5 0.988 ± 0.001, E6 0.980 ± 0.003, B2 0.964 ± 0.000, B3 0.972 ± 0.000
- [x] Hybrids (5-fold OOF, 3 seeds): H-AND B4/B5 0.974/0.977, H-CASCADE 0.970/0.972, H-FUSE 0.971/0.977
## Experiments (one key number each)
- [x] E0 label check: SQI-0 rate normal 7.5% vs abnormal 14.0%
- [x] E1 raw: ZCHSound pooled CHD minus control, B5 CRNN +27.4 pp (22.7, 32.0)
- [x] E2 equal noise: CirCor systolic murmur minus normal (clean): B6a +6.8 pp, B5 +6.2 pp, B1 SVM -3.5 pp; GEE OR B5 1.28*, B1 SVM 0.67*
- [x] E3 breakdown: grade III/VI murmurs over-rejected by B6a +11.6 pp (E1); synthetic premature beats +61 pp (B6a, E2 clean)
- [x] E4 by dataset: gap largest in ZCHSound (E2 clean +3 to +18 pp), none in PASCAL
- [x] E5 downstream (mean over beats, 3 seeds): AUROC unfiltered 0.909 +- 0.008, filtered 0.887-0.924
- [x] E6 mitigation: rhythm check +8 pp extra on premature beats; balanced CNN does not close the ZCHSound gap (+18.1 vs +10.7 pp)
- [x] D clean-beat pass rates: ZCHSound CHD 44-51 % vs control 69 %
- [x] B3 bias: ZCHSound CHD over-rejected at equal noise (OR 1.5-5.3, all Holm-sig.); CirCor systolic murmur -2.9 pp
- [x] Fix 1: B1 SVM excluded on CirCor (Platt score ceiling 0.75 at matched threshold 0.72; 0 dB rejection 18 %)
- [x] Fix 2: pooled threshold rejects 0-3 % of BMD-HS/PASCAL clean normals; BMD multi B1/B6/B7 gap persists at published thr
- [x] Fix 3: ZCHSound re-segmented (paediatric HR); PFO OR after HR+age: B3 3.78*, B5 2.50*, B4 1.11
- [x] Fix 4: covariate GEE + counts; late-systolic = 1 patient; exch vs indep GEE disagree in 30/209 near-null fits
- [x] Fix 6: B6 retrained on PhysioNet only: CirCor beat match 77 % (vs 99 %), systolic-murmur gap +6.6 pp (vs +6.8)
- [x] Option A tie rule: B6 ZCHSound PFO OR 2.74* unadjusted, 1.58 (n.s.) after HR+age; verdict matrix unchanged
- [x] Seed robustness: 24 of 31 seed-42 adjusted significant cells seed-robust; B4, E6, B2 no robust bias; B5 only MR
- [x] Hybrid hypothesis: H-FUSE-B4 noise sensitivity 65.4 % (B4 65.5 %), 0 seed-robust biased cells, mean |log OR| 0.24 vs 0.35
- [x] E2s (S1+S2-referenced noise): same direction for every seed-robust cell; 21 of 25 seed-42 cells robust
- [ ] E5 hybrids (3 filter x 3 E5 seeds): running
## Blockers
- B3 CardioPHON: repo lacks feature code -> 416 features re-implemented from column names (user request)
- ZCHSound official site unreachable; used user-supplied GitHub mirror (clean subset)
- B3 CardioPHON skipped: no feature-extraction code in repo
- BMD-HS repo moved: used github.com/sani002/BMD-HS-Dataset
- leakage check passed (before hybrid-debias scoring)
- Segmenter HR cap (120 bpm) mis-segmented ZCHSound infants: fixed (paediatric range), ZCHSound rerun
- integrity audit (src/verify.py): all 25 checks passed (after option A)
