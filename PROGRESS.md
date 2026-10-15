# Progress   (last updated: 2026-10-16 04:18)
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
- [ ] B3 CardioPHON: skipped (repo has no feature-extraction code)
- [x] B4 CNN (fold 0): 0.987 / 0.936
- [x] B5 CRNN (fold 0): 0.987 / 0.935
- [x] B6a / B6b PCGKolect (all PN windows, seen in training): 0.924 / 0.839, 0.921 / 0.834
- [x] B7 SNR (all PN windows, annotation-based): 0.805 / 0.730
- [x] E6 balanced CNN (fold 0): 0.982 / 0.915
## Experiments (one key number each)
- [x] E0 label check: SQI-0 rate normal 7.5% vs abnormal 14.0%
- [x] E1 raw: ZCHSound pooled CHD minus control, B5 CRNN +27.4 pp (22.7, 32.0)
- [x] E2 equal noise: CirCor systolic murmur minus normal (clean): B6a +6.8 pp, B5 +6.2 pp, B1 SVM -3.5 pp; GEE OR B5 1.28*, B1 SVM 0.67*
- [x] E3 breakdown: grade III/VI murmurs over-rejected by B6a +11.6 pp (E1); synthetic premature beats +61 pp (B6a, E2 clean)
- [x] E4 by dataset: gap largest in ZCHSound (E2 clean +3 to +18 pp), none in PASCAL
- [x] E5 downstream: murmur AUROC unfiltered 0.917, filtered 0.860-0.923
- [x] E6 mitigation: rhythm check +8 pp extra on premature beats; balanced CNN does not close the ZCHSound gap (+18.1 vs +10.7 pp)
- [x] D clean-beat pass rates: ZCHSound CHD 44-51 % vs control 69 %
## Blockers
- leakage check passed (final)
- ZCHSound official site unreachable; used user-supplied GitHub mirror (clean subset)
- B3 CardioPHON skipped: no feature-extraction code in repo
- BMD-HS repo moved: used github.com/sani002/BMD-HS-Dataset
