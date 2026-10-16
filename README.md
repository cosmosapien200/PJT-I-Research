# Do heart-sound quality filters throw away the sick heartbeats?

**Pathology bias in phonocardiogram (PCG) quality filters**: an implementation and evaluation of 7 published or
baseline quality filters on 4 independent heart-sound datasets plus synthetic irregular rhythms.

> Heart-sound research pipelines automatically discard recordings or beats that look "too noisy".
> Murmurs, congenital defects and irregular rhythms can look like noise.
> **At the same noise level, do these filters reject pathological beats more often than normal ones?**

---

## TL;DR

| | |
|---|---|
| **Answer** | **Sometimes, and it depends on the filter.** Of 7 filters tested on 15 heartbeat conditions, one is fair everywhere and one is biased against 6 conditions. |
| **Worst case** | The segmentation-based filter (**B6**, U-Net + HMM) rejects **premature beats 61 percentage points more often** than normal beats, and over-rejects valve disease. |
| **Children with congenital heart disease** | Deep and CardioPHON-style filters (**B5, B3**) over-reject defects such as PFO (odds ratio 2.5–3.8). This survives correction for the children's faster heart rate and younger age. |
| **Fairest filter** | **B2** (Das et al. features + random forest) detects noise well but is blind to pathology: no bias on any of the 15 conditions. |
| **Before any model** | The expert quality labels the filters learn from already mark abnormal recordings unusable twice as often (14.0 % vs 7.5 %). |
| **Practical takeaway** | Check a quality filter's rejection rate **per diagnostic class, on clean beats, at a locally calibrated threshold** before using it for pathology research. |

---

## The result in one picture

**Is each filter biased against each type of heartbeat?** (equal noise conditions; corrected for heart rate and age)

![verdict matrix](figures/paper/fig12a_verdict_matrix.png)

**How to read it:** each cell compares how often the filter rejects that kind of heartbeat with how often it rejects
normal heartbeats from the same dataset, under identical conditions.
- 🟥 **Removes more:** the filter throws this pathology away significantly more often, so useful data is lost.
- ⬜ **Fair:** no significant difference.
- 🟦 **Keeps more:** the filter rejects this pathology *less* often than normal beats.

OR = odds ratio of rejection vs normal (GEE logistic regression over all noise levels, clustered by patient,
Holm-corrected across models).

### Scorecard

![scorecard](figures/paper/fig12b_scorecard.png)

| Rank | Filter | Biased against (significant after heart-rate/age correction) |
|---|---|---|
| 1 (fairest) | **B2** Das et al. features + random forest | none of 15 |
| 2 | **B4** CNN (log-mel) | none (keeps *more* irregular-rhythm beats) |
| 3 | **B7** SNR rule (Giordano et al.) | VSD |
| 4 | **B1** Springer SQIs + logistic regression | premature, AF-like rhythm (keeps more systolic murmurs) |
| 5 | **B3** CardioPHON (416 features + ensemble) | ASD, PFO, premature, AF-like |
| 5 | **B5** CRNN (CNN + BiLSTM) | systolic murmur, VSD, PFO, MR |
| 7 | **B6** PCGKolect U-Net + duration HMM | premature (OR 19.8), AF-like, diastolic murmur, MR, MS, multi-valve |

---

## Key findings

### 1. How big are the gaps? (clean beats, identical conditions)

![gap heatmap](figures/paper/fig09a_e2_clean_gap_heatmap.png)

Each number is the **extra rejection rate (percentage points) for that pathology compared with normal beats**, on beats
that are equally clean. Red = removes the pathology more often. Bold / "CI≠0" = the 95 % patient-bootstrap
confidence interval excludes zero.

* **Irregular rhythm** splits the filters. Filters that rely on beat segmentation or rhythm periodicity (B6, B1, B3)
  reject premature and AF-like beats more often. Window-level deep classifiers (B4, B5) do not.
* **Congenital heart disease** (ZCHSound): B3, B4 and B5 reject PFO beats +29 to +40 points more often.
* **Valve disease** (BMD-HS): B6 rejects AS, MR and multi-valve beats +29 to +38 points more often.
* **CirCor murmurs:** small effects in both directions. B5/B6 over-reject (+6 to +7), while B1/B3 under-reject (−3 to −4).

### 2. Why some filters are biased and others are not

![noise vs pathology](figures/paper/fig07_noise_vs_pathology_sensitivity.png)

**Left:** every filter detects added noise well (AUROC 0.83–1.00). **Right:** a fair filter cannot tell pathological
from normal *clean* beats (AUROC ≈ 0.5). Red cells mean the filter scores that pathology as "lower quality", and
that is exactly where the bias appears. B2 stays near 0.5 because its random forest needs both a spectral change and
a loss of rhythmic periodicity. Murmurs add energy inside a regular heartbeat; noise destroys the rhythm.

### 3. Is it just heart rate and age? (Partly.)

![confound](figures/paper/fig13_heart_rate_age_confound.png)

PFO patients are mostly infants (78 % neonates or infants vs 12 % of controls) with faster heart rates (median 130 vs
103 bpm). Correcting for heart rate and age removes the PFO effect for the CNN (OR 3.33 → 1.11) and
for B6 (OR 2.74 → 1.58), but it **survives for B3 (OR 3.78) and B5 (OR 2.50)**. The gap also stays positive *within* heart-rate bands.

### 4. How the bias behaves as noise increases

![gap vs SNR](figures/paper/fig10_e2_gap_vs_snr.png)

Gaps are largest on clean and mildly noisy beats. They shrink at 0 dB only because nearly every beat is rejected
there, whatever its class.

### 5. Effect sizes with uncertainty

![forest](figures/paper/fig11_forest_gee_odds_ratios.png)

Grey: unadjusted. Black: adjusted for heart rate (+ age). Red/blue: significant after Holm correction.

### 6. Other results

| Experiment | Result | Figure |
|---|---|---|
| **E0** Label check | Experts labelled 14.0 % of abnormal vs 7.5 % of normal PhysioNet recordings unusable | [fig03](figures/paper/fig03_e0_label_check.png) |
| Threshold transfer | Published 0.5 thresholds reject 73–82 % of clean normal CirCor beats for B2–B5, so thresholds must be calibrated per dataset | [fig05](figures/paper/fig05_threshold_transfer.png) |
| **E3** Murmur grade/timing | B5/B6 over-reject loud (grade III/VI) and holosystolic murmurs most | [fig14](figures/paper/fig14_e3_murmur_grade_timing.png) |
| **E5** Downstream | Training a murmur detector on filtered data changed AUROC by at most ±0.02 (unfiltered 0.909 ± 0.008, 3 seeds) | [fig15](figures/paper/fig15_e5_downstream.png) |
| **E6** Mitigation | B6's explicit rhythm check adds only ≈ 8 points of its 61-point premature-beat bias; balancing the training data did **not** remove the bias | [fig16](figures/paper/fig16_e6_mitigation_and_b6_retrain.png) |
| B6 retrained without CirCor | Murmur bias unchanged (+6.6 vs +6.8 points), so it is not an artefact of having seen CirCor | [fig16](figures/paper/fig16_e6_mitigation_and_b6_retrain.png) |

---

## How the study works

![pipeline](figures/paper/fig01_pipeline.png)

1. **Train** every quality filter only on **PhysioNet/CinC 2016**: 3,153 recordings, expert quality labels and
   marked noise stretches, cut into 30,173 five-second windows, with 5 patient-grouped folds.
2. **Test** on datasets no filter was trained on. Each recording is cut into heartbeats (S1 onset to next S1 onset),
   and each beat is scored in a 5 s window centred on it.
3. **Calibrate** each filter separately on each dataset: 20 % of normal patients are held out, and the threshold
   is set to reject exactly 10 % of their clean beats. These patients are never evaluated.
4. **Compare fairly (E2, the decisive experiment):** take only beats that pass an objective cleanliness rule. Add
   *the same* unseen noise clip to every beat (LibriSpeech test-clean or DEMAND) at 20, 10, 5 and 0 dB. Compare
   rejection rates of pathological vs normal beats.
5. **Statistics:** patient-level bootstrap CIs; GEE logistic regression clustered by patient, adjusted for noise
   level, heart rate and age; Holm correction across models.

![E2 protocol](figures/paper/fig06_e2_protocol_example.png)

### Data

| Dataset | Role | Population | Size used | Classes |
|---|---|---|---|---|
| [PhysioNet/CinC 2016](https://physionet.org/content/challenge-2016/1.0.0/) | training only | mixed | 3,153 recordings | usable / unusable |
| [CirCor DigiScope 1.0.3](https://physionet.org/content/circor-heart-sound/1.0.3/) | main test | children | 942 patients, 24k beats | normal, systolic/diastolic murmur, ... |
| [ZCHSound](https://doi.org/10.1109/tbme.2023.3348800) (clean subset) | test | children 0–14 y | 941 patients | control, ASD, VSD, PFO, PDA |
| [BMD-HS](https://github.com/sani002/BMD-HS-Dataset) | test (supportive) | adults | 109 patients | normal, AS, AR, MR, MS, multi-valve |
| [PASCAL B](http://www.peterjbentley.com/heartchallenge/) | test (supportive) | adults | 165 patients | normal, murmur, extrasystole |
| Synthetic rhythm | test | built from clean normal CirCor beats | 2,147 beats | premature, AF-like irregular |
| ESC-50, Mini LibriSpeech | training noise | | | |
| LibriSpeech test-clean, DEMAND | **unseen** test noise | | | |

### Filters

| ID | Filter | Type |
|---|---|---|
| B1 | Springer et al. signal-quality indices + logistic regression / SVM | hand-crafted |
| B2 | Das et al. autocorrelation + spectral features + random forest | hand-crafted |
| B3 | CardioPHON (Despotovic et al.): 416 features, mutual-information selection, SVM + RF + GB soft vote (features re-implemented) | hand-crafted ensemble |
| B4 | CNN on log-mel spectrogram | deep |
| B5 | CRNN (CNN + BiLSTM) | deep |
| E6 | B4 trained with normal/abnormal-balanced usable data (mitigation) | deep |
| B6 | PCGKolect 1-D U-Net + duration-constrained HMM (released checkpoint) | deep segmentation |
| B7 | SNR rule (Giordano et al.) | rule |

---

## Caveats

* BMD-HS and PASCAL have very small calibration sets (4 and 18 normal patients), and BMD-HS healthy subjects were
  recorded in a lab, so treat those columns as supporting evidence only.
* B3's 416 features are re-implemented from the published feature names; the original extraction code was not released.
* ZCHSound, BMD-HS and PASCAL beats come from an automatic segmenter (Springer LR-HSMM, with a paediatric heart-rate
  range for ZCHSound).
* The CirCor diastolic-murmur group has only 5 patients.
* **B6 on ZCHSound:** B6 gives every beat it cannot match the maximum score, and 17.5 % of clean control beats are
  unmatched. All tied beats are therefore rejected there (calibration rejection 17.5 % instead of 10 %). B6 then
  over-rejects PFO (OR 2.74), but this disappears after heart-rate/age correction: its adult beat-duration limits fail
  on fast infant heartbeats ([fig18](figures/paper/fig18_sensitivity_b6_ties_zchsound.png)).

Every deviation from the original plan is logged in [`DEVIATIONS.md`](DEVIATIONS.md).

---

## Documents

| File | What's inside |
|---|---|
| [`PROJECT_REPORT.md`](PROJECT_REPORT.md) | **Full report:** exact methodology for every step, all results, corrections, limitations |
| [`RESULTS.md`](RESULTS.md) | Compact results tables, plus the "Fixes" section from the follow-up review |
| [`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md) | Pre-specified thresholds, metrics and tests, with every later change logged |
| [`DEVIATIONS.md`](DEVIATIONS.md) | Every deviation from the plan or the original papers, one line each |
| [`REPRODUCE.md`](REPRODUCE.md) | Commands to rerun everything from the raw data |
| [`AGENT_PLAN.md`](AGENT_PLAN.md) | The original study plan |
| [`figures/paper/`](figures/paper/) | 20 paper-ready figures (PNG + PDF) |
| [`results/`](results/) | Every CSV behind every number |

## Repository layout

```
src/            all code: data prep, features, models, scoring, analysis, figures, leakage checks
configs/        one YAML per model
manifests/      record / patient / split lists per dataset, noise-bank file lists
results/        CSV outputs per experiment
logs/train/     per-epoch and per-fold training logs
figures/paper/  paper figure set
data/models/    trained classic models (B1, B2, B3) and the segmenter
```

Raw datasets and processed audio are not included (licences and size); `REPRODUCE.md` downloads and rebuilds them.

## Reproduce

```bash
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -p .venv/bin/python numpy scipy pandas scikit-learn statsmodels librosa soundfile matplotlib pyyaml pyPCG-toolbox pywavelets cython setuptools
uv pip install -p .venv/bin/python --no-build-isolation git+https://github.com/jvkersch/hsmmlearn
```

Then follow [`REPRODUCE.md`](REPRODUCE.md). The full pipeline runs in a few hours on one RTX 3060 (12 GB) and a 10-core
CPU. Integrity is checked by `src/leakage.py` (leakage asserts) and `src/verify.py` (25 checks).
