# Deviations from AGENT_PLAN.md and from the original papers (one line each: what, why)

## Data
- CirCor downloaded file-by-file from the PhysioNet S3 mirror instead of the zip: the physionet.org zip ran at a few kB/s.
- BMD-HS cloned from github.com/sani002/BMD-HS-Dataset: the URL in the plan/paper (mHealthBuet) returns "repository not found"; sani002 is the mirror listed on the lab's publications page.
- PASCAL B taken from the Kaggle mirror (kinguistics/heartbeat-sounds, `set_b`): the original istethoscope.peterjbentley.com host does not respond. Only labelled recordings used (461: 320 normal, 95 murmur, 46 extrasystole; unlabelled test files ignored). The noisy-normal / noisy-murmur sublabels are kept (column `sublabel`).
- PASCAL patient ID = the numeric subject token in the file name (e.g. `127` in `extrastole__127_...`); 165 patients.
- ZCHSound: the official site (zchsound.ncrcch.org.cn) did not respond; the dataset was taken from the user-supplied GitHub LFS mirror (WeiJieOvO/ZCHSound-Dataset). Only the "clean" subset (941 recordings with diagnoses) is used; the 318-recording noisy subset has no diagnosis column.
- ZCHSound has no patient IDs; each recording is treated as one patient (the paper reports ~one recording per participant).
- EPHNOGRAM (optional extra negatives) not used: optional, large download, and the common corpus already has 8.5k unusable windows.
- DEMAND: 16 kHz versions of 17 environments (SCAFE has no 16 kHz file on Zenodo), channel 1 only, cut into 30 s clips.
- LibriSpeech/Mini LibriSpeech utterances capped at 12 s (as in PCGKolect_model/seg/prep_noise.py).
- All noise clips are 20 Hz high-passed and scaled to unit RMS (as in PCGKolect_model/seg/prep_noise.py).
- PhysioNet split: StratifiedGroupKFold (5 folds, seed 42) stratified by abnormal x SQI, grouped by subject for subset b and by recording elsewhere (no subject IDs in a, c, d, e, f): records of one subject in subsets other than b may cross folds.
- PhysioNet noise stretch = from the `(N` marker to the `N)` marker (the repo's prep_seg.py also labels the segment after `N)` as noise).

## Training windows
- Augmented windows (30 % of usable windows + Bank 1 noise at U[-5, 5] dB) are added as extra copies; the clean originals are kept.
- Bank 1 clips are assigned to PhysioNet folds (noise stretches inherit their record's fold, other clips random) and a window is only augmented with a clip of its own fold, so no noise clip appears in both training and validation folds.
- Window audio is stored peak-normalised in float16; every model is gain-invariant (z-score / log-mean / ratio features).
- Log-mels for B4/B5/E6 are computed on the GPU from the cached audio at load time (deterministic) instead of being stored on disk (would be ~10 GB across all test conditions).

## Models
- B1: only the five SQIs listed in the plan (kurtosis, envelope std, max envelope autocorrelation at 0.5-2 s lag, envelope sample entropy at 30 Hz, pyPCG cyclostationary degree of periodicity); no further thesis SQIs. Band-pass 25-400 Hz Butterworth order 4 (zero-phase) is an assumption, not confirmed from the thesis. Envelope autocorrelation computed on the 20 Hz low-passed Hilbert envelope at 50 Hz.
- B1 SVM: trained on a stratified random subsample of 12,000 windows (seed 42) per fold / final model so that RBF-SVM + Platt probabilities stay tractable; LR uses all windows.
- B2: 11 of 17 features implemented: sample entropy of the processed autocorrelation (variant m=2, r=0.2·SD, at 50 Hz; the other 3 variants skipped), kurtosis and variance of the processed autocorrelation, min ratio s2/s1 of singular values (matrix of consecutive autocorrelation segments of length L, minimum over L = 0.4-1.5 s; our interpretation of the patent), signal power (mean square of the 20 Hz low-passed envelope of the z-scored window), correlation with a least-squares fit of 1, 2, 3 rectified sinusoids (harmonics of a fundamental grid-searched over 0.5-3 Hz), log power ratio 0-240 vs 240-1000 Hz, mean and variance of log(0-500 Hz / rest) over 2 s frames with 0.5 s overlap. Decaying-rectified-sinusoid features skipped (under-specified).
- B2: class-weighted single random forest instead of the original 16 balanced models averaged.
- B3 CardioPHON not implemented: the repository (git.lih.lu/vdespotovic/cardiophon) contains only pre-extracted feature CSVs and the classifier script, no feature-extraction code and no trained quality model (plan fallback).
- B4/B5: 20 Hz high-pass = 2nd-order Butterworth (causal), as in the PCGKolect front-end. STFT: 50-sample (25 ms) Hann window zero-padded to 512 points, hop 20 (10 ms). Validation = PhysioNet fold 0 (~20 %, patient-grouped where IDs exist).
- E6 balanced CNN: balancing implemented by weighted sampling with replacement (usable windows from normal and abnormal recordings get equal total sampling weight, unusable windows unchanged) rather than discarding data.
- B6: each 5 s beat window is reflection-padded to the model's 10,328-sample input and decoded on its own, instead of running the streaming scheduler with 20 s history (one code path for E1 and E2; rhythm check therefore sees the ~4-6 beats in the window). Threshold confidence 0.80 (device default) used, README says 0.70.
- B6 window-level score on PhysioNet (Phase B table): 1 − mean beat confidence in the window (B6b: confidence set to 0 for beats failing duration/rhythm); 1 if no beat decoded. B6 was trained on all of PhysioNet, so its PhysioNet numbers are optimistic.
- B7 uses the reference segmentation positions (annotations, or the Springer segmenter run on the clean recording) also under added noise (oracle positions). On PhysioNet, windows without hand-corrected annotations (e.g. inside noise stretches) get score +50, so its Phase B numbers are optimistic. Score = −SNR dB (not in [0, 1]); threshold −14 corresponds to the published 14 dB.

## Test beats
- At most 8 beats per recording (random, seed from record ID) to keep E2 tractable.
- Beat-window and clean-rule SNR computed after a 25-400 Hz band-pass.
- Springer LR-HSMM (pyPCG port, hsmmlearn from GitHub) trained on 400 random SQI-1 PhysioNet records with hand-corrected S1/S2 centres; run at 1 kHz with per-record heart-rate re-estimation; beats with duration outside 0.3-2.0 s are discarded and counted as segmentation failures.
- CirCor patients with murmur Absent and outcome Abnormal (263 patients) form their own group `no_murmur_abnormal_outcome` (not covered by the plan's classes; not merged into normal).
- CirCor `murmur_diastolic` = any diastolic murmur at an audible location (only 5 patients; 4 also have a systolic murmur, whose beats have no murmur-free phase and use the quieter phase in the clean rule).
- Clean-rule murmur phase for diseases: AS/MR systolic, AR/MS diastolic, BMD multi = phases of its diseases; ZCHSound VSD/ASD systolic, PDA both (continuous), PFO unknown; PASCAL murmur unknown (quieter phase used when both or unknown).
- PASCAL: a patient counts as normal (calibration-eligible) only if all their recordings are normal.
- Synthetic rhythm sets: 300 clean normal CirCor evaluation recordings; beats concatenated to 20 s, durations changed only by trimming / extending (reflected tail) the diastole; premature beats shift 25-40 % of the previous RR earlier with a compensatory pause; up to 4 modified beats per sequence are scored. They use CirCor's thresholds and are compared with CirCor normal beats.
- E2 noise: Bank 2 source chosen 50/50 (LibriSpeech vs DEMAND) before picking the clip, otherwise 94 % of clips would be speech.

## Analysis
- Matched threshold ties: deep scores are computed in float64 (float32 sigmoids saturated at exactly 1.0) and beats exactly at the 90th-percentile threshold are rejected with the probability that makes calibration rejection exactly 10 % (deterministic per beat ID). Needed because B6 gives 1.0 to every unmatched beat (e.g. 15 % of ZCHSound calibration beats). Logged in ANALYSIS_PLAN.md.
- GEE: noise levels with no variation in a contrast (e.g. 100 % rejection at 0 dB) are dropped from that fit (complete separation); if the exchangeable fit does not converge, independence working correlation (still patient-clustered robust SEs) is used (18 of 162 fits).
- Bootstrap CIs for differences resample patients independently in the pathological and normal groups; synthetic rhythm beats come from CirCor evaluation patients who also contribute normal beats, so their CIs ignore that pairing.
- E5: early-stopping validation = 15 % of training patients (not specified in plan); sensitivity/specificity at 0.5 on the max-over-beats patient score; patients with murmur Unknown excluded.
- E3 "disease present" breakdown for BMD-HS counts multi-disease patients under each of their diseases (in addition to the class breakdown).
- E4 pools pathological classes as: CirCor systolic + diastolic murmur; ZCHSound ASD+VSD+PFO+PDA; BMD-HS AS+AR+MR+MS+multi; PASCAL murmur+extrasystole (unknown, other-location, no-murmur-abnormal-outcome and synthetic beats excluded).
- B6 at its published threshold rejects 27 % of clean ZCHSound control calibration beats and leaves 15 % of them unmatched (paediatric heart rates); reported as found.
