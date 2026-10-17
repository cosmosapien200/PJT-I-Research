# Reproduce

Python 3.12 environment (uv), from the project root:

```bash
uv venv -p 3.12 .venv && uv pip install -p .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -p .venv/bin/python numpy scipy pandas scikit-learn statsmodels librosa soundfile matplotlib pyyaml pyPCG-toolbox pywavelets cython setuptools
uv pip install -p .venv/bin/python --no-build-isolation git+https://github.com/jvkersch/hsmmlearn
```

Data (into `data/raw/`):

```bash
src/download.sh                                   # PhysioNet 2016, ESC-50, LibriSpeech dev-clean-2 + test-clean
.venv/bin/python src/dl_s3.py circor-heart-sound/1.0.3 data/raw/circor
src/dl_demand.sh                                  # DEMAND 16 kHz
git clone https://github.com/cosmosapien200/PCGKolect_model data/raw/repos/PCGKolect_model
git clone https://git.lih.lu/vdespotovic/cardiophon data/raw/repos/cardiophon
git clone https://github.com/sani002/BMD-HS-Dataset data/raw/repos/BMD-HS-Dataset
# PASCAL B: Kaggle kinguistics/heartbeat-sounds -> unzip into data/raw/pascal (set_b/, set_b.csv)
# ZCHSound: https://media.githubusercontent.com/media/WeiJieOvO/ZCHSound-Dataset/refs/heads/main/ZCHSound.zip -> data/raw/zchsound/
```
Extract archives: `physionet2016_training.zip` + `physionet2016_annotations.zip` → `data/raw/physionet2016/`,
`esc50.zip` → `data/raw/esc50/`, both LibriSpeech tarballs → `data/raw/librispeech/`.

Pipeline (run from `src/`):

```bash
cd src
PY=../.venv/bin/python
$PY prep_physionet.py                 # manifest, folds, E0, noise stretches, segments
$PY noise_banks.py bank1
$PY noise_banks.py bank2
$PY make_windows.py                   # common corpus
$PY segmenter.py train                # Springer LR-HSMM for unannotated test sets
for ds in circor bmdhs pascal zch; do $PY prep_test.py $ds; done
$PY prep_test.py synth
$PY leakage.py "end of Phase A"
$PY cache_features.py pn_windows_audio.npy pn
$PY classic.py                        # B1, B2 (5-fold CV + final)
OMP_NUM_THREADS=1 $PY cache_b3.py      # B3 CardioPHON features (416) on the corpus
OMP_NUM_THREADS=1 $PY b3_train.py      # B3 (5-fold CV + final)
for m in b4_cnn b5_crnn e6_cnn_balanced; do $PY deep.py $m; done
$PY score.py pn
for ds in circor synth bmdhs pascal zch; do $PY score.py e1 $ds; $PY score.py e2 $ds; done
$PY leakage.py "before Phase E"
$PY analysis.py                       # thresholds, E1-E4, E6, GEE  (score.py runs B3 by default; export OMP_NUM_THREADS=1)
$PY e5.py                             # E5: mean-over-beats, Youden threshold on held-out training patients, 3 seeds
$PY seg_validate.py                   # fix 3: segmenter heart-rate range check on PhysioNet
$PY analysis_fixes.py                 # fixes 2-4: pooled/published recalibration, HR/age covariates, counts, late-systolic
# fix 6 (optional): PhysioNet-only U-Net, in a copy of PCGKolect_model at data/raw/repos/PCGKolect_pnonly
#   python seg/prep_seg.py --pn ../../physionet2016 --pn-ann ../../physionet2016/annotations/hand_corrected --out data/seg_corpus.npz
#   python seg/prep_noise.py --esc50 ../../esc50 --speech ../../librispeech/LibriSpeech/dev-clean-2 --corpus data/seg_corpus.npz --out data/noise_bank.npz
#   python seg/train_seg.py --final --tag _pnonly
#   then: for ds in circor synth zch bmdhs pascal; do $PY score.py e1 $ds b6pn; $PY score.py e2 $ds b6pn; done; rerun analysis.py
# post hoc (2026-10-17): seed robustness + hybrids (ANALYSIS_PLAN.md, "Post-hoc analyses")
$PY prep_test.py synth                # rebuild synthetic set with synthetic_regular (old beats bit-identical)
./run_seeds_train.sh                  # B4/B5/E6 seeds 43, 44; B4/B5 fold models (folds 1-4) for seeds 42-44
for s in 43 44; do $PY classic.py b2_rf $s; $PY b3_train.py $s; done
$PY precache_post.py                  # B1/B2/B3 feature caches for E2s and the synthetic set
./run_seeds_hybrid.sh                 # scoring (E2s, synth, seeds), hybrid.py pn/test, analysis.py, analysis_fixes.py, seeds.py
for h in h_and_b4 h_and_b5 h_cas_b4 h_cas_b5 h_fuse_b4 h_fuse_b5; do $PY e5.py $h ${h}_s43 ${h}_s44; done
$PY fairness_rms.py
$PY plots.py
$PY summary_figs.py
$PY paper_figs.py
$PY md_tables.py
$PY verify.py                         # integrity / leakage audit
$PY leakage.py "final"
```
