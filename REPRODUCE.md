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
for m in b4_cnn b5_crnn e6_cnn_balanced; do $PY deep.py $m; done
$PY score.py pn
for ds in circor synth bmdhs pascal zch; do $PY score.py e1 $ds; $PY score.py e2 $ds; done
$PY leakage.py "before Phase E"
$PY analysis.py                       # thresholds, E1-E4, E6, GEE
$PY e5.py
$PY plots.py
$PY leakage.py "final"
```
