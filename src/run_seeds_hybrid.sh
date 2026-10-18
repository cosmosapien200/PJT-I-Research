#!/usr/bin/env bash
# Post-hoc pipeline after run_seeds_train.sh / precache_post.py / classic + B3 seed refits:
# scoring (seed 42 + deterministic on E2s and the rebuilt synthetic set; seeds 43/44), hybrids, analyses, E5, figures, audits.
cd "$(dirname "$0")"
PY=../.venv/bin/python
L=../logs/run
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
waitfor() { while ! grep -q ALLDONE "$1" 2>/dev/null; do sleep 60; done; }
waitdone() { while [ ! -f "$L/train_$1.done" ]; do sleep 60; done; }
step() { echo "$(date +%H:%M) $*"; "$@" >> $L/seeds_hybrid_steps.log 2>&1 || { echo "FAILED: $*"; exit 1; }; }

waitfor $L/precache_post.log
waitfor $L/classic_seeds.log
DET="b1_lr b1_svm b2_rf b3_cardiophon b4_cnn b5_crnn e6_cnn_balanced b6 b6pn b7 b8 b8noaug b8v4"
for exp in e1 e2 e2s; do step $PY score.py $exp synth $DET; done
for ds in circor zch bmdhs pascal; do step $PY score.py e2s $ds $DET; done
for ds in circor zch bmdhs pascal; do for exp in e1 e2; do step $PY score.py $exp $ds b4_cnn b5_crnn; done; done
for s in 43 44; do
  for m in b4_cnn b5_crnn e6_cnn_balanced; do waitdone ${m}_$s; done
  for ds in circor synth zch bmdhs pascal; do for exp in e1 e2 e2s; do
    step $PY score.py $exp $ds b2_rf_s$s b3_cardiophon_s$s b4_cnn_s$s b5_crnn_s$s e6_cnn_balanced_s$s
  done; done
done
step $PY -c "
import glob, numpy as np
for fa in glob.glob('../data/processed/scores/*_b8noaug.npy'):
    np.save(fa.replace('_b8noaug.npy', '_b8c.npy'), np.minimum(np.load(fa), np.load(fa.replace('_b8noaug.npy', '_b8.npy'))))"
waitfor $L/seeds_train.log
step $PY hybrid.py pn
step $PY hybrid.py test
step $PY leakage.py "before seed/hybrid analysis"
step $PY analysis.py
step $PY analysis_fixes.py
step $PY seeds.py
echo ALLDONE
