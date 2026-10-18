#!/usr/bin/env bash
# Post-hoc scoring of existing (seed-42 / deterministic) models: E2s on all test sets, E1/E2/E2s on the rebuilt synthetic
# set (now with synthetic_regular), and B4/B5 seed-42 embeddings for E1/E2 (scores kept). Seeded models: run_score_seeds.sh
cd "$(dirname "$0")"
PY=../.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ALLM="b1_lr b1_svm b2_rf b3_cardiophon b4_cnn b5_crnn e6_cnn_balanced b6 b6pn b7 b8 b8noaug b8v4"
for exp in e1 e2 e2s; do $PY score.py $exp synth $ALLM; done
for ds in circor zch bmdhs pascal; do $PY score.py e2s $ds $ALLM; done
for ds in circor zch bmdhs pascal; do for exp in e1 e2; do $PY score.py $exp $ds b4_cnn b5_crnn; done; done
# B8c = min(B8-noaug, B8) (selected on PhysioNet fold 0, see b8c_select.py)
$PY -c "
import glob, numpy as np
for fa in glob.glob('../data/processed/scores/*_b8noaug.npy'):
    np.save(fa.replace('_b8noaug.npy', '_b8c.npy'), np.minimum(np.load(fa), np.load(fa.replace('_b8noaug.npy', '_b8.npy'))))
print('b8c written')"
echo ALLDONE
