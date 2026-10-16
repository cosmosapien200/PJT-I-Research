#!/usr/bin/env bash
# Re-segment ZCHSound, BMD-HS, PASCAL with the 30-220 bpm segmenter and re-score every model on them
cd "$(dirname "$0")"
PY=../.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for ds in zch bmdhs pascal; do $PY prep_test.py $ds > ../logs/run/prep_${ds}_v2.log 2>&1; done
for ds in zch bmdhs pascal; do
  for exp in e1 e2; do
    $PY score.py $exp $ds b1_lr b1_svm b2_rf b6 b7 b4_cnn b5_crnn e6_cnn_balanced
  done
done
$PY precache_b3.py 10 > ../logs/run/b3_precache_v2.log 2>&1
for ds in zch bmdhs pascal; do $PY score.py e1 $ds b3_cardiophon; $PY score.py e2 $ds b3_cardiophon; done
echo ALLDONE
