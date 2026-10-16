#!/usr/bin/env bash
# B3 end-to-end: PN features -> (train || test-feature cache) -> scoring
cd "$(dirname "$0")"
PY=../.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
[ -f ../data/processed/pn_b3feats.npz ] || $PY cache_b3.py > ../logs/run/b3_feats_pn.log 2>&1
$PY b3_train.py > ../logs/run/b3_train.log 2>&1 &
TP=$!
$PY precache_b3.py 9 > ../logs/run/b3_precache.log 2>&1
wait $TP
for ds in circor synth bmdhs pascal zch; do
  $PY score.py e1 $ds b3_cardiophon
  $PY score.py e2 $ds b3_cardiophon
done
echo ALLDONE
