#!/usr/bin/env bash
# Deep-model scoring for all test sets (E1 then E2)
cd "$(dirname "$0")"
PY=../.venv/bin/python
for ds in circor synth bmdhs pascal zch; do
  $PY score.py e1 $ds b4_cnn b5_crnn e6_cnn_balanced
  $PY score.py e2 $ds b4_cnn b5_crnn e6_cnn_balanced
done
echo ALLDONE
