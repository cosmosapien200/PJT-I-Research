#!/usr/bin/env bash
# Non-deep model scoring for all test sets (E1 then E2) + PhysioNet B6/B7
cd "$(dirname "$0")"
PY=../.venv/bin/python
$PY score.py pn
for ds in circor synth bmdhs pascal zch; do
  $PY score.py e1 $ds b1_lr b1_svm b2_rf b6 b7
  $PY score.py e2 $ds b1_lr b1_svm b2_rf b6 b7
done
echo ALLDONE
