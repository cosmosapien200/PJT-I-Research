#!/usr/bin/env bash
# B8 v4: wait for both candidates, select on PhysioNet fold 0 only (pre-registered rule), score, analyse.
cd "$(dirname "$0")"
PY=../.venv/bin/python
export OMP_NUM_THREADS=2
until [ -f ../logs/train/b8v4a_dev.csv ] && [ -f ../logs/train/b8v4b_dev.csv ]; do sleep 10; done
$PY - <<'EOF'
import pandas as pd, shutil
d = pd.concat([pd.read_csv('../logs/train/b8v4a_dev.csv'), pd.read_csv('../logs/train/b8v4b_dev.csv')])
d['dev_dist'] = (d.rhythm_shift_auroc - 0.5).abs()
ok = d[d.fold0_auroc >= 0.963].sort_values(['dev_dist', 'model'])
pick = ok.model.iloc[0] if len(ok) else d.sort_values('fold0_auroc').model.iloc[-1]
d['selected'] = d.model == pick
d.to_csv('../results/b8v4_selection.csv', index=False)
shutil.copy(f'../data/models/{pick}.pt', '../data/models/b8v4.pt')
shutil.copy(f'../data/processed/val_preds_{pick}.csv', '../data/processed/val_preds_b8v4.csv')
print(d.to_string(index=False)); print('SELECTED', pick, '(C1 passed)' if len(ok) else '(no candidate passed C1)')
EOF
for ds in circor synth zch bmdhs pascal; do $PY score.py e1 $ds b8v4; $PY score.py e2 $ds b8v4; done
$PY analysis.py
$PY analysis_fixes.py
echo ALLDONE
