"""check_leakage(): assertions required by AGENT_PLAN Section 4.8.

  python leakage.py [phase-label]
"""
import os, sys, glob
import pandas as pd
from common import MAN, PROC, write_progress_line

TEST = ['circor', 'bmdhs', 'pascal', 'zch']


def check_leakage():
    pn = pd.read_csv(os.path.join(MAN, 'physionet2016.csv'))
    win = pd.read_csv(os.path.join(PROC, 'pn_windows.csv'), usecols=['record_id', 'patient_id', 'fold'])
    train_rec, train_pat = set(pn.record_id) | set(win.record_id), set(pn.patient_id) | set(win.patient_id)
    assert set(win.record_id) <= set(pn.record_id), 'training windows from a record outside the training manifest'
    # 1. no test record / patient in a training manifest
    for ds in TEST:
        p = os.path.join(MAN, f'{ds}.csv')
        if not os.path.exists(p):
            continue
        t = pd.read_csv(p)
        assert not (set(t.record_id) & train_rec), f'{ds}: record in training manifest'
        assert not (set(t.patient_id) & train_pat), f'{ds}: patient in training manifest'
        # 2. no patient in two splits
        assert (t.groupby('patient_id').split.nunique() == 1).all(), f'{ds}: patient in two splits'
        if 'e5_split' in t:
            assert (t.groupby('patient_id').e5_split.nunique() == 1).all(), f'{ds}: patient in two E5 splits'
            assert not (t[t.split == 'calibration'].e5_split != 'none').any(), f'{ds}: calibration patient in E5'
    assert (pn.groupby('patient_id').fold.nunique() == 1).all(), 'PhysioNet patient in two folds'
    assert (win.groupby('patient_id').fold.nunique() == 1).all(), 'PhysioNet window patient in two folds'
    # 3. no calibration patient in any evaluation table
    cal = set()
    for ds in TEST:
        p = os.path.join(MAN, f'{ds}.csv')
        if os.path.exists(p):
            t = pd.read_csv(p)
            cal |= set(t[t.split == 'calibration'].patient_id)
    for p in glob.glob(os.path.join(PROC, 'eval_*.csv')) + glob.glob(os.path.join(os.path.dirname(MAN), 'results', '*.csv')):
        cols = pd.read_csv(p, nrows=0).columns
        if 'patient_id' in cols:
            ids = set(pd.read_csv(p, usecols=['patient_id']).patient_id)
            assert not (ids & cal), f'calibration patient in evaluation table {os.path.basename(p)}'
    # 4. no Bank 2 file in Bank 1
    nb = pd.read_csv(os.path.join(MAN, 'noise_banks.csv'))
    b1, b2 = nb[nb.bank == 'bank1'], nb[nb.bank == 'bank2']
    base = lambda s: {os.path.basename(f.split('#')[0]) for f in s}
    assert not (set(b1.file) & set(b2.file)), 'Bank 2 file in Bank 1'
    assert not (base(b1.file) & base(b2.file)), 'Bank 2 file name in Bank 1'
    spk = lambda s: {os.path.basename(f).split('-')[0] for f in s if f.endswith('.flac')}
    assert not (spk(b1.file) & spk(b2.file)), 'LibriSpeech speaker in both banks'
    return True


if __name__ == '__main__':
    check_leakage()
    label = sys.argv[1] if len(sys.argv) > 1 else ''
    write_progress_line('Blockers', 'leakage check', f'- leakage check passed ({label})')
    print('leakage check passed', label)
