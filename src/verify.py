"""Extra integrity / leakage audit beyond check_leakage(). Prints one line per check; exits non-zero on failure.

  python verify.py
"""
import os, sys, glob
import numpy as np, pandas as pd
from common import PROC, MAN, RES

SC = os.path.join(PROC, 'scores')
fails = []


def check(name, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ') + name + (f'  [{detail}]' if detail else ''))
    if not ok:
        fails.append(name)


pn = pd.read_csv(os.path.join(MAN, 'physionet2016.csv'))
win = pd.read_csv(os.path.join(PROC, 'pn_windows.csv'))
nb = pd.read_csv(os.path.join(PROC, 'noise_bank1.csv'))
# 1. augmentation noise clip comes from the window's own fold
aug = win[win.aug == 1].merge(nb[['clip_id', 'fold']], left_on='noise_clip', right_on='clip_id', suffixes=('', '_clip'))
check('augmentation clips from same PhysioNet fold', (aug.fold == aug.fold_clip).all() and len(aug) == (win.aug == 1).sum())
# 2. PhysioNet noise stretches used as Bank-1 clips come from records of the same fold as the windows they augment
pnclip = aug[aug.noise_clip.isin(nb[nb.source == 'physionet_noise'].clip_id)]
src_rec = pnclip.noise_clip.map(dict(zip(nb.clip_id, nb.file.str.split(':').str[0])))
check('PN noise clip record fold == window fold', (src_rec.map(dict(zip(pn.record_id, pn.fold))) == pnclip.fold).all())
# 3. deep-model validation = fold 0 only, training = folds 1-4 (no overlap of patients)
check('PN fold-0 patients disjoint from folds 1-4', not (set(win[win.fold == 0].patient_id) & set(win[win.fold != 0].patient_id)))
# 4. test sets: every beat's patient/split agrees with manifest; synthetic beats only from eval patients
for ds in ['circor', 'zch', 'bmdhs', 'pascal']:
    b = pd.read_csv(os.path.join(PROC, f'{ds}_beats.csv'), usecols=['beat_id', 'record_id', 'patient_id', 'split'])
    m = pd.read_csv(os.path.join(MAN, f'{ds}.csv'), usecols=['record_id', 'patient_id', 'split']).set_index('record_id')
    check(f'{ds}: beat patient/split match manifest', (b.patient_id.values == m.loc[b.record_id, 'patient_id'].values).all()
          and (b.split.values == m.loc[b.record_id, 'split'].values).all())
    check(f'{ds}: unique beat ids', b.beat_id.is_unique)
syn = pd.read_csv(os.path.join(PROC, 'synth_beats.csv'), usecols=['patient_id', 'split'])
check('synthetic beats only from CirCor eval patients', (syn.split == 'eval').all())
# 5. clean-beat cut-off derived from calibration normal beats only (recompute)
for ds in ['circor', 'zch', 'bmdhs', 'pascal']:
    b = pd.read_csv(os.path.join(PROC, f'{ds}_beats.csv'), usecols=['cls', 'split', 'patient_normal', 'beat_snr_db', 'clean_thr_db'], low_memory=False)
    cal = b[(b.split == 'calibration') & b.patient_normal.astype(bool) & b.cls.isin(['normal', 'control'])]
    check(f'{ds}: clean cut-off = P30 of calibration normal beats', np.isclose(np.percentile(cal.beat_snr_db, 30), b.clean_thr_db.iloc[0]))
# 6. score arrays aligned with their index files, finite
bad = []
for idx in glob.glob(os.path.join(SC, '*_index.csv')):
    n = len(pd.read_csv(idx, usecols=['beat_id']))
    pre = idx[:-len('index.csv')]
    for f in glob.glob(pre + '*.npy'):
        v = np.load(f)
        if len(v) != n or not np.isfinite(v).all():
            bad.append(os.path.basename(f))
check('all score arrays aligned with index and finite', not bad, ','.join(bad[:5]))
# 7. matched thresholds use only calibration patients: thresholds recomputed from calibration rows reproduce the file
thr = pd.read_csv(os.path.join(RES, 'c_thresholds.csv'))
e1 = pd.read_csv(os.path.join(SC, 'e1_circor_index.csv'))
cal = ((e1.split == 'calibration') & (e1.cls == 'normal') & (e1.clean == 1)).values
s = np.load(os.path.join(SC, 'e1_circor_b4_cnn.npy'))[cal]
check('CirCor B4 matched threshold reproducible from calibration beats',
      np.isclose(np.percentile(s, 90, method='higher'), thr.query('dataset=="circor" and model=="b4_cnn"').matched.iloc[0]))
# 8. E2 noise is identical across SNR levels for a beat and is a Bank-2 clip
e2 = pd.read_csv(os.path.join(SC, 'e2_circor_index.csv'))
check('E2: one noise clip per beat across levels', (e2.groupby('beat_id').noise_clip.nunique() == 1).all())
b2 = set(pd.read_csv(os.path.join(PROC, 'noise_bank2.csv')).clip_id)
check('E2: all noise clips from Bank 2', set(e2.noise_clip) <= b2)
# 9. E5 patients: train/test disjoint, no calibration patients, Unknown excluded
cm = pd.read_csv(os.path.join(MAN, 'circor.csv'))
tr, te = set(cm[cm.e5_split == 'e5_train'].patient_id), set(cm[cm.e5_split == 'e5_test'].patient_id)
check('E5 train/test patients disjoint', not (tr & te))
check('E5 excludes calibration and murmur-Unknown patients',
      not ((tr | te) & set(cm[cm.split == 'calibration'].patient_id)) and not ((tr | te) & set(cm[cm.murmur == 'Unknown'].patient_id)))
# 10. B6 decoded-beat match rate on clean CirCor beats (sanity of frame->sample mapping)
mt = np.load(os.path.join(SC, 'e1_circor_b6_matched.npy'))
check('B6 matches >= 95 % of CirCor reference beats as recorded', mt.mean() >= 0.95, f'{mt.mean():.3f}')
# 11. PhysioNet window labels follow the rules
check('SQI-0 records give only unusable windows', (win[win.sqi == 0].unusable == 1).all())
check('usable windows have no noise overlap and SQI 1', ((win[win.unusable == 0].noise_frac == 0) & (win[win.unusable == 0].sqi == 1)).all())
print('FAILED:' if fails else 'ALL CHECKS PASSED', fails if fails else '')
sys.exit(1 if fails else 0)
