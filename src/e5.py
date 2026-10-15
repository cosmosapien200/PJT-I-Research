"""E5: downstream effect of quality filtering on a murmur classifier (CirCor).

B4 CNN architecture, target = patient murmur Present (1) / Absent (0); Unknown excluded.
CirCor non-calibration patients split 60/40 by patient (manifests/circor.csv: e5_split).
Train once unfiltered and once per quality model with training beats rejected at that model's matched
threshold removed. Early stopping on 15 % of training patients. Test on the unfiltered 40 %.
Patient score = max over beats; sensitivity / specificity at 0.5, AUROC.
"""
import os
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
from common import PROC, MAN, RES, LOGT, SEED, stable_seed
import deep
from analysis import load, MODELS, add_rejections

THR = os.path.join(RES, 'c_thresholds.csv')


def main():
    e1 = load('e1', 'circor')
    man = pd.read_csv(os.path.join(MAN, 'circor.csv'))[['record_id', 'e5_split', 'murmur_present']]
    b = e1.merge(man, on='record_id')
    thr_all = pd.read_csv(THR)
    thr = thr_all[thr_all.dataset == 'circor'].set_index('model').matched
    b = add_rejections(b, thr_all, [m for m in MODELS if m in b and m in thr.index])
    W = np.load(os.path.join(PROC, 'circor_beat_windows.npy'), mmap_mode='r')
    use = b.e5_split.isin(['e5_train', 'e5_test']).values
    sel = np.flatnonzero(use)
    X = deep.prep_audio(W[sel])
    b = b.iloc[sel].reset_index(drop=True)
    y = b.murmur_present.values
    test = np.flatnonzero(b.e5_split.values == 'e5_test')
    train_all = np.flatnonzero(b.e5_split.values == 'e5_train')
    tr_pat = np.array(sorted(b.patient_id.values[train_all].astype(str)))
    tr_pat = np.unique(tr_pat)
    rng = np.random.default_rng(SEED)
    val_pat = set(rng.choice(tr_pat, int(round(0.15 * len(tr_pat))), replace=False))
    is_val = b.patient_id.isin(val_pat).values
    rows = []
    variants = ['unfiltered'] + [m for m in MODELS if m in b and m in thr.index]
    for v in variants:
        keep = np.ones(len(b), bool) if v == 'unfiltered' else (b[f'{v}__matched'].values == 0)
        tr = np.flatnonzero((b.e5_split.values == 'e5_train') & ~is_val & keep)
        va = np.flatnonzero((b.e5_split.values == 'e5_train') & is_val & keep)
        model = deep.train_model(f'e5_{v}', X, y, tr, va, deep.CNN, log_path=os.path.join(LOGT, f'e5_murmur_{v}.csv'))
        p = 1 / (1 + np.exp(-deep.predict_logits(model, X[test])))
        pt = pd.DataFrame(dict(patient_id=b.patient_id.values[test], y=y[test], p=p)).groupby('patient_id').agg(y=('y', 'first'), p=('p', 'max'))
        pr = pt.p >= 0.5
        n_tr_all = int(((b.e5_split.values == 'e5_train')).sum())
        rows.append(dict(filter_model=v, n_train_beats=len(tr) + len(va), frac_train_beats_kept=(len(tr) + len(va)) / n_tr_all,
                         frac_train_murmur_beats_kept=keep[train_all][y[train_all] == 1].mean(),
                         frac_train_absent_beats_kept=keep[train_all][y[train_all] == 0].mean(),
                         n_test_patients=len(pt), n_test_murmur_patients=int(pt.y.sum()),
                         sensitivity=pr[pt.y == 1].mean(), specificity=1 - pr[pt.y == 0].mean(),
                         auroc=roc_auc_score(pt.y, pt.p)))
        print(rows[-1], flush=True)
        del model; torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(os.path.join(RES, 'e5_downstream.csv'), index=False)


if __name__ == '__main__':
    main()
