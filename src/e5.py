"""E5: downstream effect of quality filtering on a murmur classifier (CirCor), revised protocol.

B4 CNN architecture, target = patient murmur Present (1) / Absent (0); Unknown excluded.
CirCor non-calibration patients split 60/40 by patient (manifests/circor.csv: e5_split).
Train once unfiltered and once per quality model with training beats rejected at that model's matched
threshold removed. 15 % of training patients (fixed, seeded) are held out from fitting: they drive early stopping
and the choice of the decision threshold (Youden's J on their patient scores). Test on the unfiltered 40 %.
Patient score = mean over beats. 3 seeds per variant (network init / batch order); report mean +- SD.
Outputs: results/e5_downstream_seeds.csv (one row per variant x seed), results/e5_downstream.csv (mean, SD).
"""
import os, sys
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, roc_curve
from common import PROC, MAN, RES, LOGT, SEED
import deep
from analysis import load, MODELS, add_rejections, EXCLUDE

THR = os.path.join(RES, 'c_thresholds.csv')
SEEDS = [SEED, SEED + 1, SEED + 2]


def patient_scores(b, idx, p):
    return pd.DataFrame(dict(patient_id=b.patient_id.values[idx], y=b.murmur_present.values[idx], p=p)) \
        .groupby('patient_id').agg(y=('y', 'first'), p=('p', 'mean'))


def youden(y, p):
    fpr, tpr, t = roc_curve(y, p)
    return float(t[np.argmax(tpr - fpr)])


def main():
    e1 = load('e1', 'circor')
    man = pd.read_csv(os.path.join(MAN, 'circor.csv'))[['record_id', 'e5_split', 'murmur_present']]
    b = e1.merge(man, on='record_id')
    thr_all = pd.read_csv(THR)
    models = [m for m in MODELS if m in b and m in set(thr_all[thr_all.dataset == 'circor'].model) and ('circor', m) not in EXCLUDE]
    b = add_rejections(b, thr_all, models)
    W = np.load(os.path.join(PROC, 'circor_beat_windows.npy'), mmap_mode='r')
    sel = np.flatnonzero(b.e5_split.isin(['e5_train', 'e5_test']).values)
    X = deep.prep_audio(W[sel])
    b = b.iloc[sel].reset_index(drop=True)
    y = b.murmur_present.values
    test = np.flatnonzero(b.e5_split.values == 'e5_test')
    train_all = np.flatnonzero(b.e5_split.values == 'e5_train')
    tr_pat = np.unique(b.patient_id.values[train_all].astype(str))
    rng = np.random.default_rng(SEED)
    val_pat = set(rng.choice(tr_pat, int(round(0.15 * len(tr_pat))), replace=False))
    is_val = b.patient_id.isin(val_pat).values
    variants = ['unfiltered'] + models
    if len(sys.argv) > 1:
        variants = [v for v in variants if v in sys.argv[1:]]
    rows = []
    for v in variants:
        keep = np.ones(len(b), bool) if v == 'unfiltered' else (b[f'{v}__matched'].values == 0)
        tr = np.flatnonzero((b.e5_split.values == 'e5_train') & ~is_val & keep)
        va = np.flatnonzero((b.e5_split.values == 'e5_train') & is_val & keep)
        for sd in SEEDS:
            model = deep.train_model(f'e5_{v}', X, y, tr, va, deep.CNN, seed=sd,
                                     log_path=os.path.join(LOGT, f'e5_murmur_{v}_s{sd}.csv'))
            pv = patient_scores(b, va, 1 / (1 + np.exp(-deep.predict_logits(model, X[va]))))
            t = youden(pv.y, pv.p)
            pt = patient_scores(b, test, 1 / (1 + np.exp(-deep.predict_logits(model, X[test]))))
            pr = pt.p >= t
            rows.append(dict(filter_model=v, seed=sd, n_train_beats=len(tr) + len(va),
                             frac_train_beats_kept=keep[train_all].mean(),
                             frac_train_murmur_beats_kept=keep[train_all][y[train_all] == 1].mean(),
                             frac_train_absent_beats_kept=keep[train_all][y[train_all] == 0].mean(),
                             n_val_patients=len(pv), threshold=t, n_test_patients=len(pt), n_test_murmur_patients=int(pt.y.sum()),
                             auroc=roc_auc_score(pt.y, pt.p), sensitivity=pr[pt.y == 1].mean(), specificity=1 - pr[pt.y == 0].mean()))
            print(v, sd, round(rows[-1]['auroc'], 3), flush=True)
            del model; torch.cuda.empty_cache()
    per = pd.DataFrame(rows)
    p = os.path.join(RES, 'e5_downstream_seeds.csv')
    if len(sys.argv) > 1 and os.path.exists(p):
        old = pd.read_csv(p)
        per = pd.concat([old[~old.filter_model.isin(per.filter_model)], per], ignore_index=True)
    per.to_csv(p, index=False)
    agg = per.groupby('filter_model', sort=False).agg(
        frac_train_beats_kept=('frac_train_beats_kept', 'first'), frac_train_murmur_beats_kept=('frac_train_murmur_beats_kept', 'first'),
        frac_train_absent_beats_kept=('frac_train_absent_beats_kept', 'first'), n_test_patients=('n_test_patients', 'first'),
        n_test_murmur_patients=('n_test_murmur_patients', 'first'), n_seeds=('seed', 'size'),
        auroc_mean=('auroc', 'mean'), auroc_sd=('auroc', 'std'), sensitivity_mean=('sensitivity', 'mean'),
        sensitivity_sd=('sensitivity', 'std'), specificity_mean=('specificity', 'mean'), specificity_sd=('specificity', 'std')).reset_index()
    order = {m: i for i, m in enumerate(['unfiltered'] + MODELS)}
    agg.sort_values('filter_model', key=lambda c: c.map(order)).to_csv(os.path.join(RES, 'e5_downstream.csv'), index=False)


if __name__ == '__main__':
    main()
