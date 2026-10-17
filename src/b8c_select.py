"""B8c: combine B8-noaug (a) and B8 (b) noise scores; select the combination on PhysioNet fold 0 only (pre-registered),
then write test-set score arrays for the selected combination.  Output: results/b8c_selection.csv, scores e*_<ds>_b8c.npy
"""
import os, sys, glob
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
from common import PROC, MODELS, RES, SEED, WIN, highpass
import train_b8 as T

COMBOS = {'geomean': lambda a, b: np.sqrt(np.clip(a, 0, 1) * np.clip(b, 0, 1)), 'mean': lambda a, b: (a + b) / 2,
          'min': np.minimum}


def dev_crops():
    """Exactly the crops of the training script's rhythm check (seed SEED+7)."""
    d = T.TS.Data()
    man = pd.read_csv(os.path.join(os.path.dirname(PROC), '..', 'manifests', 'physionet2016.csv'))
    fold = dict(zip('pn' + man.record_id, man.fold))
    rec_fold = np.array([fold[r] for r in d.rid])
    r2 = np.random.default_rng(SEED + 7)
    cand = [r for r in np.flatnonzero(rec_fold == 0) if d.sqi[r] == 1 and d.len[r] >= 3 * T.SAMPLES][:300]
    A, B = [], []
    for r in cand:
        s0 = d.off[r]; x = d.audio[s0:s0 + 3 * T.SAMPLES].astype(np.float32); y = d.lab[s0:s0 + 3 * T.SAMPLES]
        if (y == T.IGNORE).mean() > 0.2 or (y == 4).any():
            continue
        x2, _ = T.rr_aug(x, y, r2, (0.6, 1.0, 0.7, 1.3, 0.4))
        A.append(x[:WIN]); B.append(x2[:WIN])
    return np.stack(A), np.stack(B)


def model_scores(name, X):
    T.TS.highpass = highpass
    fe, net = T.Frontend().to('cuda'), T.SegNet().to('cuda')
    net.load_state_dict(torch.load(os.path.join(MODELS, f'{name}.pt'), map_location='cuda')); net.eval()
    return T.window_noise_scores(fe, net, X)


def main():
    a = pd.read_csv(os.path.join(PROC, 'val_preds_b8noaug.csv')); b = pd.read_csv(os.path.join(PROC, 'val_preds_b8.csv'))
    assert (a.window_id == b.window_id).all()
    A, B = dev_crops()
    sa = {k: model_scores(k, A) for k in ('b8noaug', 'b8')}; sb = {k: model_scores(k, B) for k in ('b8noaug', 'b8')}
    rows = []
    for name, f in COMBOS.items():
        s = f(a.score.values, b.score.values)
        c0, c1 = f(sa['b8noaug'], sa['b8']), f(sb['b8noaug'], sb['b8'])
        rh = roc_auc_score(np.r_[np.zeros(len(c0)), np.ones(len(c1))], np.r_[c0, c1])
        rows.append(dict(combo=name, fold0_auroc=roc_auc_score(a.y, s), rhythm_shift_auroc=rh, dev_dist=abs(rh - 0.5)))
    for k in ('b8noaug', 'b8'):
        rh = roc_auc_score(np.r_[np.zeros(len(sa[k])), np.ones(len(sb[k]))], np.r_[sa[k], sb[k]])
        v = pd.read_csv(os.path.join(PROC, f'val_preds_{k}.csv'))
        rows.append(dict(combo=f'(component {k})', fold0_auroc=roc_auc_score(v.y, v.score), rhythm_shift_auroc=rh, dev_dist=abs(rh - 0.5)))
    t = pd.DataFrame(rows)
    cand = t[~t.combo.str.startswith('(') & (t.fold0_auroc >= 0.963)]
    order = {'geomean': 0, 'mean': 1, 'min': 2}
    if len(cand):
        pick = cand.assign(o=cand.combo.map(order)).sort_values(['dev_dist', 'o']).combo.iloc[0]; passed = True
    else:
        pick = t[~t.combo.str.startswith('(')].sort_values('fold0_auroc').combo.iloc[-1]; passed = False
    t['selected'] = t.combo == pick
    t.to_csv(os.path.join(RES, 'b8c_selection.csv'), index=False)
    print(t.round(4).to_string(index=False)); print('SELECTED', pick, 'C1 passed' if passed else 'C1 FAILED')
    f = COMBOS[pick]
    pd.DataFrame(dict(window_id=a.window_id, fold=0, y=a.y, score=f(a.score.values, b.score.values))).to_csv(
        os.path.join(PROC, 'val_preds_b8c.csv'), index=False)
    sc = os.path.join(PROC, 'scores')
    for fa in glob.glob(os.path.join(sc, '*_b8noaug.npy')):
        fb = fa.replace('_b8noaug.npy', '_b8.npy')
        np.save(fa.replace('_b8noaug.npy', '_b8c.npy'), f(np.load(fa), np.load(fb)))
    print('test-set b8c arrays written')


if __name__ == '__main__':
    main()
