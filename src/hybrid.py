"""Hybrid hypothesis (post hoc, pre-specified in ANALYSIS_PLAN.md): periodicity (Das, B2) + deep model (B4 CNN / B5 CRNN).

  H-AND      score = min(Das percentile, DL percentile)               (reject only when both look noisy)
  H-CASCADE  score = Das percentile if < 0.4 or > 0.9, else the DL percentile
  H-FUSE     random forest (120 trees, sqrt, balanced, seed 42) on 11 Das features + the deep model's 128-d penultimate embedding

Percentile = mid-rank empirical CDF among the clean normal calibration beats of the test dataset (E1, as recorded; synthetic
sets use CirCor's). On PhysioNet (Phase B) the reference is the out-of-fold scores of usable windows of normal recordings
(not augmented) in the four other folds. Hybrid seed s = deep model seed s (+ B2 seed s for H-AND / H-CASCADE).
Training data: PhysioNet 2016 only. H-FUSE uses out-of-fold embeddings from fold-specific deep models (fold 0 = main model).

  python hybrid.py pn      PhysioNet: out-of-fold embeddings, H-FUSE forests, val_preds_<hybrid>.csv
  python hybrid.py test    test-set score arrays data/processed/scores/<exp>_<ds>_<hybrid>.npy
"""
import os, sys
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from common import PROC, MODELS, SEED
from score import SC, SNRS

SEEDS = [SEED, SEED + 1, SEED + 2]
DL = {'b4': 'b4_cnn', 'b5': 'b5_crnn'}
KINDS = ('and', 'cas', 'fuse')
CAS_LO, CAS_HI = 0.4, 0.9
DATASETS = ['circor', 'zch', 'bmdhs', 'pascal', 'synth']
NORMAL = ('normal', 'control')


def tag(m, s):
    return m if s == SEED else f'{m}_s{s}'


def hybrid_tags():
    return [tag(f'h_{k}_{a}', s) for k in KINDS for a in DL for s in SEEDS]


def pct(ref, x):
    """Mid-rank empirical CDF of x within ref."""
    r = np.sort(np.asarray(ref, np.float64))
    return (np.searchsorted(r, x, 'left') + np.searchsorted(r, x, 'right')) / (2.0 * len(r))


def combine(kind, pd_, pl):
    if kind == 'and':
        return np.minimum(pd_, pl)
    return np.where((pd_ < CAS_LO) | (pd_ > CAS_HI), pd_, pl)


def rf():
    return RandomForestClassifier(n_estimators=120, max_features='sqrt', class_weight='balanced', random_state=SEED, n_jobs=8)


# ---------------------------------------------------------------- PhysioNet
def oof_dl(meta, m, s):
    """Out-of-fold deep scores and embeddings for every PhysioNet window (fold 0 = main model, folds 1-4 = fold models)."""
    import deep
    sc = np.full(len(meta), np.nan); E = np.full((len(meta), 128), np.nan, np.float32)
    for k in range(5):
        t = tag(m, s) + ('' if k == 0 else f'_f{k}')
        idx = np.flatnonzero(meta.fold.values == k)
        v = pd.read_csv(os.path.join(PROC, f'val_preds_{t}.csv'))
        assert (v.window_id.values == meta.window_id.values[idx]).all()
        sc[idx] = v.score.values
        fe = os.path.join(PROC, f'oof_embed_{t}.npy')
        if not os.path.exists(fe):          # seed-42 main models predate the embedding hook: compute once, keep scores
            W = np.load(os.path.join(PROC, 'pn_windows_audio.npy'), mmap_mode='r')
            p, e = deep.score(t, W[idx], embed=True)
            assert np.abs(p - sc[idx]).max() < 1e-4
            np.save(fe, e.astype(np.float32))
        E[idx] = np.load(fe)
    return sc, E


def run_pn():
    meta = pd.read_csv(os.path.join(PROC, 'pn_windows.csv'))
    das = np.load(os.path.join(PROC, 'pn_feats.npz'))['b2']
    y, fold = meta.unusable.values, meta.fold.values
    ref_ok = ((meta.unusable == 0) & (meta.abnormal == 0) & (meta.aug == 0)).values
    for s in SEEDS:
        b2 = pd.read_csv(os.path.join(PROC, f'val_preds_{tag("b2_rf", s)}.csv'))
        assert (b2.window_id == meta.window_id).all()
        for a, m in DL.items():
            sc, E = oof_dl(meta, m, s)
            # H-AND / H-CASCADE: percentiles against the other folds' usable normal-recording windows
            P = {}
            for k in range(5):
                te = fold == k; ref = ref_ok & (fold != k)
                P.setdefault('das', np.zeros(len(meta)))[te] = pct(b2.score.values[ref], b2.score.values[te])
                P.setdefault('dl', np.zeros(len(meta)))[te] = pct(sc[ref], sc[te])
            for kind in ('and', 'cas'):
                pd.DataFrame(dict(window_id=meta.window_id, fold=fold, y=y, score=combine(kind, P['das'], P['dl']))).to_csv(
                    os.path.join(PROC, f'val_preds_{tag(f"h_{kind}_{a}", s)}.csv'), index=False)
            # H-FUSE: nested out-of-fold validation, then the final forest on all windows
            X = np.hstack([das, E]).astype(np.float32)
            oof = np.zeros(len(meta))
            for k in range(5):
                oof[fold == k] = rf().fit(X[fold != k], y[fold != k]).predict_proba(X[fold == k])[:, 1]
            t = tag(f'h_fuse_{a}', s)
            pd.DataFrame(dict(window_id=meta.window_id, fold=fold, y=y, score=oof)).to_csv(os.path.join(PROC, f'val_preds_{t}.csv'), index=False)
            joblib.dump(rf().fit(X, y), os.path.join(MODELS, f'{t}.joblib'))
            print('pn', t, flush=True)


# ---------------------------------------------------------------- test sets
def das_test(exp, ds):
    if exp == 'e1':
        return np.load(os.path.join(SC, f'e1_{ds}_none_feats.npz'))['b2']
    return np.concatenate([np.load(os.path.join(SC, f'{exp}_{ds}_{lv}_feats.npz'))['b2'] for lv in SNRS])


def run_test():
    idx1 = {ds: pd.read_csv(os.path.join(SC, f'e1_{ds}_index.csv'), low_memory=False) for ds in DATASETS}
    for s in SEEDS:
        refs = {}
        for ds in DATASETS:
            src = 'circor' if ds == 'synth' else ds
            i1 = idx1[src]
            cal = ((i1.split == 'calibration') & i1.cls.isin(NORMAL) & (i1.clean == 1)).values
            refs[ds] = {k: np.load(os.path.join(SC, f'e1_{src}_{tag(m, s)}.npy'))[cal] for k, m in [('das', 'b2_rf')] + list(DL.items())}
        for exp in ('e1', 'e2', 'e2s'):
            for ds in DATASETS:
                n = len(pd.read_csv(os.path.join(SC, f'{exp}_{ds}_index.csv'), usecols=['beat_id']))
                d = np.load(os.path.join(SC, f'{exp}_{ds}_{tag("b2_rf", s)}.npy'))
                F = das_test(exp, ds)
                assert len(d) == n == len(F)
                pdas = pct(refs[ds]['das'], d)
                for a, m in DL.items():
                    l = np.load(os.path.join(SC, f'{exp}_{ds}_{tag(m, s)}.npy'))
                    pl = pct(refs[ds][a], l)
                    for kind in ('and', 'cas'):
                        np.save(os.path.join(SC, f'{exp}_{ds}_{tag(f"h_{kind}_{a}", s)}.npy'), combine(kind, pdas, pl))
                    E = np.load(os.path.join(SC, f'{exp}_{ds}_{tag(m, s)}_emb.npy')).astype(np.float32)
                    t = tag(f'h_fuse_{a}', s)
                    clf = joblib.load(os.path.join(MODELS, f'{t}.joblib'))
                    np.save(os.path.join(SC, f'{exp}_{ds}_{t}.npy'), clf.predict_proba(np.hstack([F, E]))[:, 1])
                print('test', s, exp, ds, n, flush=True)


if __name__ == '__main__':
    run_pn() if sys.argv[1] == 'pn' else run_test()
