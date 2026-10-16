"""Phase C thresholds, Phase E tables (E1-E4, E6) and Phase F statistics, following ANALYSIS_PLAN.md.

  python analysis.py
Outputs: results/b_quality_performance.csv, c_thresholds.csv, d_clean_pass_rates.csv, e1_raw.csv,
e2_equal_noise.csv, e3_breakdown.csv, e4_by_dataset.csv, e6_mitigation.csv, f_gee_e2.csv
"""
import os, warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, roc_curve
from common import PROC, RES, MAN, SEED, stable_seed

warnings.filterwarnings('ignore')
SC = os.path.join(PROC, 'scores')
MODELS = ['b1_lr', 'b1_svm', 'b2_rf', 'b3_cardiophon', 'b4_cnn', 'b5_crnn', 'e6_cnn_balanced', 'b6a', 'b6b', 'b7', 'b6pna', 'b6pnb']
PUBLISHED = {m: 0.5 for m in MODELS} | {'b6a': 0.20, 'b6b': 0.20, 'b7': -14.0, 'b6pna': 0.20, 'b6pnb': 0.20}
SENSITIVITY = {'b6pna', 'b6pnb'}   # fix 6 (B6 retrained on PhysioNet only): reported, not in the Holm family
DATASETS = ['circor', 'zch', 'bmdhs', 'pascal']
NORMAL = {'normal', 'control'}
NOT_PATH = NORMAL | {'unknown', 'murmur_patient_other_location', 'no_murmur_abnormal_outcome'}
NB = 1000
# option A (user decision): if more than 5 % of calibration beats sit exactly on the matched threshold, every tied
# beat is rejected (calibration rejection then exceeds 10 %) instead of randomised tie-breaking, which capped B6
TIE_MASS_MAX = 0.05
# (dataset, model) pairs excluded from every table: B1 SVM's Platt-scaled score saturates near 0.75 and CirCor's
# matched threshold (0.72) sits at that ceiling, so added noise cannot raise its rejection (see DEVIATIONS.md, fix 1)
EXCLUDE = {('circor', 'b1_svm')}


def drop_excluded(df):
    if df is None or df.empty or 'model' not in df or 'dataset' not in df:
        return df
    bad = np.zeros(len(df), bool)
    for ds, m in EXCLUDE:
        bad |= (df.dataset.values == ds) & (df.model.values == m)
    return df[~bad].reset_index(drop=True)


# ---------------------------------------------------------------- loading
def load(exp, ds):
    p = os.path.join(SC, f'{exp}_{ds}_index.csv')
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, low_memory=False)
    for m in MODELS:
        f = os.path.join(SC, f'{exp}_{ds}_{m}.npy')
        if os.path.exists(f):
            df[m] = np.load(f)
    if ds == 'synth':
        df['dataset'] = 'circor'
    return df


def all_scores():
    e1 = pd.concat([d for d in (load('e1', ds) for ds in DATASETS + ['synth']) if d is not None], ignore_index=True)
    e2 = pd.concat([d for d in (load('e2', ds) for ds in DATASETS + ['synth']) if d is not None], ignore_index=True)
    # E2 no-noise level = E1 scores of clean evaluation beats
    e2 = pd.concat([e1[(e1.clean == 1) & (e1.split == 'eval')].assign(snr_level='none'), e2], ignore_index=True)
    e2['snr_level'] = e2.snr_level.astype(str)
    models = [m for m in MODELS if m in e1 and e1[m].notna().all()]
    return e1, e2, models


def extra_labels():
    b = []
    for ds in DATASETS:
        t = pd.read_csv(os.path.join(MAN, f'{ds}.csv'), low_memory=False)
        b.append(t.drop(columns=[c for c in ('dataset', 'patient_id', 'cls', 'split') if c in t]))
    return pd.concat(b, ignore_index=True).drop_duplicates('record_id')


# ---------------------------------------------------------------- statistics
def patient_counts(df, col):
    g = df.groupby('patient_id')[col].agg(['sum', 'size'])
    return g['sum'].values.astype(float), g['size'].values.astype(float)


def boot(r, n, rng):
    if len(r) == 0:
        return np.full(NB, np.nan)
    idx = rng.integers(0, len(r), (NB, len(r)))
    return r[idx].sum(1) / n[idx].sum(1)


def rate_ci(df, col, ref=None, seed=SEED):
    """Rejection rate with patient bootstrap CI; if ref given, also difference (df - ref) with CI."""
    rng = np.random.default_rng(seed)
    r, n = patient_counts(df, col)
    bs = boot(r, n, rng)
    out = dict(n_beats=int(n.sum()), n_patients=len(r), rate=r.sum() / max(n.sum(), 1),
               ci_lo=np.nanpercentile(bs, 2.5), ci_hi=np.nanpercentile(bs, 97.5))
    if ref is not None:
        r0, n0 = patient_counts(ref, col)
        b0 = boot(r0, n0, rng)
        d = bs - b0
        out.update(diff_vs_normal=out['rate'] - r0.sum() / max(n0.sum(), 1),
                   diff_ci_lo=np.nanpercentile(d, 2.5), diff_ci_hi=np.nanpercentile(d, 97.5))
    return out


# ---------------------------------------------------------------- thresholds
def thresholds(e1, models):
    rows = []
    for ds in DATASETS:
        cal = e1[(e1.dataset == ds) & (e1.split == 'calibration') & e1.cls.isin(NORMAL) & (e1.clean == 1)]
        for m in models:
            v = cal[m].values
            thr = float(np.percentile(v, 90, method='higher'))
            p_gt, p_eq = float((v > thr).mean()), float((v == thr).mean())
            q = float(np.clip((0.10 - p_gt) / p_eq, 0, 1)) if p_eq > 0 else 0.0
            if p_eq > TIE_MASS_MAX:   # option A: a large tie (e.g. B6 unmatched beats = 1.0) is rejected in full
                q = 1.0
            rows.append(dict(dataset=ds, model=m, published=PUBLISHED[m], matched=thr, tie_reject_prob=q,
                             frac_cal_at_threshold=p_eq, n_calibration_beats=len(cal), n_calibration_patients=cal.patient_id.nunique(),
                             cal_reject_rate_matched=p_gt + q * p_eq, tie_rule='reject_all_ties' if p_eq > TIE_MASS_MAX else 'randomised',
                             cal_reject_rate_published=float((v > PUBLISHED[m]).mean())))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RES, 'c_thresholds.csv'), index=False)
    return t


def add_rejections(df, thr, models):
    # ties at the matched threshold are rejected with probability tie_reject_prob, using a uniform number
    # derived from the beat ID (same for every noise level of a beat), so calibration rejection is exactly 10 %
    u = df.beat_id.map(lambda b: stable_seed('tie', b) / 2 ** 32).values
    for m in models:
        tm = thr[thr.model == m].set_index('dataset')
        t, q = df.dataset.map(tm.matched).values, df.dataset.map(tm.tie_reject_prob).values
        s = df[m].values
        df[f'{m}__matched'] = ((s > t) | ((s == t) & (u < q))).astype(int)
        df[f'{m}__published'] = (df[m].values > PUBLISHED[m]).astype(int)
    return df


# ---------------------------------------------------------------- tables
def table(df, models, by, level_col=None):
    rows = []
    keys = ['dataset'] + ([level_col] if level_col else [])
    for k, g in df.groupby(keys):
        k = k if isinstance(k, tuple) else (k,)
        ref = g[g.cls.isin(NORMAL)]
        for c, gc in g.groupby(by):
            for m in models:
                for tt in ('matched', 'published'):
                    col = f'{m}__{tt}'
                    r = rate_ci(gc, col, None if gc.cls.isin(NORMAL).all() else ref)
                    rows.append(dict(zip(keys, k)) | {by: c, 'model': m, 'threshold': tt} | r)
    return pd.DataFrame(rows)


def b_performance():
    rows = []
    for m in ['b1_lr', 'b1_svm', 'b2_rf', 'b3_cardiophon', 'b4_cnn', 'b5_crnn', 'e6_cnn_balanced', 'b6a', 'b6b', 'b7']:
        p = os.path.join(PROC, f'val_preds_{m}.csv')
        if not os.path.exists(p):
            continue
        v = pd.read_csv(p)
        y, s = v.y.values, v.score.values
        pr = s > PUBLISHED[m]
        fpr, tpr, _ = roc_curve(y, s)
        i = np.argmin(np.abs(fpr - (1 - tpr)))
        ev = {'b1_lr': '5-fold out-of-fold', 'b1_svm': '5-fold out-of-fold', 'b2_rf': '5-fold out-of-fold', 'b3_cardiophon': '5-fold out-of-fold',
              'b6a': 'all windows (not trained on quality labels; seen in training)',
              'b6b': 'all windows (not trained on quality labels; seen in training)',
              'b7': 'all windows (annotation-based, not trained)'}.get(m, 'validation fold 0')
        rows.append(dict(model=m, eval_set=ev, n_windows=len(y), auroc=roc_auc_score(y, s),
                         balanced_acc=balanced_accuracy_score(y, pr), sensitivity=pr[y == 1].mean(),
                         specificity=1 - pr[y == 0].mean(), eer=(fpr[i] + 1 - tpr[i]) / 2, threshold=PUBLISHED[m]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, 'b_quality_performance.csv'), index=False)
    return df


def gee(e2, models):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests
    rows = []
    for ds in DATASETS:
        d = e2[e2.dataset == ds]
        classes = [c for c in d.cls.unique() if c not in NORMAL]
        for m in models:
            col = f'{m}__matched'
            for c in classes:
                sub = d[d.cls.isin([c] + list(NORMAL))].copy()
                sub['y'] = sub[col]; sub['path'] = (~sub.cls.isin(NORMAL)).astype(int)
                sub = sub.sort_values('patient_id')
                # drop noise levels with no variation in this contrast (complete separation; no class information)
                var = sub.groupby('snr_level').y.transform(lambda v: 0 < v.mean() < 1).astype(bool)
                sub = sub[var]
                base = dict(dataset=ds, model=m, cls=c, n_beats=len(sub), n_patients=sub.patient_id.nunique(),
                            snr_levels_used='/'.join(sorted(sub.snr_level.unique())))
                if sub.empty or sub.y.nunique() < 2 or sub[sub.path == 1].patient_id.nunique() < 3:
                    rows.append(base | dict(odds_ratio=np.nan, or_ci_lo=np.nan, or_ci_hi=np.nan, p=np.nan, cov_struct='', note='degenerate'))
                    continue
                form = 'y ~ path + C(snr_level)' if sub.snr_level.nunique() > 1 else 'y ~ path'
                res = None
                for cs_name, cs in (('exchangeable', sm.cov_struct.Exchangeable()), ('independence', sm.cov_struct.Independence())):
                    try:
                        with warnings.catch_warnings(record=True) as w:
                            warnings.simplefilter('always')
                            f = smf.gee(form, 'patient_id', sub, family=sm.families.Binomial(), cov_struct=cs).fit(maxiter=100)
                        b, se = f.params['path'], f.bse['path']
                        bad = any('Iteration limit' in str(x.message) or 'converge' in str(x.message).lower() for x in w)
                        if np.isfinite(b) and np.isfinite(se) and se > 1e-6 and abs(b) < 15 and not bad:
                            res = (cs_name, b, se, f.pvalues['path']); break
                    except Exception:
                        pass
                if res is None:
                    rows.append(base | dict(odds_ratio=np.nan, or_ci_lo=np.nan, or_ci_hi=np.nan, p=np.nan, cov_struct='', note='not converged'))
                    continue
                cs_name, b, se, p = res
                ind = {}
                try:   # sensitivity: independence working correlation (patient-clustered robust SE) for every fit
                    fi = smf.gee(form, 'patient_id', sub, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Independence()).fit()
                    bi, si = fi.params['path'], fi.bse['path']
                    ind = dict(or_indep=np.exp(bi), or_indep_ci_lo=np.exp(bi - 1.96 * si), or_indep_ci_hi=np.exp(bi + 1.96 * si),
                               p_indep=fi.pvalues['path'])
                except Exception:
                    pass
                rows.append(base | dict(odds_ratio=np.exp(b), or_ci_lo=np.exp(b - 1.96 * se), or_ci_hi=np.exp(b + 1.96 * se),
                                        p=p, cov_struct=cs_name, note='') | ind)
    g = drop_excluded(pd.DataFrame(rows))
    g['p_holm'] = np.nan
    for (ds, c), idx in g[~g.model.isin(SENSITIVITY)].groupby(['dataset', 'cls']).groups.items():
        ok = g.loc[idx, 'p'].notna()
        if ok.any():
            ii = g.loc[idx][ok].index
            g.loc[ii, 'p_holm'] = multipletests(g.loc[ii, 'p'], method='holm')[1]
    g.to_csv(os.path.join(RES, 'f_gee_e2.csv'), index=False)
    return g


def main():
    b_performance()
    e1, e2, models = all_scores()
    thr = thresholds(e1, models)
    e1 = add_rejections(e1, thr, models); e2 = add_rejections(e2, thr, models)
    ev1 = e1[e1.split == 'eval']
    ev1[['beat_id', 'patient_id', 'dataset', 'cls', 'clean']].to_csv(os.path.join(PROC, 'eval_e1_beats.csv'), index=False)

    # D: clean pass rates (evaluation beats)
    rows = []
    for (ds, c), g in ev1.groupby(['dataset', 'cls']):
        rows.append(dict(dataset=ds, cls=c) | rate_ci(g, 'clean', None if c in NORMAL else ev1[(ev1.dataset == ds) & ev1.cls.isin(NORMAL)]))
    pd.DataFrame(rows).rename(columns={'rate': 'clean_pass_rate'}).to_csv(os.path.join(RES, 'd_clean_pass_rates.csv'), index=False)

    # E1, E2
    t1 = drop_excluded(table(ev1, models, 'cls')); t1.to_csv(os.path.join(RES, 'e1_raw.csv'), index=False)
    t2 = drop_excluded(table(e2, models, 'cls', 'snr_level')); t2.to_csv(os.path.join(RES, 'e2_equal_noise.csv'), index=False)

    # E3 breakdowns
    lab = extra_labels()
    rows = []
    for exp, df in (('E1', ev1.assign(snr_level='raw')), ('E2', e2)):
        df = df.merge(lab, on='record_id', how='left', suffixes=('', '_m'))
        cc = df[df.dataset == 'circor']
        for var in ('sys_timing', 'sys_grade', 'sys_shape'):
            sub = cc[cc.cls == 'murmur_systolic'].copy()
            sub['level_value'] = sub[var].fillna('nan').astype(str)
            for (lv, val), g in sub.groupby(['snr_level', 'level_value']):
                ref = cc[(cc.snr_level == lv) & (cc.cls == 'normal')]
                for m in models:
                    rows.append(dict(experiment=exp, dataset='circor', snr_level=lv, breakdown=var, value=val, model=m,
                                     threshold='matched') | rate_ci(g, f'{m}__matched', ref))
        for ds, flags in (('bmdhs', ['AS', 'AR', 'MR', 'MS']),):
            d = df[df.dataset == ds]
            for f in flags:
                for lv, g in d[d[f] == 1].groupby('snr_level'):
                    ref = d[(d.snr_level == lv) & d.cls.isin(NORMAL)]
                    for m in models:
                        rows.append(dict(experiment=exp, dataset=ds, snr_level=lv, breakdown='disease_present', value=f, model=m,
                                         threshold='matched') | rate_ci(g, f'{m}__matched', ref))
        for ds, classes in (('zch', ['ASD', 'VSD', 'PFO', 'PDA']), ('bmdhs', ['AS', 'AR', 'MR', 'MS', 'multi']),
                            ('pascal', ['murmur', 'extrasystole']), ('circor', ['synthetic_premature', 'synthetic_irregular'])):
            d = df[df.dataset == ds]
            for (lv, c), g in d[d.cls.isin(classes)].groupby(['snr_level', 'cls']):
                ref = d[(d.snr_level == lv) & d.cls.isin(NORMAL)]
                for m in models:
                    rows.append(dict(experiment=exp, dataset=ds, snr_level=lv, breakdown='class', value=c, model=m,
                                     threshold='matched') | rate_ci(g, f'{m}__matched', ref))
    drop_excluded(pd.DataFrame(rows)).to_csv(os.path.join(RES, 'e3_breakdown.csv'), index=False)

    # E4: pooled pathological vs normal per dataset
    rows = []
    for exp, df in (('E1', ev1.assign(snr_level='raw')), ('E2', e2)):
        df = df[~df.cls.isin(NOT_PATH - NORMAL) & ~df.cls.str.startswith('synthetic')]
        for (ds, lv), g in df.groupby(['dataset', 'snr_level']):
            ref, path = g[g.cls.isin(NORMAL)], g[~g.cls.isin(NORMAL)]
            for m in models:
                r = rate_ci(path, f'{m}__matched', ref)
                r0 = rate_ci(ref, f'{m}__matched')
                rows.append(dict(experiment=exp, dataset=ds, snr_level=lv, model=m, threshold='matched',
                                 pathological_classes='+'.join(sorted(path.cls.unique())), normal_rate=r0['rate'],
                                 pathological_rate=r['rate'], gap=r['diff_vs_normal'], gap_ci_lo=r['diff_ci_lo'],
                                 gap_ci_hi=r['diff_ci_hi'], n_path_patients=r['n_patients'], n_normal_patients=r0['n_patients']))
    e4 = drop_excluded(pd.DataFrame(rows)); e4.to_csv(os.path.join(RES, 'e4_by_dataset.csv'), index=False)

    # E6: (a) B6a vs B6b, (b) balanced CNN vs B4 on E2 (pooled gaps from E4 + per-class E2)
    e6 = e4[e4.model.isin(['b6a', 'b6b', 'b4_cnn', 'e6_cnn_balanced'])].copy()
    e6['comparison'] = np.where(e6.model.isin(['b6a', 'b6b']), 'a: rhythm check (B6a vs B6b)', 'b: balanced training (B4 vs E6)')
    e6.to_csv(os.path.join(RES, 'e6_mitigation.csv'), index=False)

    gee(e2, models)
    print('analysis done; models:', models)


if __name__ == '__main__':
    main()
