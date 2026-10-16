"""Follow-up analyses (fixes 2-4).

  fix 2  recalibration for BMD-HS and PASCAL: published threshold and a threshold calibrated on pooled normal
         calibration patients of all datasets (patient-weighted 90th percentile, ties randomised as in the main analysis)
  fix 3  heart-rate (and age) confound: per-beat heart rate from the segmentation, age where available; E2 GEE with
         covariates and a stratified check in heart-rate bands
  fix 4  GEE  rejected ~ class + SNR level + heart rate [+ age], clustered by patient, Holm across models; counts table;
         inspection of the CirCor late-systolic group
Outputs: results/fix2_*.csv, results/fix3_*.csv, results/fix4_*.csv
"""
import os, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from common import PROC, MAN, RES, stable_seed, BMD_ROOT
from analysis import (all_scores, thresholds, add_rejections, rate_ci, drop_excluded, EXCLUDE, NORMAL, PUBLISHED,
                      DATASETS, SENSITIVITY, TIE_MASS_MAX)

warnings.filterwarnings('ignore')
HR_BANDS = [0, 80, 100, 120, 140, 400]
HR_LABELS = ['<80', '80-100', '100-120', '120-140', '>=140']


# ---------------------------------------------------------------- covariates
def covariates():
    """beat_id -> heart rate (bpm, from the beat's segmentation) and age category."""
    rows = []
    for ds in DATASETS + ['synth']:
        b = pd.read_csv(os.path.join(PROC, f'{ds}_beats.csv'), usecols=['beat_id', 'record_id', 'start', 'end'], low_memory=False)
        b['hr_bpm'] = 60 * 2000 / (b.end - b.start)
        rows.append(b[['beat_id', 'record_id', 'hr_bpm']].assign(src=ds))
    c = pd.concat(rows, ignore_index=True)
    age = {}
    cc = pd.read_csv(os.path.join(MAN, 'circor.csv'))
    age.update(dict(zip(cc.record_id, cc.age.fillna('unknown').astype(str))))
    z = pd.read_csv(os.path.join(MAN, 'zch.csv'))
    d = z.age_days.astype(float)
    zc = np.select([d < 28, d < 365, d < 12 * 365.25], ['Neonate', 'Infant', 'Child'], 'Adolescent')
    age.update(dict(zip(z.record_id, zc)))
    bm = pd.read_csv(os.path.join(MAN, 'bmdhs.csv'))
    meta = pd.read_csv(os.path.join(BMD_ROOT, 'additional_metadata.csv'))
    meta['patient_id'] = 'bmd_' + meta.patient_id
    bm = bm.merge(meta[['patient_id', 'Age']], on='patient_id', how='left')
    age.update(dict(zip(bm.record_id, bm.Age.fillna('unknown').astype(str))))
    syn = pd.read_csv(os.path.join(PROC, 'synth_beats.csv'), usecols=['record_id', 'source_record'])
    age.update({r: age.get(s, 'unknown') for r, s in zip(syn.record_id, syn.source_record)})
    c['age_cat'] = c.record_id.map(age).fillna('unknown')
    return c[['beat_id', 'hr_bpm', 'age_cat']]


# ---------------------------------------------------------------- fix 2
def weighted_matched(v, w, q=0.90):
    """Threshold t (an observed value) and tie probability so that the w-weighted rejection (> t, ties at t with
    probability p) equals 1 - q."""
    o = np.argsort(v); v, w = v[o], w[o] / w.sum()
    cw = np.cumsum(w)
    t = v[min(np.searchsorted(cw, q), len(v) - 1)]
    gt, eq = w[v > t].sum(), w[v == t].sum()
    p = float(np.clip((1 - q - gt) / eq, 0, 1)) if eq > 0 else 0.0
    if eq > TIE_MASS_MAX:           # option A, as in the matched threshold
        p = 1.0
    return float(t), p


def fix2(e1, e2, models):
    cal = e1[(e1.split == 'calibration') & e1.cls.isin(NORMAL) & (e1.clean == 1) & e1.dataset.isin(DATASETS)]
    w = 1.0 / cal.groupby('patient_id').beat_id.transform('size').values    # each calibration patient weighs equally
    rows = []
    for m in models:
        t, p = weighted_matched(cal[m].values, w)
        rows.append(dict(model=m, pooled=t, pooled_tie_prob=p, n_cal_patients=cal.patient_id.nunique(), n_cal_beats=len(cal),
                         **{f'n_cal_patients_{ds}': cal[cal.dataset == ds].patient_id.nunique() for ds in DATASETS}))
    pooled = pd.DataFrame(rows)
    pooled.to_csv(os.path.join(RES, 'fix2_thresholds_pooled.csv'), index=False)

    def rej(df, m, kind):
        s = df[m].values
        if kind == 'published':
            return (s > PUBLISHED[m]).astype(int)
        r = pooled.set_index('model').loc[m]
        u = df.beat_id.map(lambda b: stable_seed('tie', b) / 2 ** 32).values
        return ((s > r.pooled) | ((s == r.pooled) & (u < r.pooled_tie_prob))).astype(int)

    out = {}
    for exp, df in (('E1', e1[(e1.split == 'eval')].assign(snr_level='raw')), ('E2', e2)):
        df = df[df.dataset.isin(['bmdhs', 'pascal'])].copy()
        res = []
        for kind in ('published', 'pooled', 'matched'):
            for m in models:
                df['r'] = df[f'{m}__matched'] if kind == 'matched' else rej(df, m, kind)
                for (ds, lv), g in df.groupby(['dataset', 'snr_level']):
                    ref = g[g.cls.isin(NORMAL)]
                    for c, gc in g.groupby('cls'):
                        res.append(dict(experiment=exp, dataset=ds, snr_level=lv, cls=c, model=m, threshold=kind)
                                   | rate_ci(gc, 'r', None if c in NORMAL else ref))
        out[exp] = pd.DataFrame(res)
        out[exp].to_csv(os.path.join(RES, f'fix2_recal_{exp.lower()}.csv'), index=False)
    # achieved clean-normal rejection per threshold (calibration vs evaluation patients), all datasets
    rows = []
    cn = e1[e1.cls.isin(NORMAL) & (e1.clean == 1) & e1.dataset.isin(DATASETS)].copy()
    for kind in ('published', 'pooled', 'matched'):
        for m in models:
            cn['r'] = cn[f'{m}__matched'] if kind == 'matched' else rej(cn, m, kind)
            for (ds, sp), g in cn.groupby(['dataset', 'split']):
                rows.append(dict(dataset=ds, split=sp, model=m, threshold=kind) | rate_ci(g, 'r'))
    nr = drop_excluded(pd.DataFrame(rows))
    nr.to_csv(os.path.join(RES, 'fix2_clean_normal_rejection.csv'), index=False)
    return out


# ---------------------------------------------------------------- GEE with covariates (fix 3/4)
def fit(sub, form):
    for cs_name, cs in (('exchangeable', sm.cov_struct.Exchangeable()), ('independence', sm.cov_struct.Independence())):
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                f = smf.gee(form, 'patient_id', sub, family=sm.families.Binomial(), cov_struct=cs).fit(maxiter=100)
            b, se = f.params['path'], f.bse['path']
            bad = any('Iteration limit' in str(x.message) or 'converge' in str(x.message).lower() for x in w)
            if np.isfinite(b) and np.isfinite(se) and 1e-6 < se and abs(b) < 15 and not bad:
                return dict(odds_ratio=np.exp(b), or_ci_lo=np.exp(b - 1.96 * se), or_ci_hi=np.exp(b + 1.96 * se),
                            p=f.pvalues['path'], cov_struct=cs_name)
        except Exception:
            pass
    return dict(odds_ratio=np.nan, or_ci_lo=np.nan, or_ci_hi=np.nan, p=np.nan, cov_struct='not converged')


def gee_cov(e2, models):
    rows = []
    for ds in DATASETS:
        d = e2[e2.dataset == ds]
        has_age = d.age_cat.nunique() > 1
        for m in models:
            if (ds, m) in EXCLUDE:
                continue
            for c in [c for c in d.cls.unique() if c not in NORMAL]:
                sub = d[d.cls.isin([c] + list(NORMAL))].copy()
                sub['y'] = sub[f'{m}__matched']; sub['path'] = (~sub.cls.isin(NORMAL)).astype(int)
                sub = sub[sub.groupby('snr_level').y.transform(lambda v: 0 < v.mean() < 1).astype(bool)].sort_values('patient_id')
                if sub.empty or sub[sub.path == 1].patient_id.nunique() < 3 or sub.y.nunique() < 2:
                    continue
                sub['hr_z'] = (sub.hr_bpm - sub.hr_bpm.mean()) / (sub.hr_bpm.std() + 1e-9)
                snr = ' + C(snr_level)' if sub.snr_level.nunique() > 1 else ''
                specs = {'unadjusted': f'y ~ path{snr}', 'hr': f'y ~ path{snr} + hr_z + I(hr_z**2)'}
                if has_age:
                    # drop age levels with < 2 patients in this subset to keep the design estimable
                    keep = sub.groupby('age_cat').patient_id.transform('nunique') >= 2
                    specs['hr_age'] = (f'y ~ path{snr} + hr_z + I(hr_z**2) + C(age_cat)', keep)
                base = dict(dataset=ds, model=m, cls=c, n_beats=len(sub), n_patients=sub.patient_id.nunique(),
                            n_path_patients=sub[sub.path == 1].patient_id.nunique(),
                            hr_median_path=sub[sub.path == 1].hr_bpm.median(), hr_median_normal=sub[sub.path == 0].hr_bpm.median())
                for name, spec in specs.items():
                    form, keep = (spec if isinstance(spec, tuple) else (spec, None))
                    s2 = sub[keep] if keep is not None else sub
                    r = fit(s2, form)
                    try:
                        fi = smf.gee(form, 'patient_id', s2, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Independence()).fit()
                        r |= dict(or_indep=np.exp(fi.params['path']), p_indep=fi.pvalues['path'])
                    except Exception:
                        pass
                    rows.append(base | dict(adjustment=name) | r)
    g = pd.DataFrame(rows)
    g['p_holm'] = np.nan
    for _, idx in g[~g.model.isin(SENSITIVITY)].groupby(['dataset', 'cls', 'adjustment']).groups.items():
        ii = [i for i in idx if np.isfinite(g.loc[i, 'p'])]
        if ii:
            g.loc[ii, 'p_holm'] = multipletests(g.loc[ii, 'p'], method='holm')[1]
    g.to_csv(os.path.join(RES, 'fix4_gee_e2_covariates.csv'), index=False)
    return g


def hr_bands(e2, models):
    rows = []
    e2 = e2.copy()
    e2['hr_band'] = pd.cut(e2.hr_bpm, HR_BANDS, labels=HR_LABELS, right=False).astype(str)
    for ds in DATASETS:
        d = e2[e2.dataset == ds]
        for band, g in d.groupby('hr_band'):
            ref = g[g.cls.isin(NORMAL)]
            if ref.patient_id.nunique() < 3:
                continue
            for c, gc in g.groupby('cls'):
                if c in NORMAL or gc.patient_id.nunique() < 3:
                    continue
                for m in models:
                    if (ds, m) in EXCLUDE:
                        continue
                    r = rate_ci(gc, f'{m}__matched', ref)
                    rows.append(dict(dataset=ds, hr_band=band, cls=c, model=m, normal_rate=ref[f'{m}__matched'].mean(),
                                     n_normal_patients=ref.patient_id.nunique()) | r)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RES, 'fix3_hr_band_gaps.csv'), index=False)
    return t


def describe_cov(e1):
    ev = e1[(e1.split == 'eval') & e1.dataset.isin(DATASETS)]
    g = ev.groupby(['dataset', 'cls'])
    t = g.hr_bpm.describe()[['count', '25%', '50%', '75%']].rename(columns={'count': 'n_beats', '25%': 'hr_q25', '50%': 'hr_median', '75%': 'hr_q75'})
    age = ev.groupby(['dataset', 'cls']).apply(lambda x: x.drop_duplicates('patient_id').age_cat.value_counts(normalize=True).round(3).to_dict())
    t['age_distribution_patients'] = age
    t.reset_index().to_csv(os.path.join(RES, 'fix3_hr_age_by_class.csv'), index=False)


# ---------------------------------------------------------------- fix 4: counts and late-systolic
def counts(e1, e2):
    rows = []
    for ds in DATASETS + ['circor_synth']:
        src = 'synth' if ds == 'circor_synth' else ds
        man = pd.read_csv(os.path.join(PROC, f'{src}_beats.csv'), usecols=['record_id', 'patient_id', 'cls', 'split', 'clean'], low_memory=False)
        for c, g in man.groupby('cls'):
            ev = g[g.split == 'eval']
            rows.append(dict(dataset=ds, cls=c, calibration_patients=g[g.split == 'calibration'].patient_id.nunique(),
                             eval_patients=ev.patient_id.nunique(), eval_recordings=ev.record_id.nunique(),
                             e1_eval_beats=len(ev), e2_clean_eval_beats=int(ev.clean.sum()),
                             e2_clean_eval_patients=ev[ev.clean == 1].patient_id.nunique()))
    pd.DataFrame(rows).to_csv(os.path.join(RES, 'fix4_counts.csv'), index=False)


def late_systolic(e1, e2, models):
    cm = pd.read_csv(os.path.join(MAN, 'circor.csv'))
    meta = pd.read_csv(os.path.join(os.path.dirname(MAN), 'data', 'raw', 'circor', 'training_data.csv'))
    pts = cm[cm.sys_timing == 'Late-systolic'].patient_id.unique()
    out = []
    for p in pts:
        r = cm[cm.patient_id == p]
        mrow = meta[meta['Patient ID'] == int(p[2:])].iloc[0]
        b1 = e1[e1.patient_id == p]
        row = dict(patient_id=p, split=r.split.iloc[0], murmur_locations=mrow['Murmur locations'],
                   recording_locations=mrow['Recording locations:'], grade=r.sys_grade.iloc[0], shape=r.sys_shape.iloc[0],
                   age=r.age.iloc[0], n_recordings=r.record_id.nunique(),
                   n_beats_by_class=b1.cls.value_counts().to_dict(), n_clean_murmur_beats=int(((b1.cls == 'murmur_systolic') & (b1.clean == 1)).sum()))
        ms = b1[b1.cls == 'murmur_systolic']
        for m in models:
            if ('circor', m) not in EXCLUDE:
                row[f'e1_reject_{m}'] = ms[f'{m}__matched'].mean() if len(ms) else np.nan
        out.append(row)
    pd.DataFrame(out).to_csv(os.path.join(RES, 'fix4_late_systolic.csv'), index=False)


def main():
    e1, e2, models = all_scores()
    thr = thresholds(e1, models)
    e1 = add_rejections(e1, thr, models); e2 = add_rejections(e2, thr, models)
    cov = covariates()
    e1 = e1.merge(cov, on='beat_id', how='left'); e2 = e2.merge(cov, on='beat_id', how='left')
    assert e1.hr_bpm.notna().all() and e2.hr_bpm.notna().all()
    fix2(e1, e2, models)
    describe_cov(e1)
    hr_bands(e2, models)
    gee_cov(e2, models)
    counts(e1, e2)
    late_systolic(e1, e2, models)
    print('fixes analysis done')


if __name__ == '__main__':
    main()
