"""Seed robustness (post hoc, see ANALYSIS_PLAN.md): one full replicate of the benchmark per seed, mean +- SD over seeds,
pooled GEE with seed as an extra factor, and the seed-robust verdict.

  python seeds.py

Seeded models: B2 RF, B3 CardioPHON, B4 CNN, B5 CRNN, E6 balanced CNN (seeds 42, 43, 44; 42 = original models) and the six
hybrids (hybrid seed s = its deep model's seed s, and B2 seed s). Every other model is identical in every seed; its rows are
computed once and reused, but its Holm-adjusted p-values are recomputed per seed (the Holm family includes seeded models).
Seed-robust verdict per cell: Holm-significant in >= 2 of 3 seeds in one direction AND pooled GEE (+ C(seed), patient
clusters, Holm across the same family) significant in that direction; otherwise "no significant difference".
Outputs (results/): <table>_per_seed.csv (long, column seed) and <table>_seeds.csv (mean, SD over seeds) for
b_quality_performance, c_thresholds, e1_raw, e2_equal_noise, e2s_equal_noise, e3_breakdown, e4_by_dataset, f_gee_e2, f_gee_e2s,
fix2_clean_normal_rejection, fix2_recal_e2, fix3_hr_band_gaps; gee_covariates_per_seed.csv and gee_covariates_seeds.csv
(both E2 variants, all adjustments, per-seed OR / Holm p, pooled fit, verdict).
"""
import os, warnings
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from common import RES, SEED
import analysis as A
import analysis_fixes as F

warnings.filterwarnings('ignore')
SEEDS = [SEED, SEED + 1, SEED + 2]
SEEDED = ['b2_rf', 'b3_cardiophon', 'b4_cnn', 'b5_crnn', 'e6_cnn_balanced']
VARIES = SEEDED + A.HYBRIDS
NPROC = 12
G = {}   # frames shared with forked workers


def tag(m, s):
    return m if s == SEED else f'{m}_s{s}'


def tags(s):
    return {m: (tag(m, s) if m in VARIES else m) for m in A.MODELS}


# ---------------------------------------------------------------- per-seed replicate
def _rep(m):
    R = A.replicate(G['e1'], G['e2'], G['e2s'], [m])
    R['gee_cov_e2'] = F.gee_cov(G['e2'], [m], out=None, adjust=False, indep=False)
    R['gee_cov_e2s'] = F.gee_cov(G['e2s'], [m], out=None, adjust=False, indep=False)
    fx = F.fix2(G['e1'], G['e2'], [m], write=False)
    R['fix2_clean_normal'], R['fix2_recal_e2'] = fx['clean_normal'], fx['E2']
    R['hr_bands'] = F.hr_bands(G['e2'], [m], write=False)
    return R


def replicate_seed(s, cov):
    e1, e2, models = A.all_scores('e2', tags(s))
    _, e2s, ms = A.all_scores('e2s', tags(s))
    models = [m for m in models if m in ms]
    thr = A.thresholds(e1, models)
    e1, e2, e2s = (A.add_rejections(x, thr, models).merge(cov, on='beat_id', how='left') for x in (e1, e2, e2s))
    run = models if s == SEED else [m for m in models if m in VARIES]
    G.update(e1=e1, e2=e2, e2s=e2s)
    with ProcessPoolExecutor(NPROC) as ex:
        parts = list(ex.map(_rep, run))
    R = {k: pd.concat([p[k] for p in parts], ignore_index=True) for k in parts[0]}
    R['thresholds'] = thr
    R['perf'] = A.b_performance({m: tags(s)[m] for m in A.MODELS}, write=False)
    keep = ['beat_id', 'patient_id', 'dataset', 'cls', 'snr_level', 'hr_bpm', 'age_cat']
    rej = {v: x[keep + [f'{m}__matched' for m in models if m in VARIES]] for v, x in (('E2', e2), ('E2s', e2s))}
    return R, models, rej


# ---------------------------------------------------------------- pooled GEE (seed as a factor)
def _pooled(args):
    m, variant = args
    frames = G[f'rej_{variant}']
    base = frames[SEEDS[0]]
    rows = []
    for ds in A.DATASETS:
        idx = (base.dataset == ds).values
        d0 = base[idx]
        has_age = d0.age_cat.nunique() > 1
        for c, pc, ref in A.contrasts(d0, ds):
            subs = []
            for s in SEEDS:
                d = frames[s][idx]
                sub = d[d.cls.isin([pc] + ref)].copy()
                sub['y'] = sub[f'{m}__matched'].values; sub['path'] = (sub.cls == pc).astype(int); sub['seed'] = str(s)
                subs.append(sub)
            sub = pd.concat(subs, ignore_index=True)
            sub = sub[sub.groupby('snr_level').y.transform(lambda v: 0 < v.mean() < 1).astype(bool)].sort_values(['patient_id', 'seed'])
            if sub.empty or sub[sub.path == 1].patient_id.nunique() < 3 or sub.y.nunique() < 2:
                continue
            for name, spec in F.cov_designs(sub, has_age, extra=' + C(seed)').items():
                form, keep = (spec if isinstance(spec, tuple) else (spec, None))
                r = F.fit(sub[keep] if keep is not None else sub, form)
                rows.append(dict(variant=variant, dataset=ds, model=m, cls=c, adjustment=name) | {f'pooled_{k}': v for k, v in r.items()})
    return rows


# ---------------------------------------------------------------- aggregation
def per_seed(parts, key):
    return pd.concat([p[key].assign(seed=s) for s, p in parts.items()], ignore_index=True)


def fill_fixed(df, models_varies):
    """Copy the seed-42 rows of models that do not vary with the seed into the other seeds."""
    fixed = df[(df.seed == SEED) & ~df.model.isin(models_varies)]
    out = [df] + [fixed.assign(seed=s) for s in SEEDS[1:]]
    return pd.concat(out, ignore_index=True)


def agg(df):
    """mean and SD over seeds of every numeric column, per key (all non-numeric columns)."""
    df = df.copy()
    for c in ('snr_level', 'threshold', 'split', 'experiment', 'value'):
        if c in df:
            df[c] = df[c].astype(str)
    num = [c for c in df.columns if c != 'seed' and pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    keys = [c for c in df.columns if c not in num and c != 'seed']
    g = df.groupby(keys, dropna=False, sort=False)
    out = g[num].mean().add_suffix('_mean').join(g[num].std().add_suffix('_sd'))
    out['n_seeds'] = g.seed.nunique()
    cols = [c for n in num for c in (f'{n}_mean', f'{n}_sd')]
    return out[cols + ['n_seeds']].reset_index()


def main():
    cov = F.covariates()
    parts, rej, models = {}, {'E2': {}, 'E2s': {}}, None
    for s in SEEDS:
        parts[s], ms, r = replicate_seed(s, cov)
        models = ms if models is None else models
        for v in r:
            rej[v][s] = r[v]
        print('seed', s, 'replicate done', flush=True)
    for v in rej:   # identical row order across seeds (same index files)
        assert all((rej[v][s].beat_id.values == rej[v][SEED].beat_id.values).all() for s in SEEDS)
    vary = [m for m in models if m in VARIES]

    # ---- simple tables: per seed (long) + mean/SD
    names = {'perf': 'b_quality_performance', 'thresholds': 'c_thresholds', 'e1': 'e1_raw', 'e2': 'e2_equal_noise', 'e2s': 'e2s_equal_noise',
             'e3': 'e3_breakdown', 'e4': 'e4_by_dataset', 'fix2_clean_normal': 'fix2_clean_normal_rejection', 'fix2_recal_e2': 'fix2_recal_e2',
             'hr_bands': 'fix3_hr_band_gaps'}
    for k, name in names.items():
        df = per_seed(parts, k)
        if k not in ('perf', 'thresholds'):
            df = fill_fixed(df, vary)
        if k == 'perf':
            df = df.drop(columns=['eval_set']).merge(parts[SEED]['perf'][['model', 'eval_set']], on='model')
        df.to_csv(os.path.join(RES, f'{name}_per_seed.csv'), index=False)
        agg(df).to_csv(os.path.join(RES, f'{name}_seeds.csv'), index=False)

    # ---- unadjusted GEE (as f_gee_e2) per seed, Holm per seed
    for v, k in (('e2', 'gee_e2'), ('e2s', 'gee_e2s')):
        df = fill_fixed(per_seed(parts, k), vary)
        df = pd.concat([A.holm(g.reset_index(drop=True), ['dataset', 'cls']) for _, g in df.groupby('seed')], ignore_index=True)
        df.to_csv(os.path.join(RES, f'f_gee_{v}_per_seed.csv'), index=False)
        agg(df.drop(columns=['note', 'cov_struct', 'snr_levels_used'])).to_csv(os.path.join(RES, f'f_gee_{v}_seeds.csv'), index=False)

    # ---- covariate GEE per seed (Holm per seed) + pooled + verdict
    cg = []
    for v, k in (('E2', 'gee_cov_e2'), ('E2s', 'gee_cov_e2s')):
        df = fill_fixed(per_seed(parts, k), vary).assign(variant=v)
        cg.append(pd.concat([A.holm(g.reset_index(drop=True), ['dataset', 'cls', 'adjustment']) for _, g in df.groupby('seed')], ignore_index=True))
    cg = pd.concat(cg, ignore_index=True)
    cg.to_csv(os.path.join(RES, 'gee_covariates_per_seed.csv'), index=False)

    G.update({f'rej_{v}': rej[v] for v in rej})
    with ProcessPoolExecutor(NPROC) as ex:
        pooled = pd.DataFrame([r for rows in ex.map(_pooled, [(m, v) for m in vary for v in rej]) for r in rows])
    keys = ['variant', 'dataset', 'model', 'cls', 'adjustment']
    fixed = cg[(cg.seed == SEED) & ~cg.model.isin(vary)]
    fixed = fixed[keys + ['odds_ratio', 'or_ci_lo', 'or_ci_hi', 'p', 'cov_struct']].rename(
        columns={c: f'pooled_{c}' for c in ('odds_ratio', 'or_ci_lo', 'or_ci_hi', 'p', 'cov_struct')})
    pooled = pd.concat([pooled, fixed], ignore_index=True)
    pooled = A.holm(pooled.reset_index(drop=True), ['variant', 'dataset', 'cls', 'adjustment'], p='pooled_p', out='pooled_p_holm')

    wide = cg.pivot_table(index=keys, columns='seed', values=['odds_ratio', 'p_holm'], aggfunc='first')
    wide.columns = [f'{a}_s{b}' for a, b in wide.columns]
    wide = wide.reset_index()
    st = cg.groupby(keys).agg(or_mean=('odds_ratio', 'mean'), or_sd=('odds_ratio', 'std'),
                              log_or_mean=('odds_ratio', lambda x: np.log(x).mean()),
                              n_path_patients=('n_path_patients', 'first'), n_patients=('n_patients', 'first'),
                              n_sig_pos=('p_holm', lambda x: 0), n_sig_neg=('p_holm', lambda x: 0)).reset_index()
    pos = cg.assign(sp=(cg.p_holm < 0.05) & (cg.odds_ratio > 1), sn=(cg.p_holm < 0.05) & (cg.odds_ratio < 1)).groupby(keys)
    st['n_sig_pos'] = pos.sp.sum().values; st['n_sig_neg'] = pos.sn.sum().values
    out = st.merge(wide, on=keys, how='left').merge(pooled, on=keys, how='left')
    ps = (out.pooled_p_holm < 0.05)
    out['verdict'] = np.select([(out.n_sig_pos >= 2) & ps & (out.pooled_odds_ratio > 1), (out.n_sig_neg >= 2) & ps & (out.pooled_odds_ratio < 1)],
                               ['biased against', 'keeps more'], 'no significant difference')
    out['verdict_code'] = out.verdict.map({'biased against': 1, 'keeps more': -1, 'no significant difference': 0})
    out.to_csv(os.path.join(RES, 'gee_covariates_seeds.csv'), index=False)
    print('seeds analysis done:', len(out), 'GEE cells;', out.verdict.value_counts().to_dict())


if __name__ == '__main__':
    main()
