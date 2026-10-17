"""Aggregate fairness per model: RMS of the pathology-vs-normal rejection gap over the 15 disease / rhythm conditions.

  RMS gap      = sqrt(mean_c gap_c^2)                  (pp; 0 = perfectly fair; sign-agnostic)
  debiased RMS = sqrt(max(0, mean_c gap_c^2 - mean_c SE_c^2))   removes the inflation from sampling noise in small groups
  RMS log-OR   = same on the heart-rate/age-adjusted GEE log odds ratios (reported as a multiplicative factor exp(RMS))
95 % CIs: 1,000 patient bootstrap resamples (patients resampled within each class group, independently), recomputing the RMS.
Exploratory summary added after the pre-specified analysis (see DEVIATIONS.md).
Outputs: results/fairness_rms.csv, figures/paper/fig21_fairness_rms.(png|pdf)
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import RES, FIG, SEED
from analysis import all_scores, thresholds, add_rejections, NORMAL

COND = [('circor', 'murmur_systolic'), ('circor', 'murmur_diastolic'), ('zch', 'ASD'), ('zch', 'VSD'), ('zch', 'PFO'), ('zch', 'PDA'),
        ('bmdhs', 'AS'), ('bmdhs', 'AR'), ('bmdhs', 'MR'), ('bmdhs', 'MS'), ('bmdhs', 'multi'), ('pascal', 'murmur'),
        ('pascal', 'extrasystole'), ('circor', 'synthetic_premature'), ('circor', 'synthetic_irregular')]
MODELS = [('b1_lr', 'B1 Springer SQI'), ('b2_rf', 'B2 Das RF'), ('b3_cardiophon', 'B3 CardioPHON'), ('b4_cnn', 'B4 CNN'),
          ('b5_crnn', 'B5 CRNN'), ('e6_cnn_balanced', 'E6 balanced CNN'), ('b6a', 'B6 PCGKolect'), ('b7', 'B7 SNR rule'),
          ('b8', 'B8 PCGKolect-v3'), ('b8v4', 'B8v4 PCGKolect-v4'), ('b8c', 'B8c PCGKolect-v3c')]
LEVELS = ['none', '20', '10', '5', '0']
NB = 1000


def patient_counts(df, col):
    g = df.groupby('patient_id')[col].agg(['sum', 'size'])
    return g['sum'].values.astype(float), g['size'].values.astype(float)


def boot_rates(r, n, rng):
    idx = rng.integers(0, len(r), (NB, len(r)))
    return r[idx].sum(1) / n[idx].sum(1)


def main():
    e1, e2, models = all_scores()
    thr = thresholds(e1, models)
    e2 = add_rejections(e2, thr, models)
    rng = np.random.default_rng(SEED)
    rows, draws = [], {}
    for m, name in MODELS:
        col = f'{m}__matched'
        for scope, levels in (('clean', ['none']), ('all_levels', LEVELS)):
            gaps, ses, bs = [], [], []
            for ds, c in COND:
                for lv in levels:
                    g = e2[(e2.dataset == ds) & (e2.snr_level == lv)]
                    path, ref = g[g.cls == c], g[g.cls.isin(NORMAL)]
                    if path.empty or ref.empty:
                        continue
                    rp, np_ = patient_counts(path, col); r0, n0 = patient_counts(ref, col)
                    d = boot_rates(rp, np_, rng) - boot_rates(r0, n0, rng)
                    gaps.append(rp.sum() / np_.sum() - r0.sum() / n0.sum()); ses.append(d.std()); bs.append(d)
            gaps, ses, bs = np.array(gaps) * 100, np.array(ses) * 100, np.array(bs) * 100
            rms = np.sqrt(np.mean(gaps ** 2)); deb = np.sqrt(max(0, np.mean(gaps ** 2) - np.mean(ses ** 2)))
            # bootstrap draws of the debiased statistic: E[b^2] = true^2 + 2 SE^2 (observed noise + resampling noise)
            bdeb = np.sqrt(np.clip((bs ** 2).mean(0) - 2 * np.mean(ses ** 2), 0, None))
            draws[(m, scope)] = bdeb
            rows.append(dict(model=m, name=name, scope=scope, n_cells=len(gaps), rms_gap_pp=rms, debiased_rms_gap_pp=deb,
                             debiased_ci_lo=np.percentile(bdeb, 2.5), debiased_ci_hi=np.percentile(bdeb, 97.5), mean_signed_gap_pp=gaps.mean(),
                             max_abs_gap_pp=np.abs(gaps).max()))
    t = pd.DataFrame(rows)
    # adjusted log odds ratios
    a = pd.read_csv(os.path.join(RES, 'fix4_gee_e2_covariates.csv'))
    a = a[((a.dataset.isin(['circor', 'zch'])) & (a.adjustment == 'hr_age')) | ((a.dataset.isin(['bmdhs', 'pascal'])) & (a.adjustment == 'hr'))]
    orow = []
    for m, name in MODELS:
        s = pd.concat([a[(a.model == m) & (a.dataset == ds) & (a.cls == c)] for ds, c in COND])
        s = s[np.isfinite(s.odds_ratio)]
        lo, se = np.log(s.odds_ratio.values), (np.log(s.or_ci_hi.values) - np.log(s.or_ci_lo.values)) / (2 * 1.96)
        r = np.sqrt(np.mean(lo ** 2)); d = np.sqrt(max(0, np.mean(lo ** 2) - np.mean(se ** 2)))
        orow.append(dict(model=m, n_or=len(lo), rms_log_or=r, rms_or_factor=np.exp(r), debiased_rms_or_factor=np.exp(d)))
    t = t.merge(pd.DataFrame(orow), on='model', how='left')
    # paired bootstrap: is each model's clean-beat RMS larger than B2's?
    base = draws[('b2_rf', 'clean')]
    t['p_rms_not_gt_b2_clean'] = [np.mean(draws[(m, 'clean')] <= base) if s == 'clean' else np.nan for m, s in zip(t.model, t.scope)]
    b6 = draws[('b6a', 'clean')]; b6all = draws[('b6a', 'all_levels')]
    t['p_rms_not_lt_b6'] = [np.mean(draws[(m, s)] >= (b6 if s == 'clean' else b6all)) for m, s in zip(t.model, t.scope)]
    t['rms_reduction_vs_b6'] = [1 - r / t[(t.model == 'b6a') & (t.scope == s)].debiased_rms_gap_pp.iloc[0] for r, s in zip(t.debiased_rms_gap_pp, t.scope)]
    t.to_csv(os.path.join(RES, 'fairness_rms.csv'), index=False)
    pd.set_option('display.width', 230)
    print(t[t.scope == 'clean'][['name', 'rms_gap_pp', 'debiased_rms_gap_pp', 'debiased_ci_lo', 'debiased_ci_hi', 'mean_signed_gap_pp', 'rms_or_factor',
                                 'debiased_rms_or_factor', 'p_rms_not_gt_b2_clean']].round(3).to_string(index=False))
    print(t[t.scope == 'all_levels'][['name', 'rms_gap_pp', 'debiased_rms_gap_pp', 'debiased_ci_lo', 'debiased_ci_hi']].round(2).to_string(index=False))
    plot(t)


def plot(t):
    INK, INK2, GRID, SURF = '#1f1f1d', '#5c5b57', '#e4e3df', '#fcfcfb'
    prop = {'b8', 'b8v4', 'b8c'}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True, facecolor=SURF)
    c = t[t.scope == 'clean'].sort_values('debiased_rms_gap_pp', ascending=False).reset_index(drop=True)
    order = list(c.model)
    y = np.arange(len(order))
    for ax, (scope, col, lo, hi, title, xl) in zip(axes, (
            ('clean', 'debiased_rms_gap_pp', 'debiased_ci_lo', 'debiased_ci_hi', '(a) RMS rejection gap, clean beats', 'noise-corrected RMS gap, 15 conditions (pp)'),
            ('all_levels', 'debiased_rms_gap_pp', 'debiased_ci_lo', 'debiased_ci_hi', '(b) RMS gap, all noise levels', 'noise-corrected RMS gap, 15 conditions × 5 levels (pp)'),
            ('clean', 'rms_or_factor', None, None, '(c) RMS of adjusted odds ratios', 'typical OR distance from 1 (×, HR/age-adjusted)'))):
        d = t[t.scope == scope].set_index('model').loc[order]
        cols = ['#e34948' if m in prop else '#2a78d6' for m in order]
        col_ = col
        if lo:
            ax.hlines(y, d[lo], d[hi], color=cols, lw=2, alpha=0.6)
        ax.scatter(d[col], y, color=cols, s=55, zorder=3, edgecolor=SURF, linewidth=1.5)
        if col == 'debiased_rms_gap_pp':
            ax.scatter(d['rms_gap_pp'], y, marker='|', color=INK, s=120, zorder=4, label='raw RMS (inflated by small groups)')
        for yy, v in zip(y, d[col]):
            ax.text(v, yy + 0.32, f'{v:.2f}' if col == 'rms_or_factor' else f'{v:.1f}', ha='center', fontsize=7.5, color=INK2)
        ax.set_title(title, loc='left', fontweight='bold', fontsize=10.5); ax.set_xlabel(xl, color=INK2)
        ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.set_facecolor(SURF)
    axes[0].set_yticks(y, c.name); axes[0].legend(fontsize=8, loc='upper right', frameon=False)
    fig.suptitle('Aggregate fairness: RMS of the pathological-minus-normal rejection gap over all 15 disease / rhythm conditions (lower = fairer)',
                 x=0.01, ha='left', fontweight='bold', fontsize=11.5)
    fig.text(0.01, 0.01, 'Dots: noise-corrected RMS (sampling variance of small groups removed) with 95 % patient-bootstrap CI; bars "|": raw RMS. '
             'Blue = existing filters, red = PCGKolect-v3 variants. Matched thresholds; E2 equal-noise protocol. Exploratory summary.',
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG, 'paper', f'fig21_fairness_rms.{ext}'), dpi=170, facecolor=SURF)
    plt.close(fig)


if __name__ == '__main__':
    main()
