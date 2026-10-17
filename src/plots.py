"""Figures: training curves, E2 / E2s rejection vs SNR, E3 by murmur grade/timing, E5 murmur sensitivity.
Seeded models (B2-B5, E6) and hybrids: mean of seeds 42-44 (results/*_seeds.csv)."""
import os, glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import LOGT, RES, FIG

ORDER = ['none', '20', '10', '5', '0']
NAMES = {'b1_lr': 'B1 LR', 'b1_svm': 'B1 SVM', 'b2_rf': 'B2 RF', 'b3_cardiophon': 'B3 CardioPHON', 'b4_cnn': 'B4 CNN', 'b5_crnn': 'B5 CRNN',
         'e6_cnn_balanced': 'E6 balanced CNN', 'b6a': 'B6a', 'b6b': 'B6b', 'b7': 'B7 SNR', 'b6pna': 'B6pn-a (PN only)', 'b6pnb': 'B6pn-b (PN only)', 'b8': 'B8 PCGKolect-v3', 'b8noaug': 'B8 no-aug (ablation)', 'b8v4': 'B8v4 PCGKolect-v4', 'b8c': 'B8c PCGKolect-v3c',
         'h_and_b4': 'H-AND Das∧B4', 'h_and_b5': 'H-AND Das∧B5', 'h_cas_b4': 'H-CASCADE Das→B4', 'h_cas_b5': 'H-CASCADE Das→B5',
         'h_fuse_b4': 'H-FUSE Das+B4', 'h_fuse_b5': 'H-FUSE Das+B5'}


def seed_mean(name):
    d = pd.read_csv(os.path.join(RES, f'{name}_seeds.csv'))
    return d.rename(columns={c: c[:-5] for c in d.columns if c.endswith('_mean')})


def training_curves():
    import re
    fs = sorted(glob.glob(os.path.join(LOGT, '*.csv')))
    cv = [f for f in fs if f.endswith('_cv.csv')]
    deep = [f for f in fs if not f.endswith('_cv.csv') and 'e5_' not in f and 'epoch' in pd.read_csv(f, nrows=0).columns]   # B8 logs are per step
    rep_ = lambda f: re.search(r'_s\d+(_f\d)?\.csv$|_f\d\.csv$', os.path.basename(f))
    groups = [('quality models (seed 42)', [f for f in deep if not rep_(f)]),
              ('seed replicates (s43, s44) and fold models (_f1-4)', [f for f in deep if rep_(f)]),
              ('E5 murmur classifiers', [f for f in fs if 'e5_' in os.path.basename(f)])]
    fig, ax = plt.subplots(3, 3, figsize=(17, 13))
    cmap = plt.get_cmap('tab10')
    for r, (title, files) in enumerate(groups):
        for i, f in enumerate(files):
            d = pd.read_csv(f); n = os.path.basename(f)[:-4].replace('e5_murmur_', '')
            c = cmap(i % 10)
            ax[r, 0].plot(d.epoch, d.train_loss, '--', color=c, lw=1)
            ax[r, 0].plot(d.epoch, d.val_loss, '-', color=c, lw=1.5, label=n)
            ax[r, 1].plot(d.epoch, d.val_auroc, color=c, label=n)
        leg = len(files) <= 16
        ax[r, 0].set(title=f'{title}: loss (dashed train, solid val)', xlabel='epoch')
        ax[r, 1].set(title=f'{title}: validation AUROC', xlabel='epoch')
        if leg:
            ax[r, 0].legend(fontsize=7); ax[r, 1].legend(fontsize=7)
    for i, f in enumerate(cv):
        d = pd.read_csv(f)
        ax[0, 2].scatter(np.full(len(d), i), d.auroc, s=15)
    ax[0, 2].set_xticks(range(len(cv)), [os.path.basename(f)[:-7] for f in cv])
    ax[0, 2].set(title='Classic models: per-fold AUROC'); ax[0, 2].tick_params(axis='x', labelrotation=45, labelsize=7)
    ax[1, 2].axis('off'); ax[2, 2].axis('off')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'training_curves.png'), dpi=120); plt.close(fig)


def e2_figure(variant='e2'):
    t = seed_mean(f'{variant}_equal_noise')
    t = t[(t.threshold == 'matched') & ~t.cls.str.endswith('_vs_regular')]; t['snr_level'] = t.snr_level.astype(str)
    for ds, g in t.groupby('dataset'):
        models = [m for m in NAMES if m in g.model.unique()]
        n = len(models); c = 3; r = int(np.ceil(n / c))
        fig, axes = plt.subplots(r, c, figsize=(4.6 * c, 3.4 * r), sharey=True, squeeze=False)
        for ax, m in zip(axes.flat, models):
            gm = g[g.model == m]
            for cl, gc in gm.groupby('cls'):
                gc = gc.set_index('snr_level').reindex(ORDER)
                x = np.arange(len(ORDER))
                ax.errorbar(x, gc.rate, yerr=[gc.rate - gc.ci_lo, gc.ci_hi - gc.rate], marker='o', ms=3, capsize=2,
                            lw=2.5 if cl in ('normal', 'control') else 1.2, label=cl)
            ax.set_xticks(range(len(ORDER)), ['clean', '20', '10', '5', '0'])
            ax.set(title=NAMES[m], xlabel='added noise SNR (dB)', ylabel='rejection rate', ylim=(0, 1))
        for ax in axes.flat[n:]:
            ax.axis('off')
        axes.flat[0].legend(fontsize=7)
        ref = 'window power' if variant == 'e2' else 'S1+S2 power (E2s)'
        fig.suptitle(f'{variant.upper()} equal noise, {ds} (SNR vs clean {ref}; matched threshold; seeded models: mean of 3 seeds, seed-averaged 95% patient-bootstrap CI)')
        fig.tight_layout(); fig.savefig(os.path.join(FIG, f'{variant}_{ds}.png'), dpi=110); plt.close(fig)


def e3_figure():
    t = seed_mean('e3_breakdown')
    t = t[(t.dataset == 'circor') & t.breakdown.isin(['sys_grade', 'sys_timing']) & (t.experiment == 'E1')]
    if t.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    for ax, var in zip(axes, ['sys_grade', 'sys_timing']):
        g = t[t.breakdown == var]
        vals = sorted(g.value.unique()); models = [m for m in NAMES if m in g.model.unique()]
        w = 0.8 / len(models)
        for i, m in enumerate(models):
            gm = g[g.model == m].set_index('value').reindex(vals)
            ax.bar(np.arange(len(vals)) + i * w, gm.diff_vs_normal, w, label=NAMES[m],
                   yerr=[gm.diff_vs_normal - gm.diff_ci_lo, gm.diff_ci_hi - gm.diff_vs_normal], capsize=1, error_kw=dict(lw=0.6))
        ax.axhline(0, color='k', lw=0.8)
        ax.set_xticks(np.arange(len(vals)) + 0.4 - w / 2, vals)
        ax.set(title=f'CirCor systolic murmur beats by {var} (E1, matched)', ylabel='rejection rate minus normal')
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'e3_circor_grade_timing.png'), dpi=120); plt.close(fig)


def e5_figure():
    p = os.path.join(RES, 'e5_downstream.csv')
    if not os.path.exists(p):
        return
    t = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(14, 4.4))
    x = np.arange(len(t))
    ax.bar(x - 0.27, t.auroc_mean, 0.27, yerr=t.auroc_sd, capsize=2, label='AUROC')
    ax.bar(x, t.sensitivity_mean, 0.27, yerr=t.sensitivity_sd, capsize=2, label='sensitivity')
    ax.bar(x + 0.27, t.specificity_mean, 0.27, yerr=t.specificity_sd, capsize=2, label='specificity')
    ax.set_xticks(x, [NAMES.get(m, m) for m in t.filter_model], rotation=30)
    ax.set(ylim=(0.5, 1), title='E5: patient-level murmur detection (mean over beats), CirCor test 40 %, mean +- SD over runs (hybrids: 3 filter x 3 E5 seeds)')
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(FIG, 'e5_downstream.png'), dpi=120); plt.close(fig)


if __name__ == '__main__':
    training_curves()
    for f in (e2_figure, lambda: e2_figure('e2s'), e3_figure, e5_figure):
        try:
            f()
        except FileNotFoundError:
            pass
    print('figures done')
