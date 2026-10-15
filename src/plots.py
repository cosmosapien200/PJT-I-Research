"""Figures: training curves, E2 rejection vs SNR, E3 by murmur grade/timing, E5 murmur sensitivity."""
import os, glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import LOGT, RES, FIG

ORDER = ['none', '20', '10', '5', '0']
NAMES = {'b1_lr': 'B1 LR', 'b1_svm': 'B1 SVM', 'b2_rf': 'B2 RF', 'b4_cnn': 'B4 CNN', 'b5_crnn': 'B5 CRNN',
         'e6_cnn_balanced': 'E6 balanced CNN', 'b6a': 'B6a', 'b6b': 'B6b', 'b7': 'B7 SNR'}


def training_curves():
    fs = sorted(glob.glob(os.path.join(LOGT, '*.csv')))
    deep = [f for f in fs if not f.endswith('_cv.csv')]
    cv = [f for f in fs if f.endswith('_cv.csv')]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for f in deep:
        d = pd.read_csv(f); n = os.path.basename(f)[:-4]
        l, = ax[0].plot(d.epoch, d.train_loss, '--', lw=1)
        ax[0].plot(d.epoch, d.val_loss, '-', color=l.get_color(), lw=1.5, label=n)
        ax[1].plot(d.epoch, d.val_auroc, color=l.get_color(), label=n)
    ax[0].set(title='Loss (dashed train, solid val)', xlabel='epoch'); ax[0].legend(fontsize=6)
    ax[1].set(title='Validation AUROC', xlabel='epoch'); ax[1].legend(fontsize=6)
    for i, f in enumerate(cv):
        d = pd.read_csv(f)
        ax[2].scatter(np.full(len(d), i), d.auroc, s=15)
    ax[2].set_xticks(range(len(cv)), [os.path.basename(f)[:-7] for f in cv])
    ax[2].set(title='Classic models: per-fold AUROC')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'training_curves.png'), dpi=130); plt.close(fig)


def e2_figure():
    t = pd.read_csv(os.path.join(RES, 'e2_equal_noise.csv'))
    t = t[t.threshold == 'matched']; t['snr_level'] = t.snr_level.astype(str)
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
        fig.suptitle(f'E2 equal noise, {ds} (matched threshold, 95% patient-bootstrap CI)')
        fig.tight_layout(); fig.savefig(os.path.join(FIG, f'e2_{ds}.png'), dpi=120); plt.close(fig)


def e3_figure():
    t = pd.read_csv(os.path.join(RES, 'e3_breakdown.csv'))
    t = t[(t.dataset == 'circor') & t.breakdown.isin(['sys_grade', 'sys_timing']) & (t.experiment == 'E1')]
    if t.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
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
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(t))
    ax.bar(x - 0.2, t.sensitivity, 0.4, label='sensitivity'); ax.bar(x + 0.2, t.specificity, 0.4, label='specificity')
    ax.plot(x, t.auroc, 'k^', label='AUROC')
    ax.set_xticks(x, [NAMES.get(m, m) for m in t.filter_model], rotation=30)
    ax.set(ylim=(0, 1), title='E5: patient-level murmur detection, CirCor test 40 %, by training-set quality filter')
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(FIG, 'e5_downstream.png'), dpi=120); plt.close(fig)


if __name__ == '__main__':
    training_curves()
    for f in (e2_figure, e3_figure, e5_figure):
        try:
            f()
        except FileNotFoundError:
            pass
    print('figures done')
