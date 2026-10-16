"""Summary figures: model x heartbeat-condition matrices.

  figures/summary_1_gap_heatmap.png     rejection gap (pathological minus normal, pp) on clean beats, equal conditions
  figures/summary_2_verdict_matrix.png  significant over-/under-rejection after heart-rate (+age) adjustment
  figures/summary_3_scorecard.png       per model: how many conditions it over-rejects / treats fairly / under-rejects
Data: results/e2_equal_noise.csv, results/f_gee_e2.csv, results/fix4_gee_e2_covariates.csv
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
from common import RES, FIG

BLUE, RED, MID = '#2a78d6', '#e34948', '#f0efec'
INK, INK2, GRID, SURF = '#1f1f1d', '#5c5b57', '#d9d8d4', '#fcfcfb'
CMAP = LinearSegmentedColormap.from_list('div', [BLUE, MID, RED])

MODELS = [('b1_lr', 'B1  Springer SQI (LR)'), ('b2_rf', 'B2  Das RF'), ('b3_cardiophon', 'B3  CardioPHON'),
          ('b4_cnn', 'B4  CNN'), ('b5_crnn', 'B5  CRNN'), ('b6a', 'B6  PCGKolect U-Net+HMM'), ('b7', 'B7  SNR rule')]
COLS = [  # (dataset, class, label, group)
    ('circor', 'murmur_systolic', 'Systolic\nmurmur', 'CirCor (children, murmurs)'),
    ('circor', 'murmur_diastolic', 'Diastolic\nmurmur*', 'CirCor (children, murmurs)'),
    ('zch', 'ASD', 'ASD', 'ZCHSound (children, congenital)'),
    ('zch', 'VSD', 'VSD', 'ZCHSound (children, congenital)'),
    ('zch', 'PFO', 'PFO', 'ZCHSound (children, congenital)'),
    ('zch', 'PDA', 'PDA', 'ZCHSound (children, congenital)'),
    ('bmdhs', 'AS', 'AS', 'BMD-HS (adults, valve)†'),
    ('bmdhs', 'AR', 'AR', 'BMD-HS (adults, valve)†'),
    ('bmdhs', 'MR', 'MR', 'BMD-HS (adults, valve)†'),
    ('bmdhs', 'MS', 'MS', 'BMD-HS (adults, valve)†'),
    ('bmdhs', 'multi', 'Multi-\nvalve', 'BMD-HS (adults, valve)†'),
    ('pascal', 'murmur', 'Murmur', 'PASCAL (adults)†'),
    ('pascal', 'extrasystole', 'Extra-\nsystole', 'PASCAL (adults)†'),
    ('circor', 'synthetic_premature', 'Premature\nbeats', 'Synthetic rhythm'),
    ('circor', 'synthetic_irregular', 'AF-like\nirregular', 'Synthetic rhythm'),
]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'text.color': INK, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK, 'figure.facecolor': SURF, 'axes.facecolor': SURF})


def groups_axis(ax, y):
    """Dataset group labels and separators above the columns."""
    spans, start = [], 0
    for i in range(1, len(COLS) + 1):
        if i == len(COLS) or COLS[i][3] != COLS[start][3]:
            spans.append((start, i - 1, COLS[start][3])); start = i
    for a, b, name in spans:
        ax.annotate(name, xy=((a + b) / 2, y), xycoords=('data', 'axes fraction'), ha='center', va='bottom',
                    fontsize=9, color=INK2, annotation_clip=False)
        ax.plot([a - 0.4, b + 0.4], [y - 0.01, y - 0.01], transform=ax.get_xaxis_transform(), color=GRID, lw=1.5, clip_on=False)
        if a > 0:
            ax.axvline(a - 0.5, color=SURF, lw=5)


def ink_for(rgb):
    lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return INK if lum > 0.55 else 'white'


def load():
    e2 = pd.read_csv(os.path.join(RES, 'e2_equal_noise.csv'))
    e2 = e2[(e2.threshold == 'matched') & (e2.snr_level.astype(str) == 'none')]
    g = pd.read_csv(os.path.join(RES, 'f_gee_e2.csv'))
    a = pd.read_csv(os.path.join(RES, 'fix4_gee_e2_covariates.csv'))
    # most complete identifiable adjustment: HR + age where age varies (CirCor, ZCHSound); HR only for BMD-HS
    # (all healthy subjects are 20-29, age not identifiable) and PASCAL (no age)
    a = a[((a.dataset.isin(['circor', 'zch'])) & (a.adjustment == 'hr_age')) |
          ((a.dataset.isin(['bmdhs', 'pascal'])) & (a.adjustment == 'hr'))]
    return e2, g, a


def matrix(df, val):
    M = np.full((len(MODELS), len(COLS)), np.nan)
    for i, (m, _) in enumerate(MODELS):
        for j, (ds, c, _, _) in enumerate(COLS):
            r = df[(df.model == m) & (df.dataset == ds) & (df.cls == c)]
            if len(r):
                M[i, j] = r[val].iloc[0]
    return M


def fig1(e2, g):
    D = matrix(e2, 'diff_vs_normal') * 100
    lo, hi = matrix(e2, 'diff_ci_lo') * 100, matrix(e2, 'diff_ci_hi') * 100
    P = matrix(g, 'p_holm')
    fig, ax = plt.subplots(figsize=(15.5, 6.4))
    norm = TwoSlopeNorm(vcenter=0, vmin=-30, vmax=30)
    ax.imshow(np.clip(D, -30, 30), cmap=CMAP, norm=norm, aspect='auto')
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            if np.isnan(D[i, j]):
                continue
            col = ink_for(CMAP(norm(np.clip(D[i, j], -30, 30)))[:3])
            sig = P[i, j] < 0.05
            ax.text(j, i - 0.08, f'{D[i, j]:+.0f}', ha='center', va='center', fontsize=11,
                    fontweight='bold' if sig else 'normal', color=col)
            ax.text(j, i + 0.27, '● sig.' if sig else ('○ CI excl. 0' if (lo[i, j] > 0 or hi[i, j] < 0) else ''),
                    ha='center', va='center', fontsize=7, color=col)
    ax.set_xticks(range(len(COLS)), [c[2] for c in COLS], fontsize=9)
    ax.set_yticks(range(len(MODELS)), [m[1] for m in MODELS], fontsize=10)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(COLS)), minor=True); ax.set_yticks(np.arange(-.5, len(MODELS)), minor=True)
    ax.grid(which='minor', color=SURF, lw=2); ax.tick_params(which='minor', length=0)
    groups_axis(ax, 1.02)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), ax=ax, fraction=0.025, pad=0.01)
    cb.set_label('extra rejection vs normal beats (percentage points)', color=INK2)
    cb.set_ticks([-30, -15, 0, 15, 30], labels=['−30\nkeeps more', '−15', '0\nfair', '+15', '+30\nremoves more'])
    cb.outline.set_visible(False)
    fig.suptitle('Which filter removes which kind of heartbeat?  Clean beats, same conditions, threshold = 10 % of normal beats rejected',
                 x=0.01, ha='left', fontsize=13, fontweight='bold', y=0.995)
    fig.text(0.01, 0.015, 'Red = the filter throws away this pathology MORE often than normal beats (bad: biased against it).  '
             'Blue = it throws it away LESS often.  Grey = no difference.\n'
             'Number = rejection-rate gap in percentage points on clean beats.  Bold + "● sig." = significant over all noise levels '
             '(GEE, Holm-corrected across models); "○ CI excl. 0" = 95 % patient-bootstrap CI of this cell excludes 0, but not significant overall.\n'
             '* diastolic murmur: only 5 patients.  † BMD-HS / PASCAL: small calibration sets (4 / 18 normal patients); treat as supportive.',
             fontsize=8.5, color=INK2, va='bottom')
    fig.tight_layout(rect=(0, 0.075, 1, 0.95))
    fig.savefig(os.path.join(FIG, 'summary_1_gap_heatmap.png'), dpi=150)
    plt.close(fig)


def verdict(a):
    OR, P = matrix(a, 'odds_ratio'), matrix(a, 'p_holm')
    V = np.where(np.isnan(OR), np.nan, np.where((P < 0.05) & (OR > 1), 1, np.where((P < 0.05) & (OR < 1), -1, 0)))
    return V, OR


def fig2(a):
    V, OR = verdict(a)
    colors = {1: RED, 0: '#e4e3df', -1: BLUE}
    fig, ax = plt.subplots(figsize=(15.5, 6.0))
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            v = V[i, j]
            fc = '#ffffff' if np.isnan(v) else colors[int(v)]
            ax.add_patch(plt.Rectangle((j - 0.47, i - 0.45), 0.94, 0.9, facecolor=fc, edgecolor='none'))
            if np.isnan(v):
                ax.text(j, i, 'n/a', ha='center', va='center', fontsize=8, color=INK2); continue
            word = {1: 'REMOVES\nMORE', 0: 'fair', -1: 'keeps\nmore'}[int(v)]
            col = 'white' if v != 0 else INK2
            ax.text(j, i - 0.12, word, ha='center', va='center', fontsize=8 if v else 9, fontweight='bold' if v else 'normal', color=col)
            ax.text(j, i + 0.28, f'OR {OR[i, j]:.2f}', ha='center', va='center', fontsize=7.5, color=col)
    ax.set_xlim(-0.5, len(COLS) - 0.5); ax.set_ylim(len(MODELS) - 0.5, -0.5)
    ax.set_xticks(range(len(COLS)), [c[2] for c in COLS], fontsize=9)
    ax.set_yticks(range(len(MODELS)), [m[1] for m in MODELS], fontsize=10)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    groups_axis(ax, 1.02)
    ax.legend(handles=[Patch(color=RED, label='removes this pathology MORE than normal (significant)'),
                       Patch(color='#e4e3df', label='no significant difference (fair)'),
                       Patch(color=BLUE, label='removes it LESS than normal (significant)')],
              loc='upper left', bbox_to_anchor=(0, -0.13), ncol=3, frameon=False, fontsize=9)
    fig.suptitle('Verdict after correcting for heart rate and age:  is the filter biased against each heartbeat type?',
                 x=0.01, ha='left', fontsize=13, fontweight='bold', y=0.995)
    fig.text(0.01, 0.01, 'GEE logistic regression over all noise levels (clean, 20, 10, 5, 0 dB), rejected ~ class + noise level + heart rate (+ age), '
             'clustered by patient; Holm correction across the 10 models; OR = odds ratio vs normal beats.\n'
             'Age adjustment for CirCor and ZCHSound only (BMD-HS healthy subjects are all 20-29 years, PASCAL has no age).',
             fontsize=8.5, color=INK2, va='bottom')
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(os.path.join(FIG, 'summary_2_verdict_matrix.png'), dpi=150)
    plt.close(fig)
    return V


def fig3(V):
    names = [m[1] for m in MODELS]
    over = np.nansum(V == 1, 1); fair = np.nansum(V == 0, 1); under = np.nansum(V == -1, 1)
    order = np.argsort(-over * 100 - under)[::-1]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    y = np.arange(len(names))
    for k, (vals, col, lab) in enumerate(((over, RED, 'removes pathology MORE (biased)'), (fair, '#cfcdc8', 'fair'),
                                         (under, BLUE, 'removes pathology LESS'))):
        left = (0 if k == 0 else (over if k == 1 else over + fair))[order] if k else np.zeros(len(names))
        ax.barh(y, vals[order], left=left, color=col, height=0.62, edgecolor=SURF, linewidth=2, label=lab)
        for yy, v, l0 in zip(y, vals[order], left):
            if v:
                ax.text(l0 + v / 2, yy, str(int(v)), ha='center', va='center', fontsize=9,
                        color='white' if col != '#cfcdc8' else INK)
    ax.set_yticks(y, [names[i] for i in order]); ax.tick_params(length=0)
    ax.set_xlabel('number of heartbeat conditions (out of 15)')
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, fontsize=9)
    fig.suptitle('Scorecard: for how many heartbeat types is each filter biased?  (after heart-rate/age correction)',
                 x=0.01, ha='left', fontsize=12, fontweight='bold')
    fig.text(0.01, 0.01, 'Most biased at the top, least biased (B2) at the bottom. Counts from the verdict matrix (summary_2). n/a cells are not counted.',
             fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(os.path.join(FIG, 'summary_3_scorecard.png'), dpi=150)
    plt.close(fig)
    pd.DataFrame(dict(model=names, over=over, fair=fair, under=under)).to_csv(os.path.join(RES, 'summary_scorecard.csv'), index=False)


if __name__ == '__main__':
    e2, g, a = load()
    fig1(e2, g)
    V = fig2(a)
    fig3(V)
    print('summary figures done')
