"""Summary figures: model x heartbeat-condition matrices (seed-robust, 3 seeds; see seeds.py).

  figures/summary_1_gap_heatmap.png      rejection gap (pathological minus normal, pp, mean of 3 seeds) on clean beats
  figures/summary_2_verdict_matrix.png   seed-robust over-/under-rejection after heart-rate (+age) adjustment (E2)
  figures/summary_2b_verdict_matrix_e2s.png   same with the S1+S2-referenced noise level (E2s)
  figures/summary_3_scorecard.png        per model: how many conditions it over-rejects / no significant difference / under-rejects
Data: results/e2_equal_noise_per_seed.csv, results/gee_covariates_seeds.csv
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
SEEDED = {'b2_rf', 'b3_cardiophon', 'b4_cnn', 'b5_crnn', 'e6_cnn_balanced'}

MODELS = [('b1_lr', 'B1  Springer SQI (LR)'), ('b2_rf', 'B2  Das RF'), ('b3_cardiophon', 'B3  CardioPHON'),
          ('b4_cnn', 'B4  CNN'), ('b5_crnn', 'B5  CRNN'), ('e6_cnn_balanced', 'E6  balanced CNN'), ('b6a', 'B6  PCGKolect U-Net+HMM'),
          ('b7', 'B7  SNR rule'), ('b8', 'B8  PCGKolect-v3 (proposed)'), ('b8v4', 'B8v4  PCGKolect-v4'), ('b8c', 'B8c  PCGKolect-v3c'),
          ('h_and_b4', 'H-AND  Das ∧ B4'), ('h_and_b5', 'H-AND  Das ∧ B5'), ('h_cas_b4', 'H-CASCADE  Das → B4'),
          ('h_cas_b5', 'H-CASCADE  Das → B5'), ('h_fuse_b4', 'H-FUSE  Das + B4 emb.'), ('h_fuse_b5', 'H-FUSE  Das + B5 emb.'),
          ('hd_avg', 'HD1  avg(Das, B4)'), ('hd_and', 'HD2  min(Das, B4)'), ('hd_lr', 'HD3  LR Das + B4 emb.'),
          ('hd_h4_l01', 'HD4  λ0.1 (chosen)'), ('hd_h4_l03', 'HD4  λ0.3'), ('hd_h4_l10', 'HD4  λ1.0')]
HYB = {m for m, _ in MODELS if m.startswith(('h_', 'hd_'))}
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
# post hoc construction control (not among the 15 pre-registered conditions, not counted in the scorecard)
COLS_REG = [('circor', 'synthetic_premature_vs_regular', 'Premature\nvs regular', 'vs synthetic regular‡'),
            ('circor', 'synthetic_irregular_vs_regular', 'AF-like\nvs regular', 'vs synthetic regular‡')]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'text.color': INK, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK, 'figure.facecolor': SURF, 'axes.facecolor': SURF})


def adjusted(a):
    """Most complete identifiable adjustment: HR + age where age varies (CirCor, ZCHSound); HR only for BMD-HS (all healthy
    subjects are 20-29, age not identifiable) and PASCAL (no age)."""
    return a[((a.dataset.isin(['circor', 'zch'])) & (a.adjustment == 'hr_age')) | ((a.dataset.isin(['bmdhs', 'pascal'])) & (a.adjustment == 'hr'))]


def groups_axis(ax, y, cols):
    spans, start = [], 0
    for i in range(1, len(cols) + 1):
        if i == len(cols) or cols[i][3] != cols[start][3]:
            spans.append((start, i - 1, cols[start][3])); start = i
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
    e2 = pd.read_csv(os.path.join(RES, 'e2_equal_noise_per_seed.csv'))
    e2 = e2[(e2.threshold == 'matched') & (e2.snr_level.astype(str) == 'none')]
    a = pd.read_csv(os.path.join(RES, 'gee_covariates_seeds.csv'))
    return e2, a


def matrix(df, val, cols=COLS):
    M = np.full((len(MODELS), len(cols)), np.nan)
    for i, (m, _) in enumerate(MODELS):
        for j, (ds, c, _, _) in enumerate(cols):
            r = df[(df.model == m) & (df.dataset == ds) & (df.cls == c)]
            if len(r):
                M[i, j] = r[val].iloc[0]
    return M


def ytick_labels(ax):
    ax.set_yticks(range(len(MODELS)), [m[1] + ('  (3 seeds)' if (m[0] in SEEDED or m[0] in HYB) else '') for m in MODELS], fontsize=9.5)
    for t, (m, _) in zip(ax.get_yticklabels(), MODELS):
        if m in HYB:
            t.set_color('#0f8a8a' if m.startswith('hd_') else '#4a3aa7')


def fig1(e2, a):
    cols = COLS + COLS_REG
    mean = e2.groupby(['model', 'dataset', 'cls'])[['diff_vs_normal']].mean().reset_index()
    D = matrix(mean, 'diff_vs_normal', cols) * 100
    # CI excludes 0 in >= 2 of 3 seeds (same direction)
    e2 = e2.assign(pos=e2.diff_ci_lo > 0, neg=e2.diff_ci_hi < 0)
    ci = e2.groupby(['model', 'dataset', 'cls']).agg(npos=('pos', 'sum'), nneg=('neg', 'sum')).reset_index()
    ci['ci2'] = ((ci.npos >= 2) | (ci.nneg >= 2)).astype(float)
    CI = matrix(ci, 'ci2', cols)
    V = matrix(a[(a.variant == 'E2') & (a.adjustment == 'unadjusted')], 'verdict_code', cols)
    fig, ax = plt.subplots(figsize=(17.5, 9.4 * len(MODELS) / 17))
    norm = TwoSlopeNorm(vcenter=0, vmin=-30, vmax=30)
    ax.imshow(np.clip(D, -30, 30), cmap=CMAP, norm=norm, aspect='auto')
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            if np.isnan(D[i, j]):
                ax.text(j, i, 'n/a', ha='center', va='center', fontsize=7.5, color=INK2); continue
            col = ink_for(CMAP(norm(np.clip(D[i, j], -30, 30)))[:3])
            sig = np.isfinite(V[i, j]) and V[i, j] != 0
            ax.text(j, i - 0.1, f'{D[i, j]:+.0f}', ha='center', va='center', fontsize=10.5, fontweight='bold' if sig else 'normal', color=col)
            ax.text(j, i + 0.28, '● robust' if sig else ('○ CI≠0' if CI[i, j] == 1 else ''), ha='center', va='center', fontsize=6.5, color=col)
    ax.set_xticks(range(len(cols)), [c[2] for c in cols], fontsize=8.5)
    ytick_labels(ax)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(cols)), minor=True); ax.set_yticks(np.arange(-.5, len(MODELS)), minor=True)
    ax.grid(which='minor', color=SURF, lw=2); ax.tick_params(which='minor', length=0)
    ax.axhline(len(MODELS) - len(HYB) - 0.5, color=INK2, lw=1.2)
    groups_axis(ax, 1.015, cols)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), ax=ax, fraction=0.02, pad=0.01)
    cb.set_label('extra rejection vs normal beats (percentage points)', color=INK2)
    cb.set_ticks([-30, -15, 0, 15, 30], labels=['−30\nkeeps more', '−15', '0\nno sig.\ndifference', '+15', '+30\nremoves more'])
    cb.outline.set_visible(False)
    fig.suptitle('Which filter removes which kind of heartbeat?  Clean beats, same conditions, threshold = 10 % of normal beats rejected (mean of 3 seeds)',
                 x=0.01, ha='left', fontsize=13, fontweight='bold', y=0.995)
    fig.text(0.01, 0.012, 'Red = the filter throws away this pathology MORE often than normal beats (biased against it).  '
             'Blue = it throws it away LESS often.  Grey = no difference.  Purple rows = hybrids (Das periodicity + deep model).\n'
             'Number = rejection-rate gap in percentage points on clean beats, mean over seeds 42-44 (single value for models without seeds).  '
             'Bold + "● robust" = seed-robust significant over all noise levels (unadjusted GEE, Holm across models, significant in ≥ 2 of 3 seeds AND in the pooled fit);\n'
             '"○ CI≠0" = the 95 % patient-bootstrap CI of this cell excludes 0 in ≥ 2 of 3 seeds, not seed-robust overall.  '
             '* diastolic murmur: 5 patients.  † small calibration sets (4 / 18 normal patients).  ‡ post-hoc construction control (synthetic beat vs synthetic regular beat).',
             fontsize=8, color=INK2, va='bottom')
    fig.tight_layout(rect=(0, 0.07, 1, 0.955))
    fig.savefig(os.path.join(FIG, 'summary_1_gap_heatmap.png'), dpi=150)
    plt.close(fig)


def fig2(a, variant='E2', name='summary_2_verdict_matrix.png'):
    cols = COLS + COLS_REG
    a = adjusted(a[a.variant == variant])
    V, OR = matrix(a, 'verdict_code', cols), matrix(a, 'pooled_odds_ratio', cols)
    N, SP, SN = matrix(a, 'n_path_patients', cols), matrix(a, 'n_sig_pos', cols), matrix(a, 'n_sig_neg', cols)
    colors = {1: RED, 0: '#e4e3df', -1: BLUE}
    fig, ax = plt.subplots(figsize=(18, 10.2 * len(MODELS) / 17))
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            v = V[i, j]
            fc = '#ffffff' if np.isnan(v) else colors[int(v)]
            ax.add_patch(plt.Rectangle((j - 0.47, i - 0.45), 0.94, 0.9, facecolor=fc, edgecolor='none'))
            if np.isnan(v):
                ax.text(j, i, 'n/a', ha='center', va='center', fontsize=8, color=INK2); continue
            word = {1: 'REMOVES\nMORE', 0: 'no sig.\ndifference', -1: 'keeps\nmore'}[int(v)]
            col = 'white' if v != 0 else INK2
            ax.text(j, i - 0.2, word, ha='center', va='center', fontsize=7.2, fontweight='bold' if v else 'normal', color=col, linespacing=0.95)
            ax.text(j, i + 0.13, f'OR {OR[i, j]:.2f}', ha='center', va='center', fontsize=7, color=col)
            m = MODELS[i][0]
            k = int(max(SP[i, j], SN[i, j]))
            seeds = (f' · {k}/3' if (m in SEEDED or m in HYB) else '') if k else ''
            mark = '●' if v != 0 else ('◐' if k else '')
            ax.text(j, i + 0.33, f'n={int(N[i, j])}{seeds} {mark}'.strip(), ha='center', va='center', fontsize=6.3, color=col)
    ax.set_xlim(-0.5, len(cols) - 0.5); ax.set_ylim(len(MODELS) - 0.5, -0.5)
    ax.set_xticks(range(len(cols)), [c[2] for c in cols], fontsize=8.5)
    ytick_labels(ax)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axhline(len(MODELS) - len(HYB) - 0.5, color=INK2, lw=1.2)
    groups_axis(ax, 1.015, cols)
    ax.legend(handles=[Patch(color=RED, label='removes this pathology MORE than normal (seed-robust significant)'),
                       Patch(color='#e4e3df', label='no significant difference'),
                       Patch(color=BLUE, label='removes it LESS than normal (seed-robust significant)')],
              loc='upper left', bbox_to_anchor=(0, -0.075), ncol=3, frameon=False, fontsize=9)
    noise = 'noise scaled to the window power' if variant == 'E2' else 'noise scaled to the S1+S2 power (E2s variant)'
    fig.suptitle(f'Verdict after correcting for heart rate and age, robust across 3 training seeds:  is the filter biased against each heartbeat type?  ({variant})',
                 x=0.01, ha='left', fontsize=13, fontweight='bold', y=0.995)
    fig.text(0.01, 0.008, f'GEE logistic regression over all noise levels (clean, 20, 10, 5, 0 dB; {noise}), rejected ~ class + noise level + heart rate + HR² (+ age), '
             'clustered by patient; Holm correction across the main model family (each hybrid added to it on its own).\n'
             'Coloured only if Holm-significant in ≥ 2 of 3 seeds AND in the pooled fit over seeds (+ seed factor), same direction.  OR = pooled odds ratio vs normal beats;  '
             'n = pathological patients;  k/3 = seeds with a Holm-significant OR;  ● seed-robust;  ◐ significant in some seeds only.\n'
             'Age adjustment for CirCor and ZCHSound only (BMD-HS healthy subjects are all 20-29 years, PASCAL has no age).  ‡ post-hoc construction control, '
             'not counted in the scorecard.  Models without seeds (B1, B6, B7, B8) are deterministic: their pooled fit is their single fit.',
             fontsize=8, color=INK2, va='bottom')
    fig.tight_layout(rect=(0, 0.075, 1, 0.955))
    fig.savefig(os.path.join(FIG, name), dpi=150)
    plt.close(fig)
    return V[:, :len(COLS)]


def fig3(V):
    names = [m[1] for m in MODELS]
    over = np.nansum(V == 1, 1); fair = np.nansum(V == 0, 1); under = np.nansum(V == -1, 1)
    order = np.argsort(-over * 100 - under, kind='stable')[::-1]
    fig, ax = plt.subplots(figsize=(11.5, 7.2 * len(MODELS) / 17))
    y = np.arange(len(names))
    for k, (vals, col, lab) in enumerate(((over, RED, 'removes pathology MORE (biased against)'), (fair, '#cfcdc8', 'no significant difference'),
                                         (under, BLUE, 'removes pathology LESS'))):
        left = (0 if k == 0 else (over if k == 1 else over + fair))[order] if k else np.zeros(len(names))
        ax.barh(y, vals[order], left=left, color=col, height=0.62, edgecolor=SURF, linewidth=2, label=lab)
        for yy, v, l0 in zip(y, vals[order], left):
            if v:
                ax.text(l0 + v / 2, yy, str(int(v)), ha='center', va='center', fontsize=9,
                        color='white' if col != '#cfcdc8' else INK)
    ax.set_yticks(y, [names[i] for i in order]); ax.tick_params(length=0)
    for t, i in zip(ax.get_yticklabels(), order):
        if MODELS[i][0] in HYB:
            t.set_color('#0f8a8a' if MODELS[i][0].startswith('hd_') else '#4a3aa7')
    ax.set_xlabel('number of heartbeat conditions (out of 15)')
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False, fontsize=9)
    fig.suptitle('Scorecard: for how many heartbeat types is each filter biased?  (seed-robust, after heart-rate/age correction)',
                 x=0.01, ha='left', fontsize=12, fontweight='bold')
    fig.text(0.01, 0.01, 'Most biased at the top. Counts from the E2 verdict matrix (summary_2) over the 15 pre-registered conditions; n/a cells are not counted. '
             'Purple = hybrids.', fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(os.path.join(FIG, 'summary_3_scorecard.png'), dpi=150)
    plt.close(fig)
    pd.DataFrame(dict(model=[m for m, _ in MODELS], name=names, biased_against=over, no_significant_difference=fair, keeps_more=under)) \
        .to_csv(os.path.join(RES, 'summary_scorecard.csv'), index=False)


if __name__ == '__main__':
    e2, a = load()
    fig1(e2, a)
    V = fig2(a)
    fig2(a, 'E2s', 'summary_2b_verdict_matrix_e2s.png')
    fig3(V)
    print('summary figures done')
