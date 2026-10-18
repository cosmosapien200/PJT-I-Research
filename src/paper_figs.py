"""Paper figure set -> figures/paper/figNN_*.png (+ .pdf). Every number is read from results/ or data/processed/scores.

  python paper_figs.py
Also writes results/sens_b6_ties_zch.csv (B6 tie-handling sensitivity, option A: every unmatched beat rejected) and
results/noise_vs_bias.csv (fig 22).
Seeded models (B2, B3, B4, B5, E6) and the hybrids are shown as the mean of seeds 42-44 (results/*_seeds.csv); intervals
are seed-averaged 95 % patient-bootstrap CIs; significance marks are seed-robust (seeds.py).
"""
import os, warnings, glob, shutil
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, Patch
from sklearn.metrics import roc_auc_score, roc_curve
from common import RES, FIG, PROC, MAN, WIN, FS, add_noise, bandpass

warnings.filterwarnings('ignore')
OUT = os.path.join(FIG, 'paper'); os.makedirs(OUT, exist_ok=True)
SC = os.path.join(PROC, 'scores')
# palette (validated reference instance; fixed order, never cycled)
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
BLUE, RED, MID, GREY = '#2a78d6', '#e34948', '#f0efec', '#b9b8b3'
INK, INK2, GRID, SURF = '#1f1f1d', '#5c5b57', '#e4e3df', '#fcfcfb'
SEQ = LinearSegmentedColormap.from_list('seq', ['#cde2fb', '#86b6ef', '#2a78d6', '#184f95', '#0d366b'])
DIV = LinearSegmentedColormap.from_list('div', [BLUE, MID, RED])
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5, 'text.color': INK, 'axes.labelcolor': INK2,
                     'axes.edgecolor': GRID, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.titleweight': 'bold',
                     'axes.titlesize': 10.5, 'axes.titlelocation': 'left', 'figure.facecolor': SURF, 'axes.facecolor': SURF,
                     'axes.grid': False, 'legend.frameon': False, 'axes.spines.top': False, 'axes.spines.right': False,
                     'lines.linewidth': 2, 'savefig.facecolor': SURF})

MAIN = [('b1_lr', 'B1 Springer SQI'), ('b2_rf', 'B2 Das RF'), ('b3_cardiophon', 'B3 CardioPHON'), ('b4_cnn', 'B4 CNN'),
        ('b5_crnn', 'B5 CRNN'), ('b6a', 'B6 PCGKolect'), ('b7', 'B7 SNR rule'), ('b8', 'B8 PCGKolect-v3')]
HYB = [('h_and_b4', 'H-AND Das∧B4'), ('h_and_b5', 'H-AND Das∧B5'), ('h_cas_b4', 'H-CASCADE Das→B4'), ('h_cas_b5', 'H-CASCADE Das→B5'),
       ('h_fuse_b4', 'H-FUSE Das+B4'), ('h_fuse_b5', 'H-FUSE Das+B5')]
ALLM = MAIN + HYB                      # every combined result figure shows the hybrids too
SEEDED = {'b2_rf', 'b3_cardiophon', 'b4_cnn', 'b5_crnn', 'e6_cnn_balanced'} | {m for m, _ in HYB}
SEEDS = [42, 43, 44]
MCOL = {m: CAT[i] for i, (m, _) in enumerate(MAIN)} | {m: (CAT[3] if m.endswith('b4') else CAT[4]) for m, _ in HYB}
MLS = {m: '-' for m, _ in MAIN} | {m: {'and': (0, (4, 2)), 'cas': (0, (1, 1.5)), 'fus': (0, (5, 1.5, 1, 1.5))}[m[2:5]] for m, _ in HYB}
MMK = {m: 'o' for m, _ in MAIN} | {m: {'and': 'v', 'cas': 's', 'fus': 'D'}[m[2:5]] for m, _ in HYB}
MNAME = dict(MAIN) | dict(HYB) | {'b1_svm': 'B1 SVM', 'e6_cnn_balanced': 'E6 balanced CNN', 'b6b': 'B6b (+rhythm check)',
                      'b6pna': 'B6 PhysioNet-only', 'b6pnb': 'B6b PhysioNet-only', 'b8noaug': 'B8 without rhythm aug.', 'b8v4': 'B8v4 PCGKolect-v4', 'b8c': 'B8c PCGKolect-v3c'}
COLS = [('circor', 'murmur_systolic', 'Systolic murmur'), ('circor', 'murmur_diastolic', 'Diastolic murmur*'),
        ('zch', 'ASD', 'ASD'), ('zch', 'VSD', 'VSD'), ('zch', 'PFO', 'PFO'), ('zch', 'PDA', 'PDA'),
        ('bmdhs', 'AS', 'AS'), ('bmdhs', 'AR', 'AR'), ('bmdhs', 'MR', 'MR'), ('bmdhs', 'MS', 'MS'), ('bmdhs', 'multi', 'Multi-valve'),
        ('pascal', 'murmur', 'Murmur'), ('pascal', 'extrasystole', 'Extrasystole'),
        ('circor', 'synthetic_premature', 'Premature (synth.)'), ('circor', 'synthetic_irregular', 'AF-like (synth.)')]
DSN = {'circor': 'CirCor', 'zch': 'ZCHSound', 'bmdhs': 'BMD-HS', 'pascal': 'PASCAL B'}
GROUPS = [(0, 1, 'CirCor'), (2, 5, 'ZCHSound'), (6, 10, 'BMD-HS†'), (11, 12, 'PASCAL†'), (13, 14, 'Synthetic rhythm')]
rd = lambda f: pd.read_csv(os.path.join(RES, f))


def rs(name):
    """Seed-mean view of results/<name>_seeds.csv: <col>_mean -> <col>, <col>_sd kept."""
    d = rd(f'{name}_seeds.csv')
    return d.rename(columns={c: c[:-5] for c in d.columns if c.endswith('_mean')})


def tag(m, s):
    return m if s == 42 or m not in SEEDED else f'{m}_s{s}'


def robust(variant='E2'):
    """Seed-robust GEE table (pooled OR, verdicts) for one E2 variant."""
    a = rd('gee_covariates_seeds.csv')
    return a[a.variant == variant]


def adjusted(a):
    return a[((a.dataset.isin(['circor', 'zch'])) & (a.adjustment == 'hr_age')) | ((a.dataset.isin(['bmdhs', 'pascal'])) & (a.adjustment == 'hr'))]


def save(fig, name, note=None):
    if note:
        fig.text(0.01, 0.005, note, fontsize=7.5, color=INK2, va='bottom', wrap=True)
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=170)
    fig.savefig(os.path.join(OUT, name + '.pdf'))
    plt.close(fig)
    print('wrote', name)


def ink_on(rgb):
    return INK if 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2] > 0.55 else 'white'


def col_groups(ax, y=1.01):
    for a, b, name in GROUPS:
        ax.annotate(name, xy=((a + b) / 2, y), xycoords=('data', 'axes fraction'), ha='center', va='bottom', fontsize=8.5,
                    color=INK2, annotation_clip=False)
        ax.plot([a - 0.42, b + 0.42], [y - 0.005] * 2, transform=ax.get_xaxis_transform(), color=GREY, lw=1.2, clip_on=False)


def heat(ax, M, rows, cols, norm, cmap, fmt, sig=None, sub=None, clip=None):
    """Colour uses M clipped to +-clip (saturation); printed numbers are always the true values."""
    C = np.clip(M, -clip, clip) if clip else M
    ax.imshow(C, cmap=cmap, norm=norm, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                ax.text(j, i, '–', ha='center', va='center', color=INK2, fontsize=8); continue
            c = ink_on(cmap(norm(C[i, j]))[:3])
            bold = sig is not None and sig[i, j]
            ax.text(j, i - (0.1 if sub is not None else 0), fmt(M[i, j]), ha='center', va='center', fontsize=8.5,
                    color=c, fontweight='bold' if bold else 'normal')
            if sub is not None and sub[i][j]:
                ax.text(j, i + 0.3, sub[i][j], ha='center', va='center', fontsize=6.5, color=c)
    ax.set_xticks(range(len(cols)), cols, rotation=35, ha='right', fontsize=8.5)
    ax.set_yticks(range(len(rows)), rows, fontsize=9)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, M.shape[1]), minor=True); ax.set_yticks(np.arange(-.5, M.shape[0]), minor=True)
    ax.grid(which='minor', color=SURF, lw=2.5); ax.tick_params(which='minor', length=0)


def cell(df, m, ds, c, col):
    r = df[(df.model == m) & (df.dataset == ds) & (df.cls == c)]
    return r[col].iloc[0] if len(r) else np.nan


def matrix(df, col, models=MAIN):
    return np.array([[cell(df, m, ds, c, col) for ds, c, _ in COLS] for m, _ in models], float)


# ------------------------------------------------------------------ F1 pipeline
def f01_pipeline():
    fig, ax = plt.subplots(figsize=(13, 6.6)); ax.axis('off'); ax.set_xlim(0, 100); ax.set_ylim(0, 62)

    def box(x, y, w, h, title, body, col):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.25,rounding_size=1.2', fc='white', ec=col, lw=1.6))
        ax.add_patch(FancyBboxPatch((x, y + h - 3.2), w, 3.2, boxstyle='round,pad=0.25,rounding_size=1.2', fc=col, ec=col, lw=0))
        ax.text(x + 0.8, y + h - 1.6, title, color='white', fontsize=9, fontweight='bold', va='center')
        ax.text(x + 0.8, y + h - 4.4, body, color=INK, fontsize=7.6, va='top', linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color=INK2, lw=1.3))
    box(0.5, 38, 22, 23, '1  Training data', 'PhysioNet/CinC 2016\n3,153 records, 2 kHz\nexpert SQI labels +\n(N…N) noise stretches\n5 patient-grouped folds\n\nBank 1 noise (ESC-50,\nMiniLibri, PhysioNet noise)', CAT[0])
    box(25.5, 38, 22, 23, '2  Common corpus', '5 s windows, 2.5 s hop\nunusable: SQI 0 or ≥50 %\n   in a noise stretch\nusable: SQI 1, no noise\n+30 % usable windows\n   + Bank 1 noise\n   (−5…5 dB) → unusable\n30,173 windows', CAT[0])
    box(50.5, 38, 22, 23, '3  Quality filters', 'B1 Springer SQI (LR, SVM)\nB2 Das features + RF\nB3 CardioPHON (416 feat.)\nB4 CNN · B5 CRNN · E6\nB6 PCGKolect U-Net+HMM\nB7 SNR rule · B8 v3\nhybrids Das + B4/B5:\n   AND, CASCADE, FUSE\nB2-B5, E6, hybrids: 3 seeds', CAT[0])
    box(75.5, 38, 24, 23, '4  Test data (never trained on)', 'CirCor 1.0.3 (main, annotated)\nZCHSound (children, CHD)\nBMD-HS (adults, valve)\nPASCAL B (adults)\n+ synthetic premature / AF-like\n\npatients split first:\n20 % of normals = calibration', CAT[1])
    box(0.5, 6, 22, 26, '5  Test beats', 'S1 onset → next S1 onset\nCirCor: expert .tsv\nothers: Springer LR-HSMM\n   (paediatric HR prior for\n   ZCHSound)\n≤ 8 beats / recording\n5 s window centred on beat\nclean if beat SNR ≥ P30 of\n   calibration normals', CAT[1])
    box(25.5, 6, 22, 26, '6  Thresholds', 'published / default\n   (0.5; B6 conf 0.80;\n   B7 14 dB)\nmatched (main):\n   rejects exactly 10 % of\n   clean calibration normal\n   beats, per model × dataset\npooled (sensitivity)', CAT[2])
    box(50.5, 6, 22, 26, '7  Experiments', 'E1 raw rejection\nE2 equal noise (decisive):\n   clean beats + Bank 2 noise\n   (LibriSpeech test-clean,\n   DEMAND) at 20/10/5/0 dB\n   E2s: SNR vs S1+S2 power\nE3 breakdowns · E4 by dataset\nE5 downstream · E6 mitigation', CAT[2])
    box(75.5, 6, 24, 26, '8  Statistics', 'patient bootstrap 95 % CI\n   (1,000 resamples)\nGEE logistic, exchangeable,\n   clustered by patient:\n   rejected ~ class + SNR\n   [+ heart rate + age]\nHolm across 11 models\nseed-robust: ≥ 2/3 seeds\n   + pooled fit significant', CAT[2])
    for x in (23, 48, 73):
        arrow(x, 49.5, x + 2.3, 49.5)
    arrow(87.5, 37.7, 87.5, 34); arrow(75.2, 33, 23, 33)
    for x in (23, 48, 73):
        arrow(x, 19, x + 2.3, 19)
    ax.text(0.5, 61.8, 'Study pipeline', fontsize=13, fontweight='bold', va='bottom')
    save(fig, 'fig01_pipeline')


# ------------------------------------------------------------------ F2 datasets
def f02_datasets():
    c = rd('fix4_counts.csv'); d = rd('d_clean_pass_rates.csv')
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw=dict(width_ratios=[1.1, 1]))
    order = [('circor', 'normal'), ('circor', 'murmur_systolic'), ('circor', 'murmur_diastolic'), ('circor', 'murmur_patient_other_location'),
             ('circor', 'no_murmur_abnormal_outcome'), ('circor', 'unknown'), ('zch', 'control'), ('zch', 'ASD'), ('zch', 'VSD'),
             ('zch', 'PFO'), ('zch', 'PDA'), ('bmdhs', 'normal'), ('bmdhs', 'AS'), ('bmdhs', 'AR'), ('bmdhs', 'MR'), ('bmdhs', 'MS'),
             ('bmdhs', 'multi'), ('pascal', 'normal'), ('pascal', 'murmur'), ('pascal', 'extrasystole'),
             ('circor_synth', 'synthetic_premature'), ('circor_synth', 'synthetic_irregular')]
    lab = [f'{DSN.get(ds, "Synthetic")} · {cl.replace("murmur_patient_other_location", "murmur pt, other loc.").replace("no_murmur_abnormal_outcome", "no murmur, abn. outcome").replace("_", " ")}' for ds, cl in order]
    y = np.arange(len(order))[::-1]
    ax = axes[0]
    ev = [c[(c.dataset == ds) & (c.cls == cl)].eval_patients.iloc[0] for ds, cl in order]
    ca = [c[(c.dataset == ds) & (c.cls == cl)].calibration_patients.iloc[0] for ds, cl in order]
    ax.barh(y, ev, height=0.62, color=CAT[0], label='evaluation patients')
    ax.barh(y, ca, left=ev, height=0.62, color=CAT[1], label='calibration patients', edgecolor=SURF, linewidth=1.5)
    beats = [c[(c.dataset == ds) & (c.cls == cl)].e1_eval_beats.iloc[0] for ds, cl in order]
    for yy, e, k, b in zip(y, ev, ca, beats):
        ax.text(e + k + 6, yy, f'{e + k}  ({b:,} beats)', va='center', fontsize=7.5, color=INK2)
    ax.set_yticks(y, lab, fontsize=8); ax.tick_params(axis='y', length=0)
    ax.set_xlabel('patients'); ax.set_xlim(0, max(np.add(ev, ca)) * 1.35)
    ax.set_title('Test-set composition (patients; evaluation beats in brackets)'); ax.legend(loc='lower right', fontsize=8)
    ax.xaxis.grid(True, color=GRID, lw=1); ax.set_axisbelow(True)
    ax = axes[1]
    for k, (ds, cl) in enumerate(order):
        r = d[(d.dataset == ds.replace('circor_synth', 'circor')) & (d.cls == cl)]
        if r.empty:
            continue
        r = r.iloc[0]; isn = cl in ('normal', 'control')
        ax.plot([r.ci_lo * 100, r.ci_hi * 100], [y[k]] * 2, color=GREY, lw=2, solid_capstyle='round')
        ax.plot(r.clean_pass_rate * 100, y[k], 'o', ms=7, color=INK if isn else CAT[0], mec=SURF, mew=1.5)
    ax.set_yticks(y, [''] * len(y)); ax.tick_params(axis='y', length=0)
    for ds, (a, b) in {'circor': (0, 5), 'zch': (6, 10), 'bmdhs': (11, 16), 'pascal': (17, 19)}.items():
        nr = d[(d.dataset == ds) & d.cls.isin(['normal', 'control'])].clean_pass_rate.iloc[0] * 100
        ax.plot([nr, nr], [y[b] - 0.4, y[a] + 0.4], color=INK, lw=1, ls=(0, (2, 2)))
    ax.set_xlabel('clean-beat pass rate (%)'); ax.set_xlim(0, 105)
    ax.set_title('Clean-beat pass rate (95 % CI); black = normal')
    ax.xaxis.grid(True, color=GRID, lw=1); ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, 'fig02_datasets_and_clean_rule',
         'Clean = beat SNR (S1+S2 power over the quietest non-murmur phase, 25-400 Hz) ≥ 30th percentile of the dataset\'s calibration normal beats. Synthetic beats are clean by construction.')


# ------------------------------------------------------------------ F3 E0
def f03_e0():
    e = rd('e0_label_check.csv')
    subs = ['all', 'a', 'b', 'c', 'd', 'e', 'f']
    fig, ax = plt.subplots(figsize=(8.5, 4))
    x = np.arange(len(subs)); w = 0.36
    for k, (cl, col) in enumerate((('normal', CAT[0]), ('abnormal', CAT[1]))):
        v = [e[(e.subset == s) & (e.cls == cl)].sqi0_rate.iloc[0] * 100 for s in subs]
        n = [e[(e.subset == s) & (e.cls == cl)].n_records.iloc[0] for s in subs]
        ax.bar(x + (k - 0.5) * w, v, w * 0.92, color=col, label=f'{cl} recordings')
        for xx, vv, nn in zip(x + (k - 0.5) * w, v, n):
            ax.text(xx, vv + 0.6, f'{vv:.0f}', ha='center', fontsize=7.5, color=INK2)
    ax.set_xticks(x, ['all'] + [f'subset {s}' for s in subs[1:]])
    ax.set_ylabel('recordings labelled unusable (SQI 0), %'); ax.legend(fontsize=8)
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.set_title('E0  PhysioNet 2016: experts label abnormal recordings unusable twice as often (14.0 % vs 7.5 %)')
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, 'fig03_e0_label_check', 'The training labels of every quality filter already carry a pathology-quality association.')


# ------------------------------------------------------------------ F4 model performance
def f04_models():
    b = rs('b_quality_performance')
    order = ['b1_lr', 'b1_svm', 'b2_rf', 'b3_cardiophon', 'b4_cnn', 'b5_crnn', 'e6_cnn_balanced', 'b6a', 'b6b', 'b7', 'b8', 'b8noaug'] + [m for m, _ in HYB]
    b = b.set_index('model').loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw=dict(width_ratios=[1.6, 1]))
    ax = axes[0]; x = np.arange(len(order)); w = 0.38
    ax.bar(x - w / 2, b.auroc, w * 0.92, color=CAT[0], label='AUROC', yerr=b.auroc_sd.fillna(0), error_kw=dict(lw=1, ecolor=INK2))
    ax.bar(x + w / 2, b.balanced_acc, w * 0.92, color=CAT[1], label='balanced accuracy (default threshold)', yerr=b.balanced_acc_sd.fillna(0),
           error_kw=dict(lw=1, ecolor=INK2))
    for xx, v in zip(x - w / 2, b.auroc):
        ax.text(xx, v + 0.005, f'{v:.3f}'[1:], ha='center', fontsize=7, color=INK2)
    ax.set_xticks(x, [MNAME.get(m, m).replace(' ', '\n', 1) for m in order], fontsize=7)
    ax.set_ylim(0.6, 1.0); ax.legend(fontsize=8, loc='lower right'); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.set_title('Quality-filter performance on PhysioNet (usable vs unusable windows)')
    ax = axes[1]
    for i, (m, nm) in enumerate(ALLM):
        v = pd.read_csv(os.path.join(PROC, f'val_preds_{m}.csv'))
        f, t, _ = roc_curve(v.y, v.score)
        ax.plot(f, t, color=MCOL[m], ls=MLS[m], lw=1.5, label=f'{nm} ({roc_auc_score(v.y, v.score):.3f})')
    ax.plot([0, 1], [0, 1], color=GREY, lw=1)
    ax.set_xlabel('false-positive rate'); ax.set_ylabel('true-positive rate (unusable detected)')
    ax.set_title('ROC curves (seed 42)'); ax.legend(fontsize=6.5, loc='lower right', ncol=2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, 'fig04_model_validation',
         'B1-B3 and hybrids: 5-fold out-of-fold (patient-grouped; H-FUSE nested). B4/B5/E6: held-out fold 0. B6 (released checkpoint, saw all PhysioNet) and B7 (annotation-based): all windows, optimistic. '
         'Bars: mean ± SD over seeds 42-44 for B2-B5, E6 and the hybrids.')


# ------------------------------------------------------------------ F5 threshold transfer
def f05_thresholds():
    n = rs('fix2_clean_normal_rejection')
    ms = [m for m, _ in MAIN] + ['e6_cnn_balanced'] + [m for m, _ in HYB]
    dss = ['circor', 'zch', 'bmdhs', 'pascal']
    fig, axes = plt.subplots(1, 3, figsize=(14, 6.4))
    for ax, (kind, title) in zip(axes, (('published', 'published / default threshold'), ('matched', 'matched threshold (main)'),
                                        ('pooled', 'pooled threshold'))):
        M = np.array([[n[(n.model == m) & (n.dataset == d) & (n.split == 'eval') & (n.threshold == kind)].rate.iloc[0] * 100
                       if len(n[(n.model == m) & (n.dataset == d) & (n.split == 'eval') & (n.threshold == kind)]) else np.nan
                       for d in dss] for m in ms])
        heat(ax, M, [MNAME[m] for m in ms], [DSN[d] for d in dss], plt.Normalize(0, 80), SEQ, lambda v: f'{v:.0f}')
        ax.set_title(title, fontsize=10)
        ax.set_xticklabels([DSN[d] for d in dss], rotation=0, ha='center')
    fig.suptitle('Clean NORMAL evaluation beats rejected (%) — target 10 %', x=0.01, ha='left', fontweight='bold', fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    save(fig, 'fig05_threshold_transfer',
         'Published thresholds do not transfer (CirCor: 73-82 % of clean normal beats rejected by B2-B5). Matched thresholds hit 10 % on calibration patients; evaluation rates deviate where calibration sets are small (BMD-HS 4, PASCAL 18 normal patients). '
         'Pooled thresholds are dominated by the domain-shifted sets. B1 SVM omitted (excluded on CirCor). Seeded models: mean of 3 seeds. H-AND/H-CASCADE default = 0.9 (percentile).')


# ------------------------------------------------------------------ F6 example of E2 protocol
def f06_example():
    import score
    b = pd.read_csv(os.path.join(PROC, 'circor_beats.csv'), low_memory=False)
    W = np.load(os.path.join(PROC, 'circor_beat_windows.npy'), mmap_mode='r')
    seg = pd.read_csv(os.path.join(PROC, 'circor_segments.csv.gz'))
    i = int(np.flatnonzero((b.cls == 'murmur_systolic') & (b.clean == 1) & (b.split == 'eval') & (b.sys_grade == 'III/VI'))[3])
    r = b.iloc[i]; w = np.asarray(W[i], np.float32)
    N, clips = score.e2_noise([r.beat_id])
    ws = int(r.centre - WIN // 2)
    sg = seg[(seg.record_id == r.record_id) & (seg.start >= ws) & (seg.end <= ws + WIN)]
    fig, axes = plt.subplots(5, 1, figsize=(12, 7.6), sharex=True)
    t = np.arange(WIN) / FS
    shade = {'S1': CAT[0], 'S2': CAT[1], 'systole': CAT[4]}
    for ax, lev in zip(axes, ['none', 20, 10, 5, 0]):
        x = w if lev == 'none' else add_noise(w, N[0], lev)
        x = bandpass(x / np.abs(x).max())
        for _, s in sg.iterrows():
            if s.state in shade:
                ax.axvspan((s.start - ws) / FS, (s.end - ws) / FS, color=shade[s.state], alpha=0.12, lw=0)
        ax.axvspan(r.rel_start / FS, r.rel_end / FS, ymin=0.93, ymax=1, color=INK, alpha=0.8, lw=0)
        ax.plot(t, x, color=INK, lw=0.6)
        ax.set_yticks([]); ax.spines['left'].set_visible(False)
        ax.text(0.003, 0.82, 'clean' if lev == 'none' else f'+ noise at {lev} dB', transform=ax.transAxes, fontsize=9, fontweight='bold')
    axes[-1].set_xlabel('time (s)')
    fig.legend(handles=[Patch(color=shade[k], alpha=0.35, label=k) for k in shade] + [Patch(color=INK, label='scored beat (bar)')],
               ncol=4, loc='lower right', fontsize=8, bbox_to_anchor=(0.99, 0.0))
    axes[0].set_title(f'E2 protocol on one clean CirCor beat (systolic murmur, grade III/VI); noise clip {clips[0]} '
                      f'(same clip and offset at every level)', fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, 'fig06_e2_protocol_example', 'Noise from Bank 2 (unseen by every model), scaled to the clean window power. 25-400 Hz band-passed for display; expert segmentation shaded.')


# ------------------------------------------------------------------ F7 diagnostics: noise sensitivity vs class discrimination
def f07_diagnostics():
    ms = ALLM
    idx = {(e, ds): pd.read_csv(os.path.join(SC, f'{e}_{ds}_index.csv'), low_memory=False) for e in ('e1', 'e2') for ds in ['circor', 'zch', 'bmdhs', 'pascal', 'synth']}
    rows_n, rows_c = [], []
    for m, _ in ms:
        rn, rc = [], []
        for t in ([tag(m, s) for s in SEEDS] if m in SEEDED else [m]):   # AUROCs averaged over seeds
            rn.append([]); rc.append([])
            for ds in ['circor', 'zch', 'bmdhs', 'pascal']:
                e1, e2 = idx[('e1', ds)], idx[('e2', ds)]
                sel = ((e1.clean == 1) & (e1.split == 'eval')).values
                s0 = np.load(os.path.join(SC, f'e1_{ds}_{t}.npy'))[sel]; s4 = np.load(os.path.join(SC, f'e2_{ds}_{t}.npy'))[(e2.snr_level == 0).values]
                rn[-1].append(roc_auc_score(np.r_[np.zeros(len(s0)), np.ones(len(s4))], np.r_[s0, s4]))
            for ds, c, _ in COLS:
                e1 = idx[('e1', ds)]
                s = np.load(os.path.join(SC, f'e1_{ds}_{t}.npy'))
                ref = ((e1.clean == 1) & (e1.split == 'eval') & e1.cls.isin(['normal', 'control'])).values
                if c.startswith('synthetic'):
                    es = idx[('e1', 'synth')]; ss = np.load(os.path.join(SC, f'e1_synth_{t}.npy'))
                    p = ss[(es.cls == c).values]
                else:
                    p = s[((e1.clean == 1) & (e1.split == 'eval') & (e1.cls == c)).values]
                rc[-1].append(roc_auc_score(np.r_[np.zeros(ref.sum()), np.ones(len(p))], np.r_[s[ref], p]) if len(p) else np.nan)
        rows_n.append(np.mean(rn, 0)); rows_c.append(np.mean(rc, 0))
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw=dict(width_ratios=[0.33, 1]))
    heat(axes[0], np.array(rows_n), [n for _, n in ms], ['CirCor', 'ZCH', 'BMD-HS', 'PASCAL'], plt.Normalize(0.5, 1), SEQ,
         lambda v: f'{v:.2f}')
    axes[0].set_xticklabels(['CirCor', 'ZCH', 'BMD-HS', 'PASCAL'], rotation=0, ha='center')
    axes[0].set_title('Detects noise?\nAUROC clean vs 0 dB (same beats)', fontsize=9.5)
    C = np.array(rows_c)
    heat(axes[1], C, [n for _, n in ms], [c[2] for c in COLS], TwoSlopeNorm(0.5, 0.25, 0.85), DIV, lambda v: f'{v:.2f}')
    col_groups(axes[1], 1.01)
    axes[1].set_title('Does the filter score pathology as worse quality?\nAUROC pathological vs normal clean beats (0.5 = blind; red = pathology looks less usable)',
                      fontsize=9.5, pad=18)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, 'fig07_noise_vs_pathology_sensitivity',
         'Left: every filter responds to noise. Right: a filter with no pathology bias stays near 0.5; red cells explain the biases (e.g. B3/B4/B5 on ZCHSound PFO, B6 on synthetic rhythm). '
         'B2 is noise-sensitive but pathology-blind (≈0.5 everywhere). Seeded models and hybrids: mean AUROC of 3 seeds.')


# ------------------------------------------------------------------ F8/F9 gap heatmaps (E1, E2 clean, E2 10 dB)
def gap_heat(df, title, name, note, vmax=40):
    """df: per-seed rows (column seed). Number = mean gap over seeds; 'CI≠0' = 95 % CI excludes 0 in >= 2 of 3 seeds."""
    df = df.assign(pos=(df.diff_ci_lo > 0).astype(float), neg=(df.diff_ci_hi < 0).astype(float))
    m = df.groupby(['model', 'dataset', 'cls'])[['diff_vs_normal', 'pos', 'neg']].agg({'diff_vs_normal': 'mean', 'pos': 'sum', 'neg': 'sum'}).reset_index()
    D = matrix(m, 'diff_vs_normal', ALLM) * 100
    sig = (matrix(m, 'pos', ALLM) >= 2) | (matrix(m, 'neg', ALLM) >= 2)
    fig, ax = plt.subplots(figsize=(14.5, 8.2))
    heat(ax, D, [n for _, n in ALLM], [c[2] for c in COLS], TwoSlopeNorm(0, -vmax, vmax), DIV,
         lambda v: f'{v:+.0f}', sig=sig, sub=[['CI≠0' if s else '' for s in row] for row in sig], clip=vmax)
    ax.axhline(len(MAIN) - 0.5, color=INK2, lw=1.2)
    col_groups(ax)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=TwoSlopeNorm(0, -vmax, vmax), cmap=DIV), ax=ax, fraction=0.02, pad=0.01)
    cb.set_label('rejection rate: pathological − normal (pp)'); cb.outline.set_visible(False)
    ax.set_title(title, pad=18)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, name, note)


def f08_f09_gaps():
    e1 = rd('e1_raw_per_seed.csv'); e1 = e1[e1.threshold == 'matched']
    gap_heat(e1, 'E1  Raw rejection gap as recorded (all evaluation beats, matched threshold, mean of 3 seeds)', 'fig08_e1_raw_gap_heatmap',
             'Red = pathology rejected more often than normal beats. Bold/"CI≠0" = 95 % patient-bootstrap CI excludes 0 in ≥ 2 of 3 seeds. * 5 patients. † small calibration sets. '
             'E1 mixes recording quality and filter behaviour; E2 isolates the filter. Rows below the line: hybrids.')
    e2 = rd('e2_equal_noise_per_seed.csv'); e2 = e2[e2.threshold == 'matched']; e2['snr_level'] = e2.snr_level.astype(str)
    gap_heat(e2[e2.snr_level == 'none'], 'E2  Equal-conditions gap on CLEAN beats (no added noise, matched threshold, mean of 3 seeds)',
             'fig09a_e2_clean_gap_heatmap', 'Same beats quality-matched by the clean rule. Bold/"CI≠0" = 95 % patient-bootstrap CI excludes 0 in ≥ 2 of 3 seeds. Rows below the line: hybrids.')
    gap_heat(e2[e2.snr_level == '10'], 'E2  Equal-noise gap at 10 dB added Bank-2 noise (matched threshold, mean of 3 seeds)',
             'fig09b_e2_10dB_gap_heatmap', 'Identical noise clip per beat; SNR relative to the clean window power.')
    e2s = rd('e2s_equal_noise_per_seed.csv'); e2s = e2s[e2s.threshold == 'matched']; e2s['snr_level'] = e2s.snr_level.astype(str)
    gap_heat(e2s[e2s.snr_level == '10'], 'E2s  Equal-noise gap at 10 dB, noise scaled to the S1+S2 power (matched threshold, mean of 3 seeds)',
             'fig09c_e2s_10dB_gap_heatmap', 'Same beats and noise clips as E2; the SNR reference is the power of the clean window inside its S1 and S2 segments, '
             'so murmur energy does not raise the noise level. Bold/"CI≠0" = CI excludes 0 in ≥ 2 of 3 seeds.')


# ------------------------------------------------------------------ F10 gap vs SNR small multiples
def f10_gap_vs_snr():
    e2 = rs('e2_equal_noise'); e2 = e2[e2.threshold == 'matched']; e2['snr_level'] = e2.snr_level.astype(str)
    keys = [('circor', 'murmur_systolic', 'CirCor systolic murmur'), ('zch', 'PFO', 'ZCHSound PFO'), ('zch', 'VSD', 'ZCHSound VSD'),
            ('bmdhs', 'multi', 'BMD-HS multi-valve†'), ('pascal', 'murmur', 'PASCAL murmur†'), ('circor', 'synthetic_premature', 'Synthetic premature beats')]
    L = ['none', '20', '10', '5', '0']
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.4), sharex=True)
    for ax, (ds, c, title) in zip(axes.flat, keys):
        ax.axhline(0, color=INK2, lw=1)
        for m, nm in ALLM:
            g = e2[(e2.dataset == ds) & (e2.cls == c) & (e2.model == m)].set_index('snr_level').reindex(L)
            if g.diff_vs_normal.isna().all():
                continue
            ax.plot(range(5), g.diff_vs_normal * 100, color=MCOL[m], ls=MLS[m], marker=MMK[m], ms=4, mec=SURF, mew=1, lw=1.6, label=nm)
        ax.set_title(title, fontsize=10); ax.set_xticks(range(5), ['clean', '20', '10', '5', '0'])
        ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    for ax in axes[1]:
        ax.set_xlabel('added noise SNR (dB)')
    for ax in axes[:, 0]:
        ax.set_ylabel('rejection gap vs normal (pp)')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, ncol=7, loc='upper center', bbox_to_anchor=(0.5, 0.955))
    fig.suptitle('E2  How the pathology-vs-normal gap evolves with added noise (matched threshold, mean of 3 seeds)', x=0.01, ha='left', fontweight='bold', fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    save(fig, 'fig10_e2_gap_vs_snr', 'Gaps shrink at 0 dB because almost every beat is rejected (ceiling). Hybrids share the colour of their deep model (dashed AND, dotted CASCADE, dash-dot FUSE). '
         'Mean ± SD over seeds in results/e2_equal_noise_seeds.csv; per-class curves in figures/e2_<dataset>.png.')


# ------------------------------------------------------------------ F11 forest plot
def f11_forest():
    g = robust('E2').rename(columns={'pooled_odds_ratio': 'odds_ratio', 'pooled_or_ci_lo': 'or_ci_lo', 'pooled_or_ci_hi': 'or_ci_hi'})
    keys = [('circor', 'murmur_systolic', 'CirCor systolic murmur', 'hr_age'), ('zch', 'PFO', 'ZCHSound PFO', 'hr_age'),
            ('zch', 'VSD', 'ZCHSound VSD', 'hr_age'), ('zch', 'ASD', 'ZCHSound ASD', 'hr_age'),
            ('bmdhs', 'multi', 'BMD-HS multi-valve† (HR only)', 'hr'), ('circor', 'synthetic_premature', 'Synthetic premature', 'hr_age')]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharey=True)
    y = np.arange(len(ALLM))[::-1]
    for ax, (ds, c, title, adj) in zip(axes.flat, keys):
        ax.axvline(1, color=INK2, lw=1)
        for k, (spec, col, off, lab) in enumerate((('unadjusted', GREY, 0.17, 'unadjusted'), (adj, INK, -0.17, 'adjusted (HR' + (' + age)' if adj == 'hr_age' else ')')))):
            for yy, (m, _) in zip(y, ALLM):
                r = g[(g.dataset == ds) & (g.cls == c) & (g.model == m) & (g.adjustment == spec)]
                if r.empty or not np.isfinite(r.odds_ratio.iloc[0]):
                    continue
                r = r.iloc[0]
                sig = r.verdict_code != 0
                colr = (RED if r.odds_ratio > 1 else BLUE) if (sig and k == 1) else col
                ax.plot([r.or_ci_lo, r.or_ci_hi], [yy + off] * 2, color=colr, lw=1.6, solid_capstyle='round')
                ax.plot(r.odds_ratio, yy + off, 'o' if k else 's', ms=6 if k else 5, color=colr, mec=SURF, mew=1,
                        label=lab if yy == y[0] else None)
        ax.set_xscale('log'); ax.set_xlim(0.15, 30); ax.set_xticks([0.2, 0.5, 1, 2, 5, 20], ['0.2', '0.5', '1', '2', '5', '20'])
        ax.set_title(title, fontsize=10); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
        ax.axhline(len(HYB) - 0.5, color=GRID, lw=1.2)
        ax.set_yticks(y, [n for _, n in ALLM])
    for ax in axes[1]:
        ax.set_xlabel('odds ratio of rejection vs normal (log scale)')
    from matplotlib.lines import Line2D
    axes[0, 0].legend(handles=[Line2D([], [], color=GREY, marker='s', ms=5, label='unadjusted'),
                               Line2D([], [], color=INK, marker='o', ms=6, label='adjusted (HR [+ age])'),
                               Line2D([], [], color=RED, marker='o', ms=6, label='adjusted, seed-robust sig., OR > 1'),
                               Line2D([], [], color=BLUE, marker='o', ms=6, label='adjusted, seed-robust sig., OR < 1')],
                      fontsize=7, loc='lower right')
    fig.suptitle('GEE odds ratios with 95 % CI (E2, all noise levels, pooled over 3 seeds): grey = unadjusted, black = adjusted; red/blue = seed-robust significant after adjustment',
                 x=0.01, ha='left', fontweight='bold', fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    save(fig, 'fig11_forest_gee_odds_ratios', 'rejected ~ class + SNR level [+ HR + HR² + age category] (+ seed for seeded models), exchangeable correlation, patient clusters; Holm across the 11-model family '
         '(each hybrid added on its own). Seed-robust = Holm-significant in ≥ 2 of 3 seeds and pooled. OR > 1: pathology rejected more often. BMD-HS age not identifiable.')


# ------------------------------------------------------------------ F12 verdict + scorecard (copies of summary figs)
def f12_verdict():
    for src, dst in (('summary_2_verdict_matrix.png', 'fig12a_verdict_matrix'), ('summary_3_scorecard.png', 'fig12b_scorecard'),
                     ('summary_1_gap_heatmap.png', 'fig12c_gap_heatmap_with_significance'), ('summary_2b_verdict_matrix_e2s.png', 'fig12d_verdict_matrix_e2s')):
        shutil.copy(os.path.join(FIG, src), os.path.join(OUT, dst + '.png'))
    print('copied verdict / scorecard')


# ------------------------------------------------------------------ F13 confound: HR and age
def f13_confound():
    from analysis_fixes import covariates
    cov = covariates()
    rows = []
    for ds in ['circor', 'zch']:
        b = pd.read_csv(os.path.join(PROC, f'{ds}_beats.csv'), usecols=['beat_id', 'cls', 'split', 'patient_id'], low_memory=False)
        rows.append(b[b.split == 'eval'].merge(cov, on='beat_id').assign(dataset=ds))
    d = pd.concat(rows)
    fig = plt.figure(figsize=(14, 8.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
    # (a) HR by class, ZCHSound
    ax = fig.add_subplot(gs[0, 0:2])
    groups = [('zch', 'control'), ('zch', 'ASD'), ('zch', 'VSD'), ('zch', 'PFO'), ('zch', 'PDA'),
              ('circor', 'normal'), ('circor', 'murmur_systolic'), ('circor', 'unknown')]
    data = [d[(d.dataset == ds) & (d.cls == c)].hr_bpm.values for ds, c in groups]
    bp = ax.boxplot(data, vert=False, widths=0.55, showfliers=False, patch_artist=True, medianprops=dict(color=INK, lw=2))
    for p, (ds, c) in zip(bp['boxes'], groups):
        p.set_facecolor(MID if c in ('normal', 'control') else '#cde2fb'); p.set_edgecolor(GREY)
    ax.set_yticks(range(1, len(groups) + 1), [f'{DSN[ds]} · {c.replace("_", " ")}' for ds, c in groups], fontsize=8.5)
    ax.set_xlabel('heart rate from segmentation (bpm)'); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.set_title('(a) Heart rate per beat: CHD classes (esp. PFO) beat faster than controls')
    # (b) age composition ZCH
    ax = fig.add_subplot(gs[0, 2])
    z = d[d.dataset == 'zch'].drop_duplicates('patient_id')
    cats = ['Neonate', 'Infant', 'Child', 'Adolescent']
    cls_ = ['control', 'ASD', 'VSD', 'PFO', 'PDA']
    left = np.zeros(len(cls_))
    for k, a in enumerate(cats):
        v = np.array([(z[z.cls == c].age_cat == a).mean() * 100 for c in cls_])
        ax.barh(range(len(cls_)), v, left=left, color=CAT[k], height=0.6, edgecolor=SURF, linewidth=1.5, label=a)
        for i, (vv, ll) in enumerate(zip(v, left)):
            if vv > 9:
                ax.text(ll + vv / 2, i, f'{vv:.0f}', ha='center', va='center', fontsize=7.5, color='white' if k in (0, 1) else INK)
        left += v
    ax.set_yticks(range(len(cls_)), cls_); ax.invert_yaxis(); ax.set_xlabel('% of patients')
    ax.legend(fontsize=7.5, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.18))
    ax.set_title('(b) ZCHSound age mix')
    # (c) PFO gap within HR bands
    ax = fig.add_subplot(gs[1, 0:2])
    hb = rs('fix3_hr_band_gaps')
    bands = ['<80', '80-100', '100-120', '120-140', '>=140']
    w = 0.058
    for k, (m, nm) in enumerate(ALLM):
        g = hb[(hb.dataset == 'zch') & (hb.cls == 'PFO') & (hb.model == m)].set_index('hr_band').reindex(bands)
        xs = np.arange(5) + (k - (len(ALLM) - 1) / 2) * w
        ax.errorbar(xs, g.diff_vs_normal * 100, yerr=[(g.diff_vs_normal - g.diff_ci_lo) * 100, (g.diff_ci_hi - g.diff_vs_normal) * 100],
                    fmt=MMK[m], ms=4, color=MCOL[m], ecolor=MCOL[m], elinewidth=1, capsize=0, mec=SURF, label=nm)
    n_p = hb[(hb.dataset == 'zch') & (hb.cls == 'PFO') & (hb.model == 'b4_cnn')].set_index('hr_band').reindex(bands)
    ax.set_xticks(range(5), [f'{b} bpm\n(PFO n={int(p)}, ctrl n={int(q)})' for b, p, q in zip(bands, n_p.n_patients, n_p.n_normal_patients)], fontsize=8)
    ax.axhline(0, color=INK2, lw=1); ax.set_ylabel('PFO − control rejection (pp)'); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.legend(fontsize=6.5, ncol=5, loc='upper right'); ax.set_title('(c) ZCHSound PFO gap stays positive within heart-rate bands (E2, all levels pooled, mean of 3 seeds)')
    # (d) OR shrinkage after adjustment
    ax = fig.add_subplot(gs[1, 2])
    g = robust('E2')
    ends = []
    for k, (m, nm) in enumerate(ALLM):
        u = g[(g.dataset == 'zch') & (g.cls == 'PFO') & (g.model == m) & (g.adjustment == 'unadjusted')]
        a = g[(g.dataset == 'zch') & (g.cls == 'PFO') & (g.model == m) & (g.adjustment == 'hr_age')]
        if u.empty or a.empty:
            continue
        ax.plot([0, 1], [u.pooled_odds_ratio.iloc[0], a.pooled_odds_ratio.iloc[0]], color=MCOL[m], ls=MLS[m], marker=MMK[m], ms=5, mec=SURF)
        ends.append([a.pooled_odds_ratio.iloc[0], a.pooled_odds_ratio.iloc[0], nm + (' *' if a.verdict_code.iloc[0] != 0 else ''), MCOL[m]])
    ends.sort(key=lambda e: e[0])
    for k in range(1, len(ends)):                      # spread labels, keep leader lines to the true end
        ends[k][1] = max(ends[k][1], ends[k - 1][1] + 0.26)
    for v, lv, nm, c in ends:
        ax.plot([1.02, 1.12], [v, lv], color=GREY, lw=0.8)
        ax.text(1.14, lv, nm, fontsize=7.5, va='center', color=INK2)
    ax.axhline(1, color=INK2, lw=1); ax.set_xlim(-0.2, 2.1)
    ax.set_xticks([0, 1], ['unadjusted', 'HR + age']); ax.set_ylabel('PFO odds ratio')
    ax.set_title('(d) PFO OR before/after adjustment\n(pooled over seeds; * seed-robust significant after)')
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, 'fig13_heart_rate_age_confound', 'Heart rate = 60 / beat duration from the segmentation (ZCHSound: Springer LR-HSMM with paediatric 30-220 bpm prior).')


# ------------------------------------------------------------------ F14 E3 grade / timing
def f14_e3():
    e3 = rs('e3_breakdown')
    t = e3[(e3.dataset == 'circor') & (e3.experiment == 'E1') & e3.breakdown.isin(['sys_grade', 'sys_timing'])]
    vals = [('sys_grade', 'I/VI'), ('sys_grade', 'II/VI'), ('sys_grade', 'III/VI'), ('sys_timing', 'Early-systolic'),
            ('sys_timing', 'Mid-systolic'), ('sys_timing', 'Holosystolic')]
    fig, ax = plt.subplots(figsize=(14, 5.4))
    w = 0.058
    for k, (m, nm) in enumerate(ALLM):
        xs, ys, lo, hi = [], [], [], []
        for j, (br, v) in enumerate(vals):
            r = t[(t.breakdown == br) & (t.value == v) & (t.model == m)]
            if r.empty:
                continue
            r = r.iloc[0]; xs.append(j + (k - (len(ALLM) - 1) / 2) * w); ys.append(r.diff_vs_normal * 100)
            lo.append((r.diff_vs_normal - r.diff_ci_lo) * 100); hi.append((r.diff_ci_hi - r.diff_vs_normal) * 100)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt=MMK[m], ms=4, color=MCOL[m], elinewidth=1, mec=SURF, label=nm)
    n = [t[(t.breakdown == br) & (t.value == v) & (t.model == 'b4_cnn')].n_patients.iloc[0] for br, v in vals]
    ax.set_xticks(range(len(vals)), [f'{v}\n(n={k} pts)' for (_, v), k in zip(vals, n)])
    ax.axvline(2.5, color=GRID, lw=1.5)
    ax.text(1, ax.get_ylim()[1], 'by murmur grade', ha='center', va='top', fontsize=9, color=INK2)
    ax.text(4, ax.get_ylim()[1], 'by murmur timing', ha='center', va='top', fontsize=9, color=INK2)
    ax.axhline(0, color=INK2, lw=1); ax.set_ylabel('systolic-murmur − normal rejection (pp)')
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True); ax.legend(fontsize=7, ncol=7, loc='lower center', bbox_to_anchor=(0.5, -0.4))
    ax.set_title('E3  CirCor systolic murmurs: B5 and B6 over-reject loud (III/VI) and holosystolic murmurs; B1/B7 under-reject (E1, matched)')
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, 'fig14_e3_murmur_grade_timing', 'Late-systolic omitted (one patient). Seeded models and hybrids: mean of 3 seeds with seed-averaged 95 % patient-bootstrap CIs.')


# ------------------------------------------------------------------ F15 E5
def f15_e5():
    t = rd('e5_downstream.csv')
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4), gridspec_kw=dict(width_ratios=[1, 1.4]))
    ax = axes[0]
    y = np.arange(len(t))[::-1]
    for yy, (_, r) in zip(y, t.iterrows()):
        ax.plot([r.frac_train_absent_beats_kept * 100, r.frac_train_murmur_beats_kept * 100], [yy, yy], color=GREY, lw=2)
        ax.plot(r.frac_train_absent_beats_kept * 100, yy, 'o', color=CAT[0], ms=7, mec=SURF, label='murmur-absent beats' if yy == y[0] else None)
        ax.plot(r.frac_train_murmur_beats_kept * 100, yy, 'o', color=CAT[1], ms=7, mec=SURF, label='murmur beats' if yy == y[0] else None)
    ax.set_yticks(y, [MNAME.get(m, 'none (unfiltered)') for m in t.filter_model]); ax.set_xlabel('% of training beats kept by the filter')
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.set_title('(a) What the filter leaves for training\n(orange left of blue = murmurs removed more)')
    ax = axes[1]
    x = np.arange(len(t)); w = 0.27
    for k, (col, lab, c) in enumerate((('auroc', 'AUROC', CAT[0]), ('sensitivity', 'sensitivity', CAT[1]), ('specificity', 'specificity', CAT[2]))):
        ax.bar(x + (k - 1) * w, t[f'{col}_mean'], w * 0.92, yerr=t[f'{col}_sd'], color=c, label=lab, error_kw=dict(lw=1, ecolor=INK2))
    ax.axhline(t[t.filter_model == 'unfiltered'].auroc_mean.iloc[0], color=CAT[0], lw=1, ls=(0, (3, 2)))
    short = {'b1_lr': 'B1', 'b2_rf': 'B2', 'b3_cardiophon': 'B3', 'b4_cnn': 'B4', 'b5_crnn': 'B5', 'e6_cnn_balanced': 'E6',
             'b6a': 'B6', 'b6b': 'B6b', 'b7': 'B7', 'b8': 'B8', 'b8v4': 'B8v4', 'b8c': 'B8c', 'b8noaug': 'B8na', 'unfiltered': 'none',
             'h_and_b4': 'AND\nB4', 'h_and_b5': 'AND\nB5', 'h_cas_b4': 'CAS\nB4', 'h_cas_b5': 'CAS\nB5', 'h_fuse_b4': 'FUSE\nB4', 'h_fuse_b5': 'FUSE\nB5'}
    ax.set_xticks(x, [short.get(m, m) for m in t.filter_model], fontsize=7.5)
    ax.set_ylim(0.6, 1); ax.legend(fontsize=8, ncol=3, loc='upper center'); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.set_title('(b) Murmur detection on the test 40 %\n(patient level, mean ± SD over runs)')
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, 'fig15_e5_downstream', 'Murmur CNN (B4 architecture), patient score = mean over beats, threshold = Youden J on held-out training patients. '
         'Existing filters: 3 E5 seeds; hybrids: 3 filter seeds × 3 E5 seeds.')


# ------------------------------------------------------------------ F16 E6 mitigation + B6 retrain
def f16_e6():
    e2 = rs('e2_equal_noise'); e2 = e2[e2.threshold == 'matched']; e2['snr_level'] = e2.snr_level.astype(str)
    e4 = rs('e4_by_dataset'); e4['snr_level'] = e4.snr_level.astype(str)
    L = ['none', '20', '10', '5', '0']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    for c, ls in (('synthetic_premature', '-'), ('synthetic_irregular', (0, (3, 2)))):
        for m, col in (('b6a', CAT[5]), ('b6b', CAT[6])):
            g = e2[(e2.dataset == 'circor') & (e2.cls == c) & (e2.model == m)].set_index('snr_level').reindex(L)
            ax.plot(range(5), g.diff_vs_normal * 100, color=col, ls=ls, marker='o', ms=4, label=f'{MNAME[m]}: {c.split("_")[1]}')
    ax.set_xticks(range(5), ['clean', '20', '10', '5', '0']); ax.set_xlabel('added noise SNR (dB)'); ax.set_ylabel('gap vs normal (pp)')
    ax.legend(fontsize=7.5); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.set_title('(a) E6a  B6 rhythm check adds ≈ +8 pp\non premature beats')
    ax = axes[1]
    dss = ['circor', 'zch', 'bmdhs', 'pascal']; x = np.arange(4); w = 0.36
    for k, (m, col) in enumerate((('b4_cnn', CAT[3]), ('e6_cnn_balanced', CAT[6]))):
        r = [e4[(e4.dataset == d) & (e4.model == m) & (e4.snr_level == 'none') & (e4.experiment == 'E2')].iloc[0] for d in dss]
        ax.bar(x + (k - 0.5) * w, [q.gap * 100 for q in r], w * 0.92, color=col, label=MNAME[m],
               yerr=[[(q.gap - q.gap_ci_lo) * 100 for q in r], [(q.gap_ci_hi - q.gap) * 100 for q in r]], error_kw=dict(lw=1, ecolor=INK2))
    ax.axhline(0, color=INK2, lw=1); ax.set_xticks(x, [DSN[d] for d in dss]); ax.set_ylabel('pooled pathological − normal (pp)')
    ax.legend(fontsize=8); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.set_title('(b) E6b  balanced training does not\nclose the gap (E2 clean, mean of 3 seeds)')
    ax = axes[2]
    keys = [('circor', 'murmur_systolic', 'CirCor systolic'), ('circor', 'synthetic_premature', 'premature'),
            ('circor', 'synthetic_irregular', 'AF-like'), ('zch', 'PFO', 'ZCH PFO')]
    x = np.arange(len(keys))
    for k, (m, col) in enumerate((('b6a', CAT[5]), ('b6pna', CAT[2]))):
        r = [e2[(e2.dataset == d) & (e2.cls == c) & (e2.model == m) & (e2.snr_level == 'none')].iloc[0] for d, c, _ in keys]
        ax.bar(x + (k - 0.5) * w, [q.diff_vs_normal * 100 for q in r], w * 0.92, color=col, label=MNAME[m],
               yerr=[[(q.diff_vs_normal - q.diff_ci_lo) * 100 for q in r], [(q.diff_ci_hi - q.diff_vs_normal) * 100 for q in r]],
               error_kw=dict(lw=1, ecolor=INK2))
    ax.axhline(0, color=INK2, lw=1); ax.set_xticks(x, [k[2] for k in keys]); ax.legend(fontsize=8)
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.set_title('(c) B6 retrained without CirCor: murmur\nbias unchanged, rhythm bias halved (E2 clean)')
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, 'fig16_e6_mitigation_and_b6_retrain', 'B6 PhysioNet-only matches 77 % of CirCor reference beats vs 99 % for the released checkpoint. 95 % patient-bootstrap CIs.')


# ------------------------------------------------------------------ F17 recalibration
def f17_recal():
    r = rs('fix2_recal_e2'); r['snr_level'] = r.snr_level.astype(str)
    keys = [('bmdhs', 'multi', 'BMD-HS multi-valve'), ('bmdhs', 'MR', 'BMD-HS MR'), ('pascal', 'murmur', 'PASCAL murmur'),
            ('pascal', 'extrasystole', 'PASCAL extrasystole')]
    fig, axes = plt.subplots(1, 4, figsize=(15, 6.6), sharey=True)
    y = np.arange(len(ALLM))[::-1]
    for ax, (ds, c, title) in zip(axes, keys):
        ax.axvline(0, color=INK2, lw=1)
        for k, (kind, col, off) in enumerate((('matched', INK, 0.2), ('published', CAT[1], 0), ('pooled', CAT[0], -0.2))):
            for yy, (m, _) in zip(y, ALLM):
                q = r[(r.dataset == ds) & (r.cls == c) & (r.model == m) & (r.threshold == kind) & (r.snr_level == 'none')]
                if q.empty:
                    continue
                q = q.iloc[0]
                ax.plot([q.diff_ci_lo * 100, q.diff_ci_hi * 100], [yy + off] * 2, color=col, lw=1.4)
                ax.plot(q.diff_vs_normal * 100, yy + off, 'o', ms=5, color=col, mec=SURF, label=kind if yy == y[0] else None)
        ax.set_title(title, fontsize=10); ax.set_yticks(y, [n for _, n in ALLM]); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
        ax.set_xlabel('gap vs normal, E2 clean (pp)')
    axes[0].legend(fontsize=8, loc='lower left')
    fig.suptitle('Threshold choice for the small-calibration datasets (BMD-HS 4, PASCAL 18 normal calibration patients)', x=0.01, ha='left', fontweight='bold', fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    save(fig, 'fig17_recalibration_bmdhs_pascal', 'Pooled thresholds reject 0-3 % of these datasets\' clean normal beats, so gaps collapse by floor effect; published-threshold gaps for B1/B6/B7 (BMD-HS) and B2-B5 (PASCAL murmur) persist. '
         'Seeded models and hybrids: mean of 3 seeds.')


# ------------------------------------------------------------------ F18 B6 tie sensitivity (option A)
def f18_b6_ties():
    import statsmodels.api as sm, statsmodels.formula.api as smf
    from analysis import all_scores, rate_ci
    e1, e2, _ = all_scores()
    out = []
    for m in ('b6a', 'b6b'):
        z = e2[e2.dataset == 'zch'].copy()
        thr = pd.read_csv(os.path.join(RES, 'c_thresholds.csv')).query('dataset=="zch" and model==@m').iloc[0]
        u = z.beat_id.map(lambda b: __import__('common').stable_seed('tie', b) / 2 ** 32).values
        p_eq = thr.frac_cal_at_threshold
        q_rand = float(np.clip((0.10 - (thr.cal_reject_rate_matched - thr.tie_reject_prob * p_eq)) / p_eq, 0, 1))
        z['main'] = ((z[m] > thr.matched) | ((z[m] == thr.matched) & (u < q_rand))).astype(int)   # old randomised rule
        z['optA'] = ((z[m] > thr.matched) | ((z[m] == thr.matched) & (u < thr.tie_reject_prob))).astype(int)   # current main rule
        for rule in ('main', 'optA'):
            for c in ['ASD', 'VSD', 'PFO', 'PDA']:
                for lv in ['none', '20', '10', '5', '0']:
                    g = z[z.snr_level == lv]
                    q = rate_ci(g[g.cls == c], rule, g[g.cls == 'control'])
                    out.append(dict(model=m, rule=rule, cls=c, snr_level=lv, normal_rate=g[g.cls == 'control'][rule].mean()) | q)
                s = z[z.cls.isin([c, 'control'])].copy(); s['y'] = s[rule]; s['path'] = (s.cls != 'control').astype(int)
                s = s[s.groupby('snr_level').y.transform(lambda v: 0 < v.mean() < 1).astype(bool)].sort_values('patient_id')
                f = smf.gee('y ~ path + C(snr_level)', 'patient_id', s, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
                out.append(dict(model=m, rule=rule, cls=c, snr_level='GEE', odds_ratio=np.exp(f.params['path']), p=f.pvalues['path']))
    t = pd.DataFrame(out); t.to_csv(os.path.join(RES, 'sens_b6_ties_zch.csv'), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    L = ['none', '20', '10', '5', '0']
    ax = axes[0]
    for rule, ls in (('main', (0, (3, 2))), ('optA', '-')):
        g = t[(t.model == 'b6a') & (t.rule == rule) & (t.cls == 'PFO') & (t.snr_level.isin(L))].set_index('snr_level').reindex(L)
        ax.plot(range(5), g.rate * 100, color=CAT[5], ls=ls, marker='o', ms=4, label=f'PFO, {"randomised ties (previous)" if rule == "main" else "all ties rejected (option A, main)"}')
        ax.plot(range(5), g.normal_rate * 100, color=INK2, ls=ls, marker='o', ms=4, label=f'control, {"previous" if rule == "main" else "option A"}')
    ax.set_xticks(range(5), ['clean', '20', '10', '5', '0']); ax.set_xlabel('added noise SNR (dB)'); ax.set_ylabel('B6 rejection rate (%)')
    ax.legend(fontsize=7.5, handlelength=3.5); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.set_title('(a) ZCHSound: randomised ties capped B6\n(dashed = previous rule, solid = option A, now main)')
    ax = axes[1]
    gg = t[(t.model == 'b6a') & (t.snr_level == 'GEE')]
    cl = ['ASD', 'VSD', 'PFO', 'PDA']; x = np.arange(4); w = 0.36
    for k, (rule, col) in enumerate((('main', GREY), ('optA', CAT[5]))):
        v = [gg[(gg.rule == rule) & (gg.cls == c)].odds_ratio.iloc[0] for c in cl]
        p = [gg[(gg.rule == rule) & (gg.cls == c)].p.iloc[0] for c in cl]
        ax.bar(x + (k - 0.5) * w, v, w * 0.92, color=col, label='previous (randomised ties)' if rule == 'main' else 'option A (all ties rejected, main)')
        for xx, vv, pp in zip(x + (k - 0.5) * w, v, p):
            ax.text(xx, vv + 0.04, f'{vv:.2f}' + ('*' if pp < 0.05 else ''), ha='center', fontsize=8, color=INK2)
    ax.axhline(1, color=INK2, lw=1); ax.set_xticks(x, cl); ax.set_ylabel('GEE odds ratio vs control')
    ax.legend(fontsize=8); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.set_title('(b) B6 GEE odds ratio vs control, unadjusted\n(PFO/PDA significant under option A)')
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, 'fig18_sensitivity_b6_ties_zchsound', '17.5 % of clean ZCHSound control calibration beats are unmatched by B6 (> the 10 % target), so the matched threshold lies on the tie at 1.0. '
         'Option A (main analysis since the follow-up): every tied beat is rejected when > 5 % of calibration beats tie, so B6 calibration rejection on ZCHSound is 17.5 % instead of 10 %. * p < 0.05.')


# ------------------------------------------------------------------ F19 B1 SVM ceiling, training curves, B3 features
def f19_misc():
    shutil.copy(os.path.join(FIG, 'fix1_b1svm_score_vs_snr_circor.png'), os.path.join(OUT, 'fig19a_b1svm_ceiling_circor.png'))
    shutil.copy(os.path.join(FIG, 'training_curves.png'), os.path.join(OUT, 'fig19b_training_curves.png'))
    mi = rd('b3_feature_mutual_information.csv').head(25)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.barh(range(len(mi)), mi.mutual_information, color=CAT[2], height=0.65)
    ax.set_yticks(range(len(mi)), mi.feature, fontsize=7.5); ax.set_xlabel('mutual information with the usable/unusable label')
    ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.set_title('B3 CardioPHON: top 25 of 416 re-implemented features (83 selected)')
    fig.tight_layout(); save(fig, 'fig19c_b3_feature_ranking')
    for f in sorted(glob.glob(os.path.join(FIG, 'e2_*.png')) + glob.glob(os.path.join(FIG, 'e2s_*.png'))):
        shutil.copy(f, os.path.join(OUT, 'fig20_' + os.path.basename(f)))


# ------------------------------------------------------------------ F22 hybrid read-out: noise sensitivity vs bias
COND15 = [(ds, c) for ds, c, _ in COLS]


def noise_vs_bias():
    """Per model: noise sensitivity (mean rejection of noisy normal evaluation beats at 10 and 5 dB, CirCor + ZCHSound, matched
    threshold; per seed, then mean +- SD) and bias (seed-robust 'biased against' cells among the 15 conditions; mean |log OR|,
    adjusted, pooled over seeds). Both E2 variants. -> results/noise_vs_bias.csv"""
    rows = []
    a_all = rd('gee_covariates_seeds.csv')
    for variant, f in (('E2', 'e2_equal_noise_per_seed.csv'), ('E2s', 'e2s_equal_noise_per_seed.csv')):
        e = rd(f); e['snr_level'] = e.snr_level.astype(str)
        e = e[(e.threshold == 'matched') & e.dataset.isin(['circor', 'zch']) & e.cls.isin(['normal', 'control']) & e.snr_level.isin(['10', '5'])]
        ns = e.groupby(['model', 'seed']).rate.mean().groupby('model').agg(['mean', 'std', 'count'])
        a = adjusted(a_all[a_all.variant == variant])
        a = a[[(d, c) in COND15 for d, c in zip(a.dataset, a.cls)]]
        for m, _ in ALLM + [('e6_cnn_balanced', ''), ('b8v4', ''), ('b8c', '')]:
            am = a[a.model == m]
            lo = np.abs(np.log(am.pooled_odds_ratio.values)); lo = lo[np.isfinite(lo)]
            rows.append(dict(variant=variant, model=m, name=MNAME[m], noise_sensitivity=ns.loc[m, 'mean'], noise_sensitivity_sd=ns.loc[m, 'std'],
                             n_seeds=int(ns.loc[m, 'count']), n_biased_against=int((am.verdict_code == 1).sum()), n_keeps_more=int((am.verdict_code == -1).sum()),
                             n_conditions=len(am), mean_abs_log_or=lo.mean(), n_or=len(lo)))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RES, 'noise_vs_bias.csv'), index=False)
    return t


def f22_noise_vs_bias():
    t = noise_vs_bias()
    d = t[t.variant == 'E2'].set_index('model')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6))
    ms = [m for m, _ in ALLM] + ['e6_cnn_balanced']
    col = MCOL | {'e6_cnn_balanced': CAT[6]}; mk = MMK | {'e6_cnn_balanced': 'P'}
    ref = min(d.loc['b4_cnn', 'noise_sensitivity'], d.loc['b5_crnn', 'noise_sensitivity'])
    for ax, (ycol, ylab) in zip(axes, (('n_biased_against', 'seed-robust "biased against" conditions (of 15)'),
                                       ('mean_abs_log_or', 'mean |log OR| over the 15 conditions (adjusted, pooled)'))):
        ax.axvspan((ref - 0.05) * 100, 100, color='#e8f1fb', lw=0, zorder=0)
        ax.axvline((ref - 0.05) * 100, color=CAT[0], lw=1, ls=(0, (3, 2)))
        for i, m in enumerate(ms):
            r = d.loc[m]; yv = r[ycol] + (((i % 5) - 2) * 0.06 if ycol == 'n_biased_against' else 0)
            ax.errorbar(r.noise_sensitivity * 100, yv, xerr=(r.noise_sensitivity_sd if np.isfinite(r.noise_sensitivity_sd) else 0) * 100, fmt=mk[m], ms=8,
                        color=col[m], mec=INK if m.startswith('h_') else SURF, mew=1, elinewidth=1, capsize=0, zorder=3, label=MNAME[m])
        ax.set_xlim(0, 80)
        ax.set_xlabel('noise sensitivity: % of noisy NORMAL beats rejected (10 + 5 dB, CirCor + ZCHSound)')
        ax.set_ylabel(ylab); ax.grid(True, color=GRID); ax.set_axisbelow(True)
    axes[0].set_title('(a) Detects noise vs number of biased conditions')
    axes[1].set_title('(b) Detects noise vs typical size of the class effect')
    axes[0].text((ref - 0.05) * 100 + 0.5, axes[0].get_ylim()[1] * 0.97, ' ≥ min(B4, B5) − 5 pp', color=CAT[0], fontsize=8, va='top')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=8, fontsize=8, loc='lower center', bbox_to_anchor=(0.5, 0.045), frameon=False)
    fig.suptitle('Hybrid hypothesis: can a filter be as noise-sensitive as B4/B5 without their bias?  (best = right and low)', x=0.01, ha='left',
                 fontweight='bold', fontsize=12)
    fig.tight_layout(rect=(0, 0.16, 1, 0.95))
    save(fig, 'fig22_noise_vs_bias', 'E2, matched thresholds. x: mean over seeds (± SD) of the rejection rate of noisy normal evaluation beats. y: seed-robust verdict matrix (fig 12a). '
         'Hybrids (black edge) share the colour of their deep model: ▼ AND, ■ CASCADE, ◆ FUSE. Shaded: the pre-specified "close to B4/B5" region (≥ min(B4, B5) − 5 pp). Points in (a) jittered vertically.')


if __name__ == '__main__':
    for f in (f01_pipeline, f02_datasets, f03_e0, f04_models, f05_thresholds, f06_example, f07_diagnostics, f08_f09_gaps,
              f10_gap_vs_snr, f11_forest, f12_verdict, f13_confound, f14_e3, f15_e5, f16_e6, f17_recal, f18_b6_ties, f19_misc,
              f22_noise_vs_bias):
        f()
