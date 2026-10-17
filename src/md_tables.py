"""Render compact markdown tables from results/*.csv (input for RESULTS.md). Writes results/tables.md.
Gaps are the mean over seeds 42-44 (± SD across seeds for seeded models and hybrids), with the seed-averaged 95 % CI."""
import os
import numpy as np, pandas as pd
from common import RES

NAMES = {'b1_lr': 'B1 LR', 'b1_svm': 'B1 SVM', 'b2_rf': 'B2 RF', 'b3_cardiophon': 'B3 CardioPHON', 'b4_cnn': 'B4 CNN', 'b5_crnn': 'B5 CRNN',
         'e6_cnn_balanced': 'E6 bal. CNN', 'b6a': 'B6a', 'b6b': 'B6b', 'b7': 'B7 SNR', 'b6pna': 'B6pn-a (PN only)', 'b6pnb': 'B6pn-b (PN only)', 'b8': 'B8 PCGKolect-v3', 'b8noaug': 'B8 no-aug (ablation)', 'b8v4': 'B8v4 PCGKolect-v4', 'b8c': 'B8c PCGKolect-v3c',
         'h_and_b4': 'H-AND B4', 'h_and_b5': 'H-AND B5', 'h_cas_b4': 'H-CASCADE B4', 'h_cas_b5': 'H-CASCADE B5', 'h_fuse_b4': 'H-FUSE B4', 'h_fuse_b5': 'H-FUSE B5',
         'hd_avg': 'HD1 avg', 'hd_and': 'HD2 min', 'hd_lr': 'HD3 LR', 'hd_h4_l01': 'HD4 λ0.1', 'hd_h4_l03': 'HD4 λ0.3', 'hd_h4_l10': 'HD4 λ1.0'}
SEEDED = {'b2_rf', 'b3_cardiophon', 'b4_cnn', 'b5_crnn', 'e6_cnn_balanced'} | {m for m in NAMES if m.startswith(('h_', 'hd_'))}
MAIN = {'circor': 'murmur_systolic', 'zch': 'VSD', 'bmdhs': 'multi', 'pascal': 'murmur'}


def pct(r, lo, hi):
    return f'{100 * r:.1f} ({100 * lo:.1f}, {100 * hi:.1f})'


def diff(r, lo, hi, sd=None):
    return f'{100 * r:+.1f}' + (f' ± {100 * sd:.1f}' if sd is not None and np.isfinite(sd) else '') + f' ({100 * lo:+.1f}, {100 * hi:+.1f})'


def md(df):
    cols = list(df.columns)
    out = ['| ' + ' | '.join(map(str, cols)) + ' |', '|' + '---|' * len(cols)]
    out += ['| ' + ' | '.join(map(str, r)) + ' |' for r in df.values]
    return '\n'.join(out)


def main():
    s = []
    for variant in ('e2', 'e2s'):
        e2 = pd.read_csv(os.path.join(RES, f'{variant}_equal_noise_seeds.csv')); e2['snr_level'] = e2.snr_level.astype(str)
        for ds in ['circor', 'zch', 'bmdhs', 'pascal']:
            g = e2[(e2.dataset == ds) & (e2.threshold == 'matched')]
            for c in sorted(g.cls.unique()):
                if c in ('normal', 'control'):
                    continue
                t = g[g.cls == c]
                if t.empty:
                    continue
                piv = t.pivot_table(index='model', columns='snr_level', values=['diff_vs_normal_mean', 'diff_vs_normal_sd', 'diff_ci_lo_mean', 'diff_ci_hi_mean'])
                rows = []
                for m in NAMES:
                    if m not in piv.index:
                        continue
                    rows.append([NAMES[m]] + [diff(piv.loc[m, ('diff_vs_normal_mean', l)], piv.loc[m, ('diff_ci_lo_mean', l)], piv.loc[m, ('diff_ci_hi_mean', l)],
                                                   piv.loc[m, ('diff_vs_normal_sd', l)] if m in SEEDED else None)
                                              if ('diff_vs_normal_mean', l) in piv.columns else '' for l in ['none', '20', '10', '5', '0']])
                ref = 'synthetic_regular' if c.endswith('_vs_regular') else 'normal'
                s.append(f'### {variant.upper()} {ds}: {c.replace("_vs_regular", "")} minus {ref}, rejection-rate difference in pp, mean ± SD over 3 seeds '
                         f'(seed-averaged 95% CI), matched threshold\n')
                s.append(md(pd.DataFrame(rows, columns=['model', 'clean', '20 dB', '10 dB', '5 dB', '0 dB'])) + '\n')
    open(os.path.join(RES, 'tables.md'), 'w').write('\n'.join(s))
    print('tables.md written')


if __name__ == '__main__':
    main()
