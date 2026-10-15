"""Phase A for PhysioNet 2016: manifest + folds, E0 label check, noise stretches, state segments.

Outputs
  manifests/physionet2016.csv          one row per record (patient, fold, labels, sqi)
  results/e0_label_check.csv
  data/processed/pn_segments.csv.gz    hand-corrected states (S1/systole/S2/diastole) in samples
  data/processed/pn_noise.csv          '(N ... N)' stretches in samples
  data/processed/pn_audio.npz          raw 2 kHz audio of every record (float32)
"""
import os, glob
import numpy as np, pandas as pd, scipy.io as sio
from sklearn.model_selection import StratifiedGroupKFold
from common import PN_ROOT, MAN, RES, PROC, FS, SEED, load_audio, write_progress_line

ANN = os.path.join(PN_ROOT, 'annotations')


def manifest():
    rows = []
    for sub in 'abcdef':
        ref = pd.read_csv(os.path.join(ANN, 'updated', f'training-{sub}', 'REFERENCE_withSQI.csv'),
                          header=None, names=['record_id', 'label', 'sqi'])
        ref['subset'] = sub
        rows.append(ref)
    m = pd.concat(rows, ignore_index=True)
    app = pd.read_csv(os.path.join(ANN, 'Online Appendix_training set.csv'))
    app = app.rename(columns={'Challenge record name': 'record_id', 'Subject ID': 'subject', 'Diagnosis': 'diagnosis'})
    m = m.merge(app[['record_id', 'subject', 'diagnosis']], on='record_id', how='left')
    # patient id: subject id for subset b, otherwise the record itself (no subject IDs)
    m['patient_id'] = np.where((m.subset == 'b') & m.subject.notna(),
                               'pn_b_S' + m.subject.fillna(-1).astype(int).astype(str), 'pn_' + m.record_id)
    m['abnormal'] = (m.label == 1).astype(int)
    m['dataset'] = 'physionet2016'
    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    m['fold'] = -1
    strat = m.abnormal * 2 + m.sqi
    for k, (_, te) in enumerate(sgk.split(m, strat, groups=m.patient_id)):
        m.loc[m.index[te], 'fold'] = k
    m['split'] = 'train_fold' + m.fold.astype(str)
    return m


def e0(m):
    out = []
    for sub, g in [('all', m)] + [(s, m[m.subset == s]) for s in 'abcdef']:
        for name, cls in [('normal', 0), ('abnormal', 1)]:
            x = g[g.abnormal == cls]
            n0 = int((x.sqi == 0).sum())
            out.append(dict(subset=sub, cls=name, n_records=len(x), n_sqi0=n0,
                            sqi0_rate=n0 / max(len(x), 1)))
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(RES, 'e0_label_check.csv'), index=False)
    a = df[df.subset == 'all'].set_index('cls').sqi0_rate
    write_progress_line('Experiments (one key number each)', 'E0',
                        f'- [x] E0 label check: SQI-0 rate normal {a["normal"]:.1%} vs abnormal {a["abnormal"]:.1%}')
    return df


def parse_ann(sub, rid, n):
    """Returns state segments [(start, end, state)] and noise stretches [(start, end)] in samples."""
    p = os.path.join(ANN, 'hand_corrected', f'training-{sub}_StateAns', f'{rid}_StateAns.mat')
    if not os.path.exists(p):
        return [], []
    a = sio.loadmat(p)
    a = a[[k for k in a if not k.startswith('__')][0]]
    starts = [int(np.squeeze(r[0])) - 1 for r in a]
    labs = [str(r[1].flat[0].flat[0]) for r in a]
    segs, noise, n_open = [], [], None
    for i, (st, lb) in enumerate(zip(starts, labs)):
        if lb == '(N':
            n_open = st
        elif lb == 'N)':
            if n_open is not None:
                noise.append((n_open, min(st, n)))
            n_open = None
        elif lb in ('S1', 'systole', 'S2', 'diastole') and n_open is None and i + 1 < len(starts):
            en = min(starts[i + 1], n)
            if en > st:
                segs.append((st, en, lb))
    if n_open is not None:
        noise.append((n_open, n))
    return segs, noise


def main():
    m = manifest()
    m[['dataset', 'record_id', 'patient_id', 'split', 'fold', 'subset', 'label', 'abnormal', 'sqi', 'diagnosis']] \
        .to_csv(os.path.join(MAN, 'physionet2016.csv'), index=False)
    e0(m)
    audio, seg_rows, noise_rows = {}, [], []
    for r in m.itertuples():
        x = load_audio(os.path.join(PN_ROOT, f'training-{r.subset}', r.record_id + '.wav'))
        audio[r.record_id] = x
        segs, noise = parse_ann(r.subset, r.record_id, len(x))
        seg_rows += [(r.record_id, s, e, l) for s, e, l in segs]
        noise_rows += [(r.record_id, s, e) for s, e in noise]
    pd.DataFrame(seg_rows, columns=['record_id', 'start', 'end', 'state']).to_csv(
        os.path.join(PROC, 'pn_segments.csv.gz'), index=False)
    pd.DataFrame(noise_rows, columns=['record_id', 'start', 'end']).to_csv(os.path.join(PROC, 'pn_noise.csv'), index=False)
    np.savez(os.path.join(PROC, 'pn_audio.npz'), **audio)
    dur = sum(len(v) for v in audio.values()) / FS / 3600
    write_progress_line('Datasets', 'PhysioNet', f'- [x] PhysioNet 2016: {len(m):,} records, {dur:.1f} h, 2 kHz, '
                        f'{m.patient_id.nunique():,} patient groups, {len(noise_rows)} noise stretches')
    print(len(m), 'records', len(noise_rows), 'noise stretches', len(seg_rows), 'segments')


if __name__ == '__main__':
    main()
