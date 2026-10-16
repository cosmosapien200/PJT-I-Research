"""Score test beats with every quality model (E1 raw, E2 equal noise) and PhysioNet windows (Phase B for B6/B7).

  python score.py e1 <ds> [models...]     ds in circor bmdhs pascal zch synth
  python score.py e2 <ds> [models...]
  python score.py pn                      B6/B7 window scores on the PhysioNet corpus

Scores are cached per model in data/processed/scores/<exp>_<ds>_<model>.npy, aligned with the index file
data/processed/scores/<exp>_<ds>_index.csv. E2 index: clean evaluation beats x SNR levels (20, 10, 5, 0 dB);
the no-noise level is taken from E1. Noisy windows are regenerated deterministically on demand.
"""
import os, sys, time
import numpy as np, pandas as pd, joblib
from common import PROC, MODELS, WIN, stable_seed, add_noise
from noise_banks import load_bank, noise_segment

SC = os.path.join(PROC, 'scores'); os.makedirs(SC, exist_ok=True)
ALL = ['b1_lr', 'b1_svm', 'b2_rf', 'b3_cardiophon', 'b4_cnn', 'b5_crnn', 'e6_cnn_balanced', 'b6', 'b7']
SNRS = [20, 10, 5, 0]


def beats_table(ds):
    b = pd.read_csv(os.path.join(PROC, f'{ds}_beats.csv'), low_memory=False)
    b['dataset'] = 'circor_synth' if ds == 'synth' else ds
    return b


def segments(ds):
    return pd.read_csv(os.path.join(PROC, f'{ds}_segments.csv.gz'))


def e2_noise(beat_ids):
    """Deterministic Bank 2 noise segment per beat: source 50/50 (speech / DEMAND), then clip and offset."""
    aud, off, meta = load_bank('bank2')
    pools = {s: np.flatnonzero(meta.source.values == s) for s in ('librispeech_test_clean', 'demand')}
    N = np.zeros((len(beat_ids), WIN), np.float32); clips = []
    for i, bid in enumerate(beat_ids):
        rng = np.random.default_rng(stable_seed('e2', bid))
        src = 'librispeech_test_clean' if rng.random() < 0.5 else 'demand'
        idx = int(rng.choice(pools[src]))
        N[i] = noise_segment(aud, off, idx, WIN, rng); clips.append(meta.clip_id[idx])
    return N, clips


def window_sets(exp, ds):
    """Yields (name, row_index_array, windows float32) chunks + writes the index file."""
    b = beats_table(ds)
    W = np.load(os.path.join(PROC, f'{ds}_beat_windows.npy'), mmap_mode='r')
    if exp == 'e1':
        idx = b[['beat_id', 'record_id', 'patient_id', 'dataset', 'cls', 'split', 'clean']].copy()
        idx['snr_level'] = 'none'
        idx.to_csv(os.path.join(SC, f'e1_{ds}_index.csv'), index=False)
        yield 'none', np.arange(len(b)), np.asarray(W, np.float32), b
        return
    sel = np.flatnonzero((b.clean.values == 1) & (b.split.values == 'eval'))
    bs = b.iloc[sel].reset_index(drop=True)
    N, clips = e2_noise(bs.beat_id.values)
    rows = []
    for s in SNRS:
        r = bs[['beat_id', 'record_id', 'patient_id', 'dataset', 'cls', 'split', 'clean']].copy()
        r['snr_level'] = s; r['noise_clip'] = clips; rows.append(r)
    pd.concat(rows, ignore_index=True).to_csv(os.path.join(SC, f'e2_{ds}_index.csv'), index=False)
    clean = np.asarray(W[sel], np.float32)
    for k, s in enumerate(SNRS):
        X = np.stack([add_noise(clean[i], N[i], s) for i in range(len(sel))])
        X /= np.abs(X).max(1, keepdims=True) + 1e-9
        yield s, np.arange(k * len(sel), (k + 1) * len(sel)), X, bs


def run(exp, ds, models):
    t0 = time.time()
    outs = {m: [] for m in models}
    seg = segments(ds) if 'b7' in models else None
    for level, rows, X, b in window_sets(exp, ds):
        need_f = [m for m in models if m.startswith(('b1', 'b2'))]
        if need_f:
            fp = os.path.join(SC, f'{exp}_{ds}_{level}_feats.npz')
            if os.path.exists(fp):
                z = np.load(fp); b1, b2 = z['b1'], z['b2']
            else:
                import features as F
                b1, b2 = F.compute(X); np.savez(fp, b1=b1, b2=b2)
            for m in need_f:
                clf = joblib.load(os.path.join(MODELS, f'{m}.joblib'))
                outs[m].append(clf.predict_proba(b1 if m.startswith('b1') else b2)[:, 1])
        if 'b3_cardiophon' in models:
            fp = os.path.join(SC, f'{exp}_{ds}_{level}_b3feats.npz')
            if os.path.exists(fp):
                F3 = np.load(fp)['X']
            else:
                import cardiophon_features as C3
                F3 = C3.compute(X); np.savez(fp, X=F3)
            F3 = F3.astype(np.float64); F3[~np.isfinite(F3)] = np.nan
            outs['b3_cardiophon'].append(joblib.load(os.path.join(MODELS, 'b3_cardiophon.joblib')).predict_proba(F3)[:, 1])
        for m in models:
            if m in ('b4_cnn', 'b5_crnn', 'e6_cnn_balanced'):
                import deep
                outs[m].append(deep.score(m, X))
            elif m == 'b6':
                import b6
                outs[m].append(b6.score_beats(X, b.rel_start.values, b.rel_end.values).values)
            elif m == 'b7':
                import b7
                outs[m].append(b7.score(X, b.record_id.values, b.centre.values - WIN // 2, seg))
        print(exp, ds, level, len(rows), 'windows', round(time.time() - t0), 's', flush=True)
    for m in models:
        v = np.concatenate(outs[m])
        if m == 'b6':
            cols = ['b6a', 'b6b', 'b6_conf', 'b6_matched', 'b6_ok_dur', 'b6_ok_rhythm']
            for j, c in enumerate(cols):
                np.save(os.path.join(SC, f'{exp}_{ds}_{c}.npy'), v[:, j])
        else:
            np.save(os.path.join(SC, f'{exp}_{ds}_{m}.npy'), v)


def run_pn():
    """B6 and B7 window-level scores on the full PhysioNet corpus (not trained on quality labels)."""
    import b6, b7
    meta = pd.read_csv(os.path.join(PROC, 'pn_windows.csv'))
    W = np.load(os.path.join(PROC, 'pn_windows_audio.npy'), mmap_mode='r')
    seg = pd.read_csv(os.path.join(PROC, 'pn_segments.csv.gz'))
    out = []
    for i in range(0, len(W), 5000):
        X = np.asarray(W[i:i + 5000], np.float32)
        r = b6.score_windows(X)
        r['b7'] = b7.score(X, meta.record_id.values[i:i + 5000], meta.start.values[i:i + 5000], seg)
        out.append(r)
    r = pd.concat(out, ignore_index=True)
    for m in ('b6a', 'b6b', 'b7'):
        pd.DataFrame(dict(window_id=meta.window_id, fold=meta.fold, y=meta.unusable, score=r[m].values)).to_csv(
            os.path.join(PROC, f'val_preds_{m}.csv'), index=False)
    print('pn done')


if __name__ == '__main__':
    if sys.argv[1] == 'pn':
        run_pn()
    else:
        run(sys.argv[1], sys.argv[2], sys.argv[3:] or ALL)
