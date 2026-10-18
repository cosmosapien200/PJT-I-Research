"""Score test beats with every quality model (E1 raw, E2 equal noise) and PhysioNet windows (Phase B for B6/B7).

  python score.py e1 <ds> [models...]     ds in circor bmdhs pascal zch synth
  python score.py e2 <ds> [models...]
  python score.py e2s <ds> [models...]    E2 with the noise scaled against the clean window's S1+S2 power (post hoc)
  python score.py pn                      B6/B7 window scores on the PhysioNet corpus

Scores are cached per model in data/processed/scores/<exp>_<ds>_<model>.npy, aligned with the index file
data/processed/scores/<exp>_<ds>_index.csv. E2 index: clean evaluation beats x SNR levels (20, 10, 5, 0 dB);
the no-noise level is taken from E1. Noisy windows are regenerated deterministically on demand.
E2s: same beats, clips, offsets and levels as E2; only the reference power changes (S1+S2 samples instead of the whole window).
Seeded models (post hoc) use tags <model>_s<seed>; B4/B5 tags also save the penultimate embedding (<exp>_<ds>_<tag>_emb.npy).
"""
import os, sys, time
import numpy as np, pandas as pd, joblib
from common import PROC, MODELS, WIN, stable_seed, add_noise, add_noise_ref
from noise_banks import load_bank, noise_segment

SC = os.path.join(PROC, 'scores'); os.makedirs(SC, exist_ok=True)
ALL = ['b1_lr', 'b1_svm', 'b2_rf', 'b3_cardiophon', 'b4_cnn', 'b5_crnn', 'e6_cnn_balanced', 'b6', 'b7']
SNRS = [20, 10, 5, 0]
DEEP = ('b4_cnn', 'b5_crnn', 'e6_cnn_balanced')
EMBED = ('b4_cnn', 'b5_crnn')


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


def s1s2_power(clean, bs, seg):
    """Mean power of each clean window over its S1 and S2 samples: every reference S1/S2 segment fully inside the window
    (as used by B7), else the scored beat's own S1 and S2."""
    by_rec = {r: g[['start', 'end', 'state']].values for r, g in seg[seg.state.isin(['S1', 'S2'])].groupby('record_id')}
    out = np.empty(len(bs))
    for i, r in enumerate(bs.itertuples()):
        ws = int(r.centre) - WIN // 2
        sg = by_rec.get(r.record_id, np.empty((0, 3)))
        iv = [(s - ws, e - ws) for s, e, _ in sg if s >= ws and e <= ws + WIN]
        if not iv:
            iv = [(max(r.start - ws, 0), min(r.s1_end - ws, WIN)), (max(r.s2_start - ws, 0), min(r.s2_end - ws, WIN))]
        m = np.zeros(WIN, bool)
        for a, b in iv:
            m[int(a):int(b)] = True
        out[i] = np.mean(clean[i][m].astype(np.float64) ** 2)
    return out


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
    pd.concat(rows, ignore_index=True).to_csv(os.path.join(SC, f'{exp}_{ds}_index.csv'), index=False)
    clean = np.asarray(W[sel], np.float32)
    pref = s1s2_power(clean, bs, segments(ds)) if exp == 'e2s' else None
    for k, s in enumerate(SNRS):
        X = np.stack([add_noise(clean[i], N[i], s) if pref is None else add_noise_ref(clean[i], N[i], s, pref[i])
                      for i in range(len(sel))])
        X /= np.abs(X).max(1, keepdims=True) + 1e-9
        yield s, np.arange(k * len(sel), (k + 1) * len(sel)), X, bs


def run(exp, ds, models):
    t0 = time.time()
    outs = {m: [] for m in models}
    embs = {m: [] for m in models if m.startswith(EMBED)}
    seg = segments(ds) if 'b7' in models else None
    for level, rows, X, b in window_sets(exp, ds):
        need_f = [m for m in models if m.startswith(('b1', 'b2', 'hd_h4'))]
        if need_f:
            fp = os.path.join(SC, f'{exp}_{ds}_{level}_feats.npz')
            if os.path.exists(fp):
                z = np.load(fp); b1, b2 = z['b1'], z['b2']
            else:
                import features as F
                b1, b2 = F.compute(X); np.savez(fp, b1=b1, b2=b2)
            for m in need_f:
                if m.startswith('hd_h4'):   # debiased fusion network: audio + cached Das features
                    import hd
                    outs[m].append(hd.score_h4(m, X, b2)); continue
                clf = joblib.load(os.path.join(MODELS, f'{m}.joblib'))
                outs[m].append(clf.predict_proba(b1 if m.startswith('b1') else b2)[:, 1])
        b3m = [m for m in models if m.startswith('b3_cardiophon')]
        if b3m:
            fp = os.path.join(SC, f'{exp}_{ds}_{level}_b3feats.npz')
            if os.path.exists(fp):
                F3 = np.load(fp)['X']
            else:
                import cardiophon_features as C3
                F3 = C3.compute(X); np.savez(fp, X=F3)
            F3 = F3.astype(np.float64); F3[~np.isfinite(F3)] = np.nan
            for m in b3m:
                outs[m].append(joblib.load(os.path.join(MODELS, f'{m}.joblib')).predict_proba(F3)[:, 1])
        for m in models:
            if m.startswith(EMBED):
                import deep
                p, e = deep.score(m, X, embed=True)
                outs[m].append(p); embs[m].append(e.astype(np.float32))
            elif m.startswith(DEEP):
                import deep
                outs[m].append(deep.score(m, X))
            elif m.startswith('b8'):
                import b6
                outs[m].append(b6.score_noise(X, b.rel_start.values, b.rel_end.values, os.path.join(MODELS, f'{m}.pt')))
            elif m in ('b6', 'b6pn'):
                import b6
                outs[m].append(b6.score_beats(X, b.rel_start.values, b.rel_end.values,
                                              b6.PN_ONLY if m == 'b6pn' else b6.RELEASE).values)
            elif m == 'b7':
                import b7
                outs[m].append(b7.score(X, b.record_id.values, b.centre.values - WIN // 2, seg))
        print(exp, ds, level, len(rows), 'windows', round(time.time() - t0), 's', flush=True)
    for m in models:
        v = np.concatenate(outs[m])
        if m in ('b6', 'b6pn'):
            cols = ['b6a', 'b6b', 'b6_conf', 'b6_matched', 'b6_ok_dur', 'b6_ok_rhythm']
            if m == 'b6pn':
                cols = [c.replace('b6', 'b6pn') for c in cols]
            for j, c in enumerate(cols):
                np.save(os.path.join(SC, f'{exp}_{ds}_{c}.npy'), v[:, j])
        else:
            f = os.path.join(SC, f'{exp}_{ds}_{m}.npy')
            if m in embs and os.path.exists(f) and len(np.load(f)) == len(v):
                # re-run only to obtain embeddings: keep the published scores (GPU re-inference differs by float noise only)
                d = np.abs(np.load(f) - v).max()
                assert d < 1e-4, f'{f}: re-scored values differ by {d}'
                print(m, 'kept existing scores; max re-inference difference', f'{d:.1e}')
            else:
                np.save(f, v)
        if m in embs:
            np.save(os.path.join(SC, f'{exp}_{ds}_{m}_emb.npy'), np.concatenate(embs[m]))


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
