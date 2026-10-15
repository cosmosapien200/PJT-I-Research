"""Fixed beat segmenter for test sets without beat annotations: Springer LR-HSMM (pyPCG port).

Trained once on PhysioNet 2016 hand-corrected annotations (training data only), 1 kHz input.
  python segmenter.py train
  segment(x2k) -> list of segments (start, end, state) in 2 kHz samples
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy.signal import resample_poly
from common import PROC, MAN, MODELS, SEED

MODEL = os.path.join(MODELS, 'springer_lr_hsmm.json')
STATES = {1: 'S1', 2: 'systole', 3: 'S2', 4: 'diastole'}
N_TRAIN = 400


def train():
    from pyPCG.lr_hsmm import LR_HSMM
    man = pd.read_csv(os.path.join(MAN, 'physionet2016.csv'))
    seg = pd.read_csv(os.path.join(PROC, 'pn_segments.csv.gz'))
    aud = np.load(os.path.join(PROC, 'pn_audio.npz'))
    rng = np.random.default_rng(SEED)
    cand = man[man.sqi == 1].record_id.values
    data, s1s, s2s = [], [], []
    for rid in rng.permutation(cand):
        g = seg[seg.record_id == rid]
        if len(g) < 40:
            continue
        x = resample_poly(aud[rid].astype(np.float64), 1, 2)
        # annotations: S1 / S2 centres, seconds
        s1 = ((g[g.state == 'S1'].start + g[g.state == 'S1'].end) / 2 / 2000).values
        s2 = ((g[g.state == 'S2'].start + g[g.state == 'S2'].end) / 2 / 2000).values
        data.append(x); s1s.append(s1); s2s.append(s2)
        if len(data) >= N_TRAIN:
            break
    m = LR_HSMM()
    m.train_model(data, s1s, s2s)
    m.save_model(MODEL)
    print('trained on', len(data), 'records')


_M = None


def segment(x2k):
    global _M
    if _M is None:
        from pyPCG.lr_hsmm import LR_HSMM
        _M = LR_HSMM(); _M.load_model(MODEL)
    x = resample_poly(np.asarray(x2k, np.float64), 1, 2)
    st, _ = _M.segment_single(x, recalc_timing=True)
    st = np.repeat(np.asarray(st, int), 2)[:len(x2k)]
    if len(st) < len(x2k):
        st = np.pad(st, (0, len(x2k) - len(st)), mode='edge')
    edges = np.flatnonzero(np.diff(st)) + 1
    starts = np.concatenate([[0], edges]); ends = np.concatenate([edges, [len(st)]])
    return [(int(s), int(e), STATES[int(st[s])]) for s, e in zip(starts, ends) if int(st[s]) in STATES]


if __name__ == '__main__':
    globals()[sys.argv[1]]()
