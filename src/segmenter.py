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
HR_ADULT = (30, 120)   # pyPCG default (adult populations: BMD-HS, PASCAL)
HR_CHILD = (30, 220)   # paediatric populations (ZCHSound: neonates to adolescents); (30, 120) halves infant heart rates
HR_RANGE = HR_ADULT


def hr_sys(x, fs=1000, hr_range=HR_RANGE):
    """Springer heart-rate / systole estimate (pyPCG _get_hr_sys) with a paediatric-capable range; the systole
    search window [0.2 s, RR/2] is shrunk to [0.25 RR, RR/2] when RR/2 < 0.2 s (pyPCG fails there)."""
    import scipy.signal as sgn
    from pyPCG.lr_hsmm import _spike_removal, _h_envelope_feature
    f = sgn.sosfiltfilt(sgn.butter(4, (25, 400), 'bandpass', output='sos', fs=fs), x)
    h = _h_envelope_feature(_spike_removal(f, fs), fs)
    y = h - h.mean()
    c = sgn.correlate(y, y, method='fft')[len(h):]
    c = c / (np.max(np.abs(c)) + 1e-12)
    lo, hi = round(60 / hr_range[1] * fs), min(round(60 / hr_range[0] * fs), len(c) - 1)
    hr = 60 / ((np.argmax(c[lo:hi]) + lo) / fs)
    rr = 60 / hr * fs
    smin = min(round(0.2 * fs), round(0.25 * rr))
    smax = max(round(rr / 2), smin + 1)
    return hr, (np.argmax(c[smin:smax]) + smin) / fs


def segment(x2k, hr_range=HR_RANGE):
    global _M
    if _M is None:
        from pyPCG.lr_hsmm import LR_HSMM
        _M = LR_HSMM(); _M.load_model(MODEL)
    from pyPCG.lr_hsmm import _generate_features, _get_duration_distributions, _expand_states
    x = resample_poly(np.asarray(x2k, np.float64), 1, 2)
    if tuple(hr_range) == HR_ADULT:     # original pyPCG path (identical to the first analysis)
        _M.expected_hr_range = HR_ADULT
        st, _ = _M.segment_single(x, recalc_timing=True)
        return _states_to_segments(st, len(x2k))
    F = np.array(_generate_features(x, _M.signal_fs, _M.feature_fs, _M.bandpass_frq), dtype=np.float64)
    hr, sy = hr_sys(x, _M.signal_fs, hr_range)
    _M.hsmm_model.durations = _get_duration_distributions(hr, sy, _M.feature_fs, _M.mean_s1_len, _M.mean_s2_len,
                                                          _M.std_s1_len, _M.std_s2_len)
    st = _expand_states(_M.hsmm_model.decode(F.T) + 1, _M.feature_fs, _M.signal_fs, len(x))
    return _states_to_segments(st, len(x2k))


def _states_to_segments(st, n2k):
    x2k = np.empty(n2k)
    st = np.repeat(np.asarray(st, int), 2)[:len(x2k)]
    if len(st) < len(x2k):
        st = np.pad(st, (0, len(x2k) - len(st)), mode='edge')
    edges = np.flatnonzero(np.diff(st)) + 1
    starts = np.concatenate([[0], edges]); ends = np.concatenate([edges, [len(st)]])
    return [(int(s), int(e), STATES[int(st[s])]) for s, e in zip(starts, ends) if int(st[s]) in STATES]


if __name__ == '__main__':
    globals()[sys.argv[1]]()
