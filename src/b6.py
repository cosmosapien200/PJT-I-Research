"""B6: PCGKolect segmentation model (released checkpoint, not retrained) as a beat-quality filter.

Pipeline as in the repo: 2 kHz, 20 Hz high-pass (2nd-order Butterworth), 128-pt Hann STFT hop 40,
32 mel bands 25-900 Hz, log, minus mean, /4 -> 1D U-Net -> 5-state posteriors -> duration-constrained
Viterbi (seg/decode.py) -> beats with confidence.
Each 5 s window is reflection-padded to the model's 10,328-sample input (5.164 s); B6 sees only this window.
  B6a: score = 1 - confidence of the decoded beat matching the reference beat (>= 50 % overlap); no match -> 1
  B6b: as B6a, and score = 1 when the matched beat fails the duration (0.3-2.0 s) or rhythm (+-30 %) check
"""
import os, sys
import numpy as np, pandas as pd, torch
from concurrent.futures import ProcessPoolExecutor
from common import PCGK_ROOT, WIN, highpass

sys.path.insert(0, os.path.join(PCGK_ROOT, 'seg'))
sys.path.insert(0, PCGK_ROOT)
import model as segmodel          # noqa: E402  (PCGKolect_model/seg/model.py)
import decode                     # noqa: E402  (PCGKolect_model/seg/decode.py)

PAD_L = (segmodel.SAMPLES - WIN) // 2
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
RELEASE = os.path.join(PCGK_ROOT, 'release', 'seg_final.pt')
# fix 6: same architecture and recipe retrained on PhysioNet 2016 only (no CirCor), see DEVIATIONS.md
PN_ONLY = os.path.join(os.path.dirname(PCGK_ROOT), 'PCGKolect_pnonly', 'runs', 'seg_pnonly_final.pt')
_NET = {}


def net(ckpt=RELEASE):
    if ckpt not in _NET:
        fe, sn = segmodel.Frontend(), segmodel.SegNet()
        sn.load_state_dict(torch.load(ckpt, map_location='cpu'))
        _NET[ckpt] = torch.nn.Sequential(fe, sn).to(DEV).eval()
    return _NET[ckpt]


@torch.no_grad()
def posteriors(W, bs=512, ckpt=RELEASE):
    """W [N, 10000] -> P [N, 256, 5]."""
    out = []
    for i in range(0, len(W), bs):
        x = np.asarray(W[i:i + bs], np.float32)
        x = x / (np.abs(x).max(1, keepdims=True) + 1e-9)
        x = np.pad(x, ((0, 0), (PAD_L, segmodel.SAMPLES - WIN - PAD_L)), mode='reflect')
        x = highpass(x).astype(np.float32)
        lg = net(ckpt)(torch.from_numpy(x).to(DEV))[:, :, 0, :]          # [B, 5, T]
        out.append(torch.softmax(lg, 1).transpose(1, 2).cpu().numpy())
    return np.concatenate(out)


def frame_to_sample(f):
    return f * segmodel.HOP + segmodel.NFFT // 2 - segmodel.HOP // 2 - PAD_L


def _decode_one(args):
    P, rs, re_ = args
    bts = decode.beats(decode.viterbi(P), P)
    if rs is None:   # window-level summary
        if not bts:
            return 1.0, 1.0, 0
        conf = np.array([b['conf'] for b in bts]); ok = np.array([b['ok_dur'] and b['ok_rhythm'] for b in bts])
        return 1 - conf.mean(), 1 - (conf * ok).mean(), len(bts)
    best, bo = None, 0.0
    for b in bts:
        s, e = frame_to_sample(b['start']), frame_to_sample(b['end'])
        ov = max(0, min(e, re_) - max(s, rs)) / max(re_ - rs, 1)
        if ov > bo:
            best, bo = b, ov
    if best is None or bo < 0.5:
        return 1.0, 1.0, 0.0, 0, 0, 0
    a = 1 - best['conf']
    b_ = 1.0 if not (best['ok_dur'] and best['ok_rhythm']) else a
    return a, b_, best['conf'], 1, int(best['ok_dur']), int(best['ok_rhythm'])


def _decode_chunk(items):
    return [_decode_one(it) for it in items]


def _parallel(items, nproc=10, chunk=256):
    parts = [items[i:i + chunk] for i in range(0, len(items), chunk)]
    with ProcessPoolExecutor(nproc) as ex:
        return [r for p in ex.map(_decode_chunk, parts) for r in p]


def score_beats(W, rel_start, rel_end, ckpt=RELEASE):
    P = posteriors(W, ckpt=ckpt)
    res = _parallel([(P[i], int(rel_start[i]), int(rel_end[i])) for i in range(len(P))])
    return pd.DataFrame(res, columns=['b6a', 'b6b', 'b6_conf', 'b6_matched', 'b6_ok_dur', 'b6_ok_rhythm'])


def score_windows(W):
    P = posteriors(W)
    res = _parallel([(P[i], None, None) for i in range(len(P))])
    return pd.DataFrame(res, columns=['b6a', 'b6b', 'b6_nbeats'])


# ---------------------------------------------------------------------- B8 (PCGKolect-v3, proposed)
def score_noise(W, rel_start, rel_end, ckpt):
    """Mean noise-class posterior over the frames whose centre lies inside the scored beat (no decoding)."""
    P = posteriors(W, ckpt=ckpt)                                   # [N, 256, 5]
    centre = np.arange(P.shape[1]) * segmodel.HOP + segmodel.NFFT // 2 - PAD_L
    out = np.empty(len(P))
    for i in range(len(P)):
        m = (centre >= rel_start[i]) & (centre < rel_end[i])
        if not m.any():
            m = np.abs(centre - (rel_start[i] + rel_end[i]) / 2) == np.abs(centre - (rel_start[i] + rel_end[i]) / 2).min()
        out[i] = P[i, m, 4].mean()
    return out
