"""Shared paths, constants and signal helpers."""
import os, hashlib, random
import numpy as np, soundfile as sf
from scipy.signal import butter, sosfiltfilt, sosfilt, resample_poly
from math import gcd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, 'data', 'raw')
PROC = os.path.join(ROOT, 'data', 'processed')
MAN = os.path.join(ROOT, 'manifests')
RES = os.path.join(ROOT, 'results')
LOGT = os.path.join(ROOT, 'logs', 'train')
FIG = os.path.join(ROOT, 'figures')
MODELS = os.path.join(ROOT, 'data', 'models')
for d in (PROC, MAN, RES, LOGT, FIG, MODELS):
    os.makedirs(d, exist_ok=True)

FS = 2000
WIN = 5 * FS          # 5 s window
HOP = int(2.5 * FS)   # 2.5 s hop
SEED = 42
PN_ROOT = os.path.join(RAW, 'physionet2016')
CIRCOR_ROOT = os.path.join(RAW, 'circor')
BMD_ROOT = os.path.join(RAW, 'repos', 'BMD-HS-Dataset')
PASCAL_ROOT = os.path.join(RAW, 'pascal')
ZCH_ROOT = os.path.join(RAW, 'zchsound', 'ZCHSound')
PCGK_ROOT = os.path.join(RAW, 'repos', 'PCGKolect_model')


def seed_all(s=SEED):
    random.seed(s); np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    except ImportError:
        pass


def stable_seed(*parts):
    """Deterministic 32-bit seed from strings (used for per-beat noise selection)."""
    h = hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16)


def load_audio(path, max_s=None):
    """Mono float32 at 2 kHz (never upsample: sources are >= 2 kHz)."""
    x, fs = sf.read(path, dtype='float32', always_2d=True)
    x = x.mean(1)
    if max_s:
        x = x[:int(max_s * fs)]
    if fs != FS:
        assert fs > FS, f'{path}: {fs} Hz < 2 kHz'
        g = gcd(FS, fs)
        x = resample_poly(x, FS // g, fs // g)
    return x.astype(np.float32)


_SOS = {}


def _sos(kind, f, order):
    k = (kind, f if np.isscalar(f) else tuple(f), order)
    if k not in _SOS:
        _SOS[k] = butter(order, f, kind, fs=FS, output='sos')
    return _SOS[k]


def highpass(x, f=20, order=2):
    return sosfilt(_sos('highpass', f, order), x, axis=-1).astype(np.float32)


def bandpass(x, lo=25, hi=400, order=4):
    return sosfiltfilt(_sos('bandpass', [lo, hi], order), x, axis=-1).astype(np.float32)


def window_at(x, centre, n=WIN):
    """n-sample window centred on `centre`, reflection-padded at recording edges."""
    s = int(centre) - n // 2
    off = 0                              # index of original sample 0 inside xp
    xp = x
    while s + off < 0 or s + off + n > len(xp):   # repeated reflection for very short clips
        p = min(len(xp) - 1, max(-(s + off), s + off + n - len(xp)))
        xp = np.pad(xp, (p, p), mode='reflect')
        off += p
    return xp[s + off:s + off + n].astype(np.float32)


def add_noise(clean, noise, snr_db):
    """Scale `noise` (same length) so that P_clean / P_noise = snr_db."""
    pc = np.mean(clean.astype(np.float64) ** 2)
    pn = np.mean(noise.astype(np.float64) ** 2) + 1e-20
    return (clean + noise * np.sqrt(pc / (pn * 10 ** (snr_db / 10)))).astype(np.float32)


def add_noise_ref(clean, noise, snr_db, p_ref):
    """As add_noise, but against a given reference power (E2s: the clean window's S1+S2 power): p_ref / P_noise = snr_db."""
    pn = np.mean(noise.astype(np.float64) ** 2) + 1e-20
    return (clean + noise * np.sqrt(p_ref / (pn * 10 ** (snr_db / 10)))).astype(np.float32)


def write_progress_line(section, key, line):
    """Overwrite (or add) one line in PROGRESS.md under `## section`, keyed by `key` substring."""
    import time, re
    p = os.path.join(ROOT, 'PROGRESS.md')
    txt = open(p).read() if os.path.exists(p) else '# Progress\n## Datasets\n## Models (validation AUROC / balanced accuracy)\n## Experiments (one key number each)\n## Blockers\n'
    lines = txt.splitlines()
    lines[0] = f'# Progress   (last updated: {time.strftime("%Y-%m-%d %H:%M")})'
    try:
        i = lines.index(f'## {section}')
    except ValueError:
        lines.append(f'## {section}'); i = len(lines) - 1
    j = i + 1
    while j < len(lines) and not lines[j].startswith('## '):
        if key in lines[j]:
            lines[j] = line
            break
        j += 1
    else:
        lines.insert(j, line)
    open(p, 'w').write('\n'.join(lines) + '\n')
