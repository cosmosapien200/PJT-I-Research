"""Hand-crafted window features for the classic quality models.

B1 (Springer-style SQIs, 25-400 Hz Butterworth order 4):
  kurtosis, envelope std, max envelope autocorrelation (lag 0.5-2 s), envelope sample entropy,
  degree of periodicity (pyPCG implementations where available).
B2 (Das et al. 2017 / US11373757B2, the well-defined subset): see das_features().

  compute(windows [N, 10000]) -> (b1 [N, 5], b2 [N, 11])   (multiprocess)
"""
import numpy as np
from scipy.signal import hilbert, butter, sosfilt, sosfiltfilt, resample_poly, welch
from scipy.stats import kurtosis
from concurrent.futures import ProcessPoolExecutor
import warnings
from common import FS, bandpass

B1_NAMES = ['kurtosis', 'env_std', 'env_acf_max', 'env_sampen', 'periodicity']
B2_NAMES = ['acf_sampen', 'acf_kurtosis', 'svd_ratio_min', 'env_power', 'acf_var',
            'sin1_corr', 'sin2_corr', 'sin3_corr', 'pow_ratio_240', 'r500_mean', 'r500_var']
LP20 = butter(2, 20, 'lowpass', fs=FS, output='sos')
EFS = 50   # envelope / autocorrelation rate for Das features
LP10 = butter(2, 10, 'lowpass', fs=EFS, output='sos')


def _z(x):
    x = x.astype(np.float64)
    return (x - x.mean()) / (x.std() + 1e-12)


def sampen(x, m=2, r=0.2):
    from pyPCG.sqi import _sample_entropy_fast
    try:
        v = _sample_entropy_fast(np.asarray(x, np.float64), m, r)
        return v if np.isfinite(v) else 5.0
    except (ValueError, ZeroDivisionError):
        return 5.0


def acf(x):
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    a = np.fft.irfft(f * np.conj(f))[:n]
    return a / (a[0] + 1e-12)


def b1_features(w):
    import pyPCG, pyPCG.sqi as sqi
    x = _z(bandpass(w))
    env = np.abs(hilbert(x))
    sig = pyPCG.pcg_signal(x, FS)
    e50 = resample_poly(sosfiltfilt(LP20, env), 1, FS // EFS)
    a = acf(e50)
    lo, hi = int(0.5 * EFS), int(2.0 * EFS)
    try:
        per = float(sqi.periodicity_score(sig))
    except Exception:
        per = 0.0
    return [kurtosis(x), env.std(), a[lo:hi + 1].max(), sampen(resample_poly(env, 3, 200)), per]


def _rect_sin_corr(a, t, k):
    """Best correlation of a with a least-squares fit of sum_j c_j |sin(pi j f0 t)| + c0 over f0 in 0.5-3 Hz."""
    best = 0.0
    for f0 in np.arange(0.5, 3.0001, 0.025):
        B = np.column_stack([np.ones_like(t)] + [np.abs(np.sin(np.pi * j * f0 * t)) for j in range(1, k + 1)])
        c, *_ = np.linalg.lstsq(B, a, rcond=None)
        fit = B @ c
        if fit.std() > 0:
            best = max(best, float(np.corrcoef(fit, a)[0, 1]))
    return best


def _svd_ratio_min(a):
    out = 1.0
    for L in range(int(0.4 * EFS), int(1.5 * EFS) + 1):
        k = len(a) // L
        if k < 2:
            continue
        s = np.linalg.svd(a[:k * L].reshape(k, L), compute_uv=False)
        out = min(out, s[1] / (s[0] + 1e-12))
    return out


def das_features(w):
    x = _z(w)
    env = sosfilt(LP20, np.abs(hilbert(x)))
    e = resample_poly(env, 1, FS // EFS)
    a = sosfiltfilt(LP10, acf(e))
    t = np.arange(len(a)) / EFS
    f, P = welch(x, FS, nperseg=1024)
    pr = P[f <= 240].sum() / (P[f > 240].sum() + 1e-12)
    rs = []
    for s in range(0, len(x) - 2 * FS + 1, int(1.5 * FS)):        # 2 s frames, 0.5 s overlap
        f2, P2 = welch(x[s:s + 2 * FS], FS, nperseg=512)
        rs.append(P2[f2 <= 500].sum() / (P2[f2 > 500].sum() + 1e-12))
    rs = np.log10(np.array(rs) + 1e-12)
    return [sampen(a), kurtosis(a), _svd_ratio_min(a), float(np.mean(env ** 2)), float(np.var(a)),
            _rect_sin_corr(a, t, 1), _rect_sin_corr(a, t, 2), _rect_sin_corr(a, t, 3),
            np.log10(pr + 1e-12), rs.mean(), rs.var()]


def _chunk(ws):
    warnings.filterwarnings('ignore')
    return np.array([b1_features(w) for w in ws], np.float32), np.array([das_features(w) for w in ws], np.float32)


def compute(W, nproc=10, chunk=64):
    W = np.asarray(W)
    parts = [W[i:i + chunk].astype(np.float32) for i in range(0, len(W), chunk)]
    with ProcessPoolExecutor(nproc) as ex:
        res = list(ex.map(_chunk, parts))
    b1 = np.concatenate([r[0] for r in res]); b2 = np.concatenate([r[1] for r in res])
    return np.nan_to_num(b1, nan=0, posinf=0, neginf=0), np.nan_to_num(b2, nan=0, posinf=0, neginf=0)
