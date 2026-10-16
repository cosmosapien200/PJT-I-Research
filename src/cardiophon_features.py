"""B3 CardioPHON: re-implementation of its quality feature set (Grooby et al.-style heart/lung sound quality features).

The CardioPHON repository ships only pre-extracted features (416 columns) and the classifier script, so the
features are re-implemented here from the column names and the Grooby et al. descriptions. Definitions are
approximations where the names are not self-explanatory (see DEVIATIONS.md). Input: 5 s window at 2 kHz ->
resampled to 1 kHz -> extended to 6 s by concatenating copies (as CardioPHON does for short recordings).

  extract(w2k) -> np.ndarray [len(NAMES)]
"""
import warnings
import numpy as np
from scipy import signal as sg
from scipy.stats import kurtosis, skew
import pywt, librosa
from sklearn.mixture import GaussianMixture

FS = 1000
EFS = 50
DUR = 6 * FS
HEART = (20, 200)
LUNG = (100, 500)


# ---------------------------------------------------------------- helpers
def _bp(x, lo, hi, order=4):
    hi = min(hi, FS / 2 - 1)
    return sg.sosfiltfilt(sg.butter(order, [lo, hi], 'bandpass', fs=FS, output='sos'), x)


def _lp(x, f, fs, order=2):
    return sg.sosfiltfilt(sg.butter(order, f, 'lowpass', fs=fs, output='sos'), x)


def _norm(x):
    x = np.asarray(x, np.float64)
    return (x - x.mean()) / (x.std() + 1e-12)


def _sampen(x, m=2, r=0.2):
    from pyPCG.sqi import _sample_entropy_fast
    x = np.asarray(x, np.float64)
    if len(x) < m + 3 or x.std() == 0:
        return 5.0
    try:
        v = _sample_entropy_fast(x, m, r)
        return float(v) if np.isfinite(v) else 5.0
    except Exception:
        return 5.0


def _acf(x):
    x = np.asarray(x, np.float64) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    a = np.fft.irfft(f * np.conj(f))[:n]
    return a / (a[0] + 1e-12)


def _entropies(p):
    p = np.asarray(p, np.float64)
    p = p[p > 0]
    p = p / p.sum()
    sh = -np.sum(p * np.log2(p))
    ts = (1 - np.sum(p ** 2))                   # Tsallis q = 2
    rn = -np.log2(np.sum(p ** 2))               # Renyi alpha = 2
    return sh, ts, rn


def _hist_p(x, bins=64):
    h, _ = np.histogram(x, bins=bins)
    return h + 1e-12


def _mode(x, bins=32):
    h, e = np.histogram(x, bins=bins)
    i = np.argmax(h)
    return 0.5 * (e[i] + e[i + 1])


def _hjorth(x):
    d1, d2 = np.diff(x), np.diff(x, 2)
    v0, v1, v2 = np.var(x) + 1e-12, np.var(d1) + 1e-12, np.var(d2) + 1e-12
    return np.sqrt(v2 / v1) / np.sqrt(v1 / v0)


def _lsf(a):
    """Line spectral frequencies of an LPC polynomial a (a[0] = 1)."""
    a = np.asarray(a, np.float64)
    p = len(a) - 1
    a1 = np.concatenate([a, [0]])
    P = a1 + a1[::-1]
    Q = a1 - a1[::-1]
    ang = np.sort(np.concatenate([np.angle(np.roots(P)), np.angle(np.roots(Q))]))
    ang = ang[(ang > 1e-6) & (ang < np.pi - 1e-6)]
    out = np.full(p, np.nan)
    out[:min(p, len(ang))] = ang[:p]
    return out


def _peaks_info(a, lo, hi):
    """Highest and second-highest autocorrelation peak in lag range [lo, hi) samples: value, lag, prominence."""
    seg = a[lo:hi]
    pk, pr = sg.find_peaks(seg, prominence=0)
    if len(pk) == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    o = np.argsort(seg[pk])[::-1]
    v1, l1, p1 = seg[pk[o[0]]], (pk[o[0]] + lo), pr['prominences'][o[0]]
    v2, p2 = (seg[pk[o[1]]], pr['prominences'][o[1]]) if len(o) > 1 else (0.0, 0.0)
    return [v1, l1, p1, p2, v2]


def _period(env, fs=EFS):
    a = _acf(env)
    lo, hi = int(0.3 * fs), int(2.0 * fs)
    return lo + int(np.argmax(a[lo:hi])) if len(a) > hi else int(fs)


def _cycle_corr(env, per):
    k = len(env) // per
    if k < 2:
        return 0.0, 0.0
    M = env[:k * per].reshape(k, per)
    c = [np.corrcoef(M[i], M[i + 1])[0, 1] for i in range(k - 1)]
    c = np.nan_to_num(c)
    return float(np.mean(c)), float(np.std(c))


def _hr_std(env, fs=EFS):
    hrs = []
    L, step = 3 * fs, fs // 2
    for s in range(0, len(env) - L + 1, step):
        hrs.append(60 * fs / _period(env[s:s + L], fs))
    return float(np.std(hrs)) if hrs else 0.0


# ---------------------------------------------------------------- envelopes (50 Hz)
def envelopes(x):
    from pyPCG.lr_hsmm import _generate_features
    henv, env, psd, wt = _generate_features(x, FS, EFS, (25, 400))
    n = len(henv)
    xb = _bp(x, 25, 400)
    # PSD envelope over the raw (not band-passed) signal
    f, t, S = sg.spectrogram(x, FS, nperseg=50, noverlap=30)
    psd_o = np.interp(np.linspace(0, len(t) - 1, n), np.arange(len(t)), S[(f >= 40) & (f <= 60)].mean(0))
    stft = np.interp(np.linspace(0, len(t) - 1, n), np.arange(len(t)), S[(f >= 20) & (f <= 200)].sum(0))
    xn = xb / (np.abs(xb).max() + 1e-12)
    sh = -(xn ** 2) * np.log(xn ** 2 + 1e-12)
    shannon = sg.resample_poly(_lp(sh, 20, FS), 1, FS // EFS)[:n]
    E = dict(psd=psd, psd_original=psd_o, wavelet=wt, homomorphic=henv, hilbert=env, stft=stft, shannon=shannon)
    return {k: _norm(np.resize(v, n)) for k, v in E.items()}


# ---------------------------------------------------------------- segmentation (two HSMM variants)
_M = None


def _model():
    global _M
    if _M is None:
        from pyPCG.lr_hsmm import LR_HSMM
        import os
        from common import MODELS
        _M = LR_HSMM(); _M.load_model(os.path.join(MODELS, 'springer_lr_hsmm.json'))
    return _M


def segmentations(x):
    """Springer (per-recording timing) and Schmidt-like (fixed population timing) state sequences at 50 Hz."""
    from pyPCG.lr_hsmm import _generate_features, _get_hr_sys, _get_duration_distributions
    m = _model()
    F = np.array(_generate_features(x, FS, EFS, m.bandpass_frq), dtype=np.float64).T
    hr, sy = _get_hr_sys(x, FS, m.bandpass_frq, *m.expected_hr_range)
    base = m.hsmm_model.durations
    m.hsmm_model.durations = _get_duration_distributions(hr, sy, EFS, m.mean_s1_len, m.mean_s2_len, m.std_s1_len, m.std_s2_len)
    s_spr = m.hsmm_model.decode(F) + 1
    m.hsmm_model.durations = base
    s_sch = m.hsmm_model.decode(F) + 1
    return np.asarray(s_spr), np.asarray(s_sch)


RANGES = {1: (0.04, 0.20), 2: (0.10, 0.50), 3: (0.04, 0.16), 4: (0.15, 1.50)}


def _cycles(st):
    """Complete cycles (S1 start .. next S1 start) with per-state durations in s."""
    e = np.flatnonzero(np.diff(st)) + 1
    starts, ends = np.concatenate([[0], e]), np.concatenate([e, [len(st)]])
    runs = [(int(st[s]), s, en) for s, en in zip(starts, ends)]
    cyc = []
    for i in range(len(runs) - 4):
        if [r[0] for r in runs[i:i + 5]] == [1, 2, 3, 4, 1] and runs[i][1] > 0:
            cyc.append(dict(start=runs[i][1], end=runs[i + 4][1],
                            d={runs[i + k][0]: (runs[i + k][2] - runs[i + k][1]) / EFS for k in range(4)}))
    return cyc


def seg_features(st, envs, tag):
    cyc = _cycles(st)
    out = {}
    names = {1: 's1', 2: 'sys', 3: 's2', 4: 'dia'}
    bad_any = np.zeros(len(cyc), bool)
    for k, nm in names.items():
        if cyc:
            d = np.array([c['d'][k] for c in cyc])
            lo, hi = RANGES[k]
            bad = (d < lo) | (d > hi) | (np.abs(d - np.median(d)) > 0.3 * np.median(d) + 1e-9)
            bad_any |= bad
            out[f'percentage_bad_cycles_{nm}{tag}'] = bad.mean()
        else:
            out[f'percentage_bad_cycles_{nm}{tag}'] = 1.0
    out[f'percentage_bad_cycles_overall{tag}'] = bad_any.mean() if cyc else 1.0
    cov = sum(c['end'] - c['start'] for c in cyc) / len(st)
    env = envs['homomorphic']
    if len(cyc) >= 2:
        L = int(np.median([c['end'] - c['start'] for c in cyc]))
        M = np.array([np.interp(np.linspace(c['start'], c['end'] - 1, L), np.arange(len(env)), env) for c in cyc])
        tmpl = M.mean(0)
        cc = np.nan_to_num([np.corrcoef(m, tmpl)[0, 1] for m in M])
        q, qa = float(np.median(cc)), float(np.mean(cc))
    else:
        q = qa = 0.0
    out[f'seg_qual{tag}'] = q * cov
    out[f'seg_qual_avg{tag}'] = qa
    out[f'seg_qual_inv{tag}'] = 1 - q * cov
    out[f'seg_qual_inv_avg{tag}'] = 1 - qa
    for en, ek in (('psd', 'psd'), ('psd_ori', 'psd_original'), ('wav', 'wavelet'), ('homo', 'homomorphic'),
                   ('hil', 'hilbert'), ('stft', 'stft'), ('shannon', 'shannon')):
        if tag and en == 'stft':
            continue
        e = envs[ek]
        rest = e[(st == 2) | (st == 4)].mean() if np.any((st == 2) | (st == 4)) else 0.0
        for s, nm in ((1, 's1'), (3, 's2')):
            out[f'seg_hsmm_qf_{nm}_{en}{tag}'] = (e[st == s].mean() - rest) if np.any(st == s) else 0.0
    rr = np.array([c['d'][1] + c['d'][2] + c['d'][3] + c['d'][4] for c in cyc])
    if len(rr) >= 3:
        dd = np.diff(rr)
        out[f'seg_per_abnormal_rmssd{tag}'] = np.mean(np.abs(dd) > 0.15 * np.mean(rr))
        out[f'seg_per_abnormal_zero{tag}'] = np.mean(np.abs(dd) < 1e-9)
        out[f'seg_per_abnormal_sd1{tag}'] = np.std(dd) / np.sqrt(2) / (np.mean(rr) + 1e-9)
    else:
        out[f'seg_per_abnormal_rmssd{tag}'] = out[f'seg_per_abnormal_zero{tag}'] = 1.0
        out[f'seg_per_abnormal_sd1{tag}'] = 1.0
    return out, cyc


def _peak_windows(env, win_s, fs=EFS):
    pk, _ = sg.find_peaks(env, distance=int(0.15 * fs), prominence=0.3)
    L = int(win_s * fs)
    ok = []
    for s in range(0, len(env) - L + 1, L // 2):
        n = np.sum((pk >= s) & (pk < s + L))
        ok.append(2 * win_s * 0.5 <= n <= 2 * win_s * 3.5)     # 30-210 bpm, two sounds per beat
    return float(np.mean(ok)) if ok else 0.0, pk


# ---------------------------------------------------------------- main
def extract_dict(w2k):
    warnings.filterwarnings('ignore')
    x = sg.resample_poly(np.asarray(w2k, np.float64), 1, 2)
    x = np.tile(x, int(np.ceil(DUR / len(x))))[:DUR]
    peak_norm = x / (np.abs(x).max() + 1e-12)
    x = _norm(x)
    o = {}
    # --- time domain / audio
    f, P = sg.welch(x, FS, nperseg=1024)
    Pn = P / (P.sum() + 1e-12)
    o['audio_se'] = _entropies(Pn)[0] / np.log2(len(Pn))
    o['audio_clip'] = np.mean(np.abs(peak_norm) >= 0.95)
    for L, tag in ((2000, '_2000'), (1000, '_1000')):
        fr = librosa.util.frame(x, frame_length=L // 10, hop_length=L // 20)
        en = (fr ** 2).mean(0)
        o[f'audio_rate_average_energy{tag}'] = en.mean() / (en.max() + 1e-12)
        o[f'audio_rate_average_energy2{tag}'] = np.mean(en > 0.5 * en.mean())
    o['audio_heart_con1'] = P[(f >= 20) & (f <= 200)].sum() / (P.sum() + 1e-12)
    o['audio_heart_con2'] = P[(f >= 20) & (f <= 100)].sum() / (P[(f > 100)].sum() + 1e-12)
    o['audio_skew'] = skew(peak_norm); o['audio_kurtosis'] = kurtosis(peak_norm); o['audio_var'] = np.var(peak_norm)
    o['audio_high_var'] = np.var(_bp(peak_norm, 200, 499))
    o['audio_zcr'] = np.mean(np.abs(np.diff(np.sign(x))) > 0)
    a10 = librosa.lpc(x, order=10)
    for i in range(11):
        o[f'audio_lpc10_{i + 1}'] = a10[i]
    a6 = librosa.lpc(x, order=6)
    for i in range(7):
        o[f'audio_lpc6_{i + 1}'] = a6[i]
    for i, v in enumerate(_lsf(a6)):
        o[f'audio_lsf6_{i + 1}'] = v
    o['audio_shannon'], o['audio_tsallis'], o['audio_renyi'] = _entropies(_hist_p(x))
    envs = envelopes(x)
    ea = _acf(envs['hilbert'])
    o['audio_periodicity_heart'] = ea[int(0.3 * EFS):int(2 * EFS)].max()
    o['audio_periodicity_lung'] = ea[int(2 * EFS):int(5 * EFS)].max()
    o['audio_autocorr_kurtosis'] = kurtosis(ea)
    # --- wavelet
    c = pywt.wavedec(x, 'db4', level=7)
    E = np.array([np.sum(ci ** 2) for ci in c]); E = E / E.sum()
    for i in range(8):
        o[f'wav_energy_{i + 1}'] = E[i]
    o['wav_log'] = np.sum(np.log(E + 1e-12))
    c5 = pywt.wavedec(x, 'db4', level=5)
    for nm, ci in (('a5', c5[0]), ('d5', c5[1]), ('d4', c5[2])):
        o[f'wav_shannon_{nm}'], o[f'wav_tsallis_{nm}'], o[f'wav_renyi_{nm}'] = _entropies(ci ** 2)
    d4 = c5[2]
    o['wav_RMSSD'] = np.sqrt(np.mean(np.diff(d4) ** 2)) / (np.std(d4) + 1e-12)
    o['wav_RZC'] = np.mean(np.abs(np.diff(np.sign(d4))) > 0)
    # --- f0
    f0 = librosa.yin(x.astype(np.float32), fmin=30, fmax=450, sr=FS, frame_length=512, hop_length=128)
    o['f0_overall'] = np.median(f0); o['f0_min'] = f0.min(); o['f0_max'] = f0.max(); o['f0_mean'] = f0.mean()
    o['f0_median'] = np.median(f0); o['f0_mode'] = _mode(f0); o['f0_var'] = np.var(f0)
    o['cry_f0'] = np.mean((f0 >= 250) & (f0 <= 450))
    # --- truncated autocorrelation (heart lags 0.2-2 s); "strunc" = smoothed
    tr = ea[int(0.2 * EFS):int(2 * EFS)]
    st_ = _lp(tr, 5, EFS) if len(tr) > 12 else tr
    k = 1
    for m in (1, 2, 3):
        for r in (0.1, 0.2):
            o[f'trunc_autocorr_se{k}'] = _sampen(tr, m, r); o[f'strunc_autocorr_se{k}'] = _sampen(st_, m, r); k += 1
    o['trunc_autocorr_hjorth'] = _hjorth(tr); o['strunc_autocorr_hjorth'] = _hjorth(st_)
    o['trunc_autocorr_var'] = np.var(tr); o['strunc_autocorr_var'] = np.var(st_)
    for nm, rng in (('heart', (0.3, 1.5)), ('lung', (1.5, 3.0))):
        for t_, a_ in (('trunc', tr), ('strunc', st_)):
            L = int(rng[0] * EFS)
            kk = len(ea) // max(L, 1)
            s_ = np.linalg.svd(ea[:kk * L].reshape(kk, L), compute_uv=False) if kk >= 2 else np.array([1, 1])
            o[f'{t_}_autocorr_svd_{nm}'] = s_[1] / (s_[0] + 1e-12)
    for nm, rng in (('heart', (0.3, 2.0)), ('lung', (2.0, 5.0))):
        lo, hi = int(rng[0] * EFS), int(rng[1] * EFS)
        v1, l1, p1, p2, v2 = _peaks_info(ea, lo, hi)
        o[f'autocorr_cc_{nm}'] = np.corrcoef(ea[:hi - lo], ea[lo:hi])[0, 1] if hi < len(ea) else 0.0
        o[f'autocorr_max_val_{nm}'] = v1; o[f'autocorr_cycle_dur_{nm}'] = l1 / EFS
        o[f'autocorr_max_prom_{nm}'] = p1; o[f'autocorr_max_prom_{nm}2'] = p2; o[f'autocorr_max_val_{nm}2'] = v2
    o['autocorr_kurtosis_2'] = kurtosis(_acf(envs['homomorphic']))
    # --- power spectrum
    Pm = np.convolve(P, np.ones(5) / 5, 'same')
    pk, _ = sg.find_peaks(Pm, prominence=Pm.max() * 0.05)
    o['power_maf_num_peaks'] = len(pk)
    top = pk[np.argsort(Pm[pk])[::-1]] if len(pk) else np.array([np.argmax(Pm)])
    o['power_maf_f_peaks'] = f[top[0]]
    o['power_maf_diff_peaks'] = abs(f[top[0]] - f[top[1]]) if len(top) > 1 else 0.0
    rng = np.random.default_rng(0)
    smp = rng.choice(f, 3000, p=Pn)[:, None]
    gm = GaussianMixture(4, random_state=0).fit(smp)
    od = np.argsort(gm.means_[:, 0])
    gp = np.concatenate([gm.means_[od, 0], np.sqrt(gm.covariances_.reshape(-1)[od]), gm.weights_[od]])
    for i in range(12):
        o[f'power_gmm_parameters_{i + 1}'] = gp[i]
    sig_w = gm.weights_ > 0.1
    o['power_gmm_num_peaks'] = int(sig_w.sum())
    mw = gm.means_[np.argsort(gm.weights_)[::-1], 0]
    o['power_gmm_f_peaks'] = mw[0]; o['power_gmm_diff_peaks'] = abs(mw[0] - mw[1])
    sel = (f >= 20) & (f <= 490)
    lf, lp_ = np.log10(f[sel]), np.log10(P[sel] + 1e-20)
    sl, ic = np.polyfit(lf, lp_, 1)
    o['power_regression_slope'] = sl; o['power_regression_intercept'] = ic
    o['power_regression_r2'] = np.corrcoef(lf, lp_)[0, 1] ** 2
    bp = lambda lo, hi: P[(f >= lo) & (f < hi)].sum() + 1e-20
    tot = P.sum() + 1e-20
    o['power_sdr1_heart'] = bp(20, 200) / tot; o['power_sdr2_heart'] = bp(20, 200) / bp(200, 500)
    o['power_sdr1_lung'] = bp(100, 500) / tot; o['power_sdr2_lung'] = bp(100, 500) / bp(20, 100)
    o['power_sdr3_heart'] = bp(40, 100) / tot
    o['power_sdr_cry1'] = bp(250, 500) / tot; o['power_sdr_cry2'] = bp(250, 500) / bp(20, 250)
    for i in range(10):
        o[f'power_b{i + 1}'] = bp(i * 50, (i + 1) * 50) / tot
    cs = np.cumsum(Pn)
    q = lambda p_: f[min(np.searchsorted(cs, p_), len(f) - 1)]
    fm = np.sum(f * Pn)
    o['power_f_mean'] = fm; o['power_std'] = np.sqrt(np.sum(Pn * (f - fm) ** 2)); o['power_f_med'] = q(0.5)
    o['power_bw'] = q(0.975) - q(0.025); o['power_f_p25'] = q(0.25); o['power_f_p75'] = q(0.75)
    o['power_f_IQR'] = q(0.75) - q(0.25)
    for b in (0, 100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800):
        o[f'power_{b:03d}' if b < 1000 else f'power_{b}'] = bp(b, b + 100) / tot   # bands >= 500 Hz are empty at 1 kHz
    o['power_tp'] = np.log10(tot)
    o['power_ratio_low'] = bp(0, 100) / tot; o['power_ratio_high'] = bp(200, 500) / tot
    o['power_ratio_higher'] = bp(400, 500) / tot; o['power_ratio_mid'] = bp(100, 200) / tot
    o['power_freq_centroid'] = fm
    o['power_hn'] = np.exp(np.mean(np.log(P + 1e-20))) / (np.mean(P) + 1e-20)      # spectral flatness
    o['power_max_pow'] = np.log10(P.max() + 1e-20); o['power_max_freq'] = f[np.argmax(P)]
    o['power_ratio_max'] = P.max() / (P.mean() + 1e-20); o['power_var'] = np.var(Pn)
    # --- segmentation-based (Springer timing / Schmidt-like fixed timing)
    s_spr, s_sch = segmentations(x)
    n = min(len(s_spr), len(next(iter(envs.values()))))
    s_spr, s_sch = s_spr[:n], s_sch[:n]
    envs = {k: v[:n] for k, v in envs.items()}
    d1, cyc1 = seg_features(s_spr, envs, '')
    d2, cyc2 = seg_features(s_sch, envs, '_schmidt')
    o.update(d1); o.update(d2)
    # --- envelope sample entropies / variances / cycle correlation / HR stability
    xl = _bp(x, *LUNG)
    hl = np.abs(sg.hilbert(xl))
    lung_env = _norm(sg.resample_poly(_lp(hl, 20, FS), 1, FS // EFS)[:n])
    homo_l = _norm(sg.resample_poly(np.exp(_lp(np.log(hl + 1e-12), 8, FS)), 1, FS // EFS)[:n])
    fS, tS, S = sg.spectrogram(x, FS, nperseg=50, noverlap=30)
    band = lambda lo, hi: _norm(np.interp(np.linspace(0, len(tS) - 1, n), np.arange(len(tS)), S[(fS >= lo) & (fS < hi)].sum(0)))
    mv = np.convolve(x ** 2, np.ones(50) / 50, 'same')
    mvn = _norm(sg.resample_poly(mv, 1, FS // EFS)[:n])
    flux = np.r_[0, np.sqrt((np.diff(S, axis=1) ** 2).sum(0))]
    o['hil_lung_envelope_se'] = _sampen(lung_env)
    o['psd1_envelope_se'] = _sampen(band(20, 60)); o['psd2_envelope_se'] = _sampen(band(60, 120)); o['psd3_envelope_se'] = _sampen(band(120, 250))
    o['homo_lung_envelope_se'] = _sampen(homo_l)
    o['var_frac_envelope_se'] = _sampen(mvn); o['logvar_envelope_se'] = _sampen(_norm(np.log(mvn - mvn.min() + 1e-3)))
    o['spec1_envelope_se'] = _sampen(_norm(np.interp(np.linspace(0, len(tS) - 1, n), np.arange(len(tS)), flux)))
    o['spec2_envelope_se'] = _sampen(band(200, 500))
    for k, e in envs.items():
        o[f'{k}_envelope_var'] = np.var(sg.resample_poly(e, 1, 1))
    o['psd_envelope_original_var'] = o.pop('psd_original_envelope_var')
    for k, e in envs.items():
        o[f'{k}_envelope_se'] = _sampen(e)
    for k, e in envs.items():
        a_, s_ = _cycle_corr(e, _period(e))
        o[f'{k}_envelope_cycle_corr_avg'] = a_; o[f'{k}_envelope_cycle_corr_std'] = s_
    for k, e in envs.items():
        o[f'{k}_envelope_hr_std'] = _hr_std(e)
    for i, st in enumerate((1, 2, 3, 4)):
        a_, b_ = s_spr == st, s_sch == st
        o[f'seg_agreement{i + 1}'] = (a_ & b_).sum() / max((a_ | b_).sum(), 1)
    # --- PSD shape (two Welch resolutions)
    for tag, nps in (('', 256), ('_2000', 1024)):
        f2, P2 = sg.welch(x, FS, nperseg=nps)
        b2 = lambda lo, hi: P2[(f2 >= lo) & (f2 < hi)].sum() + 1e-20
        o[f'psd_rho1{tag}'] = b2(20, 150) / b2(150, 500); o[f'psd_rho2{tag}'] = b2(20, 80) / b2(80, 200)
        o[f'psd_rho3{tag}'] = b2(150, 300) / b2(300, 500)
        o[f'psd_noise{tag}'] = np.median(P2[f2 > 300]) / (np.median(P2[(f2 >= 20) & (f2 <= 150)]) + 1e-20)
    # --- MFCC
    mf = librosa.feature.mfcc(y=x.astype(np.float32), sr=FS, n_mfcc=14, n_fft=256, hop_length=64, n_mels=32, fmin=20, fmax=500)
    g = librosa.feature.mfcc(S=librosa.power_to_db(librosa.feature.melspectrogram(y=x.astype(np.float32), sr=FS, n_fft=len(x),
                                                                                   hop_length=len(x) + 1, n_mels=32, fmin=20, fmax=500)), n_mfcc=14)[:, 0]
    for i in range(14):
        r = mf[i]
        o[f'mfcc_coeffs_{i + 1}'] = g[i]; o[f'mfcc_min_coeffs_{i + 1}'] = r.min(); o[f'mfcc_max_coeffs_{i + 1}'] = r.max()
        o[f'mfcc_mean_coeffs_{i + 1}'] = r.mean(); o[f'mfcc_median_coeffs_{i + 1}'] = np.median(r)
        o[f'mfcc_mode_coeffs_{i + 1}'] = _mode(r); o[f'mfcc_var_coeffs_{i + 1}'] = r.var(); o[f'mfcc_skew_coeffs_{i + 1}'] = skew(r)
    mm = mf.mean(1)
    o['mfcc_mean_min'] = mm.min(); o['mfcc_mean_max'] = mm.max(); o['mfcc_mean_skew'] = skew(mm)
    # --- peak-window plausibility (Iga = homomorphic peaks, Liang = Shannon energy, Springer / Schmidt = S1+S2 centres)
    centres = lambda st: np.flatnonzero(np.diff((np.isin(st, (1, 3))).astype(int)) == 1)
    for nm, env in (('iga', envs['homomorphic']), ('iga_S1S2', envs['homomorphic']), ('liang_S1S2', envs['shannon'])):
        for tag, ws in (('original', 2.0), ('modified', 1.5), ('modified2', 3.0)):
            o[f'wav_{nm}_per_peak_window_{tag}'] = _peak_windows(env, ws)[0]
    for nm, st in (('springer', s_spr), ('schmidt', s_sch)):
        c_ = centres(st)
        for tag, ws in (('original', 2.0), ('modified', 1.5), ('modified2', 3.0)):
            L = int(ws * EFS)
            ok = [2 * ws * 0.5 <= np.sum((c_ >= s) & (c_ < s + L)) <= 2 * ws * 3.5 for s in range(0, n - L + 1, L // 2)]
            o[f'wav_{nm}_S1S2_per_peak_window_{tag}'] = float(np.mean(ok)) if ok else 0.0
    for i, (env) in enumerate((homo_l, homo_l, homo_l)):
        o[f'wav_iga_per_peak_window_lung_homo{i + 1}'] = _peak_windows(env, (2.0, 1.5, 3.0)[i])[0]
    for i, env in enumerate((band(20, 60), band(60, 120), band(120, 250))):
        o[f'wav_iga_per_peak_window_lung_psd{i + 1}'] = _peak_windows(env, 2.0)[0]
    h = envs['homomorphic']
    tr_ = np.sum(np.abs(np.diff((h > 0).astype(int))))
    o['wav_iga_transitCounterNorm'] = tr_ / (len(h) / EFS)
    o['wav_iga_reducedTransitCounter'] = np.sum(np.abs(np.diff((h > 1).astype(int)))) / (len(h) / EFS)
    _, pkx = _peak_windows(h, 2.0)
    ib = np.diff(pkx) / EFS
    o['wav_iga_RMSSDall'] = np.sqrt(np.mean(np.diff(ib) ** 2)) / (np.mean(ib) + 1e-9) if len(ib) > 2 else 1.0
    return o


_NAMES = None


def names():
    global _NAMES
    if _NAMES is None:
        _NAMES = list(extract_dict(np.random.default_rng(0).standard_normal(10000)).keys())
    return _NAMES


def extract(w2k):
    nm = names()
    try:
        d = extract_dict(w2k)
        return np.array([d.get(k, np.nan) for k in nm], np.float64)
    except Exception:
        return np.full(len(nm), np.nan)


def _chunk(ws):
    warnings.filterwarnings('ignore')
    return np.array([extract(w) for w in ws])


def compute(W, nproc=10, chunk=32):
    from concurrent.futures import ProcessPoolExecutor
    parts = [np.asarray(W[i:i + chunk], np.float32) for i in range(0, len(W), chunk)]
    with ProcessPoolExecutor(nproc) as ex:
        return np.concatenate(list(ex.map(_chunk, parts))).astype(np.float32)
