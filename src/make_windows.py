"""Phase A.4: PhysioNet training windows (the common corpus).

5 s windows, 2.5 s hop.
  unusable: record SQI 0, or >= 50 % of the window inside a '(N ... N)' stretch
  usable:   record SQI 1 and no overlap with any noise stretch
  other windows are dropped.
Augmentation: 30 % of usable windows are copied, a Bank 1 clip (same fold) is added at SNR ~ U[-5, 5] dB,
and the copy is labelled unusable.
Audio is stored peak-normalised (all models are gain-invariant), float16.
"""
import os
import numpy as np, pandas as pd
from common import PROC, MAN, WIN, HOP, SEED, add_noise
from noise_banks import load_bank, noise_segment


def norm(x):
    return (x / (np.abs(x).max() + 1e-9)).astype(np.float16)


def main():
    rng = np.random.default_rng(SEED)
    man = pd.read_csv(os.path.join(MAN, 'physionet2016.csv'))
    nz = pd.read_csv(os.path.join(PROC, 'pn_noise.csv'))
    aud = np.load(os.path.join(PROC, 'pn_audio.npz'))
    b_aud, b_off, b_meta = load_bank('bank1')
    by_fold = {k: np.flatnonzero(b_meta.fold.values == k) for k in range(5)}
    X, rows = [], []
    for r in man.itertuples():
        x = aud[r.record_id]
        mask = np.zeros(len(x), bool)
        for s, e in nz.loc[nz.record_id == r.record_id, ['start', 'end']].values:
            mask[s:e] = True
        for st in range(0, len(x) - WIN + 1, HOP):
            frac = mask[st:st + WIN].mean()
            if r.sqi == 0 or frac >= 0.5:
                y = 1
            elif r.sqi == 1 and frac == 0:
                y = 0
            else:
                continue
            w = x[st:st + WIN]
            base = dict(record_id=r.record_id, patient_id=r.patient_id, fold=r.fold, subset=r.subset,
                        abnormal=r.abnormal, sqi=r.sqi, start=st, unusable=y, aug=0, noise_clip='', snr_db=np.nan,
                        noise_frac=frac)
            X.append(norm(w)); rows.append(base)
            if y == 0 and rng.random() < 0.30:
                idx = rng.choice(by_fold[r.fold])
                snr = rng.uniform(-5, 5)
                wn = add_noise(w, noise_segment(b_aud, b_off, idx, WIN, rng), snr)
                X.append(norm(wn))
                rows.append(dict(base, unusable=1, aug=1, noise_clip=b_meta.clip_id[idx], snr_db=snr))
    meta = pd.DataFrame(rows)
    meta.insert(0, 'window_id', [f'pnw{i:06d}' for i in range(len(meta))])
    np.save(os.path.join(PROC, 'pn_windows_audio.npy'), np.stack(X))
    meta.to_csv(os.path.join(PROC, 'pn_windows.csv'), index=False)
    print(len(meta), 'windows;', meta.groupby(['unusable', 'aug']).size().to_dict())


if __name__ == '__main__':
    main()
