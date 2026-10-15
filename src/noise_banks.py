"""Noise banks at 2 kHz (20 Hz high-pass, unit RMS), following PCGKolect_model/seg/prep_noise.py.

  Bank 1 (training augmentation): ESC-50, Mini LibriSpeech dev-clean-2, PhysioNet '(N ... N)' stretches
  Bank 2 (E2 test noise only):    LibriSpeech test-clean, DEMAND (16 kHz, channel 1, 30 s clips)

  python noise_banks.py bank1|bank2
Bank 1 clips get a fold (0-4) so that augmentation never mixes noise across PhysioNet folds:
PhysioNet stretches inherit their record's fold, other clips are assigned at random (seed 42).
"""
import os, sys, glob
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from common import RAW, PROC, MAN, FS, SEED, load_audio, highpass

MIN_LEN = FS


def _load(args):
    path, max_s = args
    return highpass(load_audio(path, max_s=max_s))


def save(name, clips, meta):
    keep = [i for i, x in enumerate(clips) if len(x) >= MIN_LEN and np.sqrt(np.mean(x ** 2)) > 1e-5]
    clips = [clips[i] / (np.sqrt(np.mean(clips[i] ** 2)) + 1e-9) for i in keep]
    meta = meta.iloc[keep].reset_index(drop=True)
    lens = np.array([len(x) for x in clips])
    meta['clip_id'] = [f'{name}_{i:05d}' for i in range(len(meta))]
    meta['n_samples'] = lens
    np.savez(os.path.join(PROC, f'noise_{name}.npz'), audio=np.concatenate(clips).astype(np.float16),
             offset=np.concatenate([[0], np.cumsum(lens)]))
    meta.to_csv(os.path.join(PROC, f'noise_{name}.csv'), index=False)
    for s, g in meta.groupby('source'):
        print(f'{name} {s:10s} clips={len(g):5d} minutes={g.n_samples.sum() / FS / 60:6.1f}')
    # combined manifest
    parts = [pd.read_csv(os.path.join(PROC, f'noise_{b}.csv')) for b in ('bank1', 'bank2')
             if os.path.exists(os.path.join(PROC, f'noise_{b}.csv'))]
    pd.concat(parts, ignore_index=True).to_csv(os.path.join(MAN, 'noise_banks.csv'), index=False)


def bank1():
    rng = np.random.default_rng(SEED)
    esc = sorted(glob.glob(os.path.join(RAW, 'esc50', '*', 'audio', '*.wav')))
    spc = sorted(glob.glob(os.path.join(RAW, 'librispeech', 'LibriSpeech', 'dev-clean-2', '**', '*.flac'), recursive=True))
    with ProcessPoolExecutor(8) as ex:
        clips = list(ex.map(_load, [(p, None) for p in esc] + [(p, 12) for p in spc], chunksize=32))
    meta = pd.DataFrame(dict(bank='bank1', source=['esc50'] * len(esc) + ['mini_librispeech'] * len(spc),
                             file=[os.path.relpath(p, RAW) for p in esc + spc]))
    meta['fold'] = rng.integers(0, 5, len(meta))
    # PhysioNet noise stretches (>= 1 s), from the raw (unfiltered) cached audio
    man = pd.read_csv(os.path.join(MAN, 'physionet2016.csv')).set_index('record_id')
    nz = pd.read_csv(os.path.join(PROC, 'pn_noise.csv'))
    aud = np.load(os.path.join(PROC, 'pn_audio.npz'))
    pn_rows = []
    for r in nz.itertuples():
        if r.end - r.start >= MIN_LEN:
            clips.append(highpass(aud[r.record_id][r.start:r.end]))
            pn_rows.append(dict(bank='bank1', source='physionet_noise', file=f'{r.record_id}:{r.start}-{r.end}',
                                fold=int(man.loc[r.record_id, 'fold'])))
    meta = pd.concat([meta, pd.DataFrame(pn_rows)], ignore_index=True)
    save('bank1', clips, meta)


def bank2():
    spc = sorted(glob.glob(os.path.join(RAW, 'librispeech', 'LibriSpeech', 'test-clean', '**', '*.flac'), recursive=True))
    with ProcessPoolExecutor(8) as ex:
        clips = list(ex.map(_load, [(p, 12) for p in spc], chunksize=32))
    files = [os.path.relpath(p, RAW) for p in spc]
    src = ['librispeech_test_clean'] * len(spc)
    for z in sorted(glob.glob(os.path.join(RAW, 'demand', '*_16k.zip'))):
        if not os.path.exists(z + '.done'):
            print('skip incomplete', z); continue
        d = z[:-4]
        if not os.path.isdir(d):
            os.system(f'cd {os.path.dirname(z)} && unzip -q -o {os.path.basename(z)} -d {os.path.basename(d)}')
        ch = sorted(glob.glob(os.path.join(d, '**', 'ch01.wav'), recursive=True))
        for p in ch:
            x = highpass(load_audio(p))
            for k, s in enumerate(range(0, len(x) - 30 * FS + 1, 30 * FS)):
                clips.append(x[s:s + 30 * FS]); files.append(f'{os.path.relpath(p, RAW)}#{k}'); src.append('demand')
    save('bank2', clips, pd.DataFrame(dict(bank='bank2', source=src, file=files, fold=-1)))


def load_bank(name):
    z = np.load(os.path.join(PROC, f'noise_{name}.npz'))
    return z['audio'], z['offset'], pd.read_csv(os.path.join(PROC, f'noise_{name}.csv'))


def noise_segment(audio, offset, idx, n, rng):
    """n samples from clip idx at a random offset (clip tiled if shorter than n)."""
    x = audio[offset[idx]:offset[idx + 1]].astype(np.float32)
    if len(x) < n:
        x = np.tile(x, int(np.ceil(n / len(x))) + 1)
    s = rng.integers(0, len(x) - n + 1)
    return x[s:s + n]


if __name__ == '__main__':
    globals()[sys.argv[1]]()
