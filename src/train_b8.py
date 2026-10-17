"""B8 "PCGKolect-v3" (proposed filter; pre-registered in ANALYSIS_PLAN.md).

  python train_b8.py b8        # with rhythm/tempo augmentation (the proposed model)
  python train_b8.py b8noaug   # ablation: identical, without the augmentation

PCGKolect 1-D U-Net, trained on PhysioNet 2016 folds 1-4 only (fold 0 held out, same split as B4/B5), frame labels
S1/systole/S2/diastole/noise from the hand-corrected annotations + the repo's synthetic corruption recipe. No
recording-level SQI labels are used. Quality score of a beat/window = mean noise posterior over its frames.
Outputs: data/models/<name>.pt, logs/train/<name>.csv (loss every 500 steps + fold-0 validation),
data/processed/val_preds_<name>.csv (fold-0 window scores).
"""
import os, sys, time, csv
import numpy as np, pandas as pd, torch
import torch.nn.functional as Fn
from common import PROC, MAN, MODELS, LOGT, SEED, RAW, WIN

PN_ONLY = os.path.join(RAW, 'repos', 'PCGKolect_pnonly')
sys.path.insert(0, os.path.join(PN_ONLY, 'seg')); sys.path.insert(0, PN_ONLY)
import train_seg as TS                                       # noqa: E402  (repo training utilities, PhysioNet-only data)
from model import Frontend, SegNet, SAMPLES, FRAMES, HOP, NFFT, IGNORE, frame_labels, FS   # noqa: E402

DEV = 'cuda'
STEPS, BATCH = 12000, 64
DIA = 3                                                       # diastole label in the corpus


AUG = {'b8': (0.35, 1.4, 0.7, 1.3, 0.15), 'b8v4a': (0.60, 1.25, 0.85, 1.15, 0.10), 'b8v4b': (0.50, 1.30, 0.80, 1.20, 0.15)}
CFG = AUG['b8']


def rr_aug(x, y, rng, cfg=None):
    """Label-aware tempo / rhythm augmentation: rescale every diastole (no pitch change)."""
    lo, hi, jlo, jhi, pprem = cfg or CFG
    f_rec = np.exp(rng.uniform(np.log(lo), np.log(hi)))
    e = np.flatnonzero(np.diff(y.astype(np.int16))) + 1
    starts, ends = np.r_[0, e], np.r_[e, len(y)]
    xs, ys = [], []
    fade = np.linspace(1, 0, 20, dtype=np.float32)
    for s, t in zip(starts, ends):
        seg, lab = x[s:t], y[s:t]
        if y[s] == DIA and t - s >= int(0.1 * FS):
            f = f_rec * rng.uniform(jlo, jhi)
            if rng.random() < pprem:
                f *= rng.uniform(0.60, 0.75)                  # premature-like: diastole shortened 25-40 % more
            n = max(int(0.05 * FS), int((t - s) * f))
            if n <= t - s:
                seg = seg[:n].copy()
            else:
                ext = np.resize(np.concatenate([seg, seg[::-1]]), n)
                seg = ext.astype(np.float32)
            if len(seg) > 40:
                seg[-20:] *= fade
            lab = np.full(len(seg), DIA, y.dtype)
        xs.append(seg); ys.append(lab)
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys)


def make_batch(rng, d, recs, w, aug):
    X = np.empty((BATCH, SAMPLES), np.float32)
    Y = np.empty((BATCH, FRAMES), np.int64)
    for j in range(BATCH):
        while True:
            r = rng.choice(recs, p=w)
            L = d.len[r]
            Lc = min(L, 3 * SAMPLES)
            s0 = d.off[r] + rng.integers(0, L - Lc + 1)
            x = d.audio[s0:s0 + Lc].astype(np.float32)
            ys = d.lab[s0:s0 + Lc]
            if aug:
                x, ys = rr_aug(x, ys, rng)
            if len(x) < SAMPLES:
                continue
            for _ in range(8):
                s = rng.integers(0, len(x) - SAMPLES + 1)
                yf = frame_labels(ys[s:s + SAMPLES], 0, FRAMES)
                if (yf != IGNORE).mean() > 0.3:
                    break
            xx = x[s:s + SAMPLES] * rng.choice([-1, 1])
            X[j], Y[j] = TS.corrupt(rng, d, xx, ys[s:s + SAMPLES], yf)
            break
    return torch.from_numpy(X), torch.from_numpy(Y)


@torch.no_grad()
def window_noise_scores(fe, net, W, bs=256):
    """Mean noise posterior over the frames inside each 5 s window (reflection-padded to the model input)."""
    pad = (SAMPLES - WIN) // 2
    lo, hi = int(np.ceil((pad - NFFT // 2) / HOP)), int((pad + WIN - NFFT // 2) / HOP)
    out = []
    for i in range(0, len(W), bs):
        x = np.asarray(W[i:i + bs], np.float32)
        x = x / (np.abs(x).max(1, keepdims=True) + 1e-9)
        x = np.pad(x, ((0, 0), (pad, SAMPLES - WIN - pad)), mode='reflect')
        x = TS.highpass(x) if hasattr(TS, 'highpass') else x
        P = torch.softmax(net(fe(torch.from_numpy(np.asarray(x, np.float32)).to(DEV)))[:, :, 0, :], 1)
        out.append(P[:, 4, lo:hi + 1].mean(1).cpu().numpy())
    return np.concatenate(out)


def main(name):
    from sklearn.metrics import roc_auc_score
    from common import highpass
    TS.highpass = highpass
    global CFG
    aug = name in AUG
    CFG = AUG.get(name, AUG['b8'])
    rng = np.random.default_rng(SEED); torch.manual_seed(SEED)
    d = TS.Data()
    man = pd.read_csv(os.path.join(MAN, 'physionet2016.csv'))
    fold = dict(zip('pn' + man.record_id, man.fold))
    rec_fold = np.array([fold[r] for r in d.rid])
    # noise clips: PhysioNet stretches follow their record's fold; fold-0 stretches are never used in training
    d.nfold = np.array([fold.get(r, -1) if r else -1 for r in d.nrid])
    d.exclude_fold = 0
    recs = np.array([r for r in np.flatnonzero(rec_fold != 0) if d.len[r] >= SAMPLES])
    w = (d.len[recs] - SAMPLES + 1).astype(np.float64); w /= w.sum()
    fe, net = Frontend().to(DEV), SegNet().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=STEPS, pct_start=0.1)
    meta = pd.read_csv(os.path.join(PROC, 'pn_windows.csv'))
    va = np.flatnonzero(meta.fold.values == 0)
    Wv = np.load(os.path.join(PROC, 'pn_windows_audio.npy'), mmap_mode='r')[va]
    yv = meta.unusable.values[va]
    log = open(os.path.join(LOGT, f'{name}.csv'), 'w', newline=''); wr = csv.writer(log)
    wr.writerow(['step', 'train_loss', 'val_auroc', 'lr']); log.flush()
    run, t0 = 0.0, time.time()

    class Batches(torch.utils.data.IterableDataset):          # parallel batch building (seeded per worker)
        def __iter__(self):
            wi = torch.utils.data.get_worker_info()
            r = np.random.default_rng([SEED, wi.id if wi else 0])
            while True:
                yield make_batch(r, d, recs, w, aug)
    loader = iter(torch.utils.data.DataLoader(Batches(), batch_size=None, num_workers=7, prefetch_factor=4,
                                              persistent_workers=True))
    for st in range(STEPS):
        net.train()
        X, Y = next(loader)
        with torch.no_grad():
            f = TS.spec_augment(fe(X.to(DEV)), rng)
        loss = Fn.cross_entropy(net(f)[:, :, 0, :], Y.to(DEV), ignore_index=IGNORE, label_smoothing=0.05)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
        run = 0.98 * run + 0.02 * loss.item() if st else loss.item()
        if (st + 1) % 1000 == 0 or st + 1 == STEPS:
            net.eval()
            auc = roc_auc_score(yv, window_noise_scores(fe, net, Wv))
            wr.writerow([st + 1, round(run, 4), round(auc, 4), sch.get_last_lr()[0]]); log.flush()
            print(f'{name} step {st + 1} loss {run:.3f} fold0 AUROC {auc:.4f} {time.time() - t0:.0f}s', flush=True)
    net.eval()
    torch.save(net.state_dict(), os.path.join(MODELS, f'{name}.pt'))
    s = window_noise_scores(fe, net, Wv)
    pd.DataFrame(dict(window_id=meta.window_id.values[va], fold=0, y=yv, score=s)).to_csv(
        os.path.join(PROC, f'val_preds_{name}.csv'), index=False)
    print(name, 'final fold-0 AUROC', round(roc_auc_score(yv, s), 4))
    # development rhythm check (PhysioNet fold 0 only): clean normal SQI-1 crops vs the same crops after strong
    # premature/irregular rescaling; a rhythm-blind noise score gives AUROC 0.5
    r2 = np.random.default_rng(SEED + 7)
    cand = [r for r in np.flatnonzero(rec_fold == 0) if d.sqi[r] == 1 and d.len[r] >= 3 * SAMPLES][:300]
    A, B = [], []
    for r in cand:
        s0 = d.off[r]; x = d.audio[s0:s0 + 3 * SAMPLES].astype(np.float32); y = d.lab[s0:s0 + 3 * SAMPLES]
        if (y == IGNORE).mean() > 0.2 or (y == 4).any():
            continue
        x2, _ = rr_aug(x, y, r2, (0.6, 1.0, 0.7, 1.3, 0.4))
        A.append(x[:WIN]); B.append(x2[:WIN])
    sa, sb = window_noise_scores(fe, net, np.stack(A)), window_noise_scores(fe, net, np.stack(B))
    rh = roc_auc_score(np.r_[np.zeros(len(sa)), np.ones(len(sb))], np.r_[sa, sb])
    with open(os.path.join(LOGT, f'{name}_dev.csv'), 'w') as fh:
        fh.write(f'model,fold0_auroc,rhythm_shift_auroc,n_crops\n{name},{roc_auc_score(yv, s):.4f},{rh:.4f},{len(sa)}\n')
    print(name, 'rhythm-shift AUROC', round(rh, 4))


if __name__ == '__main__':
    main(sys.argv[1])
