"""Deep quality models (study baselines) on the PhysioNet common corpus.

  python deep.py b4_cnn | b5_crnn | e6_cnn_balanced

Input: 5 s @ 2 kHz, 20 Hz high-pass, per-window z-score -> log-mel (64 bands 20-1000 Hz, 25 ms window,
10 ms hop), computed on the GPU from the cached audio. One train/validation split: fold 0 (20 %, by
patient where IDs exist) is validation. Adam 1e-3, batch 64, class-weighted BCE, <= 50 epochs, early
stopping on validation loss (patience 5). Per-epoch log: logs/train/<name>.csv
"""
import os, sys, time, csv
import numpy as np, pandas as pd, torch, torch.nn as nn
import librosa
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from common import PROC, LOGT, MODELS, FS, SEED, seed_all, highpass

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


class LogMel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('win', torch.hann_window(50))
        self.register_buffer('mel', torch.tensor(librosa.filters.mel(sr=FS, n_fft=512, n_mels=64, fmin=20, fmax=1000), dtype=torch.float32))

    def forward(self, x):                                   # x [B, N] (already high-passed)
        x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-8)
        S = torch.stft(x, 512, hop_length=20, win_length=50, window=self.win, center=True, return_complex=True)
        return torch.log(self.mel @ (S.real ** 2 + S.imag ** 2) + 1e-6).unsqueeze(1)   # [B, 1, 64, T]


def block(cin, cout, pool):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.MaxPool2d(pool))


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fe = LogMel()
        self.body = nn.Sequential(block(1, 32, 2), block(32, 64, 2), block(64, 128, 2), block(128, 128, 2))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, x):
        return self.head(self.body(self.fe(x)).mean((2, 3))).squeeze(1)


class CRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fe = LogMel()
        self.body = nn.Sequential(block(1, 32, 2), block(32, 64, 2), block(64, 128, (2, 1)), block(128, 128, (2, 1)))
        self.rnn = nn.LSTM(128 * 4, 64, num_layers=2, batch_first=True, bidirectional=True)
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        h = self.body(self.fe(x))                             # [B, 128, 4, T]
        h = h.flatten(1, 2).transpose(1, 2)                   # [B, T, 512]
        return self.head(self.rnn(h)[0].mean(1)).squeeze(1)


ARCH = {'b4_cnn': CNN, 'b5_crnn': CRNN, 'e6_cnn_balanced': CNN}


def prep_audio(W):
    """float16 windows -> high-passed float32 (numpy)."""
    return np.concatenate([highpass(np.asarray(W[i:i + 4096], np.float32)) for i in range(0, len(W), 4096)])


@torch.no_grad()
def predict_logits(model, X, bs=256):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs]
        xb = xb if torch.is_tensor(xb) else torch.from_numpy(np.asarray(xb, np.float32))
        out.append(model(xb.to(DEV, torch.float32)).float().cpu())
    return torch.cat(out).numpy()


def train_model(name, X, y, tr, va, arch, sample_w=None, log_path=None, max_epochs=50, patience=5):
    """Generic trainer (also used for the E5 murmur classifier). Returns the best model (min val loss)."""
    seed_all(SEED)
    Xg = torch.from_numpy(X).to(DEV, torch.float16)
    yg = torch.from_numpy(y.astype(np.float32)).to(DEV)
    pw = torch.tensor((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1), device=DEV)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    model = arch().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(SEED)
    best, best_state, bad = np.inf, None, 0
    f = open(log_path, 'w', newline='') if log_path else None
    wr = csv.writer(f) if f else None
    if wr:
        wr.writerow(['epoch', 'train_loss', 'val_loss', 'val_auroc', 'val_balanced_acc', 'lr']); f.flush()
    tr_t = torch.from_numpy(tr)
    for ep in range(1, max_epochs + 1):
        model.train()
        if sample_w is not None:
            order = tr_t[torch.multinomial(torch.from_numpy(sample_w[tr]).double(), len(tr), replacement=True, generator=g)]
        else:
            order = tr_t[torch.randperm(len(tr), generator=g)]
        tl, nb = 0.0, 0
        for i in range(0, len(order), 64):
            idx = order[i:i + 64].to(DEV)
            if len(idx) < 2:
                continue
            loss = lossf(model(Xg[idx].float()), yg[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        lv = predict_logits(model, Xg[torch.from_numpy(va).to(DEV)])
        vl = float(lossf(torch.from_numpy(lv).to(DEV), yg[torch.from_numpy(va).to(DEV)]).item())
        pv = 1 / (1 + np.exp(-lv))
        auc = roc_auc_score(y[va], pv) if len(np.unique(y[va])) > 1 else np.nan
        bacc = balanced_accuracy_score(y[va], pv >= 0.5)
        if wr:
            wr.writerow([ep, round(tl / max(nb, 1), 5), round(vl, 5), round(auc, 5), round(bacc, 5), opt.param_groups[0]['lr']]); f.flush()
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if f:
        f.close()
    model.load_state_dict(best_state)
    del Xg
    torch.cuda.empty_cache()
    return model


def load_model(name):
    m = ARCH[name]().to(DEV)
    m.load_state_dict(torch.load(os.path.join(MODELS, f'{name}.pt'), map_location=DEV))
    m.eval()
    return m


def score(name, W):
    """Unusable probability for raw float16 windows."""
    m = load_model(name)
    return 1 / (1 + np.exp(-predict_logits(m, prep_audio(W))))


def main(name):
    t0 = time.time()
    meta = pd.read_csv(os.path.join(PROC, 'pn_windows.csv'))
    X = prep_audio(np.load(os.path.join(PROC, 'pn_windows_audio.npy'), mmap_mode='r'))
    y = meta.unusable.values
    va = np.flatnonzero(meta.fold.values == 0); tr = np.flatnonzero(meta.fold.values != 0)
    sw = None
    if name == 'e6_cnn_balanced':
        # usable windows: normal and abnormal recordings get equal total sampling weight (50/50)
        sw = np.ones(len(meta))
        u = (y == 0)
        trm = np.zeros(len(meta), bool); trm[tr] = True
        n_un = (u & trm & (meta.abnormal.values == 0)).sum(); n_ua = (u & trm & (meta.abnormal.values == 1)).sum()
        n_u = n_un + n_ua
        sw[u & (meta.abnormal.values == 0)] = 0.5 * n_u / n_un
        sw[u & (meta.abnormal.values == 1)] = 0.5 * n_u / n_ua
    model = train_model(name, X, y, tr, va, ARCH[name], sample_w=sw, log_path=os.path.join(LOGT, f'{name}.csv'))
    torch.save(model.state_dict(), os.path.join(MODELS, f'{name}.pt'))
    pv = 1 / (1 + np.exp(-predict_logits(model, X[va])))
    pd.DataFrame(dict(window_id=meta.window_id.values[va], fold=0, y=y[va], score=pv)).to_csv(
        os.path.join(PROC, f'val_preds_{name}.csv'), index=False)
    print(name, 'val AUROC', round(roc_auc_score(y[va], pv), 4), 'time', round(time.time() - t0), 's')


if __name__ == '__main__':
    main(sys.argv[1])
