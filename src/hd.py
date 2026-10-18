"""Debiased Das + CNN hybrids (post hoc, pre-registered in ANALYSIS_PLAN.md, branch hybrid-debias). PhysioNet 2016 only.

  H1 hd_avg   mean of the B2 and B4 percentiles (reference: out-of-fold scores of clean usable normal windows, folds 1-4)
  H2 hd_and   min of the same percentiles
  H3 hd_lr    class-balanced logistic regression on the B4 penultimate embedding (out-of-fold) + 11 Das features
  H4 hd_h4_l{01,03,10}  B4 trunk + Das features -> head, trained end to end with rhythm augmentation and a gradient-reversal
              adversary (normal/abnormal, + subset) with weight lambda 0.1 / 0.3 / 1.0
Candidate seed s uses B4 seed s and B2 seed s (H4: network seed s).

  python hd.py aug        rhythm-augmented training windows (+ Das features)       -> data/processed/pn_hd_aug*.{csv,npy,npz}
  python hd.py selset     fold-0 selection beats and fold-0 synthetic rhythm set   -> data/processed/pn_hd_sel*.{csv,npy,npz}
  python hd.py train <lam> <seed>   one H4 network
  python hd.py fit        H1-H3 on PhysioNet (all seeds); val_preds_<tag>.csv
  python hd.py select     selection metrics (i)-(iii) for all candidates x seeds -> results/hd_selection*.csv
  python hd.py test       H1-H3 test-set score arrays (H4 test scores come from score.py)
"""
import os, sys, time, csv
import numpy as np, pandas as pd, torch, torch.nn as nn, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from common import PROC, MODELS, RES, LOGT, MAN, SEED, FS, WIN, stable_seed, window_at, bandpass, seed_all
import deep

SEEDS = [SEED, SEED + 1, SEED + 2]
LAMS = {'l01': 0.1, 'l03': 0.3, 'l10': 1.0}
CANDS = ['hd_avg', 'hd_and', 'hd_lr'] + [f'hd_h4_{k}' for k in LAMS]
SUBSETS = list('abcdef')
BEAT_MIN, BEAT_MAX = int(0.3 * FS), int(2.0 * FS)
DEV = deep.DEV


def tag(m, s):
    return m if s == SEED else f'{m}_s{s}'


def pct(ref, x):
    r = np.sort(np.asarray(ref, np.float64))
    return (np.searchsorted(r, x, 'left') + np.searchsorted(r, x, 'right')) / (2.0 * len(r))


# ---------------------------------------------------------------- beats and rhythm sequences (same algorithm as prep_test.synth)
def beats_of(segs):
    from prep_test import beats_from_segments
    return [b for b in beats_from_segments([tuple(v) for v in segs]) if BEAT_MIN <= b['end'] - b['start'] <= BEAT_MAX]


def sequence(x, bl, kind, rng, length, prem_every=None):
    """Concatenate beats bl (cycling) of recording x to >= length samples, changing timing only by trimming / extending the
    diastole. kind: regular | premature | irregular. prem_every: force one premature beat in every k beats (else p = 0.2)."""
    out, marks, t, i = [], [], 0, 0
    off = int(rng.integers(1, prem_every)) if prem_every else 0
    while t < length:
        b = bl[i % len(bl)]; i += 1
        L = b['end'] - b['start']
        dia = b['end'] - b['dia_start']
        if kind == 'premature':
            prem = len(out) > 0 and ((i - 1 - off) % prem_every == 0 if prem_every else rng.random() < 0.2)
            target = L
        elif kind == 'irregular':
            prem, target = False, int(L * rng.uniform(0.7, 1.3))
        else:
            prem, target = False, L
        seg = x[b['start']:b['end']]
        target = max(target, (b['dia_start'] - b['start']) + int(0.05 * FS))
        if target <= L:
            seg = seg[:target]
        else:
            tail = x[b['dia_start']:b['end']]
            seg = np.concatenate([seg, np.resize(np.concatenate([tail, tail[::-1]]), target - L)])
        if prem:
            prev_len, prev_dia = marks[-1]['len'], marks[-1]['dia_len']
            shift = min(int(rng.uniform(0.25, 0.40) * prev_len), prev_dia - int(0.05 * FS))
            if shift > 0:
                out[-1] = out[-1][:prev_len - shift]
                marks[-1]['len'] -= shift; marks[-1]['dia_len'] -= shift; t -= shift
                tail = x[b['dia_start']:b['end']]
                seg = np.concatenate([seg, np.resize(np.concatenate([tail, tail[::-1]]), shift)])
        marks.append(dict(start=t, len=len(seg), dia_len=dia + (len(seg) - L), prem=prem))
        out.append(seg); t += len(seg)
    return np.concatenate(out), marks


def pn_tables():
    man = pd.read_csv(os.path.join(MAN, 'physionet2016.csv'))
    segs = pd.read_csv(os.path.join(PROC, 'pn_segments.csv.gz'))
    noise = pd.read_csv(os.path.join(PROC, 'pn_noise.csv'))
    audio = np.load(os.path.join(PROC, 'pn_audio.npz'))
    return man, {r: g[['start', 'end', 'state']].values for r, g in segs.groupby('record_id')}, \
        {r: g[['start', 'end']].values for r, g in noise.groupby('record_id')}, audio


# ---------------------------------------------------------------- (a) rhythm-augmented training windows
def build_aug():
    from features import compute
    meta = pd.read_csv(os.path.join(PROC, 'pn_windows.csv'))
    man, segs, noise, audio = pn_tables()
    rng = np.random.default_rng(SEED)
    use = meta[(meta.fold != 0) & (meta.unusable == 0) & (meta.aug == 0)]
    n_target = int(round(0.2 * len(use)))
    elig = []
    for r in use.itertuples():
        bl = [b for b in beats_of(segs.get(r.record_id, np.empty((0, 3)))) if b['start'] >= r.start and b['end'] <= r.start + WIN]
        if len(bl) >= 4:
            elig.append((r, bl))
    pick = rng.permutation(len(elig))[:n_target]
    rows, W = [], []
    for j, k in enumerate(sorted(pick)):
        r, bl = elig[k]
        kind = 'premature' if j % 2 == 0 else 'irregular'
        x = audio[r.record_id]
        y, _ = sequence(x, bl, kind, rng, WIN, prem_every=5 if kind == 'premature' else None)
        w = y[:WIN]
        W.append((w / (np.abs(w).max() + 1e-9)).astype(np.float16))
        rows.append(dict(aug_id=f'hdaug{j:05d}', window_id=r.window_id, record_id=r.record_id, patient_id=r.patient_id, fold=r.fold,
                         subset=r.subset, abnormal=r.abnormal, kind=kind, unusable=0))
    W = np.stack(W)
    np.save(os.path.join(PROC, 'pn_hd_aug_audio.npy'), W)
    pd.DataFrame(rows).to_csv(os.path.join(PROC, 'pn_hd_aug.csv'), index=False)
    b1, b2 = compute(W)
    np.savez(os.path.join(PROC, 'pn_hd_aug_feats.npz'), b2=b2)
    print('aug windows', len(rows), 'of target', n_target, 'eligible', len(elig))


# ---------------------------------------------------------------- selection sets (fold 0)
def build_selset():
    from features import compute
    from prep_test import beat_snr
    man, segs, noise, audio = pn_tables()
    f0 = man[(man.fold == 0) & (man.sqi == 1)]
    rows, W = [], []
    for r in f0.itertuples():
        x = audio[r.record_id]
        nz = noise.get(r.record_id, np.empty((0, 2)))
        bl = [b for b in beats_of(segs.get(r.record_id, np.empty((0, 3))))
              if not any((b['start'] < e) and (b['end'] > s) for s, e in nz)]
        if not bl:
            continue
        rng = np.random.default_rng(stable_seed(r.record_id))
        if len(bl) > 8:
            bl = [bl[i] for i in np.sort(rng.choice(len(bl), 8, replace=False))]
        xf = bandpass(x)
        for k, b in enumerate(bl):
            c = (b['start'] + b['end']) // 2
            w = window_at(x, c)
            W.append((w / (np.abs(w).max() + 1e-9)).astype(np.float16))
            rows.append(dict(beat_id=f'{r.record_id}_b{k:02d}', record_id=r.record_id, patient_id=r.patient_id, abnormal=r.abnormal,
                             subset=r.subset, kind='real', beat_snr_db=beat_snr(xf, b, 'none')[0]))
    sel = pd.DataFrame(rows)
    thr = float(np.percentile(sel[sel.abnormal == 0].beat_snr_db, 30))
    sel['clean'] = (sel.beat_snr_db >= thr).astype(int)
    # synthetic rhythm set from clean normal fold-0 recordings (test-set synthetic algorithm: 20 s, p(premature) = 0.2)
    rng = np.random.default_rng(SEED)
    src = sel[(sel.abnormal == 0) & (sel.clean == 1)].record_id.unique()
    srows = []
    for rid in rng.permutation(src)[:300]:
        x = audio[rid]; xf = bandpass(x)
        bl = [b for b in beats_of(segs[rid]) if beat_snr(xf, b, 'none')[0] >= thr]
        if len(bl) < 3:
            continue
        pid = man.set_index('record_id').patient_id[rid]
        for kind in ('regular', 'premature', 'irregular'):
            y, marks = sequence(x, bl, kind, rng, 20 * FS)
            cand = [j for j in range(1, len(marks) - 1) if (marks[j]['prem'] if kind == 'premature' else True)]
            for k, j in enumerate(sorted(rng.permutation(cand)[:4])):
                c = marks[j]['start'] + marks[j]['len'] // 2
                w = window_at(y, c)
                W.append((w / (np.abs(w).max() + 1e-9)).astype(np.float16))
                srows.append(dict(beat_id=f'syn_{kind[:3]}_{rid}_b{k:02d}', record_id=rid, patient_id=pid, abnormal=0, subset=rid[0],
                                  kind=f'synthetic_{kind}', beat_snr_db=np.nan, clean=1))
    sel = pd.concat([sel, pd.DataFrame(srows)], ignore_index=True)
    W = np.stack(W)
    np.save(os.path.join(PROC, 'pn_hd_sel_audio.npy'), W)
    sel.to_csv(os.path.join(PROC, 'pn_hd_sel.csv'), index=False)
    b1, b2 = compute(W)
    np.savez(os.path.join(PROC, 'pn_hd_sel_feats.npz'), b2=b2)
    print('selection set', sel.kind.value_counts().to_dict(), 'clean real', int(sel[sel.kind == 'real'].clean.sum()))


# ---------------------------------------------------------------- H4 network
class GradRev(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class H4Net(nn.Module):
    def __init__(self, mu=None, sd=None):
        super().__init__()
        self.cnn = deep.CNN()                                 # B4 trunk (front end + 4 blocks); its own head is unused
        self.register_buffer('mu', torch.zeros(11) if mu is None else torch.as_tensor(mu, dtype=torch.float32))
        self.register_buffer('sd', torch.ones(11) if sd is None else torch.as_tensor(sd, dtype=torch.float32))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(139, 1))
        self.adv = nn.Sequential(nn.Linear(139 + 6, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))

    def joint(self, x, f):
        return torch.cat([self.cnn.embed(x), (f - self.mu) / self.sd], 1)

    def forward(self, x, f):
        return self.head(self.joint(x, f)).squeeze(1)


def h4_data():
    meta = pd.read_csv(os.path.join(PROC, 'pn_windows.csv'))
    aug = pd.read_csv(os.path.join(PROC, 'pn_hd_aug.csv'))
    X = np.concatenate([deep.prep_audio(np.load(os.path.join(PROC, 'pn_windows_audio.npy'), mmap_mode='r')),
                        deep.prep_audio(np.load(os.path.join(PROC, 'pn_hd_aug_audio.npy')))])
    F = np.concatenate([np.load(os.path.join(PROC, 'pn_feats.npz'))['b2'], np.load(os.path.join(PROC, 'pn_hd_aug_feats.npz'))['b2']]).astype(np.float32)
    d = pd.concat([meta[['fold', 'unusable', 'abnormal', 'subset']].assign(real=1), aug[['fold', 'unusable', 'abnormal', 'subset']].assign(real=0)],
                  ignore_index=True)
    return meta, X, F, d


def train_h4(key, s):
    lam = LAMS[key]
    t0 = time.time(); seed_all(s)
    meta, X, F, d = h4_data()
    y = d.unusable.values.astype(np.float32); ab = d.abnormal.values.astype(np.float32)
    S = np.eye(6, dtype=np.float32)[d.subset.map({k: i for i, k in enumerate(SUBSETS)}).values]
    tr = np.flatnonzero(d.fold.values != 0); va = np.flatnonzero((d.fold.values == 0) & (d.real.values == 1))
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
    Xg = torch.from_numpy(X).to(DEV, torch.float16)
    Fg, yg, ag, Sg = (torch.from_numpy(v).to(DEV) for v in (F, y, ab, S))
    ug = torch.from_numpy((y == 0).astype(np.float32)).to(DEV)
    model = H4Net(mu, sd).to(DEV)
    pw = torch.tensor((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1), device=DEV)
    lq, la = nn.BCEWithLogitsLoss(pos_weight=pw), nn.BCEWithLogitsLoss(reduction='none')
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(s)
    name = tag(f'hd_h4_{key}', s)
    f = open(os.path.join(LOGT, f'{name}.csv'), 'w', newline=''); wr = csv.writer(f)
    wr.writerow(['epoch', 'train_loss', 'val_loss', 'val_auroc', 'val_balanced_acc', 'lr', 'adv_loss'])
    best, best_state, bad = np.inf, None, 0
    tr_t, va_g = torch.from_numpy(tr), torch.from_numpy(va).to(DEV)
    for ep in range(1, 51):
        model.train()
        order = tr_t[torch.randperm(len(tr), generator=g)]
        tl = al = 0.0; nb = 0
        for i in range(0, len(order), 64):
            idx = order[i:i + 64].to(DEV)
            if len(idx) < 2:
                continue
            j = model.joint(Xg[idx].float(), Fg[idx])
            loss = lq(model.head(j).squeeze(1), yg[idx])
            m = ug[idx]
            if m.sum() > 0:
                a = model.adv(torch.cat([GradRev.apply(j, lam), Sg[idx]], 1)).squeeze(1)
                adv = (la(a, ag[idx]) * m).sum() / m.sum()
                loss = loss + adv            # gradient reversal scales the adversary's signal to the trunk by -lambda
                al += adv.item()
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        lv = predict(model, Xg[va_g], Fg[va_g])
        vl = float(lq(torch.from_numpy(lv).to(DEV), yg[va_g]).item())
        pv = 1 / (1 + np.exp(-lv)); auc = roc_auc_score(y[va], pv)
        wr.writerow([ep, round(tl / nb, 5), round(vl, 5), round(auc, 5), round(((pv >= .5) == y[va]).mean(), 5), 1e-3, round(al / nb, 5)]); f.flush()
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 5:
                break
    f.close()
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(MODELS, f'{name}.pt'))
    pv = 1 / (1 + np.exp(-predict(model, Xg[va_g], Fg[va_g]).astype(np.float64)))
    pd.DataFrame(dict(window_id=meta.window_id.values[va], fold=0, y=y[va].astype(int), score=pv)).to_csv(
        os.path.join(PROC, f'val_preds_{name}.csv'), index=False)
    print(name, 'val AUROC', round(roc_auc_score(y[va], pv), 4), 'time', round(time.time() - t0), 's')


@torch.no_grad()
def predict(model, X, F, bs=256):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs]; fb = F[i:i + bs]
        xb = xb if torch.is_tensor(xb) else torch.from_numpy(np.asarray(xb, np.float32))
        fb = fb if torch.is_tensor(fb) else torch.from_numpy(np.asarray(fb, np.float32))
        out.append(model(xb.to(DEV, torch.float32), fb.to(DEV, torch.float32)).float().cpu())
    return torch.cat(out).numpy()


def load_h4(name):
    m = H4Net().to(DEV)
    m.load_state_dict(torch.load(os.path.join(MODELS, f'{name}.pt'), map_location=DEV)); m.eval()
    return m


def score_h4(name, W, das):
    """Unusable probability for raw windows W with their cached Das features (used by score.py)."""
    return 1 / (1 + np.exp(-predict(load_h4(name), deep.prep_audio(W), das).astype(np.float64)))


# ---------------------------------------------------------------- H1-H3 on PhysioNet
def oof(meta, m, s):
    """Out-of-fold scores (and B4 embeddings) for every PhysioNet window: folds 1-4 from fold models, fold 0 from the main model."""
    sc = np.full(len(meta), np.nan); E = np.full((len(meta), 128), np.nan, np.float32)
    for k in range(5):
        t = tag(m, s) + ('' if k == 0 else f'_f{k}')
        idx = np.flatnonzero(meta.fold.values == k)
        v = pd.read_csv(os.path.join(PROC, f'val_preds_{t}.csv'))
        assert (v.window_id.values == meta.window_id.values[idx]).all()
        sc[idx] = v.score.values
        E[idx] = np.load(os.path.join(PROC, f'oof_embed_{t}.npy'))
    return sc, E


def refs(meta, s):
    """Percentile references: out-of-fold B2 / B4 scores of clean usable normal windows in folds 1-4."""
    b2 = pd.read_csv(os.path.join(PROC, f'val_preds_{tag("b2_rf", s)}.csv')).score.values
    b4, _ = oof(meta, 'b4_cnn', s)
    ok = ((meta.unusable == 0) & (meta.aug == 0) & (meta.abnormal == 0) & (meta.fold != 0)).values
    return b2[ok], b4[ok]


def fit():
    meta = pd.read_csv(os.path.join(PROC, 'pn_windows.csv'))
    das = np.load(os.path.join(PROC, 'pn_feats.npz'))['b2']
    y, va, tr = meta.unusable.values, np.flatnonzero(meta.fold.values == 0), np.flatnonzero(meta.fold.values != 0)
    for s in SEEDS:
        r2, r4 = refs(meta, s)
        b2 = pd.read_csv(os.path.join(PROC, f'val_preds_{tag("b2_rf", s)}.csv')).score.values
        b4, E = oof(meta, 'b4_cnn', s)
        p2, p4 = pct(r2, b2[va]), pct(r4, b4[va])
        for m, sc in (('hd_avg', (p2 + p4) / 2), ('hd_and', np.minimum(p2, p4))):
            pd.DataFrame(dict(window_id=meta.window_id.values[va], fold=0, y=y[va], score=sc)).to_csv(os.path.join(PROC, f'val_preds_{tag(m, s)}.csv'), index=False)
        X = np.hstack([E, das]).astype(np.float64)
        lr = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', C=1.0, max_iter=5000)).fit(X[tr], y[tr])
        joblib.dump(lr, os.path.join(MODELS, f'{tag("hd_lr", s)}.joblib'))
        pd.DataFrame(dict(window_id=meta.window_id.values[va], fold=0, y=y[va], score=lr.predict_proba(X[va])[:, 1])).to_csv(
            os.path.join(PROC, f'val_preds_{tag("hd_lr", s)}.csv'), index=False)
        print('fit seed', s, flush=True)


# ---------------------------------------------------------------- selection
def cand_scores(m, s, W, das, b2, b4, emb, r2, r4):
    if m == 'hd_avg':
        return (pct(r2, b2) + pct(r4, b4)) / 2
    if m == 'hd_and':
        return np.minimum(pct(r2, b2), pct(r4, b4))
    if m == 'hd_lr':
        return joblib.load(os.path.join(MODELS, f'{tag(m, s)}.joblib')).predict_proba(np.hstack([emb, das]).astype(np.float64))[:, 1]
    return score_h4(tag(m, s), W, das)


def gap_ci(df, col, a, b, rng, nb=1000):
    """rate(group a) - rate(group b), 95 % patient bootstrap CI (patients resampled within each group)."""
    def counts(g):
        t = g.groupby('patient_id')[col].agg(['sum', 'size']); return t['sum'].values.astype(float), t['size'].values.astype(float)
    ra, na = counts(df[a]); rb, nb_ = counts(df[b])
    ia, ib = rng.integers(0, len(ra), (nb, len(ra))), rng.integers(0, len(rb), (nb, len(rb)))
    d = ra[ia].sum(1) / na[ia].sum(1) - rb[ib].sum(1) / nb_[ib].sum(1)
    return ra.sum() / na.sum() - rb.sum() / nb_.sum(), np.percentile(d, 2.5), np.percentile(d, 97.5)


def select():
    meta = pd.read_csv(os.path.join(PROC, 'pn_windows.csv'))
    va = np.flatnonzero(meta.fold.values == 0)
    sel = pd.read_csv(os.path.join(PROC, 'pn_hd_sel.csv'))
    W = np.load(os.path.join(PROC, 'pn_hd_sel_audio.npy'))
    das = np.load(os.path.join(PROC, 'pn_hd_sel_feats.npz'))['b2']
    rows = []
    for s in SEEDS:
        r2, r4 = refs(meta, s)
        b2 = joblib.load(os.path.join(MODELS, f'{tag("b2_rf", s)}.joblib')).predict_proba(das)[:, 1]
        b4, emb = deep.score(tag('b4_cnn', s), W, embed=True)
        for m in CANDS + ['b4_cnn', 'b2_rf']:
            sc = b4 if m == 'b4_cnn' else b2 if m == 'b2_rf' else cand_scores(m, s, W, das, b2, b4, emb, r2, r4)
            v = pd.read_csv(os.path.join(PROC, f'val_preds_{tag(m, s)}.csv'))
            v = v[v.window_id.isin(meta.window_id.values[va])]
            auc = roc_auc_score(v.y, v.score)
            d = sel.assign(score=sc)
            real = d[(d.kind == 'real') & (d.clean == 1)]
            thr = float(np.percentile(real[real.abnormal == 0].score, 90, method='higher'))
            d['rej'] = (d.score > thr).astype(int)
            rng = np.random.default_rng(SEED)
            gp = gap_ci(d, 'rej', (d.kind == 'real') & (d.clean == 1) & (d.abnormal == 1), (d.kind == 'real') & (d.clean == 1) & (d.abnormal == 0), rng)
            gr = gap_ci(d, 'rej', d.kind.isin(['synthetic_premature', 'synthetic_irregular']), d.kind == 'synthetic_regular', rng)
            rows.append(dict(candidate=m, seed=s, val_auroc=auc, threshold=thr, path_gap=gp[0], path_gap_ci_lo=gp[1], path_gap_ci_hi=gp[2],
                             rhythm_gap=gr[0], rhythm_gap_ci_lo=gr[1], rhythm_gap_ci_hi=gr[2]))
        print('select seed', s, flush=True)
    t = pd.DataFrame(rows)
    t['path_ci_incl0'] = (t.path_gap_ci_lo <= 0) & (t.path_gap_ci_hi >= 0)
    t['rhythm_ci_incl0'] = (t.rhythm_gap_ci_lo <= 0) & (t.rhythm_gap_ci_hi >= 0)
    t.to_csv(os.path.join(RES, 'hd_selection_per_seed.csv'), index=False)
    a = t.groupby('candidate', sort=False).agg(val_auroc=('val_auroc', 'mean'), val_auroc_sd=('val_auroc', 'std'), path_gap=('path_gap', 'mean'),
                                               rhythm_gap=('rhythm_gap', 'mean'), n_path_ci0=('path_ci_incl0', 'sum'),
                                               n_rhythm_ci0=('rhythm_ci_incl0', 'sum')).reset_index()
    b4auc = a.set_index('candidate').val_auroc['b4_cnn']
    c = a[a.candidate.isin(CANDS)].copy()
    c['order'] = c.candidate.map({m: i for i, m in enumerate(CANDS)})
    c['qualifies'] = (c.n_path_ci0 >= 2) & (c.n_rhythm_ci0 >= 2)
    c['within_001_of_b4'] = c.val_auroc >= b4auc - 0.01
    c['sum_abs_gap'] = c.path_gap.abs() + c.rhythm_gap.abs()
    if c.qualifies.any():
        pick, rule = c[c.qualifies].sort_values(['val_auroc', 'order'], ascending=[False, True]).candidate.iloc[0], 'qualifying, highest AUROC'
    elif c.within_001_of_b4.any():
        pick, rule = c[c.within_001_of_b4].sort_values(['sum_abs_gap', 'order']).candidate.iloc[0], 'none qualifies; smallest summed |gap| within 0.01 AUROC of B4'
    else:
        pick, rule = c.sort_values(['val_auroc', 'order'], ascending=[False, True]).candidate.iloc[0], 'none qualifies or within 0.01 of B4: highest AUROC (failure)'
    c['selected'] = c.candidate == pick; c['rule'] = rule
    out = pd.concat([c, a[~a.candidate.isin(CANDS)]], ignore_index=True)
    out.to_csv(os.path.join(RES, 'hd_selection.csv'), index=False)
    print('SELECTED', pick, '|', rule)


# ---------------------------------------------------------------- test-set scores for H1-H3
def test():
    from score import SC, SNRS
    DS = ['circor', 'zch', 'bmdhs', 'pascal', 'synth']
    meta = pd.read_csv(os.path.join(PROC, 'pn_windows.csv'))
    for s in SEEDS:
        r2, r4 = refs(meta, s)
        lr = joblib.load(os.path.join(MODELS, f'{tag("hd_lr", s)}.joblib'))
        for exp in ('e1', 'e2', 'e2s'):
            for ds in DS:
                n = len(pd.read_csv(os.path.join(SC, f'{exp}_{ds}_index.csv'), usecols=['beat_id']))
                das = np.load(os.path.join(SC, f'e1_{ds}_none_feats.npz'))['b2'] if exp == 'e1' else \
                    np.concatenate([np.load(os.path.join(SC, f'{exp}_{ds}_{lv}_feats.npz'))['b2'] for lv in SNRS])
                b2 = np.load(os.path.join(SC, f'{exp}_{ds}_{tag("b2_rf", s)}.npy'))
                b4 = np.load(os.path.join(SC, f'{exp}_{ds}_{tag("b4_cnn", s)}.npy'))
                E = np.load(os.path.join(SC, f'{exp}_{ds}_{tag("b4_cnn", s)}_emb.npy')).astype(np.float32)
                assert len(das) == len(b2) == len(b4) == len(E) == n
                p2, p4 = pct(r2, b2), pct(r4, b4)
                np.save(os.path.join(SC, f'{exp}_{ds}_{tag("hd_avg", s)}.npy'), (p2 + p4) / 2)
                np.save(os.path.join(SC, f'{exp}_{ds}_{tag("hd_and", s)}.npy'), np.minimum(p2, p4))
                np.save(os.path.join(SC, f'{exp}_{ds}_{tag("hd_lr", s)}.npy'), lr.predict_proba(np.hstack([E, das]).astype(np.float64))[:, 1])
        print('test seed', s, flush=True)


if __name__ == '__main__':
    a = sys.argv[1]
    {'aug': build_aug, 'selset': build_selset, 'fit': fit, 'select': select, 'test': test}.get(a, lambda: None)()
    if a == 'train':
        train_h4(sys.argv[2], int(sys.argv[3]))
