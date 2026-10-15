"""Phase C/D data side: test-set manifests, calibration split, beats, clean-beat rule, beat windows.

  python prep_test.py circor|bmdhs|pascal|synth

Per dataset writes
  manifests/<ds>.csv                    record_id, patient_id, split (calibration / eval), labels
  data/processed/<ds>_segments.csv.gz   state segments (2 kHz samples) used for beats and B7
  data/processed/<ds>_beats.csv         one row per beat (class, extra labels, positions, SNR, clean flag)
  data/processed/<ds>_beat_windows.npy  5 s window centred on each beat (peak-normalised float16)
  results/d_segmentation_failures_<ds>.csv   (segmenter datasets only)
Split first: patients are assigned to calibration / eval before any beat is cut.
At most MAX_BEATS beats per recording are kept (random, seed derived from the record id).
"""
import os, sys, glob, re
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from common import (CIRCOR_ROOT, BMD_ROOT, PASCAL_ROOT, ZCH_ROOT, PROC, MAN, RES, FS, WIN, SEED,
                    load_audio, bandpass, window_at, stable_seed)

MAX_BEATS = 8
CAL_FRAC = 0.20
BEAT_MIN, BEAT_MAX = int(0.3 * FS), int(2.0 * FS)


# ---------------------------------------------------------------- manifests
def circor_manifest():
    meta = pd.read_csv(os.path.join(CIRCOR_ROOT, 'training_data.csv'))
    rows = []
    for w in sorted(glob.glob(os.path.join(CIRCOR_ROOT, 'training_data', '*.wav'))):
        rid = os.path.basename(w)[:-4]
        pid, loc = rid.split('_')[0], rid.split('_')[1]
        rows.append(dict(record_id=rid, pid=int(pid), location=loc, path=w))
    m = pd.DataFrame(rows).merge(meta, left_on='pid', right_on='Patient ID')
    m['patient_id'] = 'cc' + m.pid.astype(str)
    locs = m['Murmur locations'].fillna('').str.split('+')
    audible = [l in L for l, L in zip(m.location, locs)]
    dia = m['Diastolic murmur timing'].notna() & (m['Diastolic murmur timing'] != 'nan')

    def cls(r, aud, d):
        if r.Murmur == 'Unknown':
            return 'unknown'
        if r.Murmur == 'Absent':
            return 'normal' if r.Outcome == 'Normal' else 'no_murmur_abnormal_outcome'
        if not aud:
            return 'murmur_patient_other_location'
        return 'murmur_diastolic' if d else 'murmur_systolic'
    m['cls'] = [cls(r, a, d) for r, a, d in zip(m.itertuples(), audible, dia)]
    m['patient_normal'] = (m.Murmur == 'Absent') & (m.Outcome == 'Normal')
    m['murmur_phase'] = np.where(m.cls == 'murmur_systolic', 'systole',
                                 np.where(m.cls == 'murmur_diastolic', 'diastole', 'none'))
    # diastolic-only vs systolic+diastolic: a beat with both murmurs has no murmur-free phase
    both = (m.cls == 'murmur_diastolic') & m['Systolic murmur timing'].notna() & (m['Systolic murmur timing'] != 'nan')
    m.loc[both, 'murmur_phase'] = 'both'
    m = m.rename(columns={'Systolic murmur timing': 'sys_timing', 'Systolic murmur grading': 'sys_grade',
                          'Systolic murmur shape': 'sys_shape', 'Diastolic murmur timing': 'dia_timing',
                          'Diastolic murmur grading': 'dia_grade', 'Diastolic murmur shape': 'dia_shape',
                          'Murmur': 'murmur', 'Outcome': 'outcome', 'Age': 'age'})
    m['murmur_present'] = (m.murmur == 'Present').astype(int)
    keep = ['record_id', 'patient_id', 'location', 'path', 'cls', 'patient_normal', 'murmur_phase', 'murmur',
            'murmur_present', 'outcome', 'age', 'sys_timing', 'sys_grade', 'sys_shape', 'dia_timing', 'dia_grade', 'dia_shape']
    return m[keep]


def bmdhs_manifest():
    t = pd.read_csv(os.path.join(BMD_ROOT, 'train.csv'))
    rows = []
    phase = {'AS': 'systole', 'MR': 'systole', 'AR': 'diastole', 'MS': 'diastole'}
    for r in t.itertuples():
        dz = [d for d in ('AS', 'AR', 'MR', 'MS') if getattr(r, d) == 1]
        cls = 'normal' if r.N == 1 else (dz[0] if len(dz) == 1 else 'multi')
        ph = {phase[d] for d in dz}
        mp = 'none' if not ph else (ph.pop() if len(ph) == 1 else 'both')
        for k in range(1, 9):
            rec = getattr(r, f'recording_{k}')
            if isinstance(rec, str) and os.path.exists(os.path.join(BMD_ROOT, 'train', rec + '.wav')):
                rows.append(dict(record_id=rec, patient_id='bmd_' + r.patient_id, path=os.path.join(BMD_ROOT, 'train', rec + '.wav'),
                                 cls=cls, patient_normal=(r.N == 1), murmur_phase=mp, diseases='+'.join(dz) or 'N',
                                 AS=r.AS, AR=r.AR, MR=r.MR, MS=r.MS))
    return pd.DataFrame(rows)


def pascal_manifest():
    t = pd.read_csv(os.path.join(PASCAL_ROOT, 'set_b.csv'))
    t = t[t.label.notna()]
    on_disk = {os.path.basename(p).replace('__', '_'): p for p in glob.glob(os.path.join(PASCAL_ROOT, 'set_b', '*.wav'))
               if 'unlabelled' not in p.lower()}
    rows = []
    for r in t.itertuples():
        key = os.path.basename(r.fname).replace('Btraining_', '')
        p = on_disk.get(key)
        if p is None:
            print('PASCAL: no file for', r.fname); continue
        num = re.search(r'_(\d+)_\d+', key).group(1)
        cls = {'extrastole': 'extrasystole'}.get(r.label, r.label)
        rows.append(dict(record_id='pasB_' + key[:-4], patient_id='pas_' + num, path=p, cls=cls,
                         murmur_phase='unknown' if cls == 'murmur' else 'none', sublabel=r.sublabel))
    m = pd.DataFrame(rows)
    # a patient is "normal" only if all their recordings are normal
    m['patient_normal'] = m.groupby('patient_id').cls.transform(lambda c: (c == 'normal').all())
    return m


def zch_manifest():
    t = pd.read_csv(os.path.join(ZCH_ROOT, 'Clean Heartsound Data Details.csv'), encoding='utf-8-sig')
    # murmur phase used only by the clean-beat rule: VSD/ASD systolic, PDA continuous, PFO usually silent
    phase = {'NORMAL': 'none', 'VSD': 'systole', 'ASD': 'systole', 'PDA': 'both', 'PFO': 'unknown'}
    rows = []
    for r in t.itertuples():
        p = os.path.join(ZCH_ROOT, 'clean Heartsound Data', r.fileName)
        if not os.path.exists(p):
            continue
        d = str(r.diagnosis).strip().upper()
        rid = r.fileName[:-4]
        rows.append(dict(record_id=rid, patient_id='zch_' + rid, path=p, cls='control' if d == 'NORMAL' else d,
                         patient_normal=(d == 'NORMAL'), murmur_phase=phase.get(d, 'unknown'), age_days=getattr(r, '_3')))
    return pd.DataFrame(rows)


def assign_splits(m, name):
    rng = np.random.default_rng(stable_seed(SEED, name))
    normal_p = np.array(sorted(m.loc[m.patient_normal.astype(bool), 'patient_id'].unique()))
    cal = set(rng.choice(normal_p, int(round(CAL_FRAC * len(normal_p))), replace=False))
    m['split'] = np.where(m.patient_id.isin(cal), 'calibration', 'eval')
    if name == 'circor':   # E5: non-calibration patients 60/40 by patient, stratified by murmur
        pts = m[m.split == 'eval'].groupby('patient_id').murmur.first()
        pts = pts[pts != 'Unknown']
        e5 = {}
        for lab, g in pts.groupby(pts):
            ids = rng.permutation(g.index.values)
            n_tr = int(round(0.6 * len(ids)))
            e5.update({p: 'e5_train' for p in ids[:n_tr]}); e5.update({p: 'e5_test' for p in ids[n_tr:]})
        m['e5_split'] = m.patient_id.map(e5).fillna('none')
    m.insert(0, 'dataset', name)
    return m


# ---------------------------------------------------------------- segments -> beats
def circor_segments(rid, n):
    t = np.loadtxt(os.path.join(CIRCOR_ROOT, 'training_data', rid + '.tsv'), ndmin=2)
    st = {1: 'S1', 2: 'systole', 3: 'S2', 4: 'diastole'}
    out = []
    for a, b, l in t:
        s, e = int(round(a * FS)), min(int(round(b * FS)), n)
        out.append((s, e, st.get(int(l), 'unannotated')))
    return out


def beats_from_segments(segs):
    """S1 onset -> next S1 onset with S1, systole, S2, diastole contiguous and fully annotated."""
    out = []
    for i in range(len(segs) - 4):
        lab = [segs[i + k][2] for k in range(5)]
        if lab != ['S1', 'systole', 'S2', 'diastole', 'S1']:
            continue
        if any(abs(segs[i + k][1] - segs[i + k + 1][0]) > 2 for k in range(4)):
            continue
        out.append(dict(start=segs[i][0], s1_end=segs[i][1], s2_start=segs[i + 2][0], s2_end=segs[i + 2][1],
                        dia_start=segs[i + 3][0], end=segs[i + 4][0]))
    return out


def beat_snr(xf, b, murmur_phase):
    p = lambda s, e: float(np.mean(xf[s:e].astype(np.float64) ** 2)) + 1e-20
    ps = (np.sum(xf[b['start']:b['s1_end']].astype(np.float64) ** 2) + np.sum(xf[b['s2_start']:b['s2_end']].astype(np.float64) ** 2)) \
        / max((b['s1_end'] - b['start']) + (b['s2_end'] - b['s2_start']), 1) + 1e-20
    p_sys, p_dia = p(b['s1_end'], b['s2_start']), p(b['dia_start'], b['end'])
    if murmur_phase == 'systole':
        ref = p_dia
    elif murmur_phase == 'diastole':
        ref = p_sys
    elif murmur_phase in ('both', 'unknown'):
        ref = min(p_sys, p_dia)
    else:
        ref = p_dia
    return 10 * np.log10(ps / ref), 10 * np.log10(ps / p_sys), 10 * np.log10(ps / p_dia)


def process_record(args):
    name, r = args
    x = load_audio(r['path'])
    fail = ''
    try:
        segs = circor_segments(r['record_id'], len(x)) if name == 'circor' else __import__('segmenter').segment(x)
    except Exception as e:
        segs, fail = [], f'segmenter_error:{type(e).__name__}'
    bts = beats_from_segments(segs)
    n_all = len(bts)
    bts = [b for b in bts if BEAT_MIN <= b['end'] - b['start'] <= BEAT_MAX]
    n_ok = len(bts)
    if not bts and not fail:
        fail = 'no_valid_beats'
    rng = np.random.default_rng(stable_seed(r['record_id']))
    if len(bts) > MAX_BEATS:
        idx = np.sort(rng.choice(len(bts), MAX_BEATS, replace=False))
        bts = [bts[i] for i in idx]
    xf = bandpass(x)
    wins, rows = [], []
    for k, b in enumerate(bts):
        snr, snr_sys, snr_dia = beat_snr(xf, b, r['murmur_phase'])
        c = (b['start'] + b['end']) // 2
        w = window_at(x, c)
        wins.append((w / (np.abs(w).max() + 1e-9)).astype(np.float16))
        rows.append(dict(beat_id=f"{r['record_id']}_b{k:02d}", record_id=r['record_id'], centre=c,
                         beat_snr_db=snr, snr_vs_sys_db=snr_sys, snr_vs_dia_db=snr_dia,
                         rel_start=b['start'] - (c - WIN // 2), rel_end=b['end'] - (c - WIN // 2), **b))
    seg_df = pd.DataFrame([(r['record_id'], s, e, l) for s, e, l in segs], columns=['record_id', 'start', 'end', 'state'])
    return rows, wins, seg_df, dict(record_id=r['record_id'], n_beats_raw=n_all, n_beats_valid=n_ok, failure=fail,
                                    duration_s=len(x) / FS)


def build(name):
    m = {'circor': circor_manifest, 'bmdhs': bmdhs_manifest, 'pascal': pascal_manifest, 'zch': zch_manifest}[name]()
    m = assign_splits(m, name)
    m.drop(columns=['path']).to_csv(os.path.join(MAN, f'{name}.csv'), index=False)
    recs = m.to_dict('records')
    nproc = 10
    with ProcessPoolExecutor(nproc) as ex:
        res = list(ex.map(process_record, [(name, r) for r in recs], chunksize=4))
    rows, wins, segs, stats = [], [], [], []
    for rr, ww, sg, st in res:
        rows += rr; wins += ww; segs.append(sg); stats.append(st)
    beats = pd.DataFrame(rows)
    info = m.drop(columns=['path']).set_index('record_id')
    beats = beats.join(info, on='record_id')
    # clean-beat rule: SNR >= 30th percentile of calibration normal beats
    cal_norm = beats[(beats.split == 'calibration') & beats.patient_normal.astype(bool) & (beats.cls.isin(['normal', 'control']))]
    thr = float(np.percentile(cal_norm.beat_snr_db, 30))
    beats['clean'] = (beats.beat_snr_db >= thr).astype(int)
    beats['clean_thr_db'] = thr
    beats.to_csv(os.path.join(PROC, f'{name}_beats.csv'), index=False)
    np.save(os.path.join(PROC, f'{name}_beat_windows.npy'), np.stack(wins))
    pd.concat(segs).to_csv(os.path.join(PROC, f'{name}_segments.csv.gz'), index=False)
    st = pd.DataFrame(stats).join(info[['patient_id', 'cls', 'split']], on='record_id')
    st.to_csv(os.path.join(PROC, f'{name}_record_stats.csv'), index=False)
    if name != 'circor':
        f = st.assign(failed=(st.failure != '').astype(int),
                      beat_reject_frac=1 - st.n_beats_valid / st.n_beats_raw.clip(lower=1)) \
            .groupby('cls').agg(n_records=('record_id', 'size'), record_failure_rate=('failed', 'mean'),
                                mean_frac_beats_bad_duration=('beat_reject_frac', 'mean')).reset_index()
        f.insert(0, 'dataset', name)
        f.to_csv(os.path.join(RES, f'd_segmentation_failures_{name}.csv'), index=False)
    print(name, len(m), 'records', m.patient_id.nunique(), 'patients', len(beats), 'beats, clean thr', round(thr, 2))
    print(beats.groupby('cls').clean.agg(['size', 'mean']).round(3).to_string())


# ---------------------------------------------------------------- synthetic rhythm (from clean normal CirCor eval beats)
def synth():
    rng = np.random.default_rng(SEED)
    beats = pd.read_csv(os.path.join(PROC, 'circor_beats.csv'))
    man = pd.read_csv(os.path.join(MAN, 'circor.csv'))
    segs = pd.read_csv(os.path.join(PROC, 'circor_segments.csv.gz'))
    src = beats[(beats.cls == 'normal') & (beats.split == 'eval') & (beats.clean == 1)]
    recs = rng.permutation(src.record_id.unique())[:300]
    paths = dict(zip(man.record_id, [os.path.join(CIRCOR_ROOT, 'training_data', r + '.wav') for r in man.record_id]))
    rows, wins, seg_rows = [], [], []
    for kind in ('synthetic_premature', 'synthetic_irregular'):
        for rid in recs:
            x = load_audio(paths[rid])
            # all clean-quality beats of this recording (full annotation), in order
            bl = beats_from_segments([tuple(v) for v in segs[segs.record_id == rid][['start', 'end', 'state']].values])
            xf = bandpass(x)
            bl = [b for b in bl if BEAT_MIN <= b['end'] - b['start'] <= BEAT_MAX
                  and beat_snr(xf, b, 'none')[0] >= src.clean_thr_db.iloc[0]]
            if len(bl) < 3:
                continue
            out, sg, marks, t, i = [], [], [], 0, 0
            while t < 20 * FS:
                b = bl[i % len(bl)]; i += 1
                L = b['end'] - b['start']
                dia = b['end'] - b['dia_start']
                if kind == 'synthetic_premature':
                    prem = rng.random() < 0.2 and len(out) > 0
                    target = L
                else:
                    prem = False
                    target = int(L * rng.uniform(0.7, 1.3))
                seg = x[b['start']:b['end']]
                # fit to target length by trimming / extending diastole (reflection of the diastole tail)
                min_keep = (b['dia_start'] - b['start']) + int(0.05 * FS)
                target = max(target, min_keep)
                if target <= L:
                    seg = seg[:target]
                else:
                    tail = x[b['dia_start']:b['end']]
                    ext = np.resize(np.concatenate([tail, tail[::-1]]), target - L)
                    seg = np.concatenate([seg, ext])
                if prem:   # shift this beat early by 25-40 % of the previous RR: trim previous beat's diastole
                    prev_s, prev_len, prev_dia = marks[-1]['start'], marks[-1]['len'], marks[-1]['dia_len']
                    shift = int(rng.uniform(0.25, 0.40) * prev_len)
                    shift = min(shift, prev_dia - int(0.05 * FS))
                    if shift > 0:
                        out[-1] = out[-1][:prev_len - shift]
                        marks[-1]['len'] -= shift; marks[-1]['dia_len'] -= shift
                        t -= shift
                        # compensatory pause: extend this beat's diastole by the same amount
                        tail = x[b['dia_start']:b['end']]
                        seg = np.concatenate([seg, np.resize(np.concatenate([tail, tail[::-1]]), shift)])
                marks.append(dict(start=t, len=len(seg), dia_len=dia + (len(seg) - L), prem=prem,
                                  s1=b['s1_end'] - b['start'], s2s=b['s2_start'] - b['start'], s2e=b['s2_end'] - b['start'],
                                  dias=b['dia_start'] - b['start']))
                out.append(seg); t += len(seg)
            y = np.concatenate(out)
            sid = f'{kind[10:13]}_{rid}'
            for mk in marks:
                seg_rows += [(sid, mk['start'], mk['start'] + mk['s1'], 'S1'), (sid, mk['start'] + mk['s1'], mk['start'] + mk['s2s'], 'systole'),
                             (sid, mk['start'] + mk['s2s'], mk['start'] + mk['s2e'], 'S2'),
                             (sid, mk['start'] + mk['dias'], mk['start'] + mk['len'], 'diastole')]
            cand = [j for j in range(1, len(marks) - 1) if (marks[j]['prem'] if kind == 'synthetic_premature' else True)]
            for k, j in enumerate(sorted(rng.permutation(cand)[:4])):
                mk = marks[j]
                b = dict(start=mk['start'], s1_end=mk['start'] + mk['s1'], s2_start=mk['start'] + mk['s2s'],
                         s2_end=mk['start'] + mk['s2e'], dia_start=mk['start'] + mk['dias'], end=mk['start'] + mk['len'])
                c = (b['start'] + b['end']) // 2
                w = window_at(y, c)
                wins.append((w / (np.abs(w).max() + 1e-9)).astype(np.float16))
                snr = beat_snr(bandpass(y), b, 'none')
                rows.append(dict(beat_id=f'{sid}_b{k:02d}', record_id=sid, source_record=rid, centre=c,
                                 beat_snr_db=snr[0], rel_start=b['start'] - (c - WIN // 2), rel_end=b['end'] - (c - WIN // 2),
                                 cls=kind, **b))
    df = pd.DataFrame(rows)
    info = man.set_index('record_id')[['patient_id', 'split']]
    df = df.join(info, on='source_record')
    df['dataset'] = 'circor_synth'; df['clean'] = 1; df['murmur_phase'] = 'none'
    df.to_csv(os.path.join(PROC, 'synth_beats.csv'), index=False)
    np.save(os.path.join(PROC, 'synth_beat_windows.npy'), np.stack(wins))
    pd.DataFrame(seg_rows, columns=['record_id', 'start', 'end', 'state']).to_csv(os.path.join(PROC, 'synth_segments.csv.gz'), index=False)
    print('synth', df.groupby('cls').size().to_dict())


if __name__ == '__main__':
    synth() if sys.argv[1] == 'synth' else build(sys.argv[1])
