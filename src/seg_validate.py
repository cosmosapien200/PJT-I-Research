"""Validate the segmenter heart-rate range on held-out PhysioNet records (training data only; no test-set tuning).
Compares beat-duration agreement with hand-corrected annotations for HR range (30,120) vs (30,220)."""
import os
import numpy as np, pandas as pd
from common import PROC, MAN, SEED, RES
import segmenter as S
from prep_test import beats_from_segments

man = pd.read_csv(os.path.join(MAN, 'physionet2016.csv'))
seg = pd.read_csv(os.path.join(PROC, 'pn_segments.csv.gz'))
aud = np.load(os.path.join(PROC, 'pn_audio.npz'))
rng = np.random.default_rng(SEED + 1)
recs = [r for r in rng.permutation(man[man.sqi == 1].record_id.values)[:400]]
rows = []
for rid in recs[:150]:
    g = seg[seg.record_id == rid]
    ref = beats_from_segments([tuple(v) for v in g[['start', 'end', 'state']].values])
    if len(ref) < 5:
        continue
    ref_hr = 60 * 2000 / np.median([b['end'] - b['start'] for b in ref])
    s1_ref = np.array([b['start'] for b in ref])
    for rngname, hr_range in (('30-120', (30, 120)), ('30-220', (30, 220))):
        bts = beats_from_segments(S.segment(aud[rid], hr_range))
        hr = 60 * 2000 / np.median([b['end'] - b['start'] for b in bts]) if bts else np.nan
        s1 = np.array([b['start'] for b in bts])
        hit = np.mean([np.min(np.abs(s1 - t)) <= 100 for t in s1_ref]) if len(s1) else 0.0   # S1 onset within 50 ms
        rows.append(dict(record_id=rid, hr_range=rngname, ref_hr=ref_hr, seg_hr=hr, s1_onset_recall_50ms=hit))
d = pd.DataFrame(rows)
d['abs_hr_err'] = (d.seg_hr - d.ref_hr).abs()
s = d.groupby('hr_range').agg(n=('record_id', 'size'), median_abs_hr_err=('abs_hr_err', 'median'),
                              frac_hr_err_gt10=('abs_hr_err', lambda v: (v > 10).mean()), s1_recall=('s1_onset_recall_50ms', 'mean'))
s.to_csv(os.path.join(RES, 'fix3_segmenter_validation_physionet.csv'))
print(s.round(3).to_string())
