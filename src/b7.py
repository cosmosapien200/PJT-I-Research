"""B7: SNR reference line (not ML), after Giordano et al. 2021.

score = -(SNR dB), SNR = power of the S1 + S2 segments in the window over the power of the quietest diastolic
segment in the window (25-400 Hz band-pass). Segment positions come from the reference segmentation of the
clean recording (annotations, or the fixed Springer segmenter), so under added noise B7 uses oracle positions.
Published threshold: SNR >= 14 dB usable -> score threshold -14. Windows without a complete S1/S2/diastole
get score +50 (unusable).
"""
import numpy as np, pandas as pd
from common import WIN, bandpass

NO_SEG = 50.0


def score(W, record_ids, win_starts, segments):
    """W [N, WIN] audio; win_starts: window start in recording samples; segments: record_id,start,end,state."""
    by_rec = {r: g[['start', 'end', 'state']].values for r, g in segments.groupby('record_id')}
    out = np.full(len(W), NO_SEG)
    for i in range(len(W)):
        sg = by_rec.get(record_ids[i])
        if sg is None:
            continue
        ws = int(win_starts[i])
        inside = sg[(sg[:, 0] >= ws) & (sg[:, 1] <= ws + WIN)]
        if len(inside) == 0:
            continue
        x = bandpass(np.asarray(W[i], np.float32)).astype(np.float64) ** 2
        hs = [(s - ws, e - ws) for s, e, l in inside if l in ('S1', 'S2')]
        di = [(s - ws, e - ws) for s, e, l in inside if l == 'diastole' and e - s >= 100]
        if not hs or not di:
            continue
        ps = sum(x[s:e].sum() for s, e in hs) / sum(e - s for s, e in hs)
        pd_ = min(x[s:e].mean() for s, e in di)
        out[i] = -10 * np.log10((ps + 1e-20) / (pd_ + 1e-20))
    return out
