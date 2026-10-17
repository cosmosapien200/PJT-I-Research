"""Pre-compute the B1/B2 and B3 feature caches (CPU only) for the post-hoc window sets: E2s on every test set and E1/E2/E2s on
the rebuilt synthetic set. Independent of any trained model; score.py then reads the caches.

  python precache_post.py [nproc] [rev]    rev: second worker, job list reversed (files written atomically)
"""
import os, sys, time
import numpy as np
import score
import features as F
import cardiophon_features as C3

if __name__ == '__main__':
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    rev = 'rev' in sys.argv
    jobs = [(e, 'synth') for e in ('e1', 'e2', 'e2s')] + [('e2s', ds) for ds in ('circor', 'zch', 'bmdhs', 'pascal')]

    def put(fp, **kw):   # atomic: never leave a half-written cache for the other worker
        np.savez(fp + '.tmp.npz', **kw); os.replace(fp + '.tmp.npz', fp)
    for exp, ds in (jobs[::-1] if rev else jobs):
        sets = list(score.window_sets(exp, ds)) if rev else score.window_sets(exp, ds)
        for level, rows, X, b in (sets[::-1] if rev else sets):
            t = time.time()
            fp = os.path.join(score.SC, f'{exp}_{ds}_{level}_feats.npz')
            if not os.path.exists(fp):
                b1, b2 = F.compute(X, nproc=nproc); put(fp, b1=b1, b2=b2)
            fp = os.path.join(score.SC, f'{exp}_{ds}_{level}_b3feats.npz')
            if not os.path.exists(fp):
                put(fp, X=C3.compute(X, nproc=nproc))
            print(exp, ds, level, len(X), round(time.time() - t), 's', flush=True)
    print('ALLDONE')
