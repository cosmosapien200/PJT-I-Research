"""Pre-compute B3 feature caches for every test window set (E1, E2 levels), independent of the trained model."""
import os, sys, time
import numpy as np
import score
import cardiophon_features as C3

if __name__ == '__main__':
    for ds in ['circor', 'synth', 'bmdhs', 'pascal', 'zch']:
        for exp in ('e1', 'e2'):
            for level, rows, X, b in score.window_sets(exp, ds):
                fp = os.path.join(score.SC, f'{exp}_{ds}_{level}_b3feats.npz')
                if os.path.exists(fp):
                    continue
                t = time.time()
                np.savez(fp, X=C3.compute(X, nproc=int(sys.argv[1]) if len(sys.argv) > 1 else 10))
                print(exp, ds, level, len(X), round(time.time() - t), 's', flush=True)
    print('ALLDONE')
