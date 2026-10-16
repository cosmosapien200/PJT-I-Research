"""Cache B3 (CardioPHON) features for the PhysioNet corpus: data/processed/pn_b3feats.npz"""
import os, time
import numpy as np
from common import PROC
import cardiophon_features as C

if __name__ == '__main__':
    t = time.time()
    W = np.load(os.path.join(PROC, 'pn_windows_audio.npy'), mmap_mode='r')
    X = C.compute(W)
    np.savez(os.path.join(PROC, 'pn_b3feats.npz'), X=X, names=np.array(C.names()))
    print('pn', X.shape, 'nan rows', int(np.isnan(X).any(1).sum()), round(time.time() - t), 's')
