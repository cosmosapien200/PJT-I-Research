"""Compute hand-crafted features once for a window set and cache them in data/processed/<name>_feats.npz.

  python cache_features.py pn_windows_audio.npy pn
"""
import os, sys, time
import numpy as np
from common import PROC
import features as F

if __name__ == '__main__':
    src, name = sys.argv[1], sys.argv[2]
    t = time.time()
    W = np.load(os.path.join(PROC, src), mmap_mode='r')
    b1, b2 = F.compute(W)
    np.savez(os.path.join(PROC, f'{name}_feats.npz'), b1=b1, b2=b2, b1_names=F.B1_NAMES, b2_names=F.B2_NAMES)
    print(name, len(W), 'windows', round(time.time() - t), 's')
