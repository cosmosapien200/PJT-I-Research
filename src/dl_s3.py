"""Parallel per-file download of a PhysioNet project from the open S3 mirror, using SHA256SUMS.txt as the file list."""
import sys, os, urllib.request, concurrent.futures as cf
proj, out = sys.argv[1], sys.argv[2]          # e.g. circor-heart-sound/1.0.3  data/raw/circor
base = f"https://physionet-open.s3.amazonaws.com/{proj}/"
os.makedirs(out, exist_ok=True)
lines = urllib.request.urlopen(base + "SHA256SUMS.txt").read().decode().splitlines()
files = [l.split(maxsplit=1)[1] for l in lines if l.strip()]
def get(f):
    p = os.path.join(out, f)
    if os.path.exists(p) and os.path.getsize(p) > 0: return 0
    os.makedirs(os.path.dirname(p), exist_ok=True)
    for _ in range(4):
        try:
            urllib.request.urlretrieve(base + f, p + ".part"); os.replace(p + ".part", p); return 0
        except Exception: pass
    return 1
with cf.ThreadPoolExecutor(32) as ex: fails = sum(ex.map(get, files))
print(f"done {len(files)} files, {fails} failed")
