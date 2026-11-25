"""
check_data.py - data-quality and leakage checks on the VTUAD cache (exact duplicate clips, silent clips, loudness).

  python check_data.py            -> prints a report and writes report/data_check.json
Loads one cache file at a time (hashes only), so memory use stays modest.
"""
import glob, hashlib, json, os
import numpy as np

CLASSES = ["background", "cargo", "passengership", "tanker", "tug"]
os.makedirs("report", exist_ok=True)
files = sorted(f for f in glob.glob("cache/*.npz") if "deepship" not in f)
H, out = {}, {"sets": {}, "overlaps": {}}
for f in files:
    name = os.path.basename(f)[:-4]
    d = np.load(f); X, y = d["X"], d["y"]
    hs = np.array([hashlib.md5(x.tobytes()).hexdigest() for x in X])
    peak = np.abs(X.astype(np.int32)).max(1)
    rms = np.sqrt(((X.astype(np.float64) / 32768) ** 2).mean(1))
    uniq, counts = np.unique(hs, return_counts=True)
    info = dict(n=int(len(y)), per_class=np.bincount(y, minlength=5).tolist(),
                duplicate_clips_within=int((counts - 1).sum()), unique_clips=int(len(uniq)),
                silent_clips=int((peak == 0).sum()),
                median_dBFS_per_class={c: round(float(20 * np.log10(np.median(rms[(y == i) & (peak > 0)]) + 1e-12)), 1)
                                       for i, c in enumerate(CLASSES) if ((y == i) & (peak > 0)).any()})
    out["sets"][name] = info; H[name] = set(uniq.tolist())
    print(f"{name:45s} n={info['n']:6d} unique={info['unique_clips']:6d} dup-within={info['duplicate_clips_within']:5d} "
          f"silent={info['silent_clips']:3d} classes={info['per_class']}", flush=True)
    del X, d

def overlap(a, b):
    if a in H and b in H:
        k = len(H[a] & H[b]); out["overlaps"][f"{a} & {b}"] = k
        print(f"  identical clips shared by {a} and {b}: {k}")

print("\nLeakage checks (identical 1-s clips appearing in both sets):")
scen = sorted({n.rsplit("_", 1)[0] for n in H})
for s in scen:
    overlap(f"{s}_train", f"{s}_validation"); overlap(f"{s}_train", f"{s}_test"); overlap(f"{s}_validation", f"{s}_test")
for i, s1 in enumerate(scen):
    for s2 in scen[i + 1:]:
        overlap(f"{s1}_train", f"{s2}_test"); overlap(f"{s1}_test", f"{s2}_test")
json.dump(out, open("report/data_check.json", "w"), indent=1)
print("\nwrote report/data_check.json")
