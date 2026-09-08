"""Decode corcam_1.ts / optcam_1.ts (ASCII timestamp lists) and compare to LAS gps_time."""
import sys, numpy as np
for p in sys.argv[1:]:
    t = np.array([float(x) for x in open(p).read().split()])
    d = np.diff(t)
    print(f"\n=== {p} ===")
    print(f"count={len(t)}  first={t[0]:.6f}  last={t[-1]:.6f}  span={t[-1]-t[0]:.3f} s")
    print(f"dt: mean={d.mean()*1000:.4f} ms  median={np.median(d)*1000:.4f} ms  min={d.min()*1000:.4f}  max={d.max()*1000:.4f}  std={d.std()*1000:.4f}")
    print(f"implied rate = {1.0/d.mean():.4f} Hz ; monotonic={bool(np.all(d>0))}")
    big = np.where(d > np.median(d)*1.5)[0]
    print(f"gaps >1.5x median: {len(big)}" + (f" at idx {big[:10]} dt={d[big[:10]]}" if len(big) else ""))
