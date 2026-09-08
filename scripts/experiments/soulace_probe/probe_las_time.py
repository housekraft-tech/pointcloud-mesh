"""Chunked min/max of gps_time (and header summary) for a LAS, low memory. Read-only."""
import sys, numpy as np, laspy
p = sys.argv[1]
with laspy.open(p) as f:
    h = f.header
    print(f"{p}\n  ver={h.version} pdrf={h.point_format.id} npoints={h.point_count}")
    print("  dims:", [d.name for d in h.point_format.dimensions])
    print(f"  scales={h.scales} offsets={h.offsets}")
    print(f"  mins={h.mins} maxs={h.maxs}")
    print("  global_encoding:", h.global_encoding)
    lo, hi, tot = np.inf, -np.inf, 0
    for pts in f.chunk_iterator(4_000_000):
        t = np.asarray(pts.gps_time)
        lo = min(lo, float(t.min())); hi = max(hi, float(t.max())); tot += len(t)
    print(f"  gps_time: min={lo:.6f} max={hi:.6f} span={hi-lo:.3f} s  (n={tot})")
