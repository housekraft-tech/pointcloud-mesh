"""Pack a LAS whole -- every point, no thinning -- for the browser.

27 M points will not survive the usual route of dequantising into Float32 on
the CPU: positions alone would be 324 MB of Float32 and colours another 324 MB.
So nothing is dequantised. Positions stay Int16 and colours stay Uint8, both
handed to WebGL as-is, and the mapping back to metres is done by the object's
own scale and position. That is 162 MB + 81 MB, and the GPU does the arithmetic
for free.

Read in chunks so the whole cloud is never in memory at once.
"""
import sys, json, numpy as np, laspy

SRC = sys.argv[1] if len(sys.argv) > 1 else "output/mujammel_aligned_z0.las"
OUT = sys.argv[2] if len(sys.argv) > 2 else "output/full"

with laspy.open(SRC) as r:
    hdr = r.header
    N = hdr.point_count
    mins, maxs = np.asarray(hdr.mins), np.asarray(hdr.maxs)
print(f"{SRC}: {N:,} points, extent "
      f"{maxs[0]-mins[0]:.1f} x {maxs[1]-mins[1]:.1f} x {maxs[2]-mins[2]:.1f} m")

# three.js frame: Y up, and the scan's Y runs backwards -> (x, z, -y)
lo = np.array([mins[0], mins[2], -maxs[1]], float)
hi = np.array([maxs[0], maxs[2], -mins[1]], float)
scale = (hi-lo)/65534.0
scale[scale <= 0] = 1e-9

import os
KEEP = None
if os.path.exists("output/keep_mask.npy"):
    KEEP = np.load("output/keep_mask.npy")
    if len(KEEP) != N: KEEP = None
    else: print(f"declutter mask: keeping {KEEP.sum():,} of {N:,}")

fp = open(f"{OUT}_pos.bin", "wb")
fc = open(f"{OUT}_col.bin", "wb")
done = 0; kept = 0; cmax = 0
with laspy.open(SRC) as r:
    for pts in r.chunk_iterator(2_000_000):
        V = np.column_stack([np.asarray(pts.x), np.asarray(pts.z),
                             -np.asarray(pts.y)])
        # free-standing objects and people, by the connectivity rule in
        # declutter_middle.py -- walls and parapets are untouched
        if KEEP is not None:
            sel = KEEP[done:done+len(V)]
            V = V[sel]
        Q = np.round((V-lo)/scale)
        np.clip(Q, 0, 65534, out=Q)
        fp.write((Q-32767).astype("<i2").tobytes())
        try:
            C = np.column_stack([pts.red, pts.green, pts.blue]).astype(np.float64)
        except Exception:
            it = np.asarray(pts.intensity, np.float64)
            C = np.column_stack([it, it, it])
        if KEEP is not None: C = C[sel]
        cmax = max(cmax, float(C.max()) if C.size else 0.0)
        fc.write((C/(65535.0 if cmax > 255 else 255.0)*255
                  ).clip(0, 255).astype(np.uint8).tobytes())
        done += len(sel) if KEEP is not None else len(V)
        kept += len(V)
        print(f"  {done:,} / {N:,}", flush=True)
fp.close(); fc.close()
json.dump(dict(n=int(kept), lo=lo.tolist(), scale=scale.tolist(),
               src=SRC, colour="16-bit" if cmax > 255 else "8-bit"),
          open(f"{OUT}_meta.json", "w"))
import os
print(f"\nwrote {OUT}_pos.bin {os.path.getsize(OUT+'_pos.bin')/1e6:.0f} MB  "
      f"+ {OUT}_col.bin {os.path.getsize(OUT+'_col.bin')/1e6:.0f} MB")
print(f"{kept:,} points at full resolution -- nothing downscaled")
