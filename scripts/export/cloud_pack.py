"""Pack the LiDAR into a small binary the viewer can carry inline.

Only the very top of the ceiling is cut -- just enough slab to see in from
above. Everything else stays, so the model can be checked against the scan it
came from rather than against a filtered version of it.

Positions are quantised to Int16 over the cloud's own bounding box, which is
0.25 mm across a 16 m extent -- far finer than anything being judged here -- so
the file is 6 bytes of geometry plus 3 of colour per point instead of 15.
"""
import numpy as np, laspy, json

TARGET = 4_000_000
CUT = 0.12                     # metres of ceiling slab removed, from the top down

with laspy.open("output/mujammel_structural_v6.las") as r: p = r.read()
P = np.column_stack([p.x, p.y, p.z]).astype(np.float64)
rgb = np.column_stack([p.red, p.green, p.blue]).astype(np.float64)
H = json.load(open("output/fp_walls.json"))['clear_height']
keep = P[:, 2] < H - CUT
print(f"{len(P):,} points, cutting z > {(H-CUT)*1000:.0f} mm removes "
      f"{(~keep).sum():,} ({(~keep).mean()*100:.1f}%) -- the ceiling slab only")
P = P[keep]; rgb = rgb[keep]
# Random thinning to 700k took 2.7% of the returns, which is fine on a broad
# wall and ruinous on anything narrow -- a reveal or a mullion loses most of
# its points and reads as a cut through the geometry that is not there. Thin
# on a VOXEL grid instead: one point per 12 mm cell, so density is capped where
# the scanner dwelt without ever emptying a thinly-sampled surface.
GRID = 0.012
if len(P) > TARGET:
    key = np.floor(P/GRID).astype(np.int64)
    key -= key.min(axis=0)
    span = key.max(axis=0)+1
    h = (key[:, 0]*span[1] + key[:, 1])*span[2] + key[:, 2]
    order = np.argsort(h, kind='stable')
    first = np.ones(len(h), bool)
    first[1:] = h[order][1:] != h[order][:-1]
    idx = order[first]
    print(f"voxel thinning at {GRID*1000:.0f} mm keeps {len(idx):,} of {len(P):,}")
    if len(idx) > TARGET:
        idx = idx[np.random.default_rng(0).choice(len(idx), TARGET, replace=False)]
    idx = np.sort(idx); P = P[idx]; rgb = rgb[idx]
print(f"kept {len(P):,} points for the viewer")

# three.js frame: Y up, and Y(scan) runs backwards -> (x, z, -y)
V = np.column_stack([P[:, 0], P[:, 2], -P[:, 1]])
lo = V.min(0); hi = V.max(0)
scale = (hi - lo) / 65534.0
Q = np.round((V - lo) / scale).astype(np.int16) - 32767
mx = rgb.max()
C = (rgb / (65535.0 if mx > 255 else 255.0) * 255).clip(0, 255).astype(np.uint8)
buf = Q.astype("<i2").tobytes() + C.tobytes()
open("output/cloud.bin", "wb").write(buf)
json.dump(dict(n=len(P), lo=lo.tolist(), scale=scale.tolist(),
               cut_mm=round(CUT*1000), ceiling_mm=round(H*1000)),
          open("output/cloud_meta.json", "w"))
print(f"wrote output/cloud.bin  {len(buf)/1e6:.1f} MB "
      f"({len(buf)/len(P):.0f} bytes/point), colour {'16-bit' if mx>255 else '8-bit'}")
