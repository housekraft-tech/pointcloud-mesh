"""Carve free space by ray-casting from the recovered trajectory.

A cell the scanner saw THROUGH is empty. A cell enclosed by surfaces and never
seen through is solid -- whether or not both of its faces were ever scanned.
That second half is what the cell complex was missing, and why it drew only the
walls it happened to see from both sides.

Each return is paired with its frame's pose, and the segment between them is
walked on a 20 mm grid (3D DDA, vectorised by stepping along the ray). Every
cell the segment crosses is marked seen-through. The endpoint is NOT marked --
that is the surface.
"""
import numpy as np, laspy, json, os

G = 0.02
STOP = 0.04          # leave the last 40 mm of each ray alone: that is surface
J = json.load(open("output/fp_walls.json")); H = J['clear_height']
T = json.load(open("output/raw_to_aligned.json"))
tr = np.load("output/trajectory_raw.npy")            # (gps_time, x, y, z) raw

a = np.radians(T['yaw_deg']); c, s = np.cos(a), np.sin(a)
def to_aligned(P):
    return np.column_stack([P[:, 0]*c - P[:, 1]*s + T['tx'],
                            P[:, 0]*s + P[:, 1]*c + T['ty'],
                            P[:, 2] + T['dz']])
POSE_T = tr[:, 0]
POSE = to_aligned(tr[:, 1:4])
print(f"{len(POSE)} poses, aligned z {POSE[:,2].min():.2f}..{POSE[:,2].max():.2f} m")

# the grid the cell complex uses
with laspy.open("output/mujammel_aligned_z0.las") as r: p = r.read()
A = np.column_stack([p.x, p.y, p.z]).astype(np.float64)
if os.path.exists("output/keep_mask.npy"):
    k = np.load("output/keep_mask.npy")
    if len(k) == len(A): A = A[k]
A = A[A[:, 2] < H-0.12]
lo = A.min(0); hi = A.max(0)
n = np.ceil((hi-lo)/G).astype(int) + 1
print(f"grid {n[0]}x{n[1]}x{n[2]} @ {G*1000:.0f} mm = {n.prod()/1e6:.0f}M cells")
seen = np.zeros(int(n.prod()), bool)

# returns, with their frame times, in the aligned frame
with laspy.open("mujammelexport.las") as r:
    RT, RP = [], []
    for pts in r.chunk_iterator(4_000_000):
        RT.append(np.asarray(pts.gps_time))
        RP.append(np.column_stack([pts.x, pts.y, pts.z]).astype(np.float64))
rt = np.concatenate(RT); rp = to_aligned(np.concatenate(RP))
keep = (rp[:, 2] < H-0.12) & (rp[:, 2] > -0.20)
rt, rp = rt[keep], rp[keep]
print(f"{len(rp):,} returns inside the storey")

pi = np.searchsorted(POSE_T, rt)
np.clip(pi, 0, len(POSE)-1, out=pi)
O = POSE[pi]
d = rp - O
L = np.linalg.norm(d, axis=1)
ok = L > 0.15
O, d, L = O[ok], d[ok], L[ok]
u = d/L[:, None]
L = np.maximum(L-STOP, 0.0)
print(f"casting {len(O):,} rays, median length {np.median(L):.2f} m")

# march every ray together, one step at a time
STEP = G*0.7
maxs = int(np.ceil(L.max()/STEP))
print(f"{maxs:,} steps of {STEP*1000:.0f} mm")
CH = 2_000_000
for s0 in range(0, len(O), CH):
    Oc, uc, Lc = O[s0:s0+CH], u[s0:s0+CH], L[s0:s0+CH]
    nst = int(np.ceil(Lc.max()/STEP))
    for k in range(nst):
        t = k*STEP
        live = t < Lc
        if not live.any(): break
        q = Oc[live] + uc[live]*t
        ij = ((q-lo)/G).astype(np.int64)
        m = (ij >= 0).all(1) & (ij[:, 0] < n[0]) & (ij[:, 1] < n[1]) & (ij[:, 2] < n[2])
        ij = ij[m]
        seen[(ij[:, 0]*n[1] + ij[:, 1])*n[2] + ij[:, 2]] = True
    print(f"  {min(s0+CH, len(O)):,}/{len(O):,} rays", flush=True)

SEEN = seen.reshape(n)
print(f"seen-through: {SEEN.sum():,} cells ({SEEN.mean()*100:.1f}% of the volume)")
np.save("output/freespace.npy", SEEN)
json.dump(dict(lo=lo.tolist(), n=n.tolist(), G=G),
          open("output/freespace_meta.json", "w"))
print("wrote output/freespace.npy + freespace_meta.json")
