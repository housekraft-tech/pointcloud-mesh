"""Recover the scanner trajectory from the LAS, with no trajectory file.

The export carries gps_time, and the returns fall into 406 frames of about
0.707 s. Every return in a frame radiates from wherever the scanner was, and
for a sweep that covers all directions those unit vectors balance out. The
point where they balance -- where sum (xi - p)/|xi - p| = 0 -- is exactly the
geometric median of the frame's returns. So the sensor position per frame comes
out of a few Weiszfeld iterations, and no trajectory file is needed.

This is what unlocks the free-space term: a cell that a ray passed through is
empty, and a cell enclosed and never traversed is solid, whether or not both of
its faces were ever seen. That is the half of the cell-complex labelling that is
missing, and the reason its recall is only 50%.
"""
import numpy as np, laspy, json

SRC = "mujammelexport.las"
SUB = 4000            # returns per frame used for the estimate

def geometric_median(X, iters=64, eps=1e-6):
    p = X.mean(0)
    for _ in range(iters):
        d = np.linalg.norm(X-p, axis=1)
        d = np.maximum(d, 1e-9)
        w = 1.0/d
        q = (X*w[:, None]).sum(0)/w.sum()
        if np.linalg.norm(q-p) < eps: return q
        p = q
    return p

ts, XYZ = [], []
with laspy.open(SRC) as r:
    for pts in r.chunk_iterator(4_000_000):
        ts.append(np.asarray(pts.gps_time))
        XYZ.append(np.column_stack([pts.x, pts.y, pts.z]).astype(np.float64))
t = np.concatenate(ts); P = np.concatenate(XYZ)
order = np.argsort(t, kind='stable'); t = t[order]; P = P[order]
u, start = np.unique(t, return_index=True)
bounds = list(start)+[len(t)]
print(f"{len(P):,} returns in {len(u)} frames over {u[-1]-u[0]:.1f} s")

rng = np.random.default_rng(0)
traj = np.zeros((len(u), 3))
for i in range(len(u)):
    seg = P[bounds[i]:bounds[i+1]]
    if len(seg) > SUB: seg = seg[rng.choice(len(seg), SUB, replace=False)]
    traj[i] = geometric_median(seg)

step = np.linalg.norm(np.diff(traj, axis=0), axis=1)
print(f"trajectory: {len(traj)} poses")
print(f"  step between frames: median {step.min():.3f}/{np.median(step):.3f}/"
      f"{step.max():.3f} m (min/median/max)")
print(f"  walking speed  {np.median(step)/0.707:.2f} m/s median, "
      f"{step.max()/0.707:.2f} m/s peak")
print(f"  path length {step.sum():.1f} m")
print(f"  height z: min {traj[:,2].min():.2f} median {np.median(traj[:,2]):.2f} "
      f"max {traj[:,2].max():.2f} m")
print(f"  extent x {traj[:,0].min():.1f}..{traj[:,0].max():.1f}  "
      f"y {traj[:,1].min():.1f}..{traj[:,1].max():.1f}")
np.save("output/trajectory_raw.npy", np.column_stack([u, traj]))
print("wrote output/trajectory_raw.npy  (gps_time, x, y, z) in the SCANNER frame")
