"""Carve the free space the scanner looked through, in the model's own frame.

Geometry alone cannot tell a wall from two surfaces with a gap between them: a
wall's interior and a cavity are both empty of returns, both flat-sided, both
full height. The difference is that the scanner *saw through* the cavity from
somewhere and never saw through the wall. That is a measurement, and this is
where it comes from.

No trajectory file is needed. The export carries gps_time, and the returns fall
into frames of about 0.7 s. Every return in a frame radiates from wherever the
scanner was, so for a sweep covering all directions those unit vectors balance
out at the sensor: the point where sum (xi - p)/|xi - p| = 0 is the geometric
median of the frame's returns, which a few Weiszfeld iterations find.

One pose per frame is not enough. A frame is 0.7 s and the operator walks at
1.4 m/s, so the sensor moves a metre inside it; rays cast from the frame's
average position are tilted by tens of degrees at close range and cut straight
through walls. Carved that way, the inside of every wall came out as free as
open air -- 0.84 against 0.85 for the middle of a room -- which is the signature
of a ray origin that is wrong, not of a wall that is hollow. So the pose is
interpolated across each frame by the return's position in acquisition order,
which puts the origin within a few centimetres.

Then every return is paired with its own pose and the segment between them is
marched on the grid. Each cell the segment crosses was looked through, so it is
empty. The last few centimetres are left alone -- that is the surface, not free
space.

The output is in the MODEL's frame, not the scanner's: the same 0.5-percentile
z shift poisson_mesh.py applied and the same yaw modular_poisson.py measured, so
a cell can be looked up directly from a wall's coordinates.
"""
import sys, json, time, argparse
from pathlib import Path
import numpy as np

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

STOP = 0.05          # m: leave the last 50 mm of every ray alone -- surface
STOP_REL = 0.02      # and 2% of its length on top, because a pose error of a
                     # few centimetres throws a long ray sideways by more
SUB = 4000           # returns per frame used to locate the sensor
MIN_HITS = 8         # a cell is free only if this many rays crossed it. One
                     # stray ray -- a grazing shot, a pose off by a few cm --
                     # would otherwise carve a hole through solid masonry, and
                     # about a tenth of the rays do clip something on the way.


def geometric_median(X, iters=64, eps=1e-6):
    p = X.mean(0)
    for _ in range(iters):
        d = np.maximum(np.linalg.norm(X-p, axis=1), 1e-9)
        w = 1.0/d
        q = (X*w[:, None]).sum(0)/w.sum()
        if np.linalg.norm(q-p) < eps:
            return q
        p = q
    return p


def load(las_path):
    import laspy
    ts, XYZ = [], []
    with laspy.open(las_path) as r:
        for pts in r.chunk_iterator(4_000_000):
            ts.append(np.asarray(pts.gps_time))
            XYZ.append(np.column_stack([pts.x, pts.y, pts.z]).astype(np.float64))
    return np.concatenate(ts), np.concatenate(XYZ)


def trajectory(t, P):
    """A pose per frame, and a fractional frame index per return.

    The returned index is the position along the trajectory each return was
    taken from, counting the frame's own pose as the middle of that frame, so
    interpolating between poses gives every return an origin of its own.
    """
    o = np.argsort(t, kind="stable")
    ts, Ps = t[o], P[o]
    u, start = np.unique(ts, return_index=True)
    bounds = list(start) + [len(ts)]
    rng = np.random.default_rng(0)
    traj = np.zeros((len(u), 3))
    for i in range(len(u)):
        seg = Ps[bounds[i]:bounds[i+1]]
        if len(seg) > SUB:
            seg = seg[rng.choice(len(seg), SUB, replace=False)]
        traj[i] = geometric_median(seg)
    step = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    dt = np.median(np.diff(u)) or 1.0
    log(f"{len(u)} frames over {u[-1]-u[0]:.0f} s; path {step.sum():.1f} m, "
        f"median speed {np.median(step)/dt:.2f} m/s, moving "
        f"{np.median(step)*1000:.0f} mm within a frame; "
        f"sensor height {np.median(traj[:, 2]):.2f} m over the cloud floor")

    # fractional frame index, in the ORIGINAL point order
    fi = np.empty(len(ts))
    for i in range(len(u)):
        a, b = bounds[i], bounds[i+1]
        fi[a:b] = i - 0.5 + (np.arange(b-a) + 0.5)/(b-a)
    out = np.empty(len(t))
    out[o] = fi
    return u, traj, np.clip(out, 0, len(u)-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--las", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=None,
                    help="a model manifest, to take the yaw from")
    ap.add_argument("--yaw", type=float, default=None)
    ap.add_argument("--cell", type=float, default=0.03)
    ap.add_argument("--rays", type=int, default=6_000_000)
    a = ap.parse_args()

    yaw = a.yaw
    if yaw is None and a.manifest:
        yaw = json.load(open(a.manifest))["yaw_deg"]
    if yaw is None:
        raise SystemExit("give --yaw or --manifest so the frames match")

    t, P = load(a.las)
    log(f"{len(P):,} returns")
    dz = float(np.percentile(P[:, 2], 0.5))       # as poisson_mesh.py did
    ts, traj, fidx = trajectory(t, P)

    r = np.radians(-yaw); c, s = np.cos(r), np.sin(r)
    def to_model(Q):
        return np.column_stack([Q[:, 0]*c - Q[:, 1]*s,
                                Q[:, 0]*s + Q[:, 1]*c, Q[:, 2] - dz])
    P = to_model(P); POSE = to_model(traj)
    log(f"model frame: cloud z {P[:, 2].min():.2f}..{P[:, 2].max():.2f}, "
        f"sensor z {POSE[:, 2].min():.2f}..{POSE[:, 2].max():.2f}")

    G = a.cell
    lo = np.minimum(P.min(0), POSE.min(0)) - 5*G
    hi = np.maximum(P.max(0), POSE.max(0)) + 5*G
    n = np.ceil((hi-lo)/G).astype(int) + 1
    log(f"grid {n[0]}x{n[1]}x{n[2]} @ {G*1000:.0f} mm = {n.prod()/1e6:.1f}M cells")
    hits = np.zeros(int(n.prod()), np.int32)

    if len(P) > a.rays:
        keep = np.random.default_rng(1).choice(len(P), a.rays, replace=False)
        Pr, fr = P[keep], fidx[keep]
    else:
        Pr, fr = P, fidx
    # each return gets its own origin, interpolated along the walk
    i0 = np.floor(fr).astype(int)
    i1 = np.minimum(i0+1, len(POSE)-1)
    al = (fr - i0)[:, None]
    O = POSE[i0]*(1-al) + POSE[i1]*al
    d = Pr - O
    L = np.linalg.norm(d, axis=1)
    ok = L > 0.20
    O, d, L = O[ok], d[ok], L[ok]
    u = d/L[:, None]
    L = np.maximum(L - STOP - STOP_REL*L, 0.0)
    log(f"casting {len(O):,} rays, median length {np.median(L):.2f} m")

    STEP = G*0.7
    CH = 2_000_000
    for s0 in range(0, len(O), CH):
        Oc, uc, Lc = O[s0:s0+CH], u[s0:s0+CH], L[s0:s0+CH]
        for k in range(int(np.ceil(Lc.max()/STEP))):
            live = (k*STEP) < Lc
            if not live.any():
                break
            q = Oc[live] + uc[live]*(k*STEP)
            ij = ((q - lo)/G).astype(np.int64)
            m = ((ij >= 0).all(1) & (ij[:, 0] < n[0]) & (ij[:, 1] < n[1])
                 & (ij[:, 2] < n[2]))
            ij = ij[m]
            np.add.at(hits, (ij[:, 0]*n[1] + ij[:, 1])*n[2] + ij[:, 2], 1)
        log(f"  {min(s0+CH, len(O)):,}/{len(O):,} rays")

    seen = hits >= MIN_HITS
    log(f"looked through {seen.sum():,} cells ({seen.mean()*100:.1f}% of the box) "
        f"with at least {MIN_HITS} rays; "
        f"{int((hits > 0).sum()):,} cells were touched by any ray at all")
    np.savez_compressed(a.out, seen=np.packbits(seen), n=n, lo=lo, G=G,
                        shape=np.array([len(seen)]))
    log(f"wrote {a.out} ({Path(a.out).stat().st_size/1e6:.1f} MB)")


def load_grid(path):
    """Read a carved grid back: a lookup from position to 'was looked through'."""
    d = np.load(path)
    n = d["n"]; lo = d["lo"]; G = float(d["G"])
    seen = np.unpackbits(d["seen"])[:int(d["shape"][0])].astype(bool)
    return dict(seen=seen, n=n, lo=lo, G=G)


def seen_mask(grid, Q):
    """Which of these positions the scanner looked through."""
    out = np.zeros(len(Q), bool)
    if grid is None or len(Q) == 0:
        return out
    n, lo, G, seen = grid["n"], grid["lo"], grid["G"], grid["seen"]
    ij = np.floor((Q - lo)/G).astype(np.int64)
    m = (ij >= 0).all(1) & (ij[:, 0] < n[0]) & (ij[:, 1] < n[1]) & (ij[:, 2] < n[2])
    if not m.any():
        return out
    flat = (ij[m, 0]*n[1] + ij[m, 1])*n[2] + ij[m, 2]
    out[np.where(m)[0]] = seen[flat]
    return out


def free_fraction(grid, Q):
    """How much of a set of positions the scanner looked through."""
    if grid is None or len(Q) == 0:
        return 0.0
    n, lo, G, seen = grid["n"], grid["lo"], grid["G"], grid["seen"]
    ij = np.floor((Q - lo)/G).astype(np.int64)
    m = (ij >= 0).all(1) & (ij[:, 0] < n[0]) & (ij[:, 1] < n[1]) & (ij[:, 2] < n[2])
    if not m.any():
        return 0.0
    flat = (ij[m, 0]*n[1] + ij[m, 1])*n[2] + ij[m, 2]
    out = np.zeros(len(Q), bool)
    out[np.where(m)[0]] = seen[flat]
    return float(out.mean())


if __name__ == "__main__":
    main()
