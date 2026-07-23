"""analyse_poisson.py
-----------------
What is actually IN the Poisson mesh, measured rather than assumed.

The question this answers: the Poisson surface is the highest-fidelity thing we
have (750 MB, every groove and reveal), but it is one undifferentiated blob and
it contains furniture. Before deciding what to build from it, measure:

  1. size, topology, how many pieces
  2. the normal budget -- how much area faces sideways (wall), up (floor),
     down (ceiling), and how much is diagonal clutter
  3. the vertical area, binned by wall azimuth -- does the flat have two clean
     structural axes, and how much sits off-axis
  4. how much vertical area sits ON a stage-1 major wall line vs stranded in the
     middle of a room (= furniture, the thing we want to strip)
  5. per-wall-plane: continuous height coverage, and the HOLES in it, which is
     the only direct evidence of a door or window opening

Read as: "of X m2 of vertical surface, Y% lies within T of a known wall".

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\analyse_poisson.py \\
      <poisson.obj> [major_walls.json]
"""
import sys, json, time
from pathlib import Path
import numpy as np

CELL = 0.05          # m, raster for coverage maps
WALL_NEAR = 0.25     # m, "this face belongs to that major wall"
VERT_COS = 0.34      # |nz| below this = vertical face (~70 deg from horizontal)
HORZ_COS = 0.87      # |nz| above this = horizontal face (~30 deg)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load(path):
    """Stream the OBJ: 750 MB will not survive a naive parse twice."""
    V, F = [], []
    with open(path, "r", buffering=1 << 22) as fh:
        for ln in fh:
            if ln.startswith("v "):
                p = ln.split()
                V.append((float(p[1]), float(p[2]), float(p[3])))
            elif ln.startswith("f "):
                p = ln.split()
                idx = [int(t.split("/")[0]) - 1 for t in p[1:]]
                for k in range(1, len(idx) - 1):
                    F.append((idx[0], idx[k], idx[k + 1]))
    return np.asarray(V, np.float64), np.asarray(F, np.int64)


def main(obj_path, walls_path=None):
    t0 = time.time()
    log(f"reading {obj_path} ...")
    V, F = load(obj_path)
    log(f"{len(V):,} verts, {len(F):,} tris in {time.time()-t0:.0f}s")
    lo, hi = V.min(0), V.max(0)
    log(f"bounds  x {lo[0]:.2f}..{hi[0]:.2f}  y {lo[1]:.2f}..{hi[1]:.2f}  "
        f"z {lo[2]:.2f}..{hi[2]:.2f}   ({hi[0]-lo[0]:.2f} x {hi[1]-lo[1]:.2f} "
        f"x {hi[2]-lo[2]:.2f} m)")

    # ---- per-triangle geometry -------------------------------------------
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(B - A, C - A)
    area = 0.5 * np.linalg.norm(cr, axis=1)
    n = cr / np.maximum(np.linalg.norm(cr, axis=1)[:, None], 1e-12)
    cen = (A + B + C) / 3.0
    tot = area.sum()
    log(f"total surface area {tot:,.0f} m2   "
        f"median tri edge {np.median(np.linalg.norm(B-A, axis=1))*1000:.1f} mm")

    # ---- normal budget ----------------------------------------------------
    nz = np.abs(n[:, 2])
    vert = nz < VERT_COS
    horz = nz > HORZ_COS
    diag = ~vert & ~horz
    up = horz & (n[:, 2] > 0)
    dn = horz & (n[:, 2] < 0)
    log("NORMAL BUDGET")
    for lab, m in (("vertical (wall-like)", vert), ("floor-facing up", up),
                   ("ceiling-facing down", dn), ("diagonal / clutter", diag)):
        log(f"  {lab:<22} {area[m].sum():9,.0f} m2  {100*area[m].sum()/tot:5.1f}%")

    # ---- azimuth of the vertical area ------------------------------------
    az = np.degrees(np.arctan2(n[vert, 1], n[vert, 0])) % 180.0
    aw = area[vert]
    hist, edges = np.histogram(az, bins=36, range=(0, 180), weights=aw)
    log("VERTICAL AREA BY AZIMUTH (10 deg bins, top 8)")
    for i in np.argsort(hist)[::-1][:8]:
        log(f"  {edges[i]:5.0f}-{edges[i+1]:3.0f} deg  {hist[i]:8,.0f} m2  "
            f"{100*hist[i]/aw.sum():5.1f}%")
    # how much sits on the two dominant orthogonal axes
    best = int(np.argmax(hist))
    ax0 = 0.5 * (edges[best] + edges[best + 1])
    d = np.abs(((az - ax0 + 45) % 90) - 45)
    for tol in (2.0, 5.0, 10.0):
        m = d <= tol
        log(f"  within {tol:4.1f} deg of the {ax0:.0f}/{ax0+90:.0f} grid: "
            f"{100*aw[m].sum()/aw.sum():5.1f}% of vertical area")

    # ---- height histogram of the vertical area ---------------------------
    zc = cen[vert, 2]
    log("VERTICAL AREA BY HEIGHT (0.25 m bands)")
    zb = np.arange(lo[2], hi[2] + 0.25, 0.25)
    hz, _ = np.histogram(zc, bins=zb, weights=aw)
    for i in np.argsort(hz)[::-1][:6]:
        log(f"  z {zb[i]:5.2f}-{zb[i+1]:5.2f}  {hz[i]:8,.0f} m2")
    log(f"  area below 1.0 m (furniture band): {100*aw[zc < lo[2]+1.0].sum()/aw.sum():5.1f}%")
    log(f"  area above 2.0 m (clean wall band): {100*aw[zc > lo[2]+2.0].sum()/aw.sum():5.1f}%")

    # ---- how much vertical area is ON a known major wall -----------------
    if walls_path and Path(walls_path).exists():
        W = json.load(open(walls_path))
        segs = W["walls"] if isinstance(W, dict) and "walls" in W else W
        log(f"MAJOR WALLS: {len(segs)} runs from {Path(walls_path).name}")
        from shapely.geometry import LineString, Point
        from shapely.strtree import STRtree
        lines, kinds = [], []
        for s in segs:
            p, q = s.get("p0") or s.get("a"), s.get("p1") or s.get("b")
            if p is None:
                continue
            lines.append(LineString([p[:2], q[:2]]))
            kinds.append(s.get("kind", "wall"))
        tree = STRtree(lines)
        P = cen[vert][:, :2]
        # sample to keep it tractable, weight by the area it stands for
        step = max(1, len(P) // 400_000)
        Ps, As = P[::step], aw[::step] * step
        near = np.zeros(len(Ps), bool)
        for i, (x, y) in enumerate(Ps):
            pt = Point(x, y)
            for j in tree.query(pt.buffer(WALL_NEAR)):
                if lines[j].distance(pt) <= WALL_NEAR:
                    near[i] = True
                    break
        f_on = As[near].sum() / As.sum()
        log(f"  vertical area within {WALL_NEAR*1000:.0f} mm of a major wall: "
            f"{100*f_on:5.1f}%")
        log(f"  stranded mid-room (furniture candidate):        "
            f"{100*(1-f_on):5.1f}%  ~{(1-f_on)*aw.sum():,.0f} m2")
    else:
        log("no major_walls.json given -- skipping the on-wall test")

    # ---- occupancy: where the mesh has NOTHING ---------------------------
    # A door is a hole in a wall. Raster the vertical area in plan and in
    # elevation to see whether the holes are actually resolvable.
    nx = int((hi[0] - lo[0]) / CELL) + 2
    ny = int((hi[1] - lo[1]) / CELL) + 2
    occ = np.zeros((ny, nx), np.int32)
    ij = ((cen[vert][:, :2] - lo[:2]) / CELL).astype(np.int32)
    np.add.at(occ, (ij[:, 1], ij[:, 0]), 1)
    filled = (occ > 0)
    log(f"PLAN OCCUPANCY of vertical faces: {filled.sum():,} of {nx*ny:,} cells "
        f"({100*filled.mean():.1f}%), median {np.median(occ[filled]):.0f} tris/cell")

    # column-wise height extent -- a wall cell spans floor to ceiling, a
    # furniture cell stops early; this is the separator that actually works
    zmin = np.full((ny, nx), np.inf)
    zmax = np.full((ny, nx), -np.inf)
    np.minimum.at(zmin, (ij[:, 1], ij[:, 0]), cen[vert][:, 2])
    np.maximum.at(zmax, (ij[:, 1], ij[:, 0]), cen[vert][:, 2])
    span = np.where(filled, zmax - zmin, 0.0)
    H = hi[2] - lo[2]
    log("CELL HEIGHT SPAN (the wall-vs-furniture separator)")
    for f in (0.9, 0.75, 0.6, 0.45, 0.3):
        m = filled & (span >= f * H)
        log(f"  cells spanning >= {f*100:2.0f}% of room height ({f*H:.2f} m): "
            f"{m.sum():7,}  ({100*m.sum()/max(filled.sum(),1):5.1f}% of filled)")
    top = filled & (zmax > hi[2] - 0.35) & (zmin < lo[2] + 0.35)
    log(f"  cells touching BOTH floor and ceiling bands: {top.sum():,} "
        f"({100*top.sum()/max(filled.sum(),1):.1f}%)")

    log(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(*sys.argv[1:3])
