"""walls_from_poisson.py
-------------------
Find the wall runs in the Poisson mesh itself, instead of in six thin slices.

`house_major_walls.py` fitted runs to horizontal slices of the mesh, and four
independent checks have now shown the result is the weak link: runs 25-27 carry
4-8% wall surface (they are not walls at all), run 01 sits at 34%, while real
walls went undetected and 35% of structural surface was claimed by no run.

The elevations proved the mesh knows better: well-founded walls score 76-97%
surface coverage. So derive the runs from the same evidence the rest of the
pipeline trusts -- every vertical face in the mesh, not a handful of z-bands:

  1. a plan cell is WALL if it holds real surface (not Poisson speckle),
     reaches the ceiling, and keeps going down for at least SPAN
  2. a cell is PARAPET if it is dense and floor-anchored but stops short of
     the ceiling -- balcony balustrades are fabric, not clutter
  3. find the structural grid from the vertical faces' own azimuths, rotate
     into it, pull out runs axis-aligned, rotate back

Writes a major_walls.json in the same schema as the slice detector, so every
downstream script takes it unchanged.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\walls_from_poisson.py \\
      <poisson.obj> <out_dir> [reference_major_walls.json]
"""
import sys, json, time
from pathlib import Path
import numpy as np
import cv2

CELL = 0.05
VERT_COS = 0.34
MIN_TRIS_CELL = 30    # below this a cell is Poisson speckle floating in air
TOUCH = 0.50          # m, "reaches the ceiling" (the junction curves away)
SPAN = 1.20           # m, a ceiling-reaching cell running down this far is wall
PARA_TOP = 0.50       # m, a parapet must rise at least this high
PARA_THICK = 0.35     # m, a rail is thin in plan; a wardrobe is not
PARA_TOP_RANGE = (0.70, 1.40)   # m above floor: plausible rail height
PARA_TOP_STD = 0.18   # m, a rail is level along its length; furniture is not
PARA_BASE = 0.55      # m, ...from within this of the floor
MIN_LEN = 0.55        # m, shortest run worth calling a wall
BRIDGE = 1.20         # m, close gaps along a run up to a wide doorway
RUN_COVER = 0.55      # fraction of a run's length that must carry real
                      # evidence, measured ALONG the run, not over its box
OPEN_PX = None        # set from MIN_LEN
MIN_CELLS = 10
GRID_TOL = 6.0        # deg, half-width of the on-grid acceptance band


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load(path):
    V, F = [], []
    with open(path, "r", buffering=1 << 22) as fh:
        for ln in fh:
            if ln.startswith("v "):
                p = ln.split()
                V.append((float(p[1]), float(p[2]), float(p[3])))
            elif ln.startswith("f "):
                idx = [int(t.split("/")[0]) - 1 for t in ln.split()[1:]]
                for k in range(1, len(idx) - 1):
                    F.append((idx[0], idx[k], idx[k + 1]))
    return np.asarray(V, np.float64), np.asarray(F, np.int64)


def grid_angle(az, w):
    """The structural grid, from the vertical faces' own azimuth histogram.

    Scored by how much wall AREA lands within GRID_TOL of theta or theta+90,
    which is the thing that actually matters, rather than by picking the tallest
    histogram bin -- two adjacent bins can straddle the true axis and neither
    wins on its own.
    """
    best, ba = -1.0, 0.0
    for th in np.arange(0.0, 90.0, 0.25):
        d = np.abs(((az - th + 45.0) % 90.0) - 45.0)
        s = float(w[d <= GRID_TOL].sum())
        if s > best:
            best, ba = s, th
    return ba, best / float(w.sum())


def runs_from(mask, nx, ny, lo, R, pivot, kind, bridge=None, zmax=None,
              thick_max=None, top_range=None, top_std_max=None):
    """Elongated components of a rotated mask, returned as world-frame runs.

    The optional tests exist for parapets. "Dense, floor-anchored, stops below
    the ceiling" is also an exact description of a wardrobe, so a balustrade has
    to be identified by what makes it a balustrade: it is THIN in plan, its top
    sits at a plausible rail height, and that top is LEVEL along its length.
    Furniture is deep, and a row of it has a ragged top.
    """
    out = []
    n_px = max(3, int(round(MIN_LEN / CELL)))
    b_px = max(3, int(round((BRIDGE if bridge is None else bridge) / CELL)))
    for axis, k, kb in (("x", np.ones((1, n_px), np.uint8),
                         np.ones((1, b_px), np.uint8)),
                        ("y", np.ones((n_px, 1), np.uint8),
                         np.ones((b_px, 1), np.uint8))):
        # bridge ALONG the run first: a wall is interrupted by every doorway
        # it contains, and without this each opening splits it into a new
        # object (median run came out at 1.45 m against a 4.35 m longest).
        m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kb)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        ncc, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
        for c in range(1, ncc):
            if st[c, cv2.CC_STAT_AREA] < MIN_CELLS:
                continue
            x, y = st[c, cv2.CC_STAT_LEFT], st[c, cv2.CC_STAT_TOP]
            w, h = st[c, cv2.CC_STAT_WIDTH], st[c, cv2.CC_STAT_HEIGHT]
            if axis == "x" and w < h:
                continue
            if axis == "y" and h < w:
                continue
            L = max(w, h) * CELL
            if L < MIN_LEN:
                continue
            # Bridging is what makes a wall continuous across its doorways,
            # but left unchecked it also joins unrelated specks into runs
            # straight across an open room (123 m of "wall" in a 106 m2 flat).
            # Score every candidate on the ORIGINAL evidence, measured ALONG
            # the run -- the same test that rescued the perimeter walls in the
            # slice detector, where area density over the bounding box failed.
            sub = mask[y:y + h, x:x + w] > 0
            cov = float((sub.any(axis=0) if axis == "x"
                         else sub.any(axis=1)).mean())
            if cov < RUN_COVER:
                continue
            thick = min(w, h) * CELL
            if thick_max is not None and thick > thick_max:
                continue                       # too deep in plan to be a rail
            top = None
            if zmax is not None:
                zt = zmax[y:y + h, x:x + w][sub]
                if zt.size == 0:
                    continue
                top = float(np.median(zt))
                if top_range and not (top_range[0] <= top <= top_range[1]):
                    continue                   # not at rail height
                if top_std_max is not None and float(np.std(zt)) > top_std_max:
                    continue                   # ragged top: furniture, not a rail
            if axis == "x":
                a = np.array([lo[0] + x * CELL, lo[1] + (y + h / 2.0) * CELL])
                b = np.array([lo[0] + (x + w) * CELL, a[1]])
            else:
                a = np.array([lo[0] + (x + w / 2.0) * CELL, lo[1] + y * CELL])
                b = np.array([a[0], lo[1] + (y + h) * CELL])
            # back out of the grid frame
            p0 = R @ a + pivot
            p1 = R @ b + pivot
            rec = dict(p0=[round(float(v), 3) for v in p0],
                       p1=[round(float(v), 3) for v in p1],
                       axis=axis, length_m=round(float(L), 3),
                       thickness_m=round(float(thick), 3),
                       coverage=round(cov, 3), kind=kind)
            if top is not None:
                rec["top_z"] = round(top, 3)
            out.append(rec)
    return out


def main(obj_path, out_dir, ref=None):
    t0 = time.time()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    V, F = load(obj_path)
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(B - A, C - A)
    nn = np.linalg.norm(cr, axis=1)
    area = 0.5 * nn
    n = cr / np.maximum(nn[:, None], 1e-12)
    cen = (A + B + C) / 3.0
    vert = np.abs(n[:, 2]) < VERT_COS
    log(f"{len(F):,} tris, {int(vert.sum()):,} vertical")

    # storey height from the horizontal surfaces
    horz = ~vert
    zc = cen[horz, 2]
    z0 = float(np.percentile(zc, 2))
    z1 = float(np.percentile(zc, 98))
    log(f"storey {z0:.3f} .. {z1:.3f} m  ({z1-z0:.2f} m)")

    az = np.degrees(np.arctan2(n[vert, 1], n[vert, 0])) % 180.0
    th, frac = grid_angle(az, area[vert])
    log(f"structural grid {th:.2f} deg, holds {100*frac:.1f}% of vertical area "
        f"within {GRID_TOL} deg")
    if ref and Path(ref).exists():
        log(f"  (slice detector said "
            f"{json.load(open(ref)).get('grid_angle_deg')} deg)")

    # ---- classify plan cells in the GRID frame ---------------------------
    iv = np.where(vert)[0]
    pivot = cen[iv, :2].mean(0)
    # a wall's azimuth is normal + 90, so rotate the plan by -(th+90) == -th
    a_ = np.radians(th)
    R = np.array([[np.cos(a_), -np.sin(a_)], [np.sin(a_), np.cos(a_)]])
    Q = (cen[iv, :2] - pivot) @ R          # R.T applied on the right
    lo = Q.min(0) - 0.3
    nx = int((Q[:, 0].max() - lo[0] + 0.3) / CELL) + 1
    ny = int((Q[:, 1].max() - lo[1] + 0.3) / CELL) + 1
    ij = ((Q - lo) / CELL).astype(np.int32)
    np.clip(ij[:, 0], 0, nx - 1, out=ij[:, 0])
    np.clip(ij[:, 1], 0, ny - 1, out=ij[:, 1])
    flat = ij[:, 1] * nx + ij[:, 0]
    zmin = np.full(nx * ny, np.inf)
    zmax = np.full(nx * ny, -np.inf)
    cnt = np.zeros(nx * ny, np.int64)
    np.minimum.at(zmin, flat, cen[iv, 2])
    np.maximum.at(zmax, flat, cen[iv, 2])
    np.add.at(cnt, flat, 1)
    dense = np.isfinite(zmin) & (cnt >= MIN_TRIS_CELL)
    wall_c = dense & (zmax >= z1 - TOUCH) & ((zmax - zmin) >= SPAN)
    para_c = (dense & (zmax < z1 - TOUCH) & (zmin <= z0 + PARA_BASE)
              & (zmax >= z0 + PARA_TOP))
    log(f"plan cells: {int(dense.sum()):,} dense -> {int(wall_c.sum()):,} wall, "
        f"{int(para_c.sum()):,} parapet")

    wm = wall_c.reshape(ny, nx).astype(np.uint8)
    wm = cv2.morphologyEx(wm, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    pm = para_c.reshape(ny, nx).astype(np.uint8)
    pm = cv2.morphologyEx(pm, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # forward was Q = (X - pivot) @ R, i.e. R.T @ (X - pivot), so coming back
    # out is X = R @ Q + pivot. Passing R.T here rotated every run by twice
    # the grid angle and laid the lines across their own evidence.
    walls = runs_from(wm, nx, ny, lo, R, pivot, "wall")
    # A balustrade has no doorways, so it needs almost no bridging -- and a
    # long bridge is exactly what strung isolated furniture tops into 8-10 m
    # "parapets" running through the middle of the flat.
    zg = np.where(np.isfinite(zmax), zmax, np.nan).reshape(ny, nx)
    paras = runs_from(pm, nx, ny, lo, R, pivot, "parapet", bridge=0.35,
                      zmax=zg, thick_max=PARA_THICK,
                      # PARA_TOP_RANGE is a height ABOVE THE FLOOR, and this
                      # floor sits at z=-0.25, so it has to be lifted into
                      # absolute z before it can be compared to one
                      top_range=(z0 + PARA_TOP_RANGE[0], z0 + PARA_TOP_RANGE[1]),
                      top_std_max=PARA_TOP_STD)
    # a parapet lying on top of a detected wall is that wall's lower band
    keep = []
    for p in paras:
        pc = (np.array(p["p0"]) + np.array(p["p1"])) / 2.0
        near = False
        for w in walls:
            a, b = np.array(w["p0"]), np.array(w["p1"])
            d = b - a
            L = np.linalg.norm(d)
            if L < 1e-6:
                continue
            t = np.clip((pc - a) @ d / (L * L), 0, 1)
            if np.linalg.norm(pc - (a + t * d)) < 0.35:
                near = True
                break
        if not near:
            keep.append(p)
    log(f"runs: {len(walls)} walls, {len(paras)} parapet candidates -> "
        f"{len(keep)} kept (rest sit on a wall)")
    allw = walls + keep
    tot = sum(w["length_m"] for w in walls)
    log(f"TOTAL {tot:.1f} m of wall, longest {max(w['length_m'] for w in walls):.2f} m, "
        f"median {np.median([w['length_m'] for w in walls]):.2f} m, "
        f"median thickness {1000*np.median([w['thickness_m'] for w in walls]):.0f} mm")

    js = dict(grid_angle_deg=round(float(th), 3),
              pivot=[round(float(v), 4) for v in pivot],
              z_floor=round(z0, 3), z_ceiling=round(z1, 3), cell_m=CELL,
              source="poisson vertical faces", walls=allw)
    json.dump(js, open(out / "major_walls.json", "w"), indent=1)

    # a picture of what was found, over the evidence it came from
    img = np.full((ny, nx, 3), 255, np.uint8)
    img[dense.reshape(ny, nx)] = (215, 215, 215)
    img[para_c.reshape(ny, nx)] = (150, 190, 220)
    img[wall_c.reshape(ny, nx)] = (60, 60, 60)
    img = cv2.resize(img, (nx * 3, ny * 3), interpolation=cv2.INTER_NEAREST)
    for w in allw:
        a = (np.array(w["p0"]) - pivot) @ R
        b = (np.array(w["p1"]) - pivot) @ R
        pa = tuple((((a - lo) / CELL) * 3).astype(int))
        pb = tuple((((b - lo) / CELL) * 3).astype(int))
        col = (40, 170, 40) if w["kind"] == "wall" else (200, 120, 30)
        cv2.line(img, pa, pb, col, 2, cv2.LINE_AA)
    cv2.imwrite(str(out / "major_walls.png"), cv2.flip(img, 0))
    log(f"wrote {out/'major_walls.json'} + major_walls.png in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(*sys.argv[1:4])
