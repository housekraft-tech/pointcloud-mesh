"""analyse_poisson_openings.py
--------------------------
Can the Poisson mesh be read for OPENINGS directly?

A door is a hole in a wall. So project every vertical face that belongs to a
wall run onto that run's elevation plane (along, height), raster it, and look at
where the wall surface is MISSING. If a door shows up as a clean floor-to-lintel
void about 0.9 x 2.1 m, the mesh can supply openings on its own -- no drawing
detector, no RGB classifier.

The honest failure mode this also measures: a scanner sees a doorway from one
side only, so the reveal can be a shadow rather than a hole, and glazing returns
nothing at all, so a window is indistinguishable from missing data. The report
prints hole sizes so you can judge which is which.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\analyse_poisson_openings.py \\
      <poisson.obj> <major_walls.json> [out_dir]
"""
import sys, json, time
from pathlib import Path
import numpy as np
import cv2

CELL = 0.04         # m, elevation raster
BAND = 0.22         # m, half-thickness of the slab of faces taken per wall
VERT_COS = 0.34
MIN_HOLE_W = 0.55   # m, narrower than this is noise or a pier
MIN_HOLE_H = 0.90   # m
CLOSE_PX = 3


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


def main(obj_path, walls_path, out_dir=None):
    t0 = time.time()
    V, F = load(obj_path)
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(B - A, C - A)
    nrm = np.linalg.norm(cr, axis=1)
    n = cr / np.maximum(nrm[:, None], 1e-12)
    cen = (A + B + C) / 3.0
    vert = np.abs(n[:, 2]) < VERT_COS
    P = cen[vert]
    log(f"{len(P):,} vertical face centroids")

    W = json.load(open(walls_path))
    z0, z1 = W["z_floor"], W["z_ceiling"]
    walls = [s for s in W["walls"] if s.get("kind") == "wall"]
    log(f"{len(walls)} wall runs (parapets excluded), room height "
        f"{z1-z0:.2f} m")

    out = Path(out_dir) if out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    tot_holes, tot_len, door_like, sheet = 0, 0.0, 0, []
    for wi, s in enumerate(walls):
        p0, p1 = np.array(s["p0"]), np.array(s["p1"])
        d = p1 - p0
        L = float(np.hypot(*d))
        if L < 0.6:
            continue
        u = d / L
        v = np.array([-u[1], u[0]])
        rel = P[:, :2] - p0
        a = rel @ u
        o = rel @ v
        m = (np.abs(o) <= BAND) & (a >= -0.05) & (a <= L + 0.05)
        if m.sum() < 200:
            continue
        aa, zz = a[m], P[m, 2]

        na = int(L / CELL) + 1
        nz = int((z1 - z0) / CELL) + 1
        g = np.zeros((nz, na), np.uint8)
        ia = np.clip((aa / CELL).astype(int), 0, na - 1)
        iz = np.clip(((zz - z0) / CELL).astype(int), 0, nz - 1)
        g[iz, ia] = 255
        # a surface-mesh elevation is speckled; close it before reading voids
        g = cv2.morphologyEx(g, cv2.MORPH_CLOSE,
                             np.ones((CLOSE_PX, CLOSE_PX), np.uint8))
        cover = g.mean() / 255.0

        # Voids: every gap in the wall surface, NOT only those capped above.
        # The first version required mesh above and below the hole and found
        # nothing, because a doorway's head is a soffit -- a DOWNWARD-facing
        # face, which the vertical filter throws away. So the void runs clear
        # to the ceiling and "capped above" is never true. Take raw voids and
        # let the shape decide what they are.
        void = (g == 0).astype(np.uint8)
        void = cv2.morphologyEx(void, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        ncc, lab, st, _ = cv2.connectedComponentsWithStats(void, 8)
        hits = []
        for c in range(1, ncc):
            w = st[c, cv2.CC_STAT_WIDTH] * CELL
            h = st[c, cv2.CC_STAT_HEIGHT] * CELL
            zb = z0 + st[c, cv2.CC_STAT_TOP] * CELL
            top = zb + h
            if w < MIN_HOLE_W or h < MIN_HOLE_H:
                continue
            floored = zb - z0 < 0.25
            to_ceil = z1 - top < 0.25
            if floored and to_ceil:
                # spans the whole storey: either a real archway with no
                # spandrel, or the run simply has no wall here
                kind = "full-height gap" if w < 1.6 else "no wall / occluded"
            elif floored:
                kind = "door" if w <= 1.4 else "wide opening"
            elif zb - z0 > 0.45:
                kind = "window"
            else:
                kind = "opening"
            hits.append((w, h, zb - z0, kind))
        tot_holes += len(hits)
        tot_len += L
        door_like += sum(1 for x in hits if x[3] == "door")
        sheet.append(dict(wall=wi, length_m=round(L, 2),
                          coverage=round(float(cover), 3),
                          holes=[dict(w=round(w, 2), h=round(h, 2),
                                      sill=round(sl, 2), kind=k)
                                 for w, h, sl, k in hits]))
        log(f"  wall {wi:02d} {s['axis']} L={L:5.2f} m  surface coverage "
            f"{100*cover:4.1f}%  holes: "
            + (", ".join(f"{k} {w:.2f}x{h:.2f} sill {sl:.2f}"
                         for w, h, sl, k in hits) or "-"))
        if out:
            cv2.imwrite(str(out / f"elev_{wi:02d}.png"), g[::-1])

    log(f"TOTAL {tot_holes} candidate openings over {tot_len:.1f} m of wall "
        f"({door_like} door-shaped)")
    if out:
        json.dump(sheet, open(out / "poisson_openings.json", "w"), indent=1)
        log(f"wrote {out/'poisson_openings.json'} + {len(sheet)} elevations")
    log(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(*sys.argv[1:4])
