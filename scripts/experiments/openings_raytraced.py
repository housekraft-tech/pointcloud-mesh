"""openings_raytraced.py
-------------------
Tell a real opening from an occlusion shadow, using the scanner's own path.

Both look identical in the mesh: a hole in a wall. Previtali, Diaz-Vilarino and
Scaioni (Appl. Sci. 2018, 8(9):1529) give the test that separates them --

    a cell is an OCCLUSION only if it was occluded from EVERY scan position;
    if it is empty and was visible from at least one position, it is a real
    opening, because the sensor looked straight through it and saw nothing.

Our earlier void scan found 6 clean doors and 13 "full-height gaps" it could
not adjudicate. This decides them. We have the trajectory the method needs
(entrance_path.json), so no extra capture is required.

Method: voxelise the mesh once at VOX, then for each void blob in a wall's
elevation, march rays from a spread of sensor poses to sample points inside the
blob. A blob whose samples were widely visible is an opening; one that was
never seen through is a shadow behind furniture.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\openings_raytraced.py \\
      <poisson.obj> <major_walls.json> <entrance_path.json> <out_dir>
"""
import sys, json, time
from pathlib import Path
import numpy as np
import cv2

CELL = 0.04        # m, elevation raster (matches the earlier void scan)
BAND = 0.22        # m, half-slab of faces taken per wall
VERT_COS = 0.34
MIN_W, MIN_H = 0.55, 0.90    # m, smaller voids are noise or piers
DOOR_MIN_HEAD = 1.70         # m, a floor-anchored void shorter than this is
                             # not something you walk through
VOX = 0.06         # m, occupancy voxel for ray marching
NEAR_SKIP = 0.25   # m, ignore occupancy this close to the sensor
FAR_SKIP = 0.20    # m, ...and this close to the target, so the wall's own
                   #    jamb and reveal do not block their own opening
MAX_POSES = 160    # poses sampled along the walk
MAX_SAMPLES = 40   # sample points per void blob
SEEN_FRAC = 0.25   # fraction of samples that must be visible from some pose
WALL_MIN_COVER = 0.25  # a run whose elevation is emptier than this is not a wall
VOID_MAX_SHARE = 0.45  # a void filling more of the elevation than this is not
                       # an opening in a wall -- it IS the missing wall


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
    V = np.asarray(V, np.float64)
    # Same convention check as walls_from_poisson: a storey is the shortest
    # axis of a flat, so exports that are Y-up rather than Z-up get rotated
    # here too, or the wall runs and the mesh sit in different frames.
    ext = V.max(0) - V.min(0)
    up = int(np.argmin(ext))
    if up != 2:
        log(f"vertical axis is {'xyz'[up]}, not z -- rotating to Z-up")
        V = V[:, [a for a in (0, 1, 2) if a != up] + [up]]
    return V, np.asarray(F, np.int64)


def visible(targets, poses, occ, org, dims):
    """For each target, is it visible from ANY pose? Vectorised ray marching.

    Marches all (target, pose) pairs together in fixed steps and asks whether
    any sampled voxel between them is occupied, skipping a margin at each end.
    """
    seen = np.zeros(len(targets), bool)
    nz, ny_, nx_ = dims
    for p in poses:
        todo = np.where(~seen)[0]
        if len(todo) == 0:
            break
        T = targets[todo]
        d = T - p
        L = np.linalg.norm(d, axis=1)
        ok = L > (NEAR_SKIP + FAR_SKIP)
        if not ok.any():
            continue
        todo, T, d, L = todo[ok], T[ok], d[ok], L[ok]
        u = d / L[:, None]
        nstep = int(np.ceil(float(L.max()) / (VOX * 0.7)))
        blocked = np.zeros(len(T), bool)
        for s in range(nstep):
            t = NEAR_SKIP + s * VOX * 0.7
            live = ~blocked & (t < (L - FAR_SKIP))
            if not live.any():
                break
            q = p + u[live] * t
            ijk = ((q - org) / VOX).astype(np.int32)
            good = ((ijk[:, 0] >= 0) & (ijk[:, 0] < nx_) &
                    (ijk[:, 1] >= 0) & (ijk[:, 1] < ny_) &
                    (ijk[:, 2] >= 0) & (ijk[:, 2] < nz))
            hit = np.zeros(len(ijk), bool)
            g = np.where(good)[0]
            if len(g):
                hit[g] = occ[ijk[g, 2], ijk[g, 1], ijk[g, 0]]
            idx = np.where(live)[0][hit]
            blocked[idx] = True
        seen[todo[~blocked]] = True
    return seen


def main(obj_path, walls_path, path_json, out_dir):
    t0 = time.time()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    V, F = load(obj_path)
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(B - A, C - A)
    n = cr / np.maximum(np.linalg.norm(cr, axis=1)[:, None], 1e-12)
    cen = (A + B + C) / 3.0
    P3 = cen[np.abs(n[:, 2]) < VERT_COS]
    log(f"{len(P3):,} vertical faces")

    # ---- occupancy grid --------------------------------------------------
    org = V.min(0) - VOX
    dim = np.ceil((V.max(0) + VOX - org) / VOX).astype(int) + 1
    occ = np.zeros((dim[2], dim[1], dim[0]), bool)
    ijk = ((V - org) / VOX).astype(np.int32)
    occ[ijk[:, 2], ijk[:, 1], ijk[:, 0]] = True
    log(f"occupancy {dim[0]}x{dim[1]}x{dim[2]} at {VOX*100:.0f} cm, "
        f"{occ.sum():,} occupied ({100*occ.mean():.1f}%)")

    walk = np.asarray(json.load(open(path_json))["path"], float)
    step = max(1, len(walk) // MAX_POSES)
    poses = walk[::step]
    log(f"{len(walk):,} poses -> {len(poses)} sampled for ray casting")

    W = json.load(open(walls_path))
    z0, z1 = W["z_floor"], W["z_ceiling"]
    walls = [s for s in W["walls"] if s.get("kind") == "wall"
             and s.get("length_m", 0) >= 0.6]

    rows, n_open, n_shadow, elevs = [], 0, 0, []
    for wi, s in enumerate(walls):
        p0, p1 = np.array(s["p0"], float), np.array(s["p1"], float)
        d = p1 - p0
        L = float(np.hypot(*d))
        u = d / L
        v = np.array([-u[1], u[0]])
        rel = P3[:, :2] - p0
        a = rel @ u
        o = rel @ v
        m = (np.abs(o) <= BAND) & (a >= -0.05) & (a <= L + 0.05)
        if m.sum() < 200:
            continue
        na, nz_ = int(L / CELL) + 1, int((z1 - z0) / CELL) + 1
        g = np.zeros((nz_, na), np.uint8)
        g[np.clip(((P3[m, 2] - z0) / CELL).astype(int), 0, nz_ - 1),
          np.clip((a[m] / CELL).astype(int), 0, na - 1)] = 255
        g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        void = cv2.morphologyEx((g == 0).astype(np.uint8), cv2.MORPH_OPEN,
                                np.ones((3, 3), np.uint8))
        cover = float((g > 0).mean())
        hits = []
        ncc, lab, st, _ = cv2.connectedComponentsWithStats(void, 8)
        for c in range(1, ncc):
            w = st[c, cv2.CC_STAT_WIDTH] * CELL
            h = st[c, cv2.CC_STAT_HEIGHT] * CELL
            sill = z0 + st[c, cv2.CC_STAT_TOP] * CELL
            if w < MIN_W or h < MIN_H:
                continue
            zi, ai = np.where(lab == c)
            if len(zi) > MAX_SAMPLES:
                pick = np.linspace(0, len(zi) - 1, MAX_SAMPLES).astype(int)
                zi, ai = zi[pick], ai[pick]
            xy = p0[None, :] + (ai * CELL)[:, None] * u[None, :]
            T = np.c_[xy, z0 + zi * CELL]
            seen = visible(T, poses, occ, org, (dim[2], dim[1], dim[0]))
            frac = float(seen.mean())
            share = st[c, cv2.CC_STAT_AREA] / float(void.size)
            head = sill - z0 + h
            # Two guards the raw visibility test cannot supply on its own.
            # A run whose elevation is nearly all void has no wall to put a
            # door in -- calling that void a door is meaningless, and it is
            # how walls 25-27 (4-8% surface coverage) each scored a "door".
            if cover < WALL_MIN_COVER or share > VOID_MAX_SHARE:
                kind, real = "no wall (run is not a wall)", False
            elif frac < SEEN_FRAC:
                kind, real = "occlusion shadow", False
            elif sill - z0 >= 0.25:
                kind, real = "window", True
            elif head < DOOR_MIN_HEAD:
                # floor-anchored but far too short to walk through: a recess,
                # a skirting shadow, or the gap behind a low unit. Better
                # runs surfaced several of these as 0.96-1.08 m "doors".
                kind, real = "low void (not an opening)", False
            elif z1 - (z0 + head) < 0.15:
                # nothing above it: no lintel was captured, so this is an
                # archway / cased opening, not a door in a wall
                kind, real = ("archway" if w <= 1.4 else "wide opening"), True
            else:
                kind, real = ("door" if w <= 1.4 else "wide opening"), True
            n_open, n_shadow = (n_open + 1, n_shadow) if real else (n_open, n_shadow + 1)
            rows.append(dict(wall=wi, axis=s["axis"], kind=kind,
                             width_m=round(w, 2), height_m=round(h, 2),
                             sill_m=round(sill - z0, 2), head_m=round(head, 2),
                             wall_coverage=round(cover, 2),
                             void_share=round(float(share), 2),
                             seen_frac=round(frac, 2), real=bool(real)))
            hits.append((st[c, cv2.CC_STAT_LEFT],
                         nz_ - st[c, cv2.CC_STAT_TOP] - st[c, cv2.CC_STAT_HEIGHT],
                         st[c, cv2.CC_STAT_WIDTH], st[c, cv2.CC_STAT_HEIGHT],
                         kind,
                         (f"{kind.split(' (')[0]} {w:.2f}x{h:.2f}"
                          + (f" sill {sill-z0:.2f}" if sill - z0 >= 0.25 else ""))))
            log(f"  wall {wi:02d}  {w:4.2f} x {h:4.2f} sill {sill-z0:4.2f} "
                f"head {head:4.2f}  seen {100*frac:3.0f}%  cover {100*cover:3.0f}%"
                f"  -> {kind}")
        elevs.append((wi, g, cover, hits))

    log(f"VERDICT: {n_open} real openings, {n_shadow} occlusion shadows")
    for k in ("door", "archway", "wide opening", "window"):
        v = [r for r in rows if r["kind"] == k]
        if v:
            log(f"  {k}: {len(v)}  widths "
                f"{min(r['width_m'] for r in v):.2f}-{max(r['width_m'] for r in v):.2f} m, "
                f"heads {min(r['height_m']+r['sill_m'] for r in v):.2f}-"
                f"{max(r['height_m']+r['sill_m'] for r in v):.2f} m")
    json.dump(dict(openings=rows, seen_frac_threshold=SEEN_FRAC,
                   n_poses=len(poses)),
              open(out / "openings_raytraced.json", "w"), indent=1)
    draw(elevs, out / "openings_elevations.png")
    log(f"wrote {out/'openings_raytraced.json'} in {time.time()-t0:.0f}s")


KIND_BGR = {"door": (60, 190, 60), "archway": (230, 150, 40),
            "low void (not an opening)": (60, 60, 220),
            "wide opening": (200, 90, 200), "window": (220, 200, 40),
            "occlusion shadow": (60, 60, 220),
            "no wall (run is not a wall)": (60, 60, 220)}


def draw(elevs, path, scale=5, width=1900):
    """One labelled elevation per wall: what was found and what was rejected."""
    tiles = []
    for wi, g, cover, hits in elevs:
        # drafting convention: wall solid, opening blank. g is 255 where the
        # scan found surface, so invert it -- black is fabric, white is a void.
        img = cv2.cvtColor(255 - g, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, (g.shape[1] * scale, g.shape[0] * scale),
                         interpolation=cv2.INTER_NEAREST)
        H = img.shape[0]
        for (x, y, w, h, kind, txt) in hits:
            col = KIND_BGR.get(kind, (120, 120, 120))
            # y is measured from the floor, the image from the top
            y0 = H - (y + h) * scale
            cv2.rectangle(img, (x * scale, int(y0)),
                          ((x + w) * scale, int(y0) + h * scale), col, 2)
            cv2.putText(img, txt, (x * scale + 3, int(y0) + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
        img = cv2.copyMakeBorder(img, 26, 6, 6, 6, cv2.BORDER_CONSTANT,
                                 value=(255, 255, 255))
        cv2.putText(img, f"wall {wi:02d}   {100*cover:.0f}% surface",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1,
                    cv2.LINE_AA)
        tiles.append(cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT,
                                        value=(190, 190, 190)))
    rows_, cur, cw = [], [], 0
    for t in tiles:
        if cur and cw + t.shape[1] > width:
            rows_.append(cur); cur, cw = [], 0
        cur.append(t); cw += t.shape[1]
    if cur:
        rows_.append(cur)
    bands = []
    for r in rows_:
        h = max(t.shape[0] for t in r)
        r = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0,
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))
             for t in r]
        bands.append(np.hstack(r))
    w = max(b.shape[1] for b in bands)
    bands = [cv2.copyMakeBorder(b, 0, 0, 0, w - b.shape[1],
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))
             for b in bands]
    sheet = np.vstack(bands)
    key = np.full((34, w, 3), 255, np.uint8)
    x = 10
    for k, col in (("door", KIND_BGR["door"]), ("archway", KIND_BGR["archway"]),
                   ("wide opening", KIND_BGR["wide opening"]),
                   ("window", KIND_BGR["window"]),
                   ("rejected", KIND_BGR["occlusion shadow"])):
        cv2.rectangle(key, (x, 10), (x + 22, 26), col, -1)
        cv2.putText(key, k, (x + 28, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (30, 30, 30), 1, cv2.LINE_AA)
        x += 40 + 11 * len(k)
    cv2.imwrite(str(path), np.vstack([key, sheet]))
    log(f"wrote {path.name}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
