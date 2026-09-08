"""continuous_measure.py
-----------------------
Re-measure a reconstructed floorplan in CONTINUOUS metric coordinates instead of
the 20 mm raster the visual pipeline uses. Topology (which wall belongs to which
room) is taken from the existing `skeleton_rooms.obj`; each wall's POSITION,
LENGTH and per-room ceiling/floor heights are then refined directly against the
LiDAR points so we can report ~5-10 mm figures with an honest per-wall tolerance.

Outputs (to <out_dir>):
  measurements.json / walls.csv  -- per-wall refined offset, length, thickness,
                                    point support, tolerance (RMS / sqrt(N))
  rooms.csv                       -- per-room floor/ceiling height + tolerance

This module does NOT touch the render path.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\continuous_measure.py \
      output\\koushik_iso\\isolated.las \
      output2\\koushik_all\\skeleton_3d\\skeleton_rooms.obj \
      output2\\koushik_all\\skeleton_3d\\continuous
"""
import sys, json, time, re, collections
from pathlib import Path
import numpy as np


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------- OBJ loading
def load_obj_walls(obj_path):
    """Return {room: {wall_name: Nx3 vertex array}} for wall objects only."""
    verts = []
    groups = collections.OrderedDict()
    cur = None
    with open(obj_path) as f:
        for ln in f:
            if ln.startswith('v '):
                _, x, y, z = ln.split()[:4]
                verts.append((float(x), float(y), float(z)))
            elif ln.startswith('o '):
                cur = ln[2:].strip()
                groups.setdefault(cur, set())
            elif ln.startswith('f ') and cur is not None:
                for tok in ln.split()[1:]:
                    groups[cur].add(int(tok.split('/')[0]) - 1)
    verts = np.asarray(verts, float)
    walls = collections.defaultdict(dict)
    for g, idxs in groups.items():
        if '_wall_' not in g:
            continue
        room = re.match(r'(room_\d+)', g).group(1)
        walls[room][g] = verts[sorted(idxs)]
    return walls


# ------------------------------------------------------- per-wall refinement
def wall_axes(v_xy):
    """PCA of a thin wall footprint -> (center, dir_unit, normal_unit, length)."""
    c = v_xy.mean(0)
    u, s, vt = np.linalg.svd(v_xy - c, full_matrices=False)
    d = vt[0] / np.linalg.norm(vt[0])          # long axis
    n = np.array([-d[1], d[0]])                # perpendicular
    t = (v_xy - c) @ d
    return c, d, n, float(t.max() - t.min())


def refine_wall(xy, z, v3, band=0.12, zlo=1.0, zhi=2.0, min_pts=150):
    """Fit the wall's plane to nearby points; return refined geometry + tolerance.
    Coordinates are 2D (plan); the wall is vertical so a plan line == the plane."""
    v_xy = v3[:, :2]
    c, d, n, approx_len = wall_axes(v_xy)
    tproj = (v3[:, :2] - c) @ d
    tmin, tmax = tproj.min(), tproj.max()

    rel = xy - c
    perp = rel @ n
    along = rel @ d
    in_span = (along > tmin - 0.05) & (along < tmax + 0.05)
    # primary: tight band, mid height (avoids floor clutter / furniture tops)
    m = (np.abs(perp) < band) & in_span & (z > zlo) & (z < zhi)
    N = int(m.sum())
    conf = "high"
    if N < min_pts:
        # fallback for occluded / short partitions: wider band + full wall height,
        # lower threshold -- flagged low-confidence so the report is honest.
        m = (np.abs(perp) < band + 0.06) & in_span & (z > 0.4) & (z < 2.3)
        N = int(m.sum())
        conf = "low"
        min_pts = 50
    if N < min_pts:
        return None
    P = xy[m]
    # a wall solid is ~110 mm thick, so the band holds BOTH faces + skirting.
    # isolate the single dominant face: histogram the perpendicular offset about
    # the approximate line, keep the tallest 30 mm bin cluster, then fit only that.
    off0 = (P - c) @ n
    edges = np.arange(off0.min() - 0.005, off0.max() + 0.02, 0.01)
    hist, _ = np.histogram(off0, edges)
    pk = edges[int(np.argmax(hist))] + 0.005
    face = np.abs(off0 - pk) < 0.025
    if face.sum() >= min_pts:
        P = P[face]
    # robust line fit: refine normal direction via total-least-squares on the face
    cc = P.mean(0)
    uu, ss, vv = np.linalg.svd(P - cc, full_matrices=False)
    d2 = vv[0] / np.linalg.norm(vv[0])
    n2 = np.array([-d2[1], d2[0]])
    # iterate once, rejecting outliers (furniture against wall, adjacent surfaces)
    for _ in range(2):
        off = (P - cc) @ n2
        s = np.std(off)
        keep = np.abs(off) < 2.5 * s
        if keep.sum() < min_pts:
            break
        P = P[keep]
        cc = P.mean(0)
        uu, ss, vv = np.linalg.svd(P - cc, full_matrices=False)
        d2 = vv[0] / np.linalg.norm(vv[0])
        n2 = np.array([-d2[1], d2[0]])
    off = (P - cc) @ n2
    rms = float(np.sqrt(np.mean(off ** 2)))          # in-plane scatter (mm-ish)
    se = rms / np.sqrt(len(P))                        # offset standard error
    al = (P - cc) @ d2
    length = float(al.max() - al.min())
    # a point ON the refined centerline, and its plan normal (unit)
    center = cc + d2 * ((al.max() + al.min()) / 2)
    return dict(N=N, Nfit=len(P), rms_mm=rms * 1000, se_mm=se * 1000,
                length_m=length, center=center.tolist(), dir=d2.tolist(),
                normal=n2.tolist(), tmin=float(al.min()), tmax=float(al.max()),
                approx_len_m=approx_len, conf=conf)


# ------------------------------------------------------- per-room heights
def room_polygon_bbox(walls_r):
    allv = np.vstack([v[:, :2] for v in walls_r.values()])
    return allv.min(0), allv.max(0)


def room_heights(xy, z, lo, hi, pad=-0.10, min_pts=500):
    """Floor & ceiling plane heights from points inside a room's bbox (padded in)."""
    m = (xy[:, 0] > lo[0] - pad) & (xy[:, 0] < hi[0] + pad) & \
        (xy[:, 1] > lo[1] - pad) & (xy[:, 1] < hi[1] + pad)
    zz = z[m]
    if len(zz) < min_pts:
        return None
    zmin, zmax = zz.min(), zz.max()
    # floor & ceiling are the DOMINANT horizontal surfaces, not the extreme
    # points -- reflections/skirting sit tens of mm past the real plane and
    # would inflate the height if we anchored a band on min/max. Take the
    # histogram peak (mode) within the bottom/top 40 cm, then refine with the
    # points clustered around that peak (+-30 mm).
    edges = np.arange(zmin - 0.01, zmax + 0.02, 0.01)
    hist, _ = np.histogram(zz, edges)
    ce = 0.5 * (edges[:-1] + edges[1:])

    def peak_plane(sel):
        h = np.where(sel, hist, 0)
        if h.max() == 0:
            return None
        zp = ce[int(np.argmax(h))]
        band = zz[np.abs(zz - zp) < 0.03]
        return float(np.median(band)), int(len(band)), float(np.std(band) * 1000)

    fr = peak_plane(ce < zmin + 0.40)
    cr = peak_plane(ce > zmax - 0.40)
    if fr is None or cr is None or fr[1] < min_pts or cr[1] < min_pts:
        return None
    zf, nf, frms = fr
    zc, nc, crms = cr
    return dict(z_floor=zf, z_ceiling=zc,
                height_mm=float((zc - zf) * 1000),
                floor_rms_mm=frms, ceil_rms_mm=crms,
                n_floor=nf, n_ceil=nc)


def main(las_path, obj_path, out_dir):
    import laspy
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    log(f"loading {las_path}")
    las = laspy.read(las_path)
    xy = np.c_[np.asarray(las.x), np.asarray(las.y)].astype(np.float64)
    z = np.asarray(las.z).astype(np.float64)
    log(f"{len(z):,} points")

    walls = load_obj_walls(obj_path)
    log(f"{sum(len(w) for w in walls.values())} walls in {len(walls)} rooms")

    wall_rows = []
    for room in sorted(walls):
        for name, v3 in sorted(walls[room].items()):
            r = refine_wall(xy, z, v3)
            if r is None:
                wall_rows.append(dict(room=room, wall=name, status="low_support"))
                continue
            r.update(room=room, wall=name, status="ok")
            wall_rows.append(r)

    room_rows = []
    for room in sorted(walls):
        lo, hi = room_polygon_bbox(walls[room])
        h = room_heights(xy, z, lo, hi)
        if h:
            h.update(room=room)
            room_rows.append(h)

    # ---- write outputs
    (out / "measurements.json").write_text(json.dumps(
        dict(walls=wall_rows, rooms=room_rows), indent=2))

    import csv
    with open(out / "walls.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["room", "wall", "status", "conf", "length_m", "rms_mm", "se_mm", "N", "Nfit"])
        for r in wall_rows:
            if r["status"] == "ok":
                w.writerow([r["room"], r["wall"], r["status"], r.get("conf", ""),
                            f'{r["length_m"]:.4f}', f'{r["rms_mm"]:.2f}',
                            f'{r["se_mm"]:.3f}', r["N"], r["Nfit"]])
            else:
                w.writerow([r["room"], r["wall"], r["status"], "", "", "", "", "", ""])
    with open(out / "rooms.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["room", "height_mm", "z_floor", "z_ceiling",
                    "floor_rms_mm", "ceil_rms_mm", "n_floor", "n_ceil"])
        for r in room_rows:
            w.writerow([r["room"], f'{r["height_mm"]:.1f}', f'{r["z_floor"]:.4f}',
                        f'{r["z_ceiling"]:.4f}', f'{r["floor_rms_mm"]:.2f}',
                        f'{r["ceil_rms_mm"]:.2f}', r["n_floor"], r["n_ceil"]])

    ok = [r for r in wall_rows if r["status"] == "ok"]
    log(f"refined {len(ok)}/{len(wall_rows)} walls")
    if ok:
        rmss = np.array([r["rms_mm"] for r in ok])
        ses = np.array([r["se_mm"] for r in ok])
        log(f"per-wall plane RMS  : median {np.median(rmss):.1f} mm  p90 {np.percentile(rmss,90):.1f} mm")
        log(f"per-wall offset SE  : median {np.median(ses):.2f} mm  p90 {np.percentile(ses,90):.2f} mm")
    hsumm = ", ".join("{}:{:.0f}".format(r["room"].split("_")[1], r["height_mm"]) for r in room_rows)
    log("room heights (mm): " + hsumm)
    log(f"-> {out}/measurements.json  walls.csv  rooms.csv")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
