"""explain_lidar_to_3d.py
----------------------
Teaching / QC visualizations that show EXACTLY how the pipeline turns raw
LiDAR points into (a) room clear-dimension measurements and (b) wall grooves /
cuts in the 3D model. It reproduces the real algorithms from build_modular_3d
step by step and plots each one, so you can see WHY every number comes out the
way it does -- nothing is hand-waved.

Two deliverables:
  measurement_explained.png  - how a room's clear width/depth is measured
                               directly from the 3D points (full-height wall
                               detection in a centre strip -> face-to-face span)
  groove_explained.png       - how a groove/reveal is found on a wall (per-
                               height face setback from BOTH faces) and cut into
                               the clean wall box at the right height + depth

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\explain_lidar_to_3d.py <isolated.las> <out_dir>
"""
import sys
import time
from pathlib import Path

import numpy as np
import cv2
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.recon import clean, frame
from scripts.recon.io_las import load_scan
from scripts.recon.isolate import select_z_band, isolate_unit
from scripts.isolidarflow import DEFAULT_CONFIG
from scripts.experiments.freespace_floorplan import sensor_trajectory_from_gpstime
from scripts.experiments.hough_vectorize import snap_and_merge
from scripts.experiments.make_floorplan import close_junctions, merge_parallels

# --- exact same constants as build_modular_3d ---
PPM = 50
CELL = 1.0 / PPM
WALL_HALF_BAND = 0.18
GROOVE_MIN_DEPTH = 0.03
MIN_WALL_LEN = 0.5
BIN = 0.05                 # along-axis bin for the measurement strip
STRIP = 0.12               # +-half-width of the measurement centre strip
FULL_H = 1.5               # a bin spanning >this in z is a full-height WALL
OPEN_MARGIN = 0.30


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Stage 0: reproduce the reconstruction front-end (load -> isolate -> align),
# free-space carve, walls, watershed rooms -- identical to build_modular_3d.
# ---------------------------------------------------------------------------
def reconstruct(las_path):
    cfg = dict(DEFAULT_CONFIG)
    seed = int(cfg["seed"])
    scan = load_scan(str(las_path), max_points=cfg["max_points"], rng_seed=seed)
    scan = clean.percentile_crop(scan, lo=cfg["crop_lo_pct"], hi=cfg["crop_hi_pct"], margin_m=cfg["crop_margin_m"])
    if cfg["remove_outliers"]:
        scan = clean.remove_outliers(scan, nb_neighbors=cfg["outlier_nb"], std_ratio=cfg["outlier_std_ratio"])
    traj = (sensor_trajectory_from_gpstime(scan.xyz, scan.gps_time)
            if scan.gps_time is not None else np.zeros((0, 3)))
    z_band = select_z_band(scan.xyz[:, 2], bin_m=cfg["z_bin_m"],
                           min_height_m=cfg["z_min_height_m"], max_height_m=cfg["z_max_height_m"])
    scan, _ = isolate_unit(scan, traj, z_band, cell_m=cfg["iso_cell_m"],
                           max_gap_cells=cfg["iso_max_gap_cells"], max_dist_m=cfg["iso_max_dist_m"])
    rng = np.random.default_rng(seed)
    sub = rng.choice(scan.n, size=min(cfg["normals_max_points"], scan.n), replace=False)
    normals = frame.estimate_normals(scan.xyz[sub], radius=cfg["normals_radius_m"], max_nn=cfg["normals_max_nn"])
    R = frame.dominant_axes(normals)
    scan = frame.axis_align(scan, R)
    xyz = scan.xyz
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    z_floor = float(np.percentile(z, cfg["z_floor_pct"]))
    z_ceiling = float(np.percentile(z, cfg["z_ceiling_pct"]))
    xmin, ymin, xmax, ymax = x.min(), y.min(), x.max(), y.max()
    W = int((xmax - xmin) / CELL) + 1
    H = int((ymax - ymin) / CELL) + 1

    # occupancy (waist band) -> free-space carve from the walk path
    band = (z >= z_floor + 0.9) & (z <= z_floor + 1.6)
    cc = np.clip(((x[band] - xmin) / CELL).astype(int), 0, W - 1)
    rr = np.clip(((ymax - y[band]) / CELL).astype(int), 0, H - 1)
    occ = np.zeros((H, W), np.uint8); occ[rr, cc] = 255
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    empty = (occ == 0).astype(np.uint8)
    free = np.zeros_like(empty)
    if traj.shape[0]:
        sd = np.zeros_like(empty)
        for px, py in traj[:, :2]:
            cxx = int((px - xmin) / CELL); cyy = int((ymax - py) / CELL)
            if 0 <= cyy < H and 0 <= cxx < W and empty[cyy, cxx]:
                sd[cyy, cxx] = 1
        lbl0, _ = ndimage.label(empty, structure=np.ones((3, 3)))
        keep = set(lbl0[sd == 1].tolist()) - {0}
        free = np.isin(lbl0, list(keep)).astype(np.uint8)
    footprint = ndimage.binary_fill_holes((free | (occ > 0)))
    walls_solid = (footprint & (free == 0)).astype(np.uint8)
    walls_solid = ndimage.binary_fill_holes(walls_solid).astype(np.uint8)
    walls_solid = cv2.morphologyEx(walls_solid, cv2.MORPH_CLOSE,
                                   cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    try:
        thin = cv2.ximgproc.thinning(walls_solid)
    except Exception:
        thin = walls_solid
    lines = cv2.HoughLinesP(thin, 1, np.pi / 180, threshold=28,
                            minLineLength=int(0.5 * PPM), maxLineGap=int(0.35 * PPM))
    hsegs, vsegs = ([], [])
    if lines is not None:
        hsegs, vsegs = snap_and_merge(lines.reshape(-1, 4), merge_gap_px=int(0.3 * PPM), coord_tol_px=int(0.12 * PPM))
    hsegs, vsegs = merge_parallels(hsegs), merge_parallels(vsegs)
    hsegs, vsegs = close_junctions(hsegs, vsegs, snap=int(0.5 * PPM))
    hsegs = [s for s in hsegs if s[1] - s[0] >= int(MIN_WALL_LEN * PPM)]
    vsegs = [s for s in vsegs if s[1] - s[0] >= int(MIN_WALL_LEN * PPM)]

    def px2m(col, row):
        return xmin + col * CELL, ymax - row * CELL
    walls = [(np.array(px2m(a0, yr)), np.array(px2m(a1, yr))) for a0, a1, yr in hsegs] + \
            [(np.array(px2m(xc, a0)), np.array(px2m(xc, a1))) for a0, a1, xc in vsegs]

    # watershed rooms + open-span merge (same as production)
    distm = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5) * CELL
    cores = (distm > 0.6).astype(np.uint8)
    ncore, cmark = cv2.connectedComponents(cores)
    mk = np.zeros((H, W), np.int32)
    mk[cores > 0] = cmark[cores > 0] + 1
    mk[(free > 0) & (cores == 0)] = 0
    mk[free == 0] = 1
    cv2.watershed(cv2.merge([free * 200] * 3).astype(np.uint8), mk)
    parent = list(range(ncore + 2))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    ys, xs = np.where(mk == -1)
    from collections import defaultdict
    iface = defaultdict(int)
    for yy, xx in zip(ys.tolist(), xs.tolist()):
        y0, y1 = max(0, yy - 1), min(H, yy + 2); x0, x1 = max(0, xx - 1), min(W, xx + 2)
        labs = sorted(int(v) for v in np.unique(mk[y0:y1, x0:x1]) if v > 1)
        for i in range(len(labs)):
            for j in range(i + 1, len(labs)):
                iface[(labs[i], labs[j])] += 1
    for (a, b), n in iface.items():
        if n * CELL >= 1.6:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    for r in range(2, ncore + 1):
        mk[mk == r] = find(r)
    room_labels = sorted(set(int(v) for v in np.unique(mk) if v > 1))

    return dict(x=x, y=y, z=z, z_floor=z_floor, z_ceiling=z_ceiling,
                xmin=xmin, ymax=ymax, H=H, W=W, free=free, distm=distm,
                walls=walls, mk=mk, room_labels=room_labels, occ=occ)


# ---------------------------------------------------------------------------
# The measurement primitive -- reproduces build_modular_3d.clear_span but also
# RETURNS all intermediate arrays so we can draw exactly what it did.
# ---------------------------------------------------------------------------
def measure_axis(co, zc, along_c, fs_lo, fs_hi):
    """co = along-axis coord of the centre-strip points; zc = their height.
    Returns bin centres, per-bin z-span, wall flags, chosen faces, span, open."""
    b = np.round(co / BIN).astype(int)
    ubins = np.unique(b)
    bin_pos, bin_span, bin_wall = [], [], []
    for bb in ubins:
        mm = b == bb
        span = float(zc[mm].max() - zc[mm].min()) if mm.sum() else 0.0
        bin_pos.append(bb * BIN); bin_span.append(span)
        bin_wall.append(mm.sum() >= 6 and span > FULL_H)
    bin_pos = np.array(bin_pos); bin_span = np.array(bin_span); bin_wall = np.array(bin_wall)
    wp = np.sort(bin_pos[bin_wall])
    lft = wp[wp < along_c]; rgt = wp[wp > along_c]
    if lft.size and lft.max() >= fs_lo - OPEN_MARGIN:
        left, open_l = float(lft.max()), False
    else:
        left, open_l = fs_lo, True
    if rgt.size and rgt.min() <= fs_hi + OPEN_MARGIN:
        right, open_r = float(rgt.min()), False
    else:
        right, open_r = fs_hi, True
    return dict(bin_pos=bin_pos, bin_span=bin_span, bin_wall=bin_wall,
                left=left, right=right, open_l=open_l, open_r=open_r,
                span=right - left)


def pick_enclosed_room(R):
    """Choose a fully enclosed room (both axes bounded by real walls) with a
    clean, moderate size -- the clearest measurement teaching example."""
    x, y, z = R["x"], R["y"], R["z"]
    z_floor, z_ceiling = R["z_floor"], R["z_ceiling"]
    wb = (z >= z_floor + 0.2) & (z <= z_ceiling - 0.1)
    xw, yw, zw = x[wb], y[wb], z[wb]
    best = None
    for Lr in R["room_labels"]:
        m = (R["mk"] == Lr)
        if m.sum() < (1.0 / (CELL * CELL)):
            continue
        rd = R["distm"] * m
        cy, cx = np.unravel_index(int(np.argmax(rd)), rd.shape)
        rcx = R["xmin"] + cx * CELL; rcy = R["ymax"] - cy * CELL
        # free-space extents through centre
        row = m[cy, :]; c0 = cx; c1 = cx
        while c0 > 0 and row[c0 - 1]: c0 -= 1
        while c1 < R["W"] - 1 and row[c1 + 1]: c1 += 1
        col = m[:, cx]; r0 = cy; r1 = cy
        while r0 > 0 and col[r0 - 1]: r0 -= 1
        while r1 < R["H"] - 1 and col[r1 + 1]: r1 += 1
        fsx = (R["xmin"] + c0 * CELL, R["xmin"] + c1 * CELL)
        fsy = (R["ymax"] - r1 * CELL, R["ymax"] - r0 * CELL)
        mx = np.abs(yw - rcy) <= STRIP
        my = np.abs(xw - rcx) <= STRIP
        if mx.sum() < 30 or my.sum() < 30:
            continue
        rw = measure_axis(xw[mx], zw[mx], rcx, *fsx)
        rh = measure_axis(yw[my], zw[my], rcy, *fsy)
        area = rw["span"] * rh["span"]
        enclosed = not (rw["open_l"] or rw["open_r"] or rh["open_l"] or rh["open_r"])
        if enclosed and 4.0 < area < 18.0:
            cand = (rcx, rcy, fsx, fsy, rw, rh, xw[mx], zw[mx], yw[my], zw[my], area)
            if best is None or area < best[-1]:   # smallest enclosed = crispest example
                best = cand
    return best


# ---------------------------------------------------------------------------
def plot_measurement(R, out_dir):
    pick = pick_enclosed_room(R)
    if pick is None:
        log("no clean enclosed room found for the measurement demo"); return
    rcx, rcy, fsx, fsy, rw, rh, sx, sxz, sy, syz, _ = pick
    z_floor, z_ceiling = R["z_floor"], R["z_ceiling"]

    fig = plt.figure(figsize=(15, 10), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.28, wspace=0.2)

    # --- panel 1: plan with the room + the two measurement strips ---
    ax = fig.add_subplot(gs[0, :])
    sub = np.random.default_rng(0).choice(R["x"].size, size=min(120000, R["x"].size), replace=False)
    ax.scatter(R["x"][sub], R["y"][sub], s=0.2, c="0.75", linewidths=0)
    ax.axhspan(rcy - STRIP, rcy + STRIP, xmin=0, xmax=1, color="tab:red", alpha=0.12)
    ax.axvspan(rcx - STRIP, rcx + STRIP, ymin=0, ymax=1, color="tab:blue", alpha=0.12)
    ax.plot([fsx[0], fsx[1]], [rcy, rcy], "-", color="tab:red", lw=1)
    ax.plot([rcx, rcx], [fsy[0], fsy[1]], "-", color="tab:blue", lw=1)
    ax.plot(rcx, rcy, "k+", ms=14, mew=2)
    ax.set_title("STEP 1  Pick the room centre, take a thin +-12cm strip across it on each axis\n"
                 "(red = width strip, blue = depth strip)  -  points are top-down LiDAR",
                 fontsize=11)
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    # --- panel 2 + 3: elevations of each strip with wall bins highlighted ---
    for ax_i, (co, zc, res, along, axis_name, colr) in enumerate([
            (sx, sxz, rw, rcx, "WIDTH (x)", "tab:red"),
            (sy, syz, rh, rcy, "DEPTH (y)", "tab:blue")]):
        ax = fig.add_subplot(gs[1, ax_i])
        b = np.round(co / BIN).astype(int)
        wallbins = set(np.round(res["bin_pos"][res["bin_wall"]] / BIN).astype(int).tolist())
        is_wall = np.array([bb in wallbins for bb in b])
        ax.scatter(co[~is_wall], zc[~is_wall], s=1.5, c="0.6", linewidths=0, label="furniture / partial (<1.5m tall)")
        ax.scatter(co[is_wall], zc[is_wall], s=1.5, c=colr, linewidths=0, label="full-height WALL bin (>1.5m tall)")
        ax.axhline(z_floor, color="k", ls=":", lw=1); ax.axhline(z_ceiling, color="k", ls=":", lw=1)
        ax.axvline(res["left"], color="k", ls="--", lw=1.5)
        ax.axvline(res["right"], color="k", ls="--", lw=1.5)
        ax.annotate("", xy=(res["right"], z_floor + 0.3), xytext=(res["left"], z_floor + 0.3),
                    arrowprops=dict(arrowstyle="<->", color="k", lw=1.5))
        ax.text((res["left"] + res["right"]) / 2, z_floor + 0.45,
                f"{res['span']*1000:.0f} mm", ha="center", fontsize=12, fontweight="bold",
                bbox=dict(boxstyle="round", fc="white", ec=colr))
        tag = "  (open side capped at partition)" if (res["open_l"] or res["open_r"]) else ""
        ax.set_title(f"STEP 2  {axis_name} elevation: a bin is a WALL only if its points span "
                     f">1.5m floor->ceiling\nclear span = nearest wall face L->R{tag}", fontsize=10)
        ax.set_xlabel(f"{axis_name[0].lower()} position (m)"); ax.set_ylabel("height z (m)")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.suptitle("HOW A ROOM IS MEASURED  -  directly from 3D LiDAR points (no 2D raster)",
                 fontsize=14, fontweight="bold")
    p = Path(out_dir) / "measurement_explained.png"
    fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
    log(f"wrote {p}  (room {rw['span']*1000:.0f} x {rh['span']*1000:.0f} mm)")


# ---------------------------------------------------------------------------
# Groove extraction -- reproduces the build_modular_3d per-band face-setback
# method, for the wall that shows the strongest grooves.
# ---------------------------------------------------------------------------
def wall_setback(R, p0, p1):
    x, y, z = R["x"], R["y"], R["z"]
    z_floor, z_ceiling = R["z_floor"], R["z_ceiling"]
    d = p1 - p0; L = float(np.linalg.norm(d))
    if L < MIN_WALL_LEN:
        return None
    d = d / L; n = np.array([-d[1], d[0]])
    rel = np.column_stack([x, y]) - p0
    u = rel @ d; perp = rel @ n
    near = (np.abs(perp) <= WALL_HALF_BAND) & (u >= 0) & (u <= L)
    if near.sum() < 200:
        return None
    uu, zz, pp = u[near], z[near], perp[near]
    thick = float(np.clip(np.percentile(pp, 92) - np.percentile(pp, 8), 0.06, 0.35))
    sides = {}
    for side in (+1.0, -1.0):
        sel = (pp * side) > 0
        if sel.sum() < 300:
            continue
        sp = pp[sel] * side; sz = zz[sel]
        face = np.percentile(sp, 82)
        zb = np.arange(z_floor + 0.12, z_ceiling - 0.12, 0.1)
        setb = np.full(len(zb), np.nan)
        for i, zl in enumerate(zb):
            mm = (sz >= zl) & (sz < zl + 0.1)
            if mm.sum() >= 25:
                setb[i] = face - np.percentile(sp[mm], 82)
        sides[side] = dict(uu=uu[sel] if False else u[near][ (pp*side)>0 ], sz=sz, sp=sp,
                           face=face, zb=zb, setb=setb)
    return dict(L=L, thick=thick, uu=uu, zz=zz, pp=pp, sides=sides)


def plot_groove(R, out_dir):
    # choose the wall with the most groove bands
    best = None
    for (p0, p1) in R["walls"]:
        ws = wall_setback(R, p0, p1)
        if ws is None:
            continue
        ng = 0
        for s in ws["sides"].values():
            ng += int(np.nansum(s["setb"] >= GROOVE_MIN_DEPTH))
        if best is None or ng > best[0]:
            best = (ng, p0, p1, ws)
    if best is None or best[0] == 0:
        log("no grooved wall found for the groove demo"); return
    _, p0, p1, ws = best
    z_floor, z_ceiling = R["z_floor"], R["z_ceiling"]
    side = max(ws["sides"], key=lambda s: np.nansum(ws["sides"][s]["setb"] >= GROOVE_MIN_DEPTH))
    S = ws["sides"][side]

    fig = plt.figure(figsize=(15, 10), facecolor="white")
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # panel 1: plan context - the wall + capture band
    ax = fig.add_subplot(gs[0, 0])
    sub = np.random.default_rng(1).choice(R["x"].size, size=min(80000, R["x"].size), replace=False)
    ax.scatter(R["x"][sub], R["y"][sub], s=0.2, c="0.8", linewidths=0)
    d = (p1 - p0); d = d / np.linalg.norm(d); nrm = np.array([-d[1], d[0]])
    for k in (+1, -1):
        a = p0 + nrm * k * WALL_HALF_BAND; b = p1 + nrm * k * WALL_HALF_BAND
        ax.plot([a[0], b[0]], [a[1], b[1]], "-", color="tab:orange", lw=1)
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "-", color="tab:red", lw=2)
    ax.set_title("STEP 1  One wall + its +-18cm capture band\n(all points near the wall line)", fontsize=10)
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    # panel 2: along-wall elevation coloured by outward distance (recess)
    ax = fig.add_subplot(gs[0, 1])
    uu_side = ws["uu"][(ws["pp"] * side) > 0]
    sc = ax.scatter(uu_side, S["sz"], c=S["sp"], s=2, cmap="viridis", linewidths=0)
    ax.axhline(z_floor, color="k", ls=":", lw=1); ax.axhline(z_ceiling, color="k", ls=":", lw=1)
    ax.set_title("STEP 2  This face's elevation (along-wall u vs height z),\n"
                 "colour = outward distance of each point (bright = sticks out)", fontsize=10)
    ax.set_xlabel("u along wall (m)"); ax.set_ylabel("height z (m)")
    fig.colorbar(sc, ax=ax, label="outward dist (m)", shrink=0.8)

    # panel 3: per-height face setback profile + threshold
    ax = fig.add_subplot(gs[1, 0])
    zb, setb = S["zb"], S["setb"]
    ax.plot(np.nan_to_num(setb) * 1000, zb, "-o", ms=3, color="tab:purple")
    ax.axvline(GROOVE_MIN_DEPTH * 1000, color="tab:red", ls="--", lw=1.5,
               label=f"groove threshold {GROOVE_MIN_DEPTH*1000:.0f} mm")
    active = setb >= GROOVE_MIN_DEPTH
    for i in range(len(zb)):
        if active[i]:
            ax.axhspan(zb[i], zb[i] + 0.1, color="tab:red", alpha=0.18)
    ax.set_title("STEP 3  How far the face sets back per 10cm height band\n"
                 "(bands past the threshold = a groove/reveal, shaded red)", fontsize=10)
    ax.set_xlabel("face setback (mm)"); ax.set_ylabel("height z (m)"); ax.legend(fontsize=8)

    # panel 4: resulting wall cross-section with grooves cut
    ax = fig.add_subplot(gs[1, 1])
    thick = ws["thick"]
    ax.add_patch(Rectangle((0, z_floor), thick * 1000, z_ceiling - z_floor,
                           fc="0.85", ec="k"))
    i = 0
    while i < len(active):
        if active[i]:
            j = i
            while j < len(active) and active[j]:
                j += 1
            z0 = zb[i]; z1 = zb[j - 1] + 0.1
            depth = float(np.clip(np.nanmedian(setb[i:j]), 0.03, thick * 0.8))
            # groove cut into this face (face at x=0 side)
            ax.add_patch(Rectangle((0, z0), depth * 1000, z1 - z0, fc="white", ec="tab:red", hatch="///"))
            i = j
        else:
            i += 1
    ax.set_xlim(-0.02 * 1000, thick * 1000 * 1.2); ax.set_ylim(z_floor - 0.1, z_ceiling + 0.1)
    ax.set_title(f"STEP 4  The clean wall box (thickness {thick*1000:.0f} mm) with each groove\n"
                 "cut to the measured depth at its measured height", fontsize=10)
    ax.set_xlabel("wall thickness (mm)"); ax.set_ylabel("height z (m)")

    fig.suptitle("HOW A GROOVE / CUT IS EXTRACTED  -  per-height face setback, cut into the wall",
                 fontsize=14, fontweight="bold")
    p = Path(out_dir) / "groove_explained.png"
    fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
    log(f"wrote {p}  ({best[0]} groove bands on the chosen wall)")


def main(las_path, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log(f"reconstructing from {las_path} ...")
    R = reconstruct(las_path)
    log(f"reconstructed: {len(R['walls'])} walls, {len(R['room_labels'])} rooms")
    plot_measurement(R, out_dir)
    plot_groove(R, out_dir)
    log("done")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
