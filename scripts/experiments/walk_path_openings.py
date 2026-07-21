"""walk_path_openings.py
----------------------
Use the operator's WALK PATH to decide which holes in a wall are real openings.

A hole in a wall is ambiguous from geometry alone: a doorway and a patch the
scanner never saw behind a wardrobe look identical. The walk path resolves it,
because the operator could not walk through a wall:

  crossing + hole   -> CONFIRMED opening (they walked through it)
  crossing, no hole -> MISSED opening (the hole detector failed here)
  hole, no crossing -> UNWALKED: a window, a balcony door they did not use, or
                       a false positive. Not proof either way -- reported as
                       unverified rather than silently dropped.

A crossing only counts where wall material actually stands (see column_state):
a merged plane is INFINITE, so the path crosses its line in open space too.

The path is the native-gps_time median sensor pose. measurements.json and the
modular OBJ both use raw LAS coordinates, so it needs no transform to line up.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\walk_path_openings.py \\
      <scan.las> <detailed_modular.obj> <measurements.json> <out_dir>
"""
import sys, json, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.wall_elevations import (
    parse_obj, merge_coplanar, grids, find_openings, CELL, DOOR_STD, DOOR_TOL)

MAX_STEP   = 1.20    # m: ignore a sign change across a jump this long (SLAM skip)
MATCH_TOL  = 0.60    # m: crossing within this of a hole's span = same opening
CLUSTER    = 0.50    # m: crossings this close are the same doorway walked twice
EDGE_MARG  = 0.25    # m: crossing must be this far inside the wall's extent
MAX_OPEN_W = 3.00    # m: a clear span wider than this is a gap BETWEEN wall
                     #    runs, not an opening in a wall


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def walk_path(las_path):
    """Recover the sensor path. Uses the native-gps_time MEDIAN pose, not
    approx_trajectory's time-binned centroid: the centroid is pulled 1-2 m
    toward whichever wall was open, which drags the path through walls and
    fabricates crossings. The median resists that."""
    import laspy
    las = laspy.read(las_path)
    xyz = np.c_[np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)].astype(float)
    if not hasattr(las, "gps_time"):
        raise SystemExit("LAS has no gps_time -- cannot recover the walk path")
    t = np.asarray(las.gps_time, dtype=float)
    from scripts.experiments.freespace_floorplan import sensor_trajectory_from_gpstime
    path = sensor_trajectory_from_gpstime(xyz, t)
    log(f"walk path: {len(path):,} sensor poses from {len(t):,} points over "
        f"{t.max()-t.min():.0f} s")
    return path


def plane_of(P):
    """Centroid, normal and along-direction of a merged wall plane (2D)."""
    c = P[:, :2].mean(0)
    Q = P[:, :2] - c
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    n = Vt[1] / np.linalg.norm(Vt[1])
    d = np.array([-n[1], n[0]])
    return c, n, d


def crossings(path, c, n, d, amin, amax):
    """Where the path passes through this wall plane, inside the wall's extent."""
    rel = path[:, :2] - c
    s = rel @ n
    a = rel @ d
    out = []
    for i in range(len(path) - 1):
        if s[i] == 0 or np.sign(s[i]) == np.sign(s[i + 1]):
            continue
        step = np.linalg.norm(path[i + 1, :2] - path[i, :2])
        if step > MAX_STEP:
            continue                      # SLAM jump, not a real walk-through
        f = s[i] / (s[i] - s[i + 1])      # fraction along the segment
        ac = a[i] + f * (a[i + 1] - a[i])
        if not (amin + EDGE_MARG < ac < amax - EDGE_MARG):
            continue
        zc = path[i, 2] + f * (path[i + 1, 2] - path[i, 2])
        out.append((float(ac), float(zc)))
    # a doorway walked several times gives several crossings at the same place
    out.sort()
    merged = []
    for ac, zc in out:
        if merged and ac - merged[-1][0] < CLUSTER:
            merged[-1][2] += 1
            continue
        merged.append([ac, zc, 1])
    return merged


def column_state(occ, a0, z0, ac, z_floor, z_ceil):
    """What is the wall doing at this along-position?

    A merged plane is one INFINITE plane, so the path crosses its line in
    places where no wall physically stands. Without this test every such
    crossing is reported as a missed opening. Look at the occupancy column:

      'no wall'  nothing there -- the plane is notional here, ignore
      'opening'  material above, empty at walking height -> a real doorway
      'solid'    material all the way down -- the path cannot have gone
                 through, so this is path error, not an opening
    """
    nz, na = occ.shape
    c = int(round((ac - a0) / CELL))
    lo, hi = max(0, c - 2), min(na, c + 3)
    if lo >= hi:
        return "no wall", 0.0, 0.0
    col = occ[:, lo:hi].any(axis=1)
    zs = z0 + (np.arange(nz) + 0.5) * CELL
    high = (zs > z_floor + 2.20) & (zs < z_ceil - 0.05)
    walk = (zs > z_floor + 0.30) & (zs < z_floor + 1.60)
    f_high = float(col[high].mean()) if high.any() else 0.0
    f_walk = float(col[walk].mean()) if walk.any() else 0.0
    if f_high < 0.20 and f_walk < 0.20:
        return "no wall", f_high, f_walk
    if f_walk < 0.35:
        return "opening", f_high, f_walk
    return "solid", f_high, f_walk


def measure_at(occ, a0, z0, ac, z_floor, z_ceil):
    """Measure an opening the detector missed, using the crossing as a seed.

    Blind hole detection needs a fully enclosed void, which fails whenever the
    reveal is ragged or the Poisson bridged part of the head. Here we already
    KNOW an opening is at `ac` because someone walked through it, so we do not
    have to find it -- only measure it. Grow left and right from the seed while
    the column stays clear at walking height, then read the head off the lintel
    underside above that span.

    Returns None if the seed column is not actually clear (bad crossing).
    """
    nz, na = occ.shape
    zs = z0 + (np.arange(nz) + 0.5) * CELL
    walk = (zs > z_floor + 0.30) & (zs < z_floor + 1.60)
    if not walk.any():
        return None
    clear = occ[walk, :].mean(axis=0) < 0.35          # per-column, clear at waist

    c0 = int(round((ac - a0) / CELL))
    if not (0 <= c0 < na):
        return None
    if not clear[c0]:                                  # nudge onto the void
        near = [c for c in range(max(0, c0 - 6), min(na, c0 + 7)) if clear[c]]
        if not near:
            return None
        c0 = min(near, key=lambda c: abs(c - c0))
    cl = c0
    while cl - 1 >= 0 and clear[cl - 1]:
        cl -= 1
    cr = c0
    while cr + 1 < na and clear[cr + 1]:
        cr += 1
    width = (cr - cl + 1) * CELL
    if width > MAX_OPEN_W:
        # the clear span ran straight through a gap BETWEEN two wall runs on
        # this plane -- that is open space, not an opening in a wall
        return None

    # head = underside of the lintel: lowest occupied row above the walk band,
    # taken per column and medianed so one ragged column cannot set the height
    above = np.flatnonzero(zs > z_floor + 1.60)
    heads = []
    for c in range(cl, cr + 1):
        rows = above[occ[above, c]]
        if rows.size:
            heads.append(zs[rows.min()])
    open_to_ceiling = len(heads) < 0.3 * (cr - cl + 1)
    head = float(np.median(heads)) - z_floor if heads else float(z_ceil - z_floor)

    below = np.flatnonzero(zs < z_floor + 0.30)
    sills = []
    for c in range(cl, cr + 1):
        rows = below[occ[below, c]]
        if rows.size:
            sills.append(zs[rows.max()])
    sill = float(np.median(sills)) - z_floor if sills else 0.0

    if open_to_ceiling:
        kind = "archway / open passage"
    elif abs(head - DOOR_STD) <= DOOR_TOL:
        kind = "balcony / sliding door" if width >= 1.30 else "door"
    elif width >= 1.30:
        kind = "wide opening (non-standard head)"
    else:
        kind = "opening (non-standard head)"
    return dict(kind=kind, width_mm=round(width * 1000),
                head_mm=round(head * 1000, 1), sill_mm=round(sill * 1000, 1),
                vs_7ft_mm=(round((head - DOOR_STD) * 1000, 1)
                           if "door" in kind else None),
                along_span=[round(a0 + cl * CELL, 2), round(a0 + (cr + 1) * CELL, 2)])


def main(las_path, obj_path, mj, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    d = json.load(open(mj))
    z_floor = float(np.median([r["z_floor"] for r in d["rooms"]]))
    z_ceil = float(np.median([r["z_ceiling"] for r in d["rooms"]]))

    path = walk_path(las_path)
    G = merge_coplanar({n: P for n, P in parse_obj(obj_path).items()
                        if n.startswith("wall")})
    log(f"{len(G)} wall planes")

    rows = []
    walls_plot = []
    n_noplane = 0
    for name, P in G.items():
        if len(P) < 2000:
            continue
        c, n, dv = plane_of(P)
        a = (P[:, :2] - c) @ dv
        amin, amax = float(a.min()), float(a.max())
        walls_plot.append((c, dv, amin, amax, name))

        xs = crossings(path, c, n, dv, amin, amax)
        rel, occ, a0, z0, na, nz = grids(P)
        holes = find_openings(occ, a0, z0, z_floor, z_ceil)
        # grids() measures `along` from the same centroid, so a0 is comparable
        for h in holes:
            hb = h["_box"][0]
            hspan = (hb, hb + h["_box"][2])
            hit = [x for x in xs if hspan[0] - MATCH_TOL <= x[0] <= hspan[1] + MATCH_TOL]
            rows.append(dict(wall=name, kind=h["type"],
                             width_mm=round(h["width_m"] * 1000),
                             head_mm=h["head_mm"], sill_mm=h["sill_mm"],
                             along_m=round(hb, 2),
                             walk_crossings=sum(x[2] for x in hit),
                             verdict="CONFIRMED" if hit else "unwalked"))
        # crossings with no hole nearby: only a MISSED opening if wall material
        # actually stands here (see column_state)
        for ac, zc, cnt in xs:
            near = any(h["_box"][0] - MATCH_TOL <= ac <= h["_box"][0] + h["_box"][2] + MATCH_TOL
                       for h in holes)
            if near:
                continue
            state, f_high, f_walk = column_state(occ, a0, z0, ac, z_floor, z_ceil)
            if state == "no wall":
                n_noplane += 1
                continue
            if state != "opening":
                rows.append(dict(wall=name, kind="(path crossed solid wall)",
                                 width_mm=None, head_mm=None, sill_mm=None,
                                 along_m=round(ac, 2), walk_crossings=cnt,
                                 wall_fill_walk=round(f_walk, 2),
                                 verdict="path-error"))
                continue
            m = measure_at(occ, a0, z0, ac, z_floor, z_ceil)
            if m is None:
                rows.append(dict(wall=name, kind="(opening, could not measure)",
                                 width_mm=None, head_mm=None, sill_mm=None,
                                 along_m=round(ac, 2), walk_crossings=cnt,
                                 verdict="RECOVERED-FAILED"))
                continue
            rows.append(dict(wall=name, kind=m["kind"], width_mm=m["width_mm"],
                             head_mm=m["head_mm"], sill_mm=m["sill_mm"],
                             vs_7ft_mm=m["vs_7ft_mm"], along_m=round(ac, 2),
                             along_span=m["along_span"], walk_crossings=cnt,
                             source="walk-seeded", verdict="RECOVERED"))

    json.dump(dict(path_samples=len(path),
                   crossings_ignored_no_wall=n_noplane, openings=rows),
              open(out / "walk_verified_openings.json", "w"), indent=1)

    perr = [r for r in rows if r["verdict"] == "path-error"]
    conf = [r for r in rows if r["verdict"] == "CONFIRMED"]
    miss = [r for r in rows if r["verdict"] == "RECOVERED"]
    fail = [r for r in rows if r["verdict"] == "RECOVERED-FAILED"]
    unw = [r for r in rows if r["verdict"] == "unwalked"]
    log(f"CONFIRMED {len(conf)}   RECOVERED {len(miss)}   unwalked {len(unw)}   "
        f"unmeasurable {len(fail)}   path-error {len(perr)}   "
        f"(ignored {n_noplane} crossings of a plane "
        f"with no wall standing there)")
    for r in conf:
        log(f"  CONFIRMED {r['wall']:12} {r['kind']:34} "
            f"{r['width_mm']} mm wide, head {r['head_mm']}, walked {r['walk_crossings']}x")
    for r in miss:
        v7 = f"  ({r['vs_7ft_mm']:+.0f} vs 7ft)" if r.get("vs_7ft_mm") is not None else ""
        log(f"  RECOVERED {r['wall']:12} {r['kind']:34} {r['width_mm']:5} mm wide, "
            f"head {r['head_mm']:.0f}, walked {r['walk_crossings']}x{v7}")

    # ---- plan view
    fig, ax = plt.subplots(figsize=(15, 14))
    # draw the real wall footprints, not the merged planes: a merged plane
    # spans the gaps between its runs and would draw walls that do not exist
    for _, _, _, _, name in walls_plot:
        Q = G[name]
        s = Q[::40, :2]
        ax.scatter(s[:, 0], s[:, 1], s=0.6, c="#555", marker=".",
                   linewidths=0, zorder=1)
    ax.plot(path[:, 0], path[:, 1], color="#1f77b4", lw=0.8, alpha=0.55, zorder=2)
    ax.scatter(path[0, 0], path[0, 1], s=90, c="#00c853", marker="o",
               zorder=5, label="walk start")
    ax.scatter(path[-1, 0], path[-1, 1], s=90, c="#d50000", marker="s",
               zorder=5, label="walk end")
    look = {n: (c, dv) for c, dv, _, _, n in walls_plot}
    for r in rows:
        c, dv = look[r["wall"]]
        p = c + dv * r["along_m"]
        if r["verdict"] in ("CONFIRMED", "RECOVERED"):
            ax.scatter(*p, s=150, marker="o", facecolors="none",
                       edgecolors="#00e5ff", lw=2.4, zorder=6)
        elif r["verdict"] in ("path-error", "RECOVERED-FAILED"):
            ax.scatter(*p, s=190, marker="X", c="#ff1744", zorder=7)
        else:
            ax.scatter(*p, s=110, marker="^", c="#ffd400", zorder=6)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], color="#1f77b4", lw=2, label="walk path"),
        Line2D([], [], marker="o", ls="", mfc="none", mec="#00e5ff", mew=2,
               ms=11, label=f"opening confirmed by walk ({len(conf)+len(miss)})"),
        Line2D([], [], marker="X", ls="", color="#ff1744", ms=11,
               label=f"MISSED — walked through, not detected ({len(miss)})"),
        Line2D([], [], marker="^", ls="", color="#ffd400", ms=10,
               label=f"unwalked hole — window / unused / false ({len(unw)})"),
        Line2D([], [], marker="o", ls="", color="#00c853", ms=9, label="start"),
        Line2D([], [], marker="s", ls="", color="#d50000", ms=9, label="end"),
    ], loc="upper right", fontsize=9)
    ax.set_title("Walk path vs detected openings\n"
                 "the operator cannot walk through a wall — every path crossing "
                 "is a real opening")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(out / "walk_verified_openings.png", dpi=130)
    log(f"wrote {out/'walk_verified_openings.png'}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
