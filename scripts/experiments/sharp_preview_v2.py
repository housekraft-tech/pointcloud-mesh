"""Sharp 3D preview v2 -- fixes from user review of v1:

1. DOOR OPENINGS CUT: gps_time trajectory crossings (the operator physically
   walked through every door / onto the balcony) become boolean door cuts:
   default 0.9 m wide x 2.13 m (7 ft) high; clustered crossings widen the cut
   (balcony sliders). This is plan Task 5 logic running for real.
2. NO BEAM BOXES: the raw beam set is unfiltered (curtain tracks, cabinet
   tops) and rendered as the "big unnecessary boxes" -- dropped from the
   preview entirely until Task 10's filter lands.
3. FEWER PHANTOM WALLS: runs whose member planes cover < 35% of their span
   are dropped (sparse phantom evidence).
4. FULL-HEIGHT WALLS: every main step spans floor..ceiling exactly.

Writes output/koushik_iso/model_sharp_preview_v2.glb
"""
import sys
import time
from types import SimpleNamespace

import numpy as np

from pathlib import Path

WT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, WT)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.recon.io_las import load_scan
from scripts.recon.frame import estimate_normals, dominant_axes, axis_align
from scripts.recon.planes import detect_planes
from scripts.recon.structure import extract_columns_beams
from scripts.recon.schema import WallStep
from scripts.recon.regularize import snap_walls, resolve_corners, pair_thickness
from scripts.recon.trajectory import approx_trajectory
from scripts.recon.floorplan2d import build_room_polygons
from diag_floorplan2d_v3 import group_wall_runs_v3, snap_endpoints_to_lines

OUT_DIR = WT + r"\output\koushik_iso"

DOOR_W_M = 0.9          # default door cut width per walkthrough crossing
DOOR_H_M = 2.13         # 7 ft door height prior (user-supplied)
CROSSING_CLUSTER_M = 1.2  # crossings closer than this merge into one opening
MIN_COVERAGE = 0.35     # member planes must cover >= this fraction of run span


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wall_crossings(trajectory, walls, end_margin_m=0.15):
    """Absolute world-u positions where consecutive trajectory steps cross
    each wall's centerline (plan Task 5)."""
    traj = np.asarray(trajectory, dtype=float)[:, :2]
    out = {}
    for wi, w in enumerate(walls):
        p0 = np.asarray(w["p0"], float)
        p1 = np.asarray(w["p1"], float)
        d = p1 - p0
        length = float(np.linalg.norm(d))
        if length == 0:
            continue
        u_i = 1 if w["direction"] == "x" else 0
        hits = []
        for a, b in zip(traj[:-1], traj[1:]):
            e = b - a
            denom = d[0] * e[1] - d[1] * e[0]
            if abs(denom) < 1e-12:
                continue
            r = a - p0
            t = (r[0] * e[1] - r[1] * e[0]) / denom
            s = (r[0] * d[1] - r[1] * d[0]) / denom
            if 0.0 <= s <= 1.0 and end_margin_m <= t * length <= length - end_margin_m:
                hits.append(float((p0 + t * d)[u_i]))
        if hits:
            out[wi] = sorted(hits)
    return out


def cluster_crossings(us, gap_m=CROSSING_CLUSTER_M):
    groups, cur = [], [us[0]]
    for u in us[1:]:
        if u - cur[-1] <= gap_m:
            cur.append(u)
        else:
            groups.append(cur)
            cur = [u]
    groups.append(cur)
    return groups


def main():
    import trimesh
    from scripts.recon.solids import wall_to_solid, column_to_solid, cut_openings
    from scripts.recon.assemble import build_scene, write_glb

    try:
        import open3d as o3d
        o3d.utility.random.seed(0)
    except Exception:
        pass

    scan = load_scan(OUT_DIR + r"\isolated.las")
    log(f"loaded {scan.n:,} points (gps_time: {scan.gps_time is not None})")
    rng = np.random.default_rng(0)
    sub = rng.choice(scan.n, size=min(800_000, scan.n), replace=False)
    R = dominant_axes(estimate_normals(scan.xyz[sub]))
    scan = axis_align(scan, R)
    xyz = scan.xyz
    z_floor = float(np.percentile(xyz[:, 2], 1.0))
    z_ceiling = float(np.percentile(xyz[:, 2], 99.0))

    traj = approx_trajectory(scan.gps_time, xyz) if scan.gps_time is not None else None
    log(f"trajectory: {0 if traj is None else len(traj)} vertices")

    t0 = time.time()
    planes = detect_planes(xyz, z_floor=z_floor, z_ceiling=z_ceiling)
    verticals = [p for p in planes if p.label == "vertical"]
    log(f"detect_planes: {len(planes)} planes ({len(verticals)} vertical) in {time.time()-t0:.0f}s")

    runs, _ = group_wall_runs_v3(verticals, xyz)
    cols, _beams = extract_columns_beams(verticals, xyz, z_floor, z_ceiling)
    walls = snap_walls(runs, np.eye(3))
    walls = pair_thickness(walls, xyz)
    walls = resolve_corners(walls)
    walls = snap_endpoints_to_lines(walls)

    # phantom-wall filter: member coverage fraction of the run's span
    kept_walls = []
    for w in walls:
        span = float(np.linalg.norm(np.asarray(w["p1"]) - np.asarray(w["p0"])))
        merged = []
        for lo, hi in sorted(w["members"]):
            if merged and lo <= merged[-1][1] + 0.05:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        covered = sum(hi - lo for lo, hi in merged)
        if span > 0 and covered / span >= MIN_COVERAGE:
            kept_walls.append(w)
        else:
            log(f"dropped phantom wall (coverage {covered:.1f}/{span:.1f} m)")
    walls = kept_walls

    crossings = wall_crossings(traj, walls) if traj is not None else {}
    n_cross = sum(len(v) for v in crossings.values())
    log(f"{len(walls)} walls after coverage filter; {n_cross} walkthrough crossings "
        f"on {len(crossings)} walls")

    wall_solids, door_panels = [], []
    n_cut = 0
    for i, w in enumerate(walls):
        p0 = np.asarray(w["p0"], float)
        p1 = np.asarray(w["p1"], float)
        L = float(np.linalg.norm(p1 - p0))
        if L < 0.05:
            continue
        u_i = 1 if w["direction"] == "x" else 0
        d_comp = (p1 - p0)[u_i]
        sign = 1.0 if d_comp >= 0 else -1.0
        p0_u = p0[u_i]

        rebased = []
        for st in w["steps"]:
            lo = (st.u_min_m - p0_u) * sign
            hi = (st.u_max_m - p0_u) * sign
            lo, hi = min(lo, hi), max(lo, hi)
            rebased.append(WallStep(st.offset_m, lo, hi, st.z_min_m, st.z_max_m))
        main_i = min(range(len(rebased)), key=lambda k: abs(rebased[k].offset_m))
        m = rebased[main_i]
        rebased[main_i] = WallStep(m.offset_m, min(0.0, m.u_min_m), max(L, m.u_max_m),
                                   z_floor, z_ceiling)  # full height, always

        wall_obj = SimpleNamespace(p0=tuple(w["p0"]), p1=tuple(w["p1"]),
                                   thickness_m=float(w.get("thickness_m", 0.1)),
                                   floor_z_m=z_floor)
        try:
            solid = wall_to_solid(wall_obj, rebased)
        except Exception as exc:
            log(f"wall {i}: solid failed ({exc}) -- skipped")
            continue

        # door cuts from walkthrough crossings
        openings = []
        for grp in (cluster_crossings(crossings[i]) if i in crossings else []):
            u_lo_abs, u_hi_abs = min(grp), max(grp)
            width = max(DOOR_W_M, (u_hi_abs - u_lo_abs) + 0.6)
            mid = ((u_lo_abs + u_hi_abs) / 2 - p0_u) * sign
            u0, u1 = mid - width / 2, mid + width / 2
            if u1 < 0.05 or u0 > L - 0.05:
                continue
            u0, u1 = max(u0, 0.05), min(u1, L - 0.05)
            openings.append(SimpleNamespace(
                opening_id=f"door_w{i}_{len(openings)}",
                u_min_m=u0, u_max_m=u1, sill_m=0.0,
                height_m=min(DOOR_H_M, z_ceiling - z_floor - 0.1)))
        if openings:
            try:
                solid = cut_openings(solid, openings, wall_obj)
                n_cut += len(openings)
            except Exception as exc:
                log(f"wall {i}: opening cut failed ({exc}) -- kept uncut")
        wall_solids.append(solid)

    col_solids = []
    for c in cols:
        try:
            col_solids.append(column_to_solid(c))
        except Exception as exc:
            log(f"column {c.column_id}: failed ({exc})")

    slab = {"Floor": [], "Ceiling": []}
    wall_ns = [SimpleNamespace(p0=tuple(w["p0"]), p1=tuple(w["p1"]),
                               thickness_m=float(w.get("thickness_m", 0.1)),
                               length_m=float(np.linalg.norm(np.asarray(w["p1"]) - np.asarray(w["p0"]))))
               for w in walls]
    rooms = [p for p in build_room_polygons(wall_ns, epsilon_m=0.30) if p.area >= 1.0]
    try:
        from shapely.ops import unary_union
        footprint = unary_union(rooms).buffer(0.15, join_style=2)
        polys = list(footprint.geoms) if footprint.geom_type == "MultiPolygon" else [footprint]
        import trimesh as tm
        for poly in polys:
            f = tm.creation.extrude_polygon(poly, 0.12)
            f.apply_translation([0, 0, z_floor - 0.12])
            slab["Floor"].append(f)
            c = tm.creation.extrude_polygon(poly, 0.12)
            c.apply_translation([0, 0, z_ceiling])
            slab["Ceiling"].append(c)
    except Exception as exc:
        log(f"slabs failed ({exc})")

    elements = {"Walls": wall_solids, "Columns": col_solids}
    elements.update({k: v for k, v in slab.items() if v})
    scene = build_scene(elements)
    out = OUT_DIR + r"\model_sharp_preview_v2.glb"
    write_glb(scene, out)
    log(f"wrote {out}: {len(wall_solids)} walls ({n_cut} door cuts), "
        f"{len(col_solids)} columns, {len(rooms)} rooms, no beams (filtered)")


if __name__ == "__main__":
    main()
