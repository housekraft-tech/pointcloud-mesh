"""Sharp 3D preview v3 -- fix from user review of v2: a big ceiling BEAM's
side faces were extruded floor-to-ceiling as a solid wall, walling off an
open passage. v2's mistake: every run was forced to full height across its
full span.

v3 respects the z-evidence per stretch of each run:
  - stretch whose points reach near the floor      -> WALL (floor..ceiling box;
    bottoms up to 1.2 m above floor count as furniture occlusion and fill down)
  - stretch starting > 1.2 m up but reaching the ceiling -> BEAM: box only
    from its measured soffit up to the ceiling -> HOLLOW GAP underneath
  - stretch reaching neither floor nor ceiling     -> dropped (noise)
  - coverage holes between stretches stay OPEN (no more full-span bridging --
    this also removes v2's "extra walls where there are no walls")

Runs whose every stretch is elevated go to the Beams collection (these are
the real beams, evidenced); mixed runs stay Walls with hollow gaps.
Keeps v2's: gps_time walkthrough door cuts, phantom-wall coverage filter,
no raw-beam boxes, columns, floor/ceiling slabs.

Writes output/koushik_iso/model_sharp_preview_v3.glb
"""
import sys
import time
from types import SimpleNamespace

import numpy as np

from pathlib import Path

WT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, WT)
SCRATCH = str(Path(__file__).resolve().parent)
sys.path.insert(0, SCRATCH)

from scripts.recon.io_las import load_scan
from scripts.recon.frame import estimate_normals, dominant_axes, axis_align
from scripts.recon.planes import detect_planes
from scripts.recon.structure import _plane_axis_stats, _greedy_chain, extract_columns_beams
from scripts.recon.schema import WallStep
from scripts.recon.regularize import snap_walls, resolve_corners, pair_thickness
from scripts.recon.trajectory import approx_trajectory
from scripts.recon.floorplan2d import build_room_polygons
from diag_floorplan2d_v3 import snap_endpoints_to_lines
from sharp_preview_v2 import wall_crossings, cluster_crossings, DOOR_W_M, DOOR_H_M

OUT_DIR = WT + r"\output\koushik_iso"

U_GAP_SPLIT_M = 2.8
SEG_MERGE_M = 0.35        # member intervals closer than this fuse into one stretch
CORNER_EXTEND_M = 0.8     # extend first/last stretch to snapped corners within this
OCCLUSION_FILL_M = 1.2    # bottom starting below this above floor = occluded wall -> fill down
BEAM_CEIL_TOL_M = 0.5     # beam stretch must reach within this of the ceiling
MIN_COVERAGE = 0.35


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def group_runs_with_z(verticals, xyz, merge_offset_m=0.15, max_relief_m=0.35,
                      min_run_length_m=0.5, angle_tol_deg=10.0,
                      full_height_frac=0.4, u_gap_m=U_GAP_SPLIT_M,
                      min_run_inliers=1200):
    """v3 grouping, but members carry per-plane z evidence (dicts, not tuples)."""
    xyz = np.asarray(xyz, dtype=float)
    ex, ey = np.eye(3)[:, 0], np.eye(3)[:, 1]
    cos_tol = np.cos(np.deg2rad(angle_tol_deg))
    stats = []
    for i, p in enumerate(verticals):
        s = _plane_axis_stats(p, xyz, ex, ey)
        if s["align"] < cos_tol:
            continue
        stats.append(s)
    if not stats:
        return []
    max_z_span = max(s["z_max"] - s["z_min"] for s in stats)
    stats = [s for s in stats if (s["z_max"] - s["z_min"]) >= full_height_frac * max_z_span]

    runs = []
    for axis_name in ("x", "y"):
        group = [s for s in stats if s["axis"] == axis_name]
        for cluster in _greedy_chain(group, key=lambda s: s["offset"], max_gap=max_relief_m):
            members = sorted(cluster, key=lambda s: s["u_min"])
            segs, cur, cur_max = [], [], None
            for s in members:
                if cur and s["u_min"] - cur_max > u_gap_m:
                    segs.append(cur)
                    cur, cur_max = [], None
                cur.append(s)
                cur_max = s["u_max"] if cur_max is None else max(cur_max, s["u_max"])
            if cur:
                segs.append(cur)
            for useg in segs:
                span = max(x["u_max"] for x in useg) - min(x["u_min"] for x in useg)
                if span < min_run_length_m:
                    continue
                if sum(x["n_inliers"] for x in useg) < min_run_inliers:
                    continue
                step_groups = _greedy_chain(useg, key=lambda s: s["offset"], max_gap=merge_offset_m)
                main_sg = max(step_groups, key=lambda sg: sum(x["n_inliers"] for x in sg))
                main_abs = (sum(x["offset"] * x["n_inliers"] for x in main_sg)
                            / sum(x["n_inliers"] for x in main_sg))
                axis_vec, u_vec = useg[0]["axis_vec"], useg[0]["u_vec"]
                u_lo = min(x["u_min"] for x in useg)
                u_hi = max(x["u_max"] for x in useg)
                p0 = axis_vec * main_abs + u_vec * u_lo
                p1 = axis_vec * main_abs + u_vec * u_hi
                runs.append(dict(
                    direction=axis_name,
                    normal=tuple(map(float, axis_vec)),
                    offset_m=float(main_abs),
                    p0=(float(p0[0]), float(p0[1])),
                    p1=(float(p1[0]), float(p1[1])),
                    steps=[WallStep(0.0, u_lo, u_hi,
                                    min(x["z_min"] for x in useg),
                                    max(x["z_max"] for x in useg))],
                    members=[(float(x["u_min"]), float(x["u_max"])) for x in useg],
                    members_z=[dict(u_min=float(x["u_min"]), u_max=float(x["u_max"]),
                                    z_min=float(x["z_min"]), z_max=float(x["z_max"]),
                                    offset=float(x["offset"] - main_abs),
                                    n=int(x["n_inliers"])) for x in useg],
                ))
    return runs


def build_stretches(members_z, z_floor, z_ceiling):
    """Merge member u-intervals into stretches; classify each stretch's z."""
    members = sorted(members_z, key=lambda m: m["u_min"])
    merged = []
    for m in members:
        if merged and m["u_min"] <= merged[-1]["u_max"] + SEG_MERGE_M:
            g = merged[-1]
            g["u_max"] = max(g["u_max"], m["u_max"])
            g["z_min"] = min(g["z_min"], m["z_min"])
            g["z_max"] = max(g["z_max"], m["z_max"])
        else:
            merged.append(dict(u_min=m["u_min"], u_max=m["u_max"],
                               z_min=m["z_min"], z_max=m["z_max"]))
    out = []
    for g in merged:
        reaches_ceiling = (z_ceiling - g["z_max"]) <= BEAM_CEIL_TOL_M
        bottom = g["z_min"] - z_floor
        if bottom <= OCCLUSION_FILL_M:
            out.append(dict(kind="wall", u0=g["u_min"], u1=g["u_max"],
                            z0=z_floor, z1=z_ceiling))
        elif reaches_ceiling:
            out.append(dict(kind="beam", u0=g["u_min"], u1=g["u_max"],
                            z0=g["z_min"], z1=z_ceiling))
        # neither floor-anchored nor ceiling-hung -> noise, dropped
    return out


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
    log(f"loaded {scan.n:,} points")
    rng = np.random.default_rng(0)
    sub = rng.choice(scan.n, size=min(800_000, scan.n), replace=False)
    R = dominant_axes(estimate_normals(scan.xyz[sub]))
    scan = axis_align(scan, R)
    xyz = scan.xyz
    z_floor = float(np.percentile(xyz[:, 2], 1.0))
    z_ceiling = float(np.percentile(xyz[:, 2], 99.0))
    traj = approx_trajectory(scan.gps_time, xyz)

    t0 = time.time()
    planes = detect_planes(xyz, z_floor=z_floor, z_ceiling=z_ceiling)
    verticals = [p for p in planes if p.label == "vertical"]
    log(f"detect_planes: {len(planes)} planes ({len(verticals)} vertical) in {time.time()-t0:.0f}s")

    runs = group_runs_with_z(verticals, xyz)
    cols, _ = extract_columns_beams(verticals, xyz, z_floor, z_ceiling)
    walls = snap_walls(runs, np.eye(3))
    walls = pair_thickness(walls, xyz)
    walls = resolve_corners(walls)
    walls = snap_endpoints_to_lines(walls)

    kept = []
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
            kept.append(w)
    walls = kept

    crossings = wall_crossings(traj, walls)
    log(f"{len(walls)} walls; {sum(len(v) for v in crossings.values())} crossings "
        f"on {len(crossings)} walls")

    wall_solids, beam_solids = [], []
    n_cut = n_hollow = 0
    for i, w in enumerate(walls):
        p0 = np.asarray(w["p0"], float)
        p1 = np.asarray(w["p1"], float)
        L = float(np.linalg.norm(p1 - p0))
        if L < 0.05:
            continue
        u_i = 1 if w["direction"] == "x" else 0
        sign = 1.0 if (p1 - p0)[u_i] >= 0 else -1.0
        p0_u = p0[u_i]

        stretches = build_stretches(w["members_z"], z_floor, z_ceiling)
        if not stretches:
            continue
        # extend outermost WALL stretches onto snapped corners
        rel = []
        for s in stretches:
            lo = (s["u0"] - p0_u) * sign
            hi = (s["u1"] - p0_u) * sign
            rel.append(dict(kind=s["kind"], lo=min(lo, hi), hi=max(lo, hi),
                            z0=s["z0"], z1=s["z1"]))
        rel.sort(key=lambda s: s["lo"])
        if rel[0]["lo"] <= CORNER_EXTEND_M:
            rel[0]["lo"] = min(rel[0]["lo"], 0.0)
        if L - rel[-1]["hi"] <= CORNER_EXTEND_M:
            rel[-1]["hi"] = max(rel[-1]["hi"], L)

        steps = [WallStep(0.0, s["lo"], s["hi"], s["z0"], s["z1"]) for s in rel]
        n_hollow += sum(1 for s in rel if s["kind"] == "beam")
        wall_obj = SimpleNamespace(p0=tuple(w["p0"]), p1=tuple(w["p1"]),
                                   thickness_m=float(w.get("thickness_m", 0.1)),
                                   floor_z_m=z_floor)
        try:
            solid = wall_to_solid(wall_obj, steps)
        except Exception as exc:
            log(f"wall {i}: solid failed ({exc}) -- skipped")
            continue

        openings = []
        for grp in (cluster_crossings(crossings[i]) if i in crossings else []):
            u_lo_abs, u_hi_abs = min(grp), max(grp)
            width = max(DOOR_W_M, (u_hi_abs - u_lo_abs) + 0.6)
            mid = ((u_lo_abs + u_hi_abs) / 2 - p0_u) * sign
            u0, u1 = mid - width / 2, mid + width / 2
            # only cut where a floor-anchored stretch actually exists
            if not any(s["kind"] == "wall" and s["lo"] < u1 and s["hi"] > u0 for s in rel):
                continue
            u0, u1 = max(u0, 0.05), min(u1, L - 0.05)
            if u1 - u0 < 0.3:
                continue
            openings.append(SimpleNamespace(
                opening_id=f"door_w{i}_{len(openings)}",
                u_min_m=u0, u_max_m=u1, sill_m=0.0,
                height_m=min(DOOR_H_M, z_ceiling - z_floor - 0.1)))
        if openings:
            try:
                solid = cut_openings(solid, openings, wall_obj)
                n_cut += len(openings)
            except Exception as exc:
                log(f"wall {i}: cut failed ({exc}) -- kept uncut")

        if all(s["kind"] == "beam" for s in rel):
            beam_solids.append(solid)   # pure beam run -> Beams collection
        else:
            wall_solids.append(solid)

    col_solids = []
    for c in cols:
        try:
            col_solids.append(column_to_solid(c))
        except Exception:
            pass

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
        for poly in polys:
            f = trimesh.creation.extrude_polygon(poly, 0.12)
            f.apply_translation([0, 0, z_floor - 0.12])
            slab["Floor"].append(f)
            c = trimesh.creation.extrude_polygon(poly, 0.12)
            c.apply_translation([0, 0, z_ceiling])
            slab["Ceiling"].append(c)
    except Exception as exc:
        log(f"slabs failed ({exc})")

    elements = {"Walls": wall_solids, "Columns": col_solids}
    if beam_solids:
        elements["Beams"] = beam_solids
    elements.update({k: v for k, v in slab.items() if v})
    scene = build_scene(elements)
    out = OUT_DIR + r"\model_sharp_preview_v3.glb"
    write_glb(scene, out)
    log(f"wrote {out}: {len(wall_solids)} walls ({n_cut} door cuts, "
        f"{n_hollow} beam stretches with hollow gaps), {len(beam_solids)} pure beam runs, "
        f"{len(col_solids)} columns, {len(rooms)} rooms")


if __name__ == "__main__":
    main()
