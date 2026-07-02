"""Sharp-edged 3D preview: extrude the v3-validated walls (with relief steps),
columns and beams into watertight boolean solids and write a GLB.

This is the answer to Poisson's rounded L-cuts: a column here is its own box,
a groove/relief is a WallStep at its own offset -- every edge is an exact
90-degree crease by construction. Uses only committed recon code paths
(solids.wall_to_solid / column_to_solid / beam_to_solid, assemble.build_scene).

NOTE: walls are NOT midline-recentred here -- solids._step_box expects p0 on
the run's main face with step offsets relative to it (the group_wall_runs
convention). Openings are not cut in this preview (that is plan Task 7/9).

Writes output/koushik_iso/model_sharp_preview.glb
"""
import sys
import time
from types import SimpleNamespace

import numpy as np

from pathlib import Path

WT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, WT)

from scripts.recon.io_las import load_scan
from scripts.recon.frame import estimate_normals, dominant_axes, axis_align
from scripts.recon.planes import detect_planes
from scripts.recon.structure import extract_columns_beams
from scripts.recon.schema import WallStep
from scripts.recon.regularize import snap_walls, resolve_corners, pair_thickness
from scripts.recon.floorplan2d import build_room_polygons

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_floorplan2d_v3 import group_wall_runs_v3, snap_endpoints_to_lines

OUT_DIR = WT + r"\output\koushik_iso"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    import trimesh
    from scripts.recon.solids import wall_to_solid, column_to_solid, beam_to_solid
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

    t0 = time.time()
    planes = detect_planes(xyz, z_floor=z_floor, z_ceiling=z_ceiling)
    verticals = [p for p in planes if p.label == "vertical"]
    log(f"detect_planes: {len(planes)} planes ({len(verticals)} vertical) in {time.time()-t0:.0f}s")

    runs, _used = group_wall_runs_v3(verticals, xyz)
    cols, beams = extract_columns_beams(verticals, xyz, z_floor, z_ceiling)
    walls = snap_walls(runs, np.eye(3))
    walls = pair_thickness(walls, xyz)
    walls = resolve_corners(walls)
    walls = snap_endpoints_to_lines(walls)
    log(f"{len(walls)} walls, {len(cols)} columns, {len(beams)} beams")

    wall_solids = []
    for i, w in enumerate(walls):
        p0 = np.asarray(w["p0"], float)
        p1 = np.asarray(w["p1"], float)
        L = float(np.linalg.norm(p1 - p0))
        if L < 0.05:
            continue
        u_i = 1 if w["direction"] == "x" else 0  # extent axis
        d_comp = (p1 - p0)[u_i]
        sign = 1.0 if d_comp >= 0 else -1.0
        p0_u = p0[u_i]

        rebased = []
        for st in w["steps"]:
            lo = (st.u_min_m - p0_u) * sign
            hi = (st.u_max_m - p0_u) * sign
            lo, hi = min(lo, hi), max(lo, hi)
            z0 = z_floor if st.z_min_m - z_floor < 0.35 else st.z_min_m
            z1 = z_ceiling if z_ceiling - st.z_max_m < 0.35 else st.z_max_m
            rebased.append(WallStep(st.offset_m, lo, hi, z0, z1))
        # main step (offset 0) covers the full (corner-extended) centerline
        main_i = min(range(len(rebased)), key=lambda k: abs(rebased[k].offset_m))
        m = rebased[main_i]
        rebased[main_i] = WallStep(m.offset_m, min(0.0, m.u_min_m), max(L, m.u_max_m),
                                   m.z_min_m, m.z_max_m)

        wall_obj = SimpleNamespace(p0=tuple(w["p0"]), p1=tuple(w["p1"]),
                                   thickness_m=float(w.get("thickness_m", 0.1)))
        try:
            solid = wall_to_solid(wall_obj, rebased)
            wall_solids.append(solid)
        except Exception as exc:
            log(f"wall {i}: solid failed ({exc}) -- skipped")

    col_solids, beam_solids = [], []
    for c in cols:
        try:
            col_solids.append(column_to_solid(c))
        except Exception as exc:
            log(f"column {c.column_id}: failed ({exc})")
    for b in beams:
        try:
            beam_solids.append(beam_to_solid(b))
        except Exception as exc:
            log(f"beam {b.beam_id}: failed ({exc})")

    # floor/ceiling slabs from the room union (fallback: walls bbox)
    slab_solids = {"Floor": [], "Ceiling": []}
    wall_ns = [SimpleNamespace(p0=tuple(w["p0"]), p1=tuple(w["p1"]),
                               thickness_m=float(w.get("thickness_m", 0.1)),
                               length_m=float(np.linalg.norm(np.asarray(w["p1"]) - np.asarray(w["p0"]))))
               for w in walls]
    rooms = [p for p in build_room_polygons(wall_ns, epsilon_m=0.30) if p.area >= 1.0]
    log(f"{len(rooms)} rooms for slab footprint")
    try:
        from shapely.ops import unary_union
        footprint = unary_union(rooms).buffer(0.15, join_style=2)  # mitre = sharp corners
        polys = list(footprint.geoms) if footprint.geom_type == "MultiPolygon" else [footprint]
        for poly in polys:
            floor = trimesh.creation.extrude_polygon(poly, 0.12)
            floor.apply_translation([0, 0, z_floor - 0.12])
            slab_solids["Floor"].append(floor)
            ceil = trimesh.creation.extrude_polygon(poly, 0.12)
            ceil.apply_translation([0, 0, z_ceiling])
            slab_solids["Ceiling"].append(ceil)
    except Exception as exc:
        log(f"slab extrusion failed ({exc}) -- skipping slabs")

    elements = {"Walls": wall_solids, "Columns": col_solids, "Beams": beam_solids}
    elements.update({k: v for k, v in slab_solids.items() if v})
    scene = build_scene(elements)
    out = OUT_DIR + r"\model_sharp_preview.glb"
    write_glb(scene, out)
    n_watertight = sum(1 for s in wall_solids if s.is_watertight)
    log(f"wrote {out}: {len(wall_solids)} wall solids ({n_watertight} watertight), "
        f"{len(col_solids)} columns, {len(beam_solids)} beams, "
        f"{len(slab_solids['Floor'])} floor / {len(slab_solids['Ceiling'])} ceiling slabs")


if __name__ == "__main__":
    main()
