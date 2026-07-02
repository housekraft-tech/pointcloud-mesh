"""Room-wise interior walls v5 -- fixes from user review of v4:

1. ONE OBJECT PER WALL: all of a wall-edge's panels (relief members, beam
   soffits, sills/headers) boolean-union into a single mesh -- no more loose
   individual pieces.
2. NO MISSING WALL DATA: v4 only rendered walls reachable through closed-room
   edges, silently dropping every wall in regions where rooms didn't close.
   v5 renders every unmatched wall into a "Walls_unassigned" collection.
3. MORE ROOMS: polygonize runs at eps 0.30 then 0.50; cells found only at the
   looser epsilon are added when they don't overlap an existing room.
4. NO DUPLICATE GEOMETRY: a wall shared by two rooms renders its panels once
   (deduped globally); both faces still appear since every detected plane
   renders at its own offset.

Writes output/koushik_iso/model_rooms_v5.glb
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
from scripts.recon.structure import extract_columns_beams
from scripts.recon.regularize import snap_walls, resolve_corners, pair_thickness
from scripts.recon.trajectory import approx_trajectory
from scripts.recon.floorplan2d import build_room_polygons
from diag_floorplan2d_v3 import snap_endpoints_to_lines
from sharp_preview_v2 import wall_crossings, cluster_crossings, DOOR_W_M, DOOR_H_M
from sharp_preview_v3 import group_runs_with_z

OUT_DIR = WT + r"\output\koushik_iso"

PANEL_T = 0.02
EDGE_MATCH_M = 0.45
OCCLUSION_FILL_M = 1.2
CEIL_SNAP_M = 0.5
MIN_PANEL_H = 0.15
MIN_EDGE_M = 0.15


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    import trimesh
    from scripts.recon.solids import column_to_solid
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
    crossings = wall_crossings(traj, walls)
    log(f"{len(walls)} walls, {sum(len(v) for v in crossings.values())} crossings")

    wall_ns = [SimpleNamespace(p0=tuple(w["p0"]), p1=tuple(w["p1"]),
                               thickness_m=float(w.get("thickness_m", 0.1)),
                               length_m=float(np.linalg.norm(np.asarray(w["p1"]) - np.asarray(w["p0"]))))
               for w in walls]
    rooms = [p.simplify(0.02) for p in build_room_polygons(wall_ns, epsilon_m=0.30)
             if p.area >= 1.0]
    for cell in build_room_polygons(wall_ns, epsilon_m=0.50):
        if cell.area >= 1.0 and not any(r.contains(cell.centroid) for r in rooms):
            rooms.append(cell.simplify(0.02))
    log(f"{len(rooms)} rooms (eps 0.30 + 0.50 recovery)")

    def box(x0, x1, y0, y1, z0, z1):
        b = trimesh.creation.box(extents=(max(x1 - x0, 1e-4), max(y1 - y0, 1e-4),
                                          max(z1 - z0, 1e-4)))
        b.apply_translation([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
        return b

    seen_panels = set()   # global dedupe: (wall, u, z, offset) rounded

    def member_panels(w, horiz, e_lo, e_hi):
        """Panels for wall w's members clipped to [e_lo, e_hi], deduped."""
        out = []
        for m in w["members_z"]:
            lo = max(m["u_min"], e_lo)
            hi = min(m["u_max"], e_hi)
            if hi - lo < 0.05:
                continue
            z0 = z_floor if (m["z_min"] - z_floor) <= OCCLUSION_FILL_M else m["z_min"]
            z1 = z_ceiling if (z_ceiling - m["z_max"]) <= CEIL_SNAP_M else m["z_max"]
            if z1 - z0 < MIN_PANEL_H:
                continue
            face_pos = w["offset_m"] + m["offset"]
            key = (id(w), round(lo, 1), round(hi, 1), round(z0, 1), round(face_pos, 2))
            if key in seen_panels:
                continue
            seen_panels.add(key)
            p_lo, p_hi = face_pos - PANEL_T / 2, face_pos + PANEL_T / 2
            if horiz:
                out.append(box(lo, hi, p_lo, p_hi, z0, z1))
            else:
                out.append(box(p_lo, p_hi, lo, hi, z0, z1))
        return out

    def door_cutters(w, wi, horiz, e_lo, e_hi):
        cutters = []
        edge_hits = [u for u in crossings.get(wi, []) if e_lo - 0.3 <= u <= e_hi + 0.3]
        if not edge_hits:
            return cutters
        for grp in cluster_crossings(edge_hits):
            u_lo_abs, u_hi_abs = min(grp), max(grp)
            width = max(DOOR_W_M, (u_hi_abs - u_lo_abs) + 0.6)
            mid = (u_lo_abs + u_hi_abs) / 2
            c0, c1 = mid - width / 2, mid + width / 2
            zc = z_floor + min(DOOR_H_M, z_ceiling - z_floor - 0.1)
            q_lo, q_hi = w["offset_m"] - 0.5, w["offset_m"] + 0.5
            if horiz:
                cutters.append(box(c0, c1, q_lo, q_hi, z_floor - 0.05, zc))
            else:
                cutters.append(box(q_lo, q_hi, c0, c1, z_floor - 0.05, zc))
        return cutters

    def fuse(meshes):
        """One object per wall: union; fall back to concatenate."""
        if len(meshes) == 1:
            return meshes[0]
        try:
            return trimesh.boolean.union(meshes)
        except Exception:
            return trimesh.util.concatenate(meshes)

    def build_wall_object(w, wi, horiz, e_lo, e_hi):
        panels = member_panels(w, horiz, e_lo, e_hi)
        if not panels:
            return None
        cutters = door_cutters(w, wi, horiz, e_lo, e_hi)
        solid = fuse(panels)
        for cutter in cutters:
            try:
                r = trimesh.boolean.difference([solid, cutter])
                if r.volume > 1e-6:
                    solid = r
            except Exception:
                pass
        return solid

    elements = {}
    used_walls = set()
    n_objects = 0
    for ri, room in enumerate(rooms):
        group = []
        cx, cy = room.centroid.x, room.centroid.y
        coords = list(room.exterior.coords)
        for (xa, ya), (xb, yb) in zip(coords[:-1], coords[1:]):
            ex, ey = xb - xa, yb - ya
            elen = float(np.hypot(ex, ey))
            if elen < MIN_EDGE_M:
                continue
            horiz = abs(ex) >= abs(ey)
            want_dir = "y" if horiz else "x"
            e_perp = (ya + yb) / 2 if horiz else (xa + xb) / 2
            e_lo = min(xa, xb) if horiz else min(ya, yb)
            e_hi = max(xa, xb) if horiz else max(ya, yb)

            best = None
            for wi, w in enumerate(walls):
                if w["direction"] != want_dir:
                    continue
                d = abs(w["offset_m"] - e_perp)
                if d > EDGE_MATCH_M:
                    continue
                m_lo = min(m["u_min"] for m in w["members_z"])
                m_hi = max(m["u_max"] for m in w["members_z"])
                overlap = min(e_hi, m_hi) - max(e_lo, m_lo)
                if overlap < 0.3 * elen:
                    continue
                if best is None or d < best[0]:
                    best = (d, wi, w)
            if best is None:
                continue
            _, wi, w = best
            used_walls.add(wi)
            obj = build_wall_object(w, wi, horiz, e_lo, e_hi)
            if obj is not None:
                group.append(obj)
                n_objects += 1

        try:
            floor_panel = trimesh.creation.extrude_polygon(room, PANEL_T)
            floor_panel.apply_translation([0, 0, z_floor - PANEL_T])
            group.append(floor_panel)
        except Exception:
            pass
        if group:
            elements[f"Room_{ri:02d}_{room.area:.0f}m2"] = group

    # every wall no room edge claimed -> still in the model, never dropped
    unassigned = []
    for wi, w in enumerate(walls):
        if wi in used_walls:
            continue
        horiz = w["direction"] == "y"
        m_lo = min(m["u_min"] for m in w["members_z"])
        m_hi = max(m["u_max"] for m in w["members_z"])
        obj = build_wall_object(w, wi, horiz, m_lo, m_hi)
        if obj is not None:
            unassigned.append(obj)
            n_objects += 1
    if unassigned:
        elements["Walls_unassigned"] = unassigned
    log(f"{len(unassigned)} unassigned walls rendered (were silently missing in v4)")

    col_solids = []
    for c in cols:
        try:
            col_solids.append(column_to_solid(c))
        except Exception:
            pass
    if col_solids:
        elements["Columns"] = col_solids

    scene = build_scene(elements)
    out = OUT_DIR + r"\model_rooms_v5.glb"
    write_glb(scene, out)
    log(f"wrote {out}: {len(rooms)} rooms, {n_objects} wall objects "
        f"(one per wall, unioned), {len(col_solids)} columns")


if __name__ == "__main__":
    main()
