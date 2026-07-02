"""Sharp preview v4 -- ROOM-WISE INTERIOR WALLS (user direction):

No wall-thickness solids. For each closed room, each boundary edge is snapped
to the detected interior FACE of its wall (midline +- half measured thickness,
on the room's side) and rebuilt as thin 20 mm panels -- ONE PANEL PER DETECTED
PLANE, each with that plane's own z-range:
  - wall plane reaching low        -> panel floor..ceiling
  - beam/header plane starting high -> elevated panel, HOLLOW GAP beneath
    (kitchen / bedroom beams render correctly by construction)
  - sill + header + side planes    -> window hole emerges from the data
Coverage gaps stay open; gps_time walkthrough crossings cut doors through
whatever panels they hit. Each room is its own named collection with its own
floor panel -> per-room interior dimensions are exactly the detected faces.

Writes output/koushik_iso/model_rooms_v4.glb
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

PANEL_T = 0.02          # panel thickness (visual only; interior face is exact)
EDGE_MATCH_M = 0.45     # max distance edge<->wall midline to accept the match
OCCLUSION_FILL_M = 1.2  # plane bottoms below this above floor fill to floor
CEIL_SNAP_M = 0.5       # plane tops within this of ceiling snap to ceiling
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
    # NO recentring: offsets stay on the DETECTED faces so every member panel
    # can render at its own measured offset -- wall relief (niches, boxed
    # conduits, pilaster intrusions) is the point, not a defect.
    walls = resolve_corners(walls)
    walls = snap_endpoints_to_lines(walls)
    crossings = wall_crossings(traj, walls)
    log(f"{len(walls)} walls (midline), {sum(len(v) for v in crossings.values())} crossings")

    wall_ns = [SimpleNamespace(p0=tuple(w["p0"]), p1=tuple(w["p1"]),
                               thickness_m=float(w.get("thickness_m", 0.1)),
                               length_m=float(np.linalg.norm(np.asarray(w["p1"]) - np.asarray(w["p0"]))))
               for w in walls]
    rooms = [p.simplify(0.02) for p in build_room_polygons(wall_ns, epsilon_m=0.30)
             if p.area >= 1.0]
    log(f"{len(rooms)} rooms")

    def box(x0, x1, y0, y1, z0, z1):
        b = trimesh.creation.box(extents=(max(x1 - x0, 1e-4), max(y1 - y0, 1e-4),
                                          max(z1 - z0, 1e-4)))
        b.apply_translation([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
        return b

    elements = {}
    n_panels = n_doors = 0
    for ri, room in enumerate(rooms):
        panels = []
        cx, cy = room.centroid.x, room.centroid.y
        coords = list(room.exterior.coords)
        for (xa, ya), (xb, yb) in zip(coords[:-1], coords[1:]):
            ex, ey = xb - xa, yb - ya
            elen = float(np.hypot(ex, ey))
            if elen < MIN_EDGE_M:
                continue
            horiz = abs(ex) >= abs(ey)          # edge runs along X?
            # perpendicular axis: y if horiz else x; run direction naming:
            # direction=="x" means normal along X -> wall runs along Y
            want_dir = "y" if horiz else "x"
            e_perp = (ya + yb) / 2 if horiz else (xa + xb) / 2
            e_lo = min(xa, xb) if horiz else min(ya, yb)
            e_hi = max(xa, xb) if horiz else max(ya, yb)
            c_perp = cy if horiz else cx

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
                # no detected wall -- panel at the edge itself, full height
                z0, z1 = z_floor, z_ceiling
                if horiz:
                    panels.append(box(e_lo, e_hi, e_perp - PANEL_T / 2,
                                      e_perp + PANEL_T / 2, z0, z1))
                else:
                    panels.append(box(e_perp - PANEL_T / 2, e_perp + PANEL_T / 2,
                                      e_lo, e_hi, z0, z1))
                n_panels += 1
                continue

            _, wi, w = best
            room_boxes = []
            for m in w["members_z"]:
                lo = max(m["u_min"], e_lo)
                hi = min(m["u_max"], e_hi)
                if hi - lo < 0.05:
                    continue
                z0 = z_floor if (m["z_min"] - z_floor) <= OCCLUSION_FILL_M else m["z_min"]
                z1 = z_ceiling if (z_ceiling - m["z_max"]) <= CEIL_SNAP_M else m["z_max"]
                if z1 - z0 < MIN_PANEL_H:
                    continue
                # each detected plane renders AT ITS OWN measured offset --
                # this is what preserves wall relief/intrusions/grooves
                face_pos = w["offset_m"] + m["offset"]
                p_lo = face_pos - PANEL_T / 2
                p_hi = face_pos + PANEL_T / 2
                if horiz:
                    room_boxes.append(box(lo, hi, p_lo, p_hi, z0, z1))
                else:
                    room_boxes.append(box(p_lo, p_hi, lo, hi, z0, z1))

            # door cuts from walkthrough crossings on this wall within the edge
            edge_hits = [u for u in crossings.get(wi, [])
                         if e_lo - 0.3 <= u <= e_hi + 0.3]
            if edge_hits:
                for grp in cluster_crossings(edge_hits):
                    u_lo_abs, u_hi_abs = min(grp), max(grp)
                    width = max(DOOR_W_M, (u_hi_abs - u_lo_abs) + 0.6)
                    mid = (u_lo_abs + u_hi_abs) / 2
                    c0, c1 = mid - width / 2, mid + width / 2
                    zc = z_floor + min(DOOR_H_M, z_ceiling - z_floor - 0.1)
                    # cutter spans the whole possible relief range perpendicular
                    # to the wall so it punches every member panel it overlaps
                    q_lo, q_hi = w["offset_m"] - 0.5, w["offset_m"] + 0.5
                    if horiz:
                        cutter = box(c0, c1, q_lo, q_hi, z_floor - 0.05, zc)
                    else:
                        cutter = box(q_lo, q_hi, c0, c1, z_floor - 0.05, zc)
                    cut_boxes = []
                    for b in room_boxes:
                        try:
                            r = trimesh.boolean.difference([b, cutter])
                            if r.volume > 1e-6:
                                cut_boxes.append(r)
                            n_doors += 1
                        except Exception:
                            cut_boxes.append(b)
                    room_boxes = cut_boxes
            panels.extend(room_boxes)
            n_panels += len(room_boxes)

        floor_panel = None
        try:
            floor_panel = trimesh.creation.extrude_polygon(room, PANEL_T)
            floor_panel.apply_translation([0, 0, z_floor - PANEL_T])
        except Exception:
            pass
        group = list(panels)
        if floor_panel is not None:
            group.append(floor_panel)
        if group:
            elements[f"Room_{ri:02d}_{room.area:.0f}m2"] = group

    col_solids = []
    for c in cols:
        try:
            col_solids.append(column_to_solid(c))
        except Exception:
            pass
    if col_solids:
        elements["Columns"] = col_solids

    scene = build_scene(elements)
    out = OUT_DIR + r"\model_rooms_v4.glb"
    write_glb(scene, out)
    log(f"wrote {out}: {len(rooms)} rooms, {n_panels} interior wall panels, "
        f"{n_doors} door cuts, {len(col_solids)} columns")


if __name__ == "__main__":
    main()
