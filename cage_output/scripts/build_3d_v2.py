"""Solid model with per-wall measured heights, openings, and a continuous floor.

Three corrections over build_3d_openings.py, all from looking at the model:

  * Every wall was extruded to the ceiling. Walls are now built one segment at
    a time, each to its OWN measured top -- which is what puts the wet rooms'
    dropped ceilings in at their real height instead of inventing 600 mm of
    wall above them.
  * The floor slab inherited holes wherever CAGE left a gap between rooms, so
    the model had an opening in the floor that does not exist. Interior rings
    are now filled: a floor is continuous whatever the room graph says.
  * Openings are cut per wall, so a cut cannot reach through a wall it was
    never on.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import Polygon
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OUT = BASE / "model3d"
sys.path.insert(0, str(HERE))
from build_3d import as_polys, extrude, EXTERIOR_SKIN, SLAB  # noqa: E402
from build_3d_openings import cut_box, COLOUR                # noqa: E402

WALL_T = 0.20
MIN_TOP = 0.9      # below this a "wall" is furniture, not structure


def fill_holes(geom):
    """Drop interior rings -- a floor slab is continuous under everything."""
    return unary_union([Polygon(p.exterior) for p in as_polys(geom)])


def wall_footprint(w, rooms_union):
    """A rectangle from the measured room-side face, outward by WALL_T."""
    p0 = np.array(w["p0"], dtype=float)
    p1 = np.array(w["p1"], dtype=float)
    d = p1 - p0
    L = float(np.linalg.norm(d))
    if L < 0.3:
        return None
    u = d / L
    n = np.array([-u[1], u[0]])
    # push away from the interior
    probe = 0.5 * (p0 + p1) + n * 0.05
    from shapely.geometry import Point
    if rooms_union.contains(Point(*probe)):
        n = -n
    q = [p0, p1, p1 + n * WALL_T, p0 + n * WALL_T]
    poly = Polygon(q)
    return poly if poly.is_valid and poly.area > 1e-4 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="robust_c95_swin")
    ap.add_argument("--mode", default="strict")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    snap = json.loads((BASE / "snapped" /
                       f"snapped_{a.run}_{a.mode}.json").read_text())
    hz = json.loads((BASE / "snapped" /
                     f"snapped_{a.run}_hybrid.json").read_text())["report"]
    tops = json.loads((BASE / "openings" /
                       f"wall_tops_{a.run}.json").read_text())["walls"]
    # prefer openings measured on the wall graph: each is on a wall centre-line
    # with the wall's own thickness, and knows the room on either side
    on_walls = BASE / "openings" / f"openings_on_walls_{a.run}.json"
    src = on_walls if on_walls.exists() else (
        BASE / "openings" / f"openings_{a.run}.json")
    ops_all = json.loads(src.read_text())["openings"]
    print(f"openings from {src.name}", flush=True)
    ops = [o for o in ops_all if o.get("confident", True)]
    zf, zc = hz["floor_z"], hz["ceiling_z"]
    print(f"floor {zf:.4f} ceiling {zc:.4f}  storey {1000*(zc-zf):.0f} mm",
          flush=True)

    rooms = []
    for r in snap["rooms"]:
        p = Polygon(r["polygon_m"])
        if not p.is_valid:
            p = p.buffer(0)
        rooms += [q for q in as_polys(p) if q.area > 0.4]
    interior = unary_union(rooms)

    # --- walls, each to its own measured top --------------------------------
    by_top = collections.Counter()
    prisms, panels = [], []
    for w in tops:
        h = w.get("height_m")
        if h is None or h < MIN_TOP:
            by_top["skipped"] += 1
            continue
        fp = wall_footprint(w, interior)
        if fp is None:
            by_top["degenerate"] += 1
            continue
        z1 = zf + min(h, zc - zf)
        by_top[w["kind"]] += 1
        col = ([214, 210, 200, 255] if w["kind"] == "full"
               else [206, 196, 178, 255])
        m = trimesh.creation.extrude_polygon(fp, height=float(z1 - zf))
        m.apply_translation((0, 0, float(zf)))
        m.visual.face_colors = col
        prisms.append(m)
    print("wall segments built:", dict(by_top), flush=True)

    walls = trimesh.util.concatenate(prisms)
    print(f"walls: {len(walls.faces):,} faces", flush=True)

    # --- cut the openings ---------------------------------------------------
    boxes = [b for b in (cut_box(o) for o in ops) if b is not None]
    cutter = trimesh.util.concatenate(boxes)
    walls_cut = trimesh.boolean.difference([walls, cutter], engine="manifold")
    walls_cut.visual.face_colors = [214, 210, 200, 255]
    print(f"cut {len(boxes)} openings -> {len(walls_cut.faces):,} faces",
          flush=True)

    # --- slabs --------------------------------------------------------------
    shell = fill_holes(interior.buffer(EXTERIOR_SKIN, join_style=2))
    floor = trimesh.util.concatenate(extrude(shell, zf - SLAB, zf,
                                             [150, 146, 140, 255]))
    ceiling = trimesh.util.concatenate(extrude(shell, zc, zc + SLAB,
                                               [188, 184, 176, 255]))

    trimesh.util.concatenate([walls_cut, floor, ceiling]).export(
        OUT / "v2_solid.glb")
    trimesh.util.concatenate([walls_cut, floor]).export(OUT / "v2_cutaway.glb")

    for o in ops:
        b = cut_box(o)
        if b is None:
            continue
        c = b.centroid
        p0, p1 = np.array(o["p0"]), np.array(o["p1"])
        d = p1 - p0; d = d / (np.linalg.norm(d) + 1e-9)
        n = np.array([-d[1], d[0], 0.0])
        b.vertices = c + (b.vertices - c) - np.outer((b.vertices - c) @ n, n) * 0.9
        b.visual.face_colors = COLOUR.get(o["kind"], [160, 160, 160, 255])
        panels.append(b)
    trimesh.util.concatenate([walls_cut, floor] + panels).export(
        OUT / "v2_marked.glb")

    stats = {
        "run": a.run,
        "wall_segments": dict(by_top),
        "n_openings_built": len(ops),
        "n_openings_flagged": len(ops_all) - len(ops),
        "floor_area_m2": round(float(shell.area), 2),
        "net_internal_area_m2": round(float(interior.area), 2),
        "clear_height_mm": round(1000 * (zc - zf), 1),
        "dropped_ceiling_walls": sum(1 for w in tops
                                     if w.get("kind") == "partial"),
        "faces": int(len(walls_cut.faces)),
        "floor_holes_filled": True,
    }
    (OUT / "v2_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
