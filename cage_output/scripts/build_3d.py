"""Extrude the snapped rooms into a solid 3D model.

Every surface in this model is measured, not assumed:
  * wall faces      -- recon.metrology.detect_wall_faces, ~0.16 mm stderr
  * floor / ceiling -- the same refine_face on the z histogram
  * wall thickness  -- the gap between two rooms' snapped boundaries IS the
                       measured face pair, so interior walls come out at their
                       true thickness rather than a nominal one.

Only the exterior skin is nominal, since nothing was scanned behind it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OUT = BASE / "model3d"

EXTERIOR_SKIN = 0.125   # nominal: there is no scan data outside the flat
SLAB = 0.14             # drawn thickness for the floor / ceiling slabs


def as_polys(geom):
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]


def extrude(geom, z0, z1, colour):
    parts = []
    for p in as_polys(geom):
        if p.area < 1e-4:
            continue
        m = trimesh.creation.extrude_polygon(p, height=float(z1 - z0))
        m.apply_translation((0, 0, float(z0)))
        m.visual.face_colors = colour
        parts.append(m)
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="robust_c95_swin")
    ap.add_argument("--mode", default="strict")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    snap = json.loads((BASE / "snapped" /
                       f"snapped_{a.run}_{a.mode}.json").read_text())
    heights = json.loads((BASE / "snapped" /
                          f"snapped_{a.run}_hybrid.json").read_text())["report"]
    zf, zc = heights["floor_z"], heights["ceiling_z"]
    print(f"floor {zf:.4f} m  ceiling {zc:.4f} m  clear "
          f"{1000*(zc-zf):.1f} mm", flush=True)

    rooms = []
    for r in snap["rooms"]:
        p = Polygon(r["polygon_m"])
        if not p.is_valid:
            p = p.buffer(0)
        for q in as_polys(p):
            if q.area > 0.4:
                rooms.append(q)
    print(f"{len(rooms)} rooms, {sum(r.area for r in rooms):.1f} m2 net",
          flush=True)

    interior = unary_union(rooms)
    shell = interior.buffer(EXTERIOR_SKIN, join_style=2)
    walls = shell.difference(interior)

    meshes = []
    meshes += extrude(walls, zf, zc, [214, 210, 200, 255])
    meshes += extrude(shell, zf - SLAB, zf, [150, 146, 140, 255])
    meshes += extrude(shell, zc, zc + SLAB, [188, 184, 176, 255])
    solid = trimesh.util.concatenate(meshes)
    solid.export(OUT / "flat_solid.glb")
    solid.export(OUT / "flat_solid.obj")
    print(f"solid: {len(solid.faces):,} faces -> flat_solid.glb / .obj", flush=True)

    # rooms as coloured volumes, for the exploded / cutaway view
    palette = [[66,133,244],[219,68,55],[244,180,0],[15,157,88],[171,71,188],
               [0,172,193],[255,112,67],[158,157,36],[94,53,177],[240,98,146],
               [0,151,167],[124,179,66],[255,167,38],[84,110,122],[233,30,99]]
    vol = []
    for i, r in enumerate(rooms):
        c = palette[i % len(palette)] + [255]
        vol += extrude(r, zf, zc, c)
    volumes = trimesh.util.concatenate(vol)
    volumes.export(OUT / "flat_rooms.glb")
    print(f"rooms: {len(volumes.faces):,} faces -> flat_rooms.glb", flush=True)

    # cutaway: walls + floor, no ceiling, so you can see in from above
    cut = trimesh.util.concatenate(
        extrude(walls, zf, zc, [214, 210, 200, 255]) +
        extrude(shell, zf - SLAB, zf, [150, 146, 140, 255]))
    cut.export(OUT / "flat_cutaway.glb")

    stats = {
        "run": a.run, "mode": a.mode,
        "n_rooms": len(rooms),
        "net_internal_area_m2": round(float(sum(r.area for r in rooms)), 2),
        "footprint_area_m2": round(float(shell.area), 2),
        "wall_area_m2": round(float(walls.area), 2),
        "floor_z": zf, "ceiling_z": zc,
        "clear_height_mm": heights["clear_height_mm"],
        "volume_m3": round(float(sum(r.area for r in rooms) * (zc - zf)), 2),
        "faces_solid": int(len(solid.faces)),
        "exterior_skin_m": EXTERIOR_SKIN,
    }
    (OUT / "model_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
