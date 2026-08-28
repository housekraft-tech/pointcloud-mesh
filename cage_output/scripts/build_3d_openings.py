"""Cut the detected doors, windows and balcony doors into the solid model.

Each opening becomes a box spanning the full wall thickness, subtracted from
the wall solid. Its width and head/sill heights are the measured void extents,
so the reveal you see is the one the scanner found -- not a catalogue size
dropped onto a wall.
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
sys.path.insert(0, str(HERE))
from build_3d import as_polys, extrude, EXTERIOR_SKIN, SLAB  # noqa: E402

# how far each cut box reaches through the wall, either side of the face
CUT_DEPTH = 0.45

COLOUR = {
    "door":          [70, 130, 220, 255],
    "balcony_door":  [235, 150, 40, 255],
    "wide_opening":  [120, 200, 130, 255],
    "window":        [90, 200, 220, 255],
    "high_level_void": [200, 110, 200, 255],
}


def cut_box(o):
    """A box through the wall at this opening's measured extents."""
    p0 = np.array(o["p0"], dtype=float)
    p1 = np.array(o["p1"], dtype=float)
    mid = 0.5 * (p0 + p1)
    d = p1 - p0
    L = float(np.linalg.norm(d))
    if L < 1e-6:
        return None
    u = d / L
    n = np.array([-u[1], u[0]])
    z0, z1 = o["z0"], o["z1"]

    half_u, half_n, half_z = L / 2, CUT_DEPTH, (z1 - z0) / 2
    box = trimesh.creation.box(extents=(2 * half_u, 2 * half_n, 2 * half_z))
    T = np.eye(4)
    T[:3, 0] = [u[0], u[1], 0]
    T[:3, 1] = [n[0], n[1], 0]
    T[:3, 2] = [0, 0, 1]
    T[:3, 3] = [mid[0], mid[1], 0.5 * (z0 + z1)]
    box.apply_transform(T)
    return box


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
    ops = json.loads((BASE / "openings" /
                      f"openings_{a.run}.json").read_text())["openings"]
    dropped = [o for o in ops if not o.get("confident", True)]
    ops = [o for o in ops if o.get("confident", True)]
    if dropped:
        print(f"excluding {len(dropped)} flagged opening(s) from the cut: "
              + ", ".join(f"{o['kind']} {1000*o['width_m']:.0f} mm" for o in dropped),
              flush=True)
    zf, zc = hz["floor_z"], hz["ceiling_z"]

    rooms = []
    for r in snap["rooms"]:
        p = Polygon(r["polygon_m"])
        if not p.is_valid:
            p = p.buffer(0)
        rooms += [q for q in as_polys(p) if q.area > 0.4]

    interior = unary_union(rooms)
    shell = interior.buffer(EXTERIOR_SKIN, join_style=2)
    walls_2d = shell.difference(interior)

    wall_mesh = trimesh.util.concatenate(
        extrude(walls_2d, zf, zc, [214, 210, 200, 255]))
    print(f"wall solid: {len(wall_mesh.faces):,} faces", flush=True)

    boxes = [b for b in (cut_box(o) for o in ops) if b is not None]
    print(f"cutting {len(boxes)} openings...", flush=True)
    cutter = trimesh.util.concatenate(boxes)
    cut_walls = trimesh.boolean.difference([wall_mesh, cutter], engine="manifold")
    cut_walls.visual.face_colors = [214, 210, 200, 255]
    print(f"after cut: {len(cut_walls.faces):,} faces", flush=True)

    floor = trimesh.util.concatenate(extrude(shell, zf - SLAB, zf,
                                             [150, 146, 140, 255]))
    ceiling = trimesh.util.concatenate(extrude(shell, zc, zc + SLAB,
                                               [188, 184, 176, 255]))

    solid = trimesh.util.concatenate([cut_walls, floor, ceiling])
    solid.export(OUT / "flat_openings.glb")
    solid.export(OUT / "flat_openings.obj")

    cutaway = trimesh.util.concatenate([cut_walls, floor])
    cutaway.export(OUT / "flat_openings_cutaway.glb")

    # the openings themselves as coloured panels, for a legend view
    panels = []
    for o in ops:
        b = cut_box(o)
        if b is None:
            continue
        b = b.copy()
        # thin the panel so it reads as an infill, not a block
        c = b.centroid
        S = np.eye(4)
        p0, p1 = np.array(o["p0"]), np.array(o["p1"])
        d = p1 - p0; d /= (np.linalg.norm(d) + 1e-9)
        n = np.array([-d[1], d[0], 0.0])
        b.vertices = c + (b.vertices - c) - np.outer(
            (b.vertices - c) @ n, n) * 0.88
        b.visual.face_colors = COLOUR.get(o["kind"], [160, 160, 160, 255])
        panels.append(b)
    trimesh.util.concatenate(panels).export(OUT / "flat_opening_panels.glb")
    trimesh.util.concatenate([cut_walls, floor] + panels).export(
        OUT / "flat_openings_marked.glb")

    import collections
    stats = {
        "run": a.run,
        "n_openings": len(ops),
        "by_kind": dict(collections.Counter(o["kind"] for o in ops)),
        "total_opening_area_m2": round(
            float(sum(o["width_m"] * o["height_m"] for o in ops)), 2),
        "faces": int(len(solid.faces)),
        "watertight": bool(cut_walls.is_watertight),
    }
    (OUT / "openings_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
