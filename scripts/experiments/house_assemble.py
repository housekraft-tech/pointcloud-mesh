"""house_assemble.py
----------------
STAGES 3-4 of the house rebuild: put the measured parts back together as a
solid, per-object 3D house.

Every wall is built as a 2D polygon in (along, height) space, its openings are
subtracted as holes in that polygon, and the result is extruded through the
wall's measured thickness. Doing it in 2D is what makes real openings possible
without CSG on meshes: a door, a window with a sill, and an ARCHED head are all
just different holes in the same polygon, and the extrusion carries them
through the wall automatically.

What each source contributes -- the same split used throughout the pipeline:
  major_walls.json   (stage 1) wall runs and parapets, from mesh slices
  entrance_path.json (stage 2) the front door and the walked route
  features.json      (Poisson relief) beams, pilasters, grooves, niches, ducts
  annotated_model    the openings the LiDAR actually cut, with sill and head
  fused_detections   room names, for labelling the parts

Nothing here is idealised: wall positions, thicknesses, opening sills and heads,
beam drops and column footprints are all the measured values.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\house_assemble.py \\
      <major_walls.json> <entrance_path.json> <features.json> \\
      <annotated_model.obj> <fused_detections.json> <out_dir>
"""
import sys, json, time, re
from pathlib import Path
import numpy as np
import trimesh
from shapely.geometry import Polygon, box as shp_box
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.experiments.viz_deviation_plan import parse_groups
from scripts.experiments.classify_openings_rgb import parse_named_boxes, frame_of
from scripts.experiments.walk_path_openings import plane_of

ON_WALL = 0.45        # m: an opening within this of a wall line belongs to it
EDGE_IN = 0.05        # m: opening must start this far inside the run
MIN_PIER = 0.06       # m: ignore slivers of wall left beside an opening
ARCH_KINDS = ("archway", "door_arched")
BEAM_MIN_DROP = 0.12  # m: a ceiling plateau this far below the main slab is a beam
RELIEF_MIN = 0.04     # m: relief shallower than this is roughness, not a feature
RELIEF_KINDS = {"column/pilaster": "pilaster", "beam soffit": "beamsoffit",
                "niche/recess": "niche", "duct/chase": "duct",
                "protrusion": "protrusion"}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def wall_mesh(length, z0, z1, thick, holes):
    """(along, height) face polygon minus its openings, extruded by thickness."""
    face = shp_box(0, z0, length, z1)
    if holes:
        face = face.difference(unary_union(holes))
    if face.is_empty:
        return None
    parts = list(face.geoms) if hasattr(face, "geoms") else [face]
    meshes = []
    for p in parts:
        if p.area < 0.01:
            continue
        try:
            m = trimesh.creation.extrude_polygon(p, thick)
        except Exception:
            continue
        meshes.append(m)
    if not meshes:
        return None
    m = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    # extrude gives (x=along, y=height, z=thickness); stand it up
    m.apply_transform(trimesh.transformations.rotation_matrix(
        np.pi / 2, [1, 0, 0]))
    return m


def place(m, origin, direction, thick):
    """Put a wall mesh at its run: +x along the run, centred on the line."""
    ang = np.arctan2(direction[1], direction[0])
    m.apply_transform(trimesh.transformations.rotation_matrix(ang, [0, 0, 1]))
    n = np.array([-direction[1], direction[0]])
    off = np.array([origin[0], origin[1], 0.0]) - np.array([n[0], n[1], 0]) * thick / 2
    m.apply_translation(off)
    return m


def arch_hole(a0, a1, sill, head):
    """A rectangular opening capped with a semicircular head."""
    w = a1 - a0
    r = w / 2.0
    spring = max(sill, head - r)
    rect = shp_box(a0, sill, a1, spring)
    cx = (a0 + a1) / 2.0
    th = np.linspace(0, np.pi, 24)
    cap = Polygon([(cx + r * np.cos(t), spring + r * np.sin(t)) for t in th] +
                  [(a0, spring)])
    return unary_union([rect, cap])


def main(walls_json, path_json, feats_json, obj_path, fused_json, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    W = json.load(open(walls_json))
    EP = json.load(open(path_json))
    F = json.load(open(feats_json))
    fused = json.load(open(fused_json))

    theta = np.radians(W["grid_angle_deg"])
    pivot = np.array(W["pivot"])
    z0, z1 = W["z_floor"], W["z_ceiling"]
    # stage 1 worked in the grid frame; rotate back to the model frame
    Rb = np.array([[np.cos(-theta), -np.sin(-theta)],
                   [np.sin(-theta), np.cos(-theta)]])

    def to_model(p):
        return (np.asarray(p, float) - pivot) @ Rb.T + pivot

    def to_grid(p):
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        return (np.asarray(p, float) - pivot) @ R.T + pivot

    # ---- openings the LiDAR cut, in the grid frame with sill and head
    ops = []
    for name, b in parse_named_boxes(str(obj_path),
                                     ("door", "balcony_door", "window",
                                      "archway", "opening")):
        b = np.asarray(b, float)
        c = b[:, :2].mean(0)
        sill, head = float(b[:, 2].min()), float(b[:, 2].max())
        wdt = float(np.linalg.norm(b[:, :2].max(0) - b[:, :2].min(0)))
        mm = re.search(r"_(\d+)x(\d+)mm", name)
        if mm:
            wdt = int(mm.group(1)) / 1000.0
        ops.append(dict(name=name, c=to_grid(c), sill=sill, head=head,
                        width=wdt,
                        arch=any(k in name for k in ARCH_KINDS)))
    log(f"{len(ops)} measured openings available to cut")

    scene = {}
    used_ops = set()

    # ---------- walls and parapets ----------
    n_holes = 0
    for i, s in enumerate(W["walls"]):
        p0 = np.array(s["p0"]); p1 = np.array(s["p1"])
        d = p1 - p0
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            continue
        u = d / L
        n = np.array([-u[1], u[0]])
        top = z1 if s["kind"] == "wall" else float(s.get("top_z") or 1.0)
        holes = []
        for k, o in enumerate(ops):
            rel = o["c"] - p0
            along = float(rel @ u)
            perp = abs(float(rel @ n))
            if perp > ON_WALL or not (EDGE_IN < along < L - EDGE_IN):
                continue
            a0 = max(0.0, along - o["width"] / 2)
            a1 = min(L, along + o["width"] / 2)
            if a1 - a0 < 0.25:
                continue
            sill = max(z0, min(o["sill"], top - 0.15))
            head = min(top, max(o["head"], sill + 0.3))
            holes.append(arch_hole(a0, a1, sill, head) if o["arch"]
                         else shp_box(a0, sill, a1, head))
            used_ops.add(k)
        m = wall_mesh(L, z0, top, s["thickness_m"], holes)
        if m is None:
            continue
        place(m, p0, u, s["thickness_m"])
        nm = f"{s['kind']}_{i:02d}_{L*1000:.0f}x{s['thickness_m']*1000:.0f}mm"
        if holes:
            nm += f"_{len(holes)}op"
        n_holes += len(holes)
        scene[nm] = m
    log(f"{len(scene)} wall/parapet solids, {n_holes} openings cut, "
        f"{len(used_ops)}/{len(ops)} openings placed on a wall")

    # ---------- columns / pillars ----------
    cols = parse_groups(str(obj_path), "column")
    for name, P in cols.items():
        g = to_grid(P[:, :2])
        w = max(float(np.ptp(g[:, 0])), 0.12)
        h = max(float(np.ptp(g[:, 1])), 0.12)
        zt = float(P[:, 2].max()); zb = float(P[:, 2].min())
        m = trimesh.creation.box(extents=[w, h, max(zt - zb, 0.3)])
        c = g.mean(0)
        m.apply_translation([c[0], c[1], (zb + zt) / 2])
        m.apply_transform(trimesh.transformations.rotation_matrix(
            -theta, [0, 0, 1], point=[pivot[0], pivot[1], 0]))
        scene[f"pillar_{name.split('_')[-1]}_{w*1000:.0f}x{h*1000:.0f}mm"] = m
    log(f"{len(cols)} pillars")

    # ---------- beams: ceiling plateaus below the main slab ----------
    ceil = parse_groups(str(obj_path), "ceiling")
    cinfo = {c["name"]: c for c in F["ceiling_features"]}
    main_h = max((c["height_mm"] for c in F["ceiling_features"]), default=2700)
    n_beam = 0
    for name, P in ceil.items():
        info = cinfo.get(name)
        if not info:
            continue
        drop = (main_h - info["height_mm"]) / 1000.0
        if drop < BEAM_MIN_DROP or info.get("area_m2", 0) < 0.25:
            continue
        g = to_grid(P[:, :2])
        w = float(np.ptp(g[:, 0])); h = float(np.ptp(g[:, 1]))
        if w < 0.15 or h < 0.15:
            continue
        m = trimesh.creation.box(extents=[w, h, drop])
        c = g.mean(0)
        m.apply_translation([c[0], c[1], z1 - drop / 2])
        m.apply_transform(trimesh.transformations.rotation_matrix(
            -theta, [0, 0, 1], point=[pivot[0], pivot[1], 0]))
        kind = "beam" if drop < 0.45 else "droppedceiling"
        scene[f"{kind}_{name.split('_')[-1]}_{drop*1000:.0f}mm"] = m
        n_beam += 1
    log(f"{n_beam} beams / dropped ceilings")

    # ---------- relief: pilasters, grooves, niches, ducts ----------
    wall_pts = parse_groups(str(obj_path), "wall")
    n_rel = 0
    for wf in F["walls"]:
        P = wall_pts.get(wf["name"])
        if P is None or not wf.get("features"):
            continue
        c, nrm, dv = plane_of(P)
        for k, ft in enumerate(wf["features"]):
            kind = RELIEF_KINDS.get(ft["type"])
            depth = abs(ft.get("depth_mm", 0)) / 1000.0
            wdt = ft.get("width_m", 0); hgt = ft.get("height_m", 0)
            if kind is None or depth < RELIEF_MIN or wdt < 0.08 or hgt < 0.08:
                continue
            if wdt > 6.0:                 # a "feature" as long as the wall is
                continue                  # the wall's own curvature, not relief
            a = ft["along_start"] + wdt / 2.0
            xy = c + dv * a
            zc = (ft["z_bottom"] + ft["z_top"]) / 2.0
            m = trimesh.creation.box(extents=[wdt, max(depth, 0.02), hgt])
            ang = np.arctan2(dv[1], dv[0])
            m.apply_transform(trimesh.transformations.rotation_matrix(
                ang, [0, 0, 1]))
            m.apply_translation([xy[0], xy[1], zc])
            scene[f"{kind}_{wf['name']}_{k:02d}_{depth*1000:.0f}mm"] = m
            n_rel += 1
    log(f"{n_rel} relief features (pilasters, grooves, niches, ducts, soffits)")

    # ---------- floor and ceiling slabs ----------
    # The slab must follow the flat's real outline. A bounding box was simple
    # but wrong: it projected past every wall and read as a plinth the building
    # does not have.
    from scripts.experiments.house_entrance_path import envelope_of
    try:
        env = envelope_of(obj_path)
        for nm, zc, th in (("floor_slab", z0 - 0.06, 0.12),
                           ("ceiling_slab", z1 + 0.06, 0.12)):
            m = trimesh.creation.extrude_polygon(env, th)
            m.apply_translation([0, 0, zc - th / 2])
            scene[nm] = m
        log(f"floor and ceiling slabs from the measured footprint "
            f"({env.area:.1f} m2)")
    except Exception as e_:
        log(f"slab from footprint failed ({e_}); skipped")

    # ---------- entrance marker and walked route ----------
    ent = EP["entrance"]
    e = np.array(ent["xy"], float)
    mk = trimesh.creation.box(extents=[0.30, 0.30, 2.1])
    mk.apply_translation([e[0], e[1], z0 + 1.05])
    scene[f"ENTRANCE_{ent.get('opening','from_walk')}"] = mk

    path = np.array(EP["path"], float)
    segs = []
    for a, b in zip(path[:-1], path[1:]):
        d = b[:2] - a[:2]
        L = float(np.linalg.norm(d))
        if L < 0.02:
            continue
        seg = trimesh.creation.box(extents=[L, 0.06, 0.02])
        seg.apply_transform(trimesh.transformations.rotation_matrix(
            np.arctan2(d[1], d[0]), [0, 0, 1]))
        seg.apply_translation([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, z0 + 0.02])
        segs.append(seg)
    if segs:
        scene["WALK_PATH"] = trimesh.util.concatenate(segs)
    log(f"walk route as {len(segs)} segments")

    # ---------- export ----------
    obj_out = out / "house.obj"
    with open(obj_out, "w") as fh:
        v0 = 1
        for name, m in scene.items():
            fh.write(f"o {name}\n")
            for v in m.vertices:
                fh.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
            for f_ in m.faces:
                fh.write(f"f {f_[0]+v0} {f_[1]+v0} {f_[2]+v0}\n")
            v0 += len(m.vertices)
    log(f"wrote {obj_out}  ({len(scene)} named objects)")

    # The same house without its lid. With a ceiling slab present the standard
    # cutaway clips at max_z-0.6 and beheads every wall, so the view that
    # actually shows the rebuilt interior is this one.
    open_out = out / "house_open.obj"
    with open(open_out, "w") as fh:
        v0 = 1
        for name, m in scene.items():
            if name.startswith("ceiling_slab"):
                continue
            fh.write(f"o {name}\n")
            for v in m.vertices:
                fh.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
            for f_ in m.faces:
                fh.write(f"f {f_[0]+v0} {f_[1]+v0} {f_[2]+v0}\n")
            v0 += len(m.vertices)
    log(f"wrote {open_out}  (no ceiling slab -- the interior view)")

    try:
        sc = trimesh.Scene()
        for name, m in scene.items():
            sc.add_geometry(m, geom_name=name, node_name=name)
        sc.export(str(out / "house.glb"))
        log(f"wrote {out/'house.glb'}")
    except Exception as e_:
        log(f"glb skipped: {e_}")

    from collections import Counter
    kinds = Counter(n.split("_")[0] for n in scene)
    json.dump(dict(objects=len(scene), by_kind=dict(kinds),
                   openings_cut=n_holes,
                   entrance=ent),
              open(out / "house_manifest.json", "w"), indent=1)
    log("parts: " + ", ".join(f"{k} {v}" for k, v in kinds.most_common()))


if __name__ == "__main__":
    main(*sys.argv[1:7])
