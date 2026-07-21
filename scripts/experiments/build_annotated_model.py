"""build_annotated_model.py
-------------------------
Emit ONE 3D model carrying everything we have measured: the modular house
(per-wall / shared floor / per-level ceiling parts / columns) plus a named,
correctly-placed solid for every opening -- doors, balcony doors, windows,
archways -- so the openings are objects you can see and toggle in 3D rather
than numbers in a JSON.

Openings come from two sources, and both are kept:
  blind   enclosed holes found in the wall's occupancy grid
  walked  crossings of the operator's path, measured at the seed (these are
          the ones blind detection missed, including every balcony door)

Each opening becomes a slab spanning the wall thickness, named
`<kind>_<NN>_<w>x<h>mm`, so the name carries the measurement.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\build_annotated_model.py \\
      <scan.las> <detailed_modular.obj> <measurements.json> <out_dir>
"""
import sys, json, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.wall_elevations import (
    parse_obj, merge_coplanar, grids, find_openings, CELL)
from scripts.experiments import walk_path_openings as W

THICK = 0.16          # m: slab thickness (through a ~110 mm wall, both faces)
KIND_TAG = {
    "door": "door",
    "arched door": "door_arched",
    "balcony / sliding door": "balcony_door",
    "window": "window",
    "archway / open passage": "archway",
}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def parse_obj_faces(path):
    """Full parse keeping faces, so the source objects survive into the output."""
    V = []; groups = []
    for ln in open(path):
        if ln.startswith("o "):
            groups.append([ln[2:].strip(), []])
        elif ln.startswith("v "):
            _, x, y, z = ln.split()[:4]
            V.append((float(x), float(y), float(z)))
        elif ln.startswith("f ") and groups:
            groups[-1][1].append([int(t.split("/")[0]) - 1 for t in ln.split()[1:4]])
    return np.asarray(V), groups


def box(c2, n2, d2, a_lo, a_hi, z_lo, z_hi, thick=THICK):
    """Axis-aligned-in-plane slab: spans the opening in-plane and the wall
    thickness out-of-plane."""
    corners = []
    for a in (a_lo, a_hi):
        for s in (-thick / 2, thick / 2):
            p = c2 + d2 * a + n2 * s
            for z in (z_lo, z_hi):
                corners.append((p[0], p[1], z))
    V = np.array(corners)          # order: (a, s, z) with z fastest
    # 8 corners indexed [a][s][z] -> flat index a*4 + s*2 + z
    f = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5),
         (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3),
         (2, 3, 7), (2, 7, 6), (0, 4, 5), (0, 5, 1)]
    return V, np.array(f)


def tag(kind, i, w, h):
    base = KIND_TAG.get(kind)
    if base is None:
        base = ("opening" if "opening" in kind else
                "void" if "void" in kind else "feature")
    return f"{base}_{i:02d}_{w*1000:.0f}x{h*1000:.0f}mm"


def main(las_path, obj_path, mj, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    d = json.load(open(mj))
    z_floor = float(np.median([r["z_floor"] for r in d["rooms"]]))
    z_ceil = float(np.median([r["z_ceiling"] for r in d["rooms"]]))

    path = W.walk_path(las_path)
    Vp = parse_obj(obj_path)
    G = merge_coplanar({n: P for n, P in Vp.items() if n.startswith("wall")})
    log(f"{len(G)} wall planes")

    openings = []
    for name, P in G.items():
        if len(P) < 2000:
            continue
        c2, n2, d2 = W.plane_of(P)
        a = (P[:, :2] - c2) @ d2
        rel, occ, a0, z0, na, nz = grids(P)
        xs = W.crossings(path, c2, n2, d2, float(a.min()), float(a.max()))

        blind = find_openings(occ, a0, z0, z_floor, z_ceil)
        for h in blind:
            if "void" in h["type"]:
                continue                      # occlusion, not a real opening
            ab, zb, w, hh = h["_box"]
            walked = any(ab - W.MATCH_TOL <= x[0] <= ab + w + W.MATCH_TOL for x in xs)
            openings.append(dict(wall=name, kind=h["type"], c2=c2, n2=n2, d2=d2,
                                 a_lo=ab, a_hi=ab + w, z_lo=zb, z_hi=zb + hh,
                                 w=w, h=hh, source="blind",
                                 walk_confirmed=bool(walked)))
        # walk-seeded: the ones blind detection missed
        for ac, zc, cnt in xs:
            if any(o["wall"] == name and o["a_lo"] - W.MATCH_TOL <= ac <= o["a_hi"] + W.MATCH_TOL
                   for o in openings):
                continue
            state, _, _ = W.column_state(occ, a0, z0, ac, z_floor, z_ceil)
            if state != "opening":
                continue
            m = W.measure_at(occ, a0, z0, ac, z_floor, z_ceil)
            if m is None:
                continue
            alo, ahi = m["along_span"]
            zlo = z_floor + m["sill_mm"] / 1000.0
            zhi = z_floor + m["head_mm"] / 1000.0
            openings.append(dict(wall=name, kind=m["kind"], c2=c2, n2=n2, d2=d2,
                                 a_lo=alo, a_hi=ahi, z_lo=zlo, z_hi=zhi,
                                 w=(ahi - alo), h=(zhi - zlo), source="walked",
                                 walk_confirmed=True))

    log(f"{len(openings)} openings to place "
        f"({sum(o['source']=='walked' for o in openings)} walk-seeded)")

    # ---- assemble
    V0, groups = parse_obj_faces(obj_path)
    obj_out = out / "annotated_model.obj"
    pieces = []
    with open(obj_out, "w") as fh:
        fh.write("# modular house + measured openings as named objects\n")
        voff = 0
        for name, faces in groups:
            if not faces:
                continue
            f = np.array(faces)
            used = np.unique(f)
            remap = -np.ones(len(V0), np.int64); remap[used] = np.arange(len(used))
            fh.write(f"o {name}\n")
            for p in V0[used]:
                fh.write(f"v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")
            for t in remap[f]:
                fh.write(f"f {t[0]+voff+1} {t[1]+voff+1} {t[2]+voff+1}\n")
            voff += len(used)
            pieces.append((name, V0[used], remap[f]))

        counters = {}
        manifest = []
        for o in openings:
            base = KIND_TAG.get(o["kind"], "opening")
            counters[base] = counters.get(base, 0) + 1
            nm = tag(o["kind"], counters[base], o["w"], o["h"])
            Vb, Fb = box(o["c2"], o["n2"], o["d2"], o["a_lo"], o["a_hi"],
                         o["z_lo"], o["z_hi"])
            fh.write(f"o {nm}\n")
            for p in Vb:
                fh.write(f"v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")
            for t in Fb:
                fh.write(f"f {t[0]+voff+1} {t[1]+voff+1} {t[2]+voff+1}\n")
            voff += len(Vb)
            pieces.append((nm, Vb, Fb))
            manifest.append(dict(name=nm, kind=o["kind"], wall=o["wall"],
                                 width_mm=round(o["w"] * 1000),
                                 height_mm=round(o["h"] * 1000),
                                 sill_mm=round((o["z_lo"] - z_floor) * 1000),
                                 head_mm=round((o["z_hi"] - z_floor) * 1000),
                                 source=o["source"],
                                 walk_confirmed=o["walk_confirmed"]))
    log(f"wrote {obj_out} ({voff:,} verts, {len(pieces)} objects)")

    try:
        import trimesh
        scene = trimesh.Scene()
        for nm, Vv, Ff in pieces:
            if len(Ff) == 0:
                continue
            m = trimesh.Trimesh(Vv, Ff, process=False)
            scene.add_geometry(m, geom_name=nm, node_name=nm)
        scene.export(str(out / "annotated_model.glb"))
        log(f"wrote {out/'annotated_model.glb'}")
    except Exception as e:
        log(f"glb skipped: {e}")

    json.dump(dict(openings=manifest), open(out / "annotated_manifest.json", "w"),
              indent=1)
    from collections import Counter
    c = Counter(m["kind"] for m in manifest)
    log("OPENINGS: " + ", ".join(f"{k}={v}" for k, v in c.most_common()))
    log(f"wrote {out/'annotated_manifest.json'}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
