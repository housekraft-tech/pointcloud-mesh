"""Export the model in the formats SketchUp opens, and a 2D plan to trace.

There is no open writer for .skp -- the format is closed and the only way to
create one is SketchUp's own SDK. What there is instead is a one-click path:
SketchUp imports Collada and DXF directly, and File > Save As then gives a .skp.
So this writes what SketchUp can take, keeping the parts named so they arrive as
separate groups rather than one welded lump.

Three files, in rising order of how much they leave out:

  sketch.dae   the solid walls and slabs, one named node each. This is the
               model to open in SketchUp: a few thousand faces, not millions.
  sketch.dxf   a 2D plan -- wall footprints, openings and slab outlines on
               separate layers, with the measured dimensions as text. This is
               what a drafter traces over.
  sketch.stl   the same solids welded into one mesh, for the web version of
               SketchUp, which imports STL but not Collada.

The Poisson surface is deliberately NOT here. It is 66 million triangles for one
storey of this building, and SketchUp is unusable past about a million: it would
not open, and if it did there would be nothing to edit. The relief lives in
modular.obj and in the viewer; this is the drawing.
"""
import sys, json, argparse, time
from pathlib import Path
import numpy as np

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

CUT = 1.2             # m over the floor: the height the 2D plan is cut at


def read_groups(obj_path):
    """The solid file, back as {name: (vertices, faces)}."""
    parts = {}
    name, V, F, base = None, [], [], 0
    for line in open(obj_path):
        if line.startswith("o "):
            if name and F:
                parts[name] = (np.array(V), np.array(F) - base)
                base += len(V)
            name = line[2:].strip(); V, F = [], []
        elif line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            F.append([int(x.split("/")[0]) - 1 for x in line.split()[1:4]])
    if name and F:
        parts[name] = (np.array(V), np.array(F) - base)
    return parts


def write_dae(parts, man, path):
    """Collada, written directly: one named node per part, Z up, metres.

    trimesh can export a Collada mesh but not a Collada SCENE, and a scene is
    the whole point -- SketchUp turns each node into a named group, so the model
    arrives as walls and slabs you can click, not one welded lump.
    """
    from collada import Collada, source, geometry, material, scene
    kinds = {p["name"]: p["kind"] for p in man["parts"]}
    KC = {"wall": (0.72, 0.70, 0.66), "parapet": (0.62, 0.68, 0.72),
          "floor": (0.75, 0.72, 0.67), "ceiling": (0.80, 0.80, 0.78),
          "beam": (0.70, 0.55, 0.35), "dropped_ceiling": (0.78, 0.74, 0.55),
          "column": (0.65, 0.45, 0.45)}
    doc = Collada()
    doc.assetInfo.unitname = "meter"
    doc.assetInfo.unitmeter = 1.0
    doc.assetInfo.upaxis = "Z_UP"
    mats = {}
    for k, rgb in KC.items():
        eff = material.Effect(f"eff-{k}", [], "lambert",
                              diffuse=(*rgb, 1.0), ambient=(0.1, 0.1, 0.1, 1.0))
        mat = material.Material(f"mat-{k}", k, eff)
        doc.effects.append(eff); doc.materials.append(mat)
        mats[k] = mat
    nodes = []
    for nm, (V, F) in parts.items():
        k = kinds.get(nm, "wall")
        vs = source.FloatSource(f"{nm}-v", np.asarray(V, np.float32).ravel(),
                                ("X", "Y", "Z"))
        geo = geometry.Geometry(doc, f"geo-{nm}", nm, [vs])
        il = source.InputList()
        il.addInput(0, "VERTEX", f"#{nm}-v")
        tri = geo.createTriangleSet(np.asarray(F, np.int32).ravel(), il, "ref")
        geo.primitives.append(tri)
        doc.geometries.append(geo)
        gn = scene.GeometryNode(geo, [scene.MaterialNode("ref", mats[k], inputs=[])])
        nodes.append(scene.Node(f"node-{nm}", children=[gn], name=nm))
    sc = scene.Scene("scene", nodes)
    doc.scenes.append(sc); doc.scene = sc
    doc.write(str(path))
    log(f"wrote {path.name}: {len(parts)} named groups, "
        f"{sum(len(f) for _, f in parts.values()):,} faces, "
        f"{path.stat().st_size/1e6:.1f} MB")


def write_stl(parts, path):
    import trimesh
    ms = [trimesh.Trimesh(V, F, process=False) for V, F in parts.values()]
    trimesh.util.concatenate(ms).export(str(path))
    log(f"wrote {path.name} ({path.stat().st_size/1e6:.1f} MB)")


def write_dxf(man, path, cut_z):
    """A 2D plan: wall footprints, openings, slabs, and the dimensions."""
    import ezdxf
    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.M
    msp = doc.modelspace()
    for layer, col in (("WALLS", 7), ("OPENINGS", 1), ("SLABS", 8),
                       ("DIMENSIONS", 3), ("LABELS", 4)):
        doc.layers.add(layer, color=col)

    walls = [p for p in man["parts"] if p["kind"] in ("wall", "parapet")
             and "across_m" in p]
    for p in walls:
        ax = 0 if p["axis"] == "x" else 1
        c0, c1 = p["across_m"]; s0, s1 = p["along_m"]
        pts = ([(c0, s0), (c1, s0), (c1, s1), (c0, s1)] if ax == 0
               else [(s0, c0), (s0, c1), (s1, c1), (s1, c0)])
        msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "WALLS"})
        t = p.get("thickness_mm") or p.get("thickness_raw_mm")
        mid = ((c0+c1)/2, (s0+s1)/2) if ax == 0 else ((s0+s1)/2, (c0+c1)/2)
        msp.add_text(f"{p['name']} {int(t)}mm" if t else p["name"],
                     height=0.06,
                     dxfattribs={"layer": "LABELS"}).set_placement(mid)

    wof = {p["name"]: p for p in walls}
    for f in man["features"]:
        if f["kind"] not in ("door", "window", "arch", "opening"):
            continue
        w = wof.get(f["wall"])
        if not w:
            continue
        ax = 0 if w["axis"] == "x" else 1
        c0, c1 = w["across_m"]; a0, a1 = f["along_m"]
        pts = ([(c0, a0), (c1, a0), (c1, a1), (c0, a1)] if ax == 0
               else [(a0, c0), (a0, c1), (a1, c1), (a1, c0)])
        msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "OPENINGS"})
        mid = ((c0+c1)/2, (a0+a1)/2) if ax == 0 else ((a0+a1)/2, (c0+c1)/2)
        msp.add_text(f"{f['kind']} {f['width_mm']}x{f['height_mm']}",
                     height=0.05,
                     dxfattribs={"layer": "OPENINGS"}).set_placement(mid)

    for p in man["parts"]:
        if p["kind"] == "floor" and "centre_m" in p:
            cx, cy = p["centre_m"]; ex, ey = p.get("extent_m", [0, 0])
            msp.add_lwpolyline([(cx-ex/2, cy-ey/2), (cx+ex/2, cy-ey/2),
                                (cx+ex/2, cy+ey/2), (cx-ex/2, cy+ey/2)],
                               close=True, dxfattribs={"layer": "SLABS"})
            msp.add_text(f"{p['name']} {p['area_m2']} m2", height=0.08,
                         dxfattribs={"layer": "SLABS"}).set_placement((cx, cy))

    msp.add_text(f"clear height {man['clear_height_mm']:.0f} mm   "
                 f"walls {len(walls)}   plan cut at {cut_z*1000:.0f} mm",
                 height=0.15, dxfattribs={"layer": "DIMENSIONS"}
                 ).set_placement((0, 0))
    doc.saveas(path)
    log(f"wrote {path.name}: {len(walls)} wall footprints, "
        f"{sum(1 for f in man['features'] if f['kind'] in ('door','window','arch','opening'))}"
        f" openings ({path.stat().st_size/1e3:.0f} kB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, help="a model directory")
    ap.add_argument("--out", default=None, help="defaults to <dir>/sketch")
    ap.add_argument("--solid", default=None,
                    help="the OBJ of solids to convert; defaults to "
                         "<dir>/modular_solid.obj. Point it at boxes.obj for the "
                         "box model, which is the cleaner thing to edit.")
    ap.add_argument("--stem", default="sketch", help="output file stem")
    a = ap.parse_args()
    d = Path(a.dir)
    out = Path(a.out) if a.out else d/"sketch"
    out.mkdir(parents=True, exist_ok=True)
    man = json.load(open(d/"manifest.json"))
    solid = Path(a.solid) if a.solid else d/"modular_solid.obj"
    if not solid.exists():
        raise SystemExit(f"{solid} is missing -- run solidify_walls.py first")
    parts = read_groups(solid)
    log(f"{len(parts)} solid parts, "
        f"{sum(len(f) for _, f in parts.values()):,} faces")
    write_dae(parts, man, out/f"{a.stem}.dae")
    write_stl(parts, out/f"{a.stem}.stl")
    if a.stem == "sketch":
        write_dxf(man, out/"sketch.dxf", CUT)
    (out/"README.txt").write_text(
        "For tracing in SketchUp.\n"
        "\n"
        "boxes.dae   the box model -- flat faces, sharp edges, one named group\n"
        "            per wall, slab, beam and column, with its niches and\n"
        "            pilasters already cut into it. Import this and push/pull.\n"
        "sketch.dae  the same rooms taken from the scan surface itself, so the\n"
        "            faces follow what was measured. Heavier, less tidy, truer.\n"
        "sketch.dxf  the 2D plan cut at 1.2 m, on layers WALLS, OPENINGS, SLABS,\n"
        "            DIMENSIONS and LABELS. This is the drawing to trace.\n"
        "*.stl       the same solids welded, for SketchUp Web, which takes STL\n"
        "            but neither Collada nor OBJ.\n"
        "\n"
        "File > Import > Collada for a .dae, then File > Save As for a .skp. The\n"
        "units are metres and Z is up.\n"
        "\n"
        "There is no .skp here: the format is closed, and nothing outside\n"
        "SketchUp's own SDK can write one.\n"
        "\n"
        , encoding="utf-8")

    log(f"done -> {out}")


if __name__ == "__main__":
    main()
