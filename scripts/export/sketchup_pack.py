"""One watertight mesh per storey, in the formats SketchUp imports.

The per-part model is what you edit; this is what you HAND OVER. Every part of
a storey is fused into a single closed solid -- no coincident faces between
neighbouring boxes, no internal partitions, one shell -- and written under one
folder with the storey in the filename.

Watertightness is checked, not assumed: a mesh that is not closed, not
consistently wound, or has a negative volume is reported and not written.
"""
import sys, json, shutil, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import trimesh

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

NAMES = {0: "L0_ground", 1: "L1_first", 2: "L2_second"}


def clean(m):
    """Only repair a mesh that needs it.

    The union comes out of manifold3d already closed. Welding its vertices
    afterwards does not tidy it -- it merges vertices that were deliberately
    apart and tears 378 edges open. So a closed mesh is left exactly as it is,
    and the repair is attempted only on one that arrives broken.
    """
    if m.is_watertight and m.is_winding_consistent:
        return m
    m.merge_vertices()
    m.update_faces(m.nondegenerate_faces())
    m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()
    m.fill_holes()
    m.fix_normals()
    return m


def write_dae(m, path, name):
    """Collada with one node, written directly -- trimesh writes no scene."""
    from collada import Collada, source, geometry, material, scene
    c = Collada()
    eff = material.Effect("e0", [], "lambert", diffuse=(.82, .80, .76), specular=(0, 0, 0))
    mat = material.Material("m0", "concrete", eff)
    c.effects.append(eff); c.materials.append(mat)
    V = np.asarray(m.vertices, np.float32).ravel()
    N = np.asarray(m.face_normals, np.float32).repeat(3, axis=0).ravel()
    vs = source.FloatSource(f"{name}-v", V, ("X", "Y", "Z"))
    ns = source.FloatSource(f"{name}-n", N, ("X", "Y", "Z"))
    g = geometry.Geometry(c, f"g-{name}", name, [vs, ns])
    il = source.InputList()
    il.addInput(0, "VERTEX", f"#{name}-v")
    il.addInput(1, "NORMAL", f"#{name}-n")
    idx = np.empty(m.faces.size*2, np.int32)
    idx[0::2] = np.asarray(m.faces).ravel()
    idx[1::2] = np.arange(m.faces.size)
    g.primitives.append(g.createTriangleSet(idx, il, "ref0"))
    c.geometries.append(g)
    node = scene.Node(name, children=[scene.GeometryNode(
        g, [scene.MaterialNode("ref0", mat, inputs=[])])])
    s = scene.Scene("s", [node])
    c.scenes.append(s); c.scene = s
    c.assetInfo.unitname, c.assetInfo.unitmeter = "meter", 1.0
    c.assetInfo.upaxis = "Z_UP"
    c.write(str(path))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None,
                    help="a building directory holding L0/ L1/ ... storeys")
    ap.add_argument("--model", action="append", default=[],
                    help="NAME=DIR for a single model, repeatable")
    ap.add_argument("--out", default="Soulace Sketchup")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    jobs = []
    if a.root:
        r = Path(a.root)
        for d in sorted(p for p in r.iterdir() if p.is_dir() and p.name.startswith("L")):
            lvl = int(d.name[1:])
            jobs.append((f"{r.name.split('_')[0].title()}_{NAMES.get(lvl, d.name)}", d))
    for spec in a.model:
        nm, _, dd = spec.partition("=")
        jobs.append((nm, Path(dd)))
    rows = []
    for stem, d in jobs:
        union = d/"boxes"/"boxes_union.glb"
        if not union.exists():
            union = d/"boxes_union.glb"
        s = trimesh.load(union)
        m = clean(s.to_mesh() if hasattr(s, "to_mesh") else s)
        ok = m.is_watertight and m.is_winding_consistent and m.volume > 0
        if not ok:
            log(f"{d.name}: NOT closed (watertight={m.is_watertight}, "
                f"winding={m.is_winding_consistent}, volume={m.volume:.2f}) -- not written")
            continue
        write_dae(m, out/f"{stem}.dae", stem)
        m.export(out/f"{stem}.stl")
        m.export(out/f"{stem}.obj")
        # the per-storey folder gets renamed by hand, so find the plan rather
        # than assume where it lives
        # The fused solid is one lump on purpose -- it is watertight. But a
        # designer wants to grab ONE wall, so the same geometry also goes out
        # with its parts intact: one named group per wall, slab, beam and
        # column, each already carrying its own openings, niches and pilasters
        # and already cut where it meets a neighbour.
        bobj = d/"boxes"/"boxes.obj"
        if not bobj.exists():
            bobj = d/"boxes.obj"
        if bobj.exists():
            from to_sketchup import read_groups, write_dae as write_dae_parts
            parts = read_groups(bobj)
            mf2 = d/"modular"/"manifest.json"
            man2 = json.load(open(mf2 if mf2.exists() else d/"manifest.json"))
            write_dae_parts(parts, man2, out/f"{stem}_parts.dae")
            shutil.copy(bobj, out/f"{stem}_parts.obj")
            log(f"  {stem}_parts.dae: {len(parts)} named groups")

        dxf = next((q for q in sorted(d.rglob("*.dxf"))), None)
        if dxf and dxf.resolve() != (out/f"{stem}_plan.dxf").resolve():
            shutil.copy(dxf, out/f"{stem}_plan.dxf")
        mf = d/"modular"/"manifest.json"
        man = json.load(open(mf if mf.exists() else d/"manifest.json"))
        rows.append((stem, d.name, m, man))
        log(f"{stem}: {len(m.faces):,} triangles, {m.volume:.1f} m3, closed, "
            f"z {m.bounds[0][2]:.3f}..{m.bounds[1][2]:.3f}")

    lines = ["Soulace -- one closed solid per storey.", "",
             "Import: File > Import > Collada (.dae). Tick 'Merge coplanar faces'",
             "in the importer options, or every box face arrives split into two",
             "triangles. Units are metres, Z up. File > Save As gives you a .skp.",
             "",
             "viewer.html  serve this folder and open it to see these exact",
             "        files, storey by storey, with every edge drawn", "",
             "*.dae         the storey as ONE watertight solid",
             "*_parts.dae   the same geometry with its parts intact: one named",
             "              group per wall, slab, beam and column, each already",
             "              cut where it meets its neighbours. Import THIS to",
             "              edit wall by wall.",
             "*_detail.stl  the scanned surface itself at 60k triangles, ~1 mm",
             "              from the Poisson mesh -- all the relief the boxes",
             "              flatten. A surface, not a solid.",
             "*.stl   the same, for SketchUp Web, which takes STL",
             "*.obj   the same, for anything that is not SketchUp",
             "*_plan.dxf  the 2D plan cut at 1.2 m, layers WALLS / OPENINGS /",
             "        SLABS / DIMENSIONS / LABELS, to trace or to underlay", "",
             "| storey | triangles | masonry | floor -> ceiling | clear height |",
             "|---|---|---|---|---|"]
    for stem, nm, m, man in rows:
        zf = man["floor_z"]; ch = man["modal_ceiling_height_mm"]
        lines.append(f"| {stem} | {len(m.faces):,} | {m.volume:.1f} m3 | "
                     f"{m.bounds[0][2]:.3f} -> {m.bounds[1][2]:.3f} m | {ch:.0f} mm |")
    lines += ["", "Each storey sits in its own frame with its floor near z = 0.",
              "To stack them, raise L1 by the height of L0 and L2 by the height of",
              "L0 + L1; the exact figures are in soulace_output/storeys.json.", ""]
    (out/"README.txt").write_text("\n".join(lines), encoding="utf-8")

    # the viewer, pointed at exactly these files, with three.js beside it so the
    # page works with no network at all -- a CDN that is slow or blocked shows as
    # a black screen and no error, which is how this went wrong once already
    tmpl = Path(__file__).parent/"templates"/"sketchup_viewer.html"
    if tmpl.exists() and rows:
        floors = json.dumps([{"key": stem, "label": stem.replace("_", " "),
                              "stem": stem} for stem, _, _, _ in rows])
        html = tmpl.read_text(encoding="utf-8")
        html = html.replace("@@FLOORS@@", floors).replace("@@TITLE@@", out.name)
        (out/"viewer.html").write_text(html, encoding="utf-8")
        lib = out/"lib"; src = Path("Soulace Sketchup")/"lib"
        if not lib.exists() and src.exists():
            shutil.copytree(src, lib)
        log(f"viewer.html over {len(rows)} models")

    log(f"done -> {out}")


if __name__ == "__main__":
    main()
