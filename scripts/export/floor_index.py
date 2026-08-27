"""One folder per storey, with the four datasets side by side.

A storey is only useful if everything about it is in one place: the points it
was measured from, the surface fitted to them, that surface cut into named
parts, and the clean box model. Each floor's folder holds those four, and this
writes the README that says which file is which and a building index that
links them.
"""
import json, sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "soulace_output")
STO = {s["level"]: s for s in json.load(open(ROOT/"storeys.json"))["storeys"]}

WHAT = {
 "lidar":   ("LiDAR", "the storey's own points, cut out of the building scan"),
 "poisson": ("Poisson", "the surface fitted to those points -- all the relief, no structure"),
 "modular": ("modular", "that same surface cut into named parts: walls, slabs, beams, columns"),
 "sketch":  ("SketchUp", "what a designer imports and traces over"),
 "boxes":   ("boxes", "the clean model -- one box per part, with its protrusions and recesses"),
}
NOTE = {
 "L0.las": "the points, in building coordinates",
 "L1.las": "the points, in building coordinates",
 "L2.las": "the points, in building coordinates",
 "poisson.ply": "the reconstructed surface",
 "poisson.npz": "the same, as arrays -- what every later stage reads",
 "poisson_fine.ply": "a finer reconstruction of this storey (8 mm voxel, depth 12)",
 "poisson_fine.npz": "the same, as arrays",
 "scan_view.glb": "decimated for the browser -- the viewer's scan layer",
 "modular.obj": "full resolution, one group per part -- the Blender/SketchUp import",
 "modular_view.glb": "decimated for the browser",
 "modular_solid.obj": "each wall as a closed solid, with its openings",
 "modular_solid.glb": "the same, for the browser",
 "manifest.json": "every part with its measurements, features and coverage",
 "labels.npy": "which part each triangle belongs to",
 "verts.npy": "the vertices those triangles index",
 "names.json": "the part names, in label order",
 "wall_elevations.png": "per-wall depth maps -- the check that the relief survived",
 "thickness_profiles.png": "thickness along each wall",
 "render_iso.png": "isometric render",
 "render_plan.png": "plan render",
 "render_cutaway.png": "cutaway render",
 "render_dropped.png": "the dropped ceilings",
 "boxes.obj": "the box model, one object per part",
 "boxes.glb": "the same, for the browser and the walkthrough",
 "boxes_union.glb": "all of it fused into one watertight solid",
 "boxes.dae": "the box model for SketchUp -- one named group per part",
 "boxes.stl": "the same, welded, for SketchUp Web",
 "sketch.dae": "the scan-surface solids for SketchUp -- heavier, truer",
 "sketch.stl": "the same, welded, for SketchUp Web",
 "sketch.dxf": "the 2D plan cut at 1.2 m: WALLS, OPENINGS, SLABS, DIMENSIONS, LABELS",
 "README.txt": "how to import these, and why there is no .skp",
}
CARDS = []
for d in sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("L")):
    lvl = int(d.name[1:])
    s = STO.get(lvl, {})
    man = json.load(open(d/"modular"/"manifest.json"))
    box = man.get("boxes", {})
    lines = [f"# {d.name} -- storey {lvl}", "",
             f"Floor at **{s.get('floor_z', 0):+.3f} m** and ceiling at "
             f"**{s.get('ceiling_z', 0):+.3f} m** in the building's own frame, a clear height "
             f"of **{s.get('clear_height_mm', 0):.0f} mm**, measured from "
             f"**{s.get('points', 0):,}** points.", "",
             "The model in this folder is in its own frame, with its floor near z = 0; "
             f"add **{s.get('floor_z', 0):+.3f} m** to stack it into the building.", "",
             f"{man['n_parts']} named parts, {len(man.get('features', []))} features, "
             f"walls {'/'.join(f'{t:.0f}' for t in man.get('thickness_modes_mm', []))} mm.", ""]
    for sub, (name, why) in WHAT.items():
        ff = sorted((d/sub).iterdir()) if (d/sub).is_dir() else []
        if not ff:
            continue
        lines += [f"## {sub}/ -- {name}", "", why, "",
                  "| file | size | what it is |", "|---|---:|---|"]
        for f in ff:
            mb = f.stat().st_size/1e6
            sz = f"{mb:,.0f} MB" if mb >= 1 else f"{mb*1000:.0f} kB"
            lines.append(f"| `{f.name}` | {sz} | {NOTE.get(f.name, '')} |")
        lines.append("")
    lines += ["## viewer.html", "",
              "All three layers over each other, with the walk-through. Serve the "
              "repository and open it -- opening the file directly will not load the meshes.", ""]
    (d/"README.md").write_text("\n".join(lines), encoding="utf-8")
    tot = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())/1e9
    CARDS.append((d.name, lvl, s, man, box, tot))
    print(f"{d.name}: README written, {tot:.1f} GB in the folder")

rows = "\n".join(
  f"""<div class=c><h2><a href="{n}/viewer.html">{n}</a> &mdash; storey {l}</h2>
  <p>floor {s.get('floor_z',0):+.3f} m, ceiling {s.get('ceiling_z',0):+.3f} m,
     clear height <b>{s.get('clear_height_mm',0):.0f} mm</b>, {s.get('points',0):,} points,
     {m['n_parts']} parts, {b.get('n_boxes','?')} boxes, {t:.1f} GB</p>
  <p class=f><a href="{n}/README.md">README</a>
   &middot; <a href="{n}/lidar/">lidar/</a>
   &middot; <a href="{n}/poisson/">poisson/</a>
   &middot; <a href="{n}/modular/">modular/</a>
   &middot; <a href="{n}/boxes/">boxes/</a>
   &middot; <a href="{n}/sketch/">sketch/</a></p>
  <p class=f><img src="{n}/modular/render_iso.png" width=260></p></div>"""
  for n, l, s, m, b, t in CARDS)
(ROOT/"index.html").write_text(f"""<!doctype html><meta charset=utf-8>
<title>Soulace &mdash; three storeys</title>
<style>body{{font:15px/1.6 system-ui;margin:2rem;max-width:1100px}}
.c{{border:1px solid #ddd;border-radius:8px;padding:1rem;margin:1rem 0}}
h2{{margin:0 0 .3rem}} .f{{color:#666;font-size:13px}} img{{border:1px solid #eee}}</style>
<h1>Soulace</h1>
<p>Each storey has its own folder holding the four datasets &mdash; the points, the
Poisson surface, the surface cut into named parts, and the box model &mdash; plus a
viewer that stacks all three over each other.</p>
{rows}
<div class=c><h2>the building</h2>
<p><a href="storeys.json">storeys.json</a> &mdash; where each floor and ceiling sits.
<a href="video/soulace_flat.mp4">video/soulace_flat.mp4</a> &mdash; the fisheye walk, flattened.</p></div>
""", encoding="utf-8")
print(f"wrote {ROOT/'index.html'}")
