"""One page listing every model built, with its numbers and its pictures.

Reads each model directory's manifest and writes output/model/index.html. The
numbers shown are the ones worth arguing about: what fraction of the scanned
surface ended up in a named part, how many triangles of gap the parts leave
between them, the thicknesses the building repeats, and the masonry volume.
"""
import json, os, sys
from pathlib import Path

ROOT = Path("output/model")
DIRS = [d for d in sorted(ROOT.iterdir()) if (d/"manifest.json").exists()]

CARD = """<section>
 <h2>{title}</h2>
 <p class=links>{links}</p>
 <p class=n>{summary}</p>
 <div class=g>{imgs}</div>
</section>"""


def human(d):
    man = json.load(open(d/"manifest.json"))
    cov = man["coverage"]
    kinds = {}
    for p in man["parts"]:
        kinds[p["kind"]] = kinds.get(p["kind"], 0) + 1
    feats = {}
    for f in man["features"]:
        feats[f["kind"]] = feats.get(f["kind"], 0) + 1
    sol = man.get("solid", {})
    bits = [
        f"clear height <b>{man['clear_height_mm']:.0f} mm</b>",
        f"{man['n_parts']} parts (" + ", ".join(f"{v}&nbsp;{k}" for k, v in sorted(kinds.items())) + ")",
        f"{sum(feats.values())} features (" + ", ".join(f"{v}&nbsp;{k}" for k, v in sorted(feats.items())) + ")",
        f"<b>{cov['kept_frac']*100:.1f}%</b> of the scanned surface is in a named part",
        f"<b>{cov['crack_tris']}</b> triangles of gap between parts, out of {cov['total_tris']:,}",
        f"{cov['dropped_area_m2']} m&sup2; dropped as clutter",
    ]
    if man.get("thickness_modes_mm"):
        bits.insert(1, "thicknesses the building repeats: <b>"
                    + " / ".join(f"{t:.0f}" for t in man["thickness_modes_mm"]) + " mm</b>")
    if sol:
        bits.append(f"solids: {sol['watertight']}/{sol['walls']} watertight, "
                    f"<b>{sol['masonry_volume_m3']:.1f} m&sup3;</b> of masonry")
    if any("rgb" in p for p in man["parts"]):
        bits.append("colour baked from the scan")
    links = []
    for f, label in (("viewer.html", "open the 3D viewer"),
                     ("modular.obj", "surface model (Blender)"),
                     ("modular_solid.obj", "solid walls"),
                     ("modular_rgb.glb", "coloured model"),
                     ("manifest.json", "manifest")):
        p = d/f
        if p.exists():
            mb = p.stat().st_size/1e6
            links.append(f'<a href="{d.name}/{f}">{label}</a> <span class=sz>{mb:.0f} MB</span>')
    imgs = "".join(f'<a href="{d.name}/{i}"><img loading=lazy src="{d.name}/{i}"></a>'
                   for i in sorted(os.listdir(d)) if i.endswith(".png"))
    return CARD.format(title=d.name.replace("poisson_modular", "model").replace("_", " "),
                       links=" &middot; ".join(links),
                       summary=" &middot; ".join(bits), imgs=imgs)


html = f"""<!doctype html><meta charset=utf-8>
<title>The modular house, taken from the Poisson mesh</title>
<style>
 body{{background:#0f1115;color:#dfe3ea;font:14px/1.65 ui-sans-serif,system-ui,sans-serif;
      margin:0 auto;max-width:1150px;padding:30px}}
 h1{{font-size:21px;margin-bottom:4px}} h2{{font-size:16px;margin:34px 0 6px}}
 a{{color:#7fa8ff;text-decoration:none}} a:hover{{text-decoration:underline}}
 .n{{color:#96a1b2;font-size:12.5px}} .sz{{color:#5f6875;font-size:11px}}
 .links{{font-weight:600}}
 .g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:8px;margin-top:10px}}
 .g img{{width:100%;background:#fff;border-radius:5px}}
 code{{background:#1b2029;padding:1px 5px;border-radius:3px}}
</style>
<h1>The modular house, taken from the Poisson mesh</h1>
<p class=n>Every part is the scanned surface itself, cropped and named &mdash; not a box
fitted to it. Structure comes off the mesh; the relief, the arches and the beams are the
mesh's own geometry. <code>modular.obj</code> is that surface, one named group per part.
<code>modular_solid.obj</code> is the same walls as closed solids on the measured outline
and thickness, which is what carries a volume. koushik and mujammel are the same flat,
walked twice, so they check each other.</p>
{''.join(human(d) for d in DIRS)}
"""
(ROOT/"index.html").write_text(html, encoding="utf-8")
print(f"wrote {ROOT/'index.html'} covering {len(DIRS)} models")
