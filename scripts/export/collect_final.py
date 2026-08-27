"""Every model's latest outputs, in one folder, without copying 30 GB.

Each model gets the same five drawers -- lidar, poisson, modular, boxes,
sketchup -- so a scan is one directory and nothing has to be hunted for. The
big files are HARD LINKED, not copied: same bytes, two names, no extra disk and
no wait. Delete this folder and the originals are untouched.
"""
import os, sys, json, shutil, argparse, time
from pathlib import Path

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

MODELS = [
 dict(name="koushik", src="output/latest/koushik", las="koushikexport.las",
      ply="output/model/poisson_koushik_fine.ply",
      npz="output/model/poisson_koushik_fine.npz", pack="Flat Sketchup", stem="Koushik_flat"),
 dict(name="mujammel", src="output/latest/mujammel", las="mujammelexport.las",
      ply="output/model/poisson_mujammel_fine.ply",
      npz="output/model/poisson_mujammel_fine.npz", pack="Flat Sketchup", stem="Mujammel_flat"),
 dict(name="soulace_L0", src="soulace_output/L0", storey=True,
      pack="Soulace Sketchup", stem="Soulace_L0_ground"),
 dict(name="soulace_L1", src="soulace_output/L1", storey=True,
      pack="Soulace Sketchup", stem="Soulace_L1_first"),
 dict(name="soulace_L2", src="soulace_output/L2", storey=True,
      pack="Soulace Sketchup", stem="Soulace_L2_second"),
]
MODULAR = ["modular.obj", "modular_view.glb", "modular_full.glb", "modular_solid.obj",
           "modular_solid.glb", "labels.npy", "verts.npy", "names.json", "manifest.json",
           "wall_elevations.png", "thickness_profiles.png", "render_iso.png",
           "render_plan.png", "render_cutaway.png", "render_dropped.png", "compare_plan.png"]
BOXES = ["boxes.obj", "boxes.glb", "boxes_union.glb", "render_boxes_iso.png",
         "render_boxes_plan.png"]


def put(src, dst):
    """One file, two names. Copy only if the filesystem refuses to link."""
    src, dst = Path(src), Path(dst)
    if not src.exists():
        return 0
    if dst.exists():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return src.stat().st_size


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="output_final")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    # anything already built straight into the folder by run_scan.py is a model
    # too -- it has the five drawers and a manifest, it just was not copied here
    known = {m["name"] for m in MODELS}
    for q in sorted(out.iterdir()):
        if q.is_dir() and q.name not in known and (q/"modular"/"manifest.json").exists():
            MODELS.append(dict(name=q.name, src=str(q), in_place=True,
                               pack=str(q/"sketchup"), stem=""))
    cards = []
    for m in MODELS:
        src = Path(m["src"])
        if m.get("in_place"):
            man = json.load(open(src/"modular"/"manifest.json"))
            n = sum(f.stat().st_size for f in src.rglob("*") if f.is_file())
            cards.append((m["name"], man, n/1e9, m["stem"]))
            log(f"{m['name']}: already in place, {n/1e9:.1f} GB")
            continue
        if not src.exists():
            log(f"{m['name']}: {src} is missing -- skipped")
            continue
        d = out/m["name"]
        mod_src = src/"modular" if (src/"modular").is_dir() else src
        box_src = src/"boxes" if (src/"boxes").is_dir() else src
        n = 0
        if m.get("storey"):
            for f in (src/"lidar").glob("*.las"):
                n += put(f, d/"lidar"/f.name)
            for f in (src/"poisson").glob("poisson*"):
                n += put(f, d/"poisson"/f.name)
            n += put(src/"poisson"/"scan_view.glb", d/"poisson"/"scan_view.glb")
        else:
            n += put(m["las"], d/"lidar"/f"{m['name']}.las")
            n += put(m["ply"], d/"poisson"/"poisson.ply")
            n += put(m["npz"], d/"poisson"/"poisson.npz")
            n += put(src/"scan_view.glb", d/"poisson"/"scan_view.glb")
        for f in MODULAR:
            n += put(mod_src/f, d/"modular"/f)
        for f in BOXES:
            n += put(box_src/f, d/"boxes"/f)
        pack = Path(m["pack"])
        for f in sorted(pack.glob(m["stem"]+"*")):
            n += put(f, d/"sketchup"/f.name)
        for f in ("README.txt",):
            put(pack/f, d/"sketchup"/f)
        if (pack/"lib").is_dir() and not (d/"sketchup"/"lib").exists():
            shutil.copytree(pack/"lib", d/"sketchup"/"lib")
        # its own handover viewer, listing only this model's files
        tmpl = Path(__file__).parent/"templates"/"sketchup_viewer.html"
        if tmpl.exists():
            html = tmpl.read_text(encoding="utf-8")
            html = html.replace("@@FLOORS@@", json.dumps(
                [{"key": m["stem"], "label": m["name"], "stem": m["stem"]}]))
            html = html.replace("@@TITLE@@", f"{m['name']} &mdash; for SketchUp")
            (d/"sketchup"/"viewer.html").write_text(html, encoding="utf-8")
        # the viewer, with its mesh paths pointed at the drawers
        vsrc = src/"viewer.html"
        if vsrc.exists():
            html = vsrc.read_text(encoding="utf-8")
            import re
            for pat, rep in ((r"([^/])modular_view\.glb", r"\1modular/modular_view.glb"),
                             (r"([^/])modular_full\.glb", r"\1modular/modular_full.glb"),
                             (r"([^/])scan_view\.glb", r"\1poisson/scan_view.glb"),
                             (r"([^/])boxes\.glb", r"\1boxes/boxes.glb")):
                html = re.sub(pat, rep, html)
            (d/"viewer.html").write_text(html, encoding="utf-8")
        man = json.load(open(mod_src/"manifest.json"))
        cards.append((m["name"], man, n/1e9, m["stem"]))
        log(f"{m['name']}: {n/1e9:.1f} GB linked into {d}")

    rows = "\n".join(
      f"""<div class=c><h2>{n}</h2>
      <p>{man['n_parts']} named parts &middot; ceiling {man['modal_ceiling_height_mm']:.0f} mm over the floor
         &middot; walls {'/'.join(f"{t:.0f}" for t in man.get('thickness_modes_mm', []))} mm
         &middot; {sz:.1f} GB</p>
      <p class=f><a href="{n}/viewer.html">viewer</a> &middot;
       <a href="{n}/lidar/">lidar</a> &middot; <a href="{n}/poisson/">poisson</a> &middot;
       <a href="{n}/modular/">modular</a> &middot; <a href="{n}/boxes/">boxes</a> &middot;
       <a href="{n}/sketchup/viewer.html">sketchup</a></p></div>"""
      for n, man, sz, stem in cards)
    (out/"index.html").write_text(f"""<!doctype html><meta charset=utf-8>
<title>outputs</title>
<style>body{{font:15px/1.6 system-ui;margin:2rem;max-width:1000px}}
.c{{border:1px solid #ddd;border-radius:8px;padding:1rem;margin:1rem 0}}
h2{{margin:0 0 .3rem;font-size:16px}} .f{{color:#666;font-size:13px}}</style>
<h1>outputs</h1>
<p>Five drawers per scan: <b>lidar</b> the points, <b>poisson</b> the surface fitted to
them, <b>modular</b> that surface cut into named parts, <b>boxes</b> the clean solid
model, <b>sketchup</b> what a designer imports. Files are hard links to the originals,
so this folder costs almost no disk.</p>
{rows}""", encoding="utf-8")
    log(f"index -> {out/'index.html'}")


if __name__ == "__main__":
    main()
