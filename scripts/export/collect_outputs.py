"""Gather the current best outputs into one folder, and say what each file is.

The working folder accumulates every build -- coarse meshes, superseded runs,
half-finished experiments -- and none of that is the deliverable. This moves the
two current models into `output/latest/`, drops the carved free-space grids
beside them, writes the index over the result and a README that names every
file. Moving, not copying: these are gigabytes, and a second copy of a
superseded build helps nobody.
"""
import json, shutil, sys, subprocess
from pathlib import Path

SRC = Path("output/model")
DST = Path(sys.argv[1] if len(sys.argv) > 1 else "output/latest")
MODELS = {"poisson_modular_fine": "koushik", "poisson_modular_muj_fine": "mujammel"}
GRIDS = {"free_koushik.npz": "koushik", "free_mujammel.npz": "mujammel"}

README = """# The modular house, from the LiDAR

Two scans of one flat, each rebuilt into a named, measured, modular model. The
geometry in every part is the scanned surface itself, cropped -- not a box
fitted to it -- so the niches, arch soffits, beam drops and boxed conduits are
the scan's own millimetres.

    koushik/    23.2 M point scan, 14.2 M triangle mesh
    mujammel/   25.5 M point scan, 27.4 M triangle mesh, and it carries colour

## What is in each folder

| File | What it is |
|---|---|
| `viewer.html` | **open this first**, through the local server. Orbit, switch the ceiling off, click a part to read its measurements, switch colour between part / kind / as-scanned, and press "full resolution" for every triangle |
| `modular_view.glb` | what the viewer loads first: 3 M triangles |
| `modular_full.glb` | every triangle, loaded on demand |
| `modular.obj` | the full-resolution surface, one named `o` group per part -- **this is the Blender import** |
| `modular_solid.obj` | the same walls as closed solids: measured outline, measured thickness, openings cut through. This is the file with volumes |
| `modular_solid.glb` | the same, for a viewer |
| `manifest.json` | every part and every feature with its measurements, plus the coverage audit |
| `labels.npy`, `verts.npy`, `names.json` | the segmentation itself, for the next stage |
| `vertex_rgb.npy` | the scan's colour per vertex (mujammel only) |
| `wall_elevations.png` | each wall unfolded as a depth map -- the check that the relief survived |
| `render_*.png` | overviews: iso, cutaway, plan, walls only, floor only, solids, what was dropped |
| `compare_plan.png` | both scans' wall lines on one plan, registered (koushik only) |

`freespace/` holds the carved free space -- which cells the scanner looked
through, recovered from gps_time with no trajectory file. It is what confirms an
opening is an opening and a wall is solid.

## How to read the numbers

`manifest.json` → `coverage` answers the two that matter: `kept_frac`, how much
of the scanned surface ended up in a named part, and `crack_tris`, how many
triangles of hole the parts leave between them. A triangle that merely changes
part is a junction; one with two parts around it is a gap.

Per wall: `thickness_mm` is reported only where two faces were measured AND the
thickness matches one the building repeats. `thickness_raw_mm` is what the
pairing found either way, and `interior_free` is how much of that wall's inside
the scanner could see through -- 0 for masonry, near 1 for a cavity.

The two scans are the same flat, so they check each other; see the branch's
`docs/POISSON_MODULAR.md` for the agreement figures.
"""


def main():
    DST.mkdir(parents=True, exist_ok=True)
    (DST/"freespace").mkdir(exist_ok=True)
    for src, name in MODELS.items():
        s, d = SRC/src, DST/name
        if not s.exists():
            print(f"  skip {src}: not there")
            continue
        if d.exists():
            shutil.rmtree(d)
        shutil.move(str(s), str(d))
        man = json.load(open(d/"manifest.json"))
        print(f"  {name}: {man['n_parts']} parts, "
              f"{sum(f.stat().st_size for f in d.iterdir() if f.is_file())/1e6:.0f} MB")
    for g, _ in GRIDS.items():
        if (SRC/g).exists():
            shutil.move(str(SRC/g), str(DST/"freespace"/g))
    (DST/"README.md").write_text(README, encoding="utf-8")
    subprocess.run([sys.executable, "scripts/export/make_index.py", str(DST)], check=True)
    print(f"collected into {DST}/")


if __name__ == "__main__":
    main()
