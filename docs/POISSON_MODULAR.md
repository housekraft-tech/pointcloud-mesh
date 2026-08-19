# The modular model, taken from the Poisson mesh

`scripts/export/modular_poisson.py`

## Why

Two models existed, each right about half the problem.

The **cell complex** (`cellcomplex.py` → `modular_from_cells.py`) is right about
structure. Its parts partition the masonry, its thicknesses are the gap between
two detected planes, and two parts cannot occupy one space. But every part is a
box, so a niche, an arched head, a boxed conduit and a beam soffit all come out
as flat wall.

The **Poisson mesh** (`poisson_mesh.py`) is the opposite. It carries all of that
relief, and no structure at all: one blob, with the furniture still in it, and
no part you can name, measure or switch off.

This stage takes the structure from the mesh itself and keeps the mesh as the
geometry of every part.

## How

1. **Frame.** The building's yaw comes from where the wall area points — wall
   normals cluster at four headings 90° apart, and the offset of that cluster
   from the axes is the yaw. The mesh is rotated onto its own axes.

2. **Levels.** Floor and ceiling are the two peaks of horizontal surface *area*
   against height.

3. **Orientation — settle this before anything else.** A face plane's material
   lies on the side its normal points to, and *which* side that is depends on
   the mesh's global orientation. Poisson's is **arbitrary**: it follows
   whichever way the input normals were oriented, and two meshes of the same
   flat came out opposite. Every pairing decision inverts with that sign — on
   the flipped mesh the pairing found almost nothing, and mujammel came out with
   25 walls and a thickness "vocabulary" of 75 mm. The floor settles it:
   whatever else is uncertain, the material under a floor is below it. Where the
   floor's normals point up, the normals and the triangle winding are flipped.

4. **Faces.** Each face is an axis, a side and a coordinate, taken from the
   area-weighted histogram of surface across that axis. The coordinate is the
   **peak** refined over a ±20 mm window, not the mean of the band: relief cuts
   into the masonry, so a band-wide mean is pulled inwards and every thickness
   comes out about 20 mm too big.

5. **Walls.** Every 50 mm station along a face chooses its own opposite face, so
   thickness is measured per station and a wall is cut only where that choice
   changes. Three rules make that choice survive contact with a furnished flat:

   - a doorway does not end a wall — runs on the same pair of planes with a
     door's width between them are rejoined;
   - the opposite face must span at least 70% of the face's own height — a wall
     is bounded by a wall, while a wardrobe front stops at 2 m;
   - a candidate whose thickness matches one the **building repeats** beats a
     nearer one that matches nothing. A building has two or three thicknesses,
     used everywhere; they are found as the peaks of the candidate histogram.
     On koushik those peaks are 95 / 195 / 245 mm against a drawing that says
     150 / 200 / 244.

   Afterwards, parts that face each other across a wall are fused, because the
   far side of a corridor wall is broken into a face per room and no run-to-run
   pairing can match more than one of them.

6. **Slabs.** Horizontal triangles are gridded and grown into plateaus. Height
   clustering cannot separate a beam from its ceiling — the histogram is
   continuous — but a plateau is bounded by a vertical step, which region
   growing finds. A long narrow plateau hanging below the highest ceiling is a
   **beam**; a wide one is a **dropped ceiling**.

7. **Columns.** Tall near-square clusters that reach both floor and ceiling and
   whose surface is mostly vertical.

8. **The seam pass — this is the one that matters.** A plane band cuts the mesh
   at a flat boundary, and every fillet, jamb return, reveal and arch soffit in
   a junction falls outside every band by construction. Those are exactly the
   triangles that make a model read as one house rather than a set of panels. So
   after the geometric labelling, every unclaimed triangle that *touches* a part
   grows into it, bounded by how far that kind of part may reach. That bound is
   asymmetric for walls: 350 mm into the masonry, because relief cuts inwards,
   and 100 mm into the room, because beyond that stands the furniture.

9. **What is left** is attached to nothing structural, and only that is dropped.

10. **Relief.** Each wall is unfolded onto its near face as a depth map in
    (along, height). A cell with no surface is a void — an opening if material
    spans over it. The top of the void, column by column, is a flat lintel or a
    curved **arch**. A cell deeper than the wall's own surface is a **niche**,
    one standing proud a **pilaster**. Depth is measured against the median of
    the face, not against the fitted plane, or a wall whose plane sits on the
    far face reads as one enormous niche.

## Running it

    set PM_MASK=0 & set PM_CUT=-0.60 & set PM_VOXEL=0.008 & set PM_DEPTH=12 & set PM_TRIM=1.0
    venv311\Scripts\python.exe scripts\export\poisson_mesh.py <scan.las> output\model\poisson.ply
    venv311\Scripts\python.exe scripts\export\mesh_cache.py output\model\poisson.ply output\model\poisson.npz
    venv311\Scripts\python.exe scripts\export\modular_poisson.py --cache output\model\poisson.npz --out output\model\poisson_modular
    venv311\Scripts\python.exe scripts\export\solidify_walls.py --dir output\model\poisson_modular --cache output\model\poisson.npz
    venv311\Scripts\python.exe scripts\exportake_rgb.py --dir output\model\poisson_modular --cache output\model\poisson.npz --las <scan.las>
    venv311\Scripts\python.exe scripts\export\modular_viewer.py output\model\poisson_modular output\model\poisson.npz
    venv311\Scripts\python.exe scripts\export
ender_modular.py output\model\poisson_modular output\model\poisson.npz
    venv311\Scripts\python.exe scripts\export
ender_wall_elevation.py output\model\poisson_modular output\model\poisson.npz
    venv311\Scripts\python.exe scripts\export\make_index.py

Those settings matter. `PM_VOXEL=0.008` with `PM_DEPTH=12` roughly doubles the
mesh (koushik: 7.9 M triangles to 14.2 M) and it is not cosmetic — on the coarser
mesh the thicknesses the building repeats came out 105 / 195 / 245 mm, and on the
finer one 145 / 195 / 245 against a drawing that says 150 / 200 / 244. The extra
resolution is where the millimetres are.

`PM_CUT` slices the ceiling off for a cutaway view. **Do not use it for this
stage** — without the ceiling slab, the dropped ceilings of the
wet rooms become "the ceiling" and every height in the model is measured to the
wrong surface. `modular_poisson.py` warns when the mesh it was given looks like
that.

Outputs, in the `--out` directory:

| File | What it is |
|---|---|
| `modular.obj` | full resolution, one `o` group per named part — this is the Blender import |
| `modular.glb` | the same, as a scene of named meshes |
| `modular_lite.glb` | decimated to ~900k triangles for the browser |
| `viewer.html` | self-contained viewer: toggle by kind, click a part for its measurements |
| `manifest.json` | every part with its measurements, every feature, and the coverage audit |
| `labels.npy`, `verts.npy`, `names.json` | the segmentation itself, for further stages |
| `wall_elevations.png` | per-wall depth maps — the check that the relief survived |

## Filling the walls

`solidify_walls.py` turns each wall into a closed solid. The surface model gives
a wall both of its faces, but a face is a sheet: opened in Blender the wall is
hollow, with no volume and nothing to cut into.

Each wall's face is gridded at 10 mm. Where the scan found material the wall is
solid; where it found none through either face there is a hole — a doorway, a
window, the space under a lintel. That outline is traced, simplified to 5 mm,
triangulated with its holes and extruded through the thickness measured for that
wall. On koushik's fine mesh: **39 of 39 walls watertight, 39.1 m³ of masonry**,
in a 1 MB file.

The solid carries the outline, the openings, the thickness and therefore the
volume. It does **not** carry the millimetre relief — that stays in
`modular.obj`, which is the scanned surface itself. A per-cell solid carrying
both was built and abandoned: the scan steps at almost every 10 mm cell, so the
two faces met in millions of little ribbons and 12% of their edges would not
close, in a 787 MB file. A solid whose volume can be trusted beside a surface
whose millimetres can be trusted beats one mesh that is wrong at both.

## Colour

`bake_rgb.py` puts the scan's RGB back on the model, where the scan has it
(mujammel does, koushik's is all zero). Poisson throws colour away, so each
vertex takes the colour of the nearest scanned point — after the point cloud is
put through the same rotation and z-shift the mesh went through. The residual
distance is reported and must be a few millimetres: on mujammel it is a median of
**5.2 mm**, which is the check that the two frames really match.

Colour answers what geometry cannot: what a surface *is* (tile, paint, timber,
stone are the same shape and different colours), where the glazing is (a window
is a hole to the geometry and a dark low-return patch to the colour), and what
changed between two scans (a repaint is invisible in geometry). Each part gets a
median colour and a spread in the manifest; `modular_rgb.glb` carries it per
vertex.

## The two scans check each other

koushik and mujammel are the same flat, walked twice. Neither is a ground truth,
but a quantity that comes out the same from two independent walks is one the
pipeline can measure. `compare_models.py` registers them and reports:

| | agreement |
|---|---|
| registration | 0°, 0.364 / −0.842 m — no rotation, so it is one flat |
| wall lines | 37 of 39 matched, median **15 mm**, 90th pct 63 mm |
| wall thickness | median **10 mm** over the 15 both scans vouch for |
| ceiling levels | 2706 / 2733 / 2741 / 2744 vs 2699 / 2735 / 2736 / 2741 mm |
| openings | both find the 2325×2375 door and the 1550×1175 window |

## Reading the audit

`coverage` in the manifest answers the two questions that matter:

- `kept_frac` — how much of the scanned surface ended up in a named part.
- `crack_tris` — dropped triangles with **two different parts** around them.
  A triangle that merely changes part is a junction; one with two parts around it
  is a hole in the shell. This is the number that says "seamless", and on koushik
  it is 6 out of 7.9 million triangles.

## Result

Both scans, built at 8 mm / depth 12:

| | koushik | mujammel |
|---|---|---|
| mesh | 14.2 M triangles | 27.4 M triangles |
| parts | 82 | 80 |
| walls | 39 | 34 |
| floors / ceilings | 20 / 22 (2 beams, 7 dropped) | 22 / 22 (2 beams, 7 dropped) |
| columns | 1 | 2 |
| features | 50 | 54 |
| thicknesses repeated | 145 / 195 / 245 mm | 195 / 255 mm |
| clear height | 2740 mm | 2745 mm |
| surface in a named part | 98.6% | 98.7% |
| gap between parts | 8 triangles | 11 triangles |
| solids | 39/39 watertight, 39.1 m³ | 34/34 watertight, 46.4 m³ |
| colour | scan has none | baked, 5.1 mm median |

The drawing says 150 / 200 / 244 mm walls; the site report says 2747 mm clear.

Output folders: `output/model/poisson_modular_fine/` (koushik) and
`output/model/poisson_modular_muj_fine/` (mujammel). `output/model/index.html`
lists every build with its numbers and pictures.
