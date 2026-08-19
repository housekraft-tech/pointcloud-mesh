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

3. **Faces.** Poisson normals point **into the material**, not into the room —
   the floor reads −z and the ceiling +z, because the reconstruction treats the
   space the scanner walked as the outside of the solid. So a face plane's
   material lies on the side its normal points to, and two faces bound one wall
   when they face each other across less than a wall's width.

4. **Walls.** Every 50 mm station along a face chooses its own opposite face —
   the nearest one carrying material there. Thickness is therefore measured per
   station, and a wall is cut only where that choice changes. A doorway does not
   end a wall: runs on the same pair of planes with a door's width between them
   are rejoined. Afterwards, parts that face each other across a wall are fused,
   because the far side of a corridor wall is broken into a face per room and no
   run-to-run pairing can match more than one of them.

5. **Slabs.** Horizontal triangles are gridded and grown into plateaus. Height
   clustering cannot separate a beam from its ceiling — the histogram is
   continuous — but a plateau is bounded by a vertical step, which region growing
   finds. A long narrow plateau hanging below the highest ceiling is a **beam**;
   a wide one is a **dropped ceiling**.

6. **Columns.** Tall near-square clusters that reach both floor and ceiling and
   whose surface is mostly vertical.

7. **The seam pass — this is the one that matters.** A plane band cuts the mesh
   at a flat boundary, and every fillet, jamb return, reveal and arch soffit in a
   junction falls outside every band by construction. Those are exactly the
   triangles that make a model read as one house rather than a set of panels. So
   after the geometric labelling, every unclaimed triangle that *touches* a part
   grows into it, bounded by how far that kind of part may reach: 250 mm off a
   wall's planes (a reveal, not a curtain), 550 mm below a ceiling level (a beam
   drop), 200 mm around a floor.

8. **What is left** is attached to nothing structural, and only that is dropped.

9. **Relief.** Each wall is unfolded onto its near face as a depth map in
   (along, height). A cell with no surface is a void — an opening if material
   spans over it. The top of the void, column by column, is a flat lintel or a
   curved **arch**. A cell deeper than the wall's own surface is a **niche**, one
   standing proud a **pilaster**. Depth is measured against the median of the
   face, not against the fitted plane, or a wall whose plane sits on the far
   face reads as one enormous niche.

## Running it

    venv311\Scripts\python.exe scripts\export\poisson_mesh.py <scan.las> output\model\poisson.ply
    venv311\Scripts\python.exe scripts\export\mesh_cache.py output\model\poisson.ply output\model\poisson.npz
    venv311\Scripts\python.exe scripts\export\modular_poisson.py --cache output\model\poisson.npz --out output\model\poisson_modular
    venv311\Scripts\python.exe scripts\export\modular_viewer.py output\model\poisson_modular output\model\poisson.npz

`PM_CUT` on `poisson_mesh.py` slices the ceiling off for a cutaway view. **Do not
use it for this stage** — without the ceiling slab, the dropped ceilings of the
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

## Reading the audit

`coverage` in the manifest answers the two questions that matter:

- `kept_frac` — how much of the scanned surface ended up in a named part.
- `crack_tris` — dropped triangles with **two different parts** around them.
  A triangle that merely changes part is a junction; one with two parts around it
  is a hole in the shell. This is the number that says "seamless", and on koushik
  it is 6 out of 7.9 million triangles.

## Result on koushik

82 parts — 39 walls (28 with a measured thickness), 19 floors, 22 ceilings of
which 2 beams and 7 dropped, 2 columns. 48 features: 15 doors, 4 windows, 2
arches, 10 niches, 6 pilasters. Clear height 2740 mm. 98.7% of the surface is in
a named part; 9 m² was dropped as clutter, in 76 pieces, the largest 1.2 m².
