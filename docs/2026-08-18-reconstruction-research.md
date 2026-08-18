# How to make this better: research, then what is actually buildable

Written 18 Aug 2026, after measuring the current pipeline honestly.

## Where we actually are

Two audits against 5M scanned points (`scripts/export/audit_*.py`):

| | within 10 mm | unsupported surface | notes |
|---|---|---|---|
| Voxel surface | 57% of drawn surface | **0.0%** | it IS the scan, resampled |
| Modular model | 21% of drawn surface | **30.2%** (335 m2) | a third of it is invented |

Scanned point -> nearest modelled surface: only **63.3% within 10 mm**, and
**13.9% of scanned points have no modelled surface within 100 mm**.

The `align_check` figure I quoted for days -- 3 mm median -- only ever measured
faces on walls seen from BOTH sides. That is about half the model, and the half
it got right. It never looked at invented surface.

### Why the model invents surface

1. **48 of ~94 wall faces have assumed thickness**, not measured. Single-faced
   walls get the 195 mm median. Every one is a guess.
2. **Corner closing** extends wall ends up to 450 mm to meet perpendicular
   walls. Drawn masonry, nothing behind it.
3. **Boxes are independent.** Nothing in the representation prevents two walls
   occupying the same space -- which is why duplicates had to be hunted with
   heuristics (`rejoin`) rather than being impossible.
4. **Diagonal and curved walls cannot be expressed at all** (~0.7 m2).

All four are symptoms of one thing: **the model is a pile of independent boxes,
each fitted locally.** Nothing ties them into a consistent whole.

## What the research does instead

### PolyLayout (ghanning.github.io/PolyLayout)
Manhattan layout from posed images. Parameterises rooms as **Manhattan 3D
polygons optimised JOINTLY across rooms**, with **iterative wall split and merge**
for adaptive topology, and exploits shared building structure (dominant
directions, ground plane, ceiling height). Separates a learned scoring function
from explicit model-based geometry, which is what lets it generalise.

Transferable regardless of the image/LiDAR difference:
- rooms as closed polygons, so a wall is SHARED between two rooms rather than
  detected twice -- duplicates become structurally impossible
- one global optimisation instead of per-wall greedy fitting
- split/merge as a first-class topology operation, not a post-hoc `rejoin` hack

### DeWorldSG (deworldsg2026.github.io, ECCV 2026)
3D semantic scene graphs from RGB-D, spatio-temporal evidence aggregation,
incremental merging of local graphs into a global graph. Less directly
applicable to a bare shell, but the framing matters: **a structured graph with
explicit relations** (wall bounds room, opening lies in wall, column thickens
wall) rather than a flat parts list. That is also what a cutlist/BOM consumer
wants.

### The line that actually solves our problem: space partition + labelling
PolyFit (binary integer program, watertight+manifold by hard constraint),
Kinetic Shape Reconstruction / KSR (grow planar primitives until they collide,
better than O(n^3)), and CGAL's **Kinetic Surface Reconstruction** (shipped
2024): detect planes -> regularize parallel/coplanar/orthogonal -> partition
space into convex polyhedra -> **label each volume inside/outside by min-cut**
-> extract the boundary.

The output is **watertight because it is a union of volumes**. Applied here:
- a wall's interior is a CELL, bounded by its two detected faces. Thickness is
  never assumed -- it is the distance between two planes, and the cell between
  them is labelled solid because the evidence says so
- two walls cannot occupy one space, so duplicates are impossible
- surface exists only between a solid cell and an empty one, so unsupported
  surface is bounded by construction
- `lambda` trades data faithfulness against complexity, which is exactly the
  coverage-vs-part-count dial I have been tuning by hand

Also relevant, floorplan-specific and learned: **RoomFormer** (CVPR 2023,
two-level queries, single-stage polygon prediction from point-cloud density) and
**PolyRoom** (ECCV 2024, room-aware transformer). Both output room polygons
directly. Both need training data we do not have.

## Research vs reality

| Option | Verdict here |
|---|---|
| CGAL Kinetic Surface Reconstruction | Best-in-class, but C++ and **no Python bindings**. Would need a binding layer or a C++ side-tool. |
| `abspy` (pure-Python a-BSP cell complex + graph cut) | Right abstraction, on PyPI -- but **hard-depends on SageMath** for exact rational polyhedra. Conda-scale install; not viable on this machine (3 GB free). Tried; blocked. |
| RoomFormer / PolyRoom | Need annotated training data and a GPU. Not viable. |
| open3d Poisson | Installed in `.venv311`, but too heavy for this machine (already established). |
| **Manhattan cell complex, written here** | **Viable, and most of the benefit.** See below. |

## The realistic plan

The general case needs kinetic partitioning because planes point in arbitrary
directions. **This building is rectilinear** -- which collapses the hard part.
With axis-aligned planes only, the cell complex is just the outer product of
three sorted plane lists, and every cell is a box. That is a numpy exercise, not
a computational-geometry project.

1. **Collect planes.** Already done: `refine()` finds wall faces in the raw
   points to a few mm. Take every measured face as a candidate plane, per axis,
   plus floor and ceiling. Regularize: merge planes within ~15 mm (this is
   PolyFit/KSR's coplanarity regularization, and it is also what kills
   duplicates at the source).
2. **Build the cell complex.** Sort unique X planes, Y planes, Z planes. Cells
   are the boxes between consecutive planes -- typically 40x40x4, so a few
   thousand cells. No Sage, no kinetic anything.
3. **Label cells solid/empty by graph cut.** Data term from evidence already
   computed: point density inside the cell (solid), and free-space/visibility
   from the scanner (empty). Smoothness term penalises boundary area, which is
   `lambda`. `scipy.sparse.csgraph.maximum_flow` or PyMaxflow does the min-cut;
   both are light.
4. **Extract the boundary.** Faces between a solid cell and an empty one. This
   is watertight by construction, has no duplicates by construction, and
   invents nothing that the data term did not vote for.
5. **Keep the modular output.** Group boundary faces back into named parts --
   walls, panels, openings, columns -- reusing the classification that already
   works (doors match ground truth at 9/9). The cell complex replaces HOW
   geometry is decided, not the semantics on top.

Expected effect on the measured failures: assumed thickness disappears
(1), corner-closing disappears (2), duplicates become impossible (3). Diagonal
walls (4) still need oriented planes -- that is the one thing this does not fix,
and it is ~0.7 m2.

Honest risk: the graph cut needs a free-space term to work well, which means
ray-casting from scanner positions. We do not have scanner trajectory in the
LAS. Fallback is "empty unless proven solid", which is weaker but workable
because the ceiling and floor bound the volume.

## Sources
- PolyLayout: https://ghanning.github.io/PolyLayout/
- DeWorldSG: https://deworldsg2026.github.io/
- CGAL Kinetic Surface Reconstruction: https://doc.cgal.org/latest/Kinetic_surface_reconstruction/index.html
- CGAL announcement: https://www.cgal.org/2024/05/29/Kinetic_surface_reconstruction/
- Kinetic Shape Reconstruction: https://dl.acm.org/doi/fullHtml/10.1145/3376918
- abspy: https://github.com/chenzhaiyu/abspy
- points2poly / deep implicit fields: https://3d.bk.tudelft.nl/liangliang/publications/2022/implicit_field/implicit_field.pdf
- RoomFormer: https://github.com/ywyue/RoomFormer
- PolyRoom: https://arxiv.org/abs/2407.10439
