# Detailed Modular Mesh — Poisson relief + skeleton structure

**Date:** 2026-07-21
**Status:** Design (approved approach A; pending user review of this spec)
**Scans:** koushik (geometry) / mujammel (RGB) — same flat, same frame.

## Problem

We have two 3D outputs of the same apartment, each good at half of what we want:

- **Poisson mesh** (`output2/koushik_all/mesh/poisson_complete.obj`, ~716 MB,
  3.9 M verts / 7.9 M tris): a watertight surface wrap of the raw cloud. It
  carries the **real captured detail** — solid walls, door openings + reveals,
  columns, and the ceiling/beam relief. Confirmed 2026-07-21 by ceiling-removed
  cutaway (clean readable rooms) and per-wall close-ups (flat planar wall bodies
  + preserved door openings). Weaknesses: one undifferentiated blob (no per-wall
  / per-room separation), furniture baked in, no thickness semantics.
- **Skeleton model** (`output2/*/skeleton_3d/`): clean *structure* — a
  de-duplicated per-wall plane list (`continuous/measurements.json`, 37 walls
  koushik / 45 mujammel, each a fitted plane with sub-mm placement) and 1:1 room
  polygons from drawing→LiDAR registration (`registration/transform.json`,
  `room_measurements.json`). Weakness: geometry is flat/fragmented; loses all
  relief; walls duplicated per room in the render.

**Goal:** one assembled house where **each wall is its own detailed object**
carrying the Poisson relief, **each room is its own group** (built from the
walls that bound it), **floor and ceiling are single shared slabs** (ceiling
carries the beam relief), and **furniture is removed**. Digital/on-screen —
not for physical printing (watertightness not required).

## Approach (A — skeleton-guided crop of the Poisson)

Drive **structure** from the skeleton scaffold, take **detail** from the Poisson,
by cropping the Poisson mesh into per-element slabs using the measured wall
planes and room polygons as cutters.

Rejected alternatives:
- **B (segment Poisson directly, RANSAC plane clustering):** heavy + error-prone
  on a 700 MB noisy blob; clean per-wall separation hard. Only if A's scaffold
  proves too incomplete.
- **C (clean skeleton walls + baked displacement):** most engineering; relief is
  an approximation, not the true captured surface — defeats the point.

### Coordinate frame

All three inputs share one metric Z-up frame after a single Y-up→Z-up rotation
`(x,y,z)->(x,-z,y)` on the Poisson OBJ (verified: Poisson bbox after rotation
`[-4.86,-7.97,-0.44]..[8.06,6.73,2.70]` matches skeleton
`[-4.83,-7.78,-0.34]..[7.97,6.14,2.52]` and measurements' floor/ceiling
z = -0.216..2.519). `measurements.json` is already in this frame.

## Components

Small, independently testable units. Each takes the shared Z-up Poisson mesh
(loaded + rotated once) plus scaffold JSON, and emits named sub-meshes.

1. **`frame.py` — loader/normalizer.** Load Poisson OBJ, detect Y-up, rotate to
   Z-up, compute vertex normals, return `(mesh, verts, z_floor, z_ceiling)`.
   `z_floor/z_ceiling` from `measurements.json` room medians (not mesh bbox,
   which includes overshoot). One place owns the frame.

2. **`wall_slabs.py` — per-wall crop.** For each wall in `measurements.json`
   (`center, dir, normal, tmin, tmax`): keep Poisson verts with
   `tmin-ε < along < tmax+ε`, `|perp| < HALF_THICK` (≈0.09 m, half a ~110 mm
   wall + margin), and `z_floor+SKIRT < z < z_ceiling-CROWN` (trim floor/ceiling
   returns — the leak seen in close-ups). Emit object `wall_NN`. De-duplicated by
   construction: one slab per master plane, shared partitions appear once.
   - Config: `HALF_THICK`, `SKIRT` (≈0.04 m keep skirting), `CROWN` (≈0.04 m),
     `along_margin`.
   - Furniture drops out automatically: it floats in room interiors, outside
     every `|perp| < HALF_THICK` band.

3. **`slabs_floor_ceiling.py` — shared slabs.** Floor = verts with
   `z < z_floor+FLOOR_BAND`; ceiling = verts with `z > z_ceiling-CEIL_BAND`,
   where `CEIL_BAND` is deep enough (≈0.30 m) to keep beams hanging below the
   ceiling plane. Each is ONE object (`floor`, `ceiling`), not per-room. This is
   where beam relief lives.

4. **`columns.py` — free-standing relief rescue.** The one gap in A: a column
   mid-room is near no wall plane, so steps 2–3 drop it. Detect residual Poisson
   geometry NOT claimed by any wall/floor/ceiling slab that forms a tall vertical
   cluster (spans most of floor→ceiling, small footprint), emit as `column_NN`.
   Everything else unclaimed (furniture) is discarded. Report counts of claimed
   vs discarded verts so silent loss is visible.

5. **`rooms.py` — room grouping (not new geometry).** From
   `registration/room_measurements.json` room polygons (transformed to the LiDAR
   frame via `transform.json`), assign each `wall_NN` to the room(s) whose
   polygon edge it lies on (a shared partition belongs to both — recorded as
   membership, wall geometry still stored once). Output a manifest mapping
   room → wall/column object names; floor + ceiling are global.

6. **`assemble.py` — writer.** Write one OBJ/GLB with every object named
   (`wall_00`, …, `floor`, `ceiling`, `column_00`, …) so a viewer can toggle per
   object, plus `modular_manifest.json` (room→objects, per-object vert/tri counts,
   bbox, source wall id). Optionally per-room OBJs for isolated inspection.

## Data flow

```
poisson_complete.obj ─┐
                      ├─ frame.py ─ (mesh Z-up, z_floor, z_ceiling)
measurements.json  ───┘        │
                               ├─ wall_slabs.py      → wall_00..NN
                               ├─ slabs_floor_ceiling→ floor, ceiling
                               ├─ columns.py         → column_00..NN, discard report
registration/*.json ───────────┴─ rooms.py           → room→objects manifest
                                       │
                                   assemble.py → detailed_modular.obj/.glb
                                                 + modular_manifest.json
                                                 + per-room OBJs (optional)
```

## Output

`output2/<scan>/skeleton_3d/detailed_modular/`:
- `detailed_modular.obj` / `.glb` — one house, every wall/column/floor/ceiling a
  named object.
- `modular_manifest.json` — room→object map, per-object stats, claimed/discarded
  vert counts.
- `rooms/room_NN_<label>.obj` — optional per-room isolated meshes.
- `render/` — shaded + normal + depth close-ups per object for QA (reuse
  `render_wall_closeup.py` / `render_interior.py`).

## Error handling / honesty

- **Coverage report:** log claimed vs discarded vert %, and per-wall vert counts.
  A wall with <500 verts (e.g. wall02, 681 verts — furniture-occluded) is flagged
  `low` and kept as-is from the skeleton plane rather than a broken crop.
- **No silent truncation:** if a room polygon has no wall slab on an edge
  (partition the LiDAR never saw), log it — that edge stays open, consistent with
  the known missing-partition limit; we do NOT fabricate a wall.
- **Open shells are acceptable** (digital viewing). Watertightness/thickening is
  explicitly out of scope (that was the physical-print path, deferred).

## Scope / YAGNI

**In:** per-wall detailed crop, shared floor+ceiling with beam relief, column
rescue, room grouping manifest, named assembled OBJ/GLB, QA renders.
**Out (deferred):** physical-print prep (thickening, manifold repair, scaling,
connectors); RGB/texture baking from mujammel onto the slabs; per-wall deviation
report vs drawing/survey (separate downstream task, unblocked once modules exist).

## Success criteria

1. Assembled OBJ opens with each wall/floor/ceiling/column as a separate, toggleable, named object.
2. Furniture visibly gone; wall bodies retain real relief + door openings from the Poisson (verified against the 2026-07-21 close-ups).
3. Floor and ceiling are single shared slabs; ceiling shows beams.
4. Shared partitions appear exactly once (no duplication).
5. Manifest maps every room to its bounding walls; coverage report accounts for ~all structural verts.
