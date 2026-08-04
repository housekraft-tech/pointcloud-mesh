# Drawing-registered room segmentation — accuracy fix (Spec A)

Date: 2026-08-04
Status: design, approved for planning
Branch: walk-freespace-backup

## Problem

Room dimensions in the isolidarflow output are inaccurate. Measured against the
koushik architect plan (`floorplan_original.png`), the metrology clear-spans are
off by **1–2.4 m** on several rooms and **~100–240 mm** on the best ones — not
the sub-mm the metrology *precision* implied.

Root cause (confirmed by inspection of `output/koushik_metro2/manifest.json`):

1. **Open-plan areas have no dividing walls.** The living + dining + kitchen +
   foyer form one open space (~46 m²). Wall-based `floorplan2d.build_room_polygons`
   correctly bounds it as a single 8.7×5.2 m rectangle, but the plan labels
   *sub-zones* inside it (Living/Dining 6530×5000, Kitchen 2250×5050 …). **No
   wall-only method can recover sub-zones that have no walls between them.**
2. **Closed rooms wander ±100–240 mm and are unstable run-to-run** (6 vs 8 rooms
   across RANSAC seeds).

The metrology (`metrology.clear_between`) is precise but measures whatever room
polygon it is given; with wrong polygons it precisely measures the wrong walls.

## Goal & success criteria

Replace wall-only room segmentation with **drawing-registered** segmentation:
detect rooms on the architect drawing (which carries the sub-zones and closed
rooms), register the drawing to the LiDAR frame, segment 1:1, and feed the
labeled rooms to the existing metrology.

Success:

- Each labeled room's clear W×L matches the architect plan / manual survey to
  **cm level** (target ≤ ~50 mm on closed rooms), replacing today's 1–2.4 m.
- Open-plan public zone is split into Living / Dining / Kitchen / Foyer by the
  drawing's boundaries.
- Segmentation is **deterministic** for a given (scan, drawing) pair.
- Existing pipeline outputs (manifest, dxf, svg, glb) still produced; full test
  suite stays green.

Non-goals (this spec): columns / beams / grooves / arches detail capture (that is
**Spec B**, the relief engine, built on top of this); free-standing furniture;
multi-floor; generalization beyond the koushik + mujammel flats (hardening for
new drawings is a follow-up).

## Prior art (this is largely prototyped)

`scripts/experiments/register_drawing.py` already implements the whole chain and
`scripts/experiments/validate_rooms.py` validates it against the manual survey.
This spec **productionizes** that experiment into `scripts/recon/` and wires it
into `isolidarflow.py`. Reused pieces:

- `register_drawing.drawing_masks` — RF-DETR `elements` + `walls` models →
  room boxes+labels, plan crop, drawing wall mask, room seeds, wall segments.
- `register_drawing.main` — footprint pose search (rot 0/90/180/270 × mirror,
  area-scale, centroid align, translation hill-climb → footprint IoU), global
  wall refine (scale/angle/tx/ty hill-climb on wall distance-transform),
  per-wall perpendicular snapping, seeded `cv2.watershed` segmentation.
- `explain_lidar_to_3d.reconstruct` (metric free/occ rasters, `CELL` m/px) and
  `rfdetr_infer` (model loading/inference).
- `validate_rooms.py` — manual-survey room dims for the accuracy check.

## Architecture

Three new production modules under `scripts/recon/`, each single-purpose, plus a
swap inside `isolidarflow.py`. Nothing else in the flow changes.

```
floorplan_original.png ─► drawing.py   RF-DETR elements+walls → room boxes+labels,
                            │            plan crop, drawing wall mask, room seeds,
                            │            wall segments (metric-agnostic, px)
LiDAR (walls + occupancy) ─► register.py  footprint pose → global wall refine →
                            │            per-wall snap  ⇒ Transform (drawing px → LiDAR metric)
                            ▼
                         rooms.py       seeded watershed on LiDAR free-space with
                            │            drawing seeds + snapped drawing walls as barriers
                            │            ⇒ labeled room polygons (metric, in wall frame)
                            ▼
      isolidarflow.py:  labeled rooms REPLACE build_room_polygons →
                        metrology.clear_between per room → manifest / dxf / svg / glb
```

### `recon/drawing.py`

- `detect_drawing(img_path) -> DrawingModel` with: `plan_crop` (origin offset),
  `room_seeds: [{name, cx, cy}]` (crop px), `wall_mask` (crop px), `wall_segs:
  [(p0,p1)]` (crop px), `footprint_mask`.
- Wraps `rfdetr_infer` (elements + walls models). Depends on the RF-DETR model
  weights being present. **Step 0 of the plan verifies the model runs on
  `floorplan_original.png`** before anything else is built.

### `recon/register.py`

- `register(drawing: DrawingModel, lidar_free, lidar_occ, cell_m) -> Transform`
  where `Transform.apply(px_xy) -> metric_xy` in the LiDAR wall frame.
- Pose search + global wall refine + per-wall snap, ported verbatim from
  `register_drawing.main` (deterministic: fixed angle set, fixed hill-climb
  schedule, `np.random.default_rng(0)` only for colours which move out of the
  library into the caller).
- Emits a registration quality figure (footprint IoU, % walls within tol,
  n_snapped) used as an acceptance gate: if IoU or wall-match is below threshold,
  the pipeline **falls back to today's `build_room_polygons`** and flags
  `room_source="wall_only"` rather than shipping a bad registration.

### `recon/rooms.py`

- `segment_rooms(drawing, transform, lidar_free, lidar_walls, cell_m) ->
  [Room]` where each `Room` has `name`, `polygon` (metric, wall frame), `mask`,
  `area_m2`. Seeded `cv2.watershed`; snapped drawing walls close occluded-wall
  gaps so basins don't leak. Replaces `floorplan2d.build_room_polygons` as the
  pipeline's room source.

### Integration in `isolidarflow.py`

- New stage between walls and manifest: build `drawing`+`transform`+`rooms`.
- The rooms fed to `build_manifest`, `measure_room_clear_dims`, `write_dxf/svg`,
  and `build_room_model` are the **labeled** drawing-registered rooms.
- **Clear-span measurement stays the metrology, not the raster bbox.** The
  prototype measured clear W×L from the mask pixel bbox (cm-ish). We keep
  `metrology.clear_between`, but now run it across each *correctly bounded*
  room's principal axes (from the room polygon's oriented extent), so the
  dimension is both correctly bounded (drawing) and sub-mm measured (points).
- Manifest rooms gain `name` (Bedroom-1, Kitchen, …) and `room_source`
  ("drawing" | "wall_only" fallback).

## Data flow / contracts summary

| unit | input | output | depends on |
|---|---|---|---|
| drawing.py | plan image | seeds, wall mask, wall segs, footprint | rfdetr_infer, cv2 |
| register.py | drawing masks + LiDAR rasters | Transform + quality | reconstruct rasters, cv2, scipy |
| rooms.py | drawing + transform + rasters | labeled Rooms | cv2.watershed |
| isolidarflow | Rooms | manifest/dxf/svg/glb | metrology, schema, model, floorplan2d |

## Validation (hybrid, per the accuracy discussion)

- **Real koushik:** per-room clear dims vs the architect plan
  `floorplan_original.png` labels AND the manual survey (`validate_rooms.py`
  numbers) — assert cm-level, and specifically that the living/dining/kitchen
  split exists and each sub-zone is within tolerance.
- **Determinism test:** same (scan, drawing) → identical room set/dims.
- **Registration gate test:** a deliberately mis-scaled drawing → registration
  quality below threshold → pipeline falls back to wall_only, does not ship a
  bad answer.
- **Regression:** existing suite green; a scan with no drawing supplied still
  runs via the wall_only path.
- **Synthetic:** the two_room_house / modular_house fixtures get a tiny
  synthetic "drawing" (room boxes) so segmentation is unit-testable without the
  RF-DETR model.

## Key risks

1. **RF-DETR model availability / domain.** Whole spec hinges on `elements` +
   `walls` running on `floorplan_original.png`. Memory `rfdetr-domain-gap` says
   they work great on architect drawings. Plan step 0 verifies before building.
2. **Registration robustness.** Prototyped on koushik (single flat). Ship with
   the quality gate + wall_only fallback so a bad registration degrades to
   today's behavior instead of shipping garbage. Generalization = follow-up.
3. **Drawing availability.** Requires an architect plan per scan. When absent,
   wall_only fallback. (Product already assumes a 2D drawing exists — see
   `product-vision-deviation-detection`.)
4. **Metric bbox vs metrology.** Keep metrology.clear_between; do not regress to
   the prototype's raster bbox measurement.

## Open questions (resolved)

- Drawing to use: `floorplan_original.png` (confirmed).
- Detail capture (columns/beams/grooves/arches): separate **Spec B**, after this.
- Validation: architect plan + manual survey (hybrid), cm-level target.
