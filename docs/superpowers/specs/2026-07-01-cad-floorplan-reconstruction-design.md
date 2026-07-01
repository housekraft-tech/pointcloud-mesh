# CAD-style floor plan + wall/groove reconstruction from LiDAR scan

## Problem

The current pipeline has two prior stages:

1. `reconstruct_mesh.py` (Poisson) — geometrically complete but organic/smooth, not usable as a CAD base.
2. `segment_walls_and_grooves.py` (RANSAC planes + UV-grid cells) — attempts crisp walls/openings/grooves directly from 3D point binning, but the output is messy and not usable. This module is being replaced, not extended.

Goal: from a LiDAR/SLAM scan (LAS/LAZ), produce:

- A 2D floor plan (rendered image to start; CAD vector export can follow later) with wall centerlines, thickness, and dimensions.
- A clean, editable 3D model (OBJ, convertible to FBX via existing `obj_to_fbx.py`) suitable as a base for further editing in SketchUp — flat planar walls, clean rectangular openings, and clean groove/pillar geometry, not a mesh built from thousands of small per-voxel quads.

Critically, "grooves" are not noise to be removed — they are real as-built construction features: pillars or chases that recess into or protrude out of an otherwise flat wall, both vertically and horizontally oriented. These must be captured as accurate 3D geometry because they directly affect interior cutlists (e.g. a wardrobe back panel that must be cut with an L-notch to fit around a protruding pillar). Getting this wrong costs real money in production.

A related but separate future goal (explicitly out of scope for this spec) is using this scan-derived geometry to correct/validate an existing 2D-floorplan-derived 3D model used by a downstream BOM/cutlist system (`hk-backend` / `hk-customer-webapp`). That integration is deferred until this pipeline reliably produces accurate 3D output on its own.

## Non-goals

- Full parametric BIM/IFC export.
- Curved wall reconstruction (out-of-plane construction deviations are captured as grooves, not as true curved wall surfaces).
- The correction-layer integration with `hk-backend`/`hk-customer-webapp` (future work).
- DXF/vector CAD export (may follow later; first deliverable is a rendered floor plan image).

## Approach

Reuse the algorithmic technique from the open-source Cloud2BIM project (density-histogram → binary image → contour extraction → Douglas-Peucker simplification), ported into our own Open3D/numpy codebase rather than depending on Cloud2BIM directly (it's GPL-licensed, and its wall model assumes uniform thickness with no support for recesses/protrusions — the exact gap we need to fill).

The same core primitive — *project a band of points onto a 2D plane, build a density image, threshold, find contours, simplify* — is applied twice:

- Once horizontally (top-down) to extract wall lines / floor plan.
- Once per-wall in the wall's local (u = along wall, v = height) plane, sliced across wall depth, to extract groove/pillar shapes.

Layout is assumed "Atlanta-world": walls are vertical, but not forced onto a single pair of global orthogonal axes — each wall keeps its own detected direction, since the real building is mostly-but-not-strictly rectilinear.

## Staging

### Phase 1 — walls, floor plan, openings

1. Height-slice the recentered, downsampled cloud near ceiling height (configurable fraction of story height) → 2D density image → binary threshold → contour extraction → Douglas-Peucker simplification → candidate wall line segments.
2. Pair parallel segments within a max distance/overlap to get each wall's centerline + thickness.
3. Snap wall endpoints/intersections within a tolerance so corners meet cleanly.
4. Per wall, project the full-height point band into that wall's local (u,v) plane; 1D histogram along u for gap detection → opening rectangles; classify door/window/balcony door by sill height + width/height (reusing existing classification thresholds).
5. Output:
   - `floorplan.png` — rendered top-down floor plan with wall dimensions.
   - `manifest.json` — wall centerlines, thickness, openings (type/size/position).
   - `reconstructed.obj` — walls as flat extruded polygons (planar, low face count) with opening rectangles cut out.

### Phase 2 — grooves / pillars

1. For each wall, take points within the wall's search band whose distance from the nominal wall plane exceeds a threshold (recessed inward or protruding outward candidates).
2. Slice these residual points into fine height bands (fine enough to catch e.g. a wardrobe-height niche); per band, build the 2D density image in wall-local (u, depth) space, contour + simplify — same primitive as Phase 1, applied per band.
3. Stack per-band contours; merge contiguous bands with matching (u-range, depth) into a single 3D box/L-shaped volume — giving position, width, height, depth, and orientation (vertical or horizontal cut).
4. Extrude each groove as a clean box into or out of the wall; merge into `reconstructed.obj`; record dimensions in `manifest.json`.

## Accuracy strategy

mm-level accuracy is the primary success criterion (cutlist errors cost real money). Downsampling/voxel size is used only for search and contour-extraction speed — never the source of truth for a final number:

- Once a wall plane, opening edge, or groove boundary is roughly located via the density-image/contour method, its final position/thickness/depth is refined by a least-squares plane/edge fit on the **original, non-downsampled** points inside that region (extending the existing `refine_plane_model` approach to openings and grooves).
- A finer voxel size is used for this refinement pass than for the initial coarse detection pass.
- New `validate_measurements.py`: takes a small set of hand tape-measured ground-truth values (wall lengths/thickness, opening sizes, groove position/depth) keyed by wall/groove ID, diffs them against `manifest.json`, and reports per-measurement error in mm plus overall max/mean error. Run after every pipeline change to catch regressions before they reach a real cutlist.
- The scanner's own stated accuracy spec is documented as the target baseline; the pipeline should not add meaningfully more error on top of it.

## Testing

- Reuse the existing test-patch pattern (small spatial crop, point cap) to validate Phase 1 on a subset of the real scan locally before running the full scan on the Jarvis Labs VM.
- A script renders the floor plan and prints wall/opening dimensions for a quick sanity check against the real building.
- `validate_measurements.py` (above) is run against real tape measurements before trusting output for a production cutlist.

## Deferred

- Correction-layer integration with `hk-backend` (BOM/cutlist pillar-contact logic) and `hk-customer-webapp` (2D→3D ML editor, threejs JSON schema). Investigation of both codebases has been done separately and can be picked up when this pipeline's 3D output is validated as accurate.
- DXF/vector floor plan export.
