# Plane-First Modular Reconstruction — Design Spec

- **Date:** 2026-07-02
- **Status:** Approved (design), pending implementation plan
- **Supersedes:** the density-image floorplan pipeline (`floorplan_reconstruct.py`, the density core of `floorplan_geometry.py`, `segment_walls_and_grooves.py`, `find_z_band.py`)
- **Leaves untouched:** the Poisson mesh pipeline (`reconstruct_mesh*.py`, `mesh_common.py`)

## 1. Goal

Turn a single indoor SLAM LiDAR scan into a **modular, SketchUp-clean 3D model plus a 2D vector floor plan**, where every building element — walls, floor, ceiling, doors, windows, **columns/pillars, beams** — is a **separate, named, editable object** importable into Blender.

Two hard quality bars, in priority order:

1. **Accurate openings (cutouts).** Doors/windows/balcony openings must be crisp rectangles located per real geometry, with low false-positives from furniture.
2. **Faithful relief.** Walls are **not** flat planes. Pillars/beams/pilasters/niches that **extrude in and out** of the wall face must be preserved as clean stepped geometry — never smoothed away (Poisson's failure) and never flattened into a single slab (naive-extrusion's failure).

Everything must be **crisp/low-poly** (constructed from planes + booleans, no organic smoothing) and run **CPU-only on Windows** on the existing stack.

### Why the previous approach failed (recorded so we don't repeat it)

A top-down **density image only counts points per 2D cell**. It has no free-space/visibility signal (cannot distinguish "empty = doorway" from "empty = unscanned" from "empty = furniture shadow") and it **collapses the vertical axis** (the exact signal that separates a door from a window). On a drifty handheld scan, walls smear into "gibberish." The fix, used by robotics/scan-to-BIM: detect walls as **real 3D vertical planes**, detect openings **per wall plane** in that plane's own 2D frame, and validate them with **visibility reasoning** from the scanner path.

## 2. Non-goals

- No multi-storey handling (input is single-storey; a Z-band isolates the one story). Multi-storey is future work.
- No true parametric BIM/IFC output in v1 (mesh/GLB only). IFC export is a documented future extension.
- No detection of **closed** doors or **glazed** windows that leave no physical hole — geometrically invisible; only a learned semantic prior (GPU) could catch them. Out of scope for v1.
- No colour/material reconstruction beyond passing through RGB where present.
- No curved/organic element modelling — confirmed input relief is **rectangular/prismatic**, so a single planar method covers everything (no mesh fallback).

## 3. Input data

- **Scanner:** Feima **SLAM2000** handheld SLAM LiDAR, post-processed in **SLAM GO POST PRO**.
- **Format:** LAS/LAZ, LAS 1.2 point format 3. Fields present & usable: `X,Y,Z`, **`gps_time`**, `intensity`, and **RGB** (populated on `mujammelexport`, zero on `koushikexport`). `return_number`/`scan_angle`/`point_source_id` are unpopulated.
- **Scale:** metres. ~23–27M points per scan.
- **Scene:** single-storey apartment. Measured: **floor ≈ −0.1 m, ceiling ≈ 2.4 m**, real footprint **~12 × 14 m**.
- **Contamination:** through the **balcony**, the scan captured the **neighbouring building — other apartments at multiple floor levels** (real, dense, outside the unit). The raw bbox (39 × 74 m, 23 m tall) is mostly this exterior plus SLAM drift. Must be removed automatically.
- **Trajectory:** no sidecar trajectory file, but `gps_time` is meaningful → the walk-path is **approximately recoverable** by temporal ordering. (User may export an explicit trajectory from SLAM GO POST in future; pipeline will use it if present.)

## 4. Outputs (per scan, in `out/<name>/`)

1. **`model.glb`** — modular scene; objects named `wall_00`, `floor`, `ceiling`, `door_00`, `window_00`, `column_00`, `beam_00`, sorted into Blender collections **Walls / Floor / Ceiling / Doors / Windows / Columns / Beams**.
2. **`floorplan.dxf`** and **`floorplan.svg`** — regularized wall centrelines/faces, room polygons, opening symbols, dimension labels.
3. **`manifest.json`** — every element with geometry + measurements (cutlist-ready), reusing/extending the existing `floorplan_schema` dataclasses.
4. **`report.txt`** — element counts, measured ceiling height, isolation stats (points kept vs discarded), and validation vs any tape-measured ground truth (reuses `validate_measurements.py`).
5. Optional **`scene.blend`** via headless `bpy` when Blender is available (same pattern as `obj_to_fbx.py`).

## 5. Architecture

New package `scripts/recon/` — small, single-purpose, unit-testable modules; pure-numpy/scipy where possible (Open3D only where it must be), mirroring the existing `floorplan_geometry` discipline (so tests run without Open3D).

| Module | Responsibility |
|---|---|
| `io_las.py` | Chunked LAS/LAZ load → arrays `xyz`, `gps_time`, `rgb`, `intensity`. |
| `clean.py` | Statistical-outlier removal; drift/percentile pre-cull. |
| `isolate.py` | Primary floor↔ceiling **Z-band** selection + **trajectory-anchored** interior extraction (removes neighbour building). |
| `trajectory.py` | Approximate walk-path from `gps_time` (or load explicit trajectory); provide sensor origins for visibility. |
| `frame.py` | Normal estimation; dominant-direction (Manhattan/Atlanta) estimation; gravity/axis alignment. |
| `planes.py` | Plane detection (`detect_planar_patches`/iterative `segment_plane`) + `cluster_dbscan`; label horizontal/vertical. |
| `structure.py` | Floor/ceiling extraction; group vertical faces into **stepped multi-plane walls**; extract **columns** & **beams**. |
| `regularize.py` | Axis snapping; corner resolution via **plane intersection**; wall topology/merge. |
| `openings.py` | Per-plane occupancy → interior voids → **visibility gating** → both-face cross-check → rectangle refine → door/window classify. |
| `solids.py` | Extrude polygons to watertight solids; **boolean-cut** openings (`manifold3d`); stepped-wall & column/beam boxes. |
| `floorplan2d.py` | `shapely` arrangement → room polygons → DXF/SVG. |
| `assemble.py` | Name elements, sort into collections, write `model.glb` (`trimesh`) and optional `.blend` (`bpy`). |
| `pipeline.py` | Orchestrator + CLI; config; writes all outputs + `report.txt`. |

**Dependencies added:** `trimesh`, `manifold3d`, `shapely`, `ezdxf` (DXF), `svgwrite` (SVG). Existing: `laspy`, `open3d`, `numpy`, `scipy`, `opencv-python-headless`. Optional: `bpy`.

## 6. Pipeline stages & key algorithms

**Stage A — Ingest & clean.** Chunked load (handles 27M pts). `remove_statistical_outlier`; voxel down-sample (~1–2 cm) for detection passes while keeping the full cloud for final measurement refits.

**Stage B — Isolate the unit.**
1. **Z-band:** histogram Z; the apartment is the dominant contiguous band (floor spike + ceiling spike). Select `[z_floor − ε, z_ceiling + ε]`. Robust and automatic (no manual `--z-band`).
2. **Trajectory anchor:** build the approximate walk-path (`trajectory.py`). Keep only points in the connected spatial region reachable from the path — grid/DBSCAN connectivity plus a distance-from-path cap. Neighbour building fails on both wrong-Z and across-the-gap disconnection; the **balcony** (walked, connected) is retained.
3. Report kept-vs-discarded counts.

**Stage C — Frame.** Estimate + orient normals; build a normal histogram / Gaussian-sphere vote → dominant horizontal wall axes + vertical. Rotate the cloud axis-aligned. Keep per-wall PCA fallback for non-orthogonal walls (Atlanta-world tolerant).

**Stage D — Planes.** Iteratively detect planar patches; DBSCAN each plane's inliers to split coplanar-but-separate faces; label `floor` (lowest horizontal), `ceiling` (highest horizontal), `vertical-face` (walls, pillar faces, beam faces).

**Stage E — Structure (relief-aware).**
- **Floor / ceiling:** the dominant lowest/highest horizontal planes → slab solids. Balcony region is open-top → no ceiling there.
- **Walls as stepped multi-plane solids:** cluster vertical faces by direction+location into wall runs; within a run, faces at **different perpendicular offsets** become **steps** (in/out relief preserved) rather than being averaged to one plane.
- **Columns (vertical) & beams (horizontal):** compact prismatic clusters that protrude from a wall run (or free-standing) are split out as their own named solids.

**Stage F — Regularize.** Snap face orientations/offsets to dominant axes (within tolerance); resolve wall corners by **line/plane intersection** (watertight, no dangling ends — salvages `_resolve_corner_point`); merge duplicates; pair parallel faces for **thickness** (salvages `pair_wall_surfaces` + modal-thickness fallback ~0.1 m for single-sided walls).

**Stage G — Openings (cutouts).** Per wall face:
1. Project inlier points into the plane's `(u,v)`; rasterize a 2–5 cm occupancy grid.
2. Find **interior empty connected components** enclosed by occupied cells → opening candidates (salvages `_interior_void_cells`).
3. **Visibility gate:** voxelize; 3D-DDA ray-cast from trajectory sensor origins → label cells EMPTY/OCCUPIED/UNKNOWN; accept a candidate only if its cells are EMPTY with rays crossing the plane (rejects furniture shadows). Fallback without reliable trajectory: require the void empty on **both** wall faces + interior-void bounding (salvages `cross_check_opening_both_faces`).
4. Refine edges: `cv2.minAreaRect` + per-edge RANSAC snap to the occupied/empty transition; regularize to vertical jambs + horizontal sill/lintel (salvages `refine_opening_edges`).
5. Classify **door vs window vs balcony-door** by sill height vs floor (salvages `classify_opening`).

**Stage H — Solids & cuts.** Extrude each wall step / floor / ceiling / column / beam polygon floor→ceiling (or to its measured extent) into a **watertight** solid (`trimesh.creation.extrude_polygon`). Build opening cutter solids (overshoot past wall thickness) and `manifold3d` boolean-subtract → crisp reveals.

**Stage I — 2D floor plan.** Extend regularized wall lines into a `shapely` arrangement; `polygonize` → closed room polygons; place opening symbols; export DXF (`ezdxf`) + SVG (`svgwrite`).

**Stage J — Assemble & export.** Name all solids; sort into collections; write `model.glb` (`trimesh.Scene`) and optional headless-`bpy` `.blend`; write `manifest.json` + `report.txt`.

## 7. Config / CLI

`python scripts/reconstruct.py <scan.las|.laz> <out_dir> [options]`

- `--balcony {include,stop-at-door}` (default `include`).
- `--trajectory <path>` (optional explicit trajectory; else derived from `gps_time`).
- `--voxel <m>` (default 0.015), `--wall-thickness-default <m>` (default 0.10).
- `--ground-truth <json>` (optional, triggers validation).
- `--no-blend` (skip `.blend` even if `bpy` present).

Single `DEFAULT_CONFIG` dict (same style as the existing pipeline) drives thresholds; every magic number lives there with a comment.

## 8. Salvage vs delete

**Salvage** (move into new modules, keep tests): two-pass wall-plane refit, `pair_wall_surfaces` + modal-thickness fallback, `_resolve_corner_point`/`snap_wall_endpoints`, `_interior_void_cells`, `cross_check_opening_both_faces`, `refine_opening_edges`, `classify_opening`, percentile crop (`crop_pcd_to_percentile_bounds`), `floorplan_schema` dataclasses, `validate_measurements.py`.

**Delete** after salvage: `floorplan_reconstruct.py`, `floorplan_reconstruct_test.py`, the density-image seeding in `floorplan_geometry.py`, `segment_walls_and_grooves.py`, `find_z_band.py`, and their now-dead tests. Update `README.md`.

## 9. Testing strategy

- **Synthetic fixture (extended):** grow `tests/fixtures.py` `two_room_house()` to include a **pillar (in/out relief), a beam, a balcony opening, and a stray "neighbour" point blob across a gap.** Every stage asserts against known ground-truth dims.
- **Unit tests per module** (pure-numpy modules testable without Open3D), mirroring current `test_floorplan_geometry.py` discipline.
- **CLI/integration test:** synthetic LAS → run pipeline → assert `model.glb`, `floorplan.dxf/svg`, `manifest.json` exist and element counts/dims are correct; assert the neighbour blob was **excluded**.
- **Real-scan smoke test:** small cropped patch of `koushikexport.las` (fast), like the current `*_test.py` pattern.
- **Ground-truth validation:** reuse `validate_measurements.py` against any tape measures.
- **TDD throughout:** write the failing test for a stage before its implementation.

## 10. Phased build plan (each phase independently verifiable)

- **Phase A — Skeleton + ingest + clean + isolate:** `io_las`, `clean`, `trajectory`, `isolate`. Acceptance: on `koushik`, outputs a clean single-unit cloud with the neighbour building removed; kept/discarded counts in `report.txt`; unit tests green.
- **Phase B — Frame + planes + floor/ceiling:** `frame`, `planes`, floor/ceiling in `structure`. Acceptance: correct floor/ceiling planes + axis alignment on fixture and real patch.
- **Phase C — Walls + relief + columns/beams + regularize:** rest of `structure`, `regularize`. Acceptance: fixture pillar & beam recovered as separate stepped solids with correct offsets; watertight corners.
- **Phase D — Openings + visibility:** `openings`. Acceptance: fixture door/window/balcony recovered with correct type & dims; neighbour-through-balcony does not spawn false walls; furniture-shadow test rejected.
- **Phase E — Solids + booleans + GLB + assembly:** `solids`, `assemble`. Acceptance: `model.glb` opens in Blender with correctly named collections and crisp cut openings.
- **Phase F — 2D floor plan + manifest + validation + docs:** `floorplan2d`, manifest, report, README. Acceptance: DXF/SVG render cleanly; manifest validates against ground truth within tolerance; old code deleted.

## 11. Risks & mitigations

- **Single-sided walls** (handheld sees one face) → default thickness from `DEFAULT_CONFIG`; flag `thickness_source` in manifest.
- **Trajectory approximation imperfect** → visibility is a *gate with a fallback* (both-face emptiness), never the sole signal; degrade gracefully.
- **Manhattan snap harming non-orthogonal walls** → snap only within tolerance; per-wall PCA fallback retained.
- **`manifold3d` needs watertight inputs** → construct solids as closed extrusions by design; validate manifoldness before boolean, repair or fall back to `bpy` EXACT solver.
- **Isolation over/under-cull** → `--balcony` toggle + optional bounding-box override as an escape hatch; `report.txt` surfaces counts for inspection.

## 12. Future extensions (documented, not built)

- IFC/BIM export (IfcOpenShell: `IfcWall`/`IfcSlab`/`IfcOpeningElement`/`IfcDoor`/`IfcWindow`/`IfcSpace`) → Blender via Bonsai.
- Optional GPU semantic prior (Superpoint Transformer / KPConv) to catch closed doors / glazed windows.
- Multi-storey support (Z-band storey splitting).
- PolyFit/abspy crisp watertight shell if extruded boxes ever prove insufficient.
