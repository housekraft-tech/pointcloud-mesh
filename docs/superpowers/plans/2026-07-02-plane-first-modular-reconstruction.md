# Plane-First Modular Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a single SLAM2000 indoor scan into a modular, SketchUp-clean 3D model (separate named walls/floor/ceiling/doors/windows/columns/beams as GLB) plus a 2D vector floor plan, with accurate visibility-gated openings and preserved wall relief.

**Architecture:** Plane-first geometric pipeline. Isolate the unit (Z-band + trajectory-anchored connectivity) → detect all planar patches → build stepped relief-preserving walls + column/beam solids → per-wall-plane visibility-gated openings → extrude + boolean-cut into named solids → assemble GLB + DXF/SVG. Small single-purpose modules under `scripts/recon/`, pure-numpy where possible (Open3D only where required), TDD throughout.

**Tech Stack:** Python 3.11, laspy, Open3D 0.19, numpy, scipy, opencv-python-headless, shapely, trimesh, manifold3d, ezdxf, svgwrite; optional bpy. pytest.

## Global Constraints

- **Platform:** Windows; venv at `.claude/worktrees/cad-floorplan-reconstruction/venv311` (python `venv311/Scripts/python.exe`).
- **CPU-only.** No GPU dependency in v1.
- **Units:** metres throughout. World frame = LAS coordinates.
- **Testability:** every module except `io_las`, `planes`, `frame`, `assemble` must import without Open3D/trimesh/bpy (pure numpy/scipy/cv2/shapely) so unit tests run on any machine — mirror the existing `floorplan_geometry.py` discipline.
- **Config:** one `DEFAULT_CONFIG` dict in `pipeline.py`; every threshold lives there with a comment. No magic numbers inline.
- **Manifest:** reuse `scripts/floorplan_schema.py` dataclasses (`Wall`, `Opening`); extend with `Column`, `Beam`, `WallStep` (Task 2).
- **Commit cadence:** one commit per completed task (test + impl together). Commit trailers as in repo convention.
- **Do not touch** `reconstruct_mesh*.py` / `mesh_common.py` (Poisson mesh path stays).
- **Input reality:** single-storey; floor ≈ −0.1 m, ceiling ≈ 2.4 m; ~12×14 m unit inside a 39×74 m cloud contaminated by the neighbour building seen through the balcony; `gps_time` present, no trajectory sidecar; RGB present on mujammel only.

## File Structure

| File | Responsibility | Pure? |
|---|---|---|
| `scripts/recon/__init__.py` | package marker | — |
| `scripts/recon/schema.py` | extend schema: `Column`, `Beam`, `WallStep`, `ScanData`, `Plane` dataclasses + `to_dict` | pure |
| `scripts/recon/io_las.py` | chunked LAS/LAZ load → `ScanData` | Open3D/laspy |
| `scripts/recon/clean.py` | outlier removal, percentile pre-crop | Open3D + pure |
| `scripts/recon/trajectory.py` | approx path from `gps_time`; load explicit trajectory | pure |
| `scripts/recon/isolate.py` | Z-band select + trajectory-anchored unit extraction | pure |
| `scripts/recon/frame.py` | normals, dominant axes, axis-align | Open3D |
| `scripts/recon/planes.py` | plane detection + DBSCAN + label | Open3D |
| `scripts/recon/structure.py` | floor/ceiling, stepped walls, columns, beams | pure |
| `scripts/recon/regularize.py` | axis snap, corner intersection, merge, thickness | pure |
| `scripts/recon/openings.py` | per-plane voids, visibility gate, refine, classify | pure |
| `scripts/recon/solids.py` | extrude + boolean cut → trimesh solids | trimesh/manifold3d |
| `scripts/recon/floorplan2d.py` | shapely arrangement → rooms → DXF/SVG | shapely/ezdxf/svgwrite |
| `scripts/recon/assemble.py` | name, collections, write GLB / .blend | trimesh/bpy |
| `scripts/reconstruct.py` | CLI orchestrator + `DEFAULT_CONFIG` | mixed |
| `tests/recon/test_*.py` | one test module per recon module | pure |
| `tests/fixtures.py` | extend `two_room_house()` → add pillar, beam, balcony, neighbour blob | pure |

---

## Task 1: Package scaffold + dependencies

**Files:**
- Create: `scripts/recon/__init__.py` (empty), `tests/recon/__init__.py` (empty)
- Modify: `requirements.txt` (add `trimesh`, `manifold3d`, `shapely`, `ezdxf`, `svgwrite`)

**Interfaces:** Produces: importable `scripts.recon` package.

- [ ] **Step 1:** Add deps to `requirements.txt` (append):
```
trimesh>=4.4.0
manifold3d>=2.5.0
shapely>=2.0.0
ezdxf>=1.3.0
svgwrite>=1.4.0
```
- [ ] **Step 2:** Install: `venv311/Scripts/python.exe -m pip install trimesh manifold3d shapely ezdxf svgwrite`
- [ ] **Step 3:** Create the two empty `__init__.py` files.
- [ ] **Step 4:** Verify: `venv311/Scripts/python.exe -c "import trimesh, manifold3d, shapely, ezdxf, svgwrite; import scripts.recon; print('ok')"` → prints `ok`.
- [ ] **Step 5:** Commit: `git add requirements.txt scripts/recon tests/recon && git commit -m "Add recon package scaffold and deps"`

---

## Task 2: Extend schema (Column, Beam, WallStep, ScanData, Plane)

**Files:**
- Create: `scripts/recon/schema.py`, `tests/recon/test_schema.py`

**Interfaces:**
- Consumes: `scripts/floorplan_schema.py` (`Wall`, `Opening`).
- Produces:
  - `@dataclass ScanData(xyz: np.ndarray, gps_time: np.ndarray|None, rgb: np.ndarray|None, intensity: np.ndarray|None)` with `.n` property and `.subset(mask)->ScanData`.
  - `@dataclass Plane(normal: tuple, d: float, label: str, inlier_idx: np.ndarray)`; label ∈ {"floor","ceiling","vertical"}. Method `signed_distance(pts)->np.ndarray`.
  - `@dataclass WallStep(offset_m: float, u_min_m: float, u_max_m: float, z_min_m: float, z_max_m: float)`.
  - `@dataclass Column(column_id, footprint: list[tuple], z_min_m, z_max_m)` + `new_column_id(i)`.
  - `@dataclass Beam(beam_id, p0, p1, width_m, depth_m, z_min_m, z_max_m)` + `new_beam_id(i)`.
  - Add `steps: list = field(default_factory=list)` usage note (stored on `Wall` via a parallel dict since `Wall` is in the old module — keep `WallStep`s in a `dict[wall_id, list[WallStep]]` produced by `structure`).

- [ ] **Step 1: Write failing test** `tests/recon/test_schema.py`:
```python
import numpy as np
from scripts.recon.schema import ScanData, Plane, WallStep, Column, Beam, new_column_id

def test_scandata_subset_keeps_aligned_fields():
    xyz = np.arange(12, dtype=float).reshape(4, 3)
    t = np.array([1., 2., 3., 4.])
    s = ScanData(xyz=xyz, gps_time=t, rgb=None, intensity=None)
    assert s.n == 4
    sub = s.subset(np.array([True, False, True, False]))
    assert sub.n == 2
    assert np.allclose(sub.gps_time, [1., 3.])
    assert sub.rgb is None

def test_plane_signed_distance():
    p = Plane(normal=(0., 0., 1.), d=-2.0, label="floor", inlier_idx=np.array([]))
    d = p.signed_distance(np.array([[0., 0., 2.0], [0., 0., 3.0]]))
    assert np.allclose(d, [0.0, 1.0])

def test_column_id_format():
    assert new_column_id(3) == "column_003"
```
- [ ] **Step 2:** Run → FAIL (module missing): `venv311/Scripts/python.exe -m pytest tests/recon/test_schema.py -v`
- [ ] **Step 3:** Implement `scripts/recon/schema.py` with the dataclasses above. `Plane.signed_distance(pts) = pts @ normal + d`. `ScanData.subset` indexes every non-None array with the mask.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit: `git commit -am "Add recon schema (ScanData, Plane, WallStep, Column, Beam)"`

---

## Task 3: io_las — chunked load → ScanData

**Files:** Create `scripts/recon/io_las.py`, `tests/recon/test_io_las.py`

**Interfaces:**
- Produces: `load_scan(path: str, max_points: int|None=None, rng_seed: int=0) -> ScanData`. Reads XYZ (scaled, float64), `gps_time` if present else None, RGB (uint8, downscaled from uint16) if any nonzero else None, intensity if present. Chunked read (3M) to bound memory; if `max_points` set, uniform-random subsample.

- [ ] **Step 1: Write failing test** (writes a tiny synthetic LAS with laspy, reads it back):
```python
import numpy as np, laspy
from scripts.recon.io_las import load_scan

def _write_las(path, xyz, t=None):
    h = laspy.LasHeader(point_format=3, version="1.2")
    h.scales = [0.001, 0.001, 0.001]; h.offsets = [0, 0, 0]
    las = laspy.LasData(h)
    las.x, las.y, las.z = xyz[:,0], xyz[:,1], xyz[:,2]
    if t is not None: las.gps_time = t
    las.write(str(path))

def test_load_scan_roundtrip(tmp_path):
    xyz = np.array([[0,0,0],[1,2,3],[4,5,6]], float)
    p = tmp_path/"a.las"; _write_las(p, xyz, t=np.array([10.,11.,12.]))
    s = load_scan(str(p))
    assert s.n == 3
    assert np.allclose(np.sort(s.xyz[:,0]), [0,1,4], atol=1e-3)
    assert s.gps_time is not None and np.allclose(np.sort(s.gps_time), [10,11,12])
```
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement chunked `load_scan` using `laspy.open(...).chunk_iterator(3_000_000)`, concatenating arrays; RGB present-check = `any(red|green|blue nonzero)`; subsample with `np.random.default_rng(rng_seed)`.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit.

---

## Task 4: clean — outlier removal + percentile pre-crop

**Files:** Create `scripts/recon/clean.py`, `tests/recon/test_clean.py`

**Interfaces:**
- Produces:
  - `percentile_crop(scan: ScanData, lo=1.0, hi=99.0, margin_m=0.5) -> ScanData` — keep points within [lo,hi] percentile bbox (per-axis) expanded by margin. Pure numpy (salvage logic from `mesh_common.crop_pcd_to_percentile_bounds` but operate on `ScanData`).
  - `remove_outliers(scan: ScanData, nb_neighbors=20, std_ratio=2.0) -> ScanData` — Open3D statistical outlier removal, returns filtered `ScanData` (uses returned index mask via `subset`).

- [ ] **Step 1: Write failing test** for `percentile_crop` (pure, no Open3D): cluster of 100 points in [0,1]^3 + 5 far outliers at 100 → crop keeps ~100, drops the 5.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement `percentile_crop` (numpy `np.percentile` per axis, boolean mask, `scan.subset`). Implement `remove_outliers` guarded by `import open3d` inside the function.
- [ ] **Step 4:** Run → PASS (test only exercises `percentile_crop`).
- [ ] **Step 5:** Commit.

---

## Task 5: trajectory — approximate walk-path from gps_time

**Files:** Create `scripts/recon/trajectory.py`, `tests/recon/test_trajectory.py`

**Interfaces:**
- Produces:
  - `approx_trajectory(gps_time, xyz, dt_s=0.25) -> np.ndarray (M,3)` — bin points by `gps_time` into `dt_s` slices; each slice's sensor position ≈ centroid of that slice's points (coarse but sufficient for connectivity anchoring); return ordered path.
  - `load_trajectory(path) -> np.ndarray (M,3)` — parse a whitespace/CSV file with columns containing time,x,y,z (auto-detect); returns XYZ ordered by time.

- [ ] **Step 1: Write failing test:** synthetic — 1000 points whose `gps_time` increases along a known line; assert `approx_trajectory` returns a monotone path whose endpoints match the line ends within tolerance.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement binning via `np.digitize` on `gps_time`; per-bin centroid; drop empty bins.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit.

---

## Task 6: isolate — Z-band + trajectory-anchored unit extraction

**Files:** Create `scripts/recon/isolate.py`, `tests/recon/test_isolate.py`

**Interfaces:**
- Produces:
  - `select_z_band(z, bin_m=0.05, min_height_m=1.8) -> (z_floor, z_ceiling)` — histogram Z; find the dominant contiguous band whose extent ≥ `min_height_m` bounded by the two tallest spikes (floor, ceiling). Return the band edges.
  - `isolate_unit(scan, trajectory, z_band, cell_m=0.25, max_gap_cells=1, max_dist_m=8.0) -> (ScanData, dict)` — (1) keep points in `[z_floor-0.15, z_ceiling+0.15]`; (2) rasterize XY occupancy at `cell_m`; flood-fill the connected occupied region containing the trajectory's XY footprint (`max_gap_cells` bridging); (3) drop points beyond `max_dist_m` from the nearest trajectory vertex. Return filtered scan + stats dict `{kept, dropped, z_floor, z_ceiling}`.

- [ ] **Step 1: Write failing test for `select_z_band`:** synthetic Z with a floor spike at 0.0, ceiling spike at 2.5, uniform wall fill between, plus scattered points at 6–15 m (neighbour floors). Assert band ≈ (0.0, 2.5) within 0.1.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement histogram + dominant-band search. **Step 4:** PASS.
- [ ] **Step 5: Write failing test for `isolate_unit`:** a 4×4 m occupied XY blob (the unit) + a disconnected 2×2 blob 10 m away (neighbour) + trajectory inside the unit. Assert neighbour blob points are dropped, unit points kept.
- [ ] **Step 6:** Run → FAIL. **Step 7:** Implement occupancy flood-fill (scipy `binary_dilation`/`label`) anchored at trajectory cells + distance cap (cKDTree). **Step 8:** PASS.
- [ ] **Step 9:** Commit.

---

## Task 7: Phase A integration — CLI stage 1 on real data

**Files:** Create `scripts/reconstruct.py` (partial: ingest→clean→isolate + `report.txt`), `tests/recon/test_pipeline_isolate_cli.py`

**Interfaces:** Produces `run_isolation(in_path, out_dir, config) -> ScanData` and a `__main__` that (for now) writes the isolated cloud to `<out>/isolated.las` + `report.txt`.

- [ ] **Step 1: Write failing CLI test:** synthetic LAS (unit blob + far neighbour blob + drift) → run → assert `isolated.las` exists, point count ≈ unit only, `report.txt` contains kept/dropped counts and z-band.
- [ ] **Step 2–4:** Implement, run → PASS.
- [ ] **Step 5: Real-data check (manual, recorded in commit body):** run on `koushikexport.las`; confirm neighbour building removed and footprint ≈ 12×14 m. `venv311/Scripts/python.exe scripts/reconstruct.py ../../../koushikexport.las output/koushik_iso`
- [ ] **Step 6:** Commit: `git commit -am "Phase A: ingest+clean+isolate CLI; validated on koushik"`

> **CHECKPOINT — review Phase A output before Phase B.**

---

## Task 8: fixtures — synthetic house with pillar, beam, balcony, neighbour

**Files:** Modify `tests/fixtures.py`

**Interfaces:** Produces `modular_house() -> (xyz, gps_time, meta)` where `meta` holds ground-truth dims: room 6×5×2.7 m; one interior **pillar** 0.3×0.3 m protruding from a wall; one **beam** under the ceiling; a **door** (floor-to-header) and a **window** (sill 1.0 m); a **balcony opening** (wide) in one wall with a small **neighbour point blob 10 m beyond it**; `gps_time` increasing along a walk-path that stays inside + steps to the balcony.

- [ ] **Step 1:** Implement `modular_house()` by extending the existing `two_room_house()` point-sampling helpers; return ground-truth `meta`.
- [ ] **Step 2:** Test: assert point count > 0, all four features present in expected regions, neighbour blob separated by a gap.
- [ ] **Step 3:** Commit.

---

## Task 9: frame — normals, dominant axes, axis-align

**Files:** Create `scripts/recon/frame.py`, `tests/recon/test_frame.py`

**Interfaces:**
- Produces:
  - `estimate_normals(xyz, radius=0.06, max_nn=30) -> np.ndarray (N,3)` (Open3D, guarded import).
  - `dominant_axes(normals, up=(0,0,1)) -> np.ndarray (3,3)` — project wall normals to horizontal, vote orientation histogram (2° bins over 0–90°, Manhattan), return rotation `R` aligning dominant wall dir to +X (pure numpy — **testable without Open3D**).
  - `axis_align(scan, R) -> ScanData` — rotate xyz by `R` (record `R` for later un-rotation).

- [ ] **Step 1: Write failing test for `dominant_axes`:** synthetic normals mostly along ±X and ±Y rotated by 30° → assert returned `R` rotates them back to axis-aligned within 1°.
- [ ] **Step 2–4:** Implement + PASS. **Step 5:** Commit.

---

## Task 10: planes — detection + DBSCAN + label

**Files:** Create `scripts/recon/planes.py`, `tests/recon/test_planes.py`

**Interfaces:**
- Produces: `detect_planes(xyz, normals=None, dist_thresh=0.02, min_inliers=2000, max_planes=60, dbscan_eps=0.1, dbscan_min=50) -> list[Plane]`. Iterative `segment_plane` (Open3D) removing inliers each round; DBSCAN each plane's inliers to split disjoint faces; label by normal·up (|·|>0.9 horizontal → floor if low-Z else ceiling; else vertical). Guarded Open3D import.
- Test seam (pure): `label_plane(normal, centroid_z, z_floor, z_ceiling) -> str` unit-tested without Open3D.

- [ ] **Step 1:** Failing test for `label_plane` (horizontal low → "floor", horizontal high → "ceiling", vertical → "vertical").
- [ ] **Step 2–4:** Implement both; PASS. **Step 5:** Commit.
- [ ] **Step 6:** Integration smoke on `modular_house()` cloud: assert ≥ 4 vertical planes + 1 floor + 1 ceiling detected (run with Open3D; skip-mark if unavailable).

> **Detail firms up at execution from here (Phases C–F). Interfaces below are contracts; exact thresholds/code are finalized per-phase against real output. No placeholders — each task states concrete deliverables, tests, and acceptance.**

---

## Task 11 (Phase C): structure — floor/ceiling + stepped walls + columns/beams

**Files:** Create `scripts/recon/structure.py`, `tests/recon/test_structure.py`

**Interfaces:**
- `extract_floor_ceiling(planes) -> (Plane, Plane)`.
- `group_wall_runs(vertical_planes, axes, merge_offset_m=0.15) -> list[dict]` — cluster vertical faces by (direction, colinear offset); within a run, faces at differing perpendicular offsets become `WallStep`s (relief preserved).
- `extract_columns_beams(vertical_planes, floor_z, ceiling_z, min_size_m=0.1, max_size_m=0.6) -> (list[Column], list[Beam])` — compact prismatic clusters protruding from/independent of wall runs; vertical footprint → Column, horizontal span under ceiling → Beam.

**Acceptance:** on `modular_house()`, recovers the 4 walls, the pillar as a `Column` with correct footprint/height (±3 cm), the beam as a `Beam`, and the wall bearing the pillar carries a `WallStep` at the pillar offset.

**Tasks:** TDD each function with `modular_house()` ground-truth: (a) floor/ceiling z within 2 cm; (b) wall count == 4; (c) column footprint within 3 cm; (d) beam span within 3 cm; (e) step offset sign correct. One commit per function.

---

## Task 12 (Phase C): regularize — axis snap, corner intersection, thickness

**Files:** Create `scripts/recon/regularize.py`, `tests/recon/test_regularize.py`

**Interfaces:**
- `snap_walls(walls, axes, angle_tol_deg=8) -> walls` (snap only within tol; PCA fallback beyond).
- `resolve_corners(walls, tol_m=0.25) -> walls` — salvage `_resolve_corner_point` (line intersection, guarded) so corners are watertight.
- `pair_thickness(walls, points, default_m=0.10) -> walls` — salvage `pair_wall_surfaces` + modal fallback; set `thickness_source`.

**Acceptance:** on `modular_house()`, corners meet within 1 cm; snapped wall directions within 0.5° of axes; thickness == fixture thickness for double-sided walls, `assumed` default for single-sided. TDD per function; salvaged functions get their existing tests ported to `tests/recon/`.

---

## Task 13 (Phase D): openings — voids + visibility gate + refine + classify

**Files:** Create `scripts/recon/openings.py`, `tests/recon/test_openings.py`

**Interfaces:**
- `wall_occupancy(wall, points, cell_m=0.03) -> np.ndarray` (u,v occupancy grid).
- `interior_voids(occ) -> list[rect]` — salvage `_interior_void_cells`.
- `visibility_gate(void, wall_plane, trajectory, xyz, voxel_m=0.05) -> bool` — 3D-DDA ray-cast from trajectory sensor origins; accept if void cells EMPTY with rays crossing the plane. Fallback: `both_faces_empty(...)` (salvage `cross_check_opening_both_faces`).
- `refine_opening_edges(...)` and `classify_opening(...)` — salvage from `floorplan_geometry`.
- Returns `list[Opening]` per wall.

**Acceptance:** on `modular_house()`: door, window, balcony each recovered with correct `type` and dims (±3 cm); the **neighbour blob beyond the balcony does NOT create a wall or suppress the balcony opening**; a synthetic **furniture slab in front of a wall (occlusion shadow) is NOT reported as an opening** (visibility gate rejects it). TDD each; the furniture-shadow and neighbour-blob cases are explicit tests.

---

## Task 14 (Phase E): solids — extrude + boolean cut

**Files:** Create `scripts/recon/solids.py`, `tests/recon/test_solids.py`

**Interfaces:**
- `wall_to_solid(wall, steps) -> trimesh.Trimesh` (stepped extrusion, watertight).
- `slab_to_solid(plane, polygon, thickness) -> Trimesh` (floor/ceiling).
- `column_to_solid(column) -> Trimesh`, `beam_to_solid(beam) -> Trimesh`.
- `cut_openings(wall_solid, openings, wall) -> Trimesh` — build cutter boxes overshooting thickness; `manifold3d` boolean difference; assert result watertight.

**Acceptance:** each solid `.is_watertight`; cut wall has correct volume reduction (± tolerance) matching opening size; boolean falls back to bpy EXACT solver only if manifold3d fails (documented). TDD with trimesh primitives; watertightness asserted every task.

---

## Task 15 (Phase F): floorplan2d — arrangement → rooms → DXF/SVG

**Files:** Create `scripts/recon/floorplan2d.py`, `tests/recon/test_floorplan2d.py`

**Interfaces:**
- `build_room_polygons(walls) -> list[shapely.Polygon]` (extend wall centrelines, `shapely.polygonize`).
- `write_dxf(walls, openings, rooms, path)` (ezdxf: layers WALLS/OPENINGS/ROOMS/DIMS).
- `write_svg(walls, openings, rooms, path)`.

**Acceptance:** on `modular_house()`, one closed room polygon of correct area (±2%); DXF opens with correct layers; opening symbols placed at correct u positions. TDD per function (geometry asserted numerically; file-existence + entity-count for DXF/SVG).

---

## Task 16 (Phase F): assemble — named GLB + collections + optional .blend

**Files:** Create `scripts/recon/assemble.py`, `tests/recon/test_assemble.py`

**Interfaces:**
- `build_scene(elements: dict[str, list[Trimesh]]) -> trimesh.Scene` — names `wall_00…`, groups metadata by collection.
- `write_glb(scene, path)`; `write_blend(elements, path)` (guarded bpy, collections Walls/Floor/Ceiling/Doors/Windows/Columns/Beams).

**Acceptance:** GLB reloads via trimesh with expected node names/count; (if bpy present) `.blend` has the 7 named collections. TDD with a 2-mesh scene.

---

## Task 17 (Phase F): full pipeline wiring + manifest + validation + docs

**Files:** Modify `scripts/reconstruct.py` (wire all stages, `DEFAULT_CONFIG`, all outputs), `tests/recon/test_pipeline_full_cli.py`; update `README.md`; **delete** superseded files.

**Interfaces:** `run(in_path, out_dir, config) -> None` writing `model.glb`, `floorplan.dxf`, `floorplan.svg`, `manifest.json`, `report.txt`.

- [ ] Full CLI integration test on `modular_house()` LAS → all artifacts exist; manifest element counts correct; neighbour excluded.
- [ ] Wire `validate_measurements.py` when `--ground-truth` given.
- [ ] **Delete** `floorplan_reconstruct.py`, `floorplan_reconstruct_test.py`, `segment_walls_and_grooves.py`, `find_z_band.py`, and the density-image core of `floorplan_geometry.py` (after salvaged functions are moved); remove their dead tests.
- [ ] Update `README.md` pipeline section.
- [ ] Real-scan run on `koushikexport.las` + `mujammelexport.las`; record element counts + ceiling height in commit body.
- [ ] Commit.

> **CHECKPOINT — full review + real-scan inspection.**

---

## Self-Review (author check)

- **Spec coverage:** ingest/clean/isolate (T3–7), trajectory (T5), frame/planes (T9–10), floor/ceiling + stepped walls + columns/beams (T11), regularize/thickness (T12), openings+visibility (T13), solids+boolean (T14), 2D floorplan (T15), assemble/GLB (T16), manifest/validation/delete-old/docs (T17). Balcony toggle → `DEFAULT_CONFIG["balcony"]` consumed in isolate (T6) + openings (T13). All spec sections mapped.
- **Placeholders:** Phases A–B carry full code; C–F carry concrete interfaces + acceptance + named tests (finalized per-phase by design, not deferred vaguely).
- **Type consistency:** `ScanData`/`Plane`/`WallStep`/`Column`/`Beam` defined in T2 and used verbatim downstream; `Wall`/`Opening` reused from existing schema.
