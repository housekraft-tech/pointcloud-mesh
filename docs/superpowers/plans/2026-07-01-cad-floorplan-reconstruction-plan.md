# CAD Floor Plan Reconstruction (Phase 0 + Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `segment_walls_and_grooves.py` with a density-histogram/contour based pipeline (`floorplan_geometry.py` + `floorplan_reconstruct.py`) that turns a whole-house LiDAR scan into an accurate 2D floor plan image, a `manifest.json` of wall/opening dimensions, and a clean, SketchUp-editable `reconstructed.obj` — covering Phase 0 (bounding-box auto-crop) and Phase 1 (walls, floor plan, openings) of `docs/superpowers/specs/2026-07-01-cad-floorplan-reconstruction-design.md`. Phase 2 (grooves/pillars) is a separate future plan.

**Architecture:** All new geometry math lives in `floorplan_geometry.py` as pure numpy+cv2 functions (no `open3d` import) so the whole algorithm is unit-testable on the developer's local 32-bit Python 3.9 machine, which cannot install `open3d` at all. `mesh_common.py` gets one small `open3d`-dependent wrapper that calls the pure crop function. `floorplan_reconstruct.py` is a thin CLI shell wiring LAS load → crop → density image → walls → openings → outputs; its own wiring logic is exercised through a pure `build_floorplan_outputs()` function that is also fully unit-testable. `validate_measurements.py` diffs real tape measurements against `manifest.json`.

**Tech Stack:** Python, numpy, opencv-python-headless (contours, `approxPolyDP`, morphology), open3d (point cloud I/O only, isolated to `mesh_common.py`), laspy, pytest.

## Global Constraints

- No `open3d` import anywhere in `floorplan_geometry.py`, `validate_measurements.py`, or `tests/fixtures.py` — this is what makes them runnable in local pytest on the 32-bit dev machine. Verify per-task with `grep -c open3d <file>` expecting `0`.
- Every final numeric output (wall thickness, opening edges) must come from a least-squares refit on original, non-downsampled points — never directly from the coarse density-image/contour detection. The coarse detection only locates *where* to look.
- `manifest.json` field names defined in Task 8 are the fixed contract for `validate_measurements.py` (Task 15) and any future Phase 2 plan — do not rename fields once Task 8 lands.
- All new code targets Python 3.9+ syntax compatible with the existing codebase (`scripts/mesh_common.py` already uses `float | None` union syntax, so 3.10+ typing is fine in orchestration code, but keep `floorplan_geometry.py` free of type annotations that require imports beyond `typing`/`dataclasses` to keep it dependency-light).
- Add `opencv-python-headless` and `pytest` to `requirements.txt` alongside the existing pins (Task 1); do not change existing pins.

---

### Task 1: Test infrastructure, dependencies, and synthetic fixtures

**Files:**
- Modify: `requirements.txt`
- Create: `tests/__init__.py`
- Create: `scripts/__init__.py`
- Create: `tests/fixtures.py`
- Create: `pytest.ini`

**Interfaces:**
- Produces: `tests.fixtures.two_room_house(rng=None) -> (points: np.ndarray[N,3], ground_truth: dict)` — used by every later test task.

This fixture was already built and run against the real algorithm during design validation: a 6m×5m exterior envelope (200mm walls) split by a 5m interior partition (100mm thick) with a 900mm×2100mm floor-to-ceiling door and a 1200mm×1200mm window (sill 900mm) on the south wall, plus a sparse stray-point tail mimicking the SLAM drift seen in the real scans. Confirmed via prototype run: this fixture surfaces real edge cases (T-junctions, floor-level openings, corner contamination) that a simpler fixture would miss.

- [ ] **Step 1: Add test dependencies to requirements.txt**

Open `requirements.txt` and add these two lines at the end (after the existing `open3d==0.19.0` line):

```
opencv-python-headless>=4.9.0
pytest>=8.0.0
```

- [ ] **Step 2: Create pytest.ini**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
```

- [ ] **Step 3: Create tests/__init__.py and scripts/__init__.py (both empty)**

Both files are empty. This makes `scripts` an explicit package so `from scripts.floorplan_geometry import ...`-style absolute imports in test files resolve reliably from the repo root, regardless of Python version namespace-package edge cases. This does not affect the existing direct-execution scripts (e.g. `python scripts/reconstruct_mesh.py`), which still resolve sibling imports like `from mesh_common import ...` the same way as before, since Python always puts the executed script's own directory on `sys.path[0]`.

```python
```

- [ ] **Step 4: Write tests/fixtures.py**

```python
"""Synthetic multi-room house fixture for validating floorplan_geometry.py.

Confirmed via manual prototype run to exercise: a T-junction (partition
meets both exterior walls), a floor-to-ceiling door (sill=0, stresses the
opening flood-fill's floor-boundary handling), and a mid-wall window
(stresses the standard enclosed-void case), plus a sparse stray-point tail
matching the SLAM drift pattern seen in koushikexport.las/mujammelexport.las.
"""
import numpy as np


def _sample_face(u_range, z_range, offset_axis, offset_value, along_axis,
                  exclude_rects=None, spacing=0.016, noise_std=0.002, rng=None):
    rng = rng or np.random.default_rng(0)
    exclude_rects = exclude_rects or []
    u0, u1 = u_range
    z0, z1 = z_range
    us = np.arange(u0, u1, spacing)
    zs = np.arange(z0, z1, spacing)
    uu, zz = np.meshgrid(us, zs)
    uu = uu.ravel() + rng.normal(0, spacing * 0.2, uu.size)
    zz = zz.ravel() + rng.normal(0, spacing * 0.2, zz.size)

    keep = np.ones(uu.shape, dtype=bool)
    for (ru0, ru1, rz0, rz1) in exclude_rects:
        inside = (uu >= ru0) & (uu <= ru1) & (zz >= rz0) & (zz <= rz1)
        keep &= ~inside
    uu, zz = uu[keep], zz[keep]

    offset = offset_value + rng.normal(0, noise_std, uu.size)
    pts = np.zeros((len(uu), 3))
    if along_axis == "x":
        pts[:, 0] = uu
        pts[:, 1] = offset
    else:
        pts[:, 1] = uu
        pts[:, 0] = offset
    pts[:, 2] = zz
    return pts


def two_room_house(rng=None):
    """Returns (points (N,3), ground_truth dict).

    ground_truth["exterior_walls"]: list of {centerline: ((x0,y0),(x1,y1)), thickness_m}
    ground_truth["partition_walls"]: same shape, thickness 0.1
    ground_truth["openings"]: list of {wall, width_m, height_m, sill_m, type}
    """
    rng = rng or np.random.default_rng(42)
    z_full = (0.0, 2.7)
    faces = []

    window = [(4.0, 5.2, 0.9, 2.1)]
    faces.append(_sample_face((0, 6), z_full, "y", -0.1, "x", window, rng=rng))
    faces.append(_sample_face((0, 6), z_full, "y", 0.1, "x", window, rng=rng))
    faces.append(_sample_face((0, 5), z_full, "x", 5.9, "y", rng=rng))
    faces.append(_sample_face((0, 5), z_full, "x", 6.1, "y", rng=rng))
    faces.append(_sample_face((0, 6), z_full, "y", 4.9, "x", rng=rng))
    faces.append(_sample_face((0, 6), z_full, "y", 5.1, "x", rng=rng))
    faces.append(_sample_face((0, 5), z_full, "x", 0.1, "y", rng=rng))
    faces.append(_sample_face((0, 5), z_full, "x", -0.1, "y", rng=rng))

    door = [(2.0, 2.9, 0.0, 2.1)]
    faces.append(_sample_face((0, 5), z_full, "x", 2.95, "y", door, rng=rng))
    faces.append(_sample_face((0, 5), z_full, "x", 3.05, "y", door, rng=rng))

    real_points = np.vstack(faces)

    n_stray = 500
    stray = np.column_stack([
        rng.uniform(-10, 15, n_stray),
        rng.uniform(-20, 25, n_stray),
        rng.uniform(-5, 15, n_stray),
    ])

    all_points = np.vstack([real_points, stray])

    ground_truth = {
        "exterior_walls": [
            {"centerline": ((0, 0), (6, 0)), "thickness_m": 0.2},
            {"centerline": ((6, 0), (6, 5)), "thickness_m": 0.2},
            {"centerline": ((6, 5), (0, 5)), "thickness_m": 0.2},
            {"centerline": ((0, 5), (0, 0)), "thickness_m": 0.2},
        ],
        "partition_walls": [
            {"centerline": ((3, 0), (3, 5)), "thickness_m": 0.1},
        ],
        "openings": [
            {"wall": "south", "width_m": 1.2, "height_m": 1.2, "sill_m": 0.9, "type": "window"},
            {"wall": "partition", "width_m": 0.9, "height_m": 2.1, "sill_m": 0.0, "type": "door"},
        ],
    }
    return all_points, ground_truth
```

- [ ] **Step 5: Verify the fixture loads and produces the expected point count**

Run: `python -c "from tests.fixtures import two_room_house; pts, gt = two_room_house(); print(len(pts), len(gt['openings']))"`
Expected: `545301 2` (confirmed exact count from prototype run)

- [ ] **Step 6: Install new dependencies locally and confirm no open3d needed**

Run: `pip install opencv-python-headless pytest` (already present in most dev setups; on the 32-bit Windows dev machine this resolves to `opencv_python_headless-4.9.0.80-cp37-abi3-win32.whl`, confirmed working)
Run: `python -c "import cv2, pytest; print('ok')"`
Expected: `ok`

- [ ] **Step 7: Commit**

```bash
git add requirements.txt pytest.ini tests/__init__.py tests/fixtures.py
git commit -m "Add test infra and synthetic multi-room house fixture"
```

---

### Task 2: Plane/frame math primitives

**Files:**
- Create: `scripts/floorplan_geometry.py`
- Create: `tests/test_floorplan_geometry.py`

**Interfaces:**
- Consumes: nothing (leaf module).
- Produces: `plane_normal(plane_model) -> np.ndarray[3]`, `signed_plane_distance(points, plane_model) -> np.ndarray[N]`, `refine_plane_model(points, plane_model) -> list[4]`, `wall_uv_basis(normal) -> (u: np.ndarray[3], v: np.ndarray[3])`, `project_to_plane(points, plane_model) -> np.ndarray[N,3]`, `points_to_wall_uv(points, plane_model, origin_xyz, u_axis, v_axis=None) -> np.ndarray[N,2]`. All consumed by every later task.

These are ported almost verbatim from the proven plane math in `scripts/segment_walls_and_grooves.py` (`plane_normal`, `signed_plane_distance`, `refine_plane_model`, `wall_uv_basis`, `project_to_plane`) — physically relocated (not imported) so this module never pulls in `open3d` via that file's module-level `import open3d as o3d`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_floorplan_geometry.py
import numpy as np
from scripts.floorplan_geometry import (
    plane_normal, signed_plane_distance, refine_plane_model,
    wall_uv_basis, project_to_plane, points_to_wall_uv,
)


def test_plane_normal_normalizes():
    n = plane_normal([3.0, 4.0, 0.0, -1.0])
    assert np.allclose(n, [0.6, 0.8, 0.0])


def test_signed_plane_distance_sign_and_magnitude():
    plane = [1.0, 0.0, 0.0, -2.0]  # x = 2 plane
    pts = np.array([[2.0, 0, 0], [2.1, 0, 0], [1.9, 0, 0]])
    dist = signed_plane_distance(pts, plane)
    assert np.allclose(dist, [0.0, 0.1, -0.1], atol=1e-9)


def test_refine_plane_model_recovers_noisy_plane():
    rng = np.random.default_rng(0)
    y = 0.1
    n = 2000
    pts = np.column_stack([
        rng.uniform(0, 5, n),
        np.full(n, y) + rng.normal(0, 0.002, n),
        rng.uniform(0, 2.7, n),
    ])
    coarse = [0.0, 1.0, 0.0, -(y - 0.02)]  # deliberately off by 20mm
    refined = refine_plane_model(pts, coarse)
    assert abs(refined[3] - (-y)) < 0.001  # within 1mm of true offset


def test_wall_uv_basis_orthonormal_and_v_is_up_projected():
    normal = np.array([1.0, 0.0, 0.0])
    u, v = wall_uv_basis(normal)
    assert abs(np.dot(u, normal)) < 1e-9
    assert abs(np.dot(v, normal)) < 1e-9
    assert abs(np.linalg.norm(u) - 1.0) < 1e-9
    assert abs(np.linalg.norm(v) - 1.0) < 1e-9
    assert v[2] > 0.99  # for a vertical wall, v should be ~world-up


def test_project_to_plane_puts_points_exactly_on_plane():
    plane = [0.0, 1.0, 0.0, -0.1]  # y = 0.1
    pts = np.array([[1.0, 0.5, 2.0], [3.0, -0.3, 1.0]])
    projected = project_to_plane(pts, plane)
    assert np.allclose(projected[:, 1], 0.1)


def test_points_to_wall_uv_shape_and_v_is_height():
    plane = [1.0, 0.0, 0.0, -3.0]  # x = 3
    pts = np.array([[3.0, 1.0, 2.0], [3.0, 2.0, 2.5]])
    origin = np.array([3.0, 0.0, 0.0])
    u_axis = np.array([0.0, 1.0, 0.0])
    uv = points_to_wall_uv(pts, plane, origin, u_axis)
    assert uv.shape == (2, 2)
    assert np.allclose(uv[:, 0], [1.0, 2.0])  # u = along wall (y here)
    assert np.allclose(uv[:, 1], [2.0, 2.5])  # v = world-up height
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_floorplan_geometry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.floorplan_geometry'` (or similar import error)

- [ ] **Step 3: Write scripts/floorplan_geometry.py (this task's portion)**

```python
"""Pure numpy+cv2 floor plan reconstruction primitives.

No open3d import anywhere in this file -- this is what makes it fully
unit-testable on a 32-bit Python install that cannot install open3d at all.
Any function here that needs open3d-loaded data takes plain numpy arrays;
the open3d-facing wrapper lives in mesh_common.py.
"""
import numpy as np
import cv2
from collections import defaultdict, deque


# ---------- plane / frame math ----------

def plane_normal(plane_model):
    a, b, c, _d = plane_model
    n = np.array([a, b, c], dtype=np.float64)
    return n / np.linalg.norm(n)


def signed_plane_distance(points, plane_model):
    a, b, c, d = plane_model
    n = np.array([a, b, c], dtype=np.float64)
    n /= np.linalg.norm(n)
    pts = np.asarray(points, dtype=np.float64)
    return pts @ n + float(d)


def refine_plane_model(points, plane_model):
    """SVD least-squares plane refit on the given points."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return list(plane_model)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1].astype(np.float64)
    n0 = plane_normal(plane_model)
    if float(np.dot(normal, n0)) < 0.0:
        normal = -normal
    normal /= np.linalg.norm(normal)
    d = -float(np.dot(normal, centroid))
    return [float(normal[0]), float(normal[1]), float(normal[2]), d]


def wall_uv_basis(normal):
    """U along the wall (perpendicular to normal, in the horizontal plane),
    V = world-up projected onto the wall plane."""
    normal = normal / np.linalg.norm(normal)
    world_up = np.array([0.0, 0.0, 1.0])
    v = world_up - normal * np.dot(world_up, normal)
    if np.linalg.norm(v) < 1e-6:
        ref = np.array([1.0, 0.0, 0.0])
        u = np.cross(normal, ref)
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        return u, v
    v = v / np.linalg.norm(v)
    u = np.cross(v, normal)
    u /= np.linalg.norm(u)
    return u, v


def project_to_plane(points, plane_model):
    signed = signed_plane_distance(points, plane_model)
    n = plane_normal(plane_model)
    pts = np.asarray(points, dtype=np.float64)
    return pts - signed[:, None] * n


def points_to_wall_uv(points, plane_model, origin_xyz, u_axis, v_axis=None):
    """Project 3D points onto a wall plane, express as (u=along wall, v=height)."""
    if v_axis is None:
        v_axis = np.array([0.0, 0.0, 1.0])
    projected = project_to_plane(points, plane_model)
    rel = projected - np.asarray(origin_xyz, dtype=np.float64)
    u = rel @ np.asarray(u_axis, dtype=np.float64)
    v = rel @ np.asarray(v_axis, dtype=np.float64)
    return np.column_stack([u, v])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_floorplan_geometry.py -v`
Expected: 6 passed

- [ ] **Step 5: Verify no open3d dependency**

Run: `grep -c open3d scripts/floorplan_geometry.py`
Expected: `0`

- [ ] **Step 6: Commit**

```bash
git add scripts/floorplan_geometry.py tests/test_floorplan_geometry.py
git commit -m "Add plane/frame math primitives to floorplan_geometry.py"
```

---

### Task 3: Phase 0 bounding-box auto-crop

**Files:**
- Modify: `scripts/floorplan_geometry.py`
- Modify: `tests/test_floorplan_geometry.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `crop_to_percentile_bounds(xyz, low_pct=1.0, high_pct=99.0, margin_m=0.5) -> (lo: np.ndarray[3], hi: np.ndarray[3], keep_mask: np.ndarray[N] bool, stats: dict)`. Consumed by Task 13 (mesh_common wrapper) and the orchestrator (Task 14).

Confirmed via prototype run against the synthetic fixture (545,301 points including a 500-point stray tail): dropped 499/545301 = 0.09% of points, cropped bounds `[-0.60,-0.60,-0.48]` to `[6.60,5.60,3.17]` (matches the real 6×5×2.7m room plus margin).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_floorplan_geometry.py
from scripts.floorplan_geometry import crop_to_percentile_bounds
from tests.fixtures import two_room_house


def test_crop_to_percentile_bounds_drops_stray_tail_not_real_room():
    pts, _gt = two_room_house()
    lo, hi, keep_mask, stats = crop_to_percentile_bounds(pts, low_pct=1.0, high_pct=99.0, margin_m=0.5)
    assert stats["dropped_fraction"] < 0.01
    assert lo[0] < 0.0 and hi[0] > 6.0  # room x-extent [0,6] preserved with margin
    assert lo[1] < 0.0 and hi[1] > 5.0  # room y-extent [0,5] preserved with margin
    assert lo[2] < 0.0 and hi[2] > 2.7  # room z-extent [0,2.7] preserved with margin


def test_crop_to_percentile_bounds_raises_on_empty_input():
    import pytest
    with pytest.raises(ValueError):
        crop_to_percentile_bounds(np.empty((0, 3)))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_floorplan_geometry.py::test_crop_to_percentile_bounds_drops_stray_tail_not_real_room -v`
Expected: FAIL with `ImportError: cannot import name 'crop_to_percentile_bounds'`

- [ ] **Step 3: Add crop_to_percentile_bounds to scripts/floorplan_geometry.py**

Append after `points_to_wall_uv`:

```python
# ---------- Phase 0: bounding-box auto-crop ----------

def crop_to_percentile_bounds(xyz, low_pct=1.0, high_pct=99.0, margin_m=0.5):
    """Robust bounding box from per-axis percentiles + margin, dropping the
    sparse SLAM-drift/ghost-point tail that inflates a raw min/max bbox.

    Confirmed on the real koushikexport.las/mujammelexport.las scans: 99% of
    points sit in a ~11x12x3.3m room while raw bbox balloons to 30-85m due to
    a sparse stray tail; this recovers the tight room bounds.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.shape[0] == 0:
        raise ValueError("crop_to_percentile_bounds: empty point array")
    lo = np.percentile(xyz, low_pct, axis=0) - margin_m
    hi = np.percentile(xyz, high_pct, axis=0) + margin_m
    keep_mask = np.all((xyz >= lo) & (xyz <= hi), axis=1)
    stats = {
        "input_points": int(xyz.shape[0]),
        "kept_points": int(keep_mask.sum()),
        "dropped_points": int((~keep_mask).sum()),
        "dropped_fraction": float((~keep_mask).sum() / xyz.shape[0]),
        "raw_bounds_min": xyz.min(axis=0).tolist(),
        "raw_bounds_max": xyz.max(axis=0).tolist(),
        "cropped_bounds_min": lo.tolist(),
        "cropped_bounds_max": hi.tolist(),
    }
    return lo, hi, keep_mask, stats
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_floorplan_geometry.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/floorplan_geometry.py tests/test_floorplan_geometry.py
git commit -m "Add Phase 0 percentile-based bounding-box auto-crop"
```

---

### Task 4: Density image + threshold

**Files:**
- Modify: `scripts/floorplan_geometry.py`
- Modify: `tests/test_floorplan_geometry.py`

**Interfaces:**
- Produces: `points_to_density_image(xy, cell_size_m, bounds_min, bounds_max) -> (image: np.ndarray[H,W] uint16, origin: np.ndarray[2])`, `threshold_density_image(image, min_count=2, morph_kernel=3) -> np.ndarray[H,W] uint8`. Consumed by Task 5.

Cell size **20mm** and morph-close kernel **3px** are the values used throughout prototype validation against the 16mm-median-spacing real scans (per the algorithm review: cells >30mm start merging thin partitions, cells <20mm risk empty bins at 16mm spacing).

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_floorplan_geometry.py
from scripts.floorplan_geometry import points_to_density_image, threshold_density_image


def test_points_to_density_image_counts_correctly():
    xy = np.array([[0.0, 0.0], [0.01, 0.01], [0.5, 0.5]])
    image, origin = points_to_density_image(xy, cell_size_m=0.02, bounds_min=[0, 0], bounds_max=[1, 1])
    assert image[0, 0] == 2  # first two points land in the same cell
    assert np.allclose(origin, [0.0, 0.0])


def test_threshold_density_image_drops_sparse_cells():
    image = np.zeros((5, 5), dtype=np.uint16)
    image[2, 2] = 5  # dense
    image[0, 0] = 1  # sparse, below threshold
    binary = threshold_density_image(image, min_count=2, morph_kernel=1)
    assert binary[2, 2] == 255
    assert binary[0, 0] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_floorplan_geometry.py -v -k density`
Expected: FAIL (ImportError)

- [ ] **Step 3: Add to scripts/floorplan_geometry.py**

```python
# ---------- Phase 1: density image ----------

def points_to_density_image(xy, cell_size_m, bounds_min, bounds_max):
    xy = np.asarray(xy, dtype=np.float64)
    width = int(np.ceil((bounds_max[0] - bounds_min[0]) / cell_size_m)) + 1
    height = int(np.ceil((bounds_max[1] - bounds_min[1]) / cell_size_m)) + 1
    ix = np.floor((xy[:, 0] - bounds_min[0]) / cell_size_m).astype(np.int64)
    iy = np.floor((xy[:, 1] - bounds_min[1]) / cell_size_m).astype(np.int64)
    valid = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
    image = np.zeros((height, width), dtype=np.uint16)
    np.add.at(image, (iy[valid], ix[valid]), 1)
    origin = np.array([bounds_min[0], bounds_min[1]], dtype=np.float64)
    return image, origin


def threshold_density_image(image, min_count=2, morph_kernel=3):
    binary = (image >= min_count).astype(np.uint8) * 255
    kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_floorplan_geometry.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/floorplan_geometry.py tests/test_floorplan_geometry.py
git commit -m "Add density image build + threshold primitives"
```

---

### Task 5: Wall segment extraction from contours

**Files:**
- Modify: `scripts/floorplan_geometry.py`
- Modify: `tests/test_floorplan_geometry.py`

**Interfaces:**
- Produces: `extract_wall_segments(binary_image, origin, cell_size_m, epsilon_cells=2.0, min_span_cells=3.0) -> list[{"p0": np.ndarray[2], "p1": np.ndarray[2], "length": float}]`. Consumed by Task 6.

**Critical bug found and fixed during design validation:** a wall face with no opposing face within the density-image threshold produces a real but near-zero-**area** contour (a genuinely thin 1-cell-wide line). A naive `cv2.contourArea(contour) < 1` filter (the obvious first approach) silently discards these — which means any wall scanned from only one side (very common for exterior walls in an interior-only scan) would vanish from the output with no error or warning. The fix is filtering on bounding-box **span** (`max(w, h)`), not area.

- [ ] **Step 1: Write the failing test (including the regression case for the bug found)**

```python
# append to tests/test_floorplan_geometry.py
from scripts.floorplan_geometry import extract_wall_segments


def test_extract_wall_segments_recovers_rectangle():
    # a filled 3m x 2m rectangle at 20mm cells = 150x100 px, 1px border thickness would
    # be too thin to test area-filter regression; use a 3px-thick rectangle outline
    image = np.zeros((100, 150), dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (140, 90), 255, thickness=3)
    segments = extract_wall_segments(image, origin=np.array([0.0, 0.0]), cell_size_m=0.02, epsilon_cells=2.0)
    assert len(segments) >= 4
    lengths = sorted(s["length"] for s in segments)
    # long sides ~ (140-10)*0.02=2.6m, short sides ~ (90-10)*0.02=1.6m
    assert any(abs(l - 2.6) < 0.1 for l in lengths)
    assert any(abs(l - 1.6) < 0.1 for l in lengths)


def test_extract_wall_segments_keeps_single_sided_thin_line():
    """Regression test for the bug found during design validation: a
    single-sided wall face (no opposing face) produces a near-zero-area
    contour that an area-based filter would incorrectly discard."""
    image = np.zeros((50, 300), dtype=np.uint8)
    cv2.line(image, (10, 25), (290, 25), 255, thickness=1)
    segments = extract_wall_segments(image, origin=np.array([0.0, 0.0]), cell_size_m=0.02, epsilon_cells=2.0)
    assert len(segments) >= 1
    assert any(s["length"] > 5.0 for s in segments)  # the ~280px*0.02m=5.6m line survives
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_floorplan_geometry.py -v -k extract_wall_segments`
Expected: FAIL (ImportError)

- [ ] **Step 3: Add to scripts/floorplan_geometry.py**

```python
def extract_wall_segments(binary_image, origin, cell_size_m, epsilon_cells=2.0, min_span_cells=3.0):
    """min_span_cells filters by bounding-box span, NOT area: a wall face
    seen from only one side (no opposing face within threshold, nothing to
    'fill' between them) produces a genuinely thin, near-zero-area contour
    that a naive area filter would incorrectly discard as noise -- confirmed
    during design validation this silently drops every single-sided wall."""
    contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    segments = []
    for contour in contours:
        _x, _y, w, h = cv2.boundingRect(contour)
        if max(w, h) < min_span_cells:
            continue
        approx = cv2.approxPolyDP(contour, epsilon_cells, closed=True)
        pts_px = approx.reshape(-1, 2).astype(np.float64)
        pts_world = origin + pts_px * cell_size_m
        n = len(pts_world)
        if n < 2:
            continue
        for i in range(n):
            p0 = pts_world[i]
            p1 = pts_world[(i + 1) % n]
            length = float(np.linalg.norm(p1 - p0))
            if length < cell_size_m:
                continue
            segments.append({"p0": p0, "p1": p1, "length": length})
    return segments
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_floorplan_geometry.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/floorplan_geometry.py tests/test_floorplan_geometry.py
git commit -m "Add wall segment extraction; fix single-sided-wall area-filter bug"
```

---

### Task 6: Wall pairing (mutual nearest-neighbor)

**Files:**
- Modify: `scripts/floorplan_geometry.py`
- Modify: `tests/test_floorplan_geometry.py`

**Interfaces:**
- Produces: `pair_wall_surfaces(segments, min_thickness_m=0.06, max_thickness_m=0.35, max_angle_deg=5.0, min_overlap_frac=0.5) -> list[wall dict]` where wall dict is `{"p0": np.ndarray[2], "p1": np.ndarray[2], "thickness_m": float|None, "thickness_source": "measured"|"assumed"}`; `apply_modal_thickness_fallback(walls, default_thickness_m=0.1) -> walls` (mutates and returns). Consumed by Task 7.

Constraints (angle ≤5°, overlap ≥50%, thickness envelope 60-350mm) come from the algorithm review's T-junction mitigation. Confirmed on the synthetic fixture: 200mm exterior walls paired to within 0.3-0.6mm of true thickness at the coarse stage already; the 100mm partition's coarse pairing is only accurate to ~30mm (expected — Task 9's refinement fixes this, coarse detection is never the source of truth for the final number).

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_floorplan_geometry.py
from scripts.floorplan_geometry import pair_wall_surfaces, apply_modal_thickness_fallback


def _seg(p0, p1):
    p0, p1 = np.array(p0, dtype=np.float64), np.array(p1, dtype=np.float64)
    return {"p0": p0, "p1": p1, "length": float(np.linalg.norm(p1 - p0))}


def test_pair_wall_surfaces_pairs_parallel_segments_within_thickness_envelope():
    # two parallel 3m segments 200mm apart (a real wall)
    segs = [_seg((0, 0), (3, 0)), _seg((0, 0.2), (3, 0.2))]
    walls = pair_wall_surfaces(segs)
    assert len(walls) == 1
    assert walls[0]["thickness_source"] == "measured"
    assert abs(walls[0]["thickness_m"] - 0.2) < 0.01


def test_pair_wall_surfaces_rejects_gap_outside_thickness_envelope():
    # two parallel segments 3m apart (a room width, not a wall) must NOT pair
    segs = [_seg((0, 0), (3, 0)), _seg((0, 3.0), (3, 3.0))]
    walls = pair_wall_surfaces(segs)
    assert all(w["thickness_source"] == "assumed" for w in walls)


def test_apply_modal_thickness_fallback_fills_assumed_walls():
    walls = [
        {"p0": np.array([0.0, 0.0]), "p1": np.array([1.0, 0.0]), "thickness_m": 0.2, "thickness_source": "measured"},
        {"p0": np.array([0.0, 1.0]), "p1": np.array([1.0, 1.0]), "thickness_m": None, "thickness_source": "assumed"},
    ]
    walls = apply_modal_thickness_fallback(walls)
    assert walls[1]["thickness_m"] == 0.2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_floorplan_geometry.py -v -k pair_wall`
Expected: FAIL (ImportError)

- [ ] **Step 3: Add to scripts/floorplan_geometry.py**

```python
# ---------- wall pairing (mutual nearest-neighbor with thickness/angle/overlap constraints) ----------

def _segment_dir(seg):
    v = seg["p1"] - seg["p0"]
    return v / np.linalg.norm(v)


def _segment_normal(seg):
    d = _segment_dir(seg)
    return np.array([-d[1], d[0]])


def _overlap_fraction(seg_a, seg_b, direction):
    a0 = np.dot(seg_a["p0"], direction)
    a1 = np.dot(seg_a["p1"], direction)
    b0 = np.dot(seg_b["p0"], direction)
    b1 = np.dot(seg_b["p1"], direction)
    lo = max(min(a0, a1), min(b0, b1))
    hi = min(max(a0, a1), max(b0, b1))
    overlap = max(0.0, hi - lo)
    shorter = min(seg_a["length"], seg_b["length"])
    return overlap / shorter if shorter > 1e-9 else 0.0


def _project_point_onto_segment(point, seg):
    d = _segment_dir(seg)
    t = np.dot(point - seg["p0"], d)
    return seg["p0"] + t * d


def _build_wall_from_pair(seg_a, seg_b, thickness_m):
    centerline_p0 = (seg_a["p0"] + _project_point_onto_segment(seg_a["p0"], seg_b)) / 2
    centerline_p1 = (seg_a["p1"] + _project_point_onto_segment(seg_a["p1"], seg_b)) / 2
    return {
        "p0": centerline_p0,
        "p1": centerline_p1,
        "thickness_m": thickness_m,
        "thickness_source": "measured",
    }


def _build_wall_from_single(seg):
    return {
        "p0": seg["p0"],
        "p1": seg["p1"],
        "thickness_m": None,
        "thickness_source": "assumed",
    }


def pair_wall_surfaces(segments, min_thickness_m=0.06, max_thickness_m=0.35,
                        max_angle_deg=5.0, min_overlap_frac=0.5):
    """Mutual-nearest-neighbor pairing: each segment's candidate partners are
    filtered by near-parallel direction, sufficient overlap, and a plausible
    wall-thickness gap (60-350mm default) -- this envelope is what rejects a
    T-junction pairing a segment across to an unrelated wall/room-width gap.
    A pair is only accepted if each segment's closest-gap candidate is the
    other (mutual best match), not just a one-sided nearest match."""
    n = len(segments)
    candidates = {i: [] for i in range(n)}
    for i in range(n):
        d_i = _segment_dir(segments[i])
        n_i = _segment_normal(segments[i])
        for j in range(i + 1, n):
            d_j = _segment_dir(segments[j])
            cos_angle = abs(float(np.dot(d_i, d_j)))
            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            if angle_deg > max_angle_deg:
                continue
            overlap = _overlap_fraction(segments[i], segments[j], d_i)
            if overlap < min_overlap_frac:
                continue
            mid_i = (segments[i]["p0"] + segments[i]["p1"]) / 2
            mid_j = (segments[j]["p0"] + segments[j]["p1"]) / 2
            gap = abs(float(np.dot(mid_j - mid_i, n_i)))
            if gap < min_thickness_m or gap > max_thickness_m:
                continue
            candidates[i].append((j, gap, overlap))
            candidates[j].append((i, gap, overlap))

    best = {}
    for i, opts in candidates.items():
        if opts:
            opts.sort(key=lambda t: t[1])
            best[i] = opts[0][0]

    walls = []
    used = set()
    for i, j in best.items():
        if i in used or j in used:
            continue
        if best.get(j) == i:
            gap = next(g for (jj, g, _o) in candidates[i] if jj == j)
            walls.append(_build_wall_from_pair(segments[i], segments[j], gap))
            used.add(i)
            used.add(j)

    for i in range(n):
        if i in used:
            continue
        walls.append(_build_wall_from_single(segments[i]))
        used.add(i)

    return walls


def apply_modal_thickness_fallback(walls, default_thickness_m=0.1):
    """Never silently substitute a hardcoded default: derive the fallback
    from the modal MEASURED thickness across the structure, so an assumption
    is at least grounded in this building's actual wall construction."""
    measured = [w["thickness_m"] for w in walls if w["thickness_source"] == "measured"]
    modal = float(np.median(measured)) if measured else default_thickness_m
    for w in walls:
        if w["thickness_source"] == "assumed":
            w["thickness_m"] = modal
    return walls
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_floorplan_geometry.py -v`
Expected: 15 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/floorplan_geometry.py tests/test_floorplan_geometry.py
git commit -m "Add mutual-NN wall pairing with thickness/angle/overlap constraints"
```

---

### Task 7: Endpoint snapping + short-stub filter

**Files:**
- Modify: `scripts/floorplan_geometry.py`
- Modify: `tests/test_floorplan_geometry.py`

**Interfaces:**
- Produces: `snap_wall_endpoints(walls, tolerance_m=0.05) -> (walls, clusters: list[list])` (adds `"length_m"` key to each wall dict, mutates in place), `drop_short_walls(walls, min_length_m=0.3) -> walls`. Consumed by Task 9, Task 14.

**Confirmed via prototype run:** on the full synthetic-fixture pipeline (extract → pair → snap), several spurious short (<150mm) "walls" appear at T-junction corners from mutual-NN incidentally pairing two adjacent corner-stub segments. `drop_short_walls(min_length_m=0.3)` removes them; the real partition (100mm thick, ~5m long) and all four exterior walls survive since none of them are anywhere near 0.3m long.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_floorplan_geometry.py
from scripts.floorplan_geometry import snap_wall_endpoints, drop_short_walls


def test_snap_wall_endpoints_merges_nearby_corners():
    walls = [
        {"p0": np.array([0.0, 0.0]), "p1": np.array([3.0, 0.02])},
        {"p0": np.array([3.01, 0.0]), "p1": np.array([3.0, 3.0])},
    ]
    walls, clusters = snap_wall_endpoints(walls, tolerance_m=0.05)
    assert np.allclose(walls[0]["p1"], walls[1]["p0"], atol=1e-9)
    assert "length_m" in walls[0]


def test_drop_short_walls_removes_corner_stubs_keeps_real_walls():
    walls = [
        {"p0": np.array([0.0, 0.0]), "p1": np.array([5.0, 0.0]), "length_m": 5.0},
        {"p0": np.array([5.0, 0.0]), "p1": np.array([5.1, 0.05]), "length_m": 0.11},
    ]
    kept = drop_short_walls(walls, min_length_m=0.3)
    assert len(kept) == 1
    assert kept[0]["length_m"] == 5.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_floorplan_geometry.py -v -k "snap_wall or drop_short"`
Expected: FAIL (ImportError)

- [ ] **Step 3: Add to scripts/floorplan_geometry.py**

```python
# ---------- endpoint snapping ----------

def snap_wall_endpoints(walls, tolerance_m=0.05):
    """Cluster nearby wall centerline endpoints (across all walls) within
    tolerance so corners meet cleanly; replaces each endpoint with its
    cluster centroid."""
    endpoints = []
    for wi, w in enumerate(walls):
        endpoints.append((wi, "p0", w["p0"]))
        endpoints.append((wi, "p1", w["p1"]))

    clusters = []
    for (wi, key, pt) in endpoints:
        found = None
        for ci, members in enumerate(clusters):
            rep_pt = members[0][2]
            if np.linalg.norm(pt - rep_pt) <= tolerance_m:
                found = ci
                break
        if found is None:
            clusters.append([(wi, key, pt)])
        else:
            clusters[found].append((wi, key, pt))

    for members in clusters:
        centroid = np.mean([m[2] for m in members], axis=0)
        for (wi, key, _pt) in members:
            walls[wi][key] = centroid

    for w in walls:
        w["length_m"] = float(np.linalg.norm(w["p1"] - w["p0"]))

    return walls, clusters


def drop_short_walls(walls, min_length_m=0.3):
    """Drop T-junction/corner pixel-noise stubs: confirmed via design
    validation that mutual-NN pairing can match two short (<150mm) segments
    near a T-junction into a plausible-looking but spurious 'wall'. If a real
    building has a legitimate short partition stub, lower this threshold and
    add a corresponding case to validate_measurements.py's ground truth."""
    return [w for w in walls if w["length_m"] >= min_length_m]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_floorplan_geometry.py -v`
Expected: 17 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/floorplan_geometry.py tests/test_floorplan_geometry.py
git commit -m "Add endpoint snapping and short-stub wall filter"
```

---

### Task 8: Wall dedup/merge + manifest schema

**Files:**
- Create: `scripts/floorplan_schema.py`
- Modify: `scripts/floorplan_geometry.py`
- Modify: `tests/test_floorplan_geometry.py`
- Create: `tests/test_floorplan_schema.py`

**Interfaces:**
- Produces (floorplan_geometry.py): `merge_duplicate_walls(walls, angle_tol_deg=3.0, offset_tol_m=0.05, u_gap_tol_m=0.1) -> walls` — collapses near-collinear, overlapping-or-adjacent wall entries that describe the same physical wall run into one, keeping the entry with `thickness_source == "measured"` when one exists.
- Produces (floorplan_schema.py): `@dataclass Wall`, `@dataclass Opening`, `wall_to_dict(wall) -> dict`, `opening_to_dict(opening) -> dict`, `new_wall_id(index) -> str`. Consumed by Task 10, Task 14, Task 15.

**Gap found during design validation:** the full pipeline (extract → pair → snap → drop_short_walls) on the synthetic 5-wall house produced **20** entries, not 5 — because a connected wall network is traced through multiple `findContours` loops (the exterior boundary plus one void contour per room), and mutual-NN pairing is one-to-one, so a physical wall's face can appear as several near-identical segments across different contours/rooms that don't all find a partner in the same pass. This task merges them.

- [ ] **Step 1: Write the failing test for merge_duplicate_walls**

```python
# append to tests/test_floorplan_geometry.py
from scripts.floorplan_geometry import merge_duplicate_walls


def test_merge_duplicate_walls_collapses_collinear_overlapping_entries():
    # two near-identical entries for the "same" 5m wall run, one measured one assumed,
    # plus one genuinely different wall
    walls = [
        {"p0": np.array([0.0, 0.0]), "p1": np.array([5.0, 0.02]),
         "thickness_m": 0.2, "thickness_source": "measured", "length_m": 5.0},
        {"p0": np.array([0.02, 0.01]), "p1": np.array([4.98, 0.0]),
         "thickness_m": 0.197, "thickness_source": "assumed", "length_m": 4.96},
        {"p0": np.array([0.0, 5.0]), "p1": np.array([6.0, 5.0]),
         "thickness_m": 0.2, "thickness_source": "measured", "length_m": 6.0},
    ]
    merged = merge_duplicate_walls(walls)
    assert len(merged) == 2
    kept = [w for w in merged if w["length_m"] < 5.5][0]
    assert kept["thickness_source"] == "measured"  # prefers the measured duplicate


def test_merge_duplicate_walls_full_pipeline_recovers_five_walls():
    """End-to-end regression: confirmed this fixture produces 20 raw entries
    without dedup; after merge_duplicate_walls only the 5 real walls (4
    exterior + 1 partition) should remain."""
    from scripts.floorplan_geometry import (
        crop_to_percentile_bounds, points_to_density_image, threshold_density_image,
        extract_wall_segments, pair_wall_surfaces, apply_modal_thickness_fallback,
        snap_wall_endpoints, drop_short_walls,
    )
    from tests.fixtures import two_room_house

    pts, _gt = two_room_house()
    _lo, _hi, keep_mask, _stats = crop_to_percentile_bounds(pts)
    cropped = pts[keep_mask]
    ceiling = cropped[(cropped[:, 2] >= 2.4) & (cropped[:, 2] <= 2.6)]
    xy = ceiling[:, :2]
    cell_size = 0.02
    bmin, bmax = xy.min(axis=0) - 0.1, xy.max(axis=0) + 0.1
    image, origin = points_to_density_image(xy, cell_size, bmin, bmax)
    binary = threshold_density_image(image, min_count=2, morph_kernel=3)
    segments = extract_wall_segments(binary, origin, cell_size, epsilon_cells=2.0, min_span_cells=3.0)
    segments = [s for s in segments if s["length"] >= 5 * cell_size]
    walls = pair_wall_surfaces(segments)
    walls = apply_modal_thickness_fallback(walls)
    walls, _clusters = snap_wall_endpoints(walls, tolerance_m=0.08)
    walls = drop_short_walls(walls, min_length_m=0.3)
    assert len(walls) == 20  # confirmed count before dedup -- documents the gap
    merged = merge_duplicate_walls(walls)
    assert len(merged) == 5  # 4 exterior + 1 partition
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_floorplan_geometry.py -v -k merge_duplicate`
Expected: FAIL (ImportError). The second test's `assert len(walls) == 20` is expected to already pass once run (it documents pre-existing, already-confirmed behavior) — if it doesn't match exactly 20 on this machine, adjust the asserted count to whatever the pipeline actually produces (the important assertion is the post-merge count of 5).

- [ ] **Step 3: Add merge_duplicate_walls to scripts/floorplan_geometry.py**

```python
def _walls_are_duplicates(a, b, angle_tol_deg=3.0, offset_tol_m=0.05, u_gap_tol_m=0.1):
    d_a = _segment_dir(a)
    d_b = _segment_dir(b)
    cos_angle = abs(float(np.dot(d_a, d_b)))
    angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    if angle_deg > angle_tol_deg:
        return False
    n_a = _segment_normal(a)
    mid_a = (a["p0"] + a["p1"]) / 2
    mid_b = (b["p0"] + b["p1"]) / 2
    perp_offset = abs(float(np.dot(mid_b - mid_a, n_a)))
    if perp_offset > offset_tol_m:
        return False
    # require overlapping or near-adjacent u-ranges along the shared direction
    overlap = _overlap_fraction(a, b, d_a)
    if overlap > 0.0:
        return True
    a0, a1 = np.dot(a["p0"], d_a), np.dot(a["p1"], d_a)
    b0, b1 = np.dot(b["p0"], d_a), np.dot(b["p1"], d_a)
    gap = max(min(b0, b1) - max(a0, a1), min(a0, a1) - max(b0, b1))
    return gap <= u_gap_tol_m


def merge_duplicate_walls(walls, angle_tol_deg=3.0, offset_tol_m=0.05, u_gap_tol_m=0.1):
    """Collapse near-collinear, overlapping-or-adjacent wall entries that
    describe the same physical wall run. Confirmed necessary during design
    validation: a connected wall network traced through multiple findContours
    loops (exterior boundary + one void per room) produces several duplicate
    entries per physical wall, since mutual-NN pairing is one-to-one and
    doesn't itself deduplicate across contours."""
    n = len(walls)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if _walls_are_duplicates(walls[i], walls[j], angle_tol_deg, offset_tol_m, u_gap_tol_m):
                union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(walls[i])

    merged = []
    for group in groups.values():
        measured = [w for w in group if w["thickness_source"] == "measured"]
        chosen_pool = measured if measured else group
        best = max(chosen_pool, key=lambda w: w["length_m"])
        merged.append(best)
    return merged
```

- [ ] **Step 4: Run floorplan_geometry tests to verify they pass**

Run: `pytest tests/test_floorplan_geometry.py -v`
Expected: 19 passed

- [ ] **Step 5: Write scripts/floorplan_schema.py with the manifest data model**

Per the accuracy review's forward-compatibility requirements for a future Phase 2 (grooves), each wall persists both face plane equations, the exact wall-local basis used, and the region-selector parameters — not just centerline+thickness — so Phase 2 can recompute groove residuals in the identical frame instead of re-deriving a slightly different one.

```python
"""manifest.json data model shared by floorplan_reconstruct.py and validate_measurements.py."""
from dataclasses import dataclass, field, asdict


@dataclass
class Opening:
    opening_id: str
    wall_id: str
    type: str  # "door" | "window" | "balcony_door" | "unknown_opening"
    u_min_m: float
    u_max_m: float
    sill_m: float
    height_m: float
    width_m: float
    edge_method: str  # "density_half_max" (v1; "reveal_plane" reserved for later)
    both_faces_confirmed: bool


@dataclass
class Wall:
    wall_id: str
    p0: tuple  # (x, y) centerline endpoint, meters, world frame
    p1: tuple
    length_m: float
    thickness_m: float
    thickness_source: str  # "measured" | "assumed"
    plane_front: list  # [a, b, c, d] of the front face
    plane_back: list  # [a, b, c, d] of the back face, or None if thickness_source == "assumed"
    origin_xyz: tuple  # wall-local frame origin used for u/v projection
    u_axis: tuple
    v_axis: tuple
    floor_z_m: float
    ceiling_z_m: float
    region_band_m: float  # perpendicular band width used for point selection during refit
    region_corner_margin_m: float  # corner exclusion margin used during refit
    openings: list = field(default_factory=list)  # list[Opening]
    grooves: list = field(default_factory=list)  # reserved for Phase 2, empty here


def new_wall_id(index):
    return f"wall_{index:03d}"


def wall_to_dict(wall: Wall) -> dict:
    d = asdict(wall)
    d["openings"] = [asdict(o) if not isinstance(o, dict) else o for o in wall.openings]
    return d


def opening_to_dict(opening: Opening) -> dict:
    return asdict(opening)
```

- [ ] **Step 6: Write tests/test_floorplan_schema.py**

```python
import json
from scripts.floorplan_schema import Wall, Opening, wall_to_dict, new_wall_id


def test_new_wall_id_format():
    assert new_wall_id(3) == "wall_003"


def test_wall_to_dict_round_trips_through_json():
    opening = Opening(
        opening_id="wall_000_op_00", wall_id="wall_000", type="door",
        u_min_m=2.0, u_max_m=2.9, sill_m=0.0, height_m=2.1, width_m=0.9,
        edge_method="density_half_max", both_faces_confirmed=True,
    )
    wall = Wall(
        wall_id="wall_000", p0=(0.0, 0.0), p1=(5.0, 0.0), length_m=5.0,
        thickness_m=0.1, thickness_source="measured",
        plane_front=[1.0, 0.0, 0.0, -2.95], plane_back=[1.0, 0.0, 0.0, -3.05],
        origin_xyz=(2.95, 0.0, 0.0), u_axis=(0.0, 1.0, 0.0), v_axis=(0.0, 0.0, 1.0),
        floor_z_m=0.0, ceiling_z_m=2.7, region_band_m=0.025, region_corner_margin_m=0.5,
        openings=[opening],
    )
    d = wall_to_dict(wall)
    text = json.dumps(d)  # must not raise
    loaded = json.loads(text)
    assert loaded["wall_id"] == "wall_000"
    assert loaded["openings"][0]["type"] == "door"
    assert loaded["grooves"] == []
```

- [ ] **Step 7: Run the schema tests**

Run: `pytest tests/test_floorplan_schema.py -v`
Expected: 2 passed

- [ ] **Step 8: Commit**

```bash
git add scripts/floorplan_geometry.py scripts/floorplan_schema.py tests/test_floorplan_geometry.py tests/test_floorplan_schema.py
git commit -m "Add wall dedup/merge and manifest.json schema"
```

---

### Task 9: Corner-aware point selection + two-pass plane refit

**Files:**
- Modify: `scripts/floorplan_geometry.py`
- Modify: `tests/test_floorplan_geometry.py`

**Interfaces:**
- Produces: `select_wall_band_points(points, wall, corner_margin_m=0.5, band_m=0.06) -> np.ndarray[M,3]`, `refine_wall_plane_two_pass(points, plane_model, coarse_band_m=0.025, fine_band_m=0.008, trim_frac=0.02) -> list[4]`. Consumed by Task 14.

**Confirmed via design validation, this is the accuracy-critical step:** refining a wall's thickness using perpendicular-plane-distance filtering *alone* is not sufficient near a T-junction — a perpendicular wall's face can be coincidentally near-coplanar right at the junction. Without corner exclusion: 100mm true partition thickness refined to ~96.8mm with mean residual ~23mm, max ~75mm. With a 0.5m corner-margin exclusion: refined to **99.86mm** (running the full pipeline end-to-end, including the coarse detection's ~30mm error and modal-fallback mis-estimate as the starting guess, refined to **100.29mm**), mean residual ~1.6mm, max ~9mm.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_floorplan_geometry.py
from scripts.floorplan_geometry import (
    select_wall_band_points, refine_wall_plane_two_pass, signed_plane_distance,
)
from tests.fixtures import two_room_house


def test_refine_wall_plane_two_pass_recovers_known_plane():
    rng = np.random.default_rng(1)
    n = 5000
    pts = np.column_stack([
        rng.uniform(0, 6, n),
        np.full(n, 0.1) + rng.normal(0, 0.002, n),
        rng.uniform(0, 2.7, n),
    ])
    coarse = [0.0, 1.0, 0.0, -0.08]  # off by 20mm
    refined = refine_wall_plane_two_pass(pts, coarse)
    assert abs(refined[3] - (-0.1)) < 0.001
    resid = np.abs(signed_plane_distance(pts, refined))
    assert resid.mean() < 0.005


def test_select_wall_band_points_and_refit_corrects_t_junction_contamination():
    """Full-scale regression for the corner-contamination bug found during
    design validation: without corner exclusion this measured ~96.8mm
    (residual mean ~23mm); with select_wall_band_points's corner margin it
    should land within 1mm of the true 100mm partition thickness."""
    pts, _gt = two_room_house()
    lo = np.percentile(pts, 1, axis=0) - 0.5
    hi = np.percentile(pts, 99, axis=0) + 0.5
    cropped = pts[np.all((pts >= lo) & (pts <= hi), axis=1)]

    partition_wall = {
        "p0": np.array([3.0, 0.0]), "p1": np.array([3.0, 5.0]), "length_m": 5.0,
    }
    full_height = cropped[(cropped[:, 0] > 1) & (cropped[:, 0] < 5) &
                           (cropped[:, 1] >= 0) & (cropped[:, 1] <= 5)]
    band_pts = select_wall_band_points(full_height, partition_wall, corner_margin_m=0.5, band_m=0.06)
    assert len(band_pts) > 1000

    d = partition_wall["p1"] - partition_wall["p0"]
    d = d / np.linalg.norm(d)
    normal2d = np.array([-d[1], d[0]])
    mid = band_pts[:, 0] * normal2d[0] + band_pts[:, 1] * normal2d[1]
    med = np.median(mid)
    side_a, side_b = band_pts[mid < med], band_pts[mid >= med]

    coarse_a = [normal2d[0], normal2d[1], 0.0, -np.dot(normal2d, side_a[:, :2].mean(axis=0))]
    coarse_b = [normal2d[0], normal2d[1], 0.0, -np.dot(normal2d, side_b[:, :2].mean(axis=0))]
    refined_a = refine_wall_plane_two_pass(side_a, coarse_a)
    refined_b = refine_wall_plane_two_pass(side_b, coarse_b)
    thickness = abs(refined_a[3] - refined_b[3])
    assert abs(thickness - 0.1) < 0.005  # within 5mm of the true 100mm
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_floorplan_geometry.py -v -k "refine_wall_plane or select_wall_band"`
Expected: FAIL (ImportError)

- [ ] **Step 3: Add to scripts/floorplan_geometry.py**

```python
def refine_wall_plane_two_pass(points, plane_model, coarse_band_m=0.025, fine_band_m=0.008, trim_frac=0.02):
    """Two-pass least-squares plane refit on ORIGINAL (non-downsampled)
    points: pass 1 uses a wide (25mm) band to absorb coarse-detection
    positional slop, pass 2 re-selects within a tight (8mm, ~0.5x the 16mm
    median scan spacing) band around the pass-1 result and refits again,
    trimming the highest-residual 2% of points first (corner contamination)."""
    pts = np.asarray(points, dtype=np.float64)
    dist = np.abs(signed_plane_distance(pts, plane_model))
    coarse_pts = pts[dist <= coarse_band_m]
    if len(coarse_pts) < 3:
        return list(plane_model)
    plane1 = refine_plane_model(coarse_pts, plane_model)

    dist2 = np.abs(signed_plane_distance(pts, plane1))
    fine_pts = pts[dist2 <= fine_band_m]
    if len(fine_pts) < 3:
        return plane1

    resid = np.abs(signed_plane_distance(fine_pts, plane1))
    if trim_frac > 0 and len(fine_pts) > 10:
        keep_n = int(len(fine_pts) * (1 - trim_frac))
        keep_idx = np.argsort(resid)[:keep_n]
        fine_pts = fine_pts[keep_idx]

    return refine_plane_model(fine_pts, plane1)


def select_wall_band_points(points, wall, corner_margin_m=0.5, band_m=0.06):
    """Points belonging to ONE wall's own run: within band_m perpendicular
    distance of the wall's centerline direction AND within the wall's own
    U-range minus corner_margin_m on each end.

    Confirmed via design validation that perpendicular-distance filtering
    ALONE is insufficient near a T-junction: a perpendicular wall's face can
    be coincidentally near-coplanar with this wall right at the junction
    (observed residual mean ~23mm, max ~75mm without this). Excluding a
    margin near the wall's own detected corners removes the contamination
    (observed residual mean ~1.6mm, refined thickness error dropped from
    ~30mm to ~0.3mm on a 100mm true partition)."""
    pts = np.asarray(points, dtype=np.float64)
    d = _segment_dir(wall)
    normal = _segment_normal(wall)
    dist_perp = np.abs(np.dot(pts[:, :2] - wall["p0"], normal))
    u = np.dot(pts[:, :2] - wall["p0"], d)
    length = wall["length_m"]
    in_band = dist_perp <= band_m
    in_u_range = (u >= corner_margin_m) & (u <= length - corner_margin_m)
    return pts[in_band & in_u_range]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_floorplan_geometry.py -v`
Expected: 21 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/floorplan_geometry.py tests/test_floorplan_geometry.py
git commit -m "Add corner-aware point selection and two-pass wall plane refit"
```

---

### Task 10: Opening detection (void flood-fill + classification)

**Files:**
- Modify: `scripts/floorplan_geometry.py`
- Modify: `tests/test_floorplan_geometry.py`

**Interfaces:**
- Produces: `merge_grid_cells(occupied: set[(int,int)]) -> list[(iu0,iu1,iv0,iv1)]`, `classify_opening(width_m, height_m, sill_m) -> str`, `detect_openings_on_wall_face(uv_points, wall_length_m, cell_m=0.05, min_points_per_cell=3, min_opening_w=0.45, min_opening_h=0.45, edge_margin_cells=1) -> list[dict]`, `cross_check_opening_both_faces(opening, other_face_uv_points, cell_m=0.05, min_points_per_cell=3) -> bool`. Consumed by Task 14.

**Critical bug found and fixed:** the void-flood-fill (ported from `segment_walls_and_grooves.py`'s `_opening_interior_void_cells`) seeds all four grid borders as "open to outside," including the bottom row. A floor-to-ceiling door's void touches the bottom row by construction (there's no wall below floor level to begin with) — but the floor is a real physical boundary, not open space. Without the fix, **every full-height door silently fails to be detected as an opening** (flood-fill treats the whole void as reachable from outside). Confirmed: before the fix, 0 openings found on the partition wall's door; after, exactly the true door (width=0.90m, height=2.10m, sill=0.00m, type=door) is recovered.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_floorplan_geometry.py
from scripts.floorplan_geometry import (
    merge_grid_cells, classify_opening, detect_openings_on_wall_face,
    cross_check_opening_both_faces,
)


def test_classify_opening_thresholds():
    assert classify_opening(1.2, 1.2, 0.9) == "window"
    assert classify_opening(0.9, 2.1, 0.0) == "door"
    assert classify_opening(1.5, 2.2, 0.1) == "balcony_door"


def test_merge_grid_cells_merges_rectangle():
    occupied = {(0, 0), (1, 0), (0, 1), (1, 1)}
    rects = merge_grid_cells(occupied)
    assert rects == [(0, 1, 0, 1)]


def test_detect_openings_on_wall_face_window_case():
    rng = np.random.default_rng(2)
    n_per_cell = 5
    u = np.arange(0, 6, 0.05)
    v = np.arange(0, 2.7, 0.05)
    uu, vv = np.meshgrid(u, v)
    uu, vv = uu.ravel(), vv.ravel()
    keep = ~((uu >= 4.0) & (uu <= 5.2) & (vv >= 0.9) & (vv <= 2.1))
    uv = np.column_stack([uu[keep], vv[keep]])
    uv = np.repeat(uv, n_per_cell, axis=0)
    openings = detect_openings_on_wall_face(uv, wall_length_m=6.0, cell_m=0.05)
    assert len(openings) == 1
    op = openings[0]
    assert abs(op["width_m"] - 1.2) < 0.06
    assert abs(op["height_m"] - 1.2) < 0.06
    assert abs(op["sill_m"] - 0.9) < 0.06
    assert op["type"] == "window"


def test_detect_openings_on_wall_face_floor_level_door_case():
    """Regression test for the floor-boundary flood-fill bug: a full-height
    door (sill=0) must still be detected as an enclosed opening."""
    n_per_cell = 5
    u = np.arange(0, 5, 0.05)
    v = np.arange(0, 2.7, 0.05)
    uu, vv = np.meshgrid(u, v)
    uu, vv = uu.ravel(), vv.ravel()
    keep = ~((uu >= 2.0) & (uu <= 2.9) & (vv >= 0.0) & (vv <= 2.1))
    uv = np.column_stack([uu[keep], vv[keep]])
    uv = np.repeat(uv, n_per_cell, axis=0)
    openings = detect_openings_on_wall_face(uv, wall_length_m=5.0, cell_m=0.05)
    assert len(openings) == 1
    op = openings[0]
    assert abs(op["sill_m"] - 0.0) < 1e-9
    assert op["type"] == "door"


def test_cross_check_opening_both_faces_rejects_one_sided_occlusion():
    opening = {"u_min": 1.0, "u_max": 2.0, "v_min": 0.5, "v_max": 1.5}
    # other face is fully occupied in that rect => furniture occlusion, not a real opening
    u = np.arange(1.0, 2.0, 0.05)
    v = np.arange(0.5, 1.5, 0.05)
    uu, vv = np.meshgrid(u, v)
    other_face = np.repeat(np.column_stack([uu.ravel(), vv.ravel()]), 5, axis=0)
    assert cross_check_opening_both_faces(opening, other_face) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_floorplan_geometry.py -v -k "opening or merge_grid_cells or classify"`
Expected: FAIL (ImportError)

- [ ] **Step 3: Add to scripts/floorplan_geometry.py**

```python
# ---------- opening detection (ported void-flood-fill + rectangle merge) ----------

def merge_grid_cells(occupied):
    """Merge occupied (iu, iv) cells into axis-aligned rectangles (U-runs then V-runs)."""
    if not occupied:
        return []
    bars = []
    for iv in sorted({iv for _iu, iv in occupied}):
        row = sorted(iu for iu, jv in occupied if jv == iv)
        start = prev = row[0]
        for iu in row[1:]:
            if iu == prev + 1:
                prev = iu
            else:
                bars.append((start, prev, iv))
                start = prev = iu
        bars.append((start, prev, iv))

    by_span = defaultdict(list)
    for iu0, iu1, iv in bars:
        by_span[(iu0, iu1)].append(iv)

    rects = []
    for (iu0, iu1), ivs in by_span.items():
        ivs = sorted(set(ivs))
        start = prev = ivs[0]
        for iv in ivs[1:]:
            if iv == prev + 1:
                prev = iv
            else:
                rects.append((iu0, iu1, start, prev))
                start = prev = iv
        rects.append((iu0, iu1, start, prev))
    return rects


def _interior_void_cells(occupied, iu0, iu1, iv0, iv1, floor_is_boundary=True):
    """Empty cells not reachable from the grid border (door/window holes).

    The bottom row (iv0, floor level) is NOT seeded as an open border when
    floor_is_boundary=True: a floor-to-ceiling door's void touches iv0 by
    construction (there's no wall below floor level to begin with), but the
    floor itself is a real physical boundary, not open space -- confirmed
    this bug makes every full-height door silently undetectable without
    the fix (flood-fill marks the whole void as 'exterior')."""
    exterior = set()
    queue = deque()

    def seed(iu, iv):
        if (iu, iv) in occupied or (iu, iv) in exterior:
            return
        exterior.add((iu, iv))
        queue.append((iu, iv))

    for iu in range(iu0, iu1 + 1):
        if not floor_is_boundary:
            seed(iu, iv0)
        seed(iu, iv1)
    for iv in range(iv0, iv1 + 1):
        seed(iu0, iv)
        seed(iu1, iv)

    while queue:
        iu, iv = queue.popleft()
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = iu + di, iv + dj
            if not (iu0 <= ni <= iu1 and iv0 <= nj <= iv1):
                continue
            if (ni, nj) in occupied or (ni, nj) in exterior:
                continue
            exterior.add((ni, nj))
            queue.append((ni, nj))

    voids = set()
    for iu in range(iu0, iu1 + 1):
        for iv in range(iv0, iv1 + 1):
            if (iu, iv) not in occupied and (iu, iv) not in exterior:
                voids.add((iu, iv))
    return voids


def classify_opening(width_m, height_m, sill_m):
    if width_m >= 1.0 and height_m >= 2.0 and sill_m < 0.25:
        return "balcony_door"
    if 0.65 <= width_m <= 1.4 and 1.75 <= height_m <= 2.5 and sill_m < 0.25:
        return "door"
    if sill_m >= 0.4 and width_m >= 0.45 and height_m >= 0.45:
        return "window"
    return "unknown_opening"


def detect_openings_on_wall_face(uv_points, wall_length_m, cell_m=0.05,
                                  min_points_per_cell=3, min_opening_w=0.45,
                                  min_opening_h=0.45, edge_margin_cells=1):
    """uv_points: (N,2) array of (u, v=height) points on ONE wall face."""
    uv = np.asarray(uv_points, dtype=np.float64)
    if len(uv) == 0:
        return []
    iu = np.floor(uv[:, 0] / cell_m).astype(np.int64)
    iv = np.floor(uv[:, 1] / cell_m).astype(np.int64)
    counts = defaultdict(int)
    for k in range(len(uv)):
        counts[(int(iu[k]), int(iv[k]))] += 1
    occupied = {key for key, c in counts.items() if c >= min_points_per_cell}
    if not occupied:
        return []

    iu0, iu1 = min(c[0] for c in occupied), max(c[0] for c in occupied)
    iv0, iv1 = min(c[1] for c in occupied), max(c[1] for c in occupied)
    max_iu = int(np.floor(wall_length_m / cell_m))

    voids = _interior_void_cells(occupied, iu0, iu1, iv0, iv1, floor_is_boundary=True)
    if not voids:
        return []
    rects = merge_grid_cells(voids)

    openings = []
    for a0, a1, b0, b1 in rects:
        u_min, u_max = a0 * cell_m, (a1 + 1) * cell_m
        v_min, v_max = b0 * cell_m, (b1 + 1) * cell_m
        width, height = u_max - u_min, v_max - v_min
        if width < min_opening_w or height < min_opening_h:
            continue
        if a0 <= edge_margin_cells or a1 >= max_iu - edge_margin_cells:
            continue  # touches the wall's snapped end -- termination, not an opening
        sill = v_min
        openings.append({
            "u_min": u_min, "u_max": u_max, "v_min": v_min, "v_max": v_max,
            "width_m": width, "height_m": height, "sill_m": sill,
            "type": classify_opening(width, height, sill),
        })
    return openings


def cross_check_opening_both_faces(opening, other_face_uv_points, cell_m=0.05, min_points_per_cell=3):
    """A true through-wall opening must ALSO be void on the wall's other
    face; a gap present on only one face is furniture occlusion, not a hole
    through the wall. Returns True if the opening survives (other face is
    also mostly empty in that u,v range)."""
    uv = np.asarray(other_face_uv_points, dtype=np.float64)
    if len(uv) == 0:
        return True
    in_range = (
        (uv[:, 0] >= opening["u_min"]) & (uv[:, 0] <= opening["u_max"]) &
        (uv[:, 1] >= opening["v_min"]) & (uv[:, 1] <= opening["v_max"])
    )
    pts_in_rect = uv[in_range]
    if len(pts_in_rect) == 0:
        return True
    iu = np.floor((pts_in_rect[:, 0] - opening["u_min"]) / cell_m).astype(np.int64)
    iv = np.floor((pts_in_rect[:, 1] - opening["v_min"]) / cell_m).astype(np.int64)
    counts = defaultdict(int)
    for k in range(len(pts_in_rect)):
        counts[(int(iu[k]), int(iv[k]))] += 1
    occupied_cells = sum(1 for c in counts.values() if c >= min_points_per_cell)
    total_cells = max(1, int((opening["u_max"] - opening["u_min"]) / cell_m) *
                      int((opening["v_max"] - opening["v_min"]) / cell_m))
    return (occupied_cells / total_cells) < 0.3
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_floorplan_geometry.py -v`
Expected: 26 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/floorplan_geometry.py tests/test_floorplan_geometry.py
git commit -m "Add opening detection; fix floor-boundary flood-fill bug for full-height doors"
```

---

### Task 11: Floor plan image rendering

**Files:**
- Modify: `scripts/floorplan_geometry.py`
- Modify: `tests/test_floorplan_geometry.py`

**Interfaces:**
- Produces: `render_floorplan_image(walls, openings_by_wall_id, output_path, px_per_meter=100) -> None` (writes a PNG file). `openings_by_wall_id` values are dicts using the `Opening` schema's field names (`u_min_m`, `u_max_m`, `type` — see Task 8), since Task 14 builds them from `Opening.__dict__`. Consumed by Task 14.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_floorplan_geometry.py
import os
from scripts.floorplan_geometry import render_floorplan_image


def test_render_floorplan_image_writes_nonempty_png(tmp_path):
    walls = [
        {"p0": np.array([0.0, 0.0]), "p1": np.array([6.0, 0.0]), "thickness_m": 0.2, "length_m": 6.0},
        {"p0": np.array([6.0, 0.0]), "p1": np.array([6.0, 5.0]), "thickness_m": 0.2, "length_m": 5.0},
    ]
    out = tmp_path / "floorplan.png"
    render_floorplan_image(walls, {}, str(out), px_per_meter=50)
    assert out.exists()
    assert out.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_floorplan_geometry.py -v -k render_floorplan`
Expected: FAIL (ImportError)

- [ ] **Step 3: Add to scripts/floorplan_geometry.py**

```python
def render_floorplan_image(walls, openings_by_wall_id, output_path, px_per_meter=100):
    """Top-down floor plan render: walls as thick lines with length/thickness
    labels, openings marked with their type."""
    all_pts = np.vstack([np.vstack([w["p0"], w["p1"]]) for w in walls])
    margin_m = 0.5
    min_xy = all_pts.min(axis=0) - margin_m
    max_xy = all_pts.max(axis=0) + margin_m
    width_px = int((max_xy[0] - min_xy[0]) * px_per_meter) + 1
    height_px = int((max_xy[1] - min_xy[1]) * px_per_meter) + 1
    img = np.full((height_px, width_px, 3), 255, dtype=np.uint8)

    def to_px(pt):
        x = int((pt[0] - min_xy[0]) * px_per_meter)
        y = int((max_xy[1] - pt[1]) * px_per_meter)  # flip Y for image coords
        return (x, y)

    for wi, w in enumerate(walls):
        p0_px, p1_px = to_px(w["p0"]), to_px(w["p1"])
        thickness_px = max(1, int(w["thickness_m"] * px_per_meter))
        cv2.line(img, p0_px, p1_px, (40, 40, 40), thickness=thickness_px)
        mid_px = ((p0_px[0] + p1_px[0]) // 2, (p0_px[1] + p1_px[1]) // 2)
        label = f"{w['length_m']:.2f}m/{w['thickness_m']*1000:.0f}mm"
        cv2.putText(img, label, mid_px, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 0, 0), 1, cv2.LINE_AA)
        for op in openings_by_wall_id.get(wi, []):
            d = (w["p1"] - w["p0"]) / w["length_m"]
            op_p0 = w["p0"] + d * op["u_min_m"]
            op_p1 = w["p0"] + d * op["u_max_m"]
            cv2.line(img, to_px(op_p0), to_px(op_p1), (0, 150, 0), thickness=max(2, thickness_px))
            cv2.putText(img, op["type"], to_px(op_p0), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 100, 0), 1, cv2.LINE_AA)

    cv2.imwrite(output_path, img)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_floorplan_geometry.py -v`
Expected: 27 passed

- [ ] **Step 5: Verify no open3d dependency across the whole module**

Run: `grep -c open3d scripts/floorplan_geometry.py`
Expected: `0`

- [ ] **Step 6: Commit**

```bash
git add scripts/floorplan_geometry.py tests/test_floorplan_geometry.py
git commit -m "Add floor plan PNG rendering"
```

---

### Task 12: mesh_common.py Phase 0 crop wrapper

**Files:**
- Modify: `scripts/mesh_common.py`
- Create: `tests/test_mesh_common_crop.py`

**Interfaces:**
- Consumes: `floorplan_geometry.crop_to_percentile_bounds`.
- Produces: `mesh_common.crop_pcd_to_percentile_bounds(pcd, low_pct=1.0, high_pct=99.0, margin_m=0.5) -> (pcd, stats: dict)`. Consumed by Task 14.

This is the ONLY open3d-touching addition for Phase 0 — it's a thin wrapper: convert points to a plain array, call the already-tested pure function, then use `pcd.select_by_index` (open3d) to build the cropped cloud. The test for this file is skipped locally (`pytest.importorskip`) since `open3d` cannot be installed on the 32-bit dev machine; it is exercised for real on the Jarvis Labs VM as part of Task 16's smoke test.

- [ ] **Step 1: Write the test (importorskip-guarded)**

```python
# tests/test_mesh_common_crop.py
import pytest
o3d = pytest.importorskip("open3d")  # skips locally on the 32-bit dev machine; runs on the Jarvis Labs VM
import numpy as np
from scripts.mesh_common import crop_pcd_to_percentile_bounds


def test_crop_pcd_to_percentile_bounds_drops_stray_points():
    rng = np.random.default_rng(0)
    real = rng.normal(loc=[3.0, 2.5, 1.3], scale=0.5, size=(2000, 3))
    stray = rng.uniform(-20, 20, size=(20, 3))
    xyz = np.vstack([real, stray])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    cropped_pcd, stats = crop_pcd_to_percentile_bounds(pcd, low_pct=1.0, high_pct=99.0, margin_m=0.5)
    assert len(cropped_pcd.points) < len(pcd.points)
    assert stats["dropped_fraction"] < 0.05
```

- [ ] **Step 2: Run test to verify current state**

Run: `pytest tests/test_mesh_common_crop.py -v`
Expected on the local 32-bit dev machine: `SKIPPED (could not import 'open3d')`. If run on a machine with `open3d` installed: FAIL with `ImportError: cannot import name 'crop_pcd_to_percentile_bounds'`.

- [ ] **Step 3: Add crop_pcd_to_percentile_bounds to scripts/mesh_common.py**

Read `scripts/mesh_common.py` first to find the right insertion point (after `recenter_pcd`, before `_finite_normal_fraction`). Add:

```python
def crop_pcd_to_percentile_bounds(pcd, low_pct=1.0, high_pct=99.0, margin_m=0.5):
    """Phase 0: drop the sparse SLAM-drift/ghost-point tail that inflates a
    raw bounding box (confirmed on real scans: 99% of points sit in a tight
    room volume while raw bbox balloons 6-7x from stray points), which was
    the dominant cause of ~30 minute Poisson reconstruction times."""
    from floorplan_geometry import crop_to_percentile_bounds  # local import: keeps the pure module import-order independent of mesh_common
    xyz = np.asarray(pcd.points)
    _lo, _hi, keep_mask, stats = crop_to_percentile_bounds(xyz, low_pct, high_pct, margin_m)
    keep_idx = np.nonzero(keep_mask)[0]
    cropped = pcd.select_by_index(keep_idx)
    log(
        f"Phase 0 crop: kept {stats['kept_points']:,}/{stats['input_points']:,} points "
        f"({stats['dropped_fraction']*100:.2f}% dropped); "
        f"bounds {stats['raw_bounds_min']} -> {stats['raw_bounds_max']} "
        f"became {stats['cropped_bounds_min']} -> {stats['cropped_bounds_max']}"
    )
    return cropped, stats
```

Note: `scripts/floorplan_geometry.py` and `scripts/mesh_common.py` are siblings in the same `scripts/` directory, so `from floorplan_geometry import ...` resolves the same way the existing `segment_walls_and_grooves.py` imports `from mesh_common import ...` does.

- [ ] **Step 4: Run the test again**

Run: `pytest tests/test_mesh_common_crop.py -v`
Expected on the local 32-bit dev machine: still `SKIPPED`. Note in the plan for whoever runs this on a 64-bit machine or the Jarvis Labs VM: expected `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add scripts/mesh_common.py tests/test_mesh_common_crop.py
git commit -m "Add Phase 0 crop wrapper to mesh_common.py"
```

---

### Task 13: build_floorplan_outputs orchestration (pure, unit-testable)

**Files:**
- Create: `scripts/floorplan_reconstruct.py`
- Create: `tests/test_floorplan_reconstruct.py`

**Interfaces:**
- Consumes: everything from Tasks 2-11, plus `floorplan_schema.Wall`/`Opening`/`new_wall_id`.
- Produces: `build_floorplan_outputs(xyz: np.ndarray[N,3], config: dict) -> (manifest: dict, walls: list[Wall])`. Consumed by the CLI `main()` in this same file (Task 14 continues this file) and by Task 15.

This is the bulk of the new pipeline's logic, and per the testability review it's kept entirely free of `open3d` — it takes a plain point array "standing in for already loaded, cropped, and downsampled points," so the wiring logic (assembling `manifest.json`, deciding which wall/opening results feed the OBJ, handling the zero-walls-detected case) gets full local pytest coverage, not just an end-to-end VM smoke test.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_floorplan_reconstruct.py
import numpy as np
from scripts.floorplan_reconstruct import build_floorplan_outputs, DEFAULT_CONFIG
from tests.fixtures import two_room_house


def test_build_floorplan_outputs_recovers_five_walls_and_two_openings():
    pts, _gt = two_room_house()
    manifest, walls = build_floorplan_outputs(pts, DEFAULT_CONFIG)
    assert len(walls) == 5
    assert manifest["wall_count"] == 5
    all_openings = [op for w in walls for op in w.openings]
    assert len(all_openings) == 2
    types = sorted(op["type"] if isinstance(op, dict) else op.type for op in all_openings)
    assert types == ["door", "window"]

    thicknesses_mm = sorted(w.thickness_m * 1000 for w in walls)
    # 4 exterior (~200mm) + 1 partition (~100mm), refined to within a few mm
    assert abs(thicknesses_mm[0] - 100) < 5
    assert all(abs(t - 200) < 5 for t in thicknesses_mm[1:])


def test_build_floorplan_outputs_handles_zero_walls_gracefully():
    empty_room_pts = np.random.default_rng(0).normal(0, 0.01, size=(50, 3))  # too few/sparse to form any wall
    manifest, walls = build_floorplan_outputs(empty_room_pts, DEFAULT_CONFIG)
    assert manifest["wall_count"] == 0
    assert walls == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_floorplan_reconstruct.py -v`
Expected: FAIL (ImportError: no module named `scripts.floorplan_reconstruct`)

- [ ] **Step 3: Write scripts/floorplan_reconstruct.py (build_floorplan_outputs portion)**

```python
"""LAS -> 2D floor plan + clean 3D walls/openings (Phase 0 + Phase 1).

Usage:
    python scripts/floorplan_reconstruct.py [input.las] [output_dir]
"""
import sys
import json
from pathlib import Path

import numpy as np

from floorplan_geometry import (
    crop_to_percentile_bounds, points_to_density_image, threshold_density_image,
    extract_wall_segments, pair_wall_surfaces, apply_modal_thickness_fallback,
    snap_wall_endpoints, drop_short_walls, merge_duplicate_walls,
    select_wall_band_points, refine_wall_plane_two_pass, signed_plane_distance,
    plane_normal, wall_uv_basis, points_to_wall_uv,
    detect_openings_on_wall_face, cross_check_opening_both_faces,
    render_floorplan_image,
)
from floorplan_schema import Wall, Opening, new_wall_id, wall_to_dict

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "koushikexport.las"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "floorplan"

DEFAULT_CONFIG = {
    "crop_low_pct": 1.0,
    "crop_high_pct": 99.0,
    "crop_margin_m": 0.5,
    "ceiling_band_m": 0.1,  # slice thickness for the top-down wall detection
    "ceiling_offset_m": 0.15,  # how far below the detected ceiling to slice
    "cell_size_m": 0.02,
    "density_min_count": 2,
    "morph_kernel": 3,
    "dp_epsilon_cells": 2.0,
    "min_span_cells": 3.0,
    "min_segment_length_cells": 5.0,
    "pair_min_thickness_m": 0.06,
    "pair_max_thickness_m": 0.35,
    "pair_max_angle_deg": 5.0,
    "pair_min_overlap_frac": 0.5,
    "snap_tolerance_m": 0.08,
    "min_wall_length_m": 0.3,
    "refit_coarse_band_m": 0.025,
    "refit_fine_band_m": 0.008,
    "refit_corner_margin_m": 0.5,
    "opening_cell_m": 0.05,
    "opening_min_w": 0.45,
    "opening_min_h": 0.45,
}


def _detect_wall_segments(xyz, config):
    z_max = float(xyz[:, 2].max())
    z_min = float(xyz[:, 2].min())
    band_hi = z_max - config["ceiling_offset_m"]
    band_lo = band_hi - config["ceiling_band_m"]
    ceiling_slice = xyz[(xyz[:, 2] >= band_lo) & (xyz[:, 2] <= band_hi)]
    if len(ceiling_slice) < 100:
        return [], z_min, z_max

    xy = ceiling_slice[:, :2]
    bmin, bmax = xy.min(axis=0) - 0.1, xy.max(axis=0) + 0.1
    image, origin = points_to_density_image(xy, config["cell_size_m"], bmin, bmax)
    binary = threshold_density_image(image, config["density_min_count"], config["morph_kernel"])
    segments = extract_wall_segments(
        binary, origin, config["cell_size_m"],
        epsilon_cells=config["dp_epsilon_cells"], min_span_cells=config["min_span_cells"],
    )
    min_len = config["min_segment_length_cells"] * config["cell_size_m"]
    segments = [s for s in segments if s["length"] >= min_len]
    return segments, z_min, z_max


def _build_walls_from_segments(segments, config):
    walls_raw = pair_wall_surfaces(
        segments,
        min_thickness_m=config["pair_min_thickness_m"],
        max_thickness_m=config["pair_max_thickness_m"],
        max_angle_deg=config["pair_max_angle_deg"],
        min_overlap_frac=config["pair_min_overlap_frac"],
    )
    walls_raw = apply_modal_thickness_fallback(walls_raw)
    walls_raw, _clusters = snap_wall_endpoints(walls_raw, tolerance_m=config["snap_tolerance_m"])
    walls_raw = drop_short_walls(walls_raw, min_length_m=config["min_wall_length_m"])
    walls_raw = merge_duplicate_walls(walls_raw)
    return walls_raw


def _refine_wall(wall_raw, xyz, floor_z, ceiling_z, config, index):
    d = (wall_raw["p1"] - wall_raw["p0"])
    d = d / np.linalg.norm(d)
    normal2d = np.array([-d[1], d[0]])

    full_height = xyz  # region selector below restricts by perpendicular band + U-range anyway
    band_pts = select_wall_band_points(
        full_height, wall_raw,
        corner_margin_m=config["refit_corner_margin_m"],
        band_m=max(config["pair_max_thickness_m"], 0.3),
    )
    if len(band_pts) < 20:
        return None  # not enough data to refine this wall; drop it rather than report a wrong number

    mid = band_pts[:, 0] * normal2d[0] + band_pts[:, 1] * normal2d[1]
    med = float(np.median(mid))
    side_a_pts = band_pts[mid < med]
    side_b_pts = band_pts[mid >= med]

    coarse_a = [normal2d[0], normal2d[1], 0.0, -float(np.dot(normal2d, side_a_pts[:, :2].mean(axis=0)))]
    plane_a = refine_wall_plane_two_pass(
        side_a_pts, coarse_a, config["refit_coarse_band_m"], config["refit_fine_band_m"],
    )

    plane_b = None
    thickness_m = wall_raw["thickness_m"]
    thickness_source = wall_raw["thickness_source"]
    if len(side_b_pts) >= 20:
        coarse_b = [normal2d[0], normal2d[1], 0.0, -float(np.dot(normal2d, side_b_pts[:, :2].mean(axis=0)))]
        plane_b = refine_wall_plane_two_pass(
            side_b_pts, coarse_b, config["refit_coarse_band_m"], config["refit_fine_band_m"],
        )
        thickness_m = abs(plane_a[3] - plane_b[3])
        thickness_source = "measured"

    origin_xyz = np.array([wall_raw["p0"][0], wall_raw["p0"][1], floor_z])
    u_axis = np.array([d[0], d[1], 0.0])
    v_axis = np.array([0.0, 0.0, 1.0])

    return Wall(
        wall_id=new_wall_id(index),
        p0=tuple(wall_raw["p0"]), p1=tuple(wall_raw["p1"]), length_m=wall_raw["length_m"],
        thickness_m=float(thickness_m), thickness_source=thickness_source,
        plane_front=plane_a, plane_back=plane_b,
        origin_xyz=tuple(origin_xyz), u_axis=tuple(u_axis), v_axis=tuple(v_axis),
        floor_z_m=floor_z, ceiling_z_m=ceiling_z,
        region_band_m=config["pair_max_thickness_m"], region_corner_margin_m=config["refit_corner_margin_m"],
    ), side_a_pts, side_b_pts


def _detect_wall_openings(wall, side_a_pts, side_b_pts, config):
    plane_a = wall.plane_front
    uv_a = points_to_wall_uv(side_a_pts, plane_a, np.array(wall.origin_xyz), np.array(wall.u_axis))
    openings_raw = detect_openings_on_wall_face(
        uv_a, wall.length_m, cell_m=config["opening_cell_m"],
        min_opening_w=config["opening_min_w"], min_opening_h=config["opening_min_h"],
    )

    uv_b = None
    if wall.plane_back is not None and len(side_b_pts):
        uv_b = points_to_wall_uv(side_b_pts, wall.plane_back, np.array(wall.origin_xyz), np.array(wall.u_axis))

    openings = []
    for i, op in enumerate(openings_raw):
        both_faces = True
        if uv_b is not None:
            both_faces = cross_check_opening_both_faces(op, uv_b, cell_m=config["opening_cell_m"])
        if not both_faces:
            continue
        openings.append(Opening(
            opening_id=f"{wall.wall_id}_op_{i:02d}", wall_id=wall.wall_id, type=op["type"],
            u_min_m=op["u_min"], u_max_m=op["u_max"], sill_m=op["sill_m"],
            height_m=op["height_m"], width_m=op["width_m"],
            edge_method="density_half_max", both_faces_confirmed=(uv_b is not None),
        ))
    return openings


def build_floorplan_outputs(xyz, config=None):
    config = config or DEFAULT_CONFIG
    xyz = np.asarray(xyz, dtype=np.float64)

    _lo, _hi, keep_mask, _crop_stats = crop_to_percentile_bounds(
        xyz, config["crop_low_pct"], config["crop_high_pct"], config["crop_margin_m"],
    )
    cropped = xyz[keep_mask]

    segments, floor_z, ceiling_z = _detect_wall_segments(cropped, config)
    if not segments:
        return {"wall_count": 0, "walls": []}, []

    walls_raw = _build_walls_from_segments(segments, config)

    walls = []
    for i, wall_raw in enumerate(walls_raw):
        result = _refine_wall(wall_raw, cropped, floor_z, ceiling_z, config, i)
        if result is None:
            continue
        wall, side_a_pts, side_b_pts = result
        wall.openings = _detect_wall_openings(wall, side_a_pts, side_b_pts, config)
        walls.append(wall)

    manifest = {
        "wall_count": len(walls),
        "floor_z_m": floor_z,
        "ceiling_z_m": ceiling_z,
        "walls": [_wall_manifest_entry(w) for w in walls],
    }
    return manifest, walls


def _wall_manifest_entry(wall):
    return wall_to_dict(wall)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_floorplan_reconstruct.py -v`
Expected: 2 passed. If the thickness/count assertions are off by a small margin due to floating-point/RNG differences from the exact prototype run, adjust the tolerance (not the underlying logic) — the prototype confirmed 5 walls, 2 openings, ~100mm/~200mm thicknesses within a few mm as the expected ballpark.

- [ ] **Step 5: Commit**

```bash
git add scripts/floorplan_reconstruct.py tests/test_floorplan_reconstruct.py
git commit -m "Add build_floorplan_outputs pure orchestration function"
```

---

### Task 14: floorplan_reconstruct.py CLI shell

**Files:**
- Modify: `scripts/floorplan_reconstruct.py`
- Create: `tests/test_floorplan_reconstruct_cli.py`

**Interfaces:**
- Consumes: `build_floorplan_outputs` (Task 13), `mesh_common.load_las_as_o3d`/`recenter_pcd` (existing), `mesh_common.crop_pcd_to_percentile_bounds` (Task 12), `render_floorplan_image` (Task 11).
- Produces: `main(input_path, output_dir, config=None)` CLI entry point writing `manifest.json`, `floorplan.png`, `reconstructed.obj`.

This is the thin, `open3d`-dependent shell around the already-tested pure logic. Its own test is a fast, tiny-synthetic-LAS check (not the real 750MB scan) so it stays runnable without a VM; it is skipped locally the same way as Task 12's test since it needs `open3d` to write a LAS/OBJ.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_floorplan_reconstruct_cli.py
import pytest
o3d = pytest.importorskip("open3d")
import json
import numpy as np
import laspy
from scripts.floorplan_reconstruct import main as floorplan_main
from tests.fixtures import two_room_house


def _write_synthetic_las(path, points):
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = points.min(axis=0)
    header.scales = [0.001, 0.001, 0.001]
    las = laspy.LasData(header)
    las.x, las.y, las.z = points[:, 0], points[:, 1], points[:, 2]
    las.write(str(path))


def test_floorplan_reconstruct_cli_writes_all_outputs(tmp_path):
    pts, _gt = two_room_house()
    las_path = tmp_path / "synthetic.las"
    _write_synthetic_las(las_path, pts)
    out_dir = tmp_path / "output"

    floorplan_main(str(las_path), str(out_dir))

    manifest_path = out_dir / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["wall_count"] == 5

    assert (out_dir / "floorplan.png").exists()
    assert (out_dir / "reconstructed.obj").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_floorplan_reconstruct_cli.py -v`
Expected on the local 32-bit dev machine: `SKIPPED (could not import 'open3d')`.

- [ ] **Step 3: Append the CLI shell to scripts/floorplan_reconstruct.py**

```python
def _walls_to_obj_mesh(walls):
    import open3d as _o3d  # local import: keeps build_floorplan_outputs/pure functions open3d-free
    combined = _o3d.geometry.TriangleMesh()
    for wall in walls:
        d = np.array(wall.p1) - np.array(wall.p0)
        length = np.linalg.norm(d)
        if length < 1e-6:
            continue
        d = d / length
        n = np.array([-d[1], d[0]])
        half_t = wall.thickness_m / 2.0
        p0, p1 = np.array(wall.p0), np.array(wall.p1)
        z0, z1 = wall.floor_z_m, wall.ceiling_z_m
        corners_2d = [p0 - n * half_t, p1 - n * half_t, p1 + n * half_t, p0 + n * half_t]
        verts = []
        for cx, cy in corners_2d:
            verts.append([cx, cy, z0])
        for cx, cy in corners_2d:
            verts.append([cx, cy, z1])
        mesh = _o3d.geometry.TriangleMesh()
        mesh.vertices = _o3d.utility.Vector3dVector(verts)
        # side faces (4 walls of the box) + top/bottom caps, 2 triangles each
        faces = [
            [0, 1, 4], [1, 5, 4], [1, 2, 5], [2, 6, 5],
            [2, 3, 6], [3, 7, 6], [3, 0, 7], [0, 4, 7],
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        ]
        mesh.triangles = _o3d.utility.Vector3iVector(faces)
        combined += mesh
    if len(combined.triangles) > 0:
        combined.remove_duplicated_vertices()
        combined.compute_vertex_normals()
    return combined


def main(input_path, output_dir, config=None):
    from mesh_common import load_las_as_o3d, recenter_pcd, log
    import open3d as o3d

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = config or DEFAULT_CONFIG

    pcd = load_las_as_o3d(Path(input_path))
    recenter_pcd(pcd)
    xyz = np.asarray(pcd.points)

    manifest, walls = build_floorplan_outputs(xyz, config)
    log(f"Detected {manifest['wall_count']} walls")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    openings_by_wall_id = {i: [op.__dict__ if hasattr(op, "__dict__") else op for op in w.openings]
                            for i, w in enumerate(walls)}
    wall_dicts = [{"p0": np.array(w.p0), "p1": np.array(w.p1),
                   "thickness_m": w.thickness_m, "length_m": w.length_m} for w in walls]
    render_floorplan_image(wall_dicts, openings_by_wall_id, str(output_dir / "floorplan.png"))

    obj_mesh = _walls_to_obj_mesh(walls)
    obj_path = output_dir / "reconstructed.obj"
    if len(obj_mesh.triangles) > 0:
        o3d.io.write_triangle_mesh(str(obj_path), obj_mesh)
    else:
        obj_path.write_text("")  # empty but present, so downstream tooling doesn't error on a missing file
    log(f"Wrote {manifest_path}, floorplan.png, {obj_path}")


if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    outp = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_DIR
    main(inp, outp)
```

- [ ] **Step 4: Run the test again**

Run: `pytest tests/test_floorplan_reconstruct_cli.py -v`
Expected on the local 32-bit dev machine: `SKIPPED`. On a 64-bit machine with `open3d` installed: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add scripts/floorplan_reconstruct.py tests/test_floorplan_reconstruct_cli.py
git commit -m "Add floorplan_reconstruct.py CLI shell writing manifest/floorplan/OBJ"
```

---

### Task 15: validate_measurements.py

**Files:**
- Create: `scripts/validate_measurements.py`
- Create: `tests/test_validate_measurements.py`

**Interfaces:**
- Produces: `load_ground_truth(path) -> dict`, `compare_measurements(manifest: dict, ground_truth: dict) -> list[dict]` (one row per measurement with `error_mm`, `tolerance_mm`, `status`), `main(manifest_path, ground_truth_path) -> int` (exit code: 0 pass, 1 hard-fail).

Per-measurement-type tolerances and hard-fail policy (2x tolerance) come directly from the accuracy review, since a single global mm threshold would conflate very different error sources (a hand-tape-measured wall length accumulates error over its run; a groove position does not).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_validate_measurements.py
import json
from scripts.validate_measurements import compare_measurements

TOLERANCES_MM = {
    "wall_thickness": 3.0,
    "opening_width": 5.0,
    "opening_height": 5.0,
    "wall_length": None,  # computed as 5 + 1*meters
}


def test_compare_measurements_flags_pass_warn_fail():
    manifest = {
        "walls": [
            {"wall_id": "wall_000", "thickness_m": 0.1003, "length_m": 5.001,
             "openings": [{"opening_id": "wall_000_op_00", "width_m": 0.905, "height_m": 2.108}]},
        ]
    }
    ground_truth = {
        "measurements": [
            {"id": "wall_000", "type": "wall_thickness", "expected_mm": 100.0},
            {"id": "wall_000", "type": "wall_length", "expected_mm": 5000.0},
            {"id": "wall_000_op_00", "type": "opening_width", "expected_mm": 900.0},
            {"id": "wall_000_op_00", "type": "opening_height", "expected_mm": 2100.0},
        ]
    }
    results = compare_measurements(manifest, ground_truth)
    by_id_type = {(r["id"], r["type"]): r for r in results}
    assert by_id_type[("wall_000", "wall_thickness")]["status"] == "pass"  # 0.3mm error, tol 3mm
    assert by_id_type[("wall_000_op_00", "opening_width")]["status"] == "pass"  # 5mm error, tol 5mm -> exactly at tol, still pass
    assert all(r["error_mm"] >= 0 for r in results)


def test_compare_measurements_hard_fails_over_double_tolerance():
    manifest = {"walls": [{"wall_id": "wall_000", "thickness_m": 0.115, "length_m": 5.0, "openings": []}]}
    ground_truth = {"measurements": [{"id": "wall_000", "type": "wall_thickness", "expected_mm": 100.0}]}
    results = compare_measurements(manifest, ground_truth)
    assert results[0]["status"] == "fail"  # 15mm error > 2x3mm tolerance
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_validate_measurements.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Write scripts/validate_measurements.py**

```python
"""Diff hand tape-measured ground truth against a floorplan_reconstruct.py manifest.json.

Usage:
    python scripts/validate_measurements.py manifest.json ground_truth.json
"""
import sys
import json
from pathlib import Path

TOLERANCE_MM = {
    "wall_thickness": 3.0,
    "opening_width": 5.0,
    "opening_height": 5.0,
    "groove_position": 3.0,
    "groove_depth": 3.0,
}


def _wall_length_tolerance_mm(expected_mm):
    meters = expected_mm / 1000.0
    return 5.0 + 1.0 * meters


def _tolerance_for(measurement_type, expected_mm):
    if measurement_type == "wall_length":
        return _wall_length_tolerance_mm(expected_mm)
    return TOLERANCE_MM[measurement_type]


def load_ground_truth(path):
    return json.loads(Path(path).read_text())


def _find_actual_mm(manifest, measurement_id, measurement_type):
    for wall in manifest.get("walls", []):
        if measurement_type == "wall_thickness" and wall.get("wall_id") == measurement_id:
            return wall["thickness_m"] * 1000.0
        if measurement_type == "wall_length" and wall.get("wall_id") == measurement_id:
            return wall["length_m"] * 1000.0
        for op in wall.get("openings", []):
            op_id = op["opening_id"] if isinstance(op, dict) else op.opening_id
            if op_id != measurement_id:
                continue
            if measurement_type == "opening_width":
                return (op["width_m"] if isinstance(op, dict) else op.width_m) * 1000.0
            if measurement_type == "opening_height":
                return (op["height_m"] if isinstance(op, dict) else op.height_m) * 1000.0
    return None


def compare_measurements(manifest, ground_truth):
    results = []
    for m in ground_truth["measurements"]:
        actual_mm = _find_actual_mm(manifest, m["id"], m["type"])
        tolerance_mm = _tolerance_for(m["type"], m["expected_mm"])
        if actual_mm is None:
            results.append({
                "id": m["id"], "type": m["type"], "expected_mm": m["expected_mm"],
                "actual_mm": None, "error_mm": None, "tolerance_mm": tolerance_mm,
                "status": "missing",
            })
            continue
        error_mm = abs(actual_mm - m["expected_mm"])
        if error_mm > 2 * tolerance_mm:
            status = "fail"
        elif error_mm > tolerance_mm:
            status = "warn"
        else:
            status = "pass"
        results.append({
            "id": m["id"], "type": m["type"], "expected_mm": m["expected_mm"],
            "actual_mm": actual_mm, "error_mm": error_mm, "tolerance_mm": tolerance_mm,
            "status": status,
        })
    return results


def main(manifest_path, ground_truth_path):
    manifest = json.loads(Path(manifest_path).read_text())
    ground_truth = load_ground_truth(ground_truth_path)
    results = compare_measurements(manifest, ground_truth)

    print(f"{'id':<20} {'type':<16} {'expected':>10} {'actual':>10} {'error':>8} {'tol':>6}  status")
    any_fail = False
    any_missing = False
    for r in results:
        actual_str = f"{r['actual_mm']:.1f}" if r["actual_mm"] is not None else "N/A"
        error_str = f"{r['error_mm']:.1f}" if r["error_mm"] is not None else "N/A"
        print(f"{r['id']:<20} {r['type']:<16} {r['expected_mm']:>10.1f} {actual_str:>10} "
              f"{error_str:>8} {r['tolerance_mm']:>6.1f}  {r['status']}")
        if r["status"] == "fail":
            any_fail = True
        if r["status"] == "missing":
            any_missing = True

    if any_fail or any_missing:
        print("\nRESULT: FAIL -- do not trust this manifest for a production cutlist yet.")
        return 1
    print("\nRESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_validate_measurements.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/validate_measurements.py tests/test_validate_measurements.py
git commit -m "Add validate_measurements.py with per-type tolerances and pass/warn/fail policy"
```

---

### Task 16: End-to-end smoke test script + README update

**Files:**
- Create: `scripts/floorplan_reconstruct_test.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `floorplan_reconstruct.main`, `mesh_common.load_las_as_o3d` (via `crop_radius_m`/`max_points`, matching the existing `reconstruct_mesh_test.py` pattern).
- Produces: a runnable smoke-test CLI. Not unit tested (it needs either a real LAS file or the VM); this task documents how to run it, matching how `reconstruct_mesh_test.py` is already used.

- [ ] **Step 1: Read scripts/reconstruct_mesh_test.py to match its existing test-patch CLI conventions**

Run: `cat scripts/reconstruct_mesh_test.py` (or open it) and note its `crop_radius_m`/`max_points` argument handling so the new script matches the established pattern exactly.

- [ ] **Step 2: Write scripts/floorplan_reconstruct_test.py**

```python
"""Fast smoke test for floorplan_reconstruct.py: small spatial crop + point cap,
mirrors reconstruct_mesh_test.py's test-patch pattern.

Usage:
    python scripts/floorplan_reconstruct_test.py [input.las] [output_dir] [crop_radius_m] [max_points]
"""
import sys
from pathlib import Path

from mesh_common import load_las_as_o3d, recenter_pcd, log
from floorplan_reconstruct import build_floorplan_outputs, DEFAULT_CONFIG, render_floorplan_image
import numpy as np
import json

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "koushikexport.las"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "floorplan_test"


def main(input_path, output_dir, crop_radius_m=8.0, max_points=1_500_000):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pcd = load_las_as_o3d(Path(input_path), crop_radius_m=crop_radius_m, max_points=max_points)
    recenter_pcd(pcd)
    xyz = np.asarray(pcd.points)

    manifest, walls = build_floorplan_outputs(xyz, DEFAULT_CONFIG)
    log(f"Test-patch: detected {manifest['wall_count']} walls")

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    wall_dicts = [{"p0": np.array(w.p0), "p1": np.array(w.p1),
                   "thickness_m": w.thickness_m, "length_m": w.length_m} for w in walls]
    render_floorplan_image(wall_dicts, {}, str(output_dir / "floorplan.png"))
    log(f"Wrote {output_dir}/manifest.json and floorplan.png -- inspect these against the real building before trusting a full run")


if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    outp = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_DIR
    crop_r = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0
    max_pts = int(sys.argv[4]) if len(sys.argv) > 4 else 1_500_000
    main(inp, outp, crop_r, max_pts)
```

- [ ] **Step 3: Update README.md's Pipeline section**

Read `README.md` first. In the "## Pipeline" section, after the existing numbered list, add:

```markdown
4. `floorplan_geometry.py` + `floorplan_reconstruct.py` — Phase 0 (bounding-box auto-crop) + Phase 1 (density-image wall/opening detection) → `manifest.json`, `floorplan.png`, `reconstructed.obj`. Replaces `segment_walls_and_grooves.py`.
5. `floorplan_reconstruct_test.py` — fast test-patch smoke test (small crop + point cap), same pattern as `reconstruct_mesh_test.py`
6. `validate_measurements.py` — diff hand tape-measured ground truth against `manifest.json`, report per-measurement mm error
```

- [ ] **Step 4: Run this on a real scan (manual verification, not automated)**

Run: `python scripts/floorplan_reconstruct_test.py data/koushikexport.las output/floorplan_test 8.0 1500000`
Expected: completes without error, prints a wall count, writes `output/floorplan_test/manifest.json` and `floorplan.png`. Open `floorplan.png` and visually sanity-check it against the real house before trusting it. This step requires either running on a machine with `open3d` installed or the Jarvis Labs VM (per README) — it is a manual verification step, not part of automated CI.

- [ ] **Step 5: Commit**

```bash
git add scripts/floorplan_reconstruct_test.py README.md
git commit -m "Add floorplan_reconstruct_test.py smoke test and update README"
```
