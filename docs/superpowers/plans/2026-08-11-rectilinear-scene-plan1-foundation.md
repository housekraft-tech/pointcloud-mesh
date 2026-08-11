# Rectilinear Scene Rebuild — Plan 1: Foundation & Patch Extraction

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a SLAM LAS scan into a set of independently-fitted planar patches with a patch graph, serialised into a schema-validated scene document, via a CLI — proving that a 75 mm rectangular step survives extraction as its own patch instead of being absorbed into its parent wall.

**Architecture:** Pure-numpy geometry core under `src/rscene/core/`, with LAS I/O behind a thin adapter in `src/rscene/io/`. Patches are found by normal-based region growing with a connectivity requirement, each plane fitted only to its own points — never snapped to a global frame. Correctness is proven against synthetic scenes built from known boxes, so recovered dimensions can be asserted against ground truth in CI.

**Tech Stack:** Python 3.13, numpy, scipy (`cKDTree`), laspy, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-08-11-rectilinear-scene-rebuild-design.md`

## Global Constraints

- **Never snap one surface to another.** Every plane is fitted only to its own supporting points. No global Manhattan grid, no cross-surface merging, no endpoint snapping. Violating this reintroduces the exact failure this rebuild exists to fix.
- **Coplanarity is recorded, never applied.** Patches that share a plane are grouped into a reported class; their individual fitted planes are left untouched.
- **`src/rscene/core/` imports only numpy and scipy.** No laspy, trimesh, open3d, shapely or cv2. The core must stay runnable in an environment where those are absent.
- **Determinism.** Same input plus same config produces byte-identical output. Every iteration order is explicitly sorted; every RNG is seeded and passed in, never global.
- **No silent drops.** Points that end in no patch are counted and reported, never discarded silently.
- **Manhattan is measured, never enforced.** Frame orientation is reported as data; no geometry is rotated to match it.
- Tolerance defaults, exact values from the spec: `tau_fit = 0.003` m, `tau_feature = 0.008` m.
- Python floor: 3.13. `open3d` and `trimesh` are NOT dependencies of this plan.

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, dependency pins, pytest config |
| `src/rscene/__init__.py` | Package marker, version |
| `src/rscene/config.py` | `DEFAULT_CONFIG` — every threshold, one commented line each |
| `src/rscene/core/prim.py` | `Box` rectilinear primitive and its surface sampler |
| `src/rscene/core/points.py` | `PointSet` container, noise model |
| `src/rscene/core/fitting.py` | Plane fitting, canonical normal orientation, plane basis |
| `src/rscene/core/normals.py` | kNN PCA normal and curvature estimation |
| `src/rscene/core/patches.py` | `Patch` and region-growing extraction — the heart |
| `src/rscene/core/graph.py` | Coplanarity classes, adjacency, plane intersection lines |
| `src/rscene/core/level.py` | Frame estimation (gravity Z, reported XY rotation, storey levels) |
| `src/rscene/core/scene.py` | `Measurement`, `Provenance`, `Frame`, `Scene`, JSON round-trip |
| `src/rscene/io/las.py` | LAS load/save adapter |
| `src/rscene/cli.py` | `rscene patches <input> <outdir>` |
| `tests/golden/apartment.py` | Golden synthetic bare-shell room with known truth |

---

### Task 1: Project scaffold and the `Box` primitive

Everything downstream is tested against geometry built from boxes, so this comes first. Scaffolding is folded in here because this is the first task that needs a working environment.

**Files:**
- Create: `pyproject.toml`
- Modify: `pytest.ini` (register the `slow` marker)
- Create: `src/rscene/__init__.py`
- Create: `src/rscene/core/__init__.py`
- Create: `src/rscene/core/prim.py`
- Create: `tests/rscene/__init__.py`
- Create: `tests/rscene/test_prim.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `FACE_KEYS: tuple[str, ...]` = `("x-", "x+", "y-", "y+", "z-", "z+")`
  - `Box(name: str, lo: tuple[float,float,float], hi: tuple[float,float,float])`
  - `Box.sample_surface(spacing_m: float, faces: Sequence[str] = FACE_KEYS) -> np.ndarray` shape `(N, 3)` float64

- [ ] **Step 1: Create the package scaffold**

Create `pyproject.toml`:

```toml
[project]
name = "rscene"
version = "0.1.0"
description = "Rectilinear scene reconstruction from indoor LiDAR point clouds"
requires-python = ">=3.13"
dependencies = [
    "numpy>=2.1",
    "scipy>=1.14",
]

[project.optional-dependencies]
io = ["laspy>=2.5", "lazrs>=0.5"]
dev = ["pytest>=8.0"]

[project.scripts]
rscene = "rscene.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/rscene"]

```

Modify the existing `pytest.ini` at the repo root to register the `slow` marker. `pytest.ini` takes precedence over `pyproject.toml`'s pytest section, so this is where the setting must live. Do not delete `pytest.ini` — the old `scripts/` tests still use it.

```ini
[pytest]
testpaths = tests
python_files = test_*.py
markers =
    slow: full-resolution golden-scene runs
```

Create `src/rscene/__init__.py`:

```python
"""Rectilinear scene reconstruction from indoor LiDAR point clouds."""

__version__ = "0.1.0"
```

Create empty `src/rscene/core/__init__.py` and `tests/rscene/__init__.py`.

- [ ] **Step 2: Create the environment**

Run:

```bash
uv venv --python 3.13
uv pip install -e ".[io,dev]"
```

Expected: installs numpy, scipy, laspy, lazrs, pytest without error.

Because `pytest.ini` sets `testpaths = tests` for the whole repo including the old `scripts/` tests, always invoke pytest with an explicit path, as every later step does.

- [ ] **Step 3: Write the failing test**

Create `tests/rscene/test_prim.py`:

```python
import numpy as np

from rscene.core.prim import FACE_KEYS, Box


def test_sampled_points_lie_on_the_box_surface():
    box = Box(name="unit", lo=(0.0, 0.0, 0.0), hi=(1.0, 2.0, 3.0))
    pts = box.sample_surface(spacing_m=0.05)

    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([1.0, 2.0, 3.0])
    # every point sits inside the box, and touches at least one face
    assert np.all(pts >= lo - 1e-9)
    assert np.all(pts <= hi + 1e-9)
    on_a_face = np.isclose(pts, lo).any(axis=1) | np.isclose(pts, hi).any(axis=1)
    assert on_a_face.all()


def test_single_face_sampling_is_planar_and_correctly_placed():
    box = Box(name="unit", lo=(0.0, 0.0, 0.0), hi=(1.0, 2.0, 3.0))
    pts = box.sample_surface(spacing_m=0.1, faces=("x+",))

    assert np.allclose(pts[:, 0], 1.0)
    assert pts[:, 1].min() == 0.0 and np.isclose(pts[:, 1].max(), 2.0)
    assert pts[:, 2].min() == 0.0 and np.isclose(pts[:, 2].max(), 3.0)


def test_sampling_is_deterministic():
    box = Box(name="unit", lo=(0.0, 0.0, 0.0), hi=(1.0, 1.0, 1.0))
    assert np.array_equal(box.sample_surface(0.05), box.sample_surface(0.05))


def test_all_six_faces_are_produced():
    box = Box(name="unit", lo=(0.0, 0.0, 0.0), hi=(1.0, 1.0, 1.0))
    per_face = {k: box.sample_surface(0.25, faces=(k,)) for k in FACE_KEYS}
    assert len(per_face) == 6
    assert all(len(v) > 0 for v in per_face.values())
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/rscene/test_prim.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rscene.core.prim'`

- [ ] **Step 5: Implement `prim.py`**

Create `src/rscene/core/prim.py`:

```python
"""Rectilinear box primitive and its surface sampler.

Boxes are the vocabulary of a bare-shell concrete building: formwork produces
flat faces meeting at sharp edges. Synthetic scenes are built from boxes so
tests can assert recovered dimensions against exact ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

FACE_KEYS = ("x-", "x+", "y-", "y+", "z-", "z+")

_AXIS = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class Box:
    """An axis-aligned rectilinear box, named for traceability into tests."""

    name: str
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]

    def sample_surface(
        self, spacing_m: float, faces: Sequence[str] = FACE_KEYS
    ) -> np.ndarray:
        """Sample a regular grid on the requested faces.

        Returns (N, 3) float64. Grid spacing is adjusted per face so samples
        land exactly on the face boundary, which keeps extents exact for the
        dimension assertions in the golden tests. Deterministic: no RNG here,
        noise is added separately by points.add_gaussian_noise.
        """
        if spacing_m <= 0:
            raise ValueError(f"spacing_m must be positive, got {spacing_m}")

        lo = np.asarray(self.lo, dtype=np.float64)
        hi = np.asarray(self.hi, dtype=np.float64)
        if np.any(hi < lo):
            raise ValueError(f"box {self.name!r} has hi < lo: {self.lo} {self.hi}")

        chunks = []
        for key in faces:
            if key not in FACE_KEYS:
                raise ValueError(f"unknown face key {key!r}, expected one of {FACE_KEYS}")
            axis = _AXIS[key[0]]
            at_hi = key[1] == "+"
            u_ax, v_ax = [a for a in (0, 1, 2) if a != axis]

            nu = max(2, int(round((hi[u_ax] - lo[u_ax]) / spacing_m)) + 1)
            nv = max(2, int(round((hi[v_ax] - lo[v_ax]) / spacing_m)) + 1)
            us = np.linspace(lo[u_ax], hi[u_ax], nu)
            vs = np.linspace(lo[v_ax], hi[v_ax], nv)
            uu, vv = np.meshgrid(us, vs, indexing="ij")

            pts = np.empty((uu.size, 3), dtype=np.float64)
            pts[:, axis] = hi[axis] if at_hi else lo[axis]
            pts[:, u_ax] = uu.ravel()
            pts[:, v_ax] = vv.ravel()
            chunks.append(pts)

        if not chunks:
            return np.zeros((0, 3), dtype=np.float64)
        return np.concatenate(chunks, axis=0)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/rscene/test_prim.py -v`
Expected: 4 passed

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/rscene tests/rscene
git commit -m "feat(core): rectilinear Box primitive and surface sampler"
```

---

### Task 2: `PointSet` container and noise model

**Files:**
- Create: `src/rscene/core/points.py`
- Create: `tests/rscene/test_points.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `PointSet(xyz, gps_time=None, intensity=None, rgb=None)` with `.n: int` and `.subset(mask) -> PointSet`
  - `add_gaussian_noise(xyz: np.ndarray, sigma_m: float, rng: np.random.Generator) -> np.ndarray`

- [ ] **Step 1: Write the failing test**

Create `tests/rscene/test_points.py`:

```python
import numpy as np
import pytest

from rscene.core.points import PointSet, add_gaussian_noise


def _sample_set(n=10):
    return PointSet(
        xyz=np.arange(3 * n, dtype=np.float64).reshape(n, 3),
        gps_time=np.arange(n, dtype=np.float64),
        intensity=np.arange(n, dtype=np.uint16),
        rgb=np.zeros((n, 3), dtype=np.uint8),
    )


def test_n_reports_point_count():
    assert _sample_set(7).n == 7


def test_subset_keeps_every_attribute_row_aligned():
    ps = _sample_set(10)
    keep = np.array([0, 3, 9])
    sub = ps.subset(keep)

    assert sub.n == 3
    assert np.array_equal(sub.xyz, ps.xyz[keep])
    assert np.array_equal(sub.gps_time, ps.gps_time[keep])
    assert np.array_equal(sub.intensity, ps.intensity[keep])
    assert np.array_equal(sub.rgb, ps.rgb[keep])


def test_subset_tolerates_absent_optional_attributes():
    ps = PointSet(xyz=np.zeros((5, 3)))
    sub = ps.subset(np.array([True, False, True, False, True]))
    assert sub.n == 3
    assert sub.gps_time is None and sub.intensity is None and sub.rgb is None


def test_mismatched_attribute_length_is_rejected():
    with pytest.raises(ValueError, match="row count"):
        PointSet(xyz=np.zeros((5, 3)), gps_time=np.zeros(4))


def test_noise_has_the_requested_sigma_and_is_reproducible():
    xyz = np.zeros((200_000, 3))
    a = add_gaussian_noise(xyz, sigma_m=0.002, rng=np.random.default_rng(0))
    b = add_gaussian_noise(xyz, sigma_m=0.002, rng=np.random.default_rng(0))

    assert np.array_equal(a, b)                      # same seed, same result
    assert abs(a.std() - 0.002) < 0.0001
    assert a is not xyz                              # input not mutated
    assert np.array_equal(xyz, np.zeros((200_000, 3)))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/rscene/test_points.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rscene.core.points'`

- [ ] **Step 3: Implement `points.py`**

Create `src/rscene/core/points.py`:

```python
"""Point container and noise model.

PointSet keeps every per-point attribute row-aligned with xyz so that any
subsetting operation anywhere in the pipeline cannot silently desynchronise
intensity or timestamps from geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PointSet:
    """A point cloud plus its optional per-point attributes, all row-aligned."""

    xyz: np.ndarray                          # (N, 3) float64, metres
    gps_time: Optional[np.ndarray] = None    # (N,) float64
    intensity: Optional[np.ndarray] = None   # (N,)
    rgb: Optional[np.ndarray] = None         # (N, 3) uint8

    def __post_init__(self) -> None:
        self.xyz = np.asarray(self.xyz, dtype=np.float64)
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3:
            raise ValueError(f"xyz must be (N, 3), got {self.xyz.shape}")
        for name in ("gps_time", "intensity", "rgb"):
            attr = getattr(self, name)
            if attr is not None and len(attr) != len(self.xyz):
                raise ValueError(
                    f"{name} row count {len(attr)} != xyz row count {len(self.xyz)}"
                )

    @property
    def n(self) -> int:
        return int(self.xyz.shape[0])

    def subset(self, mask) -> "PointSet":
        """Return a new PointSet keeping rows selected by a boolean or index mask."""
        mask = np.asarray(mask)
        return PointSet(
            xyz=self.xyz[mask],
            gps_time=None if self.gps_time is None else self.gps_time[mask],
            intensity=None if self.intensity is None else self.intensity[mask],
            rgb=None if self.rgb is None else self.rgb[mask],
        )


def add_gaussian_noise(
    xyz: np.ndarray, sigma_m: float, rng: np.random.Generator
) -> np.ndarray:
    """Return a copy of xyz with isotropic Gaussian noise of the given sigma.

    The generator is passed in rather than seeded internally so that callers
    control reproducibility. Never mutates the input.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if sigma_m < 0:
        raise ValueError(f"sigma_m must be non-negative, got {sigma_m}")
    if sigma_m == 0:
        return xyz.copy()
    return xyz + rng.normal(0.0, sigma_m, size=xyz.shape)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/rscene/test_points.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/rscene/core/points.py tests/rscene/test_points.py
git commit -m "feat(core): PointSet container with row-aligned attributes and noise model"
```

---

### Task 3: Plane fitting with canonical orientation

Canonical orientation is what makes two parallel patches comparable. Without it, a wall face and the step in front of it may get opposite normal signs from PCA, and the offset between them becomes meaningless.

**Files:**
- Create: `src/rscene/core/fitting.py`
- Create: `tests/rscene/test_fitting.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `canonical_normal(n: np.ndarray) -> np.ndarray` — flips sign so the largest-magnitude component is positive
  - `fit_plane(xyz: np.ndarray) -> tuple[np.ndarray, float]` — returns `(normal, d)` with `normal @ x + d == 0`, normal canonically oriented
  - `plane_distance(xyz: np.ndarray, normal: np.ndarray, d: float) -> np.ndarray` — signed distances, shape `(N,)`
  - `plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]` — deterministic orthonormal `(u, v)` spanning the plane

- [ ] **Step 1: Write the failing test**

Create `tests/rscene/test_fitting.py`:

```python
import numpy as np
import pytest

from rscene.core.fitting import canonical_normal, fit_plane, plane_basis, plane_distance


def test_canonical_normal_makes_dominant_component_positive():
    assert np.allclose(canonical_normal(np.array([0.0, 0.0, -1.0])), [0.0, 0.0, 1.0])
    assert np.allclose(canonical_normal(np.array([-0.9, 0.1, 0.0])), [0.9, -0.1, 0.0])


def test_opposite_normals_canonicalise_to_the_same_direction():
    n = np.array([0.6, -0.8, 0.0])
    assert np.allclose(canonical_normal(n), canonical_normal(-n))


def test_fit_plane_recovers_a_known_plane():
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(0, 4, 5000), rng.uniform(0, 3, 5000),
                           np.full(5000, 2.75)])
    normal, d = fit_plane(pts)

    assert np.allclose(np.abs(normal), [0.0, 0.0, 1.0], atol=1e-9)
    # plane is z = 2.75  ->  1*z - 2.75 = 0
    assert abs(d + 2.75) < 1e-9


def test_fit_plane_recovers_offset_between_two_parallel_planes():
    """The core property the whole rebuild depends on: a 75 mm step is 75 mm."""
    rng = np.random.default_rng(1)
    face = np.column_stack([np.zeros(4000), rng.uniform(0, 4, 4000),
                            rng.uniform(0, 2.75, 4000)])
    step = np.column_stack([np.full(4000, 0.075), rng.uniform(2.8, 3.15, 4000),
                            rng.uniform(0, 2.75, 4000)])

    n1, d1 = fit_plane(face)
    n2, d2 = fit_plane(step)

    assert np.allclose(n1, n2, atol=1e-9)          # canonical -> same direction
    assert abs(abs(d1 - d2) - 0.075) < 1e-6


def test_plane_distance_is_signed_and_scaled_in_metres():
    normal, d = np.array([0.0, 0.0, 1.0]), -2.0
    pts = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.5], [0.0, 0.0, 1.5]])
    assert np.allclose(plane_distance(pts, normal, d), [0.0, 0.5, -0.5])


def test_plane_basis_is_orthonormal_and_deterministic():
    n = canonical_normal(np.array([0.0, 0.0, 1.0]))
    u, v = plane_basis(n)

    assert abs(u @ v) < 1e-12
    assert abs(u @ n) < 1e-12 and abs(v @ n) < 1e-12
    assert abs(np.linalg.norm(u) - 1) < 1e-12 and abs(np.linalg.norm(v) - 1) < 1e-12
    assert np.array_equal(u, plane_basis(n)[0])


def test_fit_plane_rejects_degenerate_input():
    with pytest.raises(ValueError, match="at least 3"):
        fit_plane(np.zeros((2, 3)))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/rscene/test_fitting.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rscene.core.fitting'`

- [ ] **Step 3: Implement `fitting.py`**

Create `src/rscene/core/fitting.py`:

```python
"""Plane fitting with a canonical normal orientation.

Every plane in the pipeline is fitted to its own points and to nothing else.
Canonical orientation exists so that two parallel patches -- a wall face and
the 75 mm step in front of it -- yield comparable plane offsets regardless of
which way PCA happened to point their normals.
"""
from __future__ import annotations

import numpy as np


def canonical_normal(n: np.ndarray) -> np.ndarray:
    """Flip a normal so its largest-magnitude component is positive.

    Ties are broken toward the lowest axis index, which keeps the choice
    deterministic for normals like (0.5, -0.5, 0).
    """
    n = np.asarray(n, dtype=np.float64)
    dominant = int(np.argmax(np.abs(n)))
    return -n if n[dominant] < 0 else n.copy()


def fit_plane(xyz: np.ndarray) -> tuple[np.ndarray, float]:
    """Total-least-squares plane through the points.

    Returns (normal, d) satisfying normal @ x + d == 0, with a unit,
    canonically-oriented normal.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.shape[0] < 3:
        raise ValueError(f"need at least 3 points to fit a plane, got {xyz.shape[0]}")

    centroid = xyz.mean(axis=0)
    centred = xyz - centroid
    # smallest singular vector of the centred cloud is the plane normal
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    normal = canonical_normal(vt[-1])
    normal /= np.linalg.norm(normal)
    d = float(-normal @ centroid)
    return normal, d


def plane_distance(xyz: np.ndarray, normal: np.ndarray, d: float) -> np.ndarray:
    """Signed perpendicular distance of each point to the plane, in metres."""
    xyz = np.asarray(xyz, dtype=np.float64)
    return xyz @ np.asarray(normal, dtype=np.float64) + d


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A deterministic orthonormal basis (u, v) spanning the plane.

    u is built from the world axis least aligned with the normal, so the basis
    is stable for a given normal and never degenerate.
    """
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)

    seed = np.zeros(3)
    seed[int(np.argmin(np.abs(normal)))] = 1.0

    u = np.cross(normal, seed)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)
    return u, v
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/rscene/test_fitting.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/rscene/core/fitting.py tests/rscene/test_fitting.py
git commit -m "feat(core): plane fitting with canonical normal orientation"
```

---

### Task 4: Normal and curvature estimation

**Files:**
- Create: `src/rscene/core/normals.py`
- Create: `tests/rscene/test_normals.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `estimate_normals(xyz: np.ndarray, k: int = 24) -> tuple[np.ndarray, np.ndarray]` — returns `(normals (N,3) canonically oriented, curvature (N,) in [0, 1/3])`

Curvature is the ratio of the smallest eigenvalue to the sum of eigenvalues: near zero on a flat surface, higher at an edge. Region growing seeds from the flattest points first, which keeps patch discovery deterministic and starts each patch in the middle of a face rather than on a corner.

- [ ] **Step 1: Write the failing test**

Create `tests/rscene/test_normals.py`:

```python
import numpy as np

from rscene.core.normals import estimate_normals
from rscene.core.prim import Box


def test_normals_on_a_flat_face_point_along_the_face_normal():
    pts = Box("f", (0, 0, 0), (2, 2, 0)).sample_surface(0.02, faces=("z+",))
    normals, curvature = estimate_normals(pts, k=16)

    assert np.abs(normals[:, 2]).min() > 0.99
    assert curvature.max() < 0.01


def test_curvature_is_higher_at_an_edge_than_on_a_face():
    box = Box("b", (0, 0, 0), (1, 1, 1))
    pts = box.sample_surface(0.02, faces=("z+", "x+"))
    normals, curvature = estimate_normals(pts, k=16)

    near_edge = np.abs(pts[:, 0] - 1.0) < 0.01
    on_face = pts[:, 0] < 0.5
    assert curvature[near_edge].mean() > curvature[on_face].mean() * 5


def test_normals_are_canonically_oriented_and_unit_length():
    pts = Box("f", (0, 0, 0), (1, 1, 0)).sample_surface(0.05, faces=("z+",))
    normals, _ = estimate_normals(pts, k=12)

    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)
    # canonical: dominant component positive, so all z-normals point +z
    assert (normals[:, 2] > 0).all()


def test_estimation_is_deterministic():
    pts = Box("f", (0, 0, 0), (1, 1, 0)).sample_surface(0.05, faces=("z+",))
    a, ca = estimate_normals(pts, k=12)
    b, cb = estimate_normals(pts, k=12)
    assert np.array_equal(a, b) and np.array_equal(ca, cb)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/rscene/test_normals.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rscene.core.normals'`

- [ ] **Step 3: Implement `normals.py`**

Create `src/rscene/core/normals.py`:

```python
"""kNN PCA normal and curvature estimation.

Normals are canonically oriented (see fitting.canonical_normal) rather than
consistently oriented toward a viewpoint: patch growing compares normals with
abs(dot), so sign carries no information and a canonical choice keeps results
reproducible.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def estimate_normals(xyz: np.ndarray, k: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Estimate per-point normals and surface variation from k nearest neighbours.

    Returns (normals, curvature). curvature is lambda_0 / sum(lambda), which is
    ~0 on a flat surface and rises toward 1/3 at a corner.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.shape[0] < k:
        raise ValueError(f"need at least k={k} points, got {xyz.shape[0]}")

    tree = cKDTree(xyz)
    _, idx = tree.query(xyz, k=k, workers=-1)

    nbr = xyz[idx]                                       # (N, k, 3)
    centred = nbr - nbr.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centred, centred) / k

    # eigh returns ascending eigenvalues; columns of v are the eigenvectors
    w, v = np.linalg.eigh(cov)
    normals = v[:, :, 0]
    curvature = w[:, 0] / np.clip(w.sum(axis=1), 1e-18, None)

    # canonical orientation, vectorised: flip rows whose dominant component < 0
    dominant = np.argmax(np.abs(normals), axis=1)
    sign = np.sign(normals[np.arange(len(normals)), dominant])
    sign[sign == 0] = 1.0
    normals = normals * sign[:, None]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    return normals, curvature
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/rscene/test_normals.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/rscene/core/normals.py tests/rscene/test_normals.py
git commit -m "feat(core): kNN PCA normal and curvature estimation"
```

---

### Task 5: Config defaults

**Files:**
- Create: `src/rscene/config.py`
- Create: `tests/rscene/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `DEFAULT_CONFIG: dict`, `merged_config(overrides: dict | None) -> dict`

- [ ] **Step 1: Write the failing test**

Create `tests/rscene/test_config.py`:

```python
import pytest

from rscene.config import DEFAULT_CONFIG, merged_config


def test_spec_tolerances_have_their_specified_values():
    assert DEFAULT_CONFIG["tau_fit_m"] == 0.003
    assert DEFAULT_CONFIG["tau_feature_m"] == 0.008


def test_merged_config_overrides_without_mutating_the_default():
    cfg = merged_config({"tau_fit_m": 0.005})
    assert cfg["tau_fit_m"] == 0.005
    assert cfg["min_patch_points"] == DEFAULT_CONFIG["min_patch_points"]
    assert DEFAULT_CONFIG["tau_fit_m"] == 0.003


def test_unknown_config_keys_are_rejected():
    with pytest.raises(KeyError, match="unknown config key"):
        merged_config({"tau_fitt_m": 0.005})


def test_merged_config_of_none_equals_the_default():
    assert merged_config(None) == DEFAULT_CONFIG
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/rscene/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rscene.config'`

- [ ] **Step 3: Implement `config.py`**

Create `src/rscene/config.py`:

```python
"""Every threshold in the pipeline, in one place, one commented line each.

Values marked "calibration pending" are starting points to be tuned against
the golden synthetic scenes and the reference scans. They are not claims of
achieved accuracy.
"""
from __future__ import annotations

from copy import deepcopy

DEFAULT_CONFIG: dict = {
    # --- reproducibility ---
    "seed": 0,                     # seeds every RNG in the pipeline
    # --- normals ---
    "normal_k": 24,                # neighbours used for PCA normal estimation
    # --- patch extraction (spec tolerances) ---
    "tau_fit_m": 0.003,            # plane inlier distance
    "tau_feature_m": 0.008,        # min depth to count as a feature, not roughness
    "patch_angle_tol_deg": 8.0,    # max normal deviation when growing a patch
    "patch_connect_radius_m": 0.05,  # CALIBRATION PENDING -- neighbour radius enforcing patch connectivity; must stay below the narrowest feature width
    "min_patch_points": 100,       # CALIBRATION PENDING -- smallest patch kept; floor set by switch-box sample count
    "refit_interval": 200,         # points added between plane refits while growing
    # --- coplanarity (recorded, never applied) ---
    "coplanar_dist_tol_m": 0.005,  # max plane-offset difference within a class
    "coplanar_angle_tol_deg": 2.0,  # max normal deviation within a class
    # --- adjacency ---
    "adjacency_radius_m": 0.05,    # max gap between patches counted as adjacent
    # --- frame (measured, never enforced) ---
    "floor_normal_tol_deg": 15.0,  # max tilt from world Z for a floor/ceiling patch
}


def merged_config(overrides: dict | None = None) -> dict:
    """Return DEFAULT_CONFIG updated with overrides, rejecting unknown keys."""
    cfg = deepcopy(DEFAULT_CONFIG)
    if not overrides:
        return cfg
    for key, value in overrides.items():
        if key not in cfg:
            raise KeyError(f"unknown config key {key!r}")
        cfg[key] = value
    return cfg
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/rscene/test_config.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/rscene/config.py tests/rscene/test_config.py
git commit -m "feat: central config with spec tolerances"
```

---

### Task 6: Region-growing patch extraction

**This is the heart of the rebuild.** Whole-cloud RANSAC fuses a 75 mm step into its parent wall because one dominant plane wins and swallows its neighbours. Region growing with a connectivity requirement makes the step its own patch by construction. The test in Step 1 is the test that encodes the entire purpose of this rewrite — if it passes, the primary failure mode is fixed.

**Files:**
- Create: `src/rscene/core/patches.py`
- Create: `tests/rscene/test_patches.py`

**Interfaces:**
- Consumes: `fit_plane`, `plane_basis`, `plane_distance` (Task 3); `estimate_normals` (Task 4); `DEFAULT_CONFIG` (Task 5)
- Produces:
  - `Patch(patch_id: int, normal: np.ndarray, d: float, point_idx: np.ndarray, n_points: int, p95_residual_m: float, centroid: np.ndarray, u_range: tuple[float,float], v_range: tuple[float,float])`
  - `extract_patches(xyz, normals, curvature, config) -> tuple[list[Patch], np.ndarray]` — the array is per-point labels, `-1` for unassigned
  - `unassigned_count(labels: np.ndarray) -> int`

**Performance note:** growth calls `query_ball_point` once per point, which is a Python-level loop. On the ~700k-point golden room expect tens of seconds — acceptable for a `slow`-marked test. If it exceeds a couple of minutes, the sanctioned fix is to replace the radius query with a precomputed fixed-k neighbour array (`tree.query(xyz, k=32)`), which is one vectorised call. Do **not** widen `patch_connect_radius_m` to speed it up — that erases the features this plan exists to preserve.

- [ ] **Step 1: Write the failing test**

Create `tests/rscene/test_patches.py`:

```python
import numpy as np

from rscene.config import merged_config
from rscene.core.normals import estimate_normals
from rscene.core.patches import extract_patches
from rscene.core.points import add_gaussian_noise
from rscene.core.prim import Box


def _wall_with_step(spacing=0.008, noise_m=0.001):
    """A wall face at x=0 with a 75 mm rectangular extrusion standing on it.

    The wall's own face is sampled only where the extrusion does not cover it,
    exactly as a scanner would see it.
    """
    below = Box("wall_lo", (0.0, 0.0, 0.0), (0.0, 2.80, 2.75))
    above = Box("wall_hi", (0.0, 3.15, 0.0), (0.0, 4.00, 2.75))
    step_face = Box("step_face", (0.075, 2.80, 0.0), (0.075, 3.15, 2.75))
    step_side_a = Box("step_a", (0.0, 2.80, 0.0), (0.075, 2.80, 2.75))
    step_side_b = Box("step_b", (0.0, 3.15, 0.0), (0.075, 3.15, 2.75))

    pts = np.concatenate([
        below.sample_surface(spacing, faces=("x+",)),
        above.sample_surface(spacing, faces=("x+",)),
        step_face.sample_surface(spacing, faces=("x+",)),
        step_side_a.sample_surface(spacing, faces=("y-",)),
        step_side_b.sample_surface(spacing, faces=("y+",)),
    ])
    return add_gaussian_noise(pts, noise_m, np.random.default_rng(0))


def test_a_75mm_step_survives_as_its_own_patch():
    """The defining test of the rebuild: the step must NOT be absorbed."""
    xyz = _wall_with_step()
    normals, curvature = estimate_normals(xyz, k=24)
    patches, labels = extract_patches(xyz, normals, curvature, merged_config())

    x_facing = [p for p in patches if abs(p.normal[0]) > 0.99]
    assert len(x_facing) >= 2, "the step was absorbed into the wall face"

    offsets = sorted(abs(p.d) for p in x_facing)
    assert abs((offsets[-1] - offsets[0]) - 0.075) < 0.003


def test_the_step_side_faces_are_found_as_separate_patches():
    xyz = _wall_with_step()
    normals, curvature = estimate_normals(xyz, k=24)
    patches, _ = extract_patches(xyz, normals, curvature, merged_config())

    y_facing = [p for p in patches if abs(p.normal[1]) > 0.99]
    assert len(y_facing) >= 2

    offsets = sorted(abs(p.d) for p in y_facing)
    assert abs((offsets[-1] - offsets[0]) - 0.35) < 0.003   # step is 350 mm wide


def test_no_surface_is_snapped_to_another():
    """Two nearly-parallel faces stay distinct rather than collapsing to one."""
    a = Box("a", (0.0, 0.0, 0.0), (0.0, 2.0, 2.0)).sample_surface(0.008, faces=("x+",))
    b = Box("b", (0.02, 2.5, 0.0), (0.02, 4.5, 2.0)).sample_surface(0.008, faces=("x+",))
    xyz = np.concatenate([a, b])

    normals, curvature = estimate_normals(xyz, k=24)
    patches, _ = extract_patches(xyz, normals, curvature, merged_config())

    x_facing = [p for p in patches if abs(p.normal[0]) > 0.99]
    assert len(x_facing) == 2
    assert abs(abs(x_facing[0].d - x_facing[1].d) - 0.02) < 0.002


def test_labels_cover_every_point_or_mark_it_unassigned():
    xyz = _wall_with_step()
    normals, curvature = estimate_normals(xyz, k=24)
    patches, labels = extract_patches(xyz, normals, curvature, merged_config())

    assert labels.shape == (len(xyz),)
    for p in patches:
        assert np.array_equal(np.sort(p.point_idx), np.sort(np.flatnonzero(labels == p.patch_id)))
    assert set(np.unique(labels)) <= {-1} | {p.patch_id for p in patches}


def test_patch_records_its_own_residual_and_extent():
    xyz = _wall_with_step()
    normals, curvature = estimate_normals(xyz, k=24)
    patches, _ = extract_patches(xyz, normals, curvature, merged_config())

    big = max(patches, key=lambda p: p.n_points)
    assert big.p95_residual_m < 0.004          # ~1 mm noise, 3 mm tolerance
    assert big.u_range[1] > big.u_range[0]
    assert big.v_range[1] > big.v_range[0]


def test_extraction_is_deterministic():
    xyz = _wall_with_step()
    normals, curvature = estimate_normals(xyz, k=24)
    a, la = extract_patches(xyz, normals, curvature, merged_config())
    b, lb = extract_patches(xyz, normals, curvature, merged_config())

    assert np.array_equal(la, lb)
    assert [p.d for p in a] == [p.d for p in b]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/rscene/test_patches.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rscene.core.patches'`

- [ ] **Step 3: Implement `patches.py`**

Create `src/rscene/core/patches.py`:

```python
"""Region-growing planar patch extraction.

Whole-cloud RANSAC is what fuses a 75 mm step into its parent wall: one
dominant plane wins and swallows its neighbours. Growing regions from
low-curvature seeds, gated on BOTH normal agreement and spatial connectivity,
makes each formwork face its own patch by construction.

Each patch's plane is fitted only to its own points. Nothing here consults a
global frame, and no patch is ever merged into another.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .fitting import fit_plane, plane_basis, plane_distance


@dataclass
class Patch:
    """A planar surface fitted to its own supporting points."""

    patch_id: int
    normal: np.ndarray                 # (3,) unit, canonically oriented
    d: float                           # normal @ x + d == 0
    point_idx: np.ndarray              # indices into the source cloud
    n_points: int
    p95_residual_m: float
    centroid: np.ndarray               # (3,)
    u_range: tuple[float, float]       # in-plane extent along plane_basis u
    v_range: tuple[float, float]       # in-plane extent along plane_basis v

    def area_bound_m2(self) -> float:
        """Area of the patch's in-plane bounding rectangle."""
        return (self.u_range[1] - self.u_range[0]) * (self.v_range[1] - self.v_range[0])


def _finalise(patch_id: int, xyz: np.ndarray, members: np.ndarray) -> Patch:
    """Fit the final plane and measure the patch's residual and extent."""
    pts = xyz[members]
    normal, d = fit_plane(pts)
    residual = np.abs(plane_distance(pts, normal, d))
    u, v = plane_basis(normal)
    centroid = pts.mean(axis=0)
    rel = pts - centroid
    us, vs = rel @ u, rel @ v
    return Patch(
        patch_id=patch_id,
        normal=normal,
        d=d,
        point_idx=np.sort(members),
        n_points=int(len(members)),
        p95_residual_m=float(np.percentile(residual, 95)),
        centroid=centroid,
        u_range=(float(us.min()), float(us.max())),
        v_range=(float(vs.min()), float(vs.max())),
    )


def extract_patches(
    xyz: np.ndarray,
    normals: np.ndarray,
    curvature: np.ndarray,
    config: dict,
) -> tuple[list[Patch], np.ndarray]:
    """Grow planar patches from low-curvature seeds.

    Returns (patches, labels). labels is (N,) int64 with the patch_id owning
    each point, or -1 where a point joined no patch. Unassigned points are the
    caller's to report -- they are never silently dropped.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    curvature = np.asarray(curvature, dtype=np.float64)

    tau = float(config["tau_fit_m"])
    cos_tol = float(np.cos(np.radians(config["patch_angle_tol_deg"])))
    radius = float(config["patch_connect_radius_m"])
    min_points = int(config["min_patch_points"])
    refit_interval = int(config["refit_interval"])

    tree = cKDTree(xyz)
    n = len(xyz)
    labels = np.full(n, -1, dtype=np.int64)
    patches: list[Patch] = []

    # flattest points first: seeds land mid-face, never on an edge
    seed_order = np.argsort(curvature, kind="stable")

    for seed in seed_order:
        if labels[seed] != -1:
            continue

        pending_id = len(patches)
        members = [int(seed)]
        labels[seed] = pending_id

        plane_n = normals[seed].copy()
        plane_d = float(-plane_n @ xyz[seed])

        stack = [int(seed)]
        since_refit = 0
        while stack:
            current = stack.pop()
            # sorted() keeps neighbour visit order deterministic
            for j in sorted(tree.query_ball_point(xyz[current], radius)):
                if labels[j] != -1:
                    continue
                if abs(float(normals[j] @ plane_n)) < cos_tol:
                    continue
                if abs(float(xyz[j] @ plane_n + plane_d)) > tau:
                    continue

                labels[j] = pending_id
                members.append(j)
                stack.append(j)

                since_refit += 1
                if since_refit >= refit_interval:
                    plane_n, plane_d = fit_plane(xyz[np.asarray(members)])
                    since_refit = 0

        member_arr = np.asarray(members, dtype=np.int64)
        if len(member_arr) < min_points:
            labels[member_arr] = -1        # release; may join a later patch
            continue

        patches.append(_finalise(pending_id, xyz, member_arr))

    return patches, labels


def unassigned_count(labels: np.ndarray) -> int:
    """Number of points that joined no patch. Reported, never dropped."""
    return int(np.count_nonzero(np.asarray(labels) == -1))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/rscene/test_patches.py -v`
Expected: 6 passed

If `test_a_75mm_step_survives_as_its_own_patch` fails, do NOT loosen the assertion — it is the acceptance criterion for the whole rebuild. Diagnose in this order: (a) is `patch_connect_radius_m` larger than the 75 mm step, letting growth jump the gap? (b) is `tau_fit_m` larger than the step depth? (c) is the plane refit drifting the plane toward the step? Reduce the connect radius first.

- [ ] **Step 5: Commit**

```bash
git add src/rscene/core/patches.py tests/rscene/test_patches.py
git commit -m "feat(core): region-growing patch extraction preserving rectangular steps"
```

---

### Task 7: Patch graph — coplanarity classes and adjacency

Coplanarity is **recorded, never applied**. Two wall segments either side of a door share a plane; grouping them into a reported class lets later stages know they belong to one wall, while leaving both fitted planes untouched.

**Files:**
- Create: `src/rscene/core/graph.py`
- Create: `tests/rscene/test_graph.py`

**Interfaces:**
- Consumes: `Patch` (Task 6); `DEFAULT_CONFIG` (Task 5)
- Produces:
  - `coplanarity_classes(patches: list[Patch], config: dict) -> list[list[int]]` — lists of `patch_id`, each sorted, outer list sorted by first element
  - `patch_adjacency(patches: list[Patch], xyz: np.ndarray, config: dict) -> list[tuple[int, int]]` — sorted `(lo, hi)` pairs
  - `intersection_line(a: Patch, b: Patch) -> tuple[np.ndarray, np.ndarray] | None` — `(point, direction)`, or `None` if parallel

- [ ] **Step 1: Write the failing test**

Create `tests/rscene/test_graph.py`:

```python
import numpy as np

from rscene.config import merged_config
from rscene.core.graph import coplanarity_classes, intersection_line, patch_adjacency
from rscene.core.normals import estimate_normals
from rscene.core.patches import Patch, extract_patches
from rscene.core.prim import Box


def _patch(pid, normal, d):
    return Patch(
        patch_id=pid, normal=np.array(normal, dtype=float), d=d,
        point_idx=np.array([0]), n_points=1, p95_residual_m=0.0,
        centroid=np.zeros(3), u_range=(0.0, 1.0), v_range=(0.0, 1.0),
    )


def test_coplanar_patches_share_a_class_but_keep_their_own_planes():
    a = _patch(0, (1, 0, 0), -2.000)
    b = _patch(1, (1, 0, 0), -2.002)     # 2 mm apart: same plane within tolerance
    c = _patch(2, (1, 0, 0), -2.500)     # clearly a different plane

    classes = coplanarity_classes([a, b, c], merged_config())

    assert [0, 1] in classes and [2] in classes
    assert a.d == -2.000 and b.d == -2.002      # untouched, never averaged


def test_parallel_but_offset_patches_are_not_coplanar():
    a = _patch(0, (1, 0, 0), 0.0)
    b = _patch(1, (1, 0, 0), -0.075)             # the 75 mm step
    assert coplanarity_classes([a, b], merged_config()) == [[0], [1]]


def test_patches_with_different_normals_are_not_coplanar():
    a = _patch(0, (1, 0, 0), 0.0)
    b = _patch(1, (0, 1, 0), 0.0)
    assert coplanarity_classes([a, b], merged_config()) == [[0], [1]]


def test_intersection_line_of_two_perpendicular_planes():
    a = _patch(0, (1, 0, 0), 0.0)        # x = 0
    b = _patch(1, (0, 1, 0), 0.0)        # y = 0
    point, direction = intersection_line(a, b)

    assert np.allclose(np.abs(direction), [0, 0, 1])
    assert abs(point[0]) < 1e-9 and abs(point[1]) < 1e-9


def test_intersection_line_of_parallel_planes_is_none():
    assert intersection_line(_patch(0, (1, 0, 0), 0.0),
                             _patch(1, (1, 0, 0), -0.2)) is None


def test_adjacent_patches_are_detected_and_distant_ones_are_not():
    corner = np.concatenate([
        Box("a", (0, 0, 0), (0, 2, 2)).sample_surface(0.01, faces=("x+",)),
        Box("b", (0, 0, 0), (2, 0, 2)).sample_surface(0.01, faces=("y+",)),
    ])
    far = Box("c", (5, 5, 0), (5, 7, 2)).sample_surface(0.01, faces=("x+",))
    xyz = np.concatenate([corner, far])

    normals, curvature = estimate_normals(xyz, k=24)
    patches, _ = extract_patches(xyz, normals, curvature, merged_config())
    pairs = patch_adjacency(patches, xyz, merged_config())

    ids_near = {p.patch_id for p in patches if p.centroid[0] < 3 and p.centroid[1] < 3}
    ids_far = {p.patch_id for p in patches if p.centroid[0] > 3}

    assert any(a in ids_near and b in ids_near for a, b in pairs)
    assert not any((a in ids_far) != (b in ids_far) for a, b in pairs)


def test_classes_and_pairs_are_deterministically_ordered():
    ps = [_patch(2, (1, 0, 0), 0.0), _patch(0, (1, 0, 0), 0.0), _patch(1, (0, 1, 0), 0.0)]
    classes = coplanarity_classes(ps, merged_config())
    assert classes == sorted(classes)
    assert all(c == sorted(c) for c in classes)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/rscene/test_graph.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rscene.core.graph'`

- [ ] **Step 3: Implement `graph.py`**

Create `src/rscene/core/graph.py`:

```python
"""Patch relationships: coplanarity classes, adjacency, intersection lines.

Coplanarity is RECORDED, never applied. Two wall segments either side of a
door belong to one wall, and saying so is useful -- but their individually
fitted planes are left exactly as measured. Averaging them would be a global
snap by another name.

Intersection lines are how sharp edges get computed later: an edge is where
two fitted planes meet, never a polyline meshed from points.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .patches import Patch


def coplanarity_classes(patches: list[Patch], config: dict) -> list[list[int]]:
    """Group patches that lie on the same plane within tolerance.

    Returns lists of patch_id, each sorted ascending, the outer list sorted by
    first element. Patch planes are not modified.
    """
    dist_tol = float(config["coplanar_dist_tol_m"])
    cos_tol = float(np.cos(np.radians(config["coplanar_angle_tol_deg"])))

    ordered = sorted(patches, key=lambda p: p.patch_id)
    parent = {p.patch_id: p.patch_id for p in ordered}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            if abs(float(a.normal @ b.normal)) < cos_tol:
                continue
            if abs(a.d - b.d) > dist_tol:
                continue
            union(a.patch_id, b.patch_id)

    groups: dict[int, list[int]] = {}
    for p in ordered:
        groups.setdefault(find(p.patch_id), []).append(p.patch_id)
    return sorted([sorted(v) for v in groups.values()])


def intersection_line(a: Patch, b: Patch) -> tuple[np.ndarray, np.ndarray] | None:
    """Line where two planes meet, as (point, unit direction).

    Returns None when the planes are parallel and therefore never meet.
    """
    direction = np.cross(a.normal, b.normal)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        return None
    direction = direction / norm

    # pick the point on the line closest to the origin
    matrix = np.vstack([a.normal, b.normal, direction])
    rhs = np.array([-a.d, -b.d, 0.0], dtype=np.float64)
    point = np.linalg.solve(matrix, rhs)
    return point, direction


def patch_adjacency(
    patches: list[Patch], xyz: np.ndarray, config: dict
) -> list[tuple[int, int]]:
    """Pairs of patches with supporting points within adjacency_radius_m.

    Returns sorted (lo, hi) patch_id pairs.
    """
    radius = float(config["adjacency_radius_m"])
    xyz = np.asarray(xyz, dtype=np.float64)

    owner = {}
    clouds = []
    for p in sorted(patches, key=lambda q: q.patch_id):
        owner[len(clouds)] = p.patch_id
        clouds.append(xyz[p.point_idx])

    pairs: set[tuple[int, int]] = set()
    trees = [cKDTree(c) for c in clouds]
    for i in range(len(clouds)):
        for j in range(i + 1, len(clouds)):
            if trees[i].count_neighbors(trees[j], radius) > 0:
                pairs.add((min(owner[i], owner[j]), max(owner[i], owner[j])))

    return sorted(pairs)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/rscene/test_graph.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/rscene/core/graph.py tests/rscene/test_graph.py
git commit -m "feat(core): patch coplanarity classes, adjacency and intersection lines"
```

---

### Task 8: Scene document with measurement provenance

**Files:**
- Create: `src/rscene/core/scene.py`
- Create: `tests/rscene/test_scene.py`

**Interfaces:**
- Consumes: `Patch` (Task 6)
- Produces:
  - `Measurement(value: float, method: str, n_points: int, p95_residual: float)`
  - `Provenance(scan_path: str, scan_sha256: str, pipeline_version: str, timestamp: str, config: dict)`
  - `Frame(z_axis: list[float], xy_rotation_deg: float, floor_z: float | None, ceiling_z: float | None)`
  - `Scene(provenance, frame, patches, coplanarity_classes, adjacency, unassigned_points, diagnostics)`
  - `scene_to_json(scene: Scene) -> str`, `scene_from_json(text: str) -> Scene`

- [ ] **Step 1: Write the failing test**

Create `tests/rscene/test_scene.py`:

```python
import json

import numpy as np

from rscene.core.patches import Patch
from rscene.core.scene import (
    Frame, Measurement, Provenance, Scene, scene_from_json, scene_to_json,
)


def _scene():
    return Scene(
        provenance=Provenance(
            scan_path="koushik.las", scan_sha256="abc123",
            pipeline_version="0.1.0", timestamp="2026-08-11T00:00:00Z",
            config={"tau_fit_m": 0.003},
        ),
        frame=Frame(z_axis=[0.0, 0.0, 1.0], xy_rotation_deg=1.4,
                    floor_z=0.0, ceiling_z=2.75),
        patches=[Patch(
            patch_id=0, normal=np.array([1.0, 0.0, 0.0]), d=-2.0,
            point_idx=np.array([0, 1, 2]), n_points=3, p95_residual_m=0.0021,
            centroid=np.array([2.0, 1.0, 1.4]), u_range=(-1.0, 1.0), v_range=(-1.4, 1.4),
        )],
        coplanarity_classes=[[0]],
        adjacency=[],
        unassigned_points=42,
        diagnostics={"p95_residual_m": 0.0021},
    )


def test_measurement_carries_provenance_and_uncertainty():
    m = Measurement(value=0.2031, method="face-to-face raw points",
                    n_points=18422, p95_residual=0.0021)
    assert m.to_dict() == {
        "value": 0.2031, "method": "face-to-face raw points",
        "n_points": 18422, "p95_residual": 0.0021,
    }


def test_scene_round_trips_through_json():
    original = _scene()
    restored = scene_from_json(scene_to_json(original))

    assert restored.provenance.scan_sha256 == "abc123"
    assert restored.frame.ceiling_z == 2.75
    assert restored.unassigned_points == 42
    assert len(restored.patches) == 1
    assert np.allclose(restored.patches[0].normal, [1.0, 0.0, 0.0])
    assert restored.patches[0].d == -2.0
    assert np.array_equal(restored.patches[0].point_idx, [0, 1, 2])


def test_serialisation_is_deterministic():
    assert scene_to_json(_scene()) == scene_to_json(_scene())


def test_json_keys_are_sorted_so_diffs_stay_readable():
    payload = json.loads(scene_to_json(_scene()))
    assert list(payload.keys()) == sorted(payload.keys())


def test_unassigned_points_are_recorded_not_dropped():
    payload = json.loads(scene_to_json(_scene()))
    assert payload["unassigned_points"] == 42
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/rscene/test_scene.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rscene.core.scene'`

- [ ] **Step 3: Implement `scene.py`**

Create `src/rscene/core/scene.py`:

```python
"""The scene document -- the single source of truth.

JSON rather than executable code: the LLM never edits geometry here, so
code-as-truth would cost determinism and diffability without buying anything.
A readable room.py is emitted FROM this document later, not into it.

Every dimension is a Measurement carrying how it was measured, how many points
backed it and how well they fitted. A number a designer cannot audit is a
number they will eventually stop trusting.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .patches import Patch


@dataclass
class Measurement:
    """A dimension plus the evidence behind it."""

    value: float
    method: str
    n_points: int
    p95_residual: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "method": self.method,
            "n_points": self.n_points,
            "p95_residual": self.p95_residual,
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "Measurement":
        return Measurement(
            value=payload["value"], method=payload["method"],
            n_points=payload["n_points"], p95_residual=payload["p95_residual"],
        )


@dataclass
class Provenance:
    """Where the scene came from, so any result can be reproduced."""

    scan_path: str
    scan_sha256: str
    pipeline_version: str
    timestamp: str
    config: dict

    def to_dict(self) -> dict[str, Any]:
        return {
            "scan_path": self.scan_path, "scan_sha256": self.scan_sha256,
            "pipeline_version": self.pipeline_version, "timestamp": self.timestamp,
            "config": self.config,
        }

    @staticmethod
    def from_dict(p: dict[str, Any]) -> "Provenance":
        return Provenance(
            scan_path=p["scan_path"], scan_sha256=p["scan_sha256"],
            pipeline_version=p["pipeline_version"], timestamp=p["timestamp"],
            config=p["config"],
        )


@dataclass
class Frame:
    """The measured frame. Reported as data; no geometry is rotated to match."""

    z_axis: list[float]
    xy_rotation_deg: float
    floor_z: Optional[float] = None
    ceiling_z: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "z_axis": list(self.z_axis), "xy_rotation_deg": self.xy_rotation_deg,
            "floor_z": self.floor_z, "ceiling_z": self.ceiling_z,
        }

    @staticmethod
    def from_dict(p: dict[str, Any]) -> "Frame":
        return Frame(
            z_axis=p["z_axis"], xy_rotation_deg=p["xy_rotation_deg"],
            floor_z=p["floor_z"], ceiling_z=p["ceiling_z"],
        )


def _patch_to_dict(p: Patch) -> dict[str, Any]:
    return {
        "patch_id": p.patch_id,
        "normal": [float(x) for x in p.normal],
        "d": float(p.d),
        "point_idx": [int(i) for i in p.point_idx],
        "n_points": p.n_points,
        "p95_residual_m": p.p95_residual_m,
        "centroid": [float(x) for x in p.centroid],
        "u_range": list(p.u_range),
        "v_range": list(p.v_range),
    }


def _patch_from_dict(p: dict[str, Any]) -> Patch:
    return Patch(
        patch_id=p["patch_id"],
        normal=np.array(p["normal"], dtype=np.float64),
        d=p["d"],
        point_idx=np.array(p["point_idx"], dtype=np.int64),
        n_points=p["n_points"],
        p95_residual_m=p["p95_residual_m"],
        centroid=np.array(p["centroid"], dtype=np.float64),
        u_range=tuple(p["u_range"]),
        v_range=tuple(p["v_range"]),
    )


@dataclass
class Scene:
    """The whole scene document. Plan 1 populates patches; parts arrive in Plan 2."""

    provenance: Provenance
    frame: Frame
    patches: list[Patch] = field(default_factory=list)
    coplanarity_classes: list[list[int]] = field(default_factory=list)
    adjacency: list[tuple[int, int]] = field(default_factory=list)
    unassigned_points: int = 0
    diagnostics: dict = field(default_factory=dict)


def scene_to_json(scene: Scene) -> str:
    """Serialise deterministically: sorted keys, fixed indent."""
    payload = {
        "adjacency": [list(pair) for pair in scene.adjacency],
        "coplanarity_classes": scene.coplanarity_classes,
        "diagnostics": scene.diagnostics,
        "frame": scene.frame.to_dict(),
        "patches": [_patch_to_dict(p) for p in scene.patches],
        "provenance": scene.provenance.to_dict(),
        "unassigned_points": scene.unassigned_points,
    }
    return json.dumps(payload, sort_keys=True, indent=2)


def scene_from_json(text: str) -> Scene:
    payload = json.loads(text)
    return Scene(
        provenance=Provenance.from_dict(payload["provenance"]),
        frame=Frame.from_dict(payload["frame"]),
        patches=[_patch_from_dict(p) for p in payload["patches"]],
        coplanarity_classes=payload["coplanarity_classes"],
        adjacency=[tuple(pair) for pair in payload["adjacency"]],
        unassigned_points=payload["unassigned_points"],
        diagnostics=payload["diagnostics"],
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/rscene/test_scene.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/rscene/core/scene.py tests/rscene/test_scene.py
git commit -m "feat(core): scene document with measurement provenance and JSON round-trip"
```

---

### Task 9: Frame estimation — measured, never enforced

**Files:**
- Create: `src/rscene/core/level.py`
- Create: `tests/rscene/test_level.py`

**Interfaces:**
- Consumes: `Patch` (Task 6); `Frame` (Task 8); `DEFAULT_CONFIG` (Task 5)
- Produces: `estimate_frame(patches: list[Patch], config: dict) -> Frame`

- [ ] **Step 1: Write the failing test**

Create `tests/rscene/test_level.py`:

```python
import numpy as np

from rscene.config import merged_config
from rscene.core.level import estimate_frame
from rscene.core.normals import estimate_normals
from rscene.core.patches import extract_patches
from rscene.core.prim import Box


def _room_points(spacing=0.02):
    floor = Box("floor", (0, 0, 0.0), (3, 2.5, 0.0)).sample_surface(spacing, faces=("z+",))
    ceil = Box("ceil", (0, 0, 2.75), (3, 2.5, 2.75)).sample_surface(spacing, faces=("z-",))
    wall = Box("wall", (0, 0, 0), (0, 2.5, 2.75)).sample_surface(spacing, faces=("x+",))
    return np.concatenate([floor, ceil, wall])


def _frame_from(xyz):
    normals, curvature = estimate_normals(xyz, k=24)
    patches, _ = extract_patches(xyz, normals, curvature, merged_config())
    return estimate_frame(patches, merged_config())


def test_gravity_axis_is_recovered_from_the_floor():
    frame = _frame_from(_room_points())
    assert np.allclose(frame.z_axis, [0.0, 0.0, 1.0], atol=1e-3)


def test_storey_levels_are_recovered():
    frame = _frame_from(_room_points())
    assert abs(frame.floor_z - 0.0) < 0.003
    assert abs(frame.ceiling_z - 2.75) < 0.003


def test_xy_rotation_is_reported_not_applied():
    """A yawed room reports its rotation; patch planes stay exactly as measured."""
    xyz = _room_points()
    yaw = np.radians(12.0)
    rot = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    rotated = xyz @ rot.T

    normals, curvature = estimate_normals(rotated, k=24)
    patches, _ = extract_patches(rotated, normals, curvature, merged_config())
    frame = estimate_frame(patches, merged_config())

    assert abs(frame.xy_rotation_deg - 12.0) < 1.0

    # the wall patch still sits where it was measured, un-rotated
    wall = max((p for p in patches if abs(p.normal[2]) < 0.2),
               key=lambda p: p.n_points)
    assert abs(abs(wall.normal[0]) - np.cos(yaw)) < 0.02


def test_frame_with_no_horizontal_patches_reports_none_levels():
    wall = Box("w", (0, 0, 0), (0, 2, 2.5)).sample_surface(0.02, faces=("x+",))
    frame = _frame_from(wall)
    assert frame.floor_z is None and frame.ceiling_z is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/rscene/test_level.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rscene.core.level'`

- [ ] **Step 3: Implement `level.py`**

Create `src/rscene/core/level.py`:

```python
"""Frame estimation: gravity axis, reported XY rotation, storey levels.

Everything here is MEASURED AND REPORTED. Nothing is rotated, snapped or
regularised. Downstream stages may read the frame to interpret results; none
of them may use it to move a surface.
"""
from __future__ import annotations

import numpy as np

from .patches import Patch
from .scene import Frame


def estimate_frame(patches: list[Patch], config: dict) -> Frame:
    """Derive the gravity axis, dominant XY rotation and storey levels.

    The gravity axis comes from the largest horizontal patch (the floor slab).
    The XY rotation is the circular mean of vertical-patch azimuths modulo 90
    degrees -- reported so a designer knows how far off-square the building is,
    never applied.
    """
    tol_cos = float(np.cos(np.radians(config["floor_normal_tol_deg"])))

    horizontal = [p for p in patches if abs(float(p.normal[2])) >= tol_cos]
    vertical = [p for p in patches if abs(float(p.normal[2])) < 0.2]

    if horizontal:
        floor_like = max(horizontal, key=lambda p: p.n_points)
        z_axis = floor_like.normal.copy()
        if z_axis[2] < 0:
            z_axis = -z_axis
    else:
        z_axis = np.array([0.0, 0.0, 1.0])

    floor_z: float | None = None
    ceiling_z: float | None = None
    if horizontal:
        heights = sorted(float(p.centroid[2]) for p in horizontal)
        floor_z, ceiling_z = heights[0], heights[-1]
        if ceiling_z == floor_z:
            ceiling_z = None

    # circular mean of azimuths folded into [0, 90): the building's squareness
    xy_rotation_deg = 0.0
    if vertical:
        azimuths = np.array([
            np.degrees(np.arctan2(float(p.normal[1]), float(p.normal[0])))
            for p in vertical
        ])
        folded = np.radians((azimuths % 90.0) * 4.0)     # 90 deg -> full circle
        weights = np.array([p.n_points for p in vertical], dtype=np.float64)
        mean_angle = np.arctan2(
            float(np.sum(weights * np.sin(folded))),
            float(np.sum(weights * np.cos(folded))),
        )
        xy_rotation_deg = float((np.degrees(mean_angle) / 4.0) % 90.0)

    return Frame(
        z_axis=[float(x) for x in z_axis],
        xy_rotation_deg=xy_rotation_deg,
        floor_z=floor_z,
        ceiling_z=ceiling_z,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/rscene/test_level.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/rscene/core/level.py tests/rscene/test_level.py
git commit -m "feat(core): frame estimation reporting rotation without applying it"
```

---

### Task 10: Golden apartment and the end-to-end accuracy test

This is what makes "accurate" a CI-checkable property rather than an opinion.

**Files:**
- Create: `tests/golden/__init__.py`
- Create: `tests/golden/apartment.py`
- Create: `tests/golden/test_accuracy.py`

**Interfaces:**
- Consumes: `Box` (Task 1); `add_gaussian_noise` (Task 2); `estimate_normals` (Task 4); `extract_patches` (Task 6); `merged_config` (Task 5)
- Produces: `GoldenScene(points: np.ndarray, truth: dict[str, float])`, `build_golden_room(spacing_m=0.008, noise_m=0.001, seed=0) -> GoldenScene`

- [ ] **Step 1: Write the golden scene builder and the failing test**

Create empty `tests/golden/__init__.py`.

Create `tests/golden/apartment.py`:

```python
"""A synthetic bare-shell room with exactly known dimensions.

Only interior-visible faces are sampled, mirroring what a scanner standing in
the room actually sees. Truth values are the numbers the pipeline must recover.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rscene.core.points import add_gaussian_noise
from rscene.core.prim import Box

# Room interior: x in [0, 3.0], y in [0, 2.5], z in [0, 2.75]
CLEAR_X = 3.0
CLEAR_Y = 2.5
HEIGHT = 2.75
EXTRUSION_DEPTH = 0.075
EXTRUSION_WIDTH = 0.35
GROOVE_DEPTH = 0.012
GROOVE_WIDTH = 0.06
BOX_SIZE = 0.08
BOX_DEPTH = 0.045


@dataclass
class GoldenScene:
    points: np.ndarray
    truth: dict[str, float]


def build_golden_room(
    spacing_m: float = 0.008, noise_m: float = 0.001, seed: int = 0
) -> GoldenScene:
    """Build the room and sample its interior surfaces."""
    faces: list[np.ndarray] = []

    # floor and ceiling
    faces.append(Box("floor", (0, 0, 0), (CLEAR_X, CLEAR_Y, 0)).sample_surface(
        spacing_m, faces=("z+",)))
    faces.append(Box("ceiling", (0, 0, HEIGHT), (CLEAR_X, CLEAR_Y, HEIGHT)).sample_surface(
        spacing_m, faces=("z-",)))

    # wall at y = 0, carrying the extrusion (x 1.20 -> 1.55)
    ex0, ex1 = 1.20, 1.20 + EXTRUSION_WIDTH
    faces.append(Box("y0_left", (0, 0, 0), (ex0, 0, HEIGHT)).sample_surface(
        spacing_m, faces=("y+",)))
    faces.append(Box("y0_right", (ex1, 0, 0), (CLEAR_X, 0, HEIGHT)).sample_surface(
        spacing_m, faces=("y+",)))
    faces.append(Box("ext_face", (ex0, EXTRUSION_DEPTH, 0),
                     (ex1, EXTRUSION_DEPTH, HEIGHT)).sample_surface(
        spacing_m, faces=("y+",)))
    faces.append(Box("ext_side_l", (ex0, 0, 0), (ex0, EXTRUSION_DEPTH, HEIGHT)).sample_surface(
        spacing_m, faces=("x+",)))
    faces.append(Box("ext_side_r", (ex1, 0, 0), (ex1, EXTRUSION_DEPTH, HEIGHT)).sample_surface(
        spacing_m, faces=("x-",)))

    # wall at y = CLEAR_Y, plain
    faces.append(Box("y1", (0, CLEAR_Y, 0), (CLEAR_X, CLEAR_Y, HEIGHT)).sample_surface(
        spacing_m, faces=("y-",)))

    # wall at x = 0, carrying a full-height groove (y 1.00 -> 1.06)
    gr0, gr1 = 1.00, 1.00 + GROOVE_WIDTH
    faces.append(Box("x0_a", (0, 0, 0), (0, gr0, HEIGHT)).sample_surface(
        spacing_m, faces=("x+",)))
    faces.append(Box("x0_b", (0, gr1, 0), (0, CLEAR_Y, HEIGHT)).sample_surface(
        spacing_m, faces=("x+",)))
    faces.append(Box("groove_base", (-GROOVE_DEPTH, gr0, 0),
                     (-GROOVE_DEPTH, gr1, HEIGHT)).sample_surface(
        spacing_m, faces=("x+",)))

    # wall at x = CLEAR_X, carrying a recessed switch box at z 1.20 -> 1.28
    sb_y0, sb_y1 = 1.10, 1.10 + BOX_SIZE
    sb_z0, sb_z1 = 1.20, 1.20 + BOX_SIZE
    # the wall face is sampled as a ring around the recess -- a scanner cannot
    # see wall surface where the box has been cut out of it
    faces.append(Box("x1_below", (CLEAR_X, 0, 0), (CLEAR_X, CLEAR_Y, sb_z0)).sample_surface(
        spacing_m, faces=("x-",)))
    faces.append(Box("x1_above", (CLEAR_X, 0, sb_z1),
                     (CLEAR_X, CLEAR_Y, HEIGHT)).sample_surface(spacing_m, faces=("x-",)))
    faces.append(Box("x1_left", (CLEAR_X, 0, sb_z0),
                     (CLEAR_X, sb_y0, sb_z1)).sample_surface(spacing_m, faces=("x-",)))
    faces.append(Box("x1_right", (CLEAR_X, sb_y1, sb_z0),
                     (CLEAR_X, CLEAR_Y, sb_z1)).sample_surface(spacing_m, faces=("x-",)))
    # the recess base, sampled finely enough to clear the min_patch_points floor
    faces.append(Box("sb_base", (CLEAR_X + BOX_DEPTH, sb_y0, sb_z0),
                     (CLEAR_X + BOX_DEPTH, sb_y1, sb_z1)).sample_surface(
        0.004, faces=("x-",)))

    points = np.concatenate(faces)
    points = add_gaussian_noise(points, noise_m, np.random.default_rng(seed))

    truth = {
        "clear_span_x_m": CLEAR_X,
        "clear_span_y_m": CLEAR_Y,
        "ceiling_height_m": HEIGHT,
        "extrusion_depth_m": EXTRUSION_DEPTH,
        "extrusion_width_m": EXTRUSION_WIDTH,
        "groove_depth_m": GROOVE_DEPTH,
        "switch_box_depth_m": BOX_DEPTH,
    }
    return GoldenScene(points=points, truth=truth)
```

Create `tests/golden/test_accuracy.py`:

```python
"""Accuracy assertions against exactly known ground truth.

Plan 1 asserts what PATCHES can prove. Part-level dimensions (wall thickness,
opening widths) arrive with Plan 2.
"""
import numpy as np
import pytest

from rscene.config import merged_config
from rscene.core.level import estimate_frame
from rscene.core.normals import estimate_normals
from rscene.core.patches import extract_patches, unassigned_count

from .apartment import build_golden_room

TOL = 0.003     # 3 mm, matching tau_fit


@pytest.fixture(scope="module")
def extracted():
    scene = build_golden_room()
    normals, curvature = estimate_normals(scene.points, k=24)
    patches, labels = extract_patches(scene.points, normals, curvature, merged_config())
    return scene, patches, labels


def _offsets_along(patches, axis):
    """Sorted plane offsets of patches whose normal points along `axis`."""
    return sorted(abs(p.d) for p in patches if abs(p.normal[axis]) > 0.99)


@pytest.mark.slow
def test_ceiling_height_is_recovered(extracted):
    scene, patches, _ = extracted
    frame = estimate_frame(patches, merged_config())
    assert abs((frame.ceiling_z - frame.floor_z) - scene.truth["ceiling_height_m"]) < TOL


@pytest.mark.slow
def test_extrusion_depth_is_recovered(extracted):
    scene, patches, _ = extracted
    offsets = _offsets_along(patches, axis=1)
    gaps = [round(b - a, 4) for a, b in zip(offsets, offsets[1:])]
    assert any(abs(g - scene.truth["extrusion_depth_m"]) < TOL for g in gaps), \
        f"no 75 mm step among y-facing offsets {offsets}"


@pytest.mark.slow
def test_groove_depth_is_recovered(extracted):
    scene, patches, _ = extracted
    offsets = _offsets_along(patches, axis=0)
    gaps = [round(b - a, 4) for a, b in zip(offsets, offsets[1:])]
    assert any(abs(g - scene.truth["groove_depth_m"]) < TOL for g in gaps), \
        f"no 12 mm groove among x-facing offsets {offsets}"


@pytest.mark.slow
def test_switch_box_is_found_as_its_own_patch(extracted):
    scene, patches, _ = extracted
    offsets = _offsets_along(patches, axis=0)
    gaps = [round(b - a, 4) for a, b in zip(offsets, offsets[1:])]
    assert any(abs(g - scene.truth["switch_box_depth_m"]) < TOL for g in gaps), \
        f"no 45 mm switch box among x-facing offsets {offsets}"


@pytest.mark.slow
def test_almost_every_point_is_explained(extracted):
    scene, _, labels = extracted
    unassigned = unassigned_count(labels)
    assert unassigned / len(scene.points) < 0.05, \
        f"{unassigned} of {len(scene.points)} points joined no patch"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/golden/ -v -m slow`
Expected: FAIL — `ModuleNotFoundError: No module named 'tests.golden.apartment'` until `__init__.py` exists, then assertion failures if extraction is not behaving.

- [ ] **Step 3: Make them pass**

The implementation already exists (Tasks 1–9). If a test fails, tune only `min_patch_points` and `patch_connect_radius_m` in `DEFAULT_CONFIG`, then re-run. Record the final values and the reason in the config comments.

Two constraints on that tuning: `patch_connect_radius_m` must stay **below** `GROOVE_WIDTH` (0.06 m) or growth jumps across the groove and erases it; `min_patch_points` must stay **below** the switch-box sample count (0.08 m / 0.004 m spacing → about 21×21 ≈ 441 points) or the box is discarded.

- [ ] **Step 4: Run the whole suite**

Run: `.venv/bin/pytest tests/rscene tests/golden -v`
Expected: all pass. Note the runtime of the slow tests; if the golden room exceeds 60 s, raise `spacing_m` to 0.01 and keep the switch-box face at 0.004.

- [ ] **Step 5: Commit**

```bash
git add tests/golden
git commit -m "test: golden bare-shell room with ground-truth accuracy assertions"
```

---

### Task 11: LAS adapter

**Files:**
- Create: `src/rscene/io/__init__.py`
- Create: `src/rscene/io/las.py`
- Create: `tests/rscene/test_io_las.py`

**Interfaces:**
- Consumes: `PointSet` (Task 2)
- Produces:
  - `load_las(path: str, max_points: int | None = None, seed: int = 0) -> PointSet`
  - `save_las(points: PointSet, path: str) -> None`
  - `file_sha256(path: str) -> str`

- [ ] **Step 1: Write the failing test**

Create `tests/rscene/test_io_las.py`:

```python
import numpy as np
import pytest

from rscene.core.points import PointSet

laspy = pytest.importorskip("laspy")

from rscene.io.las import file_sha256, load_las, save_las


def _sample(tmp_path):
    n = 500
    rng = np.random.default_rng(0)
    ps = PointSet(
        xyz=rng.uniform(0, 5, (n, 3)),
        gps_time=np.arange(n, dtype=np.float64),
        intensity=rng.integers(0, 65535, n).astype(np.uint16),
        rgb=rng.integers(0, 255, (n, 3)).astype(np.uint8),
    )
    path = tmp_path / "scan.las"
    save_las(ps, str(path))
    return ps, str(path)


def test_round_trip_preserves_geometry_to_the_las_scale(tmp_path):
    original, path = _sample(tmp_path)
    restored = load_las(path)

    assert restored.n == original.n
    assert np.allclose(restored.xyz, original.xyz, atol=0.0005)   # 0.1 mm scale


def test_round_trip_preserves_every_attribute(tmp_path):
    original, path = _sample(tmp_path)
    restored = load_las(path)

    assert np.allclose(restored.gps_time, original.gps_time)
    assert np.array_equal(restored.intensity, original.intensity)
    assert np.array_equal(restored.rgb, original.rgb)


def test_max_points_subsamples_deterministically(tmp_path):
    _, path = _sample(tmp_path)
    a = load_las(path, max_points=100, seed=7)
    b = load_las(path, max_points=100, seed=7)

    assert a.n == 100
    assert np.array_equal(a.xyz, b.xyz)


def test_max_points_above_the_count_is_a_no_op(tmp_path):
    original, path = _sample(tmp_path)
    assert load_las(path, max_points=10_000).n == original.n


def test_sha256_is_stable_and_content_dependent(tmp_path):
    _, path = _sample(tmp_path)
    assert file_sha256(path) == file_sha256(path)

    other = tmp_path / "other.las"
    save_las(PointSet(xyz=np.zeros((10, 3))), str(other))
    assert file_sha256(path) != file_sha256(str(other))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/rscene/test_io_las.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rscene.io'`

- [ ] **Step 3: Implement the adapter**

Create empty `src/rscene/io/__init__.py`.

Create `src/rscene/io/las.py`:

```python
"""LAS/LAZ adapter.

The only module allowed to import laspy. Reads every attribute the point
format carries so nothing the sensor gave us is discarded at the door --
intensity in particular is a material cue the pipeline should keep available.
"""
from __future__ import annotations

import hashlib

import numpy as np

try:
    import laspy
except ImportError as exc:                                  # pragma: no cover
    raise ImportError(
        "laspy is required for LAS I/O; install with: uv pip install -e '.[io]'"
    ) from exc

from ..core.points import PointSet

_CHUNK = 3_000_000


def file_sha256(path: str) -> str:
    """Content hash of the scan, recorded in the scene's provenance."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_las(path: str, max_points: int | None = None, seed: int = 0) -> PointSet:
    """Load a LAS/LAZ file, retaining every attribute the format carries."""
    with laspy.open(path) as reader:
        dims = set(reader.header.point_format.dimension_names)
        has_time = "gps_time" in dims
        has_rgb = {"red", "green", "blue"}.issubset(dims)
        has_intensity = "intensity" in dims

        xyz_chunks, t_chunks, rgb_chunks, i_chunks = [], [], [], []
        for pts in reader.chunk_iterator(_CHUNK):
            xyz_chunks.append(np.column_stack(
                [np.asarray(pts.x), np.asarray(pts.y), np.asarray(pts.z)]
            ).astype(np.float64))
            if has_time:
                t_chunks.append(np.asarray(pts.gps_time, dtype=np.float64))
            if has_rgb:
                rgb_chunks.append(np.column_stack(
                    [np.asarray(pts.red), np.asarray(pts.green), np.asarray(pts.blue)]
                ))
            if has_intensity:
                i_chunks.append(np.asarray(pts.intensity))

    xyz = np.concatenate(xyz_chunks) if xyz_chunks else np.zeros((0, 3))
    rgb = None
    if has_rgb:
        rgb16 = np.concatenate(rgb_chunks)
        if np.any(rgb16):
            rgb = (rgb16 >> 8).astype(np.uint8) if rgb16.max() > 255 else rgb16.astype(np.uint8)

    points = PointSet(
        xyz=xyz,
        gps_time=np.concatenate(t_chunks) if has_time else None,
        intensity=np.concatenate(i_chunks) if has_intensity else None,
        rgb=rgb,
    )

    if max_points is not None and points.n > max_points:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(points.n, size=max_points, replace=False))
        points = points.subset(idx)

    return points


def save_las(points: PointSet, path: str) -> None:
    """Write a PointSet to LAS point format 3 at 0.1 mm scale."""
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = [0.0001, 0.0001, 0.0001]
    header.offsets = [0.0, 0.0, 0.0]

    las = laspy.LasData(header)
    las.x = points.xyz[:, 0]
    las.y = points.xyz[:, 1]
    las.z = points.xyz[:, 2]
    if points.gps_time is not None:
        las.gps_time = points.gps_time
    if points.intensity is not None:
        las.intensity = np.asarray(points.intensity).astype(np.uint16)
    if points.rgb is not None:
        las.red = points.rgb[:, 0].astype(np.uint16) << 8
        las.green = points.rgb[:, 1].astype(np.uint16) << 8
        las.blue = points.rgb[:, 2].astype(np.uint16) << 8
    las.write(path)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/rscene/test_io_las.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/rscene/io tests/rscene/test_io_las.py
git commit -m "feat(io): LAS adapter retaining all point attributes"
```

---

### Task 12: CLI and report

**Files:**
- Create: `src/rscene/cli.py`
- Create: `tests/rscene/test_cli.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `main(argv: list[str] | None = None) -> int`, writing `scene.json` and `report.md` into the output directory.

- [ ] **Step 1: Write the failing test**

Create `tests/rscene/test_cli.py`:

```python
import json

import numpy as np
import pytest

from rscene.core.points import PointSet
from rscene.core.prim import Box

pytest.importorskip("laspy")

from rscene.cli import main
from rscene.io.las import save_las


def _write_scan(tmp_path):
    pts = np.concatenate([
        Box("floor", (0, 0, 0), (2, 2, 0)).sample_surface(0.01, faces=("z+",)),
        Box("wall", (0, 0, 0), (0, 2, 2.5)).sample_surface(0.01, faces=("x+",)),
    ])
    path = tmp_path / "scan.las"
    save_las(PointSet(xyz=pts), str(path))
    return str(path)


def test_cli_writes_scene_and_report(tmp_path):
    scan = _write_scan(tmp_path)
    out = tmp_path / "out"

    assert main(["patches", scan, str(out)]) == 0
    assert (out / "scene.json").exists()
    assert (out / "report.md").exists()


def test_scene_json_records_patches_and_provenance(tmp_path):
    scan = _write_scan(tmp_path)
    out = tmp_path / "out"
    main(["patches", scan, str(out)])

    payload = json.loads((out / "scene.json").read_text())
    assert len(payload["patches"]) >= 2
    assert payload["provenance"]["scan_sha256"]
    assert payload["provenance"]["config"]["tau_fit_m"] == 0.003


def test_report_states_the_unassigned_count(tmp_path):
    scan = _write_scan(tmp_path)
    out = tmp_path / "out"
    main(["patches", scan, str(out)])

    report = (out / "report.md").read_text()
    assert "Unassigned points" in report


def test_two_runs_produce_identical_scene_json(tmp_path):
    scan = _write_scan(tmp_path)
    a, b = tmp_path / "a", tmp_path / "b"
    main(["patches", scan, str(a)])
    main(["patches", scan, str(b)])

    assert (a / "scene.json").read_text() == (b / "scene.json").read_text()


def test_missing_input_exits_nonzero(tmp_path):
    assert main(["patches", str(tmp_path / "nope.las"), str(tmp_path / "out")]) == 2
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/rscene/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rscene.cli'`

- [ ] **Step 3: Implement `cli.py`**

Create `src/rscene/cli.py`:

```python
"""Command-line entry point.

    rscene patches <scan.las> <out_dir> [--max-points N] [--seed N]

Writes scene.json (the source of truth) and report.md (human-readable).
Determinism note: the provenance timestamp is intentionally derived from the
scan's content hash rather than the wall clock, so two runs of the same scan
produce byte-identical output.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from . import __version__
from .config import merged_config
from .core.graph import coplanarity_classes, patch_adjacency
from .core.level import estimate_frame
from .core.normals import estimate_normals
from .core.patches import extract_patches, unassigned_count
from .core.scene import Provenance, Scene, scene_to_json


def _report(scene: Scene, n_points: int) -> str:
    lines = [
        "# rscene patch extraction report",
        "",
        f"- Scan: `{scene.provenance.scan_path}`",
        f"- SHA256: `{scene.provenance.scan_sha256}`",
        f"- Pipeline version: {scene.provenance.pipeline_version}",
        "",
        "## Points",
        "",
        f"- Loaded: {n_points:,}",
        f"- Unassigned points: {scene.unassigned_points:,} "
        f"({scene.unassigned_points / max(n_points, 1):.1%})",
        "",
        "## Patches",
        "",
        f"- Extracted: {len(scene.patches)}",
        f"- Coplanarity classes: {len(scene.coplanarity_classes)}",
        f"- Adjacent pairs: {len(scene.adjacency)}",
        "",
        "## Frame (measured, not applied)",
        "",
        f"- Gravity axis: {scene.frame.z_axis}",
        f"- XY rotation: {scene.frame.xy_rotation_deg:.2f} deg",
        f"- Floor Z: {scene.frame.floor_z}",
        f"- Ceiling Z: {scene.frame.ceiling_z}",
        "",
        "## Largest patches",
        "",
        "| id | n_points | p95 residual (mm) | normal |",
        "|----|----------|-------------------|--------|",
    ]
    for p in sorted(scene.patches, key=lambda q: -q.n_points)[:15]:
        normal = ", ".join(f"{v:+.3f}" for v in p.normal)
        lines.append(
            f"| {p.patch_id} | {p.n_points:,} | {p.p95_residual_m * 1000:.1f} | {normal} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rscene")
    sub = parser.add_subparsers(dest="command", required=True)

    patches_cmd = sub.add_parser("patches", help="extract planar patches from a scan")
    patches_cmd.add_argument("input", help="path to a LAS/LAZ scan")
    patches_cmd.add_argument("out_dir", help="output directory")
    patches_cmd.add_argument("--max-points", type=int, default=None)
    patches_cmd.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)

    if not os.path.exists(args.input):
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2

    from .io.las import file_sha256, load_las      # imported late: optional dependency

    config = merged_config({"seed": args.seed})
    points = load_las(args.input, max_points=args.max_points, seed=args.seed)
    if points.n < config["normal_k"]:
        print(f"error: only {points.n} points, need at least {config['normal_k']}",
              file=sys.stderr)
        return 2

    normals, curvature = estimate_normals(points.xyz, k=config["normal_k"])
    patch_list, labels = extract_patches(points.xyz, normals, curvature, config)

    scene = Scene(
        provenance=Provenance(
            scan_path=os.path.basename(args.input),
            scan_sha256=file_sha256(args.input),
            pipeline_version=__version__,
            timestamp=f"content:{file_sha256(args.input)[:16]}",
            config=config,
        ),
        frame=estimate_frame(patch_list, config),
        patches=patch_list,
        coplanarity_classes=coplanarity_classes(patch_list, config),
        adjacency=patch_adjacency(patch_list, points.xyz, config),
        unassigned_points=unassigned_count(labels),
        diagnostics={
            "n_input_points": points.n,
            "p95_residual_m": float(np.percentile(
                [p.p95_residual_m for p in patch_list], 95)) if patch_list else 0.0,
        },
    )

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "scene.json"), "w") as handle:
        handle.write(scene_to_json(scene))
    with open(os.path.join(args.out_dir, "report.md"), "w") as handle:
        handle.write(_report(scene, points.n))

    print(f"{len(patch_list)} patches, {scene.unassigned_points:,} unassigned "
          f"-> {args.out_dir}")
    return 0


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/rscene/test_cli.py -v`
Expected: 5 passed

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/pytest tests/rscene tests/golden -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/rscene/cli.py tests/rscene/test_cli.py
git commit -m "feat(cli): rscene patches command writing scene.json and report.md"
```

---

## Definition of done

- `.venv/bin/pytest tests/rscene tests/golden` passes, slow tests included.
- `test_a_75mm_step_survives_as_its_own_patch` passes without any assertion having been loosened.
- Two runs of `rscene patches` on the same scan produce byte-identical `scene.json`.
- `src/rscene/core/` imports nothing beyond numpy and scipy — verify with:
  `grep -rE "^(import|from) (laspy|trimesh|open3d|shapely|cv2|ezdxf)" src/rscene/core/` returns nothing.
- Every point is accounted for: patch members plus `unassigned_points` equals the input count.

## Deferred to later plans

**Plan 2 (parts and critique):** classification, part assembly, wall thickness pairing, rectangularity gate, openings, rooms, solidify, residual critique loop, free-space and trajectory, LLM advisory labelling.

**Plan 3 (exports):** plan DXF, per-wall elevation DXFs, OBJ/FBX, GLB, `room.py` emitter, diagnostic PNGs.

Deleting the old `scripts/` tree happens once Plan 3 reaches parity, per the spec.
