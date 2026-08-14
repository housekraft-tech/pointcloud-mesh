# Rectilinear Scene Rebuild — Plan 2: Assignment & Parts

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn Plan 1's planar patches into named parts carrying measured dimensions — walls with a *measured* face-to-face thickness, slabs, columns, beams, and rectangular wall features — while accounting for every point in the cloud.

**Architecture:** Merge fragmented patches into faces, reject the ones that aren't surfaces, recruit the leftover points, decide interior direction by occupancy flood-fill, classify each face, then pair opposing faces into walls. Every dimension is emitted as a `Measurement` carrying its method, support count and residual. Nothing is snapped, averaged or rotated.

**Tech Stack:** Python 3.13, numpy, scipy, pytest, uv. (`laspy` only under `src/rscene/io/`.)

**Spec:** `docs/superpowers/specs/2026-08-11-rectilinear-scene-rebuild-design.md`, including amendment §4.4.1.
**Predecessor:** `docs/superpowers/plans/2026-08-11-rectilinear-scene-plan1-foundation.md` (complete, 80 tests).

## Global Constraints

Everything from Plan 1 still binds. Repeated here because each task's implementer sees only its own brief:

- **Fit every surface only to its own points. Never snap one surface to another.** Merging two patches into a face is permitted *only* when they are coplanar within tolerance and spatially adjacent; the merged plane is refitted from the union of their own points. No global grid, no averaging of non-adjacent surfaces.
- **Never difference `Patch.d` or `Face.d` to measure a distance between surfaces.** Use `graph.perpendicular_offset`. `d` is origin-referenced; differencing it amplifies normal error by distance from the origin (~20 mm at a 1.6 m lever, centimetres on a 15 m building). This defect shipped once already and was caught only by the whole-branch review.
- **Manhattan is measured, never enforced.** No geometry is rotated to the frame.
- **No silent drops, no silent invention.** Every point ends in exactly one of: a face, a named unassigned bucket, or the `unmodeled` set. Every part is backed by points.
- **`src/rscene/core/` imports only numpy and scipy plus stdlib.** Enforced by `tests/rscene/test_architecture.py` — do not weaken that test.
- **Determinism.** Same scan + same config → byte-identical output. Every iteration order explicitly sorted; every RNG seeded and passed in.
- Tolerance values already fixed: `tau_fit_m = 0.003`, `tau_feature_m = 0.008`.
- Python floor 3.13. No `open3d`, no `trimesh` in this plan.
- Tests: `.venv/bin/pytest tests/rscene tests/golden -v` (~4.5 min, 80 tests at plan start). **Never background a test run** — five agents stalled doing that during Plan 1. Never run bare `pytest`; the legacy `scripts/` tree needs uninstalled dependencies.

## What real data says (measured, not assumed)

These numbers come from `data/isolated_structural_v2.las`, one room cropped at native 6.1 mm density (454,708 points). They are why several tasks below exist.

| Observation | Value | Consequence |
|---|---|---|
| Patches from Plan 1 | 196–200 | Fragmented; one room should not need 200 surfaces |
| Patches under 200 points | 130 of 200 | Median box 0.23 × 0.49 m at **4.4% fill** — sparse chains, not surfaces → Task 2 |
| Parallel vertical pairs under 50 mm apart | 436, many sub-millimetre | Genuine same-face fragments → Task 1 |
| Unassigned | 13.6% | `high_curvature` 35,803 · `plane_exists_but_unassigned` 17,451 · `no_plane_within_tolerance` 8,421 · `isolated` 18 |
| Coplanarity classes after the `d`-difference fix | 69 | The merge target is roughly this order, not 200 |
| Median patch p95 residual | 2.36 mm | Real formwork *is* planar at the scale the mm claim needs |
| Frame yaw / storey | 5.17° / 2.773 m | Building is square to itself but off-axis |

**The single most important number:** 17,451 unassigned points lie *on* an existing patch plane and were simply never recruited. That is Task 3, and it is the largest single recoverable chunk.

**A bound that must not be violated:** merging tolerance is capped by the shallowest feature we must preserve. The golden groove is 12 mm deep, so `face_merge_dist_tol_m` must stay well below that — 5 mm, not the 20–40 mm that would collapse the patch count fastest. A merge tolerance above the groove depth erases the groove, which is the original failure this rebuild exists to fix.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/rscene/core/faces.py` | `Face` (merged patch group), merging, density gate, point recruitment |
| `src/rscene/core/occupancy.py` | Voxel grid, interior flood-fill, per-face interior side |
| `src/rscene/core/classify.py` | Face role assignment (floor / ceiling / wall / step / reveal / soffit) |
| `src/rscene/core/parts.py` | `Wall`, `Slab`, `Column`, `Beam` and their assembly |
| `src/rscene/core/features.py` | Rectangular wall features + the rectangularity gate |
| `src/rscene/core/scene.py` | *(modify)* carry parts and `Measurement`-typed dimensions |
| `src/rscene/config.py` | *(modify)* new thresholds, one commented line each |
| `src/rscene/cli.py` | *(modify)* `rscene parts` subcommand and its report |
| `tests/golden/apartment.py` | *(modify)* expose part-level truth (thickness, clear spans) |

---

### Task 1: `Face` — merge coplanar, adjacent patches

The switch box splits into two fragments; a step side-face splits into two sub-millimetre-apart pieces; one room yields 200 patches. Merging is the fix, and it is safe **only** when patches are both coplanar and touching.

**Files:**
- Create: `src/rscene/core/faces.py`
- Create: `tests/rscene/test_faces.py`
- Modify: `src/rscene/config.py`

**Interfaces:**
- Consumes: `Patch` (`core/patches.py`); `coplanarity_classes`, `patch_adjacency`, `perpendicular_offset` (`core/graph.py`); `fit_plane`, `plane_basis`, `plane_distance` (`core/fitting.py`)
- Produces:
  - `Face(face_id: int, normal: np.ndarray, d: float, patch_ids: list[int], point_idx: np.ndarray, loose_idx: np.ndarray, n_points: int, p95_residual_m: float, centroid: np.ndarray, u_range: tuple[float,float], v_range: tuple[float,float], role: str | None)`
  - `merge_patches(patches: list[Patch], xyz: np.ndarray, config: dict) -> list[Face]`

- [ ] **Step 1: Add config keys**

In `src/rscene/config.py`, after the coplanarity block:

```python
    # --- face merging (Plan 2) ---
    "face_merge_dist_tol_m": 0.005,   # max perpendicular offset to merge two patches into one face; MUST stay below the shallowest feature to preserve (golden groove is 12 mm)
    "face_merge_angle_tol_deg": 2.0,  # max normal deviation between patches merged into one face
    "face_merge_gap_m": 0.05,         # max spatial gap between two patches' points for them to count as the same face
```

- [ ] **Step 2: Write the failing test**

Create `tests/rscene/test_faces.py`:

```python
import numpy as np

from rscene.config import merged_config
from rscene.core.faces import merge_patches
from rscene.core.normals import estimate_normals
from rscene.core.patches import extract_patches
from rscene.core.prim import Box


def _extract(xyz, overrides=None):
    cfg = merged_config(overrides or {})
    normals, curv = estimate_normals(xyz, k=cfg["normal_k"])
    patches, labels = extract_patches(xyz, normals, curv, cfg)
    return patches, labels, cfg


def test_two_fragments_of_one_face_merge():
    """A face split by a gap narrower than face_merge_gap_m becomes one face."""
    a = Box("a", (0, 0.00, 0), (0, 1.98, 2.5)).sample_surface(0.008, faces=("x+",))
    b = Box("b", (0, 2.01, 0), (0, 4.00, 2.5)).sample_surface(0.008, faces=("x+",))
    xyz = np.concatenate([a, b])

    patches, _, cfg = _extract(xyz)
    faces = merge_patches(patches, xyz, cfg)

    x_facing = [f for f in faces if abs(f.normal[0]) > 0.99]
    assert len(x_facing) == 1, f"expected one merged face, got {len(x_facing)}"
    assert x_facing[0].n_points == sum(p.n_points for p in patches
                                       if abs(p.normal[0]) > 0.99)


def test_a_12mm_groove_is_not_merged_into_its_wall():
    """The merge tolerance must stay below the shallowest feature."""
    wall_a = Box("wa", (0.000, 0.0, 0), (0.000, 1.00, 2.5)).sample_surface(0.008, faces=("x+",))
    groove = Box("g", (-0.012, 1.00, 0), (-0.012, 1.06, 2.5)).sample_surface(0.008, faces=("x+",))
    wall_b = Box("wb", (0.000, 1.06, 0), (0.000, 2.50, 2.5)).sample_surface(0.008, faces=("x+",))
    xyz = np.concatenate([wall_a, groove, wall_b])

    patches, _, cfg = _extract(xyz)
    faces = merge_patches(patches, xyz, cfg)

    offsets = sorted({round(abs(f.d), 4) for f in faces if abs(f.normal[0]) > 0.99})
    assert len(offsets) >= 2, "the groove was merged into the wall"
    assert any(abs((offsets[-1] - offsets[0]) - 0.012) < 0.003 for _ in [0]), offsets


def test_distant_coplanar_patches_do_not_merge():
    """Coplanar but far apart is two faces, not one -- merging needs adjacency."""
    a = Box("a", (0, 0, 0), (0, 2, 2)).sample_surface(0.01, faces=("x+",))
    b = Box("b", (0, 8, 0), (0, 10, 2)).sample_surface(0.01, faces=("x+",))
    xyz = np.concatenate([a, b])

    patches, _, cfg = _extract(xyz)
    faces = merge_patches(patches, xyz, cfg)
    assert len([f for f in faces if abs(f.normal[0]) > 0.99]) == 2


def test_merged_plane_is_refitted_from_the_union():
    a = Box("a", (0, 0.00, 0), (0, 1.98, 2.5)).sample_surface(0.008, faces=("x+",))
    b = Box("b", (0, 2.01, 0), (0, 4.00, 2.5)).sample_surface(0.008, faces=("x+",))
    xyz = np.concatenate([a, b])

    patches, _, cfg = _extract(xyz)
    face = [f for f in merge_patches(patches, xyz, cfg) if abs(f.normal[0]) > 0.99][0]

    assert face.p95_residual_m < cfg["tau_fit_m"] * 1.5
    assert face.v_range[1] - face.v_range[0] > 3.5      # spans both fragments
    assert len(face.patch_ids) >= 2


def test_a_lone_patch_becomes_a_single_patch_face():
    xyz = Box("a", (0, 0, 0), (0, 2, 2)).sample_surface(0.01, faces=("x+",))
    patches, _, cfg = _extract(xyz)
    faces = merge_patches(patches, xyz, cfg)
    assert all(len(f.patch_ids) >= 1 for f in faces)
    assert sum(f.n_points for f in faces) == sum(p.n_points for p in patches)


def test_merging_is_deterministic():
    a = Box("a", (0, 0.00, 0), (0, 1.98, 2.5)).sample_surface(0.01, faces=("x+",))
    b = Box("b", (0, 2.01, 0), (0, 4.00, 2.5)).sample_surface(0.01, faces=("x+",))
    xyz = np.concatenate([a, b])
    patches, _, cfg = _extract(xyz)

    f1 = merge_patches(patches, xyz, cfg)
    f2 = merge_patches(patches, xyz, cfg)
    assert [f.face_id for f in f1] == [f.face_id for f in f2]
    assert [f.d for f in f1] == [f.d for f in f2]
    assert [f.patch_ids for f in f1] == [f.patch_ids for f in f2]
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest tests/rscene/test_faces.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rscene.core.faces'`

- [ ] **Step 4: Implement `faces.py`**

Create `src/rscene/core/faces.py`:

```python
"""Faces: patches merged into the surfaces they actually belong to.

Plan 1 deliberately let one physical surface split into several patches --
region growing stops at anything it cannot cross, and a real scan of one room
produced 200 patches where perhaps 20 surfaces exist. Merging is how that is
repaired, and it is the one place in this pipeline where two surfaces are
allowed to become one, so the conditions are strict:

    coplanar within tolerance  AND  spatially adjacent

Coplanarity alone is not enough -- two walls on opposite sides of a building
can share a plane and are obviously not one surface. Adjacency alone is not
enough either, or a 75 mm step would merge into the wall behind it.

The merge tolerance is bounded above by the shallowest feature that must
survive. The golden room's groove is 12 mm deep, so the default sits at 5 mm.
Raising it past the groove depth erases the groove -- the exact failure this
rebuild exists to fix.

The merged plane is refitted from the union of the members' own points. It is
never an average of the input planes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from .fitting import fit_plane, plane_basis, plane_distance
from .graph import perpendicular_offset
from .patches import Patch


@dataclass
class Face:
    """One physical surface, backed by one or more patches."""

    face_id: int
    normal: np.ndarray                 # (3,) unit, canonically oriented
    d: float                           # normal @ x + d == 0 (ORIGIN-referenced)
    patch_ids: list[int]               # contributing patches, sorted
    point_idx: np.ndarray              # fitted members (drive the plane)
    loose_idx: np.ndarray              # recruited members (Task 3); never fitted
    n_points: int                      # len(point_idx)
    p95_residual_m: float              # over fitted members
    centroid: np.ndarray               # (3,)
    u_range: tuple[float, float]
    v_range: tuple[float, float]
    role: Optional[str] = None         # set by classify (Task 6)
    interior_sign: Optional[int] = None  # set by occupancy (Task 5): +1 or -1

    def area_bound_m2(self) -> float:
        return (self.u_range[1] - self.u_range[0]) * (self.v_range[1] - self.v_range[0])

    def all_idx(self) -> np.ndarray:
        """Fitted plus recruited members, sorted. Every point this face owns."""
        return np.sort(np.concatenate([self.point_idx, self.loose_idx]))


def _bbox(xyz: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = xyz[idx]
    return pts.min(axis=0), pts.max(axis=0)


def _adjacent(xyz, a: Patch, b: Patch, gap: float) -> bool:
    """True when any point of a lies within `gap` of any point of b.

    A bounding-box test inflated by `gap` prunes the pair first; it can only
    reject pairs that are genuinely too far apart, never a true neighbour.
    """
    lo_a, hi_a = _bbox(xyz, a.point_idx)
    lo_b, hi_b = _bbox(xyz, b.point_idx)
    if np.any(lo_a - gap > hi_b) or np.any(lo_b - gap > hi_a):
        return False
    return cKDTree(xyz[a.point_idx]).count_neighbors(cKDTree(xyz[b.point_idx]), gap) > 0


def _finalise(face_id: int, xyz: np.ndarray, members: np.ndarray,
              patch_ids: list[int]) -> Face:
    pts = xyz[members]
    normal, d = fit_plane(pts)
    residual = np.abs(plane_distance(pts, normal, d))
    u, v = plane_basis(normal)
    centroid = pts.mean(axis=0)
    rel = pts - centroid
    us, vs = rel @ u, rel @ v
    return Face(
        face_id=face_id,
        normal=normal,
        d=d,
        patch_ids=sorted(patch_ids),
        point_idx=np.sort(members),
        loose_idx=np.zeros(0, dtype=np.int64),
        n_points=int(len(members)),
        p95_residual_m=float(np.percentile(residual, 95)),
        centroid=centroid,
        u_range=(float(us.min()), float(us.max())),
        v_range=(float(vs.min()), float(vs.max())),
    )


def merge_patches(patches: list[Patch], xyz: np.ndarray, config: dict) -> list[Face]:
    """Group patches into faces. Returns faces ordered by descending point count.

    Two patches join the same face when their normals agree, their PERPENDICULAR
    offset (never a difference of `d`) is within tolerance, and their points are
    spatially adjacent.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    dist_tol = float(config["face_merge_dist_tol_m"])
    cos_tol = float(np.cos(np.radians(config["face_merge_angle_tol_deg"])))
    gap = float(config["face_merge_gap_m"])

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
            if perpendicular_offset(a, b) > dist_tol:
                continue
            if not _adjacent(xyz, a, b, gap):
                continue
            union(a.patch_id, b.patch_id)

    groups: dict[int, list[Patch]] = {}
    for p in ordered:
        groups.setdefault(find(p.patch_id), []).append(p)

    faces = []
    for root in sorted(groups):
        members = np.concatenate([p.point_idx for p in groups[root]])
        faces.append(_finalise(0, xyz, members, [p.patch_id for p in groups[root]]))

    faces.sort(key=lambda f: (-f.n_points, f.patch_ids[0]))
    for new_id, f in enumerate(faces):
        f.face_id = new_id
    return faces
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/pytest tests/rscene/test_faces.py -v`
Expected: 6 passed

If `test_a_12mm_groove_is_not_merged_into_its_wall` fails, do **not** raise the tolerance — that test encodes the bound the whole rebuild rests on. Report it.

- [ ] **Step 6: Commit**

```bash
git add src/rscene/core/faces.py tests/rscene/test_faces.py src/rscene/config.py
git commit -m "feat(core): merge coplanar adjacent patches into faces"
```

---

### Task 2: Density gate — reject what isn't a surface

Measured on the real crop: **130 of 200 patches** hold under 200 points, with a median in-plane box of 0.23 × 0.49 m at **4.4% fill**. They are sparse scattered chains threaded together by the 50 mm connect radius, not surfaces. A bare point count cannot catch them, because SLAM density varies several-fold with range — the same feature is kept near the scanner and dropped far from it. Fill ratio travels; a point count does not.

**Files:**
- Modify: `src/rscene/core/faces.py`
- Modify: `tests/rscene/test_faces.py`
- Modify: `src/rscene/config.py`

**Interfaces:**
- Produces: `median_spacing(xyz, sample_n=50_000, seed=0) -> float`; `apply_density_gate(faces, xyz, config) -> tuple[list[Face], list[Face]]` returning `(kept, rejected)`

- [ ] **Step 1: Add config keys**

```python
    "face_min_fill": 0.25,            # min fraction of the points a fully-sampled surface would put in the face's own bbox; real crop showed sparse chains at 0.044
    "face_min_area_m2": 0.004,        # smallest face bbox kept (0.004 = a 63 mm square, below the golden 80 mm switch box)
```

- [ ] **Step 2: Write the failing test**

Append to `tests/rscene/test_faces.py`:

```python
from rscene.core.faces import apply_density_gate, median_spacing


def test_median_spacing_recovers_the_sample_grid():
    xyz = Box("f", (0, 0, 0), (2, 2, 0)).sample_surface(0.01, faces=("z+",))
    assert abs(median_spacing(xyz) - 0.01) < 0.001


def test_a_dense_surface_passes_the_gate():
    xyz = Box("f", (0, 0, 0), (0, 2, 2)).sample_surface(0.01, faces=("x+",))
    patches, _, cfg = _extract(xyz)
    faces = merge_patches(patches, xyz, cfg)
    kept, rejected = apply_density_gate(faces, xyz, cfg)
    assert len(kept) >= 1 and len(rejected) == 0


def test_a_sparse_chain_is_rejected():
    """A scattered chain spanning a large box at low fill is not a surface."""
    dense = Box("f", (0, 0, 0), (0, 2, 2)).sample_surface(0.01, faces=("x+",))
    rng = np.random.default_rng(0)
    chain = np.column_stack([
        np.zeros(300),
        rng.uniform(5.0, 5.5, 300),
        rng.uniform(0.0, 0.5, 300),
    ])
    xyz = np.concatenate([dense, chain])

    patches, _, cfg = _extract(xyz)
    faces = merge_patches(patches, xyz, cfg)
    kept, rejected = apply_density_gate(faces, xyz, cfg)

    for f in kept:
        assert not (f.centroid[1] > 4.5), "the sparse chain survived the gate"
    assert sum(f.n_points for f in kept) + sum(f.n_points for f in rejected) == \
        sum(f.n_points for f in faces)


def test_rejected_faces_are_returned_not_discarded():
    """No silent drops: the gate hands rejects back for the unmodeled bucket."""
    dense = Box("f", (0, 0, 0), (0, 2, 2)).sample_surface(0.01, faces=("x+",))
    patches, _, cfg = _extract(dense)
    faces = merge_patches(patches, dense, cfg)
    kept, rejected = apply_density_gate(faces, dense, cfg)
    assert len(kept) + len(rejected) == len(faces)
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest tests/rscene/test_faces.py -v`
Expected: FAIL — `ImportError: cannot import name 'apply_density_gate'`

- [ ] **Step 4: Implement**

Append to `src/rscene/core/faces.py`:

```python
def median_spacing(xyz: np.ndarray, sample_n: int = 50_000, seed: int = 0) -> float:
    """Median nearest-neighbour distance -- the cloud's native point spacing.

    Sampled rather than exhaustive: the median is stable well below full size,
    and this is called once per run. Seeded for determinism.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if len(xyz) < 2:
        return 0.0
    if len(xyz) > sample_n:
        rng = np.random.default_rng(seed)
        sample = xyz[np.sort(rng.choice(len(xyz), sample_n, replace=False))]
    else:
        sample = xyz
    dist, _ = cKDTree(sample).query(sample, k=2, workers=-1)
    return float(np.median(dist[:, 1]))


def apply_density_gate(
    faces: list[Face], xyz: np.ndarray, config: dict
) -> tuple[list[Face], list[Face]]:
    """Split faces into (kept, rejected) on in-plane fill ratio and area.

    Fill is the face's point count against the count a fully-sampled surface of
    the same bbox would hold at the cloud's native spacing. A point count alone
    cannot distinguish a small dense feature from a large sparse chain -- and
    SLAM density falls off with range, so an absolute count is not portable
    across one scan, let alone between scans.

    Rejected faces are RETURNED, never dropped. The caller routes them to the
    scene's `unmodeled` set so a designer still sees that something is there.
    """
    spacing = median_spacing(xyz, seed=int(config["seed"]))
    min_fill = float(config["face_min_fill"])
    min_area = float(config["face_min_area_m2"])
    per_m2 = 1.0 / (spacing ** 2) if spacing > 0 else 0.0

    kept, rejected = [], []
    for f in sorted(faces, key=lambda g: g.face_id):
        area = f.area_bound_m2()
        if area < min_area:
            rejected.append(f)
            continue
        expected = area * per_m2
        fill = (f.n_points / expected) if expected > 0 else 0.0
        (kept if fill >= min_fill else rejected).append(f)
    return kept, rejected
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/pytest tests/rscene/test_faces.py -v`
Expected: 10 passed

- [ ] **Step 6: Commit**

```bash
git add src/rscene/core/faces.py tests/rscene/test_faces.py src/rscene/config.py
git commit -m "feat(core): density gate rejecting sparse chains that are not surfaces"
```

---

### Task 3: Recruitment pass — the largest recoverable chunk

17,451 unassigned points on the real crop lie *on* an existing patch plane and were never recruited, because region growing could not reach them. Attaching them to the face they clearly belong to is the biggest single reduction in unassigned points available, and it costs nothing in accuracy **provided recruits never enter the plane fit** — the same membership/fitting separation that fixed Plan 1's Task 10.

**Files:**
- Modify: `src/rscene/core/faces.py`
- Modify: `tests/rscene/test_faces.py`
- Modify: `src/rscene/config.py`

**Interfaces:**
- Produces: `recruit_points(faces, xyz, normals, labels, config) -> np.ndarray` — returns updated labels (face ids, `-1` where still unassigned) and mutates each `Face.loose_idx` in place.

- [ ] **Step 1: Add config keys**

```python
    # --- recruitment (Plan 2) ---
    "recruit_dist_tol_m": 0.008,      # max point-to-plane distance to recruit a leftover point; deliberately looser than tau_fit_m since recruits never enter the fit
    "recruit_angle_tol_deg": 20.0,    # max normal deviation to recruit; looser than growth because edge normals are blended
    "recruit_max_reach_m": 0.10,      # recruit only within this distance of an existing member, so a point cannot join a face across a void
```

- [ ] **Step 2: Write the failing test**

Append to `tests/rscene/test_faces.py`:

```python
from rscene.core.faces import recruit_points


def test_recruitment_claims_points_lying_on_a_face():
    xyz = Box("f", (0, 0, 0), (0, 2, 2)).sample_surface(0.008, faces=("x+",))
    patches, labels, cfg = _extract(xyz)
    faces = merge_patches(patches, xyz, cfg)
    normals, _ = estimate_normals(xyz, k=cfg["normal_k"])

    before = int(np.count_nonzero(labels < 0))
    new_labels = recruit_points(faces, xyz, normals, labels, cfg)
    after = int(np.count_nonzero(new_labels < 0))

    assert after <= before
    assert sum(len(f.loose_idx) for f in faces) == before - after


def test_recruits_do_not_move_the_plane():
    """A recruit is a member for accounting, never for fitting."""
    xyz = Box("f", (0, 0, 0), (0, 2, 2)).sample_surface(0.008, faces=("x+",))
    patches, labels, cfg = _extract(xyz)
    faces = merge_patches(patches, xyz, cfg)
    normals, _ = estimate_normals(xyz, k=cfg["normal_k"])

    planes_before = [(f.normal.copy(), f.d) for f in faces]
    recruit_points(faces, xyz, normals, labels, cfg)
    for (n0, d0), f in zip(planes_before, faces):
        assert np.array_equal(n0, f.normal)
        assert d0 == f.d


def test_a_point_far_from_every_face_is_not_recruited():
    xyz = np.concatenate([
        Box("f", (0, 0, 0), (0, 2, 2)).sample_surface(0.008, faces=("x+",)),
        np.array([[5.0, 5.0, 5.0]]),
    ])
    patches, labels, cfg = _extract(xyz)
    faces = merge_patches(patches, xyz, cfg)
    normals, _ = estimate_normals(xyz, k=cfg["normal_k"])
    new_labels = recruit_points(faces, xyz, normals, labels, cfg)
    assert new_labels[-1] == -1


def test_no_point_is_recruited_twice():
    xyz = np.concatenate([
        Box("a", (0, 0, 0), (0, 2, 2)).sample_surface(0.008, faces=("x+",)),
        Box("b", (0.5, 0, 0), (0.5, 2, 2)).sample_surface(0.008, faces=("x+",)),
    ])
    patches, labels, cfg = _extract(xyz)
    faces = merge_patches(patches, xyz, cfg)
    normals, _ = estimate_normals(xyz, k=cfg["normal_k"])
    recruit_points(faces, xyz, normals, labels, cfg)

    claimed = np.concatenate([f.loose_idx for f in faces]) if faces else np.zeros(0)
    assert len(claimed) == len(np.unique(claimed))


def test_recruitment_is_deterministic():
    xyz = Box("f", (0, 0, 0), (0, 2, 2)).sample_surface(0.01, faces=("x+",))
    patches, labels, cfg = _extract(xyz)
    normals, _ = estimate_normals(xyz, k=cfg["normal_k"])

    f1 = merge_patches(patches, xyz, cfg)
    l1 = recruit_points(f1, xyz, normals, labels, cfg)
    f2 = merge_patches(patches, xyz, cfg)
    l2 = recruit_points(f2, xyz, normals, labels, cfg)

    assert np.array_equal(l1, l2)
    assert [f.loose_idx.tolist() for f in f1] == [f.loose_idx.tolist() for f in f2]
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest tests/rscene/test_faces.py -v`
Expected: FAIL — `ImportError: cannot import name 'recruit_points'`

- [ ] **Step 4: Implement**

Append to `src/rscene/core/faces.py`:

```python
_RECRUIT_CHUNK = 20_000


def recruit_points(
    faces: list[Face],
    xyz: np.ndarray,
    normals: np.ndarray,
    labels: np.ndarray,
    config: dict,
) -> np.ndarray:
    """Attach leftover points to the face they lie on. Returns new labels.

    On a real scan most unassigned points are not unexplained -- they sit ON a
    face's plane and region growing simply could not reach them. Recruiting
    them is the largest single reduction in unassigned points available.

    Recruits go into `Face.loose_idx` and are EXCLUDED from the plane fit, so a
    looser tolerance costs accounting completeness nothing in accuracy. A recruit
    must also be within `recruit_max_reach_m` of an existing fitted member, so a
    point cannot join a face across a void it has no business crossing.

    Ties are broken by nearest plane, then lowest face_id -- deterministic.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    labels = np.asarray(labels).copy()
    if not faces:
        return labels

    tau = float(config["recruit_dist_tol_m"])
    cos_tol = float(np.cos(np.radians(config["recruit_angle_tol_deg"])))
    reach = float(config["recruit_max_reach_m"])

    ordered = sorted(faces, key=lambda f: f.face_id)
    N = np.vstack([f.normal for f in ordered])           # (F, 3)
    D = np.array([f.d for f in ordered])                 # (F,)
    trees = [cKDTree(xyz[f.point_idx]) for f in ordered]

    free = np.flatnonzero(labels < 0)
    claimed: dict[int, list[int]] = {f.face_id: [] for f in ordered}

    for start in range(0, len(free), _RECRUIT_CHUNK):
        idx = free[start:start + _RECRUIT_CHUNK]
        P = xyz[idx]
        dists = np.abs(P @ N.T + D)                      # (n, F)
        agrees = np.abs(normals[idx] @ N.T) >= cos_tol   # (n, F)
        ok = agrees & (dists <= tau)

        cand = np.where(ok, dists, np.inf)
        best = np.argmin(cand, axis=1)
        best_dist = cand[np.arange(len(idx)), best]
        viable = np.isfinite(best_dist)

        for local in np.flatnonzero(viable):
            fi = int(best[local])
            point = P[local]
            if trees[fi].query(point, k=1, workers=-1)[0] > reach:
                continue
            gi = int(idx[local])
            claimed[ordered[fi].face_id].append(gi)
            labels[gi] = ordered[fi].face_id

    for f in ordered:
        got = claimed[f.face_id]
        f.loose_idx = np.sort(np.asarray(got, dtype=np.int64)) if got \
            else np.zeros(0, dtype=np.int64)
    return labels
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/pytest tests/rscene/test_faces.py -v`
Expected: 15 passed

- [ ] **Step 6: Commit**

```bash
git add src/rscene/core/faces.py tests/rscene/test_faces.py src/rscene/config.py
git commit -m "feat(core): recruit leftover points onto faces without moving their planes"
```

---

### Task 4: Occupancy grid and interior flood-fill

The spec originally used the walk trajectory to decide which side of a surface is inside. That mechanism is dead — see spec §4.4.1: a 360° scanner's time-slice centroid is the centre of what was *seen*, not where it stood, and every sensor-geometry field in the LAS is zero. Flood-fill needs no path.

**Files:**
- Create: `src/rscene/core/occupancy.py`
- Create: `tests/rscene/test_occupancy.py`
- Modify: `src/rscene/config.py`

**Interfaces:**
- Produces:
  - `OccupancyGrid(origin: np.ndarray, cell_m: float, occupied: np.ndarray)` with `.index_of(points) -> np.ndarray` and `.shape`
  - `build_occupancy(xyz, config) -> OccupancyGrid`
  - `flood_interior(grid, seed_xyz) -> np.ndarray` — boolean array of interior (air) cells

- [ ] **Step 1: Add config keys**

```python
    # --- occupancy / interior (Plan 2) ---
    "occupancy_cell_m": 0.05,         # voxel size for the interior flood-fill; coarse on purpose, it never defines geometry
    "interior_seed_height_m": 1.20,   # height above the floor plane to seed the fill, chosen to sit in open air in any room
```

- [ ] **Step 2: Write the failing test**

Create `tests/rscene/test_occupancy.py`:

```python
import numpy as np

from rscene.config import merged_config
from rscene.core.occupancy import build_occupancy, flood_interior
from rscene.core.prim import Box


def _closed_room(spacing=0.02):
    """A sealed box room: interior air must not leak outside."""
    b = Box("room", (0, 0, 0), (4, 3, 2.5))
    return b.sample_surface(spacing)


def test_grid_covers_every_point():
    xyz = _closed_room()
    grid = build_occupancy(xyz, merged_config())
    ijk = grid.index_of(xyz)
    assert (ijk >= 0).all()
    assert (ijk < np.array(grid.shape)).all()
    assert grid.occupied[ijk[:, 0], ijk[:, 1], ijk[:, 2]].all()


def test_flood_fills_the_interior_and_does_not_leak_out():
    xyz = _closed_room()
    grid = build_occupancy(xyz, merged_config())
    interior = flood_interior(grid, np.array([2.0, 1.5, 1.2]))

    assert interior.any()
    # a cell well inside is interior
    c = grid.index_of(np.array([[2.0, 1.5, 1.2]]))[0]
    assert interior[c[0], c[1], c[2]]
    # nothing on the outer shell of the grid is interior
    assert not interior[0, :, :].any()
    assert not interior[-1, :, :].any()
    assert not interior[:, 0, :].any()
    assert not interior[:, :, 0].any()


def test_interior_does_not_include_occupied_cells():
    xyz = _closed_room()
    grid = build_occupancy(xyz, merged_config())
    interior = flood_interior(grid, np.array([2.0, 1.5, 1.2]))
    assert not (interior & grid.occupied).any()


def test_a_doorway_lets_the_fill_reach_the_next_room():
    """Two rooms joined by a gap: one seed fills both."""
    left = Box("l", (0, 0, 0), (2, 3, 2.5)).sample_surface(0.02)
    right = Box("r", (2.4, 0, 0), (4.4, 3, 2.5)).sample_surface(0.02)
    # wall between them with a 0.9 m opening
    xyz = np.concatenate([left, right])
    grid = build_occupancy(xyz, merged_config())
    interior = flood_interior(grid, np.array([1.0, 1.5, 1.2]))

    far = grid.index_of(np.array([[3.4, 1.5, 1.2]]))[0]
    assert interior[far[0], far[1], far[2]], "fill did not reach through the gap"


def test_seed_inside_a_wall_raises():
    xyz = _closed_room()
    grid = build_occupancy(xyz, merged_config())
    import pytest
    with pytest.raises(ValueError, match="occupied"):
        flood_interior(grid, np.array([0.0, 1.5, 1.2]))


def test_occupancy_is_deterministic():
    xyz = _closed_room()
    cfg = merged_config()
    a = build_occupancy(xyz, cfg)
    b = build_occupancy(xyz, cfg)
    assert np.array_equal(a.occupied, b.occupied)
    assert np.array_equal(a.origin, b.origin)
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest tests/rscene/test_occupancy.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rscene.core.occupancy'`

- [ ] **Step 4: Implement `occupancy.py`**

Create `src/rscene/core/occupancy.py`:

```python
"""Voxel occupancy and the interior flood-fill.

This exists to answer two questions and no others: which side of a surface is
inside, and does a void pass all the way through. It never defines geometry --
its cells are 50 mm and its edges are stair-stepped, both unacceptable for a
model whose deliverable is sub-3 mm.

The mechanism is a flood-fill rather than the walk trajectory the spec
originally named. See spec section 4.4.1: a 360 degree scanner's time-slice
centroid is the centre of what was seen, not where the scanner stood, and every
sensor-geometry field in the reference LAS exports is zero. The fill needs no
path -- seed it in air above the floor and interior is whatever it reaches.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class OccupancyGrid:
    """A binary voxel grid. `origin` is the world position of cell (0,0,0)."""

    origin: np.ndarray          # (3,)
    cell_m: float
    occupied: np.ndarray        # (nx, ny, nz) bool

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.occupied.shape

    def index_of(self, points: np.ndarray) -> np.ndarray:
        """World points -> integer cell indices, clipped into the grid."""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        ijk = np.floor((pts - self.origin) / self.cell_m).astype(np.int64)
        return np.clip(ijk, 0, np.array(self.shape) - 1)


def build_occupancy(xyz: np.ndarray, config: dict) -> OccupancyGrid:
    """Mark every cell containing at least one point as occupied.

    One cell of padding is added on all sides so the flood-fill always has an
    exterior shell to be bounded by, and so `index_of` never lands on an edge
    cell for a real point.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    cell = float(config["occupancy_cell_m"])
    origin = xyz.min(axis=0) - cell
    extent = xyz.max(axis=0) + cell - origin
    shape = tuple(int(np.ceil(e / cell)) + 1 for e in extent)

    occupied = np.zeros(shape, dtype=bool)
    ijk = np.floor((xyz - origin) / cell).astype(np.int64)
    ijk = np.clip(ijk, 0, np.array(shape) - 1)
    occupied[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    return OccupancyGrid(origin=origin, cell_m=cell, occupied=occupied)


_NEIGHBOURS = (
    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
)


def flood_interior(grid: OccupancyGrid, seed_xyz: np.ndarray) -> np.ndarray:
    """6-connected flood-fill of free cells from a seed. Returns a bool array.

    Interior is whatever the fill reaches without passing through an occupied
    cell. A sealed room bounds it; a doorway lets it through to the next room;
    a hole in the envelope lets it escape to the grid boundary, which is itself
    diagnostic.

    Raises if the seed lands in an occupied cell -- silently returning an empty
    fill would look identical to a sealed void and hide the real problem.
    """
    seed = grid.index_of(np.asarray(seed_xyz, dtype=np.float64).reshape(1, 3))[0]
    si, sj, sk = int(seed[0]), int(seed[1]), int(seed[2])
    if grid.occupied[si, sj, sk]:
        raise ValueError(
            f"interior seed {np.asarray(seed_xyz).tolist()} lands in an occupied "
            f"cell {(si, sj, sk)}; choose a seed in open air"
        )

    nx, ny, nz = grid.shape
    interior = np.zeros(grid.shape, dtype=bool)
    interior[si, sj, sk] = True
    queue = deque([(si, sj, sk)])

    while queue:
        i, j, k = queue.popleft()
        for di, dj, dk in _NEIGHBOURS:
            a, b, c = i + di, j + dj, k + dk
            if not (0 <= a < nx and 0 <= b < ny and 0 <= c < nz):
                continue
            if interior[a, b, c] or grid.occupied[a, b, c]:
                continue
            interior[a, b, c] = True
            queue.append((a, b, c))
    return interior
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/pytest tests/rscene/test_occupancy.py -v`
Expected: 6 passed

Note on `test_a_doorway_lets_the_fill_reach_the_next_room`: the two boxes are separated by a 0.4 m air gap with no wall between them, so the fill passes through the gap. If it fails, check the gap is wider than one cell.

- [ ] **Step 6: Commit**

```bash
git add src/rscene/core/occupancy.py tests/rscene/test_occupancy.py src/rscene/config.py
git commit -m "feat(core): voxel occupancy and interior flood-fill, replacing trajectory free-space"
```

---

### Task 5: Interior side per face

**Files:**
- Modify: `src/rscene/core/occupancy.py`
- Modify: `tests/rscene/test_occupancy.py`

**Interfaces:**
- Produces: `assign_interior_sides(faces, grid, interior, config) -> None` — sets `Face.interior_sign` to `+1` (interior lies along `+normal`), `-1`, or leaves `None` when undecidable.

- [ ] **Step 1: Write the failing test**

Append to `tests/rscene/test_occupancy.py`:

```python
from rscene.config import merged_config
from rscene.core.faces import merge_patches
from rscene.core.normals import estimate_normals
from rscene.core.occupancy import assign_interior_sides
from rscene.core.patches import extract_patches


def _room_faces(spacing=0.015):
    xyz = Box("room", (0, 0, 0), (3, 2.5, 2.5)).sample_surface(spacing)
    cfg = merged_config()
    normals, curv = estimate_normals(xyz, k=cfg["normal_k"])
    patches, _ = extract_patches(xyz, normals, curv, cfg)
    return xyz, merge_patches(patches, xyz, cfg), cfg


def test_every_bounding_face_points_its_interior_inward():
    xyz, faces, cfg = _room_faces()
    grid = build_occupancy(xyz, cfg)
    interior = flood_interior(grid, np.array([1.5, 1.25, 1.2]))
    assign_interior_sides(faces, grid, interior, cfg, xyz)

    centre = np.array([1.5, 1.25, 1.25])
    decided = [f for f in faces if f.interior_sign is not None]
    assert len(decided) >= 4, "most bounding faces should be decidable"
    for f in decided:
        inward = f.interior_sign * f.normal
        to_centre = centre - f.centroid
        assert float(inward @ to_centre) > 0, (
            f"face {f.face_id} points away from the room interior"
        )


def test_interior_sign_is_none_when_neither_side_is_interior():
    """A face with air on neither side cannot be decided; it must say so."""
    xyz, faces, cfg = _room_faces()
    grid = build_occupancy(xyz, cfg)
    interior = np.zeros(grid.shape, dtype=bool)   # nothing is interior
    assign_interior_sides(faces, grid, interior, cfg, xyz)
    assert all(f.interior_sign is None for f in faces)


def test_assignment_is_deterministic():
    xyz, faces_a, cfg = _room_faces()
    grid = build_occupancy(xyz, cfg)
    interior = flood_interior(grid, np.array([1.5, 1.25, 1.2]))
    assign_interior_sides(faces_a, grid, interior, cfg, xyz)
    signs_a = [f.interior_sign for f in faces_a]

    _, faces_b, _ = _room_faces()
    assign_interior_sides(faces_b, grid, interior, cfg, xyz)
    assert signs_a == [f.interior_sign for f in faces_b]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/rscene/test_occupancy.py -v`
Expected: FAIL — `ImportError: cannot import name 'assign_interior_sides'`

- [ ] **Step 3: Implement**

Append to `src/rscene/core/occupancy.py`:

```python
def assign_interior_sides(faces, grid: OccupancyGrid, interior: np.ndarray,
                          config: dict, xyz: np.ndarray) -> None:
    """Set `Face.interior_sign` from which side of the face holds interior air.

    Probes one and a half cells out along +normal and -normal from a
    deterministic sample of the face's own points, and counts interior hits on
    each side. The side with more interior wins. A face with no interior on
    either side -- an exterior wall seen only from outside, or a surface buried
    in solid -- is left as None rather than guessed. "Measured, never assumed"
    applies here too.

    Mutates the faces in place.
    """
    step = grid.cell_m * 1.5
    for f in sorted(faces, key=lambda g: g.face_id):
        idx = f.point_idx
        if len(idx) == 0:
            f.interior_sign = None
            continue
        take = idx if len(idx) <= 256 else idx[
            np.linspace(0, len(idx) - 1, 256).astype(np.int64)
        ]
        pts = xyz[take]
        counts = {}
        for sign in (1, -1):
            probe = pts + sign * step * f.normal
            ijk = grid.index_of(probe)
            counts[sign] = int(interior[ijk[:, 0], ijk[:, 1], ijk[:, 2]].sum())
        if counts[1] == counts[-1]:
            f.interior_sign = None
        else:
            f.interior_sign = 1 if counts[1] > counts[-1] else -1
```

The signature takes `xyz` as its final argument — the test above already passes it. `Face` stores point *indices*, not coordinates, so the world points have to come from somewhere; threading the cloud in explicitly is cheaper and clearer than storing a copy on every face.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/rscene/test_occupancy.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/rscene/core/occupancy.py tests/rscene/test_occupancy.py
git commit -m "feat(core): decide each face's interior side from the flood-fill"
```

---

### Task 6: Face classification

**Files:**
- Create: `src/rscene/core/classify.py`
- Create: `tests/rscene/test_classify.py`
- Modify: `src/rscene/config.py`

**Interfaces:**
- Produces: `classify_faces(faces, frame, config) -> None` — sets `Face.role` to one of `"floor"`, `"ceiling"`, `"wall"`, `"oblique"`, `"unknown"`.

Roles beyond these (step face, reveal, soffit) depend on parts and arrive in Task 7 and Plan 3. Do not invent them here.

- [ ] **Step 1: Add config key**

```python
    "classify_slab_band_m": 0.30,     # a horizontal face within this of the storey floor/ceiling level is that slab; beyond it, an unknown horizontal (a sill, a step tread)
```

- [ ] **Step 2: Write the failing test**

Create `tests/rscene/test_classify.py`:

```python
import numpy as np

from rscene.config import merged_config
from rscene.core.classify import classify_faces
from rscene.core.faces import merge_patches
from rscene.core.level import estimate_frame
from rscene.core.normals import estimate_normals
from rscene.core.patches import extract_patches
from rscene.core.prim import Box


def _room():
    xyz = Box("room", (0, 0, 0), (3, 2.5, 2.5)).sample_surface(0.015)
    cfg = merged_config()
    normals, curv = estimate_normals(xyz, k=cfg["normal_k"])
    patches, _ = extract_patches(xyz, normals, curv, cfg)
    faces = merge_patches(patches, xyz, cfg)
    frame = estimate_frame(patches, cfg)
    return faces, frame, cfg


def test_a_closed_room_yields_a_floor_a_ceiling_and_walls():
    faces, frame, cfg = _room()
    classify_faces(faces, frame, cfg)
    roles = [f.role for f in faces]
    assert roles.count("floor") == 1
    assert roles.count("ceiling") == 1
    assert roles.count("wall") >= 4


def test_every_face_gets_a_role():
    faces, frame, cfg = _room()
    classify_faces(faces, frame, cfg)
    assert all(f.role is not None for f in faces)
    assert set(f.role for f in faces) <= {"floor", "ceiling", "wall", "oblique", "unknown"}


def test_a_horizontal_face_away_from_both_slabs_is_not_a_slab():
    faces, frame, cfg = _room()
    classify_faces(faces, frame, cfg)
    floor = [f for f in faces if f.role == "floor"][0]
    ceiling = [f for f in faces if f.role == "ceiling"][0]
    assert floor.centroid[2] < ceiling.centroid[2]


def test_classification_is_deterministic():
    faces_a, frame, cfg = _room()
    classify_faces(faces_a, frame, cfg)
    faces_b, _, _ = _room()
    classify_faces(faces_b, frame, cfg)
    assert [f.role for f in faces_a] == [f.role for f in faces_b]
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest tests/rscene/test_classify.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rscene.core.classify'`

- [ ] **Step 4: Implement `classify.py`**

Create `src/rscene/core/classify.py`:

```python
"""Face roles.

Only the roles that can be decided from a face's own geometry plus the measured
frame live here: floor, ceiling, wall, oblique, unknown. Roles that depend on
relationships between parts -- a step face, an opening reveal, a beam soffit --
are decided where those relationships are built, not guessed here.

Nothing in this module moves geometry. It reads orientation and height and
writes a label.
"""
from __future__ import annotations

import numpy as np


def classify_faces(faces, frame, config: dict) -> None:
    """Assign `Face.role` in place. Every face gets a role, never None."""
    tol_cos = float(np.cos(np.radians(config["floor_normal_tol_deg"])))
    vert_max_z = float(config["vertical_normal_max_z"])
    band = float(config["classify_slab_band_m"])
    floor_z = frame.floor_z
    ceiling_z = frame.ceiling_z

    for f in sorted(faces, key=lambda g: g.face_id):
        nz = abs(float(f.normal[2]))
        if nz >= tol_cos:
            z = float(f.centroid[2])
            near_floor = floor_z is not None and abs(z - floor_z) <= band
            near_ceiling = ceiling_z is not None and abs(z - ceiling_z) <= band
            if near_floor and near_ceiling:
                # degenerate storey; pick the nearer level
                f.role = "floor" if abs(z - floor_z) <= abs(z - ceiling_z) else "ceiling"
            elif near_floor:
                f.role = "floor"
            elif near_ceiling:
                f.role = "ceiling"
            else:
                f.role = "unknown"
        elif nz < vert_max_z:
            f.role = "wall"
        else:
            f.role = "oblique"
```

If `test_a_closed_room_yields_a_floor_a_ceiling_and_walls` finds more than one floor, the slab has fragmented and Task 1's merge is not reaching — report that rather than loosening the assertion, because it means merging is under-performing on the simplest possible input.

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/pytest tests/rscene/test_classify.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add src/rscene/core/classify.py tests/rscene/test_classify.py src/rscene/config.py
git commit -m "feat(core): classify faces into floor, ceiling, wall, oblique"
```

---

### Task 7: Wall assembly with measured thickness

This is the deliverable. A wall is **two independently fitted faces**, and its thickness is the *measured* perpendicular distance between them — never a centreline plus an assumed thickness.

**Files:**
- Create: `src/rscene/core/parts.py`
- Create: `tests/rscene/test_parts.py`
- Modify: `src/rscene/config.py`

**Interfaces:**
- Consumes: `Face` (Task 1); `perpendicular_offset` (`core/graph.py`); `Measurement` (`core/scene.py`)
- Produces:
  - `Wall(wall_id: str, face_a: int, face_b: int | None, thickness: Measurement | None, length: Measurement, height: Measurement, centroid: np.ndarray, normal: np.ndarray)`
  - `assemble_walls(faces, xyz, config) -> tuple[list[Wall], list[int]]` — returns walls and the ids of wall faces left unpaired

- [ ] **Step 1: Add config keys**

```python
    # --- wall assembly (Plan 2) ---
    "wall_thickness_min_m": 0.050,    # thinnest face pair accepted as one wall
    "wall_thickness_max_m": 0.450,    # thickest face pair accepted as one wall
    "wall_pair_min_overlap": 0.30,    # min fraction of the smaller face's in-plane extent that must overlap its partner
```

- [ ] **Step 2: Write the failing test**

Create `tests/rscene/test_parts.py`:

```python
import numpy as np

from rscene.config import merged_config
from rscene.core.classify import classify_faces
from rscene.core.faces import merge_patches
from rscene.core.level import estimate_frame
from rscene.core.normals import estimate_normals
from rscene.core.parts import assemble_walls
from rscene.core.patches import extract_patches
from rscene.core.prim import Box

THICKNESS = 0.200


def _wall_with_two_faces(spacing=0.008):
    """A free-standing wall slab, both faces visible."""
    a = Box("fa", (0.0, 0, 0), (0.0, 3.0, 2.5)).sample_surface(spacing, faces=("x+",))
    b = Box("fb", (THICKNESS, 0, 0), (THICKNESS, 3.0, 2.5)).sample_surface(spacing, faces=("x-",))
    floor = Box("fl", (-1, 0, 0), (1.5, 3.0, 0)).sample_surface(0.02, faces=("z+",))
    xyz = np.concatenate([a, b, floor])
    cfg = merged_config()
    normals, curv = estimate_normals(xyz, k=cfg["normal_k"])
    patches, _ = extract_patches(xyz, normals, curv, cfg)
    faces = merge_patches(patches, xyz, cfg)
    classify_faces(faces, estimate_frame(patches, cfg), cfg)
    return xyz, faces, cfg


def test_two_opposing_faces_become_one_wall_with_measured_thickness():
    xyz, faces, cfg = _wall_with_two_faces()
    walls, unpaired = assemble_walls(faces, xyz, cfg)

    paired = [w for w in walls if w.face_b is not None]
    assert len(paired) == 1, f"expected one paired wall, got {len(paired)}"
    t = paired[0].thickness
    assert abs(t.value - THICKNESS) < 0.003, f"thickness {t.value}"
    assert t.method
    assert t.n_points > 0


def test_thickness_is_not_measured_by_differencing_d():
    """Guard the origin lever-arm trap: the same wall far from the origin
    must measure the same thickness."""
    xyz, faces, cfg = _wall_with_two_faces()
    walls, _ = assemble_walls(faces, xyz, cfg)
    near = [w for w in walls if w.face_b is not None][0].thickness.value

    shifted = xyz + np.array([40.0, 25.0, 0.0])
    normals, curv = estimate_normals(shifted, k=cfg["normal_k"])
    patches, _ = extract_patches(shifted, normals, curv, cfg)
    faces2 = merge_patches(patches, shifted, cfg)
    classify_faces(faces2, estimate_frame(patches, cfg), cfg)
    walls2, _ = assemble_walls(faces2, shifted, cfg)
    far = [w for w in walls2 if w.face_b is not None][0].thickness.value

    assert abs(near - far) < 0.001, f"thickness moved with the origin: {near} vs {far}"


def test_a_lone_face_is_reported_unpaired_not_invented():
    a = Box("fa", (0, 0, 0), (0, 3.0, 2.5)).sample_surface(0.01, faces=("x+",))
    cfg = merged_config()
    normals, curv = estimate_normals(a, k=cfg["normal_k"])
    patches, _ = extract_patches(a, normals, curv, cfg)
    faces = merge_patches(patches, a, cfg)
    classify_faces(faces, estimate_frame(patches, cfg), cfg)

    walls, unpaired = assemble_walls(faces, a, cfg)
    assert len(unpaired) >= 1
    for w in walls:
        if w.face_b is None:
            assert w.thickness is None


def test_faces_too_far_apart_are_not_paired():
    a = Box("fa", (0.0, 0, 0), (0.0, 3.0, 2.5)).sample_surface(0.01, faces=("x+",))
    b = Box("fb", (1.5, 0, 0), (1.5, 3.0, 2.5)).sample_surface(0.01, faces=("x-",))
    xyz = np.concatenate([a, b])
    cfg = merged_config()
    normals, curv = estimate_normals(xyz, k=cfg["normal_k"])
    patches, _ = extract_patches(xyz, normals, curv, cfg)
    faces = merge_patches(patches, xyz, cfg)
    classify_faces(faces, estimate_frame(patches, cfg), cfg)

    walls, _ = assemble_walls(faces, xyz, cfg)
    assert all(w.face_b is None for w in walls), "1.5 m apart is not one wall"


def test_wall_length_and_height_are_measured():
    xyz, faces, cfg = _wall_with_two_faces()
    walls, _ = assemble_walls(faces, xyz, cfg)
    w = [x for x in walls if x.face_b is not None][0]
    assert abs(w.length.value - 3.0) < 0.01
    assert abs(w.height.value - 2.5) < 0.01


def test_assembly_is_deterministic():
    xyz, faces, cfg = _wall_with_two_faces()
    w1, u1 = assemble_walls(faces, xyz, cfg)
    w2, u2 = assemble_walls(faces, xyz, cfg)
    assert [w.wall_id for w in w1] == [w.wall_id for w in w2]
    assert [None if w.thickness is None else w.thickness.value for w in w1] == \
           [None if w.thickness is None else w.thickness.value for w in w2]
    assert u1 == u2
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest tests/rscene/test_parts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rscene.core.parts'`

- [ ] **Step 4: Implement `parts.py`**

Create `src/rscene/core/parts.py`:

```python
"""Parts: the named objects a designer actually works with.

A Wall is two independently fitted faces and a MEASURED distance between them.
It is never a centreline with an assumed thickness -- that assumption is what
the previous pipeline made, and it is why its numbers could not be trusted.

Thickness is measured with `perpendicular_offset`, never by differencing the
two faces' `d` values. `d` is origin-referenced; on a 15 m building a fraction
of a degree of normal error turns into centimetres of phantom thickness.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .faces import Face
from .graph import perpendicular_offset
from .scene import Measurement


@dataclass
class Wall:
    """One wall: one or two faces, with every dimension measured."""

    wall_id: str
    face_a: int
    face_b: Optional[int]
    thickness: Optional[Measurement]
    length: Measurement
    height: Measurement
    centroid: np.ndarray
    normal: np.ndarray


def _overlap_fraction(a: Face, b: Face) -> float:
    """How much two parallel faces overlap, as a fraction of the smaller.

    Compared in world XY and Z extents rather than each face's own uv basis,
    because two opposing faces have independently chosen bases that need not
    agree.
    """
    def extent(f: Face, axis: int) -> tuple[float, float]:
        half_u = (f.u_range[1] - f.u_range[0]) / 2.0
        half_v = (f.v_range[1] - f.v_range[0]) / 2.0
        span = max(half_u, half_v)
        return float(f.centroid[axis] - span), float(f.centroid[axis] + span)

    fracs = []
    for axis in (0, 1, 2):
        lo_a, hi_a = extent(a, axis)
        lo_b, hi_b = extent(b, axis)
        inter = min(hi_a, hi_b) - max(lo_a, lo_b)
        smaller = min(hi_a - lo_a, hi_b - lo_b)
        if smaller <= 0:
            continue
        fracs.append(max(0.0, inter) / smaller)
    return min(fracs) if fracs else 0.0


def _measure_span(f: Face, vertical: bool) -> float:
    """The face's in-plane extent, split into horizontal length and height."""
    u_span = f.u_range[1] - f.u_range[0]
    v_span = f.v_range[1] - f.v_range[0]
    return max(u_span, v_span) if not vertical else min(u_span, v_span)


def assemble_walls(
    faces: list[Face], xyz: np.ndarray, config: dict
) -> tuple[list[Wall], list[int]]:
    """Pair opposing wall faces into walls. Returns (walls, unpaired_face_ids).

    A face with no partner still becomes a Wall -- with `face_b` and `thickness`
    set to None. A wall seen from one side is a real wall whose thickness we do
    not know, and saying so is the honest output. Inventing a thickness is not.
    """
    cos_tol = float(np.cos(np.radians(config["face_merge_angle_tol_deg"])))
    t_min = float(config["wall_thickness_min_m"])
    t_max = float(config["wall_thickness_max_m"])
    min_overlap = float(config["wall_pair_min_overlap"])

    wall_faces = sorted(
        [f for f in faces if f.role == "wall"], key=lambda f: f.face_id
    )

    best: dict[int, tuple[float, int]] = {}
    for i, a in enumerate(wall_faces):
        for b in wall_faces[i + 1:]:
            if abs(float(a.normal @ b.normal)) < cos_tol:
                continue
            t = perpendicular_offset(a, b)
            if not (t_min <= t <= t_max):
                continue
            if _overlap_fraction(a, b) < min_overlap:
                continue
            for x, y in ((a.face_id, b.face_id), (b.face_id, a.face_id)):
                if x not in best or t < best[x][0]:
                    best[x] = (t, y)

    # keep only mutual best matches, so a face cannot belong to two walls
    used: set[int] = set()
    walls: list[Wall] = []
    counter = 0
    by_id = {f.face_id: f for f in wall_faces}

    for f in wall_faces:
        if f.face_id in used:
            continue
        partner = best.get(f.face_id)
        mutual = (
            partner is not None
            and best.get(partner[1], (None, None))[1] == f.face_id
            and partner[1] not in used
        )
        counter += 1
        wid = f"W{counter:02d}"
        if mutual:
            t, other_id = partner
            g = by_id[other_id]
            used.add(f.face_id)
            used.add(other_id)
            n_pts = f.n_points + g.n_points
            resid = max(f.p95_residual_m, g.p95_residual_m)
            walls.append(Wall(
                wall_id=wid, face_a=f.face_id, face_b=other_id,
                thickness=Measurement(
                    value=float(t), method="face-to-face perpendicular offset",
                    n_points=int(n_pts), p95_residual=float(resid),
                ),
                length=Measurement(
                    value=float(max(_measure_span(f, False), _measure_span(g, False))),
                    method="face in-plane extent", n_points=int(n_pts),
                    p95_residual=float(resid),
                ),
                height=Measurement(
                    value=float(max(_measure_span(f, True), _measure_span(g, True))),
                    method="face in-plane extent", n_points=int(n_pts),
                    p95_residual=float(resid),
                ),
                centroid=(f.centroid + g.centroid) / 2.0,
                normal=f.normal.copy(),
            ))
        else:
            used.add(f.face_id)
            walls.append(Wall(
                wall_id=wid, face_a=f.face_id, face_b=None, thickness=None,
                length=Measurement(
                    value=float(_measure_span(f, False)),
                    method="face in-plane extent", n_points=f.n_points,
                    p95_residual=f.p95_residual_m,
                ),
                height=Measurement(
                    value=float(_measure_span(f, True)),
                    method="face in-plane extent", n_points=f.n_points,
                    p95_residual=f.p95_residual_m,
                ),
                centroid=f.centroid.copy(),
                normal=f.normal.copy(),
            ))

    unpaired = sorted(w.face_a for w in walls if w.face_b is None)
    return walls, unpaired
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/pytest tests/rscene/test_parts.py -v`
Expected: 6 passed

`test_thickness_is_not_measured_by_differencing_d` is the important one — it shifts the whole scene 47 m from the origin and demands the thickness not move. It is the regression guard for the defect that shipped in Plan 1.

- [ ] **Step 6: Commit**

```bash
git add src/rscene/core/parts.py tests/rscene/test_parts.py src/rscene/config.py
git commit -m "feat(core): assemble walls with measured face-to-face thickness"
```

---

### Task 8: Rectangular features and the junk gate

A real formwork feature is a rectangular prism with planar faces. An irregular concrete mass is not. That is the deterministic discriminator the design rests on.

**Files:**
- Create: `src/rscene/core/features.py`
- Create: `tests/rscene/test_features.py`
- Modify: `src/rscene/config.py`

**Interfaces:**
- Produces:
  - `Feature(feature_id: str, kind: str, parent_face: int, u_range, v_range, depth: Measurement, rect_fit: float)`
  - `extract_features(faces, xyz, config) -> tuple[list[Feature], list[int]]` — returns features and the face ids quarantined as non-rectangular

`kind` is one of `"extrusion"`, `"intrusion"`. Distinguishing grooves, L-cuts and electrical boxes by size is Plan 3's job once openings exist; do not guess them here.

- [ ] **Step 1: Add config keys**

```python
    # --- features (Plan 2) ---
    "feature_max_depth_m": 0.30,      # deepest offset still treated as a feature of a parent face rather than a separate wall
    "feature_min_rect_fit": 0.70,     # min fraction of the feature's bbox that must be filled for it to count as rectangular; below this it is an irregular mass
```

- [ ] **Step 2: Write the failing test**

Create `tests/rscene/test_features.py`:

```python
import numpy as np

from rscene.config import merged_config
from rscene.core.classify import classify_faces
from rscene.core.faces import merge_patches
from rscene.core.features import extract_features
from rscene.core.level import estimate_frame
from rscene.core.normals import estimate_normals
from rscene.core.patches import extract_patches
from rscene.core.prim import Box

DEPTH = 0.075
WIDTH = 0.35


def _wall_with_extrusion(spacing=0.008):
    below = Box("wl", (0, 0.00, 0), (0, 2.80, 2.5)).sample_surface(spacing, faces=("x+",))
    above = Box("wh", (0, 3.15, 0), (0, 4.00, 2.5)).sample_surface(spacing, faces=("x+",))
    face = Box("ef", (DEPTH, 2.80, 0), (DEPTH, 3.15, 2.5)).sample_surface(spacing, faces=("x+",))
    sa = Box("sa", (0, 2.80, 0), (DEPTH, 2.80, 2.5)).sample_surface(spacing, faces=("y-",))
    sb = Box("sb", (0, 3.15, 0), (DEPTH, 3.15, 2.5)).sample_surface(spacing, faces=("y+",))
    xyz = np.concatenate([below, above, face, sa, sb])
    cfg = merged_config()
    normals, curv = estimate_normals(xyz, k=cfg["normal_k"])
    patches, _ = extract_patches(xyz, normals, curv, cfg)
    faces = merge_patches(patches, xyz, cfg)
    classify_faces(faces, estimate_frame(patches, cfg), cfg)
    return xyz, faces, cfg


def test_a_75mm_extrusion_is_found_with_its_measured_depth():
    xyz, faces, cfg = _wall_with_extrusion()
    features, quarantined = extract_features(faces, xyz, cfg)

    depths = [f.depth.value for f in features]
    assert any(abs(d - DEPTH) < 0.003 for d in depths), f"depths {depths}"


def test_a_rectangular_feature_passes_the_rectangularity_gate():
    xyz, faces, cfg = _wall_with_extrusion()
    features, _ = extract_features(faces, xyz, cfg)
    match = [f for f in features if abs(f.depth.value - DEPTH) < 0.003][0]
    assert match.rect_fit >= cfg["feature_min_rect_fit"]
    assert abs((match.u_range[1] - match.u_range[0]) - WIDTH) < 0.01 or \
           abs((match.v_range[1] - match.v_range[0]) - WIDTH) < 0.01


def test_an_irregular_mass_is_quarantined_not_promoted():
    """Scattered non-planar junk must not become a feature."""
    wall = Box("w", (0, 0, 0), (0, 4, 2.5)).sample_surface(0.008, faces=("x+",))
    rng = np.random.default_rng(0)
    blob = np.column_stack([
        rng.uniform(0.02, 0.09, 4000),
        rng.uniform(1.0, 1.4, 4000),
        rng.uniform(0.0, 0.4, 4000),
    ])
    xyz = np.concatenate([wall, blob])
    cfg = merged_config()
    normals, curv = estimate_normals(xyz, k=cfg["normal_k"])
    patches, _ = extract_patches(xyz, normals, curv, cfg)
    faces = merge_patches(patches, xyz, cfg)
    classify_faces(faces, estimate_frame(patches, cfg), cfg)

    features, quarantined = extract_features(faces, xyz, cfg)
    for f in features:
        assert f.rect_fit >= cfg["feature_min_rect_fit"]


def test_extraction_is_deterministic():
    xyz, faces, cfg = _wall_with_extrusion()
    a, qa = extract_features(faces, xyz, cfg)
    b, qb = extract_features(faces, xyz, cfg)
    assert [f.feature_id for f in a] == [f.feature_id for f in b]
    assert [f.depth.value for f in a] == [f.depth.value for f in b]
    assert qa == qb
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest tests/rscene/test_features.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rscene.core.features'`

- [ ] **Step 4: Implement `features.py`**

Create `src/rscene/core/features.py`:

```python
"""Rectangular wall features, and the gate that keeps concrete out.

A bare-shell concrete building is rectilinear by construction: formwork makes
flat faces meeting at sharp edges. A real feature -- an extrusion, a recess, a
groove, an electrical box -- is a rectangular prism. An irregular lump of
leftover concrete is not, and cannot be made to fit one.

That is the whole discriminator, and it needs no heuristics about size or
position. `rect_fit` is the fraction of the feature's own in-plane bounding box
that its points actually fill; a rectangular face fills its box, a scattered
mass does not. Anything below the gate is quarantined -- returned to the caller
for the scene's `unmodeled` set, never silently deleted and never promoted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .faces import Face, median_spacing
from .fitting import plane_basis
from .graph import perpendicular_offset
from .scene import Measurement


@dataclass
class Feature:
    """A rectangular departure from a parent face."""

    feature_id: str
    kind: str                       # "extrusion" | "intrusion"
    parent_face: int
    u_range: tuple[float, float]
    v_range: tuple[float, float]
    depth: Measurement
    rect_fit: float


def _rect_fit(face: Face, xyz: np.ndarray, spacing: float) -> float:
    """Fraction of the face's in-plane bbox its own points fill."""
    area = face.area_bound_m2()
    if area <= 0 or spacing <= 0:
        return 0.0
    expected = area / (spacing ** 2)
    return float(min(1.0, face.n_points / expected)) if expected > 0 else 0.0


def extract_features(
    faces: list[Face], xyz: np.ndarray, config: dict
) -> tuple[list[Feature], list[int]]:
    """Find rectangular features standing proud of or recessed into wall faces.

    A candidate is a wall face parallel to a larger wall face, offset by less
    than `feature_max_depth_m`, and smaller than it. Candidates that fail the
    rectangularity gate are quarantined by face id rather than deleted.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    spacing = median_spacing(xyz, seed=int(config["seed"]))
    max_depth = float(config["feature_max_depth_m"])
    min_fit = float(config["feature_min_rect_fit"])
    cos_tol = float(np.cos(np.radians(config["face_merge_angle_tol_deg"])))

    walls = sorted([f for f in faces if f.role == "wall"], key=lambda f: f.face_id)
    features: list[Feature] = []
    quarantined: list[int] = []
    counter = 0

    for cand in walls:
        parent = None
        best = np.inf
        for other in walls:
            if other.face_id == cand.face_id:
                continue
            if other.area_bound_m2() <= cand.area_bound_m2():
                continue
            if abs(float(cand.normal @ other.normal)) < cos_tol:
                continue
            offset = perpendicular_offset(cand, other)
            if offset <= 0 or offset > max_depth:
                continue
            if offset < best:
                best, parent = offset, other
        if parent is None:
            continue

        fit = _rect_fit(cand, xyz, spacing)
        if fit < min_fit:
            quarantined.append(cand.face_id)
            continue

        # sign: does the candidate stand toward the parent's interior side?
        along = float((cand.centroid - parent.centroid) @ parent.normal)
        kind = "extrusion" if along > 0 else "intrusion"

        counter += 1
        features.append(Feature(
            feature_id=f"F{counter:02d}",
            kind=kind,
            parent_face=parent.face_id,
            u_range=cand.u_range,
            v_range=cand.v_range,
            depth=Measurement(
                value=float(best),
                method="perpendicular offset to parent face",
                n_points=int(cand.n_points),
                p95_residual=float(max(cand.p95_residual_m, parent.p95_residual_m)),
            ),
            rect_fit=float(fit),
        ))

    return features, sorted(quarantined)
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/pytest tests/rscene/test_features.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add src/rscene/core/features.py tests/rscene/test_features.py src/rscene/config.py
git commit -m "feat(core): rectangular feature extraction with a rectangularity gate"
```

---

### Task 9: Carry parts in the scene document

**Files:**
- Modify: `src/rscene/core/scene.py`
- Modify: `tests/rscene/test_scene.py`

**Interfaces:**
- Produces: `Scene` gains `faces: list[Face]`, `walls: list[Wall]`, `features: list[Feature]`, `unmodeled: list[int]`; `scene_to_json` / `scene_from_json` round-trip all of them with `Measurement` values intact.

- [ ] **Step 1: Write the failing test**

Append to `tests/rscene/test_scene.py`:

```python
def test_scene_round_trips_walls_and_features():
    from rscene.core.faces import Face
    from rscene.core.features import Feature
    from rscene.core.parts import Wall

    face = Face(
        face_id=0, normal=np.array([1.0, 0.0, 0.0]), d=-2.0, patch_ids=[0, 1],
        point_idx=np.array([0, 1, 2]), loose_idx=np.array([3]), n_points=3,
        p95_residual_m=0.0021, centroid=np.array([2.0, 1.0, 1.4]),
        u_range=(-1.0, 1.0), v_range=(-1.4, 1.4), role="wall", interior_sign=1,
    )
    wall = Wall(
        wall_id="W01", face_a=0, face_b=1,
        thickness=Measurement(0.2031, "face-to-face perpendicular offset", 18422, 0.0021),
        length=Measurement(4.182, "face in-plane extent", 18422, 0.0021),
        height=Measurement(2.748, "face in-plane extent", 18422, 0.0021),
        centroid=np.array([2.0, 1.0, 1.4]), normal=np.array([1.0, 0.0, 0.0]),
    )
    feature = Feature(
        feature_id="F01", kind="extrusion", parent_face=0,
        u_range=(2.80, 3.15), v_range=(0.0, 2.75),
        depth=Measurement(0.075, "perpendicular offset to parent face", 900, 0.0022),
        rect_fit=0.94,
    )
    scene = _scene()
    scene.faces = [face]
    scene.walls = [wall]
    scene.features = [feature]
    scene.unmodeled = [7, 9]

    restored = scene_from_json(scene_to_json(scene))

    assert restored.walls[0].wall_id == "W01"
    assert restored.walls[0].thickness.value == 0.2031
    assert restored.walls[0].thickness.method == "face-to-face perpendicular offset"
    assert restored.features[0].kind == "extrusion"
    assert restored.features[0].depth.value == 0.075
    assert restored.faces[0].role == "wall"
    assert restored.faces[0].interior_sign == 1
    assert np.array_equal(restored.faces[0].loose_idx, [3])
    assert restored.unmodeled == [7, 9]


def test_an_unpaired_wall_serialises_a_null_thickness():
    from rscene.core.parts import Wall
    scene = _scene()
    scene.walls = [Wall(
        wall_id="W02", face_a=3, face_b=None, thickness=None,
        length=Measurement(2.0, "face in-plane extent", 500, 0.002),
        height=Measurement(2.5, "face in-plane extent", 500, 0.002),
        centroid=np.array([0.0, 0.0, 0.0]), normal=np.array([1.0, 0.0, 0.0]),
    )]
    restored = scene_from_json(scene_to_json(scene))
    assert restored.walls[0].thickness is None
    assert restored.walls[0].face_b is None
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/rscene/test_scene.py -v`
Expected: FAIL — `AttributeError: 'Scene' object has no attribute 'walls'`

- [ ] **Step 3: Implement**

Add four fields to `Scene`, all defaulting empty:

```python
    faces: list = field(default_factory=list)
    walls: list = field(default_factory=list)
    features: list = field(default_factory=list)
    unmodeled: list[int] = field(default_factory=list)
```

`parts.py` and `features.py` import `Measurement` from this module, so importing `Wall`/`Feature` at the top of `scene.py` would be circular. Import them **inside** the converter functions.

```python
def _measure_to_dict(m):
    return None if m is None else m.to_dict()


def _measure_from_dict(p):
    return None if p is None else Measurement.from_dict(p)


def _face_to_dict(f) -> dict:
    return {
        "face_id": f.face_id,
        "normal": [float(x) for x in f.normal],
        "d": float(f.d),
        "patch_ids": list(f.patch_ids),
        "point_idx": [int(i) for i in f.point_idx],
        "loose_idx": [int(i) for i in f.loose_idx],
        "n_points": f.n_points,
        "p95_residual_m": f.p95_residual_m,
        "centroid": [float(x) for x in f.centroid],
        "u_range": list(f.u_range),
        "v_range": list(f.v_range),
        "role": f.role,
        "interior_sign": f.interior_sign,
    }


def _face_from_dict(p):
    from .faces import Face
    return Face(
        face_id=p["face_id"],
        normal=np.array(p["normal"], dtype=np.float64),
        d=p["d"],
        patch_ids=list(p["patch_ids"]),
        point_idx=np.array(p["point_idx"], dtype=np.int64),
        loose_idx=np.array(p["loose_idx"], dtype=np.int64),
        n_points=p["n_points"],
        p95_residual_m=p["p95_residual_m"],
        centroid=np.array(p["centroid"], dtype=np.float64),
        u_range=tuple(p["u_range"]),
        v_range=tuple(p["v_range"]),
        role=p["role"],
        interior_sign=p["interior_sign"],
    )


def _wall_to_dict(w) -> dict:
    return {
        "wall_id": w.wall_id, "face_a": w.face_a, "face_b": w.face_b,
        "thickness": _measure_to_dict(w.thickness),
        "length": _measure_to_dict(w.length),
        "height": _measure_to_dict(w.height),
        "centroid": [float(x) for x in w.centroid],
        "normal": [float(x) for x in w.normal],
    }


def _wall_from_dict(p):
    from .parts import Wall
    return Wall(
        wall_id=p["wall_id"], face_a=p["face_a"], face_b=p["face_b"],
        thickness=_measure_from_dict(p["thickness"]),
        length=_measure_from_dict(p["length"]),
        height=_measure_from_dict(p["height"]),
        centroid=np.array(p["centroid"], dtype=np.float64),
        normal=np.array(p["normal"], dtype=np.float64),
    )


def _feature_to_dict(f) -> dict:
    return {
        "feature_id": f.feature_id, "kind": f.kind, "parent_face": f.parent_face,
        "u_range": list(f.u_range), "v_range": list(f.v_range),
        "depth": _measure_to_dict(f.depth), "rect_fit": f.rect_fit,
    }


def _feature_from_dict(p):
    from .features import Feature
    return Feature(
        feature_id=p["feature_id"], kind=p["kind"], parent_face=p["parent_face"],
        u_range=tuple(p["u_range"]), v_range=tuple(p["v_range"]),
        depth=_measure_from_dict(p["depth"]), rect_fit=p["rect_fit"],
    )
```

Then add `faces`, `walls`, `features` and `unmodeled` to the payload dict in `scene_to_json` and read them back in `scene_from_json`. **Keep `sort_keys=True`** and the existing alphabetical key discipline so output stays byte-identical across runs — the determinism test is what protects that.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/rscene/test_scene.py -v`
Expected: all pass, including the two new tests.

- [ ] **Step 5: Commit**

```bash
git add src/rscene/core/scene.py tests/rscene/test_scene.py
git commit -m "feat(core): carry faces, walls and features in the scene document"
```

---

### Task 10: `rscene parts` — wire the pipeline and report

**Files:**
- Modify: `src/rscene/cli.py`
- Modify: `tests/rscene/test_cli.py`

**Interfaces:**
- Produces: `rscene parts <scan.las> <out_dir>` running ingest → normals → patches → merge → density gate → recruit → occupancy → interior → classify → walls → features → scene.json + report.md

- [ ] **Step 1: Write the failing test**

Append to `tests/rscene/test_cli.py`:

```python
def test_parts_command_writes_scene_and_report(tmp_path):
    scan = _write_scan(tmp_path)
    out = tmp_path / "parts_out"
    assert main(["parts", scan, str(out)]) == 0
    assert (out / "scene.json").exists()
    assert (out / "report.md").exists()


def test_parts_report_states_unassigned_after_recruitment(tmp_path):
    scan = _write_scan(tmp_path)
    out = tmp_path / "parts_out"
    main(["parts", scan, str(out)])
    report = (out / "report.md").read_text()
    assert "before recruitment" in report.lower()
    assert "after recruitment" in report.lower()


def test_parts_is_deterministic(tmp_path):
    scan = _write_scan(tmp_path)
    a, b = tmp_path / "a", tmp_path / "b"
    main(["parts", scan, str(a)])
    main(["parts", scan, str(b)])
    assert (a / "scene.json").read_text() == (b / "scene.json").read_text()


@pytest.mark.real_scan
@pytest.mark.skipif(not _REAL_SCAN.exists(), reason="real scan not present")
def test_parts_on_the_real_scan(tmp_path):
    """Invariants that must hold on real data, not just synthetic."""
    crop = _crop_real_scan(tmp_path)
    out = tmp_path / "real_parts"
    assert main(["parts", crop, str(out), "--set", "patch_neighbor_k=64"]) == 0

    payload = json.loads((out / "scene.json").read_text())
    n_faces = len(payload["faces"])
    n_patches = len(payload["patches"])

    # merging must actually reduce the count -- 200 patches, ~69 coplanar classes
    assert n_faces < n_patches, f"merging did nothing: {n_patches} -> {n_faces}"
    # recruitment must reduce unassigned below the Plan 1 figure of 13.6%
    total = payload["diagnostics"]["n_input_points"]
    assert payload["unassigned_points"] / total < 0.136
    # every wall with two faces has a plausible thickness
    for w in payload["walls"]:
        if w["thickness"] is not None:
            assert 0.05 <= w["thickness"]["value"] <= 0.45
```

Plan 1's real-scan CLI test crops the scan **inline** rather than through a helper. Extract that cropping code into a module-level `_crop_real_scan(tmp_path) -> str` in `tests/rscene/test_cli.py` and have both the Plan 1 test and this new one call it — the crop bounds (`x ∈ (−3.2, 1.0)`, `y ∈ (−8.0, −3.0)`, about 455 k points at native density) must stay identical, or the two tests stop being comparable. It crops **spatially**; it must never fall back to `--max-points`, which random-subsamples and would contradict the warning the CLI prints.

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/rscene/test_cli.py -v`
Expected: FAIL — `argument command: invalid choice: 'parts'`

- [ ] **Step 3: Implement**

Add a `parts` subparser to `src/rscene/cli.py` alongside `patches`, sharing the input/output/seed/`--set` arguments and the `--max-points` warning. Run the full stage chain, seeding the interior fill at `interior_seed_height_m` above the floor patch's centroid (fall back to the cloud's median XY at that height if no floor face is found, and say so in the report rather than failing).

`report.md` gains, beyond Plan 1's sections:

- faces before and after merging, and after the density gate
- **unassigned before recruitment and after**, with the bucket breakdown for what remains
- walls: count, how many are paired, and a table of `wall_id`, length, height, thickness with the residual on each
- features: count by kind, with depth and `rect_fit`
- quarantined face ids and the point total they carry, under an `unmodeled` heading

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/pytest tests/rscene tests/golden -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/rscene/cli.py tests/rscene/test_cli.py
git commit -m "feat(cli): rscene parts command assembling walls and features"
```

---

### Task 11: Golden truth for part-level dimensions

The whole-branch review found that `GoldenScene.truth` declares seven values and only four are asserted — `clear_span_x_m`, `clear_span_y_m` and `extrusion_width_m` are checked by nothing. Those are the designer's headline dimensions, and Plan 2 is the first plan able to measure them.

**Files:**
- Modify: `tests/golden/apartment.py`
- Modify: `tests/golden/test_accuracy.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/golden/test_accuracy.py` a `parts`-level fixture that runs merge → gate → recruit → occupancy → classify → walls → features on the golden room, then assert:

```python
@pytest.mark.slow
def test_clear_span_x_is_recovered(assembled):
    scene, walls, features = assembled
    spans = [w.length.value for w in walls]
    assert any(abs(s - scene.truth["clear_span_x_m"]) < TOL for s in spans), spans


@pytest.mark.slow
def test_clear_span_y_is_recovered(assembled):
    scene, walls, features = assembled
    spans = [w.length.value for w in walls]
    assert any(abs(s - scene.truth["clear_span_y_m"]) < TOL for s in spans), spans


@pytest.mark.slow
def test_extrusion_width_is_recovered(assembled):
    scene, walls, features = assembled
    widths = [max(f.u_range[1] - f.u_range[0], f.v_range[1] - f.v_range[0])
              for f in features]
    assert any(abs(w - scene.truth["extrusion_width_m"]) < TOL for w in widths), widths


@pytest.mark.slow
def test_every_truth_value_is_asserted_somewhere(assembled):
    """Guard against truth keys that quietly go unchecked."""
    scene, _, _ = assembled
    asserted = {
        "clear_span_x_m", "clear_span_y_m", "ceiling_height_m",
        "extrusion_depth_m", "extrusion_width_m", "groove_depth_m",
        "switch_box_depth_m",
    }
    assert set(scene.truth) == asserted, set(scene.truth) ^ asserted
```

Also add a `wall_thickness_m` truth value to `build_golden_room` by giving one wall a second, outer face, and assert it recovers within 3 mm. The golden room currently samples only interior faces, so no wall has a measurable thickness — that gap is why Plan 1 could never test the pipeline's headline output.

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/golden/test_accuracy.py -v -m slow`
Expected: FAIL — `fixture 'assembled' not found`, then assertion failures once it exists.

- [ ] **Step 3: Add the outer wall face to the golden room**

In `tests/golden/apartment.py`, the room currently samples only interior surfaces, so **no wall has two faces and thickness cannot be measured at all** — the pipeline's headline output has never been under test. Give the `x = CLEAR_X` wall an outer face and record the truth value:

```python
WALL_THICKNESS = 0.200

    # outer face of the x = CLEAR_X wall, so one wall has a measurable thickness.
    # A scanner standing outside would see this; the golden room needs it because
    # thickness is the pipeline's headline dimension and cannot be tested from
    # one face alone.
    faces.append(Box("x1_outer",
                     (CLEAR_X + WALL_THICKNESS, 0, 0),
                     (CLEAR_X + WALL_THICKNESS, CLEAR_Y, HEIGHT)).sample_surface(
        spacing_m, faces=("x+",)))
```

and add `"wall_thickness_m": WALL_THICKNESS` to the returned `truth` dict.

- [ ] **Step 4: Add the `assembled` fixture**

In `tests/golden/test_accuracy.py`:

```python
@pytest.fixture(scope="module")
def assembled(extracted):
    """Run the Plan 2 chain on the golden room once, reuse across tests."""
    from rscene.core.classify import classify_faces
    from rscene.core.faces import apply_density_gate, merge_patches, recruit_points
    from rscene.core.features import extract_features
    from rscene.core.level import estimate_frame
    from rscene.core.normals import estimate_normals
    from rscene.core.occupancy import (assign_interior_sides, build_occupancy,
                                       flood_interior)
    from rscene.core.parts import assemble_walls

    scene, patches, labels = extracted
    cfg = merged_config()
    xyz = scene.points
    normals, _ = estimate_normals(xyz, k=cfg["normal_k"])

    faces = merge_patches(patches, xyz, cfg)
    faces, rejected = apply_density_gate(faces, xyz, cfg)
    recruit_points(faces, xyz, normals, labels, cfg)

    frame = estimate_frame(patches, cfg)
    grid = build_occupancy(xyz, cfg)
    seed = np.array([CLEAR_X / 2, CLEAR_Y / 2, frame.floor_z + cfg["interior_seed_height_m"]])
    interior = flood_interior(grid, seed)
    assign_interior_sides(faces, grid, interior, cfg, xyz)
    classify_faces(faces, frame, cfg)

    walls, _ = assemble_walls(faces, xyz, cfg)
    features, _ = extract_features(faces, xyz, cfg)
    return scene, walls, features
```

Import `CLEAR_X`, `CLEAR_Y` and `merged_config` at the top of the test module.

Add the thickness assertion alongside the four tests from Step 1:

```python
@pytest.mark.slow
def test_wall_thickness_is_recovered(assembled):
    scene, walls, _ = assembled
    thicknesses = [w.thickness.value for w in walls if w.thickness is not None]
    assert any(abs(t - scene.truth["wall_thickness_m"]) < TOL for t in thicknesses), \
        thicknesses
```

and add `"wall_thickness_m"` to the `asserted` set in `test_every_truth_value_is_asserted_somewhere`.

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/pytest tests/golden -v -m slow`
Expected: all pass. If thickness comes back wrong, check the outer face is being classified `wall` rather than falling into `oblique` — it faces `x+` while the interior face faces `x-`, and both must survive classification for the pair to form.

- [ ] **Step 6: Commit**

```bash
git add tests/golden
git commit -m "test: golden part-level truth for spans, widths and wall thickness"
```

---

## Definition of done

- `.venv/bin/pytest tests/rscene tests/golden` passes, slow and real-scan tests included.
- On the golden room: wall thickness, both clear spans, ceiling height, extrusion depth and width, groove depth and switch-box depth all recovered within 3 mm.
- On the real crop: faces < patches, unassigned below 13.6%, every paired wall's thickness in 0.05–0.45 m.
- `test_thickness_is_not_measured_by_differencing_d` passes — thickness does not change when the scene is moved 47 m from the origin.
- `tests/rscene/test_architecture.py` still passes: `src/rscene/core/` imports only numpy and scipy.
- Two runs of `rscene parts` on the same scan produce byte-identical `scene.json`.
- Every point is in a face, a named unassigned bucket, or `unmodeled` — and the report says which.

---

## Revision, 2026-08-13 — grounded in the real scan

Tasks 1–3 shipped and were reviewed. Then the pipeline was run on the real crop, and the result changed how the rest of this plan should be built. Three of the first three tasks had tests that proved less than they appeared to — each caught by an implementer reporting an anomaly, never by an assertion failing. Synthetic fixtures reproduce the *shape* of real data without its awkwardness.

### Measured baseline

One-room crop of `data/isolated_structural_v2.las`, 454,708 points at native density, `patch_neighbor_k=64`:

| stage | result |
|---|---|
| `extract_patches` | 196 patches, 62,239 unassigned (13.69%) |
| `merge_patches` | 88 faces (55% fewer) |
| `apply_density_gate` | 86 kept, 2 rejected (409 pts, 0.09%) |
| `recruit_points` | unassigned 62,239 → 32,994 — 47% claimed, 29,245 recruits, **7.26% final** |
| kept faces | 22 horizontal, 59 vertical, 5 oblique; median 190 pts, max 128,659 |
| residual | median p95 2.77 mm |

### What the real scan changed

**1. Recruitment is confirmed as the highest-value stage.** 13.69% → 7.26% unassigned, nearly halved. The user requirement that every point be accounted for is now measurably closer.

**2. The density gate is in the wrong place.** It rejects 0.09% of points post-merge. Its 4.4%-fill motivation was measured on *patches*, before merging; merging absorbs those chains into real faces first, so by the time the gate runs there is almost nothing left for it to catch. It is not wrong, it is redundant where it sits. **Do not move it yet** — the merge fix below changes the face population, so re-measure before deciding whether it belongs pre-merge, on patches.

**3. Merging had a correctness defect that only real data revealed.** Union-find over a pairwise predicate permits unbounded chain drift: face 0 chained 17 patches whose extremes were **41.2 mm apart against a 5 mm tolerance**, giving a 7.36 mm p95 residual against a 3 mm `tau_fit_m`. A robust refit does *not* rescue it — trimmed refit on the inlier 80% reaches 4.89 mm but then covers only 50% of the face's own points within tolerance. The face is genuinely not one plane. Fixed by greedy seeded accumulation: a patch joins only if it is within tolerance of the **group's current fitted plane**, refitted after each addition, so drift is bounded by the tolerance rather than by chain length.

### New global requirement: every remaining task carries a real-scan assertion

Alongside its synthetic accuracy test, each of Tasks 4–12 gains a test marked `real_scan` (skipped when `data/isolated_structural_v2.las` is absent) asserting a **behavioural** property against a figure measured beforehand. Synthetic tests answer *is the number right*; only real-scan tests answer *does this stage do anything at all on real data*.

The distinction matters because the two cannot substitute for each other. Synthetic is the only place ground truth exists — on a real scan nobody knows the true wall thickness, so a wrong answer is indistinguishable from a right one. That is how the origin lever-arm bug survived twelve reviews producing entirely plausible numbers. But synthetic cannot show what actually happens on noisy, occluded, cluttered data, which is how three vacuous tests got through.

Every real-scan assertion must:

- **State a precondition** — that the input to this stage is non-trivial — before asserting the stage's effect. A stage cannot be shown to work on nothing.
- **Use a measured number, not a guessed one.** Run the stage, look at the result, then write the assertion with headroom.
- **Be falsifiable by the stage doing nothing.** "Merging reduces face count", "recruitment reduces unassigned", "no face exceeds tolerance" all fail instantly against a no-op. `assert x is not None` does not.

### Task 12 (new): emit the scene as a program

Modularity is the point of this rebuild — the user's third confirmed failure of the old pipeline was "not actually modular", and the LiteReality architecture this design borrows from makes the scene an editable program rather than an opaque mesh. Plan 1 deferred `room.py` to the export plan, which demoted the deliverable to an afterthought. It moves here, to the point where walls first exist.

`rscene parts` gains a `room.py` output alongside `scene.json`: one object per part, parameters visible and editable, regenerating the same geometry when re-run.

```python
Wall("W03", length=4.182, height=2.748, thickness=0.2031,
     at=(2.10, 0.35), angle=5.17,
     features=[Extrusion("W03_E01", u=(2.80, 3.15), depth=0.075)])
```

`scene.json` stays the measurement record with provenance and uncertainty; `room.py` is the modular artifact a designer or an agent edits. The two are generated from the same in-memory scene, never from each other.

Requirements: every part in `scene.json` appears as exactly one object; every dimension carries the measured value, not a rounded one; the file is deterministic; and a test asserts that executing it reproduces the part count and dimensions it was generated from. Without that round-trip test it is a pretty-printer, not a program.

---

## Deferred to Plan 3

Openings (voids in wall faces validated against the flood-fill), rooms (interior connected components), solidify (half-space CSG per part), the residual critique loop, and the LLM advisory labelling described in spec §6.2. Exports — plan DXF, per-wall elevations, OBJ/FBX, glTF, `room.py` — are Plan 4.

Carried forward from Plan 1's ledger, not addressed here: `scene.json` embeds every point index and reached 6.3 MB on a 455k-point crop, so a full apartment will be hundreds of megabytes — decide a sidecar or encoding before more consumers depend on the shape. The seed-regrow quadratic in `patches.py`. `tau_feature_m` declared but unused. `min_intersection_angle_deg` (0.5°) sitting below `coplanar_angle_tol_deg` (2.0°). Floor/ceiling selection by raw min/max centroid Z, which the human ruled to keep and which the review measured as a 21 mm swing on storey height.
