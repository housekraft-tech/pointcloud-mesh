# isolidarflow: Sharp-Edged Modular 3D Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn an isolated single-storey SLAM scan into a sharp-edged, modular 3D model (separate named walls / floor / ceiling / doors / windows / balcony doors / columns / beams, GLB + Blender collections) plus an architectural 2D floorplan, with accurate measured dimensions on every element.

**Architecture:** Productionize the experimentally-validated isolidarflow wall chain (u-gap-split runs → midline recentring → T-junction closure — see `scripts/experiments/README.md` for real-scan evidence), then detect openings from two independent signals: **gps_time trajectory crossings** (the scanner physically walked through every door/pathway/balcony door) and per-wall occupancy voids gated by **trajectory visibility rays**; strip furniture as points unexplained by structural planes; extrude + boolean-cut watertight solids with the existing `recon.solids`/`recon.assemble`.

**Tech Stack:** Python 3.11 (`venv311\Scripts\python.exe`), laspy, Open3D 0.19, numpy, scipy, shapely, trimesh, manifold3d, ezdxf, svgwrite, opencv-headless, pytest.

## Global Constraints

- **Platform:** Windows; venv at `.claude/worktrees/cad-floorplan-reconstruction/venv311` — run everything with `venv311\Scripts\python.exe`. CPU-only.
- **Branch:** work on `isolidarflow`.
- **Units:** metres, world frame = LAS coordinates (axis-aligned after `frame.axis_align`).
- **Purity discipline:** every module except `io_las`, `planes`, `frame`, `solids`, `assemble` must import without Open3D/trimesh (numpy/scipy/shapely only) so unit tests run anywhere — same rule the existing `recon` package follows.
- **Determinism:** all RANSAC/detection code paths must accept and apply a seed (Task 1). Real-scan runs use seed 0.
- **Priors config (user-supplied, config values not hard-coded):** door height prior 2.13 m (7 ft), acceptable 1.85–2.35 m; ceiling height prior 2.75 m, acceptable 2.35–3.05 m; door width varies by room — annotate confidence, never reject on width alone; balcony door: floor-touching and width ≥ 1.3 m or trajectory exits the room-polygon union through it.
- **Do not touch** `reconstruct_mesh*.py` / `mesh_common.py` (Poisson reference path) or `scripts/floorplan_*` / `segment_walls_and_grooves.py` (owned by another in-flight effort).
- **Input reality (koushik scan):** ~3.75M isolated points, floor ≈ −0.25 m, ceiling ≈ 2.50 m (storey ≈ 2.75 m — matches the ceiling prior), `gps_time` present (356 trajectory vertices from Phase A), furniture clutter in room interiors, balcony on an exterior wall.
- **Commit cadence:** one commit per task (test + impl together), trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

## File Structure

| File | Change | Responsibility | Pure? |
|---|---|---|---|
| `scripts/recon/planes.py` | modify | add `seed` param → `o3d.utility.random.seed` | Open3D |
| `scripts/recon/structure.py` | modify | u-gap run splitting, coverage `members`, rescue defaults | pure |
| `scripts/recon/regularize.py` | modify | `recenter_walls`, `snap_endpoints_to_lines` | pure |
| `scripts/recon/trajectory.py` | modify | `wall_crossings` (gps_time walkthrough detection) | pure |
| `scripts/recon/clean.py` | modify | `remove_furniture` (structural-plane distance filter) | pure |
| `scripts/recon/openings.py` | create | occupancy voids + visibility gate + refine + classify | pure (scipy cKDTree) |
| `scripts/recon/schema.py` | modify | `OpeningV2`, `dims`/`confidence` fields, manifest helpers | pure |
| `scripts/recon/solids.py` | modify | floor/ceiling slab polygons from room union | trimesh |
| `scripts/isolidarflow.py` | create | end-to-end CLI + `DEFAULT_CONFIG` + debug PNG | mixed |
| `tests/recon/test_structure.py` etc. | modify/create | one test module per touched module | pure |

Interfaces reused verbatim from the existing codebase: `Plane`, `WallStep`, `Column`, `Beam`, `ScanData` (`recon/schema.py`); run-dict contract of `group_wall_runs` (`structure.py` docstring); `wall_to_solid`, `cut_openings`, `column_to_solid`, `beam_to_solid`, `slab_to_solid` (`solids.py`); `build_scene`, `write_glb` (`assemble.py`); `build_room_polygons`, `write_dxf`, `write_svg` (`floorplan2d.py`); `approx_trajectory` (`trajectory.py`); `modular_house()` fixture meta contract: `{"openings": [{"wall","u_m","sill_m","type"}], "pillar", "beam", "neighbour_bbox"}`.

---

## Task 1: Deterministic plane detection (seed)

**Files:**
- Modify: `scripts/recon/planes.py` (`detect_planes` signature)
- Test: `tests/recon/test_planes.py`

**Interfaces:**
- Produces: `detect_planes(xyz, ..., seed: int | None = 0)` — when `seed is not None`, calls `o3d.utility.random.seed(seed)` before the RANSAC loop. All other behavior unchanged.

Real-scan motivation: unseeded runs gave 191 / 238 / 244 vertical planes on the identical cloud — pipeline output must be reproducible.

- [ ] **Step 1: Write the failing test** (append to `tests/recon/test_planes.py`):

```python
import inspect
from scripts.recon.planes import detect_planes

def test_detect_planes_accepts_seed_param():
    sig = inspect.signature(detect_planes)
    assert "seed" in sig.parameters
    assert sig.parameters["seed"].default == 0
```

- [ ] **Step 2:** Run: `venv311\Scripts\python.exe -m pytest tests/recon/test_planes.py -v` → FAIL (`seed` not in parameters).
- [ ] **Step 3:** Implement: add `seed: int | None = 0` to `detect_planes`; immediately after `import open3d as o3d` add:

```python
    if seed is not None:
        o3d.utility.random.seed(seed)
```

- [ ] **Step 4:** Run test → PASS. Also run an Open3D-marked smoke if available: `venv311\Scripts\python.exe -m pytest tests/recon/test_planes.py -v`.
- [ ] **Step 5:** Commit: `git add scripts/recon/planes.py tests/recon/test_planes.py && git commit -m "Seed Open3D RNG in detect_planes for reproducible plane detection"`

---

## Task 2: u-gap run splitting + coverage members + rescue defaults

**Files:**
- Modify: `scripts/recon/structure.py` (`group_wall_runs`)
- Test: `tests/recon/test_structure.py`

**Interfaces:**
- Consumes: existing `_plane_axis_stats`, `_greedy_chain`.
- Produces: `group_wall_runs(vertical_planes, xyz, axes, merge_offset_m=0.15, max_relief_m=0.35, min_run_length_m=0.5, angle_tol_deg=10.0, full_height_frac=0.4, u_gap_m=2.8, min_run_inliers=1200, *, return_used_indices=False)`.
  - New behavior: within each offset-cluster, members are split into separate runs wherever consecutive u-intervals leave a hole wider than `u_gap_m`.
  - Each run dict gains a key `members: list[tuple[float, float]]` — the (u_min, u_max) interval of every member plane, ABSOLUTE world-u. Holes between merged member intervals narrower than `u_gap_m` stay inside the run (they are opening candidates for Task 7).
  - Default changes (real-scan validated, see `scripts/experiments/README.md`): `max_relief_m` 1.0→0.35 (corridor walls stay separate), `full_height_frac` 0.5→0.4 and `min_run_length_m` 1.0→0.5 (occluded/short partitions survive), `min_run_inliers=1200` (furniture guard).

- [ ] **Step 1: Write the failing tests** (append to `tests/recon/test_structure.py`):

```python
import numpy as np
from scripts.recon.schema import Plane
from scripts.recon.structure import group_wall_runs


def _wall_plane_points(u0, u1, offset, axis="y", z0=0.0, z1=2.7, n=4000, rng_seed=0):
    """Points of one vertical wall face: normal along `axis`, extent along the
    other axis from u0..u1, at perpendicular position `offset`."""
    rng = np.random.default_rng(rng_seed)
    u = rng.uniform(u0, u1, n)
    z = rng.uniform(z0, z1, n)
    perp = np.full(n, offset) + rng.normal(0, 0.003, n)
    if axis == "y":
        return np.column_stack([u, perp, z])
    return np.column_stack([perp, u, z])


def _mk_plane(xyz_all, idx, axis="y"):
    normal = (0.0, 1.0, 0.0) if axis == "y" else (1.0, 0.0, 0.0)
    return Plane(normal=normal, d=0.0, label="vertical", inlier_idx=np.asarray(idx))


def test_u_gap_splits_collinear_distinct_walls():
    # two same-offset walls 4 m apart along u -> MUST be two separate runs
    a = _wall_plane_points(0.0, 3.0, offset=5.0)
    b = _wall_plane_points(7.0, 10.0, offset=5.0)
    xyz = np.vstack([a, b])
    planes = [_mk_plane(xyz, np.arange(len(a))),
              _mk_plane(xyz, np.arange(len(a), len(a) + len(b)))]
    runs = group_wall_runs(planes, xyz, np.eye(3))
    assert len(runs) == 2
    lengths = sorted(round(np.linalg.norm(np.subtract(r["p1"], r["p0"])), 1) for r in runs)
    assert lengths == [3.0, 3.0]


def test_doorway_gap_stays_one_run_with_members():
    # one wall broken by a 1.0 m doorway -> ONE run, hole visible in members
    a = _wall_plane_points(0.0, 2.0, offset=5.0)
    b = _wall_plane_points(3.0, 6.0, offset=5.0)
    xyz = np.vstack([a, b])
    planes = [_mk_plane(xyz, np.arange(len(a))),
              _mk_plane(xyz, np.arange(len(a), len(a) + len(b)))]
    runs = group_wall_runs(planes, xyz, np.eye(3))
    assert len(runs) == 1
    assert "members" in runs[0] and len(runs[0]["members"]) == 2
    (a0, a1), (b0, b1) = sorted(runs[0]["members"])
    assert abs(a1 - 2.0) < 0.1 and abs(b0 - 3.0) < 0.1  # the hole is recoverable
```

- [ ] **Step 2:** Run: `venv311\Scripts\python.exe -m pytest tests/recon/test_structure.py -v -k "u_gap or doorway"` → FAIL (first test: one 10 m run; second: no `members` key).
- [ ] **Step 3:** Implement in `group_wall_runs`: add the two new params with the new defaults, change the three existing defaults, and inside the per-offset-cluster loop insert an interval-chaining split before step grouping:

```python
def _chain_by_u_interval(members, u_gap_m):
    """Split plane-stat dicts into groups of near/overlapping u-intervals:
    sorted by u_min, a new group starts when the next member's u_min is more
    than u_gap_m past the running max u_max."""
    members = sorted(members, key=lambda s: s["u_min"])
    groups, cur, cur_max = [], [], None
    for s in members:
        if cur and s["u_min"] - cur_max > u_gap_m:
            groups.append(cur)
            cur, cur_max = [], None
        cur.append(s)
        cur_max = s["u_max"] if cur_max is None else max(cur_max, s["u_max"])
    if cur:
        groups.append(cur)
    return groups
```

then replace the body of the `for cluster in _greedy_chain(...)` loop so it iterates `for useg in _chain_by_u_interval(cluster, u_gap_m):` and builds each run from `useg` exactly as before (steps, main step, length/inlier guards, rebase), with two additions: the run length guard uses the *whole segment's* span (`max(u_max) - min(u_min)`), the inlier guard is `sum(x["n_inliers"] for x in useg) >= min_run_inliers`, and the run dict gains `members=[(float(x["u_min"]), float(x["u_max"])) for x in useg]`. The run's `p0`/`p1` span `min(u_min)..max(u_max)` of the segment (holes included).
- [ ] **Step 4:** Run the two new tests → PASS.
- [ ] **Step 5:** Run the whole suite: `venv311\Scripts\python.exe -m pytest tests/recon -v`. Existing `group_wall_runs` tests may assert the OLD defaults' behavior — where a test fails purely because a fixture wall is now (correctly) split or rescued, update that test's expectation and say so in the commit body. Any other failure is a regression: fix before committing.
- [ ] **Step 6:** Commit: `git commit -am "Split wall runs at u-coverage gaps; expose members; rescue-tuned defaults"`

---

## Task 3: Midline recentring (fixes rooms sharing wall internals)

**Files:**
- Modify: `scripts/recon/regularize.py`
- Test: `tests/recon/test_regularize.py`

**Interfaces:**
- Consumes: run dicts with `steps` (rebased: main at 0.0), `offset_m`, `thickness_m`, `thickness_source` (i.e. call AFTER `pair_thickness`).
- Produces: `recenter_walls(walls, points, z_floor, z_ceiling) -> list[dict]` — for each wall with `thickness_source == "measured"` and `thickness_m >= 0.04`, shifts `p0`/`p1`/`offset_m` by `±thickness_m/2` along the wall normal toward its back face, so the centerline is the face-pair midline. Back-face side: a step at `|offset_m| ≈ thickness_m` (±0.06); fallback: the perpendicular point-density peak at `±thickness_m` (±0.07 window, mid-height band `z_floor+0.35 .. z_ceiling−0.35`, ≥50 points). No side found → wall unchanged. Non-mutating (returns new dicts).

- [ ] **Step 1: Write the failing test:**

```python
import numpy as np
from scripts.recon.schema import WallStep
from scripts.recon.regularize import recenter_walls


def _run(offset, steps, thickness, source="measured", direction="x"):
    p0 = (offset, 0.0) if direction == "x" else (0.0, offset)
    p1 = (offset, 4.0) if direction == "x" else (4.0, offset)
    return dict(direction=direction, normal=(1.0, 0.0, 0.0) if direction == "x" else (0.0, 1.0, 0.0),
                offset_m=offset, p0=p0, p1=p1, steps=steps,
                thickness_m=thickness, thickness_source=source)


def test_recenter_shifts_to_midline_using_back_step():
    steps = [WallStep(0.0, 0.0, 4.0, 0.0, 2.7), WallStep(0.20, 0.0, 4.0, 0.0, 2.7)]
    w = _run(5.0, steps, thickness=0.20)
    out = recenter_walls([w], points=np.zeros((0, 3)), z_floor=0.0, z_ceiling=2.7)
    assert abs(out[0]["offset_m"] - 5.10) < 1e-6
    assert abs(out[0]["p0"][0] - 5.10) < 1e-6
    assert w["offset_m"] == 5.0  # input not mutated


def test_recenter_leaves_assumed_walls_alone():
    w = _run(5.0, [WallStep(0.0, 0.0, 4.0, 0.0, 2.7)], thickness=0.10, source="assumed")
    out = recenter_walls([w], points=np.zeros((0, 3)), z_floor=0.0, z_ceiling=2.7)
    assert out[0]["offset_m"] == 5.0
```

- [ ] **Step 2:** Run → FAIL (`recenter_walls` missing).
- [ ] **Step 3:** Implement (port `recenter_to_midline` from `scripts/experiments/diag_floorplan2d_v3.py` — validated on the real scan, it shifted 19/19 walls and made adjacent rooms stop absorbing the shared wall body). Rename to `recenter_walls`, match this module's non-mutating shallow-copy convention, derive the perpendicular axis from `direction` exactly as the experiment does.
- [ ] **Step 4:** Run → PASS. Full suite: `venv311\Scripts\python.exe -m pytest tests/recon -v`.
- [ ] **Step 5:** Commit: `git commit -am "Add recenter_walls: place centerlines on the face-pair midline"`

---

## Task 4: Endpoint-to-line snapping (T-junction closure)

**Files:**
- Modify: `scripts/recon/regularize.py`
- Test: `tests/recon/test_regularize.py`

**Interfaces:**
- Produces: `snap_endpoints_to_lines(walls, reach_m=0.7, dangling_tol_m=0.15) -> list[dict]` — an endpoint farther than `dangling_tol_m` from every other wall's segment is extended/trimmed **along its own centerline** to the nearest intersection with another wall's line, if that intersection is within `reach_m` of the endpoint and within 0.3 m of the other wall's own extent. Run AFTER `resolve_corners` (which handles endpoint-endpoint corners; this handles T-junctions, which endpoint clustering cannot close — real-scan evidence: the 45 m² living room only closed with this pass).

- [ ] **Step 1: Write the failing test:**

```python
from scripts.recon.regularize import snap_endpoints_to_lines


def test_t_junction_endpoint_extends_onto_crossing_wall():
    # wall A vertical x=2, y 0..3.6 (ends 0.4 short of wall B at y=4)
    a = dict(direction="x", offset_m=2.0, p0=(2.0, 0.0), p1=(2.0, 3.6), steps=[])
    b = dict(direction="y", offset_m=4.0, p0=(0.0, 4.0), p1=(6.0, 4.0), steps=[])
    out = snap_endpoints_to_lines([a, b])
    assert abs(out[0]["p1"][1] - 4.0) < 1e-6   # A now reaches B's line
    assert out[1]["p0"] == (0.0, 4.0)          # B untouched


def test_endpoint_beyond_reach_stays():
    a = dict(direction="x", offset_m=2.0, p0=(2.0, 0.0), p1=(2.0, 2.9), steps=[])
    b = dict(direction="y", offset_m=4.0, p0=(0.0, 4.0), p1=(6.0, 4.0), steps=[])
    out = snap_endpoints_to_lines([a, b])  # gap 1.1 m > reach 0.7
    assert out[0]["p1"] == (2.0, 2.9)
```

- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement (port `snap_endpoints_to_lines` from `diag_floorplan2d_v3.py` unchanged in logic: dangling check via point-to-segment distance against all other walls; 2D line-line intersection via cross products; nearest qualifying intersection wins; non-mutating copies).
- [ ] **Step 4:** Run → PASS. Full suite.
- [ ] **Step 5:** Commit: `git commit -am "Add snap_endpoints_to_lines: close T-junction corners"`

---

## Task 5: Trajectory walkthrough detection (gps_time → doors/pathways)

**Files:**
- Modify: `scripts/recon/trajectory.py`
- Test: `tests/recon/test_trajectory.py`

**Interfaces:**
- Consumes: `approx_trajectory` output `(M,3)` (already in this module) and Task 2 wall dicts.
- Produces: `wall_crossings(trajectory, walls, end_margin_m=0.15) -> dict[int, list[float]]` — for each wall index, the sorted list of absolute world-u positions where a consecutive trajectory segment crosses that wall's centerline segment (crossing param within `[end_margin_m, length−end_margin_m]` so corner grazes don't count). Every real doorway the operator walked through appears here; these are the door/pathway seeds for Task 7 — the single strongest opening signal in the data (the scanner PHYSICALLY passed through every door, and through the balcony door onto the balcony).

- [ ] **Step 1: Write the failing test:**

```python
import numpy as np
from scripts.recon.trajectory import wall_crossings


def test_walkthrough_crossing_found_at_door_u():
    wall = dict(direction="y", offset_m=2.0, p0=(0.0, 2.0), p1=(6.0, 2.0), steps=[])
    # walk from room A (y=1) through the wall at x=3.2 into room B (y=3)
    traj = np.array([[3.2, 1.0, 1.2], [3.2, 1.6, 1.2], [3.2, 2.4, 1.2], [3.2, 3.0, 1.2]])
    hits = wall_crossings(traj, [wall])
    assert 0 in hits and len(hits[0]) == 1
    assert abs(hits[0][0] - 3.2) < 1e-6


def test_parallel_walk_never_crosses():
    wall = dict(direction="y", offset_m=2.0, p0=(0.0, 2.0), p1=(6.0, 2.0), steps=[])
    traj = np.array([[0.5, 1.0, 1.2], [5.5, 1.0, 1.2]])
    assert wall_crossings(traj, [wall]) == {}
```

- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement with standard 2D segment-segment intersection:

```python
def wall_crossings(trajectory, walls, end_margin_m: float = 0.15) -> dict:
    """World-u positions where consecutive trajectory steps cross each wall's
    centerline. A crossing = the operator walked through that wall line ->
    a doorway/pathway/balcony-door seed (uses gps_time ordering upstream)."""
    traj = np.asarray(trajectory, dtype=float)[:, :2]
    out = {}
    for wi, w in enumerate(walls):
        p0 = np.asarray(w["p0"], float)
        p1 = np.asarray(w["p1"], float)
        d = p1 - p0
        length = float(np.linalg.norm(d))
        if length == 0:
            continue
        u_i = 1 if w["direction"] == "x" else 0
        hits = []
        for a, b in zip(traj[:-1], traj[1:]):
            e = b - a
            denom = d[0] * e[1] - d[1] * e[0]
            if abs(denom) < 1e-12:
                continue
            r = a - p0
            t = (r[0] * e[1] - r[1] * e[0]) / denom        # along the wall
            s = (r[0] * d[1] - r[1] * d[0]) / denom        # along the step
            if 0.0 <= s <= 1.0 and end_margin_m <= t * length <= length - end_margin_m:
                hits.append(float((p0 + t * d)[u_i]))
        if hits:
            out[wi] = sorted(hits)
    return out
```

- [ ] **Step 4:** Run → PASS. Full suite.
- [ ] **Step 5:** Commit: `git commit -am "Add wall_crossings: gps_time walkthrough detection for door seeds"`

---

## Task 6: Furniture removal

**Files:**
- Modify: `scripts/recon/clean.py`
- Test: `tests/recon/test_clean.py`

**Interfaces:**
- Consumes: `ScanData`, Task 2 wall dicts (with `steps`, `thickness_m`), floor/ceiling z, `Column`/`Beam` lists.
- Produces: `remove_furniture(scan, walls, z_floor, z_ceiling, columns=(), beams=(), dist_m=0.15, floor_band_m=0.12, ceiling_band_m=0.20) -> tuple[ScanData, int]` — keeps a point iff it is within `floor_band_m` of the floor plane, within `ceiling_band_m` of the ceiling plane, within `dist_m` of any wall step's face plane (perpendicular distance to the step's offset, inside the step's u and z extent ± `dist_m`), or inside any column footprint / beam box (± `dist_m`). Everything else (sofas, tables, clutter in room interiors) is dropped. Returns (kept scan, dropped count). Pure numpy.

- [ ] **Step 1: Write the failing test:**

```python
import numpy as np
from scripts.recon.schema import ScanData, WallStep
from scripts.recon.clean import remove_furniture


def test_furniture_blob_dropped_structure_kept():
    rng = np.random.default_rng(0)
    floor = np.column_stack([rng.uniform(0, 6, 3000), rng.uniform(0, 4, 3000), np.zeros(3000)])
    wall = np.column_stack([rng.uniform(0, 6, 3000), np.full(3000, 4.0), rng.uniform(0, 2.7, 3000)])
    sofa = np.column_stack([rng.uniform(2, 3, 2000), rng.uniform(1.5, 2.5, 2000), rng.uniform(0.3, 1.0, 2000)])
    scan = ScanData(xyz=np.vstack([floor, wall, sofa]), gps_time=None, rgb=None, intensity=None)
    walls = [dict(direction="y", offset_m=4.0, p0=(0.0, 4.0), p1=(6.0, 4.0),
                  thickness_m=0.1, steps=[WallStep(0.0, 0.0, 6.0, 0.0, 2.7)])]
    kept, dropped = remove_furniture(scan, walls, z_floor=0.0, z_ceiling=2.7)
    assert dropped >= 1900                      # sofa gone
    assert kept.n >= 5900                       # floor + wall kept
    assert kept.xyz[:, 1].max() > 3.9           # wall points survived
```

- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement: start with a boolean keep-mask of `|z − z_floor| <= floor_band_m` OR `|z − z_ceiling| <= ceiling_band_m`; for each wall and each of its steps, OR-in `(|perp − (offset_m + step.offset_m)| <= dist_m + thickness_m/2) & (u within [step.u_min_m − dist_m, step.u_max_m + dist_m]) & (z within [step.z_min_m − dist_m, step.z_max_m + dist_m])` where `perp`/`u` are the direction-appropriate XY components; for columns use footprint AABB ± `dist_m` over full height; for beams the p0–p1 box ± (`width`, `depth`) ± `dist_m`. Return `scan.subset(mask)` and `int((~mask).sum())`.
- [ ] **Step 4:** Run → PASS. Full suite.
- [ ] **Step 5:** Commit: `git commit -am "Add remove_furniture: keep only points explained by structural elements"`

---

## Task 7: Openings — voids + visibility gate + walkthrough seeds + classify

**Files:**
- Create: `scripts/recon/openings.py`
- Test: `tests/recon/test_openings.py`

**Interfaces:**
- Consumes: wall dicts (Task 2–4 final form), furniture-free `xyz` (Task 6), trajectory `(M,3)`, `wall_crossings` output (Task 5), priors from config.
- Produces (all pure; scipy `cKDTree` allowed):
  - `wall_occupancy(wall, xyz, cell_m=0.03, band_m=None) -> (occ, u0, z0)` — boolean (nu, nz) grid of points within `band_m` (default `thickness_m/2 + 0.08`) of the wall midline, cells `cell_m`; `u0`/`z0` are the grid origin in world u / world z.
  - `find_voids(occ, u0, z0, cell_m, min_w_m=0.55, min_h_m=0.55) -> list[dict]` — rectangles `{u0,u1,z0,z1}` of connected empty regions (scipy `ndimage.label` on `~occ`), each tightened to its bounding box, filtered by min dims. Regions touching the grid's side edges are kept (doors touch the floor edge; balcony voids can touch a side).
  - `visibility_gate(void, wall, trajectory, kdtree, ray_step_m=0.10, clear_r_m=0.07, min_rays=3, max_sensor_dist_m=7.0) -> bool` — cast rays from each trajectory vertex within `max_sensor_dist_m` to 5 sample points inside the void rect (center + 4 midpoints of half-edges), extended 0.4 m past the wall plane; a ray passes if none of its samples (spaced `ray_step_m`, starting 0.4 m from the sensor) has a cloud point within `clear_r_m` (query the prebuilt `cKDTree`). True (a real see-through opening) if ≥ `min_rays` rays pass. A furniture occlusion shadow fails this gate (rays from other positions hit the wall surface); a real opening passes.
  - `refine_edges(void, wall, xyz, search_m=0.15, bin_m=0.02) -> void` — move each of the 4 edges to the density half-max crossing of the wall-band points' u (resp. z) marginal within `±search_m` of the coarse edge; keep the coarse edge where no crossing is found.
  - `classify_opening(void, crossing_us, z_floor, z_ceiling, priors) -> str` with `priors = {"door_h_m": 2.13, "door_h_tol_m": 0.25, "balcony_min_w_m": 1.3, "window_min_sill_m": 0.25}`:
    - sill = `void.z0 − z_floor`; height = `z1 − z0`; width = `u1 − u0`; walked = any crossing u within the void's u-range.
    - floor-touching (sill ≤ 0.15) and (width ≥ `balcony_min_w_m` or (walked and exterior wall)) → `"balcony_door"`;
    - floor-touching and walked → `"door"`; floor-touching, not walked, height within door prior ± tol → `"door"` (low confidence);
    - sill ≥ `window_min_sill_m` and top below `z_ceiling − 0.2` → `"window"`; else `"unknown_opening"`.
  - `detect_openings(walls, xyz, trajectory, crossings, z_floor, z_ceiling, priors, exterior_flags) -> dict[int, list[dict]]` — per wall: occupancy → voids, UNION with a seeded void around every walkthrough crossing that landed inside a `members` coverage hole (rect `u ± 0.55, z_floor..z_floor+priors["door_h_m"]`); gate every void; refine survivors; classify; return `{wall_idx: [{u0,u1,z0,z1,type,width_m,height_m,sill_m,walked,confidence}]}`. Voids failing the gate are healing decisions: the wall stays solid there.

**Design note — horizontal slicing:** the occupancy grid IS a stack of horizontal slices (u × z): a door reads as a floor-to-header void column, a window as a mid-band void, a furniture shadow as a void that fails the visibility gate. Never reduce a wall to a single top-down projection; all evidence stays height-resolved. Additionally, `wall_occupancy` accepts `z_band=(z_lo, z_hi)` (default full height) so callers can request the furniture-free mid-height band `(z_floor + 1.2, z_ceiling − 0.5)` — used by Task 10 as secondary wall-extent evidence where the full-height grid is noisy from clutter.

- [ ] **Step 1: Write failing tests** (`tests/recon/test_openings.py`) — pure-geometry tests first:

```python
import numpy as np
from scripts.recon.openings import wall_occupancy, find_voids, classify_opening

PRIORS = {"door_h_m": 2.13, "door_h_tol_m": 0.25, "balcony_min_w_m": 1.3, "window_min_sill_m": 0.25}
WALL = dict(direction="y", offset_m=2.0, p0=(0.0, 2.0), p1=(6.0, 2.0), thickness_m=0.1, steps=[])


def _wall_points_with_hole(hole_u=(2.0, 2.9), hole_z=(0.0, 2.1), n=60000):
    rng = np.random.default_rng(0)
    u = rng.uniform(0, 6, n); z = rng.uniform(0, 2.7, n)
    keep = ~((u > hole_u[0]) & (u < hole_u[1]) & (z > hole_z[0]) & (z < hole_z[1]))
    return np.column_stack([u[keep], np.full(keep.sum(), 2.0), z[keep]])


def test_occupancy_and_void_finds_door_hole():
    xyz = _wall_points_with_hole()
    occ, u0, z0 = wall_occupancy(WALL, xyz)
    voids = find_voids(occ, u0, z0, cell_m=0.03)
    assert len(voids) == 1
    v = voids[0]
    assert abs(v["u0"] - 2.0) < 0.1 and abs(v["u1"] - 2.9) < 0.1
    assert v["z0"] < 0.1 and abs(v["z1"] - 2.1) < 0.1


def test_classify_door_window_balcony():
    door = {"u0": 2.0, "u1": 2.9, "z0": 0.0, "z1": 2.1}
    window = {"u0": 4.0, "u1": 5.2, "z0": 0.9, "z1": 2.1}
    balcony = {"u0": 1.0, "u1": 3.0, "z0": 0.0, "z1": 2.2}
    assert classify_opening(door, [2.4], 0.0, 2.7, PRIORS) == "door"
    assert classify_opening(window, [], 0.0, 2.7, PRIORS) == "window"
    assert classify_opening(balcony, [1.8], 0.0, 2.7, PRIORS) == "balcony_door"
```

- [ ] **Step 2:** Run → FAIL (module missing).
- [ ] **Step 3:** Implement `wall_occupancy`, `find_voids`, `classify_opening` per the interface above (occupancy: project band points to (u, z), `np.zeros((nu, nz), bool)` + index fill; voids: `scipy.ndimage.label(~occ)`, per-label bounding box, min-dims filter, drop the label containing the full outside border if the grid margin is empty — pad the grid by one occupied cell on all sides first so only interior+edge-touching voids label separately).
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5: Write failing gate + end-to-end fixture tests** (same file — these are the acceptance bar; fixture meta contract: `meta["openings"] = [{"wall": "south", "u_m": (2.0, 2.9), "sill_m": 0.0, "type": "door"}, {"wall": "north", ...window sill 0.9...}, {"wall": "east", "u_m": (1.0, 4.0), "sill_m": 0.0, "type": "balcony_door"}]`):

```python
from scipy.spatial import cKDTree
from scripts.recon.openings import visibility_gate, detect_openings
from tests.fixtures import modular_house


def test_furniture_shadow_fails_gate_real_hole_passes():
    xyz = _wall_points_with_hole()                      # real hole at u 2.0-2.9
    slab = np.column_stack([np.random.default_rng(1).uniform(4.0, 5.0, 8000),
                            np.full(8000, 1.4),         # slab 0.6 m in front of wall
                            np.random.default_rng(2).uniform(0.0, 2.0, 8000)])
    cloud = np.vstack([xyz, slab])
    tree = cKDTree(cloud)
    traj = np.array([[1.0, 0.5, 1.3], [3.0, 0.7, 1.3], [5.0, 0.5, 1.3]])
    real = {"u0": 2.0, "u1": 2.9, "z0": 0.0, "z1": 2.1}
    shadow = {"u0": 4.1, "u1": 4.9, "z0": 0.2, "z1": 1.9}  # behind the slab: unseen, NOT open
    assert visibility_gate(real, WALL, traj, tree) is True
    assert visibility_gate(shadow, WALL, traj, tree) is False


def test_modular_house_openings_end_to_end():
    xyz, gps_time, meta = modular_house()
    # build walls via the real chain (planes are synthetic-perfect in fixture,
    # so construct wall dicts straight from meta walls or run group_wall_runs
    # on fixture planes -- follow test_structure.py's existing pattern)
    ...  # see Step 6
```

- [ ] **Step 6:** Implement `visibility_gate`, `refine_edges`, `detect_openings`. For the end-to-end test, reuse whatever helper `tests/recon/test_structure.py` already uses to get runs from `modular_house()` (do not invent a second fixture path); assert: the three fixture openings are recovered with correct `type` and `u`/`sill` dims within ±0.05 m; the neighbour blob does not suppress the balcony door; no opening is reported on any wall stretch the fixture lists as solid.
- [ ] **Step 7:** Run → PASS: `venv311\Scripts\python.exe -m pytest tests/recon/test_openings.py -v`. Full suite.
- [ ] **Step 8:** Commit: `git commit -am "Add recon.openings: voids, trajectory visibility gate, walkthrough seeds, prior-aware classification"`

---

## Task 8: Dimensions, confidence, manifest

**Files:**
- Modify: `scripts/recon/schema.py`
- Test: `tests/recon/test_schema.py`

**Interfaces:**
- Produces:
  - `@dataclass ElementDims(width_m: float, height_m: float, depth_m: float, source: str, flags: list)` (`source` ∈ {"measured", "refined", "assumed"}).
  - `check_priors(kind: str, dims: ElementDims, priors: dict) -> list[str]` — pure; returns human-readable flags, e.g. door height outside 1.85–2.35 → `"door_height_unusual: 2.55m (prior 2.13m)"`; ceiling outside 2.35–3.05 → `"ceiling_height_unusual"`. Flags never delete elements — they surface for review (same philosophy as `extract_unclassified`).
  - `build_manifest(walls, openings, columns, beams, rooms, z_floor, z_ceiling, config) -> dict` — JSON-ready: per-element ids (`wall_000`…, reuse `new_wall_id`/`new_column_id`/`new_beam_id`), `p0/p1/length_m/thickness_m/thickness_source/steps`, opening dims + `type` + `walked` + `confidence` + flags, column footprint + height, beam span/width/depth, room areas, storey heights, config echo.

- [ ] **Step 1: Write the failing tests:**

```python
from scripts.recon.schema import ElementDims, check_priors

PRIORS = {"door_h_m": 2.13, "door_h_tol_m": 0.25, "ceiling_m": 2.75, "ceiling_tol_m": 0.35}

def test_door_height_prior_flags():
    ok = ElementDims(0.9, 2.10, 0.1, "measured", [])
    odd = ElementDims(0.9, 2.60, 0.1, "measured", [])
    assert check_priors("door", ok, PRIORS) == []
    assert any("door_height" in f for f in check_priors("door", odd, PRIORS))

def test_manifest_roundtrips_to_json():
    import json
    from scripts.recon.schema import build_manifest
    m = build_manifest([], {}, [], [], [], z_floor=-0.25, z_ceiling=2.50, config={"seed": 0})
    assert json.loads(json.dumps(m))["storey"]["height_m"] == 2.75
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS, full suite.
- [ ] **Step 5:** Commit: `git commit -am "Add element dims, prior sanity flags, and manifest builder"`

---

## Task 9: Solids wiring — sharp-edged model assembly

**Files:**
- Modify: `scripts/recon/solids.py` (one new function), `scripts/recon/assemble.py` only if a collection name is missing
- Test: `tests/recon/test_solids.py`

**Interfaces:**
- Consumes: existing `wall_to_solid(wall, steps)`, `cut_openings(wall_solid, openings, wall)`, `column_to_solid`, `beam_to_solid`, `slab_to_solid`, `build_room_polygons`, `build_scene`, `write_glb`.
- Produces: `build_elements(walls, openings_by_wall, columns, beams, rooms, z_floor, z_ceiling, slab_m=0.12) -> dict[str, list[trimesh.Trimesh]]` with keys `"Walls","Floor","Ceiling","Doors","Windows","Columns","Beams"`:
  - each wall → `wall_to_solid` then `cut_openings` with its openings (all openings cut; sharp edges come from the boolean, no smoothing anywhere);
  - floor/ceiling slab footprint = `shapely.unary_union(rooms).buffer(max wall thickness/2, join_style=2)` (mitre join keeps corners sharp) extruded `slab_m` down (floor) / up (ceiling) via `slab_to_solid`;
  - every door/window/balcony-door also emits a thin marker panel solid (opening rect × wall thickness) into `"Doors"`/`"Windows"` (balcony doors go to `"Doors"` with name prefix `balcony_door_`) so Blender shows openings as selectable named objects;
  - every returned mesh must be watertight (`_ensure_watertight`, already in the module).

- [ ] **Step 1: Write the failing test:**

```python
import numpy as np
from shapely.geometry import Polygon
from scripts.recon.solids import build_elements
from scripts.recon.schema import WallStep


def test_build_elements_cuts_door_and_is_watertight():
    wall = dict(direction="y", offset_m=0.0, p0=(0.0, 0.0), p1=(6.0, 0.0),
                thickness_m=0.1, floor_z_m=0.0, ceiling_z_m=2.7,
                steps=[WallStep(0.0, 0.0, 6.0, 0.0, 2.7)])
    door = {"u0": 2.0, "u1": 2.9, "z0": 0.0, "z1": 2.1, "type": "door"}
    rooms = [Polygon([(0, 0), (6, 0), (6, 4), (0, 4)])]
    els = build_elements([wall], {0: [door]}, [], [], rooms, 0.0, 2.7)
    w = els["Walls"][0]
    assert w.is_watertight
    full = 6.0 * 0.1 * 2.7
    cut = 0.9 * 0.1 * 2.1
    assert abs(w.volume - (full - cut)) < 0.02
    assert len(els["Doors"]) == 1 and els["Floor"][0].is_watertight
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement `build_elements` (openings dicts are adapted into whatever rect/`Opening` shape `cut_openings` already consumes — read its signature in `solids.py` first and adapt there, in one place). **Step 4:** Run → PASS, full suite.
- [ ] **Step 5:** Commit: `git commit -am "Add build_elements: watertight sharp-edged model assembly with opening panels"`

---

## Task 10: isolidarflow CLI — end-to-end + real-scan validation

**Files:**
- Create: `scripts/isolidarflow.py`
- Test: `tests/recon/test_isolidarflow_cli.py`
- Modify: `README.md` (new pipeline section)

**Interfaces:**
- Produces: `run(in_path, out_dir, config=DEFAULT_CONFIG) -> dict` and CLI `venv311\Scripts\python.exe scripts\isolidarflow.py <scan.las> <out_dir> [--keep-furniture] [--seed N] [--debug-png]`. Stage order: `load_scan → percentile_crop → remove_outliers → approx_trajectory → select_z_band → isolate_unit → estimate_normals/dominant_axes/axis_align → detect_planes(seed) → group_wall_runs → extract_columns_beams → extract_unclassified (report) → snap_walls → pair_thickness → recenter_walls → resolve_corners → snap_endpoints_to_lines → remove_furniture → wall_crossings → detect_openings → build_room_polygons → build_manifest → build_elements → write_glb + write_dxf + write_svg + manifest.json + report.txt (+ debug PNG in the v3 experiment's solid-wall style when --debug-png)`.
- `DEFAULT_CONFIG` dict holds every threshold from Tasks 1–9 with a one-line comment each (including the priors block) — no magic numbers inline.

- [ ] **Step 1: Write the failing CLI test** — synthetic `modular_house()` written to a temp LAS (reuse the writer helper already in `tests/recon/test_io_las.py`), run `run(...)`, assert: `model.glb`, `floorplan.dxf`, `floorplan.svg`, `manifest.json`, `report.txt` all exist; manifest has ≥4 walls, exactly 1 column, ≥1 beam, and openings containing one `door`, one `window`, one `balcony_door`; no element inside `meta["neighbour_bbox"]`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement `run` + `__main__` (thin orchestration only — every stage is an existing tested function; on stage failure write the partial `report.txt` and re-raise).
- [ ] **Step 4:** Run → PASS, full suite green.
- [ ] **Step 5: Real-scan validation (manual, results recorded in the commit body):**

```
venv311\Scripts\python.exe scripts\isolidarflow.py ..\..\..\koushikexport.las output\koushik_isolidar --debug-png
venv311\Scripts\python.exe scripts\isolidarflow.py ..\..\..\mujammelexport.las output\mujammel_isolidar --debug-png
```

Acceptance on koushik: ≥7 rooms close; every trajectory walkthrough produced a door or balcony_door; balcony door present on the balcony wall; door heights within prior tolerance or flagged; GLB opens in Blender with the 7 collections and separate named objects; walls read sharp (no rounded corners) in the debug PNG and GLB.

Acceptance on mujammel: pipeline completes with comparable element counts; since this scan has RGB, the debug PNG additionally renders the mean RGB of each wall band (evidence for a future color-assisted opening/frame detector — record observations in the commit body, no RGB logic required in v1).
- [ ] **Step 6:** Update `README.md` (pipeline section: how to run, outputs, config). Commit: `git commit -am "Add isolidarflow CLI: end-to-end sharp-edged modular reconstruction"`

---

## Self-Review (author check)

- **Spec coverage:** sharp edges → boolean-cut watertight solids, mitre-join slabs (T9); beams/columns → existing extractors wired through manifest + solids (T8–T10); doors/windows/balcony doors → walkthrough seeds (T5) + voids + visibility gate + prior classification (T7); accurate 3D dims → refine_edges + measured thickness + midline recentring + prior flags (T3, T7, T8); gps_time usage → T5 (crossings) and T7 (gate rays), both feeding T10; furniture cleanup → T6; balcony detection → exterior-wall + walked + width rule (T7); reproducibility → T1. User priors (7 ft doors, 2750 mm ceilings, room-varying door widths) → priors config + confidence flags, never hard rejections (T7/T8).
- **Placeholder scan:** the `...` in Task 7 Step 5's second test is resolved by Step 6's explicit instruction (reuse `test_structure.py`'s existing fixture→runs helper); all other steps carry complete code or exact port instructions from committed, real-scan-validated experiment code.
- **Type consistency:** wall dicts flow T2→T3→T4→T6→T7→T9 with the same keys (`direction`, `offset_m`, `p0`, `p1`, `steps`, `members`, `thickness_m`, `thickness_source`); openings dicts `{u0,u1,z0,z1,type,...}` defined in T7 and consumed in T9/T8; `ElementDims`/`build_manifest` defined in T8, consumed in T10.
- **Supersedes:** Tasks 13 and 17 of `2026-07-02-plane-first-modular-reconstruction.md` are replaced by Tasks 5–10 here; that plan's Tasks 1–12 and 14–16 are complete and reused as-is.
