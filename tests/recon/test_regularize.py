import numpy as np
import pytest

from scripts.recon.schema import WallStep
from scripts.recon.regularize import snap_walls, resolve_corners, pair_thickness, recenter_walls


def _wall(p0, p1, normal=None, offset_m=None, steps=None, direction="x"):
    w = dict(direction=direction, p0=tuple(p0), p1=tuple(p1))
    if normal is not None:
        w["normal"] = tuple(normal)
    if offset_m is not None:
        w["offset_m"] = offset_m
    if steps is not None:
        w["steps"] = steps
    return w


# ---------------------------------------------------------------------------
# snap_walls
# ---------------------------------------------------------------------------

def test_snap_walls_snaps_within_tolerance_to_exact_axis():
    """A wall a few degrees off the X axis (within angle_tol_deg) must come
    back exactly axis-aligned (direction parallel to (1, 0)), with its
    length and midpoint preserved."""
    axes = np.eye(3)
    # ~3 degrees off the x-axis: direction (cos3, sin3)
    theta = np.radians(3.0)
    d = np.array([np.cos(theta), np.sin(theta)])
    mid = np.array([2.0, 5.0])
    length = 4.0
    p0 = mid - d * length / 2
    p1 = mid + d * length / 2

    walls = [_wall(p0, p1, normal=(0.0, 1.0, 0.0), offset_m=5.0)]
    out = snap_walls(walls, axes, angle_tol_deg=8.0)

    new_p0 = np.asarray(out[0]["p0"])
    new_p1 = np.asarray(out[0]["p1"])
    new_d = new_p1 - new_p0
    new_d = new_d / np.linalg.norm(new_d)

    # exactly axis-aligned (parallel to x-axis)
    assert abs(abs(new_d[0]) - 1.0) < 1e-9
    assert abs(new_d[1]) < 1e-9
    # length preserved
    assert abs(np.linalg.norm(new_p1 - new_p0) - length) < 1e-6
    # midpoint preserved
    new_mid = (new_p0 + new_p1) / 2
    assert np.allclose(new_mid, mid, atol=1e-6)
    # normal rotated to stay perpendicular, correct sign preserved
    new_normal = np.asarray(out[0]["normal"])
    assert abs(new_normal[0]) < 1e-6
    assert new_normal[1] > 0


def test_snap_walls_leaves_off_axis_wall_unchanged():
    """A wall clearly not aligned to either Manhattan axis (e.g. 40 degrees
    off both) must be left with its own original PCA-fit direction --
    snapping it would falsely force a non-orthogonal real feature onto a
    Manhattan axis."""
    axes = np.eye(3)
    theta = np.radians(40.0)
    d = np.array([np.cos(theta), np.sin(theta)])
    p0 = np.array([0.0, 0.0])
    p1 = p0 + d * 3.0

    walls = [_wall(p0, p1)]
    out = snap_walls(walls, axes, angle_tol_deg=8.0)

    assert np.allclose(out[0]["p0"], p0)
    assert np.allclose(out[0]["p1"], p1)


def test_snap_walls_does_not_mutate_input():
    axes = np.eye(3)
    theta = np.radians(2.0)
    d = np.array([np.cos(theta), np.sin(theta)])
    p0 = np.array([0.0, 0.0])
    p1 = p0 + d * 3.0
    walls = [_wall(p0, p1)]
    original_p1 = walls[0]["p1"]
    snap_walls(walls, axes, angle_tol_deg=8.0)
    assert walls[0]["p1"] == original_p1


# ---------------------------------------------------------------------------
# resolve_corners
# ---------------------------------------------------------------------------

def _perp_dist_to_line(pt, line_p0, line_p1):
    p0 = np.asarray(line_p0, dtype=float)
    p1 = np.asarray(line_p1, dtype=float)
    d = p1 - p0
    d = d / np.linalg.norm(d)
    v = np.asarray(pt, dtype=float) - p0
    perp = v - np.dot(v, d) * d
    return float(np.linalg.norm(perp))


def test_resolve_corners_true_intersection_not_average():
    """Two walls at a genuine ~90 degree corner, with a realistic
    150-200mm raw endpoint gap (their own lines are exact -- only the
    endpoint truncation is off, mirroring real RANSAC-plane wall
    geometry): the resolved shared corner must land near BOTH walls' own
    infinite lines (proving real intersection), not merely split the raw
    gap in half (a plain average would NOT land on either line)."""
    # Wall A: exact line y = 0, but its own endpoint stops short of the
    # true corner (x=5).
    a_p0 = (0.0, 0.0)
    a_p1 = (4.85, 0.0)
    # Wall B: exact line x = 5, endpoint stops short of true corner (y=0).
    b_p0 = (5.0, 0.15)
    b_p1 = (5.0, 3.0)

    walls = [_wall(a_p0, a_p1), _wall(b_p0, b_p1, direction="y")]
    out = resolve_corners(walls, tol_m=0.25)

    resolved_a = out[0]["p1"]
    resolved_b = out[1]["p0"]

    # both walls' corner endpoint resolved to the SAME point
    assert np.allclose(resolved_a, resolved_b, atol=1e-6)

    # that point lands almost exactly at the true geometric corner (5, 0)
    assert np.linalg.norm(np.asarray(resolved_a) - np.array([5.0, 0.0])) < 1e-6

    # perpendicular distance to EACH wall's own original line is ~0 --
    # a plain centroid of the raw gap would NOT satisfy this.
    dist_to_a_line = _perp_dist_to_line(resolved_a, a_p0, a_p1)
    dist_to_b_line = _perp_dist_to_line(resolved_a, b_p0, b_p1)
    assert dist_to_a_line < 1e-6
    assert dist_to_b_line < 1e-6

    # sanity: naive midpoint-of-raw-gap would NOT be on either line
    naive_mid = (np.asarray(a_p1) + np.asarray(b_p0)) / 2
    assert _perp_dist_to_line(naive_mid, a_p0, a_p1) > 0.01


def test_resolve_corners_guards_against_self_wall_collapse():
    """A short wall X, positioned so that BOTH of its own endpoints sit
    within tol_m of a third wall's bridging endpoint (and thus within
    tol_m of each other too), must NOT have both endpoints merged into one
    shared cluster -- that would collapse X's own length to ~0."""
    x_p0 = (0.0, 0.0)
    x_p1 = (0.15, 0.15)  # length ~0.212, well within tol_m of itself
    y_p0 = (0.05, 0.05)  # bridging point, within tol_m of BOTH x_p0/x_p1
    y_p1 = (10.0, 0.05)  # far away, not involved in clustering

    walls = [
        _wall(x_p0, x_p1, direction="xy"),
        _wall(y_p0, y_p1, direction="x"),
    ]
    out = resolve_corners(walls, tol_m=0.25)

    resolved_x_p0 = np.asarray(out[0]["p0"])
    resolved_x_p1 = np.asarray(out[0]["p1"])
    collapsed_length = np.linalg.norm(resolved_x_p1 - resolved_x_p0)

    # X's own two endpoints must not have been pulled onto each other
    assert collapsed_length > 0.1
    assert out[0]["length_m"] > 0.1


def test_resolve_corners_near_parallel_falls_back_to_centroid():
    """Two near-parallel (same-direction) wall segments with a small gap
    between them (e.g. two pieces of one wall run that should just
    concatenate) must resolve to the plain centroid of the raw endpoints,
    not an unstable/undefined line-intersection extrapolation."""
    a_p0 = (0.0, 0.0)
    a_p1 = (3.0, 0.0)
    b_p0 = (3.1, 0.05)  # nearly collinear with A, tiny gap
    b_p1 = (6.0, 0.05)

    walls = [_wall(a_p0, a_p1), _wall(b_p0, b_p1)]
    out = resolve_corners(walls, tol_m=0.25)

    expected_centroid = (np.asarray(a_p1) + np.asarray(b_p0)) / 2
    assert np.allclose(out[0]["p1"], expected_centroid, atol=1e-9)
    assert np.allclose(out[1]["p0"], expected_centroid, atol=1e-9)


def test_resolve_corners_does_not_mutate_input():
    walls = [_wall((0.0, 0.0), (4.85, 0.0)), _wall((5.0, 0.15), (5.0, 3.0), direction="y")]
    original = walls[0]["p1"]
    resolve_corners(walls, tol_m=0.25)
    assert walls[0]["p1"] == original


# ---------------------------------------------------------------------------
# pair_thickness
# ---------------------------------------------------------------------------

def test_pair_thickness_measured_from_run_steps():
    """A wall run whose own steps already carry a full-length back-face
    step (as structure.group_wall_runs produces for a normal double-sided
    wall) gets thickness_source='measured' with the correct gap."""
    run = _wall(
        (0.0, 0.0), (0.0, 5.0),
        normal=(1.0, 0.0, 0.0), offset_m=0.0,
        steps=[
            WallStep(offset_m=0.0, u_min_m=0.0, u_max_m=5.0, z_min_m=0.0, z_max_m=2.7),
            WallStep(offset_m=0.2, u_min_m=0.0, u_max_m=5.0, z_min_m=0.0, z_max_m=2.7),
        ],
    )
    out = pair_thickness([run], points=np.zeros((0, 3)), default_m=0.10)
    assert out[0]["thickness_source"] == "measured"
    assert abs(out[0]["thickness_m"] - 0.2) < 1e-6


def test_pair_thickness_ignores_short_relief_step_as_back_face():
    """A pillar/pilaster relief step (short along-wall extent, fully
    contained within the main step's u-range) must NOT be mistaken for the
    wall's back face, even though it is within the plausible thickness
    range."""
    run = _wall(
        (0.0, 0.0), (0.0, 5.0),
        normal=(1.0, 0.0, 0.0), offset_m=0.0,
        steps=[
            WallStep(offset_m=0.0, u_min_m=0.0, u_max_m=5.0, z_min_m=0.0, z_max_m=2.7),
            # short pillar-front relief step, well within thickness bounds
            # but only covering u in [2.0, 2.3] -- not a real back face.
            WallStep(offset_m=0.3, u_min_m=2.0, u_max_m=2.3, z_min_m=0.0, z_max_m=2.7),
        ],
    )
    out = pair_thickness([run], points=np.zeros((0, 3)), default_m=0.10)
    assert out[0]["thickness_source"] == "assumed"
    assert out[0]["thickness_m"] == 0.10


def test_pair_thickness_measured_via_point_search_fallback():
    """When the run's own steps don't carry a back-face candidate, but the
    raw point cloud has genuine double-sided evidence nearby, pair_thickness
    must find it by searching the points directly."""
    rng = np.random.default_rng(0)
    n = 400
    ys = rng.uniform(0.0, 5.0, n)
    zs = rng.uniform(0.0, 2.7, n)
    front = np.column_stack([np.zeros(n), ys, zs])
    ys2 = rng.uniform(0.0, 5.0, n)
    zs2 = rng.uniform(0.0, 2.7, n)
    back = np.column_stack([np.full(n, 0.18), ys2, zs2])
    points = np.vstack([front, back])

    run = _wall(
        (0.0, 0.0), (0.0, 5.0),
        normal=(1.0, 0.0, 0.0), offset_m=0.0,
        steps=[WallStep(offset_m=0.0, u_min_m=0.0, u_max_m=5.0, z_min_m=0.0, z_max_m=2.7)],
    )
    out = pair_thickness([run], points=points, default_m=0.10)
    assert out[0]["thickness_source"] == "measured"
    assert abs(out[0]["thickness_m"] - 0.18) < 0.02


def test_pair_thickness_assumed_for_single_sided_wall():
    """A wall with only ever one face detected (common with a handheld
    scanner that only walks the interior) gets thickness_source='assumed'
    and thickness_m == default_m."""
    rng = np.random.default_rng(1)
    n = 400
    ys = rng.uniform(0.0, 5.0, n)
    zs = rng.uniform(0.0, 2.7, n)
    front = np.column_stack([np.zeros(n), ys, zs])

    run = _wall(
        (0.0, 0.0), (0.0, 5.0),
        normal=(1.0, 0.0, 0.0), offset_m=0.0,
        steps=[WallStep(offset_m=0.0, u_min_m=0.0, u_max_m=5.0, z_min_m=0.0, z_max_m=2.7)],
    )
    out = pair_thickness([run], points=front, default_m=0.10)
    assert out[0]["thickness_source"] == "assumed"
    assert out[0]["thickness_m"] == 0.10


def test_pair_thickness_does_not_mutate_input():
    run = _wall(
        (0.0, 0.0), (0.0, 5.0),
        normal=(1.0, 0.0, 0.0), offset_m=0.0,
        steps=[
            WallStep(offset_m=0.0, u_min_m=0.0, u_max_m=5.0, z_min_m=0.0, z_max_m=2.7),
            WallStep(offset_m=0.2, u_min_m=0.0, u_max_m=5.0, z_min_m=0.0, z_max_m=2.7),
        ],
    )
    walls = [run]
    pair_thickness(walls, points=np.zeros((0, 3)))
    assert "thickness_m" not in walls[0]


# ---------------------------------------------------------------------------
# recenter_walls
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Integration test on modular_house(): full regularize pipeline
# ---------------------------------------------------------------------------

try:
    import open3d  # noqa: F401
    HAVE_O3D = True
except Exception:
    HAVE_O3D = False


@pytest.mark.skipif(not HAVE_O3D, reason="Open3D not available")
def test_regularize_pipeline_on_modular_house():
    from scripts.recon.planes import detect_planes
    from scripts.recon.structure import group_wall_runs
    from tests.fixtures import modular_house

    pts, _, meta = modular_house()
    pts = pts[pts[:, 0] < 10.0]  # drop the far neighbour blob
    planes = detect_planes(
        pts, z_floor=meta["z_floor_m"], z_ceiling=meta["z_ceiling_m"], min_inliers=800
    )
    vertical_planes = [p for p in planes if p.label == "vertical"]
    axes = np.eye(3)  # cloud already Manhattan-aligned by construction

    runs = group_wall_runs(vertical_planes, pts, axes)
    assert len(runs) == 4

    snapped = snap_walls(runs, axes, angle_tol_deg=8.0)

    # snapped directions within 0.5 degrees of the nearest dominant axis
    ex, ey = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    for w in snapped:
        d = np.asarray(w["p1"]) - np.asarray(w["p0"])
        d = d / np.linalg.norm(d)
        angle_x = np.degrees(np.arccos(np.clip(abs(np.dot(d, ex)), -1, 1)))
        angle_y = np.degrees(np.arccos(np.clip(abs(np.dot(d, ey)), -1, 1)))
        assert min(angle_x, angle_y) < 0.5

    resolved = resolve_corners(snapped, tol_m=0.3)

    # every wall's endpoint must be shared, within 1cm, by some other wall's
    # endpoint (each of the 4 corners of the rectangular room is shared by
    # exactly two of the four wall runs).
    all_endpoints = []
    for wi, w in enumerate(resolved):
        all_endpoints.append((wi, "p0", np.asarray(w["p0"])))
        all_endpoints.append((wi, "p1", np.asarray(w["p1"])))

    for wi, key, pt in all_endpoints:
        best = min(
            (np.linalg.norm(pt - pt2) for (wi2, _k2, pt2) in all_endpoints if wi2 != wi),
            default=None,
        )
        assert best is not None and best < 0.01, (
            f"wall {wi} endpoint {key}={pt} has no matching corner within 1cm (closest {best})"
        )

    thick = pair_thickness(resolved, pts, default_m=0.10)
    for w in thick:
        assert w["thickness_source"] == "measured"
        assert abs(w["thickness_m"] - meta["wall_thickness_m"]) < 0.03
