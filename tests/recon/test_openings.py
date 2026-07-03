import numpy as np
import pytest
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


# ---------------------------------------------------------------------------
# visibility gate + end-to-end fixture test
# ---------------------------------------------------------------------------

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


try:
    import open3d  # noqa: F401
    HAVE_O3D = True
except Exception:
    HAVE_O3D = False


@pytest.mark.skipif(not HAVE_O3D, reason="Open3D not available")
def test_modular_house_openings_end_to_end():
    from scripts.recon.planes import detect_planes
    from scripts.recon.structure import group_wall_runs
    from scripts.recon.regularize import snap_walls, pair_thickness, recenter_walls, resolve_corners, snap_endpoints_to_lines
    from scripts.recon.trajectory import approx_trajectory, wall_crossings

    pts, gps_time, meta = modular_house()
    pts_local = pts[pts[:, 0] < 10.0]  # drop the far neighbour blob for plane detection
    keep_mask = pts[:, 0] < 10.0

    planes = detect_planes(
        pts_local, z_floor=meta["z_floor_m"], z_ceiling=meta["z_ceiling_m"], min_inliers=800
    )
    vertical_planes = [p for p in planes if p.label == "vertical"]
    axes = np.eye(3)

    runs = group_wall_runs(vertical_planes, pts_local, axes)
    assert len(runs) == 4

    z_floor = meta["z_floor_m"]
    z_ceiling = meta["z_ceiling_m"]

    walls = snap_walls(runs, axes)
    walls = pair_thickness(walls, pts_local)
    walls = recenter_walls(walls, pts_local, z_floor, z_ceiling)
    walls = resolve_corners(walls)
    walls = snap_endpoints_to_lines(walls)

    # identify each wall by its ground-truth centerline side (south/north/east/west)
    def _side(w):
        # direction is the wall's NORMAL axis (group_wall_runs convention):
        # a wall with normal along y (i.e. direction="y") is the south/north
        # wall (its centerline runs along x); a wall with normal along x
        # (direction="x") is the west/east wall (centerline runs along y).
        p0 = np.array(w["p0"]); p1 = np.array(w["p1"])
        mid = (p0 + p1) / 2.0
        if w["direction"] == "y":
            return "south" if mid[1] < 2.5 else "north"
        return "west" if mid[0] < 3.0 else "east"

    sides = {wi: _side(w) for wi, w in enumerate(walls)}

    # use the FULL cloud (including neighbour blob) for occupancy/gate, as a
    # real pipeline would -- the balcony's neighbour blob must not suppress
    # detection of the balcony void itself.
    traj = approx_trajectory(gps_time[keep_mask], pts_local)
    crossings = wall_crossings(traj, walls)

    priors = {"door_h_m": 2.13, "door_h_tol_m": 0.25, "balcony_min_w_m": 1.3, "window_min_sill_m": 0.25}
    # exterior_flags intentionally omitted (defaults to "not asserted
    # exterior" for every wall): this pipeline doesn't yet have a real
    # exterior/interior wall classifier upstream (a later task's job), and
    # the fixture's own ground truth requires the narrow (0.9 m), walked
    # south door to classify as "door" even though it in fact sits on an
    # exterior wall -- only the wide (3.0 m >= balcony_min_w_m) east opening
    # should read as "balcony_door", via the width rule alone. Asserting
    # exterior=True for every wall here would make the (walked and
    # exterior) clause misclassify the ordinary south door as a balcony
    # door too (see classify_opening's own docstring for why that clause
    # defaults off).
    result = detect_openings(walls, pts, traj, crossings, z_floor, z_ceiling, priors)

    by_side = {}
    for wi, opes in result.items():
        for o in opes:
            by_side.setdefault(sides[wi], []).append(o)

    gt_by_side = {g["wall"]: g for g in meta["openings"]}

    for side, gt in gt_by_side.items():
        assert side in by_side and len(by_side[side]) >= 1, f"no opening recovered on {side} wall"
        # pick the opening on this side closest in u to ground truth
        gt_u0, gt_u1 = gt["u_m"]
        best = min(by_side[side], key=lambda o: abs(o["u0"] - gt_u0) + abs(o["u1"] - gt_u1))
        assert best["type"] == gt["type"], f"{side}: expected {gt['type']}, got {best['type']}"
        assert abs(best["u0"] - gt_u0) < 0.05, f"{side} u0: {best['u0']} vs {gt_u0}"
        assert abs(best["u1"] - gt_u1) < 0.05, f"{side} u1: {best['u1']} vs {gt_u1}"
        assert abs(best["sill_m"] - gt["sill_m"]) < 0.05, f"{side} sill: {best['sill_m']} vs {gt['sill_m']}"

    # no opening reported on a stretch the fixture lists as solid: west wall
    # (pillar wall) has no openings in meta.
    assert "west" not in by_side or len(by_side.get("west", [])) == 0
