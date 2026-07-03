import numpy as np

from scripts.recon.trajectory import approx_trajectory, load_trajectory, wall_crossings


def test_approx_trajectory_follows_time_ordered_line():
    # points march along +X as gps_time increases; small y/z jitter
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 10.0, 2000)
    x = t  # 0..10 along X
    xyz = np.column_stack([x, rng.normal(0, 0.01, t.size), rng.normal(0, 0.01, t.size)])
    path = approx_trajectory(t, xyz, dt_s=0.25)
    assert path.shape[0] >= 30
    # path is monotonic in X and spans the line
    assert path[0, 0] < 1.0
    assert path[-1, 0] > 9.0
    assert np.all(np.diff(path[:, 0]) > -0.1)  # roughly increasing


def test_approx_trajectory_empty():
    assert approx_trajectory(np.array([]), np.zeros((0, 3))).shape == (0, 3)


def test_load_trajectory_time_xyz(tmp_path):
    p = tmp_path / "traj.txt"
    p.write_text("# t x y z\n2.0 5 5 5\n1.0 0 0 0\n3.0 9 0 0\n")
    path = load_trajectory(str(p))
    # sorted by time -> starts at origin, ends at (9,0,0)
    assert np.allclose(path[0], [0, 0, 0])
    assert np.allclose(path[-1], [9, 0, 0])


def test_load_trajectory_xyz_only(tmp_path):
    p = tmp_path / "traj.csv"
    p.write_text("0,0,0\n1,2,3\n")
    path = load_trajectory(str(p))
    assert path.shape == (2, 3)
    assert np.allclose(path[1], [1, 2, 3])


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
