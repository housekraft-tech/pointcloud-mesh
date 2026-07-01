import numpy as np
import laspy

from scripts.reconstruct import run_isolation, DEFAULT_CONFIG, _write_report


def _write_las(path, xyz, t):
    h = laspy.LasHeader(point_format=3, version="1.2")
    h.scales = [0.001, 0.001, 0.001]
    h.offsets = [0.0, 0.0, 0.0]
    las = laspy.LasData(h)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.gps_time = t
    las.write(str(path))


def _unit(rng, n=6000, footprint=4.0, z_ceiling=2.5):
    fx = rng.uniform(0, footprint, n)
    fy = rng.uniform(0, footprint, n)
    floor = np.column_stack([fx, fy, np.zeros(n)])
    ceil = np.column_stack([fx, fy, np.full(n, z_ceiling)])
    wz = rng.uniform(0, z_ceiling, n)
    wall = np.column_stack([rng.choice([0.0, footprint], n), rng.uniform(0, footprint, n), wz])
    return np.vstack([floor, ceil, wall])


def test_run_isolation_removes_neighbour_and_drift(tmp_path):
    rng = np.random.default_rng(0)
    unit = _unit(rng)
    # neighbour building across a 10 m gap, other floor heights
    neigh = np.column_stack([
        rng.uniform(14, 18, 8000),
        rng.uniform(0, 4, 8000),
        rng.uniform(0, 12, 8000),
    ])
    # sparse drift specks very far away
    drift = np.column_stack([rng.uniform(-40, 40, 50), rng.uniform(-40, 40, 50), rng.uniform(-6, 15, 50)])
    xyz = np.vstack([unit, neigh, drift])
    # gps_time: unit scanned first (walk inside), then neighbour glimpsed
    t = np.concatenate([
        np.linspace(0, 100, len(unit)),
        np.linspace(100, 130, len(neigh)),
        np.linspace(130, 131, len(drift)),
    ])
    p = tmp_path / "scan.las"
    _write_las(p, xyz, t)

    config = dict(DEFAULT_CONFIG)
    config["remove_outliers"] = False  # keep the test fast + deterministic
    out, stats = run_isolation(str(p), str(tmp_path / "out"), config)

    # z-band is the unit storey (0 .. 2.5)
    assert abs(stats["z_floor"] - 0.0) < 0.2
    assert abs(stats["z_ceiling"] - 2.5) < 0.2
    # neighbour (x in 14..18) removed
    assert out.n > 0
    assert out.xyz[:, 0].max() < 6.0
    assert stats["dropped"] >= 8000

    lines = _write_report(str(tmp_path / "out"), str(p), stats, out)
    assert any("footprint" in ln for ln in lines)
