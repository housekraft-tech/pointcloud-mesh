import numpy as np

from scripts.recon.schema import ScanData
from scripts.recon.isolate import select_z_band, isolate_unit


def _floor_ceiling_walls(rng, z_floor=0.0, z_ceiling=2.5, footprint=4.0, n=4000):
    # floor slab
    fx = rng.uniform(0, footprint, n)
    fy = rng.uniform(0, footprint, n)
    floor = np.column_stack([fx, fy, np.full(n, z_floor)])
    ceil = np.column_stack([fx, fy, np.full(n, z_ceiling)])
    # wall fill (four edges), fewer points per height
    wz = rng.uniform(z_floor, z_ceiling, n)
    wx = rng.choice([0.0, footprint], n)
    wall = np.column_stack([wx, rng.uniform(0, footprint, n), wz])
    return np.vstack([floor, ceil, wall])


def test_select_z_band_finds_floor_and_ceiling():
    rng = np.random.default_rng(0)
    pts = _floor_ceiling_walls(rng, z_floor=0.0, z_ceiling=2.5)
    # neighbour floors seen through balcony: scattered at 6..15 m up
    neigh = np.column_stack([rng.uniform(20, 24, 500), rng.uniform(20, 24, 500), rng.uniform(6, 15, 500)])
    z = np.vstack([pts, neigh])[:, 2]
    z_floor, z_ceiling = select_z_band(z)
    assert abs(z_floor - 0.0) < 0.1
    assert abs(z_ceiling - 2.5) < 0.1


def test_isolate_unit_removes_disconnected_neighbour():
    rng = np.random.default_rng(1)
    unit = _floor_ceiling_walls(rng, z_floor=0.0, z_ceiling=2.5, footprint=4.0)
    # neighbour blob at SAME height but 10 m away across a gap
    neigh = np.column_stack([
        rng.uniform(14, 16, 3000),
        rng.uniform(0, 2, 3000),
        rng.uniform(0, 2.5, 3000),
    ])
    xyz = np.vstack([unit, neigh])
    scan = ScanData(xyz=xyz)
    # trajectory walks inside the unit
    traj = np.column_stack([np.linspace(0.5, 3.5, 20), np.full(20, 2.0), np.full(20, 1.2)])
    out, stats = isolate_unit(scan, traj, z_band=(0.0, 2.5), cell_m=0.25, max_dist_m=8.0)
    # neighbour (x in 14..16) must be gone
    assert out.xyz[:, 0].max() < 6.0
    assert stats["dropped"] >= 2500
    assert stats["kept"] > 0


def test_isolate_unit_keeps_connected_balcony_step():
    rng = np.random.default_rng(2)
    unit = _floor_ceiling_walls(rng, footprint=4.0)
    # balcony: a connected patch just outside one wall (adjacent, walked)
    balc = np.column_stack([rng.uniform(4.0, 5.2, 1500), rng.uniform(1, 3, 1500), rng.uniform(0, 2.5, 1500)])
    xyz = np.vstack([unit, balc])
    scan = ScanData(xyz=xyz)
    traj = np.column_stack([np.linspace(0.5, 4.8, 25), np.full(25, 2.0), np.full(25, 1.2)])
    out, _ = isolate_unit(scan, traj, z_band=(0.0, 2.5), cell_m=0.25, max_dist_m=8.0)
    # balcony points (x > 4.0) survive because connected + within distance
    assert (out.xyz[:, 0] > 4.0).sum() > 0
