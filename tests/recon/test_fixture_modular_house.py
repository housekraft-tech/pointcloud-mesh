import numpy as np

from tests.fixtures import modular_house


def test_modular_house_has_all_features():
    pts, gps, meta = modular_house()
    assert pts.shape[0] == gps.shape[0]
    assert pts.shape[0] > 0

    # floor and ceiling present
    z = pts[:, 2]
    assert (np.abs(z - 0.0) < 0.05).sum() > 100
    assert (np.abs(z - 2.7) < 0.05).sum() > 100

    # pillar front face near x=0.4, y in [2.0,2.3]
    pillar = (np.abs(pts[:, 0] - 0.4) < 0.03) & (pts[:, 1] > 1.9) & (pts[:, 1] < 2.4)
    assert pillar.sum() > 50

    # beam bottom near z=2.4, y in [2.4,2.6]
    beam = (np.abs(z - 2.4) < 0.03) & (pts[:, 1] > 2.35) & (pts[:, 1] < 2.65)
    assert beam.sum() > 50

    # neighbour blob is far in +x and disconnected from the room (x<=6.2)
    neigh = pts[:, 0] > 15.0
    assert neigh.sum() > 1000
    room = pts[pts[:, 0] < 10.0]
    assert room[:, 0].max() < 6.3  # gap between room (<=6.1) and neighbour (>=16)


def test_modular_house_meta_consistent():
    _, _, meta = modular_house()
    assert meta["z_ceiling_m"] == 2.7
    assert len(meta["exterior_walls"]) == 4
    assert {o["type"] for o in meta["openings"]} == {"door", "window", "balcony_door"}
