"""Rectangle completion fills only where the scan shows nothing, keeps openings
and recesses open, and reaches the level datums."""
import sys
from pathlib import Path

import numpy as np
import shapely
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/export'))
from complete_exterior_rectangles import complete_plane, level_datums


def plane_x(poly):
    """A vertical plane at x = 0 with u along +z and v along -y, as planes_of yields for n = +x."""
    return {'normal': np.array([1., 0, 0]), 'offset': 0., 'origin': np.zeros(3), 'u': np.array([0, 0, 1.]),
            'v': np.array([0, -1., 0]), 'poly': poly, 'key': (1., 0., 0., 0.)}


def test_fills_unseen_area_to_datums_but_not_openings_or_recesses():
    # Measured plane: u (height) 0.4..2.6, v (-y) 0..6, with a window hole and a
    # missing bottom band; a recess behind v in [4, 5] shows returns 150 mm off.
    window = shapely.box(1, 2, 2, 3)
    recess = shapely.box(0, 4, 3, 5)
    measured = shapely.box(.4, 0, 2.6, 6).difference(window).difference(recess)
    returns = []
    for v in np.arange(4.02, 5, .02):
        for u in np.arange(.02, 3, .02):
            returns.append([.15, -v, u])           # recess surface 150 mm behind, spans the full height
    tree = cKDTree(np.array(returns))
    completed, row = complete_plane(plane_x(measured), 0, {0: 0., 1: 3.}, tree, None)
    assert completed.bounds[0] < .03 and completed.bounds[2] > 2.97          # reaches floor 0 and slab 3
    assert completed.intersection(window).area < 1e-6                        # window stays open
    assert completed.intersection(shapely.box(0, 4.05, 3, 4.95)).area < .05  # recess stays open
    assert row['inferred_fill_m2'] > 2.                                       # unseen bottom and top bands filled


def test_level_datums_are_area_weighted_floor_heights():
    parts = [{'kind': 'floor_wall_joined', 'level': 1, 'v': [[0, 0, 3.2], [4, 0, 3.2], [4, 4, 3.2], [0, 4, 3.2]], 'f': [[0, 1, 2], [0, 2, 3]]},
             {'kind': 'floor_junction_patch', 'level': 1, 'v': [[0, 0, 9], [1, 0, 9], [1, 1, 9]], 'f': [[0, 1, 2]]}]
    assert abs(level_datums(parts)[1] - 3.2) < 1e-9
