import sys
from pathlib import Path

import numpy as np
import shapely
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/export'))
from floor_wall_junctions import extension_strips, rebuild_floor


def wall(x):
    return {'v': [[x, 0, 0], [x, 1, 0], [x, 1, 3], [x, 0, 3]],
            'f': [[0, 1, 2], [0, 2, 3]]}


def test_strip_ends_at_wall_and_preserves_floor_boundary():
    floor = shapely.box(0, 0, 1, 1)
    strip = extension_strips(wall(1.01), floor, .015)
    assert np.isclose(strip.area, .01)
    assert strip.bounds[0] >= 1 - 1e-10
    assert strip.bounds[2] <= 1.01 + 1e-10
    assert strip.intersection(floor).area == 0


def test_tiny_gap_pass_does_not_bridge_large_opening():
    strip = extension_strips(wall(1.1), shapely.box(0, 0, 1, 1), .015)
    assert strip is None or strip.is_empty


def test_rebuilding_one_datum_preserves_other_floor_height():
    # Two separate horizontal floor surfaces in a single group.
    vertices = [[x, y, z] for z in (0, 3) for x, y in ((0, 0), (1, 0), (1, 1), (0, 1))]
    part = {'v': vertices, 'f': [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]]}
    result = rebuild_floor(part, {0.: shapely.box(0, 0, 1.01, 1)})
    assert set(np.round(result.vertices[:, 2], 6)) == {0, 3}
    assert result.area > 2
