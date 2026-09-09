"""The exterior test is physical: the air in front of an exterior face reaches
open sky in the raw returns. Synthetic returns build one covered room next to
an open terrace; no property name, floor footprint or wall list is involved."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/export'))
from select_exterior_walls import Occupancy, select


def surface(x0, x1, y0, y1, z0, z1, pitch=.02):
    """Dense returns on an axis-aligned rectangle (one of the ranges is degenerate)."""
    xs = np.arange(x0, x1 + 1e-9, pitch) if x1 > x0 else np.array([x0])
    ys = np.arange(y0, y1 + 1e-9, pitch) if y1 > y0 else np.array([y0])
    zs = np.arange(z0, z1 + 1e-9, pitch) if z1 > z0 else np.array([z0])
    return np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), -1).reshape(-1, 3)


def wall_part(name, x, y0, y1, z0=0., z1=3.):
    return {'name': name, 'kind': 'wall', 'level': 0,
            'v': [[x, y0, z0], [x, y1, z0], [x, y1, z1], [x, y0, z1]], 'f': [[0, 1, 2], [0, 2, 3]]}


def scene():
    """Room x in [0,4], y in [0,4], ceiling at z=3; terrace x in [4,8] open to the sky
    with a 1 m parapet at x=8; a compound wall 3 m out at x=-3 blocks horizontal rays."""
    returns = [
        surface(0, 4, 0, 4, 3, 3),          # room ceiling
        surface(-3, 8, -2, 6, 0, 0),        # ground, room and terrace floor
        surface(0, 0, 0, 4, 0, 3),          # west wall, inner face
        surface(-.23, -.23, 0, 4, 0, 3),    # west wall, outer face towards the compound
        surface(4, 4, 0, 4, 0, 3),          # wall between room and terrace
        surface(0, 4, 0, 0, 0, 3), surface(0, 4, 4, 4, 0, 3),   # north/south room walls
        surface(8, 8, 0, 4, 0, 1),          # parapet
        surface(-3, -3, -2, 6, 0, 2.5),     # compound wall
    ]
    points = np.vstack(returns)
    # Sparse floating noise must not close the sky: well under 3 returns per 50 mm voxel.
    rng = np.random.default_rng(3)
    noise = rng.uniform([-4, -2, 0], [9, 6, 11], (400, 3))
    return np.vstack([points, noise])


def test_faces_are_classified_by_open_air_not_by_floor_footprint():
    occupancy = Occupancy(scene())
    parts = [wall_part('west outer face', -.23, 0, 4), wall_part('west inner face', 0, 0, 4),
             wall_part('terrace wall', 4, 0, 4), wall_part('room partition', 2, 0, 4),
             wall_part('parapet', 8, 0, 4, 0, 1)]
    selection, audit = select(parts, occupancy)
    classes = {r['wall']: r['classification'] for r in audit}
    assert classes['west outer face'] == 'exterior_face'      # compound wall blocks the ray, sky is open
    assert classes['terrace wall'] == 'exterior_face'         # no ceiling over the terrace
    assert classes['parapet'] == 'exterior_face'
    assert classes['west inner face'] == 'interior_face'      # room side covered, other side is masonry
    assert classes['room partition'] == 'interior_face'
    assert set(selection['walls']) == {'west outer face', 'terrace wall', 'parapet'}


def test_selection_is_per_plane_so_inner_faces_of_one_part_stay_untouched():
    occupancy = Occupancy(scene())
    part = {'name': 'west wall both faces', 'kind': 'wall_measured', 'level': 0,
            'v': [[-.23, 0, 0], [-.23, 4, 0], [-.23, 4, 3], [-.23, 0, 3],
                  [0, 0, 0], [0, 4, 0], [0, 4, 3], [0, 0, 3]],
            'f': [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]]}
    selection, audit = select([part], occupancy)
    planes = selection['walls']['west wall both faces']
    assert len(planes) == 1
    assert abs(abs(planes[0]['key'][3]) - .23) < 1e-6
    assert planes[0]['mask_uv'] is None
    assert sorted(r['classification'] for r in audit) == ['exterior_face', 'interior_face']


def test_wall_leaving_the_house_is_partly_exterior_with_a_mask_over_the_open_part():
    # A covered annex south of the room makes the y=0 wall an internal partition
    # for x in [0, 4]; the wall continues 4 m east as a garden wall between the
    # terrace and the compound, where both sides look at open sky.
    annex = [surface(0, 4, -4, 0, 3, 3), surface(0, 0, -4, 0, 0, 3), surface(4, 4, -4, 0, 0, 3),
             surface(0, 4, -4, -4, 0, 3)]
    points = np.vstack([scene(), *annex, surface(4, 8, 0, 0, 0, 3)])
    occupancy = Occupancy(points)
    part = wall_part('south wall and garden wall', 0, 0, 8)
    part['v'] = [[0, 0, 0], [8, 0, 0], [8, 0, 3], [0, 0, 3]]
    selection, audit = select([part], occupancy)
    row = audit[0]
    assert row['classification'] == 'partly_exterior_face'
    assert .3 < row['open_air_fraction'] < .7
    mask = selection['walls'][part['name']][0]['mask_uv']
    assert mask is not None
    import shapely
    region = shapely.union_all([shapely.Polygon(r) for r in mask])
    # Plane u axis runs along the wall; the open part is the x in [4, 8] half.
    bounds = region.bounds
    assert bounds[2] - bounds[0] < 5.       # mask does not span the whole 8 m wall
    assert row['open_area_m2'] > 8.         # at least most of the 4 m x 3 m garden section
