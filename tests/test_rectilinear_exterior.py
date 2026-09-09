"""Every exterior ring comes out axis-aligned: snapped when close, boxed when not."""
import sys
from pathlib import Path

import numpy as np
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/export'))
from rectilinear_exterior import rectilinear


def axis_aligned(poly):
    for ring in [poly.exterior, *poly.interiors]:
        xy = np.asarray(ring.coords)
        d = np.diff(xy, axis=0)
        if not np.all((np.abs(d[:, 0]) < 1e-6) | (np.abs(d[:, 1]) < 1e-6)):
            return False
    return True


def test_jagged_wall_with_window_and_blob_becomes_rectangles_only():
    rng = np.random.default_rng(4)
    xs = np.linspace(0, 6, 61)
    outline = [(x, 3 + rng.uniform(-.08, .08)) for x in xs] + [(x, rng.uniform(-.08, .08)) for x in xs[::-1]]
    wall = shapely.Polygon(outline).buffer(0)
    window = shapely.box(1, 1, 2, 2.2)
    blob = shapely.Point(4, 1.5).buffer(.5)           # 0.8 m2 irregular recess void
    speck = shapely.Point(5, .5).buffer(.1)            # small hole: filled
    source = wall.difference(window).difference(blob).difference(speck)
    result, report = rectilinear(source)
    assert result.geom_type == 'Polygon'
    assert axis_aligned(result)
    holes = [shapely.Polygon(r) for r in result.interiors]
    assert len(holes) == 2                              # window and boxed recess, speck gone
    assert any(h.symmetric_difference(window).area < .05 for h in holes)
    assert any(h.symmetric_difference(shapely.box(*blob.bounds)).area < .05 for h in holes)


def test_a_curved_piece_comes_out_axis_aligned():
    curved = shapely.Point(0, 0).buffer(1.)
    result, report = rectilinear(curved)
    assert axis_aligned(result)
    assert report['boxed_rings'] + report['snapped_rings'] == 1
    assert curved.area * .9 < result.area <= 4. + 1e-6         # between the circle and its box


def test_tiny_fragments_are_dropped():
    source = shapely.union_all([shapely.box(0, 0, 3, 3), shapely.box(5, 5, 5.2, 5.2)])
    result, report = rectilinear(source)
    assert result.geom_type == 'Polygon' and report['dropped_m2'] > .03
