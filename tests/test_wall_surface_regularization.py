import sys
from pathlib import Path

import numpy as np
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/export'))
from regularize_wall_surfaces import regularize, regularize_within


def test_door_window_and_small_rectangular_vent_stay_open():
    wall = shapely.box(0, 0, 6, 3)
    openings = [shapely.box(.3, 0, 1.2, 2.1), shapely.box(2, 1, 3, 2), shapely.box(4, 1, 4.1, 1.1)]
    source = wall.difference(shapely.union_all(openings))
    result, _ = regularize(source)
    assert result.intersection(shapely.union_all(openings)).area < 1e-9


def test_narrow_crack_closed_without_new_wall_run():
    source = shapely.box(0, 0, 3, 3).difference(shapely.box(1.5, 0, 1.52, 3))
    result, _ = regularize(source)
    assert result.area > source.area + .05
    assert result.difference(source.buffer(.05, join_style=2)).area < 1e-9


def test_separate_wall_runs_do_not_bridge():
    source = shapely.union_all([shapely.box(0, 0, 2, 3), shapely.box(3, 0, 5, 3)])
    result, _ = regularize(source)
    assert result.intersection(shapely.box(2.01, 0, 2.99, 3)).area == 0


def test_blobby_scan_shadow_is_filled_but_a_window_of_the_same_size_stays_open():
    wall = shapely.box(0, 0, 8, 3)
    shadow = shapely.Point(2, 1.5).buffer(.35)                  # 0.38 m2, round: a parked car's shadow
    window = shapely.box(5, 1, 5.7, 1.7)                        # 0.49 m2, rectangular
    source = wall.difference(shapely.union_all([shadow, window]))
    result, row = regularize(source)
    assert result.intersection(shadow).area > .95 * shadow.area
    assert result.intersection(window).area < 1e-9
    assert row['filled_gap_area_m2'] > .3
    assert row['protected_openings'] == 1


def test_large_irregular_void_is_not_invented():
    wall = shapely.box(0, 0, 8, 3)
    void = shapely.Point(4, 1.5).buffer(.8)                     # 2 m2: too large to call a shadow
    result, _ = regularize(wall.difference(void))
    assert result.intersection(void.buffer(-.06)).area < 1e-9


def test_dropped_surface_is_recovered_only_where_the_returns_carry_it():
    # Three islands of one plane separated by 0.2 m and 1.0 m gaps; the returns
    # cover the first gap (a surface the cleanup dropped) but not the second.
    islands = [shapely.box(0, 0, 1, 3), shapely.box(1.2, 0, 2.2, 3), shapely.box(3.2, 0, 4.2, 3)]
    source = shapely.union_all(islands)
    supported = lambda uv: uv[:, 0] < 1.3          # returns exist up to x = 1.3 only
    result, row = regularize(source, support=supported)
    assert result.intersection(shapely.box(1.0, 0, 1.2, 3)).area > .9 * .6
    assert result.intersection(shapely.box(2.4, .1, 3.0, 2.9)).area < 1e-9
    assert .55 < row['recovered_measured_area_m2'] < .65
    untouched, _ = regularize(source)               # no evidence: gaps over the notch limit stay open
    assert untouched.intersection(shapely.box(2.4, .1, 3.0, 2.9)).area < 1e-9


def test_jagged_outline_snaps_to_a_rectangle_within_50mm():
    rng = np.random.default_rng(1)
    xs = np.linspace(0, 6, 121)
    top = [(x, 3 + rng.uniform(-.02, .02)) for x in xs]
    bottom = [(x, rng.uniform(-.02, .02)) for x in xs[::-1]]
    jagged = shapely.Polygon(top + bottom)
    result, row = regularize(jagged)
    assert len(result.exterior.coords) <= 6
    assert result.symmetric_difference(shapely.box(0, 0, 6, 3)).area < .05
    assert row['snapped_rings'] == 1 and row['unsnapped_rings'] == 0


def test_a_real_step_in_the_outline_survives_snapping():
    stepped = shapely.union_all([shapely.box(0, 0, 3, 3), shapely.box(3, 0, 6, 2.4)])
    result, _ = regularize(stepped)
    assert result.symmetric_difference(stepped).area < .02


def test_jagged_window_becomes_a_rectangle():
    wall = shapely.box(0, 0, 6, 3)
    rng = np.random.default_rng(2)
    ring = [(1 + rng.uniform(-.03, .03), y) for y in np.linspace(1, 2.2, 30)] + \
           [(x, 2.2 + rng.uniform(-.03, .03)) for x in np.linspace(1, 2.2, 30)] + \
           [(2.2 + rng.uniform(-.03, .03), y) for y in np.linspace(2.2, 1, 30)] + \
           [(x, 1 + rng.uniform(-.03, .03)) for x in np.linspace(2.2, 1, 30)]
    window = shapely.Polygon(ring).buffer(0)
    result, row = regularize(wall.difference(window))
    hole = shapely.Polygon(result.interiors[0])
    assert row['protected_openings'] == 1
    assert hole.symmetric_difference(shapely.box(1, 1, 2.2, 2.2)).area < .03
    assert len(hole.exterior.coords) <= 6


def test_partial_mask_limits_the_repair_to_the_open_part_of_the_plane():
    wall = shapely.box(0, 0, 8, 3)
    holes = [shapely.Point(1.5, 1.5).buffer(.2), shapely.Point(6.5, 1.5).buffer(.2)]
    source = wall.difference(shapely.union_all(holes))
    result, row = regularize_within(source, shapely.box(4, 0, 8, 3))
    assert row['accepted']
    assert result.intersection(holes[1]).area > .95 * holes[1].area   # inside the mask: filled
    assert result.intersection(holes[0]).area < 1e-9                  # outside the mask: untouched
