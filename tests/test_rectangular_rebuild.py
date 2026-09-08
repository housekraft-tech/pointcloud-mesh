import sys
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import box
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/export'))
from rectangular_rebuild import (Settings, rigid, extract_patches, grid,
                                 largest_rectangle, rectangle_partition, robust_position)


def test_transform_rejects_scale_and_reflection():
    for v in [-1., 2.]:
        m = np.eye(4); m[0,0] = v
        with pytest.raises(ValueError):
            rigid(m)


def test_robust_face_fit_does_not_average_outliers_or_opposite_face():
    cfg = Settings()
    rng = np.random.default_rng(3)
    points = np.r_[rng.normal(.009,.002,500), rng.uniform(.025,.045,100), rng.normal(.15,.002,500)]
    fitted = robust_position(points,0,cfg)
    assert abs(fitted['position_m']-.009) < .001


def test_sparse_plane_is_not_a_wall():
    assert robust_position([0]*4,0,Settings()) is None


def test_uniform_rectangle_remains_one_rectangle():
    xs,ys,xy,domain,_ = grid(box(0,0,3,2),.05)
    r = rectangle_partition(domain,domain,xs,ys,Settings())
    assert len(r) == 1
    assert np.allclose(r[0]['bounds'],[0,0,3,2])


def test_tiny_gap_is_continuity_not_measured():
    xs,ys,xy,domain,_ = grid(box(0,0,3,2),.05)
    support = domain.copy(); support[15,15] = False
    r = rectangle_partition(support,domain,xs,ys,Settings())
    assert len(r) == 1
    assert r[0]['grid_unknown_area_m2'] > 0
    assert r[0]['grid_supported_fraction'] < 1


def test_door_hole_is_never_closed_by_continuity():
    shape = box(0,0,4,3).difference(box(1,0,2,2))
    xs,ys,xy,domain,_ = grid(shape,.05)
    r = rectangle_partition(domain,domain,xs,ys,Settings())
    assert len(r) > 1
    assert all(box(*q['bounds']).difference(shape).area < 1e-9 for q in r)
    assert abs(sum(box(*q['bounds']).area for q in r)-shape.area) < 1e-6


def test_large_missing_band_is_not_bridged():
    xs,ys,xy,domain,_ = grid(box(0,0,4,3),.05)
    support = domain & ((xy[:,:,0] < 1.5) | (xy[:,:,0] > 2.5))
    r = rectangle_partition(support,domain,xs,ys,Settings())
    gap = box(1.6,0,2.4,3)
    assert len(r) >= 2
    assert all(box(*q['bounds']).intersection(gap).area < 1e-8 for q in r)


def test_l_shaped_floor_not_its_bounding_box():
    shape = box(0,0,3,3).difference(box(1,1,3,3))
    xs,ys,xy,domain,_ = grid(shape,.05)
    rectangles = rectangle_partition(domain,domain,xs,ys,Settings())
    assert abs(sum(box(*r['bounds']).area for r in rectangles)-5) < 1e-6


def test_variable_grid_uses_physical_area_not_cell_count():
    mask = np.array([[True,False],[True,True]])
    area,y0,y1,x0,x1 = largest_rectangle(mask,np.array([0,.1,2]),np.array([0,.1,.3]))
    assert abs(area-.4) < 1e-10
    assert (y0,y1,x0,x1) == (1,2,0,2)


def test_patch_extraction_is_project_name_independent():
    mesh = trimesh.creation.box([3,.2,2])
    for name in ['alpha','unseen_building']:
        p, excluded = extract_patches({'parts':[{'name':name,'kind':'wall','v':mesh.vertices,'f':mesh.faces}]},Settings())
        assert len(p) == 6 and not excluded
        assert abs(sum(q['polygon'].area for q in p)-mesh.area) < 1e-8


def test_oblique_geometry_is_flagged_not_forced_rectangular():
    mesh = trimesh.creation.box([3,.2,2])
    mesh.apply_transform(trimesh.transformations.rotation_matrix(.2,[0,0,1]))
    p, excluded = extract_patches({'parts':[{'name':'rotated','kind':'wall','v':mesh.vertices,'f':mesh.faces}]},Settings())
    assert excluded
    assert len(p) == 2


def test_strict_mode_never_fills_missing_cells():
    xs,ys,xy,domain,_ = grid(box(0,0,3,2),.025)
    support=domain.copy(); support[25:28,35:38]=False
    rectangles=rectangle_partition(support,domain,xs,ys,Settings(allow_continuity=False,grid_m=.025))
    gap=box(xs[35],ys[25],xs[38],ys[28])
    assert all(box(*r['bounds']).intersection(gap).area < 1e-10 for r in rectangles)
    assert all(r['grid_unknown_area_m2'] < 1e-10 for r in rectangles)
