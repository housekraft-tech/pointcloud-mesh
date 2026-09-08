import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts/export"))
from filter_soulace_overlap import mask_rectangles, supported_polygon, mesh_from_polygon, plane_patches
import trimesh


def test_mask_union_has_no_added_hole_or_missing_cell():
    mask = np.ones((4, 5), bool)
    mask[1:3, 2] = False
    poly = mask_rectangles(mask, 0, 0, .1)
    assert abs(poly.area - .18) < 1e-10
    assert not poly.contains(box(.22,.12,.28,.18))


def test_far_plane_is_empty():
    tree = cKDTree([[0, 0, 1]])
    p, info = supported_polygon(box(0,0,.1,.1), np.zeros(3), np.array([1,0,0]),
                               np.array([0,1,0]), tree, min_area=0)
    assert p.is_empty


def test_retained_polygon_every_sample_is_within_cutoff():
    xy = np.arange(0,.121,.002)
    x,y = np.meshgrid(xy,xy)
    tree = cKDTree(np.column_stack((x.ravel(),y.ravel(),np.full(x.size,.004))))
    p, info = supported_polygon(box(0,0,.12,.12),np.zeros(3),np.array([1,0,0]),
                               np.array([0,1,0]),tree,min_area=.0001)
    mesh = mesh_from_polygon(p,np.zeros(3),np.array([1,0,0]),np.array([0,1,0]))
    assert mesh is not None
    assert mesh.area <= .12**2+1e-10
    assert mesh.area > .01
    assert tree.query(mesh.sample(5000))[0].max() <= .01
    assert info['distance_upper_bound_mm'] <= 10


def test_box_plane_groups_preserve_area():
    mesh = trimesh.creation.box([1,2,3])
    patches = list(plane_patches(mesh))
    assert len(patches) == 6
    assert abs(sum(p[0].area for p in patches)-mesh.area) < 1e-8


def test_50mm_envelope_keeps_dense_plane_at_45mm_offset():
    xy = np.arange(0,.121,.002)
    x,y = np.meshgrid(xy,xy)
    tree = cKDTree(np.column_stack((x.ravel(),y.ravel(),np.full(x.size,.045))))
    p, info = supported_polygon(box(0,0,.12,.12),np.zeros(3),np.array([1,0,0]),
                               np.array([0,1,0]),tree,cutoff=.05,min_area=.0001)
    assert p.area > .013
    assert info['distance_upper_bound_mm'] < 50


def test_no_bridge_between_separate_scan_patches():
    x = np.r_[np.arange(0,.041,.002),np.arange(.20,.241,.002)]
    y = np.arange(0,.121,.002)
    xx,yy = np.meshgrid(x,y)
    tree = cKDTree(np.column_stack((xx.ravel(),yy.ravel(),np.zeros(xx.size))))
    p,_ = supported_polygon(box(0,0,.24,.12),np.zeros(3),np.array([1,0,0]),
                            np.array([0,1,0]),tree,cutoff=.05,min_area=.0001)
    assert p.intersection(box(.11,.02,.13,.10)).area == 0
