import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import box

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/export'))
from lidar_support_baseline import ReferenceUnion
from filter_soulace_overlap import supported_polygon


def test_reference_union_does_not_average_different_scan_surfaces():
    refs=[{'tree':cKDTree([[0,0,.010]])},{'tree':cKDTree([[0,0,.040]])}]
    distances,_=ReferenceUnion(refs).query(np.array([[0,0,0]]))
    assert abs(distances[0]-.010)<1e-12


def test_narrow_supported_patch_is_not_rejected_by_rectangle_minimum_area():
    x=np.arange(0,.0401,.002);y=np.arange(0,1.0001,.002)
    xx,yy=np.meshgrid(x,y)
    points=np.column_stack((xx.ravel(),yy.ravel(),np.zeros(xx.size)))
    selected,_=supported_polygon(box(0,0,.04,1),np.zeros(3),np.array([1,0,0]),np.array([0,1,0]),cKDTree(points),cutoff=.05,min_area=.02)
    assert selected.area>.035
    assert selected.area<.080  # the rectangular version would reject this size


def test_oblique_supported_patch_is_retained_without_axis_alignment_gate():
    angle=np.deg2rad(17)
    u=np.array([np.cos(angle),0,np.sin(angle)]);v=np.array([0,1,0])
    x,y=np.meshgrid(np.arange(0,.301,.003),np.arange(0,.501,.003))
    points=x.ravel()[:,None]*u+y.ravel()[:,None]*v
    selected,_=supported_polygon(box(0,0,.3,.5),np.zeros(3),u,v,cKDTree(points),cutoff=.05)
    assert selected.area>.145
