import sys
from pathlib import Path
import numpy as np
from shapely.geometry import Point
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/export'))
from close_soulace_top_floor import continuous_profile,extend_junctions,height_steps
from model_soulace_top_walls import solid_from_profile


def test_sparse_pores_do_not_make_open_walls():
    u,z=np.meshgrid(np.arange(.01,3,.02),np.arange(.01,2.5,.02));uv=np.c_[u.ravel(),z.ravel()]
    uv=uv[np.arange(len(uv))%17!=0];p=np.c_[np.zeros(len(uv)),uv]
    poly,holes=continuous_profile(p,0,0,3,[[0,3,2.5]])
    assert not holes
    assert abs(poly.area-7.5)<1e-8


def test_clear_window_is_not_filled():
    u,z=np.meshgrid(np.arange(.01,3,.02),np.arange(.01,2.5,.02));uv=np.c_[u.ravel(),z.ravel()]
    uv=uv[~((uv[:,0]>.9)&(uv[:,0]<1.9)&(uv[:,1]>.8)&(uv[:,1]<1.8))]
    p=np.c_[np.zeros(len(uv)),uv];poly,holes=continuous_profile(p,0,0,3,[[0,3,2.5]])
    assert len(holes)==1 and not poly.contains(Point(1.4,1.3))


def test_corner_join_reaches_perpendicular_wall_thickness():
    runs=[{'name':'x','axis':0,'cross':[0,.2],'along':[0,2.9]},
          {'name':'y','axis':1,'cross':[3,3.2],'along':[0,3]}]
    changes=extend_junctions(runs)
    assert changes and runs[0]['along'][1]==3.2


def test_stair_side_section_closes_without_filling_space_below():
    from shapely.geometry import Polygon
    p=Polygon([(0,0),(0,.2),(-.25,.2),(-.25,.4),(-.5,.4),(-.5,.2),(-.35,.2),(-.35,0)])
    m=solid_from_profile(p,1,-.3,.3)
    assert m.is_watertight and abs(abs(m.volume)-p.area*.6)<1e-9
