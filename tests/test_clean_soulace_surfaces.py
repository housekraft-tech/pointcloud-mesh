import sys
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon,box
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/export'))
from clean_soulace_surfaces import robust_plane,fill_small_holes,polygon_triangles,height_components,fit_riser


def test_plane_retains_drainage_fall_with_outliers():
    rng=np.random.default_rng(15);xy=rng.uniform(0,6,(3000,2));z=xy@np.array([-.013,.002])-.475+rng.normal(0,.003,len(xy));z[::12]+=.15
    a=robust_plane(np.column_stack((xy,z)))
    assert np.allclose(a,[-.013,.002,-.475],atol=.001)


def test_only_small_enclosed_pores_are_filled():
    small=box(.2,.2,.3,.3);large=box(1,1,2,2);p=Polygon(box(0,0,3,3).exterior,[small.exterior,large.exterior])
    result=fill_small_holes(p,.04)
    assert abs(result.area-8)<1e-9
    assert len(result.interiors)==1


def test_triangulation_is_coplanar_and_respects_large_opening():
    p=Polygon(box(0,0,3,3).exterior,[box(1,1,2,2).exterior]);a=np.array([-.013,.002,-.475]);t=polygon_triangles(p,a)
    assert np.max(abs(t[:,:,2]-(t[:,:,:2]@a[:2]+a[2])))<1e-12
    u=t[:,1,:2]-t[:,0,:2];v=t[:,2,:2]-t[:,0,:2];area=(u[:,0]*v[:,1]-u[:,1]*v[:,0]).sum()/2
    assert abs(area-8)<1e-9


def test_neighboring_parking_and_interior_never_merge():
    x,y=np.meshgrid(np.arange(0,2,.02),np.arange(0,1,.02));z=np.where(x<1,0,-.54)
    p=np.column_stack((x.ravel(),y.ravel(),z.ravel()));lab,*_=height_components(p,cell=.04)
    left=set(lab[p[:,0]<.96]);right=set(lab[p[:,0]>1.04])
    assert left.isdisjoint(right)


def test_riser_fits_observed_position_not_regular_template():
    rng=np.random.default_rng(19);n=2000
    p=np.column_stack((rng.uniform(-4.7,-4.0,n),rng.normal(2.315,.003,n),rng.uniform(3.56,3.73,n)))
    y,qa=fit_riser(p,(-4.8,-3.9),2.3,3.56,3.73)
    assert abs(y-2.315)<.001
    assert qa['points']>1000
