import sys
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon,box
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/export'))
from model_soulace_top_walls import solid_from_profile,clean_outline,pair_choice


def test_solid_has_measured_thickness_and_preserves_window():
    p=Polygon(box(0,0,4,3).exterior,[box(1,1,2,2).exterior])
    for axis in (0,1):
        m=solid_from_profile(p,axis,2.1,2.31)
        assert m.is_watertight
        assert abs(abs(m.volume)-11*.21)<1e-8
        assert np.allclose(m.bounds[:,axis],[2.1,2.31])


def test_no_second_face_means_no_invented_thickness():
    a=np.ones((30,60),bool)
    modes=[{'d':1.0,'p95_plane_mm':8.0}]
    assert pair_choice(modes,[a]) is None


def test_parallel_but_disjoint_surfaces_do_not_make_a_wall():
    a=np.zeros((100,100),bool);b=a.copy();a[:35,:]=True;b[65:,:]=True
    modes=[{'d':1.0,'p95_plane_mm':8.0},{'d':1.2,'p95_plane_mm':8.0}]
    assert pair_choice(modes,[a,b]) is None


def test_two_matching_faces_can_form_a_wall():
    a=np.ones((50,70),bool)
    modes=[{'d':1.0,'p95_plane_mm':8.0},{'d':1.2,'p95_plane_mm':8.0}]
    assert set(pair_choice(modes,[a,a]))=={0,1}


def test_large_opening_survives_small_pore_cleanup():
    a=np.ones((120,120),bool);a[40:80,40:80]=False;a[10,10]=False
    p=clean_outline(a,np.array([0.,0.]))
    assert len(p.interiors)==1
    assert abs(Polygon(p.interiors[0]).area-1)<.01
