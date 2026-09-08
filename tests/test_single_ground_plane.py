import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/export'))
from make_soulace_single_ground import single_ground_rectangle


def test_single_level_and_complete_xy_envelope():
    p=np.array([[-10,-5,.10],[7,-2,-.60],[5,7,0],[-6,4,-.30]])
    result=single_ground_rectangle(p,-.5334)
    assert result.shape==(4,3)
    assert np.all(result[:,2]==-.5334)
    assert np.array_equal(result[:,:2].min(0),p[:,:2].min(0))
    assert np.array_equal(result[:,:2].max(0),p[:,:2].max(0))
    assert np.cross(result[1]-result[0],result[2]-result[0])[2]>0


def test_single_plane_builder_does_not_mutate_source():
    p=np.array([[0,0,0],[4,0,-.5],[4,3,-.6],[0,3,.1]]);before=p.copy()
    single_ground_rectangle(p,-.5334)
    assert np.array_equal(p,before)


@pytest.mark.parametrize('p',[
    [[0,0,0],[0,1,0],[0,2,0]],
    [[0,0,0],[1,0,0],[float('nan'),1,0]],
])
def test_invalid_plane_envelope_is_rejected(p):
    with pytest.raises(ValueError):single_ground_rectangle(p,-.5)
