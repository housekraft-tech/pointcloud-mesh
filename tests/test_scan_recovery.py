import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/export'))
from recover_mesh_evidence import bounded_triangles
from recover_soulace_details import ground_mesh


def test_fully_supported_children_coalesce_without_shape_change():
    x,y=np.meshgrid(np.arange(-.01,.62,.005),np.arange(-.01,.62,.005))
    points=np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size)))
    triangle=np.array([[[0,0,0],[.6,0,0],[0,.6,0]]])
    result,report=bounded_triangles(triangle,cKDTree(points))
    assert result.shape==(1,3,3)
    assert np.allclose(result,triangle)
    assert report['max_bound_m']<.05


def test_unsupported_hole_is_not_filled_by_parent_coalescing():
    x,y=np.meshgrid(np.arange(-.01,.82,.005),np.arange(-.01,.82,.005))
    keep=~((x>.13)&(x<.43)&(y>.13)&(y<.43))
    points=np.column_stack((x[keep],y[keep],np.zeros(keep.sum())))
    tree=cKDTree(points);triangle=np.array([[[0,0,0],[.8,0,0],[0,.8,0]]])
    result,report=bounded_triangles(triangle,tree)
    assert len(result)>1
    rng=np.random.default_rng(9210)
    w=rng.dirichlet([1,1,1],size=(len(result),40))
    q=np.einsum('nij,njk->nik',w,result).reshape(-1,3)
    assert tree.query(q)[0].max()<.05
    area=np.linalg.norm(np.cross(result[:,1]-result[:,0],result[:,2]-result[:,0]),axis=1).sum()/2
    assert area<.8*.8/2-.02


def test_floor_keeps_level_break_and_local_slope():
    x,y=np.meshgrid(np.arange(0,1,.02),np.arange(0,1,.02))
    z=np.where(x<.5,0,-.54+.02*y)
    points=np.column_stack((x.ravel(),y.ravel(),z.ravel()))
    tri=ground_mesh(points)
    assert len(tri)>0
    assert np.max(np.ptp(tri[:,:,2],axis=1))<.04
    assert np.any(tri.mean(axis=1)[:,2]<-.5)
    assert np.any(abs(tri.mean(axis=1)[:,2])<.01)
    assert np.ptp(tri[tri.mean(axis=1)[:,2]<-.5,:,2])>.01
