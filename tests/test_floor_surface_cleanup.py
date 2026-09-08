import sys
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from shapely.geometry import Polygon

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/export'))
from floor_surface_cleanup import consolidate_floor_surfaces
from filter_soulace_overlap import mesh_from_polygon
from audit_floor_artifacts import audit_floors


def panel(name,z=0,hole=False):
    holes=[[(.49,.49),(.51,.49),(.51,.51),(.49,.51)]] if hole else []
    poly=Polygon([(0,0),(1,0),(1,1),(0,1)],holes)
    mesh=mesh_from_polygon(poly,np.array([0,0,z]),np.array([1,0,0]),np.array([0,1,0]))
    return {'name':name,'kind':'floor','level':0,'v':mesh.vertices.tolist(),'f':mesh.faces.tolist()}


def test_duplicate_floor_faces_removed_without_datum_changes():
    parts,report=consolidate_floor_surfaces([panel('a'),panel('b'),panel('lower',-.5)])
    mesh=trimesh.Trimesh(parts[0]['v'],parts[0]['f'],process=False)
    assert np.isclose(mesh.area,2)
    assert set(np.round(mesh.vertices[:,2],6))=={0,-.5}
    assert np.isclose(sum(p['duplicate_area_removed_m2'] for p in report[0]['planes']),1)
    assert audit_floors(parts)['floors'][0]['degenerate_triangles']==0


def test_hole_not_filled_without_evidence():
    parts,_=consolidate_floor_surfaces([panel('hole',hole=True)])
    assert np.isclose(trimesh.Trimesh(parts[0]['v'],parts[0]['f'],process=False).area,.9996)


def test_small_hole_filled_only_with_whole_cell_raw_bound():
    x,y=np.meshgrid(np.arange(0,1.001,.01),np.arange(0,1.001,.01))
    points=np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size)))
    parts,report=consolidate_floor_surfaces([panel('hole',hole=True)],cKDTree(points))
    assert np.isclose(trimesh.Trimesh(parts[0]['v'],parts[0]['f'],process=False).area,1)
    assert sum(p['pores_filled'] for p in report[0]['planes'])==1
    far=points+[0,0,.2]
    parts,report=consolidate_floor_surfaces([panel('hole',hole=True)],cKDTree(far))
    assert np.isclose(trimesh.Trimesh(parts[0]['v'],parts[0]['f'],process=False).area,.9996)
    assert sum(p['pores_filled'] for p in report[0]['planes'])==0
