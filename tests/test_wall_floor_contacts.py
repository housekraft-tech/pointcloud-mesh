import sys
from pathlib import Path
import numpy as np
from shapely.geometry import box,Polygon,Point
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/export'))
from build_soulace_wall_floor_planes import close_small_contacts,top_polygon,physical_wall_bases


def test_small_inner_side_gap_closes_at_fixed_wall_face():
    wall=box(-.2,0,0,2);floor=box(.021,0,2,2)
    joined=close_small_contacts(floor,wall,Polygon())
    assert joined.covers(Point(0,1))
    assert joined.intersection(wall).area<1e-12


def test_both_sides_of_partition_can_contact_without_moving_wall():
    wall=box(0,0,.2,2);floor=box(-2,0,-.02,2).union(box(.22,0,2,2))
    joined=close_small_contacts(floor,wall,Polygon())
    assert joined.covers(Point(0,1)) and joined.covers(Point(.2,1))
    assert joined.intersection(wall).area<1e-12


def test_stair_void_is_never_filled_by_contact_closure():
    wall=box(-.2,0,0,2);floor=box(.021,0,2,2);void=box(0,1,1,2)
    joined=close_small_contacts(floor,wall,void)
    assert joined.intersection(void).area==0
    assert joined.covers(Point(0,.5))


def test_large_gap_is_not_bridged_by_local_contact_rule():
    wall=box(-.2,0,0,2);floor=box(.25,0,2,2)
    joined=close_small_contacts(floor,wall,Polygon())
    assert joined.symmetric_difference(floor).area==0


def test_floor_footprint_uses_top_not_a_projected_underside():
    mesh=trimesh.creation.box(extents=[2,3,.1]);mesh.apply_translation([1,1.5,-.05])
    poly=top_polygon({'v':np.asarray(mesh.vertices).tolist(),'f':np.asarray(mesh.faces).tolist()})
    assert abs(poly.area-6)<1e-10
    assert poly.symmetric_difference(box(0,0,2,3)).area<1e-10


def test_door_opening_is_not_subtracted_from_floor_as_wall():
    profile=box(0,0,3,3).difference(box(1,0,2,2))
    bases,thresholds=physical_wall_bases([{'run':{'axis':0,'along':[0,3],'cross':[0,.2]},'polygon_wkt':profile.wkt}])
    assert len(thresholds)==1 and abs(thresholds[0].area-.2)<1e-12
    floors=box(-1,0,0,3).union(box(.2,0,1,3)).union(thresholds[0])
    joined=close_small_contacts(floors,bases,Polygon())
    assert joined.covers(Point(.1,1.5))
    assert joined.intersection(bases).area<1e-12
