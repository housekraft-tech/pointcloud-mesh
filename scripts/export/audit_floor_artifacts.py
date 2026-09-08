"""Inspect floor planes for numerical defects and possible sampling artifacts.

Holes/islands are review items, not automatically defects: true voids must not
be filled merely to make a cleaner picture.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import shapely
from shapely.geometry import Polygon
import trimesh
from architectural_surface_refinement import _all_planar_patches, _polygons


def audit_floors(parts):
    rows=[];by_level_datum={}
    for part in parts:
        if 'floor' not in part.get('kind','') or part.get('reference_only'):
            continue
        mesh=trimesh.Trimesh(part['v'],part['f'],process=False)
        horizontal=np.abs(mesh.face_normals[:,2])>1-1e-9
        top=[]
        for z in np.unique(np.round(mesh.triangles_center[horizontal,2],7)):
            ids=horizontal & (np.abs(mesh.triangles_center[:,2]-z)<1e-7)
            triangles=mesh.triangles[ids]
            region=shapely.union_all(shapely.polygons(triangles[:,:,:2]))
            islands=_polygons(region)
            holes=[Polygon(h).area for p in islands for h in p.interiors]
            key=(part.get('level',0),float(z))
            by_level_datum.setdefault(key,[]).append((part['name'],region))
            top.append({'datum_m':float(z),'area_m2':float(region.area),
                        'max_planarity_residual_mm':float(np.abs(triangles[:,:,2]-z).max()*1000),
                        'duplicate_coplanar_area_m2':float(mesh.area_faces[ids].sum()-region.area),
                        'islands':len(islands),'islands_under_50cm2':sum(p.area<.005 for p in islands),
                        'holes':len(holes),'holes_under_50cm2':sum(h<.005 for h in holes),
                        'small_hole_area_m2':sum(h for h in holes if h<.005),
                        'largest_hole_m2':max(holes,default=0)})
        rows.append({'name':part['name'],'triangles':len(mesh.faces),
                     'degenerate_triangles':int((mesh.area_faces<1e-12).sum()),
                     'horizontal_planes':top,
                     'nonhorizontal_area_m2':float(mesh.area_faces[~horizontal].sum())})
    overlap=[]
    for (level,z),regions in by_level_datum.items():
        area=sum(r.area for _,r in regions)
        union=shapely.union_all([r for _,r in regions])
        overlap.append({'level':level,'datum_m':z,'group_count':len(regions),
                        'coplanar_cross_group_overlap_m2':max(0,float(area-union.area))})
    return {'floors':rows,'coplanar_overlap':overlap,
            'interpretation':'Small holes and islands require evidence review. Horizontal plane flatness is a numerical check, not a measurement of site accuracy.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args()
    result=audit_floors(json.loads(Path(args.model).read_text())['parts'])
    Path(args.out).write_text(json.dumps(result,indent=2))
    print(json.dumps({'floor_groups':len(result['floors']),
        'degenerate_triangles':sum(r['degenerate_triangles'] for r in result['floors']),
        'small_holes':sum(p['holes_under_50cm2'] for r in result['floors'] for p in r['horizontal_planes']),
        'cross_group_overlap_m2':sum(r['coplanar_cross_group_overlap_m2'] for r in result['coplanar_overlap'])},indent=2))

if __name__=='__main__':main()
