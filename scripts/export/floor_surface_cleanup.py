"""Exact floor-plane union and bounded small-pore repair, never a footprint box."""
from collections import defaultdict
from pathlib import Path
import argparse
import json
import numpy as np
import shapely
from shapely.geometry import Polygon
import trimesh
from scipy.spatial import cKDTree
from filter_soulace_overlap import supported_polygon, mesh_from_polygon, polygon_parts


def consolidate_floor_surfaces(parts, reference=None, pore_area=.005):
    floors=defaultdict(list);other=[];audit=[]
    for part in parts:
        if part.get('kind','').startswith('floor') and not part.get('reference_only'):
            floors[part.get('level',0)].append(part)
        else:other.append(part)
    for level,items in floors.items():
        by_plane=defaultdict(list)
        for part in items:
            mesh=trimesh.Trimesh(part['v'],part['f'],process=False)
            for triangle,normal,area in zip(mesh.triangles,mesh.face_normals,mesh.area_faces):
                if area<1e-12:continue
                normal=normal.copy()
                if normal[np.argmax(abs(normal))]<0:normal=-normal
                by_plane[tuple(np.round(normal,8))+(round(float(normal@triangle[0]),7),)].append(triangle)
        pieces=[];records=[]
        for key,triangles in by_plane.items():
            normal=np.asarray(key[:3]);normal/=np.linalg.norm(normal)
            triangles=np.asarray(triangles)
            origin=triangles[0,0]
            if np.max(abs((triangles-origin)@normal))>1e-7:
                raise ValueError('Cannot merge offset or noncoplanar floor faces')
            seed=np.eye(3)[np.argmin(abs(normal))]
            u=np.cross(normal,seed);u/=np.linalg.norm(u);v=np.cross(normal,u)
            xy=np.stack(((triangles-origin)@u,(triangles-origin)@v),axis=-1)
            region=shapely.union_all(shapely.polygons(xy))
            source_area=float(shapely.area(shapely.polygons(xy)).sum())
            before_area=region.area;filled=[];unresolved=[]
            if reference is not None and abs(normal[2])>1-1e-8:
                for polygon in polygon_parts(region):
                    for ring in polygon.interiors:
                        hole=Polygon(ring)
                        if hole.area>pore_area:continue
                        supported,info=supported_polygon(hole,origin,u,v,reference,cutoff=.05,min_area=1e-8)
                        # Admit the entire hole only; no partially filled pore.
                        if hole.difference(supported).area<1e-10:
                            filled.append(hole)
                        else:unresolved.append(hole.area)
                if filled:region=shapely.union_all([region,*filled])
            piece=mesh_from_polygon(region,origin,u,v)
            if piece is not None:pieces.append(piece)
            records.append({'normal':normal.tolist(),'offset_m':float(normal@origin),
                'source_area_m2':source_area,'union_area_m2':before_area,
                'duplicate_area_removed_m2':max(0,source_area-before_area),
                'pores_filled':len(filled),'added_area_m2':sum(p.area for p in filled),
                'small_pores_remaining':len(unresolved),
                'distance_bound_for_added_pores_mm':50 if filled else None})
        if not pieces:raise ValueError('Floor cleanup produced no valid surfaces')
        mesh=trimesh.util.concatenate(pieces)
        mesh.merge_vertices(digits_vertex=8)
        mesh.update_faces(mesh.nondegenerate_faces(height=1e-8) & (mesh.area_faces>=1e-12))
        mesh.remove_unreferenced_vertices()
        other.append({'name':f'L{level} consolidated floor surfaces','kind':'floor_planar_surfaces','level':level,
            'v':mesh.vertices.tolist(),'f':mesh.faces.tolist(),'colour':items[0].get('colour',[200,197,180]),
            'merge_coplanar_faces':True,'open_surface':True,'construction_solid':False,
            'evidence_status':'exact_existing_plane_union_with_bounded_small_pore_repair',
            'source_group_names':[p['name'] for p in items],'thickness_verified':False})
        audit.append({'level':level,'source_groups':len(items),'planes':records,
            'footprint_extension':False,'different_elevations_preserved':True})
    return other,audit


def main():
    from run_architectural_flow import read_manifest, read_registered_points, compare_coverage, render_review, validate_parts
    from audit_floor_artifacts import audit_floors
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--manifest',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args();source=Path(args.source).resolve();out=Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new floor-cleanup output folder')
    out.mkdir(parents=True,exist_ok=True)
    payload=json.loads((source/'model.build.json').read_text());parts=payload['parts']
    manifest=read_manifest(args.manifest)
    levels={p.get('level',0) for p in parts}
    if levels!={manifest['level']}:raise ValueError('CLI floor review requires one level; function accepts more')
    matrix=np.asarray(manifest['model_to_building'])
    scans=[{**s,'scan_to_model':(matrix@np.asarray(s['scan_to_model'])).tolist()} for s in manifest['scans']]
    xyz=np.vstack([p['v'] for p in parts if p['kind'].startswith('floor')])
    bounds=np.array([xyz.min(0)-.10,xyz.max(0)+.10])
    points,provenance=read_registered_points(scans,bounds,voxel_m=.01)
    refined,audit=consolidate_floor_surfaces(parts,cKDTree(points))
    validate_parts(refined)
    new_payload={**payload,'parts':refined}
    (out/'model.build.json').write_text(json.dumps(new_payload))
    before=[p for p in parts if p['kind'].startswith('floor')]
    after=[p for p in refined if p['kind'].startswith('floor')]
    coverage,raw,distances=compare_coverage(before,after,points)
    (out/'floor_cleanup_audit.json').write_text(json.dumps({'source':str(source),'scans':provenance,
        'cleanup':audit,'before':audit_floors(before),'after':audit_floors(after),
        'floor_coverage':coverage,'site_accuracy_certified':False},indent=2))
    render_review(refined,raw,distances,out,payload['label']+' | floor QA')
    additions=json.loads((source/'additions.build.json').read_text())
    additions['parts']=[{**p,'name':'Refined '+p['name']} for p in refined]
    additions['note']+=' Exact coplanar floor overlap removed; only wholly raw-bounded small pores filled. Existing floor datums and outer boundaries unchanged.'
    (out/'additions.build.json').write_text(json.dumps(additions))
    report=json.loads((source/'flow_report.json').read_text())
    report.update(parts=len(refined),floor_cleanup_source=str(source),native_export_pending=True,
                  floor_qa='floor_cleanup_audit.json')
    (out/'flow_report.json').write_text(json.dumps(report,indent=2))
    scene=trimesh.Scene()
    for p in refined:
        mesh=trimesh.Trimesh(p['v'],p['f'],process=False)
        mesh.visual.face_colors=np.r_[p.get('colour',[194,190,177]),255]
        scene.add_geometry(mesh,node_name=p['name'])
    scene.export(out/'model.glb')
    print(json.dumps({'floor_planes':sum(len(a['planes']) for a in audit),
        'duplicate_area_removed_m2':sum(p['duplicate_area_removed_m2'] for a in audit for p in a['planes']),
        'pores_filled':sum(p['pores_filled'] for a in audit for p in a['planes']),
        'parts':len(refined)},indent=2))

if __name__=='__main__':main()
