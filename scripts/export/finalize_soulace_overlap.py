"""Clean overlap boundaries inward, recheck against LAS, and prepare handover."""
import gc
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from audit_soulace_overlap import ROOT, STEMS, raw_points, uniform_surface, say
from filter_soulace_overlap import plane_patches, mesh_from_polygon, polygon_parts, overlap_stats
from shapely import union_all


SOURCE = ROOT / 'output_final/soulace_overlap_only_50mm'
OUT = SOURCE / 'release'


def finish():
    OUT.mkdir(exist_ok=True)
    combined, reports = [], []
    offsets = [0,3.2,6.5]
    for level in [0,1,2]:
        rng = np.random.default_rng(18400+level)
        input_payload = json.loads((SOURCE/f'Soulace_L{level}_overlap_only.build.json').read_text())['parts']
        first_audit = json.loads((SOURCE/f'Soulace_L{level}_filter_audit.json').read_text())
        metadata = json.loads((ROOT/'output_final/soulace_asbuilt_v2'/f'{STEMS[level]}_asbuilt_manifest.json').read_text())
        say(f'L{level}: inward-only boundary cleanup and full LAS recheck')
        raw, _ = raw_points(ROOT/f'output_final/soulace_L{level}/lidar/L{level}.las',metadata['source_yaw_deg'],metadata['source_floor_z_m'])
        tree = cKDTree(raw,leafsize=32,compact_nodes=False)
        payload, records, weighted_distances, weights = [], [], [], []
        for index, item in enumerate(input_payload):
            source_mesh = trimesh.Trimesh(item['v'],item['f'],process=False)
            pieces = []
            for poly, origin, u, v in plane_patches(source_mesh):
                # Erosion provides a margin for line simplification. Clipping
                # back to input is mandatory: no new area can be introduced.
                candidate = poly.buffer(-.003,join_style=2).simplify(.002,preserve_topology=True).intersection(poly)
                selected = union_all([p for p in polygon_parts(candidate) if p.area >= .02])
                if selected.is_empty:
                    continue
                if selected.difference(poly).area > 1e-9:
                    raise ValueError('Boundary cleanup added unsupported area')
                piece = mesh_from_polygon(selected,origin,u,v)
                if piece is not None:
                    pieces.append(piece)
            if not pieces:
                records.append({'name':item['name'],'removed_during_cleanup':True})
                continue
            mesh = trimesh.util.concatenate(pieces)
            mesh.merge_vertices(digits_vertex=9)
            mesh.update_faces(mesh.nondegenerate_faces())
            mesh.remove_unreferenced_vertices()
            mesh.vertices = np.round(mesh.vertices,8)
            sample, _ = uniform_surface(mesh,np.arange(len(mesh.faces)),max(500,int(mesh.area*1000)),rng)
            d = tree.query(sample,workers=8)[0]
            edge_mid = mesh.vertices[mesh.edges_unique].mean(axis=1)
            bound_check = tree.query(np.vstack((mesh.vertices,edge_mid)),workers=8)[0]
            maximum = max(float(d.max()),float(bound_check.max()))
            if maximum > .05+1e-7:
                raise ValueError(f'{item["name"]} failed recheck: {maximum}')
            weighted_distances.append(d)
            weights.append(np.full(len(d),mesh.area/len(d)))
            item = {**item,'v':mesh.vertices.tolist(),'f':mesh.faces.tolist(),'level':level}
            payload.append(item)
            record = {'name':item['name'],'kind':item['kind'],'area_m2':round(float(mesh.area),6),
                      'triangles':len(mesh.faces),'maximum_checked_distance_mm':round(maximum*1000,5),
                      'area_uniform_sample':overlap_stats(d),'boundary_checks':len(bound_check)}
            records.append(record)
            if index%10 == 0:
                say(f'L{level} {index+1}/{len(input_payload)}: {item["name"]}, {len(source_mesh.faces):,} -> {len(mesh.faces):,} triangles')
        distances = np.concatenate(weighted_distances)
        weight = np.concatenate(weights)
        order = np.argsort(distances)
        cumulative = np.cumsum(weight[order]) / weight.sum()
        summary = {'area_m2':float(weight.sum()),'uniform_samples':len(distances),
                   'median_mm':round(float(distances[order[np.searchsorted(cumulative,.5)]]*1000),3),
                   **{f'within_{t}mm_pct':round(float(weight[distances<=t/1000].sum()/weight.sum()*100),2) for t in [10,20,30,50]}}
        report = {'level':level,'source_objects':first_audit['source_objects'],'retained_objects':len(payload),
                  'source_area_m2':first_audit['source_surface_area_m2'],
                  'area_weighted_sample_support':summary,'objects':records,
                  'note':'Surface area and distance to LAS, not independent room dimensional accuracy.'}
        reports.append(report)
        (OUT/f'Soulace_L{level}_overlap_only.build.json').write_text(json.dumps({'parts':payload}))
        for item in payload:
            q = dict(item)
            vertices = np.asarray(q['v']); vertices[:,2] += offsets[level]
            q['v'] = np.round(vertices,8).tolist()
            q['name'] = f'L{level}_{q["name"]}'
            combined.append(q)
        del raw,tree
        gc.collect()
    (OUT/'Soulace_overlap_only.build.json').write_text(json.dumps({'parts':combined}))
    (OUT/'release_audit.json').write_text(json.dumps({'target_mm':10,'maximum_support_distance_mm':50,
       'levels':reports,'boundary_cleanup':'3 mm inward erosion; 2 mm simplification clipped to original supported area.',
       'open_surfaces':True,'added_thickness':False,'hole_filling':False,
       'geometry_note':'Subset of original CAD; distant surfaces are removed, not moved. Proximity does not prove wall identity or building accuracy.'},indent=2))
    say(f'Release prepared: {len(combined)} objects, {sum(len(p["f"]) for p in combined):,} triangles')


if __name__ == '__main__':
    finish()
