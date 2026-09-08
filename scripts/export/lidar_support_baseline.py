"""Generic coverage-first CAD/LAS overlap baseline, without rectangle-fit gates.

Preserves arbitrary planar support shapes. No re-fitting, minimum rectangle
extent, normal-vote gate, or maximum rectangle count. This is the older
LiDAR-filtering approach, parameterized by the common input manifest.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import trimesh
from shapely import union_all

from rectangular_rebuild import load_reference, sha, distribution
from filter_soulace_overlap import plane_patches, supported_polygon, mesh_from_polygon, polygon_parts
from audit_soulace_overlap import uniform_surface


class ReferenceUnion:
    def __init__(self, references):
        self.references=references

    def query(self, points, **kwargs):
        distances=np.min([r['tree'].query(points,**kwargs)[0] for r in self.references],axis=0)
        return distances,np.zeros(len(points),dtype=np.int64)


def main(manifest_path, out):
    manifest_path=Path(manifest_path).resolve();out=Path(out).resolve()
    if (out/'model.build.json').exists():
        raise FileExistsError(out/'model.build.json')
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(manifest_path.read_text());base=manifest_path.parent
    model=(base/manifest['candidate_model']).resolve()
    payload=json.loads(model.read_text())['parts']
    xyz=np.concatenate([p['v'] for p in payload])
    references=[load_reference(s,base,[xyz.min(0)-.2,xyz.max(0)+.2]) for s in manifest['scans']]
    tree=ReferenceUnion(references)
    rng=np.random.default_rng(9210)
    parts=[];records=[];samples=[];weights=[]
    for i,item in enumerate(payload):
        original=trimesh.Trimesh(item['v'],item['f'],process=False)
        pieces=[]
        for polygon,origin,u,v in plane_patches(original):
            clipped,info=supported_polygon(polygon,origin,u,v,tree,cutoff=.05)
            if clipped.is_empty:
                continue
            # Match the earlier delivered LiDAR baseline's inward-only cleanup.
            clean=clipped.buffer(-.003,join_style=2).simplify(.002,preserve_topology=True).intersection(clipped)
            clean=union_all([p for p in polygon_parts(clean) if p.area>=.02])
            if clean.is_empty:
                continue
            result=mesh_from_polygon(clean,origin,u,v)
            if result is not None:
                pieces.append(result)
        record={'name':item['name'],'kind':item['kind'],'source_area_m2':float(original.area),'retained_area_m2':0.0}
        if pieces:
            mesh=trimesh.util.concatenate(pieces)
            mesh.merge_vertices(digits_vertex=9)
            mesh.update_faces(mesh.nondegenerate_faces());mesh.remove_unreferenced_vertices()
            mesh.vertices=np.round(mesh.vertices,8)
            q,_=uniform_surface(mesh,np.arange(len(mesh.faces)),max(500,int(mesh.area*1000)),rng)
            distances=tree.query(q,workers=8)[0]
            edge=mesh.vertices[mesh.edges_unique].mean(axis=1)
            maximum=max(float(distances.max()),float(tree.query(np.vstack((mesh.vertices,edge)),workers=8)[0].max()))
            if maximum>.0500001:
                raise ValueError(f'{item["name"]}: exceeded 50 mm bound')
            parts.append({**item,'v':mesh.vertices.tolist(),'f':mesh.faces.tolist(),'level':manifest.get('level',0),
                          'support_cutoff_mm':50,'open_surface':True,'evidence_status':'bounded_lidar_baseline'})
            record.update(retained_area_m2=float(mesh.area),triangles=len(mesh.faces),max_checked_distance_mm=maximum*1000,
                          area_sample_support=distribution(distances))
            samples.append(distances);weights.append(np.full(len(distances),mesh.area/len(distances)))
        records.append(record)
        print(f'{i+1}/{len(payload)} {item["name"]}: {record["retained_area_m2"]:.2f} / {original.area:.2f} m2',flush=True)
    report={'label':manifest['label']+' - coverage-first LiDAR baseline','manifest':manifest,
            'source_model_sha256':sha(model),'references':[r['provenance'] for r in references],
            'source_objects':len(payload),'retained_objects':len(parts),
            'retained_surface_area_m2':sum(r['retained_area_m2'] for r in records),
            'area_weighted_support':distribution(np.concatenate(samples),np.concatenate(weights)),
            'objects':records,'refit':False,'rectangular_selection':False,'hole_filling':False,
            'note':'Candidate-guided planar overlap; not all raw scan geometry and not proof of semantic identity.'}
    (out/'model.build.json').write_text(json.dumps({'label':report['label'],'parts':parts}))
    (out/'audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ['retained_objects','retained_surface_area_m2','area_weighted_support']},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--manifest',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();main(args.manifest,args.out)
