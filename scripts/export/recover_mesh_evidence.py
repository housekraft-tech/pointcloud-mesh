"""Recover scan-supported, unclassified mesh detail outside an existing CAD model.

The Poisson mesh is only a source of surface hypotheses. Every output triangle
must lie wholly within the raw-return distance envelope. This is not semantic
wall recognition, and does not certify survey accuracy.
"""
import gc
import argparse
import json
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from rectangular_rebuild import load_reference, sha
from lidar_support_baseline import ReferenceUnion
from inspect_scan_coverage import distance_to_parts

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output_final/scan_recovered_v1/engrance'


def bounded_triangles(triangles, tree, cutoff=.05, max_depth=8):
    """A nearest-return distance plus enclosing radius bounds the whole triangle."""
    accepted=[];discarded=0;max_bound=0.
    for start in range(0,len(triangles),100000):
        original=triangles[start:start+100000];todo=original
        roots=np.arange(len(todo));nodes=np.zeros(len(todo),np.int64)
        stride=4**(max_depth+1)
        by_depth=[[] for _ in range(max_depth+1)];keys_by_depth=[[] for _ in range(max_depth+1)]
        for depth in range(max_depth+1):
            if not len(todo):break
            centres=todo.mean(axis=1);radius=np.linalg.norm(todo-centres[:,None],axis=2).max(axis=1)
            d=tree.query(centres,workers=8)[0]
            good=d+radius<=cutoff-.000002
            if good.any():
                by_depth[depth].append(todo[good]);keys_by_depth[depth].append(roots[good]*stride+nodes[good]);max_bound=max(max_bound,float(np.max(d[good]+radius[good])))
            uncertain=(~good)&(d-radius<=cutoff)&(radius>.0025)
            discarded+=int((~good&~uncertain).sum())
            if depth==max_depth:
                discarded+=int(uncertain.sum());break
            t=todo[uncertain];a,b,c=t[:,0],t[:,1],t[:,2];ab=(a+b)/2;bc=(b+c)/2;ca=(c+a)/2
            todo=np.concatenate([np.stack(s,axis=1) for s in [(a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)]])
            roots=np.tile(roots[uncertain],4)
            nodes=np.concatenate([nodes[uncertain]*4+k for k in [1,2,3,4]])
        # Four certified children cover their parent exactly, including below a
        # partially clipped root. Coalesce that evidence without changing shape.
        for depth in range(max_depth,-1,-1):
            if not by_depth[depth]:continue
            leaves=np.concatenate(by_depth[depth]);keys=np.concatenate(keys_by_depth[depth])
            if depth==0:accepted.append(leaves);continue
            node=keys%stride;parent=(keys//stride)*stride+(node-1)//4
            unique,counts=np.unique(parent,return_counts=True)
            complete=unique[counts==4];can_merge=np.isin(parent,complete)
            accepted.append(leaves[~can_merge])
            if len(complete):
                order=np.lexsort(((node[can_merge]-1)%4,parent[can_merge]))
                children=leaves[can_merge][order].reshape(-1,4,3,3)
                merged=np.stack((children[:,0,0],children[:,1,1],children[:,2,2]),axis=1)
                by_depth[depth-1].append(merged);keys_by_depth[depth-1].append(complete)
    if not accepted:return np.empty((0,3,3)),{'max_bound_m':0,'discarded_terminal_triangles':discarded}
    return np.concatenate(accepted),{'max_bound_m':max_bound,'discarded_terminal_triangles':discarded}


def mesh_part(triangles,name,kind,colour):
    mesh=trimesh.Trimesh(triangles.reshape(-1,3),np.arange(len(triangles)*3).reshape(-1,3),process=False)
    mesh.merge_vertices(digits_vertex=8)
    return {'name':name,'kind':kind,'colour':colour,'level':0,'v':np.round(mesh.vertices,8).tolist(),'f':mesh.faces.tolist()}


def main(rebuild=False):
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'model.build.json').exists() and not rebuild:raise FileExistsError('Recovery exists; use a new revision')
    if rebuild and list(OUT.glob('*.skp')):raise FileExistsError('Native delivery exists; do not alter its source')
    metadata=json.loads((ROOT/'output_final/scan_first_diagnostics/engrance/old_mesh_frame.json').read_text())
    matrix=np.array(metadata['poisson_to_model'])
    source=ROOT/'output_final/mujammel_fine/poisson/poisson.npz'
    cache=OUT/'simplified_scaffold.npz'
    if cache.exists():z=np.load(cache);v=z['v'];f=z['f']
    else:
        import fast_simplification
        print('Read existing detailed mesh',flush=True)
        z=np.load(source);v=z['V'].astype(float);f=z['T'].astype(np.int32)
        print('Simplify scaffold',len(v),len(f),flush=True)
        v,f=fast_simplification.simplify(v,f,target_count=1_200_000,agg=5)
        v=v@matrix[:3,:3].T+matrix[:3,3]
        np.savez_compressed(cache,v=v,f=f);print('Simplified',len(v),len(f),flush=True)
        gc.collect()
    baseline=json.loads((ROOT/'output_final/coverage_restored_v1/engrance/model.build.json').read_text())
    mesh=trimesh.Trimesh(v,f,process=False)
    dc=distance_to_parts(mesh.triangles_center.astype(np.float32),baseline)
    # Keep non-CAD evidence: never label these automatically as walls/columns.
    wanted=dc>.030
    tri=mesh.triangles[wanted].copy();del mesh,v,f,dc;gc.collect()
    print('Candidate residual triangles',len(tri),flush=True)
    manifest=json.loads((ROOT/'output_final/rectangular_rebuild_v1/mujammel.manifest.json').read_text())
    bounds=np.array([tri.reshape(-1,3).min(0)-.1,tri.reshape(-1,3).max(0)+.1])
    refs=[load_reference(s,ROOT,bounds) for s in manifest['scans']]
    tree=ReferenceUnion(refs)
    # A 3-D overlap test with full-density raw returns, not mesh-to-mesh support.
    supported,report=bounded_triangles(tri,tree)
    print('Raw-supported residual triangles',len(supported),report,flush=True)
    np.savez_compressed(OUT/'supported_residual.npz',triangles=supported)
    norms=np.cross(supported[:,1]-supported[:,0],supported[:,2]-supported[:,0]);norms/=np.maximum(np.linalg.norm(norms,axis=1)[:,None],1e-12)
    centres=supported.mean(axis=1)
    ceiling=(centres[:,2]>2.25)&(abs(norms[:,2])>.65)
    added=[]
    for mask,name,kind,colour in [(~ceiling,'Observed scan detail - UNCLASSIFIED','scan_detail',[94,159,171]),(ceiling,'Observed ceiling detail - UNCLASSIFIED','scan_ceiling_detail',[127,177,180])]:
        if mask.any():added.append(mesh_part(supported[mask],name,kind,colour))
    result={**baseline,'label':'Engrance | coverage baseline + observed unclassified scan detail','parts':baseline['parts']+added}
    (OUT/'model.build.json').write_text(json.dumps(result,separators=(',',':')))
    sample=np.load(ROOT/'output_final/scan_first_diagnostics/engrance/sample.npz')['p']
    before=distance_to_parts(sample,baseline);after=distance_to_parts(sample,result)
    np.savez_compressed(OUT/'coverage_samples.npz',p=sample,before=before,after=after)
    report.update(source_mesh=str(source),source_sha256=sha(source),old_mesh_frame=metadata,
                  source_baseline_parts=len(baseline['parts']),added_parts=len(added),added_triangles=len(supported),
                  sample_voxels=len(sample),sample_within_50mm_before_pct=float(np.mean(before<=.05)*100),
                  sample_within_50mm_after_pct=float(np.mean(after<=.05)*100),
                  original_parts_unchanged=True,classification='UNCLASSIFIED observed detail, may include doors/furniture/scaffolding',
                  reference_scans=[r['provenance'] for r in refs],
                  limitation='50 mm is a proximity envelope to the scan, not absolute accuracy or proof of a wall. Unsupported or unseen detail remains absent.')
    (OUT/'recovery_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['reference_scans','old_mesh_frame']},indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--rebuild-intermediate',action='store_true');args=ap.parse_args();main(args.rebuild_intermediate)
