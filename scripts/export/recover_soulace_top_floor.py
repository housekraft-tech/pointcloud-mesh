"""Restore missing top-storey surface evidence against full-density raw returns."""
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import laspy
import trimesh
from scipy.spatial import cKDTree
from recover_mesh_evidence import bounded_triangles,mesh_part
from recover_soulace_details import original_payload
from inspect_scan_coverage import distance_to_parts

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output_final/soulace_top_floor_recovery_v4'


def top_reference():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'top_raw.npy').exists():return np.load(OUT/'top_raw.npy',mmap_mode='r'),np.load(OUT/'top_times.npy',mmap_mode='r')
    s=np.load(ROOT/'output_final/scan_first_diagnostics/soulace/sample.npz');matrix=s['matrix'];points=[];times=[]
    path=ROOT/'data/Soulace/clip_texture_optimize_optimised_2026-08-20_12-07-54_514-003.las'
    with laspy.open(path) as reader:
        for block in reader.chunk_iterator(2_000_000):
            q=np.column_stack((block.x,block.y,block.z))@matrix[:3,:3].T+matrix[:3,3]
            mask=(q[:,2]>6.25)&(q[:,2]<10.15)
            points.append(q[mask].astype(np.float32));times.append(np.asarray(block.gps_time)[mask])
    p=np.concatenate(points);t=np.concatenate(times);np.save(OUT/'top_raw.npy',p);np.save(OUT/'top_times.npy',t)
    print('Full-density top reference',len(p),flush=True);return p,t


def scaffold():
    cache=OUT/'scaffold.npz'
    if cache.exists():a=np.load(cache);return a['v'],a['f']
    import fast_simplification
    source=ROOT/'output_final/soulace_L2/poisson/poisson.npz';a=np.load(source)
    v=a['V'].astype(float);f=a['T'].astype(np.int32)
    metadata=json.loads((ROOT/'output_final/soulace_overlap_audit_20260903/Soulace_L2_overlap.json').read_text())['transform']
    v[:,2]+=metadata['las_z_percentile_0_5_m'];s=np.load(ROOT/'output_final/scan_first_diagnostics/soulace/sample.npz');matrix=s['matrix']
    v=v@matrix[:3,:3].T+matrix[:3,3]
    keep=(v[:,2]>6.5)&(v[:,2]<9.97);f=f[np.all(keep[f],axis=1)];used,ids=np.unique(f,return_inverse=True);v=v[used];f=ids.reshape(-1,3)
    print('Simplifying top surface',len(f),flush=True)
    v,f=fast_simplification.simplify(v,f.astype(np.int32),target_count=min(850000,len(f)),agg=5)
    np.savez_compressed(cache,v=v,f=f);print('Top scaffold',len(v),len(f),flush=True);return v,f


def recover():
    OUT.mkdir(parents=True,exist_ok=True)
    if list(OUT.glob('*.skp')):raise FileExistsError('Native revision exists; do not rewrite its sources')
    p,t=top_reference();tree=cKDTree(p,leafsize=32,compact_nodes=False)
    base=original_payload();base['parts']=[x for x in base['parts'] if x['name'].startswith('L2_')]
    v,f=scaffold();tri=v[f];centres=tri.mean(axis=1);dist=distance_to_parts(centres.astype(np.float32),base)
    normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-12)
    roof=(centres[:,0]>-4.15)&(centres[:,0]<.25)&(centres[:,1]>-.46)&(centres[:,1]<.24)&(centres[:,2]>6.52)&(centres[:,2]<9.68)
    roof &= centres[:,2]>(6.56-.83*centres[:,0]-.35)
    roof &= centres[:,2]<(6.56-.83*centres[:,0]+.55)
    # Do not turn horizontal roofs/ceilings into visible walls. Retain their evidence separately.
    missing=dist>.025
    labels=[(missing&~roof&(abs(normal[:,2])<.65),'Top floor - recovered vertical surface evidence','top_wall_scan',[209,181,113]),
            (roof,'Roof-access stair - raw surface reference','roof_stair_reference',[196,139,75]),
            (missing&~roof&(abs(normal[:,2])>=.65),'Top floor - horizontal surface reference','top_ceiling_scan',[142,151,156])]
    parts=[];reports=[]
    for mask,name,kind,colour in labels:
        candidate=tri[mask];print('Raw-checking',name,len(candidate),flush=True)
        supported,report=bounded_triangles(candidate,tree)
        part=mesh_part(supported,name,kind,colour);part['level']=2;parts.append(part)
        report.update(name=name,kind=kind,triangles=len(supported),area_m2=float(trimesh.Trimesh(part['v'],part['f'],process=False).area));reports.append(report)
        print(json.dumps(report),flush=True)
    (OUT/'recovered_parts.build.json').write_text(json.dumps({'parts':parts},separators=(',',':')))
    sample=np.load(ROOT/'output_final/soulace_top_floor_diagnostics/top_floor_coverage.npz');q=sample['p']
    prior=distance_to_parts(q,base);after=distance_to_parts(q,{'parts':base['parts']+parts})
    np.savez_compressed(OUT/'top_coverage.npz',p=q,before=prior,after=after)
    report={'source_mesh':str(ROOT/'output_final/soulace_L2/poisson/poisson.npz'),'raw_points':len(p),'parts':reports,
            'sample_count':len(q),'sample_within_50mm_before_after_pct':[float(np.mean(prior<=.05)*100),float(np.mean(after<=.05)*100)],
            'limitation':'Recovered surfaces, not automatic wall identity or inferred wall solids. Proximity bound is to raw returns, not certified site accuracy.'}
    (OUT/'recovery_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='parts'}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--cache-only',action='store_true');args=ap.parse_args()
    if args.cache_only:top_reference()
    else:recover()
