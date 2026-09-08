"""Recover an old isolated-cloud coordinate transform from identical LAS records.

No geometric ICP: correspondences use millisecond time and 8-bit RGB.
Half of the matches validate the rigid transform without fitting it.
"""
import json
from pathlib import Path
import laspy
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output_final/scan_first_diagnostics/engrance'


def keys(p,stride=1):
    k=np.rint(np.asarray(p.gps_time)[::stride]*1000).astype(np.uint64)
    # The old PLY/LAS round-trip reset intensity/return flags and used RGB*256
    # instead of RGB*257. Neither difference changes the preserved 8-bit RGB.
    for field in ['red','green','blue']:
        k=(k ^ (np.asarray(p[field],dtype=np.uint64)[::stride]>>8))*np.uint64(1099511628211)
    return k


def main():
    path=ROOT/'output_final/mujammel_fine/lidar/mujammel_fine.las'
    cols=[];kk=[];heights=[]
    with laspy.open(path) as r:
        for p in r.chunk_iterator(2_000_000):
            cols.append(np.column_stack((p.x[::1000],p.y[::1000],p.z[::1000])));kk.append(keys(p,1000));heights.append(np.asarray(p.z).copy())
    iso=np.vstack(cols);ks=np.concatenate(kk)
    vals,ids,cnt=np.unique(ks,return_index=True,return_counts=True);ids=ids[cnt==1];ks=ks[ids];iso=iso[ids]
    order=np.argsort(ks);ks=ks[order];iso=iso[order]
    sources=[];targets=[];matched_keys=[]
    with laspy.open(ROOT/'mujammelexport.las') as r:
        for p in r.chunk_iterator(2_000_000):
            k=keys(p);i=np.searchsorted(ks,k);i=np.minimum(i,len(ks)-1);ok=ks[i]==k
            sources.append(np.column_stack((p.x[ok],p.y[ok],p.z[ok])));targets.append(iso[i[ok]])
            matched_keys.append(k[ok])
    a=np.vstack(sources);b=np.vstack(targets)
    _,unique_ids,counts=np.unique(np.concatenate(matched_keys),return_index=True,return_counts=True)
    unique_ids=unique_ids[counts==1];a=a[unique_ids];b=b[unique_ids]
    print('Unique record matches',len(a),flush=True)
    if len(a)<100:raise RuntimeError('Insufficient shared record attributes; do not guess the old frame')
    # Duplicate raw attributes can create bad pairs: rigid-fit residual consensus.
    mask=np.ones(len(a),bool)
    for _ in range(8):
        aa=a[mask][::2];bb=b[mask][::2];ca=aa.mean(0);cb=bb.mean(0)
        u,_,vt=np.linalg.svd((aa-ca).T@(bb-cb));r=vt.T@u.T
        if np.linalg.det(r)<0:vt[-1]*=-1;r=vt.T@u.T
        t=cb-r@ca;res=np.linalg.norm(a@r.T+t-b,axis=1)
        mask=res<=max(.002,float(np.median(res))*3)
    validation=res[mask][1::2]
    if np.percentile(validation,99)>.002:raise RuntimeError('Record-derived transform did not validate to 2 mm')
    raw_to_iso=np.eye(4);raw_to_iso[:3,:3]=r;raw_to_iso[:3,3]=t
    zshift=float(np.percentile(np.concatenate(heights),.5))
    manifest=json.loads((ROOT/'output_final/rectangular_rebuild_v1/mujammel.manifest.json').read_text())
    raw_to_cad=np.array(manifest['scans'][0]['scan_to_model'])
    # poisson_mesh.py subtracts this isolated LAS's .5-percentile Z.
    mesh_to_iso=np.eye(4);mesh_to_iso[2,3]=zshift
    mesh_to_cad=raw_to_cad@np.linalg.inv(raw_to_iso)@mesh_to_iso
    result={'raw_to_isolated':raw_to_iso.tolist(),'poisson_to_model':mesh_to_cad.tolist(),
            'matches':len(a),'inliers':int(mask.sum()),'validation_p99_mm':float(np.percentile(validation,99)*1000),
            'validation_max_mm':float(validation.max()*1000),'isolated_z_percentile_0_5_m':zshift,
            'note':'Corresponding record attributes establish old LAS frame; no geometric ICP or scale change. Poisson shift still must be checked against raw returns.'}
    (OUT/'old_mesh_frame.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
