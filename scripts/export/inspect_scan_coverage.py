"""Read-only, scan-first diagnostics. Samples are for inspection, not certification."""
import argparse
import json
from pathlib import Path

import laspy
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'output_final/scan_first_diagnostics'


def sampled_scan(path, matrix, bounds, stride=8, voxel=.02):
    chunks=[]; total=0
    with laspy.open(path) as reader:
        for block in reader.chunk_iterator(2_000_000):
            p=np.column_stack((block.x[::stride],block.y[::stride],block.z[::stride]))
            p=p@matrix[:3,:3].T+matrix[:3,3]
            p=p[np.all((p>=bounds[0]) & (p<=bounds[1]),axis=1)]
            chunks.append(p);total+=len(block)
    points=np.concatenate(chunks)
    _,ids=np.unique(np.floor(points/voxel).astype(np.int32),axis=0,return_index=True)
    print(path.name,total,'raw;',len(ids),'display voxels',flush=True)
    return points[ids].astype(np.float32)


def distance_to_parts(points, payload):
    scene=o3d.t.geometry.RaycastingScene(nthreads=8)
    for part in payload['parts']:
        scene.add_triangles(o3d.core.Tensor(np.asarray(part['v'],np.float32)),
                            o3d.core.Tensor(np.asarray(part['f'],np.uint32)))
    out=[]
    for start in range(0,len(points),200_000):
        out.append(scene.compute_distance(o3d.core.Tensor(points[start:start+200_000])).numpy())
    return np.concatenate(out)


def horizontal_mask(p):
    tree=cKDTree(p);out=np.zeros(len(p),bool)
    for start in range(0,len(p),50000):
        q=p[start:start+50000]
        dist,ids=tree.query(q,k=20,workers=8)
        neighbors=p[ids];neighbors=neighbors-neighbors.mean(axis=1,keepdims=True)
        cov=np.einsum('nki,nkj->nij',neighbors,neighbors)/20
        vals,vecs=np.linalg.eigh(cov)
        out[start:start+len(q)]=(abs(vecs[:,2,0])>.95)&(vals[:,0]<.0001)&(dist[:,-1]<.13)
    return out


def inspect_soulace():
    folder=OUT/'soulace';folder.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'output_final/rectangular_rebuild_v1/soulace_l0.manifest.json').read_text())
    matrix=np.array(manifest['scans'][0]['scan_to_model'])
    cache=folder/'sample.npz'
    if cache.exists():
        data=np.load(cache);p=data['p'];horizontal=data['horizontal']
    else:
        path=ROOT/'data/Soulace/clip_texture_optimize_optimised_2026-08-20_12-07-54_514-003.las'
        # Same XY unit frame as old L0; retain below its old crop and all stairs.
        p=sampled_scan(path,matrix,np.array([[-13,-14,-2],[14,14,10.1]]),stride=8,voxel=.025)
        horizontal=horizontal_mask(p)
        np.savez_compressed(cache,p=p,horizontal=horizontal,matrix=matrix)
    odo=np.loadtxt(ROOT/'data/Soulace/odometerdata.txt')
    trajectory=odo[:,2:5]@matrix[:3,:3].T+matrix[:3,3]
    fig,ax=plt.subplots(2,3,figsize=(19,12),layout='constrained')
    for a,(lo,hi) in zip(ax[0], [(-1.5,.6),(.6,2.95),(3.4,6.3)]):
        q=p[horizontal & (p[:,2]>lo)&(p[:,2]<hi)]
        sc=a.scatter(q[:,0],q[:,1],c=q[:,2],s=.35,cmap='turbo',vmin=lo,vmax=hi)
        fig.colorbar(sc,ax=a,label='Z in common L0 frame (m)')
        a.set_title(f'Horizontal scan surfaces: {lo} < Z < {hi} m')
        a.set_aspect('equal');a.set_xlabel('X (m)');a.set_ylabel('Y (m)');a.grid(alpha=.2)
    for a,axis,label in [(ax[1,0],0,'X'),(ax[1,1],1,'Y')]:
        q=p[::4]
        a.scatter(q[:,axis],q[:,2],c='lightgrey',s=.15)
        a.plot(trajectory[:,axis],trajectory[:,2],c='crimson',linewidth=.8,label='Scanner trajectory (not floor)')
        a.set_aspect('equal');a.set_xlabel(label+' (m)');a.set_ylabel('Z (m)');a.legend();a.grid(alpha=.2)
    ax[1,2].hist(p[horizontal,2],bins=np.arange(-1.5,10.1,.01),orientation='horizontal')
    ax[1,2].set_xlabel('Occupied sample voxels');ax[1,2].set_ylabel('Horizontal surface Z (m)')
    fig.suptitle('Soulace | scan-only height investigation; no candidate-floor flattening',fontsize=18)
    fig.savefig(folder/'height_investigation.png',dpi=150);plt.close(fig)
    print(folder/'height_investigation.png',flush=True)


def inspect_engrance():
    folder=OUT/'engrance';folder.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'output_final/rectangular_rebuild_v1/mujammel.manifest.json').read_text())
    source=json.loads(Path(manifest['candidate_model']).read_text())
    v=np.vstack([p['v'] for p in source['parts']]);bounds=np.array([v.min(0)-[.35,.35,.3],v.max(0)+[.35,.35,.3]])
    cache=folder/'sample.npz'
    if cache.exists():p=np.load(cache)['p']
    else:
        p=np.vstack([sampled_scan(Path(s['path']),np.array(s['scan_to_model']),bounds,voxel=.02) for s in manifest['scans']])
        _,ids=np.unique(np.floor(p/.02).astype(np.int32),axis=0,return_index=True);p=p[ids]
        np.savez_compressed(cache,p=p,bounds=bounds)
    earlier=distance_to_parts(p,source)
    baseline=json.loads((ROOT/'output_final/coverage_restored_v1/engrance/model.build.json').read_text())
    current=distance_to_parts(p,baseline)
    np.savez_compressed(folder/'distance.npz',source=earlier,baseline=current)
    fig,ax=plt.subplots(2,3,figsize=(18,12),layout='constrained')
    for col,(mask,axes,title) in enumerate([
        ((p[:,2]>.65)&(p[:,2]<1.6),(0,1),'Plan band 0.65-1.60 m'),
        (np.ones(len(p),bool),(0,2),'Front elevation, all depths'),
        (np.ones(len(p),bool),(1,2),'Side elevation, all depths')]):
        for row,(d,label) in enumerate([(earlier,'Original CAD candidates'),(current,'LiDAR-filtered CAD')]):
            ids=np.flatnonzero(mask)[::2];good=d[ids]<=.05
            ax[row,col].scatter(p[ids[good],axes[0]],p[ids[good],axes[1]],s=.25,c='#85999e',rasterized=True)
            ax[row,col].scatter(p[ids[~good],axes[0]],p[ids[~good],axes[1]],s=.4,c='#d54638',rasterized=True)
            ax[row,col].set_title(label+' | '+title);ax[row,col].set_aspect('equal');ax[row,col].grid(alpha=.15)
    fig.suptitle('Engrance | red = scan samples >50 mm from the model\nIncludes furnishings/door leaves: red is missing evidence, not automatically a missing wall',fontsize=17)
    fig.savefig(folder/'missing_evidence.png',dpi=160);plt.close(fig)
    report={'sample_voxel_m':.02,'sampling_note':'Every 8th raw return before voxel uniqueness; diagnostic, not certified area coverage',
            'points':len(p),'original_candidate_within_50mm_percent':float(np.mean(earlier<=.05)*100),
            'baseline_within_50mm_percent':float(np.mean(current<=.05)*100),
            'warning':'Unmatched includes real details, doors, furnishings, noise, and surfaces outside CAD scope.'}
    (folder/'coverage_report.json').write_text(json.dumps(report,indent=2));print(report,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('project',choices=['engrance','soulace']);args=ap.parse_args()
    {'engrance':inspect_engrance,'soulace':inspect_soulace}[args.project]()
