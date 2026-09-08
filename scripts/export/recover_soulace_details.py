"""Restore observed staircase skins and multiple ground-surface elevations.

No equal-riser template, no single parking-floor datum, and no slab extrusion.
"""
import gc
import json
from pathlib import Path
import laspy
import numpy as np
import trimesh
from scipy.spatial import Delaunay, cKDTree
from scipy.stats import binned_statistic_2d
from recover_mesh_evidence import bounded_triangles,mesh_part
from inspect_scan_coverage import distance_to_parts

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output_final/scan_recovered_v1/soulace'


def ground_mesh(points,cell=.03):
    """Local height field. Large XY gaps and height jumps are left open."""
    lo=points[:,:2].min(0)-cell;hi=points[:,:2].max(0)+cell
    edges=[np.arange(lo[k],hi[k]+cell,cell) for k in range(2)]
    values=[]
    for k in range(3):
        values.append(binned_statistic_2d(points[:,0],points[:,1],points[:,k],statistic='median',bins=edges)[0])
    low=binned_statistic_2d(points[:,0],points[:,1],points[:,2],statistic='min',bins=edges)[0]
    high=binned_statistic_2d(points[:,0],points[:,1],points[:,2],statistic='max',bins=edges)[0]
    valid=np.isfinite(values[2])&((high-low)<.04)
    v=np.column_stack([a[valid] for a in values]);faces=Delaunay(v[:,:2]).simplices
    tri=v[faces];sides=tri-np.roll(tri,1,axis=1)
    keep=(np.linalg.norm(sides[:,:,:2],axis=2).max(axis=1)<.085)&(np.ptp(tri[:,:,2],axis=1)<.04)
    return tri[keep]


def original_payload():
    config=json.loads((ROOT/'output_final/coverage_restored_v1/soulace/restore_manifest.json').read_text())
    parts=[]
    for level,nominal in enumerate([0,3.2,6.5]):
        source=ROOT/f'output_final/soulace_overlap_only_50mm/release/Soulace_L{level}_overlap_only.build.json'
        payload=json.loads(source.read_text());matrix=np.array(config['source_to_common'][str(level)])
        for part in payload['parts']:
            v=np.array(part['v']);v[:,2]+=nominal;v=v@matrix[:3,:3].T+matrix[:3,3]
            parts.append({**part,'name':f'L{level}_'+part['name'],'v':v.tolist()})
    return {'parts':parts}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'additions.build.json').exists():raise FileExistsError('Use a new revision')
    sample=np.load(ROOT/'output_final/scan_first_diagnostics/soulace/sample.npz')
    p=sample['p'];horizontal=sample['horizontal'];matrix=sample['matrix']
    source_model=original_payload();v=np.vstack([x['v'] for x in source_model['parts']]);bounds=np.array([v.min(0),v.max(0)])
    stair_box=np.array([[-4.85,1.35,-.08],[-2.65,5.72,6.64]])
    floor_bounds=np.array([bounds[0,:2]-.08,bounds[1,:2]+.08])
    print('Model bounds',bounds.tolist(),flush=True)
    floor_points=p[horizontal&(p[:,2]>-1.5)&(p[:,2]<.1)&np.all((p[:,:2]>floor_bounds[0])&(p[:,:2]<floor_bounds[1]),axis=1)]
    floor=ground_mesh(floor_points)
    floor=floor[distance_to_parts(floor.mean(axis=1).astype(np.float32),source_model)>.025]
    print('Ground triangles absent from prior CAD',len(floor),flush=True)
    stair_parts=[]
    for level in range(3):
        metadata=json.loads((ROOT/f'output_final/soulace_overlap_audit_20260903/Soulace_L{level}_overlap.json').read_text())['transform']
        path=ROOT/f'output_final/soulace_L{level}/poisson/poisson.npz'
        z=np.load(path);v=z['V'].astype(float);f=z['T'];v[:,2]+=metadata['las_z_percentile_0_5_m']
        v=v@matrix[:3,:3].T+matrix[:3,3]
        inside=np.all((v>=stair_box[0])&(v<=stair_box[1]),axis=1)
        faces=f[np.all(inside[f],axis=1)];used,ids=np.unique(faces,return_inverse=True)
        vv=v[used];ff=ids.reshape(-1,3);del v,f,z;gc.collect()
        if len(ff)>220000:
            import fast_simplification
            vv,ff=fast_simplification.simplify(vv,ff.astype(np.int32),target_count=220000,agg=5)
        # Remove overlapping storey crops by assigning centroid height intervals.
        tri=vv[ff];centres=tri.mean(axis=1)
        mask=(centres[:,2]>=[-.08,3.1,6.4][level])&(centres[:,2]<[3.1,6.4,6.64][level])
        stair_parts.append(tri[mask]);print('Stair scaffold level',level,len(stair_parts[-1]),flush=True)
    stairs=np.concatenate(stair_parts)
    stairs=stairs[distance_to_parts(stairs.mean(axis=1).astype(np.float32),source_model)>.025]
    del source_model;gc.collect()
    path=ROOT/'data/Soulace/clip_texture_optimize_optimised_2026-08-20_12-07-54_514-003.las'
    raw=[];total=0
    with laspy.open(path) as reader:
        for block in reader.chunk_iterator(2_000_000):
            q=np.column_stack((block.x,block.y,block.z))@matrix[:3,:3].T+matrix[:3,3];total+=len(q)
            ground=(q[:,2]>-1.65)&(q[:,2]<.25)&np.all((q[:,:2]>floor_bounds[0]-.1)&(q[:,:2]<floor_bounds[1]+.1),axis=1)
            stair=np.all((q>stair_box[0]-.1)&(q<stair_box[1]+.1),axis=1)
            raw.append(q[ground|stair])
    raw=np.concatenate(raw);tree=cKDTree(raw,leafsize=32,compact_nodes=False)
    print('Full-density local raw reference',len(raw),'of',total,flush=True)
    parts=[];audit=[];supported_sets=[]
    for tri,name,kind,colour in [(floor,'Lower floor and perimeter grade - OBSERVED','lower_floor_observed',[85,161,187]),(stairs,'Staircase surfaces - OBSERVED','stair_observed',[216,133,64])]:
        supported,report=bounded_triangles(tri,tree);part=mesh_part(supported,name,kind,colour)
        parts.append(part);report.update(name=name,triangles=len(supported),area_m2=float(trimesh.Trimesh(part['v'],part['f'],process=False).area))
        audit.append(report);supported_sets.append(supported);print(report,flush=True)
    # Split staircase tag at actual storey levels, without shifting any vertex.
    stair_part=parts.pop();tri=np.array(stair_part['v'])[np.array(stair_part['f'])]
    for level,(lo,hi) in enumerate([(-.1,3.2041),(3.2041,6.5544),(6.5544,6.7)]):
        subset=tri[(tri.mean(axis=1)[:,2]>=lo)&(tri.mean(axis=1)[:,2]<hi)]
        if not len(subset):continue
        part=mesh_part(subset,f'L{level} Staircase surfaces - OBSERVED','stair_observed',[216,133,64]);part['level']=level;parts.append(part)
    payload={'label':'Soulace | restored coverage + observed stairs and lower floors',
             'source_native':str(ROOT/'output_final/coverage_restored_v1/soulace/Soulace_LiDAR_coverage_restored.skp'),
             'expected_base_groups':295,'parts':parts}
    (OUT/'additions.build.json').write_text(json.dumps(payload,separators=(',',':')))
    # Describe ground elevations spatially; never substitute a universal datum.
    floor_stats=[]
    for name,xy in [('interior datum',[[-8,-3],[-6,1]]),('lower east floor',[[3,-3],[5,-1]]),('east entrance strip',[[6,-3.8],[7,-2]])]:
        q=floor_points[np.all((floor_points[:,:2]>xy[0])&(floor_points[:,:2]<xy[1]),axis=1)]
        if len(q):floor_stats.append({'region':name,'xy_bounds_m':xy,'sample_voxels':len(q),'median_z_m':float(np.median(q[:,2])),'p05_p95_z_m':np.percentile(q[:,2],[5,95]).tolist()})
    report={'source_las':str(path),'raw_returns_in_file':total,'raw_returns_in_local_reference':len(raw),'scan_to_common':matrix.tolist(),
            'stairs_roi_m':stair_box.tolist(),'floor_xy_bounds_m':floor_bounds.tolist(),'parts':audit,'floor_height_samples':floor_stats,
            'existing_geometry_unchanged':True,'equal_risers_imposed':False,'single_parking_floor_imposed':False,
            'floor_method':'30 mm local median height cells; no triangles spanning >85 mm XY or >40 mm Z; full-density 50 mm envelope afterward',
            'stair_method':'Existing measured Poisson surface in staircase ROI; independent full-density raw-scan envelope afterward. Includes tread, riser, underside and surrounding observed detail.',
            'limitations':'Open surface evidence, not watertight construction solids. Scan proximity is not a certified site dimension. Some occlusions and scan clutter remain.'}
    (OUT/'recovery_audit.json').write_text(json.dumps(report,indent=2));np.savez_compressed(OUT/'detail_triangles.npz',floor=supported_sets[0],stairs=supported_sets[1]);print(json.dumps(floor_stats,indent=2),flush=True)


if __name__=='__main__':main()
