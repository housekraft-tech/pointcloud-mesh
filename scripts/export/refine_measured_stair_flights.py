"""Fit individually observed tread and riser planes in registered stair lanes.

The numeric fitting functions accept arbitrary lanes and do not impose equal
risers, fixed orientation, or a property-specific dimension template.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from clean_soulace_surfaces import fit_riser,quad,polygon_triangles
from recover_mesh_evidence import mesh_part
from shapely.geometry import Polygon,box
from scipy.spatial import cKDTree
from architectural_surface_refinement import proximity_evidence

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output_final/soulace_architectural_v9'


def diagnose():
    sample=np.load(ROOT/'output_final/scan_first_diagnostics/soulace/sample.npz')
    p=sample['p'];h=p[sample['horizontal']]
    roi=(h[:,0]>-4.86)&(h[:,0]<-2.8)&(h[:,1]>1.2)&(h[:,1]<5.8)&(h[:,2]>-.05)&(h[:,2]<3.24)
    h=h[roi]
    fig,axes=plt.subplots(2,2,figsize=(16,13),layout='constrained');rows=[]
    for col,(name,xlo,xhi) in enumerate([('left',-4.8,-3.93),('right',-3.80,-2.95)]):
        q=h[(h[:,0]>xlo)&(h[:,0]<xhi)]
        hist,edges=np.histogram(q[:,2],np.arange(-.05,3.251,.003))
        ids=find_peaks(hist,prominence=12,distance=30)[0]
        for i in ids:
            z=(edges[i]+edges[i+1])/2;t=q[abs(q[:,2]-z)<.015]
            rows.append({'lane':name,'z':float(np.median(t[:,2])),'count':len(t),
                         'xyz_p05_p50_p95':np.percentile(t,[5,50,95],axis=0).tolist()})
        axes[0,col].scatter(q[:,1],q[:,2],s=1,c=q[:,0],cmap='viridis');axes[0,col].set_aspect('equal');axes[0,col].grid(alpha=.2);axes[0,col].set_title(name+' lane Y/Z')
        axes[1,col].scatter(q[:,0],q[:,1],s=2,c=q[:,2],cmap='turbo');axes[1,col].set_aspect('equal');axes[1,col].grid(alpha=.2);axes[1,col].set_title(name+' lane X/Y')
    fig.savefig(OUT/'lower_stair_diagnostic.png',dpi=140);plt.close(fig)
    (OUT/'lower_stair_modes.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2),flush=True)


def fit_flight(raw, horizontal, *, name, lane, direction, start_z, end_z, seeds, level):
    """Independently fit reviewed top-height modes and actual vertical risers."""
    supports=[];heights=[];widths=[]
    for along,z_seed in seeds:
        p=horizontal[(horizontal[:,0]>lane[0])&(horizontal[:,0]<lane[1])&
                     (abs(horizontal[:,1]-along)<.13)&(abs(horizontal[:,2]-z_seed)<.028)]
        if len(p)<25:raise ValueError(f'{name} tread {along}: insufficient horizontal evidence ({len(p)})')
        heights.append(float(np.median(p[:,2])));supports.append(p)
        widths.append(np.percentile(p[:,0],[.5,99.5]))
    xl,xh=np.median(widths,axis=0)+[-.012,.012]
    risers=[];riser_audit=[]
    for i,(p,z) in enumerate(zip(supports,heights)):
        edge=float(np.percentile(p[:,1],1 if direction>0 else 99))-direction*.015
        y,qa=fit_riser(raw,(xl,xh),edge,start_z if i==0 else heights[i-1],z)
        risers.append(y);riser_audit.append(qa)
    y,qa=fit_riser(raw,(xl,xh),seeds[-1][0]+direction*.14,heights[-1],end_z)
    risers.append(y);riser_audit.append(qa)
    if not np.all(np.diff(risers)*direction>.12):raise ValueError('Measured riser sequence reverses')
    tread=[];riser=[]
    for i,z in enumerate(heights):
        lo,hi=sorted(risers[i:i+2]);tread.extend(quad([[xl,lo,z],[xh,lo,z],[xh,hi,z],[xl,hi,z]]))
        y=risers[i];z0=start_z if i==0 else heights[i-1]
        riser.extend(quad([[xl,y,z0],[xh,y,z0],[xh,y,z],[xl,y,z]]))
    y=risers[-1];riser.extend(quad([[xl,y,heights[-1]],[xh,y,heights[-1]],[xh,y,end_z],[xl,y,end_z]]))
    parts=[]
    for tri,label,colour in [(tread,'treads',[219,149,76]),(riser,'risers',[197,123,61])]:
        part=mesh_part(np.asarray(tri),name+' - measured '+label,'stair_clean',colour)
        part.update(level=level,merge_coplanar_faces=True);parts.append(part)
    return parts,{'name':name,'lane_m':[float(xl),float(xh)],'direction':direction,
                  'tread_heights_m':heights,'riser_positions_m':risers,'riser_evidence':riser_audit,
                  'rises_mm':(np.diff([start_z]+heights+[end_z])*1000).tolist(),
                  'goings_mm':(abs(np.diff(risers))*1000).tolist(),'equal_dimensions_imposed':False,
                  'tread_plane_p95_mm':[float(np.percentile(abs(p[:,2]-z),95)*1000) for p,z in zip(supports,heights)]}


def build_lower():
    sample=np.load(ROOT/'output_final/scan_first_diagnostics/soulace/sample.npz')
    p=sample['p'];horizontal=p[sample['horizontal']]
    cache=OUT/'lower_stair_raw.npy'
    if cache.exists():raw=np.load(cache)
    else:
        import laspy
        matrix=sample['matrix'];blocks=[]
        with laspy.open(ROOT/'data/Soulace/clip_texture_optimize_optimised_2026-08-20_12-07-54_514-003.las') as reader:
            for chunk in reader.chunk_iterator(2000000):
                q=np.column_stack((chunk.x,chunk.y,chunk.z))@matrix[:3,:3].T+matrix[:3,3]
                keep=np.all((q>[-4.96,1.1,-.1])&(q<[-2.65,5.9,3.35]),axis=1)
                blocks.append(q[keep].astype(np.float32))
        raw=np.concatenate(blocks);np.save(cache,raw)
    landing=horizontal[(horizontal[:,0]>-4.80)&(horizontal[:,0]<-2.95)&(horizontal[:,1]>4.4)&(horizontal[:,1]<5.65)&(abs(horizontal[:,2]-1.857)<.025)]
    mid_z=float(np.median(landing[:,2]));end_z=3.2041
    specs=[{'name':'Lower outbound','lane':[-4.85,-3.90],'direction':1,'start_z':0.0,'end_z':mid_z,
            'seeds':[(1.76,.167),(2.01,.332),(2.26,.501),(2.51,.670),(2.76,.840),(3.01,1.006),(3.26,1.177),(3.51,1.348),(3.76,1.518),(4.01,1.692)]},
           {'name':'Lower return','lane':[-3.84,-2.88],'direction':-1,'start_z':mid_z,'end_z':end_z,
            'seeds':[(4.10,2.028),(3.85,2.200),(3.60,2.369),(3.35,2.542),(3.10,2.711),(2.85,2.883),(2.60,3.054)]}]
    parts=[];audit=[]
    for spec in specs:
        built,row=fit_flight(raw,horizontal,level=0,**spec);parts.extend(built);audit.append(row)
    left,right=audit;xl=left['lane_m'][0];xh=right['lane_m'][1];mid=(left['lane_m'][1]+right['lane_m'][0])/2
    far=float(np.percentile(landing[:,1],99.5))+.012
    outline=Polygon([(xl,left['riser_positions_m'][-1]),(mid,left['riser_positions_m'][-1]),
                     (mid,right['riser_positions_m'][0]),(xh,right['riser_positions_m'][0]),(xh,far),(xl,far)])
    part=mesh_part(polygon_triangles(outline,[0,0,mid_z]),'Lower intermediate landing - measured plane','stair_clean',[224,156,85]);part.update(level=0,merge_coplanar_faces=True);parts.append(part)
    near=horizontal[(horizontal[:,0]>xl)&(horizontal[:,0]<xh)&(horizontal[:,1]>1.35)&(horizontal[:,1]<right['riser_positions_m'][-1]+.02)&(abs(horizontal[:,2]-end_z)<.02)]
    lo_y=float(np.percentile(near[:,1],.5))
    part=mesh_part(polygon_triangles(box(xl,lo_y,xh,right['riser_positions_m'][-1]),[0,0,end_z]),'Lower floor arrival landing - measured plane','stair_clean',[224,156,85]);part.update(level=1,merge_coplanar_faces=True);parts.append(part)
    tree=cKDTree(raw)
    for part in parts:
        evidence=proximity_evidence(np.asarray(part['v'])[np.asarray(part['f'])],tree)
        part['support_evidence']=evidence
    (OUT/'lower_stairs.build.json').write_text(json.dumps({'parts':parts},separators=(',',':')))
    (OUT/'lower_stairs_fit.json').write_text(json.dumps({'flights':audit,'landing_z_m':mid_z,'arrival_z_m':end_z,
        'raw_points':len(raw),'proximity':[{'name':p['name'],**p['support_evidence']} for p in parts],
        'limitations':'Reviewed top-surface modes; underside modes excluded. Treads and risers independently fit; horizontal landings modeled within observed extents. No equal risers or wall additions.'},indent=2))
    fig,axes=plt.subplots(1,2,figsize=(18,8),layout='constrained')
    for ax,row in zip(axes,audit):
        q=raw[(raw[:,0]>row['lane_m'][0]+.08)&(raw[:,0]<row['lane_m'][1]-.08)]
        ax.scatter(q[::3,1],q[::3,2],s=.1,c='#aeb4b7')
        for i,z in enumerate(row['tread_heights_m']):ax.plot(row['riser_positions_m'][i:i+2],[z,z],c='#d57323',lw=2)
        ax.set_aspect('equal');ax.grid(alpha=.2);ax.set_title(row['name']+' - measured treads over raw scan')
    fig.savefig(OUT/'lower_stair_scan_check.png',dpi=150);plt.close(fig)
    print(json.dumps({'parts':len(parts),'flights':audit,'support':[{p['name']:p['support_evidence']} for p in parts]},indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--build-lower',action='store_true');args=ap.parse_args()
    if args.build_lower:build_lower()
    else:diagnose()
