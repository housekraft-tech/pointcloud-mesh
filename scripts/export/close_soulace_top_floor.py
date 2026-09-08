"""Continuous top-floor wall runs from measured faces and explicit modeled closure.

Unlike a point-coverage mesh, missing samples are not automatically wall openings.
Thickness retained from an earlier hypothesis is explicitly unverified.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.signal import find_peaks
from shapely.geometry import box,Polygon
from shapely.ops import unary_union
import shapely
import trimesh
from model_soulace_top_walls import source_data,local_raw,solid_from_profile

ROOT=Path(__file__).resolve().parents[2]
PRIOR=ROOT/'output_final/soulace_top_floor_walls_v5'
OUT=ROOT/'output_final/soulace_top_floor_closed_v6'


def inputs():
    parts,meta,matrix=source_data()
    q=local_raw(matrix)
    records=json.loads((PRIOR/'wall_plane_diagnostics.json').read_text())
    fits=json.loads((PRIOR/'wall_fit_audit.json').read_text())['walls']
    fits={r['name']:r for r in fits if 'measured_faces_m' in r}
    return parts,meta,matrix,q,records,fits


def wall_returns(q,item,faces,pad=.12):
    lo,hi=np.array(item['bounds']);axis=item['axis'];along=1-axis
    local=q[(q[:,along]>=lo[along]-pad)&(q[:,along]<=hi[along]+pad)&
            (q[:,2]>=-.02)&(q[:,2]<min(3.45,hi[2]+.01))]
    distance=np.min(abs(local[:,axis,None]-np.array(faces)[None,:]),axis=1)
    return local[distance<.03]


def diagnostics():
    OUT.mkdir(parents=True,exist_ok=True)
    parts,meta,matrix,q,records,fits=inputs()
    fig,axes=plt.subplots(6,4,figsize=(23,23),layout='constrained')
    fig2,ax=plt.subplots(figsize=(13,11),layout='constrained')
    p=q[::35];p=p[(p[:,2]>.2)&(p[:,2]<2.8)]
    ax.scatter(p[:,0],p[:,1],s=.3,c='#bec9cf')
    for item,a in zip(records,axes.flat):
        faces=fits[item['name']]['measured_faces_m'] if item['name'] in fits else [item['modes'][0]['d']]
        p=wall_returns(q,item,faces);axis=item['axis'];along=1-axis
        sample=p[::max(1,len(p)//18000)]
        a.scatter(sample[:,along],sample[:,2],s=.2,c='#276e87')
        lo,hi=np.array(item['bounds']);a.set_xlim(lo[along]-.2,hi[along]+.2);a.set_ylim(-.1,3.45)
        a.set_title(f"{item['name']} | {'paired' if len(faces)==2 else 'single face'}")
        a.grid(alpha=.15)
        xy=np.zeros((2,2));xy[:,axis]=np.mean(faces);xy[:,along]=[lo[along],hi[along]]
        ax.plot(xy[:,0],xy[:,1],lw=2);mid=xy.mean(0);ax.text(*mid,item['name'].replace('wall_','w').replace('parapet_','p'),fontsize=9,bbox={'facecolor':'white','alpha':.75,'pad':1})
    axes.flat[-1].axis('off');fig.savefig(OUT/'wall_elevation_diagnostics.png',dpi=130);plt.close(fig)
    ax.set_aspect('equal');ax.set_xlim(-10,6.6);ax.set_ylim(-4.5,6.8);ax.grid(alpha=.2);fig2.savefig(OUT/'wall_run_map.png',dpi=140);plt.close(fig2)


def infer_back_side(q,item,d,thickness):
    """Keep the measured front fixed; provisional back faces do not size rooms."""
    axis=item['axis'];along=1-axis;lo,hi=np.array(item['bounds'])
    p=q[(abs(q[:,2])<.055)&(q[:,along]>lo[along]+.08)&(q[:,along]<hi[along]-.08)]
    counts=[int((((p[:,axis]-d)*side>.15)&((p[:,axis]-d)*side<.60)).sum()) for side in (-1,1)]
    if max(counts)>max(100,3*min(counts)):
        back_sign=1 if counts[0]>counts[1] else -1
        method='away_from_better_observed_finish_floor'
    else:
        back_sign=1 if abs(d-lo[axis])<abs(d-hi[axis]) else -1
        method='retained_previous_wall_side_unverified'
    limits=sorted([d,d+back_sign*thickness])
    return limits,{'method':method,'floor_returns_negative_positive':counts,'thickness_verified':False}


def height_steps(p,along,a0,a1):
    """Piecewise horizontal wall heads; persistent height changes are retained."""
    cell=.10;edges=np.linspace(a0,a1,max(3,int(np.ceil((a1-a0)/cell))+1));xc=(edges[:-1]+edges[1:])/2
    ids=np.clip(np.searchsorted(edges,p[:,along])-1,0,len(xc)-1)
    top=np.full(len(xc),np.nan)
    for i in range(len(xc)):
        z=p[ids==i,2];z=z[z>.10]
        if len(z)>=20 and np.ptp(z)>.10:
            hist,e=np.histogram(z,np.arange(.1,3.56,.02));band=int(hist.argmax())
            if hist[band]/len(z)>.60:
                zz=z[(z>=e[band])&(z<e[band+1])];top[i]=np.percentile(zz,99)
            else:top[i]=np.percentile(z,99)
    good=np.isfinite(top)
    if good.sum()<2:raise ValueError('No supported wall height profile')
    top=np.interp(xc,xc[good],top[good]);top=ndimage.median_filter(top,size=5,mode='nearest')
    # Cluster stable head heights with a 90 mm separation, then use the median
    # within each run. This does not impose one storey height on parapets.
    sorted_values=np.sort(top);groups=np.split(sorted_values,np.flatnonzero(np.diff(sorted_values)>.09)+1)
    centres=np.array([np.median(g) for g in groups]);labels=np.argmin(abs(top[:,None]-centres),axis=1)
    runs=[];start=0
    for i in range(1,len(labels)+1):
        if i==len(labels) or labels[i]!=labels[start]:
            h=float(np.median(top[start:i]));runs.append([float(edges[start]),float(edges[i]),h]);start=i
    # A noisy 100–200 mm corner strip is not a genuine staircase-shaped head.
    while len(runs)>1 and runs[0][1]-runs[0][0]<.30:
        runs[1][0]=runs[0][0];runs.pop(0)
    while len(runs)>1 and runs[-1][1]-runs[-1][0]<.30:
        runs[-2][1]=runs[-1][1];runs.pop()
    return runs


def continuous_profile(p,axis,a0,a1,heads):
    along=1-axis;cell=.025
    outline=unary_union([box(a,0,b,h) for a,b,h in heads])
    origin=np.array([a0,0.]);size=np.ceil((np.array([a1,max(h for _,_,h in heads)])-origin)/cell).astype(int)+1
    mask=np.zeros(size,bool);uv=p[:,[along,2]];ij=np.floor((uv-origin)/cell).astype(int)
    ij=ij[np.all((ij>=0)&(ij<size),axis=1)];mask[ij[:,0],ij[:,1]]=True
    # Close sampling pores only for opening detection. The final wall uses
    # measured straight head lines, not the edges of the sampling mask.
    mask=ndimage.binary_closing(np.pad(mask,3),structure=np.ones((5,5)))[3:-3,3:-3]
    xx,zz=np.meshgrid(origin[0]+(np.arange(size[0])+.5)*cell,(np.arange(size[1])+.5)*cell,indexing='ij')
    domain=shapely.contains_xy(outline,xx,zz)
    void=domain&~mask;labels,count=ndimage.label(void);holes=[]
    for label in range(1,count+1):
        loc=np.argwhere(labels==label)
        if len(loc)*cell**2<.13:continue
        low=origin+loc.min(0)*cell;high=origin+(loc.max(0)+1)*cell
        w,h=high-low;rectangle_ratio=len(loc)*cell**2/(w*h)
        if w<.30 or h<.35 or rectangle_ratio<.73:continue
        # A strip along a wall head is not a window, nor is a lost scan corner.
        if h<.55 and high[1]>max(v[2] for v in heads)-.04:continue
        if low[0]<=a0+.03 or high[0]>=a1-.03:
            if rectangle_ratio<.9:continue
        inner=box(low[0]+.035,low[1]+.035,high[0]-.035,high[1]-.035)
        interior=uv[shapely.contains_xy(inner,uv[:,0],uv[:,1])]
        presence=len(np.unique(np.floor(interior/.025).astype(int),axis=0))*.025**2/max(inner.area,1e-6)
        if presence>.08:continue
        if low[1]<.10:low[1]=0.
        holes.append({'bounds_uz_m':[low.tolist(),high.tolist()],'void_rectangularity':float(rectangle_ratio),
                      'interior_occupied_fraction':presence,'classification':'opening_from_clear_elevation_void_not_semantic_door_label'})
        outline=outline.difference(box(*low,*high))
    return outline,holes


def extend_junctions(runs,max_gap=.28):
    changes=[]
    for run in runs:
        axis=run['axis'];along=1-axis
        for end in (0,1):
            value=run['along'][end];candidates=[]
            for other in runs:
                if other['axis']==axis:continue
                if max(run['cross'])<other['along'][0]-.10 or min(run['cross'])>other['along'][1]+.10:continue
                target=other['cross'][end]
                distance=min(abs(value-x) for x in other['cross'])
                if distance<=max_gap and ((end==0 and target<value) or (end==1 and target>value)):
                    candidates.append((distance,target,other['name']))
            if candidates:
                distance,target,other=min(candidates)
                if abs(target-value)<=max_gap+.35:
                    run['along'][end]=target
                    changes.append({'wall':run['name'],'end':end,'from_m':value,'to_m':target,'meets':other,'modeled_corner_closure':True})
    return changes


def build():
    OUT.mkdir(parents=True,exist_ok=True)
    if list(OUT.glob('Soulace*.skp')):raise FileExistsError('Use another revision')
    source,meta,matrix,q,records,fits=inputs();runs=[];audit=[]
    info={r['wall']:r for r in meta['walls']}
    for item in records:
        name=item['name']
        # The reviewed elevations identify a diagonal staircase and an overhead
        # thin rail/beam, not missing full-height walls. Preserve their originals
        # as reference, but do not turn them into invented room partitions.
        if name in ('wall_18','wall_20'):
            audit.append({'name':name,'status':'not_a_full_height_wall','reason':'diagonal_roof_stair' if name=='wall_18' else 'overhead_linear_evidence_only'})
            continue
        axis=item['axis'];along=1-axis;lo,hi=np.array(item['bounds'])
        if name in fits:
            cross=fits[name]['measured_faces_m'];faces=cross;thickness={'method':'paired_scan_faces','thickness_verified':True}
        else:
            faces=[item['modes'][0]['d']]
            cross,thickness=infer_back_side(q,item,faces[0],info[name]['thickness_mm']/1000)
        measured=wall_returns(q,item,faces)
        a0,a1=np.percentile(measured[:,along],[.1,99.9]);a0=max(a0,lo[along]-.12);a1=min(a1,hi[along]+.12)
        runs.append({'name':name,'axis':axis,'along':[float(a0),float(a1)],'cross':cross,'faces':faces,'thickness':thickness,'source':item})
    changes=extend_junctions(runs)
    parts=[];profiles=[];fig,axes=plt.subplots(6,4,figsize=(23,23),layout='constrained')
    for run,a in zip(runs,axes.flat):
        axis=run['axis'];along=1-axis;a0,a1=run['along'];dl,dh=run['cross']
        p=wall_returns(q,run['source'],run['faces'],pad=.35)
        p=p[(p[:,along]>=a0)&(p[:,along]<=a1)]
        heads=height_steps(p,along,a0,a1)
        # Include returns from depth-only recess backs when identifying holes.
        # Ignoring wall profiles must not turn a niche into a through-opening.
        local=q[(q[:,axis]>=dl-.035)&(q[:,axis]<=dh+.035)&(q[:,along]>=a0)&(q[:,along]<=a1)&(q[:,2]>=0)&(q[:,2]<=max(h[2] for h in heads)+.02)]
        profile,holes=continuous_profile(local,axis,a0,a1,heads)
        # Preserve existing door/window topology during wall repair. Scan frames,
        # leaves and recess returns must not silently shrink known openings.
        for opening in meta['openings']:
            if opening['wall']!=run['name'] or opening['kind'] not in ('door','window','arch','opening'):continue
            aa,bb=opening['along_m'];z0=opening['sill_mm']/1000;z1=opening['head_mm']/1000
            if opening['kind']=='door' and z0<.10:z0=0.
            cutter=box(aa,z0,bb,z1)
            profile=profile.difference(cutter)
            holes.append({'bounds_uz_m':[[aa,z0],[bb,z1]],'classification':'retained_prior_'+opening['kind'],
                          'dimensions_independently_verified':False})
        profile=shapely.set_precision(profile,.0001).buffer(.0002,join_style=2).buffer(-.0002,join_style=2)
        polygons=[profile] if profile.geom_type=='Polygon' else list(profile.geoms)
        meshes=[solid_from_profile(poly,axis,dl,dh) for poly in polygons if poly.area>.015]
        if not meshes or not all(m.is_watertight for m in meshes):raise ValueError('Invalid wall solid '+run['name'])
        mesh=trimesh.util.concatenate(meshes);common=(mesh.vertices+[0,0,6.5])@matrix[:3,:3].T+matrix[:3,3]
        verified=run['thickness']['thickness_verified'];name='L2 continuous '+run['name']+('' if verified else ' - thickness unverified')
        part={'name':name,'kind':'wall_solid_continuous','level':2,'colour':[205,198,181] if verified else [220,203,166],
              'v':np.round(common,7).tolist(),'f':mesh.faces.tolist(),'construction_solid':True,
              'thickness_verified':verified,'measured_face_positions_m':run['faces'],'modeled_thickness_m':dh-dl}
        parts.append(part);profiles.append({'run':{k:v for k,v in run.items() if k!='source'},'polygon_wkt':profile.wkt})
        row={'name':run['name'],'new_name':name,'status':'continuous_modeled_wall','thickness':run['thickness'],
             'thickness_mm':(dh-dl)*1000,'observed_faces_local_m':run['faces'],'cross_bounds_m':[dl,dh],
             'along_bounds_m':[a0,a1],'height_steps_m':heads,'openings':holes,'profile_area_m2':profile.area,
             'volume_m3':sum(abs(m.volume) for m in meshes),'watertight':True}
        audit.append(row)
        sample=p[::max(1,len(p)//9000)];a.scatter(sample[:,along],sample[:,2],s=.15,c='#477b91')
        for poly in polygons:
            xy=np.array(poly.exterior.coords);a.plot(xy[:,0],xy[:,1],c='#c85437',lw=1)
            for hole in poly.interiors:
                xy=np.array(hole.coords);a.plot(xy[:,0],xy[:,1],c='#238651',lw=1)
        a.set_title(run['name']);a.set_ylim(-.1,3.5);a.grid(alpha=.15)
        print(run['name'],'openings',len(holes),'heads',np.round(heads,3).tolist(),'thickness verified',verified,flush=True)
    for a in list(axes.flat)[len(runs):]:a.axis('off')
    fig.savefig(OUT/'wall_profile_scan_check.png',dpi=125);plt.close(fig)
    (OUT/'walls.build.json').write_text(json.dumps({'parts':parts},separators=(',',':')))
    (OUT/'wall_profiles.json').write_text(json.dumps(profiles,indent=2))
    (OUT/'closure_audit.json').write_text(json.dumps({'walls':audit,'junction_extensions':changes,
        'not_certified_accuracy':True,'method':'Observed wall faces, continuous stepped heads and rectangular elevation voids. Back thickness retained from prior model when not scanned. No zero-evidence new wall runs.'},indent=2))
    from recover_soulace_details import original_payload
    from render_scan_recovery import render
    from PIL import Image
    base=[p for p in original_payload()['parts'] if p['name'].startswith('L2_') and 'wall' not in p['kind'] and 'ceiling' not in p['kind']]
    roof=next(p for p in json.loads((PRIOR/'additions.build.json').read_text())['parts'] if p['kind']=='roof_stair_scan')
    for name,direction in [('3d',(1.3,-1.5,1.25)),('top',(0,0,1)),('rear',(-1.3,1.5,1.1))]:
        Image.fromarray(render(base+parts+[roof],direction=direction,w=1500,h=1100)).save(OUT/f'closure_preview_{name}.png')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--diagnostics',action='store_true');args=ap.parse_args()
    if args.diagnostics:diagnostics()
    else:build()
