"""Planar ground patches and measured top-surface stairs, with explicit evidence QA.

This is a revision adapter for Soulace; numerical fitting/footprint helpers are
reusable. It does not assert that unseen floor areas or absolute accuracy are
known. Original files are never overwritten.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.stats import binned_statistic_2d
from scipy.signal import find_peaks
import shapely
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recover_mesh_evidence import mesh_part

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output_final/soulace_clean_floor_stairs_v2'


def robust_plane(p,cutoff=.025):
    """Robust z=a*x+b*y+c; no flattening of measured drainage fall."""
    a=np.column_stack((p[:,:2],np.ones(len(p))))
    coeff=np.linalg.lstsq(a,p[:,2],rcond=None)[0]
    for _ in range(12):
        err=p[:,2]-a@coeff
        scale=max(.002,1.4826*np.median(abs(err-np.median(err))))
        w=np.minimum(1,1.5*scale/np.maximum(abs(err),1e-9))
        coeff=np.linalg.lstsq(a*w[:,None],p[:,2]*w,rcond=None)[0]
    good=abs(p[:,2]-a@coeff)<cutoff
    if good.sum()>=12:coeff=np.linalg.lstsq(a[good],p[good,2],rcond=None)[0]
    return coeff


def polygon_parts(g):
    if g.is_empty:return []
    if g.geom_type=='Polygon':return [g]
    return [p for x in g.geoms for p in polygon_parts(x)] if hasattr(g,'geoms') else []


def fill_small_holes(g,max_area=.035):
    """Only enclosed small sampling holes, never the outside or a large opening."""
    return unary_union([Polygon(p.exterior,[r for r in p.interiors if Polygon(r).area>max_area]) for p in polygon_parts(g)])


def raster_polygon(mask,origin,cell):
    """Merge horizontal runs, not one polygon per scan point."""
    runs=[]
    for row in range(mask.shape[1]):
        q=np.pad(mask[:,row].astype(np.int8),(1,1));starts=np.flatnonzero(np.diff(q)==1);ends=np.flatnonzero(np.diff(q)==-1)
        runs.extend(box(origin[0]+a*cell,origin[1]+row*cell,origin[0]+b*cell,origin[1]+(row+1)*cell) for a,b in zip(starts,ends))
    return unary_union(runs)


def polygon_triangles(g,plane):
    result=[]
    for poly in polygon_parts(shapely.make_valid(g)):
        if poly.area<.0025:continue
        triangulation=shapely.constrained_delaunay_triangles(poly)
        for face in triangulation.geoms:
            xy=np.asarray(face.exterior.coords)[:3]
            z=xy@plane[:2]+plane[2]
            t=np.column_stack((xy,z))
            if np.cross(t[1]-t[0],t[2]-t[0])[2]<0:t=t[::-1]
            result.append(t)
    return np.asarray(result).reshape(-1,3,3)


def height_components(p,cell=.04,jump=.055):
    """Join observed neighboring cells only if they do not cross a height step."""
    origin=np.floor(p[:,:2].min(0)/cell)*cell-cell
    ij=np.floor((p[:,:2]-origin)/cell).astype(int);shape=ij.max(0)+2
    edges=[origin[k]+np.arange(shape[k]+1)*cell for k in range(2)]
    h=binned_statistic_2d(p[:,0],p[:,1],p[:,2],statistic='median',bins=edges)[0]
    # Sparse graph rather than label() because adjacent cells can be different levels.
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    occupied=np.isfinite(h);ids=np.full(h.shape,-1,int);ids[occupied]=np.arange(occupied.sum())
    aa=[];bb=[]
    for axis in (0,1):
        a=[slice(None)]*2;b=a.copy();a[axis]=slice(None,-1);b[axis]=slice(1,None);a=tuple(a);b=tuple(b)
        good=occupied[a]&occupied[b]&(abs(h[a]-h[b])<jump)
        aa.extend(ids[a][good]);bb.extend(ids[b][good])
    graph=coo_matrix((np.ones(len(aa)),(aa,bb)),shape=(occupied.sum(),occupied.sum()))
    _,labs=connected_components(graph,directed=False);grid=np.full(h.shape,-1,int);grid[occupied]=labs
    return grid[ij[:,0],ij[:,1]],grid,h,origin,ij


def floor_components_diagnostic():
    s=np.load(ROOT/'output_final/scan_first_diagnostics/soulace/sample.npz');p=s['p']
    p=p[s['horizontal']&(p[:,2]>-1.5)&(p[:,2]<.10)&(p[:,0]<7.6)]
    lab,grid,h,origin,ij=height_components(p)
    keys,counts=np.unique(lab,return_counts=True);order=keys[np.argsort(counts)[::-1]]
    fig,ax=plt.subplots(figsize=(13,10),layout='constrained');records=[]
    for i,k in enumerate(order[:35]):
        q=p[lab==k]
        if len(q)<200:continue
        pl=robust_plane(q);d=abs(q[:,2]-(q[:,:2]@pl[:2]+pl[2]))
        records.append({'id':int(k),'points':len(q),'plane':pl.tolist(),'p95_mm':float(np.percentile(d,95)*1000),'xy':np.percentile(q[:,:2],[1,99],axis=0).tolist()})
        ax.scatter(q[:,0],q[:,1],s=.5,color=plt.cm.tab20(i%20));c=np.median(q,axis=0);ax.text(c[0],c[1],str(k),bbox=dict(facecolor='white',alpha=.8))
    ax.set_aspect('equal');ax.grid(alpha=.2);fig.savefig(OUT/'floor_components.png',dpi=130);plt.close(fig)
    (OUT/'floor_components.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2),flush=True)


def load_raw_detail():
    cache=OUT/'raw_detail.npy'
    if cache.exists():return np.load(cache,mmap_mode='r')
    import laspy
    s=np.load(ROOT/'output_final/scan_first_diagnostics/soulace/sample.npz');matrix=s['matrix'];kept=[]
    path=ROOT/'data/Soulace/clip_texture_optimize_optimised_2026-08-20_12-07-54_514-003.las'
    with laspy.open(path) as reader:
        for block in reader.chunk_iterator(2_000_000):
            q=np.column_stack((block.x,block.y,block.z))@matrix[:3,:3].T+matrix[:3,3]
            ground=(q[:,2]>-1.65)&(q[:,2]<.25)&(q[:,0]<7.75)
            stairs=(q[:,0]>-4.95)&(q[:,0]<-2.55)&(q[:,1]>1.2)&(q[:,1]<5.85)&(q[:,2]>3.1)&(q[:,2]<6.75)
            kept.append(q[ground|stairs].astype(np.float32))
    raw=np.concatenate(kept);np.save(cache,raw);print('Cached full-density ground + upper stairs:',len(raw),flush=True);return raw


def quad(v):
    v=np.asarray(v,float);return v[[[0,1,2],[0,2,3]]]


def fit_riser(raw,xlim,approx_y,zlo,zhi):
    """Locate a riser from its vertical interior, away from nosings/undersides."""
    p=raw[(raw[:,0]>xlim[0]+.09)&(raw[:,0]<xlim[1]-.09)&(abs(raw[:,1]-approx_y)<.105)&(raw[:,2]>zlo+.025)&(raw[:,2]<zhi-.025)]
    if len(p)<40:raise ValueError(f'Insufficient riser evidence at {approx_y}: {len(p)}')
    hist,e=np.histogram(p[:,1],np.arange(approx_y-.11,approx_y+.113,.003));i=hist.argmax();mode=(e[i]+e[i+1])/2
    p=p[abs(p[:,1]-mode)<.018];y=float(np.median(p[:,1]))
    return y,{'points':len(p),'y_m':y,'p95_residual_mm':float(np.percentile(abs(p[:,1]-y),95)*1000)}


def fit_upper_stairs(raw,sample):
    """Seeds select observed top-surface modes; no regular-riser dimensions imposed."""
    p=sample['p'];p=p[sample['horizontal']]
    mids=p[(p[:,0]>-4.79)&(p[:,0]<-2.96)&(p[:,1]>4.45)&(p[:,1]<5.65)&(abs(p[:,2]-5.097)<.02)]
    tops=p[(p[:,0]>-4.79)&(p[:,0]<-2.96)&(p[:,1]>1.4)&(p[:,1]<2.35)&(abs(p[:,2]-6.5526)<.02)]
    mid_z=float(np.median(mids[:,2]));top_z=float(np.median(tops[:,2]))
    # Reviewed local height modes, not assumed tread heights. Each is independently refit.
    specs=[('Upper outbound',(-4.85,-3.90),1,3.2041,
            [(1.90,3.397),(2.16,3.557),(2.42,3.732),(2.67,3.902),(2.93,4.067),(3.18,4.242),(3.42,4.417),(3.67,4.587),(3.92,4.757),(4.17,4.927)]),
           ('Upper return',(-3.84,-2.88),-1,mid_z,
            [(4.29,5.262),(4.03,5.427),(3.78,5.592),(3.52,5.752),(3.26,5.917),(3.01,6.082),(2.77,6.247),(2.53,6.417)])]
    parts=[];audit=[];flight_details=[]
    for name,xlim,direction,start_z,seeds in specs:
        heights=[];support=[];widths=[]
        for yc,zc in seeds:
            h=p[(p[:,0]>xlim[0])&(p[:,0]<xlim[1])&(abs(p[:,1]-yc)<.17)&(abs(p[:,2]-zc)<.025)]
            z=float(np.median(h[:,2]));heights.append(z);support.append(h)
            widths.append(np.percentile(h[:,0],[.5,99.5]))
        xl,xh=np.median(widths,axis=0)+np.array([-.012,.012])
        risers=[];riser_reports=[]
        for i,((yc,_),z,h) in enumerate(zip(seeds,heights,support)):
            edge=float(np.percentile(h[:,1],1 if direction>0 else 99))-direction*.018
            y,qa=fit_riser(raw,(xl,xh),edge,start_z if i==0 else heights[i-1],z)
            risers.append(y);riser_reports.append(qa)
        end_z=mid_z if direction>0 else top_z
        # Last measured vertical face joins the actual landing, not a crop-height seam.
        end_guess=seeds[-1][0]+direction*.14
        end_y,qa=fit_riser(raw,(xl,xh),end_guess,heights[-1],end_z);risers.append(end_y);riser_reports.append(qa)
        tread_tri=[];riser_tri=[]
        for i,z in enumerate(heights):
            ya,yb=sorted(risers[i:i+2]);t=quad([[xl,ya,z],[xh,ya,z],[xh,yb,z],[xl,yb,z]]);tread_tri.extend(t)
            y=risers[i];z0=start_z if i==0 else heights[i-1]
            riser_tri.extend(quad([[xl,y,z0],[xh,y,z0],[xh,y,z],[xl,y,z]]))
        y=risers[-1];riser_tri.extend(quad([[xl,y,heights[-1]],[xh,y,heights[-1]],[xh,y,end_z],[xl,y,end_z]]))
        for tri,label in [(tread_tri,'treads'),(riser_tri,'risers')]:
            part=mesh_part(np.asarray(tri),name+' - measured '+label,'stair_clean',[219,149,76] if label=='treads' else [197,123,61]);part['level']=1;parts.append(part)
        record={'name':name,'top_surface_seed_modes_m':seeds,'x_bounds_m':[float(xl),float(xh)],'tread_heights_m':heights,'riser_y_m':risers,
                'riser_evidence':riser_reports,'rises_mm':(np.diff([start_z]+heights+[end_z])*1000).tolist(),
                'goings_mm':(abs(np.diff(risers))*1000).tolist(),'equal_dimensions_imposed':False,
                'tread_horizontal_p95_mm':[float(np.percentile(abs(h[:,2]-z),95)*1000) for h,z in zip(support,heights)]}
        audit.append(record);flight_details.append((xl,xh,risers,heights))
    left,right=flight_details
    # L-shaped half-landing joins the two independently measured first/last risers.
    landing_pts=p[(p[:,0]>left[0]-.025)&(p[:,0]<right[1]+.025)&(p[:,1]>4.3)&(p[:,1]<5.75)&(abs(p[:,2]-5.097)<.025)]
    landing_z=mid_z;far=float(np.percentile(landing_pts[:,1],99.5))+.012
    mid=(left[1]+right[0])/2
    landing=Polygon([(left[0],left[2][-1]),(mid,left[2][-1]),(mid,right[2][0]),(right[1],right[2][0]),(right[1],far),(left[0],far)])
    part=mesh_part(polygon_triangles(landing,[0,0,landing_z]),'Upper intermediate landing - measured plane','stair_clean',[224,156,85]);part['level']=1;parts.append(part)
    # Top landing is restricted to observed horizontal support, including its rear outline.
    top=p[(p[:,0]>left[0]-.025)&(p[:,0]<right[1]+.025)&(p[:,1]>1.35)&(p[:,1]<right[2][-1]+.02)&(abs(p[:,2]-6.5526)<.02)]
    lo_y=float(np.percentile(top[:,1],.5))
    top_outline=box(left[0],lo_y,right[1],right[2][-1]);part=mesh_part(polygon_triangles(top_outline,[0,0,top_z]),'Upper floor arrival landing - measured plane','stair_clean',[224,156,85]);part['level']=2;parts.append(part)
    audit.append({'intermediate_landing_z_m':landing_z,'intermediate_far_y_m':far,'top_landing_z_m':top_z,'top_landing_near_y_m':lo_y})
    return parts,audit


def split_planar_regions(points,bounds,depth=0):
    """Subdivide only nonplanar grade; never force a multi-level surface into one plane."""
    plane=robust_plane(points);residual=abs(points[:,2]-(points[:,:2]@plane[:2]+plane[2]))
    if np.percentile(residual,95)<.024 or depth>=5 or len(points)<500:
        return [(points,bounds,plane)]
    cost=np.mean(np.minimum(residual,.07)**2);best=None
    for axis in (0,1):
        if np.ptp(points[:,axis])<1.0:continue
        for split in np.unique(np.quantile(points[:,axis],[.2,.4,.6,.8])):
            masks=[points[:,axis]<split,points[:,axis]>=split]
            if min(m.sum() for m in masks)<180:continue
            score=0
            for m in masks:
                q=points[m];a=robust_plane(q);d=abs(q[:,2]-(q[:,:2]@a[:2]+a[2]));score+=np.minimum(d,.07).dot(np.minimum(d,.07))/len(points)
            if best is None or score<best[0]:best=(score,axis,split,masks)
    if best is None or best[0]>cost*.80:return [(points,bounds,plane)]
    _,axis,split,masks=best;result=[]
    for i,m in enumerate(masks):
        b=bounds.copy();b[1-i,axis]=split
        result.extend(split_planar_regions(points[m],b,depth+1))
    return result


def fit_floors(sample):
    """Clean the previous cyan footprint, preserving its large openings and all old CAD."""
    old=np.load(ROOT/'output_final/scan_recovered_v1/soulace/detail_triangles.npz')['floor']
    # Work from the already scan-supported footprint. The small morphological repair
    # is deliberate interpolation, documented separately from direct scan support.
    polygons=shapely.polygons(old[:,:,:2])
    valid=shapely.area(polygons)>1e-10
    domain=shapely.union_all(polygons[valid])
    old_area=domain.area;old_holes=sum(len(p.interiors) for p in polygon_parts(domain))
    domain=domain.buffer(.045,join_style=2).buffer(-.045,join_style=2)
    domain=fill_small_holes(domain,.04).simplify(.022,preserve_topology=True)
    p=sample['p'];p=p[sample['horizontal']&(p[:,2]>-1.5)&(p[:,2]<.1)&(p[:,0]<7.6)]
    lab,grid,h,origin,ij=height_components(p)
    keys,counts=np.unique(lab,return_counts=True);order=keys[np.argsort(counts)[::-1]]
    parts=[];audit=[];used=Polygon()
    for k in order:
        q=p[lab==k]
        if len(q)<100:continue
        bounds=np.array([q[:,:2].min(0)-.021,q[:,:2].max(0)+.021])
        regions=split_planar_regions(q,bounds)
        for j,(q,b,plane) in enumerate(regions):
            inds=np.floor((q[:,:2]-origin)/.04).astype(int);mask=np.zeros(grid.shape,bool);mask[inds[:,0],inds[:,1]]=True
            # Keep existing occupied cells and close only short missing runs.
            mask|=ndimage.binary_closing(mask,structure=np.ones((5,5)))
            poly=raster_polygon(mask,origin,.04)
            poly=fill_small_holes(poly,.04).simplify(.022,preserve_topology=True)
            poly=poly.intersection(box(b[0,0],b[0,1],b[1,0],b[1,1])).intersection(domain).difference(used)
            poly=fill_small_holes(poly,.04)
            # Remove only isolated specks, not room-scale evidence.
            poly=unary_union([g for g in polygon_parts(poly) if g.area>.012])
            if poly.is_empty or poly.area<.025:continue
            tri=polygon_triangles(poly,plane)
            if not len(tri):continue
            part=mesh_part(tri,f'Planar ground patch {len(parts)+1:02d}','floor_clean',[85,161,187]);parts.append(part)
            used=used.union(poly)
            residual=abs(q[:,2]-(q[:,:2]@plane[:2]+plane[2]))
            audit.append({'name':part['name'],'source_component':int(k),'samples':len(q),'plane_z_ax_by_c':plane.tolist(),
                          'area_xy_m2':poly.area,'holes_remaining':sum(len(x.interiors) for x in polygon_parts(poly)),
                          'fit_p50_p95_mm':(np.percentile(residual,[50,95])*1000).tolist(),
                          'xy_bounds_m':list(poly.bounds)})
    report={'old_cyan_xy_area_m2':old_area,'old_cyan_hole_count':old_holes,'new_cyan_xy_area_m2':used.area,
            'new_cyan_hole_count':sum(len(p.interiors) for p in polygon_parts(used)),
            'original_footprint_covered_pct':100*used.intersection(shapely.union_all(polygons[valid])).area/old_area,
            'repair_method':'Planar fitting by connected height zone; 45 mm boundary closing, 40 mm grid with short-run closing; enclosed holes <=0.04 m2 repaired; 22 mm outline simplification.',
            'small_gap_repair_is_interpolation':True,'large_unknown_openings_kept':True,'patches':audit}
    return parts,report


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    ap=argparse.ArgumentParser();ap.add_argument('--diagnostic',action='store_true');ap.add_argument('--stairs',action='store_true');ap.add_argument('--floors',action='store_true');args=ap.parse_args()
    if args.diagnostic:floor_components_diagnostic()
    if args.stairs:
        from render_scan_recovery import render
        from PIL import Image
        raw=load_raw_detail();s=np.load(ROOT/'output_final/scan_first_diagnostics/soulace/sample.npz')
        parts,audit=fit_upper_stairs(raw,s)
        (OUT/'upper_stairs.build.json').write_text(json.dumps({'parts':parts},separators=(',',':')))
        (OUT/'upper_stairs_fit.json').write_text(json.dumps(audit,indent=2))
        Image.fromarray(render(parts,direction=(-1.4,-1.4,1.0),w=1200,h=1200)).save(OUT/'upper_stairs_preview.png')
        print(json.dumps(audit,indent=2),flush=True)
    if args.floors:
        from render_scan_recovery import render
        from PIL import Image
        sample=np.load(ROOT/'output_final/scan_first_diagnostics/soulace/sample.npz');parts,audit=fit_floors(sample)
        (OUT/'floors.build.json').write_text(json.dumps({'parts':parts},separators=(',',':')))
        (OUT/'floor_fit.json').write_text(json.dumps(audit,indent=2))
        for name,d in [('top',(0,0,1)),('3d',(1.3,-1.5,1.5))]:
            Image.fromarray(render(parts,direction=d,w=1600,h=1100)).save(OUT/f'floors_preview_{name}.png')
        print(json.dumps({k:v for k,v in audit.items() if k!='patches'},indent=2),flush=True)
