"""Fit editable top-storey wall solids, with measured paired faces and explicit QA.

Prior wall objects are hypotheses, never evidence. The full raw scan determines
face locations and supported elevations; unsupported templates are not restored.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
from scipy import ndimage
from scipy.spatial import cKDTree
import trimesh
import shapely
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from clean_soulace_surfaces import raster_polygon, polygon_parts, fill_small_holes
from build_soulace_asbuilt import extrude_along_cross

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output_final/soulace_top_floor_walls_v5'
REC=ROOT/'output_final/soulace_top_floor_recovery_v4'


def source_data():
    folder=ROOT/'output_final/soulace_asbuilt_v2'
    parts=json.loads((folder/'Soulace_L2_second_asbuilt.build.json').read_text())['parts']
    meta=json.loads((folder/'Soulace_L2_second_asbuilt_manifest.json').read_text())
    cfg=json.loads((ROOT/'output_final/coverage_restored_v1/soulace/restore_manifest.json').read_text())
    matrix=np.array(cfg['source_to_common']['2'])
    return parts,meta,matrix


def local_raw(matrix):
    cache=OUT/'local_raw.npy'
    if cache.exists():return np.load(cache,mmap_mode='r')
    p=np.load(REC/'top_raw.npy',mmap_mode='r')
    q=(p-matrix[:3,3])@matrix[:3,:3];q[:,2]-=6.5
    np.save(cache,q.astype(np.float32));return q


def face_modes(q,axis,lo,hi):
    along=1-axis
    p=q[(q[:,along]>lo[along]+.08)&(q[:,along]<hi[along]-.08)&
        (q[:,2]>lo[2]+.13)&(q[:,2]<hi[2]-.13)&
        (q[:,axis]>lo[axis]-.25)&(q[:,axis]<hi[axis]+.25)]
    hist,edges=np.histogram(p[:,axis],np.arange(lo[axis]-.251,hi[axis]+.257,.005))
    smooth=ndimage.gaussian_filter1d(hist.astype(float),1)
    peaks=find_peaks(smooth,prominence=max(25,smooth.max()*.025),distance=8)[0]
    records=[]
    for i in peaks:
        d=(edges[i]+edges[i+1])/2
        near=p[abs(p[:,axis]-d)<.025]
        if len(near)<100:continue
        d=float(np.median(near[:,axis]));res=abs(near[:,axis]-d)
        occupied=np.unique(np.floor(near[:,[along,2]]/.05).astype(int),axis=0)
        records.append({'d':d,'returns':len(near),'occupied_area_m2':len(occupied)*.0025,
                        'p95_plane_mm':float(np.percentile(res,95)*1000)})
    return sorted(records,key=lambda x:-x['occupied_area_m2'])


def inspect():
    OUT.mkdir(parents=True,exist_ok=True)
    parts,meta,matrix=source_data();q=local_raw(matrix);records=[]
    for info in meta['walls']:
        part=next(p for p in parts if p['name']==info['wall'])
        v=np.array(part['v']);lo=v.min(0);hi=v.max(0);axis=0 if info['axis']=='x' else 1
        modes=face_modes(q,axis,lo,hi)
        item={'name':part['name'],'axis':axis,'bounds':[lo.tolist(),hi.tolist()],
              'prior_kind':part['kind'],'modes':modes}
        records.append(item);print(json.dumps(item),flush=True)
    (OUT/'wall_plane_diagnostics.json').write_text(json.dumps(records,indent=2))


def silhouette(q, axis, d, bounds, cell=.025):
    along=1-axis;lo,hi=np.asarray(bounds)
    origin=np.floor(lo[[along,2]]/cell)*cell
    size=np.ceil((hi[[along,2]]-origin)/cell).astype(int)
    p=q[(abs(q[:,axis]-d)<.03)&(q[:,along]>=lo[along])&(q[:,along]<=hi[along])&
        (q[:,2]>=lo[2])&(q[:,2]<=hi[2])]
    mask=np.zeros(size,bool)
    if len(p):
        ij=np.floor((p[:,[along,2]]-origin)/cell).astype(int)
        good=np.all((ij>=0)&(ij<size),axis=1);ij=ij[good]
        mask[ij[:,0],ij[:,1]]=True
    return mask,origin,p


def clean_outline(mask,origin,cell=.025):
    # Close at most a 50 mm sampling gap. Preserve large holes and boundary gaps.
    padded=np.pad(mask,3)
    clean=ndimage.binary_closing(padded,structure=np.ones((3,3)))[3:-3,3:-3]
    labels,n=ndimage.label(clean)
    sizes=np.bincount(labels.ravel())*cell**2;keep=sizes>=.10;keep[0]=False
    shape=raster_polygon(keep[labels],origin,cell)
    shape=fill_small_holes(shape,.01)
    return shape.simplify(.018,preserve_topology=True)


def pair_choice(modes,masks):
    candidates=[]
    for i,a in enumerate(modes):
        for j,b in enumerate(modes[:i]):
            thickness=abs(a['d']-b['d'])
            if not .09<=thickness<=.35:continue
            shared=(ndimage.binary_dilation(masks[i],iterations=1)&
                    ndimage.binary_dilation(masks[j],iterations=1)).sum()*.025**2
            minimum=min(masks[i].sum(),masks[j].sum())*.025**2
            if minimum<.40 or shared<.25 or shared/minimum<.20:continue
            if max(a['p95_plane_mm'],b['p95_plane_mm'])>22.9:continue
            candidates.append((shared+.2*minimum,j,i))
    return max(candidates)[1:] if candidates else None


def solid_from_profile(poly,axis,dl,dh):
    # Constrained triangulation avoids earcut's occasional pinched-hole caps.
    faces=shapely.constrained_delaunay_triangles(poly)
    triangles=np.asarray([np.asarray(t.exterior.coords)[:3] for t in faces.geoms])
    vertices,ids=np.unique(np.round(triangles.reshape(-1,2),9),axis=0,return_inverse=True)
    mesh=trimesh.creation.extrude_triangulation(vertices,ids.reshape(-1,3),dh-dl)
    v=mesh.vertices.copy();target=np.empty_like(v)
    target[:,axis]=v[:,2]+dl;target[:,1-axis]=v[:,0];target[:,2]=v[:,1]
    mesh.vertices=target;mesh.fix_normals()
    return mesh


def make_models():
    OUT.mkdir(parents=True,exist_ok=True)
    if list(OUT.glob('*.skp')):raise FileExistsError('Use a new native revision')
    if not (OUT/'wall_plane_diagnostics.json').exists():inspect()
    source,meta,matrix=source_data();q=local_raw(matrix)
    records=json.loads((OUT/'wall_plane_diagnostics.json').read_text());parts=[];audit=[]
    for item in records:
        modes=item['modes'];axis=item['axis'];along=1-axis;bounds=item['bounds'];lo,hi=np.array(bounds)
        masks=[];points=[]
        for mode in modes:
            m,origin,p=silhouette(q,axis,mode['d'],bounds);masks.append(m);points.append(p)
        pair=pair_choice(modes,masks)
        if pair is None:
            audit.append({'name':item['name'],'status':'not_replaced_no_confirmed_face_pair','modes':modes})
            continue
        a,b=sorted(pair,key=lambda k:modes[k]['d']);dl,dh=modes[a]['d'],modes[b]['d']
        combined=masks[a]|masks[b]
        shape=clean_outline(combined,origin)
        # A detected prior opening is a hypothesis. Preserve it only where its
        # interior has almost no returns on either measured wall face.
        retained_openings=[]
        for opening in meta['openings']:
            if opening['wall']!=item['name']:continue
            x0,x1=opening['along_m'];z0=opening['sill_mm']/1000;z1=opening['head_mm']/1000
            inside=box(x0+.05,z0+.05,x1-.05,z1-.05)
            uv=np.concatenate([points[a][:,[along,2]],points[b][:,[along,2]]])
            area=max(inside.area,1e-9)
            occupied=np.unique(np.floor(uv[shapely.contains_xy(inside,uv[:,0],uv[:,1])]/.025).astype(int),axis=0)
            support=len(occupied)*.025**2/area
            if support<.10:
                shape=shape.difference(box(x0,z0,x1,z1));retained_openings.append({**opening,'interior_support_fraction':support})
        meshes=[]
        for poly in polygon_parts(shape):
            if poly.area>=.12:
                mesh=solid_from_profile(poly,axis,dl,dh)
                if not mesh.is_watertight:raise ValueError('Non-solid reconstructed '+item['name'])
                meshes.append(mesh)
        if not meshes:continue
        # One wall assembly may contain disconnected lintel/panel solids.
        mesh=trimesh.util.concatenate(meshes)
        common=(mesh.vertices+np.array([0,0,6.5]))@matrix[:3,:3].T+matrix[:3,3]
        name='L2 modeled '+item['name']
        parts.append({'name':name,'kind':'wall_solid_refit','level':2,'colour':[205,198,181],
                      'v':np.round(common,7).tolist(),'f':mesh.faces.tolist(),
                      'scene_levels':[2],'construction_solid':True})
        # Area-uniform probes on the two large wall faces, excluding inferred
        # thickness closures. This is sampled QA, not a continuous certificate.
        probes=[]
        for d in [dl,dh]:
            rng=np.random.default_rng(402);n=max(1000,int(shape.area/.0004))
            xy=rng.uniform(shape.bounds[:2],shape.bounds[2:],(n*3,2))
            xy=xy[shapely.contains_xy(shape,xy[:,0],xy[:,1])][:n]
            p=np.empty((len(xy),3));p[:,axis]=d;p[:,[along,2]]=xy;probes.append(p)
        local=np.concatenate(points)
        distance=cKDTree(local).query(np.concatenate(probes),workers=8)[0]
        row={'name':item['name'],'new_name':name,'status':'paired_face_wall_solid','measured_faces_m':[dl,dh],
             'thickness_mm':(dh-dl)*1000,'paired_modes':[modes[a],modes[b]],
             'solid_components':len(meshes),'watertight':all(m.is_watertight for m in meshes),
             'profile_area_m2':shape.area,'volume_m3':sum(abs(m.volume) for m in meshes),
             'sampled_face_within_50mm_pct':float(np.mean(distance<=.05)*100),
             'sampled_face_p95_mm':float(np.percentile(distance,95)*1000),
             'preserved_openings':retained_openings,
             'closure_note':'End caps and concealed opposite-face regions close the measured thickness; not directly observed everywhere.'}
        audit.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ['paired_modes','preserved_openings']}),flush=True)
    (OUT/'walls.build.json').write_text(json.dumps({'parts':parts},separators=(',',':')))
    (OUT/'wall_fit_audit.json').write_text(json.dumps({'walls':audit,'source':'Full raw LAS in common frame',
        'limitations':'Paired-plane reconstruction is not proof of wall identity or certified site accuracy. Small sampling gaps up to 50 mm are interpolated; end caps are modeled. Unpaired walls retain their prior evidence, without guessed thickness.'},indent=2))
    from recover_soulace_details import original_payload
    from render_scan_recovery import render
    from PIL import Image
    replaced={'L2_'+a['name'] for a in audit if a['status']=='paired_face_wall_solid'}
    base=[p for p in original_payload()['parts'] if p['name'].startswith('L2_') and p['name'] not in replaced and 'ceiling' not in p['kind']]
    roof=next(p for p in json.loads((REC/'recovered_parts.build.json').read_text())['parts'] if p['kind']=='roof_stair_reference')
    Image.fromarray(render(base+parts+[roof],direction=(1.3,-1.5,1.25),w=1600,h=1200)).save(OUT/'modeled_walls_preview.png')
    print('Modeled wall assemblies',len(parts),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--inspect',action='store_true');args=ap.parse_args()
    if args.inspect:inspect()
    else:make_models()
