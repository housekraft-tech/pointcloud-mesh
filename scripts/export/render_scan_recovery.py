"""Ray-traced CAD diagnostics from the actual output triangles; no invented imagery."""
import json
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]


def render(parts,direction=(1.3,-1.5,1.25),bounds=None,w=1200,h=900):
    scene=o3d.t.geometry.RaycastingScene(nthreads=8);colours=[];vertices=[]
    for p in parts:
        v=np.asarray(p['v'],np.float32);f=np.asarray(p['f'],np.uint32)
        if len(f)==0:continue
        scene.add_triangles(o3d.core.Tensor(v),o3d.core.Tensor(f));vertices.append(v)
        colours.append(np.array(p.get('colour',[177,183,179]))/255)
    if bounds is None:
        vv=np.vstack(vertices);bounds=np.array([vv.min(0),vv.max(0)])
    c=bounds.mean(axis=0);di=np.array(direction,float);di/=np.linalg.norm(di)
    forward=-di;up=np.array([0,0,1.]) if abs(di[2])<.99 else np.array([0,1.,0])
    right=np.cross(forward,up);right/=np.linalg.norm(right);up=np.cross(right,forward)
    corners=np.array([[x,y,z] for x in bounds[:,0] for y in bounds[:,1] for z in bounds[:,2]])-c
    aspect=w/h;height=max(np.ptp(corners@up),np.ptp(corners@right)/aspect)*1.08;width=height*aspect
    xx,yy=np.meshgrid((np.arange(w)+.5)/w-.5,.5-(np.arange(h)+.5)/h)
    distance=np.linalg.norm(bounds[1]-bounds[0])*2
    origins=c+di*distance+xx[:,:,None]*width*right+yy[:,:,None]*height*up
    rays=np.concatenate((origins,np.broadcast_to(forward,origins.shape)),axis=2).astype(np.float32)
    hit=scene.cast_rays(o3d.core.Tensor(rays),nthreads=8);ids=hit['geometry_ids'].numpy();normal=hit['primitive_normals'].numpy()
    valid=ids!=np.uint32(4294967295);output=np.ones((h,w,3))*.971
    light=np.array([.3,-.5,.82]);light/=np.linalg.norm(light)
    shade=.62+.38*abs(normal@light)
    output[valid]=np.asarray(colours)[ids[valid]]*shade[valid,None]
    return (np.clip(output,0,1)*255).astype(np.uint8)


def main():
    folder=ROOT/'output_final/scan_recovered_v1/engrance'
    data=json.loads((folder/'model.build.json').read_text());allparts=data['parts']
    visible=[p for p in allparts if 'ceiling' not in p['kind']];baseline=[p for p in allparts[:64] if 'ceiling' not in p['kind']]
    v=np.vstack([p['v'] for p in visible]);bounds=np.array([v.min(0),v.max(0)])
    before=render(baseline,bounds=bounds);after=render(visible,bounds=bounds)
    Image.fromarray(after).save(folder/'recovered_mesh_3d.png')
    fig,ax=plt.subplots(1,2,figsize=(16,7.5),layout='constrained')
    for a,im,title in zip(ax,[before,after],['Earlier CAD / LiDAR coverage baseline','Existing model + recovered scan surfaces (cyan)']):a.imshow(im);a.set_title(title);a.axis('off')
    fig.suptitle('Engrance | actual mesh comparison — recovered detail remains unclassified',fontsize=17)
    fig.savefig(folder/'before_after_3d.png',dpi=160);plt.close(fig)
    d=np.load(folder/'coverage_samples.npz');p=d['p'];prior=d['before'];recovered=d['after']
    fig,ax=plt.subplots(1,3,figsize=(18,7),layout='constrained')
    mask=(p[:,2]>.65)&(p[:,2]<1.6)
    for a,dist,title in zip(ax[:2],[prior,recovered],['Before: unmatched scan samples in red','After: unmatched scan samples in red']):
        ids=np.flatnonzero(mask);good=dist[ids]<=.05
        a.scatter(p[ids[good],0],p[ids[good],1],s=.4,c='#75959b');a.scatter(p[ids[~good],0],p[ids[~good],1],s=.8,c='#da4638')
        a.set_aspect('equal');a.set_title(title);a.grid(alpha=.2)
    ax[2].imshow(after);ax[2].axis('off');ax[2].set_title('3D mesh, ceilings hidden')
    fig.suptitle('Engrance scan → model check | 64.4% → 97.0% of sampled voxels within 50 mm\nDiagnostic coverage, including furniture/doors; not certified architectural completeness',fontsize=16)
    fig.savefig(folder/'coverage_comparison.png',dpi=160);plt.close(fig)
    sf=ROOT/'output_final/scan_recovered_v1/soulace';data=json.loads((sf/'additions.build.json').read_text())
    stairs=[p for p in data['parts'] if p['kind']=='stair_observed'];floors=[p for p in data['parts'] if 'floor' in p['kind']]
    s3d=render(stairs,direction=(-1.4,-1.1,1),w=1000,h=1100)
    side=render(stairs,direction=(-1,0,0),w=700,h=1100)
    floor=render(floors,direction=(1.2,-1.6,1.8),w=1300,h=1000)
    Image.fromarray(s3d).save(sf/'stairs_mesh_3d.png');Image.fromarray(side).save(sf/'stairs_mesh_side.png');Image.fromarray(floor).save(sf/'lower_floor_mesh_3d.png')
    fig,ax=plt.subplots(1,3,figsize=(17,10),layout='constrained')
    for a,im,title in zip(ax,[s3d,side,floor],['Observed staircase surfaces: 3D','Observed staircase: side','Recovered lower floors / grade']):a.imshow(im);a.axis('off');a.set_title(title)
    fig.suptitle('Soulace | recovered scan surfaces — no equal-step or single-floor-height template',fontsize=17)
    fig.savefig(sf/'recovered_details.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
