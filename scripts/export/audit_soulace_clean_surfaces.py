"""Independent full-density nearest-return checks and multi-view scan overlays."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from clean_soulace_surfaces import ROOT,OUT,load_raw_detail
from render_scan_recovery import render
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def surface_probes(triangles,spacing=.035):
    probes=[];weights=[]
    for t in triangles:
        n=max(1,int(np.ceil(np.linalg.norm(t-np.roll(t,1,axis=0),axis=1).max()/spacing)))
        a,b=np.triu_indices(n+1);u=a/n;v=(b-a)/n
        q=t[0]+u[:,None]*(t[1]-t[0])+v[:,None]*(t[2]-t[0])
        area=np.linalg.norm(np.cross(t[1]-t[0],t[2]-t[0]))/2
        probes.append(q);weights.append(np.full(len(q),area/len(q)))
    return np.concatenate(probes),np.concatenate(weights)


def stats(d,w):
    order=np.argsort(d);cdf=np.cumsum(w[order]);cdf/=cdf[-1]
    return {'sample_count':len(d),'area_weighted_within_10_20_50mm_pct':[float(100*np.sum(w[d<=x])/w.sum()) for x in [.01,.02,.05]],
            'weighted_p50_p95_mm':(np.interp([.5,.95],cdf,d[order])*1000).tolist(),'max_sample_distance_mm':float(d.max()*1000)}


def main():
    raw=load_raw_detail();tree=cKDTree(raw,leafsize=32,compact_nodes=False)
    floor=json.loads((OUT/'floors.build.json').read_text())['parts'];stairs=json.loads((OUT/'upper_stairs.build.json').read_text())['parts']
    reports={};sets={}
    for category,parts in [('floor',floor),('upper_stairs',stairs)]:
        probes=[];distances=[];weights=[];part_audits=[]
        for part in parts:
            tri=np.asarray(part['v'])[np.asarray(part['f'])];q,w=surface_probes(tri);d=tree.query(q,workers=8)[0]
            probes.append(q);weights.append(w);distances.append(d);part_audits.append({'name':part['name'],**stats(d,w)})
        q=np.concatenate(probes);d=np.concatenate(distances);w=np.concatenate(weights)
        sets[category]=(q,d);reports[category]={**stats(d,w),'parts':part_audits}
        print(category,json.dumps({k:v for k,v in reports[category].items() if k!='parts'}),flush=True)
    reports['method']='35 mm maximum triangle-edge subdivision spacing; triangle-area weights; nearest 3D raw return from full clipped LAS (not Poisson); discrete diagnostic, not continuous-distance certificate.'
    reports['warning']='Plane fitting, regular outlines and small-hole filling interpolate between returns. >50 mm probes remain reported; no claim of literal scan overlap everywhere or surveyed absolute accuracy.'
    (OUT/'scan_proximity_audit.json').write_text(json.dumps(reports,indent=2))
    s=np.load(ROOT/'output_final/scan_first_diagnostics/soulace/sample.npz');p=s['p'];h=s['horizontal']
    stair_roi=(p[:,0]>-4.87)&(p[:,0]<-2.8)&(p[:,1]>1.3)&(p[:,1]<5.75)&(p[:,2]>3.17)&(p[:,2]<6.65)
    fig,ax=plt.subplots(1,3,figsize=(18,8),layout='constrained')
    for a,xlim,label in zip(ax[:2],[(-4.78,-4.05),(-3.75,-3.0)],['Outbound flight: side','Return flight: side']):
        sel=stair_roi&(p[:,0]>xlim[0])&(p[:,0]<xlim[1]);a.scatter(p[sel,1],p[sel,2],s=.1,c='#899299',alpha=.3,rasterized=True)
        sel&=h;a.scatter(p[sel,1],p[sel,2],s=1,c='#346981',alpha=.45,rasterized=True)
        q,d=sets['upper_stairs'];sel=(q[:,0]>xlim[0])&(q[:,0]<xlim[1]);a.scatter(q[sel,1],q[sel,2],s=.35,c='#d98224',alpha=.7,rasterized=True)
        a.set_title(label+'\nBlue/grey: scan; orange: fitted walking surface');a.set_xlabel('Y (m)');a.set_ylabel('Z (m)');a.set_aspect('equal');a.grid(alpha=.2)
    ax[2].imshow(render(stairs,direction=(1.6,-1.8,1.6),w=900,h=1100));ax[2].axis('off');ax[2].set_title('New measured staircase surface: 3D')
    fig.suptitle('Soulace upper staircase | top surface separated from underside and scan clutter',fontsize=17)
    fig.savefig(OUT/'upper_stairs_scan_overlay.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(17,8),layout='constrained');q,d=sets['floor']
    a=ax[0];im=a.scatter(q[:,0],q[:,1],c=np.minimum(d*1000,100),s=.45,cmap='turbo',vmin=0,vmax=100);a.set_aspect('equal');a.set_title('Planar floor → raw 3D return distance');fig.colorbar(im,ax=a,label='Nearest-return distance (mm)')
    a=ax[1];a.imshow(render(floor,direction=(1.3,-1.5,1.8),w=1100,h=900));a.axis('off');a.set_title('Clean planar patches; larger voids retained')
    fig.suptitle('Soulace floor cleanup | interpolation is measured and reported, not hidden',fontsize=17)
    fig.savefig(OUT/'floor_scan_overlay.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
