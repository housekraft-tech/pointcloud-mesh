"""Diagnostic only: dominant ground planes and scan-height map."""
import json
from pathlib import Path
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output_final/soulace_clean_floor_stairs_v2'

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    s=np.load(ROOT/'output_final/scan_first_diagnostics/soulace/sample.npz')
    p=s['p'];p=p[s['horizontal']&(p[:,2]>-1.5)&(p[:,2]<.1)&(p[:,0]<7.6)]
    ids=np.arange(len(p));labels=np.full(len(p),-1);planes=[]
    o3d.utility.random.seed(43)
    for k in range(24):
        cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p[ids]))
        plane,ii=cloud.segment_plane(.015,3,800)
        ii=np.asarray(ii);plane=np.asarray(plane)
        if len(ii)<650:break
        labels[ids[ii]]=k
        plane=plane/(-plane[2]);planes.append(plane.tolist())
        print(k,len(ii),'z =',plane[[0,1,3]].tolist(),'xy',np.percentile(p[ids[ii],:2],[1,99],axis=0).tolist(),flush=True)
        ids=np.delete(ids,ii)
    np.savez_compressed(OUT/'floor_planes_diagnostic.npz',p=p,labels=labels,planes=np.asarray(planes))
    fig,ax=plt.subplots(1,2,figsize=(17,8),layout='constrained')
    im=ax[0].scatter(p[:,0],p[:,1],s=.3,c=p[:,2],cmap='turbo',vmin=-.65,vmax=.02);fig.colorbar(im,ax=ax[0],label='Common-frame elevation (m)')
    ax[1].scatter(p[:,0],p[:,1],s=.4,c=labels,cmap='tab20',vmin=0,vmax=20)
    for k in range(len(planes)):
        q=p[labels==k];c=np.median(q,axis=0);ax[1].text(c[0],c[1],str(k),fontsize=12,bbox=dict(facecolor='white',alpha=.8))
    for a in ax:a.set_aspect('equal');a.grid(alpha=.2)
    fig.savefig(OUT/'floor_planes_diagnostic.png',dpi=150);plt.close(fig)

if __name__=='__main__':main()
