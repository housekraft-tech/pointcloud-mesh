"""Reconstruct the crosswise roof stair, separating top and underside returns."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from model_soulace_top_walls import solid_from_profile
from scipy import ndimage

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output_final/soulace_top_floor_closed_v6'
RAW=ROOT/'output_final/soulace_top_floor_recovery_v4'


def points():
    cache=OUT/'roof_normal_points.npz'
    if cache.exists():
        a=np.load(cache);return a['p'],a['n'],a['flat']
    raw=np.load(RAW/'top_raw.npy',mmap_mode='r')
    p=raw[(raw[:,0]>-4.4)&(raw[:,0]<.5)&(raw[:,1]>-.55)&(raw[:,1]<.25)&(raw[:,2]>6.52)&(raw[:,2]<10.0)]
    _,ids=np.unique(np.floor(p/.012).astype(int),axis=0,return_index=True);p=p[ids]
    tree=cKDTree(p);distance,ids=tree.query(p,k=24,workers=8);near=p[ids];near-=near.mean(axis=1,keepdims=True)
    covariance=np.einsum('nki,nkj->nij',near,near)/24;eigen,vectors=np.linalg.eigh(covariance)
    n=vectors[:,:,0];flat=(eigen[:,0]/np.maximum(eigen.sum(axis=1),1e-12)<.045)&(distance[:,-1]<.12)
    np.savez_compressed(cache,p=p,n=n,flat=flat);return p,n,flat


def diagnose():
    OUT.mkdir(parents=True,exist_ok=True);p,n,flat=points()
    h=p[flat&(abs(n[:,2])>.94)&(p[:,1]<.17)&(p[:,1]>-.4)]
    fig,axes=plt.subplots(2,1,figsize=(17,10),layout='constrained')
    axes[0].scatter(p[:,0],p[:,2],s=.2,c='#b5b5b5');axes[0].scatter(h[:,0],h[:,2],s=1,c='#207994')
    axes[0].set_aspect('equal');axes[0].grid(alpha=.2)
    rows=[]
    for x in np.arange(-.10,-4.16,-.26):
        q=h[abs(h[:,0]-x)<.075]
        hist,e=np.histogram(q[:,2],np.arange(6.52,10.002,.002));peaks=find_peaks(hist,prominence=3,distance=25)[0]
        records=[]
        for i in peaks:
            z=(e[i]+e[i+1])/2;near=q[abs(q[:,2]-z)<.012]
            if len(near)>=8:records.append({'z':float(np.median(near[:,2])),'count':len(near),'y_range':np.percentile(near[:,1],[5,95]).tolist()})
        rows.append({'x':float(x),'modes':sorted(records,key=lambda a:-a['count'])})
    hist,e=np.histogram(h[:,2],np.arange(6.52,10.002,.002));peaks=find_peaks(hist,prominence=15,distance=20)[0]
    bands=[]
    for i in peaks:
        z=(e[i]+e[i+1])/2;near=h[abs(h[:,2]-z)<.012]
        if len(near)<30:continue
        bands.append({'z':float(np.median(near[:,2])),'count':len(near),'x_range':np.percentile(near[:,0],[5,50,95]).tolist(),'y_range':np.percentile(near[:,1],[5,95]).tolist()})
    axes[1].scatter(h[:,0],h[:,1],c=h[:,2],s=.8,cmap='turbo');axes[1].set_aspect('equal');axes[1].grid(alpha=.2)
    fig.savefig(OUT/'roof_horizontal_mode_audit.png',dpi=150);plt.close(fig)
    (OUT/'roof_height_modes.json').write_text(json.dumps({'by_x':rows,'bands':bands},indent=2))
    fig,axes=plt.subplots(3,2,figsize=(17,15),layout='constrained')
    for a,(yl,yh) in zip(axes.flat,[(-.5,-.38),(-.38,-.25),(-.25,-.1),(-.1,.05),(.05,.17),(.17,.25)]):
        r=p[(p[:,1]>yl)&(p[:,1]<yh)]
        a.scatter(r[:,0],r[:,2],s=.5,c=r[:,1],cmap='viridis');a.set_title(f'Y {yl} .. {yh} m');a.set_aspect('equal');a.grid(alpha=.2)
        a.set_xlim(-4.3,.3);a.set_ylim(6.5,9.95)
    fig.savefig(OUT/'roof_cross_lanes.png',dpi=160);plt.close(fig)
    print(json.dumps(bands,indent=2),flush=True)


def vertical_diagnostic():
    p,n,flat=points();v=p[flat&(abs(n[:,0])>.88)&(p[:,1]>-.35)&(p[:,1]<.15)]
    rows=[]
    for i in range(16):
        z=6.631+i*.2218;x=.027-i*.248
        r=v[(abs(v[:,0]-x)<.12)&(v[:,2]>z-.19)&(v[:,2]<z+.045)]
        hist,e=np.histogram(r[:,0],np.arange(x-.121,x+.122,.002))
        if hist.max()<3:rows.append({'i':i,'missing':True,'seed_x':x,'seed_z':z});continue
        j=hist.argmax();d=(e[j]+e[j+1])/2;r=r[abs(r[:,0]-d)<.012]
        allp=p[(abs(p[:,0]-d)<.012)&(p[:,1]>-.35)&(p[:,1]<.15)&(p[:,2]>z-.19)&(p[:,2]<z+.045)]
        rows.append({'i':i,'x':float(np.median(r[:,0])),'count':len(r),'z_quantiles':np.percentile(allp[:,2],[1,50,95,99]).tolist(),'seed_z':z})
    print(json.dumps(rows,indent=2),flush=True);(OUT/'roof_riser_diagnostics.json').write_text(json.dumps(rows,indent=2))


def robust_mode(values,seed,half=.12):
    values=np.asarray(values);hist,e=np.histogram(values,np.arange(seed-half,seed+half+.001,.002))
    if not len(values) or hist.max()<5:return None
    i=int(hist.argmax());mode=(e[i]+e[i+1])/2;near=values[abs(values-mode)<.012]
    return float(np.median(near))


def build():
    OUT.mkdir(parents=True,exist_ok=True)
    if list(OUT.glob('Soulace*.skp')):raise FileExistsError('Use another native revision')
    p,n,flat=points();raw=np.load(RAW/'top_raw.npy',mmap_mode='r')
    raw=raw[(raw[:,0]>-4.4)&(raw[:,0]<.4)&(raw[:,1]>-.55)&(raw[:,1]<.25)&(raw[:,2]>6.51)&(raw[:,2]<10.0)]
    h=p[flat&(abs(n[:,2])>.94)]
    sides=p[flat&(abs(n[:,1])>.9)&(p[:,0]<-.2)&(p[:,0]>-3.6)&(p[:,2]>(6.45-.88*p[:,0]))&(p[:,2]<(6.80-.88*p[:,0]))]
    yl=robust_mode(sides[:,1],-.385,.045);yh=robust_mode(sides[:,1],.225,.04)
    if yl is None or yh is None:raise ValueError('Missing measured stair width')
    inner=raw[(raw[:,1]>yl+.07)&(raw[:,1]<yh-.07)]
    fronts=[];heights=[];rows=[]
    for i in range(15):
        expected_z=6.631+i*.2218;expected_x=.027-i*.25
        q=inner[(abs(inner[:,0]-expected_x)<.12)&(inner[:,2]>expected_z-(.065 if i==0 else .19))&(inner[:,2]<expected_z-.025)]
        x=robust_mode(q[:,0],expected_x)
        if x is None:
            q=h[(abs(h[:,2]-expected_z)<.015)&(abs(h[:,0]-(expected_x-.10))<.18)]
            if len(q)<10:raise ValueError('Missing first riser edge evidence')
            x=float(np.percentile(q[:,0],99.5))+.01
            x_source='horizontal_front_edge_plus_10mm_sampling_allowance'
        else:x_source='measured_vertical_riser'
        face=inner[(abs(inner[:,0]-x)<.014)&(inner[:,2]>expected_z-.19)&(inner[:,2]<expected_z+.07)]
        horizontal=h[(abs(h[:,2]-expected_z)<.035)&(h[:,0]<x-.018)&(h[:,0]>x-.23)&(h[:,1]>yl+.03)&(h[:,1]<yh-.03)]
        if len(horizontal)>=6 and i<=6:
            z=float(np.median(horizontal[:,2]));source='observed_tread_top';z_points=len(horizontal)
        else:
            if len(face)<50:raise ValueError(f'Missing riser upper edge {i}: {len(face)}')
            z=float(np.percentile(face[:,2],99.5));source='top_of_observed_vertical_riser';z_points=len(face)
        fronts.append(x);heights.append(z)
        rows.append({'step':i+1,'front_x_m':x,'top_z_m':z,'x_source':x_source,'height_source':source,'height_points':z_points,
                     'riser_inlier_points':len(q),'height_requires_edge_interpretation':source!='observed_tread_top'})
    fronts=np.array(fronts);heights=np.array(heights)
    if not np.all(np.diff(fronts)<-.15) or not np.all((np.diff(heights)>.15)&(np.diff(heights)<.29)):
        raise ValueError('Step sequence inconsistent with the measured run')
    floor=float(np.median(inner[(inner[:,0]>-.2)&(inner[:,0]<.2)&(inner[:,2]<6.57),2]))
    # Measured upper undersides, at their own X edges. These are NOT tread tops.
    under_seeds=[(-1.90,8.051),(-2.15,8.268),(-2.40,8.484),(-2.65,8.710),(-2.90,8.934),(-3.15,9.149),(-3.40,9.372),(-3.64,9.577)]
    underside=[]
    for xs,zs in under_seeds:
        q=h[(abs(h[:,2]-zs)<.02)&(h[:,0]<xs+.03)&(h[:,0]>xs-.26)&(h[:,1]>yl+.03)&(h[:,1]<yh-.03)]
        if len(q)<20:raise ValueError('Missing measured underside band')
        z=float(np.median(q[:,2]))
        vert=inner[(abs(inner[:,0]-xs)<.09)&(inner[:,2]>zs-.18)&(inner[:,2]<zs-.025)]
        x=robust_mode(vert[:,0],xs,.09)
        if x is None:x=float(np.percentile(q[:,0],99))+.012
        underside.append([x,z])
    under=np.array(underside);pitch=float(np.median(-np.diff(under[:,0])));rise=float(np.median(np.diff(under[:,1])))
    # Lower underside patches are too occluded for independent per-step fitting.
    # Continue the measured underside cadence, explicitly separate from walking
    # levels; trim the continuation at the measured floor.
    low=[]
    for k in range(7,0,-1):
        x=under[0,0]+pitch*k;z=under[0,1]-rise*k
        if z>floor+.005:low.append([x,z])
    under=np.vstack([low,under])
    landing=h[(abs(h[:,2]-under[-1,1])<.025)&(h[:,0]<-3.65)&(h[:,1]>yl+.025)&(h[:,1]<yh-.025)]
    back=float(np.percentile(landing[:,0],1));back=max(back,-4.18)
    top_path=[[fronts[0],heights[0]]]
    for i in range(1,len(fronts)):
        top_path.extend([[fronts[i],heights[i-1]],[fronts[i],heights[i]]])
    top_path.append([back,heights[-1]])
    bottom_path=[[fronts[0],floor]];current=floor
    for x,z in under:
        if back<x<fronts[0]:bottom_path.extend([[x,current],[x,z]]);current=z
    bottom_path.append([back,current])
    poly=Polygon(top_path+bottom_path[::-1])
    if not poly.is_valid:raise ValueError('Stair side section intersects itself')
    mesh=solid_from_profile(poly,1,yl,yh)
    if not mesh.is_watertight:raise ValueError('Clean roof stair is not closed')
    part={'name':'Top-floor roof stair - clean measured run','kind':'roof_stair_clean','level':2,'colour':[203,142,77],
          'construction_solid':True,'v':np.round(mesh.vertices,7).tolist(),'f':mesh.faces.tolist()}
    (OUT/'roof_stair.build.json').write_text(json.dumps({'parts':[part]},separators=(',',':')))
    report={'steps':rows,'width_m':yh-yl,'side_y_m':[yl,yh],'bottom_floor_z_m':floor,'landing_back_x_m':back,
            'upper_underside_measured':underside,'lower_underside_continuation_modeled':low,
            'riser_heights_m':np.diff(np.r_[floor,heights]).tolist(),'tread_depths_m':(-np.diff(np.r_[fronts,back])).tolist(),
            'volume_m3':abs(mesh.volume),'watertight':True,
            'limitations':'Clean geometric reconstruction, not certified perfect site dimensions. Upper tread heights use observed vertical-riser top edges; lower undersides and landing closure are modeled continuations. No railings fabricated.'}
    (OUT/'roof_stair_fit.json').write_text(json.dumps(report,indent=2))
    # Same geometry, independent views; gray points are unedited raw evidence.
    fig=plt.figure(figsize=(17,11),layout='constrained');a=fig.add_subplot(221);b=fig.add_subplot(222);c=fig.add_subplot(223,projection='3d');d=fig.add_subplot(224)
    near=p[(p[:,1]>yl+.04)&(p[:,1]<yh-.04)]
    a.scatter(near[:,0],near[:,2],s=.3,c='#929da4');tp=np.array(top_path);bp=np.array(bottom_path);a.plot(tp[:,0],tp[:,1],c='#ce5938',lw=1.5,label='Modeled walking profile');a.plot(bp[:,0],bp[:,1],c='#386cb0',lw=1,label='Underside profile');a.legend();a.set_aspect('equal');a.grid(alpha=.2);a.set_title('Side: model over unedited scan')
    b.scatter(near[:,0],near[:,1],s=.2,c='#929da4')
    for x in fronts:b.plot([x,x],[yl,yh],c='#ce5938',lw=1)
    b.plot([fronts[0],back,back,fronts[0],fronts[0]],[yl,yl,yh,yh,yl],c='#ce5938');b.set_aspect('equal');b.set_title('Top: width and step positions')
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    c.add_collection3d(Poly3DCollection(mesh.triangles,facecolor='#c88b4b',edgecolor='#855f3b',linewidth=.2));c.set_xlim(back,fronts[0]);c.set_ylim(yl,yh);c.set_zlim(floor,heights[-1]);c.set_box_aspect((4.2,.7,3.3));c.view_init(24,-65);c.set_title('Clean solid staircase')
    d.plot(np.arange(1,16),np.diff(np.r_[floor,heights])*1000,'o-');d.set_title('Measured/edge-fitted rises (mm), not forced equal');d.grid(alpha=.2)
    fig.savefig(OUT/'roof_stair_model_scan_check.png',dpi=140);plt.close(fig)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--risers',action='store_true');ap.add_argument('--build',action='store_true');args=ap.parse_args()
    if args.build:build()
    elif args.risers:vertical_diagnostic()
    else:diagnose()
