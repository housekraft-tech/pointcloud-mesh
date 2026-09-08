"""Plot measured stair/parking regions and camera evidence; no geometry changes."""
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output_final/scan_first_diagnostics/soulace'


def main():
    z=np.load(OUT/'sample.npz');p=z['p'];h=z['horizontal'];matrix=z['matrix']
    stair=(p[:,0]>-4.85)&(p[:,0]<-2.65)&(p[:,1]>1.8)&(p[:,1]<6.35)
    q=p[stair];hq=p[stair&h]
    fig=plt.figure(figsize=(17,10),layout='constrained');axes=[fig.add_subplot(231),fig.add_subplot(232),fig.add_subplot(233,projection='3d'),fig.add_subplot(234),fig.add_subplot(235),fig.add_subplot(236)]
    for a,(i,j,title) in zip(axes[:2],[(0,1,'Plan: horizontal returns'),(1,2,'Stair profile: horizontal returns')]):
        a.scatter(hq[:,i],hq[:,j],c=hq[:,2],cmap='turbo',s=1);a.set_title(title);a.set_aspect('equal');a.grid(alpha=.2)
    axes[2].scatter(hq[::3,0],hq[::3,1],hq[::3,2],c=hq[::3,2],cmap='turbo',s=1);axes[2].set_box_aspect((2.2,4.55,9.8));axes[2].set_title('3D horizontal returns');axes[2].view_init(20,-50)
    for a,(lo,hi) in zip(axes[3:5],[(-4.75,-3.65),(-3.60,-2.7)]):
        sub=q[(q[:,0]>lo)&(q[:,0]<hi)]
        a.scatter(sub[:,1],sub[:,2],s=.2,c='slategrey');a.set_aspect('equal');a.set_title(f'Stair lane X={lo}..{hi} m');a.set_xlabel('Y (m)');a.set_ylabel('Z (m)');a.grid(alpha=.2)
    hst,edges=np.histogram(hq[:,2],bins=np.arange(-.2,9.9,.005));peaks=find_peaks(hst,prominence=40,distance=16)[0]
    axes[5].plot(hst,(edges[:-1]+edges[1:])/2);axes[5].set_title('Measured horizontal-height modes');axes[5].set_ylabel('Z (m)')
    fig.suptitle('Soulace staircase | raw scan evidence, no regular risers imposed',fontsize=17)
    fig.savefig(OUT/'stair_investigation.png',dpi=160);plt.close(fig)
    print('stair modes',[(round(float((edges[i]+edges[i+1])/2),4),int(hst[i])) for i in peaks],flush=True)
    # Existing rectified video, sampled at trajectory times in the two regions.
    odo=np.loadtxt(ROOT/'data/Soulace/odometerdata.txt');pos=odo[:,2:5]@matrix[:3,:3].T+matrix[:3,3]
    cap=cv2.VideoCapture(str(ROOT/'soulace_output/video/soulace_flat.mp4'))
    fps=cap.get(cv2.CAP_PROP_FPS);count=cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print('video',fps,count,flush=True)
    filters=[('Parking region',(pos[:,0]>2)&(pos[:,1]<1.5)&(pos[:,2]<1.3)),('Stair region',(pos[:,0]>-4.9)&(pos[:,0]<-2.6)&(pos[:,1]>2)&(pos[:,2]>1.6)&(pos[:,2]<4))]
    fig,ax=plt.subplots(2,3,figsize=(17,9),layout='constrained')
    for row,(name,mask) in enumerate(filters):
        mask &= (odo[:,1]-9947.105441>=0)&(odo[:,1]-9947.105441<count/fps)
        ids=np.flatnonzero(mask)
        for a,ix in zip(ax[row],ids[np.linspace(0,len(ids)-1,3).astype(int)]):
            seconds=odo[ix,1]-9947.105441
            cap.set(cv2.CAP_PROP_POS_MSEC,seconds*1000);ok,frame=cap.read()
            if ok:a.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            a.set_title(f'{name} candidate | video {seconds:.1f}s\nOdometer XYZ={np.round(pos[ix],2)} m');a.axis('off')
    fig.suptitle('Unvalidated odometer/video association — NOT confirmed location evidence',fontsize=15)
    fig.savefig(OUT/'parking_stair_photos.png',dpi=140);plt.close(fig);cap.release()


if __name__=='__main__':main()
