"""Time-associated camera candidates; no unvalidated odometer projection."""
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from recover_soulace_top_floor import ROOT,OUT,top_reference


def main():
    p,t=top_reference()
    mask=(p[:,0]>-3.9)&(p[:,0]<.25)&(p[:,1]>-.45)&(p[:,1]<.25)&(p[:,2]>6.6)&(p[:,2]<9.6)
    mask &= abs(p[:,2]-(6.5-.85*p[:,0]))<.22
    times=t[mask];hist,e=np.histogram(times,np.arange(t.min(),t.max()+2,2));peaks=find_peaks(hist,distance=10)[0]
    peaks=peaks[np.argsort(hist[peaks])[::-1][:9]];peaks=np.sort(peaks)
    timestamps=np.loadtxt(ROOT/'data/Soulace/corcam_1.ts');cap=cv2.VideoCapture(str(ROOT/'soulace_output/video/soulace_flat.mp4'))
    fig,axes=plt.subplots(3,3,figsize=(18,12),layout='constrained');records=[]
    for ax,i in zip(axes.flat,peaks):
        timestamp=(e[i]+e[i+1])/2;ix=int(np.argmin(abs(timestamps-timestamp)))
        cap.set(cv2.CAP_PROP_POS_FRAMES,ix);ok,frame=cap.read()
        if ok:
            ax.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));cv2.imwrite(str(OUT/f'roof_camera_candidate_{ix}.jpg'),frame)
        ax.set_title(f'Frame {ix} | video {ix/30:.1f}s | stair-region returns {hist[i]}');ax.axis('off')
        records.append({'frame':ix,'camera_timestamp':float(timestamps[ix]),'stair_return_time':float(timestamp),'return_count':int(hist[i])})
    fig.suptitle('Camera frames near roof-stair return timestamps | candidates, not calibrated overlays',fontsize=16)
    fig.savefig(OUT/'roof_access_camera_candidates.png',dpi=130);plt.close(fig);cap.release()
    (OUT/'roof_camera_candidates.json').write_text(json.dumps(records,indent=2));print(records,flush=True)


if __name__=='__main__':main()
