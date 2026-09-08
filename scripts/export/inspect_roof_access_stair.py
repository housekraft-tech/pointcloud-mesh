"""Locate the separate top-storey stepped run, outside the lower stairwell ROI."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from inspect_soulace_top_floor import ROOT,OUT


def main():
    s=np.load(ROOT/'output_final/scan_first_diagnostics/soulace/sample.npz');p=s['p'];h=s['horizontal']
    mask=(p[:,0]>-5.2)&(p[:,0]<1.2)&(p[:,1]>-1.8)&(p[:,1]<.5)&(p[:,2]>6.4)&(p[:,2]<9.95)
    q=p[mask];h=q if False else p[mask&h]
    fig=plt.figure(figsize=(17,11),layout='constrained');ax=[fig.add_subplot(221),fig.add_subplot(222),fig.add_subplot(223,projection='3d'),fig.add_subplot(224)]
    ax[0].scatter(q[:,0],q[:,2],s=.3,c='#89969c',alpha=.35);ax[0].scatter(h[:,0],h[:,2],s=1,c=h[:,2],cmap='turbo');ax[0].set_xlabel('X');ax[0].set_ylabel('Z');ax[0].set_aspect('equal')
    ax[1].scatter(q[:,0],q[:,1],s=.2,c='#89969c',alpha=.35);ax[1].scatter(h[:,0],h[:,1],s=1,c=h[:,2],cmap='turbo');ax[1].set_xlabel('X');ax[1].set_ylabel('Y');ax[1].set_aspect('equal')
    ax[2].scatter(h[:,0],h[:,1],h[:,2],s=1,c=h[:,2],cmap='turbo');ax[2].set_box_aspect((6.4,2.3,3.55));ax[2].view_init(25,-70)
    narrow=p[(p[:,0]>-4.2)&(p[:,0]<1.1)&(p[:,1]>-.33)&(p[:,1]<.23)&(p[:,2]>6.45)&(p[:,2]<9.95)]
    ax[3].scatter(narrow[:,0],narrow[:,2],s=.7,c='#47728c');ax[3].set_aspect('equal');ax[3].set_title('Narrow lane: all raw sampled returns')
    for a in [ax[0],ax[1],ax[3]]:a.grid(alpha=.2)
    fig.suptitle('Separate roof-access candidate | raw scan only, not the earlier stairwell',fontsize=17);fig.savefig(OUT/'roof_access_stair_scan.png',dpi=150);plt.close(fig)
    h=h[(h[:,1]>-.4)&(h[:,1]<.3)&(h[:,0]>-4.1)]
    records=[]
    for x in np.arange(-3.8,.81,.125):
        r=h[(h[:,0]>x)&(h[:,0]<x+.105)]
        hist,e=np.histogram(r[:,2],np.arange(6.45,9.955,.005));peaks=find_peaks(hist,prominence=2,distance=14)[0]
        peaks=peaks[np.argsort(hist[peaks])[::-1][:4]]
        records.append({'x':round(float(x),3),'modes':[(round(float(e[i]+.0025),4),int(hist[i])) for i in peaks]})
    (OUT/'roof_access_height_modes.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2),flush=True)


if __name__=='__main__':main()
