"""Read-only raw scan diagnosis of the parking/house elevation junction."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

def main():
    out = ROOT/'output_final/architectural_flow_astra/parking_diagnostic'
    out.mkdir(parents=True, exist_ok=True)
    raw = np.load(ROOT/'output_final/soulace_clean_floor_stairs_v2/raw_detail.npy', mmap_mode='r')
    p = raw[(raw[:,0] > 1.1) & (raw[:,0] < 6.3) & (raw[:,1] > -4.3) & (raw[:,1] < 1.8) & (raw[:,2] < .08)]
    fig, ax = plt.subplots(2,2,figsize=(16,11),layout='constrained')
    q = p[::3]
    im=ax[0,0].scatter(q[:,0],q[:,1],c=q[:,2],s=.4,vmin=-.6,vmax=0,cmap='turbo')
    ax[0,0].set_title('Parking junction: all raw returns below interior floor')
    fig.colorbar(im,ax=ax[0,0],label='z (m)')
    sections=[('house long side',(p[:,0]>1.3)&(p[:,0]<1.85),1),
              ('porch/front threshold',(p[:,1]>-.1)&(p[:,1]<.4),0),
              ('entry steps',(p[:,1]>.65)&(p[:,1]<1.15),0)]
    for a,(title,mask,horizontal) in zip([ax[0,1],ax[1,0],ax[1,1]],sections):
        q=p[mask]
        a.scatter(q[:,horizontal],q[:,2],s=.5,c=q[:,1-horizontal],cmap='viridis')
        a.axhline(-.5334,color='cyan',label='single parking plane')
        a.axhline(-.1,color='gray',label='model slab bottom')
        a.axhline(0,color='black',label='interior datum')
        a.set_title(title);a.set_xlabel('XY'[horizontal]+' (m)');a.set_ylabel('z (m)')
        a.legend(fontsize=8)
    for a in ax.flat:a.grid(alpha=.2)
    ax[0,0].set_aspect('equal')
    fig.savefig(out/'raw_parking_junction_sections.png',dpi=140)
    plt.close(fig)
    # Dominant vertical-strip coordinates, not a claim of wall semantics.
    q=p[(p[:,2]>-.43)&(p[:,2]<-.12)]
    hist,edges=np.histogram(q[:,0],bins=np.arange(1.1,6.31,.01))
    peaks=sorted(zip(hist, (edges[1:]+edges[:-1])/2),reverse=True)[:25]
    report={'low_junction_returns':len(p),'intermediate_height_returns':len(q),
            'x_density_peaks_count_and_m':[(int(n),float(x)) for n,x in peaks]}
    (out/'diagnosis.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
