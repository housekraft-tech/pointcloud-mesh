"""Read-only diagnostics of top-storey omissions and staircase scene membership."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from recover_soulace_details import original_payload
from inspect_scan_coverage import distance_to_parts
from render_scan_recovery import render

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output_final/soulace_top_floor_diagnostics'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    s=np.load(ROOT/'output_final/scan_first_diagnostics/soulace/sample.npz');p=s['p'];h=s['horizontal']
    data=original_payload();parts=[g for g in data['parts'] if g['name'].startswith('L2_')]
    stairs=json.loads((ROOT/'output_final/soulace_clean_floor_stairs_v2/upper_stairs.build.json').read_text())['parts']
    q=p[(p[:,2]>6.53)&(p[:,2]<9.8)];dist=distance_to_parts(q,{'parts':parts+stairs})
    np.savez_compressed(OUT/'top_floor_coverage.npz',p=q,distance=dist)
    report={'sample_points':len(q),'within_10_20_50mm_pct':[float(np.mean(dist<t)*100) for t in [.01,.02,.05]],
            'p50_p95_mm':(np.percentile(dist,[50,95])*1000).tolist(),
            'current_top_floor_parts':len(parts),'stairs_in_top_floor_scene':[g['name'] for g in stairs if g['level']==2],
            'stairs_excluded_from_top_floor_scene':[g['name'] for g in stairs if g['level']!=2]}
    print(json.dumps(report,indent=2),flush=True)
    (OUT/'coverage_report.json').write_text(json.dumps(report,indent=2))
    fig,axes=plt.subplots(2,3,figsize=(18,12),layout='constrained')
    for a,(lo,hi) in zip(axes[0],[(6.65,7.05),(7.45,7.85),(8.4,8.8)]):
        ids=(q[:,2]>lo)&(q[:,2]<hi);good=ids&(dist<=.05);bad=ids&(dist>.05)
        a.scatter(q[good,0],q[good,1],s=.3,c='#547c8d');a.scatter(q[bad,0],q[bad,1],s=.5,c='#e45435')
        a.set_aspect('equal');a.set_title(f'Z {lo}–{hi} m | red >50 mm from model');a.grid(alpha=.2)
    visible=[g for g in parts if 'ceiling' not in g['kind']]
    axes[1,0].imshow(render(visible+stairs,direction=(1.3,-1.5,1.25),w=1000,h=750));axes[1,0].axis('off');axes[1,0].set_title('Current top-floor CAD + arriving stairs')
    upper=p[h&(p[:,2]>6.65)&(p[:,2]<9.8)]
    axes[1,1].scatter(upper[:,0],upper[:,1],c=upper[:,2],s=.6,cmap='turbo',vmin=6.65,vmax=9.8);axes[1,1].set_aspect('equal');axes[1,1].set_title('All horizontal returns above top-floor landing')
    hist,e=np.histogram(upper[:,2],np.arange(6.65,9.805,.005));peaks=find_peaks(hist,prominence=30,distance=12)[0]
    axes[1,2].plot(hist,(e[1:]+e[:-1])/2);axes[1,2].set_title('Horizontal height modes above Z6.65 m');axes[1,2].grid(alpha=.2)
    fig.suptitle('Soulace top floor | raw LiDAR / current model audit',fontsize=18);fig.savefig(OUT/'top_floor_scan_audit.png',dpi=140);plt.close(fig)
    print('Upper horizontal modes',[(round(float(e[i]+.0025),3),int(hist[i])) for i in peaks],flush=True)
    np.savez_compressed(OUT/'upper_horizontal.npz',p=upper)


if __name__=='__main__':main()
