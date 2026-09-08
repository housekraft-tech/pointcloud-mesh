"""Static before/after sections from actual CAD, against unchanged LAS slices."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from audit_soulace_overlap import ROOT, STEMS, section_segments

OUT=ROOT/'output_final/soulace_overlap_only_50mm/release'

for level in [0,1,2]:
    original=json.loads((ROOT/'output_final/soulace_asbuilt_v2'/f'{STEMS[level]}_asbuilt.build.json').read_text())['parts']
    filtered=json.loads((OUT/f'Soulace_L{level}_overlap_only.build.json').read_text())['parts']
    with np.load(ROOT/f'output_final/soulace_overlap_audit_20260903/L{level}_section_1.5.npz') as data:
        raw=data['raw']
    fig,axes=plt.subplots(1,2,figsize=(15,7),dpi=180)
    bounds=[]
    for ax,payload,label,color in zip(axes,[original,filtered],['Before: full CAD','After: only scan-overlapping surfaces'],['#c75839','#137b57']):
        sections=[]
        for item in payload:
            if item['kind'].startswith('wall') or item['kind']=='column':
                vertices=np.asarray(item['v'])
                sections.append(section_segments(vertices,np.asarray(item['f']),1.5))
                bounds.extend(vertices[:,:2].tolist())
        segments=np.concatenate(sections)
        ax.scatter(raw[:,0],raw[:,1],s=.5,c='#008d9f',alpha=.55,rasterized=True)
        ax.add_collection(LineCollection(segments[:,:,:2],colors=color,linewidths=1.0))
        ax.set_title(label,loc='left',fontsize=14)
        ax.set_aspect('equal');ax.set_xlabel('X (m)');ax.set_ylabel('Y (m)')
        ax.grid(color='#e3e5e8',linewidth=.4)
    bounds=np.asarray(bounds)
    lo=bounds.min(0)-.3; hi=bounds.max(0)+.3
    for ax in axes:
        ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1])
    fig.suptitle(f'SOULACE L{level} | wall section 1.50 m above floor | no refitting or filled gaps',x=.06,ha='left',fontsize=15)
    fig.legend(handles=[Line2D([],[],marker='.',linestyle='none',color='#008d9f',label='LiDAR returns'),
                        Line2D([],[],color='#c75839',label='Original CAD'),
                        Line2D([],[],color='#137b57',label='Retained CAD: within 50 mm of raw scan')],
               loc='lower center',ncol=3,frameon=False)
    fig.tight_layout(rect=(0,.06,1,.94))
    fig.savefig(OUT/f'Soulace_L{level}_before_after.png',facecolor='white')
    plt.close(fig)
