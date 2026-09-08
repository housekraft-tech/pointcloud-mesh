"""Three-dimensional and orthographic evidence for the suspect L0 wall_32."""
import json
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D

from audit_soulace_overlap import ROOT, raw_points, section_segments, thin

OUT=ROOT/'output_final/soulace_overlap_only_50mm/release'
source_dir=ROOT/'output_final/soulace_asbuilt_v2'
name='wall_32'
metadata=json.loads((source_dir/'Soulace_L0_ground_asbuilt_manifest.json').read_text())
before=next(p for p in json.loads((source_dir/'Soulace_L0_ground_asbuilt.build.json').read_text())['parts'] if p['name']==name)
after=next(p for p in json.loads((OUT/'Soulace_L0_overlap_only.build.json').read_text())['parts'] if p['name']==name)
original=trimesh.Trimesh(before['v'],before['f'],process=False)
filtered=trimesh.Trimesh(after['v'],after['f'],process=False)
raw,_=raw_points(ROOT/'output_final/soulace_L0/lidar/L0.las',metadata['source_yaw_deg'],metadata['source_floor_z_m'])
lo,hi=original.bounds
cross=int(np.argmin((hi-lo)[:2]))
along=1-cross
order=[along,cross,2]
take=((raw[:,0]>=lo[0]-.10)&(raw[:,0]<=hi[0]+.10)&(raw[:,1]>=lo[1]-.08)&(raw[:,1]<=hi[1]+.08)&(raw[:,2]>=-.04)&(raw[:,2]<=hi[2]+.08))
local=raw[take]
del raw
rng=np.random.default_rng(18793)
points=thin(local,24000,rng)
slice_points=thin(local[np.abs(local[:,2]-1.5)<.025],16000,rng)
fig=plt.figure(figsize=(20,10.8),dpi=170)
for row,(mesh,label,colour) in enumerate([(original,'BEFORE — full-height CAD','#d2603f'),(filtered,'AFTER — retained overlap only','#218764')]):
    ax=fig.add_subplot(2,3,row*3+1)
    ax.scatter(slice_points[:,along],slice_points[:,cross],s=2,c='#008fa7',alpha=.65)
    seg=section_segments(mesh.vertices,mesh.faces,1.5)
    if len(seg):
        ax.add_collection(LineCollection(seg[:,:,[along,cross]],colors=colour,linewidths=1.4))
    ax.set_xlim(lo[along]-.10,hi[along]+.10);ax.set_ylim(lo[cross]-.10,hi[cross]+.10)
    ax.set_xlabel('Along wall (m)');ax.set_ylabel('Depth (m)')
    ax.set_title(f'{label}\nTop section at 1.50 m',loc='left',fontsize=12)
    ax.grid(color='#e1e4e8',linewidth=.4)
    ax=fig.add_subplot(2,3,row*3+2)
    ids=np.abs(mesh.face_normals[:,cross])>.95
    ax.add_collection(PolyCollection(mesh.triangles[ids][:,:,[along,2]],facecolors=colour,edgecolors='none',alpha=.36))
    # Both sides are projected here; source crop is the same narrow depth band.
    ax.scatter(points[:,along],points[:,2],s=.55,c='#008fa7',alpha=.55,rasterized=True)
    ax.set_xlim(lo[along]-.10,hi[along]+.10);ax.set_ylim(-.05,hi[2]+.10)
    ax.set_aspect('equal',adjustable='box')
    ax.set_xlabel('Along wall (m)');ax.set_ylabel('Height above floor (m)')
    ax.set_title('Side / elevation — wall depth band',loc='left',fontsize=12)
    ax.grid(color='#e1e4e8',linewidth=.4)
    ax=fig.add_subplot(2,3,row*3+3,projection='3d')
    ax.add_collection3d(Poly3DCollection(mesh.triangles[:,:,order],facecolor=colour,edgecolor='none',alpha=.38,rasterized=True))
    ax.scatter(points[:,along],points[:,cross],points[:,2],s=.45,c='#008fa7',alpha=.6,depthshade=False,rasterized=True)
    ax.set_xlim(lo[along]-.10,hi[along]+.10);ax.set_ylim(lo[cross]-.10,hi[cross]+.10);ax.set_zlim(-.05,hi[2]+.10)
    ax.set_box_aspect([hi[along]-lo[along],max(.5,hi[cross]-lo[cross]),hi[2]+.1])
    ax.view_init(elev=18,azim=-68)
    ax.set_xlabel('Along (m)',labelpad=9);ax.set_ylabel('Depth (m)',labelpad=9);ax.set_zlabel('Height (m)',labelpad=7)
    ax.set_yticks([round((lo[cross]+hi[cross])/2,2)])
    ax.set_zticks([0,1,2,3])
    ax.set_title('3D overlay — actual model and LiDAR',loc='left',fontsize=12,pad=4)
fig.suptitle('SOULACE L0 / wall_32 | top, side and 3D overlap | fixed alignment',x=.035,ha='left',fontsize=18)
fig.legend(handles=[Line2D([],[],marker='.',linestyle='none',color='#008fa7',label='Raw LiDAR (display sampled)'),
                    Line2D([],[],color='#d2603f',label='Original CAD'),Line2D([],[],color='#218764',label='Filtered CAD: 50 mm maximum envelope')],loc='lower center',ncol=3,frameon=False,fontsize=11)
fig.subplots_adjust(left=.055,right=.975,bottom=.085,top=.9,wspace=.25,hspace=.30)
fig.savefig(OUT/'Soulace_wall32_top_side_3d.png',facecolor='white')
plt.close(fig)
print(OUT/'Soulace_wall32_top_side_3d.png')
