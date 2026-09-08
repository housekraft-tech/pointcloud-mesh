"""Scientific diagnostic: unchanged scan samples, candidates and rebuilt faces."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np
import trimesh


def boundary_edges(part):
    mesh=trimesh.Trimesh(part['v'],part['f'],process=False)
    mesh.merge_vertices()
    skip=set()
    for pair, edge in zip(mesh.face_adjacency, mesh.face_adjacency_edges):
        if abs(mesh.face_normals[pair[0]]@mesh.face_normals[pair[1]])>1-1e-8:
            skip.add(tuple(sorted(edge)))
    edges=[edge for edge in mesh.edges_unique if tuple(sorted(edge)) not in skip]
    return mesh.vertices[np.asarray(edges)] if edges else np.empty((0,2,3))


def sections(parts, z):
    segments=[]
    for p in parts:
        v=np.asarray(p['v']); f=np.asarray(p['f'])
        for t in v[f]:
            crossings=[]
            for a,b in zip(t,np.roll(t,-1,axis=0)):
                if (a[2]<=z<b[2]) or (b[2]<=z<a[2]):
                    crossings.append(a+(b-a)*(z-a[2])/(b[2]-a[2]))
            if len(crossings)==2:
                segments.append(np.asarray(crossings)[:,:2])
    return segments


def render(folder):
    folder=Path(folder).resolve()
    audit=json.loads((folder/'audit.json').read_text())
    before=json.loads(Path(audit['manifest']['candidate_model']).read_text())['parts']
    after=json.loads((folder/'model.build.json').read_text())['parts']
    points=np.load(folder/'scan_display.npz')['points']
    # Ceilings hidden only for display; all retained surfaces are in the audit.
    before=[p for p in before if 'ceiling' not in p['kind']]
    after=[p for p in after if 'ceiling' not in p['kind']]
    vertices=np.concatenate([p['v'] for p in before])
    lo,hi=vertices.min(0),vertices.max(0)
    floors=[np.asarray(p['v'])[:,2].max() for p in before if p['kind']=='floor']
    z=float(np.median(floors))+1.2 if floors else float(lo[2]+1.2)
    rng=np.random.default_rng(6)
    shown=points[rng.choice(len(points),min(28_000,len(points)),replace=False)]
    fig=plt.figure(figsize=(18,11),facecolor='#f7f8fa')
    for row,parts in enumerate([before,after]):
        colour='#dc6b3d' if row==0 else '#245e79'
        title='CANDIDATES - BEFORE' if row==0 else 'RECTANGULAR REBUILD - AFTER'
        ax=fig.add_subplot(2,3,row*3+1)
        sl=points[abs(points[:,2]-z)<.075]
        ax.scatter(sl[:,0],sl[:,1],s=.7,c='#8b939b',alpha=.6,rasterized=True)
        ax.add_collection(LineCollection(sections(parts,z),colors=colour,linewidths=.9))
        ax.set(xlim=(lo[0]-.2,hi[0]+.2),ylim=(lo[1]-.2,hi[1]+.2),xlabel='X (m)',ylabel='Y (m)',title=f'{title}\nTop section at {z:.2f} m')
        ax.set_aspect('equal'); ax.grid(alpha=.15)
        ax=fig.add_subplot(2,3,row*3+2)
        ax.scatter(shown[:,0],shown[:,2],s=.3,c='#8b939b',alpha=.25,rasterized=True)
        wires=np.concatenate([boundary_edges(p) for p in parts])
        ax.add_collection(LineCollection(wires[:,:,[0,2]],colors=colour,linewidths=.45))
        ax.set(xlim=(lo[0]-.2,hi[0]+.2),ylim=(lo[2]-.1,hi[2]+.1),xlabel='X (m)',ylabel='Z (m)',title='Front elevation - all depths projected')
        ax.set_aspect('equal');ax.grid(alpha=.15)
        ax=fig.add_subplot(2,3,row*3+3,projection='3d')
        polys=[]
        for p in parts:
            v=np.asarray(p['v'])
            polys.extend(v[face] for face in p.get('quads',p['f']))
        ax.add_collection3d(Poly3DCollection(polys,facecolors=colour,edgecolors='none',alpha=.10))
        ax.add_collection3d(Line3DCollection(wires,colors=colour,linewidths=.20,alpha=.65))
        ax.scatter(shown[:,0],shown[:,1],shown[:,2],s=.15,c='#374b56',alpha=.20,depthshade=False,rasterized=True)
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)',title='3D overlay - no display realignment')
        ax.set_box_aspect(hi-lo);ax.view_init(elev=34,azim=-55)
    s=audit['summary']['area_weighted_scan_support']
    fig.suptitle(audit['label'],fontsize=21,x=.04,ha='left')
    fig.text(.04,.92,f"Same scan reference in both rows | Median {s['median_mm']:.1f} mm | {s['within_50mm_pct']:.1f}% of retained sampled surface within 50 mm",fontsize=12)
    fig.text(.04,.025,'Grey: sampled LiDAR for display; numeric tests use full-resolution ROI.  No-return regions are unknown, not proven openings.\nCandidate openings are inherited, not independently detected here. Ceilings hidden for visibility. Scan-fit statistics are not room-dimension certification.',fontsize=10)
    fig.subplots_adjust(left=.045,right=.97,top=.86,bottom=.105,wspace=.25,hspace=.35)
    path=folder/'scan_comparison_top_front_3d.png'
    fig.savefig(path,dpi=160);plt.close(fig)
    print(path)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--folder',required=True)
    render(ap.parse_args().folder)
