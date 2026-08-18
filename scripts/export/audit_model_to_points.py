"""Is the surface each model DRAWS actually supported by the scan?

Run alongside audit_points_to_model.py, which asks the opposite question. Both
are needed: align_check reports only faces on walls seen from BOTH sides, which
is about half the model and the half it got right. Quoting its 3 mm median as
the model's accuracy was wrong of me -- it never looks at invented surface.

CAVEAT on the unsupported figure: a box has six faces, and its top, bottom and
the ends abutting other walls are never scanned because they are never visible.
Some of the unsupported surface is those hidden faces rather than error. The
points-to-model direction has no such caveat and is the cleaner number.

Sample points on each mesh's own surface, then ask how far the nearest scanned
point is. Surface with no scan near it is invented.
"""
import struct, json, numpy as np, laspy
from scipy.spatial import cKDTree
J=json.load(open("output/fp_walls.json")); H=J['clear_height']
with laspy.open("output/mujammel_aligned_z0.las") as r: p=r.read()
P=np.column_stack([p.x,p.y,p.z]).astype(np.float64)
k=np.load("output/keep_mask.npy")
if len(k)==len(P): P=P[k]
P=P[P[:,2] < H-0.12]
rng=np.random.default_rng(0)
R=P[rng.choice(len(P), 5_000_000, replace=False)]
G=np.column_stack([R[:,0], R[:,2], -R[:,1]])       # glTF frame
print(f"building KD-tree on {len(G):,} scanned points...", flush=True)
tree=cKDTree(G)

def tris(path):
    f=open(path,'rb'); struct.unpack("<III",f.read(12))
    ln,_=struct.unpack("<II",f.read(8)); js=json.loads(f.read(ln))
    bl,_=struct.unpack("<II",f.read(8)); buf=f.read(bl)
    acc,bv=js['accessors'],js['bufferViews']
    def read(i):
        a=acc[i]; v=bv[a['bufferView']]
        off=v.get('byteOffset',0)+a.get('byteOffset',0)
        ct={5126:('<f4',3),5125:('<u4',1),5123:('<u2',1)}[a['componentType']]
        n=a['count']*(3 if a['type']=='VEC3' else 1)
        return np.frombuffer(buf,dtype=ct[0],count=n,offset=off).reshape(
            a['count'],-1).astype(np.float64)
    V=[]
    for n in js.get('nodes',[]):
        if 'mesh' not in n: continue
        for pr in js['meshes'][n['mesh']]['primitives']:
            pos=read(pr['attributes']['POSITION'])
            idx=read(pr['indices']).ravel().astype(int)
            V.append(pos[idx].reshape(-1,3,3))
    return np.concatenate(V) if V else np.zeros((0,3,3))

for label,path in (("MODULAR model","output/model/shell_fp.glb"),
                   ("VOXEL surface","output/model/scan_surface.glb")):
    T=tris(path)
    if not len(T): print(f"{label}: none"); continue
    # sample uniformly over triangle area
    e1,e2=T[:,1]-T[:,0], T[:,2]-T[:,0]
    area=0.5*np.linalg.norm(np.cross(e1,e2),axis=1)
    w=area/area.sum()
    N=300_000
    pick=rng.choice(len(T), N, p=w)
    u,v=rng.random(N),rng.random(N)
    fold=u+v>1; u[fold],v[fold]=1-u[fold],1-v[fold]
    S=T[pick,0]+u[:,None]*e1[pick]+v[:,None]*e2[pick]
    d,_=tree.query(S, workers=-1)
    print(f"\n{label}: {len(T):,} triangles, {area.sum():.1f} m2 of surface")
    for t in (0.005,0.010,0.020,0.050,0.100,0.200):
        print(f"   surface within {t*1000:>4.0f} mm of a scanned point: "
              f"{(d<=t).mean()*100:>5.1f}%")
    print(f"   median {np.median(d)*1000:.0f} mm | 90th {np.percentile(d,90)*1000:.0f} mm"
          f" | 99th {np.percentile(d,99)*1000:.0f} mm")
    print(f"   UNSUPPORTED (>100 mm from any scanned point): "
          f"{(d>0.100).mean()*100:.1f}% of drawn surface "
          f"= {(d>0.100).mean()*area.sum():.1f} m2")
