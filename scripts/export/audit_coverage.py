"""Coverage, measured the same way for every mesh: how far is each scanned
point from the nearest modelled SURFACE?

The bounding-box version of this only works when a file has one part per
element. Merged meshes make its boxes whole-scene slabs and the answer is
nonsense. This samples each mesh's triangles densely and puts the KD-tree on
those samples, so it is valid for any mesh and comparable across all of them.
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
S=P[rng.choice(len(P), 400_000, replace=False)]
Q=np.column_stack([S[:,0], S[:,2], -S[:,1]])          # glTF frame

def tris(path):
    f=open(path,'rb'); struct.unpack("<III",f.read(12))
    ln,_=struct.unpack("<II",f.read(8)); js=json.loads(f.read(ln))
    bl,_=struct.unpack("<II",f.read(8)); buf=f.read(bl)
    acc,bv=js['accessors'],js['bufferViews']
    def read(i):
        a=acc[i]; v=bv[a['bufferView']]
        off=v.get('byteOffset',0)+a.get('byteOffset',0)
        ct={5126:'<f4',5125:'<u4',5123:'<u2'}[a['componentType']]
        cnt=a['count']*(3 if a['type']=='VEC3' else 1)
        return np.frombuffer(buf,dtype=ct,count=cnt,offset=off).reshape(
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
                   ("CELL COMPLEX","output/model/cellcomplex.glb"),
                   ("VOXEL surface","output/model/scan_surface.glb")):
    T=tris(path)
    if not len(T): print(f"{label}: none"); continue
    e1,e2=T[:,1]-T[:,0], T[:,2]-T[:,0]
    area=0.5*np.linalg.norm(np.cross(e1,e2),axis=1)
    N=3_000_000                                       # dense, so gaps are real
    pick=rng.choice(len(T), N, p=area/area.sum())
    u,v=rng.random(N),rng.random(N)
    fold=u+v>1; u[fold],v[fold]=1-u[fold],1-v[fold]
    M=T[pick,0]+u[:,None]*e1[pick]+v[:,None]*e2[pick]
    d,_=cKDTree(M).query(Q, workers=-1)
    print(f"\n{label}: {len(T):,} triangles, {area.sum():.0f} m2")
    for t in (0.005,0.010,0.020,0.050,0.100):
        print(f"   scanned point within {t*1000:>4.0f} mm of surface: "
              f"{(d<=t).mean()*100:>5.1f}%")
    print(f"   median {np.median(d)*1000:.0f} mm | 90th {np.percentile(d,90)*1000:.0f} mm"
          f" | 99th {np.percentile(d,99)*1000:.0f} mm")
    print(f"   MISSED (>100 mm from any modelled surface): {(d>0.1).mean()*100:.1f}%")
