"""Honest accuracy audit: distance from EVERY scanned point to the model.

align_check only reports faces on walls seen from both sides. That is about
half the model, and it is the easy half. This measures the other question:
for every point the scanner actually returned, how far is the nearest surface
the model provides? A point with no surface near it is geometry the model does
not have.
"""
import struct, json, numpy as np, laspy
J=json.load(open("output/fp_walls.json")); H=J['clear_height']
with laspy.open("output/mujammel_aligned_z0.las") as r: p=r.read()
P=np.column_stack([p.x,p.y,p.z]).astype(np.float64)
k=np.load("output/keep_mask.npy")
if len(k)==len(P): P=P[k]
P=P[P[:,2] < H-0.12]
rng=np.random.default_rng(0)
S=P[rng.choice(len(P), 400_000, replace=False)]
Q=np.column_stack([S[:,0], S[:,2], -S[:,1]])          # into the glTF frame

def boxes(path):
    f=open(path,'rb'); struct.unpack("<III",f.read(12))
    ln,_=struct.unpack("<II",f.read(8)); js=json.loads(f.read(ln))
    acc=js['accessors']; out=[]; names=[]
    for ni,n in enumerate(js.get('nodes',[])):
        if 'mesh' not in n: continue
        me=js['meshes'][n['mesh']]
        for pr in me['primitives']:
            a=acc[pr['attributes']['POSITION']]
            if 'min' in a and 'max' in a:
                out.append((a['min'],a['max'])); names.append(n.get('name',''))
    return np.array([b[0] for b in out]), np.array([b[1] for b in out]), names

def dist_to_boxes(Q, lo, hi, chunk=20000):
    """|signed distance| from each point to the nearest box SURFACE."""
    best=np.full(len(Q), np.inf)
    for s in range(0, len(Q), chunk):
        q=Q[s:s+chunk][:,None,:]                       # (c,1,3)
        d_out=np.maximum(np.maximum(lo[None]-q, q-hi[None]), 0.0)
        out=np.sqrt((d_out**2).sum(-1))                # 0 when inside
        inside=(q>=lo[None]).all(-1) & (q<=hi[None]).all(-1)
        d_in=np.minimum(q-lo[None], hi[None]-q).min(-1)  # to nearest face
        d=np.where(inside, d_in, out)
        best[s:s+chunk]=d.min(1)
    return best

for label,path in (("MODULAR model","output/model/shell_fp.glb"),
                   ("CELL COMPLEX","output/model/cellcomplex.glb"),
                   ("VOXEL surface","output/model/scan_surface.glb")):
    try:
        lo,hi,names=boxes(path)
    except Exception as e:
        print(f"{label}: {e}"); continue
    if len(lo)==0: print(f"{label}: no boxes"); continue
    # the voxel mesh is 6 huge parts, so its AABBs are meaningless -- skip
    if len(lo) < 20 and 'cell' not in path:
        print(f"\n{label}: {len(lo)} parts -- AABBs are whole-scene, not "
              f"per-element, so a box test says nothing. Skipped.")
        continue
    d=dist_to_boxes(Q, lo, hi)
    print(f"\n{label}: {len(lo)} parts, {len(Q):,} scanned points tested")
    for t in (0.005,0.010,0.020,0.050,0.100,0.200,0.500):
        print(f"   within {t*1000:>4.0f} mm of a modelled surface: "
              f"{(d<=t).mean()*100:>5.1f}%")
    print(f"   median {np.median(d)*1000:.0f} mm | 90th {np.percentile(d,90)*1000:.0f} mm"
          f" | 99th {np.percentile(d,99)*1000:.0f} mm | worst {d.max()*1000:.0f} mm")
