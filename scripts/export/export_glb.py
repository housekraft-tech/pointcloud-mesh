"""Emit the modular building shell as GLB, in the target file's conventions:
Y-up, metres, one named node per element, flat-shaded boxes with crisp edges.

Elements: walls (openings cut out), the grooves/intrusions and extrusions measured
off each wall face, beams and arches hanging from the ceiling, and columns.
Furniture is not modelled -- this is the bare shell."""
import sys, numpy as np, laspy, json
sys.path.insert(0,"/private/tmp/claude-501/-Users-vallerikoushik-Documents-pointcloud-latest-pointcloud-mesh/7d4da44b-305a-4a79-a748-5cb594356906/scratchpad")
from glb import GLB
from scipy import ndimage
CELL=0.05
with laspy.open("output/mujammel_structural_v6.las") as r: p=r.read()
P=np.column_stack([p.x,p.y,p.z]).astype(np.float64); z=P[:,2]
H=float(np.percentile(z[z>1.8],99.0))
mins=P[:,:2].min(axis=0); ij=np.floor((P[:,:2]-mins)/CELL).astype(np.int64)
nx,ny=ij[:,0].max()+1,ij[:,1].max()+1; flat=ij[:,0]*ny+ij[:,1]
def occ(m):
    g=np.zeros(nx*ny,bool); g[flat[m]]=True; return g.reshape(nx,ny)
body=occ((z>1.10)&(z<1.85))
# ---- wall lines, paired into walls of measured thickness ----
lines=[]
for axis in (0,1):
    B=body if axis==0 else body.T
    cnt=B.sum(axis=1); thr=max(12,int(0.10*B.shape[1]))
    cand=np.flatnonzero(cnt>=thr)
    for grp in np.split(cand,np.flatnonzero(np.diff(cand)>2)+1):
        if not len(grp): continue
        i=int(grp[np.argmax(cnt[grp])]); pos=mins[axis]+i*CELL
        sel=np.abs(P[:,axis]-pos)<0.30
        if sel.sum()<1500: continue
        al=P[sel,1-axis]; a0,a1=np.percentile(al,[0.3,99.7])
        lines.append(dict(axis=axis,pos=float(pos),a0=float(a0),a1=float(a1),n=int(sel.sum())))
lines.sort(key=lambda w:-w['n']); lines=lines[:16]
used=set(); walls=[]
for i,L in enumerate(lines):
    if i in used: continue
    best=None
    for j,M in enumerate(lines):
        if j<=i or j in used or M['axis']!=L['axis']: continue
        t=abs(M['pos']-L['pos']); ov=min(L['a1'],M['a1'])-max(L['a0'],M['a0'])
        if 0.08<t<0.45 and ov>1.0 and (best is None or t<best[1]): best=(j,t)
    if best:
        j,t=best; used|={i,j}; M=lines[j]
        walls.append(dict(axis=L['axis'],c=(L['pos']+M['pos'])/2,th=float(t),
                          a0=max(L['a0'],M['a0']),a1=min(L['a1'],M['a1']),paired=True))
    else:
        used.add(i); walls.append(dict(axis=L['axis'],c=L['pos'],th=0.200,
                          a0=L['a0'],a1=L['a1'],paired=False))
# ---- openings, gated ----
ALL=json.load(open("output/openings.json")); OP=[]
for r in ALL:
    if not r.get('jambs'): continue
    if r['sill']>=0.35:
        if r['w']>=0.40: r['kind']="window"; OP.append(r)
        continue
    if r['w']<0.70 or not (1.95<=r['head']<=2.35): continue
    w=r['w']*1000
    r['kind']=("balcony_door" if r.get('ext') and w<=1600 else
               "balcony_opening" if r.get('ext') else
               "door" if (700<=w<=1000 and r['wk']) else "passage")
    OP.append(r)
for w in walls: w['op']=[]
for r in OP:
    c=r['x0'] if r['ax']==0 else r['y0']
    s0,s1=(r['y0'],r['y1']) if r['ax']==0 else (r['x0'],r['x1'])
    cd=[w for w in walls if w['axis']==r['ax'] and abs(w['c']-c)<0.40
        and min(s0,s1)>=w['a0']-0.3 and max(s0,s1)<=w['a1']+0.3]
    if cd: min(cd,key=lambda w:abs(w['c']-c))['op'].append(
        dict(s0=min(s0,s1),s1=max(s0,s1),sill=r['sill'],head=r['head'],kind=r['kind']))
def partition(a0,a1,H,ops):
    xs=sorted({a0,a1}|{v for o in ops for v in (o['s0'],o['s1'])})
    xs=[v for v in xs if a0-1e-6<=v<=a1+1e-6]; out=[]
    for k in range(len(xs)-1):
        L,R=xs[k],xs[k+1]
        if R-L<0.02: continue
        cov=sorted([o for o in ops if o['s0']<R-1e-6 and o['s1']>L+1e-6],key=lambda o:o['sill'])
        if not cov: out.append((L,R,0.0,H)); continue
        zc=0.0
        for o in cov:
            if o['sill']>zc+0.02: out.append((L,R,zc,o['sill']))
            zc=max(zc,o['head'])
        if zc<H-0.02: out.append((L,R,zc,H))
    return out
def yup(axis,c,th,L,R,z0,z1):
    """our Z-up (x,y,z) -> glTF Y-up (x, z, -y)"""
    if axis==0: xlo,xhi,ylo,yhi=c-th/2,c+th/2,L,R
    else:       xlo,xhi,ylo,yhi=L,R,c-th/2,c+th/2
    return (xlo,z0,-yhi),(xhi,z1,-ylo)
G=GLB(); stats={}
# ---- walls (one node each, openings cut) ----
for wi,w in enumerate(sorted(walls,key=lambda q:-(q['a1']-q['a0'])),1):
    acc=G.new_group()
    for (L,R,z0,z1) in partition(w['a0'],w['a1'],H,w['op']):
        lo,hi=yup(w['axis'],w['c'],w['th'],L,R,z0,z1)
        G.add_box("",lo,hi,acc)
    kinds="".join(sorted({o['kind'][0].upper() for o in w['op']}))
    G.add_group(f"WALL_{wi:02d}_{'X' if w['axis']==0 else 'Y'}"
                f"_t{w['th']*1000:.0f}{'_'+kinds if kinds else ''}",acc)
stats['walls']=len(walls)
# ---- relief on each wall face: grooves (in) and extrusions (out) ----
nrel=0
for wi,w in enumerate(sorted(walls,key=lambda q:-(q['a1']-q['a0'])),1):
    axis=w['axis']
    for side in (-1,+1):
        # Locate the face by the MODE of the offset histogram, not by
        # centre +/- thickness/2. The geometric guess misses the real surface,
        # and then the whole face reads as one giant extrusion.
        nom=w['c']+side*w['th']/2
        near=(np.abs(P[:,axis]-nom)<0.16)&(P[:,1-axis]>w['a0'])&(P[:,1-axis]<w['a1'])
        if near.sum()<3000: continue
        hh,ee=np.histogram(P[near,axis]-nom,bins=64,range=(-0.16,0.16))
        face=nom+0.5*(ee[hh.argmax()]+ee[hh.argmax()+1])
        sel=(np.abs(P[:,axis]-face)<0.09)&(P[:,1-axis]>w['a0'])&(P[:,1-axis]<w['a1'])
        Q=P[sel]
        if len(Q)<3000: continue
        al=Q[:,1-axis]; up=Q[:,2]; off=(Q[:,axis]-face)*side*-1000  # +ve = into the wall
        na=max(int((w['a1']-w['a0'])/CELL)+1,2); nz=int(H/CELL)+1
        ai=np.clip(((al-w['a0'])/CELL).astype(int),0,na-1)
        zi=np.clip((up/CELL).astype(int),0,nz-1)
        acc=np.full(na*nz,np.nan); k=ai*nz+zi
        o=np.argsort(k); ks=k[o]; ov=off[o]
        for grp in np.split(np.arange(len(ks)),np.flatnonzero(np.diff(ks))+1):
            acc[ks[grp[0]]]=np.median(ov[grp])
        D=acc.reshape(na,nz); pres=np.isfinite(D); rel=D-np.nanmedian(D)
        # the site's relief is a rectangular step of about 75 mm; anything beyond
        # 150 mm is a different wall, not a bump on this one
        for tag,mask,sgn in (("GROOVE",pres&(rel>40)&(rel<150),+1),
                             ("EXTRUSION",pres&(rel<-40)&(rel>-150),-1)):
            lab,n=ndimage.label(mask,ndimage.generate_binary_structure(2,2))
            for b in range(1,n+1):
                a_i,z_i=np.where(lab==b)
                if len(a_i)<24: continue
                Lm,Rm=w['a0']+a_i.min()*CELL, w['a0']+(a_i.max()+1)*CELL
                z0,z1=z_i.min()*CELL,(z_i.max()+1)*CELL
                if (Rm-Lm)<0.15 or (z1-z0)<0.15: continue
                # a genuine step is compact; a sliver spanning the whole wall is
                # a mis-located face, not a bump
                if (Rm-Lm)>0.75*(w['a1']-w['a0']) and (z1-z0)>0.75*H: continue
                d=float(np.nanmedian(np.abs(rel[lab==b])))/1000.0
                d=min(max(d,0.02),0.12)
                if sgn>0: f0,f1=face-side*d,face          # groove: cut into the wall
                else:     f0,f1=face,face+(-side)*d       # extrusion: stands proud
                lo=min(f0,f1); hi=max(f0,f1)
                a2,b2=yup(axis,(lo+hi)/2,hi-lo,Lm,Rm,z0,z1)
                nrel+=1
                G.add_box(f"{tag}_W{wi:02d}_{'A' if side<0 else 'B'}{nrel:03d}",a2,b2)
stats['relief']=nrel
# ---- beams / arches hanging from the ceiling ----
hi_pts=(z>H-1.00)&(z<H-0.12)
gh=occ(hi_pts); ceil=occ(z>H-0.20)
beam=gh&~body&ndimage.binary_dilation(ceil,np.ones((3,3),bool))
beam=ndimage.binary_opening(beam,np.ones((3,3),bool))
lab,n=ndimage.label(beam,ndimage.generate_binary_structure(2,2))
nb=0
for b in range(1,n+1):
    xs_,ys_=np.where(lab==b)
    if len(xs_)<40: continue
    x0=mins[0]+xs_.min()*CELL; x1=mins[0]+(xs_.max()+1)*CELL
    y0=mins[1]+ys_.min()*CELL; y1=mins[1]+(ys_.max()+1)*CELL
    if max(x1-x0,y1-y0)<0.8: continue
    m=(P[:,0]>x0)&(P[:,0]<x1)&(P[:,1]>y0)&(P[:,1]<y1)&(z>H-1.05)
    if m.sum()<400: continue
    soffit=float(np.percentile(z[m],2))
    if H-soffit<0.15: continue
    nb+=1
    G.add_box(f"BEAM_{nb:02d}_drop{(H-soffit)*1000:.0f}",
              (x0,soffit,-y1),(x1,H,-y0))
stats['beams']=nb
sz,parts,tris=G.write("output/model/shell.glb")
print(f"clear height {H*1000:.0f} mm")
print(f"walls {stats['walls']}  relief parts {stats['relief']}  beams {stats['beams']}")
print(f"openings cut: {sum(len(w['op']) for w in walls)}")
from collections import Counter
print("  " + ", ".join(f"{k} {v}" for k,v in
      Counter(o['kind'] for w in walls for o in w['op']).items()))
print(f"\nwrote output/model/shell.glb  {sz/1e6:.2f} MB, {parts} named parts, {tris:,} triangles")
print(f"  (target file: 38 shell elements; a plain box = 12 triangles)")
