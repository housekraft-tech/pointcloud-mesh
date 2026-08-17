"""Step 3: build the modular shell FROM the drawn wall network.

Topology comes from the drawing, so walls have exact endpoints, meet at corners,
and none go missing. Millimetres come from the scan: face positions, thickness,
and every opening measured off the vertical profile along each wall.
Also emits the floor and the ceiling, which the previous shell lacked entirely."""
import sys, json, numpy as np, laspy
sys.path.insert(0,"/private/tmp/claude-501/-Users-vallerikoushik-Documents-pointcloud-latest-pointcloud-mesh/7d4da44b-305a-4a79-a748-5cb594356906/scratchpad")
from glb import GLB
from scipy import ndimage
W=json.load(open("output/fp_walls.json"))
H=W['clear_height']; mins=np.array(W['mins']); walls=W['walls']
CELL=0.05
with laspy.open("output/mujammel_structural_v6.las") as r: p=r.read()
P=np.column_stack([p.x,p.y,p.z]).astype(np.float64); z=P[:,2]
# The wall coordinates are in a frame whose origin is the BODY-height slice.
# The full cloud extends outside that (floor and ceiling reach further), so
# clipping negatives to zero piles them on the edge and corrupts every layer.
# Keep the frame, drop what falls outside it.
XY=P[:,:2]-mins
nx=int(np.ceil((XY[:,0].max()+CELL)/CELL)); ny=int(np.ceil((XY[:,1].max()+CELL)/CELL))
inside=(XY[:,0]>=0)&(XY[:,1]>=0)
ij=np.floor(XY/CELL).astype(np.int64)
ij[:,0]=np.clip(ij[:,0],0,nx-1); ij[:,1]=np.clip(ij[:,1],0,ny-1)
flat=ij[:,0]*ny+ij[:,1]
def occ(m):
    g=np.zeros(nx*ny,bool); g[flat[m&inside]]=True; return g.reshape(nx,ny)
body=occ((z>1.10)&(z<1.85))
# The above-head band must start low enough to catch soffited rooms: the wet
# areas are ceiled at ~2410 mm, so a band starting at 2250 finds nothing there
# and every opening in those rooms is missed. 2.10-2.65 lifts coverage 60%->86%.
above=occ((z>2.10)&(z<H-0.10))
DEF_T=0.200
for w in walls:
    # thickness: measured where both faces were seen, else the drawn value
    w['t']=w['t'] if w['t'] else (w['drawn_t'] if w['drawn_t']>0.08 else DEF_T)
    # centre: midpoint of measured faces, else the single face pushed in by t/2,
    # else the registered drawn centreline
    if w['fa'] is not None and w['fb'] is not None: w['cc']=(w['fa']+w['fb'])/2
    elif w['fa'] is not None:
        w['cc']=w['fa']+(w['t']/2 if w['fa']<w['c'] else -w['t']/2)
    else: w['cc']=w['c']
    w['src']=("measured both faces" if w['fb'] is not None else
              "one face + drawn thickness" if w['fa'] is not None else "drawing only")
# ---- openings: walk each wall and read the vertical profile ----
def profile(w):
    """Occupancy along a wall, read off its FACE PLANES -- not its centreline.

    A doorway's reveals are scanned surfaces sitting at the centreline, so a
    centreline test reports 'wall present' straight through every door. The face
    planes are what a door actually interrupts.
    """
    axis=w['axis']; lo,hi=w['lo'],w['hi']
    faces=[f for f in (w['fa'],w['fb']) if f is not None] or [w['c']]
    n=max(int((hi-lo)/CELL),1)
    bod=np.zeros(n,bool); abv=np.zeros(n,bool)
    for f in faces:
        near=inside&(np.abs(XY[:,axis]-f)<0.05)
        al=XY[near,1-axis]; zz=z[near]
        k=np.floor((al-lo)/CELL).astype(np.int64)
        ok=(k>=0)&(k<n)
        mb=ok&(zz>1.10)&(zz<1.85)
        ma=ok&(zz>2.10)&(zz<H-0.10)
        kb=k[mb]; ka=k[ma]
        if len(kb): bod[np.unique(kb)]=True
        if len(ka): abv[np.unique(ka)]=True
    return bod,abv
def zprof(axis,c,a,b_):
    """Measure the void directly instead of hunting for an empty histogram run.
    A narrow band is essential: at +/-160 mm the door reveal and the wall beside
    it fill every height, so the column never looks empty and no door is found.
      head = underside of the lintel  = lowest structure above 1.9 m
      sill = top of the wall below it = highest structure under 1.9 m (0 for a door)
    """
    m=inside&(np.abs(XY[:,axis]-c)<0.08)&(XY[:,1-axis]>a+0.05)&(XY[:,1-axis]<b_-0.05)
    if m.sum()<30: return None,None
    zz=z[m]
    hi_=zz[zz>1.90]
    if len(hi_)<20: return None,None
    head=float(np.percentile(hi_,3))
    lo_=zz[zz<1.90]
    sill=float(np.percentile(lo_,97)) if len(lo_)>=40 else 0.0
    if sill<0.30: sill=0.0
    if head-sill<0.8: return None,None
    return sill,head
G=GLB(); nop=0; kinds={}
def yup(axis,c,t,L,R,z0,z1):
    if axis==0: xl,xh,yl,yh=c-t/2,c+t/2,L,R
    else:       xl,xh,yl,yh=L,R,c-t/2,c+t/2
    return (xl+mins[0],z0,-(yh+mins[1])),(xh+mins[0],z1,-(yl+mins[1]))
for wi,w in enumerate(sorted(walls,key=lambda q:-(q['hi']-q['lo'])),1):
    axis=w['axis']; c=w['cc']; lo,hi=w['lo'],w['hi']; t=w['t']
    bod,abv=profile(w)
    ops=[]; k=0
    while k<len(bod):
        if bod[k]: k+=1; continue
        j=k
        while j+1<len(bod) and not bod[j+1]: j+=1
        L=(j-k+1)*CELL
        if L>=0.45 and abv[k:j+1].mean()>0.35:      # roof over it => a real opening
            a=lo+k*CELL; b_=lo+(j+1)*CELL
            sill,head=zprof(axis,c,a,b_)
            if sill is not None and head is not None and head>1.9:
                kind=("window" if sill>0.35 else
                      "door" if 0.70<=L<=1.00 else
                      "passage")
                ops.append(dict(a=a,b=b_,sill=sill,head=head,kind=kind))
                kinds[kind]=kinds.get(kind,0)+1; nop+=1
        k=j+1
    xs=sorted({lo,hi}|{v for o in ops for v in (o['a'],o['b'])})
    acc=G.new_group()
    for q in range(len(xs)-1):
        A,B=xs[q],xs[q+1]
        if B-A<0.02: continue
        cov=sorted([o for o in ops if o['a']<B-1e-6 and o['b']>A+1e-6],key=lambda o:o['sill'])
        if not cov:
            a2,b2=yup(axis,c,t,A,B,0.0,H); G.add_box("",a2,b2,acc); continue
        zc=0.0
        for o in cov:
            if o['sill']>zc+0.02:
                a2,b2=yup(axis,c,t,A,B,zc,o['sill']); G.add_box("",a2,b2,acc)
            zc=max(zc,o['head'])
        if zc<H-0.02:
            a2,b2=yup(axis,c,t,A,B,zc,H); G.add_box("",a2,b2,acc)
    tag="".join(sorted({o['kind'][0].upper() for o in ops}))
    G.add_group(f"WALL_{wi:02d}_{'X' if axis==0 else 'Y'}_t{t*1000:.0f}"
                f"{'_'+tag if tag else ''}",acc)
    w['ops']=ops
# ---- floor and ceiling ----
fx0,fx1=float(XY[:,0].min()+mins[0]),float(XY[:,0].max()+mins[0])
fy0,fy1=float(XY[:,1].min()+mins[1]),float(XY[:,1].max()+mins[1])
G.add_box("FLOOR",(fx0,-0.05,-fy1),(fx1,0.0,-fy0))
G.add_box("CEILING",(fx0,H,-fy1),(fx1,H+0.05,-fy0))
# ---- beams ----
ceil=occ(z>H-0.20)
beam=occ((z>H-1.0)&(z<H-0.12))&~body&ndimage.binary_dilation(ceil,np.ones((3,3),bool))
beam=ndimage.binary_opening(beam,np.ones((3,3),bool))
lab,n=ndimage.label(beam,ndimage.generate_binary_structure(2,2)); nb=0
for b_ in range(1,n+1):
    xs_,ys_=np.where(lab==b_)
    if len(xs_)<40: continue
    x0=mins[0]+xs_.min()*CELL; x1=mins[0]+(xs_.max()+1)*CELL
    y0=mins[1]+ys_.min()*CELL; y1=mins[1]+(ys_.max()+1)*CELL
    if max(x1-x0,y1-y0)<0.8: continue
    m=(P[:,0]>x0)&(P[:,0]<x1)&(P[:,1]>y0)&(P[:,1]<y1)&(z>H-1.05)
    if m.sum()<400: continue
    so=float(np.percentile(z[m],2))
    if H-so<0.15: continue
    nb+=1; G.add_box(f"BEAM_{nb:02d}_drop{(H-so)*1000:.0f}",(x0,so,-y1),(x1,H,-y0))
sz,parts,tris=G.write("output/model/shell_fp.glb")
print(f"clear height {H*1000:.0f} mm")
print(f"walls {len(walls)} (all from the drawing, none missing)")
for s in ("measured both faces","one face + drawn thickness","drawing only"):
    print(f"   {s:<28} {sum(1 for w in walls if w['src']==s)}")
print(f"openings {nop}: " + ", ".join(f"{k} {v}" for k,v in sorted(kinds.items())))
print(f"beams {nb}   + floor + ceiling")
print(f"\nwrote output/model/shell_fp.glb  {sz/1e6:.2f} MB, {parts} parts, {tris:,} triangles")
json.dump(dict(clear_height_mm=H*1000,walls=[
    dict(id=i,axis=w['axis'],centre=w['cc'],lo=w['lo'],hi=w['hi'],
         length_mm=(w['hi']-w['lo'])*1000,thickness_mm=w['t']*1000,source=w['src'],
         openings=[dict(kind=o['kind'],width_mm=(o['b']-o['a'])*1000,
                        sill_mm=o['sill']*1000,head_mm=o['head']*1000,
                        arch_mm=(H-o['head'])*1000) for o in w.get('ops',[])])
    for i,w in enumerate(sorted(walls,key=lambda q:-(q['hi']-q['lo'])),1)]),
    open("output/model/shell_fp.json","w"),indent=1)
print("wrote output/model/shell_fp.json")
