"""Step 2: register the drawn wall network to the scan, then MEASURE each wall.

The drawing supplies topology -- which walls exist, where they meet, how long
they run. The scan supplies the millimetres -- exact face positions and true
thickness. Where the scan cannot see a wall, the drawing still carries it, so
nothing is silently missing."""
import json, numpy as np, laspy, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
SEG=json.load(open("output/fp_segments.json"))
CELL=0.05
with laspy.open("output/mujammel_structural_v6.las") as r: p=r.read()
P=np.column_stack([p.x,p.y,p.z]).astype(np.float64); z=P[:,2]
H=float(np.percentile(z[z>1.8],99.0))
S=P[(z>1.10)&(z<1.85)]
mins=S[:,:2].min(axis=0); XY=S[:,:2]-mins
nx=int(np.ceil((XY[:,0].max()+CELL)/CELL)); ny=int(np.ceil((XY[:,1].max()+CELL)/CELL))
ij=np.floor(XY/CELL).astype(np.int64)
scan=np.zeros((nx,ny),bool); scan[ij[:,0],ij[:,1]]=True
a,b=np.where(scan); SC=np.column_stack([a,b])*CELL
tree=cKDTree(SC)
def place(mm,rot,mir,dx,dy):
    """drawing pixels -> scan metres, for a given hypothesis"""
    pts=[]
    for s in SEG:
        n=max(int(s['a1']-s['a0'])+1,2)
        t=np.linspace(s['a0'],s['a1'],n)
        if s['dir']=="H": q=np.column_stack([t,np.full(n,s['pos'])])
        else:             q=np.column_stack([np.full(n,s['pos']),t])
        pts.append(q)
    Q=np.vstack(pts).astype(float)
    Q-=Q.mean(axis=0); Q*=mm/1000.0
    if mir: Q[:,0]*=-1
    th=np.deg2rad(rot); R=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
    Q=Q@R.T
    Q+= (SC.mean(axis=0)-Q.mean(axis=0))+np.array([dx,dy])
    return Q
best=None
for rot in (0,90,180,270):
    for mir in (False,True):
        for mm in np.arange(16.6,18.9,0.1):
            Q=place(mm,rot,mir,0,0)
            d,_=tree.query(Q,k=1,workers=4)
            m=float(np.median(d))
            if best is None or m<best[0]: best=(m,rot,mir,mm,0.0,0.0)
m,rot,mir,mm,_,_=best
for _ in range(4):
    for dx in np.arange(-0.5,0.51,0.05):
        for dy in np.arange(-0.5,0.51,0.05):
            for mmv in (mm-0.05,mm,mm+0.05):
                Q=place(mmv,rot,mir,dx,dy)
                d,_=tree.query(Q,k=1,workers=4)
                v=float(np.median(d))
                if v<best[0]: best=(v,rot,mir,mmv,dx,dy)
    m,rot,mir,mm,dx,dy=best
print(f"registration: rot {rot}deg  mirror {mir}  scale {mm:.2f} mm/px  "
      f"shift ({dx*1000:+.0f},{dy*1000:+.0f}) mm")
print(f"  median drawn-wall -> scanned-wall distance {m*1000:.0f} mm")
th=np.deg2rad(rot); R=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
allp=[]
for s in SEG:
    n=max(int(s['a1']-s['a0'])+1,2); t=np.linspace(s['a0'],s['a1'],n)
    allp.append(np.column_stack([t,np.full(n,s['pos'])]) if s['dir']=="H"
                else np.column_stack([np.full(n,s['pos']),t]))
C0=np.vstack(allp).astype(float).mean(axis=0)
def xf(px,py):
    q=np.array([px,py],float)-C0; q*=mm/1000.0
    if mir: q[0]*=-1
    q=R@q
    return q
Q0=place(mm,rot,mir,dx,dy); OFF=Q0.mean(axis=0)-np.vstack([xf(*p_) for p_ in
     np.vstack(allp).astype(float)]).mean(axis=0)
walls=[]
print(f"\n{'#':>3} {'dir':>4} {'drawn L':>8} {'drawn t':>8} {'built t':>8} "
      f"{'face A':>8} {'face B':>8} {'pts':>8}  status")
for i,s in enumerate(sorted(SEG,key=lambda q:-(q['a1']-q['a0'])),1):
    e0=xf(s['a0'],s['pos']) if s['dir']=="H" else xf(s['pos'],s['a0'])
    e1=xf(s['a1'],s['pos']) if s['dir']=="H" else xf(s['pos'],s['a1'])
    e0=e0+OFF; e1=e1+OFF
    horiz=abs(e1[0]-e0[0])>abs(e1[1]-e0[1])
    axis=1 if horiz else 0          # axis = the coordinate held constant
    c=(e0[axis]+e1[axis])/2.0
    lo,hi=sorted([e0[1-axis],e1[1-axis]])
    L=hi-lo
    band=(np.abs(XY[:,axis]-c)<0.30)&(XY[:,1-axis]>lo+0.05)&(XY[:,1-axis]<hi-0.05)
    npts=int(band.sum())
    fa=fb=tb=None
    if npts>=300:
        v=XY[band,axis]
        h,e=np.histogram(v,bins=60,range=(c-0.30,c+0.30)); ctr=0.5*(e[:-1]+e[1:])
        pk=[k for k in range(1,len(h)-1) if h[k]>=h[k-1] and h[k]>=h[k+1] and h[k]>0.22*h.max()]
        f=[]
        for k in sorted(pk,key=lambda q:-h[q]):
            if all(abs(ctr[k]-x)>0.07 for x in f): f.append(float(ctr[k]))
            if len(f)==2: break
        if len(f)==2 and 0.07<abs(f[0]-f[1])<0.40:
            fa,fb=sorted(f); tb=fb-fa
        elif f: fa=f[0]
    st=("both faces" if tb else "one face" if fa is not None else
        "NOT SCANNED - drawing only")
    walls.append(dict(axis=int(axis),c=float(c),lo=float(lo),hi=float(hi),
                      drawn_t=s['thick']*mm/1000.0,fa=fa,fb=fb,t=tb,n=npts))
    print(f"{i:>3} {'X' if axis==0 else 'Y':>4} {L*1000:>7.0f}m {s['thick']*mm:>7.0f}m "
          f"{(tb*1000 if tb else 0):>7.0f}m {(fa if fa else 0):>8.2f} {(fb if fb else 0):>8.2f} "
          f"{npts:>8,}  {st}")
tt=[w['t'] for w in walls if w['t']]
print(f"\n{len(walls)} walls from the drawing; {len(tt)} measured on both faces, "
      f"{sum(1 for w in walls if w['t'] is None and w['fa'] is not None)} on one face, "
      f"{sum(1 for w in walls if w['fa'] is None)} not scanned")
if tt: print(f"measured thickness: median {np.median(tt)*1000:.0f} mm  "
             f"range {min(tt)*1000:.0f}-{max(tt)*1000:.0f}")
json.dump(dict(clear_height=H,mins=mins.tolist(),walls=walls),
          open("output/fp_walls.json","w"),indent=1)
print("wrote output/fp_walls.json")
