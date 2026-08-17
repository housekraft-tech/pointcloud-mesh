"""Wall extraction, second pass -- with a COVERAGE metric so 'did it get them all'
is measured, not eyeballed.

Run-length scanning per row/column instead of morphological opening: opening
erodes the ends of every wall and drops anything a junction interrupts."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage
img=plt.imread("image.png")
g=img[...,:3].mean(axis=2) if img.ndim==3 else img
if g.max()<=1.0: g=g*255.0
ink=(g<120)
# drop compact blobs: north arrow, text, furniture symbols
lab,n=ndimage.label(ink,ndimage.generate_binary_structure(2,2))
heavy=np.zeros_like(ink); drop=0
for b in range(1,n+1):
    ys,xs=np.where(lab==b)
    if len(ys)<40: drop+=1; continue
    h,w=np.ptp(ys)+1,np.ptp(xs)+1
    if max(h,w)<70 and len(ys)/float(h*w)>0.40: drop+=1; continue
    heavy|=(lab==b)
print(f"ink {ink.sum():,} px -> wall linework {heavy.sum():,} px ({drop} blobs dropped)")
MINRUN=22
def runs(M,axis):
    """runs of ink >= MINRUN along each line"""
    out=[]
    A=M if axis==0 else M.T
    for i in range(A.shape[0]):
        line=A[i]; k=0
        while k<len(line):
            if not line[k]: k+=1; continue
            j=k
            while j+1<len(line) and line[j+1]: j+=1
            if j-k+1>=MINRUN: out.append((i,k,j))
            k=j+1
    return out
segs=[]
for name,axis in (("H",0),("V",1)):
    rs=runs(heavy,axis)
    groups=[]
    for i,k,j in rs:
        hit=None
        for gp in groups:
            if abs(gp['i1']-i)<=1 and not (j<gp['k']-6 or k>gp['j']+6):
                hit=gp; break
        if hit:
            hit['i1']=i; hit['k']=min(hit['k'],k); hit['j']=max(hit['j'],j); hit['n']+=1
        else:
            groups.append(dict(i0=i,i1=i,k=k,j=j,n=1))
    for gp in groups:
        thick=gp['i1']-gp['i0']+1; length=gp['j']-gp['k']+1
        if thick<3 or thick>34 or length<MINRUN: continue
        segs.append(dict(dir=name,pos=(gp['i0']+gp['i1'])/2.0,a0=gp['k'],a1=gp['j'],
                         thick=thick,length=length))
def merge(S):
    out=[]
    for s in sorted(S,key=lambda q:-q['length']):
        hit=None
        for o in out:
            if o['dir']==s['dir'] and abs(o['pos']-s['pos'])<=7 and \
               not (s['a1']<o['a0']-10 or s['a0']>o['a1']+10): hit=o; break
        if hit:
            hit['a0']=min(hit['a0'],s['a0']); hit['a1']=max(hit['a1'],s['a1'])
            hit['thick']=max(hit['thick'],s['thick'])
            hit['length']=hit['a1']-hit['a0']+1
        else: out.append(dict(s))
    return out
segs=merge(segs)
MM=17.6
# ---- COVERAGE: how much of the drawn wall linework did we actually capture? ----
cov=np.zeros_like(heavy)
for s in segs:
    t=max(int(round(s['thick']/2)),2)
    p=int(round(s['pos']))
    if s['dir']=="H": cov[max(0,p-t):p+t+1, s['a0']:s['a1']+1]=True
    else:             cov[s['a0']:s['a1']+1, max(0,p-t):p+t+1]=True
hit=(heavy&cov).sum(); miss=(heavy&~cov).sum()
print(f"\n{len(segs)} wall segments, total {sum(s['length'] for s in segs)*MM/1000:.1f} m")
print(f"COVERAGE {hit/max(heavy.sum(),1)*100:.1f}% of wall linework captured "
      f"({miss:,} px missed)")
lab2,n2=ndimage.label(heavy&~cov,ndimage.generate_binary_structure(2,2))
sz=ndimage.sum(heavy&~cov,lab2,range(1,n2+1))
big=[b for b in np.argsort(sz)[::-1][:6] if sz[b]>=120]
if big:
    print("largest missed pieces:")
    for b in big:
        ys,xs=np.where(lab2==b+1)
        print(f"   {int(sz[b]):>5} px  {(np.ptp(xs)+1)*MM:>6.0f} x {(np.ptp(ys)+1)*MM:>6.0f} mm"
              f"  at ({xs.mean():.0f},{ys.mean():.0f})")
T=np.array([s['thick']*MM for s in segs])
print(f"\nthickness median {np.median(T):.0f} mm  range {T.min():.0f}-{T.max():.0f}")
import json
json.dump([dict(dir=s['dir'],pos=float(s['pos']),a0=int(s['a0']),a1=int(s['a1']),
                thick=int(s['thick'])) for s in segs],
          open("output/fp_segments.json","w"),indent=1)
fig,ax=plt.subplots(1,3,figsize=(24,8))
ax[0].imshow(heavy,cmap='gray_r'); ax[0].set_title(f"wall linework ({heavy.sum():,} px)")
ax[1].imshow(np.zeros_like(heavy),cmap='gray_r')
for s in segs:
    if s['dir']=="H": ax[1].plot([s['a0'],s['a1']],[s['pos']]*2,'-',lw=2.4,c='crimson')
    else:             ax[1].plot([s['pos']]*2,[s['a0'],s['a1']],'-',lw=2.4,c='#0b62d0')
ax[1].set_title(f"{len(segs)} segments — {sum(s['length'] for s in segs)*MM/1000:.1f} m")
ax[1].set_xlim(0,heavy.shape[1]); ax[1].set_ylim(heavy.shape[0],0)
ov=np.zeros(heavy.shape+(3,)); ov[...,0]=(heavy&~cov); ov[...,1]=(heavy&cov)
ax[2].imshow(1-ov); ax[2].set_title(f"green captured / red missed — {hit/max(heavy.sum(),1)*100:.1f}%")
for a in ax: a.set_xticks([]); a.set_yticks([])
fig.tight_layout(); fig.savefig("output/viz/40_fp_segments2.png",dpi=110)
print("wrote output/viz/40_fp_segments2.png and output/fp_segments.json")
