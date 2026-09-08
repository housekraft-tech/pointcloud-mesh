"""Decode the Hp ('feima-i2000-imu') stream.
Hypothesised 33-byte record: 'FM' seq(u8) msgid(u8) tag(u8) ts_u24_us  6x[val_u24_signed, tag_u8]  csum(u8)
Also classifies the minority record sizes. Read-only."""
import numpy as np, collections
p = "data/Soulace/20260419-104545_Hp_Imu.fmimr"
b = np.fromfile(p, dtype=np.uint8)
idx = np.where((b[:-1]==0x46)&(b[1:]==0x4d))[0]
idx = idx[idx >= 1024]
d = np.diff(idx)
print("FM spacing histogram:", collections.Counter(d.tolist()).most_common(8))
# keep only FM positions that begin a stride-33 run (both neighbours 33 apart)
starts = idx[:-1][d == 33]
print(f"stride-33 record starts: {len(starts)}  (file {b.size} B)")
m = np.stack([b[o:o+33] for o in starts])
print("msgid unique:", np.unique(m[:,3]), " byte4 tag unique:", np.unique(m[:,4]))
def u24(cols):
    x = cols[:,0].astype(np.int64) | (cols[:,1].astype(np.int64)<<8) | (cols[:,2].astype(np.int64)<<16)
    return x
def s24(cols):
    x = u24(cols); return np.where(x >= 1<<23, x - (1<<24), x)
ts = u24(m[:,5:8])
dt = np.diff(ts) % (1<<24)
u,c = np.unique(dt, return_counts=True)
print("ts step top6:", sorted(zip(c,u))[-6:])
print(f"ts: first={ts[0]} last={ts[-1]}  implied duration = {dt.sum()/1e6:.3f} s at 1 us/tick"
      f"  -> rate {len(m)/(dt.sum()/1e6):.2f} Hz")
print("per-slot high tag byte (should be constant per slot):")
for k in range(6):
    o = 8 + 4*k
    tags = np.unique(m[:,o+3])
    v = s24(m[:, o:o+3]).astype(float)
    print(f"  slot{k}: tag(s)={tags}  mean={v.mean():12.1f} std={v.std():11.1f} range[{v.min():.0f},{v.max():.0f}]")
for grp,label in (((0,1,2),"slots012"), ((3,4,5),"slots345")):
    vs = np.stack([s24(m[:, 8+4*k:8+4*k+3]).astype(float) for k in grp], 1)
    mag = np.linalg.norm(vs, axis=1)
    print(f"  |{label}| mean={mag.mean():.1f} std={mag.std():.1f}")
