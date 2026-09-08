"""Decode the Ec_Data.fmraster 21-byte records (payload = 15 bytes after the
6-byte FM/seq/msgid/len header + h/m/s/ns time = 8 bytes -> 7 payload bytes)."""
import numpy as np
p = "data/Soulace/20260419-104545_Ec_Data.fmraster"
b = np.fromfile(p, dtype=np.uint8)
m = b[1024:].reshape(-1, 21); n = len(m)
h,mi,s = m[:,6].astype(np.int64), m[:,7].astype(np.int64), m[:,8].astype(np.int64)
ns = np.ascontiguousarray(m[:,9:13]).view('<u4').ravel().astype(np.int64)
t = h*3600+mi*60+s+ns/1e9
print(f"nrec={n} t {t[0]:.6f}..{t[-1]:.6f} span={t[-1]-t[0]:.3f}s")
print("first 6 records hex:")
for i in range(6): print("  " + " ".join(f"{x:02x}" for x in m[i]))
print("payload byte13..20 unique counts:", [len(np.unique(m[:200000,c])) for c in range(13,21)])
for w,dt_,nm in ((2,'<u2','u16'),(2,'<i2','i16'),(4,'<u4','u32'),(4,'<i4','i32')):
    for c in range(13, 21-w+1):
        v = np.ascontiguousarray(m[:,c:c+w]).view(dt_).ravel().astype(np.float64)
        print(f"  {nm}@{c}: mean={v.mean():12.2f} std={v.std():11.2f} range[{v.min():.0f},{v.max():.0f}] uniq={len(np.unique(v))}")
d = np.diff(t)
import collections
print("dt histogram (us, top8):", collections.Counter(np.round(d*1e6).astype(int).tolist()).most_common(8))
