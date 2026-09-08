"""Walk 20260419-104545_Lidar_Imu.imu using its 'a2 a7 18' record magic (raw Livox
UDP capture: the payload embeds the sensor IP 192.168.1.100 = c0 a8 01 64)."""
import numpy as np, collections
p = "data/Soulace/20260419-104545_Lidar_Imu.imu"
b = np.fromfile(p, dtype=np.uint8)
idx = np.where((b[:-2]==0xa2)&(b[1:-1]==0xa7)&(b[2:]==0x18))[0]
idx = idx[idx >= 1024]
d = np.diff(idx)
print(f"'a2 a7 18' magic: {len(idx)} hits, first {idx[:4]}")
print("spacing histogram:", collections.Counter(d.tolist()).most_common(6))
starts = idx[:-1][d == 51]
print(f"records with 51-spacing: {len(starts)} / {len(idx)}")
m = np.stack([b[o:o+51] for o in starts])
n = len(m)
f = np.ascontiguousarray(m[:, 15:39]).view('<f4').reshape(n, 6).astype(np.float64)
ok = np.all(np.isfinite(f), 1) & (np.abs(f).max(1) < 50)
print(f"records with sane float32 block: {ok.sum()}/{n} ({ok.mean()*100:.2f}%)")
f = f[ok]
for i, nm in enumerate("gyro_x gyro_y gyro_z acc_x acc_y acc_z".split()):
    print(f"  {nm}: mean={f[:,i].mean():10.5f} std={f[:,i].std():9.5f} range[{f[:,i].min():8.3f},{f[:,i].max():8.3f}]")
mag = np.linalg.norm(f[:,3:], axis=1)
print(f"  |accel| mean={mag.mean():.5f} std={mag.std():.5f}  (units of g if ~1.0)")
print(f"  ip bytes @40..43 unique rows: {np.unique(m[ok][:,40:44],axis=0)[:3]}")
ts = (m[ok][:,46].astype(np.int64) | (m[ok][:,47].astype(np.int64)<<8) |
      (m[ok][:,48].astype(np.int64)<<16) | (m[ok][:,49].astype(np.int64)<<24) |
      (m[ok][:,50].astype(np.int64)<<32))
dts = np.diff(ts)
pos = dts[dts > 0]
print(f"  5-byte LE field @46: first={ts[0]} last={ts[-1]} frac_increasing={np.mean(dts>0):.4f}")
print(f"    median step={np.median(pos):.0f} ns -> {1e9/np.median(pos):.2f} Hz ; "
      f"total span={(ts.max()-ts.min())/1e9:.3f} s")
