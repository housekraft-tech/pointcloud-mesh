"""Decode 20260419-104545_Lidar_Imu.imu (Livox built-in ICM40609 IMU).
Derived: 1152-byte header, 51-byte fixed records, 6x float32 at offset 15..38
(gyro xyz rad/s, accel xyz g). Read-only."""
import numpy as np
p = "data/Soulace/20260419-104545_Lidar_Imu.imu"
b = np.fromfile(p, dtype=np.uint8)
HDR, STRIDE = 1152, 51
assert (b.size - HDR) % STRIDE == 0, (b.size - HDR) % STRIDE
n = (b.size - HDR) // STRIDE
m = b[HDR:].reshape(n, STRIDE)
print(f"size={b.size} header={HDR} stride={STRIDE} nrec={n} leftover=0")
print("first 3 records hex:")
for i in range(3): print("  " + " ".join(f"{x:02x}" for x in m[i]))
print("per-column unique counts:", [len(np.unique(m[:20000,c])) for c in range(STRIDE)])
f = np.ascontiguousarray(m[:, 15:39]).view('<f4').reshape(n, 6).astype(np.float64)
for i, nm in enumerate("gx gy gz ax ay az".split()):
    print(f"  {nm}: mean={f[:,i].mean():10.5f} std={f[:,i].std():9.5f} range[{f[:,i].min():.3f},{f[:,i].max():.3f}]")
print(f"  |accel| mean={np.linalg.norm(f[:,3:],axis=1).mean():.5f} g  std={np.linalg.norm(f[:,3:],axis=1).std():.5f}")
print(f"  |gyro|  mean={np.linalg.norm(f[:,:3],axis=1).mean():.5f} rad/s")
# hunt for the timestamp
for w, dt_, nm in ((8,'<u8','u64'), (4,'<u4','u32'), (8,'<f8','f64')):
    for c in list(range(0, 16)) + list(range(39, STRIDE - w + 1)):
        if c + w > STRIDE: continue
        v = np.ascontiguousarray(m[:, c:c+w]).view(dt_).ravel().astype(np.float64)
        if not np.all(np.isfinite(v)): continue
        d = np.diff(v)
        if len(d) and np.mean(d > 0) > 0.99:
            print(f"  MONOTONIC {nm}@{c}: first={v[0]:.3f} last={v[-1]:.3f} med_step={np.median(d):.4f} "
                  f"span={(v[-1]-v[0]):.3f} -> rate {n/((v[-1]-v[0]) or 1):.4f} per unit")
