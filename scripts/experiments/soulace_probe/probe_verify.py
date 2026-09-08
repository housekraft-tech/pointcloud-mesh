"""Cross-validation of the decoded Soulace streams:
 - Ec_Data.fmraster  = rotary ENCODER of the spinning lidar head
 - Lidar_Imu.imu     = Livox ICM40609 IMU mounted ON the spinning head
 - Lp_Imu.fmimr      = body ICM40609 IMU (static frame)
Checks that encoder revolutions/s agrees with the head-IMU gyro spin rate."""
import numpy as np
# ---- encoder
b = np.fromfile("data/Soulace/20260419-104545_Ec_Data.fmraster", np.uint8)
m = b[1024:].reshape(-1, 21)
t = m[:,6].astype(np.int64)*3600 + m[:,7].astype(np.int64)*60 + m[:,8].astype(np.int64) + np.ascontiguousarray(m[:,9:13]).view('<u4').ravel()/1e9
ang = np.ascontiguousarray(m[:,14:16]).view('<u2').ravel().astype(np.int64)
rev = np.ascontiguousarray(m[:,16:18]).view('<u2').ravel().astype(np.int64)
dang = np.diff(ang)
print(f"ENCODER: n={len(m)} span={t[-1]-t[0]:.3f}s")
print(f"  angle field: range[{ang.min()},{ang.max()}] uniq={len(np.unique(ang))} "
      f"(2048 counts/rev -> {360/2048:.4f} deg/count)")
u,c = np.unique(dang, return_counts=True); print(f"  angle step top4: {sorted(zip(c,u))[-4:]}")
print(f"  rev field: range[{rev.min()},{rev.max()}] uniq={len(np.unique(rev))} monotonic={bool(np.all(np.diff(rev)>=0))}")
nrev = rev.max()-rev.min()
enc_hz = nrev/(t[-1]-t[0])
print(f"  -> {nrev} revolutions in {t[-1]-t[0]:.3f}s = {enc_hz:.4f} rev/s = {enc_hz*2*np.pi:.4f} rad/s = {enc_hz*60:.2f} rpm")
print(f"  record rate = {len(m)/(t[-1]-t[0]):.1f} Hz ; counts/s = {enc_hz*2048:.1f}")
# ---- head IMU spin rate
b2 = np.fromfile("data/Soulace/20260419-104545_Lidar_Imu.imu", np.uint8)
idx = np.where((b2[:-2]==0xa2)&(b2[1:-1]==0xa7)&(b2[2:]==0x18))[0]; idx = idx[idx>=1024]
st = idx[:-1][np.diff(idx)==51]
mm = np.stack([b2[o:o+51] for o in st])
f = np.ascontiguousarray(mm[:,15:39]).view('<f4').reshape(-1,6).astype(np.float64)
print(f"HEAD IMU gyro_x mean = {f[:,0].mean():.4f} rad/s  -> {f[:,0].mean()/(2*np.pi):.4f} rev/s")
print(f"  AGREEMENT with encoder: {100*abs(f[:,0].mean()/(2*np.pi) - enc_hz)/enc_hz:.2f}% difference")
# ---- Hp IMU, 28-bit signed channels
b3 = np.fromfile("data/Soulace/20260419-104545_Hp_Imu.fmimr", np.uint8)
i3 = np.where((b3[:-1]==0x46)&(b3[1:]==0x4d))[0]; i3 = i3[i3>=1024]
s3 = i3[:-1][np.diff(i3)==33]
m3 = np.stack([b3[o:o+33] for o in s3])
raw = np.ascontiguousarray(m3[:,8:32]).view('<u4').reshape(-1,6).astype(np.int64)
tag = raw >> 28
val = raw & 0x0FFFFFFF
val = np.where(val >= (1<<27), val - (1<<28), val)
print(f"HP IMU: n={len(m3)}  per-slot tag (top nibble) unique:")
for k in range(6):
    print(f"  slot{k}: tags={np.unique(tag[:,k])} mean={val[:,k].mean():12.1f} std={val[:,k].std():11.1f} "
          f"range[{val[:,k].min()},{val[:,k].max()}]")
for grp,lbl in (((0,1,2),'slots 0-2'), ((3,4,5),'slots 3-5')):
    mag = np.linalg.norm(val[:,list(grp)].astype(float),axis=1)
    print(f"  |{lbl}| mean={mag.mean():.1f}  /2**16={mag.mean()/65536:.4f}  /2**20={mag.mean()/(1<<20):.5f}")
