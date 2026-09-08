"""Decode the feimarobotics 'FM' record files (Lp_Imu 28B, Hp_Imu 33B, Ec_Data 21B).

Confirmed layout (evidence in docs/SOULACE_SENSOR_DATA.md):
  offset 0..1  'FM' magic
  offset 2     u8  rolling sequence counter (wraps 255->0)
  offset 3     u8  message/device id (constant per file)
  offset 4..5  u16 payload length (= stride - 6)
  offset 6     u8  hour
  offset 7     u8  minute
  offset 8     u8  second
  offset 9..12 u32 nanoseconds-within-second (resets at 1e9)
  offset 13..  payload
  last byte    u8  checksum
Read-only probe.
"""
import numpy as np, sys

def load(p, stride):
    b = np.fromfile(p, dtype=np.uint8)
    body = b[1024:]
    n = len(body) // stride
    return b, body[:n*stride].reshape(n, stride), len(body) - n*stride

def report(p, stride):
    b, m, leftover = load(p, stride)
    n = len(m)
    ok = np.mean((m[:,0]==0x46)&(m[:,1]==0x4d))
    print(f"\n=== {p}\n  size={b.size} stride={stride} nrec={n} leftover={leftover} magic_ok={ok:.6f}")
    if ok < 0.999:
        print("  -> stride hypothesis FAILS, aborting"); return None
    print(f"  byte3 (msg id) unique={np.unique(m[:,3])}  len field unique={np.unique(np.ascontiguousarray(m[:,4:6]).view('<u2'))}")
    h, mi, s = m[:,6].astype(np.int64), m[:,7].astype(np.int64), m[:,8].astype(np.int64)
    ns = np.ascontiguousarray(m[:,9:13]).view('<u4').ravel().astype(np.int64)
    t = h*3600 + mi*60 + s + ns/1e9
    d = np.diff(t)
    print(f"  TIME  first={t[0]:.6f} s  last={t[-1]:.6f} s  span={t[-1]-t[0]:.3f} s")
    print(f"        h range {h.min()}..{h.max()}  ns max={ns.max()} (<1e9: {ns.max()<1e9})")
    print(f"        monotonic={bool(np.all(d>0))} frac_inc={np.mean(d>0):.6f}  dt med={np.median(d)*1e6:.1f} us -> {1/np.median(d):.2f} Hz")
    return m, t

# Lp IMU
m, t = report("data/Soulace/20260419-104545_Lp_Imu.fmimr", 28)
n = len(m)
f = np.ascontiguousarray(m[:,13:27]).view('<i2').reshape(n,7).astype(np.float64)
names = ["p0","p1","p2","p3","p4","p5","p6"]
for i in range(7):
    print(f"    field{i}: mean={f[:,i].mean():10.2f} std={f[:,i].std():9.2f} range [{f[:,i].min():.0f},{f[:,i].max():.0f}]")
for grp in ((0,1,2),(3,4,5)):
    mag = np.linalg.norm(f[:,list(grp)],axis=1)
    print(f"    |{grp}| mean={mag.mean():9.2f} std={mag.std():8.2f}   /4096 = {mag.mean()/4096:.4f} g   /2048 = {mag.mean()/2048:.4f} g")

report("data/Soulace/20260419-104545_Hp_Imu.fmimr", 33)
report("data/Soulace/20260419-104545_Ec_Data.fmraster", 21)
