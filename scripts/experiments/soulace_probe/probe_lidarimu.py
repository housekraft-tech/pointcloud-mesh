"""Probe 20260419-104545_Lidar_Imu.imu -- no 'FM' framing, unknown layout."""
import numpy as np, collections
p = "data/Soulace/20260419-104545_Lidar_Imu.imu"
b = np.fromfile(p, dtype=np.uint8)
print(f"size={b.size}")
print("header 0x00-0x80:", b[:0x80].tobytes())
for off in (0x80, 0x100, 0x200, 0x400):
    print(f"@{off:#x}:", " ".join(f"{x:02x}" for x in b[off:off+48]))
# where does the header padding end?
nz = np.where(b[0x50:4096] != 0)[0]
print("first non-zero after 0x50:", (nz[:5] + 0x50) if len(nz) else "none in 4096")
for start in (0x50, 0x80, 0x100, 0x200, 0x400, 0x1000):
    rem = b.size - start
    print(f"  start {start:#6x}: rem={rem}  divisible by:",
          [s for s in range(8, 65) if rem % s == 0])
