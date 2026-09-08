"""Characterise the two opaque blobs in slam_calib.yaml: 'code' (base64) and 'correct'."""
import base64, numpy as np, re
txt = open("data/Soulace/slam_calib.yaml", encoding="utf-8", errors="replace").read()
code = re.search(r"code:\s*(\S+)", txt).group(1)
raw = base64.b64decode(code)
print(f"code: {len(code)} b64 chars -> {len(raw)} bytes")
print("  first 16 bytes:", raw[:16].hex())
print("  u32[0..3]:", np.frombuffer(raw[:16], '<u4'))
body = raw[4:4 + ((len(raw)-4)//4)*4]
f = np.frombuffer(body, '<f4')
print(f"  as {len(f)} float32 after a 4-byte header: finite={np.isfinite(f).all()} "
      f"min={f.min():.4g} max={f.max():.4g} mean={f.mean():.4g} zeros={np.sum(f==0)}")
print("  first 12 floats:", np.round(f[:12], 8))
arr = np.array([int(x) for x in re.search(r"correct:\s*\[([^\]]*)\]", txt).group(1).split(",")])
print(f"\ncorrect: {arr.size} ints, head={arr[:10]}, min={arr.min()} max={arr.max()}")
print(f"  declared payload {arr[2]}, actual after 10-int header = {arr.size-10}"
      f" (diff {arr.size-10-arr[2]})")
p = arr[10:10+arr[2]]
for period in (86, 87, 88, 89, 90, 128, 256):
    if p.size % period == 0:
        print(f"  divisible by {period}: {p.size//period} rows")
print(f"  value histogram: zeros={np.sum(p==0)} ({100*np.sum(p==0)/p.size:.1f}%)")
