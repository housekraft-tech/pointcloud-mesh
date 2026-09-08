"""Dump raw records at a given stride and score every byte offset for a
monotonically-increasing integer field (timestamp/counter). Read-only."""
import sys, numpy as np
p, off, stride = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
nshow = int(sys.argv[4]) if len(sys.argv) > 4 else 6
b = np.fromfile(p, dtype=np.uint8)
n = (b.size - off) // stride
m = b[off:off + n*stride].reshape(n, stride)
print(f"{p}: off={off} stride={stride} nrec={n} leftover={b.size-off-n*stride}")
print("\nfirst records (hex):")
for i in range(nshow):
    print(f"  [{i}] " + " ".join(f"{x:02x}" for x in m[i]))
print("last records (hex):")
for i in range(n-3, n):
    print(f"  [{i}] " + " ".join(f"{x:02x}" for x in m[i]))
print("\nper-column unique counts (first 20k rec):", [len(np.unique(m[:20000,c])) for c in range(stride)])

sub = m[:200000]
for width, dt, name in ((4,'<u4','u32le'), (8,'<u8','u64le'), (4,'<i4','i32le'), (2,'<u2','u16le')):
    for c in range(stride - width + 1):
        v = np.ascontiguousarray(sub[:, c:c+width]).view(dt).ravel().astype(np.float64)
        d = np.diff(v)
        if len(d) and np.all(d > 0):
            frac = 1.0
        else:
            frac = float(np.mean(d > 0))
        if frac > 0.98:
            med = float(np.median(d))
            print(f"  MONOTONIC {name} @off{c}: frac_inc={frac:.4f} first={v[0]:.0f} last={v[-1]:.0f} med_step={med:.3f}")
