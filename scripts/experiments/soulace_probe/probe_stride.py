"""Find fixed record stride in feimarobotics binary files by locating the repeating
'FM' (0x464d) marker and histogramming its spacing. Read-only."""
import sys, numpy as np, collections
for p in sys.argv[1:]:
    b = np.fromfile(p, dtype=np.uint8)
    print(f"\n=== {p}  size={b.size}")
    # locate 'FM'
    idx = np.where((b[:-1] == 0x46) & (b[1:] == 0x4D))[0]
    print(f"  'FM' occurrences: {len(idx)}  first at {idx[:6]}")
    if len(idx) > 2:
        d = np.diff(idx)
        c = collections.Counter(d.tolist()).most_common(6)
        print(f"  spacing histogram (top6): {c}")
    # brute-force stride search: for each candidate stride, how self-consistent is
    # the byte at a fixed offset?
    best = []
    for stride in range(8, 129):
        n = (b.size - 128) // stride
        if n < 1000: continue
        m = b[128:128 + n*stride].reshape(n, stride)
        # score = number of columns that are near-constant
        const_cols = sum(1 for c in range(stride) if len(np.unique(m[:5000, c])) <= 2)
        best.append((const_cols, stride))
    best.sort(reverse=True)
    print(f"  top strides by near-constant column count: {best[:6]}")
