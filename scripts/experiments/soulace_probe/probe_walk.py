"""Walk a length-prefixed feimarobotics 'FM' stream (handles variable record sizes)."""
import numpy as np, sys, collections
p = sys.argv[1]
b = np.fromfile(p, dtype=np.uint8)
i, recs, bad = 1024, [], 0
while i + 6 <= b.size:
    if b[i] != 0x46 or b[i+1] != 0x4d:
        bad += 1; i += 1; continue
    ln = int(b[i+4]) | (int(b[i+5]) << 8)
    if ln < 1 or i + 6 + ln > b.size:
        bad += 1; i += 1; continue
    recs.append((i, int(b[i+3]), ln)); i += 6 + ln
print(f"{p}: parsed {len(recs)} records, {bad} resync bytes, ended at {i}/{b.size}")
byid = collections.Counter((mid, ln) for _, mid, ln in recs)
print("(msgid,payload_len) -> count:", byid.most_common(10))
for (mid, ln), cnt in byid.most_common(4):
    offs = np.array([o for o, m_, l_ in recs if m_ == mid and l_ == ln])
    m = np.stack([b[o:o+6+ln] for o in offs[:400000]])
    h, mi, s = m[:,6].astype(np.int64), m[:,7].astype(np.int64), m[:,8].astype(np.int64)
    ns = np.ascontiguousarray(m[:,9:13]).view('<u4').ravel().astype(np.int64)
    t = h*3600 + mi*60 + s + ns/1e9
    d = np.diff(t)
    print(f"\n  --- msgid={mid} paylen={ln} n={len(offs)} (analysed {len(m)})")
    print(f"      t {t[0]:.6f} -> {t[-1]:.6f}  span={t[-1]-t[0]:.3f}s  mono={bool(np.all(d>0))} "
          f"dt_med={np.median(d)*1e6:.1f}us -> {1/np.median(d):.2f} Hz  ns_ok={ns.max()<1e9}")
    print("      example hex:", " ".join(f"{x:02x}" for x in m[0]))
    k = (ln - 7) // 2
    if k > 0:
        f = np.ascontiguousarray(m[:, 13:13+2*k]).view('<i2').reshape(len(m), k).astype(float)
        for j in range(k):
            print(f"        i16[{j}] mean={f[:,j].mean():10.2f} std={f[:,j].std():9.2f} "
                  f"range[{f[:,j].min():.0f},{f[:,j].max():.0f}]")
