"""Settle the Hp channel bit-split by correlating candidate decodings against the
fully-decoded Lp IMU (both nominally 1000 Hz, same rigid body)."""
import numpy as np
b = np.fromfile("data/Soulace/20260419-104545_Lp_Imu.fmimr", np.uint8)
m = b[1024:].reshape(-1,28)
lp = np.ascontiguousarray(m[:,13:27]).view('<i2').reshape(-1,7).astype(float)
lp_t = (m[:,6].astype(np.int64)*3600 + m[:,7].astype(np.int64)*60 + m[:,8].astype(np.int64)
        + np.ascontiguousarray(m[:,9:13]).view('<u4').ravel()/1e9)
b3 = np.fromfile("data/Soulace/20260419-104545_Hp_Imu.fmimr", np.uint8)
i3 = np.where((b3[:-1]==0x46)&(b3[1:]==0x4d))[0]; i3 = i3[i3>=1024]
s3 = i3[:-1][np.diff(i3)==33]
m3 = np.stack([b3[o:o+33] for o in s3])
raw = np.ascontiguousarray(m3[:,8:32]).view('<u4').reshape(-1,6).astype(np.int64)

def sx(v, bits):
    return np.where(v >= (1 << (bits-1)), v - (1 << bits), v)

# Hp has no absolute clock; align by cross-correlating a strong signal.
# Use a window in the middle of both streams and search the lag.
N = 200000
lp_mid = len(lp)//2; hp_mid = len(raw)//2
best = None
for bits in (20, 24, 28, 32):
    val = sx(raw & ((1 << bits) - 1), bits).astype(float)
    for hs in range(6):
        h = val[hp_mid:hp_mid+N, hs]
        h = (h - h.mean()) / (h.std() + 1e-9)
        for ls in range(6):
            l = lp[lp_mid:lp_mid+N, ls]
            l = (l - l.mean()) / (l.std() + 1e-9)
            # coarse lag search via FFT cross-correlation
            n = 1 << int(np.ceil(np.log2(2*N)))
            cc = np.fft.irfft(np.fft.rfft(h, n) * np.conj(np.fft.rfft(l, n)), n) / N
            k = int(np.argmax(np.abs(cc))); r = float(cc[k])
            lag = k if k < n//2 else k - n
            if best is None or abs(r) > abs(best[0]):
                best = (r, bits, hs, ls, lag)
            if abs(r) > 0.5:
                print(f"  bits={bits} hp_slot{hs} <-> lp_field{ls}: r={r:+.3f} lag={lag}")
print("\nbest overall:", best)
r, bits, hs, ls, lag = best
val = sx(raw & ((1 << bits) - 1), bits).astype(float)
print(f"\nUsing bits={bits}, lag={lag}: full slot<->field correlation matrix (|r|>0.3 shown)")
for h_ in range(6):
    hh = val[hp_mid+lag:hp_mid+lag+N, h_]; hh = (hh-hh.mean())/(hh.std()+1e-9)
    row = []
    for l_ in range(7):
        ll = lp[lp_mid:lp_mid+N, l_]; ll = (ll-ll.mean())/(ll.std()+1e-9)
        row.append(float(np.dot(hh, ll)/N))
    print(f"  hp_slot{h_}: " + " ".join(f"{x:+.2f}" for x in row))
    print(f"      scale vs best lp field: ratio of stds = "
          f"{val[:,h_].std()/lp[:,int(np.argmax(np.abs(row)))].std():.1f}")
