"""Put the architect's wall network onto the scan's own frame.

The drawing knows how the flat is CONSTRUCTED: 24 walls, each one continuous
from end to end, junctions where the designer meant them. The scan knows where
every face actually is, to a few hundredths of a millimetre. Neither is much
use without the other -- measuring alone splits one wall into five wherever a
pier or a crossing wall interrupts it, and the drawing alone has no thickness
worth the name (every wall in it is one 10-pixel line weight).

This solves the rigid transform between them: the drawing is axis-aligned and
so is the scan, so the search is four right angles, an optional mirror (image
y runs downwards), a scale near 1, and a shift found by cross-correlating the
two wall rasters.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from numpy.fft import rfft2, irfft2

CELL = 0.025


def drawing_walls(path):
    d = json.load(open(path))
    sc = d["dimensions"][0]["scale"]
    out = []
    for w in d["global_walls"]:
        p0 = np.array([w["x1"], w["y1"]])*sc
        p1 = np.array([w["x2"], w["y2"]])*sc
        out.append((p0, p1))
    return out, d


def model_walls(path):
    S = json.load(open(path))["parts"]
    segs = []
    for r in S:
        if r["op"] != "run":
            continue
        ax = 0 if (r["hi"][0]-r["lo"][0]) < (r["hi"][1]-r["lo"][1]) else 1
        c = 0.5*(r["lo"][ax] + r["hi"][ax])
        a = np.zeros(2); b = np.zeros(2)
        a[ax] = c; b[ax] = c
        a[1-ax] = r["lo"][1-ax]; b[1-ax] = r["hi"][1-ax]
        segs.append((a, b))
    return segs


def raster(segs, lo, n):
    r = np.zeros(n, np.float32)
    for a, b in segs:
        L = max(int(np.linalg.norm(b-a)/CELL), 1)
        for t in np.linspace(0, 1, L*2):
            p = a + (b-a)*t
            i = int((p[0]-lo[0])/CELL); j = int((p[1]-lo[1])/CELL)
            if 0 <= i < n[0] and 0 <= j < n[1]:
                r[i, j] = 1.0
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drawing", required=True)
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    D, meta = drawing_walls(a.drawing)
    M = model_walls(a.schedule)
    mpts = np.vstack([np.vstack(s) for s in M])
    lo = mpts.min(0) - 1.0
    ext = mpts.max(0) - mpts.min(0) + 2.0
    n = tuple((np.ceil(ext/CELL).astype(int) + 1).tolist())
    RM = raster(M, lo, n)
    FM = rfft2(RM)
    dl = sum(np.linalg.norm(b-a) for a, b in D)
    ml = sum(np.linalg.norm(b-a) for a, b in M)
    print(f"drawing {len(D)} walls / {dl:.1f} m   model {len(M)} runs / {ml:.1f} m")

    best = None
    for mirror in (1, -1):
        for k in range(4):
            th = np.radians(90*k); c, s = np.cos(th), np.sin(th)
            R = np.array([[c, -s], [s, c]]) @ np.array([[1, 0], [0, mirror]])
            Dr = [(p0 @ R.T, p1 @ R.T) for p0, p1 in D]
            dpts = np.vstack([np.vstack(x) for x in Dr])
            dlo = dpts.min(0)
            RD = raster([(p0-dlo, p1-dlo) for p0, p1 in Dr], np.zeros(2), n)
            cc = irfft2(FM * np.conj(rfft2(RD)), RM.shape)
            idx = np.unravel_index(np.argmax(cc), cc.shape)
            score = float(cc[idx])
            shift = np.array(idx, float)*CELL
            shift = np.where(shift > np.array(n)*CELL/2, shift - np.array(n)*CELL, shift)
            if best is None or score > best[0]:
                best = (score, k, mirror, shift + lo - dlo, R)
    score, k, mirror, t, R = best
    print(f"registered: rotate {90*k} deg, mirror {'yes' if mirror < 0 else 'no'}, "
          f"shift [{t[0]:+.3f} {t[1]:+.3f}] m, overlap {score:.0f}")

    placed = [dict(a=(p0 @ R.T + t).round(4).tolist(),
                   b=(p1 @ R.T + t).round(4).tolist()) for p0, p1 in D]
    # how well does each drawing wall land on a measured one?
    res = []
    for w in placed:
        p0 = np.array(w["a"]); p1 = np.array(w["b"])
        ax = 0 if abs(p1[0]-p0[0]) < abs(p1[1]-p0[1]) else 1
        c = 0.5*(p0[ax]+p1[ax])
        near = [abs(0.5*(m0[ax]+m1[ax]) - c) for m0, m1 in M
                if abs(m1[ax]-m0[ax]) < abs(m1[1-ax]-m0[1-ax])]
        if near:
            res.append(min(near))
    if res:
        r = np.array(res)*1000
        print(f"drawing walls landing on a measured wall: median {np.median(r):.0f} mm, "
              f"90th {np.percentile(r, 90):.0f} mm, worst {r.max():.0f} mm")
    json.dump(dict(rotation_deg=90*k, mirror=int(mirror),
                   shift=t.round(5).tolist(), walls=placed),
              open(a.out, "w"), indent=1)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
