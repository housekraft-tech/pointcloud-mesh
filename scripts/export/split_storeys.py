"""Cut a multi-storey scan into one scan per floor.

Everything downstream assumes a single storey: the level detector takes one
floor and one ceiling, walls run between them, and slabs are plateaus in one
height map. Handed three floors at once it would find the largest slab in the
building and measure everything to that.

So the storeys are separated first, and they separate themselves. A floor slab
and the ceiling under it are the two densest horizontal surfaces in a building,
and they come out of the height histogram as a pair of spikes about three
metres apart. Each storey is then written out with its own floor and its own
ceiling included, overlapping its neighbours slightly, because a storey without
its slabs has no levels to measure from.

The plan is cropped at the same time. A handheld scan of a house carries stray
returns hundreds of metres out -- through a window, down a street -- and they
cost nothing in point count but set the size of every grid built later: this
export is 21 x 17 m of building inside an 87 x 96 m extent, which is a 20-fold
waste in every occupancy volume downstream.
"""
import sys, json, argparse, time
from pathlib import Path
import numpy as np

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

ZBIN = 0.05           # m: height histogram bin
MIN_GAP = 2.0         # m: the least a storey can be, floor to ceiling
MAX_GAP = 5.0         # m: and the most
BELOW = 0.45          # m: keep this much below a floor -- the slab itself
ABOVE = 0.35          # m: and this much above a ceiling
PLAN_CELL = 0.25      # m: plan occupancy used to find the building
PLAN_MIN = 8          # points in a cell before it is the building
PLAN_PAD = 1.5        # m: keep this much around it


def storeys(z, floor_frac=0.10, merge=0.60, slab_max=0.50, break_frac=0.30):
    """Floor and ceiling heights, from the spikes in the height histogram.

    A level break is not one spike but two: the ceiling of the storey below and
    the slab of the storey above, a couple of hundred millimetres apart. So the
    spikes are clustered, and inside a cluster the strongest is the ceiling and
    a companion just above it is the floor over it. Pairing spikes without that
    -- taking each one as a floor and looking upward for a partner -- read this
    building as four storeys, two of them sharing a ceiling, because a dropped
    ceiling at 2.32 m is a perfectly good spike too.
    """
    bins = np.arange(z.min(), z.max()+ZBIN, ZBIN)
    h, e = np.histogram(z, bins=bins)
    mid = 0.5*(e[:-1]+e[1:])
    sm = np.convolve(h, np.ones(3)/3, "same")
    peaks = [(mid[i], sm[i]) for i in range(1, len(sm)-1)
             if sm[i] >= sm[i-1] and sm[i] >= sm[i+1] and sm[i] > floor_frac*sm.max()]
    if not peaks:
        return []
    log("horizontal spikes at " + ", ".join(f"{p:.2f}" for p, _ in peaks) + " m")

    groups = [[peaks[0]]]
    for pk in peaks[1:]:
        if pk[0] - groups[-1][-1][0] <= merge:
            groups[-1].append(pk)
        else:
            groups.append([pk])
    breaks = []
    for gr in groups:
        zc, w = max(gr, key=lambda q: q[1])          # the strongest: a ceiling
        above = [q for q in gr if 0.05 < q[0] - zc < slab_max]
        zf = max(above, key=lambda q: q[1])[0] if above else zc
        breaks.append((zc, zf, w))
    # A level break is a whole floor of slab, so it is one of the biggest
    # horizontal surfaces in the building. A dropped ceiling is a real spike and
    # not a break: taking every cluster as one cut this house's top storey at
    # 8.42 m and called it 2.1 m high, when 6.32 to 9.37 matches the two below
    # it to within 50 mm.
    top = max(w for _, _, w in breaks)
    weak = [c for c, _, w in breaks if w < break_frac*top]
    breaks = [b for b in breaks if b[2] >= break_frac*top]
    if weak:
        log("not level breaks, too little surface: "
            + ", ".join(f"{c:.2f}" for c in weak) + " m")
    log("level breaks: " + ", ".join(
        f"{c:.2f}" + (f"/{f:.2f}" if f != c else "") for c, f, _ in breaks))

    out = []
    for (c0, f0, w0), (c1, f1, w1) in zip(breaks, breaks[1:]):
        if MIN_GAP < c1 - f0 < MAX_GAP:
            out.append((f0, c1))
    return out


def main():
    import laspy
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--las", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=8,
                    help="every Nth point when looking for the storeys")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    log(f"reading {a.las}")
    with laspy.open(a.las) as r:
        hdr = r.header
        Z, XY = [], []
        for p in r.chunk_iterator(4_000_000):
            Z.append(np.asarray(p.z)[::a.sample].astype(np.float32))
            XY.append(np.column_stack([p.x, p.y])[::a.sample].astype(np.float32))
    z = np.concatenate(Z); xy = np.concatenate(XY)
    log(f"{hdr.point_count:,} points, sampled {len(z):,}")

    # the building, in plan
    lo = xy.min(0)
    n = np.ceil((xy.max(0)-lo)/PLAN_CELL).astype(int)+1
    ij = ((xy-lo)/PLAN_CELL).astype(int)
    g = np.zeros(n, np.int32)
    np.add.at(g, (ij[:, 0], ij[:, 1]), 1)
    occ = g >= PLAN_MIN
    ii, jj = np.where(occ)
    x0 = lo[0] + ii.min()*PLAN_CELL - PLAN_PAD
    x1 = lo[0] + (ii.max()+1)*PLAN_CELL + PLAN_PAD
    y0 = lo[1] + jj.min()*PLAN_CELL - PLAN_PAD
    y1 = lo[1] + (jj.max()+1)*PLAN_CELL + PLAN_PAD
    log(f"the building is {x1-x0:.1f} x {y1-y0:.1f} m inside an extent of "
        f"{np.ptp(xy[:, 0]):.0f} x {np.ptp(xy[:, 1]):.0f} m")

    lv = storeys(z)
    if not lv:
        raise SystemExit("no floor/ceiling pair found -- is this one storey?")
    log(f"{len(lv)} storeys:")
    for k, (zf, zc) in enumerate(lv):
        log(f"   L{k}: floor {zf:6.2f}  ceiling {zc:6.2f}  clear "
            f"{(zc-zf)*1000:.0f} mm")

    counts = [0]*len(lv)
    writers = []
    for k, (zf, zc) in enumerate(lv):
        h2 = laspy.LasHeader(version=hdr.version, point_format=hdr.point_format)
        h2.scales = hdr.scales; h2.offsets = hdr.offsets
        writers.append(laspy.open(out/f"L{k}.las", mode="w", header=h2))
    with laspy.open(a.las) as r:
        for chunk in r.chunk_iterator(4_000_000):
            x = np.asarray(chunk.x); y = np.asarray(chunk.y); zz = np.asarray(chunk.z)
            inplan = (x > x0) & (x < x1) & (y > y0) & (y < y1)
            for k, (zf, zc) in enumerate(lv):
                m = inplan & (zz > zf - BELOW) & (zz < zc + ABOVE)
                if m.any():
                    writers[k].write_points(chunk[m])
                    counts[k] += int(m.sum())
    for w in writers:
        w.close()
    meta = []
    for k, ((zf, zc), c) in enumerate(zip(lv, counts)):
        log(f"   L{k}.las: {c:,} points")
        meta.append(dict(level=k, floor_z=round(float(zf), 3),
                         ceiling_z=round(float(zc), 3),
                         clear_height_mm=round((zc-zf)*1000, 1), points=c,
                         band=[round(float(zf-BELOW), 3), round(float(zc+ABOVE), 3)]))
    json.dump(dict(source=a.las, plan=[round(float(v), 3) for v in (x0, y0, x1, y1)],
                   storeys=meta), open(out/"storeys.json", "w"), indent=1)
    log(f"wrote {out}/storeys.json")


if __name__ == "__main__":
    main()
