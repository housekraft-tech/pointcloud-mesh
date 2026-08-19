"""Two scans of one flat, measured twice -- how far apart do they land?

koushik and mujammel are the same apartment, scanned separately. Neither is a
ground truth for the other, but a quantity that comes out the same from two
independent walks is a quantity the pipeline can measure, and one that does not
is one it cannot. That is an error bar obtained without a tape measure.

The two models live in their own frames, so they are registered first: each
model's walls are drawn as a plan raster and the pair is aligned by
cross-correlation over the four right-angle rotations. Then wall lines, wall
thicknesses, ceiling levels and openings are compared across the transform.
"""
import sys, json
from pathlib import Path
import numpy as np
from numpy.fft import rfft2, irfft2

A_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "output/model/poisson_modular")
B_DIR = Path(sys.argv[2] if len(sys.argv) > 2 else "output/model/poisson_modular_mujammel")
A_CACHE = sys.argv[3] if len(sys.argv) > 3 else "output/model/poisson_koushik.npz"
B_CACHE = sys.argv[4] if len(sys.argv) > 4 else "output/model/poisson_mujammel.npz"
CELL = 0.025


def wall_plan(d, cache):
    """A plan raster of everything the model calls wall, and the model itself."""
    man = json.load(open(d/"manifest.json"))
    T = np.load(cache)["T"].astype(np.int64)
    V = np.load(d/"verts.npy").astype(np.float64)
    L = np.load(d/"labels.npy")
    names = json.load(open(d/"names.json"))
    C = V[T].mean(axis=1)
    wall_ids = [i for i, n in enumerate(names) if n.startswith(("wall", "parapet", "column"))]
    m = np.isin(L, wall_ids)
    return man, C[m][:, :2]


def raster(P, lo, n):
    ij = np.clip(((P - lo)/CELL).astype(int), 0, np.array(n)-1)
    r = np.zeros(n, np.float32)
    r[ij[:, 0], ij[:, 1]] = 1.0
    return r


def align(PA, PB):
    """Rotation (a right angle) and shift that put B's walls on A's."""
    best = None
    lo = PA.min(0) - 1.0
    ext = np.maximum(PA.max(0) - PA.min(0), 1.0) + 2.0
    n = (np.ceil(ext/CELL).astype(int) + 1)
    RA = raster(PA, lo, n)
    FA = rfft2(RA)
    for k in range(4):
        th = np.radians(90*k); c, s = np.cos(th), np.sin(th)
        Q = PB @ np.array([[c, s], [-s, c]])
        q_lo = Q.min(0)
        RB = raster(Q - q_lo + lo*0, lo*0, n)      # B rastered at its own origin
        cc = irfft2(FA * np.conj(rfft2(RB)), RA.shape)
        idx = np.unravel_index(np.argmax(cc), cc.shape)
        score = float(cc[idx])
        shift = np.array(idx, float)*CELL
        shift = np.where(shift > np.array(n)*CELL/2, shift - np.array(n)*CELL, shift)
        if best is None or score > best[0]:
            best = (score, k, shift + lo - q_lo)
    score, k, t = best
    th = np.radians(90*k); c, s = np.cos(th), np.sin(th)
    R = np.array([[c, s], [-s, c]])
    print(f"registration: rotate {90*k} deg, shift [{t[0]:+.3f} {t[1]:+.3f}] m "
          f"(overlap score {score:.0f})")
    return R, t


def wall_lines(man, R=None, t=None):
    """Each wall as (axis, centre coordinate across, span along, thickness)."""
    out = []
    for p in man["parts"]:
        if p["kind"] not in ("wall", "parapet") or "across_m" not in p:
            continue
        ax = 0 if p["axis"] == "x" else 1
        c = 0.5*(p["across_m"][0] + p["across_m"][1])
        s0, s1 = p["along_m"]
        pts = np.array([[c, s0], [c, s1]]) if ax == 0 else np.array([[s0, c], [s1, c]])
        if R is not None:
            pts = pts @ R + t
            ax = ax if abs(R[0, 0]) > 0.5 else 1-ax
        cc = pts[:, ax].mean(); ss = sorted(pts[:, 1-ax])
        out.append(dict(name=p["name"], axis=ax, c=float(cc), s0=float(ss[0]),
                        s1=float(ss[1]), thickness=p["thickness_mm"],
                        area=p["area_m2"]))
    return out


def main():
    manA, PA = wall_plan(A_DIR, A_CACHE)
    manB, PB = wall_plan(B_DIR, B_CACHE)
    print(f"A {A_DIR.name}: {manA['n_parts']} parts, clear height "
          f"{manA['clear_height_mm']:.0f} mm")
    print(f"B {B_DIR.name}: {manB['n_parts']} parts, clear height "
          f"{manB['clear_height_mm']:.0f} mm")
    R, t = align(PA, PB)

    WA = wall_lines(manA)
    WB = wall_lines(manB, R, t)
    # The raster fixes the shift only to its own cell. Refine it on the matched
    # wall lines themselves: walls across x pin the x shift, walls across y pin
    # the y shift, and the median is robust to the walls that did not match.
    for _ in range(3):
        adj = np.zeros(2)
        for ax in (0, 1):
            dd = []
            for a in WA:
                if a["axis"] != ax:
                    continue
                cand = [b for b in WB if b["axis"] == ax
                        and min(a["s1"], b["s1"]) - max(a["s0"], b["s0"]) > 0.4
                        and abs(a["c"]-b["c"]) < 0.35]
                if cand:
                    dd.append(min(cand, key=lambda b: abs(a["c"]-b["c"]))["c"] - a["c"])
            if dd:
                adj[ax] = float(np.median(dd))
        if np.abs(adj).max() < 1e-4:
            break
        for b in WB:
            b["c"] -= adj[b["axis"]]
    print(f"refined on matched wall lines by [{-adj[0]*1000:+.0f} {-adj[1]*1000:+.0f}] mm")
    print(f"\nwalls: A {len(WA)}, B {len(WB)}")
    rows = []
    for a in WA:
        best = None
        for b in WB:
            if b["axis"] != a["axis"]:
                continue
            ov = min(a["s1"], b["s1"]) - max(a["s0"], b["s0"])
            if ov < 0.4:
                continue
            d = abs(a["c"] - b["c"])
            if best is None or d < best[0]:
                best = (d, b, ov)
        if best and best[0] < 0.35:
            rows.append((a, best[1], best[0], best[2]))
    d = np.array([r[2] for r in rows])*1000
    print(f"{len(rows)} of {len(WA)} A-walls have a B-wall on the same line "
          f"within 350 mm")
    if len(d):
        print(f"  wall line agreement: median {np.median(d):.0f} mm, "
              f"90th pct {np.percentile(d, 90):.0f} mm, worst {d.max():.0f} mm")
    tt = [(r[0]["thickness"], r[1]["thickness"]) for r in rows
          if r[0]["thickness"] and r[1]["thickness"]]
    if tt:
        dif = np.array([abs(x-y) for x, y in tt])
        print(f"  thickness, both measured ({len(tt)} walls): median difference "
              f"{np.median(dif):.0f} mm, worst {dif.max():.0f} mm")
        for x, y in sorted(tt):
            print(f"     {x:6.0f} vs {y:6.0f} mm")

    print("\nceiling levels (mm over floor, area-ranked):")
    for man, tag in ((manA, "A"), (manB, "B")):
        lv = [(p["height_mm"], p["area_m2"], p["kind"]) for p in man["parts"]
              if p["kind"] in ("ceiling", "dropped_ceiling", "beam")]
        lv.sort(key=lambda x: -x[1])
        print(f"  {tag}: " + ", ".join(f"{h:.0f}({a:.0f}m2)" for h, a, k in lv[:8]))

    print("\nopenings, sorted by width:")
    for man, tag in ((manA, "A"), (manB, "B")):
        op = sorted([f for f in man["features"]
                     if f["kind"] in ("door", "window", "arch")],
                    key=lambda f: -f["width_mm"])
        print(f"  {tag}: " + ", ".join(f"{f['kind'][0]}{f['width_mm']}x{f['height_mm']}"
                                       for f in op))

    fa = [p["area_m2"] for p in manA["parts"] if p["kind"] == "floor"]
    fb = [p["area_m2"] for p in manB["parts"] if p["kind"] == "floor"]
    print(f"\nfloor area in named parts: A {sum(fa):.1f} m2 in {len(fa)} parts, "
          f"B {sum(fb):.1f} m2 in {len(fb)} parts")


if __name__ == "__main__":
    main()
