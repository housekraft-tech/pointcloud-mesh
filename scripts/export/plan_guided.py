"""Let the drawing group the runs; let the scan place the faces.

The measurement cuts a wall wherever anything changes -- a pier, a crossing
wall, a doorway -- and gives 105 runs for a flat the architect drew with 24
walls. Neither is wrong: the runs are what was measured, the 24 are what was
built. So the drawing is used for one thing only, the thing it is actually
authoritative about: WHICH runs are the same wall.

Every run is assigned to the drawing wall it lies along. Each group then
becomes one continuous wall, spanning the measured extent of its runs, with the
thickness those runs measured -- and where a run is thicker than the rest, the
extra comes back as a pier, so the wall still runs through behind it.

Runs no drawing wall claims are kept as they are. The drawing is a guide, not
an authority: it cannot delete something the scanner saw.
"""
import argparse, json, collections
from pathlib import Path
import numpy as np

TOL_ACROSS = 0.18     # m: how far a run may sit from the drawing's centreline
PIER_TOL = 0.030


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--registered", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    S = json.load(open(a.schedule))["parts"]
    plan = json.load(open(a.registered))["walls"]
    runs = [r for r in S if r["op"] == "run"]
    other = [r for r in S if r["op"] != "run"]

    lines = []
    for i, w in enumerate(plan):
        p0 = np.array(w["a"]); p1 = np.array(w["b"])
        ax = 0 if abs(p1[0]-p0[0]) < abs(p1[1]-p0[1]) else 1   # across axis
        lines.append(dict(i=i, ax=ax, c=0.5*(p0[ax]+p1[ax]),
                          s0=min(p0[1-ax], p1[1-ax]), s1=max(p0[1-ax], p1[1-ax])))

    claimed = collections.defaultdict(list)
    orphan = []
    for r in runs:
        ax = 0 if (r["hi"][0]-r["lo"][0]) < (r["hi"][1]-r["lo"][1]) else 1
        c = 0.5*(r["lo"][ax] + r["hi"][ax])
        s0, s1 = r["lo"][1-ax], r["hi"][1-ax]
        best, bd = None, TOL_ACROSS
        for L in lines:
            if L["ax"] != ax:
                continue
            if min(s1, L["s1"]) - max(s0, L["s0"]) < 0.10:      # must overlap
                continue
            d = abs(L["c"] - c)
            if d < bd:
                best, bd = L, d
        (claimed[best["i"]] if best else orphan).append((r, ax))

    out, piers = [], []
    for i, items in sorted(claimed.items()):
        ax = items[0][1]
        near = np.array([r["lo"][ax] for r, _ in items])
        far = np.array([r["hi"][ax] for r, _ in items])
        length = np.array([r["hi"][1-ax] - r["lo"][1-ax] for r, _ in items])
        # the wall's own faces: the length-weighted middle of what was measured
        n0 = float(np.average(near, weights=length))
        t = float(np.average(far - near, weights=length))
        s0 = float(min(r["lo"][1-ax] for r, _ in items))
        s1 = float(max(r["hi"][1-ax] for r, _ in items))
        z0 = float(min(r["lo"][2] for r, _ in items))
        z1 = float(max(r["hi"][2] for r, _ in items))
        lo = [0, 0, z0]; hi = [0, 0, z1]
        lo[ax], hi[ax] = n0, n0 + t
        lo[1-ax], hi[1-ax] = s0, s1
        name = f"wall_{len(out):02d}"
        out.append(dict(name=name, kind="wall", op="run",
                        lo=[round(v, 5) for v in lo], hi=[round(v, 5) for v in hi]))
        # anything measurably thicker stays, as a pier on this wall
        for r, _ in items:
            tr = r["hi"][ax] - r["lo"][ax]
            if tr > t + PIER_TOL:
                plo = list(r["lo"]); phi = list(r["hi"])
                plo[ax] = n0 + t
                phi[ax] = max(r["hi"][ax], n0 + t + 0.001)
                piers.append(dict(name=name, kind="wall", op="relief",
                                  lo=[round(v, 5) for v in plo],
                                  hi=[round(v, 5) for v in phi]))
    for r, ax in orphan:
        r = dict(r); r["name"] = f"wall_{len(out):02d}"
        out.append(r)

    # the openings follow whichever wall now covers them
    fixed = []
    for r in other:
        if r["op"] in ("opening", "niche", "relief"):
            ax = 0 if (r["hi"][0]-r["lo"][0]) < (r["hi"][1]-r["lo"][1]) else 1
            c = 0.5*(r["lo"][ax]+r["hi"][ax])
            host = None
            for w in out:
                wax = 0 if (w["hi"][0]-w["lo"][0]) < (w["hi"][1]-w["lo"][1]) else 1
                if wax != ax:
                    continue
                if min(r["hi"][1-ax], w["hi"][1-ax]) - max(r["lo"][1-ax], w["lo"][1-ax]) < 0.05:
                    continue
                if abs(0.5*(w["lo"][ax]+w["hi"][ax]) - c) < 0.35:
                    host = w["name"]; break
            r = dict(r)
            if host:
                r["name"] = host
        fixed.append(r)

    json.dump(dict(parts=out + piers + fixed), open(a.out, "w"))
    L = sum(max(w["hi"][0]-w["lo"][0], w["hi"][1]-w["lo"][1]) for w in out)
    print(f"{len(runs)} runs -> {len(out)} walls "
          f"({len(claimed)} the drawing grouped, {len(orphan)} it did not know about)")
    print(f"{len(piers)} piers kept on their wall")
    print(f"wall length {L:.1f} m -> {a.out}")


if __name__ == "__main__":
    main()
