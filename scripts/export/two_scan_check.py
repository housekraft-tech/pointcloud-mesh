"""Two walks of one flat, measured against each other.

Precision -- how tightly a face is pinned within one scan -- is not accuracy.
The only accuracy statement available without a tape measure is agreement
between two independent scans of the same building: whatever they both say is
probably true, and whatever they disagree about is the error bar.

The two models sit in their own frames, so the box models are registered by
their wall lines first (a right-angle rotation and a shift), then compared:
where each wall face sits, how thick each wall is, and how high the ceiling is.
"""
import sys, json, argparse
from pathlib import Path
import numpy as np


def faces(d):
    """Every wall's two faces, from the boxes manifest."""
    d = Path(d)
    mf = d/"modular"/"manifest.json"          # the drawer layout
    if not mf.exists():
        mf = d/"manifest.json"                # the older flat layout
    man = json.load(open(mf))
    out = []
    for p in man["parts"]:
        if p["kind"] not in ("wall", "parapet") or "across_m" not in p:
            continue
        ax = 0 if p["axis"] == "x" else 1
        lo, hi = p["across_m"]
        s0, s1 = p["along_m"]
        out.append(dict(axis=ax, lo=lo, hi=hi, s0=s0, s1=s1,
                        t=(hi-lo)*1000, name=p["name"], area=p.get("area_m2", 0)))
    return man, out


def align(A, B):
    """The rotation (a right angle) and shift that put B's walls on A's."""
    best = None
    for k in range(4):
        th = np.radians(90*k); c, s = np.cos(th), np.sin(th)
        R = np.array([[c, s], [-s, c]])
        BB = []
        for b in B:
            p0 = np.array([b["lo"], b["s0"]]) if b["axis"] == 0 else np.array([b["s0"], b["lo"]])
            p1 = np.array([b["hi"], b["s1"]]) if b["axis"] == 0 else np.array([b["s1"], b["hi"]])
            q0, q1 = p0 @ R, p1 @ R
            ax = b["axis"] if abs(R[0, 0]) > 0.5 else 1-b["axis"]
            BB.append(dict(b, axis=ax, lo=min(q0[ax], q1[ax]), hi=max(q0[ax], q1[ax]),
                           s0=min(q0[1-ax], q1[1-ax]), s1=max(q0[1-ax], q1[1-ax])))
        # the shift that lines up the most faces, found on the face coordinates
        sh = np.zeros(2)
        for _ in range(4):
            adj = np.zeros(2)
            for ax in (0, 1):
                dd = [min((a["lo"]-b["lo"] for b in BB if b["axis"] == ax
                           and min(a["s1"], b["s1"]+sh[1-ax]) - max(a["s0"], b["s0"]+sh[1-ax]) > 0.4),
                          key=abs, default=None)
                      for a in A if a["axis"] == ax]
                dd = [x for x in dd if x is not None and abs(x) < 1.2]
                if dd:
                    adj[ax] = float(np.median(dd))
            sh += adj
            if np.abs(adj).max() < 1e-4:
                break
        for b in BB:
            b["lo"] += sh[b["axis"]]; b["hi"] += sh[b["axis"]]
        score = 0
        for a in A:
            for b in BB:
                if b["axis"] == a["axis"] and abs(a["lo"]-b["lo"]) < 0.05:
                    score += 1
        if best is None or score > best[0]:
            best = (score, k, sh, BB)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a"); ap.add_argument("b")
    x = ap.parse_args()
    manA, A = faces(x.a); manB, B = faces(x.b)
    print(f"A {x.a}: {len(A)} walls, ceiling {manA['modal_ceiling_height_mm']:.0f} mm")
    print(f"B {x.b}: {len(B)} walls, ceiling {manB['modal_ceiling_height_mm']:.0f} mm")
    score, k, sh, BB = align(A, B)
    print(f"registered: rotate {90*k} deg, shift [{sh[0]:+.3f} {sh[1]:+.3f}] m, "
          f"{score} faces within 50 mm")

    pairs = []
    for a in A:
        cand = [b for b in BB if b["axis"] == a["axis"]
                and min(a["s1"], b["s1"]) - max(a["s0"], b["s0"]) > 0.4]
        if not cand:
            continue
        b = min(cand, key=lambda b: abs(a["lo"]-b["lo"]) + abs(a["hi"]-b["hi"]))
        if abs(a["lo"]-b["lo"]) < 0.25:
            pairs.append((a, b))
    if not pairs:
        print("no walls matched -- the registration failed"); return
    near = np.array([(a["lo"]-b["lo"]) for a, b in pairs])*1000
    far = np.array([(a["hi"]-b["hi"]) for a, b in pairs])*1000
    th = np.array([(a["t"]-b["t"]) for a, b in pairs])
    print(f"\n{len(pairs)} walls matched between the two scans")
    for nm, v in (("near face", near), ("far face", far), ("thickness", th)):
        print(f"  {nm:10s}: median |diff| {np.median(np.abs(v)):5.1f} mm, "
              f"90th {np.percentile(np.abs(v), 90):6.1f}, worst {np.abs(v).max():6.1f}, "
              f"bias {np.median(v):+5.1f}")
    ch = manA["modal_ceiling_height_mm"] - manB["modal_ceiling_height_mm"]
    print(f"  ceiling height: {ch:+.0f} mm apart")
    print(f"  thickness vocabulary: A {manA.get('thickness_modes_mm')}  "
          f"B {manB.get('thickness_modes_mm')}")


if __name__ == "__main__":
    main()
