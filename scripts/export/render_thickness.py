"""Does the box model follow the wall, or has it flattened it?

The viewer can stack the boxes over the scan and you can look for daylight, but
eyes are poor at judging a 20 mm step across a room. This measures it instead:
for every wall, the two faces are read off the scan at each 100 mm station and
plotted against the boxes that were fitted to them.

Read it as three lines. The grey band is the scan -- the wall's real front and
back faces, station by station. The coloured steps are the boxes. Where they sit
on top of each other the model is following the wall, pilasters and all; where
the steps cut across the band, a thickness change has been flattened.

The number in each title is the worst disagreement on that wall, in millimetres.
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STATION = 0.10
MIN_RUN_MM = 250


def profile(P, ax, s0, s1):
    u = P[:, 1-ax]; w = P[:, ax]
    n = max(1, int(np.ceil((s1-s0)/STATION)))
    lo = np.full(n, np.nan); hi = np.full(n, np.nan)
    idx = np.clip(((u-s0)/STATION).astype(int), 0, n-1)
    for i in range(n):
        m = idx == i
        if m.sum() >= 12:
            lo[i] = np.percentile(w[m], 2)
            hi[i] = np.percentile(w[m], 98)
    return s0 + (np.arange(n)+0.5)*STATION, lo, hi


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--walls", type=int, default=8)
    a = ap.parse_args()
    d = Path(a.dir)
    man = json.load(open(d/"manifest.json"))
    T = np.load(a.cache)["T"].astype(np.int64)
    V = np.load(d/"verts.npy").astype(np.float64)
    L = np.load(d/"labels.npy")
    names = json.load(open(d/"names.json"))
    C = V[T].mean(axis=1)

    # the boxes, grouped back to the wall they came from
    boxes = {}
    cur, vs = None, []
    for line in open(d/"boxes.obj"):
        if line.startswith("o "):
            if cur and vs:
                boxes.setdefault(cur.rsplit("_", 1)[0] if cur not in names else cur,
                                 []).append(np.array(vs))
            cur = line[2:].strip(); vs = []
        elif line.startswith("v "):
            vs.append([float(x) for x in line.split()[1:4]])
    if cur and vs:
        boxes.setdefault(cur.rsplit("_", 1)[0] if cur not in names else cur,
                         []).append(np.array(vs))

    walls = [p for p in man["parts"] if p["kind"] in ("wall", "parapet")
             and "across_m" in p]
    walls.sort(key=lambda p: -p.get("area_m2", 0))
    walls = walls[:a.walls]
    fig, axes = plt.subplots(len(walls), 1, figsize=(13, 2.3*len(walls)))
    if len(walls) == 1:
        axes = [axes]
    worst_all, med_all = [], []
    for axp, p in zip(axes, walls):
        nm = p["name"]
        ax = 0 if p["axis"] == "x" else 1
        pid = names.index(nm)
        P = C[L == pid]
        s, lo, hi = profile(P, ax, *p["along_m"])
        med = []
        axp.fill_between(s, lo*1000, hi*1000, color="#b9bec7", alpha=.85,
                         label="scan: the wall's two faces", step="mid")
        worst = 0.0
        for k, vs in enumerate(boxes.get(nm, [])):
            b0, b1 = vs.min(0), vs.max(0)
            xs = [b0[1-ax], b1[1-ax]]
            axp.plot(xs, [b0[ax]*1000]*2, color="#2f6fd0", lw=2.2,
                     label="box" if k == 0 else None)
            axp.plot(xs, [b1[ax]*1000]*2, color="#2f6fd0", lw=2.2)
            inside = (s >= b0[1-ax]) & (s <= b1[1-ax]) & ~np.isnan(lo)
            if inside.any():
                e = np.concatenate([np.abs(lo[inside]-b0[ax]),
                                    np.abs(hi[inside]-b1[ax])])*1000
                worst = max(worst, float(np.nanmax(e)))
                med.extend(e[~np.isnan(e)].tolist())
        worst_all.append(worst)
        med_all.append(float(np.median(med)) if med else 0.0)
        t = p.get("thickness_mm") or p.get("thickness_raw_mm")
        axp.set_title(f"{nm} — {p['length_mm']} mm long, "
                      f"thickness {int(t) if t else 'one face only'}"
                      f"{' mm' if t else ''}, {len(boxes.get(nm, []))} box"
                      f"{'es' if len(boxes.get(nm, [])) != 1 else ''}, "
                      f"typical {np.median(med) if med else 0:.0f} mm, "
                      f"worst {worst:.0f} mm", fontsize=9)
        axp.set_ylabel("across, mm", fontsize=8)
        axp.tick_params(labelsize=8)
        axp.grid(alpha=.25)
        if p is walls[0]:
            axp.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(d/"thickness_profiles.png", dpi=130)
    print(f"wrote {d/'thickness_profiles.png'}")
    print(f"box vs scan over {len(walls)} walls: typical disagreement "
          f"{np.median(med_all):.0f} mm; worst per wall median "
          f"{np.median(worst_all):.0f} mm, max {max(worst_all):.0f} mm "
          f"(the worst values sit on features narrower than the {MIN_RUN_MM:.0f} mm "
          f"run threshold)")


if __name__ == "__main__":
    main()
