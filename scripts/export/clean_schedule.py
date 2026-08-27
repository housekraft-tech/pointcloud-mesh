"""Fewer walls, no fragments: consolidate the schedule before it is drawn.

The measurement stage is deliberately literal -- it cuts a new run wherever the
faces step, and keeps every scrap it can justify. That is right for measuring
and wrong for a model someone has to work in: 31 walls and 105 runs for a flat
with maybe 20 real walls in it.

Three passes, all of them reversible by turning the tolerances down:

  snap     face positions that agree within SNAP are the same plane. A wall
           that wanders 8 mm along its length is one wall, not four.
  merge    runs on the same plane pair that touch, overlap, or are separated
           only by a doorway's width become one run. A doorway is cut later
           anyway, so splitting the wall there gains nothing.
  prune    what is left shorter than MIN_LEN, or thinner than MIN_THICK, was
           never a wall.
"""
import argparse, json, collections
from pathlib import Path
import numpy as np

SNAP = 0.030        # m: face positions within this are the same plane
VOCAB_TOL = 0.0     # m: nudging thickness onto the vocabulary moved walls
                    # 22 mm at the median and merged nothing extra -- not worth
                    # the accuracy. Set it to 0.025 to try it again.
JOIN = 1.40         # m: a gap up to a door's width does not end a wall
MIN_LEN = 0.50      # m: shorter than this is a stub, not a wall
PIER_TOL = 0.030    # m: thicker than the wall by more than this is a pier
MIN_THICK = 0.07    # m: 55-62 mm "walls" are a face measured once, not masonry


def cluster(vals, tol):
    """Map each value to the centre of the group of values it belongs to."""
    out, order = {}, sorted(set(vals))
    grp = [order[0]] if order else []
    for v in order[1:]:
        if v - grp[-1] <= tol:
            grp.append(v)
        else:
            c = float(np.mean(grp))
            for g in grp:
                out[g] = c
            grp = [v]
    if grp:
        c = float(np.mean(grp))
        for g in grp:
            out[g] = c
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=None,
                    help="the modular manifest, for the thickness vocabulary")
    a = ap.parse_args()
    S = json.load(open(a.inp))["parts"]
    modes = []
    if a.manifest and Path(a.manifest).exists():
        modes = [t/1000.0 for t in
                 json.load(open(a.manifest)).get("thickness_modes_mm", [])]
    runs = [r for r in S if r["op"] == "run"]
    other = [r for r in S if r["op"] != "run"]

    # --- snap the faces -----------------------------------------------------
    coords = []
    for r in runs:
        ax = 0 if (r["hi"][0]-r["lo"][0]) < (r["hi"][1]-r["lo"][1]) else 1
        r["_ax"] = ax
        coords += [r["lo"][ax], r["hi"][ax]]
    snap = cluster(coords, SNAP)
    for r in runs:
        ax = r["_ax"]
        r["lo"][ax] = snap[r["lo"][ax]]
        r["hi"][ax] = snap[r["hi"][ax]]

    # --- snap the THICKNESS to what the building repeats --------------------
    #
    # A wall that measures 191, 196 and 203 mm along its length is a 196 mm
    # wall measured three times, and splitting it into three runs is precision
    # theatre: the scan's own noise is 1.7 mm per point and the faces are
    # settled to a few hundredths, but the plaster really is that uneven. The
    # far face is moved onto the nearest thickness the building repeats, and
    # only when it is already within VOCAB_TOL -- so a genuine step to a
    # different wall type is left alone.
    moved = []
    if modes and VOCAB_TOL > 0:
        for r in runs:
            ax = r["_ax"]
            near, far = r["lo"][ax], r["hi"][ax]
            t = far - near
            m_best = min(modes, key=lambda mm: abs(mm - t))
            if abs(m_best - t) <= VOCAB_TOL:
                r["hi"][ax] = near + m_best
                moved.append(abs(m_best - t))
    if moved:
        print(f"thickness snapped to {['%.0f' % (m*1000) for m in modes]} mm on "
              f"{len(moved)} runs; median move {np.median(moved)*1000:.1f} mm, "
              f"worst {max(moved)*1000:.1f} mm")

    # --- one thickness per wall line ---------------------------------------
    #
    # This is what still reads as messy. A wall measured at 179, 185, 190 and
    # 196 mm along its length is drawn as four runs with a jog between each --
    # and a jog every couple of metres is what makes a plan look like scan
    # noise rather than a building. Grouped by the face they share, each line
    # takes ONE thickness: the length-weighted median of its own measurements.
    # That is not a guess and not a vocabulary; it is what this wall measures,
    # once.
    lines = collections.defaultdict(list)
    for r in runs:
        ax = r["_ax"]
        lines[(ax, round(r["lo"][ax], 3), round(r["lo"][2], 2), round(r["hi"][2], 2))].append(r)
    spread = []; kept_steps = []; piers = []
    for k, rs in lines.items():
        ax = k[0]
        ts = np.array([r["hi"][ax] - r["lo"][ax] for r in rs])
        ls = np.array([r["hi"][1-ax] - r["lo"][1-ax] for r in rs])
        if len(rs) < 2:
            continue
        # A wall does not stop being one wall because a pier stands in it.
        #
        # Splitting the line at every thickness change is what turns a flat
        # with twenty walls into thirty-two: the wall either side of a pier
        # becomes two objects. A designer draws the wall through, then adds
        # the pier onto its face. So the line takes its dominant thickness for
        # its whole length, and anything thicker comes back as relief -- which
        # the builder unions into that same wall solid.
        order = np.argsort(ts)
        cum = np.cumsum(ls[order])
        med = float(ts[order][np.searchsorted(cum, cum[-1]/2)])
        spread.append((ts.max()-ts.min())*1000)
        for r in rs:
            t = r["hi"][ax] - r["lo"][ax]
            if t > med + PIER_TOL:
                # the extra depth, kept as a pier on this wall
                lo = list(r["lo"]); hi = list(r["hi"])
                lo[ax] = r["lo"][ax] + med
                piers.append(dict(name=r["name"], kind="wall", op="relief",
                                  lo=[round(v, 5) for v in lo],
                                  hi=[round(v, 5) for v in hi]))
                kept_steps.append((t - med)*1000)
            r["hi"][ax] = r["lo"][ax] + med
    if spread:
        print(f"one thickness per wall line: {len(spread)} lines flattened, steps of "
              f"{np.median(spread):.0f} mm median, {max(spread):.0f} worst")
    if kept_steps:
        print(f"{len(piers)} piers kept as relief on their wall, standing "
              f"{', '.join('%.0f' % k for k in sorted(kept_steps, reverse=True)[:5])} mm "
              f"proud -- the wall runs through behind them")

    # --- merge along the wall ----------------------------------------------
    key = lambda r: (r["_ax"], round(r["lo"][r["_ax"]], 3), round(r["hi"][r["_ax"]], 3),
                     round(r["lo"][2], 2), round(r["hi"][2], 2))
    groups = collections.defaultdict(list)
    for r in runs:
        groups[key(r)].append(r)
    merged, before = [], len(runs)
    for k, rs in groups.items():
        ax = k[0]
        rs.sort(key=lambda r: r["lo"][1-ax])
        cur = None
        for r in rs:
            if cur and r["lo"][1-ax] - cur["hi"][1-ax] <= JOIN:
                cur["hi"][1-ax] = max(cur["hi"][1-ax], r["hi"][1-ax])
                cur["_n"] += 1
            else:
                if cur:
                    merged.append(cur)
                cur = dict(r); cur["_n"] = 1
        if cur:
            merged.append(cur)

    # --- prune what was never a wall ---------------------------------------
    kept = []
    for r in merged:
        ax = r["_ax"]
        if (r["hi"][1-ax] - r["lo"][1-ax]) < MIN_LEN:
            continue
        if (r["hi"][ax] - r["lo"][ax]) < MIN_THICK:
            continue
        kept.append(r)

    # one name per merged wall, so the model has as many walls as the flat does
    kept.sort(key=lambda r: (r["_ax"], r["lo"][r["_ax"]], r["lo"][1-r["_ax"]]))
    rename = {}
    for i, r in enumerate(kept):
        new = f"wall_{i:02d}"
        rename.setdefault(r["name"], []).append(new)
        r["name"] = new
        for f in ("_ax", "_n"):
            r.pop(f, None)

    # the cuts follow the wall they were measured on
    fixed = []
    for r in other:
        who = rename.get(r["name"])
        if who:
            r = dict(r); r["name"] = who[0]
        fixed.append(r)

    # Re-fit every cut to the wall it now lives in.
    #
    # A door was measured against the run it was found on. Merging gave that
    # wall a single thickness, so a cutter sized for the old run may no longer
    # reach through -- and Solid Tools then fails rather than leaving a blind
    # hole. On the first merged build that cost 6 of 26 openings and 9 of 34
    # niches. Each cut is therefore stretched across its host wall, plus an
    # overshoot, and niches keep their measured depth from the face.
    OVER = 0.20
    refit = 0
    for r in fixed:
        if r["op"] not in ("opening", "niche"):
            continue
        ax = 0 if (r["hi"][0]-r["lo"][0]) < (r["hi"][1]-r["lo"][1]) else 1
        c = 0.5*(r["lo"][ax] + r["hi"][ax])
        host = None
        for w in kept:
            wax = 0 if (w["hi"][0]-w["lo"][0]) < (w["hi"][1]-w["lo"][1]) else 1
            if wax != ax:
                continue
            if min(r["hi"][1-ax], w["hi"][1-ax]) - max(r["lo"][1-ax], w["lo"][1-ax]) < 0.05:
                continue
            if w["lo"][ax] - 0.25 <= c <= w["hi"][ax] + 0.25:
                host = w
                break
        if host is None:
            continue
        r["name"] = host["name"]
        if r["op"] == "opening":
            r["lo"][ax] = host["lo"][ax] - OVER
            r["hi"][ax] = host["hi"][ax] + OVER
        else:
            depth = r["hi"][ax] - r["lo"][ax]
            near = abs(c - host["lo"][ax]) < abs(c - host["hi"][ax])
            if near:
                r["lo"][ax] = host["lo"][ax] - 0.01
                r["hi"][ax] = host["lo"][ax] + depth
            else:
                r["hi"][ax] = host["hi"][ax] + 0.01
                r["lo"][ax] = host["hi"][ax] - depth
        refit += 1
    print(f"{refit} cuts re-fitted to the wall they now belong to")

    json.dump(dict(parts=kept + fixed + piers), open(a.out, "w"))
    print(f"runs {before} -> {len(kept)}  (snap {SNAP*1000:.0f} mm, join {JOIN*1000:.0f} mm, "
          f"drop under {MIN_LEN*1000:.0f} mm)")
    print(f"walls {len({r['name'] for r in S if r['op']=='run'})} -> {len(kept)}")
    print(f"other ops carried over: " + ", ".join(
        f"{k} {v}" for k, v in collections.Counter(r['op'] for r in fixed).items()))


if __name__ == "__main__":
    main()
