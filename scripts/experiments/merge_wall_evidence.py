"""merge_wall_evidence.py
--------------------
Take each measurement from the channel that actually measures it well.

Two wall detectors now exist over the same mesh and neither wins outright:

  walls_v2  vertical faces only. Sharp runs, so its elevations are clean and
            the opening detector reads 9 doors off them. But it cannot resolve
            wall thickness: one face is all it usually sees.
  walls_v3  faces fused with top-view slices. Resolves thickness properly --
            100 mm internal partitions against 250-300 mm external walls --
            but the fused runs are fatter, the elevations blur, and doors
            degrade into unclassified "wide openings" (9 -> 6).

So do not choose. Take the GEOMETRY from v2 and the THICKNESS from v3, matching
runs by collinearity and overlap. Nothing is averaged: each number keeps the
provenance of the channel that earned it, recorded in the output.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\merge_wall_evidence.py \\
      <geometry_major_walls.json> <thickness_major_walls.json> <out_dir>
"""
import sys, json, time
from pathlib import Path
import numpy as np

PAR_ANG = 12.0     # deg, two runs this parallel may be the same wall
PAR_OFF = 0.35     # m, ...if their centre-lines are also this close
MIN_OVER = 0.30    # fraction of the shorter run that must overlap
EXTERNAL = 0.20    # m, at or above this a wall is external, not a partition


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def geom(w):
    a, b = np.array(w["p0"], float), np.array(w["p1"], float)
    d = b - a
    L = float(np.hypot(*d))
    return a, b, (d / L if L > 1e-9 else np.array([1.0, 0.0])), L


def overlap(w1, w2):
    """Overlap length of two runs, or 0 if they are not the same wall."""
    a1, b1, u1, L1 = geom(w1)
    a2, b2, u2, L2 = geom(w2)
    if abs(float(u1 @ u2)) < np.cos(np.radians(PAR_ANG)):
        return 0.0
    nrm = np.array([-u1[1], u1[0]])
    if abs(float((a2 - a1) @ nrm)) > PAR_OFF:
        return 0.0
    t = sorted([float((a2 - a1) @ u1), float((b2 - a1) @ u1)])
    lo, hi = max(0.0, t[0]), min(L1, t[1])
    return max(0.0, hi - lo)


def main(geo_path, thick_path, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    G = json.load(open(geo_path))
    T = json.load(open(thick_path))
    tw = [w for w in T["walls"] if w["kind"] == "wall"]
    log(f"geometry: {len(G['walls'])} runs from {Path(geo_path).parent.name}")
    log(f"thickness: {len(tw)} runs from {Path(thick_path).parent.name}")

    if abs(G.get("grid_angle_deg", 0) - T.get("grid_angle_deg", 0)) > 1.0:
        log(f"  WARNING grid angles differ: {G.get('grid_angle_deg')} vs "
            f"{T.get('grid_angle_deg')} -- runs may not correspond")

    out_walls, matched, ext = [], 0, 0
    for w in G["walls"]:
        rec = dict(w)
        rec["thickness_source"] = "geometry channel"
        if w["kind"] == "wall":
            _, _, _, L = geom(w)
            best, bo = None, 0.0
            for t in tw:
                o = overlap(w, t)
                if o > bo and o >= MIN_OVER * min(L, t["length_m"]):
                    bo, best = o, t
            if best is not None and best.get("thickness_two_sided"):
                matched += 1
                rec["thickness_m"] = best["thickness_m"]
                rec["thickness_source"] = "slice channel, both faces"
                rec["thickness_overlap_m"] = round(float(bo), 2)
            th = rec["thickness_m"]
            rec["wall_type"] = "external" if th >= EXTERNAL else "partition"
            ext += rec["wall_type"] == "external"
        out_walls.append(rec)

    walls = [w for w in out_walls if w["kind"] == "wall"]
    log(f"matched {matched} of {len(walls)} runs to a two-sided thickness")
    log(f"  {ext} external (>= {EXTERNAL*1000:.0f} mm), "
        f"{len(walls)-ext} partitions")
    for kind in ("external", "partition"):
        v = [w for w in walls if w.get("wall_type") == kind]
        if v:
            log(f"  {kind:<10} n={len(v):2d}  total {sum(x['length_m'] for x in v):5.1f} m"
                f"  thickness {min(1000*x['thickness_m'] for x in v):.0f}-"
                f"{max(1000*x['thickness_m'] for x in v):.0f} mm")

    js = dict(G)
    js["walls"] = out_walls
    js["source"] = (f"geometry from {Path(geo_path).parent.name}, "
                    f"thickness from {Path(thick_path).parent.name}")
    json.dump(js, open(out / "major_walls.json", "w"), indent=1)
    log(f"wrote {out/'major_walls.json'}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
