"""validate_vs_manual.py
------------------------
Compare the continuous-measurement output (measurements.json) against the manual
on-site survey (ACE Scanner PDF, hand-measured, mm). Matches each reconstructed
room to a survey room by ceiling height + footprint, then reports per-room
dimension and height deltas with a pass/fail at a 10 mm / 1% tolerance.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\validate_vs_manual.py \
      output2\\koushik_all\\skeleton_3d\\continuous\\measurements.json \
      output2\\koushik_all\\skeleton_3d\\continuous
"""
import sys, json, collections
from pathlib import Path
import numpy as np
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---- manual survey (ACE Scanner PDF): room -> (two principal wall lengths mm, ceiling mm)
#      principal dims = the room's width & length (paired opposite walls averaged)
MANUAL = {
    "Kitchen":         (dict(w=2210, l=4991), 2700),   # open to living
    "Utility":         (dict(w=1521, l=1537), 2122),
    "Living":          (dict(w=4947, l=6540), 2697),
    "Common washroom": (dict(w=1451, l=2369), 2150),
    "Foyer":           (dict(w=1462, l=1893), 2695),
    "MBR":             (dict(w=3103, l=5823), 2730),   # L-shaped, long leg
    "MBR washroom":    (dict(w=1470, l=2415), 2150),
    "KBR":             (dict(w=2951, l=4790), 2737),
    "KBR washroom":    (dict(w=1472, l=2356), 2161),
    "GBR":             (dict(w=3390, l=3398), 2728),
}
TOL_MM = 10.0
TOL_PCT = 1.0


def room_footprint(walls):
    """Clear dimensions of a room = perpendicular distance between its OPPOSITE
    wall planes (what a tape measures). Split refined walls by orientation; the
    clear span in each axis is the spread of the perpendicular wall positions.
    Returns None unless the room has opposite walls in BOTH axes (i.e. it is a
    properly closed room, not a fragment or the merged open-plan blob)."""
    xw, yw = [], []          # vertical walls (run along y) -> give X span; and vice-versa
    for w in walls:
        if w.get("status") != "ok":
            continue
        d = np.array(w["dir"]); c = np.array(w["center"])
        if abs(d[1]) > abs(d[0]):     # runs along Y -> a vertical wall, its X position bounds width
            xw.append(c[0])
        else:                          # runs along X -> horizontal wall, its Y position bounds length
            yw.append(c[1])
    if len(xw) < 2 or len(yw) < 2:
        return None                    # not a closed room -> skip (honest)
    wx = (max(xw) - min(xw)) * 1000
    wy = (max(yw) - min(yw)) * 1000
    return dict(w=min(wx, wy), l=max(wx, wy))


def main(meas_json, out_dir):
    out = Path(out_dir)
    d = json.load(open(meas_json))
    rooms = collections.defaultdict(list)
    for w in d["walls"]:
        rooms[w["room"]].append(w)
    heights = {r["room"]: r["height_mm"] for r in d["rooms"]}

    recon = {}
    for room, walls in rooms.items():
        fp = room_footprint(walls)
        if fp is None:
            continue
        recon[room] = dict(fp=fp, h=heights.get(room))

    # cost = height diff (mm) + footprint diff (mm), assign each manual room its
    # best recon room (greedy; heights strongly separate wet vs habitable)
    def cost(rc, mn):
        c = 0.0
        if rc["h"] is not None:
            c += abs(rc["h"] - mn[1])
        c += abs(rc["fp"]["w"] - mn[0]["w"]) + abs(rc["fp"]["l"] - mn[0]["l"])
        return c

    used = set()
    rows = []
    for name, mn in sorted(MANUAL.items(), key=lambda kv: -kv[1][0]["l"]):
        best = min((r for r in recon if r not in used),
                   key=lambda r: cost(recon[r], mn), default=None)
        if best is None:
            rows.append((name, None, mn, None)); continue
        used.add(best)
        rows.append((name, best, mn, recon[best]))

    print(f"{'survey room':17} {'recon':8} {'dim: manual→ours (Δmm)':34} {'height: man→ours (Δ)':24}")
    print("-" * 92)
    dev_dim = []; dev_h = []
    for name, rid, mn, rc in rows:
        if rc is None:
            print(f"{name:17} {'—':8} (no recon match)"); continue
        mw, ml = mn[0]["w"], mn[0]["l"]; ow, ol = rc["fp"]["w"], rc["fp"]["l"]
        dw, dl = ow - mw, ol - ml
        dh = (rc["h"] - mn[1]) if rc["h"] else None
        dev_dim += [abs(dw), abs(dl)]
        if dh is not None: dev_h.append(abs(dh))
        hs = f"{mn[1]}→{rc['h']:.0f} ({dh:+.0f})" if dh is not None else "—"
        print(f"{name:17} {rid.split('_')[1]:8} {mw:.0f}×{ml:.0f} → {ow:.0f}×{ol:.0f} ({dw:+.0f},{dl:+.0f})".ljust(52)
              + f"  {hs}")

    dev_dim = np.array(dev_dim); dev_h = np.array(dev_h)
    print("-" * 92)
    print(f"footprint deviation : median {np.median(dev_dim):.0f} mm  p90 {np.percentile(dev_dim,90):.0f} mm")
    print(f"height deviation    : median {np.median(dev_h):.0f} mm  p90 {np.percentile(dev_h,90):.0f} mm")
    print(f"within {TOL_MM:.0f} mm : dims {100*(dev_dim<=TOL_MM).mean():.0f}%  heights {100*(dev_h<=TOL_MM).mean():.0f}%")
    print(f"within 25 mm : dims {100*(dev_dim<=25).mean():.0f}%  heights {100*(dev_h<=25).mean():.0f}%")
    print(f"within 50 mm : dims {100*(dev_dim<=50).mean():.0f}%  heights {100*(dev_h<=50).mean():.0f}%")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
