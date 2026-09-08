"""validate_rooms.py
--------------------
Per-room validation vs the manual survey using the WATERSHED ROOM MASKS (which
bound every segmented room cleanly) instead of the fragmentary reconstructed
walls. For each room: clear W x L from the room-mask extent + ceiling height
from the points inside it (peak/mode planes). Then match to the survey and
report deltas.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\validate_rooms.py \
      output/koushik_iso/isolated.las output2\\koushik_all\\skeleton_3d\\continuous
"""
import sys, json
from pathlib import Path
import numpy as np
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.experiments.explain_lidar_to_3d import reconstruct, CELL

MANUAL = {   # survey room -> (clear w, l mm), ceiling mm
    "Living":          (dict(w=4947, l=6540), 2697),
    "MBR":             (dict(w=3103, l=5823), 2730),
    "KBR":             (dict(w=2951, l=4790), 2737),
    "Kitchen":         (dict(w=2210, l=4991), 2700),
    "GBR":             (dict(w=3390, l=3398), 2728),
    "Foyer":           (dict(w=1462, l=1893), 2695),
    "Common washroom": (dict(w=1451, l=2369), 2150),
    "MBR washroom":    (dict(w=1470, l=2415), 2150),
    "KBR washroom":    (dict(w=1472, l=2356), 2161),
    "Utility":         (dict(w=1521, l=1537), 2122),
}


def peak(vals, sel_lo, sel_hi):
    v = vals[(vals > sel_lo) & (vals < sel_hi)]
    if len(v) < 200:
        return None
    e = np.arange(v.min() - 0.01, v.max() + 0.02, 0.01)
    h, _ = np.histogram(v, e)
    zp = 0.5 * (e[:-1] + e[1:])[int(np.argmax(h))]
    return float(np.median(v[np.abs(v - zp) < 0.03]))


def main(las, out_dir):
    out = Path(out_dir)
    R = reconstruct(las)
    mk = R["mk"]; xmin, ymax = R["xmin"], R["ymax"]
    x, y, z = R["x"], R["y"], R["z"]
    # map every point to a pixel so we can pull per-room z
    W, H = R["W"], R["H"]
    pc = np.clip(((x - xmin) / CELL).astype(int), 0, W - 1)
    pr = np.clip(((ymax - y) / CELL).astype(int), 0, H - 1)
    plab = mk[pr, pc]

    rooms = {}
    for L in R["room_labels"]:
        rows, cols = np.where(mk == L)
        if len(rows) * CELL * CELL < 1.0:
            continue
        wx = (cols.max() - cols.min()) * CELL * 1000
        wy = (rows.max() - rows.min()) * CELL * 1000
        zz = z[plab == L]
        h = None
        if len(zz) > 500:
            zf = peak(zz, zz.min() - 0.01, zz.min() + 0.40)
            zc = peak(zz, zz.max() - 0.40, zz.max() + 0.01)
            if zf is not None and zc is not None:
                h = (zc - zf) * 1000
        rooms[int(L)] = dict(w=min(wx, wy), l=max(wx, wy),
                             area=len(rows) * CELL * CELL, h=h)

    # greedy match: height gates wet vs habitable, footprint refines
    def cost(rc, mn):
        c = abs(rc["w"] - mn[0]["w"]) + abs(rc["l"] - mn[0]["l"])
        if rc["h"] is not None:
            c += 2.0 * abs(rc["h"] - mn[1])     # weight height (separates wet/dry)
        return c

    used = set(); rows_out = []
    for name, mn in sorted(MANUAL.items(), key=lambda kv: -kv[1][0]["l"]):
        cand = [r for r in rooms if r not in used]
        if not cand:
            rows_out.append((name, None, mn, None)); continue
        best = min(cand, key=lambda r: cost(rooms[r], mn))
        used.add(best); rows_out.append((name, best, mn, rooms[best]))

    print(f"reconstructed rooms >=1m2: {len(rooms)}   survey rooms: {len(MANUAL)}")
    print(f"\n{'survey room':17}{'clear W×L: survey → ours (Δ mm)':44}{'height Δ':10}")
    print("-" * 78)
    dd, dh = [], []
    for name, rid, mn, rc in rows_out:
        if rc is None:
            print(f"{name:17}(no match)"); continue
        mw, ml = mn[0]["w"], mn[0]["l"]
        dw, dl = rc["w"] - mw, rc["l"] - ml
        dd += [abs(dw), abs(dl)]
        hs = "—"
        if rc["h"] is not None:
            d = rc["h"] - mn[1]; dh.append(abs(d)); hs = f"{d:+.0f} mm"
        print(f"{name:17}{mw:.0f}×{ml:.0f} → {rc['w']:.0f}×{rc['l']:.0f} ({dw:+.0f},{dl:+.0f})".ljust(61) + hs)
    dd = np.array(dd); dh = np.array(dh)
    print("-" * 78)
    print(f"clear-dim deviation : median {np.median(dd):.0f} mm  p90 {np.percentile(dd,90):.0f} mm")
    print(f"height   deviation : median {np.median(dh):.0f} mm  p90 {np.percentile(dh,90):.0f} mm")
    for t in (25, 50, 100):
        print(f"within {t:3d} mm: dims {100*(dd<=t).mean():.0f}%  heights {100*(dh<=t).mean():.0f}%")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
