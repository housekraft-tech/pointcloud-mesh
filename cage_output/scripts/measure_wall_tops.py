"""How high does each wall actually go?

Everything so far has extruded every wall from floor to ceiling. That is wrong
for a balcony: its boundary is a parapet or railing about a metre tall, and
building it full height both invents a wall that is not there and hides the
balcony opening above it -- which is why a balcony door goes missing. It is the
same top-down blindness as the beam case, at the other end of the wall.

For each wall the occupancy grid is read row by row in z. The wall's top is the
highest height at which it still covers most of its own length; above that it
is a post, a coping, or nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"C:/Users/PC/Documents/pointcloud-mesh")
sys.path.insert(0, str(ROOT / "scripts"))
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(HERE))

from recon.openings import wall_occupancy          # noqa: E402
from recon.metrology import refine_face            # noqa: E402
from detect_openings import load_all, walls_from_rooms  # noqa: E402

CELL = 0.03
COVER = 0.45          # a wall must cover this much of its length to count
PARAPET_MAX = 1.60    # anything topping out below this is not a full wall


def wall_top(w, xyz, z_floor, z_ceiling):
    """Measured top of this wall, plus the coverage profile it came from."""
    occ, u0, z0 = wall_occupancy(w, xyz, cell_m=CELL,
                                 z_band=(z_floor, z_ceiling))
    if occ.size == 0:
        return None, None
    cover = occ.mean(axis=0)                     # fraction of u occupied per z row
    zs = z0 + (np.arange(occ.shape[1]) + 0.5) * CELL
    ok = np.where(cover >= COVER)[0]
    if ok.size == 0:
        return None, cover
    top_idx = int(ok.max())
    seed = float(zs[top_idx])
    # refine against the raw z of points near the top, so the answer is not
    # quantised to the 30 mm occupancy row
    p0 = np.asarray(w["p0"]); p1 = np.asarray(w["p1"])
    d = p1 - p0; L = float(np.linalg.norm(d)); u = d / max(L, 1e-9)
    n = np.array([-u[1], u[0]])
    rel = xyz[:, :2] - p0
    band = np.abs(rel @ n) <= w["thickness_m"] / 2 + 0.08
    along = rel @ u
    m = band & (along >= 0) & (along <= L) & (np.abs(xyz[:, 2] - seed) <= 0.25)
    if m.sum() >= 200:
        top = np.percentile(xyz[m, 2], 98.0)
    else:
        top = seed
    return float(top), cover


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="robust_c95_swin")
    ap.add_argument("--mode", default="strict")
    a = ap.parse_args()

    snap = json.loads((BASE / "snapped" /
                       f"snapped_{a.run}_{a.mode}.json").read_text())
    hz = json.loads((BASE / "snapped" /
                     f"snapped_{a.run}_hybrid.json").read_text())["report"]
    zf, zc = hz["floor_z"], hz["ceiling_z"]

    xyz, z_band, traj = load_all()
    walls = walls_from_rooms(snap["rooms"])
    print(f"{len(walls)} wall segments; storey {1000*(zc-zf):.0f} mm\n",
          flush=True)

    out = []
    for wi, w in enumerate(walls):
        top, cover = wall_top(w, xyz, zf, zc)
        if top is None:
            kind = "no-coverage"
            h = None
        else:
            h = top - zf
            kind = "full" if h >= (zc - zf) - 0.30 else (
                "parapet" if h <= PARAPET_MAX else "partial")
        L = float(np.hypot(w["p1"][0] - w["p0"][0], w["p1"][1] - w["p0"][1]))
        out.append({"wall": wi, "room": w["room"], "axis": w["axis"],
                    "coord": w["coord"], "p0": w["p0"], "p1": w["p1"],
                    "length_m": round(L, 4),
                    "top_z": None if top is None else round(top, 5),
                    "height_m": None if h is None else round(h, 4),
                    "kind": kind})

    import collections
    counts = collections.Counter(o["kind"] for o in out)
    print("wall height classes:", dict(counts), "\n")
    print(f"{'wall':>4s} {'room':>4s} {'len mm':>7s} {'top mm':>7s}  kind")
    for o in sorted(out, key=lambda o: (o["kind"], -(o["height_m"] or 0))):
        if o["kind"] in ("parapet", "partial", "no-coverage"):
            hm = "     -" if o["height_m"] is None else f"{1000*o['height_m']:7.0f}"
            print(f"{o['wall']:4d} {o['room']:4d} {1000*o['length_m']:7.0f} "
                  f"{hm}  {o['kind']}")

    dest = BASE / "openings" / f"wall_tops_{a.run}.json"
    dest.write_text(json.dumps({"floor_z": zf, "ceiling_z": zc,
                                "counts": dict(counts), "walls": out}, indent=1))
    print("\nwrote", dest)


if __name__ == "__main__":
    main()
