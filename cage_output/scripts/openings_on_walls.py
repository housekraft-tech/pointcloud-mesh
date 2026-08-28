"""Detect openings on the WALL graph rather than on room edges.

Two things change, and both matter for the doors that were going missing:

  * A wall is measured once, on its own centre-line, using its own measured
    thickness -- so a doorway is looked for in the right place with the right
    band, instead of twice from two faces 190 mm apart.
  * The wall already knows which room is on each side, so an opening is joined
    to both rooms by construction. That is what was failing before: an opening
    found from a bedroom never registered against the balcony beyond it.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"C:/Users/PC/Documents/pointcloud-mesh")
sys.path.insert(0, str(ROOT / "scripts"))
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(HERE))

from recon.openings import (wall_occupancy, find_voids, refine_edges,   # noqa: E402
                            visibility_gate, classify_opening)
from recon.trajectory import wall_crossings                              # noqa: E402
from detect_openings import load_all, PRIORS                             # noqa: E402

CELL = 0.03


def to_wall_dict(w):
    lo, hi = w["lo"], w["hi"]
    c = w["centre"]
    if w["axis"] == "x":                       # holds x, runs along y
        p0, p1, direction = (c, lo), (c, hi), "y"
    else:
        p0, p1, direction = (lo, c), (hi, c), "x"
    return {"p0": list(p0), "p1": list(p1), "direction": direction,
            "thickness_m": float(w["thickness_m"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="robust_c95_swin")
    a = ap.parse_args()

    g = json.loads((BASE / "openings" /
                    f"wall_graph_{a.run}.json").read_text())
    hz = json.loads((BASE / "snapped" /
                     f"snapped_{a.run}_hybrid.json").read_text())["report"]
    zf, zc = hz["floor_z"], hz["ceiling_z"]
    storey = zc - zf

    xyz, z_band, traj = load_all()
    leaf_path = BASE / "openings" / f"leaves_{a.run}.json"
    if leaf_path.exists():
        for lf in json.loads(leaf_path.read_text())["leaves"]:
            a0 = np.array(lf["end_a"]); b0 = np.array(lf["end_b"])
            d = b0 - a0; L2 = float(d @ d)
            if L2 < 1e-9:
                continue
            t = np.clip(((xyz[:, :2] - a0) @ d) / L2, 0, 1)
            dist = np.linalg.norm(xyz[:, :2] - (a0 + t[:, None] * d), axis=1)
            xyz = xyz[dist > lf["thickness_m"] / 2 + 0.05]

    from scipy.spatial import cKDTree
    kdt = cKDTree(xyz)

    walls = [w for w in g["walls"] if w["hi"] - w["lo"] >= 0.6]
    wds = [to_wall_dict(w) for w in walls]
    crossings = wall_crossings(traj, wds) if len(traj) else {}
    print(f"{len(walls)} walls of usable length "
          f"({sum(1 for w in walls if w['kind']=='interior')} interior)",
          flush=True)

    found = []
    for wi, (w, wd) in enumerate(zip(walls, wds)):
        occ, u0, z0 = wall_occupancy(wd, xyz, cell_m=CELL, z_band=(zf, zc))
        if occ.size == 0:
            continue
        L = w["hi"] - w["lo"]
        ext = w["kind"] == "exterior"
        cu = crossings.get(wi, []) if isinstance(crossings, dict) else []
        for vc in find_voids(occ, u0, z0, cell_m=CELL,
                             min_w_m=0.55, min_h_m=0.55):
            v = refine_edges(vc, wd, xyz, search_m=0.15, bin_m=0.02)
            width = v["u1"] - v["u0"]
            height = v["z1"] - v["z0"]
            sill = v["z0"] - zf
            head = v["z1"] - zf
            if width > 0.88 * L and height > 0.90 * storey:
                continue
            unscanned = ext and width > 0.85 * L and width > 3.0
            if sill >= 1.80 and head >= storey - 0.25:
                kind, seen = "high_level_void", None
            else:
                kind = classify_opening(v, cu, zf, zc, PRIORS, exterior=ext)
                if kind == "balcony_door" and not ext:
                    kind = "wide_opening"
                seen = visibility_gate(v, wd, traj, kdt) if len(traj) else None
            if kind == "unknown_opening":
                continue
            uvec = ((np.array(wd["p1"]) - np.array(wd["p0"])) / max(L, 1e-9))
            c0 = np.array(wd["p0"]) + uvec * v["u0"]
            c1 = np.array(wd["p0"]) + uvec * v["u1"]
            found.append({
                "wall": wi, "kind": kind, "axis": w["axis"],
                "coord": w["centre"], "wall_kind": w["kind"],
                "rooms": w["rooms"], "exterior": bool(ext),
                "thickness_m": w["thickness_m"],
                "p0": [round(float(c0[0]), 4), round(float(c0[1]), 4)],
                "p1": [round(float(c1[0]), 4), round(float(c1[1]), 4)],
                "width_m": round(width, 4), "height_m": round(height, 4),
                "sill_m": round(sill, 4), "head_m": round(head, 4),
                "z0": round(v["z0"], 5), "z1": round(v["z1"], 5),
                "seen_through": None if seen is None else bool(seen),
                "confident": bool(not unscanned and seen is not False),
                "note": ("spans an exterior wall - likely unscanned"
                         if unscanned else
                         ("no clear line of sight - likely a furniture shadow"
                          if seen is False else "")),
            })

    counts = collections.Counter(o["kind"] for o in found)
    conf = [o for o in found if o["confident"]]
    joined = {frozenset(o["rooms"]) for o in conf if len(o["rooms"]) == 2}
    print(f"\n{len(found)} openings ({len(conf)} confident)  {dict(counts)}")
    print(f"room pairs now connected by an opening: {len(joined)}")
    print(f"\n{'kind':16s} {'w mm':>6s} {'h mm':>6s} {'sill':>6s} "
          f"{'thk':>5s}  rooms")
    for o in sorted(conf, key=lambda o: (o["kind"], -o["width_m"])):
        print(f"{o['kind']:16s} {1000*o['width_m']:6.0f} "
              f"{1000*o['height_m']:6.0f} {1000*o['sill_m']:6.0f} "
              f"{1000*o['thickness_m']:5.0f}  {o['rooms']}")

    dest = BASE / "openings" / f"openings_on_walls_{a.run}.json"
    dest.write_text(json.dumps({
        "report": {"n_walls": len(walls), "n_openings": len(found),
                   "n_confident": len(conf), "counts": dict(counts),
                   "room_pairs_connected": sorted(sorted(p) for p in joined),
                   "z_floor": zf, "z_ceiling": zc},
        "openings": found}, indent=1))
    print("\nwrote", dest)


if __name__ == "__main__":
    main()
