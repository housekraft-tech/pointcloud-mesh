"""Measure openings on each wall FACE, not on the wall's centre-line.

Measuring on the centre-line uses a band half the wall thick plus slack, so it
projects both faces into one grid. That is fine only if the two faces are
parallel planes of one uniform slab. They are not: walls carry niches, the
thickness steps where a beam or column lands, and a doorway reveal is open on
one face while the return beside it is solid on the other. Projected together,
the solid face fills the open one and the doorway disappears -- which is how a
bedroom lost its balcony door while reading 100% occupied at every height.

Per face, with a thin band, the picture is unambiguous:

    void on BOTH faces, overlapping  ->  a through opening (door, window)
    void on ONE face only            ->  a niche: a recess into the wall,
                                         whose depth is bounded by the
                                         thickness

which is also the only way to tell those two apart from geometry.
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

from recon.openings import (wall_occupancy, find_voids, refine_edges,  # noqa: E402
                            visibility_gate)
from detect_openings import load_all, PRIORS                            # noqa: E402

CELL = 0.03
FACE_BAND = 0.07      # thin: this face only, not the one behind it
MIN_W, MIN_H = 0.55, 0.55


def face_voids(face_coord, axis, lo, hi, xyz, zf, zc):
    if axis == "x":
        wd = {"p0": [face_coord, lo], "p1": [face_coord, hi],
              "direction": "y", "thickness_m": 2 * (FACE_BAND - 0.08)}
    else:
        wd = {"p0": [lo, face_coord], "p1": [hi, face_coord],
              "direction": "x", "thickness_m": 2 * (FACE_BAND - 0.08)}
    occ, u0, z0 = wall_occupancy(wd, xyz, cell_m=CELL, band_m=FACE_BAND,
                                 z_band=(zf, zc))
    if occ.size == 0:
        return wd, []
    out = []
    for vc in find_voids(occ, u0, z0, cell_m=CELL, min_w_m=MIN_W, min_h_m=MIN_H):
        out.append(refine_edges(vc, wd, xyz, search_m=0.15, bin_m=0.02))
    return wd, out


def classify(width, height, sill, head, storey, through, exterior):
    """Sill and head decide the type; through-vs-niche decides the class.

    A full-height void is a doorway with no header captured, not an unknown:
    the old rule wanted a door 2.05 m tall and dropped anything reaching the
    slab, which is precisely what a balcony opening does.
    """
    floor_touching = sill <= 0.15
    if not through:
        if floor_touching and height < 1.4:
            return "low_niche"
        return "niche"
    if floor_touching:
        if height >= storey - 0.20:
            return "full_height_opening"
        if width >= PRIORS["balcony_min_w_m"]:
            return "balcony_door" if exterior else "wide_opening"
        return "door"
    if sill >= 0.35 and head < storey - 0.15:
        return "window"
    if sill >= 1.80:
        return "high_level_void"
    return "opening"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="robust_c95_swin_clean")
    a = ap.parse_args()

    g = json.loads((BASE / "openings" /
                    f"wall_graph_{a.run}.json").read_text())
    snap = json.loads((BASE / "snapped" /
                       f"snapped_{a.run}_strict.json").read_text())
    hz = json.loads((BASE / "snapped" /
                     f"snapped_{a.run}_hybrid.json").read_text())["report"]
    zf, zc = hz["floor_z"], hz["ceiling_z"]
    storey = zc - zf
    edges = {r["id"]: r["edges"] for r in snap["rooms"]}

    xyz, z_band, traj = load_all()
    from scipy.spatial import cKDTree
    kdt = cKDTree(xyz)

    found = []
    for wi, w in enumerate(g["walls"]):
        # use the FULL extent of the contributing edges, not just their overlap:
        # a door near one end of a wall falls outside the overlap entirely
        spans = []
        for rid in w["rooms"]:
            for e in edges.get(rid, []):
                if e["axis"] != w["axis"]:
                    continue
                if (abs(e["snapped"] - w["face_a"]) < 0.02 or
                        (w["face_b"] is not None and
                         abs(e["snapped"] - w["face_b"]) < 0.02)):
                    spans.append((e["lo"], e["hi"]))
        lo = min([s[0] for s in spans], default=w["lo"])
        hi = max([s[1] for s in spans], default=w["hi"])
        if hi - lo < 0.6:
            continue
        L = hi - lo
        ext = w["kind"] == "exterior"

        faces = [("a", w["face_a"])]
        if w["face_b"] is not None:
            faces.append(("b", w["face_b"]))
        per = {}
        wds = {}
        for tag, fc in faces:
            wds[tag], per[tag] = face_voids(fc, w["axis"], lo, hi, xyz, zf, zc)

        for tag, vs in per.items():
            other = "b" if tag == "a" else "a"
            for v in vs:
                width = v["u1"] - v["u0"]
                height = v["z1"] - v["z0"]
                sill, head = v["z0"] - zf, v["z1"] - zf
                if width > 0.92 * L and height > 0.92 * storey:
                    continue          # the wall simply is not there
                through = False
                other_solid = None
                if other in per:
                    # does the opposite face carry returns over this stretch?
                    oc, ou0, oz0 = None, None, None
                    fc_other = w["face_b"] if tag == "a" else w["face_a"]
                    if fc_other is not None:
                        _wd, _ = face_voids(fc_other, w["axis"], lo, hi,
                                            xyz, zf, zc)
                        oc, ou0, oz0 = wall_occupancy(
                            _wd, xyz, cell_m=CELL, band_m=FACE_BAND,
                            z_band=(zf, zc))
                        i0 = max(0, int((v["u0"] - ou0) / CELL))
                        i1 = min(oc.shape[0], int((v["u1"] - ou0) / CELL) + 1)
                        other_solid = (float(oc[i0:i1].mean())
                                       if i1 > i0 else 0.0)
                    for o in per[other]:
                        ov_u = min(v["u1"], o["u1"]) - max(v["u0"], o["u0"])
                        ov_z = min(v["z1"], o["z1"]) - max(v["z0"], o["z0"])
                        if ov_u > 0.35 * min(width, o["u1"] - o["u0"]) and \
                                ov_z > 0.35 * min(height, o["z1"] - o["z0"]):
                            through = True
                            break
                else:
                    through = True     # single-face exterior wall
                # "a void on one face only" is a niche only if the other face
                # is genuinely there and solid; if it was never measured, this
                # is a coverage gap and claiming a niche would be inventing one
                if (not through) and other_solid is not None and other_solid < 0.25:
                    continue
                kind = classify(width, height, sill, head, storey, through, ext)
                seen = visibility_gate(v, wds[tag], traj, kdt) if len(traj) else None
                uvec = ((np.array(wds[tag]["p1"]) - np.array(wds[tag]["p0"])) /
                        max(L, 1e-9))
                c0 = np.array(wds[tag]["p0"]) + uvec * v["u0"]
                c1 = np.array(wds[tag]["p0"]) + uvec * v["u1"]
                found.append({
                    "wall": wi, "face": tag, "kind": kind, "through": through,
                    "axis": w["axis"], "rooms": w["rooms"],
                    "exterior": bool(ext), "thickness_m": w["thickness_m"],
                    "coord": (w["face_a"] if tag == "a" else w["face_b"]),
                    "p0": [round(float(c0[0]), 4), round(float(c0[1]), 4)],
                    "p1": [round(float(c1[0]), 4), round(float(c1[1]), 4)],
                    "width_m": round(width, 4), "height_m": round(height, 4),
                    "sill_m": round(sill, 4), "head_m": round(head, 4),
                    "z0": round(v["z0"], 5), "z1": round(v["z1"], 5),
                    "seen_through": None if seen is None else bool(seen),
                    "confident": bool(seen is not False or not through),
                })

    # one record per physical opening: keep the wider face measurement
    found.sort(key=lambda o: -o["width_m"])
    uniq = []
    for o in found:
        dup = False
        for k in uniq:
            # the wall graph holds several walls along one physical partition,
            # so the same opening is measured more than once; match on WHERE it
            # is, not on which wall record found it
            if k["axis"] != o["axis"] or abs(k["coord"] - o["coord"]) > 0.35:
                continue
            if abs(k["z0"] - o["z0"]) > 0.25 or abs(k["z1"] - o["z1"]) > 0.25:
                continue
            j = 0 if o["axis"] == "y" else 1
            lo_a, hi_a = sorted([o["p0"][j], o["p1"][j]])
            lo_b, hi_b = sorted([k["p0"][j], k["p1"][j]])
            if min(hi_a, hi_b) - max(lo_a, lo_b) > 0.4 * min(hi_a - lo_a,
                                                             hi_b - lo_b):
                dup = True
                for r in o["rooms"]:
                    if r not in k["rooms"]:
                        k["rooms"].append(r)
                k["rooms"].sort()
                break
        if not dup:
            uniq.append(o)

    counts = collections.Counter(o["kind"] for o in uniq)
    conf = [o for o in uniq if o["confident"]]
    joined = {frozenset(o["rooms"]) for o in conf
              if o["through"] and len(o["rooms"]) == 2}
    print(f"\n{len(uniq)} openings ({len(conf)} confident)  {dict(counts)}")
    print(f"room pairs connected by a through opening: {len(joined)}")
    print(f"\n{'kind':20s} {'w mm':>6s} {'h mm':>6s} {'sill':>6s} "
          f"{'thru':>5s}  rooms")
    for o in sorted(conf, key=lambda o: (o["kind"], -o["width_m"])):
        print(f"{o['kind']:20s} {1000*o['width_m']:6.0f} "
              f"{1000*o['height_m']:6.0f} {1000*o['sill_m']:6.0f} "
              f"{str(o['through']):>5s}  {o['rooms']}")

    dest = BASE / "openings" / f"openings_per_face_{a.run}.json"
    dest.write_text(json.dumps({
        "report": {"n": len(uniq), "n_confident": len(conf),
                   "counts": dict(counts),
                   "room_pairs_connected": sorted(sorted(p) for p in joined),
                   "face_band_m": FACE_BAND},
        "openings": uniq}, indent=1))
    print("\nwrote", dest)


if __name__ == "__main__":
    main()
