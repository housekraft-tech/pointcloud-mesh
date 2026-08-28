"""One wall per wall, instead of one wall per room edge.

CAGE hands back each room as an independent polygon, so two rooms either side
of a partition never share a boundary: their edges sit 100-250 mm apart with
the wall in between. Every consumer so far has treated those as two separate
walls, which is why the floor slab has gaps it cannot bridge, why an opening
found from one room does not register against its neighbour, and why rooms that
obviously connect show no door.

Nothing needs moving. Both edges are already correctly measured -- they are the
two faces of one wall, and the gap between them is its thickness. What was
missing is a structure that says so. This builds it:

  * two edges on the same axis, a plausible wall thickness apart, whose spans
    overlap  ->  one interior wall, with a room on each side
  * anything left over                          ->  one exterior wall, one face
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(HERE))

THICK_RANGE = (0.07, 0.34)
MIN_OVERLAP = 0.30
EXT_SKIN = 0.125


def edges_of(rooms):
    out = []
    for r in rooms:
        for ei, e in enumerate(r["edges"]):
            out.append({"room": r["id"], "edge": ei, "axis": e["axis"],
                        "coord": float(e["snapped"]),
                        "lo": float(e["lo"]), "hi": float(e["hi"]),
                        "stderr_mm": e.get("stderr_mm")})
    return out


def pair_walls(edges):
    """Match edges into walls by GLOBAL score order, not index order.

    Pairing greedily as edges come up lets an early mediocre match consume an
    edge that a much better pair needed, and the better pair is then lost --
    which cost 9 of 22 walls, including the bedroom-to-living partition.
    Scoring every candidate first and taking them best-first recovers them.
    """
    cands = []
    for ax in ("x", "y"):
        es = [e for e in edges if e["axis"] == ax]
        for i in range(len(es)):
            for j in range(i + 1, len(es)):
                a, b = es[i], es[j]
                if a["room"] == b["room"]:
                    continue
                t = abs(a["coord"] - b["coord"])
                if not (THICK_RANGE[0] <= t <= THICK_RANGE[1]):
                    continue
                ov = min(a["hi"], b["hi"]) - max(a["lo"], b["lo"])
                if ov < MIN_OVERLAP:
                    continue
                cands.append((ov - 2 * t, ax, a, b, t, ov))
    cands.sort(key=lambda c: -c[0])

    # An edge is NOT consumed by its first wall. One long bedroom wall faces a
    # bathroom, then a corridor, then another bedroom along its length: those
    # are three walls sharing one face, over disjoint spans. Exclusive matching
    # kept only the first and dropped the rest.
    walls = []
    covered = {}
    for _, ax, a, b, t, ov in cands:
        lo, hi = max(a["lo"], b["lo"]), min(a["hi"], b["hi"])
        clash = False
        for key in (id(a), id(b)):
            for (c_lo, c_hi) in covered.get(key, []):
                if min(hi, c_hi) - max(lo, c_lo) > 0.25:
                    clash = True     # same stretch of face already walled
                    break
            if clash:
                break
        if clash:
            continue
        for key in (id(a), id(b)):
            covered.setdefault(key, []).append((lo, hi))
        walls.append({
            "axis": ax, "kind": "interior",
            "face_a": a["coord"], "face_b": b["coord"],
            "centre": 0.5 * (a["coord"] + b["coord"]),
            "thickness_m": round(t, 5),
            "lo": lo, "hi": hi,
            "rooms": sorted({a["room"], b["room"]}),
            "span_m": round(ov, 4),
            "stderr_mm": [a.get("stderr_mm"), b.get("stderr_mm")],
        })

    # Whatever stretch of a face no interior wall claimed is exterior.
    for e in edges:
        spans = sorted(covered.get(id(e), []))
        free, cur = [], e["lo"]
        for lo, hi in spans:
            if lo - cur > 0.25:
                free.append((cur, lo))
            cur = max(cur, hi)
        if e["hi"] - cur > 0.25:
            free.append((cur, e["hi"]))
        for lo, hi in free:
            walls.append({
                "axis": e["axis"], "kind": "exterior",
                "face_a": e["coord"], "face_b": None,
                "centre": e["coord"], "thickness_m": EXT_SKIN,
                "lo": lo, "hi": hi, "rooms": [e["room"]],
                "span_m": round(hi - lo, 4),
                "stderr_mm": [e.get("stderr_mm"), None],
            })
    return walls


def footprint(w, rooms_union):
    """The wall's own 2-D rectangle, between its two measured faces."""
    lo, hi = w["lo"], w["hi"]
    if hi - lo < 0.05:
        return None
    if w["kind"] == "interior":
        c0, c1 = sorted([w["face_a"], w["face_b"]])
    else:
        c0 = w["face_a"]
        from shapely.geometry import Point
        # push outward, away from the room this face belongs to
        probe_pos = (c0 + 0.06, 0.5 * (lo + hi)) if w["axis"] == "x" \
            else (0.5 * (lo + hi), c0 + 0.06)
        outward = -1 if rooms_union.contains(Point(*probe_pos)) else +1
        c1 = c0 + outward * EXT_SKIN
        c0, c1 = sorted([c0, c1])
    if w["axis"] == "x":
        return Polygon([(c0, lo), (c1, lo), (c1, hi), (c0, hi)])
    return Polygon([(lo, c0), (lo, c1), (hi, c1), (hi, c0)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="robust_c95_swin")
    ap.add_argument("--mode", default="strict")
    a = ap.parse_args()

    snap = json.loads((BASE / "snapped" /
                       f"snapped_{a.run}_{a.mode}.json").read_text())
    rooms = snap["rooms"]
    room_polys = {r["id"]: Polygon(r["polygon_m"]) for r in rooms}
    rooms_union = unary_union(list(room_polys.values()))

    edges = edges_of(rooms)
    walls = pair_walls(edges)
    counts = collections.Counter(w["kind"] for w in walls)
    print(f"{len(edges)} room edges -> {len(walls)} walls  {dict(counts)}",
          flush=True)

    th = [1000 * w["thickness_m"] for w in walls if w["kind"] == "interior"]
    if th:
        print(f"interior wall thickness: median {np.median(th):.1f} mm, "
              f"range {min(th):.0f}-{max(th):.0f} mm", flush=True)

    # adjacency the wall graph asserts
    adj = set()
    for w in walls:
        if len(w["rooms"]) == 2:
            adj.add(frozenset(w["rooms"]))
    print(f"room pairs joined by a shared wall: {len(adj)}", flush=True)

    fps, bad = [], 0
    for w in walls:
        fp = footprint(w, rooms_union)
        if fp is None or not fp.is_valid:
            bad += 1
            continue
        w["footprint"] = [[round(x, 5), round(y, 5)]
                          for x, y in fp.exterior.coords[:-1]]
        fps.append(fp)
    walls_2d = unary_union(fps)

    slab = unary_union([rooms_union, walls_2d])
    slab = unary_union([Polygon(p.exterior) for p in
                        (slab.geoms if hasattr(slab, "geoms") else [slab])])
    parts = len(slab.geoms) if hasattr(slab, "geoms") else 1
    print(f"floor slab: {slab.area:.2f} m2 in {parts} piece(s) "
          f"(rooms alone were {rooms_union.area:.2f} m2 in "
          f"{len(rooms_union.geoms) if hasattr(rooms_union,'geoms') else 1})",
          flush=True)

    out = {"run": a.run, "mode": a.mode,
           "n_edges": len(edges), "n_walls": len(walls),
           "counts": dict(counts), "degenerate": bad,
           "interior_thickness_mm": {
               "median": round(float(np.median(th)), 1) if th else None,
               "min": round(min(th), 1) if th else None,
               "max": round(max(th), 1) if th else None},
           "room_pairs_joined": sorted([sorted(p) for p in adj]),
           "slab_area_m2": round(float(slab.area), 2),
           "slab_pieces": parts,
           "rooms_area_m2": round(float(rooms_union.area), 2),
           "walls": walls,
           "slab_polygon": [[[round(x, 5), round(y, 5)] for x, y in
                             p.exterior.coords[:-1]]
                            for p in (slab.geoms if hasattr(slab, "geoms")
                                      else [slab])]}
    dest = BASE / "openings" / f"wall_graph_{a.run}.json"
    dest.write_text(json.dumps(out, indent=1))
    print("wrote", dest)


if __name__ == "__main__":
    main()
