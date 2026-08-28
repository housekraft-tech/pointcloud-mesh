"""Doors, windows and balcony doors on the snapped walls.

Reuses recon.openings, which keeps height resolved: a wall's points are binned
into a (u, z) grid rather than collapsed top-down, so a door reads as a
floor-to-header void column, a window as a mid-band void, and a furniture
shadow as a void that the scanner never actually walked or saw through.

That (u, z) grid is the same idea as slicing the storey by height -- here
applied per wall instead of over the whole plan, which is what lets an opening
be located along a specific wall rather than merely detected somewhere.
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

from recon.io_las import load_scan                       # noqa: E402
from recon.isolate import select_z_band, isolate_unit    # noqa: E402
from recon.trajectory import approx_trajectory, wall_crossings  # noqa: E402
from recon.openings import (wall_occupancy, find_voids, refine_edges,  # noqa: E402
                            visibility_gate, classify_opening)

YAW_DEG = 5.229
PRIORS = {
    "door_h_m": 2.05, "door_h_tol_m": 0.30,
    "balcony_min_w_m": 1.30, "window_min_sill_m": 0.35,
}
WALL_THICK = 0.20
CELL = 0.03


def load_all(max_points=8_000_000):
    scan = load_scan(str(ROOT / "koushikexport.las"), max_points=max_points)
    z_band = select_z_band(scan.xyz[:, 2])
    unit, _ = isolate_unit(scan, np.zeros((0, 3)), z_band)
    xyz = unit.xyz.copy()
    centre = xyz[:, :2].mean(axis=0)
    xyz[:, :2] -= centre
    th = np.deg2rad(-YAW_DEG)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    xyz[:, :2] = xyz[:, :2] @ R.T

    traj = np.zeros((0, 3))
    if unit.gps_time is not None:
        t = approx_trajectory(unit.gps_time, unit.xyz)
        if len(t):
            t = np.asarray(t, dtype=float).copy()
            t[:, :2] -= centre
            t[:, :2] = t[:, :2] @ R.T
            traj = t
    return xyz, z_band, traj


def walls_from_rooms(rooms):
    """One wall dict per room edge. direction = the axis the wall RUNS along."""
    walls = []
    for r in rooms:
        for e in r["edges"]:
            c = e["snapped"]
            if e["axis"] == "x":            # holds x, runs along y
                p0, p1, direction = (c, e["lo"]), (c, e["hi"]), "y"
            else:
                p0, p1, direction = (e["lo"], c), (e["hi"], c), "x"
            if np.hypot(p1[0] - p0[0], p1[1] - p0[1]) < 0.6:
                continue
            walls.append({"room": r["id"], "p0": list(p0), "p1": list(p1),
                          "direction": direction, "thickness_m": WALL_THICK,
                          "axis": e["axis"], "coord": float(c)})
    return walls


def is_exterior(wall, rooms_xy, step=0.35):
    """True if no other room sits on the far side of this wall."""
    from shapely.geometry import Point
    p0 = np.array(wall["p0"]); p1 = np.array(wall["p1"])
    mid = 0.5 * (p0 + p1)
    d = p1 - p0
    n = np.array([-d[1], d[0]])
    n = n / (np.linalg.norm(n) + 1e-9)
    own = wall["room"]
    for s in (+1, -1):
        probe = Point(*(mid + s * step * n))
        if any(i != own and poly.contains(probe) for i, poly in rooms_xy):
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="robust_c95_swin")
    ap.add_argument("--mode", default="strict")
    a = ap.parse_args()

    snap = json.loads((BASE / "snapped" /
                       f"snapped_{a.run}_{a.mode}.json").read_text())
    hz = json.loads((BASE / "snapped" /
                     f"snapped_{a.run}_hybrid.json").read_text())["report"]
    z_floor, z_ceiling = hz["floor_z"], hz["ceiling_z"]

    print("loading scan + trajectory...", flush=True)
    xyz, z_band, traj = load_all()
    print(f"  trajectory: {len(traj)} poses", flush=True)

    from shapely.geometry import Polygon
    rooms_xy = [(r["id"], Polygon(r["polygon_m"])) for r in snap["rooms"]]
    walls = walls_from_rooms(snap["rooms"])
    print(f"  {len(walls)} wall segments from {len(snap['rooms'])} rooms",
          flush=True)

    # Strip open door leaves before measuring anything. A leaf standing in its
    # own doorway makes the opening read as half its true width; removing the
    # panel's points lets the void close up into one full-width opening.
    leaf_path = BASE / "openings" / f"leaves_{a.run}.json"
    n_leaf_pts = 0
    if leaf_path.exists():
        leaves = json.loads(leaf_path.read_text())["leaves"]
        drop = np.zeros(len(xyz), bool)
        for lf in leaves:
            a0 = np.array(lf["end_a"]); b0 = np.array(lf["end_b"])
            d = b0 - a0; L2 = float(d @ d)
            if L2 < 1e-9:
                continue
            t = np.clip(((xyz[:, :2] - a0) @ d) / L2, 0.0, 1.0)
            closest = a0[None, :] + t[:, None] * d[None, :]
            dist = np.linalg.norm(xyz[:, :2] - closest, axis=1)
            drop |= dist <= (lf["thickness_m"] / 2 + 0.05)
        n_leaf_pts = int(drop.sum())
        xyz = xyz[~drop]
        print(f"  stripped {n_leaf_pts:,} points on {len(leaves)} open door "
              f"leaf/leaves", flush=True)

    crossings = wall_crossings(traj, walls) if len(traj) else {}

    # The gate asks whether the scanner ever had a clear line of sight THROUGH
    # a void. It is aimed at furniture shadows: a sofa in front of a solid wall
    # leaves an occupancy hole, but every ray toward it hits the sofa. Note the
    # limit up front -- an unscanned surface is also "clear", so the gate
    # separates shadow from opening, not glass from missing data.
    from scipy.spatial import cKDTree
    kdt = cKDTree(xyz)
    print("  built kd-tree for the visibility gate", flush=True)

    found = []
    for wi, w in enumerate(walls):
        occ, u0, z0 = wall_occupancy(w, xyz, cell_m=CELL,
                                     z_band=(z_floor, z_ceiling))
        if occ.size == 0:
            continue
        voids = find_voids(occ, u0, z0, cell_m=CELL, min_w_m=0.55, min_h_m=0.55)
        length = float(np.hypot(w["p1"][0] - w["p0"][0],
                                w["p1"][1] - w["p0"][1]))
        ext = is_exterior(w, rooms_xy)
        cu = crossings.get(wi, []) if isinstance(crossings, dict) else []
        for v_coarse in voids:
            # find_voids works on a 30 mm occupancy grid, so every extent it
            # returns is a multiple of the cell. refine_edges re-solves all
            # four edges against the raw point-density half-max crossing, so
            # the reported width and head height are measured, not quantised.
            v = refine_edges(v_coarse, w, xyz, search_m=0.15, bin_m=0.02)
            width = v["u1"] - v["u0"]
            height = v["z1"] - v["z0"]
            sill = v["z0"] - z_floor
            head = v["z1"] - z_floor
            storey = z_ceiling - z_floor
            # Reject only a void that spans the wall in BOTH directions --
            # that is a missing wall, not an opening. Spanning only in u is
            # normal: a window can run the full width of a short wall and
            # still be bounded by a sill below and a head above.
            if width > 0.88 * length and height > 0.90 * storey:
                continue
            # A partition that stops short of the ceiling: the void sits high
            # and reaches the slab. Not a window -- the thing above a door or
            # over a half-height wall, and exactly what a top-down density map
            # would have drawn as solid wall.
            # A void running nearly the whole length of an EXTERIOR wall is
            # far more likely to be surface the scanner never saw than a
            # single 8 m window. Keep it in the record, flag it, and do not
            # cut it into the model.
            unscanned = ext and width > 0.85 * length and width > 3.0
            if sill >= 1.80 and head >= storey - 0.25:
                kind = "high_level_void"
            else:
                kind = classify_opening(v, cu, z_floor, z_ceiling, PRIORS,
                                        exterior=ext)
                # classify_opening calls any wide floor-touching void a balcony
                # door. On an INTERIOR wall that is a cased opening between two
                # rooms -- a doorway with no door -- which is a different thing
                # to build and a different thing to check against a drawing.
                if kind == "balcony_door" and not ext:
                    kind = "wide_opening"
            if kind == "unknown_opening":
                continue
            # The gate's 70 mm clearance assumes a void with air around it.
            # A void hugging the slab has every ray grazing the ceiling within
            # that radius, so it fails for geometric reasons rather than
            # obstruction. Record the answer there, but do not act on it.
            gate_valid = kind != "high_level_void"
            seen = (visibility_gate(v, w, traj, kdt)
                    if (len(traj) and gate_valid) else None)
            uvec = (np.array(w["p1"]) - np.array(w["p0"])) / max(length, 1e-9)
            c0 = np.array(w["p0"]) + uvec * v["u0"]
            c1 = np.array(w["p0"]) + uvec * v["u1"]
            found.append({
                "wall": wi, "room": w["room"], "kind": kind,
                "axis": w["axis"], "coord": w["coord"], "exterior": bool(ext),
                "p0": [round(float(c0[0]), 4), round(float(c0[1]), 4)],
                "p1": [round(float(c1[0]), 4), round(float(c1[1]), 4)],
                "width_m": round(width, 4),
                "sill_m": round(v["z0"] - z_floor, 4),
                "head_m": round(v["z1"] - z_floor, 4),
                "height_m": round(height, 4),
                "width_coarse_m": round(v_coarse["u1"] - v_coarse["u0"], 4),
                "height_coarse_m": round(v_coarse["z1"] - v_coarse["z0"], 4),
                "z0": round(v["z0"], 5), "z1": round(v["z1"], 5),
                "walked": bool(any(v["u0"] <= u <= v["u1"] for u in cu)),
                "seen_through": None if seen is None else bool(seen),
                "gate_applies": bool(gate_valid),
                "confident": bool(not unscanned and (seen is not False)),
                "note": ("spans an exterior wall - likely unscanned, not glazed" if unscanned
                          else ("no clear line of sight through it - likely a furniture shadow"
                                if seen is False else "")),
            })

    # --- rejoin openings that an open door leaf split in two ----------------
    # A leaf standing open is a solid vertical panel across its own doorway, so
    # the void finder sees the clear side and stops at the leaf. Two voids on
    # the same wall sharing a sill and a head, separated by about a leaf width,
    # are one opening -- which is how an 1800 mm balcony door with one leaf
    # open and one leaf glazed reads as a ~900 mm "door".
    by_wall = {}
    for o in found:
        by_wall.setdefault(o["wall"], []).append(o)
    merged = []
    for wi, group in by_wall.items():
        j = 0 if group[0]["axis"] == "y" else 1
        group.sort(key=lambda o: min(o["p0"][j], o["p1"][j]))
        cur = None
        for o in group:
            if cur is None:
                cur = dict(o)
                continue
            lo_c, hi_c = sorted([cur["p0"][j], cur["p1"][j]])
            lo_o, hi_o = sorted([o["p0"][j], o["p1"][j]])
            gap = lo_o - hi_c
            same_sill = abs(o["sill_m"] - cur["sill_m"]) <= 0.06
            same_head = abs(o["head_m"] - cur["head_m"]) <= 0.08
            if 0 < gap <= 1.10 and same_sill and same_head:
                pts = sorted([cur["p0"], cur["p1"], o["p0"], o["p1"]],
                             key=lambda q: q[j])
                cur["p0"], cur["p1"] = pts[0], pts[-1]
                cur["width_m"] = round(abs(pts[-1][j] - pts[0][j]), 4)
                cur["leaf_gap_mm"] = round(1000 * gap, 1)
                cur["merged_parts"] = cur.get("merged_parts", 1) + 1
                cur["note"] = (cur.get("note") or "") +                     " rejoined across an open door leaf;"
                # a rejoined floor-touching opening this wide is a balcony door
                if cur["sill_m"] <= 0.15 and cur["width_m"] >= PRIORS["balcony_min_w_m"]:
                    cur["kind"] = "balcony_door" if cur["exterior"] else "wide_opening"
            else:
                merged.append(cur)
                cur = dict(o)
        if cur is not None:
            merged.append(cur)
    n_before, found = len(found), merged
    print(f"  rejoined {n_before - len(found)} leaf-split void pair(s)",
          flush=True)

    # A door between two rooms is found twice, once from each side.
    found.sort(key=lambda o: -o["width_m"])
    uniq = []
    for o in found:
        dup = False
        for k in uniq:
            if k["axis"] != o["axis"] or abs(k["coord"] - o["coord"]) > 0.35:
                continue
            j = 0 if o["axis"] == "y" else 1
            lo_a, hi_a = sorted([o["p0"][j], o["p1"][j]])
            lo_b, hi_b = sorted([k["p0"][j], k["p1"][j]])
            ov = min(hi_a, hi_b) - max(lo_a, lo_b)
            if ov > 0.5 * min(hi_a - lo_a, hi_b - lo_b):
                dup = True
                k.setdefault("also_room", []).append(o["room"])
                break
        if not dup:
            uniq.append(o)

    import collections
    counts = collections.Counter(o["kind"] for o in uniq)
    report = {"run": a.run, "mode": a.mode, "priors": PRIORS,
              "n_walls": len(walls), "n_raw": len(found), "n_unique": len(uniq),
              "counts": dict(counts),
              "z_floor": z_floor, "z_ceiling": z_ceiling,
              "trajectory_poses": int(len(traj)),
              "leaf_points_stripped": n_leaf_pts}
    out = BASE / "openings"
    out.mkdir(exist_ok=True)
    (out / f"openings_{a.run}.json").write_text(
        json.dumps({"report": report, "openings": uniq}, indent=1))

    print(json.dumps(report, indent=1))
    print(f"\n{'kind':14s} {'w mm':>7s} {'h mm':>7s} {'sill mm':>8s} "
          f"{'head mm':>8s} {'ext':>4s} {'walked':>7s}")
    for o in sorted(uniq, key=lambda x: (x["kind"], -x["width_m"])):
        print(f"{o['kind']:14s} {1000*o['width_m']:7.0f} {1000*o['height_m']:7.0f} "
              f"{1000*o['sill_m']:8.0f} {1000*o['head_m']:8.0f} "
              f"{str(o['exterior']):>4s} {str(o['walked']):>7s}")
    print("\nwrote", out / f"openings_{a.run}.json")


if __name__ == "__main__":
    main()
