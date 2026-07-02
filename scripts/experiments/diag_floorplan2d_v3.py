"""Diagnostic v3: toward a true architectural 2D floorplan on real data.

Builds on v2 and adds the three missing pieces the v2 image exposed:

1. u_gap split raised 1.2 -> 2.8 m: doorway/balcony-scale coverage holes stay
   INSIDE their wall run (recorded as holes) instead of splitting it in two.
2. Endpoint-to-LINE snapping: a dangling run endpoint within reach of a
   perpendicular run's interior gets extended onto it (closes T-junctions,
   which resolve_corners's endpoint-to-endpoint clustering cannot).
3. Wall-internal healing: every in-run coverage hole is tested against the
   raw cloud -- if the hole's footprint is genuinely EMPTY of mid-height
   points it is kept as an OPENING CANDIDATE (orange); if points/partial
   structure exist there it was occlusion, and the wall is healed solid.

Render: walls as thickness-filled solids (architectural style), opening
candidates orange, closed rooms blue with areas. Writes
floorplan2d_diag_v3.png/json into output/koushik_iso/.
"""
import json
import sys
import time

import numpy as np

from pathlib import Path

WT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, WT)

import cv2
from types import SimpleNamespace

from scripts.recon.io_las import load_scan
from scripts.recon.frame import estimate_normals, dominant_axes, axis_align
from scripts.recon.planes import detect_planes
from scripts.recon.structure import _plane_axis_stats, _greedy_chain, extract_columns_beams
from scripts.recon.schema import WallStep
from scripts.recon.regularize import snap_walls, resolve_corners, pair_thickness
from scripts.recon.floorplan2d import build_room_polygons

OUT_DIR = WT + r"\output\koushik_iso"
PPM = 60

U_GAP_SPLIT_M = 2.8       # holes wider than this = genuinely different walls
HOLE_MIN_M = 0.45         # narrower holes are noise, heal silently
LINE_SNAP_REACH_M = 0.7   # how far a dangling endpoint may extend to meet a line
DANGLING_TOL_M = 0.15     # endpoint already this close to another wall = not dangling
EMPTY_FRac = 0.08         # hole counts as empty if its mid-height point density
                          # is below this fraction of the wall's own density


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _chain_by_u_interval(members, u_gap_m):
    members = sorted(members, key=lambda s: s["u_min"])
    groups, cur, cur_max = [], [], None
    for s in members:
        if cur and s["u_min"] - cur_max > u_gap_m:
            groups.append(cur)
            cur, cur_max = [], None
        cur.append(s)
        cur_max = s["u_max"] if cur_max is None else max(cur_max, s["u_max"])
    if cur:
        groups.append(cur)
    return groups


def group_wall_runs_v3(verticals, xyz, merge_offset_m=0.15, max_relief_m=0.35,
                       min_run_length_m=0.5, angle_tol_deg=10.0,
                       full_height_frac=0.4, u_gap_m=U_GAP_SPLIT_M,
                       min_run_inliers=1200):
    """v2 grouping + per-run member interval list (for hole detection)."""
    xyz = np.asarray(xyz, dtype=float)
    ex, ey = np.eye(3)[:, 0], np.eye(3)[:, 1]
    cos_tol = np.cos(np.deg2rad(angle_tol_deg))
    stats = []
    for i, p in enumerate(verticals):
        s = _plane_axis_stats(p, xyz, ex, ey)
        if s["align"] < cos_tol:
            continue
        s["src_index"] = i
        stats.append(s)
    if not stats:
        return [], set()
    max_z_span = max(s["z_max"] - s["z_min"] for s in stats)
    stats = [s for s in stats if (s["z_max"] - s["z_min"]) >= full_height_frac * max_z_span]

    runs, used = [], set()
    for axis_name in ("x", "y"):
        group = [s for s in stats if s["axis"] == axis_name]
        for offset_cluster in _greedy_chain(group, key=lambda s: s["offset"], max_gap=max_relief_m):
            for useg in _chain_by_u_interval(offset_cluster, u_gap_m):
                step_groups = _greedy_chain(useg, key=lambda s: s["offset"], max_gap=merge_offset_m)
                steps = []
                for sg in step_groups:
                    total_n = sum(x["n_inliers"] for x in sg)
                    offset_avg = sum(x["offset"] * x["n_inliers"] for x in sg) / total_n
                    steps.append(WallStep(
                        offset_m=offset_avg,
                        u_min_m=min(x["u_min"] for x in sg),
                        u_max_m=max(x["u_max"] for x in sg),
                        z_min_m=min(x["z_min"] for x in sg),
                        z_max_m=max(x["z_max"] for x in sg),
                    ))
                main_step = max(steps, key=lambda st: st.u_max_m - st.u_min_m)
                run_len = max(x["u_max"] for x in useg) - min(x["u_min"] for x in useg)
                if run_len < min_run_length_m:
                    continue
                if sum(x["n_inliers"] for x in useg) < min_run_inliers:
                    continue
                main_abs = main_step.offset_m
                axis_vec, u_vec = useg[0]["axis_vec"], useg[0]["u_vec"]
                u_lo = min(x["u_min"] for x in useg)
                u_hi = max(x["u_max"] for x in useg)
                p0 = axis_vec * main_abs + u_vec * u_lo
                p1 = axis_vec * main_abs + u_vec * u_hi
                runs.append(dict(
                    direction=axis_name,
                    normal=(float(axis_vec[0]), float(axis_vec[1]), float(axis_vec[2])),
                    offset_m=main_abs,
                    p0=(float(p0[0]), float(p0[1])),
                    p1=(float(p1[0]), float(p1[1])),
                    steps=sorted((WallStep(st.offset_m - main_abs, st.u_min_m, st.u_max_m,
                                           st.z_min_m, st.z_max_m) for st in steps),
                                 key=lambda st: st.offset_m),
                    members=[(float(x["u_min"]), float(x["u_max"])) for x in useg],
                ))
                used.update(s["src_index"] for s in useg)
    return runs, used


def snap_endpoints_to_lines(walls, reach_m=LINE_SNAP_REACH_M, dangling_tol_m=DANGLING_TOL_M):
    """Extend dangling endpoints along their own centerline to the nearest
    crossing wall's line (T-junction closure). 2D, in place on dict copies."""
    def seg_dist(pt, a, b):
        a, b, pt = map(np.asarray, (a, b, pt))
        ab = b - a
        L2 = float(ab @ ab)
        t = 0.0 if L2 == 0 else float(np.clip((pt - a) @ ab / L2, 0.0, 1.0))
        return float(np.linalg.norm(pt - (a + t * ab)))

    out = [dict(w) for w in walls]
    moved = 0
    for i, w in enumerate(out):
        p0, p1 = np.asarray(w["p0"], float), np.asarray(w["p1"], float)
        d = p1 - p0
        L = np.linalg.norm(d)
        if L == 0:
            continue
        d = d / L
        for key, e in (("p0", p0), ("p1", p1)):
            near = min(seg_dist(e, o["p0"], o["p1"]) for j, o in enumerate(out) if j != i)
            if near <= dangling_tol_m:
                continue  # already touching something
            best = None
            for j, o in enumerate(out):
                if j == i:
                    continue
                a = np.asarray(o["p0"], float)
                d2 = np.asarray(o["p1"], float) - a
                L2 = np.linalg.norm(d2)
                if L2 == 0:
                    continue
                d2 = d2 / L2
                denom = d[0] * d2[1] - d[1] * d2[0]
                if abs(denom) < 1e-6:
                    continue  # parallel
                r = a - p0
                t = (r[0] * d2[1] - r[1] * d2[0]) / denom   # along this wall
                s = (r[0] * d[1] - r[1] * d[0]) / (-denom)  # along the other wall
                X = p0 + t * d
                if np.linalg.norm(X - e) > reach_m:
                    continue
                if s < -0.3 or s > L2 + 0.3:
                    continue  # crossing point is far outside the other wall
                dist = float(np.linalg.norm(X - e))
                if best is None or dist < best[0]:
                    best = (dist, X)
            if best is not None:
                w[key] = (float(best[1][0]), float(best[1][1]))
                moved += 1
        # keep p0/p1 fresh for the second endpoint's math
        p0, p1 = np.asarray(w["p0"], float), np.asarray(w["p1"], float)
    log(f"endpoint-to-line snap: moved {moved} endpoints")
    return out


def recenter_to_midline(walls, xyz, z_floor, z_ceiling):
    """Shift each measured wall's centerline from its detected FACE to the
    midline between its two faces. Without this, the whole wall body gets
    attributed to one adjacent room ('rooms sharing wall internals'): the
    run's offset is the main face's offset, and pair_thickness only sets
    thickness_m -- it never moves p0/p1.

    Back-face side: prefer an opposite step at |offset| ~ thickness; fall
    back to a perpendicular point-density peak at +-thickness."""
    xy = xyz[:, :2]
    z = xyz[:, 2]
    mid = (z > z_floor + 0.35) & (z < z_ceiling - 0.35)
    out, moved = [], 0
    for w in walls:
        w = dict(w)
        t = float(w.get("thickness_m", 0.1))
        if w.get("thickness_source") != "measured" or t < 0.04:
            out.append(w)
            continue
        sign = None
        for st in w.get("steps", []):
            if abs(abs(st.offset_m) - t) < 0.06 and abs(st.offset_m) > 0.03:
                sign = 1.0 if st.offset_m > 0 else -1.0
                break
        axis_i = 0 if w["direction"] == "x" else 1
        u_i = 1 - axis_i
        if sign is None:
            p0, p1 = np.asarray(w["p0"], float), np.asarray(w["p1"], float)
            u_lo, u_hi = sorted([p0[u_i], p1[u_i]])
            m = mid & (xy[:, u_i] > u_lo) & (xy[:, u_i] < u_hi)
            dperp = xy[m, axis_i] - w["offset_m"]
            n_pos = int(np.count_nonzero(np.abs(dperp - t) < 0.07))
            n_neg = int(np.count_nonzero(np.abs(dperp + t) < 0.07))
            if max(n_pos, n_neg) >= 50:
                sign = 1.0 if n_pos >= n_neg else -1.0
        if sign is None:
            out.append(w)
            continue
        shift = sign * t / 2.0
        delta = np.zeros(2)
        delta[axis_i] = shift
        p0, p1 = np.asarray(w["p0"], float), np.asarray(w["p1"], float)
        w["p0"] = (float(p0[0] + delta[0]), float(p0[1] + delta[1]))
        w["p1"] = (float(p1[0] + delta[0]), float(p1[1] + delta[1]))
        w["offset_m"] = float(w["offset_m"] + shift)
        moved += 1
        out.append(w)
    log(f"recenter_to_midline: shifted {moved}/{len(walls)} walls to face-pair midline")
    return out


def find_holes_and_heal(walls, xyz, z_floor, z_ceiling):
    """For each run: coverage holes between member u-intervals; classify each
    as OPENING CANDIDATE (mid-height footprint empty) or OCCLUDED (heal)."""
    xy = xyz[:, :2]
    z = xyz[:, 2]
    mid = (z > z_floor + 0.35) & (z < z_ceiling - 0.35)
    results = []
    for w in walls:
        axis_i = 0 if w["direction"] == "x" else 1  # normal axis
        u_i = 1 - axis_i
        off = w["offset_m"]
        half_band = max(w.get("thickness_m", 0.15), 0.15) / 2 + 0.12
        in_band = mid & (np.abs(xy[:, axis_i] - off) <= half_band)
        u_band = xy[in_band, u_i]

        ivals = sorted(w["members"])
        merged = []
        for lo, hi in ivals:
            if merged and lo <= merged[-1][1] + 0.05:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        covered = sum(hi - lo for lo, hi in merged)
        wall_density = max(u_band.size / max(covered, 0.1), 1.0)

        holes = []
        for (lo0, hi0), (lo1, hi1) in zip(merged, merged[1:]):
            gap = lo1 - hi0
            if gap < HOLE_MIN_M:
                continue
            n_in_gap = int(np.count_nonzero((u_band > hi0 + 0.05) & (u_band < lo1 - 0.05)))
            dens = n_in_gap / max(gap - 0.1, 0.05)
            is_open = dens < EMPTY_FRac * wall_density
            holes.append({"u0": round(hi0, 3), "u1": round(lo1, 3),
                          "width_m": round(gap, 2),
                          "density_frac": round(dens / wall_density, 3),
                          "opening_candidate": bool(is_open)})
        results.append(holes)
    return results


def main():
    try:
        import open3d as o3d
        o3d.utility.random.seed(0)
        log("seeded Open3D RNG")
    except Exception as exc:
        log(f"could not seed Open3D RNG ({exc}) -- results stay nondeterministic")

    scan = load_scan(OUT_DIR + r"\isolated.las")
    log(f"loaded {scan.n:,} points")
    rng = np.random.default_rng(0)
    sub = rng.choice(scan.n, size=min(800_000, scan.n), replace=False)
    R = dominant_axes(estimate_normals(scan.xyz[sub]))
    scan = axis_align(scan, R)
    xyz = scan.xyz
    z_floor = float(np.percentile(xyz[:, 2], 1.0))
    z_ceiling = float(np.percentile(xyz[:, 2], 99.0))

    t0 = time.time()
    planes = detect_planes(xyz, z_floor=z_floor, z_ceiling=z_ceiling)
    verticals = [p for p in planes if p.label == "vertical"]
    log(f"detect_planes: {len(planes)} planes ({len(verticals)} vertical) in {time.time()-t0:.0f}s")

    runs, used = group_wall_runs_v3(verticals, xyz)
    cols, beams, used_cb = extract_columns_beams(
        verticals, xyz, z_floor, z_ceiling, return_used_indices=True)
    log(f"v3 grouping: runs={len(runs)} columns={len(cols)} beams={len(beams)} "
        f"unclaimed={len(verticals) - len(used | used_cb)}/{len(verticals)}")

    walls = snap_walls(runs, np.eye(3))
    walls = pair_thickness(walls, xyz)
    walls = recenter_to_midline(walls, xyz, z_floor, z_ceiling)
    walls = resolve_corners(walls)
    walls = snap_endpoints_to_lines(walls)
    holes_per_wall = find_holes_and_heal(walls, xyz, z_floor, z_ceiling)

    wall_ns = [SimpleNamespace(
        p0=tuple(w["p0"]), p1=tuple(w["p1"]),
        thickness_m=float(w.get("thickness_m", 0.1)),
        length_m=float(np.linalg.norm(np.asarray(w["p1"]) - np.asarray(w["p0"]))),
    ) for w in walls]
    rooms = {}
    for eps in (0.05, 0.30):
        polys = build_room_polygons(wall_ns, epsilon_m=eps)
        big = [p for p in polys if p.area >= 1.0]
        rooms[eps] = big
        log(f"polygonize eps={eps}: {len(big)} rooms >=1m2, "
            f"areas={sorted(round(p.area, 1) for p in big)}")

    n_open = sum(1 for hs in holes_per_wall for h in hs if h["opening_candidate"])
    n_heal = sum(1 for hs in holes_per_wall for h in hs if not h["opening_candidate"])
    log(f"holes: {n_open} opening candidates, {n_heal} healed (occlusion)")

    # ---------- render: architectural style ----------
    xy = xyz[:, :2]
    xmin, ymin = np.percentile(xy, 0.2, axis=0) - 0.5
    xmax, ymax = np.percentile(xy, 99.8, axis=0) + 0.5
    W, H = int((xmax - xmin) * PPM), int((ymax - ymin) * PPM)

    def to_px(pt):
        return (int((pt[0] - xmin) * PPM), int((ymax - pt[1]) * PPM))

    cols_i = np.clip(((xy[:, 0] - xmin) * PPM).astype(int), 0, W - 1)
    rows_i = np.clip(((ymax - xy[:, 1]) * PPM).astype(int), 0, H - 1)
    density = np.zeros((H, W))
    np.add.at(density, (rows_i, cols_i), 1.0)
    gray = np.log1p(density)
    gray = (gray / max(gray.max(), 1e-9) * 60).astype(np.uint8)  # dim background
    img = cv2.merge([gray, gray, gray])

    overlay = img.copy()
    for poly in rooms[0.30]:
        pts = np.array([to_px(c) for c in poly.exterior.coords], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (120, 60, 10))
    img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

    def wall_rect(w, u0=None, u1=None, extra=0.0):
        p0, p1 = np.asarray(w["p0"], float), np.asarray(w["p1"], float)
        d = p1 - p0
        L = np.linalg.norm(d)
        d = d / L if L else np.array([1.0, 0.0])
        n = np.array([-d[1], d[0]])
        ht = w.get("thickness_m", 0.1) / 2 + extra
        a = p0 + d * (0.0 if u0 is None else u0)
        b = p0 + d * (L if u1 is None else u1)
        return np.array([to_px(a + n * ht), to_px(b + n * ht),
                         to_px(b - n * ht), to_px(a - n * ht)], dtype=np.int32)

    # walls solid white-ish, holes: orange = opening candidate, gray-blue = healed
    for w, holes in zip(walls, holes_per_wall):
        cv2.fillPoly(img, [wall_rect(w)], (235, 235, 235))
        p0 = np.asarray(w["p0"], float)
        d = np.asarray(w["p1"], float) - p0
        L = np.linalg.norm(d)
        if L == 0:
            continue
        d = d / L
        # holes carry ABSOLUTE world-u coordinates (the extent axis value);
        # map them onto the current (snapped/recentered) centerline
        u_i = 1 if w["direction"] == "x" else 0
        if abs(d[u_i]) < 1e-6:
            continue
        for h in holes:
            t0 = (h["u0"] - p0[u_i]) / d[u_i]
            t1 = (h["u1"] - p0[u_i]) / d[u_i]
            lo, hi = sorted((t0, t1))
            lo, hi = max(lo, 0.0), min(hi, L)
            if hi - lo < 0.05:
                continue
            color = (0, 150, 255) if h["opening_candidate"] else (160, 130, 110)
            cv2.fillPoly(img, [wall_rect(w, u0=lo, u1=hi, extra=0.02)], color)
    for w in walls:
        cv2.polylines(img, [wall_rect(w)], True, (90, 90, 90), 1)
    for poly in rooms[0.30]:
        c = poly.centroid
        cv2.putText(img, f"{poly.area:.1f}m2", to_px((c.x, c.y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 210, 150), 1, cv2.LINE_AA)
    for c in cols:
        pts = np.array([to_px(p) for p in c.footprint], dtype=np.int32)
        cv2.fillPoly(img, [pts], (0, 165, 255))

    banner = (f"V3: solid walls + T-closure + hole healing | {len(walls)} walls | "
              f"holes: {n_open} opening-candidates (orange) / {n_heal} healed | "
              f"rooms: {len(rooms[0.30])} (eps 0.30) / {len(rooms[0.05])} (eps 0.05)")
    cv2.putText(img, banner, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    out_png = OUT_DIR + r"\floorplan2d_diag_v3.png"
    cv2.imwrite(out_png, img)
    log(f"wrote {out_png}")

    summary = {
        "walls": len(walls),
        "rooms_eps030": sorted(round(p.area, 2) for p in rooms[0.30]),
        "rooms_eps005": sorted(round(p.area, 2) for p in rooms[0.05]),
        "opening_candidates": n_open, "healed_holes": n_heal,
        "per_wall": [
            {"direction": w["direction"],
             "p0": [round(float(v), 3) for v in w["p0"]],
             "p1": [round(float(v), 3) for v in w["p1"]],
             "thickness_m": round(w.get("thickness_m", 0.1), 3),
             "thickness_source": w.get("thickness_source", "?"),
             "holes": hs}
            for w, hs in zip(walls, holes_per_wall)
        ],
    }
    with open(OUT_DIR + r"\floorplan2d_diag_v3.json", "w") as f:
        json.dump(summary, f, indent=2)
    log("done")


if __name__ == "__main__":
    main()
