"""Use CAGE for topology, our metrology for geometry.

CAGE decides which edges close a room and where the corners are, at 62 mm/px.
That is the part we currently hand-tune with u-gap splitting and rescue
thresholds. It cannot place a wall: a density pixel is 62 mm wide.

So: take its room graph, throw away its coordinates, and re-solve every edge
onto a wall face measured from the raw points by recon.metrology (sub-mm,
with a standard error). Corners then fall out as exact intersections of two
measured planes rather than as regressed pixel positions.

Outputs:
  snapped_<run>.json   rooms as metric polygons with per-edge provenance
  snap_report.json     how far each edge moved, and the re-scored accuracy
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
OUT = HERE.parent / "snapped"

from recon.io_las import load_scan                      # noqa: E402
from recon.isolate import select_z_band, isolate_unit   # noqa: E402
from recon.metrology import detect_wall_faces           # noqa: E402

YAW_DEG = 5.229
WALL_BAND = (1.0, 2.2)      # above floor: above furniture, below ceiling
SNAP_TOL_M = 0.30           # ~5 density pixels; beyond this CAGE is not "near" a wall


# --------------------------------------------------------------------------
# frame + faces

def load_frame(las_name="koushikexport.las", max_points=8_000_000):
    scan = load_scan(str(ROOT / las_name), max_points=max_points)
    z_band = select_z_band(scan.xyz[:, 2])
    unit, stats = isolate_unit(scan, np.zeros((0, 3)), z_band)
    xyz = unit.xyz.copy()
    centre = xyz[:, :2].mean(axis=0)
    xyz[:, :2] -= centre
    th = np.deg2rad(-YAW_DEG)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    xyz[:, :2] = xyz[:, :2] @ R.T
    return xyz, z_band, centre, stats


def measure_faces(xyz, z_band):
    """Measured wall-face positions along x and along y."""
    z_floor = z_band[0]
    m = ((xyz[:, 2] >= z_floor + WALL_BAND[0]) &
         (xyz[:, 2] <= z_floor + WALL_BAND[1]))
    band = xyz[m]
    faces = {}
    for axis, idx in (("x", 0), ("y", 1)):
        # min_bin_frac rejects walls PARALLEL to this axis, which are
        # full-height too and would otherwise register as phantom faces.
        f = detect_wall_faces(band[:, idx], band[:, 2],
                              bin_m=0.05, min_points=200, min_span_m=0.9,
                              merge_tol=0.02, min_bin_frac=0.02)
        faces[axis] = f
        print(f"  {axis}: {len(f)} measured faces, "
              f"median stderr {1000*np.median([x.stderr for x in f]):.3f} mm"
              if f else f"  {axis}: none", flush=True)
    return faces


# --------------------------------------------------------------------------
# polygon regularisation

def axis_of(p, q):
    """'x' if the edge holds x constant (runs along y), else 'y'."""
    return "x" if abs(q[0] - p[0]) < abs(q[1] - p[1]) else "y"


def regularise(poly):
    """Force every edge onto an axis, dropping degenerate ones.

    CAGE's corners are already near-axis-aligned; this makes the assumption
    explicit so each edge has a single coordinate to re-measure.
    """
    pts = [np.asarray(p, dtype=float) for p in poly]
    n = len(pts)
    edges = []
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        if np.hypot(*(b - a)) < 0.05:
            continue
        ax = axis_of(a, b)
        k = 0 if ax == "x" else 1
        edges.append({"axis": ax, "coord": 0.5 * (a[k] + b[k]),
                      "lo": min(a[1 - k], b[1 - k]),
                      "hi": max(a[1 - k], b[1 - k])})
    # collapse consecutive same-axis edges (a staircase that should be one line)
    merged = []
    for e in edges:
        if merged and merged[-1]["axis"] == e["axis"] and \
                abs(merged[-1]["coord"] - e["coord"]) < 0.06:
            m = merged[-1]
            m["lo"] = min(m["lo"], e["lo"])
            m["hi"] = max(m["hi"], e["hi"])
        else:
            merged.append(dict(e))
    return merged


def structural(cand, frac=0.15):
    """Keep only well-supported faces.

    detect_wall_faces returns every full-height surface it can measure, which
    on a lived-in scan includes wardrobe fronts and curtain lines. Those are
    real surfaces, but they are not the room boundary, and a nearest-face snap
    will happily land on one. Support (point count) separates them: a wall
    carries an order of magnitude more returns than a piece of furniture.
    """
    if not cand:
        return cand
    thresh = frac * max(f.n for f in cand)
    return [f for f in cand if f.n >= thresh] or cand


def snap_edges(edges, faces, centroid, mode, tol=SNAP_TOL_M):
    """Re-solve each edge's constant coordinate onto a measured wall face.

    mode 'nearest'   -- the closest measured face, whatever it is
    mode 'interior'  -- nearest face, preferring the room's own side
    mode 'strict'    -- structural faces only, and only on the interior side.
                        A wall has two faces ~200 mm apart (the pipeline's
                        measured thickness modes are 194 and 253 mm); picking
                        the far one puts the room boundary a whole wall out,
                        which is where the residual error was coming from.
    """
    out = []
    for e in edges:
        cand = faces[e["axis"]]
        if mode == "strict":
            cand = structural(cand)
        k = 0 if e["axis"] == "x" else 1
        c = e["coord"]
        best, moved, why = None, None, "none"
        if cand:
            d = np.array([abs(f.value - c) for f in cand])
            order = np.argsort(d)
            inward = np.sign(centroid[k] - c)
            pick = None
            if mode == "nearest":
                pick = order[0] if d[order[0]] <= tol else None
                why = "nearest"
            elif mode == "interior":
                for i in order:
                    if d[i] > tol:
                        break
                    if np.sign(cand[i].value - c) == inward or d[i] < 0.02:
                        pick, why = i, "interior"
                        break
                if pick is None and d[order[0]] <= tol:
                    pick, why = order[0], "fallback-nearest"
            else:  # strict / wall
                # Identify the WALL, then take its room-side face. CAGE's edge
                # can sit either side of the true boundary, so "which side of
                # the edge" is not the question -- "which face of this wall
                # faces the room" is. A wall is a face pair separated by a
                # plausible thickness (the scan's measured modes are 194 and
                # 253 mm; allow 80-320 mm).
                if d[order[0]] <= tol:
                    i0 = order[0]
                    v0 = cand[i0].value
                    partner = None
                    for j in range(len(cand)):
                        if j == i0:
                            continue
                        t = abs(cand[j].value - v0)
                        if 0.08 <= t <= 0.32:
                            if partner is None or t < abs(cand[partner].value - v0):
                                partner = j
                    if partner is None:
                        pick, why = i0, "single-face"
                    else:
                        # the face of the pair closer to this room's centroid
                        pair = [i0, partner]
                        pick = min(pair, key=lambda j: abs(cand[j].value - centroid[k]))
                        why = "wall-inner-face"
                else:
                    why = "kept"
            if pick is not None:
                best = cand[pick]
                moved = float(best.value - c)
        out.append({**e,
                    "snapped": best.value if best is not None else e["coord"],
                    "moved_mm": None if moved is None else round(1000 * moved, 2),
                    "stderr_mm": None if best is None else round(1000 * best.stderr, 4),
                    "n_points": None if best is None else int(best.n),
                    "how": why,
                    "matched": best is not None})
    return out


def rebuild(edges):
    """Corners = intersections of consecutive perpendicular measured lines."""
    n = len(edges)
    if n < 4:
        return None
    pts = []
    for i in range(n):
        a, b = edges[i], edges[(i + 1) % n]
        if a["axis"] == b["axis"]:
            continue
        x = a["snapped"] if a["axis"] == "x" else b["snapped"]
        y = a["snapped"] if a["axis"] == "y" else b["snapped"]
        pts.append((x, y))
    return pts if len(pts) >= 4 else None


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="robust_c95_swin")
    ap.add_argument("--mode", default="interior", choices=["interior", "nearest", "strict"])
    ap.add_argument("--max-points", type=int, default=8_000_000)
    a = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    res = json.loads((HERE.parent / "polygons" / f"{a.run}.json").read_text())
    var = json.loads((HERE.parent / "scores" / "variants.json").read_text())
    key = Path(res["density"]).stem.replace("density_", "")
    v = var["variants"][key]
    mn, mx = np.array(v["min_coords"]), np.array(v["max_coords"])
    sx, sy = (mx[0] - mn[0]) / 256.0, (mx[1] - mn[1]) / 256.0
    print(f"run {a.run}: {len(res['polys_refined'])} rooms, "
          f"{1000*sx:.1f} x {1000*sy:.1f} mm/px", flush=True)

    print("loading scan and measuring wall faces...", flush=True)
    xyz, z_band, centre, stats = load_frame(max_points=a.max_points)
    faces = measure_faces(xyz, z_band)

    rooms, all_moves = [], []
    for i, p in enumerate(res["polys_refined"]):
        arr = np.array(p, dtype=float)
        xy = np.stack([mn[0] + arr[:, 0] * sx, mn[1] + arr[:, 1] * sy], axis=1)
        cen = xy.mean(axis=0)
        edges = regularise(xy)
        snapped = snap_edges(edges, faces, cen, a.mode)
        pts = rebuild(snapped)
        if pts is None:
            print(f"  room {i}: degenerate after regularisation, skipped")
            continue
        moves = [e["moved_mm"] for e in snapped if e["moved_mm"] is not None]
        all_moves += moves
        rooms.append({
            "id": i,
            "polygon_m": [[round(x, 5), round(y, 5)] for x, y in pts],
            "n_edges": len(snapped),
            "n_snapped": sum(1 for e in snapped if e["matched"]),
            "edges": [{k: e[k] for k in
                       ("axis", "lo", "hi", "coord", "snapped", "moved_mm", "stderr_mm", "n_points", "how", "matched")}
                      for e in snapped],
        })

    mv = np.abs(np.array(all_moves)) if all_moves else np.array([0.0])
    report = {
        "run": a.run, "mode": a.mode,
        "mm_per_px": [round(1000 * sx, 2), round(1000 * sy, 2)],
        "n_rooms": len(rooms),
        "n_faces_x": len(faces["x"]), "n_faces_y": len(faces["y"]),
        "median_face_stderr_um": round(1e6 * float(np.median(
            [f.stderr for f in faces["x"] + faces["y"]])), 1),
        "edges_total": sum(r["n_edges"] for r in rooms),
        "edges_snapped": sum(r["n_snapped"] for r in rooms),
        "edge_move_mm": {
            "median": round(float(np.median(mv)), 1),
            "p90": round(float(np.percentile(mv, 90)), 1),
            "max": round(float(mv.max()), 1),
        },
    }
    (OUT / f"snapped_{a.run}_{a.mode}.json").write_text(
        json.dumps({"report": report, "rooms": rooms}, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
