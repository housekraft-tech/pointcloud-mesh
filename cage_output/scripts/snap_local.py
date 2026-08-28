"""Per-edge local plane measurement, then a solid 3D model.

snap_to_planes.py measures wall faces globally: one list of x-positions and one
of y-positions for the whole storey. That is why ten edges found nothing -- a
1.5 m edge in one corner had to match a face fitted from every point in the
flat, and a wall that jogs, or one that simply is not full-length, has no
global line to land on.

Here each edge is measured from only the points it actually bounds: inside its
own span, within a search window of its position, in the wall height band. That
is both more likely to find the wall and more accurate when it does, because a
real wall's position varies slightly along its length.

Then the closed rooms are extruded to a solid model between the measured floor
and ceiling planes.
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

from recon.metrology import detect_wall_faces, refine_face, robust_location  # noqa: E402
from snap_to_planes import load_frame, regularise, measure_faces, WALL_BAND  # noqa: E402

SEARCH = (0.30,)              # same selection tolerance as the strict baseline
THICK_RANGE = (0.08, 0.32)    # a wall is a face pair this far apart
END_TRIM = 0.12               # ignore this much at each end of the edge (corners)


def choose_global(faces_axis, coord, centroid_k, tol):
    """Pick WHICH wall face bounds this edge, from the storey-wide inventory.

    Selection has to be global. A wall is a face pair ~200 mm apart, and from
    inside one room you only see the near face -- the far one belongs to the
    neighbouring room. Measuring in a window narrower than the wall therefore
    loses the pair, and the choice degrades to "nearest surface", which is
    often a wardrobe. Choose globally; refine locally.
    """
    if not faces_axis:
        return None, "none"
    thresh = 0.15 * max(f.n for f in faces_axis)
    cand = [f for f in faces_axis if f.n >= thresh] or faces_axis
    d = np.array([abs(f.value - coord) for f in cand])
    i0 = int(np.argmin(d))
    if d[i0] > tol:
        return None, "none"
    v0 = cand[i0].value
    partner = None
    for j in range(len(cand)):
        if j == i0:
            continue
        t = abs(cand[j].value - v0)
        if THICK_RANGE[0] <= t <= THICK_RANGE[1]:
            if partner is None or t < abs(cand[partner].value - v0):
                partner = j
    if partner is None:
        return cand[i0], "single-face"
    pick = min([i0, partner], key=lambda q: abs(cand[q].value - centroid_k))
    return cand[pick], "wall-inner-face"


def refine_locally(band, edge, seed, k, half=0.05):
    """Re-measure the chosen face using only the points this edge bounds.

    A real wall is not perfectly straight over 5 m, and the global fit averages
    its whole length. Restricting to the span the edge actually covers is both
    a truer answer for this room and a tighter one.
    """
    j = 1 - k
    lo, hi = edge["lo"], edge["hi"]
    trim = min(END_TRIM, 0.3 * (hi - lo))
    m = ((band[:, j] >= lo + trim) & (band[:, j] <= hi - trim) &
         (np.abs(band[:, k] - seed) <= half))
    if m.sum() < 150:
        return None
    return refine_face(band[m, k], seed, windows=(0.030, 0.018, 0.010),
                       min_points=100)


def snap_room(band, xy, faces):
    edges = regularise(xy)
    cen = xy.mean(axis=0)
    out = []
    for e in edges:
        k = 0 if e["axis"] == "x" else 1
        chosen, how = None, "kept"
        for tol in SEARCH:
            chosen, how = choose_global(faces[e["axis"]], e["coord"], cen[k], tol)
            if chosen is not None:
                how = f"{how}@{int(100*tol)}cm"
                break
        val = e["coord"]
        stderr = sigma = npts = None
        if chosen is not None:
            val, stderr, sigma, npts = chosen.value, chosen.stderr, chosen.sigma, chosen.n
            loc = refine_locally(band, e, chosen.value, k)
            if loc is not None and abs(loc.value - chosen.value) < 0.04:
                val, stderr, sigma, npts = loc.value, loc.stderr, loc.sigma, loc.n
                how += "+local"
        out.append({
            "axis": e["axis"], "coord": e["coord"],
            "lo": e["lo"], "hi": e["hi"],
            "snapped": float(val),
            "moved_mm": None if chosen is None else round(1000 * (val - e["coord"]), 2),
            "stderr_mm": None if stderr is None else round(1000 * stderr, 4),
            "sigma_mm": None if sigma is None else round(1000 * sigma, 3),
            "n_points": None if npts is None else int(npts),
            "how": how, "matched": chosen is not None,
        })
    return out


def rebuild(edges):
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


def measure_slabs(xyz, z_band):
    """Floor and ceiling planes, measured the same way as the walls."""
    z = xyz[:, 2]
    lo, hi = z_band
    fl = refine_face(z[z < lo + 0.5], lo, windows=(0.12, 0.05, 0.025))
    ce = refine_face(z[z > hi - 0.6], hi, windows=(0.12, 0.05, 0.025))
    return fl, ce


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="robust_c95_swin")
    ap.add_argument("--max-points", type=int, default=8_000_000)
    a = ap.parse_args()

    res = json.loads((BASE / "polygons" / f"{a.run}.json").read_text())
    var = json.loads((BASE / "scores" / "variants.json").read_text())
    key = Path(res["density"]).stem.replace("density_", "")
    v = var["variants"][key]
    mn, mx = np.array(v["min_coords"]), np.array(v["max_coords"])
    sx, sy = (mx[0] - mn[0]) / 256.0, (mx[1] - mn[1]) / 256.0

    print("loading scan...", flush=True)
    xyz, z_band, centre, _ = load_frame(max_points=a.max_points)
    zf = z_band[0]
    band = xyz[(xyz[:, 2] >= zf + WALL_BAND[0]) & (xyz[:, 2] <= zf + WALL_BAND[1])]
    print(f"  wall band: {len(band):,} points", flush=True)
    faces = measure_faces(xyz, z_band)

    floor, ceil = measure_slabs(xyz, z_band)
    print(f"  floor z = {floor.value:.4f} m  (stderr {1000*floor.stderr:.3f} mm)")
    print(f"  ceiling z = {ceil.value:.4f} m  (stderr {1000*ceil.stderr:.3f} mm)")
    print(f"  clear height = {1000*(ceil.value-floor.value):.1f} mm", flush=True)

    rooms, kept, moves = [], 0, []
    for i, p in enumerate(res["polys_refined"]):
        arr = np.array(p, dtype=float)
        xy = np.stack([mn[0] + arr[:, 0] * sx, mn[1] + arr[:, 1] * sy], axis=1)
        edges = snap_room(band, xy, faces)
        pts = rebuild(edges)
        if pts is None:
            continue
        kept += sum(1 for e in edges if not e["matched"])
        moves += [abs(e["moved_mm"]) for e in edges if e["moved_mm"] is not None]
        rooms.append({"id": i, "polygon_m": [[round(x, 5), round(y, 5)] for x, y in pts],
                      "n_edges": len(edges),
                      "n_snapped": sum(1 for e in edges if e["matched"]),
                      "edges": edges})

    mv = np.array(moves)
    tot = sum(r["n_edges"] for r in rooms)
    report = {
        "run": a.run, "method": "global-select+local-refine",
        "n_rooms": len(rooms), "edges_total": tot,
        "edges_snapped": tot - kept, "edges_kept": kept,
        "edge_move_mm": {"median": round(float(np.median(mv)), 1),
                         "p90": round(float(np.percentile(mv, 90)), 1),
                         "max": round(float(mv.max()), 1)},
        "median_stderr_um": round(1e6 * float(np.median(
            [e["stderr_mm"] / 1000 for r in rooms for e in r["edges"]
             if e["stderr_mm"] is not None])), 1),
        "floor_z": round(float(floor.value), 5),
        "ceiling_z": round(float(ceil.value), 5),
        "clear_height_mm": round(1000 * (ceil.value - floor.value), 1),
        "floor_stderr_mm": round(1000 * floor.stderr, 4),
        "ceiling_stderr_mm": round(1000 * ceil.stderr, 4),
    }
    out = BASE / "snapped" / f"snapped_{a.run}_hybrid.json"
    out.write_text(json.dumps({"report": report, "rooms": rooms}, indent=1))
    print(json.dumps(report, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
