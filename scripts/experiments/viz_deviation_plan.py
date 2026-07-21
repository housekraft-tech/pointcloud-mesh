"""viz_deviation_plan.py
---------------------
The as-built plan, drawn the way the sources should be trusted.

GEOMETRY IS LIDAR. Wall lines, wall thickness, room extents, ceiling heights,
columns, beams, grooves and niches all come from the scan. The scan is the
site: where it says a wall is, that is where the wall is.

THE DRAWING ONLY TAGS. It supplies room names and the type of each opening
(door / window / balcony door) -- things a point cloud cannot know. It never
supplies a position or a dimension here.

The deviation is then the difference between where the drawing PUT an opening
and where the scan FOUND it, drawn as an arrow from design to as-built with the
offset in millimetres. That is the readjustment.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\viz_deviation_plan.py \\
      <annotated_model.obj> <fused_detections.json> <modular_manifest.json> \\
      <features.json> <out_png>
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.experiments.classify_openings_rgb import parse_named_boxes, frame_of

TYPE_COLOR = {
    "Door": "#00e5ff", "window": "#ffd400", "balcony door": "#00ff90",
}
MATCH_MAX = 1.00       # m: beyond this a drawing tag matches no scan opening
FLAG_MM = 150          # offsets above this are a readjustment to report
MIN_DIM_LEN = 0.90     # m: shorter wall runs are not worth dimensioning
DIM_OFFSET = 0.16      # m: dimension line offset from the wall face
UNCERTAIN_MM = 600     # above this the "offset" is more likely a wrong pairing
                       # than a built error -- a door is not 1.8 m out of place


def parse_groups(path, prefix):
    V = []; g = []
    for ln in open(path):
        if ln.startswith("o "):
            g.append([ln[2:].strip(), len(V), len(V)])
        elif ln.startswith("v "):
            _, x, y, z = ln.split()[:4]
            V.append((float(x), float(y), float(z)))
            if g:
                g[-1][2] = len(V)
    V = np.asarray(V)
    return {n: V[a:b] for n, a, b in g if n.startswith(prefix) and b > a}


def wall_grid_angle(walls):
    """The building's own orthogonal grid direction, from the wall runs.

    Averaged on 4*angle so that walls 90 deg apart reinforce rather than cancel
    -- a plain mean of headings on an orthogonal plan returns nothing useful.
    Weighted by run length so the long structural walls set the grid and short
    stubs do not."""
    from scripts.experiments.walk_path_openings import plane_of
    vx = vy = 0.0
    for P in walls.values():
        if len(P) < 2000:
            continue
        c, n, dv = plane_of(P)
        a = (P[:, :2] - c) @ dv
        L = float(a.max() - a.min())
        if L < 0.9:
            continue
        ang = np.arctan2(dv[1], dv[0])
        vx += L * np.cos(4 * ang)
        vy += L * np.sin(4 * ang)
    if vx == 0 and vy == 0:
        return 0.0
    theta = np.arctan2(vy, vx) / 4.0
    # bring into (-45, 45] deg: any multiple of 90 is the same grid
    while theta > np.pi / 4:
        theta -= np.pi / 2
    while theta <= -np.pi / 4:
        theta += np.pi / 2
    return float(theta)


def main(obj_path, fused_path, manifest_path, features_path, out_png):
    fused = json.load(open(fused_path))
    man = json.load(open(manifest_path))
    feats = json.load(open(features_path))

    walls = parse_groups(obj_path, "wall")
    ceil = parse_groups(obj_path, "ceiling")
    cols = parse_groups(obj_path, "column")
    boxes = parse_named_boxes(obj_path, ("door", "balcony_door", "window",
                                         "archway", "opening"))
    cinfo = {o["name"]: o for o in man["objects"] if o["name"].startswith("ceiling")}

    # ---- straighten the DISPLAY to the building's own wall grid.
    # The flat sits ~5 deg off the scan axes, which tilts every dimension line
    # and label and makes the drawing hard to read against. This rotates the
    # view only -- no measurement changes, because length, thickness and offset
    # are all computed from the geometry, and rotation preserves distance.
    theta = wall_grid_angle(walls)
    R = np.array([[np.cos(-theta), -np.sin(-theta)],
                  [np.sin(-theta), np.cos(-theta)]])
    pivot = np.vstack(list(walls.values()))[:, :2].mean(0)

    def rot(P):
        P = np.asarray(P, float)
        flat = P.ndim == 1
        Q = np.atleast_2d(P).copy()
        Q[:, :2] = (Q[:, :2] - pivot) @ R.T + pivot
        return Q[0] if flat else Q

    for dd in (walls, ceil, cols):
        for k in list(dd):
            dd[k] = rot(dd[k])
    boxes = [(n, rot(b)) for n, b in boxes]
    for r in fused["rooms"]:
        r["centre_xy"] = list(rot(np.array(r["centre_xy"]))[:2])
    for o in fused["openings"]:
        o["model_xy"] = list(rot(np.array(o["model_xy"]))[:2])
    print(f"straightened display by {np.degrees(-theta):+.2f} deg "
          "(wall grid -> axes); measurements unchanged")

    fig, ax = plt.subplots(figsize=(19, 17))

    # ---- ceiling plateaus (LiDAR) tinted by height
    hs = [cinfo[n]["height_mm"] for n in ceil if n in cinfo]
    lo, hi = (min(hs), max(hs)) if hs else (2100, 2750)
    for n, P in ceil.items():
        if n not in cinfo:
            continue
        t = (cinfo[n]["height_mm"] - lo) / max(hi - lo, 1)
        ax.scatter(P[::40, 0], P[::40, 1], s=2.0, marker=".", linewidths=0,
                   color=plt.cm.YlGnBu(0.25 + 0.55 * (1 - t)), zorder=0)

    # ---- walls (LiDAR) -- the geometry of record
    for P in walls.values():
        s = P[::35]
        ax.scatter(s[:, 0], s[:, 1], s=1.0, c="#1c2226", marker=".",
                   linewidths=0, zorder=2)

    # ---- wall dimensions, measured off the merged planes.
    # Length is the extent of actual geometry on the plane. Thickness is the
    # 5-95 percentile perpendicular spread, which is robust to the stray
    # returns that make a min-max thickness meaningless.
    # Dimension the wall RUNS as built, not merged planes. Merging puts every
    # run that shares an infinite plane into one object, so a "wall" came out
    # 12048 mm long with its dimension line cutting diagonally across the flat.
    # A wall run is a thing you can stand in front of and measure.
    from scripts.experiments.walk_path_openings import plane_of
    dims = []
    for name, P in walls.items():
        if len(P) < 2000:
            continue
        c, n, dv = plane_of(P)
        rel = P[:, :2] - c
        a = rel @ dv
        perp = rel @ n
        length = float(a.max() - a.min())
        if length < MIN_DIM_LEN:
            continue
        thick = float(np.percentile(perp, 95) - np.percentile(perp, 5))
        mid = c + dv * float((a.max() + a.min()) / 2)
        p0 = c + dv * float(a.min()); p1 = c + dv * float(a.max())
        off = n * DIM_OFFSET
        ax.plot([p0[0] + off[0], p1[0] + off[0]],
                [p0[1] + off[1], p1[1] + off[1]],
                color="#455a64", lw=0.9, alpha=0.85, zorder=3)
        for e in (p0, p1):
            ax.plot([e[0], e[0] + off[0]], [e[1], e[1] + off[1]],
                    color="#455a64", lw=0.7, alpha=0.7, zorder=3)
        ang = np.degrees(np.arctan2(dv[1], dv[0]))
        if ang > 90: ang -= 180
        if ang < -90: ang += 180
        ax.text(mid[0] + off[0], mid[1] + off[1],
                f"{length*1000:.0f} × {thick*1000:.0f}t",
                fontsize=6.0, ha="center", va="center", rotation=ang,
                rotation_mode="anchor", color="#263238", zorder=4,
                bbox=dict(fc="white", alpha=0.72, pad=0.8, lw=0))
        dims.append((name, length, thick))

    # ---- columns (LiDAR)
    for n, P in cols.items():
        c = P[:, :2].mean(0)
        w = np.ptp(P[:, 0]); h = np.ptp(P[:, 1])
        ax.add_patch(Rectangle((c[0] - w / 2, c[1] - h / 2), w, h, fill=True,
                               fc="#ff6d00", ec="k", lw=0.6, alpha=0.95, zorder=6))

    # ---- beams / dropped ceilings (LiDAR): plateaus below the main level
    main_h = max(hs) if hs else 2750
    for n, P in ceil.items():
        if n not in cinfo:
            continue
        drop = main_h - cinfo[n]["height_mm"]
        if drop < 120:
            continue
        ax.scatter(P[::40, 0], P[::40, 1], s=2.2, marker=".", linewidths=0,
                   color="#8e24aa", alpha=0.5, zorder=1)

    # ---- room tags (names from the drawing, geometry+height from the LiDAR)
    for r in fused["rooms"]:
        parts = [p for p in r["ceiling_parts"] if p in cinfo]
        if not parts:
            continue
        big = max(parts, key=lambda p: cinfo[p]["area_m2"])
        P = ceil[big]
        c = P[:, :2].mean(0)
        area = sum(cinfo[p]["area_m2"] for p in parts)
        ax.text(c[0], c[1], f"{r['room']}\n{area:.1f} m²  h {cinfo[big]['height_mm']:.0f}",
                ha="center", va="center", fontsize=9, weight="bold", color="#0d1b2a",
                bbox=dict(fc="white", alpha=0.82, pad=2.4, lw=0), zorder=9)

    # ---- openings: position + width from the LiDAR, TYPE from the drawing
    centres = []
    for name, box in boxes:
        c, n, along, up, w, h = frame_of(box)
        centres.append((name, c[:2], along[:2], w, n[:2]))

    flagged = 0
    for o in fused["openings"]:
        p = np.array(o["model_xy"])
        near = sorted(centres, key=lambda t: np.linalg.norm(t[1] - p))
        if not near:
            continue
        name, c, al, w, nrm = near[0]
        dist = float(np.linalg.norm(c - p))
        col = TYPE_COLOR.get(o["cls"], "#bbbbbb")
        if dist > MATCH_MAX:
            # the drawing tags an opening the scan never found here
            ax.scatter(*p, s=170, marker="P", c=col, ec="k", lw=0.8, zorder=8)
            ax.text(p[0], p[1] - 0.30, f"{o['cls']}\nnot found in scan",
                    ha="center", fontsize=6.5, color="#b71c1c",
                    bbox=dict(fc="white", alpha=0.85, pad=1.2, lw=0), zorder=9)
            continue
        # as-built bar, drawn at the LiDAR position and width
        p0 = c - al * w / 2; p1 = c + al * w / 2
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=col, lw=7,
                solid_capstyle="butt", zorder=7)
        off_mm = dist * 1000
        # A door is not built 1.8 m out of place. Past UNCERTAIN_MM the number
        # is far more likely to be a wrong pairing than a construction error,
        # so it is shown grey and labelled as uncertain rather than reported as
        # a readjustment someone might act on.
        uncertain = off_mm > UNCERTAIN_MM
        if off_mm > FLAG_MM:
            acol = "#9e9e9e" if uncertain else "#d50000"
            if not uncertain:
                flagged += 1
            ax.annotate("", xy=(c[0], c[1]), xytext=(p[0], p[1]),
                        arrowprops=dict(arrowstyle="->", color=acol, lw=2.0),
                        zorder=10)
            ax.scatter(*p, s=45, marker="o", facecolors="none",
                       edgecolors=acol, lw=1.6, zorder=10)
        lbl = f"{o['cls']}\n{w*1000:.0f} mm"
        if uncertain:
            lbl += "\npairing uncertain"
        elif off_mm > FLAG_MM:
            lbl += f"\nreadjust {off_mm:.0f} mm"
        ax.text(c[0] + nrm[0] * 0.42, c[1] + nrm[1] * 0.42, lbl, fontsize=6.6,
                ha="center", va="center", color="white", zorder=11,
                bbox=dict(fc=(col if off_mm <= FLAG_MM
                              else ("#9e9e9e" if uncertain else "#d50000")),
                          alpha=0.9, pad=1.4, lw=0))

    nfeat = sum(len(w["features"]) for w in feats["walls"])
    kinds = {}
    for wl in feats["walls"]:
        for f in wl["features"]:
            kinds[f["type"]] = kinds.get(f["type"], 0) + 1
    sub = ", ".join(f"{k} {v}" for k, v in sorted(kinds.items(), key=lambda t: -t[1])[:5])

    handles = [
        Line2D([], [], color="#1c2226", lw=6, label="wall (LiDAR — geometry of record)"),
        Line2D([], [], marker="s", ls="", c="#ff6d00", ms=11, label=f"column (LiDAR, {len(cols)})"),
        Line2D([], [], marker=".", ls="", c="#8e24aa", ms=16, label="beam soffit / dropped ceiling (LiDAR)"),
        Line2D([], [], color="#00e5ff", lw=6, label="door (type from drawing)"),
        Line2D([], [], color="#ffd400", lw=6, label="window (type from drawing)"),
        Line2D([], [], color="#00ff90", lw=6, label="balcony door (type from drawing)"),
        Line2D([], [], color="#d50000", lw=2, marker=">", label=f"design → as-built offset >{FLAG_MM} mm ({flagged})"),
        Line2D([], [], marker="P", ls="", c="#bbb", mec="k", ms=11, label="tagged in drawing, not found in scan"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.96)
    ax.set_title("As-built plan — geometry and position from LiDAR, tags from the drawing\n"
                 f"arrows show where an opening sits vs where the drawing put it   |   "
                 f"{len(dims)} walls dimensioned  |  wall relief: {nfeat} features ({sub})",
                 fontsize=11)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(out_png, dpi=130)
    print(f"wrote {out_png}  ({flagged} openings offset >{FLAG_MM} mm)")


if __name__ == "__main__":
    main(*sys.argv[1:6])
