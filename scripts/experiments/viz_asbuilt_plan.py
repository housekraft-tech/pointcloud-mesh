"""viz_asbuilt_plan.py
--------------------
A proper as-built floor plan. Same data as viz_deviation_plan.py, drawn as a
drawing rather than a scatter plot.

What changed and why:
  walls are RECTANGLES built from each run's fitted plane (length x thickness),
    not clouds of points -- a point cloud reads as fuzz and hides the wall line
  dimensions go on ROOMS, not on all 69 wall stubs, which overlapped each other
    into noise; a room's width x depth is what anyone actually checks
  the view is rotated onto the building's own wall grid so every dimension and
    label is horizontal or vertical

GEOMETRY IS LIDAR throughout: wall lines, thickness, room extents, ceiling
heights, columns, beams. The drawing supplies only names and opening types.

The building is square to within about a degree on its long walls (measured,
not assumed), so any wall more than OUT_OF_SQUARE off the grid is called out --
that is a real as-built deviation, not a drawing artefact.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\viz_asbuilt_plan.py \\
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
from matplotlib.patches import Rectangle, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.experiments.classify_openings_rgb import parse_named_boxes, frame_of
from scripts.experiments.viz_deviation_plan import parse_groups, wall_grid_angle
from scripts.experiments.walk_path_openings import plane_of

TYPE_COLOR = {"Door": "#00b8d4", "window": "#f9a825", "balcony door": "#00a152"}
MIN_WALL_LEN = 0.55     # m: shorter runs are drawn but not dimensioned
MATCH_MAX = 1.00
FLAG_MM = 150
UNCERTAIN_MM = 600
OUT_OF_SQUARE = 1.5     # deg: called out as an as-built deviation
MAX_SKEW = 6.0          # deg: past this it is a segmentation artefact, not a
                        # skewed wall. Measured basis: the long runs of this
                        # flat sit within +-3 deg of the grid, so a 14-16 deg
                        # "wall" is a bad object, and drawing it as a solid
                        # rectangle invents a diagonal bar that is not there.
THICK_MIN, THICK_MAX = 0.06, 0.35   # m: plausible wall thickness
MIN_ELONG = 3.0         # length/spread below this: not a clean single run
MIN_ON_PLANE = 0.60     # fraction of points that must lie on the fitted plane


def rect_of(P):
    """Wall run -> (corners, centre, dir, length, thickness, trusted).

    `trusted` is the important part. A wall object that is blobby, or that
    holds more than one run, gets a principal direction from the SVD that is
    not the wall's length -- drawing that as a rectangle invents a diagonal
    stick that is not in the building, and measuring its angle produced
    "walls" 14-16 deg out of square. A rectangle is only honest when the run is
    clearly elongated AND its points actually lie on the plane."""
    c, n, dv = plane_of(P)
    rel = P[:, :2] - c
    a = rel @ dv
    perp = rel @ n
    length = float(a.max() - a.min())
    spread = float(np.percentile(perp, 95) - np.percentile(perp, 5))
    t = min(max(spread, THICK_MIN), THICK_MAX)
    mid = c + dv * float((a.max() + a.min()) / 2)
    corners = np.array([mid + dv * length / 2 + n * t / 2,
                        mid + dv * length / 2 - n * t / 2,
                        mid - dv * length / 2 - n * t / 2,
                        mid - dv * length / 2 + n * t / 2])
    elong = length / max(spread, 1e-6)
    on_plane = float((np.abs(perp) < max(spread, THICK_MIN)).mean())
    trusted = (elong >= MIN_ELONG) and (on_plane >= MIN_ON_PLANE) \
        and (spread <= THICK_MAX)
    return corners, mid, dv, length, t, trusted


def dim_line(ax, p0, p1, text, off, fs=7.5, col="#37474f"):
    """A dimension line with ticks, offset perpendicular to the run."""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    d = p1 - p0
    L = np.linalg.norm(d)
    if L < 1e-6:
        return
    u = d / L
    nvec = np.array([-u[1], u[0]]) * off
    q0, q1 = p0 + nvec, p1 + nvec
    ax.annotate("", xy=q1, xytext=q0,
                arrowprops=dict(arrowstyle="<->", color=col, lw=0.9,
                                shrinkA=0, shrinkB=0), zorder=8)
    for a, b in ((p0, q0), (p1, q1)):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=col, lw=0.6, alpha=0.7, zorder=8)
    m = (q0 + q1) / 2
    ang = np.degrees(np.arctan2(u[1], u[0]))
    if ang > 90: ang -= 180
    if ang < -90: ang += 180
    ax.text(m[0], m[1], text, fontsize=fs, ha="center", va="center",
            rotation=ang, rotation_mode="anchor", color=col, zorder=9,
            bbox=dict(fc="white", alpha=0.9, pad=1.0, lw=0))


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

    # ---- rotate the view onto the building's wall grid (display only)
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

    fig, ax = plt.subplots(figsize=(20, 17))
    ax.set_facecolor("white")

    # ---- rooms as tinted fills, grouped from their ceiling plateaus
    room_of = {}
    for r in fused["rooms"]:
        for p in r["ceiling_parts"]:
            room_of[p] = r
    for n, P in ceil.items():
        if n not in cinfo:
            continue
        shade = "#eceff1" if n not in room_of else "#e3f2fd"
        ax.scatter(P[::30, 0], P[::30, 1], s=3.0, marker="s", linewidths=0,
                   color=shade, zorder=0)
    # the rooms themselves, as the measured wall-bounded regions. Alternating
    # tints: one shared colour makes neighbouring rooms read as a single space,
    # which is exactly what this plan exists to disprove.
    TINTS = ["#d6e9fb", "#e6f4ea", "#fdf0d5", "#f3e5f5", "#e0f7fa", "#fce4ec"]
    for i, r in enumerate(fused["rooms"]):
        if r.get("outline"):
            ax.add_patch(Polygon(rot(np.asarray(r["outline"], float))[:, :2],
                                 closed=True, fc=TINTS[i % len(TINTS)],
                                 ec="#b0bec5", lw=0.5, zorder=1))

    # ---- walls as solid rectangles
    squint = []
    n_raw = 0
    for name, P in walls.items():
        if len(P) < 800:
            continue
        corners, mid, dv, L, t, trusted = rect_of(P)
        ang0 = np.degrees(np.arctan2(dv[1], dv[0])) % 90
        skew = ang0 - 90 if ang0 > 45 else ang0
        if abs(skew) > MAX_SKEW:
            trusted = False
        if not trusted:
            # no rectangle invented here -- show the measured points instead
            ax.scatter(P[::25, 0], P[::25, 1], s=1.2, c="#455a64", marker=".",
                       linewidths=0, zorder=4)
            n_raw += 1
            continue
        ax.add_patch(Polygon(corners, closed=True, fc="#263238", ec="#263238",
                             lw=0.3, zorder=4))
        ang = np.degrees(np.arctan2(dv[1], dv[0]))
        res = (ang % 90)
        if res > 45:
            res -= 90
        if L >= 2.0 and abs(res) > OUT_OF_SQUARE:
            squint.append((name, L, res, mid))

    # ---- columns
    for n, P in cols.items():
        c = P[:, :2].mean(0)
        w = max(np.ptp(P[:, 0]), 0.12); h = max(np.ptp(P[:, 1]), 0.12)
        ax.add_patch(Rectangle((c[0] - w / 2, c[1] - h / 2), w, h,
                               fc="#ff6d00", ec="#3e2723", lw=0.8, zorder=6))

    # ---- rooms: label + width x depth dimensions from the LiDAR extents
    for r in fused["rooms"]:
        parts = [p for p in r["ceiling_parts"] if p in cinfo and p in ceil]
        if not parts:
            continue
        # Extents come from the wall-bounded room region. Taking them from the
        # ceiling plateaus put the living room's box over the kitchen, because
        # they share one slab.
        if r.get("outline"):
            Q = rot(np.asarray(r["outline"], float))[:, :2]
            x0, y0 = Q.min(0)
            x1, y1 = Q.max(0)
        else:
            Q = np.vstack([ceil[p] for p in parts])[:, :2]
            x0, y0 = np.percentile(Q, 1, axis=0)
            x1, y1 = np.percentile(Q, 99, axis=0)
        # Clear floor area from the wall-bounded region. Summing the ceiling
        # plateaus here was wrong by construction: a plateau is a height level,
        # and one slab covers the kitchen and the living room together.
        area = r.get("area_m2")
        big = max(parts, key=lambda p: cinfo[p]["area_m2"])
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        # Only a near-rectangular room can honestly carry a width x depth. The
        # living room wraps round the kitchen wall, so its bounding box is far
        # bigger than the room -- quoting that box would overstate it by metres.
        boxed = (x1 - x0) * (y1 - y0)
        rectish = area is not None and boxed > 0 and area / boxed >= 0.85
        label = (f"{r['room'].upper()}\n{area:.1f} m²   "
                 f"h {cinfo[big]['height_mm']:.0f}" if area else
                 f"{r['room'].upper()}\nh {cinfo[big]['height_mm']:.0f}")
        if not rectish and area:
            label += "\n(not rectangular)"
        ax.text(cx, cy + 0.16, label,
                ha="center", va="center", fontsize=9.5, weight="bold",
                color="#0d1b2a", zorder=10,
                bbox=dict(fc="white", alpha=0.9, pad=2.5, lw=0))
        if rectish:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   ec="#90a4ae", lw=0.7, ls=(0, (4, 3)),
                                   zorder=3))
            dim_line(ax, (x0, y0), (x1, y0), f"{(x1-x0)*1000:.0f}", -0.20, 7.0)
            dim_line(ax, (x1, y0), (x1, y1), f"{(y1-y0)*1000:.0f}", -0.20, 7.0)

    # ---- openings: LiDAR position and width, drawing type
    centres = [(n,) + frame_of(b)[:1] + (frame_of(b)[2], frame_of(b)[4],
                                         frame_of(b)[1]) for n, b in boxes]
    centres = []
    for n, b in boxes:
        c, nv, al, up, w, h = frame_of(b)
        centres.append((n, c[:2], al[:2], w, nv[:2]))

    flagged = 0
    for o in fused["openings"]:
        p = np.array(o["model_xy"])
        near = sorted(centres, key=lambda t: np.linalg.norm(t[1] - p))
        if not near:
            continue
        name, c, al, w, nrm = near[0]
        dist = float(np.linalg.norm(c - p))
        col = TYPE_COLOR.get(o["cls"], "#9e9e9e")
        if dist > MATCH_MAX:
            ax.scatter(*p, s=150, marker="P", c=col, ec="#3e2723", lw=0.8, zorder=8)
            ax.text(p[0], p[1] - 0.28, f"{o['cls']} — not in scan", ha="center",
                    fontsize=6.5, color="#b71c1c", zorder=9,
                    bbox=dict(fc="white", alpha=0.9, pad=1.0, lw=0))
            continue
        p0 = c - al * w / 2; p1 = c + al * w / 2
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=col, lw=8,
                solid_capstyle="butt", zorder=7)
        off_mm = dist * 1000
        uncertain = off_mm > UNCERTAIN_MM
        txt = f"{o['cls']} {w*1000:.0f}"
        if off_mm > FLAG_MM and not uncertain:
            flagged += 1
            ax.annotate("", xy=(c[0], c[1]), xytext=(p[0], p[1]),
                        arrowprops=dict(arrowstyle="->", color="#d50000", lw=2.0),
                        zorder=11)
            txt += f"\nREADJUST {off_mm:.0f}"
            fc = "#d50000"
        elif uncertain:
            txt += "\n(pairing uncertain)"
            fc = "#9e9e9e"
        else:
            fc = col
        ax.text(c[0] + nrm[0] * 0.40, c[1] + nrm[1] * 0.40, txt, fontsize=6.8,
                ha="center", va="center", color="white", zorder=12,
                bbox=dict(fc=fc, alpha=0.95, pad=1.4, lw=0))

    # ---- out-of-square walls: a real as-built deviation
    for name, L, res, mid in squint:
        ax.text(mid[0], mid[1], f"{res:+.1f}°", fontsize=7, color="#c62828",
                ha="center", va="center", weight="bold", zorder=12,
                bbox=dict(fc="#fff3e0", alpha=0.95, pad=1.2, lw=0))

    # ---- overall building dimensions, outside the envelope
    allw = np.vstack(list(walls.values()))[:, :2]
    bx0, by0 = np.percentile(allw, 0.5, axis=0)
    bx1, by1 = np.percentile(allw, 99.5, axis=0)
    dim_line(ax, (bx0, by0), (bx1, by0), f"{(bx1-bx0)*1000:.0f}", -0.85, 10)
    dim_line(ax, (bx1, by0), (bx1, by1), f"{(by1-by0)*1000:.0f}", -0.85, 10)

    nfeat = sum(len(w["features"]) for w in feats["walls"])
    handles = [
        Line2D([], [], color="#263238", lw=7, label="wall — LiDAR (length × thickness)"),
        Line2D([], [], marker="s", ls="", c="#ff6d00", ms=11, label=f"column — LiDAR ({len(cols)})"),
        Line2D([], [], color="#00b8d4", lw=7, label="door"),
        Line2D([], [], color="#f9a825", lw=7, label="window"),
        Line2D([], [], color="#00a152", lw=7, label="balcony door"),
        Line2D([], [], color="#d50000", lw=2, marker=">", label=f"readjust {FLAG_MM}–{UNCERTAIN_MM} mm ({flagged})"),
        Line2D([], [], marker="P", ls="", c="#9e9e9e", mec="#3e2723", ms=11, label="tagged in drawing, not in scan"),
        Line2D([], [], marker="s", ls="", c="#fff3e0", mec="#c62828", ms=11, label=f"wall out of square >{OUT_OF_SQUARE}° ({len(squint)})"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.97)
    ax.set_title("AS-BUILT FLOOR PLAN — all geometry measured from LiDAR; "
                 "names and opening types from the architect drawing\n"
                 f"dimensions in mm   |   view squared to the building's wall grid "
                 f"({np.degrees(-theta):+.2f}°)   |   {nfeat} wall relief features",
                 fontsize=12)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout(); fig.savefig(out_png, dpi=140, facecolor="white")
    print(f"wrote {out_png}")
    print(f"  {flagged} openings to readjust, {len(squint)} walls out of square, "
          f"{n_raw} runs left as points (untrusted fit)")


if __name__ == "__main__":
    main(*sys.argv[1:6])
