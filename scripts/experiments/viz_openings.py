"""viz_openings.py
----------------
Plan-view of every classified opening: where it is, what role its width implies,
how far off standard it is, and whether it leads outside.

The ceiling footprint is drawn as the shaded field. That is deliberate -- it is
the evidence for the interior/exterior call, so the reader can see for
themselves that the EXTERIOR openings sit on the edge of the roofed area.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\viz_openings.py \\
      <annotated_model.obj> <openings_classified.json> <out_png>
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.experiments.classify_openings_rgb import parse_named_boxes, frame_of

ROLE_COLOR = {
    "main entrance":     "#ff1744",
    "bedroom":           "#2979ff",
    "kitchen / utility": "#00c853",
    "bathroom / WC":     "#ff9100",
}
OTHER = "#9e9e9e"


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
    return [V[a:b] for n, a, b in g if n.startswith(prefix) and b > a]


def main(obj_path, cls_path, out_png):
    cls = {r["name"]: r for r in json.load(open(cls_path))}
    boxes = parse_named_boxes(obj_path, ("door", "balcony_door", "window",
                                         "archway", "opening"))
    ceil = np.vstack(parse_groups(obj_path, "ceiling"))
    walls = parse_groups(obj_path, "wall")

    fig, ax = plt.subplots(figsize=(17, 15))
    # Rasterise the ceiling into a solid field. Drawn as scattered points it is
    # invisible, and this is the EVIDENCE for every interior/exterior call --
    # the reader has to be able to see where the roof stops.
    cell = 0.10
    lo = ceil[:, :2].min(0) - 0.5
    hi = ceil[:, :2].max(0) + 0.5
    nx, ny = (np.ceil((hi - lo) / cell)).astype(int)
    grid = np.zeros((ny, nx))
    ij = np.floor((ceil[:, :2] - lo) / cell).astype(int)
    ij = ij[(ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny)]
    grid[ij[:, 1], ij[:, 0]] = 1
    ax.imshow(np.ma.masked_where(grid == 0, grid), origin="lower",
              extent=[lo[0], lo[0] + nx * cell, lo[1], lo[1] + ny * cell],
              cmap="Blues", vmin=0, vmax=2.2, alpha=0.55, zorder=0,
              interpolation="nearest")
    for W in walls:
        s = W[::50]
        ax.scatter(s[:, 0], s[:, 1], s=0.8, c="#37474f", marker=".",
                   linewidths=0, zorder=1)

    for name, box in boxes:
        r = cls.get(name, {})
        c, n, along, up, w, h = frame_of(box)
        role = r.get("inferred_role")
        col = ROLE_COLOR.get(role, OTHER)
        p0 = c[:2] - along[:2] * w / 2
        p1 = c[:2] + along[:2] * w / 2
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=col, lw=6, solid_capstyle="butt",
                zorder=4)

        leads = r.get("leads", "")
        ext = leads.startswith("EXTERIOR")
        unc = leads.startswith("uncertain")
        if ext:
            ax.scatter(*c[:2], s=320, marker="*", c="#ffd600",
                       edgecolors="#000", lw=0.6, zorder=6)
        elif unc:
            ax.scatter(*c[:2], s=90, marker="D", facecolors="none",
                       edgecolors="#ffd600", lw=1.8, zorder=6)

        dev = r.get("width_dev_mm")
        lbl = f"{r.get('width_mm', round(w*1000))}"
        if dev is not None:
            lbl += f" ({dev:+.0f})"
        if role:
            lbl += f"\n{role.split(' /')[0]}"
        elif name.startswith("window"):
            lbl += "\nwindow"
        elif name.startswith("balcony"):
            lbl += "\nbalcony"
        elif name.startswith("archway"):
            lbl += "\narchway"
        off = n[:2] * 0.42
        ax.text(c[0] + off[0], c[1] + off[1], lbl, fontsize=7.0, ha="center",
                va="center", color="white", zorder=7,
                bbox=dict(fc=col, alpha=0.88, pad=1.4, lw=0))

    handles = [Line2D([], [], color=v, lw=6, label=f"{k} ({sum(1 for r in cls.values() if r.get('inferred_role')==k)})")
               for k, v in ROLE_COLOR.items()]
    handles += [
        Line2D([], [], color=OTHER, lw=6, label="window / balcony / archway"),
        Line2D([], [], marker="*", ls="", c="#ffd600", ms=17, mec="k", mew=0.5,
               label="EXTERIOR — no ceiling beyond"),
        Line2D([], [], marker="D", ls="", mfc="none", mec="#ffd600", mew=1.8,
               ms=9, label="uncertain — partial roof"),
        Line2D([], [], marker=".", ls="", c="#cfd8dc", ms=16,
               label="ceiling footprint (roofed area)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.95)
    ax.set_title("Openings — role from standard width, exterior from ceiling cover\n"
                 "label = measured width mm (deviation vs NBC standard)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(out_png, dpi=130)
    print("wrote", out_png)


if __name__ == "__main__":
    main(*sys.argv[1:4])
