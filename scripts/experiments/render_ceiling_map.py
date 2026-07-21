"""render_ceiling_map.py
----------------------
Plan-view height map of the ceiling, annotated with each part's height. The
cutaway render deletes the ceiling to show the rooms, so beams and dropped
ceilings never appear in it -- this is the view that shows them.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\render_ceiling_map.py \\
      <detailed_modular.obj> <features.json> <out_png>
"""
import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CELL = 0.04


def parse_obj(path, prefix):
    V = []; groups = []
    for ln in open(path):
        if ln.startswith("o "):
            groups.append([ln[2:].strip(), len(V), len(V)])
        elif ln.startswith("v "):
            _, x, y, z = ln.split()[:4]
            V.append((float(x), float(y), float(z)))
            groups[-1][2] = len(V)
    V = np.asarray(V)
    return {n: V[a:b] for n, a, b in groups if b > a and n.startswith(prefix)}


def main(obj_path, feat_path, out_png):
    G = parse_obj(obj_path, "ceiling")
    feats = {f["name"]: f for f in json.load(open(feat_path))["ceiling_features"]}
    allv = np.vstack(list(G.values()))
    x0, y0 = allv[:, 0].min(), allv[:, 1].min()
    nx = int((allv[:, 0].max() - x0) / CELL) + 1
    ny = int((allv[:, 1].max() - y0) / CELL) + 1
    H = np.full((ny, nx), np.nan)
    for name, P in G.items():
        h = feats.get(name, {}).get("height_mm")
        if h is None:
            continue
        i = ((P[:, 0] - x0) / CELL).astype(int)
        j = ((P[:, 1] - y0) / CELL).astype(int)
        H[j, i] = h

    fig, ax = plt.subplots(figsize=(15, 13))
    im = ax.imshow(H, origin="lower", cmap="viridis",
                   extent=[x0, x0 + nx * CELL, y0, y0 + ny * CELL],
                   interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.set_label("floor-to-ceiling height (mm)")

    for name, P in G.items():
        f = feats.get(name)
        if not f or f["area_m2"] < 0.6:
            continue
        cx, cy = P[:, 0].mean(), P[:, 1].mean()
        ax.text(cx, cy, f"{f['height_mm']:.0f}\n{f['type'].split('/')[0]}",
                ha="center", va="center", fontsize=7, color="white",
                bbox=dict(fc="black", alpha=0.55, pad=1.2, lw=0))
    ax.set_title("Ceiling height map — beams and dropped ceilings\n"
                 "(each labelled patch is its own measurable object)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(out_png, dpi=130)
    print("wrote", out_png)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
