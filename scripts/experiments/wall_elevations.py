"""wall_elevations.py
------------------
Per-wall elevation drawings: relief (grooves, pilasters, beam soffits) AND
openings (doors, windows, arches, balcony doors) on one image per wall.

Two different signals, deliberately kept separate:

  RELIEF   how far the surface sits from the wall's own best-fit plane.
           Positive = sticks out (pilaster, column, beam soffit); negative =
           cuts in (groove, niche, service chase).

  OPENINGS holes in the wall -- cells with no geometry at all, enclosed by wall
           on the sides and top. A door reaches the floor, a window has a sill,
           an archway runs to the ceiling with no lintel. These are invisible to
           the relief pass: an opening has no surface to deviate.

An arch is called out by the curvature of the opening's head: a flat lintel is
level across the top, an arch rises in the middle.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\wall_elevations.py \\
      <detailed_modular.obj> <measurements.json> <out_dir> [max_walls]
"""
import sys, json, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

CELL      = 0.025    # elevation grid cell
MIN_AREA  = 2.5      # m2: only draw walls at least this big
OPEN_MIN  = 0.10     # m2: minimum opening area
RELIEF_LIM= 0.06     # m: colour scale limit for the relief map
GROOVE_W  = 0.25     # m: a recess narrower than this is a groove, not a niche
ARCH_RISE = 0.06     # m: head rise above the springing that means "arched"
SILL_MIN  = 0.25     # m: an opening whose base is above this has a sill


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def parse_obj(path):
    V = []; groups = []
    for ln in open(path):
        if ln.startswith("o "):
            groups.append([ln[2:].strip(), len(V), len(V)])
        elif ln.startswith("v "):
            _, x, y, z = ln.split()[:4]
            V.append((float(x), float(y), float(z)))
            groups[-1][2] = len(V)
    V = np.asarray(V)
    return {n: V[a:b] for n, a, b in groups if b > a}


def label_regions(mask):
    """Connected components of a boolean grid (4-connected). Returns int labels."""
    try:
        from scipy import ndimage
        return ndimage.label(mask)[0]
    except Exception:
        lab = np.zeros(mask.shape, int); cur = 0
        for s in zip(*np.nonzero(mask)):
            if lab[s]:
                continue
            cur += 1; stack = [s]
            while stack:
                r, c = stack.pop()
                if not (0 <= r < mask.shape[0] and 0 <= c < mask.shape[1]):
                    continue
                if lab[r, c] or not mask[r, c]:
                    continue
                lab[r, c] = cur
                stack += [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]
        return lab


def merge_coplanar(G, ang_tol=6.0, off_tol=0.20):
    """Merge wall objects that lie on the same physical plane.

    The build step splits a plane into separate runs across gaps -- and a
    full-height opening (a balcony slider, an open passage) IS such a gap. That
    turns the opening into empty space BETWEEN two objects instead of a hole
    INSIDE one, so hole detection never sees it. Merging the runs back onto one
    plane before detection restores those openings.
    """
    keys = {}
    for n, P in G.items():
        Q = P[:, :2] - P[:, :2].mean(0)
        _, _, Vt = np.linalg.svd(Q, full_matrices=False)
        nv = Vt[1] / np.linalg.norm(Vt[1])
        ang = np.degrees(np.arctan2(nv[1], nv[0])) % 180.0
        r = np.radians(ang)
        keys[n] = (ang, float(P[:, :2].mean(0) @ np.array([np.cos(r), np.sin(r)])))
    names = list(G)
    parent = {n: n for n in names}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            da = abs(keys[a][0] - keys[b][0]); da = min(da, 180 - da)
            if da < ang_tol and abs(keys[a][1] - keys[b][1]) < off_tol:
                parent[find(a)] = find(b)
    out = {}
    for n in names:
        out.setdefault(find(n), []).append(n)
    merged = {}
    for root, members in out.items():
        members.sort()
        name = members[0] if len(members) == 1 else f"{members[0]}+{len(members)-1}"
        merged[name] = np.vstack([G[m] for m in members])
    return merged


def grids(P):
    """Wall -> (relief grid, occupancy grid, axes). Rows = z, cols = along."""
    c = P.mean(0)
    Q = P[:, :2] - c[:2]
    _, _, Vt = np.linalg.svd(Q - Q.mean(0), full_matrices=False)
    n = Vt[1] / np.linalg.norm(Vt[1])
    d = np.array([-n[1], n[0]])
    dev = Q @ n
    hist, edges = np.histogram(dev, bins=200)          # re-centre on wall BODY
    dev = dev - 0.5 * (edges[hist.argmax()] + edges[hist.argmax() + 1])
    along = Q @ d
    z = P[:, 2]

    a0, z0 = along.min(), z.min()
    na = int((along.max() - a0) / CELL) + 1
    nz = int((z.max() - z0) / CELL) + 1
    ai = np.clip(((along - a0) / CELL).astype(int), 0, na - 1)
    zi = np.clip(((z - z0) / CELL).astype(int), 0, nz - 1)

    occ = np.zeros((nz, na), bool)
    occ[zi, ai] = True
    acc = np.zeros((nz, na)); cnt = np.zeros((nz, na))
    np.add.at(acc, (zi, ai), dev)
    np.add.at(cnt, (zi, ai), 1)
    rel = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    return rel, occ, a0, z0, na, nz


def find_openings(occ, a0, z0, z_floor, z_ceil):
    """Enclosed empty regions = openings. Flood the empty space inward from the
    left/right/top borders to mark what is merely OUTSIDE the wall; whatever
    empty region is left is a real hole. The bottom border is excluded because a
    door legitimately runs to the floor and would otherwise leak to 'outside'."""
    nz, na = occ.shape
    empty = ~occ
    lab = label_regions(empty)
    outside = set(lab[:, 0]) | set(lab[:, -1]) | set(lab[-1, :])
    outside.discard(0)

    out = []
    for L in range(1, lab.max() + 1):
        if L in outside:
            continue
        rs, cs = np.nonzero(lab == L)
        area = len(rs) * CELL * CELL
        if area < OPEN_MIN:
            continue
        w = (cs.max() - cs.min() + 1) * CELL
        h = (rs.max() - rs.min() + 1) * CELL
        if w < 0.35 or h < 0.35:
            continue
        zb = z0 + rs.min() * CELL
        zt = z0 + (rs.max() + 1) * CELL
        ab = a0 + cs.min() * CELL
        sill = zb - z_floor

        # Head profile: top row of the hole per column. A flat lintel is level
        # across the whole width; an arch rises in the MIDDLE. Compare the
        # median of the central third against the median of the two outer
        # thirds -- using medians, not max, so one ragged cell cannot fake an
        # arch (raw max called every opening in this flat arched).
        head = np.array([rs[cs == c].max() for c in np.unique(cs)]) * CELL + z0
        k = len(head)
        if k >= 9:
            t = k // 3
            rise = float(np.median(head[t:2 * t]) -
                         np.median(np.r_[head[:t], head[2 * t:]]))
        else:
            rise = 0.0
        arched = rise > ARCH_RISE

        if sill < SILL_MIN:                                  # reaches the floor
            if zt > z_ceil - 0.30:
                kind = "archway / open passage"
            elif w >= 1.30:
                kind = "balcony / sliding door"
            elif 0.55 <= w <= 1.30 and 1.70 <= h <= 2.40:
                kind = "door"
            else:
                kind = "floor-level opening"
        else:
            kind = "window" if 0.35 <= sill <= 1.60 else "high-level opening"
        if arched and "window" not in kind:
            kind = "arched " + kind

        out.append(dict(type=kind, width_m=round(w, 3), height_m=round(h, 3),
                        sill_mm=round(sill * 1000, 1),
                        head_mm=round((zt - z_floor) * 1000, 1),
                        head_rise_mm=round(rise * 1000, 1),
                        area_m2=round(area, 2),
                        along_m=round(ab, 3),
                        _box=(ab, zb, w, h)))
    out.sort(key=lambda o: -o["area_m2"])
    return out


def find_relief(rel, a0, z0, z_ceil):
    """Protrusions and recesses on the wall face."""
    feats = []
    for sign, thr in ((1, 0.030), (-1, -0.030)):
        m = (rel > thr) if sign > 0 else (rel < thr)
        m = np.where(np.isnan(rel), False, m)
        lab = label_regions(m)
        for L in range(1, lab.max() + 1):
            rs, cs = np.nonzero(lab == L)
            if len(rs) < 40:
                continue
            w = (cs.max() - cs.min() + 1) * CELL
            h = (rs.max() - rs.min() + 1) * CELL
            depth = float(np.nanmax(np.abs(rel[rs, cs])))
            zb = z0 + rs.min() * CELL; zt = z0 + (rs.max() + 1) * CELL
            ab = a0 + cs.min() * CELL
            if sign > 0:
                if w <= 0.80 and h > 1.6:
                    k = "column / pilaster"
                elif zt > z_ceil - 0.45 and w > 2 * h:
                    k = "beam soffit"
                else:
                    k = "protrusion"
            else:
                if min(w, h) < GROOVE_W and max(w, h) > 3 * min(w, h):
                    k = "groove"
                elif h > 1.6:
                    k = "duct / chase"
                else:
                    k = "niche"
            feats.append(dict(type=k, width_m=round(w, 3), height_m=round(h, 3),
                              depth_mm=round(depth * 1000, 1),
                              area_m2=round(len(rs) * CELL * CELL, 2),
                              _box=(ab, zb, w, h)))
    feats.sort(key=lambda f: -f["area_m2"])
    return feats


STYLE = {                      # label -> (edge colour, text colour)
    "door": ("#00e5ff", "#00e5ff"),
    "balcony / sliding door": ("#00ff90", "#00ff90"),
    "window": ("#ffd400", "#ffd400"),
    "archway / open passage": ("#ff8a00", "#ff8a00"),
    "column / pilaster": ("#ff2d95", "#ff2d95"),
    "beam soffit": ("#b388ff", "#b388ff"),
    "groove": ("#7CFC00", "#7CFC00"),
    "niche": ("#ff6e6e", "#ff6e6e"),
    "duct / chase": ("#ff9e80", "#ff9e80"),
}


def style_for(kind):
    for k, v in STYLE.items():
        if k in kind:
            return v
    return ("#cccccc", "#cccccc")


def draw(name, rel, occ, a0, z0, opens, feats, z_floor, out_png):
    nz, na = occ.shape
    ext = [a0, a0 + na * CELL, z0, z0 + nz * CELL]
    fig, ax = plt.subplots(figsize=(max(7, na * CELL * 1.5), max(4.5, nz * CELL * 1.6)))
    img = np.ma.masked_invalid(rel)
    ax.imshow(img, origin="lower", extent=ext, cmap="RdBu_r",
              vmin=-RELIEF_LIM, vmax=RELIEF_LIM, interpolation="nearest")
    ax.imshow(np.ma.masked_where(occ, np.ones_like(occ, float)), origin="lower",
              extent=ext, cmap="gray", vmin=0, vmax=1, interpolation="nearest")

    for f in feats + opens:
        x, y, w, h = f["_box"]
        ec, tc = style_for(f["type"])
        ls = "-" if f in opens else "--"
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=ec, lw=2.0, ls=ls))
        dim = (f"{f['type']}\n{w*1000:.0f} x {h*1000:.0f} mm"
               + (f"\nsill {f['sill_mm']:.0f}" if "sill_mm" in f else
                  f"\n{f['depth_mm']:.0f} mm deep"))
        ax.text(x + w / 2, y + h / 2, dim, ha="center", va="center", fontsize=7,
                color=tc, bbox=dict(fc="black", alpha=0.65, pad=1.5, lw=0))

    ax.axhline(z_floor, color="w", lw=0.8, alpha=0.5)
    ax.set_title(f"{name} — elevation   "
                 f"({na*CELL:.2f} x {nz*CELL:.2f} m)   "
                 f"solid box = opening, dashed = relief\n"
                 f"blue = cuts in (groove/niche), red = sticks out "
                 f"(pilaster/beam), grey = hole")
    ax.set_xlabel("along wall (m)"); ax.set_ylabel("z (m)")
    ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(out_png, dpi=115); plt.close(fig)


def main(obj_path, mj, out_dir, max_walls=40):
    max_walls = int(max_walls)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    d = json.load(open(mj))
    z_floor = float(np.median([r["z_floor"] for r in d["rooms"]]))
    z_ceil = float(np.median([r["z_ceiling"] for r in d["rooms"]]))

    G0 = {n: P for n, P in parse_obj(obj_path).items() if n.startswith("wall")}
    G = merge_coplanar(G0)
    order = sorted(G, key=lambda n: -len(G[n]))
    log(f"{len(G0)} wall objects -> {len(G)} planes; drawing up to {max_walls}")

    index = []
    for name in order[:max_walls]:
        P = G[name]
        if len(P) < 2000:
            continue
        rel, occ, a0, z0, na, nz = grids(P)
        if na * CELL * nz * CELL < MIN_AREA:
            continue
        opens = find_openings(occ, a0, z0, z_floor, z_ceil)
        feats = find_relief(rel, a0, z0, z_ceil)
        png = out / f"{name}_elevation.png"
        draw(name, rel, occ, a0, z0, opens, feats, z_floor, png)
        index.append(dict(name=name, png=png.name,
                          size_m=[round(na * CELL, 2), round(nz * CELL, 2)],
                          openings=[{k: v for k, v in o.items() if k != "_box"}
                                    for o in opens],
                          relief=[{k: v for k, v in f.items() if k != "_box"}
                                  for f in feats]))
        kinds = ", ".join(sorted({o["type"] for o in opens})) or "-"
        log(f"{name:10} {na*CELL:5.2f}x{nz*CELL:4.2f} m  "
            f"{len(opens)} openings [{kinds}]  {len(feats)} relief")
    json.dump(index, open(out / "elevations.json", "w"), indent=1)

    tally = {}
    for w in index:
        for o in w["openings"]:
            tally[o["type"]] = tally.get(o["type"], 0) + 1
        for f in w["relief"]:
            tally[f["type"]] = tally.get(f["type"], 0) + 1
    log("TOTALS: " + ", ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    log(f"wrote {len(index)} elevations to {out}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
