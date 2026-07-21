"""analyze_features.py
--------------------
Measure the modular house and find its UNDULATIONS -- the places where a surface
departs from its own best-fit plane. That is where columns, pilasters, beams,
niches and steps live, and it is the same signal the deviation product needs.

Two passes:

  WALLS   fit a plane to each wall object, build a (along x height) relief grid
          of signed deviation, and cluster cells that stick out (+) or cut in (-).
          A full-height narrow protrusion is a column/pilaster; a band hugging
          the ceiling is a down-stand beam; a recess is a niche or a service duct.

  CEILING build an (x,y) height map of the shared ceiling slab and cluster cells
          that hang below the room's ceiling plane. Those are the beams. The
          cutaway render hides them because it removes the ceiling object -- they
          are in the model, they were just never rendered or listed.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\analyze_features.py \\
      <detailed_modular.obj> <measurements.json> <out_dir>
"""
import sys, json, time
from pathlib import Path
import numpy as np

# ---- config (metres) ----
CELL      = 0.025    # wall relief grid cell
PROTRUDE  = 0.030    # dev above this = sticks out of the wall
RECESS    = -0.030   # dev below this = cuts into the wall
MIN_CELLS = 40       # minimum cells for a wall feature
CCELL     = 0.05     # ceiling height-map cell
BEAM_DROP = 0.060    # ceiling hanging this far below its plane = beam
BEAM_CELLS= 120      # minimum cells for a beam
COL_MAXW  = 0.80     # protrusion narrower than this (and tall) = column/pilaster
BEAM_ZTOP = 0.45     # a wall band within this of the ceiling = beam soffit


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def parse_obj(path):
    """Read `o <name>` groups -> {name: vertex array}. Faces not needed here."""
    V = []; groups = []; cur = None
    for ln in open(path):
        if ln.startswith("o "):
            cur = ln[2:].strip(); groups.append([cur, len(V), len(V)])
        elif ln.startswith("v "):
            _, x, y, z = ln.split()[:4]
            V.append((float(x), float(y), float(z)))
            groups[-1][2] = len(V)
    V = np.asarray(V)
    return {n: V[a:b] for n, a, b in groups if b > a}


def components(cells):
    """4-connected components over a set of (i,j) integer cells."""
    cs = set(cells); seen = set(); out = []
    for c in cs:
        if c in seen:
            continue
        stack = [c]; comp = []
        while stack:
            x = stack.pop()
            if x in seen or x not in cs:
                continue
            seen.add(x); comp.append(x)
            i, j = x
            stack += [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]
        out.append(comp)
    return out


def fit_plane(P):
    """Best-fit vertical plane: horizontal normal from the smallest-variance
    horizontal direction. Returns (centroid, normal2d, along_dir2d)."""
    c = P.mean(0)
    Q = P[:, :2] - c[:2]
    _, _, Vt = np.linalg.svd(Q - Q.mean(0), full_matrices=False)
    n = Vt[1]                          # least-variance horizontal dir = normal
    n = n / np.linalg.norm(n)
    d = np.array([-n[1], n[0]])
    return c, n, d


def analyse_wall(name, P, z_ceil):
    c, n, d = fit_plane(P)
    dev = (P[:, 0] - c[0]) * n[0] + (P[:, 1] - c[1]) * n[1]
    # re-centre on the wall BODY (the modal plane), not the mean, so that a big
    # pilaster does not drag the reference plane out with it
    hist, edges = np.histogram(dev, bins=200)
    dev = dev - 0.5 * (edges[hist.argmax()] + edges[hist.argmax() + 1])
    along = (P[:, 0] - c[0]) * d[0] + (P[:, 1] - c[1]) * d[1]
    z = P[:, 2]
    length = float(along.max() - along.min())
    height = float(z.max() - z.min())

    ai = np.floor((along - along.min()) / CELL).astype(int)
    zi = np.floor((z - z.min()) / CELL).astype(int)
    key = ai.astype(np.int64) * 100000 + zi
    order = np.argsort(key)
    k_s, dev_s = key[order], dev[order]
    bounds = np.r_[0, np.flatnonzero(np.diff(k_s)) + 1, len(k_s)]
    cell_ij, cell_dev = [], []
    for s, e in zip(bounds[:-1], bounds[1:]):
        cell_ij.append((int(k_s[s] // 100000), int(k_s[s] % 100000)))
        cell_dev.append(float(np.median(dev_s[s:e])))
    cell_ij = np.array(cell_ij); cell_dev = np.array(cell_dev)

    feats = []
    for sign, thr, kind in ((1, PROTRUDE, "protrusion"), (-1, RECESS, "recess")):
        sel = cell_dev > thr if sign > 0 else cell_dev < thr
        if sel.sum() < MIN_CELLS:
            continue
        idx = np.flatnonzero(sel)
        lut = {tuple(cell_ij[i]): i for i in idx}
        for comp in components(lut.keys()):
            if len(comp) < MIN_CELLS:
                continue
            ii = np.array([lut[c2] for c2 in comp])
            a0 = cell_ij[ii, 0]; z0 = cell_ij[ii, 1]
            w = (a0.max() - a0.min() + 1) * CELL
            h = (z0.max() - z0.min() + 1) * CELL
            depth = float(np.abs(cell_dev[ii]).max())
            ztop = z.min() + (z0.max() + 1) * CELL
            zbot = z.min() + z0.min() * CELL
            if kind == "protrusion":
                if w <= COL_MAXW and h > 0.6 * height:
                    label = "column/pilaster"
                elif ztop > z_ceil - BEAM_ZTOP and w > 2 * h:
                    label = "beam soffit"
                else:
                    label = "protrusion"
            else:
                label = "niche/recess" if h < 0.7 * height else "duct/chase"
            feats.append(dict(type=label, width_m=round(w, 3), height_m=round(h, 3),
                              depth_mm=round(depth * 1000, 1),
                              z_bottom=round(float(zbot), 3), z_top=round(float(ztop), 3),
                              along_start=round(float(along.min() + a0.min() * CELL), 3),
                              n_cells=len(comp)))
    feats.sort(key=lambda f: -f["depth_mm"])
    body = cell_dev[(cell_dev > RECESS) & (cell_dev < PROTRUDE)]
    return dict(name=name, length_m=round(length, 3), height_m=round(height, 3),
                area_m2=round(length * height, 2), n_verts=int(len(P)),
                flatness_rms_mm=round(float(np.std(body) * 1000), 1) if body.size else None,
                heading_deg=round(float(np.degrees(np.arctan2(n[1], n[0])) % 180), 1),
                features=feats)


def analyse_ceiling(parts, z_floor):
    """Classify each ceiling plateau. The build step already separates the slab
    into flat plateaus; here each one gets a height and a name.

    A plateau sitting below the main ceiling IS a beam soffit or a dropped
    ceiling -- long and narrow means beam, broad means a dropped/false ceiling
    over a wet room. Nothing was missing from the model; it was one merged
    object with no per-part height until now."""
    if not parts:
        return []
    areas = np.array([p["area_m2"] for p in parts])
    heights = np.array([p["height_mm"] for p in parts])
    main = float(heights[np.argmax(areas)])        # tallest-by-area = main ceiling
    log(f"main ceiling height {main:.0f} mm (largest plateau)")
    out = []
    for p, h, a in zip(parts, heights, areas):
        P = p["verts"]
        w = float(np.ptp(P[:, 0])); l = float(np.ptp(P[:, 1]))
        span, thick = max(w, l), min(w, l)
        drop = main - h
        if drop < 40:
            kind = "main ceiling"
        elif span > 3 * thick and a < 6:
            kind = "beam soffit"
        elif a >= 2.0:
            kind = "dropped ceiling"
        else:
            kind = "bulkhead/step"
        out.append(dict(name=p["name"], type=kind, height_mm=round(float(h), 1),
                        drop_mm=round(float(drop), 1), area_m2=round(float(a), 2),
                        span_m=round(span, 2), width_m=round(thick, 2),
                        flatness_mm=p.get("flatness_mm")))
    out.sort(key=lambda b: (-b["area_m2"]))
    return out


def main(obj_path, mj, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    d = json.load(open(mj))
    z_ceil = float(np.median([r["z_ceiling"] for r in d["rooms"]]))
    z_floor = float(np.median([r["z_floor"] for r in d["rooms"]]))
    log(f"reference z_floor={z_floor:.3f} z_ceil={z_ceil:.3f} "
        f"(height {(z_ceil-z_floor)*1000:.0f} mm)")

    G = parse_obj(obj_path)
    log(f"parsed {len(G)} objects")

    walls = []
    for name, P in G.items():
        if not name.startswith("wall") or len(P) < 500:
            continue
        walls.append(analyse_wall(name, P, z_ceil))
    walls.sort(key=lambda w: -w["area_m2"])

    man = json.load(open(Path(obj_path).parent / "modular_manifest.json"))
    cparts = [dict(o, verts=G[o["name"]]) for o in man["objects"]
              if o["name"].startswith("ceiling") and o["name"] in G
              and "height_mm" in o]
    beams = analyse_ceiling(cparts, z_floor)

    cols = [dict(name=n, footprint_m=[round(float(np.ptp(P[:, 0])), 2),
                                      round(float(np.ptp(P[:, 1])), 2)],
                 height_m=round(float(np.ptp(P[:, 2])), 2))
            for n, P in G.items() if n.startswith("column")]

    rep = dict(reference=dict(z_floor=z_floor, z_ceiling=z_ceil,
                              height_mm=round((z_ceil - z_floor) * 1000, 1)),
               rooms=d["rooms"], walls=walls, ceiling_features=beams, columns=cols)
    json.dump(rep, open(out / "features.json", "w"), indent=1)

    nf = sum(len(w["features"]) for w in walls)
    log(f"{len(walls)} walls measured, {nf} wall features, {len(beams)} ceiling features")
    for w in walls[:12]:
        fs = ", ".join(f"{f['type']} {f['depth_mm']:.0f}mm" for f in w["features"][:3])
        log(f"  {w['name']:10} {w['length_m']:5.2f} x {w['height_m']:4.2f} m  "
            f"rms={w['flatness_rms_mm']}mm  {fs}")
    for b in beams[:20]:
        log(f"  {b['name']:12} {b['type']:16} h={b['height_mm']:6.0f} mm  "
            f"drop {b['drop_mm']:5.0f} mm  {b['span_m']:5.2f} x {b['width_m']:4.2f} m  "
            f"{b['area_m2']:5.2f} m2")
    log(f"wrote {out/'features.json'}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
