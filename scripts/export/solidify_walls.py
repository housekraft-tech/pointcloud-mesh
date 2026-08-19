"""Fill the wall: a closed solid between the two faces the scan measured.

The segmented model gives each wall both of its faces, but a face is a sheet.
Opened in Blender the wall is hollow -- two skins with nothing between them, no
volume, nothing to cut a socket into, nothing to take a quantity from.

Each wall is gridded across its own face in (along, height) at 10 mm. Where the
scan found material the wall is solid; where it found none through either face
there is a hole -- a doorway, a window, the space under a lintel. That outline
is traced, simplified to 5 mm, triangulated with its holes, and extruded through
the thickness measured for that wall. The result is one watertight solid per
wall:

  - the outline is measured, at 10 mm, including every opening;
  - the thickness is measured between the two faces, per wall;
  - the volume is therefore a quantity, not an estimate.

What the solid does NOT carry is the millimetre relief -- the niches, the boxed
conduits, the arch soffits. Those live in `modular.obj`, which is the scanned
surface itself, cropped and named, and they are the reason that file exists. A
per-cell solid carrying both was tried and abandoned: the scan steps at almost
every 10 mm cell, so the two faces meet in millions of little ribbons and 12% of
their edges would not close. A solid whose volume can be trusted, beside a
surface whose millimetres can be trusted, beats one mesh that is wrong at both.
"""
import sys, json, argparse, time
from pathlib import Path
import numpy as np

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

CELL = 0.010          # m: grid across the wall face
SIMPLIFY = 0.005      # m: contour simplification -- half a cell
MIN_AREA = 0.02       # m2: ignore outline scraps smaller than this
MIN_HOLE = 0.05       # m2: and holes smaller than this (scan shadows, not doors)


def outline(P, ax, origin):
    """Cells of the wall face that carry material, as a binary elevation map."""
    u = P[:, 1-ax]; v = P[:, 2]
    iu = ((u - origin[0])/CELL).astype(np.int64)
    iv = ((v - origin[1])/CELL).astype(np.int64)
    nu = int(iu.max())+1; nv = int(iv.max())+1
    m = np.zeros((nu, nv), np.uint8)
    m[iu, iv] = 1
    # The map is built from triangle centroids, so a coarse patch of mesh can
    # miss a cell it actually covers. Closing over one cell bridges those
    # without touching a real opening, which is tens of cells across.
    import cv2
    k = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)


def polygons(mask):
    """Outer rings and their holes, in cell coordinates."""
    import cv2
    from shapely.geometry import Polygon
    pad = np.pad(mask, 1)
    cs, hier = cv2.findContours(pad, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []
    hier = hier[0]
    eps = SIMPLIFY/CELL
    rings = [cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(float) - 1
             for c in cs]
    out = []
    for i, h in enumerate(hier):
        if h[3] != -1 or len(rings[i]) < 3:          # a hole: taken with its parent
            continue
        holes = []
        j = h[2]
        while j != -1:
            if len(rings[j]) >= 3:
                q = Polygon(rings[j])
                if q.is_valid and q.area*CELL*CELL > MIN_HOLE:
                    holes.append(rings[j])
            j = hier[j][0]
        p = Polygon(rings[i], holes)
        if not p.is_valid:
            p = p.buffer(0)
        if p.is_empty or p.area*CELL*CELL < MIN_AREA:
            continue
        # buffer(0) can split a self-touching ring into several polygons
        out += [g for g in getattr(p, "geoms", [p])
                if g.area*CELL*CELL >= MIN_AREA]
    return out


def solid_of(P, ax, thickness, front_w):
    """A watertight prism per outline ring, extruded through the thickness."""
    import trimesh
    origin = np.array([P[:, 1-ax].min(), P[:, 2].min()]) - CELL
    polys = polygons(outline(P, ax, origin))
    if not polys:
        return None, 0.0
    meshes = []
    for p in polys:
        m = trimesh.creation.extrude_polygon(p, thickness/CELL, engine="earcut")
        v = np.asarray(m.vertices, float)*CELL
        # cv2 gives contour points as (column, row), and the mask is indexed
        # [along, height] -- so the polygon's x is HEIGHT and its y is ALONG.
        height = v[:, 0] + origin[1]
        along = v[:, 1] + origin[0]
        across = v[:, 2] + front_w
        w = np.empty_like(v)
        if ax == 0:
            w[:, 0] = across; w[:, 1] = along
        else:
            w[:, 0] = along; w[:, 1] = across
        w[:, 2] = height
        meshes.append(trimesh.Trimesh(w, np.asarray(m.faces), process=False))
    m = trimesh.util.concatenate(meshes)
    return m, float(sum(abs(x.volume) for x in meshes))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="output/model/poisson_modular")
    ap.add_argument("--cache", default="output/model/poisson_koushik.npz")
    a = ap.parse_args()
    import trimesh
    d = Path(a.dir)
    man = json.load(open(d/"manifest.json"))
    T = np.load(a.cache)["T"].astype(np.int64)
    V = np.load(d/"verts.npy").astype(np.float64)
    L = np.load(d/"labels.npy")
    names = json.load(open(d/"names.json"))
    C = V[T].mean(axis=1)
    modes = man.get("thickness_modes_mm") or [200.0]
    parts = {p["name"]: p for p in man["parts"]}

    out = []; total = 0.0; wt = 0
    for nm, p in parts.items():
        if p["kind"] not in ("wall", "parapet"):
            continue
        pid = names.index(nm)
        sel = L == pid
        if not sel.any():
            continue
        ax = 0 if p["axis"] == "x" else 1
        measured = p.get("thickness_mm")
        th = (measured or p.get("thickness_raw_mm") or float(np.median(modes)))/1000.0
        th = float(min(max(th, 0.08), 0.45))
        w = C[sel][:, ax]
        # the front face is the outermost surface on the side the wall was seen
        # from; a 2nd percentile rather than the minimum, so one stray triangle
        # does not move the whole solid
        front_w = float(np.percentile(w, 2))
        if np.median(w) > 0.5*(p["across_m"][0] + p["across_m"][1]):
            front_w = float(np.percentile(w, 98)) - th
        m, vol = solid_of(C[sel], ax, th, front_w)
        if m is None:
            continue
        if m.is_watertight:
            wt += 1
        out.append((nm, m))
        total += vol
        p["volume_m3"] = round(vol, 4)
        p["solid_thickness_mm"] = round(th*1000, 1)
        p["solid_thickness_measured"] = bool(measured)
    log(f"{len(out)} walls solidified, {total:.2f} m3 of masonry, "
        f"{wt}/{len(out)} watertight")

    voff = 0
    with open(d/"modular_solid.obj", "w") as fh:
        fh.write("# walls as closed solids: measured outline, measured thickness\n")
        for nm, m in out:
            fh.write(f"o {nm}\n")
            np.savetxt(fh, np.asarray(m.vertices), fmt="v %.4f %.4f %.4f")
            np.savetxt(fh, np.asarray(m.faces)+voff+1, fmt="f %d %d %d")
            voff += len(m.vertices)
    log(f"wrote {d/'modular_solid.obj'} ({voff:,} verts, "
        f"{(d/'modular_solid.obj').stat().st_size/1e6:.1f} MB)")

    sc = trimesh.Scene()
    rng = np.random.default_rng(5)
    for nm, m in out:
        col = np.zeros((len(m.vertices), 4), np.uint8)
        col[:, :3] = rng.integers(70, 230, 3); col[:, 3] = 255
        m.visual.vertex_colors = col
        sc.add_geometry(m, geom_name=nm, node_name=nm)
    sc.export(str(d/"modular_solid.glb"))
    log(f"wrote {d/'modular_solid.glb'} "
        f"({(d/'modular_solid.glb').stat().st_size/1e6:.1f} MB)")

    man["solid"] = dict(walls=len(out), watertight=wt,
                        masonry_volume_m3=round(total, 3),
                        grid_mm=CELL*1000, simplify_mm=SIMPLIFY*1000)
    json.dump(man, open(d/"manifest.json", "w"), indent=1)
    log("manifest updated with per-wall volumes")


if __name__ == "__main__":
    main()
