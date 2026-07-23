"""house_poisson_parts.py
--------------------
Rebuild the house by CROPPING the Poisson mesh, never by re-synthesising it.

The extruded-box house was clean and lost everything: a wall became a rectangle,
so the grooves, reveals, arch soffits and beam faces that the scan actually
resolved (median triangle edge 15.7 mm) were thrown away. Here the measured wall
runs are used only as a STENCIL -- every triangle that survives is an original
Poisson triangle, so all of that relief comes through untouched.

The classifier is what a surface TOUCHES, not how tall it is:

    reaches floor AND ceiling  ->  wall / pillar
    reaches ceiling only       ->  beam, arch soffit, dropped ceiling
    reaches floor only         ->  furniture (dropped, but exported separately)

A height-span threshold would have deleted the beams and arches, which is
exactly the fabric worth keeping. Furniture is not deleted, it is written to its
own file: for deviation work you have to be able to see what was discarded.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\house_poisson_parts.py \\
      <poisson.obj> <major_walls.json> <out_dir>
"""
import sys, json, time
from pathlib import Path
from collections import Counter
import numpy as np
import cv2

CELL = 0.05          # m, plan raster
VERT_COS = 0.34      # |nz| below this = vertical face
NEAR = 0.30          # m, a wall column this close to a run joins that run
TOUCH = 0.50         # m, "reaches" the ceiling within this. Not tighter: the
                     # wall/ceiling junction curves in a Poisson surface and
                     # those triangles read as diagonal, so a wall's vertical
                     # faces stop ~0.35 m short of the slab.
SPAN = 1.20          # m, a ceiling-attached surface running down further than
                     # this is a wall; less is a beam / arch soffit
MIN_TRIS_CELL = 30   # a plan cell holding fewer triangles than this is Poisson
                     # speckle floating in mid-air. A quarter of all "filled"
                     # cells hold 1-2 triangles and they swamp every statistic.
MIN_CELLS = 12       # a part smaller than this is noise
PILLAR_MAX = 1.00    # m, free-standing full-height blob wider than this is a wall
MERGE_ANG = 8.0      # deg, two fragments this parallel can be the same wall
MERGE_OFF = 0.20     # m, ...if their centre-lines are also this close
MERGE_GAP = 1.60     # m, ...and the gap between them is no wider than a doorway
ATTACH = 0.60        # m, a beam / soffit this close to a wall is parented to it
DILATE = 1           # cells, glue a part's own speckle together


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load(path):
    """Stream the OBJ. 750 MB does not survive a careless parse."""
    V, F = [], []
    with open(path, "r", buffering=1 << 22) as fh:
        for ln in fh:
            if ln.startswith("v "):
                p = ln.split()
                V.append((float(p[1]), float(p[2]), float(p[3])))
            elif ln.startswith("f "):
                idx = [int(t.split("/")[0]) - 1 for t in ln.split()[1:]]
                for k in range(1, len(idx) - 1):
                    F.append((idx[0], idx[k], idx[k + 1]))
    return np.asarray(V, np.float64), np.asarray(F, np.int64)


def write_obj(path, V, groups):
    """One OBJ, one 'o' group per part, sharing the original vertex block.

    Re-indexing per part would triple the file; the vertex list is written once
    and every group indexes into it.
    """
    with open(path, "w", buffering=1 << 22) as fh:
        fh.write(f"# {len(groups)} objects cropped from the Poisson mesh\n")
        for x, y, z in V:
            fh.write(f"v {x:.5f} {y:.5f} {z:.5f}\n")
        for name, F in groups:
            fh.write(f"o {name}\n")
            for a, b, c in F:
                fh.write(f"f {a+1} {b+1} {c+1}\n")


def main(obj_path, walls_path, out_dir):
    t0 = time.time()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    W = json.load(open(walls_path))
    z0, z1 = W["z_floor"], W["z_ceiling"]
    runs = [s for s in W["walls"] if s.get("length_m", 0) >= 0.5]
    log(f"{len(runs)} wall runs, storey {z0:.2f}..{z1:.2f} m")

    V, F = load(obj_path)
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(B - A, C - A)
    nn = np.linalg.norm(cr, axis=1)
    area = 0.5 * nn
    n = cr / np.maximum(nn[:, None], 1e-12)
    cen = (A + B + C) / 3.0
    log(f"{len(V):,} verts / {len(F):,} tris / {area.sum():,.0f} m2")

    vert = np.abs(n[:, 2]) < VERT_COS
    lo = V.min(0)
    nx = int((V[:, 0].max() - lo[0]) / CELL) + 2
    ny = int((V[:, 1].max() - lo[1]) / CELL) + 2

    # ---- what each plan column of VERTICAL surface touches ---------------
    iv = np.where(vert)[0]
    ij = ((cen[iv, :2] - lo[:2]) / CELL).astype(np.int32)
    np.clip(ij[:, 0], 0, nx - 1, out=ij[:, 0])
    np.clip(ij[:, 1], 0, ny - 1, out=ij[:, 1])
    flat = ij[:, 1] * nx + ij[:, 0]
    zmin = np.full(nx * ny, np.inf)
    zmax = np.full(nx * ny, -np.inf)
    cnt = np.zeros(nx * ny, np.int64)
    np.minimum.at(zmin, flat, cen[iv, 2])
    np.maximum.at(zmax, flat, cen[iv, 2])
    np.add.at(cnt, flat, 1)
    filled = np.isfinite(zmin) & (cnt >= MIN_TRIS_CELL)
    on_ceil = filled & (zmax >= z1 - TOUCH)
    # Reaching the FLOOR is not a usable test: almost every wall is occluded at
    # its base by furniture standing against it, so it has no vertical face at
    # ankle height. Requiring floor contact filed 5,607 wall columns as beams.
    # Reaching the CEILING is reliable -- nothing hides the top of a wall -- so
    # the split is on how far DOWN the surface then continues.
    structural = on_ceil & ((zmax - zmin) >= SPAN)   # wall or pillar
    overhead = on_ceil & ~structural                 # beam, soffit, dropped slab
    clutter = filled & ~on_ceil                      # furniture
    log(f"VERTICAL PLAN COLUMNS: {int(filled.sum()):,} filled -> "
        f"{int(structural.sum()):,} structural, {int(overhead.sum()):,} overhead, "
        f"{int(clutter.sum()):,} furniture")
    log(f"  by area: structural {area[iv][structural[flat]].sum():,.0f} m2, "
        f"overhead {area[iv][overhead[flat]].sum():,.0f} m2, "
        f"furniture {area[iv][clutter[flat]].sum():,.0f} m2")

    # ---- assign structural columns to the measured wall runs -------------
    gx = lo[0] + (np.arange(nx) + 0.5) * CELL
    gy = lo[1] + (np.arange(ny) + 0.5) * CELL
    GX, GY = np.meshgrid(gx, gy)
    P = np.c_[GX.ravel(), GY.ravel()]
    owner = np.full(nx * ny, -1, np.int32)
    dist = np.full(nx * ny, np.inf)
    # Claim OVERHEAD cells too, not just structural ones. The strip where a wall
    # meets the slab reaches the ceiling but spans less than SPAN, so it scored
    # as "overhead" and every wall got a red beam painted along its top edge.
    # It is not a beam, it is the top of the wall underneath it -- so anything
    # sitting over a measured run belongs to that run whatever its span.
    cidx = np.where(structural | overhead)[0]
    for wi, s in enumerate(runs):
        p0, p1 = np.array(s["p0"], float), np.array(s["p1"], float)
        d = p1 - p0
        L = float(np.hypot(*d))
        if L < 1e-6:
            continue
        u = d / L
        rel = P[cidx] - p0
        t = np.clip(rel @ u, 0.0, L)
        perp = np.linalg.norm(rel - t[:, None] * u, axis=1)
        better = perp < np.minimum(dist[cidx], NEAR)
        tgt = cidx[better]
        owner[tgt] = wi
        dist[tgt] = perp[better]
    claimed = owner >= 0
    log(f"  claimed by a run: {int((claimed & structural).sum()):,} structural "
        f"+ {int((claimed & overhead).sum()):,} overhead (wall-head strips)")

    def blobs(mask, tag):
        g = mask.reshape(ny, nx).astype(np.uint8)
        if DILATE:
            g = cv2.morphologyEx(g, cv2.MORPH_CLOSE,
                                 np.ones((2 * DILATE + 1,) * 2, np.uint8))
        ncc, lab, st, _ = cv2.connectedComponentsWithStats(g, 8)
        labf = lab.ravel()
        out = []
        for c in range(1, ncc):
            if st[c, cv2.CC_STAT_AREA] < MIN_CELLS:
                continue
            w = st[c, cv2.CC_STAT_WIDTH] * CELL
            h = st[c, cv2.CC_STAT_HEIGHT] * CELL
            out.append((c, w, h, labf == c))
        log(f"  {len(out)} unclaimed {tag} blobs")
        return out

    # ---- unclaimed structural: a pillar is COMPACT, anything else is wall --
    # The old guard let a blob up to 3 m through as a "pillar", which is how a
    # whole wall came out orange. A column is near-square and small in plan;
    # a long unclaimed blob is a wall the run detector missed, so keep it as a
    # wall rather than dropping it (35% of structural area was falling through).
    pillar_id = np.zeros(nx * ny, np.int32)
    extra_id = np.zeros(nx * ny, np.int32)
    npil = nex = 0
    for c, w, h, m in blobs(structural & ~claimed, "structural"):
        short, long_ = min(w, h), max(w, h)
        if long_ <= PILLAR_MAX and short >= 0.10 and long_ / max(short, CELL) <= 2.5:
            npil += 1
            pillar_id[m] = npil
        else:
            nex += 1
            extra_id[m] = nex
    log(f"  -> {npil} pillars (compact, <= {PILLAR_MAX} m), "
        f"{nex} recovered walls (elongated, missed by the run detector)")

    # ---- unclaimed overhead = the real beams, soffits and dropped slabs ---
    over_id = np.zeros(nx * ny, np.int32)
    nov = 0
    for c, w, h, m in blobs(overhead & ~claimed, "overhead"):
        nov += 1
        over_id[m] = nov
    log(f"  -> {nov} beams / arch soffits / dropped ceilings "
        f"(free of any wall head)")

    # ---- merge the fragments into CONTINUOUS runs ------------------------
    # A wall is one object from end to end, not a row of tiles. The run
    # detector and the recovery pass between them produced 42 pieces of what
    # are physically about 20 walls, split at doorways, at T-junctions and
    # wherever a band went sparse. Fit a centre-line to each piece and union
    # any two that are collinear, side by side and nearly touching.
    def centreline(cells):
        pts = P[cells]
        c = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
        d = vt[0]
        t = (pts - c) @ d
        return c + t.min() * d, c + t.max() * d, d

    pieces = []
    for wi in range(len(runs)):
        cells = np.where(owner == wi)[0]
        if len(cells) >= MIN_CELLS:
            pieces.append([cells, runs[wi].get("kind", "wall")])
    for xid in range(1, nex + 1):
        cells = np.where(extra_id == xid)[0]
        if len(cells) >= MIN_CELLS:
            pieces.append([cells, "wall"])
    geom = [centreline(c) for c, _ in pieces]

    parent = list(range(len(pieces)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            if pieces[i][1] != pieces[j][1]:
                continue                       # never fuse a parapet to a wall
            (ai, bi, di), (aj, bj, dj) = geom[i], geom[j]
            if abs(float(di @ dj)) < np.cos(np.radians(MERGE_ANG)):
                continue                       # not parallel
            nrm = np.array([-di[1], di[0]])
            if abs(float((aj - ai) @ nrm)) > MERGE_OFF:
                continue                       # parallel but a different wall
            ti = sorted([float((ai - ai) @ di), float((bi - ai) @ di)])
            tj = sorted([float((aj - ai) @ di), float((bj - ai) @ di)])
            gap = max(tj[0] - ti[1], ti[0] - tj[1])
            if gap <= MERGE_GAP:               # overlapping or nearly touching
                parent[find(i)] = find(j)
    fused = {}
    for i in range(len(pieces)):
        fused.setdefault(find(i), []).append(i)
    log(f"CONTINUITY: {len(pieces)} fragments -> {len(fused)} continuous runs")

    wall_cells, wall_kind, wall_geom = [], [], []
    for root, members in fused.items():
        cells = np.concatenate([pieces[m][0] for m in members])
        wall_cells.append(cells)
        wall_kind.append(pieces[members[0]][1])
        wall_geom.append(centreline(cells))

    # ---- parent every beam / arch soffit to the wall it sits on ----------
    # CAD wants a hierarchy, not a pile: the soffit over a doorway belongs to
    # that doorway's wall. A blob too far from any wall stays free-standing.
    over_parent = {}
    for oid in range(1, nov + 1):
        cells = np.where(over_id == oid)[0]
        if len(cells) < MIN_CELLS:
            continue
        c = P[cells].mean(0)
        best, bd = -1, np.inf
        for k, (a, b, d) in enumerate(wall_geom):
            L = float(np.linalg.norm(b - a))
            t = np.clip(float((c - a) @ d), 0.0, L) if L > 1e-6 else 0.0
            dd = float(np.linalg.norm(c - (a + t * d)))
            if dd < bd:
                bd, best = dd, k
        over_parent[oid] = (best, bd) if bd <= ATTACH else (None, bd)
    att = sum(1 for v in over_parent.values() if v[0] is not None)
    log(f"  attached {att} of {len(over_parent)} overhead parts to a wall "
        f"(within {ATTACH} m); {len(over_parent)-att} free-standing")

    # ---- horizontal surfaces --------------------------------------------
    horz = ~vert
    ih = np.where(horz)[0]
    is_floor = cen[ih, 2] < (z0 + z1) / 2.0

    # ---- gather triangles into named groups ------------------------------
    groups, furn, sched = [], [], []
    member = np.full(nx * ny, -1, np.int32)
    for k, cells in enumerate(wall_cells):
        member[cells] = k
    nwall = 0
    wall_name = {}
    for k, cells in enumerate(wall_cells):
        m = iv[member[flat] == k]
        if len(m) < 200:
            continue
        nwall += 1
        a, b, d = wall_geom[k]
        nm = f"{wall_kind[k]}_{nwall:02d}"
        wall_name[k] = nm
        groups.append((nm, F[m]))
        zc = cen[m, 2]
        sched.append(dict(name=nm, kind=wall_kind[k], parent=None,
                          start=[round(float(v), 3) for v in a],
                          end=[round(float(v), 3) for v in b],
                          length_m=round(float(np.linalg.norm(b - a)), 3),
                          z_bottom=round(float(zc.min()), 3),
                          z_top=round(float(zc.max()), 3),
                          ntris=int(len(m)), children=[]))
    by_name = {s["name"]: s for s in sched}
    npar = 0
    for oid in range(1, nov + 1):
        m = iv[over_id[flat] == oid]
        if len(m) < 200:
            continue
        pk, pd = over_parent.get(oid, (None, np.inf))
        zc = cen[m, 2]
        # a soffit hanging well below the slab spans an opening -- that is an
        # arch or lintel head; one sitting tight under it is a beam
        kind = "arch" if zc.min() < z1 - 0.55 else "beam"
        if pk is not None and pk in wall_name:
            npar += 1
            pname = wall_name[pk]
            nm = f"{pname}__{kind}_{npar:02d}"
            by_name[pname]["children"].append(nm)
        else:
            pname = None
            nm = f"{kind}_free_{oid:02d}"
        groups.append((nm, F[m]))
        sched.append(dict(name=nm, kind=kind, parent=pname,
                          z_bottom=round(float(zc.min()), 3),
                          z_top=round(float(zc.max()), 3),
                          soffit_drop_m=round(float(z1 - zc.min()), 3),
                          ntris=int(len(m)), children=[]))
    for pid in range(1, npil + 1):
        m = iv[pillar_id[flat] == pid]
        if len(m) >= 200:
            groups.append((f"pillar_{pid:02d}", F[m]))
            sched.append(dict(name=f"pillar_{pid:02d}", kind="pillar",
                              parent=None, ntris=int(len(m)), children=[]))
    groups.append(("floor", F[ih[is_floor]]))
    groups.append(("ceiling", F[ih[~is_floor]]))
    furn.append(("furniture", F[iv[clutter[flat]]]))

    kept = sum(len(g) for _, g in groups)
    log(f"KEPT {kept:,} tris in {len(groups)} objects "
        f"({100*kept/len(F):.1f}% of the mesh)")
    log("  " + ", ".join(f"{k} {v}" for k, v in
                         Counter(s["kind"] for s in sched).most_common()))
    log(f"  {npar} beams/arches parented to a wall")
    wl = [s["length_m"] for s in sched if s["kind"] in ("wall", "parapet")]
    if wl:
        log(f"  wall runs: {len(wl)}, total {sum(wl):.1f} m, "
            f"longest {max(wl):.2f} m, median {np.median(wl):.2f} m")
    log(f"DROPPED {len(furn[0][1]):,} tris as furniture "
        f"({100*len(furn[0][1])/len(F):.1f}%)")

    write_obj(out / "house_poisson.obj", V, groups)
    write_obj(out / "furniture.obj", V, furn)
    json.dump(dict(schedule=sched, kept_tris=int(kept), total_tris=int(len(F)),
                   furniture_tris=int(len(furn[0][1])),
                   z_floor=z0, z_ceiling=z1),
              open(out / "poisson_parts.json", "w"), indent=1)
    log(f"wrote {out/'house_poisson.obj'} + furniture.obj in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(*sys.argv[1:4])
