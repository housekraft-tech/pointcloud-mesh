"""build_detailed_modular.py
---------------------------
Approach A (spec 2026-07-21, rev2): SEGMENT the Poisson mesh into a modular
house, keeping COMPLETE wall surfaces. Uses the Poisson mesh as the base
geometry and partitions its triangles by surface orientation + nearest measured
wall plane:
  - vertical-normal triangles  -> nearest wall plane within its extent  (full walls)
  - horizontal-normal, high     -> shared ceiling (beams)
  - horizontal-normal, low      -> shared floor
  - leftover vertical clusters   -> free-standing columns
  - everything else              -> discarded (furniture)

Keeps every triangle of a wall (both faces, full height, all relief) instead of
sampling a thin vertex band. Load the 716 MB Poisson once.

IMPORTANT: crop a Poisson mesh with the measurements.json FROM THE SAME SCAN
(koushik Poisson <-> koushik measurements) or the planes are offset and walls
are missed.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\build_detailed_modular.py \\
      <poisson.obj> <measurements.json> <out_dir>
"""
import sys, json, time
from pathlib import Path
import numpy as np
import open3d as o3d

# ---- config (metres) ----
HALF_THICK = 0.13     # half wall band around plane (grabs both faces of ~110mm wall)
ALONG_MARG = 0.20     # extend past wall ends
VERT_NZ    = 0.55     # |normal.z| below this = vertical surface (wall); above = flat (slab)
CEIL_FROM_MID = 0.30  # ceiling = flat tris above mid-height + this
FLOOR_BAND = 0.12     # floor = flat tris below z_floor + this
CEIL_BAND  = 0.30     # ceiling also = any flat tri within this of z_ceiling (beams)
DUP_ANG    = 6.0      # deg: merge walls within this heading
DUP_OFF    = 0.15     # m: merge walls within this perpendicular offset
COL_CELL   = 0.12     # column-detect grid cell
COL_ZSPAN  = 0.60     # frac of room height a column must span
COL_MAXFOOT= 0.80     # m: max column footprint dimension (tighter -> less furniture)
COL_TOUCH  = 0.25     # m: column must reach within this of BOTH floor and ceiling
COL_ASPECT = 3.0      # max footprint aspect ratio (a column is near-square)
MIN_TRIS   = 300      # below this a wall is flagged 'low'
CEIL_CELL  = 0.05     # ceiling height-map cell
CEIL_STEP  = 0.030    # m: cell-to-cell height step that breaks a ceiling plateau
CEIL_MINCELL = 60     # minimum cells for a ceiling part (0.15 m2)
# --- unmeasured-wall recovery (rev3): plane-detect the residual vertical tris ---
EX_ANG     = 5.0      # deg: heading bin (seeding only)
EX_ANGW    = 35.0     # deg: membership heading tolerance -- Poisson surface noise
                      #      wobbles triangle normals; too tight and one wall
                      #      shatters into many interleaved objects
EX_OFF     = 0.10     # m: offset bin
EX_COLL    = 0.17     # m: collect band -- wide enough to take BOTH faces of a
                      #    ~110 mm wall into one object instead of two
EX_TOP     = 0.40     # m: a wall must reach within this of the ceiling
EX_GAP     = 0.40     # m: split a plane into separate walls across gaps this big
EX_MINLEN  = 0.50     # m: minimum run length to be a wall
EX_ZFRAC   = 0.55     # frac of room height the run must span
EX_MINTRIS = 400      # minimum triangles for a recovered wall


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_frame(poisson, mj):
    log(f"loading {poisson} ...")
    mesh = o3d.io.read_triangle_mesh(str(poisson))
    v = np.asarray(mesh.vertices)
    if np.argmin(v.max(0) - v.min(0)) == 1:                # Y-up -> Z-up
        Rz = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
        mesh.rotate(Rz, center=(0, 0, 0))
    mesh.compute_triangle_normals()
    V = np.asarray(mesh.vertices)
    tri = np.asarray(mesh.triangles)
    tn = np.asarray(mesh.triangle_normals)
    tc = V[tri].mean(axis=1)                               # triangle centroids
    d = json.load(open(mj))
    rooms = d["rooms"]
    z_floor = float(np.median([r["z_floor"] for r in rooms]))
    z_ceil  = float(np.median([r["z_ceiling"] for r in rooms]))
    log(f"{len(V):,} verts / {len(tri):,} tris; z_floor={z_floor:.3f} z_ceil={z_ceil:.3f}")
    return mesh, V, tri, tn, tc, [w for w in d["walls"] if w.get("status") == "ok"], z_floor, z_ceil


def cluster_walls(walls):
    for w in walls:
        c = np.array(w["center"]); n = np.array(w["normal"])
        w["_off"] = float(n @ c)
        w["_ang"] = float(np.degrees(np.arctan2(n[1], n[0])) % 180)
    parent = list(range(len(walls)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(len(walls)):
        for j in range(i + 1, len(walls)):
            da = abs(walls[i]["_ang"] - walls[j]["_ang"]); da = min(da, 180 - da)
            if da < DUP_ANG and abs(walls[i]["_off"] - walls[j]["_off"]) < DUP_OFF:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(walls)):
        groups.setdefault(find(i), []).append(walls[i])
    return list(groups.values())


def submesh(mesh, V, tri, tri_idx):
    """Build a sub-mesh from a set of triangle indices (keeps whole triangles)."""
    t = tri[tri_idx]
    used = np.unique(t)
    remap = -np.ones(len(V), np.int64); remap[used] = np.arange(len(used))
    nm = o3d.geometry.TriangleMesh()
    nm.vertices = o3d.utility.Vector3dVector(V[used])
    nm.triangles = o3d.utility.Vector3iVector(remap[t])
    return nm


def segment(V, tri, tn, tc, groups, z_floor, z_ceil):
    Ntri = len(tri)
    zmid = (z_floor + z_ceil) / 2
    vertical = np.abs(tn[:, 2]) < VERT_NZ                  # wall-like surface
    flat = ~vertical
    label = np.full(Ntri, "", object)

    # slabs first (flat tris only)
    ceil_m = flat & ((tc[:, 2] > zmid + CEIL_FROM_MID) | (tc[:, 2] > z_ceil - CEIL_BAND))
    floor_m = flat & (tc[:, 2] < z_floor + FLOOR_BAND) & ~ceil_m
    label[ceil_m] = "ceiling"; label[floor_m] = "floor"

    # walls: vertical tris -> nearest plane within extent
    wall_of = np.full(Ntri, -1, np.int64)
    best = np.full(Ntri, 1e9)
    cand0 = vertical & (tc[:, 2] > z_floor + 0.02) & (tc[:, 2] < z_ceil - 0.02)
    reps = []
    for k, g in enumerate(groups):
        wsum = sum(w["N"] for w in g)
        c = sum(np.array(w["center"]) * w["N"] for w in g) / wsum
        n = sum(np.array(w["normal"]) * w["N"] for w in g) / wsum
        n = n / np.linalg.norm(n)
        dvec = np.array([-n[1], n[0]])
        tmin = min(w["tmin"] for w in g) - ALONG_MARG
        tmax = max(w["tmax"] for w in g) + ALONG_MARG
        reps.append((k, c, n, dvec, tmin, tmax, sorted({w["room"] for w in g}),
                     [w["wall"] for w in g]))
        perp = np.abs((tc[:, 0] - c[0]) * n[0] + (tc[:, 1] - c[1]) * n[1])
        along = (tc[:, 0] - c[0]) * dvec[0] + (tc[:, 1] - c[1]) * dvec[1]
        m = cand0 & (perp < HALF_THICK) & (along > tmin) & (along < tmax) & (perp < best)
        best[m] = perp[m]; wall_of[m] = k
    for k in range(len(groups)):
        label[wall_of == k] = f"wall_{k:02d}"

    return label, wall_of, reps, vertical


def extra_walls(tn, tc, label, vertical, z_floor, z_ceil):
    """Recover walls the skeleton never measured a plane for.

    The measured list is sparse (~37 planes), so plane-gating alone leaves the
    enclosure gappy. Here EVERY unclaimed vertical triangle is a wall candidate:
    detect planes in the residual by binning (heading, perpendicular offset),
    split each plane into runs across gaps, and keep runs that span most of the
    floor->ceiling height. Furniture is rejected by height span, not by distance
    to a measured plane.
    """
    room_h = z_ceil - z_floor
    free = np.where((label == "") & vertical &
                    (tc[:, 2] > z_floor + 0.03) & (tc[:, 2] < z_ceil - 0.03))[0]
    if free.size == 0:
        return []
    n2 = tn[free, :2]
    ln = np.linalg.norm(n2, axis=1)
    ok = ln > 1e-6
    free = free[ok]; n2 = n2[ok] / ln[ok, None]
    p = tc[free]
    ang = np.degrees(np.arctan2(n2[:, 1], n2[:, 0])) % 180.0     # heading, mod pi
    ar = np.radians(ang)
    nc = np.stack([np.cos(ar), np.sin(ar)], 1)                   # canonical normal
    off = p[:, 0] * nc[:, 0] + p[:, 1] * nc[:, 1]
    log(f"residual vertical tris: {free.size:,} -> plane detection")

    abin = np.floor(ang / EX_ANG).astype(int)
    obin = np.floor(off / EX_OFF).astype(int)
    keys, inv, cnt = np.unique(np.stack([abin, obin], 1), axis=0,
                               return_inverse=True, return_counts=True)
    avail = np.ones(free.size, bool)
    out = []; n_short = 0
    for bi in np.argsort(-cnt):
        if cnt[bi] < EX_MINTRIS:
            break
        seed = (inv == bi) & avail
        if seed.sum() < EX_MINTRIS:
            continue
        # refine the plane from the seed, then re-collect across bin boundaries
        a0 = float(np.median(ang[seed])); o0 = float(np.median(off[seed]))
        da = np.abs(ang - a0); da = np.minimum(da, 180 - da)
        # measure distance with the SEED plane's normal, not each triangle's own
        # (noisy) normal -- otherwise one wall shatters into interleaved objects
        r = np.radians(a0)
        ns = np.array([np.cos(r), np.sin(r)])
        dist = p[:, 0] * ns[0] + p[:, 1] * ns[1] - o0
        mem = np.where(avail & (da < EX_ANGW) & (np.abs(dist) < EX_COLL))[0]
        if mem.size < EX_MINTRIS:
            continue
        d = np.array([-ns[1], ns[0]])                            # along-wall dir
        along = p[mem, 0] * d[0] + p[mem, 1] * d[1]
        order = np.argsort(along)
        a_s = along[order]
        brk = np.where(np.diff(a_s) > EX_GAP)[0]
        starts = np.r_[0, brk + 1]; ends = np.r_[brk + 1, len(a_s)]
        for s, e in zip(starts, ends):
            run = mem[order[s:e]]
            if run.size < EX_MINTRIS:
                continue
            length = a_s[e - 1] - a_s[s]
            z = p[run, 2]
            zspan = z.max() - z.min()
            if length < EX_MINLEN or zspan < EX_ZFRAC * room_h:
                continue
            if z.max() < z_ceil - EX_TOP:      # furniture stops short of the ceiling
                n_short += 1
                continue
            avail[run] = False
            name = f"wall_x{len(out):02d}"
            label[free[run]] = name
            out.append(dict(name=name, tris=free[run], ntris=int(run.size),
                            length_m=round(float(length), 2),
                            zspan_m=round(float(zspan), 2),
                            heading_deg=round(a0, 1)))
    log(f"recovered {len(out)} unmeasured walls "
        f"({sum(o['ntris'] for o in out):,} tris); "
        f"{n_short} tall runs rejected for not reaching the ceiling")
    return out


def split_ceiling(tc, label, z_floor):
    """Break the single ceiling slab into clean, separately-measurable parts.

    One merged ceiling object gives no usable height reading -- this flat has a
    2.70 m main ceiling and 2.16 m dropped wet-room ceilings, plus down-stand
    beams, all averaged together. Here the slab is gridded in (x,y), the cell
    heights are clustered into discrete LEVELS, and each connected region of a
    level becomes its own object carrying one height number.
    """
    ci = np.where(label == "ceiling")[0]
    if ci.size == 0:
        return []
    p = tc[ci]
    gx = np.floor(p[:, 0] / CEIL_CELL).astype(int)
    gy = np.floor(p[:, 1] / CEIL_CELL).astype(int)
    key = gx.astype(np.int64) * 100000 + gy
    order = np.argsort(key)
    k_s = key[order]
    bounds = np.r_[0, np.flatnonzero(np.diff(k_s)) + 1, len(k_s)]
    cell_ij, cell_z, cell_tris = [], [], []
    for s, e in zip(bounds[:-1], bounds[1:]):
        sl = order[s:e]
        cell_ij.append((int(k_s[s] // 100000), int(k_s[s] % 100000)))
        cell_z.append(float(np.max(p[sl, 2])))          # underside of the slab
        cell_tris.append(ci[sl])
    cell_ij = np.array(cell_ij); cell_z = np.array(cell_z)

    # Grow FLAT PLATEAUS: a neighbouring cell joins only if it is at essentially
    # the same height. Global height clustering does not work here -- beams and
    # slopes make the height histogram continuous, with no gaps to cut on -- but
    # plateaus are separated by sharp vertical steps, which region growing finds.
    lut = {tuple(c): i for i, c in enumerate(cell_ij)}
    seen = set()
    out = []
    for c0 in lut:
        if c0 in seen:
            continue
        seen.add(c0)
        stack = [c0]; comp = [c0]
        while stack:
            x = stack.pop()
            zx = cell_z[lut[x]]
            i, j = x
            for nb in ((i+1, j), (i-1, j), (i, j+1), (i, j-1)):
                if nb in seen or nb not in lut:
                    continue
                if abs(cell_z[lut[nb]] - zx) < CEIL_STEP:
                    seen.add(nb); stack.append(nb); comp.append(nb)
        if len(comp) < CEIL_MINCELL:
            continue
        ii = np.array([lut[c] for c in comp])
        tris = np.concatenate([cell_tris[i] for i in ii])
        zc = float(np.median(cell_z[ii]))
        name = f"ceiling_{len(out):02d}"
        label[tris] = name
        out.append(dict(name=name, tris=tris, ntris=int(tris.size),
                        z=round(zc, 4),
                        height_mm=round((zc - z_floor) * 1000, 1),
                        area_m2=round(len(comp) * CEIL_CELL ** 2, 2),
                        flatness_mm=round(float(np.std(cell_z[ii]) * 1000), 1)))
    out.sort(key=lambda c: -c["area_m2"])
    for c in out:
        log(f"{c['name']}: h={c['height_mm']:.0f} mm  area={c['area_m2']:.1f} m2  "
            f"flat±{c['flatness_mm']:.0f} mm  {c['ntris']:,} tris")
    return out


def columns(V, tri, tc, label, z_floor, z_ceil):
    room_h = z_ceil - z_floor
    free = np.where((label == "") & (tc[:, 2] > z_floor + 0.1) & (tc[:, 2] < z_ceil - 0.1))[0]
    if free.size == 0:
        return []
    p = tc[free]
    gx = np.floor(p[:, 0] / COL_CELL).astype(int)
    gy = np.floor(p[:, 1] / COL_CELL).astype(int)
    from collections import defaultdict
    cz = defaultdict(list)
    for i, (a, b) in enumerate(zip(gx, gy)):
        cz[(a, b)].append(p[i, 2])
    tall = {c for c, zs in cz.items() if (max(zs) - min(zs)) > COL_ZSPAN * room_h and len(zs) > 25}
    seen = set(); comps = []
    for c in tall:
        if c in seen:
            continue
        st = [c]; comp = []
        while st:
            x = st.pop()
            if x in seen or x not in tall:
                continue
            seen.add(x); comp.append(x)
            st += [(x[0]+dx, x[1]+dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
        comps.append(comp)
    cols = []
    for ci, comp in enumerate(comps):
        cs = set(comp)
        sel = free[[(a, b) in cs for a, b in zip(gx, gy)]]
        if sel.size < 150:
            continue
        pp = tc[sel]; foot = pp[:, :2].max(0) - pp[:, :2].min(0)
        if max(foot) > COL_MAXFOOT:
            continue
        # a real column runs floor->ceiling and is near-square in plan;
        # furniture fragments fail one or the other.
        if pp[:, 2].min() > z_floor + COL_TOUCH or pp[:, 2].max() < z_ceil - COL_TOUCH:
            continue
        if max(foot) > COL_ASPECT * max(min(foot), 1e-3):
            continue
        label[sel] = f"column_{ci:02d}"
        cols.append(dict(name=f"column_{ci:02d}", tris=sel,
                         footprint=[round(float(foot[0]), 2), round(float(foot[1]), 2)],
                         ntris=int(sel.size)))
        log(f"column_{ci:02d}: {sel.size:,} tris foot={foot.round(2)}")
    return cols


def main(poisson, mj, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    mesh, V, tri, tn, tc, walls, z_floor, z_ceil = load_frame(poisson, mj)
    groups = cluster_walls(walls)
    log(f"{len(walls)} ok walls -> {len(groups)} physical walls (deduped)")
    label, wall_of, reps, vertical = segment(V, tri, tn, tc, groups, z_floor, z_ceil)
    extras = extra_walls(tn, tc, label, vertical, z_floor, z_ceil)
    ceilparts = split_ceiling(tc, label, z_floor)
    cols = columns(V, tri, tc, label, z_floor, z_ceil)

    # build pieces
    pieces = []
    wall_meta = []
    for (k, c, n, dvec, tmin, tmax, rooms, members) in reps:
        idx = np.where(label == f"wall_{k:02d}")[0]
        pieces.append((f"wall_{k:02d}", idx))
        wall_meta.append(dict(name=f"wall_{k:02d}", rooms=rooms, members=members,
                              length_m=round(float(tmax - tmin), 3), ntris=int(idx.size),
                              conf="low" if idx.size < MIN_TRIS else "ok"))
        log(f"wall_{k:02d}: {idx.size:,} tris rooms={rooms} "
            f"{'LOW' if idx.size < MIN_TRIS else ''}")
    for ex in extras:
        pieces.append((ex["name"], ex["tris"]))
        wall_meta.append(dict(name=ex["name"], rooms=[], members=[], source="recovered",
                              length_m=ex["length_m"], ntris=ex["ntris"],
                              conf="low" if ex["ntris"] < MIN_TRIS else "ok"))
    fi = np.where(label == "floor")[0]
    pieces.append(("floor", fi))
    for cp in ceilparts:
        pieces.append((cp["name"], cp["tris"]))
    ci = np.where(label == "ceiling")[0]          # unclustered remainder
    if ci.size:
        pieces.append(("ceiling_misc", ci))
    ceil_t = sum(c["ntris"] for c in ceilparts) + int(ci.size)
    log(f"floor {fi.size:,} tris   ceiling {ceil_t:,} tris in "
        f"{len(ceilparts)} parts (+{ci.size:,} unclustered)")
    for cdef in cols:
        pieces.append((cdef["name"], cdef["tris"]))

    # write named OBJ + glb
    obj_path = out / "detailed_modular.obj"
    voff = 0
    subs = []
    with open(obj_path, "w") as fh:
        fh.write("# detailed modular house (Poisson segmented, complete walls)\n")
        for name, idx in pieces:
            if idx.size == 0:
                fh.write(f"o {name}\n"); subs.append((name, None)); continue
            sm = submesh(mesh, V, tri, idx)
            subs.append((name, sm))
            vv = np.asarray(sm.vertices); tt = np.asarray(sm.triangles)
            fh.write(f"o {name}\n")
            for p in vv:
                fh.write(f"v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")
            for t in tt:
                fh.write(f"f {t[0]+voff+1} {t[1]+voff+1} {t[2]+voff+1}\n")
            voff += len(vv)
    log(f"wrote {obj_path} ({voff:,} verts, {len(pieces)} objects)")
    try:
        import trimesh
        scene = trimesh.Scene()
        for name, sm in subs:
            if sm is None:
                continue
            tmsh = trimesh.Trimesh(np.asarray(sm.vertices), np.asarray(sm.triangles), process=False)
            if len(tmsh.faces):
                scene.add_geometry(tmsh, geom_name=name, node_name=name)
        scene.export(str(out / "detailed_modular.glb"))
        log(f"wrote {out/'detailed_modular.glb'}")
    except Exception as e:
        log(f"glb skipped: {e}")

    tot = len(tri)
    wall_t = int((wall_of >= 0).sum())
    extra_t = sum(e["ntris"] for e in extras)
    col_t = sum(c["ntris"] for c in cols)
    disc = int((label == "").sum())
    manifest = dict(
        objects=wall_meta +
                [dict(name="floor", ntris=int(fi.size))] +
                [{k: c[k] for k in ("name", "ntris", "z", "height_mm", "area_m2",
                                    "flatness_mm")} for c in ceilparts] +
                [{k: c[k] for k in ("name", "footprint", "ntris")} for c in cols],
        rooms={r: [w["name"] for w in wall_meta if r in w["rooms"]]
               for r in sorted({r for w in wall_meta for r in w["rooms"]})},
        coverage=dict(total_tris=tot, wall_tris=wall_t, recovered_wall_tris=extra_t,
                      n_recovered_walls=len(extras), floor_tris=int(fi.size),
                      ceiling_tris=ceil_t, n_ceiling_parts=len(ceilparts), column_tris=col_t, discarded_tris=disc,
                      structural_frac=round((tot - disc) / tot, 3)))
    json.dump(manifest, open(out / "modular_manifest.json", "w"), indent=1)
    c = manifest["coverage"]
    log(f"COVERAGE structural={c['structural_frac']*100:.1f}%  walls={wall_t:,} "
        f"+recovered={extra_t:,} ({len(extras)} walls) floor={fi.size:,} "
        f"ceil={ceil_t:,}({len(ceilparts)} parts) col={col_t:,} discarded(furniture)={disc:,}")
    log("done")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
