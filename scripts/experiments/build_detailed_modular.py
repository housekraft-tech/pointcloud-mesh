"""build_detailed_modular.py
---------------------------
Approach A (spec 2026-07-21): carve the Poisson mesh into a MODULAR house using
the measured wall planes + room tags as cutters. One assembled house, each
physical wall its own detailed object (carrying the real Poisson relief), a
single shared floor and a single shared ceiling (beams), free-standing columns
rescued, furniture dropped. Digital / on-screen (open shells OK).

Loads the 716 MB Poisson once; every component below is a function over the
shared Z-up mesh + measurements.json scaffold.

Usage:
  venv311\\Scripts\\python.exe scripts\\experiments\\build_detailed_modular.py \\
      <poisson.obj> <measurements.json> <out_dir>
"""
import sys, json, time
from pathlib import Path
import numpy as np
import open3d as o3d

# ---- config (metres) ----
HALF_THICK = 0.11     # half wall band around plane (grabs both faces of ~110mm wall)
ALONG_MARG = 0.15     # extend past wall ends
SKIRT      = 0.03     # trim floor return from wall body
CROWN      = 0.03     # trim ceiling return from wall body
FLOOR_BAND = 0.08     # floor slab thickness above z_floor
CEIL_BAND  = 0.30     # ceiling slab depth below z_ceiling (deep -> keeps beams)
DUP_ANG    = 6.0      # deg: merge walls within this heading
DUP_OFF    = 0.15     # m: merge walls within this perpendicular offset
COL_CELL   = 0.10     # column-detect grid cell
COL_ZSPAN  = 0.60     # frac of room height a column must span
COL_MAXFOOT= 1.20     # m: max column footprint dimension
MIN_VERTS  = 500      # below this a wall crop is flagged 'low'


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------- 1. frame ----------
def load_frame(poisson, mj):
    log(f"loading {poisson} ...")
    mesh = o3d.io.read_triangle_mesh(str(poisson))
    mesh.compute_vertex_normals()
    v = np.asarray(mesh.vertices)
    if np.argmin(v.max(0) - v.min(0)) == 1:                # Y-up -> Z-up
        Rz = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
        mesh.rotate(Rz, center=(0, 0, 0))
    v = np.asarray(mesh.vertices)
    d = json.load(open(mj))
    rooms = d["rooms"]
    z_floor = float(np.median([r["z_floor"] for r in rooms]))
    z_ceil  = float(np.median([r["z_ceiling"] for r in rooms]))
    log(f"{len(v):,} verts; z_floor={z_floor:.3f} z_ceil={z_ceil:.3f}")
    return mesh, v, [w for w in d["walls"] if w.get("status") == "ok"], z_floor, z_ceil


# ---------- 2. de-duplicate wall planes (union-find) ----------
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


# ---------- helpers ----------
def append_obj(fh, name, mesh, voff):
    v = np.asarray(mesh.vertices); tri = np.asarray(mesh.triangles)
    fh.write(f"o {name}\n")
    for p in v:
        fh.write(f"v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")
    for t in tri:
        fh.write(f"f {t[0]+voff+1} {t[1]+voff+1} {t[2]+voff+1}\n")
    return voff + len(v)


# ---------- 3. per-wall slabs ----------
def wall_slabs(mesh, v, groups, z_floor, z_ceil):
    zlo, zhi = z_floor + SKIRT, z_ceil - CROWN
    objs = []; claimed = np.zeros(len(v), bool)
    for k, g in enumerate(groups):
        # representative plane = mean of members (weighted by fit count)
        wsum = sum(w["N"] for w in g)
        c = sum(np.array(w["center"]) * w["N"] for w in g) / wsum
        n = sum(np.array(w["normal"]) * w["N"] for w in g) / wsum
        n = n / np.linalg.norm(n)
        dvec = np.array([-n[1], n[0]])
        along = (v[:, 0] - c[0]) * dvec[0] + (v[:, 1] - c[1]) * dvec[1]
        perp  = (v[:, 0] - c[0]) * n[0] + (v[:, 1] - c[1]) * n[1]
        tmin = min(w["tmin"] for w in g) - ALONG_MARG
        tmax = max(w["tmax"] for w in g) + ALONG_MARG
        m = (along > tmin) & (along < tmax) & (np.abs(perp) < HALF_THICK) & \
            (v[:, 2] > zlo) & (v[:, 2] < zhi)
        idx = np.where(m)[0]
        rooms = sorted({w["room"] for w in g})
        name = f"wall_{k:02d}"
        conf = "low" if idx.size < MIN_VERTS else "ok"
        claimed[idx] = True
        objs.append(dict(name=name, idx=idx, rooms=rooms, conf=conf,
                         members=[w["wall"] for w in g],
                         length_m=round(float(tmax - tmin), 3), nverts=int(idx.size)))
        log(f"{name}: {idx.size:,} v  rooms={rooms} conf={conf}")
    return objs, claimed


# ---------- 4. shared floor + ceiling ----------
def shared_slabs(v, z_floor, z_ceil):
    floor = np.where(v[:, 2] < z_floor + FLOOR_BAND)[0]
    ceil  = np.where(v[:, 2] > z_ceil - CEIL_BAND)[0]
    log(f"floor: {floor.size:,} v   ceiling: {ceil.size:,} v (band {CEIL_BAND} m -> beams)")
    return floor, ceil


# ---------- 5. free-standing columns from unclaimed verts ----------
def columns(v, claimed, z_floor, z_ceil):
    room_h = z_ceil - z_floor
    free = np.where(~claimed & (v[:, 2] > z_floor + 0.1) & (v[:, 2] < z_ceil - 0.1))[0]
    if free.size == 0:
        return [], free
    p = v[free]
    gx = np.floor(p[:, 0] / COL_CELL).astype(int)
    gy = np.floor(p[:, 1] / COL_CELL).astype(int)
    from collections import defaultdict
    cell_z = defaultdict(list)
    for i, (a, b) in enumerate(zip(gx, gy)):
        cell_z[(a, b)].append(p[i, 2])
    tall = {c for c, zs in cell_z.items()
            if (max(zs) - min(zs)) > COL_ZSPAN * room_h and len(zs) > 30}
    # connected components on tall cells
    seen = set(); comps = []
    for c in tall:
        if c in seen:
            continue
        stack = [c]; comp = []
        while stack:
            x = stack.pop()
            if x in seen or x not in tall:
                continue
            seen.add(x); comp.append(x)
            stack += [(x[0]+dx, x[1]+dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
        comps.append(comp)
    cols = []
    for ci, comp in enumerate(comps):
        cs = set(comp)
        sel = free[[(a, b) in cs for a, b in zip(gx, gy)]]
        if sel.size < 200:
            continue
        pp = v[sel]
        foot = pp[:, :2].max(0) - pp[:, :2].min(0)
        if max(foot) > COL_MAXFOOT:            # too wide -> not a column (missed wall/big furniture)
            continue
        cols.append(dict(name=f"column_{ci:02d}", idx=sel,
                         footprint=[round(float(foot[0]), 2), round(float(foot[1]), 2)],
                         nverts=int(sel.size)))
        log(f"column_{ci:02d}: {sel.size:,} v foot={foot.round(2)}")
    return cols, free


# ---------- 6. assemble ----------
def assemble(mesh, wall_objs, floor_idx, ceil_idx, col_objs, out, claimed, free):
    out.mkdir(parents=True, exist_ok=True)
    pieces = []
    for w in wall_objs:
        pieces.append((w["name"], mesh.select_by_index(w["idx"])))
    pieces.append(("floor", mesh.select_by_index(floor_idx)))
    pieces.append(("ceiling", mesh.select_by_index(ceil_idx)))
    for c in col_objs:
        pieces.append((c["name"], mesh.select_by_index(c["idx"])))

    obj_path = out / "detailed_modular.obj"
    voff = 0
    with open(obj_path, "w") as fh:
        fh.write("# detailed modular house (Poisson relief + skeleton structure)\n")
        for name, m in pieces:
            m.compute_vertex_normals()
            voff = append_obj(fh, name, m, voff)
    log(f"wrote {obj_path}  ({voff:,} verts, {len(pieces)} objects)")

    # glb via trimesh scene (named nodes) -- optional/best-effort
    try:
        import trimesh
        scene = trimesh.Scene()
        for name, m in pieces:
            tm = trimesh.Trimesh(np.asarray(m.vertices), np.asarray(m.triangles),
                                 process=False)
            if len(tm.faces):
                scene.add_geometry(tm, geom_name=name, node_name=name)
        scene.export(str(out / "detailed_modular.glb"))
        log(f"wrote {out/'detailed_modular.glb'}")
    except Exception as e:
        log(f"glb skipped: {e}")

    total = len(claimed)
    struct_claimed = int(claimed.sum()) + int(floor_idx.size) + int(ceil_idx.size)
    col_v = sum(c["nverts"] for c in col_objs)
    manifest = dict(
        objects=[{k: w[k] for k in ("name", "rooms", "conf", "length_m", "nverts", "members")}
                 for w in wall_objs] +
                [dict(name="floor", nverts=int(floor_idx.size)),
                 dict(name="ceiling", nverts=int(ceil_idx.size))] +
                [{k: c[k] for k in ("name", "footprint", "nverts")} for c in col_objs],
        rooms={r: [w["name"] for w in wall_objs if r in w["rooms"]]
               for r in sorted({r for w in wall_objs for r in w["rooms"]})},
        coverage=dict(total_verts=total,
                      wall_verts=int(claimed.sum()),
                      floor_verts=int(floor_idx.size),
                      ceiling_verts=int(ceil_idx.size),
                      column_verts=col_v,
                      discarded_verts=int(total - struct_claimed - col_v),
                      structural_frac=round((struct_claimed + col_v) / total, 3)),
        config=dict(HALF_THICK=HALF_THICK, CEIL_BAND=CEIL_BAND, FLOOR_BAND=FLOOR_BAND,
                    SKIRT=SKIRT, CROWN=CROWN, DUP_ANG=DUP_ANG, DUP_OFF=DUP_OFF))
    json.dump(manifest, open(out / "modular_manifest.json", "w"), indent=1)
    log(f"wrote {out/'modular_manifest.json'}")
    cov = manifest["coverage"]
    log(f"COVERAGE structural={cov['structural_frac']*100:.1f}%  "
        f"walls={cov['wall_verts']:,} floor={cov['floor_verts']:,} "
        f"ceil={cov['ceiling_verts']:,} col={cov['column_verts']:,} "
        f"discarded(furniture)={cov['discarded_verts']:,}")


def main(poisson, mj, out_dir):
    out = Path(out_dir)
    mesh, v, walls, z_floor, z_ceil = load_frame(poisson, mj)
    groups = cluster_walls(walls)
    log(f"{len(walls)} ok walls -> {len(groups)} physical walls (deduped)")
    wall_objs, claimed = wall_slabs(mesh, v, groups, z_floor, z_ceil)
    floor_idx, ceil_idx = shared_slabs(v, z_floor, z_ceil)
    claimed_fc = claimed.copy(); claimed_fc[floor_idx] = True; claimed_fc[ceil_idx] = True
    col_objs, free = columns(v, claimed_fc, z_floor, z_ceil)
    assemble(mesh, wall_objs, floor_idx, ceil_idx, col_objs, out, claimed, free)
    log("done")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
