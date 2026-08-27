"""The clean model: sharp boxes that follow the measured thickness, cut to fit.

The Poisson model is the scanned surface -- honest, and 25 million triangles of
slightly-wobbly everything. This is the other end: sharp-edged boxes, a few
thousand triangles, every dimension traceable to a measurement. Three things
make it a building rather than a pile of blocks.

**A wall is a wall, plus what sticks out of it and what is cut into it.** The
relief is real -- pilasters, columns cast into the wall, boxed conduits, beam
soffits, niches -- and it happens over a PATCH, not over the whole height, so a
profile taken only along the length flattens all of it.

But giving every patch its own box is worse: the wall stops being a wall. Tried
that way, one storey came out as 523 disconnected blocks that no longer read as
a network, which is the opposite of a clean model.

So each wall gets ONE core box at the thickness it holds over most of its face,
spanning its full length and height -- that is the network, and it is what makes
the model coherent. The face is then gridded (100 mm along, 250 mm up), and only
where it departs from the core by more than 25 mm does anything else happen: a
patch standing proud becomes an added box, a patch sunk in becomes a subtraction.
A plain wall is one box. A wall with a pilaster and a niche is one box, plus one,
minus one.

**Parts must meet, not interpenetrate.** Measured independently, two walls at a
corner can each stop short and leave a slot of daylight. Every box is therefore
grown 120 mm along its length first, so contact is guaranteed -- and then the
overlaps are cut away, in priority order, so each part keeps only the volume
that is its own. A wall meeting another wall comes out L-shaped; a wall meeting
a slab stops at the slab's face. Nothing is missed and nothing is counted twice,
which is checked at the end: the parts' volumes must sum to the union's.

**Openings go all the way through.** Cutters are made oversize across the wall
and exact along it, so a millimetre of disagreement cannot leave a film of
geometry across a doorway.
"""
import sys, json, argparse, time
from pathlib import Path
import numpy as np

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

GROW = 0.12           # m: how far a box reaches into what it abuts, before cutting
SLAB = 0.15           # m: assumed depth of a floor or ceiling slab
CUT_OVER = 0.20       # m: how far an opening cutter overshoots the wall faces
MIN_LEN = 0.25        # m: ignore parts shorter than this
STATION = 0.10        # m: spacing of the thickness profile along a wall
QUANT = 0.020         # m: thickness steps below this are not steps
MIN_CELLS = 6         # a rectangle smaller than this is scan noise
RUN_STEP = 0.030      # m: a face that steps this far along the wall is a new run
RUN_MIN  = 0.40       # m: ...but only if it holds for this far, or it is a saw
SNAP     = 0.005      # m: face positions are quoted to this, so runs come out flush
RELIEF = 0.040        # m: a face this far off the core is relief, not the wall.
                      # The Poisson surface wobbles by 10-20 mm, so a lower bar
                      # turns scan noise into a field of little fins.
MIN_SIDE = 2          # cells: relief must be at least this in BOTH directions,
                      # or a one-cell-wide fin at a wall's end becomes a pilaster
FILL_HOLE = 10        # cells: a gap this small is missing data, not an opening
# Who keeps the material where two boxes meet. The masonry wins: a slab is
# poured BETWEEN the walls, so the wall keeps its full thickness up to the
# ceiling and the slab is trimmed back to the wall face. The other way round
# -- which is what this did -- a floor grown 120 mm into the room cut a notch
# out of every wall it touched, and that notch was most of the wall error.
PRIORITY = ["column", "wall", "parapet", "beam", "floor", "ceiling", "dropped_ceiling"]


CLAMP = [None, None]      # set from --z0/--z1: the storey's own band


def settle(P, W, ax, face, win=0.020, nmin=200):
    """Put a face where the surface's own mean puts it, and say how well it is known.

    Every face so far has come from a histogram peak or a median of per-cell
    percentiles. Both are quantised -- to the 5 mm bin, to the cell -- and both
    throw away the one thing that beats noise: averaging. The scan's noise on a
    flat wall is 1.74 mm RMS per point, so the MEAN of n points on that wall is
    known to 1.74/sqrt(n) mm. A face with 20,000 triangles on it is pinned to
    about a hundredth of a millimetre, and what is left is not noise but the
    question of whether the wall is really flat.

    Returns the settled coordinate and its standard error in mm.
    """
    d = P[:, ax] - face
    m = np.abs(d) < win
    if m.sum() < nmin:
        return face, None
    w = W[m]
    q = P[m, ax]
    mu = float(np.average(q, weights=w))
    var = float(np.average((q-mu)**2, weights=w))
    # effective sample size for a weighted mean
    neff = (w.sum()**2)/np.sum(w**2)
    return mu, float(np.sqrt(max(var, 0)/max(neff, 1))*1000)


def faces_of(P, ax, hint=None, lo_hi=None, span=(0.05, 0.65)):
    """The two faces of a part, as the two peaks of where its surface sits.

    min/max is the wrong tool here. The seam pass deliberately grows a part into
    the fillets and returns around it, so its extreme triangles reach 200-350 mm
    past the face -- and a box drawn to those extremes swallows the very surface
    it was measured from (a column came out 230 mm fat, and 99% of its own scan
    ended up inside it).

    Surface, though, piles up ON the faces. So the faces are the two tallest
    peaks of the histogram across the part, far enough apart to be two sides of
    something. If only one peak stands out the part was seen from one side, and
    the thickness comes from the hint.
    """
    q = P[:, ax]
    if q.size < 30:
        return lo_hi
    e = np.arange(q.min(), q.max()+0.005, 0.005)
    if e.size < 3:
        return (float(q.min()), float(q.max()))
    h, _ = np.histogram(q, bins=e)
    c = 0.5*(e[:-1]+e[1:])
    i1 = int(np.argmax(h))
    far = np.abs(c - c[i1])
    ok = (far > span[0]) & (far < span[1])
    if ok.any() and h[ok].max() > 0.25*h[i1]:
        i2 = int(np.flatnonzero(ok)[np.argmax(h[ok])])
        return (float(min(c[i1], c[i2])), float(max(c[i1], c[i2])))
    if hint:
        # one face seen: put the other one a measured thickness behind it, on
        # the side the modular stage says the material lies
        t = hint/1000.0
        if lo_hi and abs(lo_hi[1]-c[i1]) > abs(lo_hi[0]-c[i1]):
            return (float(c[i1]), float(c[i1]+t))
        return (float(c[i1]-t), float(c[i1]))
    return lo_hi


SCHEDULE = []             # every primitive, so SketchUp can draw it natively


def sched(name, kind, op, lo, hi):
    """Record a box as numbers, not triangles.

    Everything here is an axis-aligned box before it is anything else -- a core
    wall, a pilaster added to it, a niche or a doorway cut out of it. Turning
    those into triangles and handing the triangles to SketchUp is what makes a
    wall arrive as a hundred little faces. A designer would draw the rectangle
    and push/pull it, and SketchUp can be told to do exactly that -- but only
    if it is given the rectangle.
    """
    SCHEDULE.append(dict(name=name, kind=kind, op=op,
                         lo=[round(float(v), 5) for v in lo],
                         hi=[round(float(v), 5) for v in hi]))


def box(lo, hi, slab=False):
    """An axis-aligned box, clipped to the storey it belongs to.

    Boxes are grown to guarantee contact, which for a single flat is harmless
    but in a three-storey building pushes each floor 120 mm into the one above.
    Clipping here rather than after the booleans keeps every box exact: these
    are axis-aligned, so a clamp is a min and a max, not a cut.
    """
    import trimesh
    lo = np.asarray(lo, float).copy(); hi = np.asarray(hi, float).copy()
    # A slab is the thing that CLOSES the storey, so it lives in the band from
    # the ceiling surface to the top of the structure; everything else stops at
    # the ceiling. Clamping a slab to the ceiling plane the way a wall is
    # clamped flattens it to nothing and leaves the storey open.
    pad = SLAB if slab else 0.0
    if CLAMP[0] is not None:
        lo[2] = max(lo[2], CLAMP[0]-pad); hi[2] = max(hi[2], CLAMP[0]-pad)
    if CLAMP[1] is not None:
        lo[2] = min(lo[2], CLAMP[1]+pad); hi[2] = min(hi[2], CLAMP[1]+pad)
    m = trimesh.creation.box(extents=np.maximum(hi-lo, 1e-4))
    m.apply_translation((lo+hi)/2)
    return m


def face_grid(P, ax, s0, s1, z0, z1, ucell=0.10, vcell=0.25):
    """Both faces of a wall, per cell of its own surface."""
    from scipy import ndimage
    u = P[:, 1-ax]; v = P[:, 2]; w = P[:, ax]
    nu = max(1, int(np.ceil((s1-s0)/ucell)))
    nv = max(1, int(np.ceil((z1-z0)/vcell)))
    iu = np.clip(((u-s0)/ucell).astype(int), 0, nu-1)
    iv = np.clip(((v-z0)/vcell).astype(int), 0, nv-1)
    flat = iu*nv + iv
    lo = np.full(nu*nv, np.nan); hi = np.full(nu*nv, np.nan)
    o = np.argsort(flat); fs = flat[o]; ws = w[o]
    b = np.r_[0, np.flatnonzero(np.diff(fs))+1, len(fs)]
    for a0, b0 in zip(b[:-1], b[1:]):
        if b0-a0 >= 8:
            lo[fs[a0]] = np.percentile(ws[a0:b0], 4)
            hi[fs[a0]] = np.percentile(ws[a0:b0], 96)
    lo = lo.reshape(nu, nv); hi = hi.reshape(nu, nv)
    if np.isfinite(lo).sum() < 4:
        return None
    return dict(lo=lo, hi=hi, s0=s0, z0=z0, ucell=ucell, vcell=vcell)


def rectangles(mask, val_lo, val_hi):
    """Greedy rectangles over a mask, each carrying its own median faces."""
    todo = mask.copy()
    nu, nv = mask.shape
    out = []
    for i in range(nu):
        j = 0
        while j < nv:
            if not todo[i, j]:
                j += 1; continue
            j1 = j
            while j1+1 < nv and todo[i, j1+1]:
                j1 += 1
            i1 = i
            while i1+1 < nu and todo[i1+1, j:j1+1].all():
                i1 += 1
            todo[i:i1+1, j:j1+1] = False
            if ((i1-i+1) >= MIN_SIDE and (j1-j+1) >= MIN_SIDE
                    and (i1-i+1)*(j1-j+1) >= MIN_CELLS):
                out.append((i, i1+1, j, j1+1,
                            float(np.nanmedian(val_lo[i:i1+1, j:j1+1])),
                            float(np.nanmedian(val_hi[i:i1+1, j:j1+1]))))
            j = j1+1
    return out


def relief(g):
    """The core thickness, and the patches that stand off it either way.

    The core is the median face position over the whole wall, so it is the
    thickness the wall holds over most of itself rather than the thickest place
    on it. Everything within RELIEF of that is the wall; the rest is a pilaster
    to add or a niche to subtract.
    """
    lo, hi = g["lo"], g["hi"]
    core_lo = float(np.nanmedian(lo)); core_hi = float(np.nanmedian(hi))
    have = np.isfinite(lo)
    # A long wall does not hold one thickness end to end. A run that changes
    # plane by more than RUN_STEP is a different piece of masonry -- a 12.8 m
    # perimeter wall taken as one core sat 20 mm off along most of its length.
    # Each run keeps its own two faces; the pieces abut, so the wall is still
    # one solid, and relief is then measured against the run it belongs to.
    with np.errstate(all="ignore"):
        cl = np.nanmedian(lo, axis=1); ch = np.nanmedian(hi, axis=1)
    runs, start = [], 0
    for i in range(1, len(cl)+1):
        end = i == len(cl)
        step = False
        if not end:
            a = np.nanmedian(cl[start:i]); b = np.nanmedian(ch[start:i])
            if np.isfinite(cl[i]) and np.isfinite(a):
                step = abs(cl[i]-a) > RUN_STEP or abs(ch[i]-b) > RUN_STEP
        if end or step:
            seg_lo = np.nanmedian(cl[start:i]); seg_hi = np.nanmedian(ch[start:i])
            if np.isfinite(seg_lo) and np.isfinite(seg_hi):
                runs.append([start, i, float(seg_lo), float(seg_hi)])
            start = i
    # A step every cell is not a wall, it is a saw. A change of plane only
    # counts if the wall HOLDS it: runs shorter than RUN_MIN are absorbed into
    # whichever neighbour they are closer to, and what survives is snapped to
    # SNAP so two runs that agree to a few millimetres come out flush.
    nmin = max(1, int(round(RUN_MIN/g["ucell"])))
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for i, r in enumerate(runs):
            if r[1]-r[0] >= nmin:
                continue
            left = runs[i-1] if i > 0 else None
            right = runs[i+1] if i+1 < len(runs) else None
            pick = left if right is None else right if left is None else (
                left if abs(left[2]-r[2])+abs(left[3]-r[3])
                     <= abs(right[2]-r[2])+abs(right[3]-r[3]) else right)
            pick[0] = min(pick[0], r[0]); pick[1] = max(pick[1], r[1])
            runs.pop(i); changed = True
            break
    for r in runs:
        r[2] = round(r[2]/SNAP)*SNAP; r[3] = round(r[3]/SNAP)*SNAP
    # neighbours that ended up on the same planes are one run again
    merged = [runs[0]]
    for r in runs[1:]:
        if abs(r[2]-merged[-1][2]) < 1e-9 and abs(r[3]-merged[-1][3]) < 1e-9:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    runs = [tuple(r) for r in merged]
    if not runs:
        runs = [(0, len(cl), core_lo, core_hi)]
    g["runs"] = runs
    # Relief has to be INTERIOR to the face. The seam pass grows a wall into the
    # ceiling above it, the floor below it and the walls it meets, so the border
    # cells always read thicker than the wall -- and turned into a rail running
    # the full length of every wall, with a fin at each end. Those rows are
    # still part of the wall; they are just no longer allowed to define relief.
    inner = np.zeros_like(have)
    inner[1:-1, 1:-1] = True
    if inner.sum() < MIN_CELLS:
        inner = have.copy()
    have = have & inner

    def rects(mask, a, b):
        out = []
        for (i0, i1, j0, j1, vlo, vhi) in rectangles(mask & have, a, b):
            out.append(dict(u0=g["s0"]+i0*g["ucell"], u1=g["s0"]+i1*g["ucell"],
                            v0=g["z0"]+j0*g["vcell"], v1=g["z0"]+j1*g["vcell"],
                            lo=vlo, hi=vhi))
        return out

    # the core each cell belongs to
    CL = np.full_like(lo, np.nan); CH = np.full_like(hi, np.nan)
    for (i0, i1, a, b) in g["runs"]:
        CL[i0:i1, :] = a; CH[i0:i1, :] = b

    adds, subs = [], []
    for r in rects(have & (lo < CL - RELIEF), lo, CL):
        adds.append(r)                             # proud of the near face
    for r in rects(have & (hi > CH + RELIEF), CH, hi):
        adds.append(r)                             # proud of the far face
    for r in rects(have & (lo > CL + RELIEF), CL, lo):
        r["lo"] -= 0.01; subs.append(r)            # sunk into the near face
    for r in rects(have & (hi < CH - RELIEF), hi, CH):
        r["hi"] += 0.01; subs.append(r)            # sunk into the far face
    return core_lo, core_hi, adds, subs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--z0", type=float, default=None,
                    help="clamp every box to start here (model frame)")
    ap.add_argument("--z1", type=float, default=None,
                    help="clamp every box to stop here -- the storey's true "
                         "ceiling, so one floor cannot poke into the next")
    ap.add_argument("--no-cut", action="store_true",
                    help="leave the boxes overlapping instead of cutting them to fit")
    a = ap.parse_args()
    import trimesh
    CLAMP[0], CLAMP[1] = a.z0, a.z1
    d = Path(a.dir); out = Path(a.out) if a.out else d
    man = json.load(open(d/"manifest.json"))
    zf = man["floor_z"]
    # The storey's ceiling is the height at which its horizontal surface piles
    # up -- modal_ceiling_height_mm -- NOT structural_ceiling_z, which is the
    # slab above and sits with the next storey's floor, and NOT the splitter's
    # figure, which is a histogram of the whole building and came out 250 mm
    # short on two of the three storeys here.
    zc = zf + man["modal_ceiling_height_mm"]/1000.0

    T = np.load(a.cache)["T"].astype(np.int64)
    V = np.load(d/"verts.npy").astype(np.float64)
    L = np.load(d/"labels.npy")
    names = json.load(open(d/"names.json"))
    C = V[T].mean(axis=1)
    tri = V[T]
    nrm = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    area = np.linalg.norm(nrm, axis=1)/2.0
    flat = np.abs(nrm[:, 2])/(2*area + 1e-12) > 0.90     # a horizontal triangle
    pts, bounds, wts, lie = {}, {}, {}, {}
    for pid, nm in enumerate(names):
        m = L == pid
        if m.any():
            q = C[m]
            pts[nm] = q
            wts[nm] = area[m]
            lie[nm] = flat[m]
            bounds[nm] = (q.min(0), q.max(0))
    if a.z1 is not None:
        zc = min(zc, a.z1)
    log(f"{man['n_parts']} parts, floor {zf:.3f}, ceiling {zc:.3f}"
        + (f", clamped to {a.z0:.3f}..{a.z1:.3f}" if a.z1 is not None else ""))

    wof = {p["name"]: p for p in man["parts"]}
    cutters = {}
    for f in man["features"]:
        if f["kind"] not in ("door", "window", "arch", "opening"):
            continue
        w = wof.get(f["wall"])
        if not w or "across_m" not in w:
            continue
        ax = 0 if w["axis"] == "x" else 1
        c0, c1 = w["across_m"]; a0, a1 = f["along_m"]
        lo = [0, 0, zf + f["sill_mm"]/1000.0]; hi = [0, 0, zf + f["head_mm"]/1000.0]
        lo[ax], hi[ax] = c0 - CUT_OVER, c1 + CUT_OVER
        lo[1-ax], hi[1-ax] = a0, a1
        cutters.setdefault(f["wall"], []).append((lo, hi))

    # ---- one box per part, or per thickness run for a wall ------------------
    made, n_steps, n_cut, n_relief, n_flat, n_runs = [], 0, 0, 0, 0, 0
    stderr = []
    for p in man["parts"]:
        k, nm = p["kind"], p["name"]
        if k in ("wall", "parapet"):
            if p.get("length_mm", 0) < MIN_LEN*1000 or nm not in pts:
                continue
            ax = 0 if p["axis"] == "x" else 1
            s0, s1 = p["along_m"]
            z0 = min(zf + p["z_mm"][0]/1000.0, zf)
            z1 = zf + p["z_mm"][1]/1000.0
            if k == "wall":
                z1 = max(z1, zc)
            grid = face_grid(pts[nm], ax, s0, s1, z0, z1)
            if grid is None:
                core_lo, core_hi = p["across_m"]
                adds, subs = [], []
            else:
                core_lo, core_hi, adds, subs = relief(grid)
                # Where a wall was scanned from one side only, every cell holds
                # one face and the two medians collapse onto each other. That
                # is not a 9 mm wall -- it is a wall with one face measured, so
                # fall back to the thickness the modular stage established.
                if core_hi - core_lo < 0.02:
                    core_lo, core_hi = faces_of(
                        pts[nm], ax, p.get("thickness_mm") or p.get("thickness_raw_mm"),
                        p["across_m"])
                    n_flat += 1
                n_relief += len(adds) + len(subs)
                if adds or subs:
                    n_steps += 1

            def span(lo_c, hi_c, u0, u1, v0, v1):
                lo = [0, 0, v0]; hi = [0, 0, v1]
                lo[ax], hi[ax] = lo_c, hi_c
                lo[1-ax], hi[1-ax] = u0, u1
                return box(lo, hi)

            runs = grid.get("runs") if grid else None
            if runs and nm in pts:
                settled = []
                for (i0, i1, fa, fb) in runs:
                    u0 = s0 + i0*grid["ucell"]; u1 = s0 + i1*grid["ucell"]
                    keep = ((pts[nm][:, 1-ax] >= u0-0.02)
                            & (pts[nm][:, 1-ax] <= u1+0.02))
                    sel, selw = pts[nm][keep], wts[nm][keep]
                    a2, ea = settle(sel, selw, ax, fa)
                    b2, eb = settle(sel, selw, ax, fb)
                    if b2 - a2 > 0.04:            # still a wall, not a sliver
                        fa, fb = a2, b2
                        for e in (ea, eb):
                            if e is not None:
                                stderr.append(e)
                    settled.append((i0, i1, fa, fb))
                runs = settled
            if runs:
                u = lambda i: s0 + i*grid["ucell"]
                # NB: not `a`/`b` -- this is inside main(), where `a` is the
                # argparse namespace, and shadowing it here cost a whole run
                def _rec(c_lo, c_hi, u0, u1, v0, v1, op):
                    lo = [0, 0, v0]; hi = [0, 0, v1]
                    lo[ax], hi[ax] = c_lo, c_hi
                    lo[1-ax], hi[1-ax] = u0, u1
                    sched(nm, k, op, lo, hi)
                for (i0, i1, c_lo, c_hi) in runs:
                    _rec(c_lo, c_hi, (s0-GROW) if i0 == 0 else u(i0),
                         (s1+GROW) if i1 >= len(grid["lo"]) else u(i1),
                         z0-GROW, z1+GROW, "run")
                for r in adds:
                    _rec(r["lo"], r["hi"], r["u0"], r["u1"], r["v0"], r["v1"],
                         "relief")          # a pilaster: push/pull OUT of the face
                for r in subs:
                    _rec(r["lo"], r["hi"], r["u0"], r["u1"], r["v0"], r["v1"],
                         "niche")           # push/pull IN
                for lo_c, hi_c in cutters.get(nm, []):
                    sched(nm, k, "opening", lo_c, hi_c)   # push/pull THROUGH
                pieces = [span(a, b, (s0-GROW) if i0 == 0 else u(i0),
                               (s1+GROW) if i1 >= len(grid["lo"]) else u(i1),
                               z0-GROW, z1+GROW) for (i0, i1, a, b) in runs]
                m = pieces[0] if len(pieces) == 1 else trimesh.boolean.union(
                    pieces, engine="manifold")
                if len(pieces) > 1:
                    n_runs += len(pieces) - 1
            else:
                m = span(core_lo, core_hi, s0-GROW, s1+GROW, z0-GROW, z1+GROW)
            # Relief goes on FIRST and the openings are cut afterwards. The
            # other way round, a pilaster unioned over a doorway seals it shut
            # again -- which is what put panels across the openings.
            if adds:
                try:
                    m = trimesh.boolean.union(
                        [m] + [span(r["lo"], r["hi"], r["u0"], r["u1"],
                                    r["v0"], r["v1"]) for r in adds],
                        engine="manifold")
                except Exception as e:
                    log(f"  {nm}: relief union failed ({e})")
            cs = list(cutters.get(nm, []))
            for r in subs:                      # niches: cut into the wall
                lo = [0, 0, r["v0"]]; hi = [0, 0, r["v1"]]
                lo[ax], hi[ax] = r["lo"], r["hi"]
                lo[1-ax], hi[1-ax] = r["u0"], r["u1"]
                cs.append((lo, hi))
            if cs:
                try:
                    holes = trimesh.util.concatenate([box(l, h) for l, h in cs])
                    m = trimesh.boolean.difference([m, holes], engine="manifold")
                    n_cut += len(cs)
                except Exception as e:
                    log(f"  {nm}: cut failed ({e})")
            if m.volume > 1e-4:
                made.append([nm, k, m])
            else:
                # Silence here loses a whole wall from a "watertight" model.
                log(f"  {nm}: no volume ({p.get('area_m2',0):.1f} m2 of surface "
                    f"dropped) -- faces at {core_lo:.3f}/{core_hi:.3f}")
        elif k in ("floor", "ceiling", "dropped_ceiling", "beam"):
            if nm not in bounds or p.get("area_m2", 0) < 0.4:
                continue
            b0, b1 = bounds[nm]
            z = zf + p["height_mm"]/1000.0
            # What was scanned is a SURFACE, and which way the material lies
            # behind it depends on which surface it is. A floor is walked on,
            # so its slab hangs below it. A ceiling, a dropped ceiling and a
            # beam soffit are all seen from underneath, so their material sits
            # ABOVE the surface -- built downwards they hang into the room,
            # which is where most of the error was.
            if k == "floor":
                if z > zc - 0.3 or z < zf - 0.6:
                    continue               # the storey above's floor, or below the building
                lo, hi = z - SLAB, z
            else:
                if z > zc + 0.10 or z < zf + 0.3:
                    continue               # not this storey's overhead surface
                if abs(z - zc) < 0.12:
                    lo, hi = z, z + SLAB           # this IS the ceiling
                elif k == "beam":
                    lo, hi = z, max(z + 0.12, zc)  # a beam runs up to the ceiling
                else:
                    lo, hi = z, z + 0.06           # a false ceiling is a plate
            # A plateau is one PART, but not always one height: a ceiling
            # part picked up over several rooms steps between them, and a
            # single plate at the part's average sits 40-80 mm off most of it.
            # So the part is split where its own surface steps, and each level
            # gets its own plate over its own footprint.
            q, w, fl = pts[nm], wts[nm], lie[nm]
            # Not everything labelled floor or ceiling is a plane. A step
            # riser, a plinth, a stepped bulkhead round a duct: mostly VERTICAL
            # surface, and a 150 mm plate at its average height misses it by
            # more than its own size. Anything less than a third horizontal is
            # modelled as the block it is.
            if w[fl].sum() < 0.30*w.sum():
                x0, x1 = np.percentile(q[:, 0], [1, 99])
                y0, y1 = np.percentile(q[:, 1], [1, 99])
                zl, zh = np.percentile(q[:, 2], [2, 98])
                if (x1-x0) > 0.12 and (y1-y0) > 0.12 and (zh-zl) > 0.08:
                    sched(nm, k, "slab", [x0, y0, zl], [x1, y1, zh])
                    made.append([nm, k, box([x0, y0, zl], [x1, y1, zh], slab=True)])
                continue
            qh, wh = (q[fl], w[fl]) if fl.sum() > 30 else (q, w)
            o = np.argsort(qh[:, 2]); zz = qh[o, 2]; ww = wh[o]
            cut = np.flatnonzero(np.diff(zz) > 0.06) + 1
            groups = np.split(np.arange(len(zz)), cut)
            levels = [g for g in groups if ww[g].sum() > 0.4]
            if not levels:
                levels = [np.arange(len(zz))]
            for j, g in enumerate(levels):
                zj = float(np.average(zz[g], weights=ww[g]))
                sd = float(np.sqrt(np.average((zz[g]-zj)**2, weights=ww[g])
                                   / max((ww[g].sum()**2)/np.sum(ww[g]**2), 1))*1000)
                stderr.append(sd)
                if k == "floor":
                    if zj > zc - 0.3 or zj < zf - 0.6:
                        continue
                    lo, hi = zj - SLAB, zj
                else:
                    if zj > zc + 0.10 or zj < zf + 0.3:
                        continue
                    if abs(zj - zc) < 0.12:
                        lo, hi = zj, zj + SLAB
                    elif k == "beam":
                        lo, hi = zj, max(zj + 0.12, zc)
                    else:
                        lo, hi = zj, zj + 0.06
                sub = qh[o][g]
                x0, x1 = np.percentile(sub[:, 0], [1, 99])
                y0, y1 = np.percentile(sub[:, 1], [1, 99])
                if x1-x0 < 0.15 or y1-y0 < 0.15:
                    continue
                sched(nm if j == 0 else f"{nm}_{j:02d}", k, "slab",
                      [x0-GROW, y0-GROW, lo], [x1+GROW, y1+GROW, hi])
                made.append([nm if j == 0 else f"{nm}_{j:02d}", k,
                             box([x0-GROW, y0-GROW, lo],
                                 [x1+GROW, y1+GROW, hi], slab=True)])
        elif k == "column":
            if nm not in bounds:
                continue
            b0, b1 = bounds[nm]
            fp = p.get("footprint_m") or [None, None]
            x0, x1 = faces_of(pts[nm], 0, (fp[0] or 0)*1000 or None,
                              (b0[0], b1[0])) or (b0[0], b1[0])
            y0, y1 = faces_of(pts[nm], 1, (fp[1] or 0)*1000 or None,
                              (b0[1], b1[1])) or (b0[1], b1[1])
            sched(nm, k, "column", [x0, y0, zf], [x1, y1, zc])
            made.append([nm, k, box([x0, y0, zf], [x1, y1, zc])])
    log(f"{len(made)} boxes from {man['n_parts']} parts; {n_steps} walls carry "
        f"relief ({n_relief} pilasters, beams and niches worked into their own "
        f"wall); {n_cut} cuts made; {n_flat} walls were scanned from one side "
        f"only and took their thickness from the modular stage; "
        f"{n_runs} places where a wall steps to a new thickness along its run")
    if stderr:
        st = np.array(stderr)
        log(f"faces settled on the surface mean: standard error median "
            f"{np.median(st):.3f} mm, 90th {np.percentile(st, 90):.3f} mm, "
            f"worst {st.max():.2f} mm over {len(st)} faces")

    # ---- cut the overlaps away, so every part owns its own volume -----------
    raw_vol = sum(m.volume for _, _, m in made)
    if not a.no_cut:
        order = sorted(range(len(made)),
                       key=lambda i: (PRIORITY.index(made[i][1])
                                      if made[i][1] in PRIORITY else 99,
                                      -made[i][2].volume))
        placed = []
        n_l = 0
        for i in order:
            nm, k, m = made[i]
            b = m.bounds
            hits = [q for q in placed
                    if (q.bounds[0] < b[1]).all() and (q.bounds[1] > b[0]).all()]
            if hits:
                try:
                    cut = trimesh.boolean.difference(
                        [m, trimesh.util.concatenate(hits)], engine="manifold")
                    if cut.volume > 1e-4:
                        if len(cut.faces) > len(m.faces):
                            n_l += 1
                        made[i][2] = m = cut
                except Exception as e:
                    log(f"  {nm}: overlap cut failed ({e})")
            placed.append(m)
        log(f"{n_l} parts took an L or T cut where they meet another part")
    parts_out = [(nm, m) for nm, k, m in made if m.volume > 1e-4]

    # ---- does it add up? ----------------------------------------------------
    vol = sum(m.volume for _, m in parts_out)
    try:
        u = trimesh.boolean.union([m for _, m in parts_out], engine="manifold")
        u.export(str(out/"boxes_union.glb"))
        log(f"volumes: parts {vol:.2f} m3, union {u.volume:.2f} m3, "
            f"overlap {vol-u.volume:+.3f} m3 (was {raw_vol-u.volume:+.2f} before cutting); "
            f"union watertight={u.is_watertight}")
    except Exception as e:
        u = None
        log(f"union skipped: {e}")

    voff = 0
    with open(out/"boxes.obj", "w") as fh:
        fh.write("# clean box model: measured thickness runs, openings cut, "
                 "overlaps removed\n")
        for nm, m in parts_out:
            fh.write(f"o {nm}\n")
            np.savetxt(fh, m.vertices, fmt="v %.4f %.4f %.4f")
            np.savetxt(fh, m.faces + voff + 1, fmt="f %d %d %d")
            voff += len(m.vertices)
    sc = trimesh.Scene()
    rng = np.random.default_rng(3)
    for nm, m in parts_out:
        # Bake the normals. A glTF without NORMAL is lit as black whatever the
        # material says, and a boolean result carries none of its own -- which
        # is how a 400 kB file of 79 boxes ended up looking like a broken page.
        mm = trimesh.Trimesh(np.asarray(m.vertices), np.asarray(m.faces),
                             vertex_normals=np.asarray(m.vertex_normals),
                             process=False)
        col = np.zeros((len(mm.vertices), 4), np.uint8)
        col[:, :3] = rng.integers(80, 235, 3); col[:, 3] = 255
        mm.visual.vertex_colors = col
        sc.add_geometry(mm, geom_name=nm, node_name=nm)
    sc.export(str(out/"boxes.glb"))
    log(f"wrote boxes.obj ({voff:,} verts) and boxes.glb "
        f"({(out/'boxes.glb').stat().st_size/1e3:.0f} kB), {len(parts_out)} objects")

    for nm, m in parts_out:
        base = nm.rsplit("_", 1)[0] if nm not in wof else nm
        if base in wof:
            wof[base].setdefault("box_volume_m3", 0.0)
            wof[base]["box_volume_m3"] = round(
                wof[base]["box_volume_m3"] + float(m.volume), 4)
    man["boxes"] = dict(objects=len(parts_out), walls_with_relief=n_steps,
                        grow_mm=GROW*1000, cell_mm=[100, 150],
                        quant_mm=QUANT*1000,
                        volume_m3=round(vol, 3),
                        union_m3=round(float(u.volume), 3) if u is not None else None)
    json.dump(man, open(d/"manifest.json", "w"), indent=1)
    if SCHEDULE:
        js = out/"boxes_schedule.json"
        json.dump(dict(parts=SCHEDULE), open(js, "w"))
        kinds = {}
        for r in SCHEDULE:
            kinds[r["op"]] = kinds.get(r["op"], 0) + 1
        log("schedule for SketchUp: " + ", ".join(
            f"{v} {k}" for k, v in sorted(kinds.items()))
            + f"  over {len({r['name'] for r in SCHEDULE})} parts -> {js.name}")
    log("manifest updated")


if __name__ == "__main__":
    main()
