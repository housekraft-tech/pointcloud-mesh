"""The modular model taken from the POISSON MESH, not fitted to it.

The cell-complex model is right about structure and wrong about surface: every
part is a box, so a niche, an arched head, a boxed conduit and a beam soffit all
come out as flat wall. The Poisson mesh is the opposite -- it carries all of
that relief, and none of the structure: it is one blob with the furniture still
in it.

This takes the structure FROM the mesh itself and keeps the mesh's own surface
as the geometry of every part:

  planes      detected on the mesh by area, not on the points, and paired into
              physical walls whose thickness is the gap between two real faces
  parts       a part owns whole triangles of the original mesh, so a niche, a
              reveal, an arch soffit and a beam drop survive exactly as scanned
  seamless    after the geometric labelling, every unclaimed triangle that
              touches a part grows into it, so junctions have no cracks and the
              parts partition the structural surface instead of sampling it
  furniture   what is left after that growth is what is attached to nothing
              structural, and only that is dropped

The seam-closing growth is the difference between this and the earlier
crop-by-plane build: a plane band cuts the mesh at a flat boundary, and every
fillet, jamb return and soffit within a junction falls outside every band. Those
are exactly the triangles that make the model read as one house rather than a
set of panels.
"""
import sys, os, json, time, argparse
from pathlib import Path
import numpy as np

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

# ---- geometry constants (metres, degrees) ---------------------------------
VERT_NZ    = 0.35     # |n.z| below this is a wall surface
HORIZ_NZ   = 0.85     # |n.z| above this is a slab surface
HEAD_BIN   = 3.0      # deg: heading bin for plane seeding
HEAD_TOL   = 30.0     # deg: membership tolerance -- Poisson normals wobble
OFF_BIN    = 0.01     # m: offset histogram bin
FACE_AREA  = 0.35     # m2: a face plane must have this much scanned area
FACE_BAND  = 0.05     # m: how far off the plane a triangle still belongs to it
RUN_GAP    = 0.35     # m: split a face into separate runs across gaps this big
RUN_MIN    = 0.40     # m: shortest run that is a wall
WALL_ZFRAC = 0.55     # frac of room height a wall run must span
WALL_TOP   = 1.70     # m above the floor a wall must reach. Measured against
                      # the FLOOR, not the ceiling: a wet room's wall stops at
                      # its dropped ceiling and is still a full wall.
WALL_AREA  = 1.00     # m2: below this a "wall" is the leftover sliver of one
                      # already claimed, and is released back for growth
JOIN_GAP   = 1.30     # m: two runs on the same pair of planes with a gap this
                      # short are one wall with a doorway in it
PAR_MIN    = 1.00     # m: a parapet (half-height wall) must be at least this long
PAR_ZMIN   = 0.45     # m: and this tall
MAX_THICK  = 0.45     # m: thicker than this is not one wall
MIN_THICK  = 0.08     # m: thinner than this is not masonry. Two scans of one
                      # flat disagreed on thickness by 86 mm at the median, and
                      # the thin readings were all a panel standing off a wall
                      # being paired as its far face.
MODE_TOL   = 0.020    # m: how close a candidate must be to one of the
                      # building's own thicknesses to be preferred as one
MODE_SUPP  = 0.25
PT_CELL    = 0.02     # m: grid the scan is checked on
PT_SUPPORT = 0.50     # a face plane must have scanned points behind this much
                      # of itself, or it is not a surface the scanner saw     # a peak carrying less than this share of the strongest
                      # one is not a thickness the building repeats -- it is a
                      # few panels standing off a few walls
PAIR_ZCOVER = 0.70    # the opposite face must span this much of the face's own
                      # height. A wall is bounded by a wall; a wardrobe front
                      # stops at 2 m and is not the other side of anything.
STATION    = 0.05     # m: station spacing along a face when pairing
SLAB_CELL  = 0.05     # m: slab height-map cell
SLAB_STEP  = 0.030    # m: height step that breaks a plateau
SLAB_MIN   = 40       # cells: smallest slab part (0.10 m2)
BEAM_MAXW  = 0.90     # m: a beam soffit is no wider than this
BEAM_MINL  = 1.00     # m: and no shorter than this
SLAB_KEEP  = 0.25     # m2: a smaller plateau is a fragment, not a part: it is
                      # released so it grows into the surface around it
COL_MINFOOT = 0.15    # m: a column is at least this wide both ways
COL_MINAREA = 0.50    # m2: and carries at least this much surface
COL_CELL   = 0.10
COL_FOOT   = 0.80     # m: max column footprint
COL_ASPECT = 3.0
COL_TOUCH  = 0.30     # m: must reach this close to both floor and ceiling
COL_VERT   = 0.55     # frac of a column's surface that must be vertical
SPECK_AREA = 0.05     # m2: a connected piece smaller than this is Poisson dust
LEVEL_SAME = 0.025    # m: plateaus within this of each other are one surface
GROW_OUT   = 0.10     # m: how far INTO THE ROOM a wall may grow -- a skirting,
                      # a cornice, the lip of a reveal. Beyond it stands the
                      # furniture, and a curtain absorbed into a wall is worse
                      # than a missing one.
GROW_IN    = 0.35     # m: how far INTO THE MASONRY it may grow, for the wall
                      # whose far face was never seen. Relief cuts inwards, so
                      # this is the direction that has to stay generous.
GROW_ALONG = 0.30     # m: past the ends of the run
GROW_SLAB  = 0.55     # m: how far below its level a ceiling may grow (beam drop)
GROW_FLOOR = 0.20     # m
GROW_COL   = 0.20     # m beyond the column footprint
GROW_ROUNDS = 4000   # growth stops when nothing joins; this is only a backstop


# ---------------------------------------------------------------- mesh -----
def load(cache):
    d = np.load(cache)
    V = d["V"].astype(np.float64); T = d["T"].astype(np.int64)
    N = d["N"].astype(np.float64); C = d["C"].astype(np.float64)
    P = V[T]
    A = 0.5 * np.linalg.norm(np.cross(P[:, 1]-P[:, 0], P[:, 2]-P[:, 0]), axis=1)
    log(f"{len(T):,} tris, {A.sum():.1f} m2")
    return V, T, N, C, A


def manhattan_yaw(N, A):
    """The building's own yaw, from where the wall area points.

    Wall normals cluster at four headings 90 deg apart; the offset of that
    cluster from the axes is the yaw. Taken mod 90 and averaged as a circular
    quantity, so all four clusters vote for the same answer.
    """
    vert = np.abs(N[:, 2]) < VERT_NZ
    n2 = N[vert, :2]; a = A[vert]
    ln = np.linalg.norm(n2, axis=1); ok = ln > 1e-6
    ang = np.degrees(np.arctan2(n2[ok, 1], n2[ok, 0])) % 90.0
    w = a[ok]
    r = np.radians(ang * 4.0)                      # mod 90 -> full circle
    yaw = np.degrees(np.arctan2((w*np.sin(r)).sum(), (w*np.cos(r)).sum())) / 4.0
    return float(yaw)


def rotate(V, N, C, yaw_deg):
    r = np.radians(-yaw_deg); c, s = np.cos(r), np.sin(r)
    R = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    return V @ R.T, N @ R.T, C @ R.T


def levels(N, C, A):
    """Floor and ceiling from where the horizontal area is."""
    hz = np.abs(N[:, 2]) > HORIZ_NZ
    z = C[hz, 2]; a = A[hz]
    bins = np.arange(z.min()-0.01, z.max()+0.02, 0.01)
    h, e = np.histogram(z, bins=bins, weights=a)
    mid = 0.5*(e[:-1]+e[1:])
    lo = mid < mid.min() + 0.5*(mid.max()-mid.min())
    z_floor = float(mid[lo][np.argmax(h[lo])])
    z_ceil = float(mid[~lo][np.argmax(h[~lo])])
    ceil_area = float(h[~lo].max())
    log(f"floor z={z_floor:.3f}  ceiling z={z_ceil:.3f}  clear height "
        f"{(z_ceil-z_floor)*1000:.0f} mm  (ceiling slab {ceil_area:.1f} m2)")
    if ceil_area < 0.25*float(h[lo].max()):
        # A mesh built with the ceiling sliced off reads its dropped ceilings
        # as THE ceiling, and then every height in the model is wrong. Say so.
        log("WARNING: the ceiling carries far less area than the floor -- this "
            "mesh was probably built with the ceiling slab cut off. Rebuild it "
            "with the whole scan or every height here is measured to the wrong "
            "surface.")
    return z_floor, z_ceil


# --------------------------------------------------------------- planes ----
# A face plane's material lies on the side its normal points to, and two faces
# bound one wall when they face each other across a gap narrower than a wall.
# Which side that is depends on the mesh's global orientation, and Poisson's
# is arbitrary -- it follows whichever way the input normals were oriented, so
# two meshes of the same flat can come out opposite. orient() settles it
# against the floor, which is the one surface whose material side is known.


def orient(N, T, C, z_floor):
    """Make the normals point INTO the material, using the floor to decide.

    Everything downstream -- which side of a face is masonry, which face is the
    other side of a wall, how thick that wall is -- inverts with this sign. The
    floor settles it: whatever else is uncertain, the material under a floor is
    below it. On a mesh that comes out the other way, both the normals and the
    triangle winding are flipped, so the exported parts still face outward.
    """
    band = (np.abs(N[:, 2]) > 0.85) & (C[:, 2] < z_floor + 0.10)
    if band.sum() < 100:
        log("orientation: too little floor to check -- assuming normals point "
            "into the material")
        return N, T
    nz = float(np.mean(N[band, 2]))
    if nz < 0:
        log(f"orientation: floor normals read {nz:+.2f}, pointing down into the "
            f"ground -- normals already point into the material")
        return N, T
    log(f"orientation: floor normals read {nz:+.2f}, pointing up into the room "
        f"-- flipping the mesh so normals point into the material")
    return -N, T[:, ::-1].copy()

def scan_grid(las_path, yaw_deg):
    """The scan itself, as a 20 mm occupancy set in the model's frame.

    Poisson does not only interpolate, it extrapolates: where the cloud is thin
    it wraps a shell around the surface, and a wall seen from one side comes
    back as a slab about 100 mm thick with a face on each side. That phantom
    face is indistinguishable from a real one by geometry -- it is flat, it is
    vertical, it is full height, it faces the right way -- and it is what put a
    95 mm entry in koushik's thickness vocabulary, on a scan four times sparser
    than mujammel's, which sees no such wall.

    What it does not have is points. This grid is what a face is checked
    against.
    """
    import laspy
    with laspy.open(las_path) as r:
        pts = r.read()
    P = np.column_stack([pts.x, pts.y, pts.z]).astype(np.float64)
    P[:, 2] -= float(np.percentile(P[:, 2], 0.5))       # as poisson_mesh.py did
    a = np.radians(-yaw_deg); c, sn = np.cos(a), np.sin(a)
    P[:, :2] = np.column_stack([P[:, 0]*c - P[:, 1]*sn, P[:, 0]*sn + P[:, 1]*c])
    lo = P.min(0) - 0.1
    n = np.ceil((P.max(0) + 0.1 - lo)/PT_CELL).astype(int) + 1
    idx = ((P - lo)/PT_CELL).astype(np.int64)
    flat = (idx[:, 0]*n[1] + idx[:, 1])*n[2] + idx[:, 2]
    occ = np.zeros(int(n.prod()), bool)
    occ[flat] = True
    log(f"scan grid: {len(P):,} points, {int(occ.sum()):,} cells of {PT_CELL*1000:.0f} mm")
    return dict(occ=occ, lo=lo, n=n)


def point_support(grid, Q):
    """Fraction of the given positions that have a scanned point beside them."""
    if grid is None:
        return 1.0
    lo, n, occ = grid["lo"], grid["n"], grid["occ"]
    idx = np.floor((Q - lo)/PT_CELL).astype(np.int64)
    ok = np.zeros(len(Q), bool)
    for d in (0, 1, -1):
        for e in (0, 1, -1):
            for f in (0, 1, -1):
                j = idx + np.array([d, e, f])
                good = np.all((j >= 0) & (j < n), axis=1)
                flat = (j[good, 0]*n[1] + j[good, 1])*n[2] + j[good, 2]
                w = np.where(good)[0]
                ok[w[occ[flat]]] = True
    return float(ok.mean())


def face_planes(N, C, A, z_floor, z_ceil, grid=None):
    """Every wall face: an axis, a side, and a coordinate, found by area."""
    vert = (np.abs(N[:, 2]) < VERT_NZ) & (C[:, 2] > z_floor + 0.03) & (C[:, 2] < z_ceil - 0.03)
    out = []
    dropped = [0]
    for ax in (0, 1):
        for sgn in (+1, -1):
            m = np.where(vert & (N[:, ax]*sgn > np.cos(np.radians(HEAD_TOL))))[0]
            if m.size == 0:
                continue
            c = C[m, ax]; a = A[m]
            bins = np.arange(c.min()-OFF_BIN, c.max()+2*OFF_BIN, OFF_BIN)
            h, e = np.histogram(c, bins=bins, weights=a)
            mid = 0.5*(e[:-1]+e[1:])
            h3 = np.convolve(h, np.ones(3), "same")            # 30 mm window
            taken = np.zeros(len(h3), bool)
            for j in np.argsort(-h3):
                if h3[j] < FACE_AREA or taken[max(0, j-4):j+5].any():
                    continue
                taken[j] = True
                band = np.abs(c - mid[j]) < FACE_BAND
                if a[band].sum() < FACE_AREA:
                    continue
                tris = m[band]
                # a face the scanner never actually hit is a Poisson artefact
                sub = tris if len(tris) < 4000 else tris[::max(1, len(tris)//4000)]
                if point_support(grid, C[sub]) < PT_SUPPORT:
                    dropped[0] += 1
                    continue
                # The face position is the PEAK of the histogram refined over a
                # narrow window, not the mean of the whole band: relief is all
                # on one side of a face, so a band-wide mean is pulled into the
                # masonry and every thickness comes out ~20 mm too big.
                core = np.abs(c - mid[j]) < 0.02
                if a[core].sum() < 0.2*a[band].sum():
                    core = band
                z = C[tris, 2]
                out.append(dict(axis=ax, sign=sgn,
                                coord=float(np.average(c[core], weights=a[core])),
                                tris=tris, area=float(a[band].sum()),
                                z0=float(np.percentile(z, 1)),
                                z1=float(np.percentile(z, 99))))
    out.sort(key=lambda f: -f["area"])
    if dropped[0]:
        log(f"{dropped[0]} face planes had no scanned points behind them "
            f"-- Poisson shells, dropped")
    log(f"{len(out)} face planes "
        + ", ".join(f"{sum(1 for f in out if f['axis']==ax and f['sign']==s)}"
                    f"@{'xy'[ax]}{'+-'[s<0]}" for ax in (0, 1) for s in (+1, -1)))
    return out


def profile(f, C, A, lo, hi):
    """Where along its own length a face actually carries material."""
    al = C[f["tris"], 1-f["axis"]]
    idx = np.clip(((al - lo)/STATION).astype(int), 0, int((hi-lo)/STATION))
    return np.bincount(idx, weights=A[f["tris"]], minlength=int((hi-lo)/STATION)+1)


def thickness_modes(planes, prof, st_area):
    """The thicknesses this building was actually built with.

    Choosing the NEAREST opposite face makes a wardrobe standing 80 mm off a
    wall into that wall's far side, and two scans of one flat then disagree
    about thickness by 86 mm. A building does not have arbitrary thicknesses:
    it has two or three, repeated everywhere. Those repeats are found here as
    the peaks of the candidate histogram, weighted by how many stations support
    each candidate, and a candidate near a peak is then preferred over a nearer
    one that matches nothing else in the building.
    """
    h = np.zeros(int(MAX_THICK/0.01)+1)
    for i, f in enumerate(planes):
        for j, g in enumerate(planes):
            if g["axis"] != f["axis"] or g["sign"] == f["sign"]:
                continue
            t = (g["coord"] - f["coord"])*f["sign"]
            if not (MIN_THICK < t < MAX_THICK):
                continue
            zs = f["z1"] - f["z0"]
            cov = min(f["z1"], g["z1"]) - max(f["z0"], g["z0"])
            if zs > 0.1 and cov < PAIR_ZCOVER*zs:
                continue
            n = min(len(prof[i]), len(prof[j]))
            both = (prof[i][:n] > st_area) & (prof[j][:n] > st_area)
            # Weight a candidate by the SURFACE the two faces carry where they
            # face each other, not by how many stations they share. Counting
            # stations lets a shelf and a wall outvote two walls, and that is
            # what put a phantom 145 mm in koushik's vocabulary while mujammel,
            # the same flat better scanned, read a clean 200 and 243.
            sup = float(np.minimum(prof[i][:n], prof[j][:n])[both].sum())
            if sup > 0:
                h[int(t/0.01)] += sup
    if h.sum() == 0:
        return []
    sm = np.convolve(h, np.ones(3), "same")
    modes = []
    for b in np.argsort(-sm):
        if sm[b] < MODE_SUPP*sm.max() or any(abs(b*0.01 - m) < 0.05 for m in modes):
            continue
        # the peak's own centre of mass, not the bin centre: the bins are 10 mm
        # and the number is quoted against a drawing in millimetres
        w = sm[max(0, b-1):b+2]
        c = np.arange(max(0, b-1), b+2)*0.01 + 0.005
        modes.append(float((w*c).sum()/w.sum()))
        if len(modes) == 3:
            break
    log("thicknesses this building repeats: "
        + ", ".join(f"{m*1000:.0f} mm" for m in sorted(modes)))
    return sorted(modes)


def physical_walls(planes, C, A, z_floor, z_ceil, label, name_of):
    """Pair the faces into walls, and measure each wall where it stands.

    Pairing plane-to-plane rather than run-to-run is what makes a corridor wall
    one wall: its far side is broken into three faces by the rooms it passes,
    and a run-to-run pairing can only ever match one of them. Here each 50 mm
    station along a face chooses its own opposite face -- the nearest one that
    has material there -- so the thickness is measured per station and the wall
    is cut only where that choice changes.
    """
    H = z_ceil - z_floor
    lo = C[:, :2].min(0) - 0.5; hi = C[:, :2].max(0) + 0.5
    prof = []
    for f in planes:
        prof.append(profile(f, C, A, lo[1-f["axis"]], hi[1-f["axis"]]))
    ST_AREA = 0.002          # m2 of surface in a 50 mm station = material there

    modes = thickness_modes(planes, prof, ST_AREA)

    runs = []
    for i, f in enumerate(planes):
        L0 = lo[1-f["axis"]]
        has = prof[i] > ST_AREA
        partner = np.full(len(has), -1, np.int64)
        thick = np.zeros(len(has))
        score = np.full(len(has), 1e9)
        vouched = np.zeros(len(has), bool)
        for j, g in enumerate(planes):
            if g["axis"] != f["axis"] or g["sign"] == f["sign"]:
                continue
            t = (g["coord"] - f["coord"])*f["sign"]      # material side of f
            if not (MIN_THICK < t < MAX_THICK):
                continue
            span = f["z1"] - f["z0"]
            cover = min(f["z1"], g["z1"]) - max(f["z0"], g["z0"])
            if span > 0.1 and cover < PAIR_ZCOVER*span:
                continue
            pj = prof[j]
            n = min(len(has), len(pj))
            # the opposite face may be occluded for a few stations; smear it
            gh = np.convolve((pj > ST_AREA).astype(float), np.ones(7), "same")[:n] > 0
            # rank: a thickness the building repeats beats a nearer one that
            # matches nothing, and among equals the thinner wins
            near = min((abs(t-m) for m in modes), default=1e9)
            rank = (0.0 if near < MODE_TOL else 1.0) + t/100.0
            take = has[:n] & gh & ((partner[:n] < 0) | (rank < score[:n]))
            partner[:n][take] = j; thick[:n][take] = t; score[:n][take] = rank
            vouched[:n][take] = near < MODE_TOL
        # cut the face into runs of constant partner
        cut = np.r_[True, (partner[1:] != partner[:-1]) | (~has[1:]) | (~has[:-1])]
        starts = np.flatnonzero(cut & has)
        for s in starts:
            e = s
            while e+1 < len(has) and has[e+1] and partner[e+1] == partner[s]:
                e += 1
            if (e-s+1)*STATION < RUN_MIN:
                continue
            runs.append(dict(face=i, partner=int(partner[s]),
                             s0=L0 + s*STATION, s1=L0 + (e+1)*STATION,
                             vouched=bool(vouched[s]),
                             thickness=float(thick[s]) if partner[s] >= 0 else None))
    # a doorway does not end a wall: rejoin runs that sit on the same pair of
    # planes with only a door's width between them
    runs.sort(key=lambda r: (r["face"], r["partner"], r["s0"]))
    joined = []
    for r in runs:
        if (joined and joined[-1]["face"] == r["face"]
                and joined[-1]["partner"] == r["partner"]
                and r["s0"] - joined[-1]["s1"] < JOIN_GAP):
            joined[-1]["s1"] = max(joined[-1]["s1"], r["s1"])
            continue
        joined.append(dict(r))
    log(f"{len(runs)} face runs -> {len(joined)} after rejoining across doorways")
    runs = joined

    # a wall found from both of its faces is one wall: keep the pairing once
    seen = set(); walls = []
    for r in sorted(runs, key=lambda r: -(r["s1"]-r["s0"])):
        f = planes[r["face"]]
        key = None
        if r["partner"] >= 0:
            a, b = sorted((r["face"], r["partner"]))
            key = (a, b, round(r["s0"], 1), round(r["s1"], 1))
            dup = any(k[0] == a and k[1] == b and
                      min(k[3], r["s1"]) - max(k[2], r["s0"]) > 0.5*(r["s1"]-r["s0"])
                      for k in seen)
            if dup:
                continue
            seen.add(key)
        walls.append(dict(planes=[r["face"]] + ([r["partner"]] if r["partner"] >= 0 else []),
                          axis=f["axis"], s0=r["s0"], s1=r["s1"],
                          vouched=r["vouched"], thickness=r["thickness"]))
    log(f"{len(runs)} runs -> {len(walls)} physical walls "
        f"({sum(1 for w in walls if w['thickness'] is None)} single-faced)")

    # ---- claim the masonry between the two faces --------------------------
    keep = []
    for w in walls:
        ax = w["axis"]; ps = [planes[i] for i in w["planes"]]
        if w["thickness"] is None:
            f = ps[0]
            c_lo = f["coord"] - (FACE_BAND if f["sign"] > 0 else 0.0)
            c_hi = f["coord"] + (FACE_BAND if f["sign"] < 0 else 0.0)
            c_lo -= 0.02; c_hi += 0.02
        else:
            cs = sorted(p["coord"] for p in ps)
            c_lo, c_hi = cs[0] - 0.03, cs[1] + 0.03
        m = ((C[:, ax] > c_lo) & (C[:, ax] < c_hi) &
             (C[:, 1-ax] > w["s0"] - 0.05) & (C[:, 1-ax] < w["s1"] + 0.05) &
             (C[:, 2] > z_floor - 0.05) & (C[:, 2] < z_ceil + 0.05) & (label == -1))
        if not m.any():
            continue
        z = C[m, 2]
        zspan = float(z.max() - z.min())
        length = w["s1"] - w["s0"]
        full = zspan > WALL_ZFRAC*H and z.max() > z_floor + WALL_TOP
        if not full and not (length > PAR_MIN and zspan > PAR_ZMIN and
                             z.min() < z_floor + 0.35):
            continue
        if float(A[m].sum()) < WALL_AREA:
            # the far side of a wall already claimed, or a fragment of one:
            # leave it unlabelled so it grows into the wall it belongs to
            continue
        w["kind"] = "wall" if full else "parapet"
        w["z0"] = float(z.min()); w["z1"] = float(z.max())
        w["length"] = float(length)
        w["c_lo"] = float(c_lo); w["c_hi"] = float(c_hi)
        # how far this wall may reach when the seams are grown: into the
        # masonry generously, into the room hardly at all
        if w["thickness"] is None:
            f = ps[0]
            w["g_lo"] = c_lo - (GROW_IN if f["sign"] < 0 else GROW_OUT)
            w["g_hi"] = c_hi + (GROW_IN if f["sign"] > 0 else GROW_OUT)
        else:
            w["g_lo"] = c_lo - GROW_OUT
            w["g_hi"] = c_hi + GROW_OUT
        w["part_id"] = len(name_of)
        w["name"] = f"{w['kind']}_{len(keep):02d}"
        label[m] = w["part_id"]
        name_of.append(w["name"])
        keep.append(w)
    keep = fuse_walls(keep, C, A, label, name_of)
    log(f"{len(keep)} walls kept ("
        f"{sum(1 for w in keep if w['kind']=='parapet')} parapets, "
        f"{sum(1 for w in keep if w['thickness'] and w.get('vouched'))} with a "
        f"thickness the building repeats, "
        f"{sum(1 for w in keep if w['thickness'] and not w.get('vouched'))} paired "
        f"with something else)")
    return keep, modes


def fuse_walls(walls, C, A, label, name_of):
    """Two parts facing each other across a wall's width are one wall.

    A face only pairs during detection if the opposite face was seen at the
    same stations. Where the far side was scanned in pieces -- three rooms
    along a corridor, or half a wall hidden behind a wardrobe -- the two sides
    survive as separate parts, and a model that shows a wall as two loose
    panels is not modular. Here they are fused after the fact, on the same
    test the pairing used: same axis, overlapping in plan, less than one wall
    apart. The fused thickness is then measured across both, not defaulted.
    """
    par = list(range(len(walls)))
    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]; i = par[i]
        return i
    for i, a in enumerate(walls):
        for j, b in enumerate(walls[i+1:], i+1):
            if a["axis"] != b["axis"] or find(i) == find(j):
                continue
            ov = min(a["s1"], b["s1"]) - max(a["s0"], b["s0"])
            if ov < 0.50:
                continue
            span = max(a["c_hi"], b["c_hi"]) - min(a["c_lo"], b["c_lo"])
            gap = max(a["c_lo"], b["c_lo"]) - min(a["c_hi"], b["c_hi"])
            if span > MAX_THICK or gap > 0.30:
                continue
            par[find(i)] = find(j)
    groups = {}
    for i in range(len(walls)):
        groups.setdefault(find(i), []).append(i)
    if len(groups) == len(walls):
        return walls
    out = []
    for g in groups.values():
        ws = [walls[i] for i in g]
        if len(ws) == 1:
            out.append(ws[0]); continue
        head = max(ws, key=lambda w: float(A[label == w["part_id"]].sum()))
        c_lo = min(w["c_lo"] for w in ws); c_hi = max(w["c_hi"] for w in ws)
        head["s0"] = min(w["s0"] for w in ws); head["s1"] = max(w["s1"] for w in ws)
        head["length"] = head["s1"] - head["s0"]
        head["c_lo"] = c_lo; head["c_hi"] = c_hi
        head["g_lo"] = min(w["g_lo"] for w in ws)
        head["g_hi"] = max(w["g_hi"] for w in ws)
        # Prefer a thickness that was MEASURED between a pair of faces during
        # detection. The outer span of the fused parts is an upper bound: it
        # also spans whatever sits between two walls that flank a duct.
        ts = [(float(A[label == w["part_id"]].sum()), w["thickness"], w["vouched"])
              for w in ws if w["thickness"]]
        if ts:
            _, t, v = max(ts)                 # the best-supported measurement
        else:
            t, v = float(c_hi - c_lo - 0.06), False   # an upper bound, no pair
        head["thickness"] = t if MIN_THICK < t < MAX_THICK else None
        head["vouched"] = bool(v)
        head["planes"] = sorted({q for w in ws for q in w["planes"]})
        head["z0"] = min(w["z0"] for w in ws); head["z1"] = max(w["z1"] for w in ws)
        for w in ws:
            if w is not head:
                label[label == w["part_id"]] = head["part_id"]
                # the absorbed part must not keep a wall name: the renumbering
                # below reuses those names, and two parts answering to one name
                # is how a lookup lands on an empty part
                name_of[w["part_id"]] = f"_absorbed_{w['part_id']:03d}"
        out.append(head)
    # renumber so the names run in order of size again
    out.sort(key=lambda w: -float(A[label == w["part_id"]].sum()))
    remap = {}
    for k, w in enumerate(out):
        nm = f"{w['kind']}_{k:02d}"
        remap[w["part_id"]] = nm
        name_of[w["part_id"]] = nm
        w["name"] = nm
    log(f"fused {len(walls)} wall parts into {len(out)} walls "
        f"({sum(1 for w in out if w['thickness']) } with a measured thickness)")
    return out


# ---------------------------------------------------------------- slabs ----
def plateaus(sel, C, A, prefix, label, name_of, pick_z):
    """Grow flat plateaus in a set of horizontal triangles.

    Height clustering does not separate a beam from the ceiling it hangs off --
    the height histogram is continuous. A plateau is separated from its
    neighbour by a vertical step, which region growing across a height map
    finds directly.
    """
    if sel.size == 0:
        return []
    p = C[sel]
    gx = np.floor(p[:, 0]/SLAB_CELL).astype(np.int64)
    gy = np.floor(p[:, 1]/SLAB_CELL).astype(np.int64)
    key = gx*1000000 + gy
    o = np.argsort(key); ks = key[o]
    b = np.r_[0, np.flatnonzero(np.diff(ks))+1, len(ks)]
    cij, cz, ctr = [], [], []
    for s, e in zip(b[:-1], b[1:]):
        sl = o[s:e]
        cij.append((int(ks[s]//1000000), int(ks[s] % 1000000)))
        cz.append(pick_z(p[sl, 2]))
        ctr.append(sel[sl])
    cz = np.array(cz)
    lut = {c: i for i, c in enumerate(cij)}
    seen = set(); out = []
    for c0 in lut:
        if c0 in seen:
            continue
        seen.add(c0); stack = [c0]; comp = [c0]
        while stack:
            x = stack.pop(); zx = cz[lut[x]]; i, j = x
            for nb in ((i+1, j), (i-1, j), (i, j+1), (i, j-1)):
                if nb in seen or nb not in lut:
                    continue
                if abs(cz[lut[nb]] - zx) < SLAB_STEP:
                    seen.add(nb); stack.append(nb); comp.append(nb)
        if len(comp) < SLAB_MIN:
            continue
        ii = np.array([lut[c] for c in comp])
        tris = np.concatenate([ctr[i] for i in ii])
        ij = np.array(comp)
        ext = (ij.max(0) - ij.min(0) + 1) * SLAB_CELL
        z = float(np.median(cz[ii]))
        out.append(dict(cells=ij, tris=tris, z=z,
                        area=float(len(comp)*SLAB_CELL**2),
                        width=float(min(ext)), length=float(max(ext)),
                        flat_mm=float(np.std(cz[ii])*1000)))
    # Two plateaus at the same height that touch are one surface. The reach is
    # two cells, so a furniture shadow is bridged and a wall -- four cells of
    # masonry at the thinnest -- is not. Region
    # growing splits them wherever a furniture shadow or a doorway pinches the
    # height map, and a floor cut in half by an invisible line is not modular,
    # it is an artefact.
    out.sort(key=lambda c: -c["area"])
    owner = {}
    for i, c in enumerate(out):
        for ij in map(tuple, c["cells"]):
            owner[ij] = i
    par = list(range(len(out)))
    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]; i = par[i]
        return i
    for i, c in enumerate(out):
        for (x, y) in map(tuple, c["cells"]):
            for nb in ((x+1, y), (x-1, y), (x, y+1), (x, y-1),
                       (x+2, y), (x-2, y), (x, y+2), (x, y-2)):
                j = owner.get(nb)
                if j is None or find(j) == find(i):
                    continue
                if abs(out[j]["z"] - c["z"]) < LEVEL_SAME:
                    par[find(i)] = find(j)
    merged = {}
    for i, c in enumerate(out):
        merged.setdefault(find(i), []).append(c)
    grouped = []
    for g in merged.values():
        if len(g) == 1:
            grouped.append(g[0]); continue
        cells = np.concatenate([c["cells"] for c in g])
        ext = (cells.max(0)-cells.min(0)+1)*SLAB_CELL
        grouped.append(dict(cells=cells,
                            tris=np.concatenate([c["tris"] for c in g]),
                            z=float(np.average([c["z"] for c in g],
                                               weights=[c["area"] for c in g])),
                            area=float(sum(c["area"] for c in g)),
                            width=float(min(ext)), length=float(max(ext)),
                            flat_mm=float(max(c["flat_mm"] for c in g))))
    if len(grouped) < len(out):
        log(f"{prefix}: {len(out)} plateaus -> {len(grouped)} after joining "
            f"pieces at the same level")
    out = sorted(grouped, key=lambda c: -c["area"])
    parts = []
    for c in out:
        if c["area"] < SLAB_KEEP:
            continue
        c["part_id"] = len(name_of)
        c["name"] = f"{prefix}_{len(parts):02d}"
        label[c["tris"]] = c["part_id"]
        name_of.append(c["name"])
        parts.append(c)
    return parts


def slabs(N, C, A, z_floor, z_ceil, label, name_of):
    hz = np.abs(N[:, 2]) > HORIZ_NZ
    mid = 0.5*(z_floor+z_ceil)
    fl = np.where(hz & (C[:, 2] < z_floor + 0.25) & (label == -1))[0]
    ce = np.where(hz & (C[:, 2] > mid) & (label == -1))[0]
    floor = plateaus(fl, C, A, "floor", label, name_of, lambda z: float(np.min(z)))
    ceil = plateaus(ce, C, A, "ceiling", label, name_of, lambda z: float(np.max(z)))
    # a long narrow plateau hanging below the highest ceiling is a beam
    if ceil:
        top = max(c["z"] for c in ceil)
        for c in ceil:
            drop = top - c["z"]
            c["kind"] = ("beam" if (c["width"] < BEAM_MAXW and drop > 0.08
                                    and c["length"] > max(BEAM_MINL, 2*c["width"])
                                    and c["area"] > 0.30)
                         else ("dropped_ceiling" if drop > 0.10 else "ceiling"))
            c["drop_mm"] = round(drop*1000, 1)
    for i, c in enumerate(ceil):
        if c["kind"] != "ceiling":
            nm = f"{c['kind']}_{i:02d}"
            name_of[c["part_id"]] = nm
            c["name"] = nm
    log(f"floor: {len(floor)} parts, ceiling: {len(ceil)} parts "
        f"({sum(1 for c in ceil if c['kind']=='beam')} beams, "
        f"{sum(1 for c in ceil if c['kind']=='dropped_ceiling')} dropped)")
    return floor, ceil


def columns(C, A, N, z_floor, z_ceil, label, name_of):
    free = np.where((label == -1) & (C[:, 2] > z_floor+0.1) & (C[:, 2] < z_ceil-0.1))[0]
    if free.size == 0:
        return []
    p = C[free]
    gx = np.floor(p[:, 0]/COL_CELL).astype(np.int64)
    gy = np.floor(p[:, 1]/COL_CELL).astype(np.int64)
    key = gx*1000000 + gy
    uk, inv = np.unique(key, return_inverse=True)
    zmin = np.full(len(uk), 1e9); zmax = np.full(len(uk), -1e9)
    np.minimum.at(zmin, inv, p[:, 2]); np.maximum.at(zmax, inv, p[:, 2])
    tall = (zmax - zmin) > 0.60*(z_ceil - z_floor)
    cells = {(int(k//1000000), int(k % 1000000)) for k in uk[tall]}
    seen = set(); out = []
    for c in cells:
        if c in seen:
            continue
        st = [c]; comp = []
        while st:
            x = st.pop()
            if x in seen or x not in cells:
                continue
            seen.add(x); comp.append(x)
            st += [(x[0]+a, x[1]+b) for a in (-1, 0, 1) for b in (-1, 0, 1)]
        ij = np.array(comp)
        foot = (ij.max(0)-ij.min(0)+1)*COL_CELL
        if max(foot) > COL_FOOT or max(foot) > COL_ASPECT*max(min(foot), 1e-3):
            continue
        if min(foot) < COL_MINFOOT:
            continue
        cs = set(comp)
        sel = free[[(a, b) in cs for a, b in zip(gx, gy)]]
        if sel.size < 150:
            continue
        z = C[sel, 2]
        if z.min() > z_floor + COL_TOUCH or z.max() < z_ceil - COL_TOUCH:
            continue
        vert = A[sel][np.abs(N[sel, 2]) < VERT_NZ].sum()
        if vert < COL_VERT*A[sel].sum() or A[sel].sum() < COL_MINAREA:
            continue        # a column is walls, not a stack of horizontal tops
        pid = len(name_of); nm = f"column_{len(out):02d}"
        label[sel] = pid; name_of.append(nm)
        out.append(dict(name=nm, part_id=pid, tris=sel, cells=ij,
                        footprint=[round(float(foot[0]), 3), round(float(foot[1]), 3)],
                        area=float(A[sel].sum())))
    log(f"{len(out)} columns")
    return out


# ------------------------------------------------------------ seam grow ----
def prune_specks(label, ea, eb, A):
    """Drop the dust before anything is measured on it.

    Poisson inflates a small balloon around every stray return, and those
    balloons are what a plane histogram, a plateau and a column detector all
    trip over. They are recognised structurally, not by height: a piece of the
    building is a large connected sheet, and a balloon is a few square
    centimetres attached to nothing.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(label)
    g = coo_matrix((np.ones(len(ea)), (ea, eb)), shape=(n, n))
    ncomp, comp = connected_components(g, directed=False)
    ar = np.bincount(comp, weights=A)
    dust = ar[comp] < SPECK_AREA
    label[dust] = -2
    log(f"{ncomp:,} connected pieces; {int(dust.sum()):,} triangles "
        f"({ar[ar < SPECK_AREA].sum():.2f} m2) are dust and are set aside")
    return int(dust.sum())



def adjacency(T, nv):
    """Triangle pairs sharing an edge."""
    E = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    E = np.sort(E, axis=1)
    key = E[:, 0].astype(np.int64)*nv + E[:, 1]
    o = np.argsort(key, kind="stable")
    ks = key[o]
    same = np.flatnonzero(ks[1:] == ks[:-1])
    nt = len(T)
    a = (o[same] % nt).astype(np.int64)
    b = (o[same+1] % nt).astype(np.int64)
    m = a != b
    log(f"adjacency: {m.sum():,} shared edges")
    return a[m], b[m]


def grow(label, ea, eb, C, walls, ceil, floor, cols, z_floor, z_ceil):
    """Close the seams: an unclaimed triangle joins the part it touches.

    Every junction -- wall to ceiling, jamb to reveal, beam side to soffit --
    lies outside every geometric band by construction, because the bands are
    flat and the junction is where two of them meet. Those triangles are what
    make the parts read as one continuous house, so instead of being discarded
    they grow into whichever part they are attached to, subject to a limit on
    how far a part may reach.
    """
    nparts = int(label.max())+1
    kind = np.zeros(nparts, np.int8)          # 0 wall 1 ceiling 2 floor 3 column
    ax = np.zeros(nparts, np.int8)
    c_lo = np.zeros(nparts); c_hi = np.zeros(nparts)
    s_lo = np.zeros(nparts); s_hi = np.zeros(nparts)
    lvl = np.zeros(nparts)
    box = np.zeros((nparts, 4))
    for w in walls:
        i = w["part_id"]
        kind[i] = 0; ax[i] = w["axis"]
        c_lo[i] = w["g_lo"]; c_hi[i] = w["g_hi"]
        s_lo[i] = w["s0"]; s_hi[i] = w["s1"]
    for c in ceil:
        kind[c["part_id"]] = 1; lvl[c["part_id"]] = c["z"]
    for f in floor:
        kind[f["part_id"]] = 2; lvl[f["part_id"]] = f["z"]
    for c in cols:
        i = c["part_id"]; kind[i] = 3
        ij = c["cells"]
        box[i] = [ij[:, 0].min()*COL_CELL, (ij[:, 0].max()+1)*COL_CELL,
                  ij[:, 1].min()*COL_CELL, (ij[:, 1].max()+1)*COL_CELL]

    total_new = 0
    for it in range(GROW_ROUNDS):
        la, lb = label[ea], label[eb]
        cand_i = np.r_[eb[(la >= 0) & (lb == -1)], ea[(lb >= 0) & (la == -1)]]
        cand_p = np.r_[la[(la >= 0) & (lb == -1)], lb[(lb >= 0) & (la == -1)]]
        if cand_i.size == 0:
            break
        p = C[cand_i]
        k = kind[cand_p]
        ok = np.zeros(len(cand_i), bool)
        # walls
        w = k == 0
        if w.any():
            pw = cand_p[w]
            a0 = ax[pw]
            cc = np.where(a0 == 0, p[w, 0], p[w, 1])
            ss = np.where(a0 == 0, p[w, 1], p[w, 0])
            ok[w] = ((cc > c_lo[pw]) & (cc < c_hi[pw]) &
                     (ss > s_lo[pw] - GROW_ALONG) & (ss < s_hi[pw] + GROW_ALONG) &
                     (p[w, 2] > z_floor - 0.05) & (p[w, 2] < z_ceil + 0.10))
        c = k == 1
        if c.any():
            ok[c] = (p[c, 2] > lvl[cand_p[c]] - GROW_SLAB) & (p[c, 2] > 0.5*(z_floor+z_ceil))
        f = k == 2
        if f.any():
            ok[f] = p[f, 2] < lvl[cand_p[f]] + GROW_FLOOR
        q = k == 3
        if q.any():
            bx = box[cand_p[q]]
            ok[q] = ((p[q, 0] > bx[:, 0]-GROW_COL) & (p[q, 0] < bx[:, 1]+GROW_COL) &
                     (p[q, 1] > bx[:, 2]-GROW_COL) & (p[q, 1] < bx[:, 3]+GROW_COL))
        cand_i = cand_i[ok]; cand_p = cand_p[ok]
        if cand_i.size == 0:
            break
        # deterministic: lowest part id wins a contested triangle
        o = np.lexsort((cand_p, cand_i))
        ci, cp = cand_i[o], cand_p[o]
        first = np.r_[True, ci[1:] != ci[:-1]]
        ci, cp = ci[first], cp[first]
        new = label[ci] == -1
        label[ci[new]] = cp[new]
        n_new = int(new.sum())
        total_new += n_new
        if n_new == 0:
            break
    log(f"seam growth: {total_new:,} triangles joined their part in {it+1} rounds")
    return total_new


# --------------------------------------------------------------- relief ----
FEAT_CELL  = 0.025    # m: wall depth-map cell
FEAT_DEPTH = 0.040    # m: depth that makes a recess a niche, not surface noise
FEAT_MIN   = 100      # cells (0.06 m2) before a relief feature is real
OPEN_MIN_W = 0.35     # m
OPEN_MIN_H = 0.35     # m
ARCH_RISE  = 0.045    # m: head profile bowing this much is an arch, not a lintel
DOOR_H     = 1.60     # m: a void shorter than this, standing on the floor, is
                      # not a door -- it is the shadow of whatever stood there
SILL_MAX   = 1.70     # m: a window sill above this is not a window
FEAT_MAXFRAC = 0.25   # a "niche" covering more of the wall than this is the
                      # wall itself, mis-read against a plane on the far face
FEAT_MAXAREA = 3.0    # m2: and neither a niche nor a pilaster is bigger
FEAT_MAXDEEP = 0.25   # m: nor deeper than this


def wall_features(w, planes, C, label, z_floor, z_ceil):
    """Read a wall's own surface: its openings, its arches, its niches.

    The wall is unfolded onto its near face as a depth map in (along, height).
    A cell with no surface is a void -- an opening if material spans over it,
    the end of the wall if not. A cell whose surface sits deeper into the
    masonry than the face is a niche, one standing proud of it a pilaster. All
    three come off the same map, so they cannot contradict each other.
    """
    from scipy import ndimage
    f = planes[w["planes"][0]]
    ax = w["axis"]; sgn = f["sign"]
    sel = np.where(label == w["part_id"])[0]
    if sel.size == 0:
        return []
    p = C[sel]
    al = p[:, 1-ax]
    dep = (p[:, ax] - f["coord"])*sgn          # + = into the masonry
    z = p[:, 2]
    thick = w["thickness"] if w["thickness"] else 0.10
    near = (dep > -0.06) & (dep < max(0.5*thick, 0.06))
    if near.sum() < 200:
        return []
    al = al[near]; dep = dep[near]; z = z[near]
    t0, t1 = w["s0"], w["s1"]
    nu = max(2, int(np.ceil((t1-t0)/FEAT_CELL)))
    nv = max(2, int(np.ceil((z_ceil-z_floor)/FEAT_CELL)))
    iu = np.clip(((al-t0)/FEAT_CELL).astype(int), 0, nu-1)
    iv = np.clip(((z-z_floor)/FEAT_CELL).astype(int), 0, nv-1)
    flat = iu*nv + iv
    cnt = np.bincount(flat, minlength=nu*nv).reshape(nu, nv)
    dsum = np.bincount(flat, weights=dep, minlength=nu*nv).reshape(nu, nv)
    have = cnt > 0
    depth = np.zeros_like(dsum)
    depth[have] = dsum[have]/cnt[have]
    col_any = have.any(axis=1)
    feats = []

    # ---- openings: voids with material over them ---------------------------
    void = (~have) & col_any[:, None]
    lab, k = ndimage.label(void, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    for i in range(1, k+1):
        u, v = np.where(lab == i)
        if u.size < 40:
            continue
        u0, u1, v0, v1 = u.min(), u.max(), v.min(), v.max()
        wid = (u1-u0+1)*FEAT_CELL; hei = (v1-v0+1)*FEAT_CELL
        if wid < OPEN_MIN_W or hei < OPEN_MIN_H or u0 == 0 or u1 == nu-1:
            continue
        if not have[u0:u1+1, v1+1:].any(axis=1).all():
            continue                                   # no head over it
        # the top of the void, column by column: a flat lintel or a curved arch
        headz = np.array([np.max(np.where(lab[c] == i)[0]) for c in range(u0, u1+1)],
                         float)*FEAT_CELL + z_floor
        rise = float(headz.max() - np.median(np.r_[headz[:3], headz[-3:]]))
        sill = v0*FEAT_CELL + z_floor
        head = float(headz.max())
        # An opening is classified by what it is shaped like, and a void that
        # is not shaped like any of them is an occlusion shadow, not a hole in
        # the wall. Saying so is the difference between 11 doors and 23.
        arched = (rise > ARCH_RISE and rise < 0.6*wid and wid > 0.60
                  and head > z_floor + 1.70)
        if arched:
            kind = "arch"
        elif sill < z_floor + 0.25 and hei > DOOR_H and wid > 0.50:
            kind = "door"
        elif (z_floor + 0.30 < sill < z_floor + SILL_MAX and hei > 0.45
              and wid > 0.45):
            kind = "window"
        else:
            kind = "void"                              # unexplained: occlusion
        feats.append(dict(kind=kind, wall=w["name"],
                          width_mm=round(wid*1000), height_mm=round(hei*1000),
                          sill_mm=round((sill-z_floor)*1000),
                          head_mm=round((head-z_floor)*1000),
                          arch_rise_mm=round(rise*1000, 1),
                          along_m=[round(float(t0+u0*FEAT_CELL), 3),
                                   round(float(t0+(u1+1)*FEAT_CELL), 3)]))

    # ---- niches and pilasters ---------------------------------------------
    # Depth is measured against the wall's OWN surface, not against the fitted
    # plane: the plane can sit on the far face, or a few centimetres off a
    # rendered wall, and then the whole face reads as one enormous niche.
    base = float(np.median(depth[have])) if have.any() else 0.0
    face_cells = int(have.sum())
    for mask, kn in ((have & (depth > base + FEAT_DEPTH), "niche"),
                     (have & (depth < base - FEAT_DEPTH), "pilaster")):
        lab, k = ndimage.label(mask)
        if k == 0:
            continue
        for i in range(1, k+1):
            u, v = np.where(lab == i)
            if (u.size < FEAT_MIN or u.size > FEAT_MAXFRAC*face_cells
                    or u.size*FEAT_CELL**2 > FEAT_MAXAREA):
                continue
            u0, u1, v0, v1 = u.min(), u.max(), v.min(), v.max()
            d = float(np.abs(depth[lab == i] - base).max())
            if d > FEAT_MAXDEEP:
                continue
            feats.append(dict(kind=kn, wall=w["name"],
                              width_mm=round((u1-u0+1)*FEAT_CELL*1000),
                              height_mm=round((v1-v0+1)*FEAT_CELL*1000),
                              depth_mm=round(d*1000, 1),
                              base_mm=round(v0*FEAT_CELL*1000),
                              area_m2=round(float(u.size)*FEAT_CELL**2, 3),
                              along_m=[round(float(t0+u0*FEAT_CELL), 3),
                                       round(float(t0+(u1+1)*FEAT_CELL), 3)]))
    return feats


def dedupe_openings(feats, walls):
    """One doorway is one opening, however many faces saw it.

    A wall whose two faces were not paired becomes two parts, and the same
    doorway is then read off both of them. They are the same hole: same axis,
    same position along the wall, same size.
    """
    wof = {w["name"]: w for w in walls}
    out = []
    for f in sorted(feats, key=lambda f: -(f.get("width_mm", 0)*f.get("height_mm", 0))):
        if f["kind"] not in ("door", "window", "arch", "void"):
            out.append(f); continue
        w = wof[f["wall"]]
        dup = False
        for g in out:
            if g["kind"] not in ("door", "window", "arch", "void"):
                continue
            v = wof[g["wall"]]
            if v["axis"] != w["axis"]:
                continue
            if abs(v["c_lo"] - w["c_lo"]) > 0.45 and abs(v["c_hi"] - w["c_hi"]) > 0.45:
                continue
            ov = (min(f["along_m"][1], g["along_m"][1]) -
                  max(f["along_m"][0], g["along_m"][0]))
            if ov > 0.5*(f["along_m"][1]-f["along_m"][0]):
                dup = True
                g.setdefault("also_on", []).append(f["wall"])
                break
        if not dup:
            out.append(f)
    n = len(feats) - len(out)
    if n:
        log(f"{n} openings were the same hole seen from the other face")
    return out


# ---------------------------------------------------------------- export ---
def write_obj(path, V, T, label, name_of, order):
    voff = 0
    with open(path, "w") as fh:
        fh.write("# modular house segmented from the Poisson mesh\n")
        for pid in order:
            idx = np.where(label == pid)[0]
            if idx.size == 0:
                continue
            t = T[idx]
            used = np.unique(t)
            remap = np.full(len(V), -1, np.int64); remap[used] = np.arange(len(used))
            fh.write(f"o {name_of[pid]}\n")
            for q in V[used]:
                fh.write(f"v {q[0]:.5f} {q[1]:.5f} {q[2]:.5f}\n")
            for a, b, c in remap[t]:
                fh.write(f"f {a+voff+1} {b+voff+1} {c+voff+1}\n")
            voff += len(used)
    log(f"wrote {path} ({voff:,} verts)")


def write_glb(path, V, T, label, name_of, order):
    import trimesh
    sc = trimesh.Scene()
    rng = np.random.default_rng(7)
    for pid in order:
        idx = np.where(label == pid)[0]
        if idx.size == 0:
            continue
        t = T[idx]
        used = np.unique(t)
        remap = np.full(len(V), -1, np.int64); remap[used] = np.arange(len(used))
        m = trimesh.Trimesh(V[used], remap[t], process=False)
        col = np.zeros((len(used), 4), np.uint8)
        col[:, :3] = rng.integers(60, 235, 3)
        col[:, 3] = 255
        m.visual.vertex_colors = col
        sc.add_geometry(m, geom_name=name_of[pid], node_name=name_of[pid])
    sc.export(path)
    log(f"wrote {path}")


# ----------------------------------------------------------------- audit ---
def audit(label, ea, eb, C, A, name_of):
    """Is the model seamless, and is what was dropped really furniture?

    A crack is not a triangle that changes part -- that is just a junction. It
    is a DROPPED triangle with two different parts around it: a hole left in the
    middle of the house. Those are counted directly.
    """
    tot = len(label)
    kept = int((label >= 0).sum())
    la, lb = label[ea], label[eb]
    seam = (la >= 0) & (lb >= 0) & (la != lb)
    crack_edges = ((la < 0) & (lb >= 0)) | ((lb < 0) & (la >= 0))
    # dropped triangles touching two or more distinct parts = a hole in the shell
    drop = np.where(label < 0)[0]
    touch = {}
    m = (la < 0) & (lb >= 0)
    for t, p in zip(ea[m], lb[m]):
        touch.setdefault(int(t), set()).add(int(p))
    m = (lb < 0) & (la >= 0)
    for t, p in zip(eb[m], la[m]):
        touch.setdefault(int(t), set()).add(int(p))
    cracks = sum(1 for v in touch.values() if len(v) > 1)
    log(f"AUDIT kept {kept:,}/{tot:,} tris ({100*kept/tot:.1f}%), "
        f"{seam.sum():,} part-to-part seam edges, "
        f"{len(touch):,} dropped tris touch the shell, {cracks:,} of them bridge "
        f"two parts")
    return dict(total_tris=tot, kept_tris=kept, dropped_tris=int(tot-kept),
                kept_frac=round(kept/tot, 4),
                kept_area_m2=round(float(A[label >= 0].sum()), 2),
                dropped_area_m2=round(float(A[label == -1].sum()), 2),
                dust_area_m2=round(float(A[label == -2].sum()), 2),
                seam_edges=int(seam.sum()), crack_tris=int(cracks))


def dropped_report(label, C, A, ea, eb, z_floor, z_ceil, n=8):
    """What was thrown away, largest first -- the check on the furniture rule."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    idx = np.where(label == -1)[0]
    if idx.size == 0:
        return []
    ren = np.full(len(label), -1, np.int64); ren[idx] = np.arange(len(idx))
    m = (label[ea] == -1) & (label[eb] == -1)
    g = coo_matrix((np.ones(m.sum()), (ren[ea[m]], ren[eb[m]])),
                   shape=(len(idx), len(idx)))
    ncomp, lab = connected_components(g, directed=False)
    ar = np.bincount(lab, weights=A[idx])
    out = []
    for c in np.argsort(-ar)[:n]:
        s = idx[lab == c]
        p = C[s]
        out.append(dict(area_m2=round(float(ar[c]), 2), tris=int((lab == c).sum()),
                        z=[round(float(p[:, 2].min()), 2), round(float(p[:, 2].max()), 2)],
                        size_m=[round(float(np.ptp(p[:, 0])), 2),
                                round(float(np.ptp(p[:, 1])), 2),
                                round(float(np.ptp(p[:, 2])), 2)]))
    log(f"dropped as furniture: {ncomp:,} pieces, largest "
        + ", ".join(f"{o['area_m2']} m2 (z {o['z'][0]}..{o['z'][1]})" for o in out[:5]))
    return out


# ------------------------------------------------------------------ main ---
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="output/model/poisson_koushik.npz")
    ap.add_argument("--out", default="output/model/poisson_modular")
    ap.add_argument("--no-glb", action="store_true")
    ap.add_argument("--las", default=None,
                    help="the scan itself: face planes are checked against it")
    args = ap.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    V, T, N, C, A = load(args.cache)
    yaw = manhattan_yaw(N, A)
    log(f"manhattan yaw {yaw:+.2f} deg -- rotating the scan onto its own axes")
    V, N, C = rotate(V, N, C, yaw)
    z_floor, z_ceil = levels(N, C, A)
    N, T = orient(N, T, C, z_floor)

    label = np.full(len(T), -1, np.int64)
    name_of = []
    ea, eb = adjacency(T, len(V))
    n_dust = prune_specks(label, ea, eb, A)

    grid = scan_grid(args.las, yaw) if args.las else None
    planes = face_planes(N, C, A, z_floor, z_ceil, grid)
    walls, modes = physical_walls(planes, C, A, z_floor, z_ceil, label, name_of)
    floor, ceil = slabs(N, C, A, z_floor, z_ceil, label, name_of)
    cols = columns(C, A, N, z_floor, z_ceil, label, name_of)
    log(f"before growth: {(label>=0).mean()*100:.1f}% of triangles claimed")

    grow(label, ea, eb, C, walls, ceil, floor, cols, z_floor, z_ceil)

    feats = []
    for w in walls:
        feats += wall_features(w, planes, C, label, z_floor, z_ceil)
    feats = dedupe_openings(feats, walls)
    log(f"{len(feats)} relief features: "
        + ", ".join(f"{k}={sum(1 for f in feats if f['kind']==k)}"
                    for k in ("door", "window", "arch", "niche", "pilaster", "void")))

    cov = audit(label, ea, eb, C, A, name_of)
    junk = dropped_report(label, C, A, ea, eb, z_floor, z_ceil)

    np.save(out/"labels.npy", label)
    np.save(out/"verts.npy", V.astype(np.float32))
    json.dump(name_of, open(out/"names.json", "w"))
    order = [p for p in range(len(name_of)) if (label == p).any()]
    write_obj(out/"modular.obj", V, T, label, name_of, order)
    if not args.no_glb:
        write_glb(str(out/"modular.glb"), V, T, label, name_of, order)

    parts = []
    for w in walls:
        idx = label == w["part_id"]
        # A thickness whose pair matches none of the thicknesses the building
        # repeats is a pairing with something that is not the other side of the
        # wall -- usually a panel standing off it. It is kept as a raw reading
        # and left out of the measurement, because a wrong number is worse here
        # than a missing one.
        parts.append(dict(name=w["name"], kind=w["kind"],
                          thickness_mm=(round(w["thickness"]*1000, 1)
                                        if w["thickness"] and w.get("vouched") else None),
                          thickness_raw_mm=round(w["thickness"]*1000, 1) if w["thickness"] else None,
                          faces_seen=len(w["planes"]),
                          axis="xy"[w["axis"]],
                          length_mm=round(w["length"]*1000),
                          across_m=[round(w["c_lo"], 4), round(w["c_hi"], 4)],
                          along_m=[round(w["s0"], 4), round(w["s1"], 4)],
                          z_mm=[round((w["z0"]-z_floor)*1000), round((w["z1"]-z_floor)*1000)],
                          tris=int(idx.sum()), area_m2=round(float(A[idx].sum()), 2)))
    for c in ceil + floor:
        idx = label == c["part_id"]
        cij = c["cells"]*SLAB_CELL
        parts.append(dict(name=c["name"], kind=c.get("kind", "floor"),
                          centre_m=[round(float(cij[:, 0].mean()), 3),
                                    round(float(cij[:, 1].mean()), 3)],
                          extent_m=[round(float(np.ptp(cij[:, 0])), 2),
                                    round(float(np.ptp(cij[:, 1])), 2)],
                          height_mm=round((c["z"]-z_floor)*1000, 1),
                          drop_mm=c.get("drop_mm"),
                          area_m2=round(c["area"], 2), flat_mm=round(c["flat_mm"], 1),
                          tris=int(idx.sum())))
    for c in cols:
        idx = label == c["part_id"]
        parts.append(dict(name=c["name"], kind="column", footprint_m=c["footprint"],
                          tris=int(idx.sum()), area_m2=round(float(A[idx].sum()), 2)))
    # The clear height is to the STRUCTURAL ceiling, which is the highest slab
    # carrying real area -- not the modal one. Picking the mode made the same
    # flat read 2740 mm off one mesh and 2700 off a finer one, because the two
    # plateaus swap which is larger; the highest one is the same in both.
    tops = [c for c in ceil if c.get("kind") == "ceiling"]
    if tops:
        big = max(c["area"] for c in tops)
        z_top = max(c["z"] for c in tops if c["area"] > 0.2*big)
    else:
        z_top = z_ceil

    man = dict(source=args.cache, yaw_deg=round(yaw, 3),
               structural_ceiling_z=round(z_top, 4),
               thickness_modes_mm=[round(m*1000, 1) for m in modes],
               floor_z=round(z_floor, 4), ceiling_z=round(z_ceil, 4),
               clear_height_mm=round((z_top-z_floor)*1000, 1),
               modal_ceiling_height_mm=round((z_ceil-z_floor)*1000, 1),
               n_parts=len(order), parts=parts, features=feats,
               coverage=cov, dropped=junk)
    json.dump(man, open(out/"manifest.json", "w"), indent=1)
    log(f"wrote {out/'manifest.json'}: {len(order)} parts, {len(feats)} features")
    return 0


if __name__ == "__main__":
    sys.exit(main())
