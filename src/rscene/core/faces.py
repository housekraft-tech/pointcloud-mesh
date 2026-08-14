"""Faces: patches merged into the surfaces they actually belong to.

Plan 1 deliberately let one physical surface split into several patches --
region growing stops at anything it cannot cross, and a real scan of one room
produced 200 patches where perhaps 20 surfaces exist. Merging is how that is
repaired, and it is the one place in this pipeline where two surfaces are
allowed to become one, so the conditions are strict:

    coplanar within tolerance  AND  spatially adjacent

Coplanarity alone is not enough -- two walls on opposite sides of a building
can share a plane and are obviously not one surface. Adjacency alone is not
enough either, or a 75 mm step would merge into the wall behind it.

The merge tolerance is bounded above by the shallowest feature that must
survive. The golden room's groove is 12 mm deep, so the default sits at 5 mm.
Raising it past the groove depth erases the groove -- the exact failure this
rebuild exists to fix.

The merged plane is refitted from the union of the members' own points. It is
never an average of the input planes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from .fitting import fit_plane, plane_basis, plane_distance
from .patches import Patch


@dataclass
class Face:
    """One physical surface, backed by one or more patches."""

    face_id: int
    normal: np.ndarray                 # (3,) unit, canonically oriented
    d: float                           # normal @ x + d == 0 (ORIGIN-referenced)
    patch_ids: list[int]               # contributing patches, sorted
    point_idx: np.ndarray              # fitted members (drive the plane)
    loose_idx: np.ndarray              # recruited members (Task 3); never fitted
    n_points: int                      # len(point_idx)
    p95_residual_m: float              # over fitted members
    centroid: np.ndarray               # (3,)
    u_range: tuple[float, float]
    v_range: tuple[float, float]
    role: Optional[str] = None         # set by classify (Task 6)
    interior_sign: Optional[int] = None  # set by occupancy (Task 5): +1 or -1

    def area_bound_m2(self) -> float:
        return (self.u_range[1] - self.u_range[0]) * (self.v_range[1] - self.v_range[0])

    def all_idx(self) -> np.ndarray:
        """Fitted plus recruited members, sorted. Every point this face owns."""
        return np.sort(np.concatenate([self.point_idx, self.loose_idx]))


def _bbox(xyz: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = xyz[idx]
    return pts.min(axis=0), pts.max(axis=0)


def _adjacent(xyz, a: Patch, b: Patch, gap: float) -> bool:
    """True when any point of a lies within `gap` of any point of b.

    A bounding-box test inflated by `gap` prunes the pair first; it can only
    reject pairs that are genuinely too far apart, never a true neighbour.
    """
    lo_a, hi_a = _bbox(xyz, a.point_idx)
    lo_b, hi_b = _bbox(xyz, b.point_idx)
    if np.any(lo_a - gap > hi_b) or np.any(lo_b - gap > hi_a):
        return False
    return cKDTree(xyz[a.point_idx]).count_neighbors(cKDTree(xyz[b.point_idx]), gap) > 0


def _finalise(face_id: int, xyz: np.ndarray, members: np.ndarray,
              patch_ids: list[int]) -> Face:
    pts = xyz[members]
    normal, d = fit_plane(pts)
    residual = np.abs(plane_distance(pts, normal, d))
    u, v = plane_basis(normal)
    centroid = pts.mean(axis=0)
    rel = pts - centroid
    us, vs = rel @ u, rel @ v
    return Face(
        face_id=face_id,
        normal=normal,
        d=d,
        patch_ids=sorted(patch_ids),
        point_idx=np.sort(members),
        loose_idx=np.zeros(0, dtype=np.int64),
        n_points=int(len(members)),
        p95_residual_m=float(np.percentile(residual, 95)),
        centroid=centroid,
        u_range=(float(us.min()), float(us.max())),
        v_range=(float(vs.min()), float(vs.max())),
    )


def merge_patches(patches: list[Patch], xyz: np.ndarray, config: dict) -> list[Face]:
    """Group patches into faces. Returns faces ordered by descending point count.

    Grouping is GREEDY SEEDED ACCUMULATION against the group's own fitted
    plane, not pairwise union-find over neighbours. Union-find only ever
    checks adjacent pairs: A joins B, B joins C, C joins D, each link within
    tolerance, and nothing stops A and D from ending up far apart -- a chain
    can drift arbitrarily far past `face_merge_dist_tol_m` even though every
    individual link obeyed it. Measured on a real room crop this put patches
    41 mm apart inside a single face against a 5 mm tolerance.

    Algorithm:
      1. Order patches by descending point count -- the largest is the most
         reliable plane estimate.
      2. Seed a new face with the largest unassigned patch; its fitted plane
         is the group plane.
      3. Repeatedly scan remaining unassigned patches for candidates whose
         normal agrees with the GROUP plane, whose centroid is within
         `face_merge_dist_tol_m` of the GROUP plane (never a neighbour's
         plane), and which are spatially adjacent to at least one patch
         already in the group.
      4. Add the best candidate (smallest offset to the group plane, ties
         broken by lowest patch_id), refit the group plane over the union of
         members' points, and repeat.
      5. Stop when nothing qualifies; start a new face from the largest
         remaining unassigned patch.

    Every member therefore ends up within tolerance of the face's OWN plane,
    which bounds total drift by the tolerance rather than by chain length.

    Refitting after every addition is O(n^2) in group size in the worst
    case; fine for the tens-of-patches groups seen so far.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    dist_tol = float(config["face_merge_dist_tol_m"])
    cos_tol = float(np.cos(np.radians(config["face_merge_angle_tol_deg"])))
    gap = float(config["face_merge_gap_m"])

    by_id = {p.patch_id: p for p in patches}
    priority = sorted(patches, key=lambda p: (-p.n_points, p.patch_id))
    unassigned = {p.patch_id: p for p in priority}

    groups: list[list[int]] = []

    while unassigned:
        seed = next(p for p in priority if p.patch_id in unassigned)
        group_ids = [seed.patch_id]
        del unassigned[seed.patch_id]
        group_normal, group_d = seed.normal, seed.d
        group_members = seed.point_idx

        while True:
            prelim = []  # (offset, patch_id, patch) candidates against the CURRENT group plane
            for pid, p in unassigned.items():
                if abs(float(p.normal @ group_normal)) < cos_tol:
                    continue
                offset = abs(float(group_normal @ p.centroid + group_d))
                if offset > dist_tol:
                    continue
                if not any(_adjacent(xyz, by_id[gid], p, gap) for gid in group_ids):
                    continue
                prelim.append((offset, pid, p))
            prelim.sort(key=lambda c: (c[0], c[1]))

            # A candidate that looks close to the OLD group plane can still pull
            # the refit plane away from an existing member -- e.g. a group that
            # already spans a bend, where the next step continues the bend past
            # what a single plane can explain. Refitting alone does not catch
            # that: it only checks the new point going in, never re-validates
            # points already accepted. So every candidate is tried by refitting
            # the WHOLE enlarged group and re-checking every member (old and
            # new) against that refit plane; the first one that leaves the
            # entire group within tolerance is accepted. This keeps the
            # invariant -- every member within tolerance of the face's own
            # plane -- true after every single addition, not just at the end.
            accepted = None
            for offset, pid, p in prelim:
                trial_members = np.concatenate([group_members, p.point_idx])
                trial_normal, trial_d = fit_plane(xyz[trial_members])
                ok = True
                for gid in (*group_ids, pid):
                    gp = by_id[gid]
                    if abs(float(gp.normal @ trial_normal)) < cos_tol:
                        ok = False
                        break
                    if abs(float(trial_normal @ gp.centroid + trial_d)) > dist_tol:
                        ok = False
                        break
                if ok:
                    accepted = (pid, trial_members, trial_normal, trial_d)
                    break
            if accepted is None:
                break
            best_pid, group_members, group_normal, group_d = accepted
            group_ids.append(best_pid)
            del unassigned[best_pid]

        groups.append(group_ids)

    faces = []
    for group_ids in groups:
        members = np.concatenate([by_id[pid].point_idx for pid in group_ids])
        faces.append(_finalise(0, xyz, members, group_ids))

    faces.sort(key=lambda f: (-f.n_points, f.patch_ids[0]))
    for new_id, f in enumerate(faces):
        f.face_id = new_id
    return faces


def median_spacing(xyz: np.ndarray, sample_n: int = 50_000, seed: int = 0) -> float:
    """Median nearest-neighbour distance -- the cloud's native point spacing.

    The KD-tree MUST be built on the full cloud: the tree defines what "the
    neighbourhood" is, and thinning the cloud before building it inflates
    nearest-neighbour distances by the thinning factor -- a 50k sample of a
    1M-point cloud measures the SAMPLE's spacing, not the cloud's, and returns
    a near-constant figure regardless of how dense the real cloud is.

    `sample_n` instead bounds which points are QUERIED against that full
    tree. This keeps the cost close to the sampled version (only the tree
    build is full-size; the query is bounded by sample_n) while returning the
    cloud's true spacing. Seeded for determinism.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if len(xyz) < 2:
        return 0.0
    tree = cKDTree(xyz)
    if len(xyz) > sample_n:
        rng = np.random.default_rng(seed)
        query = xyz[np.sort(rng.choice(len(xyz), sample_n, replace=False))]
    else:
        query = xyz
    dist, _ = tree.query(query, k=2, workers=4)
    return float(np.median(dist[:, 1]))


def _face_coverage(xyz: np.ndarray, f: Face, cell_mult: float, seed: int) -> float:
    """Fraction of the face's own in-plane grid cells that hold >=1 point.

    The grid cell size is `cell_mult` times the face's OWN median spacing
    (measured from the face's own points, not the whole-cloud spacing) --
    that is what makes this a COVERAGE metric rather than a DENSITY metric.
    A genuine surface, however sparse the scan made it, is dense relative to
    ITS OWN spacing and so fills nearly every one of its own cells. A
    scattered chain threaded together by the patch connect radius leaves
    most cells empty no matter how the cell size is chosen, because its
    points are far apart relative to each other, not just relative to some
    global reference. This is what makes the metric invariant to range and
    incidence-angle density falloff, which a global-spacing fill ratio is not.
    """
    pts = xyz[f.point_idx]
    if len(pts) < 2:
        return 0.0
    own_spacing = median_spacing(pts, seed=seed)
    if own_spacing <= 0:
        return 1.0  # degenerate (e.g. duplicate points): treat as fully covered

    u, v = plane_basis(f.normal)
    rel = pts - f.centroid
    us, vs = rel @ u, rel @ v
    u_lo, u_hi = np.percentile(us, [2, 98])
    v_lo, v_hi = np.percentile(vs, [2, 98])

    cell = cell_mult * own_spacing
    n_u = max(1, int(np.ceil((u_hi - u_lo) / cell))) if u_hi > u_lo else 1
    n_v = max(1, int(np.ceil((v_hi - v_lo) / cell))) if v_hi > v_lo else 1

    in_range = (us >= u_lo) & (us <= u_hi) & (vs >= v_lo) & (vs <= v_hi)
    us, vs = us[in_range], vs[in_range]
    if len(us) == 0:
        return 0.0

    ui = np.clip(((us - u_lo) / cell).astype(np.int64), 0, n_u - 1)
    vi = np.clip(((vs - v_lo) / cell).astype(np.int64), 0, n_v - 1)
    occupied = len(np.unique(ui * n_v + vi))
    return occupied / float(n_u * n_v)


def apply_density_gate(
    faces: list[Face], xyz: np.ndarray, config: dict
) -> tuple[list[Face], list[Face]]:
    """Split faces into (kept, rejected) on in-plane grid COVERAGE and area.

    Coverage answers "are these points a coherent surface, or a scattered
    chain?" -- a question about spatial layout, not about how many points
    there are. It grids the face's own in-plane extent into cells sized off
    the FACE'S OWN median spacing (see `_face_coverage`) and measures the
    fraction of cells holding at least one point.

    This replaced a global-density fill ratio (`n_points / (area / spacing^2)`
    against one whole-cloud `median_spacing`). That metric could not tell a
    real, sparse, far-from-scanner wall from actual junk: SLAM point density
    varies with range and incidence angle by a factor of eight or more on a
    single real scan, so a genuinely dense wall seen obliquely and a
    deliberately scattered chain can land on the same fill ratio. Coverage
    against the face's own spacing does not have this failure mode -- a real
    surface is dense relative to itself no matter how sparse the scan made
    it, so it still fills nearly all of its own cells; a scattered chain does
    not, regardless of overall density. See
    `.superpowers/sdd/2026-08-13-rectilinear-scene-plan2-parts/median-spacing-fix.md`
    for the real-crop numbers that forced this: at correct global spacing,
    the density-fill gate rejected a genuine 20.4 sq m, 64k-point wall (fill
    0.116) alongside actual junk (a synthetic sparse chain at fill 0.121) --
    the two were indistinguishable on that metric.

    The extent used here is a 2nd-98th percentile TRIM of the fitted members'
    in-plane coordinates, deliberately different from `area_bound_m2()` (a plain
    min/max over `u_range`/`v_range`). Region growing only requires 50 mm
    connectivity and 3 mm planarity, so a single stray member dragged a metre out
    inflates the plain bbox -- easily enough to flip a genuinely dense wall from
    kept to rejected, which is a worse failure than admitting junk (the gate
    exists to stop concrete dust becoming objects, not to delete walls).
    `u_range`/`v_range`/`area_bound_m2()` on the Face itself are left untouched
    -- they are the face's true reported extent for downstream consumers -- so
    do not "helpfully" make the two consistent.

    Coverage is measured over `f.point_idx` (fitted members only), never
    `f.all_idx()`. From Task 3 onward `loose_idx` holds recruited members
    attached under a looser tolerance than the fitted plane; deliberately, they
    cannot rescue a face whose own fitted points do not already support it.

    Rejected faces are RETURNED, never dropped. The caller routes them to the
    scene's `unmodeled` set so a designer still sees that something is there.
    """
    min_coverage = float(config["face_min_coverage"])
    min_area = float(config["face_min_area_m2"])
    cell_mult = float(config["face_coverage_cell_spacing_mult"])
    seed = int(config["seed"])

    kept, rejected = [], []
    for f in sorted(faces, key=lambda g: g.face_id):
        area = f.area_bound_m2()
        if area < min_area:
            rejected.append(f)
            continue
        coverage = _face_coverage(xyz, f, cell_mult, seed)
        (kept if coverage >= min_coverage else rejected).append(f)
    return kept, rejected


_RECRUIT_CHUNK = 20_000


def recruit_points(
    faces: list[Face],
    xyz: np.ndarray,
    normals: np.ndarray,
    labels: np.ndarray,
    config: dict,
) -> np.ndarray:
    """Attach leftover points to the face they lie on. Returns new labels.

    On a real scan most unassigned points are not unexplained -- they sit ON a
    face's plane and region growing simply could not reach them. Recruiting
    them is the largest single reduction in unassigned points available.

    Recruits go into `Face.loose_idx` and are EXCLUDED from the plane fit, so a
    looser tolerance costs accounting completeness nothing in accuracy. A recruit
    must also be within `recruit_max_reach_m` of an existing fitted member, so a
    point cannot join a face across a void it has no business crossing.

    Ties are broken by nearest plane, then lowest face_id -- deterministic.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    labels = np.asarray(labels).copy()
    if not faces:
        return labels

    tau = float(config["recruit_dist_tol_m"])
    cos_tol = float(np.cos(np.radians(config["recruit_angle_tol_deg"])))
    reach = float(config["recruit_max_reach_m"])

    ordered = sorted(faces, key=lambda f: f.face_id)
    N = np.vstack([f.normal for f in ordered])           # (F, 3)
    D = np.array([f.d for f in ordered])                 # (F,)
    trees = [cKDTree(xyz[f.point_idx]) for f in ordered]

    free = np.flatnonzero(labels < 0)
    claimed: dict[int, list[int]] = {f.face_id: [] for f in ordered}

    for start in range(0, len(free), _RECRUIT_CHUNK):
        idx = free[start:start + _RECRUIT_CHUNK]
        P = xyz[idx]
        dists = np.abs(P @ N.T + D)                      # (n, F)
        agrees = np.abs(normals[idx] @ N.T) >= cos_tol   # (n, F)
        ok = agrees & (dists <= tau)

        cand = np.where(ok, dists, np.inf)
        best = np.argmin(cand, axis=1)
        best_dist = cand[np.arange(len(idx)), best]
        viable = np.isfinite(best_dist)

        # Reach is checked one face at a time, batching every viable candidate
        # assigned to that face into a single cKDTree query. A per-point query
        # loop here scales with candidate count -- on a real scan that is
        # ~17k individual calls, the same shape that cost Plan 1 161 s.
        viable_local = np.flatnonzero(viable)
        fi_viable = best[viable_local]
        gi_viable = idx[viable_local]
        pts_viable = P[viable_local]

        for fi in np.unique(fi_viable):
            sel = fi_viable == fi
            gi_f = gi_viable[sel]
            dist_to_member, _ = trees[int(fi)].query(pts_viable[sel], k=1, workers=-1)
            face_id = ordered[int(fi)].face_id
            for gi in gi_f[dist_to_member <= reach]:
                gi = int(gi)
                claimed[face_id].append(gi)
                labels[gi] = face_id

    for f in ordered:
        got = claimed[f.face_id]
        f.loose_idx = np.sort(np.asarray(got, dtype=np.int64)) if got \
            else np.zeros(0, dtype=np.int64)
    return labels
