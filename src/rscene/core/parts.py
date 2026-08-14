"""Parts: the named objects a designer actually works with.

A Wall is two independently fitted faces and a MEASURED distance between them.
It is never a centreline with an assumed thickness -- that assumption is what
the previous pipeline made, and it is why its numbers could not be trusted.

Thickness is measured LOCALLY, in the region where the two faces actually
overlap -- never as a single global centroid-to-centroid offset, and never by
differencing the two faces' `d` values. Real faces are not flat across their
whole extent (measured neighbour-bin correlation +0.59 to +0.83 on a real
scan: genuine slab/wall curvature of several millimetres over metres, not
noise). A single global offset silently inherits that curvature. Binning the
overlap and taking the median of local offsets is immune to it, and the
spread across bins is reported as the measurement's `p95_residual` so a
curved wall says so instead of hiding it in one falsely-precise number.

`d` is origin-referenced; differencing it amplifies normal error by distance
from the origin. `perpendicular_offset` (and the local measurement here)
project the ACTUAL point positions onto the pair's bisector normal instead,
which is invariant to where the world origin happens to sit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .faces import Face
from .fitting import plane_basis
from .graph import perpendicular_offset
from .occupancy import OccupancyGrid, build_occupancy, interior_by_enclosure
from .scene import Measurement


@dataclass
class Wall:
    """One wall: one or two faces, with every dimension measured."""

    wall_id: str
    face_a: int
    face_b: Optional[int]
    thickness: Optional[Measurement]
    length: Measurement
    height: Measurement
    centroid: np.ndarray
    normal: np.ndarray


def _measure_span(f: Face, vertical: bool) -> float:
    """The face's in-plane extent, split into horizontal length and height."""
    u_span = f.u_range[1] - f.u_range[0]
    v_span = f.v_range[1] - f.v_range[0]
    return max(u_span, v_span) if not vertical else min(u_span, v_span)


def _bisector(a: Face, b: Face) -> np.ndarray:
    bisector = a.normal + b.normal
    norm = float(np.linalg.norm(bisector))
    if norm < 1e-9:
        raise ValueError(
            "measure_local_thickness: a.normal and b.normal are opposed; "
            "the bisector is degenerate. Canonically-oriented faces should "
            "never hit this."
        )
    return bisector / norm


def _shared_uv_projection(a: Face, b: Face, xyz: np.ndarray):
    """Project both faces' fitted points into ONE shared (u, v) basis,
    orthogonal to the pair's bisector normal and centred on their midpoint.

    `a` and `b` each fitted their own (u, v) basis independently (see
    `Face.u_range`/`v_range`), and the two need not agree -- comparing them
    directly, or approximating each face's footprint as an isotropic square
    of side `max(u_span, v_span)` (the original approach here), silently
    passes pairs that do not actually overlap. Measured on the real crop: a
    ~0.28 m tall band face (a lintel/sill) squared out to a ~1.2 m box read
    as overlapping a 2.5 m tall wall face it was actually 3-4 cm short of in
    real height -- a false pass on the overlap gate that fed a fallback,
    non-local thickness measurement into the deliverable's headline numbers.
    Both `_overlap_fraction` and `measure_local_thickness` use THIS shared
    projection so the overlap gate and the thickness measurement can never
    disagree about what "overlapping" means.

    Returns (ua, va, ub, vb): each face's points' (u, v) coordinates.
    """
    bisector = _bisector(a, b)
    u, v = plane_basis(bisector)
    origin = (a.centroid + b.centroid) / 2.0

    pa = np.asarray(xyz, dtype=np.float64)[a.point_idx]
    pb = np.asarray(xyz, dtype=np.float64)[b.point_idx]
    rel_a, rel_b = pa - origin, pb - origin
    return rel_a @ u, rel_a @ v, rel_b @ u, rel_b @ v


def _overlap_fraction(a: Face, b: Face, xyz: np.ndarray) -> float:
    """How much two faces' in-plane footprints actually overlap, as a
    fraction of the SMALLER face's own footprint area -- both measured in
    the shared bisector-normal basis (see `_shared_uv_projection`), never
    each face's own independently-fitted basis or an isotropic-square
    approximation.
    """
    ua, va, ub, vb = _shared_uv_projection(a, b, xyz)
    u_lo, u_hi = max(ua.min(), ub.min()), min(ua.max(), ub.max())
    v_lo, v_hi = max(va.min(), vb.min()), min(va.max(), vb.max())
    inter_area = max(0.0, u_hi - u_lo) * max(0.0, v_hi - v_lo)

    area_a = (ua.max() - ua.min()) * (va.max() - va.min())
    area_b = (ub.max() - ub.min()) * (vb.max() - vb.min())
    smaller = min(area_a, area_b)
    return float(inter_area / smaller) if smaller > 0 else 0.0


def measure_local_thickness(
    a: Face, b: Face, xyz: np.ndarray, config: dict
) -> tuple[float, int, float]:
    """Perpendicular distance between two faces, measured IN THEIR OVERLAP.

    A single global centroid-to-centroid offset (`perpendicular_offset`)
    inherits whatever curvature the two surfaces have across their whole
    extent, which real concrete surfaces measurably have (several mm over
    metres). This instead:

      1. Builds a shared (u, v) basis orthogonal to the pair's bisector
         normal -- shared because `a` and `b` each fitted their own basis
         independently and the two need not agree.
      2. Projects both faces' fitted points into that basis, centred on the
         pair's midpoint.
      3. Restricts to the (u, v) BOUNDING BOX both faces actually share --
         the overlap region -- and bins it into `wall_thickness_bin_m`
         cells.
      4. In each cell containing points from both faces, measures the local
         offset as the bisector-projected distance between the two faces'
         mean positions in that cell.
      5. Reports the MEDIAN local offset as the thickness, and the p95 of
         |local offset - median| across cells as `p95_residual` -- the
         measurement's own spread, not a plane-fit residual.

    Falls back to the single centroid-to-centroid `perpendicular_offset` when
    the overlap region or bin population is too small to bin (e.g. the two
    faces barely overlap): a coarse single-cell measurement is still a local
    one, just with one cell.

    Never touches `Face.d`. Everything here is a projection of actual point
    positions onto the bisector, so translating the whole scene changes
    nothing about the result.

    Returns (thickness_m, n_points, p95_residual_m, n_bins). `n_bins` is the
    number of overlap cells that actually qualified (>= 3 points from EACH
    face) and contributed a local measurement -- it is what distinguishes a
    genuinely flat `p95_residual == 0.0` (many qualifying bins, all
    agreeing) from a fallback measurement that never bins at all (`n_bins`
    is 0 when the overlap box is empty, 1 when the box exists but no cell
    reached the per-face minimum so the whole-overlap mean is used instead).
    A caller that wants "was this actually measured locally" should check
    `n_bins >= 2`, not just look at the spread.
    """
    bisector = _bisector(a, b)
    origin = (a.centroid + b.centroid) / 2.0
    pa = np.asarray(xyz, dtype=np.float64)[a.point_idx]
    pb = np.asarray(xyz, dtype=np.float64)[b.point_idx]
    wa = (pa - origin) @ bisector
    wb = (pb - origin) @ bisector
    ua, va, ub, vb = _shared_uv_projection(a, b, xyz)

    u_lo, u_hi = max(ua.min(), ub.min()), min(ua.max(), ub.max())
    v_lo, v_hi = max(va.min(), vb.min()), min(va.max(), vb.max())

    fallback_t = float(abs(wb.mean() - wa.mean()))
    n_pts = int(len(a.point_idx) + len(b.point_idx))
    if u_hi <= u_lo or v_hi <= v_lo:
        return fallback_t, n_pts, 0.0, 0

    cell = max(float(config["wall_thickness_bin_m"]), 1e-6)
    n_u = max(1, int(np.ceil((u_hi - u_lo) / cell)))
    n_v = max(1, int(np.ceil((v_hi - v_lo) / cell)))

    in_a = (ua >= u_lo) & (ua <= u_hi) & (va >= v_lo) & (va <= v_hi)
    in_b = (ub >= u_lo) & (ub <= u_hi) & (vb >= v_lo) & (vb <= v_hi)
    if not in_a.any() or not in_b.any():
        return fallback_t, n_pts, 0.0, 1

    def cell_id(uu, vv):
        iu = np.clip(((uu - u_lo) / cell).astype(np.int64), 0, n_u - 1)
        iv = np.clip(((vv - v_lo) / cell).astype(np.int64), 0, n_v - 1)
        return iu * n_v + iv

    ca = cell_id(ua[in_a], va[in_a])
    cb = cell_id(ub[in_b], vb[in_b])
    wa_ov, wb_ov = wa[in_a], wb[in_b]

    min_bin_pts = 3
    local_vals = []
    used_pts = 0
    shared_cells = np.intersect1d(np.unique(ca), np.unique(cb))
    for c in shared_cells:
        sel_a = ca == c
        sel_b = cb == c
        if sel_a.sum() < min_bin_pts or sel_b.sum() < min_bin_pts:
            continue
        local_vals.append(float(wb_ov[sel_b].mean() - wa_ov[sel_a].mean()))
        used_pts += int(sel_a.sum() + sel_b.sum())

    if not local_vals:
        return fallback_t, n_pts, 0.0, 1

    local_vals = np.abs(np.asarray(local_vals, dtype=np.float64))
    thickness = float(np.median(local_vals))
    spread = float(np.percentile(np.abs(local_vals - thickness), 95)) if len(local_vals) > 1 else 0.0
    return thickness, used_pts, spread, len(local_vals)


def _probe_interior_fraction(
    f: Face, grid: OccupancyGrid, interior: np.ndarray, config: dict, xyz: np.ndarray
) -> tuple[float, float]:
    """Fraction of a deterministic sample of `f`'s own points whose +/-normal
    probe lands in an interior cell. Mirrors `assign_interior_sides`'s probe
    geometry exactly (same step, same up-to-256-point sample), but returns
    the raw per-side fractions instead of collapsing them to a single sign,
    so both sides can be tested independently.
    """
    idx = f.point_idx
    if len(idx) == 0:
        return 0.0, 0.0
    take = idx if len(idx) <= 256 else idx[
        np.linspace(0, len(idx) - 1, 256).astype(np.int64)
    ]
    pts = np.asarray(xyz, dtype=np.float64)[take]
    step = grid.cell_m * float(config["interior_probe_cells"])
    fracs = []
    for sign in (1, -1):
        probe = pts + sign * step * f.normal
        ijk = grid.index_of(probe)
        fracs.append(float(interior[ijk[:, 0], ijk[:, 1], ijk[:, 2]].mean()))
    return fracs[0], fracs[1]


def _is_free_standing_leaf(
    f: Face, grid: OccupancyGrid, interior: np.ndarray, config: dict, xyz: np.ndarray
) -> bool:
    """POSITIVE test for "interior air on both sides" -- not an absence-of-
    signal proxy. A real wall has solid (non-interior) on one side; a
    free-standing leaf (an open door) has interior air on both, because the
    probe reach (`interior_probe_cells` cells) exceeds a leaf's ~40 mm
    thickness and lands in the SAME room's air beyond it on either side.
    Both sides must show a clear MAJORITY of probes landing in interior
    cells (>= 0.5 each) to count -- a face where neither side has any
    interior data at all (both fractions 0, e.g. a face far from any
    computed interior region, or a synthetic fixture with no enclosed room)
    fails this test and remains a normal pairing candidate, so this needs no
    separate opt-in flag the way the old `interior_sign is None` proxy did.
    """
    frac_pos, frac_neg = _probe_interior_fraction(f, grid, interior, config, xyz)
    return frac_pos >= 0.5 and frac_neg >= 0.5


def assemble_walls(
    faces: list[Face], xyz: np.ndarray, config: dict,
    grid: OccupancyGrid | None = None, interior: np.ndarray | None = None,
) -> tuple[list[Wall], list[int]]:
    """Pair opposing wall faces into walls. Returns (walls, unpaired_face_ids).

    A face with no partner still becomes a Wall -- with `face_b` and
    `thickness` set to None. A wall seen from one side is a real wall whose
    thickness we do not know, and saying so is the honest output. Inventing
    a thickness is not.

    A door leaf standing open is a vertical plane (~40 mm thick) that is
    already below `wall_thickness_min_m` and so can never pair -- but each of
    its two faces would otherwise still surface as its own unpaired "wall",
    which is wrong: a leaf has interior air on BOTH sides, a real wall has
    solid on one. `_is_free_standing_leaf` is a POSITIVE test for that (both
    sides probe as majority-interior), not an absence-of-signal proxy --
    `Face.interior_sign is None` also fires for faces `assign_interior_sides`
    simply could not decide (grid-boundary, low probe support), which is a
    different thing and was excluding faces it should not have. A face this
    positive test flags is excluded from PAIRING candidacy only; it still
    surfaces as its own unpaired Wall -- excluding it from the output
    entirely would violate the accounting invariant that every wall face
    ends up in exactly one Wall.

    `grid`/`interior` are optional: pass the caller's own `OccupancyGrid` and
    `interior_by_enclosure` result to avoid recomputing them (the real
    pipeline already has both from Task 5's `assign_interior_sides` stage).
    When omitted, both are built here from `xyz`/`config` -- deterministic,
    and cheap enough for small synthetic fixtures. A scene with no enclosed
    room at all (e.g. a bare wall slab with no floor/ceiling/room around it)
    has essentially no interior cells, so every probe fraction reads ~0 and
    the leaf test never fires -- no separate synthetic-vs-real branch needed.
    """
    cos_tol = float(np.cos(np.radians(config["face_merge_angle_tol_deg"])))
    t_min = float(config["wall_thickness_min_m"])
    t_max = float(config["wall_thickness_max_m"])
    min_overlap = float(config["wall_pair_min_overlap"])

    wall_faces = sorted(
        [f for f in faces if f.role == "wall"], key=lambda f: f.face_id
    )

    if grid is None or interior is None:
        grid = build_occupancy(xyz, config)
        interior = interior_by_enclosure(grid)

    leaf_ids = {
        f.face_id for f in wall_faces
        if _is_free_standing_leaf(f, grid, interior, config, xyz)
    }

    def pairing_eligible(f: Face) -> bool:
        return f.face_id not in leaf_ids

    best: dict[int, tuple[float, int]] = {}
    for i, a in enumerate(wall_faces):
        if not pairing_eligible(a):
            continue
        for b in wall_faces[i + 1:]:
            if not pairing_eligible(b):
                continue
            if abs(float(a.normal @ b.normal)) < cos_tol:
                continue
            coarse = perpendicular_offset(a, b)
            if coarse > 2.0 * t_max:
                continue
            if _overlap_fraction(a, b, xyz) < min_overlap:
                continue
            t, n_pts, resid, n_bins = measure_local_thickness(a, b, xyz, config)
            if not (t_min <= t <= t_max):
                continue
            for x, y in ((a.face_id, b.face_id), (b.face_id, a.face_id)):
                if x not in best or t < best[x][0]:
                    best[x] = (t, y)

    # keep only mutual best matches, so a face cannot belong to two walls
    used: set[int] = set()
    walls: list[Wall] = []
    counter = 0
    by_id = {f.face_id: f for f in wall_faces}

    for f in wall_faces:
        if f.face_id in used:
            continue
        partner = best.get(f.face_id)
        mutual = (
            partner is not None
            and best.get(partner[1], (None, None))[1] == f.face_id
            and partner[1] not in used
        )
        counter += 1
        wid = f"W{counter:02d}"
        if mutual:
            _, other_id = partner
            g = by_id[other_id]
            used.add(f.face_id)
            used.add(other_id)
            t, n_pts, resid, n_bins = measure_local_thickness(f, g, xyz, config)
            walls.append(Wall(
                wall_id=wid, face_a=f.face_id, face_b=other_id,
                thickness=Measurement(
                    value=float(t),
                    method=(
                        "face-to-face perpendicular offset, local overlap "
                        f"median (n_bins={n_bins})"
                    ),
                    n_points=int(n_pts), p95_residual=float(resid),
                ),
                length=Measurement(
                    value=float(max(_measure_span(f, False), _measure_span(g, False))),
                    method="face in-plane extent", n_points=int(f.n_points + g.n_points),
                    p95_residual=float(max(f.p95_residual_m, g.p95_residual_m)),
                ),
                height=Measurement(
                    value=float(max(_measure_span(f, True), _measure_span(g, True))),
                    method="face in-plane extent", n_points=int(f.n_points + g.n_points),
                    p95_residual=float(max(f.p95_residual_m, g.p95_residual_m)),
                ),
                centroid=(f.centroid + g.centroid) / 2.0,
                normal=f.normal.copy(),
            ))
        else:
            used.add(f.face_id)
            walls.append(Wall(
                wall_id=wid, face_a=f.face_id, face_b=None, thickness=None,
                length=Measurement(
                    value=float(_measure_span(f, False)),
                    method="face in-plane extent", n_points=f.n_points,
                    p95_residual=f.p95_residual_m,
                ),
                height=Measurement(
                    value=float(_measure_span(f, True)),
                    method="face in-plane extent", n_points=f.n_points,
                    p95_residual=f.p95_residual_m,
                ),
                centroid=f.centroid.copy(),
                normal=f.normal.copy(),
            ))

    unpaired = sorted(w.face_a for w in walls if w.face_b is None)
    return walls, unpaired
