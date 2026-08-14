"""Parts: the named objects a designer actually works with.

A Wall is two independently fitted faces and a MEASURED distance between them.
It is never a centreline with an assumed thickness -- that assumption is what
the previous pipeline made, and it is why its numbers could not be trusted.

Thickness is measured LOCALLY, in the region where the two faces actually
overlap -- never as a single global centroid-to-centroid offset, and never by
differencing the two faces' `d` values. Real faces are not flat across their
whole extent (measured neighbour-bin correlation +0.59 to +0.83 on a real
scan: genuine slab/wall curvature of several millimetres over metres, not
noise). A single global offset silently inherits that curvature.

Thickness also genuinely VARIES, and not as noise around a mean: a wall runs
at one thickness, steps where a structural column is embedded in it, and
steps back. Binning the overlap and SEGMENTING the bins along the wall's
length (see `ThicknessSegment`) captures that directly -- where the step is,
not a spread number that would describe it as smearing. `Wall.thickness` is
the dominant segment's own median; `Wall.thickness_field` is the full
segmented field for anything that needs to see the step.

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


#: Two or more overlap cells qualified (>= 3 points from EACH face); the
#: reported thickness is a genuine median across independent local samples.
METHOD_LOCAL_BIN_MEDIAN = "local-bin-median"
#: Exactly one overlap cell qualified. A real local measurement -- not a
#: whole-overlap average -- but there is only one of it; no spread to trust.
METHOD_LOCAL_SINGLE_BIN = "local-single-bin"
#: No cell qualified (empty overlap box, no points inside it, or every cell
#: fell short of the per-face minimum). The reported value is the mean
#: offset over the WHOLE overlap (or, if the overlap box itself was empty,
#: the whole face) -- i.e. exactly the non-local measurement this module
#: exists to avoid. Visibly labelled as such rather than silently returned
#: as if it were a bin result.
METHOD_WHOLE_OVERLAP_FALLBACK = "whole-overlap-fallback"


@dataclass
class ThicknessSample:
    """One local thickness measurement: a bin centre (in the pair's shared
    (u, v) basis -- see `_shared_uv_projection`) plus what was measured
    there. For a `METHOD_WHOLE_OVERLAP_FALLBACK` result there is exactly one
    sample, positioned at the overlap region's (or, lacking one, the whole
    pair's) centre -- honestly one giant "bin".
    """

    u: float
    v: float
    value: float
    n_points: int

    def to_dict(self) -> dict:
        return {"u": self.u, "v": self.v, "value": self.value, "n_points": self.n_points}

    @staticmethod
    def from_dict(payload: dict) -> "ThicknessSample":
        return ThicknessSample(
            u=payload["u"], v=payload["v"], value=payload["value"],
            n_points=payload["n_points"],
        )


@dataclass
class ThicknessSegment:
    """A contiguous run of bins along the wall's LONGER shared-basis axis
    (an approximation of "along its length": whichever of the pair's shared
    u/v extents is larger) whose thickness agrees within
    `wall_thickness_step_tol_m`.

    Wall thickness is not noise around a mean -- it is PIECEWISE: a wall
    runs at one thickness, steps where a structural column is embedded in
    it, and steps back. A summary statistic (min/max/percentile) describes
    that bimodal reality as if it were a spread and hides the one thing that
    matters: where the step is. Segments say that directly: "244 mm from
    u=0.00 to 1.20, then 451 mm from u=1.20 to 1.60 (candidate column), then
    244 mm again."

    `start`/`end` are positions along the length axis in the pair's shared
    basis (metres, relative to the pair's midpoint -- see
    `_shared_uv_projection`). `thickness` is the median of the segment's own
    bin values. `is_candidate_column` marks a segment thicker than the
    DOMINANT segment (the one covering the most length) by more than the
    step tolerance -- flagged, not classified: turning it into an actual
    `Column` part needs the column's other faces, which is Plan 3's job.
    """

    start: float
    end: float
    thickness: float
    n_bins: int
    n_points: int
    is_candidate_column: bool

    def to_dict(self) -> dict:
        return {
            "start": self.start, "end": self.end, "thickness": self.thickness,
            "n_bins": self.n_bins, "n_points": self.n_points,
            "is_candidate_column": self.is_candidate_column,
        }

    @staticmethod
    def from_dict(payload: dict) -> "ThicknessSegment":
        return ThicknessSegment(
            start=payload["start"], end=payload["end"], thickness=payload["thickness"],
            n_bins=payload["n_bins"], n_points=payload["n_points"],
            is_candidate_column=payload["is_candidate_column"],
        )


@dataclass
class ThicknessField:
    """The thickness FIELD behind a wall's single headline number: every
    local sample that went into it (the raw material), plus that field
    SEGMENTED along the wall's length (the structure). A wall's thickness
    genuinely varies for a structural reason -- a column embedded in it --
    not as noise around a mean; collapsing that to one median and a spread
    number would describe a step as if it were smearing, and hide exactly
    where a designer cannot route a conduit or clear joinery. See
    `ThicknessSegment`.

    `segments` always has at least one entry, covering the whole field --
    the ordinary case of a wall with no column in it is correctly reported
    as a single segment, not a missing or degenerate result.
    """

    samples: list[ThicknessSample]
    segments: list[ThicknessSegment]

    def to_dict(self) -> dict:
        return {
            "samples": [s.to_dict() for s in self.samples],
            "segments": [s.to_dict() for s in self.segments],
        }

    @staticmethod
    def from_dict(payload: dict) -> "ThicknessField":
        return ThicknessField(
            samples=[ThicknessSample.from_dict(s) for s in payload["samples"]],
            segments=[ThicknessSegment.from_dict(s) for s in payload["segments"]],
        )


@dataclass
class Wall:
    """One wall: one or two faces, with every dimension measured.

    `thickness` is the DOMINANT segment's median (the segment covering the
    most length) -- the wall's own thickness, not an average contaminated by
    an embedded column. `thickness_field` is the evidence behind it: every
    local sample and the segments they were grouped into, for anything that
    wants to see how thickness actually varies along the wall. `None` on
    both when the wall is unpaired (no partner, no measurement to report).
    """

    wall_id: str
    face_a: int
    face_b: Optional[int]
    thickness: Optional[Measurement]
    thickness_field: Optional[ThicknessField]
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
) -> tuple[Measurement, ThicknessField]:
    """The thickness FIELD between two faces, measured IN THEIR OVERLAP --
    not one number, but the per-bin measurements it was built from.

    A single global centroid-to-centroid offset (`perpendicular_offset`)
    inherits whatever curvature the two surfaces have across their whole
    extent, which real concrete surfaces measurably have (several mm over
    metres), and collapses genuine along-the-wall/up-the-wall variation
    (plaster and masonry vary for different reasons) into one falsely
    precise number. This instead:

      1. Builds a shared (u, v) basis orthogonal to the pair's bisector
         normal -- shared because `a` and `b` each fitted their own basis
         independently and the two need not agree (see
         `_shared_uv_projection`).
      2. Restricts to the (u, v) BOUNDING BOX both faces actually share --
         the overlap region.
      3. Bins it ADAPTIVELY: cell size targets
         `wall_thickness_bin_target_pts` points per bin from the SPARSER
         face's overlap population, clipped to
         [`wall_thickness_bin_min_m`, `wall_thickness_bin_max_m`]. A fixed
         cell size cannot serve both a dense, metres-wide overlap (which
         should resolve finely) and a sparse one (which should degrade
         gracefully to few bins rather than fragment into empty ones).
      4. In each cell with >= 3 points from EACH face, measures the local
         offset as the bisector-projected distance between the two faces'
         mean positions in that cell -- one `ThicknessSample`, positioned at
         the cell centre.
      5. SEGMENTS the bins along the LONGER shared-basis axis (an
         approximation of "along the wall's length"): adjacent bins whose
         median thickness agrees within `wall_thickness_step_tol_m` merge
         into one `ThicknessSegment`; a jump beyond it starts a new one.
         Wall thickness is piecewise, not noise around a mean -- a wall runs
         at one thickness, steps where a structural column is embedded in
         it, and steps back -- so this reports where the step is rather
         than a spread number that would describe a genuine step as if it
         were smearing. See `ThicknessSegment`.
      6. Reports the DOMINANT segment (the one covering the most length) as
         the headline `Measurement` -- the wall's own thickness, not an
         average contaminated by an embedded column -- with the FULL
         segmented field on the returned `ThicknessField`.

    `Measurement.method` is one of three explicit values -- never a string a
    caller has to parse to learn what happened -- and describes the DOMINANT
    segment specifically:
      - `METHOD_LOCAL_BIN_MEDIAN`   -- the dominant segment has >= 2 cells;
        trustworthy.
      - `METHOD_LOCAL_SINGLE_BIN`   -- the dominant segment has exactly 1
        cell; a genuine local measurement, but there is only one of it.
      - `METHOD_WHOLE_OVERLAP_FALLBACK` -- no cell qualified anywhere (empty
        overlap box, no points inside it, or every cell under-populated).
        The reported value is a plain mean over the whole overlap (or,
        lacking an overlap box, the whole pair) -- NOT a local measurement,
        and `Measurement.n_bins == 0` says so explicitly rather than being
        conflated with a genuine single-bin result the way an earlier
        version of this function did.

    Never touches `Face.d`. Everything here is a projection of actual point
    positions onto the bisector, so translating the whole scene changes
    nothing about the result.
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
    fallback_uv = (
        float((ua.mean() + ub.mean()) / 2.0),
        float((va.mean() + vb.mean()) / 2.0),
    )

    def fallback_result() -> tuple[Measurement, ThicknessField]:
        sample = ThicknessSample(u=fallback_uv[0], v=fallback_uv[1], value=fallback_t, n_points=n_pts)
        segment = ThicknessSegment(
            start=fallback_uv[0], end=fallback_uv[0], thickness=fallback_t,
            n_bins=0, n_points=n_pts, is_candidate_column=False,
        )
        field = ThicknessField(samples=[sample], segments=[segment])
        measurement = Measurement(
            value=fallback_t, method=METHOD_WHOLE_OVERLAP_FALLBACK,
            n_points=n_pts, p95_residual=0.0, n_bins=0,
        )
        return measurement, field

    if u_hi <= u_lo or v_hi <= v_lo:
        return fallback_result()

    in_a = (ua >= u_lo) & (ua <= u_hi) & (va >= v_lo) & (va <= v_hi)
    in_b = (ub >= u_lo) & (ub <= u_hi) & (vb >= v_lo) & (vb <= v_hi)
    if not in_a.any() or not in_b.any():
        return fallback_result()

    u_span, v_span = u_hi - u_lo, v_hi - v_lo
    sparser = min(int(in_a.sum()), int(in_b.sum()))
    bin_min = float(config["wall_thickness_bin_min_m"])
    bin_max = float(config["wall_thickness_bin_max_m"])
    target_pts = float(config["wall_thickness_bin_target_pts"])
    overlap_area = u_span * v_span
    if sparser > 0 and overlap_area > 0:
        ideal_cell = float(np.sqrt(overlap_area * target_pts / sparser))
        cell = min(max(ideal_cell, bin_min), bin_max)
    else:
        cell = bin_max

    n_u = max(1, int(np.ceil(u_span / cell)))
    n_v = max(1, int(np.ceil(v_span / cell)))
    step_u, step_v = u_span / n_u, v_span / n_v

    def cell_id(uu, vv):
        iu = np.clip(((uu - u_lo) / step_u).astype(np.int64), 0, n_u - 1)
        iv = np.clip(((vv - v_lo) / step_v).astype(np.int64), 0, n_v - 1)
        return iu * n_v + iv

    ca = cell_id(ua[in_a], va[in_a])
    cb = cell_id(ub[in_b], vb[in_b])
    wa_ov, wb_ov = wa[in_a], wb[in_b]

    min_bin_pts = 3
    samples: list[ThicknessSample] = []
    # (iu, iv, value, n_points) per qualifying cell -- kept alongside `samples`
    # for segmentation below, which groups by bin INDEX along the length
    # axis, not by the samples' float positions.
    records: list[tuple[int, int, float, int]] = []
    shared_cells = np.intersect1d(np.unique(ca), np.unique(cb))
    for c in shared_cells:
        sel_a = ca == c
        sel_b = cb == c
        if sel_a.sum() < min_bin_pts or sel_b.sum() < min_bin_pts:
            continue
        iu, iv = divmod(int(c), n_v)
        u_center = u_lo + (iu + 0.5) * step_u
        v_center = v_lo + (iv + 0.5) * step_v
        value = abs(float(wb_ov[sel_b].mean() - wa_ov[sel_a].mean()))
        n_bin_pts = int(sel_a.sum() + sel_b.sum())
        samples.append(ThicknessSample(u=u_center, v=v_center, value=value, n_points=n_bin_pts))
        records.append((iu, iv, value, n_bin_pts))

    if not samples:
        return fallback_result()

    step_tol = float(config["wall_thickness_step_tol_m"])
    length_is_u = u_span >= v_span

    # Group records by their index along the LONGER axis (an approximation
    # of "along the wall's length"), aggregating the perpendicular axis --
    # a step spans the wall's full height, not one height-bin of it.
    groups: dict[int, list[tuple[int, int, float, int]]] = {}
    for rec in records:
        key = rec[0] if length_is_u else rec[1]
        groups.setdefault(key, []).append(rec)
    keys = sorted(groups.keys())

    def group_median(k: int) -> float:
        return float(np.median([r[2] for r in groups[k]]))

    def key_extent(k: int) -> tuple[float, float]:
        if length_is_u:
            return u_lo + k * step_u, u_lo + (k + 1) * step_u
        return v_lo + k * step_v, v_lo + (k + 1) * step_v

    def finalise(run_keys: list[int]) -> ThicknessSegment:
        recs = [r for k in run_keys for r in groups[k]]
        vals = np.array([r[2] for r in recs], dtype=np.float64)
        starts = [key_extent(k)[0] for k in run_keys]
        ends = [key_extent(k)[1] for k in run_keys]
        return ThicknessSegment(
            start=float(min(starts)), end=float(max(ends)),
            thickness=float(np.median(vals)),
            n_bins=len(recs), n_points=int(sum(r[3] for r in recs)),
            is_candidate_column=False,
        )

    # Adjacent groups whose median agrees within `step_tol` merge into one
    # segment; a jump beyond it starts a new one. Comparison is against the
    # immediately preceding GROUP's median (not a running segment average),
    # so a slow drift across many groups cannot silently smuggle a step
    # past the tolerance by accumulating it one small hop at a time.
    segments: list[ThicknessSegment] = []
    run = [keys[0]]
    prev_median = group_median(keys[0])
    for k in keys[1:]:
        med = group_median(k)
        if abs(med - prev_median) <= step_tol:
            run.append(k)
        else:
            segments.append(finalise(run))
            run = [k]
        prev_median = med
    segments.append(finalise(run))

    dominant = max(segments, key=lambda s: s.end - s.start)
    for seg in segments:
        if seg is not dominant and seg.thickness > dominant.thickness + step_tol:
            seg.is_candidate_column = True

    dominant_keys = [k for k in keys if key_extent(k)[0] >= dominant.start and key_extent(k)[1] <= dominant.end]
    dominant_vals = np.array(
        [r[2] for k in dominant_keys for r in groups[k]], dtype=np.float64
    )
    spread = (
        float(np.percentile(np.abs(dominant_vals - dominant.thickness), 95))
        if len(dominant_vals) > 1 else 0.0
    )
    method = METHOD_LOCAL_SINGLE_BIN if dominant.n_bins == 1 else METHOD_LOCAL_BIN_MEDIAN

    field = ThicknessField(samples=samples, segments=segments)
    measurement = Measurement(
        value=dominant.thickness, method=method,
        n_points=dominant.n_points, p95_residual=spread, n_bins=dominant.n_bins,
    )
    return measurement, field


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
            measurement, _field = measure_local_thickness(a, b, xyz, config)
            t = measurement.value
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
            measurement, field = measure_local_thickness(f, g, xyz, config)
            walls.append(Wall(
                wall_id=wid, face_a=f.face_id, face_b=other_id,
                thickness=measurement,
                thickness_field=field,
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
                thickness_field=None,
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
