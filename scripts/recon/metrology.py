"""Sub-millimetre metrology: robust face/plane estimation from raw points.

Everything here is pure numpy and unit-testable. The rule this module exists
to enforce: *rasters decide topology, raw points decide distance*. No function
here may quantize a measurement to a grid.

The estimators are M-estimators (Huber IRLS) rather than plain means so that a
skirting board, a picture frame, or a few stray SLAM-drift points near a wall
face cannot drag the face position. Each returns its own uncertainty, because a
millimetre-level number without an error bar is not a millimetre-level claim.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

# Huber tuning constant: 95% efficiency vs the mean under clean Gaussian noise,
# while bounding the influence of anything beyond ~1.3 robust sigma.
HUBER_K = 1.345

# Floor on the robust scale estimate (metres). Guards the degenerate case where
# >50% of samples land on one exact value and MAD collapses to 0.
_SIGMA_FLOOR = 1e-5


class Location(NamedTuple):
    """A 1-D robust location estimate with its uncertainty (metres)."""

    value: float
    sigma: float        # robust scale of the residuals (surface roughness + noise)
    stderr: float       # standard error of `value` itself
    n: int              # samples inside the final window


class PlaneFit(NamedTuple):
    """A robust total-least-squares plane fit."""

    normal: np.ndarray  # unit normal
    offset: float       # d, such that the plane is {p : p @ normal == d}
    sigma: float        # robust RMS of point-to-plane residuals
    stderr: float       # standard error of `offset`
    n: int              # inliers


def robust_location(vals, k: float = HUBER_K, iters: int = 12) -> Location:
    """Huber M-estimate of the location of a 1-D sample.

    Starts from the median (breakdown 50%) and reweights toward the mean, so
    the result keeps the median's resistance to outliers but the mean's
    efficiency -- which is what buys the sub-millimetre standard error once a
    face carries thousands of points.
    """
    v = np.asarray(vals, dtype=float).ravel()
    v = v[np.isfinite(v)]
    n = v.size
    if n == 0:
        return Location(float("nan"), float("nan"), float("nan"), 0)
    if n == 1:
        return Location(float(v[0]), 0.0, float("inf"), 1)

    mu = float(np.median(v))
    sigma = max(1.4826 * float(np.median(np.abs(v - mu))), _SIGMA_FLOOR)

    w = np.ones(n)
    for _ in range(iters):
        r = np.abs(v - mu) / sigma
        w = np.where(r <= k, 1.0, k / np.maximum(r, 1e-12))
        new_mu = float(np.sum(w * v) / np.sum(w))
        if abs(new_mu - mu) < 1e-12:
            mu = new_mu
            break
        mu = new_mu
        sigma = max(1.4826 * float(np.median(np.abs(v - mu))), _SIGMA_FLOOR)

    # Effective sample size after downweighting; conservative vs plain sqrt(n).
    n_eff = float(np.sum(w) ** 2 / np.sum(w ** 2))
    return Location(mu, sigma, sigma / np.sqrt(max(n_eff, 1.0)), int(n))


def refine_face(vals, seed: float, windows=(0.040, 0.020, 0.012),
                min_points: int = 20) -> Location | None:
    """Refine a coarse face position to sub-mm using the raw coordinates.

    `seed` is a rough position (e.g. a bin centre from coarse detection) that
    may be off by half a bin. The window schedule shrinks around the running
    estimate: the first pass tolerates the seed error, later passes tighten
    onto the surface itself so that neighbouring structure (the wall's far
    face, a skirting board) is excluded rather than averaged in.

    Returns None if the face never carries enough support to be measurable.
    """
    v = np.asarray(vals, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None

    centre = float(seed)
    out: Location | None = None
    for half in windows:
        sel = v[np.abs(v - centre) <= half]
        if sel.size < min_points:
            break
        out = robust_location(sel)
        if not np.isfinite(out.value):
            return None
        centre = out.value
    return out


def detect_wall_faces(coords, heights, bin_m: float = 0.05,
                      min_points: int = 6, min_span_m: float = 1.5,
                      merge_tol: float = 0.005, **refine_kw) -> list[Location]:
    """Find full-height wall faces along one axis, measured to sub-mm.

    Coarse binning is used ONLY to locate candidates -- a bin qualifies when it
    holds points spanning more than `min_span_m` vertically, which separates
    structure from furniture. The returned position never comes from the bin;
    every candidate is re-measured from the raw coordinates via `refine_face`.

    Each qualifying bin is refined independently and duplicates are then merged
    by MEASURED position (within `merge_tol`). Deduplicating on the measurement
    rather than on bin adjacency matters: a face landing near a bin edge spills
    into its neighbour, and merging those neighbours by adjacency would chain
    across a thin partition and swallow the wall's far face. Resolving power is
    therefore set by `merge_tol`, not by `bin_m`.
    """
    co = np.asarray(coords, dtype=float).ravel()
    zc = np.asarray(heights, dtype=float).ravel()
    if co.size == 0:
        return []

    b = np.floor(co / bin_m).astype(np.int64)
    seeds = []
    for bb in np.unique(b):
        m = b == bb
        if int(m.sum()) < min_points:
            continue
        if float(zc[m].max() - zc[m].min()) <= min_span_m:
            continue
        seeds.append(float(np.median(co[m])))

    fits = []
    for s in seeds:
        fit = refine_face(co, s, **refine_kw)
        if fit is not None and np.isfinite(fit.value):
            fits.append(fit)
    if not fits:
        return []

    # Collapse fits that converged on the same surface; keep the best-supported.
    fits.sort(key=lambda f: f.value)
    faces = [fits[0]]
    for f in fits[1:]:
        if f.value - faces[-1].value <= merge_tol:
            if f.n > faces[-1].n:
                faces[-1] = f
        else:
            faces.append(f)
    return faces


def fit_plane_tls(points, normal0=None, band: float = 0.05,
                  iters: int = 6, k: float = HUBER_K,
                  min_points: int = 50) -> PlaneFit | None:
    """Robust total-least-squares plane fit to a face.

    TLS (smallest eigenvector of the weighted covariance) rather than an
    axis-wise least squares, so the fit is unbiased for walls that are not
    exactly axis-aligned -- which is the whole point of not snapping to
    Manhattan before measuring.

    `band` is the initial inlier half-width; it retightens to 3 sigma each
    iteration so the final fit sees the surface and not its surroundings.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < min_points:
        return None

    if normal0 is None:
        c = pts.mean(axis=0)
        _, vecs = np.linalg.eigh(np.cov((pts - c).T))
        n = vecs[:, 0]
    else:
        n = np.asarray(normal0, dtype=float)
        n = n / np.linalg.norm(n)
    d = float(np.median(pts @ n))

    sigma = band / 3.0
    inl = np.ones(pts.shape[0], dtype=bool)
    for _ in range(iters):
        r = pts @ n - d
        inl = np.abs(r) <= band
        if int(inl.sum()) < min_points:
            return None
        p = pts[inl]
        rin = np.abs(p @ n - d) / max(sigma, _SIGMA_FLOOR)
        w = np.where(rin <= k, 1.0, k / np.maximum(rin, 1e-12))

        c = np.sum(w[:, None] * p, axis=0) / np.sum(w)
        q = p - c
        cov = (q * w[:, None]).T @ q / np.sum(w)
        vals, vecs = np.linalg.eigh(cov)
        n_new = vecs[:, 0]
        if n_new @ n < 0:            # keep the normal's sense stable
            n_new = -n_new
        n, d = n_new, float(c @ n_new)

        sigma = max(float(np.sqrt(max(vals[0], 0.0))), _SIGMA_FLOOR)
        band = max(3.0 * sigma, 0.004)

    p = pts[inl]
    resid = p @ n - d
    n_in = int(inl.sum())
    return PlaneFit(n, d, sigma, sigma / np.sqrt(max(n_in, 1)), n_in)


def face_gap(a: Location, b: Location) -> tuple[float, float]:
    """Clear distance between two measured faces, with its combined stderr.

    Uncertainties add in quadrature: the span is only as certain as the two
    independent faces that define it.
    """
    return abs(b.value - a.value), float(np.hypot(a.stderr, b.stderr))
