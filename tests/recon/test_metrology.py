"""Accuracy tests for scripts.recon.metrology.

These assert MILLIMETRE-level recovery against exactly-known ground truth, so
they fail loudly if an estimator ever regresses to grid-quantized behaviour.
Tolerances are stated in millimetres to keep the intent obvious.
"""
import numpy as np
import pytest

from scripts.recon.metrology import (
    detect_wall_faces,
    face_gap,
    fit_plane_tls,
    refine_face,
    robust_location,
)

MM = 1e-3


def _face_strip(pos, z0=0.0, z1=2.6, n=4000, noise=0.002, rng=None):
    """Points on one vertical wall face at coordinate `pos`."""
    rng = rng or np.random.default_rng(0)
    return (pos + rng.normal(0, noise, n), rng.uniform(z0, z1, n))


class TestRobustLocation:
    def test_recovers_mean_far_below_noise(self):
        rng = np.random.default_rng(1)
        v = 2.9500 + rng.normal(0, 0.002, 5000)
        got = robust_location(v)
        assert abs(got.value - 2.95) < 0.2 * MM
        assert got.sigma == pytest.approx(0.002, rel=0.15)
        assert got.stderr < 0.1 * MM

    def test_stderr_tracks_actual_error(self):
        """The reported stderr must be honest, not decorative."""
        errs = []
        for s in range(60):
            rng = np.random.default_rng(s)
            v = 1.0 + rng.normal(0, 0.003, 800)
            errs.append(robust_location(v).value - 1.0)
        reported = robust_location(1.0 + np.random.default_rng(0)
                                   .normal(0, 0.003, 800)).stderr
        assert np.std(errs) == pytest.approx(reported, rel=0.35)

    def test_resists_skirting_board_outliers(self):
        """5% of points 3cm proud (a skirting board) must not move the face."""
        rng = np.random.default_rng(2)
        clean = 3.0 + rng.normal(0, 0.002, 4000)
        skirting = 2.97 + rng.normal(0, 0.002, 200)
        v = np.concatenate([clean, skirting])
        assert abs(robust_location(v).value - 3.0) < 0.5 * MM
        assert abs(float(np.mean(v)) - 3.0) > 1.0 * MM   # the naive mean fails

    def test_degenerate_inputs(self):
        assert robust_location([]).n == 0
        assert robust_location([1.5]).value == 1.5
        assert np.isfinite(robust_location(np.full(50, 2.0)).value)


class TestRefineFace:
    @pytest.mark.parametrize("seed_offset", [-0.024, -0.01, 0.0, 0.01, 0.024])
    def test_absorbs_half_bin_seed_error(self, seed_offset):
        """A seed off by up to half a 5cm bin must still land sub-mm."""
        vals, _ = _face_strip(2.95, rng=np.random.default_rng(3))
        got = refine_face(vals, 2.95 + seed_offset)
        assert got is not None
        assert abs(got.value - 2.95) < 0.5 * MM

    def test_excludes_far_face_of_a_thin_partition(self):
        """Both faces of a 100mm partition are present; measure only the near one."""
        rng = np.random.default_rng(4)
        near, _ = _face_strip(2.95, rng=rng)
        far, _ = _face_strip(3.05, rng=rng)
        got = refine_face(np.concatenate([near, far]), 2.95)
        assert got is not None
        assert abs(got.value - 2.95) < 0.5 * MM

    def test_returns_none_without_support(self):
        assert refine_face(np.array([1.0, 1.001]), 1.0) is None
        assert refine_face(np.array([]), 1.0) is None


class TestDetectWallFaces:
    def test_recovers_both_faces_sub_mm(self):
        rng = np.random.default_rng(5)
        a, za = _face_strip(0.10, rng=rng)
        b, zb = _face_strip(2.95, rng=rng)
        faces = detect_wall_faces(np.concatenate([a, b]),
                                  np.concatenate([za, zb]))
        assert len(faces) == 2
        assert abs(faces[0].value - 0.10) < 0.5 * MM
        assert abs(faces[1].value - 2.95) < 0.5 * MM

    def test_never_returns_a_bin_centre(self):
        """Regression guard: the old code returned bin*0.05 verbatim."""
        rng = np.random.default_rng(6)
        vals, zs = _face_strip(2.9371, rng=rng)
        faces = detect_wall_faces(vals, zs)
        assert len(faces) == 1
        assert abs(faces[0].value - 2.9371) < 0.5 * MM
        assert faces[0].value % 0.05 > 1e-6      # not snapped to the grid

    def test_rejects_furniture(self):
        """A 0.9m-tall counter must not be mistaken for a wall face."""
        rng = np.random.default_rng(7)
        wall, zw = _face_strip(0.10, rng=rng)
        counter = 1.60 + rng.normal(0, 0.002, 3000)
        zc = rng.uniform(0.0, 0.9, 3000)
        faces = detect_wall_faces(np.concatenate([wall, counter]),
                                  np.concatenate([zw, zc]))
        assert len(faces) == 1
        assert abs(faces[0].value - 0.10) < 0.5 * MM

    def test_face_on_a_bin_boundary_is_not_split(self):
        """A face sitting exactly on a bin edge must yield ONE face, not two."""
        rng = np.random.default_rng(8)
        vals, zs = _face_strip(3.00, noise=0.004, rng=rng)
        faces = detect_wall_faces(vals, zs)
        assert len(faces) == 1
        assert abs(faces[0].value - 3.00) < 0.5 * MM

    def test_clear_span_end_to_end(self):
        """The number the floorplan actually prints, against known truth."""
        rng = np.random.default_rng(9)
        a, za = _face_strip(0.10, rng=rng)
        b, zb = _face_strip(2.95, rng=rng)
        faces = detect_wall_faces(np.concatenate([a, b]),
                                  np.concatenate([za, zb]))
        span, err = face_gap(faces[0], faces[-1])
        assert abs(span - 2.85) < 0.5 * MM
        assert err < 0.2 * MM

    def test_empty(self):
        assert detect_wall_faces(np.array([]), np.array([])) == []


class TestFitPlaneTLS:
    def test_recovers_a_wall_out_of_square(self):
        """0.7 degrees off axis: TLS must not bias the offset like an axis fit."""
        rng = np.random.default_rng(10)
        th = np.deg2rad(0.7)
        n_true = np.array([np.cos(th), np.sin(th), 0.0])
        d_true = 2.95
        u = rng.uniform(-2.5, 2.5, 6000)
        z = rng.uniform(0, 2.6, 6000)
        tang = np.array([-np.sin(th), np.cos(th), 0.0])
        pts = (n_true * d_true) + u[:, None] * tang + z[:, None] * np.array([0, 0, 1.0])
        pts += rng.normal(0, 0.002, pts.shape) * n_true

        fit = fit_plane_tls(pts)
        assert fit is not None
        assert abs(fit.offset - d_true) < 0.5 * MM
        assert np.degrees(np.arccos(abs(fit.normal @ n_true))) < 0.05

    def test_rejects_undersized_input(self):
        assert fit_plane_tls(np.zeros((10, 3))) is None
        assert fit_plane_tls(np.zeros((100, 2))) is None
