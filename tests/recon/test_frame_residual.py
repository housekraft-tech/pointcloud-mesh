"""axis_residuals: quantify what the Manhattan snap costs.

dominant_axes returns a rotation with no indication of how well it fits, so a
building 0.5 deg out of square looks identical to a perfectly square one --
while costing 87mm over a 10m span. These tests pin that the residual is
actually measured, and that a building the single-grid assumption does NOT
describe is flagged rather than silently averaged.
"""
import numpy as np
import pytest

from scripts.recon.frame import axis_residuals, deviation_mm, dominant_axes


def _wall_normals(headings_deg, n_each=400, noise_deg=0.0, rng=None):
    """Horizontal unit normals at the given headings."""
    rng = rng or np.random.default_rng(0)
    out = []
    for h in headings_deg:
        a = np.deg2rad(h) + np.deg2rad(noise_deg) * rng.standard_normal(n_each)
        out.append(np.column_stack([np.cos(a), np.sin(a), np.zeros(n_each)]))
    return np.vstack(out)


class TestDeviationMM:
    def test_the_number_that_matters(self):
        """0.1 deg over 10m is ~17mm -- comparable with the rest of the budget."""
        assert deviation_mm(0.1, 10.0) == pytest.approx(17.45, abs=0.1)
        assert deviation_mm(0.5, 10.0) == pytest.approx(87.3, abs=0.5)
        assert deviation_mm(0.0, 10.0) == 0.0

    def test_scales_with_lever_arm(self):
        assert deviation_mm(0.2, 5.0) == pytest.approx(deviation_mm(0.2, 10.0) / 2, rel=1e-6)


class TestSquareBuilding:
    @pytest.mark.parametrize("theta", [0.0, 7.3, 31.0, 44.0, 89.0])
    def test_recovers_orientation_and_reports_near_zero_residual(self, theta):
        nrm = _wall_normals([theta, theta + 90, theta + 180, theta + 270])
        r = axis_residuals(nrm)
        assert r is not None
        assert r.max_dev_deg < 1e-6
        assert r.dispersion_deg < 1e-3
        assert r.frac_off_grid == 0.0

    def test_theta_agrees_with_dominant_axes(self):
        """The residual must describe the SAME grid the pipeline rotates to."""
        nrm = _wall_normals([12.5, 102.5, 192.5], noise_deg=1.0,
                            rng=np.random.default_rng(1))
        r = axis_residuals(nrm)
        # dominant_axes builds R from (c, s) = (cos(-theta), sin(-theta)) as
        # [[c, -s], [s, c]], so R[0,0] = cos(theta) and R[0,1] = sin(theta).
        R = dominant_axes(nrm)
        applied = np.degrees(np.arctan2(R[0, 1], R[0, 0])) % 90.0
        assert r.theta_deg == pytest.approx(applied, abs=1e-9)


class TestOutOfSquare:
    def test_detects_a_building_half_a_degree_out(self):
        """One wall pair rotated 0.5 deg: must surface, not average away."""
        nrm = np.vstack([_wall_normals([0.0, 180.0]),
                         _wall_normals([90.5, 270.5])])
        r = axis_residuals(nrm)
        assert r.max_dev_deg == pytest.approx(0.25, abs=0.02)
        assert deviation_mm(r.max_dev_deg, 10.0) > 40

    def test_dispersion_tracks_wall_noise(self):
        rng = np.random.default_rng(2)
        low = axis_residuals(_wall_normals([0, 90, 180, 270], noise_deg=0.2, rng=rng))
        high = axis_residuals(_wall_normals([0, 90, 180, 270], noise_deg=2.0, rng=rng))
        assert low.dispersion_deg == pytest.approx(0.2, rel=0.35)
        assert high.dispersion_deg == pytest.approx(2.0, rel=0.35)
        assert high.dispersion_deg > 5 * low.dispersion_deg

    def test_flags_a_wing_at_an_odd_angle(self):
        """A 30-deg wing breaks the single-grid assumption. frac_off_grid must
        expose it -- the circular mean is being pulled by walls it does not
        describe, so theta itself is suspect, not merely imprecise."""
        nrm = np.vstack([_wall_normals([0, 90, 180, 270], n_each=800),
                         _wall_normals([30, 120], n_each=400)])
        r = axis_residuals(nrm)
        assert r.frac_off_grid > 0.15
        assert r.max_dev_deg > 10

    def test_square_building_is_not_flagged(self):
        rng = np.random.default_rng(3)
        r = axis_residuals(_wall_normals([0, 90, 180, 270], noise_deg=0.5, rng=rng))
        assert r.frac_off_grid < 0.01
        assert r.max_dev_deg < 5.0


class TestUncertaintyAndEdges:
    def test_stderr_tracks_actual_scatter_of_theta(self):
        thetas = []
        for s in range(40):
            rng = np.random.default_rng(s)
            nrm = _wall_normals([11.0, 101.0, 191.0], n_each=300,
                                noise_deg=1.5, rng=rng)
            thetas.append(axis_residuals(nrm).theta_deg)
        reported = axis_residuals(_wall_normals(
            [11.0, 101.0, 191.0], n_each=300, noise_deg=1.5,
            rng=np.random.default_rng(0))).theta_stderr_deg
        assert np.std(thetas) == pytest.approx(reported, rel=0.5)

    def test_more_walls_means_tighter_theta(self):
        rng = np.random.default_rng(4)
        few = axis_residuals(_wall_normals([0, 90], n_each=50, noise_deg=2.0, rng=rng))
        many = axis_residuals(_wall_normals([0, 90], n_each=5000, noise_deg=2.0, rng=rng))
        assert many.theta_stderr_deg < few.theta_stderr_deg / 5

    def test_no_wall_normals(self):
        floor_only = np.tile([0.0, 0.0, 1.0], (100, 1))
        assert axis_residuals(floor_only) is None
        assert axis_residuals(np.zeros((0, 3))) is None
