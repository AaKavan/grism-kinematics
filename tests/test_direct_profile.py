"""Real-source profiling must weight R/C correctly and fit the inclination prior."""
import numpy as np
import pytest
import torch

from dingo import injection_flexible as free


class IdentitySource:
    """A small exact linear problem with no mock data or injected truth map."""

    param_names = ("V", "Rt", "PA", "Rdx", "Rdy", "inclination")
    shape = (2, 2)
    device = torch.device("cpu")
    inclination_mean = 75.
    inclination_sigma = 5.
    parameter_scales = np.array([200., 5., 1., 1., 1., np.deg2rad(5.)])
    theta_bounds = (np.array([0., .5, 0., -10., -10., np.deg2rad(5.)]),
                    np.array([1000., 40., 2*np.pi, 10., 10., np.deg2rad(85.)]))

    def __init__(self):
        self.data = {"R": torch.tensor([[1., float("nan")], [-2., 4.]]),
                     "C": torch.tensor([[5., 8.], [-4., 4.]])}
        self.sigma_by_pupil = {"R": 1., "C": 2.}
        self.valid_by_pupil = {p: torch.isfinite(a) for p, a in self.data.items()}
        self.validation_calls = 0

    def validate_inputs(self):
        self.validation_calls += 1

    @property
    def emission(self):
        raise AssertionError("There is no supplied source morphology in this fit.")

    def mapping_coordinates(self, theta):
        y, x = torch.meshgrid(torch.arange(2.), torch.arange(2.), indexing="ij")
        return {"R": (x, y), "C": (x, y)}

    def log_prior(self, theta):
        point = np.asarray(theta)
        low, high = self.theta_bounds
        if point.shape != (6,) or not np.isfinite(point).all() or np.any(point < low) or np.any(point > high):
            return -np.inf
        return -.5*((point[5]-np.deg2rad(75.))/np.deg2rad(5.))**2

    def prior_penalty(self, theta):
        return .5*((theta[5]-np.deg2rad(75.))/np.deg2rad(5.))**2

    def chi2(self, theta, *, emission, noisy=True):
        answer = theta.sum()*0.  # Differentiable, but no kinematic data information here.
        image = torch.as_tensor(emission, dtype=theta.dtype)
        for pupil in ("R", "C"):
            valid = self.valid_by_pupil[pupil]
            residual = image[valid] - self.data[pupil][valid]
            answer = answer + torch.sum((residual/self.sigma_by_pupil[pupil])**2)
        return answer


def point(inclination=75.):
    return np.array([450., 10., np.deg2rad(286.), 1., .3, np.deg2rad(inclination)])


def test_fixed_rdy_reaches_optimizer_as_equal_bounds_without_running_a_fit(monkeypatch):
    """Inspect optimizer arguments with stubbed solvers, not a source fit."""
    import scipy.optimize
    from types import SimpleNamespace

    model = IdentitySource()
    model.fixed_rdy = .271

    def profile_stub(experiment, theta, **kwargs):
        assert theta[4] == .271
        return dict(theta=theta.copy(), emission=np.ones((2, 2)), success=True,
                    chi2=1., sse=1., prior_penalty=0., message="stub")

    def inspect_arguments(objective, initial, *, bounds, **kwargs):
        assert initial[4] == .271
        assert bounds[4] == (.271, .271)
        assert bounds[0][0] < bounds[0][1]  # Velocity stays free.
        assert bounds[3][0] < bounds[3][1]  # R dx stays free.
        assert bounds[5][0] < bounds[5][1]  # Inclination stays free.
        objective(initial)
        return SimpleNamespace(success=True, message="argument inspection only")

    monkeypatch.setattr(free, "_profile_map", profile_stub)
    monkeypatch.setattr(scipy.optimize, "minimize", inspect_arguments)
    result = free.fit(model, 189., start_offset=(1.406, -.711), progress=False)
    assert result["theta"][4] == .271
    assert result["start_theta"][4] == .271


def test_real_profile_uses_separate_noise_and_excludes_invalid_pixels():
    model = IdentitySource()
    result = free.profile_map(model, point())
    # Weighted mean of 1 +/- 1 and 5 +/- 2 is 1.8, not the unweighted mean 3.
    expected = np.array([[1.8, 8.], [0., 4.]])
    np.testing.assert_allclose(result["emission"], expected, atol=1e-5)
    expected_chi2 = sum(float(torch.sum(
        ((torch.as_tensor(expected)[model.valid_by_pupil[p]]-model.data[p][model.valid_by_pupil[p]])
         / model.sigma_by_pupil[p])**2)) for p in ("R", "C"))
    assert result["chi2"] == pytest.approx(expected_chi2)
    assert result["valid_pixels"] == 7
    assert result["prior_penalty"] == pytest.approx(0., abs=1e-12)
    assert result["success"]
    assert model.validation_calls == 1


def test_profile_reports_inclination_penalty_separately_from_data_score():
    model = IdentitySource()
    center = free.profile_map(model, point(75.))
    displaced = free.profile_map(model, point(65.))
    assert displaced["prior_penalty"] == pytest.approx(2., rel=1e-6)
    assert displaced["chi2"] == pytest.approx(center["chi2"])


def test_six_parameter_fit_follows_inclination_prior_and_checkpoints_best_points():
    model = IdentitySource()
    checkpoints = []
    result = free.fit(model, 450., start_inclination=60., max_evaluations=30,
                      progress=False, callback=checkpoints.append)
    assert np.rad2deg(result["start_theta"][-1]) == pytest.approx(60.)
    assert np.rad2deg(result["theta"][-1]) == pytest.approx(75., abs=.001)
    assert result["prior_penalty"] < 1e-8
    assert result["profile_score"] == pytest.approx(.5*result["chi2"]+result["prior_penalty"])
    assert "prior" in result["objective"]
    assert len(checkpoints) >= 2
    scores = [record["profile_score"] for record in checkpoints]
    assert np.all(np.diff(scores) < 0.)
    assert checkpoints[-1]["nfev"] <= result["nfev"]
    assert all(record["inner_success"] for record in checkpoints)
    assert model.validation_calls == 1  # No file hashing inside the optimizer.


def test_real_source_cannot_accidentally_use_truth_only_scan_or_noiseless_mode():
    model = IdentitySource()
    with pytest.raises(ValueError, match="injection-only"):
        free.scan_velocity(model, [100., 200.], progress=False)
    with pytest.raises(ValueError, match="no noiseless counterpart"):
        free.profile_map(model, point(), noisy=False)


def test_real_source_rejects_nonpositive_input_noise():
    model = IdentitySource()
    model.sigma_by_pupil["C"] = 0.
    with pytest.raises(ValueError, match="input noise must be positive"):
        free.profile_map(model, point())


def test_source_model_and_sparse_profile_use_the_same_weighted_original_pixels(tmp_path, grism_cal):
    from dingo.direct_source import SourceModel
    from dingo.simple_simulation import make_mock

    mock = make_mock(tmp_path / "inputs")
    model = SourceModel(mock.source_id, data_dir=mock.data_dir, center=(40.3, 40.7),
                        inclination_mean=60., inclination_sigma=5., noise_sigma=(.02, .04))
    theta = np.array([200., 5., np.deg2rad(35.), 0., 0., np.deg2rad(60.)])
    result = free.profile_map(model, theta)
    with torch.no_grad():
        torch_chi2 = float(model.chi2(theta, emission=result["emission"]))
    assert result["success"]
    assert result["chi2"] == pytest.approx(torch_chi2, rel=2e-6)
    assert result["prior_penalty"] == pytest.approx(0., abs=1e-12)
    assert not hasattr(model, "mock")
    assert not hasattr(model, "emission")
