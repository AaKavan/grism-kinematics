"""A free source map must fit original R/C without reading its injected shape."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from dingo.injection_forward import ForwardExperiment
from dingo.simple_simulation import make_mock
from dingo import injection_flexible as free


@pytest.fixture
def experiment(tmp_path, grism_cal):
    return ForwardExperiment(make_mock(tmp_path / "toy"))


def test_sparse_operator_agrees_with_direct_prediction_for_an_arbitrary_map(experiment):
    exp = experiment
    arbitrary = np.random.default_rng(9).uniform(size=exp.shape).astype(np.float32)
    point = np.array([317., 3.7, 1.1, .23, -.41])
    sparse = free._design_matrix(exp, point) @ arbitrary.ravel()
    predicted = np.concatenate([a.detach().numpy().ravel() for a in exp.predict(point, arbitrary)])
    np.testing.assert_allclose(sparse, predicted, rtol=2e-6, atol=3e-7)


def test_free_map_closes_noiseless_truth_without_reading_generating_morphology(experiment):
    exp = experiment
    # A truth-map access now contaminates the result; the linear solver must
    # depend only on data, mapping coordinates, and the shared canvas shape.
    exp.emission = torch.full(exp.shape, float("nan"))
    result = free.profile_map(exp, exp.truth_theta, noisy=False)
    assert result["success"]
    assert result["sse"] < 1e-7
    assert np.isfinite(result["emission"]).all()
    assert np.min(result["emission"]) >= 0.
    assert result["chi2"] is None


def test_conditional_grid_recovers_noiseless_velocity_with_free_morphology(experiment):
    table, fig = free.scan_velocity(experiment, [100., 150., 200., 250., 300.], progress=False)
    try:
        assert table.loc[table["Noiseless"].idxmin(), "V_inf"] == 200.
        assert table.loc[table["With noise"].idxmin(), "V_inf"] == 200.
        assert (experiment.mock.directory / "direct_free_map" / "velocity_scan.csv").is_file()
        assert "profile-likelihood" in (experiment.mock.directory / "direct_free_map" / "scan_info.txt").read_text()
    finally:
        plt.close(fig)


def test_score_uses_original_input_noise_and_reports_inner_convergence(experiment):
    result = free.profile_map(experiment, experiment.truth_theta)
    residual = sum(np.sum((result[f"predicted_{p}"] - experiment.data[p].numpy())**2) for p in ("R", "C"))
    assert result["sse"] == pytest.approx(residual)
    assert result["chi2"] == pytest.approx(residual / experiment.sigma**2)
    assert result["success"]
    assert result["optimality"] < 1e-5


def test_profile_fit_reports_exhausted_budget_and_preserves_explicit_start(experiment):
    result = free.fit(experiment, 350., max_evaluations=1, progress=False)
    np.testing.assert_allclose(result["theta"], [350., 3., np.deg2rad(20.), .2, -.2])
    assert not result["success"]
    assert result["inner_success"]
    assert result["nfev"] == 1
    assert "Maximum profile evaluations" in result["message"]
    assert result["projected_gradient_max"] > 1e-5
    assert not result["gradient_converged"]


def test_noiseless_multivariate_recovery_has_no_truth_initialization(experiment):
    result = free.fit(experiment, 350., noisy=False, max_evaluations=100, progress=False)
    assert result["inner_success"]
    assert result["sse"] < 1e-6
    assert abs(result["theta"][0] - 200.) < .5
    assert abs(result["theta"][1] - 5.) < .05
    assert np.max(np.abs(result["theta"][3:])) < .005


def test_regenerated_mock_cannot_receive_results_from_a_stale_experiment(experiment):
    make_mock(experiment.mock.directory, seed=7, overwrite=True)
    for action in (
        lambda: free.profile_map(experiment, experiment.truth_theta),
        lambda: free.scan_velocity(experiment, [150., 200., 250.], progress=False),
        lambda: free.fit(experiment, 350., max_evaluations=1, progress=False),
    ):
        with pytest.raises(ValueError, match="saved simulation has changed"):
            action()
    assert not (experiment.mock.directory / "direct_free_map").exists()


def test_map_solver_bounds_blas_threads_and_restores_callers_settings(experiment, monkeypatch):
    import scipy.optimize
    from threadpoolctl import threadpool_info, threadpool_limits

    original = scipy.optimize.lsq_linear
    observed = []
    def checked_solver(*args, **kwargs):
        counts = [pool['num_threads'] for pool in threadpool_info() if pool['user_api'] == 'blas']
        observed.extend(counts)
        assert counts and all(count == 1 for count in counts)
        return original(*args, **kwargs)
    monkeypatch.setattr(scipy.optimize, 'lsq_linear', checked_solver)
    with threadpool_limits(limits=2, user_api='blas'):
        before = [pool['num_threads'] for pool in threadpool_info() if pool['user_api'] == 'blas']
        result = free.profile_map(experiment, experiment.truth_theta)
        after = [pool['num_threads'] for pool in threadpool_info() if pool['user_api'] == 'blas']
        assert before == after
        assert all(count == 2 for count in after)
    assert observed and result['success']


@pytest.mark.parametrize("noisy", [False, True])
def test_inclination_profile_score_and_gradient_without_optimization(experiment, monkeypatch, noisy):
    """Check six-parameter wiring; noiseless SSE must not acquire a prior weight."""
    from types import SimpleNamespace
    import scipy.optimize

    exp = ForwardExperiment(experiment.mock, inclination_mean=55., inclination_sigma=5.)
    # At the true map/kinematics, clean inputs give zero residual gradient.
    # The only nonzero score gradient should be the prior, when noisy=True.
    exp.data = exp.clean_data

    def supplied_map(*args, **kwargs):
        return SimpleNamespace(x=exp.emission.numpy().ravel(), success=True,
                               message="test supplied map", optimality=0., nit=0)

    evaluated = []

    def one_evaluation(objective, start, **kwargs):
        assert len(start) == len(kwargs["bounds"]) == 6
        score, gradient = objective(start)
        evaluated.append((score, gradient))
        return SimpleNamespace(success=False, message="test evaluation only")

    monkeypatch.setattr(scipy.optimize, "lsq_linear", supplied_map)
    monkeypatch.setattr(scipy.optimize, "minimize", one_evaluation)
    result = free.fit(exp, 200., start_turnover=5., start_pa=35., start_offset=(0., 0.),
                      start_inclination=60., noisy=noisy, progress=False)
    assert result["theta"].shape == (6,)
    assert result["prior_penalty"] == pytest.approx(.5 if noisy else 0., abs=1e-6)
    # The inclination scale is five degrees, so a one-sigma deviation gives
    # unit derivative in the optimizer's scaled coordinate.
    expected = np.zeros(6)
    expected[5] = 1. if noisy else 0.
    np.testing.assert_allclose(evaluated[0][1], expected, rtol=0, atol=2e-6)
    assert ("plus stated kinematic prior" in result["objective"]) == noisy
