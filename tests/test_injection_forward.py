"""The direct likelihood must explain the untouched simulated input pixels."""
import numpy as np
import pytest
import torch

from dingo.injection_forward import ForwardExperiment
from dingo.simple_simulation import make_mock


@pytest.fixture
def experiment(tmp_path, grism_cal):
    return ForwardExperiment(make_mock(tmp_path / 'toy'))


def test_true_forward_prediction_closes_exactly_and_scan_recovers_truth(experiment):
    exp = experiment
    for pupil, prediction in zip(('R', 'C'), exp.predict(exp.truth_theta)):
        torch.testing.assert_close(prediction, exp.clean_data[pupil], rtol=0, atol=0)
    trials = np.tile(exp.truth_theta, (41, 1))
    trials[:, 0] = np.arange(0, 401, 10)
    scores = exp.chi2(trials, noisy=False).numpy()
    assert trials[np.argmin(scores), 0] == 200.
    assert scores[20] == 0.
    assert np.all(scores[np.arange(41) != 20] > 0)


def test_batched_likelihood_matches_scalar_and_rejects_bad_proposals(experiment):
    exp = experiment
    trials = np.tile(exp.truth_theta, (4, 1))
    trials[1, 0] = 105.
    trials[2, 3:] = [.2, -.2]
    trials[3, 0] = -1.
    batch = exp.log_probability_batch(trials)
    scalar = [exp.log_probability(point) for point in trials]
    np.testing.assert_allclose(batch, scalar, rtol=0, atol=1e-7)
    assert not np.isfinite(batch[-1])


def test_supplied_emission_is_shared_between_both_grisms_and_not_truth(experiment):
    exp = experiment
    blank = torch.zeros(exp.shape)
    for image in exp.predict(exp.truth_theta, emission=blank):
        assert torch.count_nonzero(image) == 0
    double = exp.predict(exp.truth_theta, emission=2 * exp.emission)
    for pupil, prediction in zip(('R', 'C'), double):
        torch.testing.assert_close(prediction, 2 * exp.clean_data[pupil], rtol=0, atol=0)
    # The known-map prediction remains unchanged after another map was supplied.
    assert float(exp.chi2(exp.truth_theta, noisy=False)) == 0.


def test_noisy_likelihood_is_original_pixel_gaussian(experiment):
    exp = experiment
    predictions = exp.predict(exp.truth_theta)
    expected = -.5 * sum(float(torch.sum(((a - exp.data[p]) / exp.sigma).double()**2))
                         for p, a in zip(('R', 'C'), predictions))
    expected += float(exp.log_prior(exp.truth_theta))
    assert exp.log_probability(exp.truth_theta) == pytest.approx(expected, abs=1e-7)


def test_noiseless_simulation_has_no_formal_noise_likelihood(tmp_path, grism_cal):
    exp = ForwardExperiment(make_mock(tmp_path / 'zero_noise', noise_sigma=0.))
    assert float(exp.chi2(exp.truth_theta, noisy=False)) == 0.
    with pytest.raises(ValueError, match='noise_sigma'):
        exp.log_probability(exp.truth_theta)


@pytest.mark.parametrize('start', [75., 350., 800.])
def test_noisy_known_map_recovers_rotation_from_displaced_starts(experiment, start):
    result = experiment.fit(start, max_evaluations=2200)
    # This realization has finite noise and a V/R_t degeneracy; the test is a
    # scientific sanity bound, not exact equality to the generating parameters.
    assert result['success']
    assert abs(result['theta'][0] - 200.) < 25.
    assert abs(result['theta'][1] - 5.) < 1.
    assert np.max(np.abs(result['theta'][3:])) < .1
    assert result['log_probability'] >= experiment.log_probability(experiment.truth_theta) - 1.


def test_inclination_prior_preserves_true_prediction_and_supplied_brightness(experiment):
    exp = ForwardExperiment(experiment.mock, inclination_mean=55., inclination_sigma=5.)
    assert exp.param_names[-1] == "velocity.inc_v"
    assert exp.truth_theta.shape == (6,)
    assert exp.sample_inclination
    assert exp.inclination_settings == {
        "mode": "gaussian", "mean_deg": 55., "sigma_deg": 5., "bounds_deg": [5., 85.]}
    emission_before = exp.emission.clone()
    for pupil, prediction in zip(("R", "C"), exp.predict(exp.truth_theta)):
        torch.testing.assert_close(prediction, exp.clean_data[pupil], rtol=0, atol=0)

    different = exp.truth_theta.copy()
    different[-1] = np.deg2rad(70.)
    for original, changed in zip(exp.predict(exp.truth_theta), exp.predict(different)):
        assert torch.any(original != changed)
    # Releasing kinematic inclination does not rebuild the input brightness map.
    torch.testing.assert_close(exp.emission, emission_before, rtol=0, atol=0)


def test_inclination_gaussian_prior_and_batch_likelihood(experiment):
    exp = ForwardExperiment(experiment.mock, inclination_mean=55., inclination_sigma=5.)
    trials = np.tile(exp.truth_theta, (5, 1))
    trials[:, -1] = np.deg2rad([55., 60., 65., 4., 86.])
    prior = exp.log_prior(trials)
    np.testing.assert_allclose(prior[:3] - prior[0], [0., -.5, -2.], atol=1e-12)
    assert np.isneginf(prior[3:]).all()
    assert prior[0] == pytest.approx(experiment.log_prior(trials[0, :5]))
    batch = exp.log_probability_batch(trials)
    scalar = np.array([exp.log_probability(point) for point in trials])
    np.testing.assert_allclose(batch, scalar, rtol=0, atol=1e-7)
    assert np.isneginf(batch[3:]).all()

    # The free-map profile uses the same Gaussian penalty and its gradient,
    # while preserving its historical hard-bound-only treatment of V/Rt/PA/Rdx/Rdy.
    theta = torch.tensor(trials[1], dtype=torch.float32, requires_grad=True)
    penalty = exp.prior_penalty(theta)
    gradient = torch.autograd.grad(penalty, theta)[0]
    assert float(penalty.detach()) == pytest.approx(.5, abs=1e-6)
    assert float(gradient[-1]) == pytest.approx(1 / np.deg2rad(5.), rel=1e-6)
    torch.testing.assert_close(gradient[:5], torch.zeros(5), rtol=0, atol=0)
    assert float(experiment.prior_penalty(theta[:5])) == 0.


@pytest.mark.parametrize("kwargs", [
    {"inclination_mean": float("nan")}, {"inclination_mean": 4.},
    {"inclination_mean": 86.}, {"inclination_mean": True},
    {"inclination_sigma": 0.}, {"inclination_sigma": -1.},
    {"inclination_sigma": float("inf")}, {"inclination_sigma": True},
])
def test_inclination_configuration_rejects_invalid_values_before_loading(kwargs):
    with pytest.raises(ValueError, match="inclination_"):
        ForwardExperiment("not-read", **kwargs)


def test_inclination_fit_starts_from_prior_mean_or_explicit_angle(experiment, monkeypatch):
    from types import SimpleNamespace
    import scipy.optimize

    exp = ForwardExperiment(experiment.mock, inclination_mean=55., inclination_sigma=5.)
    supplied_starts = []

    def no_optimization(objective, start, *, method, bounds, options):
        # Exercise packing and score evaluation without running a fit.
        supplied_starts.append(np.asarray(start) * exp.parameter_scales)
        assert len(bounds) == 6
        assert np.isfinite(objective(start))
        return SimpleNamespace(success=False, message="test stub", nfev=1)

    monkeypatch.setattr(scipy.optimize, "minimize", no_optimization)
    default = exp.fit(350.)
    explicit = exp.fit(350., start_inclination=65.)
    assert np.rad2deg(default["start_theta"][-1]) == pytest.approx(55.)
    assert np.rad2deg(explicit["start_theta"][-1]) == pytest.approx(65.)
    np.testing.assert_array_equal(default["theta"], supplied_starts[0])
    np.testing.assert_array_equal(explicit["theta"], supplied_starts[1])
    with pytest.raises(ValueError, match="fixes inclination"):
        experiment.fit(350., start_inclination=65.)
