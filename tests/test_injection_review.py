"""Provenance and posterior display checks for the injection notebook.

The very short chains here exercise saving and plotting only. They do not
assert convergence or establish frequentist coverage of the model's intervals.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from dingo.injection_forward import ForwardExperiment
from dingo.simple_simulation import make_mock
from dingo import injection_review as review


@pytest.fixture
def experiment(tmp_path, grism_cal):
    return ForwardExperiment(make_mock(tmp_path / "experiment"))


def test_short_sampling_saves_only_retained_draws_and_explicit_conditions(experiment):
    exp = experiment
    folder = review.sample(exp, {"theta": exp.truth_theta}, warmup=2, draws=10,
                           walkers=12, seed=3, progress=False)
    info = json.loads((folder / "run_info.txt").read_text())
    with np.load(folder / "samples.npz", allow_pickle=False) as saved:
        assert saved["chain"].shape == (10, 12, 5)
        assert saved["log_prob"].shape == (10, 12)
        assert saved["param_names"].tolist() == list(exp.param_names)
        assert np.isfinite(saved["chain"]).all()
        assert np.isfinite(saved["log_prob"]).all()
    assert info["status"] == "finished"
    assert info["retained_steps"] == 10
    assert info["warmup"] == 2
    assert info["morphology_uncertainty_included"] is False
    assert info["input_sha256"] == exp.mock.metadata["input_sha256"]
    assert "true emission map" in info["condition_on"]
    assert "finished_utc" in info
    original = (folder / "samples.npz").read_bytes()
    with pytest.raises(FileExistsError, match="saved samples"):
        review.sample(exp, {"theta": exp.truth_theta}, warmup=2, draws=10,
                      walkers=12, progress=False)
    assert (folder / "samples.npz").read_bytes() == original


def test_result_summary_keeps_units_truth_and_all_draws(experiment):
    exp = experiment
    folder = review.sample(exp, {"theta": exp.truth_theta}, warmup=1, draws=10,
                           walkers=12, seed=2, progress=False)
    views = review.show_results(exp, folder)
    try:
        with np.load(folder / "samples.npz", allow_pickle=False) as saved:
            flat = saved["chain"].reshape(-1, 5)
        converted = flat * np.array([1., 1., 180 / np.pi, 1., 1.])
        quantiles = np.quantile(converted, [.16, .5, .84], axis=0)
        table = views["summary"]
        np.testing.assert_allclose(table["median"], quantiles[1])
        np.testing.assert_allclose(table["minus_16th"], quantiles[1] - quantiles[0])
        np.testing.assert_allclose(table["plus_84th"], quantiles[2] - quantiles[1])
        np.testing.assert_allclose(table["truth"], [200., 5., 35., 0., 0.])
        for filename in ("corner.png", "corner.pdf", "traces.png", "residuals.png",
                         "rotation_curve.png", "truth_comparison.csv"):
            assert (folder / filename).is_file()
        assert "known-map" in views["corner"]._suptitle.get_text().lower()
    finally:
        plt.close("all")


def test_other_noise_realization_is_rejected_before_loading_samples(experiment, tmp_path):
    exp = experiment
    folder = exp.mock.directory / "direct_known_map"
    folder.mkdir()
    # No sample file is needed: provenance must fail before any plotting.
    (folder / "run_info.txt").write_text(json.dumps({
        "input_sha256": exp.mock.metadata["input_sha256"]}))
    other = ForwardExperiment(make_mock(tmp_path / "different", seed=2))
    with pytest.raises(ValueError, match="different inputs"):
        review.show_results(other, folder)


def test_noiseless_scan_minimum_is_truth_and_mcmc_is_rejected(tmp_path, grism_cal):
    exp = ForwardExperiment(make_mock(tmp_path / "noiseless", noise_sigma=0.))
    table, figure = review.scan_velocity(exp, [100., 150., 200., 250., 300.])
    try:
        assert list(table.columns) == ["V_inf", "Noiseless"]
        assert table.loc[table["Noiseless"].idxmin(), "V_inf"] == 200.
        assert table.loc[table["V_inf"] == 200., "Noiseless"].item() == 0.
        assert (exp.mock.directory / "direct_known_map/velocity_scan.csv").is_file()
        with pytest.raises(ValueError, match="Noiseless"):
            review.sample(exp, {"theta": exp.truth_theta}, warmup=1, draws=2,
                          walkers=12, progress=False)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("interrupt_phase, expected_steps", [(1, 0), (2, 3)])
def test_interruption_distinguishes_warmup_from_retained_samples(
        experiment, monkeypatch, interrupt_phase, expected_steps):
    import emcee

    original_sample = emcee.EnsembleSampler.sample
    calls = 0

    def interrupted_sample(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        phase = calls
        for index, state in enumerate(original_sample(self, *args, **kwargs)):
            yield state
            if phase == interrupt_phase and index == (0 if phase == 1 else 2):
                raise KeyboardInterrupt("deliberate test interruption")

    monkeypatch.setattr(emcee.EnsembleSampler, "sample", interrupted_sample)
    exp = experiment
    with pytest.raises(KeyboardInterrupt, match="deliberate"):
        review.sample(exp, {"theta": exp.truth_theta}, warmup=2, draws=10,
                      walkers=12, progress=False)
    folder = exp.mock.directory / "direct_known_map"
    info = json.loads((folder / "run_info.txt").read_text())
    assert info["status"] == "interrupted"
    if expected_steps:
        with np.load(folder / "samples.npz", allow_pickle=False) as saved:
            assert saved["chain"].shape == (expected_steps, 12, 5)
        assert info["retained_steps"] == expected_steps
    else:
        assert not (folder / "samples.npz").exists()


def test_overwritten_warmup_failure_cannot_show_the_previous_chain(experiment, monkeypatch):
    import emcee

    exp = experiment
    folder = review.sample(exp, {"theta": exp.truth_theta}, warmup=0, draws=2,
                           walkers=12, progress=False)
    (folder / "corner.png").write_bytes(b"old posterior figure")

    def stop_in_warmup(self, *args, **kwargs):
        raise KeyboardInterrupt("warmup stopped")

    monkeypatch.setattr(emcee.EnsembleSampler, "run_mcmc", stop_in_warmup)
    with pytest.raises(KeyboardInterrupt, match="warmup stopped"):
        review.sample(exp, {"theta": exp.truth_theta}, warmup=2, draws=10,
                      walkers=12, overwrite=True, progress=False)
    assert not (folder / "samples.npz").exists()
    assert not (folder / "corner.png").exists()
    assert json.loads((folder / "run_info.txt").read_text())["status"] == "interrupted"


def test_simulation_regeneration_requires_a_fresh_forward_object(experiment):
    exp = experiment
    make_mock(exp.mock.directory, seed=2, overwrite=True)
    with pytest.raises(ValueError, match="fresh ForwardExperiment"):
        review.sample(exp, {"theta": exp.truth_theta}, warmup=1, draws=2,
                      walkers=12, progress=False)


def test_pa_summary_corner_truth_and_traces_share_the_circular_branch(tmp_path, grism_cal):
    exp = ForwardExperiment(make_mock(tmp_path / "seam", pa=359.8))
    folder = exp.mock.directory / "direct_known_map"
    folder.mkdir()
    rng = np.random.default_rng(11)
    chain = np.broadcast_to(exp.truth_theta, (30, 12, 5)).copy()
    chain += rng.normal(size=chain.shape) * np.array([1., .05, .02, .01, .01])
    chain[:, :, 2] %= 2 * np.pi
    assert np.any(np.rad2deg(chain[:, :, 2]) < 5.)
    assert np.any(np.rad2deg(chain[:, :, 2]) > 355.)
    np.savez_compressed(folder / "samples.npz", chain=chain,
                        log_prob=np.zeros((30, 12)))
    (folder / "run_info.txt").write_text(json.dumps({
        "input_sha256": exp.mock.metadata["input_sha256"], "status": "test_fixture"}))
    views = review.show_results(exp, folder)
    try:
        median = views["summary"].loc["PA [deg]", "median"]
        truth = views["summary"].loc["PA [deg]", "truth"]
        assert abs(median - truth) < .5
        assert abs((median - 359.8 + 180) % 360 - 180) < .5
        pa_trace = views["traces"].axes[2]
        plotted = np.concatenate([line.get_ydata() for line in pa_trace.lines])
        assert np.ptp(plotted) < 10.
        np.testing.assert_allclose(pa_trace.lines[-1].get_ydata(), truth)
        pa_corner_axis = views["corner"].axes[2 * 5 + 2]
        assert pa_corner_axis.get_xlim()[0] < truth < pa_corner_axis.get_xlim()[1]
        # Plotting must not rewrite the sampler's native wrapped angles.
        with np.load(folder / "samples.npz") as saved:
            np.testing.assert_array_equal(saved["chain"], chain)
    finally:
        plt.close("all")


@pytest.mark.parametrize("inclination_mean", [None, 60.])
def test_inclination_sampler_dimensions_and_metadata_without_mcmc(
        experiment, monkeypatch, inclination_mean):
    """Check sampler wiring and saved files with a stand-in, not a recovery run."""
    import emcee

    exp = ForwardExperiment(experiment.mock, inclination_mean=inclination_mean,
                            inclination_sigma=5.)
    ndim = len(exp.param_names)

    class StoredPositions:
        def __init__(self, walkers, dimensions, log_probability, *, vectorize):
            assert dimensions == ndim
            assert vectorize
            self.log_probability = log_probability
            self.acceptance_fraction = np.zeros(walkers)
            self.reset()

        def reset(self):
            self.iteration = 0
            self.chain = []

        def run_mcmc(self, positions, steps, *, progress):
            assert positions.shape == (12, ndim)
            return positions

        def sample(self, positions, *, iterations, progress):
            for _ in range(iterations):
                self.chain.append(positions.copy())
                self.iteration += 1
                yield positions

        def get_chain(self):
            return np.asarray(self.chain)

        def get_log_prob(self):
            return np.asarray([self.log_probability(points) for points in self.chain])

    monkeypatch.setattr(emcee, "EnsembleSampler", StoredPositions)
    folder = review.sample(exp, {"theta": exp.truth_theta}, warmup=1, draws=2,
                           walkers=12, progress=False)
    info = json.loads((folder / "run_info.txt").read_text())
    assert info["inclination_settings"] == exp.inclination_settings
    assert info["inclination_uncertainty_included"] == (inclination_mean is not None)
    assert ("true inclination" in info["condition_on"]) == (inclination_mean is None)
    assert info["morphology_uncertainty_included"] is False
    with np.load(folder / "samples.npz") as saved:
        assert saved["chain"].shape == (2, 12, ndim)
        assert saved["param_names"].tolist() == list(exp.param_names)
        assert np.isfinite(saved["log_prob"]).all()
    if inclination_mean is not None:
        with pytest.raises(ValueError, match="walkers must be an integer >= 12"):
            review.sample(exp, {"theta": exp.truth_theta}, walkers=10, progress=False)


def test_inclination_saved_draws_plot_in_degrees_and_reject_another_prior(experiment):
    """All plots use the sixth parameter and refuse to relabel a different prior."""
    exp = ForwardExperiment(experiment.mock, inclination_mean=60., inclination_sigma=5.)
    folder = exp.mock.directory / "direct_known_map"
    folder.mkdir()
    rng = np.random.default_rng(6)
    chain = np.broadcast_to(exp.truth_theta, (30, 12, 6)).copy()
    chain += rng.normal(size=chain.shape) * np.array([1., .05, .02, .01, .01, .03])
    np.savez_compressed(folder / "samples.npz", chain=chain,
                        log_prob=np.zeros((30, 12)), param_names=np.array(exp.param_names))
    (folder / "run_info.txt").write_text(json.dumps({
        "input_sha256": exp.mock.metadata["input_sha256"],
        "inclination_settings": exp.inclination_settings, "status": "test_fixture"}))
    views = review.show_results(exp, folder)
    try:
        q = np.quantile(np.rad2deg(chain[..., 5]), [.16, .5, .84])
        row = views["summary"].loc["i [deg]"]
        np.testing.assert_allclose([row["median"], row["minus_16th"], row["plus_84th"]],
                                   [q[1], q[1] - q[0], q[2] - q[1]])
        assert row["truth"] == pytest.approx(60.)
        assert len(views["corner"].axes) == 36
        assert views["traces"].axes[5].get_visible()
        assert views["traces"].axes[5].get_ylabel() == "i [deg]"
        assert (folder / "corner.pdf").is_file()
    finally:
        plt.close("all")

    for other in (experiment, ForwardExperiment(exp.mock, inclination_mean=60., inclination_sigma=10.)):
        with pytest.raises(ValueError, match="different inclination settings"):
            review.show_results(other, folder)
    # Legacy five-parameter results must not be interpreted as this new model.
    info = json.loads((folder / "run_info.txt").read_text())
    info.pop("inclination_settings")
    (folder / "run_info.txt").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="fixed inclination"):
        review.show_results(exp, folder)


def test_inclination_free_map_display_records_prior_not_fixed_truth(experiment):
    """Exercise reporting with a fabricated profile result, without optimizing."""
    exp = ForwardExperiment(experiment.mock, inclination_mean=60., inclination_sigma=5.)
    predicted = [a.numpy() for a in exp.predict(exp.truth_theta)]
    result = dict(theta=exp.truth_theta.copy(), emission=exp.emission.numpy(),
                  predicted_R=predicted[0], predicted_C=predicted[1], noisy=True,
                  success=False, message="test fixture", optimality=0.)
    views = review.show_free_result(exp, result)
    try:
        assert views["summary"].loc["i [deg]", "fitted"] == pytest.approx(60.)
        folder = exp.mock.directory / "direct_free_map"
        info = json.loads((folder / "run_info.txt").read_text())
        assert "velocity.inc_v" not in info["fixed_parameters"]
        assert info["inclination_settings"] == exp.inclination_settings
        assert info["posterior_errors_available"] is False
    finally:
        plt.close("all")

