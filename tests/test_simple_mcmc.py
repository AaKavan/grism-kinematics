"""The minimal notebook must keep one target and preserve saved experiments."""
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from dingo import simple_mcmc as sm


@pytest.fixture
def settings(tmp_path, grism_cal, staged):
    staged("ID6170")
    return dict(data_dir=Path(__file__).resolve().parents[1] / "data",
                output_root=tmp_path, run_name="check", device="cpu",
                start_velocity=300.0, start_inclination=75.0,
                inclination_mean=75.0, inclination_sigma=5.0)


def test_starts_share_one_deterministic_original_residual_target(settings):
    first = sm.prepare(6170, **settings)
    alternate = dict(settings, run_name="other_start", start_velocity=400,
                     start_pa=120, start_turnover=8, start_offset=(1, -.5))
    second = sm.prepare(6170, **alternate)
    f = first.fitter
    assert f.mcmc_param_names == second.fitter.mcmc_param_names
    assert len(f.mcmc_param_names) == 8
    assert "noise.ln_f" not in f.mcmc_param_names
    theta = first.theta.copy()
    score = f.log_probability(theta)
    images = [x.copy() for x in f.get_fitting_results()[:2]]
    f.log_probability(second.theta)
    assert f.log_probability(theta) == score
    for expected, actual in zip(images, f.get_fitting_results()[:2]):
        np.testing.assert_array_equal(expected, actual)
    assert second.fitter.log_probability(theta) == score
    np.testing.assert_array_equal(f.mcmc_foot_R, second.fitter.mcmc_foot_R)
    np.testing.assert_array_equal(f.mcmc_foot_C, second.fitter.mcmc_foot_C)
    scalar = [f.log_probability(p) for p in [theta, second.theta]]
    np.testing.assert_allclose(f.log_probability_batch(np.array([theta, second.theta])),
                               scalar, rtol=1e-6, atol=.03)
    expected_ll = -.5 * (np.sum((images[0] - images[1])**2) / f.mcmc_var_base
                         + f._mcmc_lnvar_const)
    np.testing.assert_allclose(score - f.log_prior(theta), expected_ll, rtol=1e-6)


def test_invalid_overwrite_keeps_previous_experiment(settings, monkeypatch):
    run = sm.prepare(6170, **settings)
    original = (run.directory / "run_info.txt").read_bytes()

    def broken_initialization(*args, **kwargs):
        raise RuntimeError("Example unavailable calibration")

    monkeypatch.setattr(sm, "_prepare_fitter", broken_initialization)
    with pytest.raises(RuntimeError, match="calibration"):
        sm.prepare(6170, **settings, overwrite=True)
    assert (run.directory / "run_info.txt").read_bytes() == original


def test_reopen_preserves_target_and_selected_start(settings):
    run = sm.prepare(6170, **settings)
    sm.fit_local(run, max_evaluations=4, progress=False)
    reopened = sm.open_run(run.directory, device="cpu")
    np.testing.assert_array_equal(run.theta, reopened.theta)
    assert run.fitter.log_probability(run.theta) == reopened.fitter.log_probability(reopened.theta)
    assert (run.directory / "run_info.txt").is_file()


def test_interrupted_warmup_is_not_exported_as_posterior(settings, monkeypatch):
    run = sm.prepare(6170, **settings)

    def interrupted(**kwargs):
        f = run.fitter
        f._mcmc_raw_chain = np.tile(run.theta, (2, 20, 1))
        f._mcmc_raw_log_prob = np.zeros((2, 20))
        f.mcmc_sampler = SimpleNamespace(iteration=2, acceptance_fraction=np.zeros(20))

    monkeypatch.setattr(run.fitter, "fit_MCMC", interrupted)
    sm.sample(run, warmup=10, draws=20, walkers=20, progress=False)
    with np.load(run.directory / "samples.npz", allow_pickle=False) as data:
        assert data["chain"].shape == (0, 20, 8)
        assert not data["completed"]
    assert not (run.directory / "prediction.npz").exists()
    with pytest.raises(ValueError, match="warmup"):
        sm.show_results(run.directory)


def test_refuses_output_alias_into_source_inputs(settings):
    source = settings["data_dir"] / "ID6170"
    (settings["output_root"] / "6170").symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="separate"):
        sm.prepare(6170, **settings, overwrite=True)


def test_completed_resume_preserves_draws_and_acceptance(settings):
    run = sm.prepare(6170, **settings)
    sm.sample(run, warmup=2, draws=4, walkers=20, seed=4, progress=False)
    with np.load(run.directory / "samples.npz", allow_pickle=False) as saved:
        chain, acceptance = saved["chain"].copy(), saved["acceptance_fraction"].copy()
    restored = sm.open_run(run.directory, device="cpu")
    sm.sample(restored, warmup=2, draws=4, walkers=20, seed=4, resume=True, progress=False)
    with np.load(run.directory / "samples.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["chain"], chain)
        np.testing.assert_array_equal(saved["acceptance_fraction"], acceptance)
    assert np.isfinite(restored.info["diagnostics"]["acceptance_fraction"]).all()
    sm.sample(restored, warmup=2, draws=5, walkers=20, seed=4, resume=True, progress=False)
    with np.load(run.directory / "samples.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["chain"][:4], chain)
        assert saved["chain"].shape == (5, 20, 8)


def test_fixed_geometry_excluded_from_fit_and_survives_reopen(settings):
    run = sm.prepare(6170, **settings, fixed_x0=40., fixed_y0=41., fixed_inclination=70.)
    f = run.fitter
    expected = ["velocity.V_rot", "velocity.R_v", "velocity.theta_v", "image.R.dx", "image.R.dy"]
    assert f.mcmc_param_names == expected
    assert set(run.info["fixed_parameters"]) == {"velocity.x0_v", "velocity.y0_v", "velocity.inc_v"}
    assert all(n not in f.config["mcmc"]["priors"] for n in run.info["fixed_parameters"])
    prior_fingerprint = f.mcmc_fingerprint()
    theta = run.theta.copy()
    proposals = np.stack([theta, theta.copy()])
    proposals[1, 0] += 10
    np.testing.assert_allclose(f.log_probability_batch(proposals),
        [f.log_probability(t) for t in proposals], rtol=1e-6, atol=.03)
    sm.fit_local(run, max_evaluations=4, progress=False)
    assert f.mcmc_fingerprint()["fixed_parameters"] == prior_fingerprint["fixed_parameters"]
    assert f.velocity_cfg["velocity.x0_v"].value == 40.
    assert f.velocity_cfg["velocity.y0_v"].value == 41.
    assert f.velocity_cfg["velocity.inc_v"].value == pytest.approx(np.deg2rad(70.), abs=1e-7)
    reopened = sm.open_run(run.directory, device="cpu")
    assert reopened.fitter.mcmc_param_names == expected
    assert reopened.fitter.mcmc_fingerprint() == f.mcmc_fingerprint()
    assert reopened.info["fixed_parameters"] == run.info["fixed_parameters"]
    import matplotlib.pyplot as plt
    figure = sm.show_fit(reopened)
    assert "Fixed:" in figure._suptitle.get_text()
    plt.close(figure)


def test_one_fixed_coordinate_leaves_other_coordinates_sampled(settings):
    run = sm.prepare(6170, **settings, fixed_x0=40.)
    names = run.fitter.mcmc_param_names
    assert "velocity.x0_v" not in names
    assert {"velocity.y0_v", "velocity.inc_v"}.issubset(names)
    assert len(names) == 7


def test_fixed_geometry_saved_and_changed_value_rejects_resume(settings):
    run = sm.prepare(6170, **settings, fixed_x0=40., fixed_y0=40., fixed_inclination=75.)
    sm.sample(run, warmup=1, draws=3, walkers=20, seed=4, progress=False)
    assert Path(run.fitter.config["mcmc"]["checkpoint"]).is_file()
    for filename in ("samples.npz", "prediction.npz"):
        with np.load(run.directory / filename, allow_pickle=False) as saved:
            assert len(saved["param_names"]) == 5
            actual = dict(zip(saved["fixed_names"].tolist(), saved["fixed_values"]))
            assert actual == run.info["fixed_parameters"]
    views = sm.show_results(run.directory, save=False)
    assert len(views["summary"]) == 5
    import matplotlib.pyplot as plt
    for figure in views.values():
        if hasattr(figure, "savefig"):
            plt.close(figure)
    reopened = sm.open_run(run.directory, device="cpu")
    sm.sample(reopened, warmup=1, draws=3, walkers=20, seed=4, resume=True, progress=False)
    reopened.fitter.set_params_dict({"velocity.x0_v": 40.5})
    with pytest.raises(RuntimeError, match="fixed_parameters changed"):
        sm.sample(reopened, warmup=1, draws=4, walkers=20, seed=4, resume=True, progress=False)


def test_center_prior_penalty_is_used_by_scalar_and_batched_posterior(settings, monkeypatch):
    """A one-sigma center move costs 0.5 log probability in either backend."""
    run = sm.prepare(6170, **settings, center_mean=(39., 41.), center_sigma=(1., 2.))
    f = run.fitter
    names = f.mcmc_param_names
    assert {"velocity.x0_v", "velocity.y0_v", "velocity.inc_v"}.issubset(names)
    mean = run.theta.copy()
    ix, iy = names.index("velocity.x0_v"), names.index("velocity.y0_v")
    mean[ix], mean[iy] = 39., 41.
    proposals = np.tile(mean, (4, 1))
    proposals[1, ix] += 1.
    proposals[2, iy] += 2.
    proposals[3, ix] += 1.
    proposals[3, iy] += 2.

    # Remove the image term here to test the actual posterior's prior weight,
    # rather than merely checking that arguments were copied into a dictionary.
    monkeypatch.setattr(f, "log_likelihood", lambda **extras: 0.)
    monkeypatch.setattr(f, "log_likelihood_batch", lambda values: np.zeros(len(values)))
    scalar = np.array([f.log_probability(point) for point in proposals])
    batch = f.log_probability_batch(proposals)
    np.testing.assert_allclose(scalar - scalar[0], [0., -.5, -.5, -1.])
    np.testing.assert_allclose(batch, scalar)


def test_center_prior_default_is_independent_of_start(settings):
    run = sm.prepare(6170, **settings, start_center=(39., 42.))
    for axis in ("x", "y"):
        prior = run.info["priors"][f"velocity.{axis}0_v"]
        assert prior["mu"] == 40.
        assert prior["sigma"] == 5.
    assert run.info["start_parameters"]["velocity.x0_v"] == 39.
    assert run.info["start_parameters"]["velocity.y0_v"] == 42.


def test_center_priors_saved_reopened_and_bound_to_checkpoint_identity(settings):
    run = sm.prepare(6170, **settings, center_mean=(39., 41.), center_sigma=1.)
    saved = json.loads((run.directory / "run_info.txt").read_text())
    for axis, mean in (("x", 39.), ("y", 41.)):
        prior = saved["priors"][f"velocity.{axis}0_v"]
        assert prior["mu"] == mean
        assert prior["sigma"] == 1.
        assert prior == saved["configuration"]["mcmc"]["priors"][f"velocity.{axis}0_v"]
    reopened = sm.open_run(run.directory, device="cpu")
    assert reopened.fitter.mcmc_fingerprint() == run.fitter.mcmc_fingerprint()
    assert reopened.fitter.log_probability(run.theta) == run.fitter.log_probability(run.theta)
    other = sm.prepare(6170, **dict(settings, run_name="wider"),
                       center_mean=(39., 41.), center_sigma=2.)
    assert other.fitter.mcmc_fingerprint() != run.fitter.mcmc_fingerprint()


def test_fixed_coordinate_overrides_only_its_center_prior(settings):
    run = sm.prepare(6170, **settings, center_mean=(39., 41.),
                     center_sigma=(1., 2.), fixed_x0=40.)
    assert "velocity.x0_v" not in run.info["priors"]
    assert "velocity.x0_v" not in run.fitter.mcmc_param_names
    assert run.info["fixed_parameters"]["velocity.x0_v"] == 40.
    assert run.info["priors"]["velocity.y0_v"]["mu"] == 41.
    assert run.info["priors"]["velocity.y0_v"]["sigma"] == 2.
    assert "velocity.y0_v" in run.fitter.mcmc_param_names


@pytest.mark.parametrize("option,value", [
    ("center_sigma", 0.), ("center_sigma", -1.), ("center_sigma", np.nan),
    ("center_sigma", True), ("center_sigma", (1., np.inf)),
    ("center_sigma", (1.,)), ("center_mean", (40.,)),
    ("center_mean", (np.nan, 40.)), ("center_mean", (80., 40.)),
])
def test_invalid_center_prior_does_not_replace_existing_run(settings, option, value):
    """Prior validation must precede deletion, even with overwrite requested."""
    directory = settings["output_root"] / "6170" / settings["run_name"]
    directory.mkdir(parents=True)
    marker = directory / "run_info.txt"
    original = json.dumps({"schema": sm.SCHEMA, "preserve_this": True})
    marker.write_text(original)
    with pytest.raises(ValueError, match=option):
        sm.prepare(6170, **settings, overwrite=True, **{option: value})
    assert marker.read_text() == original


@pytest.mark.parametrize("noise_sigma,expected", [(.02, (.02, .02)), ((.01, .03), (.01, .03))])
def test_supplied_noise_controls_actual_likelihood_without_estimation(settings, monkeypatch,
                                                                    noise_sigma, expected):
    """Known input noise sets the same full-image Gaussian score used by MCMC."""
    from dingo import fitting

    def unexpected_estimation(*args, **kwargs):
        raise AssertionError("Explicit noise must not be replaced by an image estimate.")

    monkeypatch.setattr(fitting, "estimate_background_sigma", unexpected_estimation)
    run = sm.prepare(6170, **settings, noise_sigma=noise_sigma, noise_correlation_area=2.)
    fitter = run.fitter
    variance = 2. * sum(sigma**2 for sigma in expected)
    assert (fitter.sigma_R, fitter.sigma_C) == expected
    assert fitter.mcmc_var_base == pytest.approx(variance)
    score = fitter.log_probability(run.theta)
    r, c = fitter.get_fitting_results()[:2]
    expected_log_likelihood = -.5 * (np.sum((r - c)**2) / variance
                                      + r.size * np.log(2 * np.pi * variance))
    assert score - fitter.log_prior(run.theta) == pytest.approx(expected_log_likelihood, abs=.01)
    assert run.info["noise"]["sigma_source"] == "supplied"
    assert "supplied explicitly" in run.info["noise_note"]


def test_supplied_noise_survives_reopening_and_changes_checkpoint_identity(settings):
    run = sm.prepare(6170, **settings, noise_sigma=(.01, .02))
    reopened = sm.open_run(run.directory, device="cpu")
    assert reopened.fitter.mcmc_fingerprint() == run.fitter.mcmc_fingerprint()
    assert reopened.fitter.log_probability(run.theta) == run.fitter.log_probability(run.theta)
    saved = json.loads((run.directory / "run_info.txt").read_text())
    assert saved["noise"]["sigma_source"] == "supplied"
    assert saved["configuration"]["mcmc"]["noise"]["sigma_R"] == .01
    assert saved["configuration"]["mcmc"]["noise"]["sigma_C"] == .02
    different = sm.prepare(6170, **dict(settings, run_name="different_noise"), noise_sigma=(.02, .02))
    assert different.fitter.mcmc_fingerprint() != run.fitter.mcmc_fingerprint()


@pytest.mark.parametrize("value", [0., -1., np.nan, np.inf, True, (.01,), (.01, 0.), (True, .02), "invalid"])
def test_invalid_noise_does_not_replace_existing_run(settings, value):
    """Invalid simulation variance cannot delete a saved experiment."""
    directory = settings["output_root"] / "6170" / settings["run_name"]
    directory.mkdir(parents=True)
    marker = directory / "run_info.txt"
    original = json.dumps({"schema": sm.SCHEMA, "preserve_this": True})
    marker.write_text(original)
    with pytest.raises(ValueError, match="noise_sigma"):
        sm.prepare(6170, **settings, overwrite=True, noise_sigma=value)
    assert marker.read_text() == original
