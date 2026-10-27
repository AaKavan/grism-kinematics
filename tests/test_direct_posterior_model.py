"""Numerical and prior checks for joint direct-image posterior sampling."""
import numpy as np
import pytest
from astropy.io import fits

jax = pytest.importorskip("jax")
pytest.importorskip("numpyro")
import jax.numpy as jnp
from numpyro.infer.util import log_density

from dingo.direct_posterior_model import DirectPosteriorModel
from dingo.direct_source import SourceModel
from dingo.simple_simulation import make_mock


@pytest.fixture
def inputs(tmp_path, grism_cal):
    mock = make_mock(tmp_path / "mock", size=31, center=(15.3, 15.7),
                     turnover=3., effective_radius=4.)
    source = SourceModel(mock.source_id, data_dir=mock.data_dir,
                         center=(15.3, 15.7), inclination_mean=60.,
                         inclination_sigma=5., noise_sigma=.02)
    theta = np.array([mock.truth[k] for k in source.param_names])
    with np.load(mock.directory / "truth.npz") as stored:
        emission = stored["image"].copy()
    return mock, source, theta, emission


def test_forward_model_matches_torch_to_float32_precision(inputs):
    _, source, theta, emission = inputs
    model = DirectPosteriorModel(source, emission_prior_scale=2.)
    points = np.stack([theta, theta + [30., .4, .037, .17, -.11, .021]])
    for point in points:
        coordinates = model.mapping_coordinates(point)
        torch_coordinates = source.mapping_coordinates(point)
        for pupil in ("R", "C"):
            for actual, expected in zip(coordinates[pupil], torch_coordinates[pupil]):
                # Float32 wavelength rounding corresponds to about 4e-4 pixel
                # here; the new arithmetic keeps the same table constants.
                np.testing.assert_allclose(actual, expected.numpy(), rtol=0., atol=5e-4)
        for actual, expected in zip(model.predict(point, emission), source.predict(point, emission)):
            np.testing.assert_allclose(actual, expected.numpy(), rtol=0., atol=2e-4)
    batched = jax.jit(jax.vmap(model.predict, in_axes=(0, None)))(jnp.asarray(points), jnp.asarray(emission))
    assert batched[0].shape == (2, *source.shape)
    assert batched[0].dtype == jnp.float64


def test_kinematic_and_source_gradients_match_finite_difference(inputs):
    _, source, theta, emission = inputs
    model = DirectPosteriorModel(source, emission_prior_scale=2.)
    point = theta + [31.2, .411, .0371, .171, -.111, .0211]
    score = jax.jit(model.chi2)
    gradient = np.asarray(jax.jit(jax.grad(model.chi2))(jnp.asarray(point), jnp.asarray(emission)))
    steps = np.array([1e-4, 1e-6, 1e-7, 1e-7, 1e-7, 1e-7])
    finite = []
    for k, step in enumerate(steps):
        delta = np.zeros(6)
        delta[k] = step
        finite.append(float((score(point + delta, emission) - score(point - delta, emission)) / (2 * step)))
    np.testing.assert_allclose(gradient, finite, rtol=4e-4, atol=.02)
    map_gradient = np.asarray(jax.grad(model.chi2, argnums=1)(point, emission))
    delta = np.zeros(source.shape)
    delta[15, 15] = 1e-6
    finite_map = float((score(point, emission + delta) - score(point, emission - delta)) / 2e-6)
    assert map_gradient[15, 15] == pytest.approx(finite_map, rel=1e-6, abs=1e-5)
    assert np.isfinite(map_gradient).all()


def test_likelihood_keeps_separate_noise_and_fixed_invalid_mask(inputs):
    mock, _, theta, emission = inputs
    path = mock.data_dir / f"ID{mock.source_id}" / "C.fits"
    pixels = fits.getdata(path)
    pixels[0, 0] = np.nan
    fits.writeto(path, pixels, overwrite=True)
    source = SourceModel(mock.source_id, data_dir=mock.data_dir,
                         center=(15.3, 15.7), inclination_mean=60.,
                         inclination_sigma=5., noise_sigma=(.02, .04))
    model = DirectPosteriorModel(source, emission_prior_scale=2.)
    predictions = model.predict(theta, emission)
    expected = 0.
    for pupil, prediction in zip(("R", "C"), predictions):
        valid = np.asarray(model.valid_by_pupil[pupil])
        residual = ((prediction - model.data[pupil]) / model.sigma_by_pupil[pupil])[valid]
        expected += float(np.sum(np.asarray(residual)**2))
    assert float(model.chi2(theta, emission)) == pytest.approx(expected, abs=1e-9)
    assert np.isfinite(jax.grad(model.chi2)(theta, emission)).all()


def test_joint_density_includes_map_prior_and_both_original_images(inputs):
    _, source, theta, emission = inputs
    model = DirectPosteriorModel(source, emission_prior_scale=.7)
    sites = model.initial_values(theta, emission)
    logp, trace = log_density(model.model, (), {}, sites)
    point = jnp.stack([sites[name] for name in model.site_names])
    expected = model.log_prior(point, sites["emission"]) - .5 * model.chi2(point, sites["emission"])
    assert float(logp) == pytest.approx(float(expected), abs=1e-8)
    assert trace["emission"]["type"] == "sample"
    assert trace["emission"]["is_observed"] is False
    np.testing.assert_allclose(trace["theta"]["value"], point)
    changed = point.at[5].add(np.deg2rad(5.))
    assert float(model.log_prior(changed, emission) - model.log_prior(point, emission)) == pytest.approx(-.5)
    doubled = model.log_prior(point, 2 * emission) - model.log_prior(point, emission)
    assert float(doubled) == pytest.approx(float(-1.5 * np.sum(emission**2) / .7**2), rel=1e-6)
    assert np.isneginf(model.log_prior(point, -emission))


def test_initialization_is_positive_without_fixing_morphology(inputs):
    _, source, theta, _ = inputs
    model = DirectPosteriorModel(source, emission_prior_scale=1.)
    sites = model.initial_values(theta, np.zeros(source.shape))
    np.testing.assert_allclose(sites["emission"], .002)
    # Values far below the initialization floor remain in posterior support.
    assert np.isfinite(model.log_prior(theta, np.full(source.shape, 1e-12)))
    for bad in (0., -1., np.nan, True, [1., 2.]):
        with pytest.raises(ValueError, match="emission_prior_scale"):
            DirectPosteriorModel(source, emission_prior_scale=bad)
    with pytest.raises(ValueError, match="nonnegative"):
        model.initial_values(theta, -np.ones(source.shape))
    with pytest.raises(ValueError, match="within their bounds"):
        model.initial_values(theta + [2000., 0., 0., 0., 0., 0.], np.zeros(source.shape))


def test_calibration_identity_is_reproducible_and_includes_cutout(inputs):
    _, source, _, _ = inputs
    first = DirectPosteriorModel(source, emission_prior_scale=1.)
    second = DirectPosteriorModel(source, emission_prior_scale=2.)
    assert first.calibration_hash == second.calibration_hash
    x, y, width, height = source.cutouts["R"]
    source.cutouts["R"] = (x + 1, y, width, height)
    changed = DirectPosteriorModel(source, emission_prior_scale=1.)
    assert changed.calibration_hash != first.calibration_hash


def test_fixed_rdy_is_deterministic_and_absent_from_latent_coordinates(repo, staged, grism_cal):
    """Inspect the conditioned model trace; this does not run MCMC or a fit."""
    from pathlib import Path
    from numpyro.infer.util import unconstrain_fn

    staged("ID6170")
    source = SourceModel(6170, data_dir=Path(repo) / "data", fixed_rdy=.271)
    model = DirectPosteriorModel(source, emission_prior_scale=.4)
    theta = np.array([189., 3.2, np.deg2rad(306.5), 1.4, -.711, np.deg2rad(80.)])
    emission = np.full(source.shape, .005)
    sites = model.initial_values(theta, emission)
    assert "rdy" not in sites
    assert model.site_names == ("velocity", "turnover", "pa", "rdx", "inclination")
    assert model.parameter_sites[4] == "rdy"
    coordinates = unconstrain_fn(model.model, (), {}, sites)
    assert "rdy" not in coordinates
    logp, trace = log_density(model.model, (), {}, sites)
    assert trace["rdy"]["type"] == "deterministic"
    assert float(trace["rdy"]["value"]) == .271
    assert trace["theta"]["value"].shape == (6,)
    assert float(trace["theta"]["value"][4]) == .271
    assert all(trace[name]["type"] == "sample" for name in model.site_names)
    point = trace["theta"]["value"]
    expected = model.log_prior(point, sites["emission"]) - .5*model.chi2(point, sites["emission"])
    assert float(logp) == pytest.approx(float(expected), abs=1e-8)
    assert np.isneginf(model.log_prior(theta, emission))
    gradient = np.asarray(jax.grad(model.chi2)(point, sites["emission"]))
    assert gradient[4] == 0.
    assert abs(gradient[3]) > 0.
    for actual, expected in zip(model.predict(point, emission), source.predict(point, emission)):
        np.testing.assert_allclose(actual, expected.numpy(), rtol=0., atol=2e-5)
