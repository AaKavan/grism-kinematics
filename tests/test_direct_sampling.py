"""Independent probability and checkpoint checks for the joint NUTS driver.

These tests use a one-pixel problem whose brightness can be integrated out
analytically. No grism files, generating brightness map, or GPU are required.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
numpyro = pytest.importorskip("numpyro")
import jax.numpy as jnp
import numpyro.distributions as dist

from dingo.direct_sampling import JointSampler
from dingo.direct_sampling import _validate_retained_checkpoint


class OnePixelModel:
    """Two measurements of one unknown positive brightness and one velocity.

    R measures ``velocity * emission``; C measures ``emission``. The name
    velocity keeps the driver interface realistic without pretending this is
    a galaxy simulation. Its marginalized target is known independently.
    """

    site_names = ("velocity",)
    sigma = np.array([.4, .3])
    data = np.array([.8, 1.1])
    scale = 1.3
    bounds = (.1, 2.5)

    def model(self):
        velocity = numpyro.sample("velocity", dist.Uniform(*self.bounds))
        emission = numpyro.sample(
            "emission", dist.HalfNormal(self.scale).expand((1, 1)).to_event(2))
        numpyro.deterministic("theta", jnp.stack([velocity]))
        prediction = jnp.array([velocity, 1.]) * emission[0, 0]
        numpyro.factor("two_measurements", -.5*jnp.sum(
            ((prediction-jnp.asarray(self.data))/jnp.asarray(self.sigma))**2))

    @staticmethod
    def initial_values():
        return [dict(velocity=jnp.array(.5), emission=jnp.array([[.7]])),
                dict(velocity=jnp.array(1.4), emission=jnp.array([[1.3]]))]


@pytest.fixture
def double_precision():
    """Keep the generic driver test independent of the real-model import."""
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def make_sampler(*, warmup=20, seed=9):
    model = OnePixelModel()
    return JointSampler(model, model.initial_values(), warmup=warmup,
                        seed=seed, target_accept=.85, max_tree_depth=7)


def assert_identical_state(left, right):
    assert left.iterations == right.iterations
    for key, array in left.state_arrays().items():
        np.testing.assert_array_equal(array, right.state_arrays()[key])


def test_warmup_chunking_and_restored_rng_preserve_the_exact_trajectory(double_precision):
    """Chunking must neither restart adaptation nor repeat random draws."""
    uninterrupted = make_sampler()
    chunked = make_sampler()
    _, all_stats = uninterrupted.advance(20, collect=False)
    _, first_stats = chunked.advance(7, collect=False)
    _, second_stats = chunked.advance(13, collect=False)
    assert_identical_state(uninterrupted, chunked)
    np.testing.assert_array_equal(
        all_stats["iteration"], np.concatenate(
            [first_stats["iteration"], second_stats["iteration"]], axis=1))
    assert np.all(all_stats["is_warmup"])
    assert uninterrupted.iterations == 20

    # A different construction seed must be completely superseded by the
    # saved state, including its random keys and adapted mass matrix.
    restored = make_sampler(seed=123)
    restored.restore(chunked.state_arrays())
    expected_samples, expected_stats = uninterrupted.advance(15)
    first_samples, first_stats = restored.advance(6)
    second_samples, second_stats = restored.advance(9)
    assert_identical_state(uninterrupted, restored)
    for key in expected_samples:
        actual = np.concatenate([first_samples[key], second_samples[key]], axis=1)
        np.testing.assert_array_equal(expected_samples[key], actual)
    for key in expected_stats:
        actual = np.concatenate([first_stats[key], second_stats[key]], axis=1)
        np.testing.assert_array_equal(expected_stats[key], actual)
    assert not expected_stats["is_warmup"].any()
    np.testing.assert_array_equal(expected_stats["iteration"],
                                  np.tile(np.arange(21, 36), (2, 1)))
    assert expected_samples["emission"].shape == (2, 15, 1, 1)


def test_restore_rejects_incompatible_arrays_and_disagreeing_chain_counts(double_precision):
    sampler = make_sampler()
    arrays = sampler.state_arrays()
    removed = dict(arrays)
    removed.pop(next(iter(removed)))
    with pytest.raises(ValueError, match="incompatible sampler state"):
        sampler.restore(removed)
    reshaped = dict(arrays)
    key = next(iter(reshaped))
    reshaped[key] = reshaped[key].reshape(-1, 1)
    with pytest.raises(ValueError, match="shapes or dtypes"):
        sampler.restore(reshaped)
    sampler.state = sampler.state._replace(i=jnp.array([1, 2]))
    with pytest.raises(ValueError, match="iteration counters disagree"):
        _ = sampler.iterations


def test_initial_zero_brightness_is_rejected_before_sampling(double_precision):
    model = OnePixelModel()
    initial = [dict(velocity=jnp.array(.7), emission=jnp.zeros((1, 1)))]
    with pytest.raises(ValueError, match="strictly inside"):
        JointSampler(model, initial, warmup=20)


def test_joint_draws_match_an_independently_marginalized_brightness_target(double_precision):
    """Velocity errors include the unknown map, not its conditional optimum."""
    from scipy.integrate import cumulative_trapezoid
    from scipy.special import log_ndtr

    model = OnePixelModel()
    grid = np.linspace(*model.bounds, 20001)
    inverse_variance = 1/model.sigma**2
    # Complete the square in the Gaussian likelihood times HalfNormal prior.
    # Integral over s>=0 is exp(b²/2q-c/2) * sqrt(2*pi/q) * Phi(b/sqrt(q)),
    # apart from the prior normalization. There is no profile fit in this check.
    q = inverse_variance[0]*grid**2 + inverse_variance[1] + model.scale**-2
    b = inverse_variance[0]*grid*model.data[0] + inverse_variance[1]*model.data[1]
    log_density = .5*b*b/q - .5*np.log(q) + log_ndtr(b/np.sqrt(q))
    density = np.exp(log_density-log_density.max())
    cdf = cumulative_trapezoid(density, grid, initial=0.)
    cdf /= cdf[-1]

    sampler = make_sampler(warmup=250, seed=19)
    sampler.advance(250, collect=False)
    samples, stats = sampler.advance(1500)
    velocity = samples["velocity"].ravel()
    # The probability integral transform is uniform for the analytical target.
    transformed = np.sort(np.interp(velocity, grid, cdf))
    empirical = (np.arange(len(transformed))+.5)/len(transformed)
    assert np.max(np.abs(transformed-empirical)) < .06
    expected_quantiles = np.interp([.16, .5, .84], cdf, grid)
    np.testing.assert_allclose(np.quantile(velocity, [.16, .5, .84]),
                               expected_quantiles, atol=.07, rtol=0.)
    assert np.std(samples["emission"]) > .1
    assert np.all(samples["emission"] > 0.)
    assert not stats["is_warmup"].any()
    assert stats["diverging"].sum() == 0


def test_corrupt_retained_history_is_rejected_before_resume():
    saved = dict(iterations=np.array(12), warmup=np.array(10))
    with pytest.raises(ValueError, match="missing retained"):
        _validate_retained_checkpoint(saved, chains=2, warmup=10, draws=20,
                                     site_names=("velocity",))
    for name in ("velocity", "theta", "emission"):
        saved[f"sample_{name}"] = np.ones((2, 2))
    for name in ("energy", "potential_energy", "num_steps", "accept_prob", "diverging", "is_warmup"):
        saved[f"stats_{name}"] = np.zeros((2, 2))
    saved["stats_iteration"] = np.array([[11, 12], [11, 12]])
    _validate_retained_checkpoint(saved, chains=2, warmup=10, draws=20,
                                 site_names=("velocity",))
    saved["stats_iteration"][1, 1] = 11
    with pytest.raises(ValueError, match="duplicated"):
        _validate_retained_checkpoint(saved, chains=2, warmup=10, draws=20,
                                     site_names=("velocity",))


def test_interrupted_real_interface_reopens_partial_draws_and_resumes(tmp_path, grism_cal, monkeypatch):
    """A lost notebook kernel must not discard or duplicate retained steps."""
    import json
    import matplotlib.pyplot as plt
    from dingo import direct_workflow as direct
    from dingo.simple_simulation import make_mock

    mock = make_mock(tmp_path / "mock", size=21, center=(10.3, 10.7),
                     effective_radius=3., noise_sigma=.02)
    run = direct.prepare(
        mock.source_id, data_dir=mock.data_dir, output_root=tmp_path / "runs",
        run_name="interrupted", center=(10.3, 10.7), inclination_mean=60.,
        noise_sigma=.02, start_velocity=200., start_turnover=5.,
        start_pa=35., start_offset=(0., 0.))
    direct.fit(run, max_evaluations=1, progress=False)
    original = JointSampler.advance

    def interrupt_after_first_draw_block(self, count, *, collect=True):
        if self.iterations >= 12:
            raise KeyboardInterrupt("test interruption")
        return original(self, count, collect=collect)

    settings = dict(warmup=8, draws=8, chains=2, max_tree_depth=3,
                    checkpoint_every=4, device="cpu", progress=False)
    monkeypatch.setattr(JointSampler, "advance", interrupt_after_first_draw_block)
    with pytest.raises(KeyboardInterrupt):
        direct.sample(run, **settings)
    with np.load(run.directory / "posterior.npz", allow_pickle=False) as saved:
        prior_draws = saved["sample_theta"].copy()
        assert prior_draws.shape == (2, 4, 6)
        assert not saved["stats_is_warmup"].any()
        assert "prediction_theta" in saved
    record = json.loads((run.directory / "run_info.txt").read_text())
    assert record["mcmc"]["status"] == "interrupted"
    assert record["status"] == "mcmc_interrupted"
    assert record["mcmc"]["retained_draws"] == 4
    # The atomic posterior contains everything required for inspection, even
    # if the redundant standalone prediction file is absent.
    (run.directory / "posterior_prediction.npz").unlink()
    try:
        views = direct.show_posterior(run.directory)
        assert "corner" in views and "residuals" in views
    finally:
        plt.close("all")
    reopened = direct.load_run(run.directory)
    monkeypatch.setattr(JointSampler, "advance", original)
    direct.sample(reopened, resume=True, **settings)
    record = json.loads((run.directory / "run_info.txt").read_text())
    assert record["status"] == "mcmc_completed"
    with np.load(run.directory / "posterior.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["sample_theta"][:, :4], prior_draws)
        assert saved["sample_theta"].shape == (2, 8, 6)
        np.testing.assert_array_equal(saved["stats_iteration"],
                                      np.tile(np.arange(9, 17), (2, 1)))
