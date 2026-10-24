"""Cross-backend physical/gradient checks and proper-prior validation."""

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
pytest.importorskip("numpyro")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from numpyro.infer import Predictive
from numpyro.infer.util import log_density

from dingo.native_forward import NativeLineProjector, projected_arctan_velocity
from dingo.native_fitting import BilinearLineMap, StripContinuumLikelihood
from dingo.native_jax import JaxNativeProjector, JaxLineMap, JaxStripLikelihood, projected_velocity
from dingo.native_posterior import NativePosterior, PosteriorPrior, log_field_basis, radial_profiles


def example():
    grid = BilinearLineMap(3, 7, half_extent=.3, flux_unit=2., dtype=torch.float64)
    coeff = np.zeros((2, 49, 2, 7))
    coeff[:, :, 0, 0] = 10.2 + grid.east.numpy()/.06
    coeff[:, :, 1, 0] = 11.4 + grid.north.numpy()/.06
    coeff[0, :, 0, 1] = 1000; coeff[1, :, 1, 1] = -900
    coeff[:, :, :, 2] = 2.
    kernel = np.array([[0., .1, .2], [.3, 2., .1], [.2, .1, 0.]])
    projector = NativeLineProjector(coeff, 4., (23, 25), kernel, [1., .8], spectral_samples=5, deposition="cubic")
    rng = np.random.default_rng(7)
    shape = (2, 23, 25)
    data, error = rng.normal(size=shape), rng.uniform(.3, 1.5, shape)
    valid = rng.uniform(size=shape) > .1
    valid[0, 0] = False
    likelihood = StripContinuumLikelihood(data, error, valid, ["R", "C"])
    return grid, projector, likelihood


def test_jax_pixels_and_kinematic_gradients_match_torch():
    grid, projector, likelihood = example()
    jgrid, jprojector, jlike = JaxLineMap(grid), JaxNativeProjector(projector), JaxStripLikelihood(likelihood)
    controls = np.array([[.1, .4, .3], [.6, 2., .7], [.1, .8, .2]])
    q = np.array([200., .2, .8, .9])
    np.testing.assert_allclose(jgrid(jnp.asarray(controls)), grid(torch.tensor(controls))[0].numpy(), atol=1e-15)
    def jt(p):
        vel = projected_velocity(jgrid.east, jgrid.north, *p)
        return jprojector(jgrid(jnp.asarray(controls)), vel)
    qtorch = torch.tensor(q, requires_grad=True)
    tline = projector(grid(torch.tensor(controls)), projected_arctan_velocity(grid.east, grid.north, *qtorch), sigma_v=45.)[0]
    np.testing.assert_allclose(jax.jit(jt)(jnp.asarray(q)), tline.detach().numpy(), atol=2e-15)
    score = likelihood.relative(tline)
    tgrad = torch.autograd.grad(score, qtorch)[0].numpy()
    jscore, jgrad = jax.jit(jax.value_and_grad(lambda p: jlike.relative(jt(p))))(jnp.asarray(q))
    np.testing.assert_allclose(jscore, score.detach().numpy(), atol=1e-12)
    np.testing.assert_allclose(jgrad, tgrad, atol=2e-11, rtol=2e-10)
    np.testing.assert_allclose(jlike.chi_square(jt(q)), likelihood.chi_square(tline).detach().numpy(), rtol=1e-13)
    batched = jax.vmap(jt)(jnp.stack([q, q]))
    np.testing.assert_allclose(batched[0], batched[1], atol=1e-15)


def test_log_field_has_proper_variance_and_no_redundant_dc():
    basis, modes, scales = log_field_basis(7, .63, .1)
    np.testing.assert_allclose(modes.T @ modes, np.eye(48), atol=3e-15)
    np.testing.assert_allclose(basis.mean(0), 0., atol=1e-14)
    assert np.mean(np.sum(basis*basis, axis=1)) == pytest.approx(1.)
    assert (scales > 0).all()
    assert np.linalg.matrix_rank(basis) == 48


def test_positive_prior_map_flux_and_initialization():
    grid, projector, likelihood = example()
    posterior = NativePosterior(JaxNativeProjector(projector), JaxLineMap(grid), JaxStripLikelihood(likelihood), half_extent=.3)
    controls = posterior.controls(np.log(1.7), jnp.arange(8)/5, 2., .2)
    assert (np.asarray(controls) > 0).all()
    assert float(posterior.line_map(controls).sum()) == pytest.approx(3.4)
    old = np.array([[0., .4, 0.], [.6, 2., .7], [.1, 0., .2]])
    initial = posterior.initialize_from_fit({"fitted_vref_rturn_theta": [499., 2., 1.1]}, old)
    assert .02 < initial["r_turn"] < 2.
    assert all(np.isfinite(v).all() for v in initial.values())
    density, _ = log_density(posterior.model, (), {}, initial)
    assert np.isfinite(density)
    # Map coefficients actually affect the likelihood and have finite gradients.
    grad = jax.grad(lambda z: posterior.likelihood.relative(posterior.predict(
        210., .22, .9, .9, 0., z, 2., .25)))(jnp.zeros(8))
    assert np.isfinite(grad).all() and np.linalg.norm(grad) > 0


@pytest.mark.parametrize("turnover,inc_error", [("uniform", 5.), ("log_uniform", None)])
def test_prior_predictive_respects_support_and_samples_inclination(turnover, inc_error):
    grid, projector, likelihood = example()
    prior = PosteriorPrior(turnover_prior=turnover, inclination_error_deg=inc_error)
    posterior = NativePosterior(JaxNativeProjector(projector), JaxLineMap(grid), JaxStripLikelihood(likelihood), prior=prior, half_extent=.3)
    draws = Predictive(posterior.model, num_samples=1024)(jax.random.key(21), likelihood_weight=0.)
    assert (np.asarray(draws["v_ref"]) > 0).all() and (np.asarray(draws["v_ref"]) < 1500).all()
    assert (np.asarray(draws["r_turn"]) > .02).all() and (np.asarray(draws["r_turn"]) < 2.).all()
    if inc_error:
        assert np.std(draws["inclination"]) == pytest.approx(np.deg2rad(5.), rel=.08)
    else:
        assert np.mean(draws["cos_i"]) == pytest.approx((np.cos(np.deg2rad(5))+np.cos(np.deg2rad(85)))/2, abs=.03)


def test_profiles_propagate_joint_inclination_and_require_distance_for_mass():
    v = np.array([[210., 210.]])
    rt = np.array([[.22, .22]])
    inc = np.array([[.9, .65]])
    profiles = radial_profiles(v, rt, inc, [.1, .5])
    np.testing.assert_allclose(profiles["projected_velocity_kms"][..., -1], v)
    assert "rotation_only_mass_msun" not in profiles
    expected = (np.sin(.9)/np.sin(.65))**2
    assert profiles["mass_proxy_arcsec_kms2"][0, 1, -1]/profiles["mass_proxy_arcsec_kms2"][0, 0, -1] == pytest.approx(expected)
    mass = radial_profiles(v, rt, inc, [.5], kpc_per_arcsec=8.)["rotation_only_mass_msun"]
    assert mass[0, 0, 0] == pytest.approx(4*(210/np.sin(.9))**2/4.30091727003628e-6)


def test_chain_gate_rejects_disjoint_modes_and_accepts_independent_draws():
    pytest.importorskip("arviz_stats")
    from diagnostics.native_posterior_check import diagnostics
    rng = np.random.default_rng(24)
    sample = rng.normal(size=(4, 1600))
    extras = {"energy": rng.normal(size=(4, 1600)), "diverging": np.zeros((4, 1600), bool),
              "num_steps": np.full((4, 1600), 31), "accept_prob": np.full((4, 1600), .95)}
    good = diagnostics({"v_ref": sample}, extras, 8)
    assert good["convergence_gate_passed"]
    stuck = sample + np.arange(4)[:, None]*5.
    bad = diagnostics({"v_ref": stuck}, extras, 8)
    assert not bad["convergence_gate_passed"]
    assert bad["max_rank_rhat"] > 1.1
    # Python max/min can silently ignore a NaN in a later element. One flat,
    # non-diagnostic parameter must fail the complete posterior's gate.
    with np.errstate(invalid="ignore", divide="ignore"):
        invalid = diagnostics({"v_ref": sample, "flat_parameter": np.ones_like(sample)}, extras, 8)
    assert not invalid["convergence_gate_passed"]
    assert not invalid["all_diagnostics_finite"]


def test_turnover_reweighting_recovers_analytic_log_uniform_quantiles():
    from diagnostics.native_posterior_sensitivity import weighted_quantiles
    lower, upper = .02, 2.
    # An evenly spaced uniform sample becomes log-uniform under weights 1/r.
    r = lower+(np.arange(100000)+.5)*(upper-lower)/100000
    probabilities = np.array([.05, .5, .95])
    expected = lower*(upper/lower)**probabilities
    np.testing.assert_allclose(weighted_quantiles(r, 1/r), expected, rtol=2e-6)


def test_oracle_control_keeps_kinematic_priors_and_flux_scaling():
    from diagnostics.native_oracle_check import KnownMapPosterior
    grid, projector, likelihood = example()
    jm, jp, jl = JaxLineMap(grid), JaxNativeProjector(projector), JaxStripLikelihood(likelihood)
    prior = PosteriorPrior()
    native = NativePosterior(jp, jm, jl, prior=prior, half_extent=.3)
    flux = jm(jnp.ones((3, 3)))
    oracle = KnownMapPosterior(jp, jl, jm.east, jm.north, flux, prior)
    key = jax.random.key(17)
    expected = Predictive(native.model, num_samples=32)(key, likelihood_weight=0.)
    actual = Predictive(oracle.model, num_samples=32)(key, likelihood_weight=0.)
    for name in ("v_ref", "r_turn", "theta", "inclination", "log_flux"):
        np.testing.assert_array_equal(actual[name], expected[name])
    first = oracle.predict(210., .22, .9, .9, 0.)
    second = oracle.predict(210., .22, .9, .9, np.log(2.))
    np.testing.assert_allclose(second, 2*first, atol=1e-14)


def test_emission_prior_reweighting_density_and_map_transport():
    from scipy.stats import lognorm
    from diagnostics.native_map_prior_comparison import lognormal_log_ratio, weight_diagnostics
    values = np.array([[.4, .8, 1.5], [.6, .9, 2.]])
    ratio = lognormal_log_ratio(values, np.log(2.), .5, np.log(4.), 1.)
    expected = lognorm.logpdf(values, 1., scale=4.)-lognorm.logpdf(values, .5, scale=2.)
    np.testing.assert_allclose(ratio, expected, atol=1e-14)
    weights, summary = weight_diagnostics(np.zeros((4, 20)))
    assert summary["kish_weight_ess"] == pytest.approx(80.)
    np.testing.assert_allclose(summary["weight_fraction_by_chain"], .25)
    old, _, oldscale = log_field_basis(7, .63, .1)
    new, _, newscale = log_field_basis(7, .63, .2)
    z = np.random.default_rng(13).normal(size=48)
    np.testing.assert_allclose(new @ (z*oldscale/newscale), old @ z, atol=2e-15)


def test_emission_moments_are_scale_invariant_and_do_not_use_inclination():
    from diagnostics.native_map_prior_comparison import shape_moments
    east = np.array([-2., 2., 0., 0.])
    north = np.array([0., 0., -1., 1.])
    np.testing.assert_allclose(shape_moments(east, north, np.ones(4)), [0., 0., .5, 0.])
    np.testing.assert_allclose(shape_moments(east, north, np.full((3, 4), 7.)),
                               np.tile([0., 0., .5, 0.], (3, 1)))


def test_adapted_step_reduction_preserves_target_covariance_and_sampling_step():
    from numpyro.infer import NUTS
    from diagnostics.native_posterior_check import reduce_adapted_step_size
    potential = lambda z: .5*jnp.sum(z*z)
    kernel = NUTS(potential_fn=potential, max_tree_depth=4)
    state = kernel.init(jax.random.key(91), num_warmup=0, init_params=jnp.array([.2, -.3]))
    scaled = reduce_adapted_step_size(state, .5)
    np.testing.assert_allclose(scaled.adapt_state.step_size, state.adapt_state.step_size*.5)
    # No position, gradient, density, key, covariance or other adapted state changes.
    restored = scaled._replace(adapt_state=scaled.adapt_state._replace(step_size=state.adapt_state.step_size))
    for actual, expected in zip(jax.tree.leaves(restored), jax.tree.leaves(state)):
        if jax.dtypes.issubdtype(actual.dtype, jax.dtypes.prng_key):
            actual, expected = jax.random.key_data(actual), jax.random.key_data(expected)
        np.testing.assert_array_equal(actual, expected)
    advanced = kernel.sample(scaled, (), {})
    np.testing.assert_array_equal(advanced.adapt_state.step_size, scaled.adapt_state.step_size)
    np.testing.assert_allclose(advanced.potential_energy, potential(advanced.z))
    for invalid in (0., -1., 1.1, np.nan):
        with pytest.raises(ValueError):
            reduce_adapted_step_size(state, invalid)
