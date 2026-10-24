"""Posterior probability/gradient checks, independent of notebook cosmetics."""
import numpy as np
import pytest
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from scipy.integrate import quad
from scipy.special import log_ndtr

from dingo.native_real_posterior import integrated_positive_amplitudes, RealNativePosterior


@pytest.mark.parametrize('data_value', [-8., -1., 0., 3., 12.])
def test_positive_amplitude_integral_matches_direct_quadrature(data_value):
    t = np.array([[[.3, 1.2, -.2]]]); d = t*data_value
    scale = 4.
    b = float((t*d).sum()); precision = float((t*t).sum()+1/scale**2)
    mode = max(0., b/precision)
    offset = -.5*precision*mode**2+b*mode
    integral = quad(lambda a: np.sqrt(2/np.pi)/scale*np.exp(-.5*precision*a*a+b*a-offset), 0, np.inf)[0]
    expected = np.log(integral)+offset
    actual, loc, sd = integrated_positive_amplitudes(jnp.array(t), jnp.array(d), scale)
    assert float(actual) == pytest.approx(expected, abs=2e-10)
    assert float(loc[0]) == pytest.approx(b/precision)
    assert float(sd[0]) == pytest.approx(precision**-.5)


def test_zero_information_returns_normalized_prior_integral_and_finite_gradient():
    template = jnp.zeros((2, 3, 4)); data = jnp.ones_like(template)
    value = integrated_positive_amplitudes(template, data)[0]
    assert float(value) == pytest.approx(0., abs=1e-12)
    gradient = jax.grad(lambda t: integrated_positive_amplitudes(t, data)[0])(template)
    assert np.isfinite(gradient).all()
    # Scaling the template changes an explicitly fixed amplitude prior. The
    # marginal likelihood must not have profiling's scale invariance.
    t = jnp.ones((1, 2, 2))
    a = integrated_positive_amplitudes(t, t, 2.)[0]
    b = integrated_positive_amplitudes(3*t, t, 2.)[0]
    assert abs(float(a-b)) > .1


def example():
    import torch
    from dingo.native_fitting import BilinearLineMap, StripContinuumLikelihood
    from dingo.native_forward import NativeLineProjector
    from dingo.native_jax import JaxLineMap, JaxNativeProjector, JaxStripLikelihood
    from dingo.native_real import RealVelocityProfile, broadened_inclination_density
    grid = BilinearLineMap(3, 7, half_extent=.3, dtype=torch.float64)
    coeff = np.zeros((2, 49, 2, 7))
    coeff[:, :, 0, 0] = 10.3+grid.east.numpy()/.06
    coeff[:, :, 1, 0] = 10.7+grid.north.numpy()/.06
    coeff[0, :, 0, 1] = 1000.; coeff[1, :, 1, 1] = -900.
    op = NativeLineProjector(coeff, 4., (23, 23), np.ones((3, 3)), [1., 1.], deposition='cubic')
    y, x = np.mgrid[:23, :23]
    data = np.stack([10*np.exp(-((x-11)**2+(y-10)**2)/8)]*2)
    likelihood = StripContinuumLikelihood(data, np.ones_like(data), np.ones_like(data, dtype=bool), ['R', 'C'])
    ig, ip = broadened_inclination_density([1.1], 10.)
    return RealVelocityProfile(JaxNativeProjector(op), JaxLineMap(grid), JaxStripLikelihood(likelihood),
                               ig, ip, half_extent=.3)


def test_real_posterior_preserves_forward_model_and_has_correct_gradients():
    jax.clear_caches()
    engine = example(); posterior = RealNativePosterior(engine)
    x = engine.initial(.7, 1.11123); x[3] = .2; x[5] = .12; x[10] = .3
    _, old = engine.components(x, 200.)
    line, flux, velocity = posterior.render(x, 200.)
    np.testing.assert_allclose(line*old[3][:, None, None], old[4], atol=1e-13)
    np.testing.assert_allclose(flux, old[5], atol=1e-14)
    np.testing.assert_allclose(velocity, old[6], atol=1e-12)
    score = jax.jit(posterior.log_likelihood)
    grad = np.asarray(jax.jit(jax.grad(posterior.log_likelihood))(x, 200.))
    for i in (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, len(x)-1):
        step = np.zeros_like(x); step[i] = 1e-5
        numeric = float(score(x+step, 200.)-score(x-step, 200.))/(2e-5)
        assert grad[i] == pytest.approx(numeric, abs=2e-5, rel=2e-4)
    # All inferred map and registration coefficients enter the target.
    assert np.linalg.norm(grad[10:10+engine.nmap]) > 0
    assert np.linalg.norm(grad[10+engine.nmap:]) > 0


def test_posterior_prior_coordinates_match_grid_penalties_up_to_constants():
    from numpyro.infer.util import log_density
    import numpyro.handlers as handlers
    jax.clear_caches()
    engine = example(); posterior = RealNativePosterior(engine)
    x = engine.initial(.7, 1.11123)
    values = posterior.initial_values(x, 200.)
    # For two states, remove their different (integrated vs profile) data terms.
    offsets = []
    for delta in (0., .12):
        p = {k: np.array(v, copy=True) for k, v in values.items()}
        p['log_width'] += delta; p['center'][0] += delta/3
        p['map_z'][0] += delta; p['inclination'] += delta/10
        vector = posterior.vector(p)
        logp, _ = log_density(posterior.model, (), {}, p)
        _, parts = engine.components(vector, p['vrot'])
        log_prior = float(logp-posterior.log_likelihood(vector, p['vrot']))
        offsets.append(log_prior+float(parts[1]+parts[2]))
    assert offsets[0] == pytest.approx(offsets[1], abs=1e-9)


def test_real_start_selection_requires_distinct_stationary_branches():
    from diagnostics.native_real_posterior import starting_branches
    base = dict(stationarity_passed=True, center_east_north_arcsec=[0., 0.])
    a = base | dict(vrot_0p5arcsec_kms=280., objective=-100.)
    b = base | dict(vrot_0p5arcsec_kms=480., objective=-90., center_east_north_arcsec=[0., -.29])
    c = b | dict(objective=-200., stationarity_passed=False)
    assert starting_branches({'fits': [a, b, c]}) == (a, b)
    with pytest.raises(ValueError, match='second'):
        starting_branches({'fits': [a]})
