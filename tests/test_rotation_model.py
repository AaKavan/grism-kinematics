"""No-width forward equivalence and the probability target of Gibbs updates."""
from types import SimpleNamespace

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import truncnorm

from dingo.rotation_model import RotationOnlyModel, positive_normal, continuum_gibbs


def models():
    from test_native_coupled import small_model
    old = small_model()
    q, w = np.polynomial.hermite.hermgauss(17)
    old.engine.projector.quadrature = jnp.asarray(q*np.sqrt(2.))
    old.engine.projector.quadrature_weights = jnp.asarray(w/np.sqrt(np.pi))
    cm = old.continuum_model
    target = SimpleNamespace(exposures=[SimpleNamespace(science=np.asarray(cm.data[i]),
        error=np.asarray(1/cm.inv[i]), valid=np.asarray(cm.valid[i]), pupil=p)
        for i, p in enumerate(('R', 'C'))])
    new = RotationOnlyModel(old.engine, target, np.asarray(old.response))
    p = new.initial(old.engine.initial(.7, 1.11), 200., flux=2.)
    return old, new, p


def test_one_node_rotation_preserves_line_objective_and_gradients_without_mutation():
    old, new, p = models()
    p[3] += .03; p[6] = .13; p[11] += .2
    assert len(old.engine.projector.quadrature) == 17
    assert len(new.engine.projector.quadrature) == 1
    assert new.bounds[5] == (0., 0.)
    old_value, old_parts = old.components(jnp.asarray(p))
    new_value, new_parts = new.components(jnp.asarray(p))
    np.testing.assert_allclose(new_parts['line'], old_parts['line'], atol=2e-14)
    np.testing.assert_allclose(new_value, old_value, atol=1e-12)
    old_grad = jax.grad(old.objective)(jnp.asarray(p))
    new_grad = jax.grad(new.objective)(jnp.asarray(p))
    np.testing.assert_allclose(new_grad, old_grad, atol=1e-11)


def test_sigma_is_not_a_latent_and_density_matches_positive_continuum_target():
    from numpyro.infer.util import log_density, initialize_model
    from numpyro.infer.initialization import init_to_value
    old, new, p = models()
    differences = []
    for delta in (0., .12):
        trial = p.copy(); trial[0] += delta; trial[6] += delta; trial[-1] += delta
        values = new.posterior_initial_values(trial)
        assert 'sigma_v' not in values
        density, trace = log_density(new.model, (), {}, values)
        old_density, _ = log_density(old.model, (), {}, dict(values, sigma_v=0.))
        differences.append(float(density-old_density))
        assert trace['sigma_v']['is_observed']
        np.testing.assert_allclose(new.line_from_sites(values), new.components(trial)[1]['line'], atol=1e-13)
    np.testing.assert_allclose(differences, 0., atol=1e-11)
    info = initialize_model(jax.random.key(23), new.model,
                           init_strategy=init_to_value(values=values))
    assert 'sigma_v' not in info.param_info.z
    assert np.isfinite(info.param_info.potential_energy)
    p[5] = .01
    with pytest.raises(ValueError, match='exactly zero'):
        new.posterior_initial_values(p)


@pytest.mark.parametrize('cutoff', [-10., 0., 5., 20., 40.])
def test_positive_normal_moments_even_far_into_tail(cutoff):
    n = 80000
    draws = np.asarray(jax.jit(positive_normal)(jax.random.key(42),
        jnp.full(n, -cutoff), jnp.ones(n)))
    mean, variance = truncnorm.stats(cutoff, np.inf, loc=-cutoff, moments='mv')
    assert np.isfinite(draws).all() and np.min(draws) >= 0.
    assert draws.mean() == pytest.approx(mean, abs=5*np.sqrt(variance/n))
    assert draws.var() == pytest.approx(variance, rel=.04)


def test_correlated_continuum_sweeps_preserve_known_joint_truncated_density():
    # Independent rejection draws from a correlated Gaussian give a reference
    # joint truncated distribution, not a reimplementation of coordinate Gibbs.
    covariance = np.array([[1., .35, -.15], [.35, .8, .2], [-.15, .2, .7]])
    location = np.array([.1, -.2, .3]); gram = np.linalg.inv(covariance)
    rng = np.random.default_rng(98)
    proposals = rng.multivariate_normal(location, covariance, size=700000)
    reference = proposals[(proposals > 0.).all(axis=1)][:80000]
    rhs = gram@location
    draws = jnp.ones((40000, 3))*.3
    def transition(index, current):
        return continuum_gibbs(jax.random.fold_in(jax.random.key(8), index),
            current, jnp.asarray(gram), jnp.asarray(rhs), sweeps=4)
    # Many independent chains, enough sweeps to erase the deliberately wrong
    # starting distribution before comparing moments and cross-correlations.
    draws = np.asarray(jax.jit(lambda x: jax.lax.fori_loop(0, 20, transition, x))(draws))
    np.testing.assert_allclose(draws.mean(0), reference.mean(0), atol=.014)
    np.testing.assert_allclose(np.cov(draws.T), np.cov(reference.T), atol=.015)


def test_model_gibbs_uses_current_line_background_and_full_continuum_precision():
    _, model, p = models()
    sites = model.posterior_initial_values(p)
    coefficients = sites.pop('continuum_coefficients')
    cm = model.continuum_model
    line = model.line_from_sites(sites)
    residual = np.asarray(cm.strips(cm.data-line))
    rhs = np.einsum('esd,esd,dk->esk', residual, np.asarray(cm.strips(cm.inv))**2, np.asarray(cm.basis))
    key = jax.random.key(17)
    expected = continuum_gibbs(key, jnp.asarray(coefficients), cm.gram, jnp.asarray(rhs), 4)
    actual = model.gibbs_fn(key, {'continuum_coefficients': jnp.asarray(coefficients)}, sites)
    np.testing.assert_allclose(actual['continuum_coefficients'], expected, atol=1e-12)
