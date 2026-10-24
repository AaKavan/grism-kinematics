"""Incremental adaptation, positive-Gibbs coordinates, and exact restarts."""
import pickle

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest

from dingo.rotation_model import positive_normal
from dingo.rotation_sampling import RotationSampler


class SmallPositiveModel:
    scalar_sites = ('x',)

    def model(self):
        x = numpyro.sample('x', dist.Normal(0., 1.))
        c = numpyro.sample('continuum_coefficients', dist.HalfNormal(1.).expand((1, 1, 3)).to_event(3))
        numpyro.sample('observed', dist.Normal(x+c.ravel(), .4).to_event(1),
                       obs=jnp.array([.2, .5, .8]))

    def gibbs_fn(self, rng_key, gibbs_sites, hmc_sites):
        precision = 1+1/.4**2
        mean = (jnp.array([.2, .5, .8])-hmc_sites['x'])/.4**2/precision
        return {'continuum_coefficients': positive_normal(rng_key, mean, precision**-.5).reshape(1, 1, 3)}


def tiny_sampler(state=None):
    starts = [dict(x=jnp.asarray(x), continuum_coefficients=jnp.full((1, 1, 3), .3))
              for x in (-.3, .4)]
    return RotationSampler(SmallPositiveModel(), chains=2, warmup=12,
        target_accept=.8, max_tree_depth=3, seed=987, initial_values=starts, state=state)


def assert_same_state(left, right):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right)):
        if jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):
            np.testing.assert_array_equal(jax.random.key_data(a), jax.random.key_data(b))
        elif np.issubdtype(a.dtype, np.inexact):
            np.testing.assert_allclose(a, b, rtol=2e-12, atol=2e-12)
        else:
            np.testing.assert_array_equal(a, b)


def test_chunked_warmup_and_checkpoint_resume_match_uninterrupted_transition_sequence():
    reference = tiny_sampler()
    expected, all_stats = reference.advance(20)
    assert reference.iterations == 20
    assert expected['x'].shape == (2, 20)
    assert expected['continuum_coefficients'].shape == (2, 20, 1, 1, 3)
    assert np.all(expected['continuum_coefficients'] > 0.)
    np.testing.assert_array_equal(all_stats['iteration'][0], np.arange(1, 21))
    np.testing.assert_array_equal(all_stats['is_warmup'][0], np.arange(1, 21) <= 12)

    partial = tiny_sampler()
    samples, stats = partial.advance(5, collect=False)
    assert samples == {} and stats['iteration'][0, -1] == 5
    saved = pickle.loads(pickle.dumps(jax.device_get(partial.state)))
    resumed = tiny_sampler(state=saved)
    assert resumed.iterations == 5
    resumed.advance(7, collect=False)
    assert resumed.iterations == 12
    step_sizes = resumed.step_sizes.copy()
    actual, retained_stats = resumed.advance(8)
    assert not retained_stats['is_warmup'].any()
    np.testing.assert_array_equal(resumed.step_sizes, step_sizes)
    for name in actual:
        np.testing.assert_allclose(actual[name], expected[name][:, 12:], rtol=2e-12, atol=2e-12)
    assert_same_state(resumed.state, reference.state)

    # A second restart after warmup must neither reset i nor re-enter adaptation.
    again = tiny_sampler(state=pickle.loads(pickle.dumps(jax.device_get(resumed.state))))
    result, result_stats = again.advance(3)
    continuation, continuation_stats = reference.advance(3)
    for name in result:
        np.testing.assert_allclose(result[name], continuation[name], rtol=2e-12, atol=2e-12)
    assert not result_stats['is_warmup'].any()
    assert_same_state(again.state, reference.state)


def test_actual_rotation_target_runs_batched_gibbs_nuts_without_sigma_latent():
    from test_rotation_model import models
    _, model, p = models()
    starts = []
    for speed in (180., 220.):
        trial = p.copy(); trial[0] = speed/100
        starts.append(model.posterior_initial_values(trial))
    sampler = RotationSampler(model, chains=2, warmup=2, target_accept=.8,
        max_tree_depth=1, seed=52, initial_values=starts)
    sampler.advance(2, collect=False)
    samples, stats = sampler.advance(2)
    assert 'sigma_v' not in samples
    assert samples['vrot'].shape == (2, 2)
    assert samples['continuum_coefficients'].shape[:2] == (2, 2)
    assert np.isfinite(samples['vrot']).all() and np.isfinite(stats['potential_energy']).all()
    assert np.min(samples['continuum_coefficients']) > 0
    assert not stats['is_warmup'].any()


def test_sampler_rejects_invalid_advances_and_mismatched_saved_chain_count():
    sampler = tiny_sampler()
    with pytest.raises(ValueError, match='positive integer'):
        sampler.advance(0)
    bad_state = jax.tree.map(lambda x: x[:1], sampler.state)
    with pytest.raises(ValueError, match='shapes or dtypes'):
        tiny_sampler(state=bad_state)
