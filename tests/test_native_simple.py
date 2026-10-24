"""Independent light-profile, geometry, and posterior-density checks."""
from types import SimpleNamespace

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest

from dingo.native_simple import ExponentialCoupledModel, exponential_weights


def example():
    from test_native_coupled import small_model
    old = small_model()
    cm = old.continuum_model
    target = SimpleNamespace(exposures=[SimpleNamespace(science=np.asarray(cm.data[i]),
        error=np.asarray(1/cm.inv[i]), valid=np.asarray(cm.valid[i]), pupil=p)
        for i, p in enumerate(('R', 'C'))])
    simple = ExponentialCoupledModel(old.engine, target, np.asarray(old.response))
    p = old.initial(old.engine.initial(.7, 1.11), 200., flux=2.)
    _, parts = old.components(p)
    return old, simple, simple.from_flexible(p, parts['source_flux'])


def test_exponential_matches_original_dingo_sersic_family():
    from dingo.galaxy import sersic_model
    east, north = np.meshgrid(np.linspace(-.9, .9, 61), np.linspace(-.9, .9, 61))
    area = np.ones(east.shape); area[[0, -1], :] *= .5; area[:, [0, -1]] *= .5
    center = [.0317, -.0213]
    expected = sersic_model(east, north, 1., .23, 1., *center, q=.43, theta=.8)*area
    expected /= expected.sum()
    actual = exponential_weights(east, north, area, np.log(.23), .43, .8, jnp.array(center))
    np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=1e-11)
    assert float(actual.sum()) == pytest.approx(1., abs=1e-14)


def test_light_and_rotation_centers_are_independent_and_map_coordinates_removed():
    old, m, p = example()
    assert len(p) == len(old.bounds)-old.engine.nmap+2 == len(m.bounds)
    assert m.engine.nmap == 0
    x, speed, _, _, _ = m.split(p)
    _, f0, v0 = m.renderer.render(jnp.asarray(x), speed)
    light = x.copy(); light[10:12] += [.3, -.1]
    _, f1, v1 = m.renderer.render(jnp.asarray(light), speed)
    np.testing.assert_array_equal(v0, v1)
    assert np.max(np.abs(f0-f1)) > 1e-4
    kin = x.copy(); kin[5:7] += [.3, -.1]
    _, f2, v2 = m.renderer.render(jnp.asarray(kin), speed)
    np.testing.assert_array_equal(f0, f2)
    assert np.max(np.abs(v0-v2)) > 1.
    assert old.engine.nmap > 0  # no mutation of archived model geometry


def test_new_shape_and_dispersion_gradients_match_finite_differences():
    _, m, p = example()
    p[5] = .35; p[11:13] = [.217, -.149]
    f = jax.jit(m.objective)
    g = np.asarray(jax.jit(jax.grad(m.objective))(jnp.asarray(p)))
    for i in [5, 6, 7, 8, 9, 10, 11, 12]:
        step = np.zeros_like(p); step[i] = 1e-5
        numeric = (float(f(p+step))-float(f(p-step)))/2e-5
        assert g[i] == pytest.approx(numeric, rel=2e-4, abs=2e-5)


def test_numpyro_density_matches_profiled_penalties_at_continuum_optimum():
    from numpyro.infer.util import log_density
    _, m, p = example()
    offsets = []
    for delta in [0., .13]:
        trial = p.copy()
        trial[0] = 2.+delta; trial[5] = .4+delta
        trial[8] = np.log(.21)+delta; trial[11:13] = [.2+delta, -.1]
        trial[-2:] = [-.2, .1+delta]
        objective, parts = m.components(jnp.asarray(trial))
        params = m.posterior_initial_values(trial)
        params['continuum_coefficients'] = np.maximum(np.asarray(parts['continuum_coefficients']), 1e-12)
        density, _ = log_density(m.model, (), {}, params)
        offsets.append(float(density+objective))
    assert offsets[0] == pytest.approx(offsets[1], abs=1e-8)


def test_zero_sigma_first_derivative_does_not_certify_a_minimum():
    # A mathematical counterexample using the real operator, not a galaxy
    # recovery experiment. Fixed-width scans must bypass this boundary trap.
    from test_native_forward import make_projector
    import torch
    op = make_projector(nodes=1, spectral_samples=9, deposition='cubic')
    target = op([1.], [0.], sigma_v=50.).detach()
    def objective(scaled_sigma):
        prediction = op([1.], [0.], sigma_v=100*scaled_sigma)
        return .5*torch.sum(((prediction-target)/.01)**2)+.5*scaled_sigma**2
    zero = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    gradient = torch.autograd.grad(objective(zero), zero, create_graph=True)[0]
    curvature = torch.autograd.grad(gradient, zero)[0]
    assert abs(float(gradient)) < 1e-12
    assert float(curvature) < 0
    assert float(objective(torch.tensor(.1))) < float(objective(zero))
