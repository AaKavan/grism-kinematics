"""Geometry and probability checks for the restrictive linked-disk control."""
from types import SimpleNamespace

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest

from dingo.native_linked import LinkedExponentialCoupledModel, projected_axis_ratio
from dingo.native_simple import B1, ExponentialCoupledModel, exponential_weights


def example(**kwargs):
    from test_native_coupled import small_model
    old = small_model()
    cm = old.continuum_model
    target = SimpleNamespace(exposures=[SimpleNamespace(science=np.asarray(cm.data[i]),
        error=np.asarray(1/cm.inv[i]), valid=np.asarray(cm.valid[i]), pupil=p)
        for i, p in enumerate(('R', 'C'))])
    linked = LinkedExponentialCoupledModel(old.engine, target, np.asarray(old.response), **kwargs)
    p = old.initial(old.engine.initial(.7, 1.11), 200., flux=2.)
    _, parts = old.components(p)
    return old, linked, linked.from_flexible(p, parts['source_flux']), target, p


def test_linked_geometry_uses_one_disk_and_does_not_mutate_original():
    old, m, p, _, _ = example()
    assert len(p) == len(old.bounds)-old.engine.nmap-2 == len(m.bounds)
    assert m.base_size == 8+2*m.n
    assert old.engine.nmap > 0 and m.engine.nmap == 0
    x, speed, _, _, _ = m.split(p)
    x = x.copy(); x[5:7] = [.27, -.19]
    _, flux, velocity = m.renderer.render(jnp.asarray(x), speed)
    expected = exponential_weights(m.engine.line_map.east, m.engine.line_map.north,
        m.engine.line_map.area, x[7], np.cos(x[2]), x[1], .1*x[5:7])
    np.testing.assert_allclose(flux, expected, atol=1e-15)
    for index, delta in [(1, .2), (2, .1), (5, .3), (6, .2)]:
        trial = x.copy(); trial[index] += delta
        _, changed_light, changed_velocity = m.renderer.render(jnp.asarray(trial), speed)
        assert np.max(np.abs(flux-changed_light)) > 1e-5
        assert np.max(np.abs(velocity-changed_velocity)) > .1
    changed = x.copy(); changed[7] += .2
    _, changed_light, changed_velocity = m.renderer.render(jnp.asarray(changed), speed)
    assert np.max(np.abs(flux-changed_light)) > 1e-5
    np.testing.assert_array_equal(velocity, changed_velocity)
    report = m.describe(np.r_[p[0], x, p[1+m.base_size:]])
    assert report['geometry_linked']
    assert report['emission_center_east_north_arcsec'] == report['center_east_north_arcsec']
    assert report['emission_angle_deg'] == pytest.approx(report['theta_east_to_north_deg'] % 180)
    assert report['emission_axis_ratio'] == pytest.approx(np.cos(x[2]))


def test_transforms_preserve_kinematics_registration_and_nuisance_tail():
    old, m, p, target, original = example()
    original[-4:] = [2.3, .13, -.21, .07]
    original[1+10+old.engine.nmap:1+len(old.engine.bounds)] = [.2, -.1, .3, -.2]
    converted = m.from_flexible(original)
    np.testing.assert_array_equal(converted[:8], original[:8])
    np.testing.assert_array_equal(converted[9:13], [.2, -.1, .3, -.2])
    np.testing.assert_array_equal(converted[-4:], original[-4:])
    independent = ExponentialCoupledModel(old.engine, target, np.asarray(old.response))
    separate = independent.from_flexible(original)
    separate[11:13] = [-2., 1.]  # independent light center must be discarded
    linked = m.from_exponential(separate)
    np.testing.assert_array_equal(linked[:9], separate[:9])
    np.testing.assert_array_equal(linked[9:], separate[13:])
    for invalid in [original[:-1], original[:, None]]:
        with pytest.raises(ValueError):
            m.from_flexible(invalid)
    with pytest.raises(ValueError):
        m.from_exponential(separate[:-1])


def test_saved_map_moments_support_different_quadrature_without_using_its_center():
    _, m, _, _, original = example()
    # Current test renderer has 7x7 nodes, saved map deliberately 5x5.
    axis = np.linspace(-m.engine.half_extent, m.engine.half_extent, 5)
    east, north = np.meshgrid(axis, axis)
    weights = np.exp(-((east-.04)**2/.11**2+(north+.02)**2/.07**2)/2).ravel()
    w = weights/weights.sum()
    xy = np.stack((east.ravel(), north.ravel()), axis=1)
    centered = xy-w@xy
    cov = np.einsum('n,ni,nj->ij', w, centered, centered)
    expected = B1*np.sqrt(np.linalg.eigvalsh(cov)[-1]/3)
    p = m.from_flexible(original, weights)
    assert np.exp(p[8]) == pytest.approx(np.clip(expected, .051, .79))
    np.testing.assert_array_equal(p[6:8], original[6:8])
    for invalid in [np.ones(6), -weights, weights*np.nan, weights[:, None], np.zeros(25)]:
        with pytest.raises(ValueError):
            m.from_flexible(original, invalid)


@pytest.mark.parametrize('thickness', [0., .2])
def test_projected_thickness_changes_light_only(thickness):
    _, m, p, _, _ = example(thickness=thickness)
    i = p[3]
    assert float(projected_axis_ratio(i, thickness)) == pytest.approx(
        np.sqrt(np.cos(i)**2+thickness**2*np.sin(i)**2))
    _, thin, thinp, _, _ = example()
    _, light, vel = m.renderer.render(jnp.asarray(p[1:1+m.base_size]), 200.)
    _, thin_light, thin_vel = thin.renderer.render(jnp.asarray(thinp[1:1+thin.base_size]), 200.)
    np.testing.assert_array_equal(vel, thin_vel)
    if thickness:
        assert np.max(np.abs(light-thin_light)) > 1e-5
    assert m.describe(p)['emission_thickness'] == thickness
    with pytest.raises(ValueError):
        LinkedExponentialCoupledModel(None, None, None, thickness=1.)


@pytest.mark.parametrize('center_kind', ['catalog', 'stellar', 'free'])
def test_shared_center_has_exactly_one_prior(center_kind):
    _, m, p, _, _ = example(center_kind=center_kind)
    x = p[1:1+m.base_size].copy()
    x[5:7] = 0.
    first = float(m.base_penalty(jnp.asarray(x)))
    x[5:7] = [.3, -.2]
    difference = float(m.base_penalty(jnp.asarray(x)))-first
    sd = .05 if center_kind == 'stellar' else .15
    expected = 0. if center_kind == 'free' else .5*np.sum((.1*x[5:7]/sd)**2)
    assert difference == pytest.approx(expected, abs=1e-14)


def test_linked_geometry_and_dispersion_gradients_match_finite_difference():
    _, m, p, _, _ = example(thickness=.2)
    p[5] = .35; p[6:8] = [.217, -.149]
    function = jax.jit(m.objective)
    gradient = np.asarray(jax.jit(jax.grad(m.objective))(jnp.asarray(p)))
    for i in range(1, 13):
        step = np.zeros_like(p); step[i] = 1e-5
        numeric = (float(function(p+step))-float(function(p-step)))/2e-5
        assert gradient[i] == pytest.approx(numeric, rel=2e-4, abs=2e-5)


@pytest.mark.parametrize('center_kind', ['catalog', 'stellar', 'free'])
@pytest.mark.parametrize('mode', ['positive', 'coupled'])
def test_numpyro_density_matches_profiled_objective(center_kind, mode):
    from numpyro.infer.util import log_density
    _, m, p, _, _ = example(center_kind=center_kind, mode=mode)
    offsets = []
    for delta in [0., .13]:
        trial = p.copy()
        trial[0] = 2.+delta; trial[5] = .4+delta
        trial[2] += delta; trial[3] += delta/10
        trial[8] = np.log(.21)+delta; trial[6:8] = [.2+delta, -.1]
        trial[-2:] = [-.2, .1+delta] if mode == 'positive' else [0., 0.]
        objective, parts = m.components(jnp.asarray(trial))
        params = m.posterior_initial_values(trial)
        if mode == 'positive':
            params['continuum_coefficients'] = np.maximum(np.asarray(parts['continuum_coefficients']), 1e-12)
        density, trace = log_density(m.model, (), {}, params)
        assert 'emission_center' not in trace and 'emission_angle' not in trace
        assert 'axis_ratio' not in trace
        offsets.append(float(density+objective))
    assert offsets[0] == pytest.approx(offsets[1], abs=1e-8)
