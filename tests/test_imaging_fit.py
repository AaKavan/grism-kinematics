"""Imaging chain restart, diagnostic gates and PSF invariance."""
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest

from dingo.imaging_fit import (ImagingSampler, boundary_diagnostics,
                               diagnostics, resample_effective_psf, moment_theta,
                               commit_checkpoint, read_checkpoint, verify_or_restore_exports)


def small_model():
    z = numpyro.sample('shape_base', dist.Normal(0., 1.))
    q = numpyro.deterministic('q', jax.nn.sigmoid(z))
    numpyro.sample('observed', dist.Normal(q, .15), obs=jnp.asarray(.7))


def sampler(state=None):
    return ImagingSampler(small_model, dict(shape_base=jnp.array([-.1, .2])),
        warmup=12, seed=129, max_tree_depth=3, target_accept=.8, state=state)


def test_checkpoint_preserves_warmup_schedule_and_exact_draws():
    reference = sampler()
    expected, expected_stats = reference.advance(20)
    assert expected['q'].shape == (2, 20)
    assert np.all((expected['q'] > 0) & (expected['q'] < 1))
    np.testing.assert_array_equal(expected_stats['iteration'][0], np.arange(1, 21))
    np.testing.assert_array_equal(expected_stats['is_warmup'][0], np.arange(1, 21) <= 12)

    partial = sampler()
    draws, _ = partial.advance(5, collect=False)
    assert draws == {}
    saved = pickle.loads(pickle.dumps(jax.device_get(partial.state)))
    continued = sampler(state=saved)
    assert continued.iterations == 5
    continued.advance(7, collect=False)
    step_size = np.asarray(continued.state.adapt_state.step_size).copy()
    actual, stats = continued.advance(8)
    np.testing.assert_array_equal(continued.state.adapt_state.step_size, step_size)
    assert not stats['is_warmup'].any()
    for name in actual:
        np.testing.assert_allclose(actual[name], expected[name][:, 12:], atol=2e-12, rtol=2e-12)
    resumed = sampler(state=pickle.loads(pickle.dumps(jax.device_get(continued.state))))
    next_actual, _ = resumed.advance(3)
    next_expected, _ = reference.advance(3)
    for name in next_actual:
        np.testing.assert_allclose(next_actual[name], next_expected[name], atol=2e-12, rtol=2e-12)


def test_boundary_gate_rejects_radius_pileup_even_when_angle_is_clear():
    healthy = dict(theta=np.zeros((4, 10)), r_eff=np.full((4, 10), 10.))
    assert boundary_diagnostics(healthy, [-np.pi/2, np.pi/2])['orientation_parameterization_ok']
    crowded = healthy | dict(r_eff=np.full((4, 10), 44.98))
    result = boundary_diagnostics(crowded, [-np.pi/2, np.pi/2])
    assert result['radius_boundary_fraction'] == 1
    assert result['angle_boundary_fraction'] == 0
    assert not result['orientation_parameterization_ok']
    assert not boundary_diagnostics(healthy | dict(theta=np.full((4, 10), np.pi/2-.001)),
                                    [-np.pi/2, np.pi/2])['orientation_parameterization_ok']


def test_convergence_never_accepts_short_or_warmup_draws():
    data = dict(q=np.array([[.5, .6], [.4, .5], [.6, .5], [.7, .6]]))
    stats = dict(energy=np.ones((4, 2)), potential_energy=np.ones((4, 2)),
        num_steps=np.ones((4, 2)), accept_prob=np.ones((4, 2)),
        diverging=np.zeros((4, 2), dtype=bool), is_warmup=np.zeros((4, 2), dtype=bool))
    assert not diagnostics(data, stats, 10)['convergence_gate_passed']
    mixed = stats | dict(is_warmup=np.ones((4, 2), dtype=bool))
    assert 'sampling_input_error' in diagnostics(data, mixed, 10)


def test_empirical_psf_matches_historical_sampler_and_normalizes():
    from diagnostics.native_imaging_prior import resample_effective_psf as historical
    yy, xx = np.indices((51, 51))
    psf = np.exp(-((xx-25.2)**2+(yy-24.9)**2)/(2*2.3**2))
    psf[0, 0] = -.001
    expected, _ = historical(psf, .015, .03)
    actual, meta = resample_effective_psf(psf, .015, .03)
    np.testing.assert_array_equal(actual, expected)
    assert actual.sum() == pytest.approx(1.)
    assert np.min(actual) >= 0 and meta['clipped_negative_fraction'] > 0
    with pytest.raises(ValueError, match='positive mass'):
        resample_effective_psf(np.zeros((5, 5)), .015, .03)


def test_sampler_rejects_wrong_chain_shape_and_nonpositive_chunks():
    current = sampler()
    with pytest.raises(ValueError, match='positive integer'):
        current.advance(0)
    state = jax.tree.map(lambda x: x[:1], current.state)
    with pytest.raises(ValueError, match='shapes or dtypes'):
        sampler(state=state)


@pytest.mark.parametrize('theta', [np.pi/4, 3*np.pi/4, .17, 2.1])
def test_moment_chart_recovers_pysersic_tilted_ellipse(theta):
    from pysersic.rendering import render_sersic_2d
    yy, xx = np.indices((101, 101))
    weights = np.asarray(render_sersic_2d(jnp.asarray(xx), jnp.asarray(yy),
        50., 50., 1., 10., .5, .55, theta))
    inferred = moment_theta(weights, xx, yy, (50., 50.))
    difference = (inferred-theta+np.pi/2) % np.pi-np.pi/2
    assert abs(difference) < 1e-6


def test_interrupted_checkpoint_commit_preserves_old_draws_and_repairs_exports(tmp_path, monkeypatch):
    import dingo.imaging_fit as module
    from dingo.real_workflow import read_json
    identity = dict(source_id=15665, settings=dict(warmup=12))
    report = dict(identity=identity, phase='sampling', completed=False)
    first = dict(identity=identity, state=12, samples=dict(q=np.ones((4, 2))*.6),
                 stats=dict(energy=np.ones((4, 2))))
    commit_checkpoint(tmp_path, report, first)
    previous = read_json(tmp_path/'imaging.json')
    second = dict(identity=identity, state=14, samples=dict(q=np.ones((4, 4))*.8),
                  stats=dict(energy=np.ones((4, 4))*2))
    original_write = module.write_json
    def interrupt_before_report(*_):
        raise KeyboardInterrupt('Simulated interruption after new checkpoint and public exports')
    monkeypatch.setattr(module, 'write_json', interrupt_before_report)
    with pytest.raises(KeyboardInterrupt):
        commit_checkpoint(tmp_path, report, second)
    committed = read_json(tmp_path/'imaging.json')
    assert committed == previous
    assert (tmp_path/committed['checkpoint_file']).exists()
    loaded = read_checkpoint(tmp_path, committed, identity)
    assert loaded['state'] == 12
    np.testing.assert_array_equal(loaded['samples']['q'], first['samples']['q'])
    monkeypatch.setattr(module, 'write_json', original_write)
    verify_or_restore_exports(tmp_path, committed, loaded)
    with np.load(tmp_path/'samples.npz') as saved:
        np.testing.assert_array_equal(saved['q'], first['samples']['q'])
    commit_checkpoint(tmp_path, committed, second)
    assert read_checkpoint(tmp_path, read_json(tmp_path/'imaging.json'), identity)['state'] == 14
    assert (tmp_path/previous['checkpoint_file']).exists()
    assert len(list(tmp_path.glob('checkpoint_*.pkl'))) == 2


def test_initialization_only_report_can_resume_but_changed_target_cannot(tmp_path):
    identity = dict(source_id=15665, settings=dict(warmup=12))
    report = dict(identity=identity, phase='initialization', completed=False)
    # Even an uncommitted orphan generation cannot make the worker load it.
    (tmp_path/'checkpoint_orphan.pkl').write_bytes(b'not a committed pickle')
    assert read_checkpoint(tmp_path, report, identity) is None
    with pytest.raises(ValueError, match='changed'):
        read_checkpoint(tmp_path, report, dict(source_id=3700))
    with pytest.raises(ValueError, match='no committed checkpoint'):
        read_checkpoint(tmp_path, report | dict(completed=True), identity)


def test_checkpoint_and_export_corruption_are_detected(tmp_path):
    identity = dict(source_id=15665)
    report = dict(identity=identity, phase='complete', completed=True)
    payload = dict(identity=identity, samples=dict(q=np.full((4, 4), .7)),
                   stats=dict(energy=np.ones((4, 4))))
    commit_checkpoint(tmp_path, report, payload)
    (tmp_path/'samples.npz').write_bytes(b'interrupted public export')
    verify_or_restore_exports(tmp_path, report, read_checkpoint(tmp_path, report, identity))
    with np.load(tmp_path/'samples.npz') as saved:
        np.testing.assert_array_equal(saved['q'], payload['samples']['q'])
    (tmp_path/report['checkpoint_file']).write_bytes(b'damaged checkpoint')
    with pytest.raises(ValueError, match='checksum mismatch'):
        read_checkpoint(tmp_path, report, identity)
