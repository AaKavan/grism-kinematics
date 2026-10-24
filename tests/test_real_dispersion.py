"""Scientific-coordinate, response-axis and artifact safety checks."""
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize

from diagnostics.native_real_dispersion import pack_free, unpack_free, scaled_kernels, choose_best
from dingo.real_dispersion import DispersionResults
from dingo.real_workflow import sha256, write_json


def test_fixed_dispersion_refit_keeps_sigma_fixed_and_moves_all_other_coordinates():
    original = np.linspace(-.1, .1, 15)
    target = np.linspace(.2, .5, 15)
    def loss(free):
        vector, speed = unpack_free(free, jnp.asarray(40.))
        return jnp.sum((vector-jnp.asarray(target))**2) + ((speed-280.)/100.)**2
    vg = jax.jit(jax.value_and_grad(loss))
    def fun(p):
        v, g = vg(p)
        return float(v), np.asarray(g)
    result = minimize(fun, pack_free(original, 500.), jac=True, method='BFGS', tol=1e-9)
    vector, speed = unpack_free(result.x, jnp.asarray(40.))
    expected = target.copy(); expected[4] = .4
    np.testing.assert_allclose(vector, expected, atol=1e-8)
    assert float(speed) == pytest.approx(280.)


@pytest.mark.parametrize('pupil', ['R', 'C'])
def test_response_changes_affect_only_requested_axis_before_deposition(pupil):
    record = dict(pupil=pupil, lsf_fraction=.6, along_sigma_pixels=[1.1, 2.5], cross_sigma_pixels=1.2)
    y, x = np.mgrid[-10:11, -10:11]
    for spectral, spatial in [(1., 1.), (.9, 1.), (1., 1.1)]:
        k = scaled_kernels([record], spectral, spatial)[0]
        assert k.sum() == pytest.approx(1.)
        moments = np.array([(k*x*x).sum(), (k*y*y).sum()])+1/3
        expected = np.array([(.6*1.1**2+.4*2.5**2)*spectral**2, 1.2**2*spatial**2])
        np.testing.assert_allclose(moments, expected if pupil == 'R' else expected[::-1], atol=1e-10)


def test_failed_lower_score_is_not_hidden_by_selecting_a_worse_stationary_fit():
    a = dict(variant='baseline', fixed_sigma_kms=20., objective=4., stationarity_passed=True)
    b = a | dict(objective=2., stationarity_passed=False)
    assert choose_best([a, b]) == [b]


def test_viewer_checks_source_and_artifact_integrity(tmp_path):
    shape = (2, 3, 3)
    metadata = dict(source_id=3700, exposures=[dict(pupil='R'), dict(pupil='C')])
    np.savez(tmp_path/'native_data.npz', metadata_json=json.dumps(metadata),
             science=np.ones(shape), error=np.ones(shape), valid=np.ones(shape, bool), emline=np.ones(shape))
    np.savez(tmp_path/'fit.npz', source_id=3700, vector=np.zeros(12))
    fit = dict(variant='baseline', fixed_sigma_kms=0., artifact='fit.npz', artifact_sha256=sha256(tmp_path/'fit.npz'))
    report = dict(format='native_real_dispersion_v1', source_id=3700, real_observations=True,
                  native_data_sha256=sha256(tmp_path/'native_data.npz'), best_by_sigma=[fit],
                  settings=dict(variants=['baseline']))
    write_json(tmp_path/'dispersion.json', report)
    with pytest.raises(ValueError, match='different source'):
        DispersionResults(tmp_path, 8910)
    view = DispersionResults(tmp_path, 3700)
    assert view.exposure_pair() == [0, 1]
    with pytest.raises(ValueError, match='Select a R'):
        view.exposure_pair(r_exposure=2)
    with pytest.raises(ValueError, match='not been fitted'):
        view.candidate(10.)
    np.savez(tmp_path/'fit.npz', source_id=3700, vector=np.ones(12))
    with pytest.raises(ValueError, match='artifact changed'):
        view.candidate(0.)
