"""Held-out line leakage and an independent constrained-fit reference."""
import numpy as np
from scipy.optimize import lsq_linear

from dingo.native_holdout_continuum import fit_sideband_continuum


def example():
    rng = np.random.default_rng(21636)
    data = rng.normal(.15, .2, (2, 9, 9))
    error = rng.uniform(.1, .3, data.shape)
    valid = rng.uniform(size=data.shape) > .1
    return data, error, valid, ['R', 'C'], np.array([[4.2, 3.8], [4.2, 3.8]])


def test_excluded_line_science_has_no_effect_even_when_nonfinite():
    data, error, valid, pupils, centers = example()
    baseline = fit_sideband_continuum(data, error, valid, pupils, centers, guard=2.5)
    changed = data.copy()
    changed[~baseline['sideband_mask']] = np.nan
    repeated = fit_sideband_continuum(changed, error, valid, pupils, centers, guard=2.5)
    for key in ('prediction', 'prediction_variance', 'coefficients', 'background'):
        np.testing.assert_array_equal(baseline[key], repeated[key])
    assert baseline['metadata'] == repeated['metadata']


def test_sideband_fit_and_active_covariance_match_independent_dense_linear_algebra():
    data, error, valid, pupils, centers = example()
    result = fit_sideband_continuum(data, error, valid, pupils, centers, guard=2.5,
                                   coefficient_scale=.7, background_scale=.8)
    size = data.shape[-1]
    t = np.linspace(0., 1., size)
    basis = np.stack(((1-t)**2, 2*t*(1-t), t*t), axis=-1)
    for e, pupil in enumerate(pupils):
        # Build a complete independent joint least-squares system, including
        # the signed background and all coefficient priors.
        values = data[e].T if pupil == 'C' else data[e]
        errors = error[e].T if pupil == 'C' else error[e]
        mask = result['sideband_mask'][e].T if pupil == 'C' else result['sideband_mask'][e]
        design = np.zeros((size*size, 3*size+1))
        for strip in range(size):
            design[strip*size:(strip+1)*size, 3*strip:3*strip+3] = basis
        design[:, -1] = result['background_sigma'][e]
        weights = np.where(mask, 1/errors, 0.).ravel()
        prior = np.diag(np.r_[np.full(3*size, 1/.7), 1.])
        a = np.vstack((design*weights[:, None], prior))
        y = np.r_[values.ravel()*weights, np.zeros(3*size+1)]
        opt = lsq_linear(a, y, bounds=(np.r_[np.zeros(3*size), -np.inf], np.inf),
                         tol=1.e-13, max_iter=1000)
        actual = np.r_[result['coefficients'][e].ravel(), result['background_z'][e]]
        np.testing.assert_allclose(actual, opt.x, atol=2.e-7)
        active = np.r_[actual[:-1] > 0., True]
        covariance = np.linalg.inv(a[:, active].T @ a[:, active])
        prediction_var = np.einsum('ik,kl,il->i', design[:, active], covariance, design[:, active])
        expected = prediction_var.reshape(size, size)
        if pupil == 'C':
            expected = expected.T
        np.testing.assert_allclose(result['prediction_variance'][e], expected, rtol=1.e-10, atol=1.e-12)
        assert result['metadata'][e]['converged']


def test_rotating_r_to_c_preserves_physical_prediction():
    data, error, valid, _, centers = example()
    r = fit_sideband_continuum(data[:1], error[:1], valid[:1], ['R'], centers[:1], guard=2.5)
    c = fit_sideband_continuum(data[:1].swapaxes(-1, -2), error[:1].swapaxes(-1, -2),
                              valid[:1].swapaxes(-1, -2), ['C'], centers[:1, ::-1], guard=2.5)
    np.testing.assert_array_equal(r['prediction'], c['prediction'].swapaxes(-1, -2))
    np.testing.assert_array_equal(r['sideband_mask'], c['sideband_mask'].swapaxes(-1, -2))
