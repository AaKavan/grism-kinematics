"""CPU sideband calibration for frozen held-out line predictions.

Only the sideband science pixels enter this fit. The excluded central pixels,
and any proposed emission-line model, are never used. This is a conditional
prediction given the held-out exposure's sidebands, not a prediction of its
continuum from other exposures. Positive continuum coefficients have the same
HalfNormal prior used by native_coupled.PositiveStripContinuum; the signed
background has a Normal prior with scale equal to the median valid ERR.
"""
import numpy as np
from scipy.optimize import root_scalar


def _fit_exposure(data, error, mask, background_sigma, coefficient_scale):
    """Rows are cross-dispersion strips; columns follow dispersion."""
    n = data.shape[0]
    t = np.linspace(0., 1., data.shape[1])
    basis = np.stack(((1-t)**2, 2*t*(1-t), t*t), axis=-1)
    # Discard excluded values before doing arithmetic: even NaN/inf in the line
    # window cannot contaminate the fitted sideband continuum.
    observed = np.where(mask, data, 0.)
    weights = np.divide(1., error**2, out=np.zeros_like(error), where=mask)
    gram = np.einsum('sd,dk,dl->skl', weights, basis, basis)
    gram += np.eye(3) / coefficient_scale**2
    rhs0 = np.einsum('sd,sd,dk->sk', observed, weights, basis)
    cross = background_sigma*np.einsum('sd,dk->sk', weights, basis)
    inverses = np.zeros((8, n, 3, 3))
    for active in range(1, 8):
        keep = np.flatnonzero([(active >> k) & 1 for k in range(3)])
        inverse = np.linalg.inv(gram[:, keep[:, None], keep])
        for j, k in enumerate(keep):
            for l, m in enumerate(keep):
                inverses[active, :, k, m] = inverse[:, j, l]

    def profile(background_z):
        rhs = rhs0-background_z*cross
        candidates = np.einsum('askl,sl->ask', inverses, rhs)
        feasible = np.all(candidates >= 0., axis=-1)
        scores = (.5*np.einsum('ask,skl,asl->as', candidates, gram, candidates)
                  - np.einsum('ask,sk->as', candidates, rhs))
        active = np.argmin(np.where(feasible, scores, np.inf), axis=0)
        coefficients = candidates[active, np.arange(n)]
        continuum = coefficients @ basis.T
        residual = observed-continuum-background_sigma*background_z
        gradient = background_z-background_sigma*np.sum(weights*residual)
        return float(gradient), coefficients, continuum, active, residual

    left, right = -1., 1.
    for _ in range(128):
        if profile(left)[0] <= 0. <= profile(right)[0]:
            break
        left *= 2.
        right *= 2.
    else:
        raise RuntimeError('Could not bracket the convex sideband background optimum.')
    root = root_scalar(lambda z: profile(z)[0], bracket=(left, right),
                       method='brentq', xtol=1.e-12, rtol=1.e-14)
    if not root.converged:
        raise RuntimeError('Sideband background optimization did not converge.')
    z = float(root.root)
    gradient, coefficients, continuum, active, residual = profile(z)
    chosen_inverse = inverses[active, np.arange(n)]
    cross_response = np.einsum('skl,sl->sk', chosen_inverse, cross)
    precision_z = (1.+background_sigma**2*np.sum(weights)
                   - np.sum(cross*cross_response))
    if not np.isfinite(precision_z) or precision_z <= 0.:
        raise RuntimeError('Nonpositive conditional background precision.')
    variance_z = 1./precision_z
    variance = np.einsum('dk,skl,dl->sd', basis, chosen_inverse, basis)
    variance += (background_sigma-cross_response @ basis.T)**2*variance_z
    chi_square = float(np.sum(weights*residual**2))
    penalty = .5*(float(np.sum((coefficients/coefficient_scale)**2))+z*z)
    return dict(continuum=continuum, coefficients=coefficients,
                background=background_sigma*z, background_z=z,
                prediction=continuum+background_sigma*z,
                prediction_variance=np.maximum(variance, 0.),
                metadata=dict(converged=True, iterations=int(root.iterations),
                    function_calls=int(root.function_calls), sideband_pixels=int(mask.sum()),
                    objective=.5*chi_square+penalty, chi_square=chi_square, penalty=penalty,
                    background_gradient=gradient,
                    positive_coefficients=int(np.count_nonzero(coefficients)),
                    standardized_background_variance=float(variance_z)))


def fit_sideband_continuum(science, error, valid, pupils, line_centers_xy, *,
                          guard=28., coefficient_scale=1., background_scale=1.):
    """Predict positive continuum plus signed background using sidebands only.

    Inputs have shape [exposure, y, x]. ``line_centers_xy`` are nominal line
    positions in these cutouts, in zero-based (x,y) pixels. For R, sidebands
    satisfy abs(x-center_x) >= guard; for C they satisfy abs(y-center_y) >= guard.
    The mask uses calibration geometry only and is identical for all candidate
    lines. Returned prediction variance conditions on the selected active
    coefficient set, ignores positivity-boundary uncertainty, and includes
    coefficient/background covariance. It is not a full predictive posterior.
    """
    science = np.asarray(science, dtype=float)
    error = np.asarray(error, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    pupils = np.asarray(pupils)
    centers = np.asarray(line_centers_xy, dtype=float)
    if (science.ndim != 3 or science.shape != error.shape or science.shape != valid.shape
            or science.shape[1] != science.shape[2]):
        raise ValueError('Matching square [exposure,y,x] arrays required.')
    nexp, size, _ = science.shape
    if (pupils.shape != (nexp,) or set(pupils)-{'R', 'C'}
            or centers.shape != (nexp, 2) or not np.isfinite(centers).all()):
        raise ValueError('One R/C pupil and finite nominal (x,y) center per exposure required.')
    if (not np.isfinite([guard, coefficient_scale, background_scale]).all()
            or min(guard, coefficient_scale, background_scale) <= 0.):
        raise ValueError('Positive finite guard and prior scales required.')
    if np.any(valid & (~np.isfinite(error) | (error <= 0.))):
        raise ValueError('Valid pixels must have finite positive ERR.')
    if np.any(np.sum(valid, axis=(1, 2)) == 0):
        raise ValueError('Every exposure requires some valid pixels.')

    yy, xx = np.indices((size, size))
    coordinate = np.where((pupils == 'R')[:, None, None], xx, yy)
    origin = np.where(pupils == 'R', centers[:, 0], centers[:, 1])
    sideband = valid & (np.abs(coordinate-origin[:, None, None]) >= guard)
    if np.any(np.sum(sideband, axis=(1, 2)) == 0):
        raise ValueError('Every exposure requires valid sideband pixels outside the guard.')
    if np.any(sideband & ~np.isfinite(science)):
        raise ValueError('Sideband science pixels must be finite.')
    scales = background_scale*np.asarray([np.median(e[v]) for e, v in zip(error, valid)])
    results = []
    for e, pupil in enumerate(pupils):
        arrays = [science[e], error[e], sideband[e]]
        if pupil == 'C':
            arrays = [a.T for a in arrays]
        result = _fit_exposure(*arrays, scales[e], coefficient_scale)
        if pupil == 'C':
            for key in ('prediction', 'prediction_variance', 'continuum'):
                result[key] = result[key].T
        results.append(result)
    output = {key: np.stack([r[key] for r in results]) for key in
              ('prediction', 'prediction_variance', 'continuum', 'coefficients',
               'background', 'background_z')}
    output.update(sideband_mask=sideband, background_sigma=scales,
        metadata=[r['metadata'] for r in results],
        variance_interpretation='Conditional active-set Gaussian curvature; includes continuum/background '
            'covariance, excludes active-set and positivity-boundary uncertainty.',
        calibration_policy='Only held-out sideband science enters; central line pixels and line '
            'candidates are excluded. This is a sideband-conditioned prediction.')
    return output
