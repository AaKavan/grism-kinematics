"""Controlled morphology-prior changes with identical physical starting maps.

These scenarios preserve the flexible gas map and its independence from the
stellar light and kinematic geometry. They change one finite-grid Gaussian
field prior setting at a time; none is a morphology-free reconstruction.
"""
import copy

import jax.numpy as jnp
import numpy as np

from .native_posterior import log_field_basis


MORPHOLOGY_SCENARIOS = {
    'baseline': {'map_sigma': 2., 'correlation_arcsec': .10},
    'amplitude1': {'map_sigma': 1., 'correlation_arcsec': .10},
    'amplitude4': {'map_sigma': 4., 'correlation_arcsec': .10},
    'correlation0p05': {'map_sigma': 2., 'correlation_arcsec': .05},
    'correlation0p20': {'map_sigma': 2., 'correlation_arcsec': .20},
}


def _field_description(engine):
    """Validate the saved basis rather than assume it uses the current default.

    The DCT modes are independent of correlation length. Their positive scales
    are recovered from the actual basis and checked against the documented
    normalized (1 + ell**2 * eigenvalue)**-1 spectrum. A rotated, truncated,
    sign-flipped or otherwise incompatible basis must not silently transport.
    """
    size, extent = int(engine.line_map.size), float(engine.half_extent)
    amplitude = float(engine.map_sigma)
    if not np.isfinite([extent, amplitude]).all() or min(extent, amplitude) <= 0:
        raise ValueError('Finite positive map amplitude and source extent required.')
    nmap = size*size-1
    if size < 2 or engine.nmap != nmap:
        raise ValueError('Expected the complete nonconstant source-map basis.')
    if len(engine.bounds) != 10+nmap+2*engine.projector.exposures:
        raise ValueError('Engine nuisance coordinates do not match the flexible map.')
    actual = np.asarray(engine.basis, dtype=float)
    if actual.shape != (size*size, nmap) or not np.isfinite(actual).all():
        raise ValueError('Invalid finite-grid morphology basis.')
    _, modes, _ = log_field_basis(size, extent, .10)
    scales = np.sum(modes*actual, axis=0)
    if np.any(scales <= 0) or not np.allclose(actual, modes*scales, rtol=1e-10, atol=1e-12):
        raise ValueError('Morphology basis is not the expected positive DCT spectrum.')
    delta = 2*extent/(size-1)
    eigen1 = 4*np.sin(np.pi*np.arange(size)/(2*size))**2/delta**2
    eigen = (eigen1[:, None]+eigen1[None, :]).ravel()[1:]
    intercept, slope = np.linalg.lstsq(np.column_stack((np.ones(nmap), eigen)),
                                      1/scales, rcond=None)[0]
    if min(intercept, slope) <= 0:
        raise ValueError('Morphology spectrum has no finite positive correlation length.')
    correlation = float(np.sqrt(slope/intercept))
    expected, _, _ = log_field_basis(size, extent, correlation)
    if not np.allclose(actual, expected, rtol=1e-10, atol=1e-12):
        raise ValueError('Morphology basis normalization or correlation spectrum changed.')
    return modes, scales, dict(map_sigma=amplitude, correlation_arcsec=correlation,
                              controls=size, half_extent_arcsec=extent, nmap=nmap)


def morphology_settings(engine):
    """Describe the validated actual field, including an inferred old correlation."""
    return _field_description(engine)[2]


def morphology_engine(engine, scenario):
    """Copy an engine with one declared field prior; leave the original untouched.

    The source grid, elliptical envelope, velocity field, inclination prior,
    response, likelihood and registration are all retained. The width envelope
    penalty is unchanged. Larger map_sigma weakens the coefficient penalty at
    a fixed physical brightness field; correlation changes redistribute it
    across spatial frequencies and are not uniformly weaker/stronger priors.
    """
    if scenario not in MORPHOLOGY_SCENARIOS:
        raise ValueError(f'Unknown morphology scenario: {scenario!r}')
    settings = morphology_settings(engine)
    selected = MORPHOLOGY_SCENARIOS[scenario]
    result = copy.copy(engine)
    result.bounds = list(engine.bounds)
    result.map_sigma = selected['map_sigma']
    result.basis = jnp.asarray(log_field_basis(settings['controls'], settings['half_extent_arcsec'],
                                             selected['correlation_arcsec'])[0])
    result.map_correlation_arcsec = selected['correlation_arcsec']
    result.morphology_scenario = scenario
    return result


def transport_parameters(parameters, original_engine, new_engine):
    """Transport a full CoupledNativeModel vector without changing its sky map.

    With B = U diag(s), set z_new = A_old*s_old/(A_new*s_new) * z_old.
    This changes only the map coordinates, which begin at full-vector index
    11. Flux, envelope, velocity, dispersion, centers and exposure nuisance
    coordinates remain bitwise unchanged. The result is an initialization,
    not a fitted solution under the new prior.
    """
    _, old_scales, old = _field_description(original_engine)
    _, new_scales, new = _field_description(new_engine)
    if any(old[k] != new[k] for k in ('controls', 'half_extent_arcsec', 'nmap')):
        raise ValueError('Map transport requires the same finite source control grid.')
    if original_engine.projector.exposures != new_engine.projector.exposures:
        raise ValueError('Map transport cannot change the number of exposures.')
    for name in ('east', 'north', 'area'):
        if not np.array_equal(np.asarray(getattr(original_engine.line_map, name)),
                              np.asarray(getattr(new_engine.line_map, name))):
            raise ValueError('Map transport requires identical quadrature and source support.')
    for name in ('east_controls', 'north_controls'):
        if not np.array_equal(np.asarray(getattr(original_engine, name)),
                              np.asarray(getattr(new_engine, name))):
            raise ValueError('Map transport cannot change the envelope coordinates.')
    original = np.asarray(parameters, dtype=float)
    expected = 1+len(original_engine.bounds)+2*original_engine.projector.exposures
    if original.shape != (expected,) or not np.isfinite(original).all():
        raise ValueError('Expected a finite full coupled-model parameter vector.')
    result = original.copy()
    section = slice(11, 11+old['nmap'])
    result[section] *= old['map_sigma']*old_scales/(new['map_sigma']*new_scales)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite transported map coefficients.')
    return result
