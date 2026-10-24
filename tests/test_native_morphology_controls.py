"""Map-prior transport must preserve physical predictions, not just coefficients."""
import copy

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest

from dingo.native_morphology_controls import (
    MORPHOLOGY_SCENARIOS, morphology_engine, morphology_settings, transport_parameters)
from dingo.native_real_posterior import RealNativePosterior


def comparison_example():
    from test_native_coupled import small_model
    model = small_model()
    engine = model.engine
    vector = engine.initial(.7, 1.11)
    vector[3:7] = [.12, .35, .21, -.13]
    vector[7:10] = [np.log(.29), .43, -.38]
    vector[10:] = np.random.default_rng(63).normal(0., .4, len(vector)-10)
    parameters = model.initial(vector, 217., flux=2.3)
    parameters[-3:] = [.2, .1, -.1]
    return model, parameters


@pytest.mark.parametrize('scenario', list(MORPHOLOGY_SCENARIOS))
def test_matched_start_preserves_actual_rc_images_and_only_changes_map_penalty(scenario):
    original, parameters = comparison_example()
    old_basis = np.asarray(original.engine.basis).copy()
    changed_engine = morphology_engine(original.engine, scenario)
    changed = copy.copy(original)
    changed.engine = changed_engine
    changed.renderer = RealNativePosterior(changed_engine)
    transported = transport_parameters(parameters, original.engine, changed_engine)
    old_objective, old_parts = original.components(jnp.asarray(parameters))
    new_objective, new_parts = changed.components(jnp.asarray(transported))
    # Test the full calibrated detector image and profiled positive continuum,
    # as well as the normalized source weights, under both R and C projections.
    for key in ('source_flux', 'source_velocity', 'line', 'continuum', 'background', 'amplitudes'):
        np.testing.assert_allclose(new_parts[key], old_parts[key], rtol=2e-12, atol=2e-13)
    section = slice(11, 11+original.engine.nmap)
    expected_delta = .5*(transported[section]@transported[section]-parameters[section]@parameters[section])
    assert float(new_objective-old_objective) == pytest.approx(expected_delta, abs=1e-11)
    mask = np.ones(len(parameters), bool); mask[section] = False
    np.testing.assert_array_equal(transported[mask], parameters[mask])
    recovered = transport_parameters(transported, changed_engine, original.engine)
    np.testing.assert_allclose(recovered, parameters, rtol=2e-14, atol=2e-14)
    np.testing.assert_array_equal(original.engine.basis, old_basis)
    assert changed_engine is not original.engine
    assert changed_engine.bounds is not original.engine.bounds
    assert original.engine.map_sigma == 2.
    settings = morphology_settings(changed_engine)
    assert settings['map_sigma'] == MORPHOLOGY_SCENARIOS[scenario]['map_sigma']
    assert settings['correlation_arcsec'] == pytest.approx(
        MORPHOLOGY_SCENARIOS[scenario]['correlation_arcsec'], rel=1e-12)


def test_transport_rejects_changed_basis_normalization_modes_and_parameter_geometry():
    model, parameters = comparison_example()
    new = morphology_engine(model.engine, 'amplitude4')
    for corrupted_basis in (2*np.asarray(model.engine.basis),
                            np.asarray(model.engine.basis)[:, ::-1]):
        corrupt = copy.copy(model.engine)
        corrupt.basis = jnp.asarray(corrupted_basis)
        with pytest.raises(ValueError, match='basis'):
            transport_parameters(parameters, corrupt, new)
    with pytest.raises(ValueError, match='full coupled-model'):
        transport_parameters(parameters[1:], model.engine, new)
    nonfinite = parameters.copy(); nonfinite[11] = np.nan
    with pytest.raises(ValueError, match='finite full'):
        transport_parameters(nonfinite, model.engine, new)


def test_transport_rejects_changed_quadrature_and_retains_source_constraints():
    model, parameters = comparison_example()
    new = morphology_engine(model.engine, 'correlation0p20')
    new.line_map = copy.copy(new.line_map)
    new.line_map.east = new.line_map.east+.001
    with pytest.raises(ValueError, match='identical quadrature'):
        transport_parameters(parameters, model.engine, new)
    with pytest.raises(ValueError, match='Unknown morphology'):
        morphology_engine(model.engine, 'none')
