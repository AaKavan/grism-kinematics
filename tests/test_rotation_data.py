"""Source identity and accepted-prior requirements for the new data loader."""
import json

import numpy as np
import pytest

from dingo.rotation_data import inclination_prior


def imaging_reports(tmp_path, source_id=21636):
    for name in ('full_pixel', 'core_pixel_clean'):
        directory = tmp_path / name
        directory.mkdir()
        # Metadata fixture only; this is not a synthetic science recovery run.
        np.savez(directory / 'samples.npz', q=np.linspace(.2, .8, 80).reshape(2, 40))
        report = dict(source_id=source_id, completed=True,
            diagnostics={'convergence_gate_passed': True},
            orientation_parameterization_ok=True, input_sha256={})
        (directory / 'imaging.json').write_text(json.dumps(report))
    return tmp_path


def test_isotropic_orientation_needs_no_imaging_files():
    angle, log_density, description = inclination_prior(21636, None, 'isotropic')
    density = np.exp(log_density)
    assert np.trapezoid(density, angle) == pytest.approx(1.)
    np.testing.assert_allclose(density / np.sin(angle), np.full_like(angle, density[0] / np.sin(angle[0])))
    assert description['kind'] == 'isotropic'
    with pytest.raises(ValueError, match='Inclination must'):
        inclination_prior(21636, None, 'automatic')


def test_imaging_requires_same_source_and_passed_checks(tmp_path):
    path = imaging_reports(tmp_path)
    with pytest.raises(ValueError, match='different source'):
        inclination_prior(3700, path, 'imaging')
    report_path = path / 'full_pixel/imaging.json'
    report = json.loads(report_path.read_text())
    report['diagnostics']['convergence_gate_passed'] = False
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='failed its diagnostics'):
        inclination_prior(21636, path, 'imaging')


def test_imaging_preserves_validated_density_definition(tmp_path):
    # The historical implementation is the numerical reference; the new loader
    # itself does not import any diagnostics runner.
    from diagnostics.native_source_grid import inclination_prior as previous_prior
    path = imaging_reports(tmp_path)
    actual = inclination_prior(21636, path, 'imaging')
    previous = previous_prior(21636, path, 'imaging')
    np.testing.assert_array_equal(actual[0], previous[0])
    np.testing.assert_array_equal(actual[1], previous[1])
    assert actual[2] == previous[2]
