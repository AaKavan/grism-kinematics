"""Imaging review happens before sampling; its optional gas prior stays soft."""
import json

import numpy as np
import pytest

from dingo import three_file_imaging as review
from dingo.real_workflow import sha256
from dingo.rotation_imaging import ImagingResults, VARIANTS


def accepted(tmp_path, full_q, core_q):
    for (name, _), q in zip(VARIANTS, (full_q, core_q)):
        path = tmp_path/name
        path.mkdir()
        np.savez(path/'samples.npz', q=q)
        (path/'imaging.json').write_text(json.dumps(dict(source_id=6170,
            completed=True, diagnostics=dict(convergence_gate_passed=True),
            orientation_parameterization_ok=True, input_sha256={},
            samples_sha256=sha256(path/'samples.npz'))))
    return ImagingResults(tmp_path, 6170)


def test_prior_mixes_alternative_fits_equally_and_preserves_added_scatter(tmp_path):
    result = accepted(tmp_path, np.full((4, 100), .5),
                      np.full((4, 20), np.cos(np.deg2rad(30.))))
    prior = review.inclination_prior(result, scatter_deg=10., q0_bounds=(0., 1e-10))
    assert prior['inclination_deg'] == pytest.approx(45.)
    assert prior['inclination_sigma_deg'] == pytest.approx(np.sqrt(15.**2+10.**2))
    assert prior['stellar_inclination_std_deg'] == pytest.approx(15.)
    assert prior['alternative_fits_not_independent_data']
    assert prior['kind'] == 'imaging_gaussian_approximation'
    assert prior['source_id'] == 6170
    assert len(prior['sources']) == 2


def test_default_thickness_prior_is_deterministic_and_soft(tmp_path):
    result = accepted(tmp_path, np.full((4, 50), .7), np.full((4, 50), .7))
    first = review.inclination_prior(result)
    assert first == review.inclination_prior(result)
    assert first['q0_uniform_before_rejection'] == [.05, .30]
    assert first['additional_scatter_deg'] == 10.
    assert first['inclination_sigma_deg'] > 10.
    assert first['inclination_deg'] > np.rad2deg(np.arccos(.7))
    assert first == review.inclination_prior(result, total_sigma_deg=None)
    assert 'total_sigma_deg' not in first and 'width_policy' not in first


def test_total_width_is_exact_preserves_imaging_mean_and_records_user_choice(tmp_path):
    from dingo.three_file_fit import validate_inclination_metadata
    result = accepted(tmp_path, np.full((4, 100), .5),
                      np.full((4, 20), np.cos(np.deg2rad(30.))))
    original = review.inclination_prior(result, q0_bounds=(0., 1e-10))
    prior = review.inclination_prior(result, total_sigma_deg=5., q0_bounds=(0., 1e-10))
    assert prior['inclination_sigma_deg'] == prior['total_sigma_deg'] == 5.
    assert prior['inclination_deg'] == original['inclination_deg']
    assert prior['stellar_inclination_std_deg'] == pytest.approx(15.)
    assert prior['kind'] == 'imaging_mean_user_width_gaussian'
    assert prior['additional_scatter_deg'] is None
    assert 'User-selected total Gaussian width' in prior['width_policy']
    for key in ('source_id', 'sources', 'q0_uniform_before_rejection',
                'stellar_inclination_mean_deg', 'invalid_q_q0_pair_fraction',
                'alternative_fits_not_independent_data', 'truncation_deg'):
        assert prior[key] == original[key]
    assert validate_inclination_metadata(prior, result.source_id,
        prior['inclination_deg'], 5.) == prior
    assert prior == review.inclination_prior(result, scatter_deg=0.,
        total_sigma_deg=np.float64(5.), q0_bounds=(0., 1e-10))


@pytest.mark.parametrize('sigma', [0., -1., np.nan, np.inf, -np.inf,
                                   True, False, np.bool_(True), '5', [5.]])
def test_total_width_rejects_invalid_values_before_reading_imaging(sigma):
    with pytest.raises(ValueError, match='total_sigma_deg must be a finite positive number'):
        review.inclination_prior(object(), total_sigma_deg=sigma)


def test_prior_rejects_unaccepted_or_changed_samples(tmp_path):
    result = accepted(tmp_path, np.full((4, 20), .7), np.full((4, 20), .7))
    path = tmp_path/'full_pixel/imaging.json'
    report = json.loads(path.read_text())
    report['diagnostics']['convergence_gate_passed'] = False
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='not yet accepted'):
        review.inclination_prior(result)
    report['diagnostics']['convergence_gate_passed'] = True
    path.write_text(json.dumps(report))
    np.savez(tmp_path/'full_pixel/samples.npz', q=np.full((4, 20), .9))
    with pytest.raises(ValueError, match='checksum'):
        review.inclination_prior(result)


@pytest.mark.parametrize('q', [1., -.1, 1.1, np.nan])
def test_prior_does_not_clip_unphysical_or_face_on_samples(tmp_path, q):
    result = accepted(tmp_path, np.full((4, 20), q), np.full((4, 20), q))
    with pytest.raises(ValueError):
        review.inclination_prior(result)


@pytest.mark.parametrize('run', [False, True])
def test_review_shows_input_before_worker_and_read_only_never_saves(tmp_path, monkeypatch, capsys, run):
    events = []
    for name, _ in VARIANTS:
        path = tmp_path/name
        path.mkdir()
        (path/'samples.npz').touch()
        (path/'fit.npz').touch()

    class Result:
        path = tmp_path
        def input_figure(self, **kwargs):
            return 'actual F150W image'
        def summary(self):
            return 'diagnostics'
        def figure(self):
            return 'model residual'
        def parameter_summary(self, **kwargs):
            return 'native parameters'
        def inclination_summary(self, **kwargs):
            import pandas as pd
            return pd.DataFrame([
                dict(q0=0., median_deg=60., minus_1sigma_deg=2., plus_1sigma_deg=3.,
                     incompatible_fraction=0., status='compatible'),
                dict(q0=.2, median_deg=62., minus_1sigma_deg=2.5, plus_1sigma_deg=3.5,
                     incompatible_fraction=0., status='compatible')]).set_index('q0')
        def corner(self, **kwargs):
            events.append(('corner saved', kwargs['save']))
            return 'PySersic corner'
        def inclination_figure(self, **kwargs):
            events.append(('inclination saved', kwargs['save']))
            return 'inclination'

    def prepare(source_id, **kwargs):
        assert source_id == 6170
        assert kwargs['run'] == run and kwargs['override'] is True
        result = Result()
        kwargs['preview_callback'](result)
        events.append('worker starts' if run else 'existing fit returned')
        return result

    monkeypatch.setattr(review, 'prepare_imaging', prepare)
    monkeypatch.setattr(review, '_show_figure', events.append)
    monkeypatch.setattr(review, '_show', events.append)
    result = review.review_imaging(6170, run=run, override=True)
    assert isinstance(result, Result)
    assert events[0] == 'actual F150W image'
    assert events[1] == ('worker starts' if run else 'existing fit returned')
    assert ('corner saved', run) in events
    assert ('inclination saved', run) in events
    assert events.count('PySersic corner') == 1
    printed = capsys.readouterr().out
    assert 'q0 = 0: i = 60.000 (+3.000/−2.000) deg' in printed
    assert 'q0 = 0.2: i = 62.000 (+3.500/−2.500) deg' in printed
