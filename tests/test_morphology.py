"""Shared morphology bookkeeping must never silently reuse a different prior."""
import fcntl
import json
from pathlib import Path

import numpy as np
import pytest

from dingo import morphology
from dingo.real_workflow import read_json, sha256, write_json


@pytest.fixture
def setup(tmp_path, monkeypatch):
    config = tmp_path/'config.yaml'
    config.write_text('summary:\n  name: ID15665\n  ra: 64.0\n  dec: -24.0\n'
                      '  z: 1.16\n  lambda_rest: 1.875\n  filter: F444W\n')
    options = dict(directory=tmp_path/'15665/morph_fits/F150W', config_path=config,
                   science=tmp_path/'science.fits', error=tmp_path/'error.fits',
                   psf=tmp_path/'psf.fits', device='cpu', warmup=10, draws=20,
                   display=False)
    for name in ('science', 'error', 'psf'):
        options[name].write_bytes(name.encode())
    calls = []

    def prepare(source_id, **kwargs):
        calls.append(kwargs)
        output = Path(kwargs['imaging'])
        variant = output/'full_pixel'
        variant.mkdir(exist_ok=True)
        (output/'cutout.npz').write_bytes(b'prepared data')
        (output/'F150W_ePSF_original.fits').write_bytes(b'prepared psf')
        provenance = dict(source_id=source_id, input_sha256={
            str(path): sha256(path) for path in (output/'cutout.npz', output/'F150W_ePSF_original.fits')})
        write_json(output/'input_provenance.json', provenance)
        write_json(output/'imaging_variants.json', {'source_id': source_id, 'variants': ['full_pixel']})
        callback = kwargs.get('preview_callback')
        if callback:
            callback(morphology.ImagingResults(output, source_id, variants=('full_pixel',)))
        count = kwargs['samples']
        q = np.linspace(.5, .8, 4*count).reshape(4, count)
        samples = dict(xc=np.full_like(q, 90), yc=np.full_like(q, 89), flux=np.full_like(q, 100),
                       r_eff=np.full_like(q, 10), n=np.full_like(q, 1.5), ellip=1-q,
                       q=q, theta=np.full_like(q, .5), sky_back=np.zeros_like(q))
        np.savez(variant/'samples.npz', **samples)
        report = dict(source_id=source_id, completed=True, phase='complete',
            warmup_completed=kwargs['warmup'], retained_draws_per_chain=count,
            diagnostics=dict(chains=4, draws_per_chain=count, convergence_gate_passed=True,
                max_rank_rhat=1.001, min_bulk_ess=70., min_tail_ess=65., divergences=0),
            orientation_parameterization_ok=True, input_sha256=provenance['input_sha256'],
            samples_sha256=sha256(variant/'samples.npz'))
        write_json(variant/'imaging.json', report)
        return morphology.ImagingResults(output, source_id, variants=('full_pixel',))

    monkeypatch.setattr(morphology, 'prepare_imaging', prepare)
    monkeypatch.setattr(morphology, '_input_preview', lambda *a, **k: None)
    monkeypatch.setattr(morphology, '_review', lambda *a, **k: {})
    return options, calls, prepare


def test_single_fit_saves_pooled_text_and_honest_provenance(setup):
    options, calls, _ = setup
    result = morphology.fit_morphology(15665, **options)
    assert result.path == options['directory']/'fit_001'
    assert result.ready
    assert calls[0]['variants'] == ('full_pixel',)
    assert calls[0]['samples'] == 20
    report = read_json(result.path/'morphology.json')
    assert report['status'] == 'complete'
    assert report['completed'] == dict(chains=4, warmup_per_chain=10, retained_draws_per_chain=20)
    assert report['identity']['settings']['chains'] == 4
    assert report['finished_at']
    assert report['prepared_input_provenance']['input_sha256']
    assert report['identity']['raw_inputs']['science']['size_bytes'] == 7
    text = (result.path/'pysersic_15665_info.cat').read_text()
    for expected in ('Sérsic', 'Finished (UTC)', 'NumPyro NUTS', 'requested 10; completed 10',
                     'requested 20; saved 20', 'normalized image sum', 'normalized image value',
                     'q0=0 and q0=0.2', 'p16', 'p84', 'not directly fitted', 'passed'):
        assert expected in text
    parameters = report['parameter_summary']
    qrow = next(row for row in parameters if row['parameter'] == 'q')
    assert qrow['median'] == pytest.approx(.65)
    assert qrow['p16'] == pytest.approx(.548)
    assert qrow['p84'] == pytest.approx(.752)
    assert len(report['inclination_summary']) == 2


def test_reuse_does_not_launch_worker_or_change_recorded_finish(setup):
    options, calls, _ = setup
    first = morphology.fit_morphology(15665, **options)
    saved = (first.path/'morphology.json').read_bytes()
    txt = (first.path/'pysersic_15665_info.cat').read_bytes()
    second = morphology.fit_morphology(15665, **options)
    assert second.path == first.path
    assert len(calls) == 1
    assert (first.path/'morphology.json').read_bytes() == saved
    assert (first.path/'pysersic_15665_info.cat').read_bytes() == txt


def test_refit_makes_new_directory_preserving_previous_posterior(setup):
    options, calls, _ = setup
    first = morphology.fit_morphology(15665, **options)
    saved = (first.path/'full_pixel/samples.npz').read_bytes()
    report = (first.path/'morphology.json').read_bytes()
    second = morphology.fit_morphology(15665, **options, refit=True)
    assert first.path.name == 'fit_001' and second.path.name == 'fit_002'
    assert len(calls) == 2
    assert (first.path/'full_pixel/samples.npz').read_bytes() == saved
    assert (first.path/'morphology.json').read_bytes() == report
    assert read_json(options['directory']/'current.json')['fit_directory'] == 'fit_002'


def test_orphan_completion_reconciles_status_without_inventing_finish_time(setup):
    options, calls, _ = setup
    result = morphology.fit_morphology(15665, **options)
    path = result.path/'morphology.json'
    metadata = read_json(path)
    metadata.update(status='running', worker_completed=False, finished_at=None,
                    last_attempt_ended_at=None)
    metadata['attempts'][-1].update(status='running', ended_at=None)
    write_json(path, metadata)
    result = morphology.fit_morphology(15665, **options)
    recovered = read_json(path)
    assert len(calls) == 1 and result.ready
    assert recovered['status'] == 'complete'
    assert recovered['finished_at'] is None
    assert recovered['completion_observed_at']
    assert recovered['completed_report_modified_at']
    assert recovered['started_at'] == metadata['started_at']
    assert len(recovered['attempts']) == 1
    text = (result.path/'pysersic_15665_info.cat').read_text()
    assert 'not the actual run finish time' in text
    assert 'not recorded; completed while controller was unavailable' in text


@pytest.mark.parametrize('change', ['warmup', 'draws', 'seed', 'science', 'config'])
def test_cache_identity_changes_require_explicit_refit(setup, change):
    options, calls, _ = setup
    first = morphology.fit_morphology(15665, **options)
    if change in ('warmup', 'draws'):
        options[change] += 1
    elif change == 'seed':
        options['seed'] = 99
    elif change == 'science':
        options['science'].write_bytes(b'new mosaic bytes')
    else:
        path = options['config_path']
        path.write_text(path.read_text().replace('64.0', '64.1'))
    with pytest.raises(ValueError, match='refit=True'):
        morphology.fit_morphology(15665, **options)
    assert len(calls) == 1
    second = morphology.fit_morphology(15665, **options, refit=True)
    assert second.path != first.path


def test_prepared_input_corruption_is_rejected(setup):
    options, calls, _ = setup
    first = morphology.fit_morphology(15665, **options)
    (first.path/'cutout.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum changed'):
        morphology.fit_morphology(15665, **options)
    assert len(calls) == 1


def test_saved_samples_corruption_is_rejected(setup):
    options, calls, _ = setup
    first = morphology.fit_morphology(15665, **options)
    with (first.path/'full_pixel/samples.npz').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='samples checksum changed'):
        morphology.fit_morphology(15665, **options)
    assert len(calls) == 1


@pytest.mark.parametrize('exception,status', [(KeyboardInterrupt(), 'interrupted'),
                                              (RuntimeError('bad initialization'), 'failed')])
def test_failed_attempt_report_and_same_settings_resume(setup, monkeypatch, exception, status):
    options, calls, prepare = setup

    def fail(*args, **kwargs):
        raise exception

    monkeypatch.setattr(morphology, 'prepare_imaging', fail)
    with pytest.raises(type(exception)):
        morphology.fit_morphology(15665, **options)
    path = options['directory']/'fit_001'
    report = read_json(path/'morphology.json')
    assert report['status'] == status
    assert report['finished_at'] is None
    assert report['last_attempt_ended_at']
    assert status in (path/'pysersic_15665_info.cat').read_text()
    first_started = report['started_at']
    monkeypatch.setattr(morphology, 'prepare_imaging', prepare)
    result = morphology.fit_morphology(15665, **options)
    assert result.path == path
    report = read_json(path/'morphology.json')
    assert report['status'] == 'complete'
    assert report['started_at'] == first_started
    assert len(report['attempts']) == 2


def test_nonconverged_completed_fit_is_inspectable_and_not_silently_retried(setup, monkeypatch):
    options, calls, prepare = setup

    def unconverged(*args, **kwargs):
        result = prepare(*args, **kwargs)
        path = result.path/'full_pixel/imaging.json'
        report = read_json(path)
        report['diagnostics']['convergence_gate_passed'] = False
        write_json(path, report)
        return morphology.ImagingResults(result.path, 15665, variants=('full_pixel',))

    monkeypatch.setattr(morphology, 'prepare_imaging', unconverged)
    first = morphology.fit_morphology(15665, **options)
    assert not first.ready
    assert read_json(first.path/'morphology.json')['status'] == 'complete_not_accepted'
    with pytest.raises(ValueError, match='not yet accepted'):
        first.require_ready()
    second = morphology.fit_morphology(15665, **options)
    assert not second.ready and second.path == first.path
    assert len(calls) == 1


def test_active_orphan_worker_blocks_resume_and_refit(setup):
    options, calls, _ = setup
    first = morphology.fit_morphology(15665, **options)
    with (first.path/'full_pixel/.fit.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for refit in (False, True):
            with pytest.raises(RuntimeError, match='still running'):
                morphology.fit_morphology(15665, **options, refit=refit)
    assert len(calls) == 1
    assert not (options['directory']/'fit_002').exists()


def test_simultaneous_notebooks_cannot_share_cache(setup):
    options, calls, _ = setup
    options['directory'].mkdir(parents=True)
    with (options['directory']/'.morphology.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='Another morphology job'):
            morphology.fit_morphology(15665, **options)
    assert not calls


def test_preview_precedes_worker_samples_and_review(setup, monkeypatch):
    options, calls, _ = setup
    events = []

    def preview(result, **kwargs):
        assert not (result.path/'full_pixel/samples.npz').exists()
        events.append('preview')

    def review(result, **kwargs):
        assert (result.path/'full_pixel/samples.npz').exists()
        events.append('review')
        return {'corner': 'optional plotting problem'}

    monkeypatch.setattr(morphology, '_input_preview', preview)
    monkeypatch.setattr(morphology, '_review', review)
    result = morphology.fit_morphology(15665, **options)
    assert events == ['preview', 'review']
    assert 'optional plotting problem' in (result.path/'pysersic_15665_info.cat').read_text()
    assert result.ready


def test_incompatible_disk_thickness_reported_without_clipping(setup, monkeypatch):
    options, _, prepare = setup

    def thin(*args, **kwargs):
        result = prepare(*args, **kwargs)
        path = result.path/'full_pixel/samples.npz'
        with np.load(path) as data:
            arrays = {name: data[name] for name in data.files}
        arrays['ellip'][:] = .9
        arrays['q'][:] = .1
        np.savez(path, **arrays)
        report_path = result.path/'full_pixel/imaging.json'
        report = read_json(report_path)
        report['samples_sha256'] = sha256(path)
        write_json(report_path, report)
        return morphology.ImagingResults(result.path, 15665, variants=('full_pixel',))

    monkeypatch.setattr(morphology, 'prepare_imaging', thin)
    result = morphology.fit_morphology(15665, **options)
    report = read_json(result.path/'morphology.json')
    thick = report['inclination_summary'][1]
    assert thick['q0'] == .2 and thick['status'] == 'incompatible'
    assert thick['median_deg'] is None and thick['incompatible_fraction'] == 1.


@pytest.mark.parametrize('changes', [dict(warmup=0), dict(draws=-1), dict(seed=-1),
                                    dict(cutout_size=180), dict(fit_radius=.1), dict(device='tpu')])
def test_invalid_request_has_no_output(setup, changes):
    options, calls, _ = setup
    options.update(changes)
    with pytest.raises(ValueError):
        morphology.fit_morphology(15665, **options)
    assert not options['directory'].exists()
    assert not calls
