"""Real-image preparation and orchestration must not silently admit failed priors."""
import json
from pathlib import Path

import numpy as np
import pytest

from dingo import rotation_imaging as imaging
from dingo.real_workflow import RealSource, sha256


@pytest.fixture
def source(tmp_path):
    config = tmp_path/'config.yaml'
    config.write_text('summary:\n  name: ID15665\n  ra: 64.0\n  dec: -24.0\n'
                      '  z: 1.16\n  lambda_rest: 1.875\n  filter: F444W\n')
    return RealSource(15665, bundle=tmp_path/'bundle', results=tmp_path/'results', config=config)


def accepted_variant(path, sid=15665):
    path.mkdir(parents=True, exist_ok=True)
    np.savez(path/'samples.npz', q=np.full((4, 100), .7))
    report = dict(source_id=sid, completed=True, band='F150W', real_imaging=True,
        diagnostics=dict(convergence_gate_passed=True), orientation_parameterization_ok=True,
        samples_sha256=sha256(path/'samples.npz'), input_sha256={})
    (path/'imaging.json').write_text(json.dumps(report))


def accepted(path, sid=15665):
    for name, _ in imaging.VARIANTS:
        accepted_variant(path/name, sid)


def options(source):
    return dict(bundle=source.bundle, results=source.results, config=source.config)


def test_accepted_existing_imaging_is_reused_without_worker(source, monkeypatch):
    old = source.directory/'imaging'
    accepted(old)
    monkeypatch.setattr(imaging, 'run_command', lambda *a, **k: pytest.fail('Unexpected worker'))
    result = imaging.prepare_imaging(source.source_id, **options(source))
    assert result.ready and result.require_ready() == old
    assert not (source.directory/'imaging_runs').exists()


def test_prepared_image_is_previewed_before_any_fitting_worker(source, monkeypatch):
    events = []

    def prepare(src, output, **kwargs):
        output.mkdir(parents=True)
        (output/'cutout.npz').write_bytes(b'prepared-image marker')
        events.append('prepared')

    def preview(result):
        assert (result.path/'cutout.npz').exists()
        assert not (result.path/'full_pixel/imaging.json').exists()
        events.append('preview')

    def worker(args, log, **kwargs):
        assert events[:2] == ['prepared', 'preview']
        path = Path(args[args.index('--output')+1])
        events.append(path.name)
        accepted_variant(path)

    monkeypatch.setattr(imaging, 'prepare_inputs', prepare)
    monkeypatch.setattr(imaging, 'run_command', worker)
    result = imaging.prepare_imaging(source.source_id, preview_callback=preview, **options(source))
    assert result.ready
    assert events == ['prepared', 'preview', 'full_pixel', 'core_pixel_clean']


@pytest.mark.parametrize('run', [False, True])
def test_accepted_imaging_preview_does_not_prepare_or_refit(source, monkeypatch, run):
    old = source.directory/'imaging'
    accepted(old)
    monkeypatch.setattr(imaging, 'prepare_inputs', lambda *a, **k: pytest.fail('Unexpected preparation'))
    monkeypatch.setattr(imaging, 'run_command', lambda *a, **k: pytest.fail('Unexpected worker'))
    previews = []
    result = imaging.prepare_imaging(source.source_id, run=run,
        preview_callback=lambda value: previews.append(value.path), **options(source))
    assert result.ready
    assert previews == [old]
    assert not (source.directory/'imaging_runs').exists()


def test_failed_historical_imaging_runs_both_new_variants_without_overwrite(source, monkeypatch):
    old = source.directory/'imaging/full_pixel'
    old.mkdir(parents=True)
    report = json.dumps(dict(source_id=source.source_id, completed=True, diagnostics=None))
    (old/'imaging.json').write_text(report)
    called = []

    def prepare(source, output, **kwargs):
        output.mkdir(parents=True)

    def run(args, log, **kwargs):
        output = Path(args[args.index('--output')+1])
        called.append(output)
        if output.name == 'full_pixel':
            raise RuntimeError('first optimizer failed')
        accepted_variant(output)

    monkeypatch.setattr(imaging, 'prepare_inputs', prepare)
    monkeypatch.setattr(imaging, 'run_command', run)
    result = imaging.prepare_imaging(source.source_id, **options(source))
    assert [p.name for p in called] == ['full_pixel', 'core_pixel_clean']
    assert result.path == source.directory/'imaging_runs/pysersic_v1'
    assert (old/'imaging.json').read_text() == report
    assert not result.ready
    assert 'first optimizer failed' in result.summary().iloc[0]['note']
    with pytest.raises(ValueError, match='not yet accepted'):
        result.require_ready()


def test_inspect_only_makes_no_files(source, monkeypatch):
    monkeypatch.setattr(imaging, 'prepare_inputs', lambda *a, **k: pytest.fail('Unexpected preparation'))
    monkeypatch.setattr(imaging, 'run_command', lambda *a, **k: pytest.fail('Unexpected worker'))
    result = imaging.prepare_imaging(source.source_id, run=False, **options(source))
    assert not result.ready
    assert not source.directory.exists()


def _single_rejected_result(path, report):
    (path/'full_pixel').mkdir(parents=True)
    (path/imaging.VARIANT_METADATA).write_text(json.dumps(
        dict(source_id=8910, variants=['full_pixel'])))
    (path/'full_pixel/imaging.json').write_text(json.dumps(dict(source_id=8910, **report)))
    return imaging.ImagingResults(path, 8910)


def test_worker_abort_reports_native_error_instead_of_convergence_advice(tmp_path):
    result = _single_rejected_result(tmp_path, dict(completed=False, phase='initialization'))
    (tmp_path/'morphology.json').write_text(json.dumps(
        dict(worker_errors={'full_pixel': 'Job exited with code -6'})))
    (tmp_path/'full_pixel.log').write_text('Initializing PySersic\nfree(): invalid size\n')
    with pytest.raises(ValueError) as error:
        result.require_ready()
    message = str(error.value)
    assert 'no completed posterior' in message and 'initialization' in message
    assert 'code -6' in message and 'free(): invalid size' in message
    assert str(tmp_path/'full_pixel.log') in message
    assert 'Increase' not in message and 'OVERRIDE' not in message and 'refit=True' not in message
    assert not result.ready


def test_completed_divergent_fit_identifies_the_only_failed_metric(tmp_path):
    result = _single_rejected_result(tmp_path, dict(completed=True,
        diagnostics=dict(convergence_gate_passed=False, all_diagnostics_finite=True,
            chains=4, max_rank_rhat=1.00094, min_bulk_ess=4057., min_tail_ess=3374.,
            divergences=1, bfmi_by_chain=[1.08, 1., .97, .99]),
        orientation_parameterization_ok=True))
    np.savez(tmp_path/'full_pixel/samples.npz', q=np.full((4, 10), .93))
    with pytest.raises(ValueError) as error:
        result.require_ready()
    message = str(error.value)
    assert 'sampling completed' in message and 'divergences=1 (requires zero)' in message
    assert 'maximum rank Rhat' not in message and 'minimum bulk ESS' not in message
    assert 'Adding draws cannot remove' in message and 'refit=True' in message
    assert 'OVERRIDE' not in message
    assert not result.ready


def test_completed_fit_lists_mixing_and_boundary_failures(tmp_path):
    result = _single_rejected_result(tmp_path, dict(completed=True,
        diagnostics=dict(convergence_gate_passed=False, chains=4,
            max_rank_rhat=1.03, min_bulk_ess=70., min_tail_ess=200.,
            divergences=0, bfmi_by_chain=[.1, .5, .6, .7]),
        orientation_parameterization_ok=False))
    with pytest.raises(ValueError) as error:
        result.require_ready()
    message = str(error.value)
    for text in ('Rhat=1.03', 'bulk ESS=70', 'tail ESS=200', 'BFMI=0.1',
                 'boundary checks failed', 'samples.npz is missing'):
        assert text in message
    assert 'divergences=' not in message


@pytest.mark.parametrize('override', [False, True])
def test_native_crash_retries_once_from_committed_checkpoint(source, monkeypatch, override):
    calls = []
    monkeypatch.setattr(imaging, 'prepare_inputs', lambda source, output, **kw: output.mkdir(parents=True))

    def run(args, log, **kwargs):
        output = Path(args[args.index('--output')+1])
        calls.append(output.name)
        if len(calls) == 1:
            output.mkdir()
            checkpoint = output/'checkpoint_123.pkl'
            checkpoint.write_bytes(b'checkpoint fixture, not deserialized by orchestration')
            (output/'imaging.json').write_text(json.dumps(dict(source_id=source.source_id,
                checkpoint_file=checkpoint.name, checkpoint_sha256=sha256(checkpoint))))
            raise RuntimeError('Job exited with code -11. See log')
        if len(calls) == 2:
            assert (output/'checkpoint_123.pkl').exists()
        accepted_variant(output)

    monkeypatch.setattr(imaging, 'run_command', run)
    result = imaging.prepare_imaging(source.source_id, override=override, **options(source))
    assert calls == ['full_pixel', 'full_pixel', 'core_pixel_clean']
    assert result.ready


def test_interruption_is_not_automatically_restarted(source, monkeypatch):
    calls = []
    monkeypatch.setattr(imaging, 'prepare_inputs', lambda source, output, **kw: output.mkdir(parents=True))

    def run(*args, **kwargs):
        calls.append(1)
        raise KeyboardInterrupt

    monkeypatch.setattr(imaging, 'run_command', run)
    with pytest.raises(KeyboardInterrupt):
        imaging.prepare_imaging(source.source_id, **options(source))
    assert len(calls) == 1


def test_discovery_rejects_wrong_source_and_detects_corrupted_samples(source):
    wrong = source.directory/'imaging'
    accepted(wrong, 3700)
    assert not imaging.imaging_ready(wrong, source.source_id)
    with pytest.raises(ValueError, match='different source'):
        imaging.ImagingResults(wrong, source.source_id)
    new = source.directory/'imaging_runs/new'
    accepted(new)
    assert imaging.imaging_directory(source) == new
    np.savez(new/'full_pixel/samples.npz', q=np.full((4, 100), .1))
    with pytest.raises(ValueError, match='samples checksum'):
        imaging.imaging_ready(new, source.source_id)


@pytest.fixture
def mosaics(tmp_path):
    from astropy.io import fits
    from astropy.wcs import WCS
    w = WCS(naxis=2)
    w.wcs.crpix = [51, 51]
    w.wcs.cdelt = [-.03/3600, .03/3600]
    w.wcs.crval = [64., -24.]
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    header = w.to_header()
    header['BUNIT'] = 'MJy/sr'
    science, error, psf = [tmp_path/name for name in ('sci.fits', 'err.fits', 'psf.fits')]
    y, x = np.indices((101, 101))
    fits.writeto(science, np.exp(-((x-50)**2+(y-50)**2)/50), header)
    fits.writeto(error, np.full((101, 101), .01), header)
    ph = fits.Header({'PIXSCALE': .03})
    fits.writeto(psf, np.ones((9, 9))/81, ph)
    return dict(science=science, error=error, psf=psf)


def test_real_cutout_alignment_cache_and_aperture(source, mosaics):
    output = source.directory/'imaging_runs/test'
    imaging.prepare_inputs(source, output, cutout_size=31, fit_radius=.3, **mosaics)
    with np.load(output/'cutout.npz') as data:
        assert data['image'].shape == (31, 31)
        np.testing.assert_allclose(data['center_xy'], [15, 15], atol=1e-6)
        assert data['error'].shape == data['image'].shape
    digest = sha256(output/'cutout.npz')
    imaging.prepare_inputs(source, output, cutout_size=31, fit_radius=.3, **mosaics)
    assert sha256(output/'cutout.npz') == digest
    with pytest.raises(ValueError, match='settings changed'):
        imaging.prepare_inputs(source, output, cutout_size=41, fit_radius=.3, **mosaics)
    with pytest.raises(ValueError, match='aperture must fit'):
        imaging.prepare_inputs(source, output, cutout_size=31, fit_radius=1., **mosaics)


def test_shifted_error_wcs_is_rejected_before_output(source, mosaics):
    from astropy.io import fits
    with fits.open(mosaics['error'], mode='update') as h:
        h[0].header['CRPIX1'] += 1
    output = source.directory/'imaging_runs/test'
    with pytest.raises(ValueError, match='not aligned'):
        imaging.prepare_inputs(source, output, cutout_size=31, fit_radius=.3, **mosaics)
    assert not output.exists()


def test_inconsistent_saved_residuals_cannot_be_displayed(source):
    import matplotlib.pyplot as plt
    output = source.directory/'imaging_runs/test'
    accepted(output)
    values = np.ones((9, 9))
    np.savez(output/'full_pixel/fit.npz', image=values, model=values, error=values,
             residual=values, mask=np.zeros_like(values, dtype=bool))
    try:
        with pytest.raises(AssertionError):
            imaging.ImagingResults(output, source.source_id).figure()
    finally:
        plt.close('all')
