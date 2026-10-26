"""Single Sérsic selection is explicit and cannot weaken legacy acceptance."""
import json
from pathlib import Path

import numpy as np
import pytest

from dingo import rotation_imaging as imaging
from dingo import three_file_imaging as review
from dingo.real_workflow import RealSource, sha256


@pytest.fixture
def source(tmp_path):
    config = tmp_path/'config.yaml'
    config.write_text('summary:\n  name: ID15665\n  ra: 64.0\n  dec: -24.0\n'
                      '  z: 1.16\n  lambda_rest: 1.875\n  filter: F444W\n')
    return RealSource(15665, bundle=tmp_path/'bundle', results=tmp_path/'results', config=config)


def accepted_variant(path, source_id=15665):
    path.mkdir(parents=True, exist_ok=True)
    np.savez(path/'samples.npz', q=np.full((4, 25), .5))
    report = dict(source_id=source_id, completed=True,
        diagnostics=dict(convergence_gate_passed=True), orientation_parameterization_ok=True,
        samples_sha256=sha256(path/'samples.npz'), input_sha256={})
    (path/'imaging.json').write_text(json.dumps(report))


def mark_single(path, source_id=15665):
    path.mkdir(parents=True, exist_ok=True)
    (path/imaging.VARIANT_METADATA).write_text(json.dumps(
        dict(source_id=source_id, variants=['full_pixel'])))


def options(source):
    return dict(bundle=source.bundle, results=source.results, config=source.config)


def test_single_selection_runs_one_worker_and_preview_knows_selection(source, monkeypatch):
    workers, previews = [], []
    monkeypatch.setattr(imaging, 'prepare_inputs',
        lambda source, output, **kwargs: output.mkdir(parents=True, exist_ok=True))

    def worker(args, log, **kwargs):
        output = Path(args[args.index('--output')+1])
        assert args[args.index('--mask-core')+1] == '0.0'
        workers.append(output.name)
        accepted_variant(output)

    monkeypatch.setattr(imaging, 'run_command', worker)
    output = source.results/'morph_fits/F150W'
    result = imaging.prepare_imaging(source.source_id, imaging=output,
        variants=('full_pixel',), preview_callback=lambda r: previews.append(r.variants),
        **options(source))
    assert workers == ['full_pixel']
    assert previews == [(('full_pixel', 0.),)]
    assert result.variants == (('full_pixel', 0.),)
    assert result.ready and result.require_ready() == output
    assert result.summary()['fit'].tolist() == ['full_pixel']
    assert not (output/'core_pixel_clean').exists()

    # Opening the output without an explicit selection respects its marker.
    reopened = imaging.prepare_imaging(source.source_id, imaging=output, run=False,
        **options(source))
    assert reopened.variants == result.variants and reopened.ready
    assert workers == ['full_pixel']


def test_legacy_output_still_requires_both_alternatives(tmp_path):
    accepted_variant(tmp_path/'full_pixel')
    result = imaging.ImagingResults(tmp_path, 15665)
    assert result.variants == imaging.VARIANTS
    assert not result.ready
    with pytest.raises(ValueError, match='not yet accepted'):
        review.inclination_prior(result, total_sigma_deg=5.)
    accepted_variant(tmp_path/'core_pixel_clean')
    assert imaging.imaging_ready(tmp_path, 15665)
    assert not (tmp_path/imaging.VARIANT_METADATA).exists()


@pytest.mark.parametrize('run', [True, False])
def test_existing_legacy_fit_cannot_be_reinterpreted_as_single(source, monkeypatch, run):
    output = source.results/'legacy'
    accepted_variant(output/'full_pixel')
    monkeypatch.setattr(imaging, 'prepare_inputs', lambda *a, **kw: pytest.fail('must not prepare'))
    with pytest.raises(ValueError, match='variant selection changed'):
        imaging.prepare_imaging(source.source_id, imaging=output,
            variants=('full_pixel',), run=run, **options(source))
    assert not (output/imaging.VARIANT_METADATA).exists()


def test_changed_declared_selection_is_rejected(source):
    output = source.results/'single'
    mark_single(output)
    accepted_variant(output/'full_pixel')
    before = (output/imaging.VARIANT_METADATA).read_bytes()
    with pytest.raises(ValueError, match='variant selection changed'):
        imaging.prepare_imaging(source.source_id, imaging=output,
            variants=('full_pixel', 'core_pixel_clean'), **options(source))
    assert (output/imaging.VARIANT_METADATA).read_bytes() == before


def test_single_prior_uses_one_posterior_and_exact_user_width(tmp_path):
    mark_single(tmp_path)
    accepted_variant(tmp_path/'full_pixel')
    result = imaging.ImagingResults(tmp_path, 15665)
    prior = review.inclination_prior(result, q0_bounds=(0., 1e-10), total_sigma_deg=5.)
    assert prior['inclination_deg'] == pytest.approx(60.)
    assert prior['inclination_sigma_deg'] == 5.
    assert prior['stellar_inclination_std_deg'] == pytest.approx(0.)
    assert [item['variant'] for item in prior['sources']] == ['full_pixel']
    assert prior['alternative_fits_not_independent_data'] is False
    assert prior['kind'] == 'imaging_mean_user_width_gaussian'


def test_single_figure_has_one_row_and_unselected_report_is_rejected(tmp_path):
    import matplotlib.pyplot as plt
    mark_single(tmp_path)
    accepted_variant(tmp_path/'full_pixel')
    values = np.arange(81).reshape(9, 9).astype(float)
    np.savez(tmp_path/'full_pixel/fit.npz', image=values, model=values,
             error=np.ones_like(values), residual=np.zeros_like(values),
             mask=np.zeros_like(values, dtype=bool))
    result = imaging.ImagingResults(tmp_path, 15665)
    with pytest.raises(ValueError, match='not selected'):
        result._report('core_pixel_clean')
    figure = result.figure()
    try:
        image_axes = [ax for ax in figure.axes if ax.images]
        assert len(image_axes) == 3
        assert all(ax.get_subplotspec().get_gridspec().nrows == 1 for ax in image_axes)
    finally:
        plt.close(figure)


def test_automatic_discovery_does_not_replace_requested_single_with_legacy_pair(source, monkeypatch):
    legacy = source.directory/'imaging'
    for name, _ in imaging.VARIANTS:
        accepted_variant(legacy/name)
    monkeypatch.setattr(imaging, 'prepare_inputs',
        lambda source, output, **kwargs: output.mkdir(parents=True, exist_ok=True))
    calls = []

    def worker(args, log, **kwargs):
        path = Path(args[args.index('--output')+1])
        calls.append(path.name)
        accepted_variant(path)

    monkeypatch.setattr(imaging, 'run_command', worker)
    result = imaging.prepare_imaging(source.source_id, variants=('full_pixel',),
        **options(source))
    assert result.path == source.directory/'imaging_runs/pysersic_v1'
    assert calls == ['full_pixel'] and result.ready
    assert not (legacy/imaging.VARIANT_METADATA).exists()


@pytest.mark.parametrize('variants', [[], 'full_pixel', ['typo'], ['full_pixel', 'full_pixel']])
def test_invalid_selection_is_rejected_without_output(source, variants):
    with pytest.raises(ValueError, match='variants must'):
        imaging.prepare_imaging(source.source_id, variants=variants, **options(source))
    assert not source.directory.exists()


def test_wrong_source_marker_is_rejected(tmp_path):
    mark_single(tmp_path, source_id=3700)
    with pytest.raises(ValueError, match='different source'):
        imaging.imaging_ready(tmp_path, 15665)


def test_inspection_does_not_write_selection_marker(source):
    output = source.results/'empty'
    result = imaging.prepare_imaging(source.source_id, run=False, imaging=output,
        variants=('full_pixel',), **options(source))
    assert not result.ready and not output.exists()
    assert result.variants == (('full_pixel', 0.),)


def test_constructor_checks_expected_variants_without_writing(tmp_path):
    missing = tmp_path/'missing'
    result = imaging.ImagingResults(missing, 15665, variants=('full_pixel',))
    assert result.variants == (('full_pixel', 0.),)
    assert not result.ready and not missing.exists()
    mark_single(tmp_path)
    with pytest.raises(ValueError, match='variant selection changed'):
        imaging.ImagingResults(tmp_path, 15665, variants=('full_pixel', 'core_pixel_clean'))


def test_single_input_is_magma_and_excluded_pixels_are_white(tmp_path):
    import matplotlib.pyplot as plt
    mark_single(tmp_path)
    values = np.arange(81).reshape(9, 9).astype(float)
    np.savez(tmp_path/'cutout.npz', image=values, error=np.ones_like(values),
             center_xy=np.array([4., 4.]), pixel_scale_arcsec=.1)
    result = imaging.ImagingResults(tmp_path, 15665)
    figure = result.input_figure(fit_radius=.25)
    try:
        axis = figure.axes[0]
        image = axis.images[0]
        assert image.get_cmap().name == 'magma'
        np.testing.assert_array_equal(image.get_cmap().get_bad(), [1., 1., 1., 1.])
        assert 'white pixels are excluded' in axis.get_title()
        assert np.ma.getmaskarray(image.get_array()).any()
    finally:
        plt.close(figure)


def test_single_empty_figure_has_only_one_row(tmp_path):
    import matplotlib.pyplot as plt
    result = imaging.ImagingResults(tmp_path/'missing', 15665, variants=('full_pixel',))
    figure = result.figure()
    try:
        assert len(figure.axes) == 3
        assert all('full_pixel: no saved model yet' in ax.texts[0].get_text()
                   for ax in figure.axes)
    finally:
        plt.close(figure)


def test_read_only_accepted_single_preserves_every_file(source):
    output = source.results/'single'
    mark_single(output)
    accepted_variant(output/'full_pixel')
    before = {path.relative_to(output): path.read_bytes()
              for path in output.rglob('*') if path.is_file()}
    result = imaging.prepare_imaging(source.source_id, run=False, override=True,
        imaging=output, variants=('full_pixel',), **options(source))
    assert result.ready
    after = {path.relative_to(output): path.read_bytes()
             for path in output.rglob('*') if path.is_file()}
    assert after == before


def test_interrupted_preparation_can_resume_as_single_fit(source, monkeypatch):
    output = source.results/'single'
    attempts = []

    def prepare(source, path, **kwargs):
        assert imaging.imaging_variants(path, source.source_id) == (('full_pixel', 0.),)
        attempts.append(path)
        # Model selection must exist before any scientific input can be saved.
        (path/'cutout.npz').write_bytes(b'partial preparation fixture')
        if len(attempts) == 1:
            raise KeyboardInterrupt

    monkeypatch.setattr(imaging, 'prepare_inputs', prepare)
    monkeypatch.setattr(imaging, 'run_command', lambda args, *a, **kw:
        accepted_variant(Path(args[args.index('--output')+1])))
    with pytest.raises(KeyboardInterrupt):
        imaging.prepare_imaging(source.source_id, imaging=output, variants=('full_pixel',),
            **options(source))
    result = imaging.prepare_imaging(source.source_id, imaging=output, variants=('full_pixel',),
        **options(source))
    assert len(attempts) == 2 and result.ready
    assert not (output/'core_pixel_clean').exists()


def test_read_only_discovery_retains_incomplete_single_selection(source):
    output = source.directory/'imaging'
    mark_single(output)
    # The fit is not accepted yet, but it still has one declared variant.
    result = imaging.prepare_imaging(source.source_id, run=False, **options(source))
    assert result.path == output
    assert result.variants == (('full_pixel', 0.),)
    assert not result.ready
    assert not (source.directory/'imaging_runs').exists()
