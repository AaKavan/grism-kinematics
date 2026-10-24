"""Review saved PySersic samples without a fitter, and expose geometry assumptions."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

from dingo.real_workflow import sha256
from dingo.rotation_imaging import ImagingResults


@pytest.fixture
def result(tmp_path):
    path = tmp_path/'imaging'
    variant = path/'full_pixel'
    variant.mkdir(parents=True)
    rng = np.random.default_rng(82)
    samples = {name: center+rng.normal(0, scale, (4, 100)) for name, center, scale in (
        ('xc', 10., .1), ('yc', 10., .1), ('flux', 200., 1.), ('r_eff', 3., .1),
        ('n', 1., .05), ('ellip', .4, .01), ('theta', 3.2, .01), ('sky_back', 0., .02))}
    samples['ellip_base'] = rng.uniform(size=(4, 100))
    samples['q'] = 1-samples['ellip']
    np.savez(variant/'samples.npz', **samples)
    y, x = np.indices((21, 21))
    image = np.exp(-((x-10)**2+(y-10)**2)/6)
    error = np.full_like(image, .01)
    np.savez(path/'cutout.npz', image=image, error=error,
             center_xy=[10, 10], pixel_scale_arcsec=.03)
    provenance = dict(source_id=6170, band='F150W', unit='MJy/sr',
                      input_sha256={str(path/'cutout.npz'): sha256(path/'cutout.npz')})
    (path/'input_provenance.json').write_text(json.dumps(provenance))
    report = dict(source_id=6170, completed=True, diagnostics={'convergence_gate_passed': True},
                  orientation_parameterization_ok=True, samples_sha256=sha256(variant/'samples.npz'),
                  arguments={'fit_radius': .2, 'mask_core': 0.}, pixel_unit_scale=.01)
    (variant/'imaging.json').write_text(json.dumps(report))
    return ImagingResults(path, 6170)


def test_summary_and_inclination_are_derived_from_actual_joint_draws(result):
    with np.load(result.path/'full_pixel/samples.npz') as arrays:
        expected = np.rad2deg(np.arccos(1-arrays['ellip']))
    np.testing.assert_allclose(result.inclination_samples(), expected)
    thick = result.inclination_samples(q0=.2)
    assert np.all(thick > expected)
    table = result.parameter_summary()
    assert list(table.index) == ['xc', 'yc', 'flux', 'r_eff', 'n', 'ellip', 'theta', 'sky_back', 'q', 'inclination_deg']
    q16, q50, q84 = np.quantile(expected, [.16, .5, .84])
    row = table.loc['inclination_deg']
    assert row['median'] == pytest.approx(q50)
    assert row['minus_1sigma'] == pytest.approx(q50-q16)
    assert row['plus_1sigma'] == pytest.approx(q84-q50)
    assert table.loc['sky_back', 'unit'] == 'normalized image value'


@pytest.mark.parametrize('q0', [np.nan, -1., 1., .9])
def test_inclination_rejects_invalid_or_incompatible_thickness(result, q0):
    with pytest.raises(ValueError):
        result.inclination_samples(q0=q0)


def _set_axis_ratios(result, values):
    """Replace the fixture's posterior while keeping its saved checksum valid."""
    path = result.path/'full_pixel/samples.npz'
    with np.load(path, allow_pickle=False) as arrays:
        samples = {name: arrays[name].copy() for name in arrays.files}
    samples['ellip'] = 1-np.asarray(values).reshape(samples['ellip'].shape)
    samples['q'] = 1-samples['ellip']
    np.savez(path, **samples)
    result.reports['full_pixel']['samples_sha256'] = sha256(path)


def test_inclination_comparison_preserves_asymmetric_intervals_for_both_geometries(result):
    # These masses make the q 16/50/84 percentiles exactly .25/.6/.95.
    # Inclination decreases with q, reversing the percentile order.
    _set_axis_ratios(result, [.25]*120 + [.6]*200 + [.95]*80)
    table = result.inclination_summary()
    assert table.index.tolist() == [0., .2]
    for q0 in (0., .2):
        low, median, high = np.degrees(np.arccos(np.sqrt(
            (np.array([.95, .6, .25])**2-q0**2)/(1-q0**2))))
        row = table.loc[q0]
        np.testing.assert_allclose(row[['median_deg', 'minus_1sigma_deg', 'plus_1sigma_deg']].astype(float),
                                   [median, median-low, high-median])
        assert row['minus_1sigma_deg'] != pytest.approx(row['plus_1sigma_deg'])
        assert row['incompatible_fraction'] == 0.
    assert table.loc[.2, 'median_deg'] > table.loc[0., 'median_deg']


@pytest.mark.parametrize('incompatible_q', [.1, .2-1e-8])
def test_inclination_comparison_reports_incompatible_draws_without_clipping_or_dropping(result, incompatible_q):
    _set_axis_ratios(result, [incompatible_q]*40 + [.6]*360)
    table = result.inclination_summary()
    thick = table.loc[.2]
    assert thick['incompatible_fraction'] == pytest.approx(.1)
    assert thick[['median_deg', 'minus_1sigma_deg', 'plus_1sigma_deg']].isna().all()
    assert 'incompat' in thick['status'].lower()
    assert table.loc[0., 'incompatible_fraction'] == 0.
    assert np.isfinite(table.loc[0., 'median_deg'])


@pytest.mark.parametrize('q0, q, expected_degrees', [(.25, 1., 0.), (.25, .25, 90.), (.2, .2, 90.)])
def test_inclination_summary_accepts_face_on_and_exact_thickness_limits(result, q0, q, expected_degrees):
    _set_axis_ratios(result, np.full(400, q))
    row = result.inclination_summary(q0_values=(q0,)).loc[q0]
    assert row['median_deg'] == pytest.approx(expected_degrees)
    assert row['minus_1sigma_deg'] == 0.
    assert row['plus_1sigma_deg'] == 0.
    assert row['incompatible_fraction'] == 0.
    np.testing.assert_allclose(result.inclination_samples(q0=q0), expected_degrees)


def test_builtin_corner_uses_native_defaults_with_physical_parameters_and_all_chains(result, monkeypatch):
    from pysersic.results import PySersicResults
    calls = []

    def capture(native, **kwargs):
        calls.append((native, kwargs))
        return plt.figure()

    monkeypatch.setattr(PySersicResults, 'corner', capture)
    before = sorted(str(p.relative_to(result.path)) for p in result.path.rglob('*'))
    fig = result.corner()
    try:
        native, kwargs = calls[0]
        assert len(calls) == 1
        assert set(native.idata.posterior.data_vars) == {'xc', 'yc', 'flux', 'r_eff', 'n', 'ellip', 'theta', 'sky_back'}
        assert native.idata.posterior['xc'].shape == (4, 100)
        with np.load(result.path/'full_pixel/samples.npz', allow_pickle=False) as saved:
            for name in native.idata.posterior.data_vars:
                np.testing.assert_array_equal(native.idata.posterior[name], saved[name])
        # Do not wrap the fitted theta chart across pi and split its distribution.
        assert np.median(native.idata.posterior['theta']) > np.pi
        # PySersic supplies its own labels, quantiles, titles, point rendering,
        # contours and colors. The wrapper must not replace those defaults.
        assert kwargs == {}
        assert before == sorted(str(p.relative_to(result.path)) for p in result.path.rglob('*'))
    finally:
        plt.close(fig)


def test_input_figure_works_before_fit_and_obeys_actual_aperture(result):
    fig = result.input_figure(fit_radius=.15)
    try:
        displayed = fig.axes[0].images[0].get_array()
        y, x = np.indices((21, 21))
        np.testing.assert_array_equal(np.ma.getmaskarray(displayed), np.hypot(x-10, y-10)*.03 > .15)
        with np.load(result.path/'cutout.npz') as arrays:
            np.testing.assert_allclose(displayed.compressed(), arrays['image'][~displayed.mask])
        assert not (result.path/'figures').exists()
    finally:
        plt.close(fig)


def test_input_figure_prefers_saved_fit_mask(result):
    with np.load(result.path/'cutout.npz') as arrays:
        image, error = arrays['image'], arrays['error']
    mask = np.zeros(image.shape, dtype=bool)
    mask[:8] = True
    path = result.path/'full_pixel/fit.npz'
    np.savez(path, image=image, error=error, mask=mask)
    result.reports['full_pixel']['fit_sha256'] = sha256(path)
    fig = result.input_figure(fit_radius=.15)
    try:
        np.testing.assert_array_equal(np.ma.getmaskarray(fig.axes[0].images[0].get_array()), mask)
    finally:
        plt.close(fig)
    result.reports['full_pixel']['fit_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='prediction checksum'):
        result.input_figure()


def test_review_rejects_changed_samples_or_inputs(result):
    result.reports['full_pixel']['samples_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='samples checksum'):
        result.parameter_summary()
    with (result.path/'cutout.npz').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='input checksum'):
        result.input_figure()


def test_inclination_figure_and_explicit_save(result):
    fig = result.inclination_figure(save=True)
    try:
        assert 'thin circular disk' in fig._suptitle.get_text()
        assert (result.path/'figures/full_pixel_inclination.png').exists()
        assert (result.path/'figures/full_pixel_inclination.pdf').exists()
    finally:
        plt.close(fig)


def test_wrong_source_is_rejected_before_plotting(result):
    with pytest.raises(ValueError, match='different source'):
        ImagingResults(result.path, 15665)
