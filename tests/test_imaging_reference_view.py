"""The imaging reference must show measured centers in the correct pixel frame."""
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

from dingo import morphology
from dingo.real_workflow import sha256, write_json
from dingo.rotation_imaging import ImagingResults


@pytest.fixture
def result(tmp_path):
    variant = tmp_path/'full_pixel'
    variant.mkdir()
    yy, xx = np.indices((11, 13))
    image = np.exp(-((xx-4.2)**2+(yy-7.1)**2)/5)
    model = .98*image
    np.savez(variant/'fit.npz', image=image, model=model, mask=np.zeros_like(image, bool))
    x = np.linspace(4., 4.4, 40).reshape(4, 10)
    y = np.linspace(6.8, 7.4, 40).reshape(4, 10)
    values = dict(xc=x, yc=y, flux=np.full_like(x, 10.), r_eff=np.full_like(x, 2.),
                  n=np.ones_like(x), ellip=np.full_like(x, .3),
                  theta=np.full_like(x, .4), sky_back=np.zeros_like(x))
    np.savez(variant/'samples.npz', **values)
    write_json(tmp_path/'imaging_variants.json', dict(source_id=6170, variants=['full_pixel']))
    write_json(tmp_path/'input_provenance.json', dict(source_id=6170, input_sha256={}))
    write_json(variant/'imaging.json', dict(source_id=6170, completed=True,
        fit_sha256=sha256(variant/'fit.npz'), samples_sha256=sha256(variant/'samples.npz')))
    return ImagingResults(tmp_path, 6170, variants=('full_pixel',))


def test_two_panels_share_flux_scale_and_annotate_posterior_center(result):
    before = {p.name: p.read_bytes() for p in (result.path/'full_pixel').iterdir()}
    figure = result.center_figure()
    try:
        assert len(figure.axes) == 2
        assert figure.axes[0].images[0].norm is figure.axes[1].images[0].norm
        with np.load(result.path/'full_pixel/fit.npz') as saved:
            np.testing.assert_array_equal(figure.axes[0].images[0].get_array(), saved['image'])
            np.testing.assert_array_equal(figure.axes[1].images[0].get_array(), saved['model'])
        for axis in figure.axes:
            marker = next(line for line in axis.lines if line.get_gid() == 'pysersic-center')
            np.testing.assert_allclose(marker.get_xdata(), [4.2])
            np.testing.assert_allclose(marker.get_ydata(), [7.1])
            text = axis.texts[0].get_text()
            assert 'x_c=4.200' in text and 'y_c=7.100' in text
            assert '0.136' in text and '0.204' in text
            assert 'zero-based' in axis.get_xlabel()
            assert 'residual' not in axis.get_title().lower()
            assert axis.images[0].get_cmap().name == 'gist_heat'
        assert 'not the R/C-stack' in figure._suptitle.get_text()
    finally:
        plt.close(figure)
    assert not (result.path/'figures').exists()
    assert before == {p.name: p.read_bytes() for p in (result.path/'full_pixel').iterdir()}


def test_center_values_can_be_hidden_without_removing_the_center_marker(result):
    figure = result.center_figure(annotate_center=False)
    try:
        for axis in figure.axes:
            assert not axis.texts
            marker = next(line for line in axis.lines if line.get_gid() == 'pysersic-center')
            np.testing.assert_allclose(marker.get_xdata(), [4.2])
            np.testing.assert_allclose(marker.get_ydata(), [7.1])
        assert 'image and model' in figure._suptitle.get_text()
        assert 'intervals' not in figure._suptitle.get_text()
    finally:
        plt.close(figure)


@pytest.mark.parametrize('filename,message', [('fit.npz', 'prediction checksum'),
                                            ('samples.npz', 'samples checksum')])
def test_changed_saved_data_are_not_silently_plotted(result, filename, message):
    with (result.path/'full_pixel'/filename).open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match=message):
        result.center_figure()


@pytest.mark.parametrize('reference,expected', [(False, 'fit'), (True, 'center')])
def test_review_selects_requested_panel_and_preserves_native_corner(tmp_path, reference, expected):
    (tmp_path/'full_pixel').mkdir()
    (tmp_path/'full_pixel/samples.npz').write_bytes(b'posterior exists')
    calls = []
    saves = []

    def make(name):
        calls.append(name)
        return plt.figure()

    result = SimpleNamespace(path=tmp_path, figure=lambda: make('fit'),
        center_figure=lambda **kwargs: make('center'), corner=lambda variant: make('corner'),
        _save_review_figure=lambda fig, name, save: saves.append((name, save)))
    errors = morphology._review(result, display=False, center_reference=reference)
    assert errors == {}
    assert calls == [expected, 'corner']
    assert saves == [(f'full_pixel_{expected}', True), ('full_pixel_corner', True)]
    assert (tmp_path/'corner.png').read_bytes().startswith(b'\x89PNG')
    assert (tmp_path/'corner.pdf').read_bytes().startswith(b'%PDF')


def test_hidden_centers_are_removed_only_from_displayed_parameter_table(result, monkeypatch):
    import IPython.display
    import pandas as pd

    displayed = []
    monkeypatch.setattr(IPython.display, 'display', displayed.append)
    monkeypatch.setattr(result, 'corner', lambda variant: plt.figure())
    monkeypatch.setattr(result, '_save_review_figure', lambda *args: None)
    parameter_file = result.path/'parameters.txt'
    parameter_file.write_text('Complete saved report including xc and yc\n')
    before = (result.path/'full_pixel/samples.npz').read_bytes()
    errors = morphology._review(result, display=True, center_reference=True,
                                show_center_values=False)
    assert errors == {}
    table = next(item for item in displayed
                 if isinstance(item, pd.DataFrame) and 'r_eff' in item.index)
    assert 'xc' not in table.index and 'yc' not in table.index
    assert 'xc' in result.parameter_summary().index
    assert 'yc' in result.parameter_summary().index
    assert (result.path/'full_pixel/samples.npz').read_bytes() == before
    assert parameter_file.read_text() == 'Complete saved report including xc and yc\n'
