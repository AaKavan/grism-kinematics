"""Imaging reference coordinates must retain scale, rotation and paired draws."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from astropy.io import fits
from astropy.wcs import WCS

from dingo.imaging_reference_coordinates import center_reference, find_native_reference


def make_wcs(scale, angle, crpix):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [64., -24.]
    wcs.wcs.crpix = crpix
    c, s = np.cos(angle), np.sin(angle)
    wcs.wcs.cd = scale/3600*np.array([[-c, s], [s, c]])
    return wcs


@pytest.fixture
def reference_setup(tmp_path):
    config = dict(summary=dict(name="ID6170", ra=64., dec=-24., filter="F444W", r_fit=40),
                  image=dict(R=dict(module="B"), C=dict(module="B")))
    path = tmp_path/"config.yaml"
    path.write_text(yaml.safe_dump(config))
    imaging = make_wcs(.03, 0., [51., 51.])
    np.savez(tmp_path/"cutout.npz", wcs_header=imaging.to_header().tostring())
    samples = dict(xc=np.array([[50., 51., 52.], [50.5, 51.5, 52.5]]),
                   yc=np.array([[50., 52., 54.], [51., 53., 55.]]))
    result = SimpleNamespace(path=tmp_path, source_id=6170,
                             _physical_samples=lambda variant: samples)
    return result, path, imaging, samples


def write_native(path, wcs, *, module="B", band="F444W"):
    path.parent.mkdir(parents=True, exist_ok=True)
    primary = fits.PrimaryHDU()
    primary.header["MODULE"] = module
    primary.header["FILTER"] = band
    science = fits.ImageHDU(np.zeros((101, 101)), header=wcs.to_header(), name="SCI")
    fits.HDUList([primary, science]).writeto(path)


def test_rotated_detector_uses_sky_offsets_and_paired_draws(reference_setup, tmp_path):
    result, config, imaging, samples = reference_setup
    native = make_wcs(.06, np.pi/2, [41.3, 42.1])
    path = tmp_path/"native.fits"
    write_native(path, native)
    table = center_reference(result, config_path=config, native_reference=path)
    # A 90-degree detector rotation swaps axes and changes the relevant sign;
    # the detector's fractional CRPIX must cancel from the offset reference.
    expected_x = 40.+.5*(samples["yc"].ravel()-50.)
    expected_y = 40.-.5*(samples["xc"].ravel()-50.)
    for name, expected in [("nominal_dingo_x0", expected_x), ("nominal_dingo_y0", expected_y)]:
        np.testing.assert_allclose(table.loc[name, ["p16", "median", "p84"]].astype(float),
                                   np.percentile(expected, [16, 50, 84]), atol=1e-7)
    assert table.loc["image_xc", "median"] == np.median(samples["xc"])
    assert table.loc["east_offset", "median"] < 0.
    assert table.loc["north_offset", "median"] > 0.
    assert "Approximate" in table.loc["nominal_dingo_x0", "meaning"]
    assert any("registration uncertainty" in note for note in table.attrs["notes"])


def test_missing_cutout_wcs_returns_only_native_imaging_pixels(reference_setup):
    result, config, _, _ = reference_setup
    np.savez(result.path/"cutout.npz", image=np.zeros((101, 101)))
    table = center_reference(result, config_path=config)
    assert table.index.tolist() == ["image_xc", "image_yc"]
    assert any("no WCS" in note for note in table.attrs["notes"])


def test_noncelestial_cutout_wcs_does_not_invent_sky_coordinates(reference_setup):
    result, config, _, _ = reference_setup
    np.savez(result.path/"cutout.npz", wcs_header=fits.Header().tostring())
    table = center_reference(result, config_path=config)
    assert "ra" not in table.index
    assert any("not celestial" in note for note in table.attrs["notes"])


def test_native_reference_discovery_checks_module_band_and_coverage(reference_setup, tmp_path):
    _, config, _, _ = reference_setup
    directory = tmp_path/"bundle/direct/F444W_imaging/obsnum116"
    good = directory/"c_nrcblong_cal_1of_a3001_crf.fits"
    write_native(directory/"a_nrcblong_cal_1of_a3001_crf.fits", make_wcs(.06, 0., [51., 51.]), band="F356W")
    write_native(directory/"b_nrcblong_cal_1of_a3001_crf.fits", make_wcs(.06, 0., [5., 5.]))
    write_native(good, make_wcs(.06, .7, [51., 51.]))
    assert find_native_reference(config, tmp_path/"bundle") == good
    assert find_native_reference(config, tmp_path/"missing") is None


def test_rejects_mismatched_source_and_invalid_samples(reference_setup):
    result, config, _, samples = reference_setup
    result.source_id = 3700
    with pytest.raises(ValueError, match="different sources"):
        center_reference(result, config_path=config)
    result.source_id = 6170
    samples["yc"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite paired"):
        center_reference(result, config_path=config)
