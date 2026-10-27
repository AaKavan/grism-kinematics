"""Coordinate and unsupported-format checks for user-drawn input masks."""

import numpy as np
import pytest

from dingo.input_masks import read_ds9_exclusion


def test_ds9_one_based_pixel_centers(tmp_path):
    path = tmp_path / "C_mask.reg"
    path.write_text("# Region file format: DS9 version 4.1\nimage\ncircle(3,4,0.25)\n")
    mask = read_ds9_exclusion(path, (7, 8))
    assert np.argwhere(mask).tolist() == [[3, 2]]


def test_physical_coordinates_apply_fits_ltm_ltv(tmp_path):
    path = tmp_path / "C_mask.reg"
    path.write_text("physical; circle(7,12,0.1)\n")
    header = {"LTM1_1": 2., "LTM2_2": 2., "LTV1": -10., "LTV2": -20.}
    mask = read_ds9_exclusion(path, (8, 8), header)
    # Image = LTM * physical + LTV = (4, 4), which is zero-based (3, 3).
    assert np.argwhere(mask).tolist() == [[3, 3]]


def test_polygon_includes_boundary_and_combines_separate_regions(tmp_path):
    path = tmp_path / "C_mask.reg"
    path.write_text("image\npolygon(2,2,4,2,4,4,2,4)\ncircle(7,7,0.1)\n")
    mask = read_ds9_exclusion(path, (9, 9))
    assert mask.sum() == 10
    assert mask[1:4, 1:4].all()
    assert mask[6, 6]


@pytest.mark.parametrize("region", [
    "fk5\ncircle(47.7,25.6,0.1)",
    "image\n-circle(3,3,1)",
    "image\ncircle(3,3,1) # include=0",
    "image\ncircle(3,3,nan)",
    "image\ncircle(1000,1000,1)",
])
def test_invalid_or_unsupported_region_is_not_silently_ignored(tmp_path, region):
    path = tmp_path / "C_mask.reg"
    path.write_text(region)
    with pytest.raises(ValueError):
        read_ds9_exclusion(path, (9, 9))


def _background_fill_files(tmp_path):
    from astropy.io import fits

    region = tmp_path / "C_mask.reg"
    region.write_text("image\ncircle(16,16,3)\n")
    data = np.full((32, 32), 3., dtype=np.float32)
    mask = read_ds9_exclusion(region, data.shape)
    data[mask] = 100.
    data[0, 0] = np.nan
    primary = fits.PrimaryHDU(data)
    primary.header["OBJECT"] = "test galaxy"
    extra = fits.ImageHDU(np.arange(16).reshape(4, 4), name="OTHER")
    path = tmp_path / "C.fits"
    fits.HDUList([primary, extra]).writeto(path, checksum=True)
    return path, region, data, mask


def test_background_fill_preserves_outside_backup_extensions_and_is_idempotent(tmp_path):
    from astropy.io import fits
    from dingo.input_masks import replace_region_with_background

    path, region, original, mask = _background_fill_files(tmp_path)
    original_bytes = path.read_bytes()
    report = replace_region_with_background(path, region, border_width=4)
    backup = tmp_path / "C_original_before_region_fill.fits"
    assert backup.read_bytes() == original_bytes
    with fits.open(path, checksum=True) as hdus:
        np.testing.assert_array_equal(hdus[0].data[~mask], original[~mask])
        assert np.all(hdus[0].data[mask] == 3.)
        np.testing.assert_array_equal(hdus[1].data, np.arange(16).reshape(4, 4))
        assert hdus[0].header["OBJECT"] == "test galaxy"
        assert all(hdu.verify_checksum() == 1 for hdu in hdus)
    assert report["pixels_replaced"] == int(mask.sum())
    assert report["background_median"] == 3.
    assert report["background_sigma"] == 0.
    filled_bytes = path.read_bytes()
    report_bytes = (tmp_path / "C_background_fill.json").read_bytes()
    repeated = replace_region_with_background(path, region, border_width=4)
    assert repeated["already_applied"]
    assert path.read_bytes() == filled_bytes
    assert (tmp_path / "C_background_fill.json").read_bytes() == report_bytes
    assert backup.read_bytes() == original_bytes


def test_background_fill_refuses_unrelated_working_file_changes(tmp_path):
    from astropy.io import fits
    from dingo.input_masks import replace_region_with_background

    path, region, _, _ = _background_fill_files(tmp_path)
    replace_region_with_background(path, region, border_width=4)
    with fits.open(path, mode="update") as hdus:
        hdus[0].data[1, 1] = 45.
    changed_bytes = path.read_bytes()
    with pytest.raises(ValueError, match="changed independently"):
        replace_region_with_background(path, region, border_width=4)
    assert path.read_bytes() == changed_bytes


def test_background_fill_changed_region_starts_from_original(tmp_path):
    from astropy.io import fits
    from dingo.input_masks import replace_region_with_background

    path, region, original, old_mask = _background_fill_files(tmp_path)
    first = replace_region_with_background(path, region, border_width=4)
    region.write_text("image\ncircle(25,25,2)\n")
    current_mask = read_ds9_exclusion(region, original.shape)
    second = replace_region_with_background(path, region, border_width=4)
    actual = fits.getdata(path)
    np.testing.assert_array_equal(actual[~current_mask], original[~current_mask])
    assert np.all(actual[old_mask] == 100.)
    assert second["original_sha256"] == first["original_sha256"]
    assert np.all(actual[current_mask] == 3.)
