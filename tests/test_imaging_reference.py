"""The imaging-only notebook wrapper must preserve source and frame identity."""
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from dingo import imaging_reference as reference


@pytest.mark.parametrize("source_id", [6170, 3700])
def test_reference_uses_calibrated_mosaics_and_exports_wcs(tmp_path, monkeypatch, source_id):
    data = tmp_path / "data" / f"ID{source_id}"
    data.mkdir(parents=True)
    config = data / "config_kinematics.yaml"
    config.write_text(f"summary:\n  name: ID{source_id}\n  ra: 64.0\n  dec: -24.0\n"
                      "  filter: F444W\n  r_fit: 40\nimage:\n  R:\n    module: B\n")
    legacy = data / "image.fits"
    legacy.write_bytes(b"original display-only file: must remain untouched")
    output = tmp_path / "output"
    saved = output / str(source_id) / "morph_fits/F150W_reference/fit_001"
    (saved / "full_pixel").mkdir(parents=True)
    (saved / "full_pixel/samples.npz").write_bytes(b"sample-existence marker")
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [3., 3.]
    wcs.wcs.crval = [64., -24.]
    wcs.wcs.cdelt = [-.03 / 3600, .03 / 3600]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    pixels = np.arange(25, dtype=float).reshape(5, 5)
    np.savez(saved / "cutout.npz", image=pixels, error=np.ones((5, 5)),
             wcs_header=wcs.to_header().tostring())
    result = SimpleNamespace(path=saved, source_id=source_id,
        _check_inputs=lambda report: {"unit": "MJy/sr"}, _report=lambda name: {},
        _physical_samples=lambda name: {"xc": np.full((4, 10), 2.1),
                                        "yc": np.full((4, 10), 2.2)})
    calls = []

    def fit(sid, **options):
        calls.append((sid, options))
        return result

    monkeypatch.setattr(reference, "fit_morphology", fit)
    observed = reference.fit_reference(source_id, data_dir=data.parent,
        output_root=output, edr_dir=tmp_path / "survey", bundle=tmp_path / "bundle",
        device="cpu", display=False)
    assert observed is result
    sid, options = calls[0]
    assert sid == source_id
    assert options["config_path"] == config
    assert options["science"].name == "4750_F150W_v05_sci.fits"
    assert options["error"].name == "4750_F150W_v05_err.fits"
    assert options["center_reference"] is True
    assert options["display"] is False
    with fits.open(saved / "F150W_image.fits") as hdus:
        np.testing.assert_array_equal(hdus[0].data, pixels)
        assert hdus[0].header["SOURCEID"] == source_id
        assert hdus[0].header["BUNIT"] == "MJy/sr"
        assert WCS(hdus[0].header).has_celestial
    text = (saved / f"pysersic_{source_id}_center.cat").read_text()
    assert "image_xc" in text and "Do not copy" in text
    assert "nominal_dingo_x0" not in text  # No invented mapping without a native WCS.
    assert legacy.read_bytes() == b"original display-only file: must remain untouched"
