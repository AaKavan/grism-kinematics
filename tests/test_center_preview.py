"""The editable marker must use source coordinates, not raw image coordinates."""
from hashlib import sha256
from pathlib import Path

from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml

from dingo.center_preview import preview_center


@pytest.mark.parametrize("source_id", [6170, 3700, 15665])
def test_preview_markers_roundtrip_to_selected_center_without_writing(
        source_id, repo, staged, grism_cal):
    import torch
    from dingo import grism, utils
    from dingo.kinematics import forward_dispersion_model

    source = Path(staged(f"ID{source_id}"))
    paths = [source / f"{name}.fits" for name in ("image", "R", "C")]
    paths.append(source / "config_kinematics.yaml")
    if any(not p.is_file() for p in paths):
        pytest.skip("Complete three-file source inputs are not staged")
    before = {p: sha256(p.read_bytes()).hexdigest() for p in paths}
    names_before = sorted(p.name for p in source.iterdir())
    config = yaml.safe_load(paths[-1].read_text())
    center = (40.25, 40.62)
    offsets = {"R": (1.0625, .2852), "C": (0., 0.)}
    fig = preview_center(source_id, data_dir=Path(repo) / "data", center=center,
                         r_offset=offsets["R"], zoom=12.)
    try:
        geometry = fig.dingo_center_preview
        assert geometry["source_center"] == list(center)
        assert "Nominal" in fig.axes[0].get_title()
        assert "no fit" in fig._suptitle.get_text()
        for index, pupil in enumerate(("R", "C"), start=1):
            ax = fig.axes[index]
            marker, = [line for line in ax.lines if line.get_gid() == "trial-kinematic-center"]
            observed = np.array([float(marker.get_xdata()[0]), float(marker.get_ydata()[0])])
            np.testing.assert_allclose(observed, geometry["points"][pupil])
            summary = config["summary"]
            _, spatial, dispersion = grism.load_nircam_wfss_model(
                pupil, config["image"][pupil]["module"], summary["filter"])
            forward = utils.get_grism_model_torch(
                spatial, dispersion, pupil, 1024, 1024, direction="forward", device="cpu")
            wavelength = torch.tensor([summary["lambda_rest"]], dtype=torch.float32) * (1 + summary["z"])
            x, y = [torch.tensor([v], dtype=torch.float32) for v in observed]
            sx, sy = forward_dispersion_model(x, y, wavelength, forward,
                                              dx=offsets[pupil][0], dy=offsets[pupil][1])
            cutout = geometry["provenance"]["cutouts"][pupil]
            np.testing.assert_allclose([float(sx) - cutout[0], float(sy) - cutout[1]],
                                       center, atol=2e-5)
            np.testing.assert_array_equal(ax.images[0].get_array(),
                                          fits.getdata(source / f"{pupil}.fits"))
        assert before == {p: sha256(p.read_bytes()).hexdigest() for p in paths}
        assert names_before == sorted(p.name for p in source.iterdir())
    finally:
        plt.close(fig)


def test_registration_moves_only_its_own_observed_marker(repo, staged, grism_cal):
    staged("ID6170")
    args = dict(source_id=6170, data_dir=Path(repo) / "data", center=(40., 40.))
    first = preview_center(**args, r_offset=(0., 0.))
    shifted = preview_center(**args, r_offset=(1.5, -.75))
    try:
        p1 = first.dingo_center_preview["points"]
        p2 = shifted.dingo_center_preview["points"]
        np.testing.assert_allclose(np.array(p2["R"]) - p1["R"], [-1.5, .75])
        assert p1["C"] == p2["C"]
        assert p1["image"] == p2["image"]
    finally:
        plt.close(first)
        plt.close(shifted)


def test_invalid_center_rejected_before_file_access(tmp_path):
    with pytest.raises(ValueError, match="center"):
        preview_center(6170, data_dir=tmp_path, center=(np.nan, 40.))


def test_geometry_mismatch_is_rejected(tmp_path):
    source = tmp_path / "ID6170"
    source.mkdir()
    config = {"summary": {"name": "ID6170", "r_fit": 40}}
    (source / "config_kinematics.yaml").write_text(yaml.safe_dump(config))
    for name in ("image", "R", "C"):
        fits.writeto(source / f"{name}.fits", np.zeros((41, 41)))
    with pytest.raises(ValueError, match="shapes"):
        preview_center(6170, data_dir=tmp_path, center=(40., 40.))
