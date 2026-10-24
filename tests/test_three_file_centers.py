"""Image markers must use the plotted sample and the rectifier's coordinates."""
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from dingo import three_file_centers as centers


@pytest.fixture
def example(tmp_path):
    names = ["velocity.x0_v", "velocity.y0_v", "image.R.dx", "image.R.dy"]
    pixels = dict(theta=np.array([43., 39., 2., -1.]), param_names=np.array(names),
                  image=np.zeros((81, 81)), observed_R=np.zeros((81, 81)),
                  observed_C=np.zeros((81, 81)))
    report = dict(source_id=15665, r_fit_pixels=40, redshift=1.2, pixel_scale_arcsec=.063,
                  fixed_params={"velocity.v_sys": 0., "image.C.dx": 0., "image.C.dy": 0.},
                  best_fit={"velocity.x0_v": 999., "velocity.y0_v": 999.})
    result = SimpleNamespace(source_id=15665, path=tmp_path, report=report, _final_bundle=None,
                             _prediction_pixels=lambda: pixels,
                             physical_scale=lambda: {"pc_per_pixel": 500.})
    return result, pixels


def test_saved_vector_not_report_median_or_best_fit(example):
    result, pixels = example
    params = centers.saved_parameters(result)
    assert params["velocity.x0_v"] == 43.
    assert params["velocity.y0_v"] == 39.
    assert params["image.C.dx"] == 0.


@pytest.mark.parametrize("corruption", ["duplicate", "nan", "missing_center", "wrong_shape"])
def test_reject_invalid_plot_vector(example, corruption):
    result, pixels = example
    if corruption == "duplicate":
        pixels["param_names"][1] = pixels["param_names"][0]
    elif corruption == "nan":
        pixels["theta"][0] = np.nan
    elif corruption == "missing_center":
        pixels["param_names"][0] = "missing"
    else:
        pixels["theta"] = pixels["theta"][None]
    with pytest.raises(ValueError):
        centers.saved_parameters(result)


def test_missing_config_keeps_honest_corrected_and_nominal_direct_marker(example):
    result, pixels = example
    pixels["image"] = np.zeros((101, 101))
    geometry = centers.center_geometry(result)
    assert geometry["points"]["rectified"] == [43., 39.]
    assert geometry["points"]["direct"] == [53., 49.]
    assert geometry["points"]["observed_R"] is None
    assert geometry["points"]["observed_C"] is None
    assert geometry["offset_pixels"] == [3., -1.]
    assert geometry["offset_pc"] == [1500., -500.]
    assert "unavailable" in geometry["projection_note"]
    assert "no astrometric registration" in geometry["direct_note"]


def test_embedded_geometry_requires_matching_prediction_and_source(example, monkeypatch):
    result, _ = example
    saved = centers.center_geometry(result)
    monkeypatch.setattr(centers, "_metadata", lambda _: {"kinematic_centers": saved})
    assert centers.center_geometry(result) is saved
    saved["params"]["velocity.x0_v"] = 40.
    with pytest.raises(ValueError, match="parameter vector"):
        centers.center_geometry(result)


def _config():
    return dict(summary=dict(name="ID15665", z=1.2, lambda_rest=1.875, r_fit=40,
                             filter="F444W", mode="kinematics", unit="pix"),
                image={p: dict(module="A", dx=0., dy=0.) for p in ("R", "C")})


def test_incompatible_source_config_is_not_used(example):
    result, _ = example
    config = _config()
    config["summary"]["name"] = "ID3700"
    (result.path / "run_config.yaml").write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="metadata disagree"):
        centers.center_geometry(result)


def test_partial_legacy_config_leaves_raw_projection_unavailable(example):
    result, _ = example
    (result.path / "run_config.yaml").write_text(yaml.safe_dump({"mcmc": {"seed": 42}}))
    geometry = centers.center_geometry(result)
    assert geometry["points"]["observed_R"] is None
    assert "lacks" in geometry["projection_note"]


@pytest.mark.parametrize("systemic_velocity", [0., 120.])
@pytest.mark.parametrize("explicit_cutouts", [False, True])
def test_projected_center_maps_back_through_actual_rectifier(example, grism_cal,
                                                            systemic_velocity, explicit_cutouts):
    """Independent forward evaluation catches sign, wavelength, and origin errors."""
    import torch
    from dingo import grism, utils
    from dingo.kinematics import forward_dispersion_model

    result, _ = example
    config = _config()
    params = centers.saved_parameters(result)
    params["velocity.v_sys"] = systemic_velocity
    if explicit_cutouts:
        config["image"]["R"]["cutout"] = (-119, 21, 81, 81)
        config["image"]["C"]["cutout"] = (44, -74, 81, 81)
    projected, provenance = centers._project(config, params)
    lambda0 = torch.tensor([config["summary"]["lambda_rest"]]) * (1 + config["summary"]["z"])
    wavelength = lambda0 * (1 + systemic_velocity / 299792.458)
    for pupil in ("R", "C"):
        _, sp, dp = grism.load_nircam_wfss_model(pupil, "A", "F444W")
        forward = utils.get_grism_model_torch(sp, dp, pupil, 1024, 1024,
                                             direction="forward", device="cpu")
        xg, yg = [torch.tensor([v], dtype=torch.float32) for v in projected[f"observed_{pupil}"]]
        sx, sy = forward_dispersion_model(xg, yg, wavelength, forward,
            dx=params[f"image.{pupil}.dx"], dy=params[f"image.{pupil}.dy"])
        cutout = provenance["cutouts"][pupil]
        np.testing.assert_allclose([float(sx)-cutout[0], float(sy)-cutout[1]],
                                   [params["velocity.x0_v"], params["velocity.y0_v"]], atol=2e-5)
        if explicit_cutouts:
            assert cutout == list(config["image"][pupil]["cutout"])


def test_changed_saved_calibration_is_not_silently_projected(example, tmp_path, monkeypatch):
    from dingo import grism
    from dingo.real_workflow import sha256
    _, _ = example
    calibration = tmp_path / "trace.dat"
    calibration.write_text("original")
    expected = sha256(calibration)
    calibration.write_text("changed")
    monkeypatch.setattr(grism, "get_grism_cal_dir", lambda: tmp_path)
    identity = {"search_identity": {"parent_identity": {"calibration": {"trace.dat": expected}}}}
    with pytest.raises(ValueError, match="calibration changed"):
        centers._project(_config(), {}, identity)
