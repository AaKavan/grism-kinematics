"""Check that excluded input intensities cannot affect the common likelihood."""

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from dingo.kinematics import bilinear_interpolte_intensity_torch
from dingo.masked_comparison import InputMaskedFitter, excluded_scatter_envelope
from dingo.three_file_fit import build_config


def test_envelope_contains_all_bilinear_destinations():
    excluded = np.zeros((12, 14), dtype=bool)
    excluded[2, 3] = True
    excluded[9, 12] = True
    x_bounds, y_bounds = (-2.3, 3.8), (-4.1, 2.2)
    envelope = excluded_scatter_envelope(excluded, x_bounds, y_bounds, excluded.shape)
    y, x = torch.meshgrid(torch.arange(12), torch.arange(14), indexing="ij")
    values = torch.as_tensor(excluded.astype(np.float32))
    # Include the exact extrema and translations that straddle integer pixels.
    for dx in [-2.3, -2., -.001, 0., .001, 3., 3.8]:
        for dy in [-4.1, -4., -.001, 0., .001, 2., 2.2]:
            scattered = bilinear_interpolte_intensity_torch(
                x + dx, y + dy, values, (0, 0, 14, 12)).numpy()
            assert not np.any((scattered > 0) & ~envelope)


@pytest.fixture
def masked_fitter(tmp_path, staged, grism_cal):
    source = Path(staged("ID6170"))
    config = build_config(6170, source.parent, source / "config_kinematics.yaml", tmp_path,
                          inclination_deg=76., inclination_sigma_deg=5., device="cpu",
                          velocity_parameter="V_rot")
    config["input_masks"] = {}
    for pupil, region in (("C", "circle(47.698245,25.656347,6.4480075)"),
                           ("R", "circle(10,65,1.5)")):
        path = tmp_path / f"{pupil}_mask.reg"
        path.write_text("physical\n" + region + "\n")
        config["input_masks"][pupil] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    config["mcmc"]["noise"].update(sigma_R=.0034, sigma_C=.0036)
    config["velocity"].update(V_rot=0., R_v=10.)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    fitter = InputMaskedFitter(str(config_path), device=torch.device("cpu"))
    fitter.prepare_mcmc()
    return fitter


def _trial_points(fitter):
    points = np.tile(fitter.prepare_mcmc()["theta0"], (5, 1))
    indices = {name: index for index, name in enumerate(fitter.mcmc_param_names)}
    values = [
        (0., 10., 40., 40., 1., 1.2, 0., 0.),
        (200., 10., 38., 42., 4.7, 1.3, 3., -2.),
        (400., 20., 43., 37., 2.5, 1.2, -5., 4.),
        (800., 35., 35., 36., 5.5, 1.25, 9.9, -9.9),
        (999., 39., 46., 44., .5, 1.18, -9.9, 9.9),
    ]
    names = ("velocity.V_rot", "velocity.R_v", "velocity.x0_v", "velocity.y0_v",
             "velocity.theta_v", "velocity.inc_v", "image.R.dx", "image.R.dy")
    for point, row in zip(points, values):
        for name, value in zip(names, row):
            point[indices[name]] = value
    return points


@pytest.mark.data
def test_masked_input_values_do_not_change_scores(masked_fitter):
    fitter = masked_fitter
    points = _trial_points(fitter)
    before = fitter.log_probability_batch(points)
    assert np.isfinite(before).all()
    for pupil, mask in fitter.input_excluded.items():
        data = getattr(fitter, f"true_grism_{pupil}")
        data[mask] = torch.linspace(-1.e5, 1.e5, int(mask.sum()), dtype=data.dtype)
    after = fitter.log_probability_batch(points)
    # This checks actual likelihood isolation, not merely the displayed mask.
    np.testing.assert_array_equal(before, after)


@pytest.mark.data
def test_scalar_batch_match_and_remapped_masks_stay_excluded(masked_fitter):
    fitter = masked_fitter
    points = _trial_points(fitter)
    batch = fitter.log_probability_batch(points)
    scalar = []
    for point in points:
        scalar.append(fitter.log_probability(point))
        assert np.isfinite(scalar[-1])
        for pupil, mask in fitter.input_excluded.items():
            scattered = bilinear_interpolte_intensity_torch(
                getattr(fitter, f"x_{pupil}"), getattr(fitter, f"y_{pupil}"),
                mask.to(torch.float32), getattr(fitter, f"cutout_{pupil}"))
            assert not bool(((scattered > 0) & fitter.mcmc_mask).any())
    np.testing.assert_allclose(batch, scalar, rtol=0., atol=.02)


@pytest.mark.data
def test_stale_comparison_that_includes_masked_support_is_rejected(masked_fitter):
    fitter = masked_fitter
    fitter.validate_input_mask_support()
    bad = torch.nonzero(fitter.input_comparison_excluded)[0]
    fitter.mcmc_mask[bad[0], bad[1]] = True
    with pytest.raises(ValueError, match="saved comparison"):
        fitter.validate_input_mask_support()
