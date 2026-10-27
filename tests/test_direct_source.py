"""Real-data direct prediction must retain the tested injection conventions."""
import numpy as np
import pytest
import torch
from astropy.io import fits

from dingo.direct_source import SourceModel
from dingo.simple_simulation import make_mock


@pytest.fixture
def mock(tmp_path, grism_cal):
    return make_mock(tmp_path / "inputs")


def _model(mock, **kwargs):
    return SourceModel(mock.source_id, data_dir=mock.data_dir,
                       center=(mock.truth["velocity.x0_v"], mock.truth["velocity.y0_v"]),
                       inclination_mean=60., inclination_sigma=5.,
                       **kwargs)


def _point(mock):
    return np.array([mock.truth[k] for k in SourceModel.param_names])


def test_exact_direct_prediction_matches_independent_saved_injection(mock):
    model = _model(mock, noise_sigma=.02)
    with np.load(mock.directory / "truth.npz") as saved:
        emission = saved["image"]
        for pupil, predicted in zip(("R", "C"), model.predict(_point(mock), emission)):
            np.testing.assert_array_equal(predicted.numpy(), saved[f"noiseless_{pupil}"])
    assert model.param_names[-1] == "velocity.inc_v"
    assert not hasattr(model, "truth_theta")
    assert not hasattr(model, "mock")
    assert not hasattr(model, "emission")
    with pytest.raises(TypeError, match="emission"):
        model.predict(_point(mock))
    model.validate_inputs()


def test_separate_noise_weights_and_nonfinite_mask_match_original_pixels(mock):
    source = mock.data_dir / f"ID{mock.source_id}"
    r = fits.getdata(source / "R.fits")
    r[0, 0] = np.nan
    fits.writeto(source / "R.fits", r, overwrite=True)
    model = _model(mock, noise_sigma=(.02, .04))
    assert not bool(model.valid_by_pupil["R"][0, 0])
    assert bool(torch.isfinite(model.data["R"]).all())
    with np.load(mock.directory / "truth.npz") as saved:
        emission = saved["image"]
    point = _point(mock)
    predictions = model.predict(point, emission)
    expected = 0.
    for pupil, predicted in zip(("R", "C"), predictions):
        valid = model.valid_by_pupil[pupil]
        delta = ((predicted - model.data[pupil]) / model.sigma_by_pupil[pupil])[valid]
        expected += float(torch.sum(delta.double()**2))
    assert float(model.chi2(point, emission=emission)) == pytest.approx(expected, abs=1e-9)
    varied = torch.tensor(point + [3., .1, .01, .02, -.01, .01],
                          dtype=torch.float32, requires_grad=True)
    score = model.chi2(varied, emission=emission) + model.prior_penalty(varied)
    gradient = torch.autograd.grad(score, varied)[0]
    assert torch.isfinite(gradient).all()
    assert abs(float(gradient[-1])) > 0.


def test_only_inclination_gets_gaussian_penalty_and_remains_fitted(mock):
    model = _model(mock, noise_sigma=.02)
    point = _point(mock)
    assert model.log_prior(point) == pytest.approx(0., abs=1e-12)
    point[-1] += np.deg2rad(5.)
    assert model.log_prior(point) == pytest.approx(-.5)
    assert float(model.prior_penalty(point)) == pytest.approx(.5, abs=2e-6)
    point[3] = 8.
    assert model.log_prior(point) == pytest.approx(-.5)
    point[-1] = np.deg2rad(86.)
    assert model.log_prior(point) == -np.inf


def test_batched_chi2_matches_scalar_and_display_image_does_not_predict(mock):
    model = _model(mock, noise_sigma=(.03, .04))
    with np.load(mock.directory / "truth.npz") as saved:
        emission = saved["image"]
    points = np.stack([_point(mock), _point(mock) + [25., .3, .03, .1, -.1, .02]])
    batch = model.chi2(points, emission=emission).numpy()
    scalar = np.array([float(model.chi2(p, emission=emission)) for p in points])
    np.testing.assert_allclose(batch, scalar, rtol=0, atol=1e-7)
    before = [a.numpy().copy() for a in model.predict(points[0], emission)]
    source = mock.data_dir / f"ID{mock.source_id}"
    fits.writeto(source / "image.fits", np.full((31, 31), 1e8), overwrite=True)
    with pytest.raises(ValueError, match="image.fits changed"):
        model.validate_inputs()
    updated = _model(mock, noise_sigma=(.03, .04))
    assert updated.image.shape == (31, 31)
    for a, b in zip(before, updated.predict(points[0], emission)):
        np.testing.assert_array_equal(a, b.numpy())


def test_noise_estimation_and_input_validation(mock):
    model = _model(mock)
    assert model.noise_source == "estimated_from_inputs"
    assert all(.015 < x < .025 for x in model.sigma_by_pupil.values())
    with pytest.raises(ValueError, match="no saved noiseless"):
        model.chi2(_point(mock), emission=np.zeros(model.shape), noisy=False)
    with pytest.raises(ValueError, match="noise_sigma"):
        _model(mock, noise_sigma=True)
    with pytest.raises(ValueError, match="positive"):
        _model(mock, noise_sigma=(0., .02))


def test_source_config_identity_and_center_validation(mock):
    with pytest.raises(ValueError, match="center"):
        SourceModel(mock.source_id, data_dir=mock.data_dir, center=(500., 40.))
    with pytest.raises(ValueError, match="inclination_sigma"):
        SourceModel(mock.source_id, data_dir=mock.data_dir, inclination_sigma=0.)
    with pytest.raises(ValueError, match="no greater than 1000"):
        SourceModel(mock.source_id, data_dir=mock.data_dir, max_velocity=1500.)


def test_fixed_rdy_is_a_constant_in_scalar_and_batch_predictions(repo, staged, grism_cal):
    """Read existing inputs and check forward coordinates; no fitter is run."""
    from pathlib import Path
    staged("ID6170")
    model = SourceModel(6170, data_dir=Path(repo) / "data", fixed_rdy=.271)
    assert model.fixed_parameters["image.R.dy"] == .271
    assert model.configuration["direct_model"]["fixed_rdy"] == .271
    point = np.array([189., 3.2, np.deg2rad(306.5), 1.4, .271, np.deg2rad(80.)])
    assert np.isfinite(model.log_prior(point.astype(np.float32)))
    changed = point.copy()
    changed[4] = -.711
    assert model.log_prior(changed) == -np.inf
    coordinates = model.mapping_coordinates(np.stack([point, changed]))
    for pupil in ("R", "C"):
        for coordinate in coordinates[pupil]:
            np.testing.assert_array_equal(coordinate[0].numpy(), coordinate[1].numpy())
    emission = np.full(model.shape, .005)
    variable = torch.tensor(point, dtype=torch.float32, requires_grad=True)
    gradient = torch.autograd.grad(model.chi2(variable, emission=emission), variable)[0]
    assert float(gradient[4]) == 0.
    assert abs(float(gradient[3])) > 0.
    free = SourceModel(6170, data_dir=Path(repo) / "data")
    assert free.fixed_rdy is None
    assert "fixed_rdy" not in free.configuration["direct_model"]


@pytest.mark.parametrize("bad", [True, np.bool_(True), np.nan, np.inf, -10.01, 10.01, [1., 2.]])
def test_invalid_fixed_rdy_is_rejected_before_input_loading(bad):
    with pytest.raises(ValueError, match="fixed_rdy"):
        SourceModel(6170, data_dir="/unused-inputs", fixed_rdy=bad)
