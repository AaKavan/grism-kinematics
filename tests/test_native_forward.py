"""Physical invariants and independent checks of the native-pixel prototype."""
import numpy as np
import json
import os
from pathlib import Path
import pytest
import torch
from astropy.io import fits

from dingo.exposures import TraceCalibration, _read_window, native_valid_mask, load_native_target
from dingo.native_forward import (NativeLineProjector, NativeGaussianLikelihood,
                                  projection_coefficients, projected_arctan_velocity)


@pytest.mark.parametrize("pupil", ["R", "C"])
def test_full_polynomial_geometry_matches_independent_calibration(pupil):
    rng = np.random.default_rng(84)
    # All spatial and wavelength terms nonzero, with detector-scale coefficients.
    trace = rng.normal(size=(3, 10)) * np.array([1, 1e-3, 1e-3, 1e-6, 1e-6, 1e-6, 1e-9, 1e-9, 1e-9, 1e-9])
    trace *= np.array([1, 1e-2, 1e-5])[:, None]
    wave = np.array([10., .02, -.01, 2e-6, 3e-6, -1e-6, 1000., .01, .02, 1e-6, 2e-6, 3e-6, 2., .001, -.002, .03])
    cal = TraceCalibration(pupil, "B", trace.ravel(), wave, {})
    xy = rng.uniform(100, 1900, (12, 2))
    origin = (320, 710)
    coeff = projection_coefficients(cal, xy, 4.1, origin)
    for lam in (3.85, 4.1, 4.8):
        got = sum(coeff[..., k] * (lam - 4.1)**k for k in range(7))
        expected = np.stack(cal.project(xy[:, 0], xy[:, 1], lam), axis=-1) - origin
        np.testing.assert_allclose(got, expected, atol=2e-12, rtol=1e-12)


def make_projector(kernel=None, nodes=2, **kwargs):
    coeff = np.zeros((2, nodes, 2, 7))
    coeff[:, :, :, 0] = np.array([[5.2, 6.3], [8.6, 7.4]])[:nodes]
    # R and C disperse along different axes, with opposite signed calibrations.
    coeff[0, :, 0, 1] = 1000
    coeff[1, :, 1, 1] = -1000
    return NativeLineProjector(coeff, 4., (17, 19), np.ones((1, 1)) if kernel is None else kernel,
                               [1., 2.], **kwargs)


def test_flux_centroid_sense_and_batch_scalar_equivalence():
    model = make_projector()
    flux = torch.tensor([2., 3.], dtype=torch.float64)
    vel = torch.tensor([[0., 0.], [100., 100.]], dtype=torch.float64)
    out = model(flux, vel)
    np.testing.assert_allclose(out.sum((-2, -1)), [[5, 10], [5, 10]], atol=1e-13)
    torch.testing.assert_close(out[1:], model(flux, vel[1]), rtol=0, atol=0)
    yy, xx = torch.meshgrid(torch.arange(17), torch.arange(19), indexing="ij")
    for e, grid, sign in [(0, xx, 1), (1, yy, -1)]:
        centroid = (out[:, e] * grid).sum((-2, -1)) / out[:, e].sum((-2, -1))
        assert float(centroid[1] - centroid[0]) == pytest.approx(sign * 1000*4*100/299792.458)


def test_off_cutout_flux_is_lost_not_renormalized_and_psf_can_scatter_in():
    kernel = np.zeros((3, 3)); kernel[1, 2] = 1.  # moves right one pixel
    model = make_projector(kernel, nodes=1)
    shifts = torch.tensor([[-6.5, 0.], [-6.5, 0.]])  # node at x=-1.3
    out = model([2.], [0.], shifts=shifts)
    np.testing.assert_allclose(out.sum((-2, -1)), [[1.4, 2.8]], atol=1e-12)
    gone = model([2.], [0.], shifts=shifts - 100)
    assert gone.sum() == 0


def test_intrinsic_width_conserves_flux_and_has_correct_second_moment():
    model = make_projector(nodes=1, spectral_samples=7)
    cold, warm = model([1.], [0.]), model([1.], [0.], sigma_v=50.)
    np.testing.assert_allclose(warm.sum((-2, -1)), [[1., 2.]], atol=1e-12)
    x = torch.arange(19, dtype=torch.float64)
    def variance(img):
        p = img[0, 0].sum(0); mu = (p*x).sum()/p.sum()
        return ((x-mu)**2*p).sum()/p.sum()
    # Bilinear deposition adds <=1/4 pixel^2 of phase-dependent variance.
    assert abs(float(variance(warm)-variance(cold)) - (4*1000*50/299792.458)**2) < .25


def test_autograd_matches_finite_differences_for_flux_velocity_and_shifts():
    model = make_projector()
    flux = torch.tensor([1., 2.], dtype=torch.float64, requires_grad=True)
    vel = torch.tensor([21., -41.], dtype=torch.float64, requires_grad=True)
    shifts = torch.tensor([[.03, .09], [.12, -.06]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda f, v, s: model(f, v, shifts=s),
                                   (flux, vel, shifts), fast_mode=True)


def test_cubic_deposition_flux_centroid_width_and_gradient_at_pixel_crossings():
    model = make_projector(nodes=1, deposition="cubic")
    for phase in (.0, .13, .5, .97):
        # Move the first node to x=6+phase, y=7 (an exact crossing in y).
        shift = torch.tensor([[.8+phase, .7], [.8+phase, .7]], dtype=torch.float64, requires_grad=True)
        image = model([1.], [0.], shifts=shift)
        torch.testing.assert_close(image.sum((-2, -1)), torch.tensor([[1., 2.]], dtype=torch.float64))
        marginal = image[0, 0].sum(0)
        x = torch.arange(19, dtype=torch.float64)
        centroid = (x*marginal).sum()
        assert float(centroid.detach()) == pytest.approx(6+phase)
        assert float(((x-centroid)**2*marginal).sum().detach()) == pytest.approx(1/3)
        assert torch.autograd.gradcheck(lambda s: model([1.], [0.], shifts=s), (shift,), fast_mode=True)


def test_projected_speed_is_defined_at_reference_radius():
    east = torch.tensor([-.5, 0., .5], dtype=torch.float64)
    vel = projected_arctan_velocity(east, east*0, 200., .3, 0., .7)
    np.testing.assert_allclose(vel, [[-200., 0., 200.]], atol=1e-10)


def test_fixed_likelihood_masks_nan_and_has_gaussian_normalization():
    data = np.array([[[1., np.nan], [2., 0.]]])
    error = np.array([[[2., np.nan], [3., 1.]]])
    valid = np.isfinite(data)
    likelihood = NativeGaussianLikelihood(data, error, valid)
    prediction = torch.tensor([[[3., np.nan], [-1., 0.]]], dtype=torch.float64)
    expected = -.5 * (2 + np.log(2*np.pi*np.array([4, 9, 1])).sum())
    assert float(likelihood(prediction)) == pytest.approx(expected)
    assert float(likelihood.relative(prediction)) == pytest.approx(
        float(likelihood(prediction) - likelihood(torch.zeros_like(prediction))))
    assert torch.isfinite(likelihood(prediction)).all()
    assert float(likelihood(prediction*0)) < float(likelihood(torch.tensor(data)))
    with pytest.raises(ValueError, match="Every exposure"):
        NativeGaussianLikelihood(data, error, valid*False)


def test_dq_policy_and_native_edge_padding(tmp_path):
    data = np.ones((3, 3)); error = data.copy()
    dq = np.array([[0, 1, 2], [4, 5, 0], [0, 0, 0]], dtype=np.uint32)
    data[2, 0] = np.nan; error[2, 1] = 0; error[2, 2] = np.inf
    np.testing.assert_array_equal(native_valid_mask(data, error, dq),
                                  [[True, False, False], [True, False, True], [False]*3])
    fits.writeto(tmp_path / "pixels.fits", np.arange(16.).reshape(4, 4))
    with fits.open(tmp_path / "pixels.fits") as hdul:
        cut = _read_window(hdul[0], (-1, 2), 3, np.nan)
    np.testing.assert_allclose(cut, [[np.nan, 8, 9], [np.nan, 12, 13], [np.nan]*3], equal_nan=True)


@pytest.mark.data
@pytest.mark.parametrize("sid,counts,excluded", [(3700, (9, 9), 0), (8910, (3, 6), 3), (15665, (9, 9), 0)])
def test_real_loader_preserves_native_pixels_and_records_coverage(repo, tmp_path, sid, counts, excluded):
    root = Path(os.environ.get("SAPPHIRES_NATIVE_DIR", Path(repo).parents[1] / "EDR_data/magnif_copy/sapphires_kinematics_inputs_2026-10-13"))
    if not root.is_dir():
        pytest.skip("Copied exposure bundle unavailable; set SAPPHIRES_NATIVE_DIR")
    target = load_native_target(root, Path(repo) / f"data/ID{sid}/config_kinematics.yaml")
    assert tuple(sum(e.pupil == p for e in target.exposures) for p in ("R", "C")) == counts
    assert len(target.excluded_exposures) == excluded
    e = target.exposures[0]
    x, y = e.origin_xy
    with fits.open(e.path, memmap=False) as hdul:
        np.testing.assert_array_equal(e.science, hdul["SCI"].section[y:y+81, x:x+81])
        np.testing.assert_array_equal(e.error, hdul["ERR"].section[y:y+81, x:x+81])
    for e in target.exposures:
        np.testing.assert_allclose(e.source_detector_xy([0.], [0.])[0], e.direct_xy, atol=1e-5)
    # Float32 source-grid tensors must not force Astropy's offset trigonometry
    # into single precision (previously up to ~0.8 detector pixel of error).
    east32 = np.linspace(-.63, .63, 25, dtype=np.float32)
    north32 = east32[::-1].copy()
    np.testing.assert_allclose(e.source_detector_xy(east32, north32),
                               e.source_detector_xy(east32.astype(float), north32.astype(float)), atol=1e-8, rtol=0)
    artifact = tmp_path / "target.npz"
    target.save(artifact)
    with np.load(artifact, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata_json"]))
        assert len(metadata["excluded_exposures"]) == excluded
        np.testing.assert_array_equal(saved["valid"], np.stack([e.valid for e in target.exposures]))
        assert saved["dq"].dtype == np.uint32
