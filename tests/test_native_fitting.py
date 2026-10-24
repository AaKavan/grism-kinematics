"""Independent least-squares and physical checks for flexible native fitting."""

import numpy as np
import pytest
import torch

from dingo.native_fitting import StripContinuumLikelihood, BilinearLineMap, NativeMapFitter
from dingo.native_forward import NativeLineProjector


@pytest.mark.parametrize("degree", [0, 1, 2])
def test_continuum_matches_explicit_weighted_least_squares(degree):
    rng = np.random.default_rng(81)
    shape = (2, 7, 9)
    data, line = rng.normal(size=shape), rng.normal(size=shape)
    error = rng.uniform(.3, 1.5, shape)
    valid = rng.uniform(size=shape) > .2
    valid[0, 0] = False  # fully empty R strip
    valid[1, :, 0] = False; valid[1, 0, 0] = True  # rank-deficient C strip
    obj = StripContinuumLikelihood(data, error, valid, ["R", "C"], degree=degree)
    fitted = obj.continuum(torch.tensor(line)).numpy()
    expected, residual = np.zeros(shape), np.zeros(shape)
    for e, pupil in enumerate(("R", "C")):
        d, m, err, keep = [(a[e] if pupil == "R" else a[e].T) for a in (data, line, error, valid)]
        basis = np.polynomial.legendre.legvander(np.linspace(-1, 1, d.shape[-1]), degree)
        out = np.zeros_like(d)
        for j in range(len(d)):
            k = keep[j]
            if k.any():
                coef = np.linalg.lstsq(basis[k]/err[j, k, None], (d[j, k]-m[j, k])/err[j, k], rcond=1e-12)[0]
                out[j] = basis @ coef
        expected[e] = out if pupil == "R" else out.T
    np.testing.assert_allclose(fitted, expected, atol=1e-12)
    residual[valid] = (data-line-expected)[valid]/error[valid]
    assert float(obj.chi_square(torch.tensor(line))) == pytest.approx(np.sum(residual**2))
    expected_relative = -.5*(obj.chi_square(torch.tensor(line))-obj.chi_square(torch.zeros(shape)))
    torch.testing.assert_close(obj.relative(torch.tensor(line)), expected_relative)


def test_continuum_uncertainty_reduces_line_information_and_gradient_is_correct():
    shape = (1, 3, 17)
    obj = StripContinuumLikelihood(np.zeros(shape), np.ones(shape), np.ones(shape, bool), ["R"], degree=1)
    template = torch.exp(-.5*(torch.arange(17, dtype=torch.float64)-8)**2)[None, None].expand(shape).clone()
    projected = obj.residualize(template)
    assert 0 < float(projected.square().sum()) < float(template.square().sum())
    template.requires_grad_(True)
    assert torch.autograd.gradcheck(obj.relative, (template,), fast_mode=True)
    # Arbitrary continuum addition leaves line-dependent relative scores unchanged.
    x = np.linspace(-1, 1, 17)
    continuum = np.stack([2+x, -1+2*x, 3-4*x])[None]
    other = StripContinuumLikelihood(continuum, np.ones(shape), np.ones(shape, bool), ["R"], degree=1)
    torch.testing.assert_close(other.relative(template), obj.relative(template), atol=1e-12, rtol=1e-12)


def test_continuum_batch_and_masked_nan_are_safe():
    data = np.ones((2, 5, 7)); data[0, 0, 0] = np.nan
    valid = np.isfinite(data)
    obj = StripContinuumLikelihood(data, np.ones_like(data), valid, ["R", "C"])
    line = torch.ones((3, 2, 5, 7), dtype=torch.float64)
    line[:, 0, 0, 0] = torch.nan
    scores = obj.relative(line)
    assert torch.isfinite(scores).all()
    torch.testing.assert_close(scores[0], obj.relative(line[0]))


def test_continuum_predictive_variance_and_unconstrained_strips():
    shape = (1, 3, 11)
    valid = np.ones(shape, bool); valid[0, 0] = False
    error = np.full(shape, 2.)
    obj = StripContinuumLikelihood(np.zeros(shape), error, valid, ["R"], degree=2)
    variance = obj.continuum_variance().numpy()
    assert np.isinf(variance[0, 0]).all()
    b = np.polynomial.legendre.legvander(np.linspace(-1, 1, 11), 2)
    expected = np.diag(b @ np.linalg.inv(b.T @ b / 4) @ b.T)
    np.testing.assert_allclose(variance[0, 1], expected)


def test_line_map_flux_and_regularization_converge_for_linear_brightness():
    for size in (3, 7, 13):
        model = BilinearLineMap(size, 49, flux_unit=3., dtype=torch.float64)
        axis = torch.linspace(-1, 1, size, dtype=torch.float64)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        uniform = torch.ones((size, size), dtype=torch.float64)
        linear = 2+x+.5*y
        assert float(model(uniform).sum()) == pytest.approx(3.)
        assert float(model(linear).sum()) == pytest.approx(6.)
        assert float(model.roughness(linear)) == pytest.approx(2.5)
        assert float(model.roughness(uniform)) == 0
        assert torch.min(model(linear)) >= 0


def test_flexible_fit_changes_initial_map_and_retains_complete_state():
    grid = BilinearLineMap(3, 13, half_extent=.3, flux_unit=1., dtype=torch.float64)
    coeff = np.zeros((2, 169, 2, 7))
    coeff[:, :, 0, 0] = 10.2 + grid.east.numpy()/.06
    coeff[:, :, 1, 0] = 10.4 + grid.north.numpy()/.06
    coeff[0, :, 0, 1] = 1000
    coeff[1, :, 1, 1] = 1000
    projector = NativeLineProjector(coeff, 4., (23, 23), np.ones((1, 1)), [1, 1])
    shape = (2, 23, 23)
    empty = NativeMapFitter(projector, grid, np.zeros(shape), np.ones(shape)*.001, np.ones(shape, bool), ["R", "C"], continuum_degree=1)
    truth = np.r_[.7, 1., .8, [.2, 1., .2, 1., 4., 1., .2, 1., .6]]
    with torch.no_grad():
        injected, _ = empty.predict(torch.tensor(truth))
    y, x = np.mgrid[:23, :23]
    continuum = np.stack([.01+.0001*x, -.02+.0002*y])
    fitter = NativeMapFitter(projector, grid, injected.numpy()+continuum, np.ones(shape)*.001,
                             np.ones(shape, bool), ["R", "C"], continuum_degree=1)
    fit = fitter.fit([110., .25, .7], maxiter=300, strength=0., gtol=1e-4)
    assert fit["report"]["train_chi_square"] < 1e-4
    np.testing.assert_allclose(fit["controls"].ravel(), truth[3:], atol=1e-3)
    np.testing.assert_allclose(fit["line"]+fit["continuum"], injected.numpy()+continuum, atol=1e-5)
    assert len(fit["vector"]) == 12
    assert fit["report"]["held_chi_square"] is None
    # Profiling fixes only the requested physical velocity, re-fits the map,
    # and excludes the fixed variable from stationarity/boundary diagnostics.
    profiled = fitter.fit([900., .3, 1.], strength=0., fixed={"v_ref": 140.},
                          maxiter=400, gtol=1e-4)
    assert profiled["vector"][0] == .7
    assert profiled["report"]["fixed_parameters"] == {"v_ref": 140.}
    assert profiled["report"]["train_chi_square"] < 1e-4
    assert profiled["report"]["stationary"]
    np.testing.assert_allclose(profiled["controls"], fit["controls"], atol=1e-3)
    with pytest.raises(ValueError, match="Unknown fixed"):
        fitter.fit([110., .25, .7], strength=0., fixed={"typo": 0.})
    with pytest.raises(ValueError, match="outside"):
        fitter.fit([110., .25, .7], strength=0., fixed={"r_turn": -1.})


def test_nonrotating_null_refits_map_and_ignores_unidentifiable_angles():
    grid = BilinearLineMap(3, 9, half_extent=.3, dtype=torch.float64)
    coeff = np.zeros((2, 81, 2, 7))
    coeff[:, :, 0, 0] = 7.2 + grid.east.numpy()/.1
    coeff[:, :, 1, 0] = 7.4 + grid.north.numpy()/.1
    coeff[0, :, 0, 1] = 1000; coeff[1, :, 1, 1] = 1000
    projector = NativeLineProjector(coeff, 4., (15, 15), np.ones((1, 1)), [1, 1], deposition="cubic")
    true_map = torch.tensor([[.2, 1., .2], [1., 4., 1.], [.2, 1., .6]], dtype=torch.float64)
    data = projector(grid(true_map), torch.zeros(81), sigma_v=45.)[0].numpy()
    fitter = NativeMapFitter(projector, grid, data, np.ones_like(data)*.001,
                             np.ones_like(data, bool), ["R", "C"], continuum_degree=1)
    for rt, theta in ((.05, -2.), (1.5, 4.)):
        fit = fitter.fit([900., .3, 1.], strength=0.,
                         fixed={"v_ref": 0., "r_turn": rt, "theta": theta}, maxiter=200)
        assert fit["vector"][0] == 0
        assert fit["report"]["stationary"]
        assert fit["report"]["boundary_hits"] == []
        assert fit["report"]["train_chi_square"] < 1e-6
        np.testing.assert_allclose(fit["controls"], true_map.numpy(), atol=1e-4)


def test_common_holdout_excludes_unused_grism_and_prevents_data_leakage():
    grid = BilinearLineMap(2, 3, dtype=torch.float64)
    coeff = np.zeros((4, 9, 2, 7))
    projector = NativeLineProjector(coeff, 4., (5, 5), np.ones((1, 1)), np.ones(4))
    data = np.zeros((4, 5, 5))
    arguments = (projector, grid, data, np.ones_like(data), np.ones_like(data, bool), ["R", "R", "C", "C"])
    single = NativeMapFitter(*arguments, train_indices=[0], held_indices=[1, 3])
    joint = NativeMapFitter(*arguments, train_indices=[0, 2], held_indices=[1, 3])
    assert single.train_indices.tolist() == [0]
    assert single.held_indices.tolist() == joint.held_indices.tolist() == [1, 3]
    default = NativeMapFitter(*arguments, train_indices=[0])
    assert default.held_indices.tolist() == [1, 2, 3]
    for held in ([0], [1, 1], [-1], [4]):
        with pytest.raises(ValueError, match="held-out"):
            NativeMapFitter(*arguments, train_indices=[0], held_indices=held)
