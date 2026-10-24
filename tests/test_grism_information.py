"""Check shared-nuisance information and deprojection error propagation."""

import numpy as np
import pytest
import torch

from diagnostics.grism_information import profile_information, log_mass_gradient, propagated_sigma


def test_complementary_directions_identify_velocity_only_with_shared_map():
    # Independent observations m+v and m-v: either alone can absorb v into m.
    # Their difference identifies v when the morphology m is shared.
    nuisance = torch.ones((2, 1), dtype=torch.float64)
    velocity = torch.tensor([[1.], [-1.]], dtype=torch.float64)
    joint, _ = profile_information(nuisance, velocity)
    r, _ = profile_information(nuisance[:1], velocity[:1])
    c, _ = profile_information(nuisance[1:], velocity[1:])
    assert float(joint) == pytest.approx(2.)
    assert float(r+c) == pytest.approx(0., abs=1e-25)


def test_profile_information_matches_explicit_linear_regression_covariance():
    rng = np.random.default_rng(902)
    a = rng.normal(size=(70, 12))
    j = rng.normal(size=(70, 4)) + a[:, :4] * 3
    information, diagnostics = profile_information(torch.tensor(a), torch.tensor(j))
    full = np.column_stack([j, a])
    expected = np.linalg.inv(full.T @ full)[:4, :4]
    np.testing.assert_allclose(np.linalg.inv(information.numpy()), expected, rtol=1e-12)
    assert diagnostics["rank"] == 12
    # Duplicating a nuisance column must not create or remove information.
    duplicated, d = profile_information(torch.tensor(np.column_stack([a, a[:, 0]])), torch.tensor(j))
    np.testing.assert_allclose(duplicated.numpy(), information.numpy(), rtol=1e-12)
    assert d["rank"] == 12


def test_unidentified_parameters_do_not_acquire_zero_uncertainty():
    result = propagated_sigma(np.diag([4., 0.]), np.array([[1., 0.], [0., 1.], [1., 1.]]))
    assert result[0] == .5
    assert np.isinf(result[1:]).all()
    assert np.isinf(propagated_sigma(np.zeros((2, 2)), np.eye(2))).all()


def test_mass_gradient_includes_turnover_and_inclination():
    q = np.array([1.05, 1.1, .9, .9])
    radii = np.array([.1, .2, .3, .5])
    def log_mass(p):
        v = p[0]*200 * np.arctan(radii/(p[1]*.2))/np.arctan(.5/(p[1]*.2))/np.sin(p[3])
        return np.log(radii*v**2)
    h = 1e-5
    finite = np.column_stack([(log_mass(q+d*h)-log_mass(q-d*h))/(2*h) for d in np.eye(4)])
    np.testing.assert_allclose(log_mass_gradient(q, radii), finite, rtol=1e-8, atol=1e-9)
    # At the reference radius the explicit turnover derivative vanishes.
    assert log_mass_gradient(q, radii)[-1, 1] == pytest.approx(0.)
