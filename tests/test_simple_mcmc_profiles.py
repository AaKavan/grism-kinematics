"""Physical postprocessing preserves posterior pairs, units, and saved chains."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from dingo.simple_mcmc_profiles import PhysicalProfiles


@pytest.fixture
def saved(tmp_path):
    # Correlated analytic arrays test propagation without fitting mock images.
    velocity = np.array([100., 150., 230., 310., 430., 570., 730., 900.]).reshape(2, 4)
    turnover = np.array([1., 2., 4., 7., 11., 17., 25., 35.]).reshape(2, 4)
    info = {"schema": "dingo-simple-mcmc-v1", "source_id": 6170, "status": "finished",
            "configuration": {"summary": {"z": 1.1147, "r_fit": 40}},
            "fixed_parameters": {"velocity.x0_v": 40., "velocity.y0_v": 40.,
                                 "velocity.inc_v": float(np.deg2rad(75.))}}
    def write():
        np.savez(tmp_path / "samples.npz", chain=np.stack([velocity, turnover], axis=-1),
                 param_names=np.array(["velocity.V_rot", "velocity.R_v"]))
        (tmp_path / "run_info.txt").write_text(json.dumps(info))
        return PhysicalProfiles(tmp_path, pixel_scale_arcsec=.063)
    return tmp_path, velocity, turnover, info, write


def test_paired_profiles_and_each_draw_turnover(saved):
    _, v, rt, _, write = saved
    result = write()
    scale = result.scale["pc_per_pixel"]
    assert scale == pytest.approx(530.265566, abs=1e-6)
    radii = np.linspace(0, 40 * scale, 35)
    actual = result.radial_profiles(radii)
    expected_v = 2 / np.pi * v[..., None] * np.arctan(radii / (rt[..., None] * scale))
    expected_m = radii * expected_v**2 / .00430091727003628
    np.testing.assert_allclose(actual["vrot_kms"], np.quantile(expected_v, [.16, .5, .84], axis=(0, 1)))
    np.testing.assert_allclose(actual["mass_msun"], np.quantile(expected_m, [.16, .5, .84], axis=(0, 1)))
    shortcut = 2 / np.pi * np.median(v) * np.arctan(radii / (np.median(rt) * scale))
    assert not np.allclose(shortcut, actual["vrot_kms"][1], rtol=.001)
    table = result.turnover_summary()
    for name, expected in (("turnover_velocity", v / 2),
                            ("turnover_radius_pixels", rt),
                            ("turnover_radius_pc", rt * scale)):
        q = np.quantile(expected, [.16, .5, .84])
        row = table.loc[name]
        np.testing.assert_allclose([row["median"], row["minus_16th"], row["plus_84th"]],
                                   [q[1], q[1] - q[0], q[2] - q[1]])


def test_arctan_mass_units_no_double_deprojection_and_scale(saved):
    path, v, rt, _, write = saved
    v[:] = 400.
    rt[:] = 10.
    result = write()
    scale = result.scale["pc_per_pixel"]
    p = result.radial_profiles(np.array([0., 10., 1e9]) * scale)
    np.testing.assert_allclose(p["vrot_kms"], np.tile([0., 200., 400.], (3, 1)), rtol=1e-8)
    np.testing.assert_allclose(p["mass_msun"][:, 1], 10 * scale * 200**2 / .00430091727003628)
    assert np.all(p["mass_msun"][:, 0] == 0)
    twice = PhysicalProfiles(path, pixel_scale_arcsec=.126)
    other = twice.radial_profiles(np.array([0., 10., 1e9]) * 2 * scale)
    np.testing.assert_allclose(p["vrot_kms"], other["vrot_kms"])
    np.testing.assert_allclose(2 * p["mass_msun"], other["mass_msun"])


@pytest.mark.parametrize("radii", [[], [1.], [-1., 1.], [2., 1.], [1., 1.], [0., np.nan]])
def test_invalid_grids_rejected(saved, radii):
    with pytest.raises(ValueError):
        saved[-1]().radial_profiles(radii)


@pytest.mark.parametrize("z", [None, 0., -1., np.nan, np.inf, True])
def test_invalid_redshift_not_guessed(saved, z):
    saved[3]["configuration"]["summary"]["z"] = z
    with pytest.raises(ValueError):
        saved[-1]()


def test_postprocessing_saves_products_without_touching_originals(saved):
    path, _, _, _, write = saved
    result = write()
    originals = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir()}
    result.turnover_summary(save=True)
    radii = np.linspace(0, result.scale["pc_per_pixel"] * 40, 20)
    import matplotlib.pyplot as plt
    for method in (result.rotation_curve, result.mass_profile):
        fig = method(radii, save=True)
        assert "pc" in fig.axes[0].get_xlabel()
        plt.close(fig)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in originals.items())
    for filename in ("turnover_summary.txt", "rotation_curve.png", "dynamical_mass.png", "physical_profiles.npz"):
        assert (path / filename).is_file()
    with np.load(path / "physical_profiles.npz", allow_pickle=False) as data:
        assert data["vrot_kms"].shape == (3, len(radii))
        assert data["mass_msun"].shape == (3, len(radii))
