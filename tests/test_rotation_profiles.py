"""Physical profiles must preserve posterior pairs and the fitted amplitude definition."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from dingo.rotation import RotationResults


pytest.importorskip("astropy")


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def profile_run(tmp_path):
    shape = (2, 4)
    # Deliberately correlated speeds/turnover radii distinguish paired curves
    # from curves built using independent marginal summaries.
    v = np.array([[90., 120., 175., 240.], [320., 410., 540., 700.]])
    rt = np.array([[.03, .05, .09, .14], [.22, .35, .6, 1.]])
    samples = dict(vrot=v, log_rturn=np.log(rt),
        inclination=np.linspace(.25, 1.3, v.size).reshape(shape),
        theta=np.full(shape, .7), center=np.zeros((*shape, 2)),
        systemic=np.zeros(shape), line_flux_1e17=np.ones(shape))
    report = dict(format="dingo_rotation_v1", source_id=15665,
        source_summary=dict(z=1.1613), completed=True, real_observations=True,
        settings=dict(chains=2), model=dict(intrinsic_sigma_kms=0),
        retained_draws_per_chain=4, diagnostics=dict(convergence_gate_passed=False))

    def save():
        np.savez(tmp_path / "samples.npz", **samples)
        report["samples_sha256"] = _digest(tmp_path / "samples.npz")
        (tmp_path / "rotation.json").write_text(json.dumps(report))
        return RotationResults(tmp_path, 15665)

    return tmp_path, report, samples, save


def test_profiles_preserve_paired_draws_and_covariance(profile_run):
    _, _, samples, save = profile_run
    result = save()
    scale = result.physical_scale()["pc_per_arcsec"]
    # More than 32 radii exercises multiple blocks without changing the oracle.
    angular_radii = np.linspace(0., .9, 75)
    profiles = result.radial_profiles(angular_radii * scale)
    v_inf = samples["vrot"] * np.pi / (2 * np.arctan(.5 / np.exp(samples["log_rturn"])))
    curves = (2 / np.pi) * v_inf[..., None] * np.arctan(
        angular_radii / np.exp(samples["log_rturn"])[..., None])
    projected = curves * np.sin(samples["inclination"])[..., None]
    mass = angular_radii * scale * curves**2 / .00430091727003628
    for name, draws in (("vrot_kms", curves), ("vsini_kms", projected), ("mass_msun", mass)):
        np.testing.assert_allclose(profiles[name], np.quantile(draws, [.16, .5, .84], axis=(0, 1)))
    np.testing.assert_allclose(profiles["vinf_kms_quantiles"], np.quantile(v_inf, [.16, .5, .84]))
    assert profiles["retained_draws"] == 8
    assert profiles["sampling_checks_passed"] is False
    assert "not established" in profiles["radial_coverage"]
    # A curve from median parameters is not the posterior median curve.
    median_rt = np.median(np.exp(samples["log_rturn"]))
    shortcut = np.median(samples["vrot"]) * np.arctan(angular_radii / median_rt) / np.arctan(.5 / median_rt)
    assert not np.allclose(profiles["vrot_kms"][1], shortcut, rtol=1e-3)


def test_arctan_identities_mass_units_and_no_extra_inclination_correction(profile_run):
    _, _, samples, save = profile_run
    samples["vrot"][:] = 200.
    samples["log_rturn"][:] = np.log(.25)
    result = save()
    scale = result.physical_scale()["pc_per_arcsec"]
    angular_radii = np.array([0., .25, .5, 1e8])
    profiles = result.radial_profiles(angular_radii * scale)
    v_inf = 200 * np.pi / (2 * np.arctan(2.))
    expected = [0., v_inf / 2, 200., v_inf]
    for percentile in profiles["vrot_kms"]:
        np.testing.assert_allclose(percentile, expected, rtol=1e-8)
    np.testing.assert_allclose(profiles["mass_msun"][:, 2], .5 * scale * 200**2 / .00430091727003628)
    np.testing.assert_allclose(profiles["vsini_kms"][:, 2],
        np.quantile(200 * np.sin(samples["inclination"]), [.16, .5, .84]))
    assert np.all(profiles["mass_msun"][:, 0] == 0.)


def test_distance_is_source_specific_and_scales_mass_linearly(profile_run):
    _, report, _, save = profile_run
    first = save().radial_profiles()
    assert first["reference_radius_pc"] == pytest.approx(4237.621650540629)
    assert first["lensing_corrected"] is False
    report["source_summary"]["z"] = .2
    second = save().radial_profiles()
    factor = second["pc_per_arcsec"] / first["pc_per_arcsec"]
    assert factor != pytest.approx(1.)
    np.testing.assert_allclose(second["radius_pc"], first["radius_pc"] * factor)
    np.testing.assert_allclose(second["vrot_kms"], first["vrot_kms"])
    np.testing.assert_allclose(second["mass_msun"], first["mass_msun"] * factor)


@pytest.mark.parametrize("redshift", [None, True, "1.1613", 0., -1., float("nan"), float("inf")])
def test_invalid_or_absent_redshift_is_not_guessed(profile_run, redshift):
    _, report, _, save = profile_run
    report["source_summary"]["z"] = redshift
    with pytest.raises(ValueError, match="positive source redshift"):
        save().physical_scale()


@pytest.mark.parametrize("radii", [[], [1.], 1., [[1., 2.]], [-1., 2.], [2., 1.],
                                  [1., 1.], [0., np.nan], [0., np.inf]])
def test_invalid_radius_grids_are_rejected(profile_run, radii):
    with pytest.raises(ValueError, match="radii_pc"):
        profile_run[3]().radial_profiles(radii)


def test_saved_profile_figures_are_cpu_only_and_leave_science_files_unchanged(profile_run):
    path, _, _, save = profile_run
    save()
    before = {p.name: _digest(p) for p in path.iterdir() if p.is_file()}
    script = """
import builtins, sys
real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'jax', 'jaxlib', 'torch', 'numpyro'}:
        raise AssertionError('Plotting imported a fitting/GPU library: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from dingo import rotation
def forbid_fit(*args, **kwargs):
    raise AssertionError('Saved-result plotting attempted to start fitting')
rotation.fit = rotation.run_command = forbid_fit
result = rotation.RotationResults(sys.argv[1], 15665)
import matplotlib.pyplot as plt
for method in (result.rotation_curve, result.mass_profile):
    figure = method(save=True)
    assert 'Provisional' in '\\n'.join(t.get_text() for t in figure.texts)
    assert figure.axes[0].get_xlabel() == 'Disk-plane radius [pc]'
    plt.close(figure)
"""
    subprocess.run([sys.executable, "-c", script, str(path)], check=True,
        cwd=Path(__file__).resolve().parents[1], env={**os.environ, "MPLBACKEND": "Agg"},
        capture_output=True, text=True, timeout=45)
    assert before == {p.name: _digest(p) for p in path.iterdir() if p.is_file()}
    assert {p.name for p in (path / "figures").iterdir()} == {
        "rotation_curve.png", "rotation_curve.pdf", "mass_profile.png", "mass_profile.pdf"}
