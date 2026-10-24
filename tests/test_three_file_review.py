"""Saved three-file posteriors retain their velocity definition and paired draws."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from dingo import three_file


@pytest.fixture
def saved_run(tmp_path):
    names = ["velocity.V_sini", "velocity.R_v", "velocity.inc_v",
             "velocity.theta_v", "velocity.x0_v", "velocity.y0_v"]
    # Strong correlations make a curve from marginal medians observably wrong.
    values = np.array([
        [80., 1., .3, .6, 39., 41.],
        [100., 2., .4, .7, 40., 42.],
        [140., 4., .5, .8, 41., 43.],
        [180., 6., .6, .9, 42., 44.],
        [230., 9., .7, 1., 43., 45.],
        [300., 12., .8, 1.1, 44., 46.],
        [400., 18., .9, 1.2, 45., 47.],
        [500., 25., 1., 1.3, 46., 48.],
    ]).reshape(4, 2, len(names))
    report = dict(schema="dingo-three-file-v1", source_id=6170,
                  intrinsic_dispersion_kms=0., redshift=1.1147,
                  pixel_scale_arcsec=.063, r_fit_pixels=40,
                  walkers=2, draws=4, warmup=1000, param_names=names,
                  fixed_params={}, sampling_pass=False, status="complete")

    def save():
        path = tmp_path / "samples.npz"
        # This file contains retained steps only, even though warmup is recorded.
        np.savez(path, chain=values, param_names=np.asarray(names), warmup=1000)
        report["samples_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (tmp_path / "report.json").write_text(json.dumps(report))
        return three_file.ThreeFileResults(tmp_path, 6170)

    return tmp_path, values, report, save


def test_asymptotic_velocity_deprojection_and_no_second_burn_removal(saved_run):
    _, values, _, save = saved_run
    result = save()
    columns = result._columns()
    assert columns["vrot"][0].shape == (4, 2)
    assert result._samples()[0].size == values.size
    np.testing.assert_allclose(columns["vsini"][0], values[..., 0])
    np.testing.assert_allclose(columns["vrot"][0], values[..., 0]/np.sin(values[..., 2]))
    assert r"V_\infty" in columns["vrot"][1]
    scale = result.physical_scale()["pc_per_pixel"]
    np.testing.assert_allclose(columns["rturn_pc"][0], values[..., 1]*scale)
    np.testing.assert_allclose(columns["x0"][0], values[..., 4]-40)
    np.testing.assert_allclose(columns["y0"][0], values[..., 5]-40)
    assert "[pixel]" in columns["x0"][1]
    assert "[pixel]" in columns["y0"][1]


def test_sampled_v_and_i_are_not_deprojected_twice(saved_run, monkeypatch):
    import corner
    import matplotlib.pyplot as plt
    _, values, report, save = saved_run
    report["param_names"][0] = "velocity.V_rot"
    report["velocity_parameter"] = "V_rot"
    result = save()
    cols = result._columns()
    np.testing.assert_array_equal(cols["vrot"][0], values[..., 0])
    np.testing.assert_allclose(cols["vsini"][0], values[..., 0]*np.sin(values[..., 2]))
    assert result.velocity_summary().index[0] == "Vinf [km/s]"
    assert result.default_parameters()[0] == "vrot"
    captured = {}

    def capture(draws, **kwargs):
        captured.update(draws=draws.copy(), **kwargs)
        fig, _ = plt.subplots(draws.shape[1], draws.shape[1])
        return fig

    monkeypatch.setattr(corner, "corner", capture)
    fig = result.corner()
    try:
        np.testing.assert_array_equal(captured["draws"][:, 0], values[..., 0].ravel())
        assert r"\sin i" not in captured["labels"][0]
        assert "sampled deprojected asymptote" in " ".join(t.get_text() for t in fig.texts)
    finally:
        plt.close(fig)
    radii = np.array([0., 1000., 3000.])
    fig = result.rotation_curve(radii)
    try:
        expected = (2/np.pi)*values[..., 0, None]*np.arctan(
            radii/(values[..., 1, None]*result.physical_scale()["pc_per_pixel"]))
        np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), np.median(expected, axis=(0, 1)))
        assert r"\sin i" not in fig.axes[0].get_ylabel()
    finally:
        plt.close(fig)


@pytest.fixture
def fixed_inclination_run(saved_run):
    path, _, report, _ = saved_run
    report["param_names"] = ["velocity.V_rot", "velocity.R_v", "velocity.theta_v",
                             "velocity.x0_v", "velocity.y0_v"]
    rng = np.random.default_rng(526)
    values = (np.asarray([400., 3., .6, 40., 41.])
              + rng.normal(size=(250, 4, 5))*[12., .15, .03, .1, .15])
    report.update(walkers=4, draws=250, velocity_parameter="V_rot",
                  fixed_params={"velocity.inc_v": np.deg2rad(30.)},
                  inclination_mode="fixed", fixed_inclination_deg=30.,
                  seed_kind="manual_start", start_velocity_kms=350.,
                  start_inclination_deg=30., base_candidate_id=1)
    np.savez(path/"samples.npz", chain=values, param_names=report["param_names"])
    report["samples_sha256"] = hashlib.sha256((path/"samples.npz").read_bytes()).hexdigest()
    (path/"report.json").write_text(json.dumps(report))
    return three_file.ThreeFileResults(path, 6170), values


def test_fixed_inclination_profiles_and_summary_use_exact_fixed_angle(fixed_inclination_run):
    result, values = fixed_inclination_run
    assert result.fixed_inclination_deg == pytest.approx(30.)
    columns = result._columns()
    np.testing.assert_allclose(columns["inclination"][0], 30.)
    np.testing.assert_allclose(columns["vrot"][0], values[..., 0])
    np.testing.assert_allclose(columns["vsini"][0], .5*values[..., 0])
    profiles = result.radial_profiles([0., 1000., 3000.])
    np.testing.assert_allclose(profiles["vsini_kms"], .5*profiles["vrot_kms"])
    summary = result.velocity_summary()
    assert "Inclination [deg]" not in summary.index
    assert "Fixed i = 30°" in result.status
    assert "intervals are conditional" in result.status
    assert "both sampled" not in result.status


@pytest.mark.parametrize("explicit", [False, True])
def test_fixed_inclination_corner_and_traces_drop_constant_dimension(fixed_inclination_run, explicit):
    import matplotlib.pyplot as plt
    result, _ = fixed_inclination_run
    parameters = ("vrot", "rturn_pixels", "inclination", "theta", "x0", "y0") if explicit else None
    assert result.default_parameters() == ("vrot", "rturn_pc", "theta", "x0", "y0")
    for figure, axes_count in ((result.corner(parameters), 25), (result.traces(parameters), 5)):
        try:
            assert len(figure.axes) == axes_count
            labels = [label for ax in figure.axes for label in (ax.get_xlabel(), ax.get_ylabel())]
            assert r"$i$ [deg]" not in labels
            assert "Fixed i = 30°" in " ".join(t.get_text() for t in figure.texts)
        finally:
            plt.close(figure)


def test_fixed_inclination_curve_labels_conditional_intervals(fixed_inclination_run):
    import matplotlib.pyplot as plt
    result, _ = fixed_inclination_run
    for figure in (result.rotation_curve([0., 1000., 3000.]),
                   result.rotation_curve(radii_pixels=[0., 2., 6.], projected=True)):
        try:
            assert "Fixed i = 30°; intervals are conditional" in " ".join(t.get_text() for t in figure.texts)
        finally:
            plt.close(figure)


def test_fixed_inclination_bookkeeping_must_match_actual_chain(fixed_inclination_run):
    result, _ = fixed_inclination_run
    result.report["fixed_inclination_deg"] = 40.
    with pytest.raises(ValueError, match="disagrees"):
        result._columns()
    result.report["param_names"].append("velocity.inc_v")
    with pytest.raises(ValueError, match="both sampled and fixed"):
        result.default_parameters()


def test_profiles_use_paired_asymptotic_velocity_and_turnover_draws(saved_run):
    _, values, _, save = saved_run
    result = save()
    scale = result.physical_scale()["pc_per_pixel"]
    # More than eight radii exercises the chunked calculation.
    radii = np.linspace(0., 30., 37)*scale
    result_profiles = result.radial_profiles(radii)
    vinf = values[..., 0]/np.sin(values[..., 2])
    rt = values[..., 1]*scale
    curves = (2/np.pi)*vinf[..., None]*np.arctan(radii/rt[..., None])
    projected = (2/np.pi)*values[..., 0, None]*np.arctan(radii/rt[..., None])
    masses = radii*curves**2/.00430091727003628
    np.testing.assert_allclose(result_profiles["vsini_kms"],
                               np.quantile(projected, [.16, .5, .84], axis=(0, 1)))
    np.testing.assert_allclose(result_profiles["vrot_kms"],
                               np.quantile(curves, [.16, .5, .84], axis=(0, 1)))
    np.testing.assert_allclose(result_profiles["mass_msun"],
                               np.quantile(masses, [.16, .5, .84], axis=(0, 1)))
    np.testing.assert_array_equal(result_profiles["radius_pc"], radii)
    shortcut = (2/np.pi)*np.median(vinf)*np.arctan(radii/np.median(rt))
    assert not np.allclose(result_profiles["vrot_kms"][1], shortcut, rtol=1e-3)
    projected_shortcut = (2/np.pi)*np.median(values[..., 0])*np.arctan(radii/np.median(rt))
    assert not np.allclose(result_profiles["vsini_kms"][1], projected_shortcut, rtol=1e-3)


def test_turnover_is_half_asymptotic_speed_and_mass_has_one_deprojection(saved_run):
    _, values, _, save = saved_run
    values[..., 0] = 150.
    values[..., 1] = 3.
    values[..., 2] = np.pi/6
    result = save()
    rt = 3*result.physical_scale()["pc_per_pixel"]
    profiles = result.radial_profiles([0., rt, rt*1e12])
    for quantile in profiles["vsini_kms"]:
        np.testing.assert_allclose(quantile, [0., 75., 150.], rtol=1e-10)
    for quantile in profiles["vrot_kms"]:
        np.testing.assert_allclose(quantile, [0., 150., 300.], rtol=1e-10)
    np.testing.assert_allclose(profiles["mass_msun"][:, 1], rt*150**2/.00430091727003628)


def test_velocity_summary_defaults_to_projected_paired_draws(saved_run):
    _, values, _, save = saved_run
    result = save()
    radius = 3000.
    scale = result.physical_scale()["pc_per_pixel"]
    summary = result.velocity_summary(radius_pc=radius)
    assert summary.index.tolist() == ["Vrot(3000 pc) sin(i) [km/s]",
                                      "Vinf sin(i) [km/s]", "Inclination [deg]"]
    projected = (2/np.pi)*values[..., 0]*np.arctan(radius/(values[..., 1]*scale))
    lo, med, hi = np.quantile(projected, [.16, .5, .84])
    np.testing.assert_allclose(summary.iloc[0], [med, med-lo, hi-med])
    extended = result.velocity_summary(radius_pc=radius, include_deprojected=True)
    np.testing.assert_allclose(extended.loc[summary.index], summary)
    assert "Vrot(3000 pc) [km/s]" in extended.index
    assert "Vinf [km/s]" in extended.index
    deprojected = projected/np.sin(values[..., 2])
    lo, med, hi = np.quantile(deprojected, [.16, .5, .84])
    np.testing.assert_allclose(extended.loc["Vrot(3000 pc) [km/s]"], [med, med-lo, hi-med])


def test_default_corner_shows_projected_amplitude_and_sampled_geometry(saved_run, monkeypatch):
    import corner
    import matplotlib.pyplot as plt
    _, values, _, save = saved_run
    result = save()
    captured = {}

    def capture(draws, **kwargs):
        captured.update(draws=draws.copy(), **kwargs)
        fig, _ = plt.subplots(draws.shape[1], draws.shape[1])
        return fig

    monkeypatch.setattr(corner, "corner", capture)
    fig = result.corner()
    try:
        assert three_file.DEFAULT_PARAMETERS == ("vsini", "rturn_pc", "inclination", "theta", "x0", "y0")
        np.testing.assert_array_equal(captured["draws"][:, 0], values[..., 0].ravel())
        np.testing.assert_allclose(captured["draws"][:, 2], np.rad2deg(values[..., 2]).ravel())
        assert r"V_\infty\sin i" in captured["labels"][0]
        assert captured["color"] == "black"
        assert "sampled projected asymptote" in " ".join(t.get_text() for t in fig.texts)
    finally:
        plt.close(fig)


def test_pixel_review_uses_paired_draws_without_distance_conversion(saved_run, monkeypatch):
    import corner
    import matplotlib.pyplot as plt
    _, values, _, save = saved_run
    result = save()

    def no_physical_scale():
        raise AssertionError("Pixel review must not require a distance conversion")

    monkeypatch.setattr(result, "physical_scale", no_physical_scale)
    parameters = ("vsini", "rturn_pixels", "theta", "x0", "y0")
    columns = result._columns(parameters)
    np.testing.assert_array_equal(columns["rturn_pixels"][0], values[..., 1])
    assert "rturn_pc" not in columns
    captured = {}

    def capture(draws, **kwargs):
        captured.update(draws=draws.copy(), **kwargs)
        fig, _ = plt.subplots(draws.shape[1], draws.shape[1])
        return fig

    monkeypatch.setattr(corner, "corner", capture)
    fig = result.corner(parameters)
    try:
        assert captured["draws"].shape == (8, 5)
        np.testing.assert_array_equal(captured["draws"][:, 1], values[..., 1].ravel())
        assert "[pixel]" in captured["labels"][1]
        assert "[pc]" not in " ".join(captured["labels"])
    finally:
        plt.close(fig)
    traces = result.traces(parameters)
    try:
        assert len(traces.axes) == 5
        assert "[pixel]" in traces.axes[1].get_ylabel()
        np.testing.assert_allclose(traces.axes[1].lines[-1].get_ydata(), np.median(values[..., 1], axis=1))
    finally:
        plt.close(traces)

    radius = 8.
    speed = (2/np.pi)*values[..., 0]*np.arctan(radius/values[..., 1])
    lo, med, hi = np.quantile(speed, [.16, .5, .84])
    summary = result.velocity_summary(radius_pixels=radius, include_inclination=False)
    assert summary.index.tolist() == ["Vrot(8 pixel) sin(i) [km/s]", "Vinf sin(i) [km/s]"]
    np.testing.assert_allclose(summary.iloc[0], [med, med-lo, hi-med])
    radii = np.linspace(0., radius, 19)
    profile = result.projected_profile(radii)
    expected = (2/np.pi)*values[..., 0, None]*np.arctan(radii/values[..., 1, None])
    np.testing.assert_allclose(profile["vsini_kms"], np.quantile(expected, [.16, .5, .84], axis=(0, 1)))
    fig = result.rotation_curve(radii_pixels=radii)
    try:
        assert fig.axes[0].get_xlabel() == "Model radius [pixel]"
        np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), profile["vsini_kms"][1])
    finally:
        plt.close(fig)


def test_pixel_and_physical_radius_arguments_cannot_be_mixed(saved_run):
    _, _, _, save = saved_run
    result = save()
    with pytest.raises(ValueError, match="not both"):
        result.velocity_summary(radius_pc=1000., radius_pixels=8.)
    with pytest.raises(ValueError, match="not both"):
        result.rotation_curve([0., 1000.], radii_pixels=[0., 8.])
    with pytest.raises(ValueError, match="projected velocity"):
        result.rotation_curve(radii_pixels=[0., 8.], projected=False)


@pytest.mark.parametrize("projected", [True, False])
def test_rotation_plot_labels_and_medians_match_requested_velocity(saved_run, projected):
    import matplotlib.pyplot as plt
    _, _, _, save = saved_run
    result = save()
    radii = np.array([0., 1000., 3000.])
    # Exercise the public default as well as explicit optional deprojection.
    fig = result.rotation_curve(radii) if projected else result.rotation_curve(radii, projected=False)
    try:
        ax = fig.axes[0]
        expected = result.radial_profiles(radii)["vsini_kms" if projected else "vrot_kms"][1]
        np.testing.assert_allclose(ax.lines[0].get_ydata(), expected)
        assert ax.get_xlabel() == "Disk-plane radius [pc]"
        assert (r"\sin i" in ax.get_ylabel()) is projected
        assert ("projected rotation curve" if projected else "deprojected rotation curve") in fig._suptitle.get_text()
        if projected:
            assert "inclination still enters the disk geometry" in " ".join(t.get_text() for t in fig.texts)
    finally:
        plt.close(fig)


def test_sample_checksum_is_checked_before_review(saved_run):
    path, _, _, save = saved_run
    result = save()
    with (path / "samples.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        result.radial_profiles([0., 1000.])


@pytest.mark.parametrize("field,value", [
    ("source_id", 15665), ("schema", "native-other-format"),
    ("intrinsic_dispersion_kms", 10.),
])
def test_wrong_source_or_model_is_rejected(saved_run, field, value):
    _, _, report, save = saved_run
    report[field] = value
    with pytest.raises(ValueError, match="this source's three-file"):
        save()


@pytest.mark.parametrize("radii", [[1.], [-1., 1.], [1., 1.], [2., 1.], [0., np.nan]])
def test_invalid_physical_radii_are_rejected(saved_run, radii):
    _, _, _, save = saved_run
    with pytest.raises(ValueError, match="strictly increasing"):
        save().radial_profiles(radii)


@pytest.mark.parametrize("custom_data", [False, True])
def test_fit_worker_receives_data_root_containing_source_directory(tmp_path, monkeypatch, custom_data):
    repo = tmp_path / "repo"
    metadata = repo / "data" / "ID6170"
    metadata.mkdir(parents=True)
    (metadata / "config_kinematics.yaml").write_text("summary: {}\n")
    directory = tmp_path / "custom-data" / "ID6170" if custom_data else metadata
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("image.fits", "R.fits", "C.fits"):
        (directory / name).write_bytes(b"not opened by orchestration")
    monkeypatch.setattr(three_file, "REPO", repo)
    commands = []
    monkeypatch.setattr(three_file, "run_command",
                        lambda args, *a, **kw: commands.append(args))
    monkeypatch.setattr(three_file, "ThreeFileResults", lambda path, sid: (path, sid))
    three_file.fit(6170, results=tmp_path / "results", device="cpu",
                   data=directory.parent if custom_data else None)
    assert len(commands) == 1
    args = commands[0]
    worker_data = Path(args[args.index("--data-dir")+1])
    assert worker_data / "ID6170" == directory
    assert Path(args[args.index("--config")+1]) == metadata / "config_kinematics.yaml"
    assert args[:2] == ["-m", "dingo.three_file_fit"]
