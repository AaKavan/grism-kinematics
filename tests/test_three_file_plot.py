"""Residual reviews keep signs, scales and actual displayed-solution velocities."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from dingo.three_file_plot import candidate_figure, physical_scale, velocity_summary


@pytest.fixture
def candidates():
    r = np.arange(60., dtype=float).reshape(2, 5, 6)
    c = r.copy()
    c[0, 2, 2] -= 1
    c[1, 2, 2] -= 10
    c[:, 1, 1] += 2
    mask = np.ones((5, 6), dtype=bool)
    mask[0, 0] = False
    pixels = dict(candidate_ids=np.array([2, 7]), rectified_R=r,
                  rectified_C=c, difference=r-c, mask=mask)
    records = {2: {"params": dict(V_sini=150., inc_v=np.pi/6, R_v=2.)},
               7: {"full_params": {"velocity.V_sini": 400.,
                    "velocity.inc_v": np.pi/6, "velocity.R_v": 4.}}}
    metadata = dict(redshift=1.1147, pixel_scale_arcsec=.063)
    return pixels, records, metadata


def test_velocity_summary_retains_projected_and_deprojected_values(candidates):
    _, records, metadata = candidates
    pc_pixel = physical_scale(metadata)["pc_per_pixel"]
    summary = velocity_summary(records[2]["params"], metadata, radius_pc=2*pc_pixel)
    assert summary["v_inf"] == pytest.approx(300.)
    assert summary["v_sini"] == 150.
    assert summary["v_at_radius"] == pytest.approx(150.)
    assert summary["v_sini_at_radius"] == pytest.approx(75.)
    assert summary["inclination_deg"] == pytest.approx(30.)
    assert summary["rturn_pc"] == pytest.approx(2*pc_pixel)
    # Deprojected quantities remain available; no solution is silently rejected.
    assert velocity_summary(records[7]["full_params"], metadata)["v_inf"] == pytest.approx(800.)
    assert summary["sampled_velocity_parameter"] == "V_sini"


@pytest.mark.parametrize("format", ["plain", "dotted", "nested"])
def test_deprojected_sampled_amplitude_and_inclination_are_not_reinterpreted(candidates, format):
    _, _, metadata = candidates
    values = dict(V_rot=300., inc_v=np.pi/6, R_v=2.)
    params = (values if format == "plain" else {"velocity": values} if format == "nested"
              else {f"velocity.{name}": value for name, value in values.items()})
    summary = velocity_summary(params, metadata,
                               radius_pc=2*physical_scale(metadata)["pc_per_pixel"])
    assert summary["sampled_velocity_parameter"] == "V_rot"
    assert summary["v_inf"] == 300.
    assert summary["v_sini"] == pytest.approx(150.)
    assert summary["v_at_radius"] == pytest.approx(150.)
    assert summary["v_sini_at_radius"] == pytest.approx(75.)


def test_conflicting_projected_and_deprojected_record_rejected(candidates):
    _, records, metadata = candidates
    values = dict(records[2]["params"], V_rot=500.)
    with pytest.raises(ValueError, match="V_rot and V_sini disagree"):
        velocity_summary(values, metadata)


def test_consistent_derived_amplitude_keeps_legacy_sampled_name(candidates):
    _, records, metadata = candidates
    values = dict(records[2]["params"], V_rot=300.)
    summary = velocity_summary(values, dict(metadata, param_names=["velocity.V_sini"]))
    assert summary["sampled_velocity_parameter"] == "V_sini"


def test_deprojected_residual_titles_match_actual_sampled_values(candidates, tmp_path):
    pixels, records, metadata = candidates
    records[2]["params"]["V_rot"] = 300.
    del records[2]["params"]["V_sini"]
    fig = candidate_figure(pixels, records, metadata, ids=[2])
    try:
        residual = [ax for ax in fig.axes if ax.images][2]
        summary = velocity_summary(records[2]["params"], metadata)
        title = residual.get_title()
        assert r"V_\infty=300" in title
        assert "sampled asymptote" in title and "deprojected" in title
        assert rf"V_{{\rm rot}}({summary['radius_pc']:.0f}\,{{\rm pc}})={summary['v_at_radius']:.0f}" in title
        assert r"\sin i" not in title
        np.testing.assert_array_equal(residual.images[0].get_array()[1:], pixels["difference"][0, 1:])
        fig.savefig(tmp_path/"sampled_rotation.png")
    finally:
        plt.close(fig)


def test_residual_sign_full_stamp_no_smoothing_and_shared_scales(candidates):
    pixels, records, metadata = candidates
    fig = candidate_figure(pixels, records, metadata)
    try:
        panels = [ax for ax in fig.axes if ax.images]
        assert len(panels) == 6
        images = [ax.images[0] for ax in panels]
        assert all(im.get_array().shape == (5, 6) for im in images)
        assert all(im.get_interpolation() == "nearest" for im in images)
        assert images[0].norm is images[1].norm is images[3].norm is images[4].norm
        assert images[2].norm is images[5].norm
        assert images[2].norm.vmin == -images[2].norm.vmax
        assert images[2].get_array()[2, 2] == 1
        assert images[5].get_array()[2, 2] == 10
        assert images[2].get_array()[1, 1] == -2
        assert images[2].get_array().mask[0, 0]
        for panel, record in zip((panels[2], panels[5]), records.values()):
            summary = velocity_summary(record.get("params", record.get("full_params")), metadata)
            title = panel.get_title()
            assert rf"V_{{\rm rot}}({summary['radius_pc']:.0f}\,{{\rm pc}})\sin i={summary['v_sini_at_radius']:.0f}" in title
            assert rf"V_\infty\sin i={summary['v_sini']:.0f}" in title
            assert "(asymptote)" in title
            assert r"V_\infty=" not in title
        assert panels[2].get_subplotspec().get_gridspec().get_width_ratios() == [1, 1, 1.5]
    finally:
        plt.close(fig)


def test_subset_keeps_global_scales_and_original_ids(candidates):
    pixels, records, metadata = candidates
    all_fig = candidate_figure(pixels, records, metadata)
    sub_fig = candidate_figure(pixels, records, metadata, ids=[7])
    try:
        all_panels = [a for a in all_fig.axes if a.images]
        sub_panels = [a for a in sub_fig.axes if a.images]
        assert "Candidate 7" in sub_panels[0].get_title()
        assert sub_panels[2].images[0].norm.vmax == all_panels[2].images[0].norm.vmax
        assert sub_panels[0].images[0].norm.vmin == all_panels[0].images[0].norm.vmin
        assert sub_panels[0].images[0].norm.vmax == all_panels[0].images[0].norm.vmax
    finally:
        plt.close(all_fig)
        plt.close(sub_fig)


def test_wrong_saved_residual_cannot_be_mislabeled(candidates):
    pixels, records, metadata = candidates
    pixels["difference"] *= -1
    with pytest.raises(ValueError, match="corrected R minus corrected C"):
        candidate_figure(pixels, records, metadata)


@pytest.mark.parametrize("changes", [{"inc_v": 0.}, {"inc_v": np.pi/2},
    {"R_v": 0.}, {"V_sini": -1.}, {"inc_v": np.nan}])
def test_invalid_kinematics_rejected(candidates, changes):
    _, records, metadata = candidates
    params = dict(records[2]["params"], **changes)
    with pytest.raises(ValueError):
        velocity_summary(params, metadata)


def test_figure_renders_without_a_sampler(candidates, tmp_path):
    pixels, records, metadata = candidates
    fig = candidate_figure(pixels, records, metadata, residual_limit=10.)
    try:
        filename = tmp_path/"residual_review.png"
        fig.savefig(filename)
        assert filename.stat().st_size > 1000
    finally:
        plt.close(fig)
