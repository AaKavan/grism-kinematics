"""Saved joint-posterior inspection must preserve draw identity and failures."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from dingo.direct_posterior_review import diagnostics, show_posterior


def _saved_run(tmp_path, chains=4, draws=1200):
    directory = tmp_path / "source_run"
    directory.mkdir()
    rng = np.random.default_rng(817)
    theta = np.array([300., 5., np.deg2rad(280.), .2, -.1, np.deg2rad(70.)])
    scales = np.array([20., .5, .03, .1, .1, .04])
    samples = theta + rng.normal(size=(chains, draws, 6))*scales
    emission = np.exp(rng.normal(-3., .5, size=(chains, draws, 3, 3)))
    info = dict(source_id=6170, fixed_center_pixels=[1., 1.], noise_sigma=dict(R=.02, C=.03),
                mcmc=dict(chains=chains, max_tree_depth=8))
    (directory / "run_info.txt").write_text(json.dumps(info))
    arrays = dict(sample_theta=samples, sample_emission=emission, fingerprint=np.array("same-target"),
                  stats_energy=rng.normal(size=(chains, draws)),
                  stats_potential_energy=rng.normal(size=(chains, draws)),
                  stats_num_steps=np.full((chains, draws), 31),
                  stats_accept_prob=np.full((chains, draws), .9),
                  stats_diverging=np.zeros((chains, draws), dtype=bool),
                  stats_is_warmup=np.zeros((chains, draws), dtype=bool),
                  stats_iteration=np.broadcast_to(np.arange(501, 501+draws), (chains, draws)))
    np.savez(directory / "posterior.npz", **arrays)
    # Deliberately choose a draw away from the pooled median. The residuals
    # must use this actual saved prediction and its own velocity label.
    chain, draw = 1, min(17, draws-1)
    pred = dict(theta=samples[chain, draw], emission=emission[chain, draw],
                chain_index=np.array(chain), draw_index=np.array(draw),
                fingerprint=np.array("same-target"), projected_centers=np.array([[1., 1.], [1.2, 1.1]]))
    for pupil, extra in (("R", .03), ("C", -.02)):
        pred[f"predicted_{pupil}"] = emission[chain, draw]*2
        pred[f"observed_{pupil}"] = pred[f"predicted_{pupil}"] + extra
        pred[f"valid_{pupil}"] = np.ones((3, 3), dtype=bool)
    np.savez(directory / "posterior_prediction.npz", **pred)
    return directory, arrays, pred


def test_diagnostics_require_joint_map_mixing_and_no_divergences(tmp_path):
    directory, arrays, _ = _saved_run(tmp_path)
    report = diagnostics(directory / "posterior.npz")
    assert report["convergence_checks_passed"]
    assert report["morphology"]["all_pixels"]["number_of_pixels"] == 9
    assert len(report["parameters"]) == 6
    assert report["divergences"] == 0
    # Good kinematics do not rescue an unchanging nuisance-map coordinate.
    arrays["sample_emission"][0, :, 0, 0] = .01
    np.savez(directory / "posterior.npz", **arrays)
    report = diagnostics(directory)
    assert not report["convergence_checks_passed"]
    assert report["morphology"]["all_pixels"]["constant_chain_coordinate"]


@pytest.mark.parametrize("bad", ["divergence", "warmup", "nan", "constant_energy"])
def test_bad_retained_diagnostics_cannot_pass(tmp_path, bad):
    directory, arrays, _ = _saved_run(tmp_path)
    if bad == "divergence":
        arrays["stats_diverging"][0, 1] = True
    elif bad == "warmup":
        arrays["stats_is_warmup"][0, 1] = True
    elif bad == "nan":
        arrays["sample_theta"][0, 1, 0] = np.nan
    else:
        arrays["stats_energy"][:] = 1.
    np.savez(directory / "posterior.npz", **arrays)
    report = diagnostics(directory)
    assert report["convergence_checks_passed"] is False
    # Saved JSON must use null for unavailable/nonfinite diagnostics.
    saved = (directory / "posterior_diagnostics.json").read_text()
    assert "NaN" not in saved and "Infinity" not in saved


def test_warmup_only_is_readable_and_not_a_posterior(tmp_path):
    directory, _, _ = _saved_run(tmp_path)
    np.savez(directory / "posterior.npz", iterations=np.array(25), warmup=np.array(100))
    report = diagnostics(directory)
    assert report["draws_per_chain"] == 0
    assert not report["convergence_checks_passed"]
    with pytest.raises(ValueError, match="No retained"):
        show_posterior(directory)


def test_tiny_runs_and_stale_prediction_are_rejected(tmp_path):
    directory, arrays, prediction = _saved_run(tmp_path, draws=2)
    report = diagnostics(directory)
    assert not report["convergence_checks_passed"]
    prediction["theta"] = prediction["theta"].copy()
    prediction["theta"][0] += 1.
    np.savez(directory / "posterior_prediction.npz", **prediction)
    with pytest.raises(ValueError, match="do not agree"):
        show_posterior(directory)


def test_plots_reopen_saved_draws_with_coherent_residual_and_pooled_summary(tmp_path):
    directory, arrays, prediction = _saved_run(tmp_path, draws=160)
    views = show_posterior(directory)
    try:
        expected = np.quantile(arrays["sample_theta"][..., 0], [.16, .5, .84])
        assert views["summary"].loc["V_inf", "median"] == pytest.approx(expected[1])
        assert views["summary"].loc["V_inf", "minus_16th"] == pytest.approx(expected[1]-expected[0])
        assert views["summary"].loc["V_at_R_t", "median"] == pytest.approx(expected[1]/2)
        axes = [ax for ax in views["residuals"].axes if ax.images]
        for row, pupil, sigma in ((0, "R", .02), (1, "C", .03)):
            np.testing.assert_array_equal(axes[row*3+1].images[0].get_array(), prediction[f"predicted_{pupil}"])
            residual = (prediction[f"observed_{pupil}"]-prediction[f"predicted_{pupil}"])/sigma
            np.testing.assert_allclose(axes[row*3+2].images[0].get_array(), residual)
        title = views["residuals"]._suptitle.get_text()
        assert f"V_inf={prediction['theta'][0]:.1f}" in title
        assert "one joint draw" in title
        for suffix in ("corner.png", "corner.pdf", "traces.png", "residuals.png", "emission.png",
                       "rotation_curve.png", "summary.txt"):
            assert (directory / f"posterior_{suffix}").is_file()
        diagonal = np.asarray(views["corner"].axes).reshape(6, 6)
        assert f"{expected[1]:.2f}" in diagonal[0, 0].get_title()
    finally:
        plt.close("all")


def _set_fixed_rdy(directory, arrays, prediction, value=.271):
    """Construct explicitly fixed saved output without running any sampler."""
    info_path = directory / "run_info.txt"
    info = json.loads(info_path.read_text())
    info["direct_model"] = {"fixed_rdy": value}
    info_path.write_text(json.dumps(info))
    arrays["sample_theta"][..., 4] = value
    prediction["theta"][4] = value
    np.savez(directory / "posterior.npz", **arrays)
    np.savez(directory / "posterior_prediction.npz", **prediction)


def test_explicit_fixed_rdy_is_verified_then_excluded_from_mixing(tmp_path):
    directory, arrays, prediction = _saved_run(tmp_path)
    _set_fixed_rdy(directory, arrays, prediction)
    report = diagnostics(directory)
    assert report["convergence_checks_passed"]
    assert report["fixed_parameters"] == {"R_dy": .271}
    assert "R_dy" not in report["parameters"]
    assert len(report["parameters"]) == 5
    assert "R_dy fixed" in report["scope"]

    # Explicitly fixing dy must not excuse a stuck coordinate that is free.
    arrays["sample_theta"][0, :, 3] = .2
    np.savez(directory / "posterior.npz", **arrays)
    report = diagnostics(directory)
    assert not report["convergence_checks_passed"]
    assert report["parameters"]["R_dx"]["constant_chain_coordinate"]


def test_constant_rdy_without_fixed_metadata_is_still_a_failed_chain(tmp_path):
    directory, arrays, _ = _saved_run(tmp_path)
    arrays["sample_theta"][..., 4] = .271
    np.savez(directory / "posterior.npz", **arrays)
    report = diagnostics(directory)
    assert report["fixed_parameters"] == {}
    assert not report["convergence_checks_passed"]
    assert report["parameters"]["R_dy"]["constant_chain_coordinate"]


def test_fixed_metadata_cannot_hide_varying_rdy_draws(tmp_path):
    directory, arrays, prediction = _saved_run(tmp_path)
    _set_fixed_rdy(directory, arrays, prediction)
    arrays["sample_theta"][0, 0, 4] += .01
    np.savez(directory / "posterior.npz", **arrays)
    report = diagnostics(directory)
    assert not report["convergence_checks_passed"]
    assert any("disagree with the explicit fixed value" in message for message in report["warnings"])
    with pytest.raises(ValueError, match="disagree with the explicit fixed value"):
        show_posterior(directory)


def test_fixed_rdy_plots_omit_the_axis_and_label_conditional_errors(tmp_path):
    directory, arrays, prediction = _saved_run(tmp_path, draws=160)
    _set_fixed_rdy(directory, arrays, prediction)
    views = show_posterior(directory)
    try:
        assert len(views["corner"].axes) == 25
        assert sum(ax.axison for ax in views["traces"].axes) == 5
        summary = views["summary"]
        assert summary.loc["R_dy", "median"] == .271
        assert summary.loc["R_dy", "status"] == "fixed"
        assert np.isnan(summary.loc["R_dy", "minus_16th"])
        assert np.isnan(summary.loc["R_dy", "plus_84th"])
        assert summary.loc["inclination", "status"] == "sampled"
        assert summary.loc["V_at_R_t", "status"] == "derived"
        assert "R_dy fixed at +0.271" in summary.attrs["scope"]
        assert "R_dy fixed at +0.271" in views["corner"]._suptitle.get_text()
        assert "R_dy fixed at +0.271" in views["rotation_curve"].axes[0].get_title()
        assert "fixed" in (directory / "posterior_summary.txt").read_text()
    finally:
        plt.close("all")
