"""The recovery demonstration must not inherit the generating parameters."""
import json
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dingo.simple_simulation import make_mock, load_mock
from dingo import simulation_review as review


def test_recovery_has_displaced_starts_and_known_noise(tmp_path, grism_cal):
    mock = make_mock(tmp_path / "toy", noise_sigma=.02)
    run = review.prepare_recovery(mock, start_velocity=350., start_turnover=3., start_pa=20.)
    p = dict(zip(run.fitter.mcmc_param_names, run.theta))
    assert p["velocity.V_rot"] == pytest.approx(350.)
    assert p["velocity.R_v"] == pytest.approx(3.)
    assert np.rad2deg(p["velocity.theta_v"]) == pytest.approx(20.)
    assert run.info["noise"]["residual_variance"] == pytest.approx(2 * .02**2)
    assert run.info["fixed_parameters"] == {
        name: mock.truth[name] for name in ("velocity.x0_v", "velocity.y0_v", "velocity.inc_v")}
    assert set(p) == {"velocity.V_rot", "velocity.R_v", "velocity.theta_v", "image.R.dx", "image.R.dy"}
    # A second fit cannot be initialized from the mutated state of the first.
    alternate = review.prepare_recovery(mock, start_velocity=75.)
    assert alternate.info["start_parameters"]["velocity.V_rot"] == 75.
    assert load_mock(mock.directory).truth == mock.truth


def test_noiseless_scan_is_finite_and_does_not_produce_a_posterior(tmp_path, grism_cal):
    mock = make_mock(tmp_path / "zero", velocity=0., noise_sigma=0.)
    table, fig = review.scan_velocity(mock, [0., 100., 200.])
    try:
        assert list(table) == ["V_inf", "Noiseless"]
        assert np.isfinite(table.to_numpy()).all()
        assert table.loc[table["Noiseless"].idxmin(), "V_inf"] == 0.
        assert not (mock.directory / "fits").exists()
        assert load_mock(mock.directory).metadata["input_sha256"] == mock.metadata["input_sha256"]
        with pytest.raises(ValueError, match="velocity scan"):
            review.prepare_recovery(mock)
    finally:
        plt.close(fig)


def test_result_from_another_noise_realization_is_rejected(tmp_path, grism_cal):
    one = make_mock(tmp_path / "one", seed=1)
    other = make_mock(tmp_path / "other", seed=2)
    run = review.prepare_recovery(one)
    # Identity must fail before attempts to load nonexistent samples or plots.
    with pytest.raises(ValueError, match="different simulation"):
        review.compare_posterior(other, run.directory)
    with pytest.raises(ValueError, match="different simulation"):
        review.compare_starts(other, {350.: run})
    assert not (other.directory / "starting_values.csv").exists()


def test_start_comparison_rechecks_inputs_after_generation(tmp_path, grism_cal):
    mock = make_mock(tmp_path / "toy")
    run = review.prepare_recovery(mock)
    source = mock.data_dir / f"ID{mock.source_id}" / "R.fits"
    source.write_bytes(source.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Simulation input changed"):
        review.compare_starts(mock, {350.: run})
    assert not (mock.directory / "starting_values.csv").exists()


@pytest.mark.parametrize("geometry", ["truth", "different_center", "different_i", "sampled_i"])
def test_posterior_title_only_claims_geometry_actually_fixed_to_truth(
        tmp_path, grism_cal, monkeypatch, geometry):
    import pandas as pd

    mock = make_mock(tmp_path / "toy")
    names = ["velocity.V_rot", "velocity.R_v", "velocity.theta_v"]
    fixed = {name: mock.truth[name]
             for name in ("velocity.x0_v", "velocity.y0_v", "velocity.inc_v")}
    if geometry == "different_center":
        fixed["velocity.x0_v"] += 1.
    elif geometry == "different_i":
        fixed["velocity.inc_v"] += np.deg2rad(10.)
    elif geometry == "sampled_i":
        fixed.pop("velocity.inc_v")
    result = tmp_path / "saved"
    result.mkdir()
    info = {"inputs": {name: {"sha256": value}
                       for name, value in mock.metadata["input_sha256"].items()},
            "fixed_parameters": fixed}
    (result / "run_info.txt").write_text(json.dumps(info))
    chain = np.broadcast_to([mock.truth[name] for name in names], (3, 4, 3)).copy()
    np.savez(result / "samples.npz", chain=chain, param_names=names)
    # Rendering stubs isolate the saved-run identity and labeling contract;
    # no sampler is needed to test whether geometry claims are truthful.
    views = {"summary": pd.DataFrame(index=names),
             "traces": plt.subplots(3, 1)[0], "corner": plt.subplots()[0]}
    monkeypatch.setattr(review.sm, "show_results", lambda *args, **kwargs: views)
    try:
        returned = review.compare_posterior(mock, result)
        title = returned["rotation_curve"].axes[0].get_title()
        assert ("fixed to truth" in title) == (geometry == "truth")
        assert (result / "truth_comparison.csv").is_file()
    finally:
        for value in views.values():
            if hasattr(value, "savefig"):
                plt.close(value)


def test_corner_keeps_truth_visible_when_posterior_misses_it():
    import corner

    fig = corner.corner(np.random.default_rng(4).normal(size=(500, 2)))
    truths = [-10., 20.]
    try:
        assert review._include_corner_truths(fig, truths)
        axes = np.asarray(fig.axes).reshape(2, 2)
        for index, truth in enumerate(truths):
            lower, upper = axes[index, index].get_xlim()
            assert lower < truth < upper
        assert axes[1, 0].get_xlim() == axes[0, 0].get_xlim()
        assert axes[1, 0].get_ylim() == axes[1, 1].get_xlim()
    finally:
        plt.close(fig)
