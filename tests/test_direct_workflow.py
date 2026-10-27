"""Real-source workflow saves coherent profile fits without claiming MCMC."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from pathlib import Path

from dingo import direct_workflow as wf
from dingo.simple_simulation import make_mock


def test_fixed_rdy_preparation_and_reopening_without_fitting(repo, staged, grism_cal, tmp_path):
    """No fitting: read existing inputs and inspect only the new run settings."""
    staged("ID6170")
    run = wf.prepare(6170, data_dir=Path(repo)/"data", output_root=tmp_path,
                     run_name="fixed_rdy", center=(40.25, 41.), inclination_mean=75.,
                     fixed_rdy=.271, start_offset=(1.406, -.711))
    assert run.theta[4] == .271
    assert run.model.fixed_rdy == .271
    assert run.model.configuration["direct_model"]["fixed_rdy"] == .271
    assert run.model.inclination_sigma == 5.
    before = (run.directory/"run_info.txt").read_bytes()
    reopened = wf.load_run(run.directory)
    assert reopened.model.fixed_rdy == .271
    assert reopened.theta[4] == .271
    assert (run.directory/"run_info.txt").read_bytes() == before
    assert not (run.directory/"fit.npz").exists()


@pytest.fixture
def mock(tmp_path, grism_cal):
    # Real-source loading uses these FITS/config files, not their truth record.
    return make_mock(tmp_path / "simulation", size=21, center=(10.3, 10.7),
                     effective_radius=3., noise_sigma=.02)


def prepare(mock, tmp_path, name="trial", **kwargs):
    settings = dict(center=(10.3, 10.7), inclination_mean=60., inclination_sigma=5.,
                    start_velocity=350., start_turnover=3., start_pa=20.,
                    start_offset=(.2, -.2), noise_sigma=(.02, .03), device="cpu")
    settings.update(kwargs)
    return wf.prepare(mock.source_id, data_dir=mock.data_dir,
                      output_root=tmp_path / "runs", run_name=name, **settings)


def test_prepare_records_fitted_inclination_and_conditional_profile_scope(mock, tmp_path):
    run = prepare(mock, tmp_path)
    info = json.loads((run.directory / "run_info.txt").read_text())
    assert info["status"] == "prepared"
    assert info["fixed_center_pixels"] == [10.3, 10.7]
    assert info["inclination_prior_deg"] == {"mean": 60., "sigma": 5.}
    assert info["parameter_names"][-1] == "velocity.inc_v"
    assert len(info["parameter_names"]) == 6
    assert info["noise_sigma"] == {"R": .02, "C": .03}
    assert info["posterior_errors_available"] is False
    assert "profiled" in info["method"]
    assert not (run.directory / "samples.npz").exists()
    assert not hasattr(run.model, "emission")


def test_overwrite_requires_ownership_and_validates_replacement_first(mock, tmp_path):
    folder = tmp_path / "runs" / str(mock.source_id) / "unmarked"
    folder.mkdir(parents=True)
    sentinel = folder / "keep.txt"
    sentinel.write_text("unrelated")
    with pytest.raises(ValueError, match="run record"):
        prepare(mock, tmp_path, "unmarked", overwrite=True)
    assert sentinel.read_text() == "unrelated"

    run = prepare(mock, tmp_path)
    sentinel = run.directory / "keep.txt"
    sentinel.write_text("old valid run")
    with pytest.raises(FileExistsError):
        prepare(mock, tmp_path)
    with pytest.raises(ValueError, match="Starting parameters"):
        prepare(mock, tmp_path, start_velocity=1500., overwrite=True)
    assert sentinel.read_text() == "old valid run"
    replacement = prepare(mock, tmp_path, overwrite=True)
    assert not sentinel.exists()
    assert json.loads((replacement.directory / "run_info.txt").read_text())["status"] == "prepared"


def test_output_symlink_and_source_overlap_are_rejected(mock, tmp_path):
    source = mock.data_dir / f"ID{mock.source_id}"
    output = tmp_path / "runs" / str(mock.source_id)
    output.mkdir(parents=True)
    (output / "linked").symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        prepare(mock, tmp_path, "linked", overwrite=True)
    with pytest.raises(ValueError, match="separate"):
        wf.prepare(mock.source_id, data_dir=mock.data_dir, output_root=source,
                   run_name="nested", center=(10.3, 10.7), inclination_mean=60.)
    assert (source / "R.fits").is_file()


def test_saved_map_predictions_parameters_and_reopened_plots_agree(mock, tmp_path):
    run = prepare(mock, tmp_path)
    result = wf.fit(run, max_evaluations=2, progress=False)
    with np.load(run.directory / "fit.npz", allow_pickle=False) as saved:
        content = {k: saved[k].copy() for k in saved.files}
    np.testing.assert_array_equal(content["theta"], result["theta"])
    np.testing.assert_array_equal(content["emission"], result["emission"])
    for pupil, predicted in zip(("R", "C"), run.model.predict(content["theta"], content["emission"])):
        np.testing.assert_allclose(content[f"predicted_{pupil}"], predicted.detach().numpy(),
                                   rtol=1e-5, atol=1e-7)
    info = json.loads((run.directory / "run_info.txt").read_text())
    assert info["status"] == "completed"
    assert info["max_evaluations"] == 2
    assert info["fit"]["nfev"] == 2
    assert info["fit"]["success"] is False  # Budget completion is not convergence.
    assert info["posterior_errors_available"] is False

    # Opening from disk must not construct or run a fitting object.
    views = wf.show_results(run.directory)
    try:
        summary = views["summary"]
        assert summary.loc["V_inf", "value"] == content["theta"][0]
        assert summary.loc["inclination", "value"] == pytest.approx(np.rad2deg(content["theta"][5]))
        assert summary.loc["V_at_R_t", "value"] == content["theta"][0] / 2
        image_axes = [a for a in views["residuals"].axes if a.images]
        for row, pupil in enumerate(("R", "C")):
            np.testing.assert_array_equal(image_axes[3*row+1].images[0].get_array(),
                                          content[f"predicted_{pupil}"])
            expected = ((content[f"observed_{pupil}"] - content[f"predicted_{pupil}"])
                        / info["noise_sigma"][pupil])
            np.testing.assert_allclose(image_axes[3*row+2].images[0].get_array(), expected)
        assert (run.directory / "residuals.png").is_file()
        assert (run.directory / "parameter_summary.txt").is_file()
        assert not (run.directory / "corner.png").exists()
    finally:
        plt.close("all")


def test_interrupt_preserves_last_complete_checkpoint(mock, tmp_path, monkeypatch):
    run = prepare(mock, tmp_path)
    result = wf.injection_flexible.profile_map(run.model, run.theta)

    def interrupted(*args, callback, **kwargs):
        callback(result)
        raise KeyboardInterrupt

    monkeypatch.setattr(wf.injection_flexible, "fit", interrupted)
    with pytest.raises(KeyboardInterrupt):
        wf.fit(run, progress=False)
    info = json.loads((run.directory / "run_info.txt").read_text())
    assert info["status"] == "interrupted"
    with np.load(run.directory / "fit.npz") as saved:
        np.testing.assert_array_equal(saved["theta"], result["theta"])
        np.testing.assert_array_equal(saved["emission"], result["emission"])
    views = wf.show_results(run.directory)
    try:
        assert views["summary"].loc["V_inf", "value"] == result["theta"][0]
    finally:
        plt.close("all")


def test_changed_inputs_prevent_fitting_or_new_input_plot(mock, tmp_path):
    run = prepare(mock, tmp_path)
    source = mock.data_dir / f"ID{mock.source_id}"
    with (source / "image.fits").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        wf.fit(run, max_evaluations=1, progress=False)
    with pytest.raises(ValueError, match="changed"):
        wf.show_inputs(run)
    assert not (run.directory / "fit.npz").exists()


@pytest.mark.parametrize("noise", [None, (.02, .03)])
def test_load_run_preserves_fixed_noise_and_settings_without_rewriting(mock, tmp_path, noise):
    run = prepare(mock, tmp_path, noise_sigma=noise)
    before = (run.directory / "run_info.txt").read_bytes()
    reopened = wf.load_run(run.directory)
    assert reopened.model.configuration["direct_model"] == run.model.configuration["direct_model"]
    assert reopened.model.input_records == run.model.input_records
    np.testing.assert_array_equal(reopened.theta, run.theta)
    assert (run.directory / "run_info.txt").read_bytes() == before


def test_replaced_named_folder_rejects_the_old_notebook_handle(mock, tmp_path):
    old = prepare(mock, tmp_path)
    replacement = prepare(mock, tmp_path, center=(10., 10.), overwrite=True)
    before = (replacement.directory / "run_info.txt").read_bytes()
    with pytest.raises(ValueError, match="no longer matches"):
        wf.sample(old, warmup=2, draws=2, overwrite=True)
    with pytest.raises(ValueError, match="no longer matches"):
        wf.fit(old, max_evaluations=1)
    assert (replacement.directory / "run_info.txt").read_bytes() == before
