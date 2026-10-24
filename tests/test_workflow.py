"""The notebook API keeps run identity, scientific choices and files separate.

These tests use tiny input files and replace expensive workers. They exercise
orchestration contracts without starting PySersic, an optimizer or MCMC.
"""
from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from astropy.io import fits

from dingo import workflow


def file_hashes(directory):
    """Snapshot content so read-only or rejected actions can be checked."""
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*")
        if path.is_file()
    }


@pytest.fixture
def inputs(tmp_path):
    """Two independent source folders, with no dependence on local survey data."""
    data = tmp_path / "data"
    for source_id in (123, 456):
        directory = data / f"ID{source_id}"
        directory.mkdir(parents=True)
        for name in ("image", "R", "C"):
            fits.writeto(directory / f"{name}.fits", np.ones((21, 21)))
        configuration = {
            "summary": {
                "name": f"ID{source_id}", "z": 1.2, "ra": 64., "dec": -24.,
                "filter": "F444W", "lambda_rest": 1.875, "r_fit": 10,
            },
            "image": {
                "R": {"module": "A", "path": ["R.fits", 0]},
                "C": {"module": "A", "path": ["C.fits", 0]},
            },
        }
        (directory / "config_kinematics.yaml").write_text(yaml.safe_dump(configuration))
    return data


@pytest.fixture
def make_run(tmp_path, inputs):
    def make(source_id=123, **changes):
        options = {
            "working_dir": tmp_path / "working",
            "data_dir": inputs,
            "config_path": inputs / f"ID{source_id}" / "config_kinematics.yaml",
            "inclination_mode": "fixed", "inclination_deg": 30.,
            "device": "cpu",
        }
        options.update(changes)
        return workflow.prepare_source(source_id, **options)
    return make


def saved_search(path, source_id=123, mean=32., sigma=5., run=None):
    """A valid reader-level candidate bundle; workers themselves are mocked."""
    path.mkdir(parents=True, exist_ok=True)
    prior = {
        "type": "gaussian", "mu": float(np.deg2rad(mean)),
        "sigma": float(np.deg2rad(sigma)),
        "min": float(np.deg2rad(5.)), "max": float(np.deg2rad(85.)),
    }
    trials = []
    for candidate_id, velocity in enumerate((350., 550.)):
        params = {
            "V_rot": velocity, "R_v": 2., "inc_v": float(np.deg2rad(mean)),
            "x0_v": 10., "y0_v": 10., "theta_v": .5,
        }
        full = {f"velocity.{key}": value for key, value in params.items()}
        full.update({"image.R.dx": 0., "image.R.dy": 0., "noise.ln_f": 0.})
        trials.append({
            "candidate_id": candidate_id, "params": params, "full_params": full,
            "loss": 10. - candidate_id, "log_posterior": 20. + candidate_id,
        })
    np.savez(
        path / "candidates.npz", candidate_ids=np.array([0, 1]),
        rectified_R=np.ones((2, 21, 21)), rectified_C=np.ones((2, 21, 21)),
        difference=np.zeros((2, 21, 21)), mask=np.ones((21, 21), dtype=bool),
        observed_R=np.ones((21, 21)), observed_C=np.ones((21, 21)),
        image=np.ones((21, 21)),
    )
    report = {
        "schema": "dingo-three-file-candidates-v1", "source_id": source_id,
        "velocity_parameter": "V_rot", "selected": 1, "trials": trials,
        "inclination_prior": prior, "inclination_metadata": None,
        "inclination_settings": {"inclination_deg": mean, "inclination_sigma_deg": sigma},
        "search_settings": {"method": "broad"},
        "candidates_sha256": hashlib.sha256((path / "candidates.npz").read_bytes()).hexdigest(),
    }
    if run is not None:
        record = json.loads((run.directory / "run.json").read_text())
        original = yaml.safe_load(Path(run.configuration["config_path"]).read_text())
        summary = original["summary"]
        # build_config normalizes these fields before recording search identity.
        summary.update(mode="kinematics", unit="pix", r_fit=int(summary.get("r_fit", 10)))
        identity = {
            "source_id": source_id,
            "inputs": {key: record["inputs"][key] for key in ("image", "R", "C")},
            "config": {
                "summary": summary,
                "image": {pupil: {"module": original["image"][pupil]["module"],
                                   "path": [record["inputs"][pupil]["path"], 0],
                                   "dx": 0., "dy": 0.} for pupil in ("R", "C")},
            },
        }
        identity_path = path / "search_identity.json"
        identity_path.write_text(json.dumps(identity))
        report["search_identity_sha256"] = hashlib.sha256(identity_path.read_bytes()).hexdigest()
    (path / "start_search.json").write_text(json.dumps(report))
    config = {
        "velocity": {"V_rot": 200., "inc_v": prior["mu"]},
        "mcmc": {"priors": {
            "velocity.V_rot": {"type": "uniform", "min": 0., "max": 1500.},
            "velocity.inc_v": prior,
        }},
    }
    (path / "search_config.yaml").write_text(yaml.safe_dump(config))
    return report


def fake_sampler(run, calls):
    """Return a mock CLI worker that writes a minimal genuine reader report."""
    def worker(arguments, log_path, *, device):
        calls.append(arguments)
        value = lambda flag: arguments[arguments.index(flag) + 1]
        fixed = "dingo.three_file_fixed_i_fit" in arguments
        parameters = ["velocity.V_rot", "velocity.R_v", "noise.ln_f"]
        if not fixed:
            parameters.append("velocity.inc_v")
        report = {
            "schema": "dingo-three-file-v1", "source_id": run.source_id,
            "intrinsic_dispersion_kms": 0., "velocity_parameter": "V_rot",
            "status": "complete", "draws": int(value("--draws")),
            "warmup": int(value("--warmup")), "walkers": int(value("--walkers")),
            "param_names": parameters, "sampling_pass": False,
            "inclination_mode": "fixed" if fixed else "sampled",
            "fixed_params": {"velocity.inc_v": float(np.deg2rad(float(value("--start-inclination"))))} if fixed else {},
            "base_candidate_id": int(value("--base-candidate")),
            "parent_run": value("--parent-run"),
            "start_velocity_kms": float(value("--start-velocity")),
            "start_inclination_deg": float(value("--start-inclination")),
        }
        report["inclination_prior"] = (
            {"type": "fixed", "value": report["fixed_params"]["velocity.inc_v"]}
            if fixed else json.loads((run.search_directory / "start_search.json").read_text())["inclination_prior"]
        )
        (run.posterior_directory / "report.json").write_text(json.dumps(report))
    return worker


def test_sources_and_experiments_have_independent_run_directories(make_run, tmp_path):
    first = make_run(start_velocity_kms=400.)
    second = make_run(start_velocity_kms=450.)
    other = make_run(456, start_velocity_kms=400.)
    assert first.directory.parent == second.directory.parent == tmp_path / "working" / "123"
    assert other.directory.parent == tmp_path / "working" / "456"
    assert first.directory != second.directory
    assert first.search_directory == first.directory / "_checkpoints" / "search"
    assert first.posterior_directory == first.directory / "_checkpoints" / "posterior"
    assert first.configuration["start_velocity_kms"] == 400.
    assert second.configuration["start_velocity_kms"] == 450.
    with pytest.raises(FrozenInstanceError):
        first.source_id = 456


def test_load_is_read_only_and_does_not_require_observation_files(make_run, inputs, tmp_path, monkeypatch):
    run = make_run(start_velocity_kms=400.)
    before = file_hashes(tmp_path / "working")
    monkeypatch.setattr(workflow, "run_command", lambda *a, **k: pytest.fail("Loading started a worker"))
    for path in inputs.rglob("*.fits"):
        path.unlink()
    reopened = workflow.load_run(run.directory)
    assert reopened.source_id == run.source_id
    assert reopened.configuration == run.configuration
    assert file_hashes(tmp_path / "working") == before


def test_changed_settings_require_explicit_overwrite_and_preserve_siblings(make_run):
    run = make_run(run_name="123_experiment", start_velocity_kms=400.)
    sibling = make_run(run_name="123_other", start_velocity_kms=500.)
    marker = sibling.directory / "keep.txt"
    marker.write_text("A different scientific experiment")
    before = file_hashes(run.directory)
    with pytest.raises(ValueError):
        make_run(run_name="123_experiment", start_velocity_kms=450.)
    assert file_hashes(run.directory) == before
    replaced = make_run(run_name="123_experiment", start_velocity_kms=450., overwrite=True)
    assert replaced.configuration["start_velocity_kms"] == 450.
    assert marker.read_text() == "A different scientific experiment"


def test_overwrite_rejects_unowned_directory(make_run, tmp_path):
    directory = tmp_path / "working" / "123" / "123_unowned"
    directory.mkdir(parents=True)
    marker = directory / "important.txt"
    marker.write_text("Not a workflow output")
    with pytest.raises(ValueError):
        make_run(run_name="123_unowned", overwrite=True)
    assert marker.read_text() == "Not a workflow output"


@pytest.mark.parametrize("name", ["../escape", "/absolute", "has/slash", ""])
def test_invalid_run_name_never_creates_output(make_run, tmp_path, name):
    with pytest.raises(ValueError):
        make_run(run_name=name)
    assert not (tmp_path / "working").exists()


def test_wrong_source_configuration_rejected_without_outputs(make_run, inputs, tmp_path):
    with pytest.raises(ValueError):
        make_run(config_path=inputs / "ID456" / "config_kinematics.yaml")
    assert not (tmp_path / "working").exists()


@pytest.mark.parametrize("changes", [
    {"inclination_deg": 5.}, {"inclination_deg": 85.},
    {"inclination_deg": np.nan}, {"inclination_deg": True},
    {"start_velocity_kms": 0.}, {"start_velocity_kms": np.inf},
    {"start_velocity_kms": True}, {"run_number": 0},
    {"device": "unavailable"},
    {"inclination_mode": "gaussian_prior", "inclination_sigma_deg": 0.},
])
def test_invalid_scientific_settings_fail_before_creating_output(make_run, tmp_path, changes):
    with pytest.raises(ValueError):
        make_run(**changes)
    assert not (tmp_path / "working").exists()


def test_batch_records_failure_and_continues_to_the_next_source(make_run, monkeypatch):
    first = make_run(run_name="123_first")
    second = make_run(456, run_name="456_second")
    calls = []

    def fit(run, **kwargs):
        calls.append((run.source_id, kwargs))
        if run.source_id == 123:
            raise RuntimeError("A worker failed; preserve its checkpoint")
        return "second source completed"

    monkeypatch.setattr(workflow, "fit_source", fit)
    results = workflow.fit_batch(
        {"first": first, "second": second}, candidate_ids={"first": 0, "second": 1},
        warmup=20, draws=100, walkers=32,
    )
    assert [source_id for source_id, _ in calls] == [123, 456]
    assert calls[0][1]["candidate_id"] == 0
    assert calls[1][1]["candidate_id"] == 1
    assert calls[1][1]["draws"] == 100
    assert not results["first"].succeeded
    assert isinstance(results["first"].error, RuntimeError)
    assert results["second"].succeeded
    assert results["second"].result == "second source completed"


def test_batch_respects_keyboard_interrupt(make_run, monkeypatch):
    first = make_run()
    second = make_run(456)
    calls = []

    def interrupted(run, **kwargs):
        calls.append(run.source_id)
        raise KeyboardInterrupt

    monkeypatch.setattr(workflow, "fit_source", interrupted)
    with pytest.raises(KeyboardInterrupt):
        workflow.fit_batch({"first": first, "second": second})
    assert calls == [123]


def test_gaussian_search_keeps_prior_center_distinct_from_start_inclination(make_run, monkeypatch):
    run = make_run(
        inclination_mode="gaussian_prior", inclination_deg=37.,
        inclination_sigma_deg=5., start_inclination_deg=60.,
    )
    calls = []

    def worker(arguments, log_path, *, device):
        calls.append(arguments)
        saved_search(run.search_directory, mean=37., sigma=5., run=run)

    monkeypatch.setattr(workflow, "run_command", worker)
    workflow.find_candidates(run)
    assert len(calls) == 1
    arguments = calls[0]
    assert arguments[:2] == ["-m", "dingo.three_file_search"]
    assert "--search-only" in arguments
    for option, expected in (
        ("--velocity-parameter", "V_rot"),
        ("--inclination-deg", "37.0"),
        ("--inclination-sigma-deg", "5.0"),
        ("--output", str(run.search_directory)),
    ):
        assert arguments[arguments.index(option) + 1] == expected
    assert run.configuration["start_inclination_deg"] == 60.


def test_foreign_source_parent_never_reaches_worker(make_run, tmp_path, monkeypatch):
    parent = tmp_path / "foreign_search"
    saved_search(parent, source_id=456)
    monkeypatch.setattr(workflow, "run_command", lambda *a, **k: pytest.fail("Used another source's search"))
    with pytest.raises(ValueError):
        run = make_run(parent_search=parent, start_velocity_kms=400.)
        workflow.prepare_start(run)


def test_sampling_without_manual_velocity_requires_explicit_candidate(make_run, monkeypatch):
    run = make_run()
    saved_search(run.search_directory, run=run)
    monkeypatch.setattr(workflow, "run_command", lambda *a, **k: pytest.fail("Silently selected a candidate"))
    with pytest.raises(ValueError, match="candidate"):
        workflow.fit_source(run, warmup=10, draws=20, walkers=32)


def test_fixed_sampler_receives_exact_angle_and_manual_velocity(make_run, monkeypatch):
    run = make_run(start_velocity_kms=400., inclination_deg=30.)
    saved_search(run.search_directory, mean=55., run=run)
    calls = []
    monkeypatch.setattr(workflow, "run_command", fake_sampler(run, calls))
    result = workflow.fit_source(run, warmup=0, draws=20, walkers=32, seed=7)
    assert result.fixed_inclination_deg == pytest.approx(30.)
    arguments = calls[0]
    assert arguments[:2] == ["-m", "dingo.three_file_fixed_i_fit"]
    for option, expected in (
        ("--start-velocity", "400.0"), ("--start-inclination", "30.0"),
        ("--base-candidate", "1"), ("--warmup", "0"), ("--draws", "20"),
        ("--walkers", "32"), ("--seed", "7"),
        ("--output", str(run.posterior_directory)),
    ):
        assert arguments[arguments.index(option) + 1] == expected
    assert "--prepare-only" not in arguments
    info = (run.directory / "run_info.txt").read_text()
    assert '"draws": 20' in info
    assert '"completed_draws": 20' in info
    assert '"sampling_pass": false' in info


def test_sampled_inclination_start_does_not_change_prior(make_run, monkeypatch):
    run = make_run(
        inclination_mode="gaussian_prior", inclination_deg=37.,
        start_velocity_kms=400., start_inclination_deg=60.,
    )
    saved_search(run.search_directory, mean=37., run=run)
    before = file_hashes(run.search_directory)
    calls = []
    monkeypatch.setattr(workflow, "run_command", fake_sampler(run, calls))
    workflow.fit_source(run, warmup=10, draws=20, walkers=32)
    arguments = calls[0]
    assert arguments[:2] == ["-m", "dingo.three_file_manual_fit"]
    assert float(arguments[arguments.index("--start-inclination") + 1]) == 60.
    assert file_hashes(run.search_directory) == before
    assert run.configuration["inclination_deg"] == 37.
    assert run.configuration["inclination_sigma_deg"] == 5.


def test_resume_increases_total_draws_but_rejects_changed_sampling_setup(make_run, monkeypatch):
    run = make_run(start_velocity_kms=400.)
    saved_search(run.search_directory, run=run)
    calls = []
    monkeypatch.setattr(workflow, "run_command", fake_sampler(run, calls))
    base = {"warmup": 10, "draws": 20, "walkers": 32, "seed": 7}
    workflow.fit_source(run, **base)
    workflow.fit_source(run, **(base | {"draws": 40}))
    assert len(calls) == 2
    before = file_hashes(run.directory)
    for changes in (
        {"draws": 39}, {"warmup": 11}, {"walkers": 34}, {"seed": 8},
        {"candidate_id": 0},
    ):
        with pytest.raises(ValueError):
            workflow.fit_source(run, **(base | {"draws": 40} | changes))
        assert len(calls) == 2
        assert file_hashes(run.directory) == before


def test_worker_failure_is_recorded_without_discarding_checkpoint(make_run, monkeypatch):
    run = make_run(start_velocity_kms=400.)
    saved_search(run.search_directory, run=run)

    def fail(arguments, log_path, *, device):
        (run.posterior_directory / "checkpoint.npz").write_bytes(b"recoverable checkpoint")
        raise RuntimeError("Worker failure for testing")

    monkeypatch.setattr(workflow, "run_command", fail)
    with pytest.raises(RuntimeError, match="Worker failure"):
        workflow.fit_source(run, warmup=10, draws=20, walkers=32)
    record = json.loads((run.directory / "run.json").read_text())
    assert record["stages"]["sampling"]["status"] == "failed"
    assert record["stages"]["sampling"]["error"]["type"] == "RuntimeError"
    assert record["stages"]["sampling"]["finished_utc"]
    assert (run.posterior_directory / "checkpoint.npz").read_bytes() == b"recoverable checkpoint"
    calls = []
    monkeypatch.setattr(workflow, "run_command", fake_sampler(run, calls))
    workflow.fit_source(run, warmup=10, draws=20, walkers=32)
    resumed = json.loads((run.directory / "run.json").read_text())
    assert resumed["stages"]["sampling"]["status"] == "complete"
    assert resumed["stages"]["sampling"]["attempt"] == 2
    assert (run.posterior_directory / "checkpoint.npz").read_bytes() == b"recoverable checkpoint"


def test_changed_observation_prevents_worker_before_resuming(make_run, inputs, monkeypatch):
    run = make_run(start_velocity_kms=400.)
    saved_search(run.search_directory, run=run)
    (inputs / "ID123" / "R.fits").write_bytes(b"changed input")
    monkeypatch.setattr(workflow, "run_command", lambda *a, **k: pytest.fail("Used changed observation"))
    with pytest.raises(ValueError, match="input"):
        workflow.fit_source(run, warmup=10, draws=20, walkers=32)


def test_external_parent_gaussian_width_must_match_requested_width(make_run, tmp_path):
    seed = make_run(run_name="123_seed")
    parent = tmp_path / "parent"
    saved_search(parent, mean=37., sigma=10., run=seed)
    before = file_hashes(parent)
    with pytest.raises(ValueError, match="width"):
        make_run(
            run_name="123_child", parent_search=parent,
            inclination_mode="gaussian_prior", inclination_deg=37.,
            inclination_sigma_deg=5., start_velocity_kms=400.,
        )
    assert file_hashes(parent) == before


def test_prepare_start_evaluates_without_sampling_or_changing_parent(make_run, monkeypatch):
    run = make_run(start_velocity_kms=450.)
    saved_search(run.search_directory, run=run)
    before = file_hashes(run.search_directory)
    calls = []

    def render(arguments, log_path, *, device):
        calls.append(arguments)
        saved_search(run.posterior_directory, mean=30.)

    monkeypatch.setattr(workflow, "run_command", render)
    preview = workflow.prepare_start(run, candidate_id=0)
    assert preview.source_id == 123
    assert "--prepare-only" in calls[0]
    assert calls[0][calls[0].index("--base-candidate") + 1] == "0"
    assert calls[0][calls[0].index("--start-velocity") + 1] == "450.0"
    assert file_hashes(run.search_directory) == before
    assert not (run.posterior_directory / "report.json").exists()


def test_attached_imaging_sets_exact_total_five_degree_width_without_running(make_run, tmp_path, monkeypatch):
    run = make_run(inclination_mode="imaging_prior", inclination_deg=None)
    imaging = tmp_path / "accepted_imaging"
    for variant in ("full_pixel", "core_pixel_clean"):
        directory = imaging / variant
        directory.mkdir(parents=True)
        np.savez(directory / "samples.npz", q=np.full((4, 30), .7))
        report = {
            "source_id": 123, "completed": True,
            "diagnostics": {"convergence_gate_passed": True},
            "orientation_parameterization_ok": True, "input_sha256": {},
            "samples_sha256": hashlib.sha256((directory / "samples.npz").read_bytes()).hexdigest(),
        }
        (directory / "imaging.json").write_text(json.dumps(report))
    before = file_hashes(imaging)
    monkeypatch.setattr(workflow, "run_command", lambda *a, **k: pytest.fail("Attaching imaging must not fit"))
    workflow.use_imaging(run, imaging)
    record = json.loads((run.directory / "run.json").read_text())
    prior = record["imaging"]["prior"]
    assert prior["inclination_sigma_deg"] == 5.
    assert prior["total_sigma_deg"] == 5.
    assert prior["kind"] == "imaging_mean_user_width_gaussian"
    assert len(prior["sources"]) == 2
    assert file_hashes(imaging) == before


def test_active_run_lock_blocks_overwrite(make_run):
    import fcntl
    run = make_run(start_velocity_kms=400.)
    before = file_hashes(run.directory)
    lock_path = run.directory.parent / f".{run.directory.name}.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="active"):
            make_run(start_velocity_kms=400., overwrite=True)
    assert file_hashes(run.directory) == before


def test_symlinked_output_cannot_redirect_overwrite(make_run, tmp_path):
    target = tmp_path / "unrelated"
    target.mkdir()
    (target / "important.txt").write_text("Keep this directory")
    parent = tmp_path / "working" / "123"
    parent.mkdir(parents=True)
    (parent / "123_link").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        make_run(run_name="123_link", overwrite=True)
    assert (target / "important.txt").read_text() == "Keep this directory"


def test_manual_start_skips_null_scores_in_parent_candidates(make_run, monkeypatch):
    run = make_run(start_velocity_kms=400.)
    report = saved_search(run.search_directory, run=run)
    report["trials"][1]["log_posterior"] = None
    (run.search_directory / "start_search.json").write_text(json.dumps(report))
    calls = []
    monkeypatch.setattr(workflow, "run_command", fake_sampler(run, calls))
    workflow.fit_source(run, warmup=10, draws=20, walkers=32)
    assert calls[0][calls[0].index("--base-candidate") + 1] == "0"


@pytest.mark.parametrize("change", [
    {"source_id": 456}, {"start_velocity_kms": 450.},
    {"base_candidate_id": 0},
    {"fixed_params": {"velocity.inc_v": float(np.deg2rad(40.))}},
])
def test_loading_rejects_posterior_from_different_source_or_experiment(make_run, monkeypatch, change):
    run = make_run(start_velocity_kms=400.)
    saved_search(run.search_directory, run=run)
    monkeypatch.setattr(workflow, "run_command", fake_sampler(run, []))
    workflow.fit_source(run, warmup=10, draws=20, walkers=32)
    path = run.posterior_directory / "report.json"
    report = json.loads(path.read_text())
    report.update(change)
    path.write_text(json.dumps(report))
    monkeypatch.setattr(workflow, "run_command", lambda *a, **k: pytest.fail("Loading must never fit"))
    with pytest.raises(ValueError):
        workflow.load_results(run)


def test_external_search_is_pinned_and_never_deleted_by_child_overwrite(make_run, tmp_path, monkeypatch):
    seed = make_run(run_name="123_seed")
    parent = tmp_path / "parent"
    saved_search(parent, run=seed)
    before = file_hashes(parent)
    options = {"run_name": "123_child", "parent_search": parent, "start_velocity_kms": 400.}
    child = make_run(**options)
    assert child.search_directory == parent
    make_run(**options, overwrite=True)
    assert file_hashes(parent) == before

    report_path = parent / "start_search.json"
    report = json.loads(report_path.read_text())
    report["trials"][0]["log_posterior"] += 1.
    report_path.write_text(json.dumps(report))
    monkeypatch.setattr(workflow, "run_command", lambda *a, **k: pytest.fail("Used a replaced external search"))
    with pytest.raises(ValueError, match="external search changed"):
        workflow.prepare_start(child)
