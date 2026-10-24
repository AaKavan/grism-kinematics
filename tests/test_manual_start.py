"""Manual starts preserve the posterior, parent science, and read-only review."""
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from dingo import manual_start as manual
from dingo.three_file import CandidateResults, ThreeFileResults


def hashes(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob("*") if p.is_file()}


@pytest.fixture
def saved_parent(tmp_path):
    results = tmp_path / "results"
    location = dict(parent_run="rotation_v_i5_v1", start_velocity_kms=300.,
                    start_inclination_deg=30., results=results)
    source, parent, child = manual._locations(3700, **location)
    parent.mkdir(parents=True)
    sibling = parent.with_name("other_saved_run")
    sibling.mkdir()
    (sibling / "checkpoint.npz").write_bytes(b"Preserve this other source run")
    prior = dict(type="gaussian", mu=float(np.deg2rad(55.)),
                 sigma=float(np.deg2rad(5.)), min=float(np.deg2rad(5.)),
                 max=float(np.deg2rad(85.)))
    config = dict(velocity={"V_rot": 200., "inc_v": float(np.deg2rad(55.))},
                  mcmc={"priors": {"velocity.V_rot": dict(type="uniform", min=0., max=1500.),
                                    "velocity.inc_v": prior}})
    (parent / "search_config.yaml").write_text(yaml.safe_dump(config))
    arrays = dict(candidate_ids=np.array([2, 0]),
                  rectified_R=np.ones((2, 5, 5)), rectified_C=np.ones((2, 5, 5)),
                  difference=np.zeros((2, 5, 5)), display_mask=np.ones((5, 5), bool))
    parent_report = dict(schema="dingo-three-file-candidates-v1", source_id=3700,
                         velocity_parameter="V_rot", selected=2, inclination_prior=prior,
                         search_settings={"method": "broad"}, trials=[
        dict(candidate_id=0, loss=8., log_posterior=5., params=dict(V_rot=400., inc_v=.8, R_v=3.)),
        dict(candidate_id=1, loss=None, params={}),
        dict(candidate_id=2, loss=3., log_posterior=10., params=dict(V_rot=250., inc_v=.6, R_v=2.))])
    for trial in parent_report["trials"]:
        trial["full_params"] = {f"velocity.{k}": v for k, v in trial["params"].items()}

    def save_candidates(path, report, pixels):
        path.mkdir(parents=True, exist_ok=True)
        np.savez(path / "candidates.npz", **pixels)
        report["candidates_sha256"] = hashlib.sha256((path / "candidates.npz").read_bytes()).hexdigest()
        (path / "start_search.json").write_text(json.dumps(report))

    def write_parent():
        save_candidates(parent, parent_report, arrays)

    write_parent()
    start_fields = dict(seed_kind="manual_start", manual_start_schema=manual.SCHEMA,
                        parent_run=str(parent), start_velocity_kms=300.,
                        start_inclination_deg=30., base_candidate_id=2)
    report = dict(schema="dingo-three-file-v1", source_id=3700,
                  intrinsic_dispersion_kms=0., velocity_parameter="V_rot",
                  inclination_prior=prior, param_names=["velocity.V_rot", "velocity.inc_v", "noise.ln_f"],
                  fixed_params={"velocity.v_sys": 0.}, **start_fields)
    start_report = dict(schema="dingo-three-file-candidates-v1", source_id=3700,
                        velocity_parameter="V_rot", selected=0, inclination_prior=prior,
                        search_settings={"method": "manual_start"}, **start_fields,
                        trials=[dict(candidate_id=0, loss=4., log_posterior=8.,
                                     params=dict(V_rot=300., inc_v=float(np.deg2rad(30.)), R_v=2.))])
    start_arrays = {name: (value[:1] if name != "display_mask" else value.copy())
                    for name, value in arrays.items()}
    start_arrays["candidate_ids"] = np.array([0])

    def write_child(path=None, *, posterior=True):
        path = Path(path or child)
        save_candidates(path, start_report, start_arrays)
        if posterior:
            (path / "report.json").write_text(json.dumps(report))
        (path / "search_identity.json").write_text(json.dumps(dict(
            schema=manual.SCHEMA, source_id=3700, provenance={"parent_run": str(parent)})))

    return SimpleNamespace(results=results, source=source, parent=parent, child=child,
                           sibling=sibling, location=location, report=report,
                           start_report=start_report, parent_report=parent_report,
                           config=config, write_parent=write_parent, write=write_child)


def test_locations_separate_manual_starts_and_do_not_write(tmp_path):
    settings = dict(parent_run="old_v_i", start_velocity_kms=275.5,
                    start_inclination_deg=32.25, results=tmp_path / "results")
    source, parent, output = manual._locations(3700, **settings)
    assert parent.name == "old_v_i"
    assert output.name == "old_v_i_manual_v1_v275p5_i32p25_cbest"
    assert parent.parent == output.parent == source.directory
    for changes in (dict(start_velocity_kms=275.6), dict(start_inclination_deg=32.2),
                    dict(base_candidate_id=0), dict(run_label="repeat2")):
        assert manual._locations(3700, **(settings | changes))[2] != output
    assert not source.results.exists()


@pytest.mark.parametrize("changes", [dict(start_velocity_kms=0.), dict(start_velocity_kms=-1.),
    dict(start_velocity_kms=np.nan), dict(start_velocity_kms=np.inf), dict(start_velocity_kms=True),
    dict(start_inclination_deg=5.), dict(start_inclination_deg=85.),
    dict(start_inclination_deg=np.nan), dict(base_candidate_id=-1),
    dict(base_candidate_id=1.5), dict(base_candidate_id=True),
    dict(parent_run="../parent"), dict(run_label="../unsafe")])
def test_invalid_locations_fail_without_creating_files(tmp_path, changes):
    settings = dict(parent_run="parent", start_velocity_kms=300.,
                    start_inclination_deg=30., results=tmp_path / "results")
    with pytest.raises(ValueError):
        manual._locations(3700, **(settings | changes))
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("prepare_only", [False, True])
def test_dispatch_passes_starts_and_budget_without_redefining_prior(saved_parent, monkeypatch, prepare_only):
    calls = []

    def worker(arguments, log_path, *, device):
        calls.append((arguments, log_path, device))
        saved_parent.write(posterior=not prepare_only)

    monkeypatch.setattr(manual, "run_command", worker)
    before = hashes(saved_parent.parent), hashes(saved_parent.sibling)
    function = manual.prepare if prepare_only else manual.fit
    result = function(3700, **saved_parent.location, warmup=20, samples=80,
                      walkers=32, seed=7, device="cpu")
    assert isinstance(result, CandidateResults if prepare_only else ThreeFileResults)
    assert len(calls) == 1
    args, log, device = calls[0]
    assert args[:2] == ["-m", "dingo.three_file_manual_fit"]
    assert ("--prepare-only" in args) == prepare_only
    paired = [a for a in args[2:] if a != "--prepare-only"]
    values = dict(zip(paired[::2], paired[1::2]))
    assert values == {"--parent-run": str(saved_parent.parent), "--output": str(saved_parent.child),
                      "--start-velocity": "300.0", "--start-inclination": "30.0",
                      "--warmup": "20", "--draws": "80", "--walkers": "32",
                      "--seed": "7", "--device": "cpu"}
    assert log == saved_parent.child / "run.log" and device == "cpu"
    assert (hashes(saved_parent.parent), hashes(saved_parent.sibling)) == before
    assert result.report["inclination_prior"]["mu"] == pytest.approx(np.deg2rad(55.))
    assert result.report["start_inclination_deg"] == 30.


def test_explicit_base_candidate_passed_and_checked(saved_parent, monkeypatch):
    calls = []
    settings = saved_parent.location | {"base_candidate_id": 0}
    output = manual._locations(3700, **settings)[2]
    saved_parent.start_report["base_candidate_id"] = 0

    def worker(args, *a, **k):
        calls.append(args)
        saved_parent.write(output, posterior=False)

    monkeypatch.setattr(manual, "run_command", worker)
    manual.prepare(3700, **settings, device="cpu")
    args = calls[0]
    assert args[args.index("--base-candidate") + 1] == "0"
    before = hashes(saved_parent.results)
    with pytest.raises(ValueError):
        manual.prepare(3700, **(saved_parent.location | {"base_candidate_id": 1}), device="cpu")
    assert len(calls) == 1 and hashes(saved_parent.results) == before


@pytest.mark.parametrize("which", ["load", "load_start"])
def test_saved_review_needs_no_parent_and_never_launches_worker(saved_parent, monkeypatch, which):
    saved_parent.write()
    shutil.rmtree(saved_parent.parent)
    before = hashes(saved_parent.results)
    monkeypatch.setattr(manual, "run_command", lambda *a, **k: pytest.fail("Read-only review must never run a worker"))
    result = getattr(manual, which)(3700, **saved_parent.location)
    assert result.path == saved_parent.child
    assert hashes(saved_parent.results) == before


@pytest.mark.parametrize("which", ["load", "load_start"])
@pytest.mark.parametrize("change", ["source", "parent", "kind", "schema", "velocity", "inclination", "base", "base_bool"])
def test_review_rejects_changed_start_interpretation(saved_parent, which, change):
    report = saved_parent.report if which == "load" else saved_parent.start_report
    key, value = {"source": ("source_id", 6170), "parent": ("parent_run", "/different/run"),
                  "kind": ("seed_kind", "candidate"), "schema": ("manual_start_schema", "unknown"),
                  "velocity": ("start_velocity_kms", 301.),
                  "inclination": ("start_inclination_deg", 31.),
                  "base": ("base_candidate_id", -1), "base_bool": ("base_candidate_id", True)}[change]
    report[key] = value
    saved_parent.write()
    before = hashes(saved_parent.results)
    with pytest.raises(ValueError):
        getattr(manual, which)(3700, **saved_parent.location)
    assert hashes(saved_parent.results) == before


@pytest.mark.parametrize("change", ["vsini", "no_inclination", "fixed_inclination", "fixed_velocity"])
def test_loaded_posterior_must_sample_both_v_and_i(saved_parent, change):
    report = saved_parent.report
    if change == "vsini":
        report["param_names"][0] = "velocity.V_sini"
        report["velocity_parameter"] = "V_sini"
    elif change == "no_inclination":
        report["param_names"].remove("velocity.inc_v")
    else:
        name = "velocity.inc_v" if change == "fixed_inclination" else "velocity.V_rot"
        report["fixed_params"][name] = 30.
    saved_parent.write()
    with pytest.raises(ValueError):
        manual.load(3700, **saved_parent.location)


@pytest.mark.parametrize("problem", ["vsini", "velocity_bound", "inclination_bound", "missing_candidate"])
def test_bad_parent_or_start_fails_before_reset(saved_parent, monkeypatch, problem):
    saved_parent.write()
    settings = saved_parent.location.copy()
    if problem == "vsini":
        saved_parent.parent_report["velocity_parameter"] = "V_sini"
        for trial in saved_parent.parent_report["trials"]:
            if "V_rot" in trial["params"]:
                trial["params"]["V_sini"] = trial["params"].pop("V_rot")
        saved_parent.write_parent()
    elif problem == "velocity_bound":
        saved_parent.config["mcmc"]["priors"]["velocity.V_rot"]["max"] = 250.
    elif problem == "inclination_bound":
        saved_parent.config["mcmc"]["priors"]["velocity.inc_v"]["min"] = float(np.deg2rad(40.))
    else:
        settings["base_candidate_id"] = 1
    (saved_parent.parent / "search_config.yaml").write_text(yaml.safe_dump(saved_parent.config))
    before = hashes(saved_parent.results)
    monkeypatch.setattr(manual, "run_command", lambda *a, **k: pytest.fail("Invalid input must not dispatch"))
    with pytest.raises(ValueError):
        manual.prepare(3700, **settings, override=True, device="cpu")
    assert hashes(saved_parent.results) == before


def test_override_resets_only_identified_manual_child(saved_parent, monkeypatch):
    saved_parent.write()
    (saved_parent.child / "old_checkpoint.npz").write_bytes(b"Old draws")
    before = hashes(saved_parent.parent), hashes(saved_parent.sibling)

    def worker(*a, **k):
        assert not (saved_parent.child / "old_checkpoint.npz").exists()
        saved_parent.write()

    monkeypatch.setattr(manual, "run_command", worker)
    manual.fit(3700, **saved_parent.location, override=True, device="cpu")
    assert (hashes(saved_parent.parent), hashes(saved_parent.sibling)) == before


@pytest.mark.parametrize("settings", [dict(warmup=-1), dict(samples=0), dict(walkers=19),
    dict(seed=-1), dict(seed=True), dict(samples=2.5), dict(device="tpu"), dict(override="yes")])
def test_invalid_sampling_settings_fail_without_reset(saved_parent, monkeypatch, settings):
    saved_parent.write()
    before = hashes(saved_parent.results)
    monkeypatch.setattr(manual, "run_command", lambda *a, **k: pytest.fail("Invalid settings must not dispatch"))
    with pytest.raises(ValueError):
        manual.fit(3700, **saved_parent.location, **(dict(override=True, device="cpu") | settings))
    assert hashes(saved_parent.results) == before


@pytest.mark.parametrize("identity", [None, {"schema": "other"},
    {"schema": manual.SCHEMA, "source_id": 6170},
    {"schema": manual.SCHEMA, "source_id": 3700, "provenance": {"parent_run": "/other"}}])
def test_override_refuses_unidentified_or_unrelated_child(saved_parent, monkeypatch, identity):
    saved_parent.write()
    path = saved_parent.child / "search_identity.json"
    if identity is None:
        path.unlink()
    else:
        path.write_text(json.dumps(identity))
    before = hashes(saved_parent.results)
    monkeypatch.setattr(manual, "run_command", lambda *a, **k: pytest.fail("Unsafe reset must not dispatch"))
    with pytest.raises(ValueError):
        manual.prepare(3700, **saved_parent.location, override=True, device="cpu")
    assert hashes(saved_parent.results) == before


def test_final_export_root_keeps_manual_result_separate_without_writing(saved_parent, tmp_path):
    working = tmp_path / "working"
    path = manual.final_root(working, source_id=3700, **saved_parent.location)
    assert path == working / saved_parent.child.name
    assert path / "ID3700" != working / "ID3700"
    assert not working.exists()


def _fixed_child(saved):
    settings = saved.location | {"fix_inclination": True}
    child = manual._locations(3700, **settings)[2]
    angle = settings["start_inclination_deg"]
    for report in (saved.report, saved.start_report):
        report.update(manual_start_schema=manual.FIXED_SCHEMA, inclination_mode="fixed",
                      fixed_inclination_deg=angle,
                      inclination_prior=dict(type="fixed", value=float(np.deg2rad(angle))))
    saved.report["param_names"].remove("velocity.inc_v")
    saved.report["fixed_params"]["velocity.inc_v"] = float(np.deg2rad(angle))
    return settings, child


@pytest.mark.parametrize("prepare_only", [False, True])
def test_fixed_inclination_dispatch_and_reload_keep_free_run(saved_parent, monkeypatch, prepare_only):
    saved_parent.write()
    before = hashes(saved_parent.child), hashes(saved_parent.parent)
    settings, child = _fixed_child(saved_parent)
    assert child.name == saved_parent.child.name + "_fixedi"
    calls = []

    def worker(arguments, log_path, *, device):
        calls.append(arguments)
        saved_parent.write(child, posterior=not prepare_only)
        (child/"search_identity.json").write_text(json.dumps(dict(schema=manual.FIXED_SCHEMA,
            source_id=3700, provenance={"parent_run": str(saved_parent.parent)})))

    monkeypatch.setattr(manual, "run_command", worker)
    result = (manual.prepare if prepare_only else manual.fit)(3700, **settings, device="cpu")
    assert calls[0][:2] == ["-m", "dingo.three_file_fixed_i_fit"]
    assert result.path == child
    assert (hashes(saved_parent.child), hashes(saved_parent.parent)) == before
    if not prepare_only:
        assert "velocity.inc_v" not in result.report["param_names"]
        assert result.report["fixed_params"]["velocity.inc_v"] == pytest.approx(np.deg2rad(30.))
    monkeypatch.setattr(manual, "run_command", lambda *a, **k: pytest.fail("Reading cannot run anything"))
    reread = (manual.load_start if prepare_only else manual.load)(3700, **settings)
    assert reread.path == child


@pytest.mark.parametrize("mutation", ["sampled_angle", "fixed_wrong_angle", "prior_wrong_angle",
                                     "metadata_wrong_angle", "wrong_mode", "free_schema"])
def test_fixed_result_cannot_be_misread_as_requested_condition(saved_parent, mutation):
    settings, child = _fixed_child(saved_parent)
    r = saved_parent.report
    if mutation == "sampled_angle":
        r["param_names"].append("velocity.inc_v")
    elif mutation == "fixed_wrong_angle":
        r["fixed_params"]["velocity.inc_v"] = float(np.deg2rad(31.))
    elif mutation == "prior_wrong_angle":
        r["inclination_prior"]["value"] = float(np.deg2rad(31.))
    elif mutation == "metadata_wrong_angle":
        r["fixed_inclination_deg"] = 31.
    elif mutation == "wrong_mode":
        r["inclination_mode"] = "sampled"
    else:
        r["manual_start_schema"] = manual.SCHEMA
    saved_parent.write(child)
    with pytest.raises(ValueError):
        manual.load(3700, **settings)


@pytest.mark.parametrize("value", ["fixed", 1, None])
def test_fixed_switch_requires_a_boolean(saved_parent, value):
    with pytest.raises(ValueError, match="True or False"):
        manual._locations(3700, **saved_parent.location, fix_inclination=value)


def test_fixed_override_preserves_free_counterpart(saved_parent, monkeypatch):
    saved_parent.write()
    before = hashes(saved_parent.child), hashes(saved_parent.parent)
    settings, child = _fixed_child(saved_parent)
    saved_parent.write(child)
    (child/"search_identity.json").write_text(json.dumps(dict(schema=manual.FIXED_SCHEMA,
        source_id=3700, provenance={"parent_run": str(saved_parent.parent)})))
    (child/"checkpoint.npz").write_bytes(b"replace only this fixed experiment")

    def worker(*a, **k):
        assert not (child/"checkpoint.npz").exists()
        saved_parent.write(child)

    monkeypatch.setattr(manual, "run_command", worker)
    manual.fit(3700, **settings, override=True, device="cpu")
    assert (hashes(saved_parent.child), hashes(saved_parent.parent)) == before
