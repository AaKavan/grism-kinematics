"""A fixed inclination is absent from theta, with exact geometry and provenance."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from dingo import three_file_fixed_i_fit as worker
from dingo.three_file_fit import _json, _npz, _search_config, build_config


FREE_NAMES = ["velocity.V_rot", "velocity.R_v", "velocity.x0_v", "velocity.y0_v",
              "velocity.theta_v", "image.R.dx", "image.R.dy", "noise.ln_f"]


def _snapshot(path):
    return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()}


def _write_parent(parent, config, full, inputs, calibration):
    parent.mkdir(exist_ok=True)
    (parent/"search_config.yaml").write_text(yaml.safe_dump(config))
    if not (parent/"comparison.npz").exists():
        _npz(parent/"comparison.npz", mask=np.ones((3, 3), bool))
    _npz(parent/"candidates.npz", candidate_ids=np.array([0, 1]))
    _npz(parent/"velocity_profile.npz", points=np.asarray([list(full.values())]),
         param_names=np.asarray(list(full)), log_posterior=np.array([0.]))
    identity = dict(source_id=3700, config=_search_config(config), inputs=inputs,
        posterior={key: config["mcmc"][key] for key in ("priors", "noise", "likelihood", "xy_iters", "xy_tol")},
        code={"fitting.py": worker._sha(worker.REPO/"dingo/fitting.py")}, calibration=calibration)
    _json(parent/"search_identity.json", identity)
    trials = []
    for idx in range(2):
        params = copy.deepcopy(full)
        params["velocity.R_v"] += idx
        trials.append(dict(candidate_id=idx, loss=1., log_posterior=float(idx), full_params=params))
    search = dict(source_id=3700, redshift=1.1231, pixel_scale_arcsec=.063, r_fit_pixels=40,
        inclination_prior=config["mcmc"]["priors"]["velocity.inc_v"],
        inclination_metadata=dict(kind="imaging", source_id=3700, note="unchanged imaging information"),
        inclination_settings=dict(inclination_deg=70., inclination_sigma_deg=5.),
        noise_note="shared comparison", display_mask_note="common mask", selected=0,
        search_settings=dict(method="broad"), trials=trials,
        search_identity_sha256=worker._sha(parent/"search_identity.json"),
        **{f"{name}_sha256": worker._sha(parent/f"{name}.npz") for name in ("candidates", "comparison", "velocity_profile")})
    _json(parent/"start_search.json", search)
    return search


@pytest.fixture
def saved(tmp_path, monkeypatch):
    parent, cal = tmp_path/"parent", tmp_path/"cal"
    parent.mkdir(); cal.mkdir()
    (cal/"test.dat").write_text("calibration")
    import dingo.grism
    monkeypatch.setattr(dingo.grism, "get_grism_cal_dir", lambda: cal)
    inputs = {}
    for name in ("image", "R", "C"):
        path = parent/f"{name}.fits"
        path.write_text(name)
        inputs[name] = dict(path=str(path), sha256=worker._sha(path))
    prior = dict(type="gaussian", mu=float(np.deg2rad(70)), sigma=float(np.deg2rad(5)),
                 min=float(np.deg2rad(5)), max=float(np.deg2rad(85)))
    config = dict(summary=dict(name="ID3700"), velocity=dict(V_rot=200., inc_v=prior["mu"]),
        fitting=[dict(override={"velocity.inc_v, velocity.R_v": {"fit": True, "min": 0., "max": 2.}}),
                 dict(override={"velocity.inc_v": {"fit": True, "min": .7, "max": 1.3}})],
        shared_comparison=str(parent/"comparison.npz"), mcmc=dict(
            priors={"velocity.inc_v": prior, "velocity.V_rot":dict(type="uniform", min=0., max=1500.)},
            noise=dict(fit_ln_f=True), likelihood="gaussian", xy_iters=40, xy_tol=.005,
            reseed_stuck=False, init="prior", init_scatter={"velocity.V_rot":2., "velocity.inc_v":.01}))
    full = dict(zip(FREE_NAMES, [550., 4., 40., 40., 1., 1., -1., .2]))
    full["velocity.inc_v"] = float(np.deg2rad(55))
    _write_parent(parent, config, full, inputs, {"test.dat": worker._sha(cal/"test.dat")})
    args = SimpleNamespace(parent_run=parent, output=tmp_path/"fixed", start_velocity=310.,
        start_inclination=30., base_candidate=None, prepare_only=True,
        warmup=1000, draws=8000, walkers=64, seed=42, device="cpu")

    def mock_render(config, parent, full, metadata, source_dir, device, directory):
        params = {name.split(".", 1)[1]: value for name, value in full.items() if name.startswith("velocity.")}
        _npz(directory/"candidates.npz", candidate_ids=np.array([0]),
            rectified_R=np.array([[[params["V_rot"]]]]), rectified_C=np.array([[[params["inc_v"]]]]))
        return [dict(candidate_id=0, full_params=copy.deepcopy(full), params=params,
                     log_posterior=10., loss=1.)], FREE_NAMES

    monkeypatch.setattr(worker, "_render_seed", mock_render)
    return args


def test_preparation_fixes_every_stage_and_preserves_parent(saved, monkeypatch):
    before = _snapshot(saved.parent_run)
    original = json.loads((saved.parent_run/"start_search.json").read_text())
    monkeypatch.setattr(worker, "sample", lambda *a: pytest.fail("prepare-only must not run MCMC"))
    result = worker.run(saved)
    assert result["seed_kind"] == "manual_start"
    assert result["manual_start_schema"] == worker.SCHEMA
    assert result["base_candidate_id"] == 1
    assert result["inclination_mode"] == "fixed"
    assert result["fixed_inclination_deg"] == result["start_inclination_deg"] == 30.
    assert result["parent_inclination_prior"] == original["inclination_prior"]
    assert result["inclination_prior"] == {"type": "fixed", "value": np.deg2rad(30.)}
    assert result["inclination_metadata"]["inherited_imaging_metadata"] == original["inclination_metadata"]
    assert result["param_names"] == FREE_NAMES
    cfg = yaml.safe_load((saved.output/"search_config.yaml").read_text())
    assert cfg["velocity"]["inc_v"] == np.deg2rad(30.)
    for stage in cfg["fitting"]:
        assert stage["override"]["velocity.inc_v"]["fit"] is False
    assert cfg["fitting"][0]["override"]["velocity.R_v"]["fit"] is True
    assert cfg["mcmc"]["init_scatter"] == {"velocity.V_rot": 2.}
    assert cfg["mcmc"]["noise"]["ln_f"] == .2
    assert worker._sha(saved.output/"comparison.npz") == worker._sha(saved.parent_run/"comparison.npz")
    assert _snapshot(saved.parent_run) == before


def test_sampling_handoff_uses_fixed_metadata_and_allows_counts_after_preparation(saved, monkeypatch):
    prepared = worker.run(saved)
    identity = (saved.output/"search_identity.json").read_bytes()
    saved.prepare_only = False
    saved.warmup, saved.draws, saved.walkers, saved.seed = 250, 500, 32, 71
    captured = {}

    def mock_sample(args, config, output, source_dir, inputs, metadata, search, identity, started):
        captured.update(args=args, config=config, metadata=metadata, search=search)
        return dict(param_names=FREE_NAMES, fixed_params={"velocity.inc_v": np.deg2rad(30.)}, status="complete")

    monkeypatch.setattr(worker, "sample", mock_sample)
    result = worker.run(saved)
    assert captured["search"] == prepared
    assert captured["config"]["mcmc"]["nburn"] == 250
    assert captured["config"]["mcmc"]["nsteps"] == 500
    assert captured["config"]["mcmc"]["nwalkers"] == 32
    assert captured["config"]["mcmc"]["seed"] == 71
    assert captured["metadata"]["kind"] == "user_fixed_inclination"
    assert result["fixed_inclination_deg"] == 30.
    assert (saved.output/"search_identity.json").read_bytes() == identity


@pytest.mark.parametrize("setting,value", [("start_velocity", 0.), ("start_velocity", np.nan),
    ("start_inclination", 5.), ("start_inclination", 85.), ("start_inclination", np.nan),
    ("start_inclination", True), ("walkers", 19), ("warmup", -1), ("draws", 0), ("base_candidate", 9)])
def test_invalid_start_leaves_no_new_run(saved, setting, value):
    setattr(saved, setting, value)
    with pytest.raises(ValueError):
        worker.run(saved)
    assert not saved.output.exists()


@pytest.mark.parametrize("setting,value", [("start_velocity", 311.), ("start_inclination", 31.), ("base_candidate", 0)])
def test_changed_start_rejected_before_writes(saved, setting, value):
    worker.run(saved)
    before = _snapshot(saved.output)
    setattr(saved, setting, value)
    with pytest.raises(ValueError, match="different starting values"):
        worker.run(saved)
    assert _snapshot(saved.output) == before


@pytest.mark.parametrize("name", ["start_search.json", "candidates.npz", "comparison.npz"])
def test_corrupted_preparation_rejected(saved, name):
    worker.run(saved)
    with (saved.output/name).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="missing or changed"):
        worker.run(saved)


@pytest.mark.parametrize("location", ["same", "nested", "ancestor"])
def test_output_must_not_overlap_parent(saved, location):
    saved.output = {"same": saved.parent_run, "nested": saved.parent_run/"fixed",
                   "ancestor": saved.parent_run.parent}[location]
    before = _snapshot(saved.parent_run.parent)
    with pytest.raises(ValueError, match="separate posterior directory"):
        worker.run(saved)
    assert _snapshot(saved.parent_run.parent) == before


@pytest.mark.parametrize("wrong", ["sampled", "value"])
def test_bad_sampler_result_rejected(saved, monkeypatch, wrong):
    saved.prepare_only = False
    result = dict(param_names=FREE_NAMES, fixed_params={"velocity.inc_v": np.deg2rad(30.)})
    if wrong == "sampled":
        result["param_names"] = FREE_NAMES + ["velocity.inc_v"]
    else:
        result["fixed_params"]["velocity.inc_v"] = np.deg2rad(31.)
    monkeypatch.setattr(worker, "sample", lambda *a: result)
    with pytest.raises(RuntimeError):
        worker.run(saved)


def test_actual_cpu_fixed_seed_and_short_chain(tmp_path, monkeypatch):
    """Real ID3700 images, 20 walkers and only three retained steps in scratch."""
    import torch
    from dingo.grism import get_grism_cal_dir
    from dingo.three_file_search import save_comparison, SharedComparisonFitter
    monkeypatch.setenv("GRISM_CAL_DIR", str(worker.REPO.parent/"grism_cal"))
    source, parent = worker.REPO/"data/ID3700", tmp_path/"parent"
    parent.mkdir()
    config = build_config(3700, worker.REPO/"data", source/"config_kinematics.yaml", parent,
        inclination_deg=70., inclination_sigma_deg=5., device="cpu", velocity_parameter="V_rot")
    config["shared_comparison"] = str(parent/"comparison.npz")
    config_path = parent/"search_config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    torch.set_num_threads(2)
    save_comparison(config_path, parent/"comparison.npz", torch.device("cpu"))
    base = SharedComparisonFitter(str(config_path), device=torch.device("cpu"))
    state = base.prepare_mcmc()
    full = dict(zip(state["param_names"], map(float, state["theta0"])))
    full.update({"velocity.V_rot": 267., "velocity.R_v": 2.015625,
        "velocity.x0_v": 40.4140625, "velocity.y0_v": 40.84375,
        "velocity.theta_v": 3.6229217, "velocity.inc_v": float(np.deg2rad(53.8778)),
        "image.R.dx": 2.7734375, "image.R.dy": 1.2265625, "noise.ln_f": .17})
    inputs = {name: dict(path=str(source/f"{name}.fits"), sha256=worker._sha(source/f"{name}.fits"))
              for name in ("image", "R", "C")}
    cal = Path(get_grism_cal_dir())
    calibration = {str(p.relative_to(cal)): worker._sha(p) for p in sorted(cal.rglob("*.dat"))}
    _write_parent(parent, config, full, inputs, calibration)
    before = _snapshot(parent)
    args = SimpleNamespace(parent_run=parent, output=tmp_path/"fixed", start_velocity=310.,
        start_inclination=30., base_candidate=0, prepare_only=True,
        warmup=2, draws=3, walkers=20, seed=43, device="cpu")
    prepared = worker.run(args)
    seed = prepared["trials"][0]
    assert seed["full_params"]["velocity.V_rot"] == 310.
    assert seed["full_params"]["velocity.inc_v"] == np.deg2rad(30.)
    assert set(prepared["param_names"]) == set(FREE_NAMES)
    fixed = SharedComparisonFitter(str(args.output/"search_config.yaml"), device=torch.device("cpu"))
    fixed.set_params_dict(seed["full_params"])
    fixed_state = fixed.prepare_mcmc()
    assert set(fixed_state["param_names"]) == set(FREE_NAMES)
    assert "velocity.inc_v" not in dict(fixed_state["priors"])
    for stage in fixed.param_config_lists["velocity"]:
        assert stage["velocity.inc_v"].fit is False
        assert float(stage["velocity.inc_v"].tensor) == pytest.approx(np.deg2rad(30), abs=1e-7)
    theta = fixed_state["theta0"]
    assert fixed.log_probability(theta) == pytest.approx(seed["log_posterior"], abs=.02)
    np.testing.assert_allclose(fixed.log_probability_batch(np.stack([theta, theta])),
                               fixed.log_probability(theta), atol=.02)
    # The fixed-i likelihood is identical to the free model evaluated at the
    # same geometry. Only the inclination-prior factor disappears.
    free_theta = np.asarray([seed["full_params"][n] for n in state["param_names"]])
    free_likelihood = base.log_probability(free_theta) - base.log_prior(free_theta)
    assert seed["log_likelihood"] == pytest.approx(free_likelihood, abs=.02)
    np.testing.assert_array_equal(fixed.mcmc_var_map, base.mcmc_var_map)
    with np.load(args.output/"candidates.npz", allow_pickle=False) as images:
        np.testing.assert_array_equal(images["difference"], images["rectified_R"]-images["rectified_C"])
        np.testing.assert_array_equal(images["mask"], base.mcmc_mask.cpu().numpy())

    args.prepare_only = False
    result = worker.run(args)
    assert result["status"] == "complete"
    assert result["sampling_pass"] is False  # This tiny run tests plumbing only.
    assert set(result["param_names"]) == set(FREE_NAMES)
    assert result["fixed_params"]["velocity.inc_v"] == pytest.approx(np.deg2rad(30), abs=1e-7)
    with np.load(args.output/"samples.npz", allow_pickle=False) as saved_samples:
        assert saved_samples["chain"].shape == (3, 20, 8)
        assert np.isfinite(saved_samples["log_prob"]).all()
    with np.load(args.output/"predictions.npz", allow_pickle=False) as predictions:
        assert "velocity.inc_v" not in predictions["param_names"].tolist()
        fixed.log_probability(predictions["theta"])
        r, c, vr, vc = fixed.get_fitting_results()
        np.testing.assert_allclose(predictions["rectified_R"], r, atol=1e-6)
        np.testing.assert_allclose(predictions["rectified_C"], c, atol=1e-6)
        np.testing.assert_allclose(predictions["velocity_R"], vr, atol=1e-5)
        np.testing.assert_allclose(predictions["velocity_C"], vc, atol=1e-5)
    original_samples = (args.output/"samples.npz").read_bytes()
    worker.run(args)
    assert (args.output/"samples.npz").read_bytes() == original_samples
    assert _snapshot(parent) == before
