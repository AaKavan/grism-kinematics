"""Manual starts change initial V/i, not the posterior or its free parameters."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from dingo import three_file_manual_fit as worker
from dingo.three_file_fit import _json, _npz, _search_config, build_config


NAMES = ["velocity.V_rot", "velocity.R_v", "velocity.x0_v", "velocity.y0_v",
         "velocity.theta_v", "velocity.inc_v", "image.R.dx", "image.R.dy", "noise.ln_f"]


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
        inclination_metadata=dict(kind="imaging", source_id=3700, note="unaltered imaging information"),
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
        shared_comparison=str(parent/"comparison.npz"), mcmc=dict(
            priors={"velocity.inc_v": prior, "velocity.V_rot":dict(type="uniform", min=0., max=1500.)},
            noise=dict(fit_ln_f=True), likelihood="gaussian", xy_iters=40, xy_tol=.005,
            reseed_stuck=False, init="prior", init_scatter={"velocity.V_rot":2., "velocity.inc_v":.01}))
    full = dict(zip(NAMES, [550., 4., 40., 40., 1., float(np.deg2rad(55)), 1., -1., .2]))
    _write_parent(parent, config, full, inputs, {"test.dat": worker._sha(cal/"test.dat")})
    args = SimpleNamespace(parent_run=parent, output=tmp_path/"manual", start_velocity=310.,
        start_inclination=43., base_candidate=None, prepare_only=True,
        warmup=1000, draws=8000, walkers=64, seed=42, device="cpu")

    def mock_render(config, parent, full, metadata, source_dir, device, directory):
        params = {name.split(".", 1)[1]: value for name, value in full.items() if name.startswith("velocity.")}
        _npz(directory/"candidates.npz", candidate_ids=np.array([0]),
            rectified_R=np.array([[[params["V_rot"]]]]), rectified_C=np.array([[[params["inc_v"]]]]))
        return [dict(candidate_id=0, full_params=copy.deepcopy(full), params=params,
                     log_posterior=10., loss=1.)], NAMES

    monkeypatch.setattr(worker, "_render_seed", mock_render)
    return args


def test_preparation_changes_only_two_starts_and_preserves_prior_and_parent(saved, monkeypatch):
    before = _snapshot(saved.parent_run)
    original = json.loads((saved.parent_run/"start_search.json").read_text())
    monkeypatch.setattr(worker, "sample", lambda *a: pytest.fail("prepare-only must not run MCMC"))
    result = worker.run(saved)
    assert result["seed_kind"] == "manual_start"
    assert result["base_candidate_id"] == 1  # highest posterior, regardless of parent.selected
    assert result["start_velocity_kms"] == 310.
    assert result["start_inclination_deg"] == 43.
    assert result["inclination_prior"] == original["inclination_prior"]
    assert result["inclination_metadata"] == original["inclination_metadata"]
    expected = copy.deepcopy(original["trials"][1]["full_params"])
    expected.update({"velocity.V_rot":310., "velocity.inc_v":float(np.deg2rad(43.))})
    assert result["trials"][0]["full_params"] == expected
    cfg = yaml.safe_load((saved.output/"search_config.yaml").read_text())
    assert cfg["mcmc"]["noise"]["ln_f"] == expected["noise.ln_f"]
    assert cfg["mcmc"]["init"] == "adam"
    assert cfg["mcmc"]["stage"] == 0
    assert cfg["mcmc"]["reseed_stuck"] is False
    assert cfg["mcmc"]["init_scatter"] == {"velocity.V_rot":2., "velocity.inc_v":.01}
    assert worker._sha(saved.output/"comparison.npz") == worker._sha(saved.parent_run/"comparison.npz")
    assert _snapshot(saved.parent_run) == before


def test_explicit_base_and_mcmc_counts_can_be_chosen_after_preparation(saved, monkeypatch):
    saved.base_candidate = 0
    prepared = worker.run(saved)
    identity_before = (saved.output/"search_identity.json").read_bytes()
    before_parent = _snapshot(saved.parent_run)
    saved.prepare_only = False
    saved.warmup, saved.draws, saved.walkers, saved.seed = 250, 500, 32, 71
    captured = {}

    def mock_sample(args, config, output, source_dir, inputs, metadata, search, identity, started):
        captured.update(args=args, config=config, search=search)
        return dict(param_names=NAMES, status="complete", inclination_prior=search["inclination_prior"])

    monkeypatch.setattr(worker, "sample", mock_sample)
    result = worker.run(saved)
    assert result["base_candidate_id"] == 0
    assert captured["search"] == prepared
    assert captured["config"]["mcmc"]["nburn"] == 250
    assert captured["config"]["mcmc"]["nsteps"] == 500
    assert captured["config"]["mcmc"]["nwalkers"] == 32
    assert captured["config"]["mcmc"]["seed"] == 71
    assert (saved.output/"search_identity.json").read_bytes() == identity_before
    assert _snapshot(saved.parent_run) == before_parent


@pytest.mark.parametrize("setting,value", [
    ("start_velocity", -100.), ("start_velocity", 0.), ("start_velocity",1500.),
    ("start_velocity", np.nan), ("start_velocity", True), ("start_inclination", 5.),
    ("start_inclination", 85.), ("start_inclination", np.inf), ("start_inclination", False),
    ("base_candidate", 1.5), ("base_candidate", True), ("base_candidate", 9),
    ("warmup", -1), ("draws", 0), ("walkers", 19), ("seed", True), ("device", "invalid")])
def test_invalid_values_leave_no_new_run(saved, setting, value):
    setattr(saved, setting, value)
    with pytest.raises(ValueError):
        worker.run(saved)
    assert not saved.output.exists()


@pytest.mark.parametrize("setting,value", [("start_velocity", 311.), ("start_inclination", 44.), ("base_candidate", 0)])
def test_changed_start_rejected_before_existing_files_are_touched(saved, setting, value):
    worker.run(saved)
    before = _snapshot(saved.output)
    setattr(saved, setting, value)
    with pytest.raises(ValueError, match="different starting values"):
        worker.run(saved)
    assert _snapshot(saved.output) == before


@pytest.mark.parametrize("file", ["start_search.json", "candidates.npz", "comparison.npz"])
def test_corrupted_preparation_rejected(saved, file):
    worker.run(saved)
    with (saved.output/file).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="missing or changed"):
        worker.run(saved)


@pytest.mark.parametrize("location", ["same", "nested", "ancestor"])
def test_output_must_not_overlap_parent(saved, location):
    saved.output = {"same":saved.parent_run, "nested":saved.parent_run/"manual",
                    "ancestor":saved.parent_run.parent}[location]
    before = _snapshot(saved.parent_run.parent)
    with pytest.raises(ValueError, match="separate posterior directory"):
        worker.run(saved)
    assert _snapshot(saved.parent_run.parent) == before


def test_nonfinite_seed_is_not_persisted(saved, monkeypatch):
    def reject(*args):
        raise ValueError("nonfinite posterior")
    monkeypatch.setattr(worker, "_render_seed", reject)
    with pytest.raises(ValueError, match="nonfinite posterior"):
        worker.run(saved)
    assert not saved.output.exists()


def test_actual_cpu_seed_and_sampler_handoff_preserve_likelihood_and_free_parameters(tmp_path, monkeypatch):
    """Real ID3700 images, one seed evaluation, zero posterior draws."""
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
    # A finite real-source geometry: the zero-registration construction is
    # solely the comparison reference, not an adequate kinematic seed.
    full.update({"velocity.V_rot":267., "velocity.R_v":2.015625,
        "velocity.x0_v":40.4140625, "velocity.y0_v":40.84375,
        "velocity.theta_v":3.6229217, "velocity.inc_v":float(np.deg2rad(53.8778)),
        "image.R.dx":2.7734375, "image.R.dy":1.2265625})
    full["noise.ln_f"] = .17
    inputs = {name:dict(path=str(source/f"{name}.fits"), sha256=worker._sha(source/f"{name}.fits"))
              for name in ("image", "R", "C")}
    cal = Path(get_grism_cal_dir())
    calibration = {str(p.relative_to(cal)):worker._sha(p) for p in sorted(cal.rglob("*.dat"))}
    _write_parent(parent, config, full, inputs, calibration)
    before = _snapshot(parent)
    args = SimpleNamespace(parent_run=parent, output=tmp_path/"child", start_velocity=310.,
        start_inclination=43., base_candidate=0, prepare_only=True,
        warmup=10, draws=20, walkers=32, seed=43, device="cpu")
    prepared = worker.run(args)
    seed = prepared["trials"][0]
    assert seed["full_params"]["velocity.V_rot"] == 310.
    assert seed["full_params"]["velocity.inc_v"] == np.deg2rad(43.)
    assert seed["full_params"]["noise.ln_f"] == .17
    with np.load(args.output/"candidates.npz", allow_pickle=False) as images:
        np.testing.assert_array_equal(images["difference"], images["rectified_R"]-images["rectified_C"])
        np.testing.assert_array_equal(images["mask"], base.mcmc_mask.cpu().numpy())
        assert images["candidate_ids"].tolist() == [0]
    assert prepared["inclination_prior"] == config["mcmc"]["priors"]["velocity.inc_v"]
    assert set(prepared["param_names"]) == set(NAMES)

    class StopBeforeSampling(Exception):
        pass

    def stop(self):
        setup = self.prepare_mcmc()
        names, theta = setup["param_names"], setup["theta0"]
        assert theta[names.index("velocity.V_rot")] == pytest.approx(310., abs=1e-5)
        assert theta[names.index("velocity.inc_v")] == pytest.approx(np.deg2rad(43), abs=1e-7)
        assert theta[names.index("noise.ln_f")] == .17
        assert self.config["mcmc"]["init"] == "adam"
        assert dict(setup["priors"])["velocity.inc_v"] == config["mcmc"]["priors"]["velocity.inc_v"]
        np.testing.assert_array_equal(self.mcmc_var_map, base.mcmc_var_map)
        assert self.log_probability(theta) == pytest.approx(seed["log_posterior"], abs=.02)
        # Scoring the same pair in both sampler modes agrees; no new noise/mask.
        batch = self.log_probability_batch(np.stack([theta, theta]))
        np.testing.assert_allclose(batch, self.log_probability(theta), atol=.02)
        raise StopBeforeSampling

    monkeypatch.setattr(SharedComparisonFitter, "fit_MCMC", stop)
    args.prepare_only = False
    with pytest.raises(StopBeforeSampling):
        worker.run(args)
    assert not (args.output/"samples.npz").exists()
    # The existing sampler pins warmup once sampling has been initialized.
    args.warmup = 11
    previous = _snapshot(args.output)
    with pytest.raises(ValueError, match="different inputs/settings/code"):
        worker.run(args)
    assert _snapshot(args.output) == previous
    assert _snapshot(parent) == before
