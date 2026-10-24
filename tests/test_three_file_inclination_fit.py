"""Scan-to-posterior handoff: sampled inclination, honest provenance and no draws."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from dingo import three_file_inclination_fit as worker
from dingo.inclination_scan_fit import inclination_log_prior
from dingo.three_file_fit import _search_config


NAMES = ["velocity.V_sini", "velocity.R_v", "velocity.x0_v", "velocity.y0_v",
         "velocity.theta_v", "velocity.inc_v", "image.R.dx", "image.R.dy", "noise.ln_f"]


def _saved_products(parent, scan, config, inputs, calibration, full, *, pixels=None):
    """Write a small fully identified parent and scan, without optimization."""
    parent.mkdir(exist_ok=True); scan.mkdir(exist_ok=True)
    (parent/"search_config.yaml").write_text(yaml.safe_dump(config))
    if not (parent/"comparison.npz").exists():
        worker._npz(parent/"comparison.npz", mask=np.ones((3, 3), bool))
    worker._npz(parent/"candidates.npz", candidate_ids=np.array([0]))
    worker._npz(parent/"velocity_profile.npz", points=np.array([list(full.values())]),
                param_names=np.array(list(full)), log_posterior=np.array([0.]))
    identity = dict(source_id=15665, config=_search_config(config), inputs=inputs,
        posterior={k: config["mcmc"][k] for k in ("priors", "noise", "likelihood", "xy_iters", "xy_tol")},
        code={"fitting.py": worker._sha(worker.REPO/"dingo/fitting.py")}, calibration=calibration)
    worker._json(parent/"search_identity.json", identity)
    prior = config["mcmc"]["priors"]["velocity.inc_v"]
    report = dict(source_id=15665, redshift=1.5, pixel_scale_arcsec=.063, r_fit_pixels=40,
        inclination_prior=prior, inclination_metadata={"kind": "imaging", "sources": ["original context"]},
        inclination_settings={}, noise_note="test", display_mask_note="common mask",
        search_settings=dict(method="broad", profile_radius_pc=4000.), trials=[],
        search_identity_sha256=worker._sha(parent/"search_identity.json"),
        **{f"{n}_sha256": worker._sha(parent/f"{n}.npz") for n in ("candidates", "comparison", "velocity_profile")})
    worker._json(parent/"start_search.json", report)
    names = list(full)
    angles = np.array([30., 40.])
    rows = []
    for i, angle in enumerate(angles):
        point = dict(full, **{"velocity.inc_v": float(np.deg2rad(angle))})
        params = {k.split(".", 1)[1]: v for k, v in point.items() if k.startswith("velocity.")}
        inc_term = inclination_log_prior(np.deg2rad(angle), prior)
        rows.append(dict(candidate_id=i, kind="fixed inclination diagnostic", params=params,
            full_params=point, loss=1., log_prior=inc_term-2., log_likelihood=10.,
            log_posterior=8.+inc_term, log_inclination_prior=inc_term, log_other_prior=-2.,
            velocity_summary=dict(V_inf_kms=params["V_sini"]/np.sin(params["inc_v"]))))
    if pixels is None:
        pixels = {k: np.zeros((3, 3)) for k in ("image", "observed_R", "observed_C")}
        pixels.update(mask=np.ones((3, 3), bool), rectified_R=np.zeros((3, 3)),
                      rectified_C=np.zeros((3, 3)), difference=np.zeros((3, 3)))
    arrays = dict(pixels)
    arrays.update({k: np.stack([pixels[k], pixels[k]]) for k in ("rectified_R", "rectified_C", "difference")})
    arrays.update(candidate_ids=np.arange(2), inclinations_deg=angles,
        points=np.array([[row["full_params"][name] for name in names] for row in rows]),
        param_names=np.array(names), **{k: np.array([r[k] for r in rows]) for k in
            ("log_prior", "log_likelihood", "log_posterior", "log_inclination_prior", "log_other_prior")})
    worker._npz(scan/"inclination_scan.npz", **arrays)
    scan_identity = dict(schema="dingo-inclination-scan-v1", run_dir=str(parent),
        **{k: report[k] for k in ("search_identity_sha256", "comparison_sha256", "candidates_sha256")},
        code={"inclination_scan_fit.py": worker._sha(worker.REPO/"dingo/inclination_scan_fit.py")})
    worker._json(scan/"identity.json", scan_identity)
    scan_report = dict(report, schema="dingo-inclination-scan-v1", status="complete", trials=rows,
        identity=scan_identity, search_run_dir=str(parent),
        parent_search_sha256=worker._sha(parent/"start_search.json"),
        scan_sha256=worker._sha(scan/"inclination_scan.npz"),
        inclination_scan_sha256=worker._sha(scan/"inclination_scan.npz"))
    worker._json(scan/"report.json", scan_report)
    return report


@pytest.fixture
def saved(tmp_path, monkeypatch):
    parent, scan, cal = tmp_path/"parent", tmp_path/"scan", tmp_path/"cal"
    parent.mkdir(); cal.mkdir()
    (cal/"test.dat").write_text("calibration")
    import dingo.grism
    monkeypatch.setattr(dingo.grism, "get_grism_cal_dir", lambda: cal)
    inputs = {}
    for name in ("image", "R", "C"):
        path = parent/f"{name}.fits"; path.write_text(name)
        inputs[name] = dict(path=str(path), sha256=worker._sha(path))
    prior = dict(type="gaussian", mu=float(np.deg2rad(32.1)), sigma=float(np.deg2rad(10.2)),
                 min=float(np.deg2rad(5)), max=float(np.deg2rad(85)))
    config = dict(summary=dict(name="ID15665"), shared_comparison=str(parent/"comparison.npz"),
        mcmc=dict(priors={"velocity.inc_v": prior}, noise=dict(fit_ln_f=True),
                  likelihood="gaussian", xy_iters=40, xy_tol=.005))
    full = dict(zip(NAMES, [270., 4., 40., 40., 1., float(np.deg2rad(30.)), 1., -1., -.1]))
    _saved_products(parent, scan, config, inputs, {"test.dat": worker._sha(cal/"test.dat")}, full)
    return SimpleNamespace(parent_run=parent, scan_dir=scan, output=tmp_path/"new_posterior",
        inclination_deg=30., inclination_sigma_deg=5., scan_inclination_deg=None,
        warmup=1000, draws=4000, walkers=64, seed=42, device="cpu")


def _snapshot(path):
    return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()}


def test_dispatch_changes_only_prior_and_keeps_source_files_unchanged(saved, monkeypatch):
    before = {p: _snapshot(p) for p in (saved.parent_run, saved.scan_dir)}
    captured = {}

    def mock_sample(args, config, output, source_dir, inputs, metadata, search, identity, started):
        captured.update(config=copy.deepcopy(config), metadata=metadata, search=search, args=args)
        return dict(param_names=NAMES, status="complete", inclination_prior=config["mcmc"]["priors"]["velocity.inc_v"],
                    inclination_metadata=metadata, initialization=search)

    monkeypatch.setattr(worker, "sample", mock_sample)
    result = worker.run(saved)
    prior = result["inclination_prior"]
    assert prior["mu"] == pytest.approx(np.deg2rad(30))
    assert prior["sigma"] == pytest.approx(np.deg2rad(5))
    assert prior["min"] == pytest.approx(np.deg2rad(5))
    assert prior["max"] == pytest.approx(np.deg2rad(85))
    assert result["seed_kind"] == "inclination_scan" and result["seed_inclination_deg"] == 30.
    assert result["selection_method"] == "inclination scan initialization with sampled inclination"
    assert result["original_inclination_prior"]["mu"] == pytest.approx(np.deg2rad(32.1))
    metadata = result["inclination_metadata"]
    assert metadata["kind"] == "user_gaussian_inclination_prior"
    assert "sources" not in metadata and metadata["inherited_imaging_metadata"]["sources"]
    assert captured["config"]["mcmc"]["checkpoint"] == str(saved.output/"checkpoint.npz")
    assert captured["search"]["search_settings"]["method"] == "inclination_scan_seed"
    row = captured["search"]["trials"][0]
    assert row["log_prior"] == pytest.approx(-2.)
    assert row["log_likelihood"] == 10.
    assert worker._sha(saved.output/"comparison.npz") == worker._sha(saved.parent_run/"comparison.npz")
    for path, value in before.items():
        assert _snapshot(path) == value
    # A longer production run preserves the child identity and seed.
    identity_before = (saved.output/"search_identity.json").read_bytes()
    saved.draws = 5000
    worker.run(saved)
    assert (saved.output/"search_identity.json").read_bytes() == identity_before
    assert captured["config"]["mcmc"]["nsteps"] == 5000
    child_before = _snapshot(saved.output)
    saved.inclination_sigma_deg = 10.
    with pytest.raises(ValueError, match="different settings, prior or inputs"):
        worker.run(saved)
    assert _snapshot(saved.output) == child_before


@pytest.mark.parametrize("setting,value", [
    ("inclination_sigma_deg", 0.), ("inclination_sigma_deg", np.nan),
    ("inclination_deg", 5.), ("inclination_deg", 85.), ("inclination_deg", np.inf),
    ("warmup", -1), ("draws", 0), ("walkers", 19), ("seed", True),
])
def test_invalid_settings_rejected_before_child_creation(saved, setting, value):
    setattr(saved, setting, value)
    with pytest.raises(ValueError):
        worker.run(saved)
    assert not saved.output.exists()


@pytest.mark.parametrize("location", ["parent", "parent_child", "scan", "scan_child", "ancestor"])
def test_output_cannot_overwrite_or_enclose_sources(saved, location):
    saved.output = {"parent": saved.parent_run, "parent_child": saved.parent_run/"new",
        "scan": saved.scan_dir, "scan_child": saved.scan_dir/"new", "ancestor": saved.parent_run.parent}[location]
    before = _snapshot(saved.parent_run.parent)
    with pytest.raises(ValueError, match="separate posterior directory"):
        worker.run(saved)
    assert _snapshot(saved.parent_run.parent) == before


def test_missing_angle_is_not_silently_rounded_to_nearby_scan(saved):
    saved.scan_inclination_deg = 31.
    with pytest.raises(ValueError, match="No saved scan row"):
        worker.run(saved)
    assert not saved.output.exists()


def test_changed_scan_arrays_are_rejected(saved):
    with (saved.scan_dir/"inclination_scan.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        worker.run(saved)
    assert not saved.output.exists()


def test_scan_row_can_differ_from_prior_mean(saved, monkeypatch):
    def mock_sample(args, config, output, source_dir, inputs, metadata, search, identity, started):
        assert search["trials"][0]["full_params"]["velocity.inc_v"] == pytest.approx(np.deg2rad(40))
        assert config["mcmc"]["priors"]["velocity.inc_v"]["mu"] == pytest.approx(np.deg2rad(30))
        return dict(param_names=NAMES)
    saved.scan_inclination_deg = 40.
    monkeypatch.setattr(worker, "sample", mock_sample)
    assert worker.run(saved)["seed_inclination_deg"] == 40.


def test_actual_cpu_handoff_samples_inclination_with_new_prior_no_chain(tmp_path, monkeypatch):
    """Run genuine fitter preparation and likelihood once, stopping before emcee."""
    import torch
    from astropy.io import fits
    from dingo.three_file_fit import build_config
    from dingo.three_file_search import SharedComparisonFitter, save_comparison
    from dingo.grism import get_grism_cal_dir
    monkeypatch.setenv("GRISM_CAL_DIR", str(worker.REPO.parent/"grism_cal"))
    parent, scan = tmp_path/"parent", tmp_path/"scan"
    parent.mkdir()
    source = worker.REPO/"data/ID15665"
    config = build_config(15665, worker.REPO/"data", source/"config_kinematics.yaml", parent,
        inclination_deg=32.1, inclination_sigma_deg=10.2, device="cpu")
    config["shared_comparison"] = str(parent/"comparison.npz")
    config_path = parent/"search_config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    torch.set_num_threads(2)
    save_comparison(config_path, parent/"comparison.npz", torch.device("cpu"))
    base = SharedComparisonFitter(str(config_path), device=torch.device("cpu"))
    base.set_params_dict({"velocity.inc_v": np.deg2rad(30.)})
    info = base.prepare_mcmc()
    original_score = base.log_probability(info["theta0"])
    full = dict(zip(info["param_names"], map(float, info["theta0"])))
    # Set precisely the scan angle, independent of float32 tensor roundoff.
    full["velocity.inc_v"] = float(np.deg2rad(30.))
    inputs = {n: dict(path=str(source/f"{n}.fits"), sha256=worker._sha(source/f"{n}.fits")) for n in ("image", "R", "C")}
    cal = Path(get_grism_cal_dir())
    calibration = {str(p.relative_to(cal)): worker._sha(p) for p in sorted(cal.rglob("*.dat"))}
    r, c, _, _ = base.get_fitting_results()
    observed_r, observed_c = base.get_true_images()
    pixels = dict(image=fits.getdata(source/"image.fits"), observed_R=observed_r,
        observed_C=observed_c, rectified_R=r, rectified_C=c, difference=r-c,
        mask=base.mcmc_mask.cpu().numpy())
    _saved_products(parent, scan, config, inputs, calibration, full, pixels=pixels)
    args = SimpleNamespace(parent_run=parent, scan_dir=scan, output=tmp_path/"child",
        inclination_deg=30., inclination_sigma_deg=5., warmup=1000, draws=4000,
        walkers=64, seed=42, device="cpu")
    before = {p: _snapshot(p) for p in (parent, scan)}

    class StopBeforeSampling(Exception):
        pass

    def stop(self):
        state = self.prepare_mcmc()
        assert set(state["param_names"]) == set(NAMES)
        idx = state["param_names"].index("velocity.inc_v")
        assert state["theta0"][idx] == pytest.approx(np.deg2rad(30), abs=1e-7)
        np.testing.assert_array_equal(self.mcmc_mask, base.mcmc_mask)
        np.testing.assert_array_equal(self.mcmc_var_map, base.mcmc_var_map)
        prior = dict(state["priors"])["velocity.inc_v"]
        assert prior["mu"] == pytest.approx(np.deg2rad(30))
        assert prior["sigma"] == pytest.approx(np.deg2rad(5))
        theta = state["theta0"].copy()
        theta[idx] = prior["mu"]
        p0 = self.log_prior(theta)
        theta[idx] += prior["sigma"]
        assert self.log_prior(theta)-p0 == pytest.approx(-.5)
        # The same line-images likelihood, with only the old Gaussian term removed.
        score = self.log_probability(state["theta0"])
        old_term = inclination_log_prior(state["theta0"][idx], config["mcmc"]["priors"]["velocity.inc_v"])
        new_term = inclination_log_prior(state["theta0"][idx], prior)
        assert score == pytest.approx(original_score-old_term+new_term, abs=.05)
        raise StopBeforeSampling

    monkeypatch.setattr(SharedComparisonFitter, "fit_MCMC", stop)
    with pytest.raises(StopBeforeSampling):
        worker.run(args)
    assert not (args.output/"samples.npz").exists()
    for path, value in before.items():
        assert _snapshot(path) == value
