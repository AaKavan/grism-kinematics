"""Fixed-inclination mechanics, provenance guards and one actual CPU evaluation."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from dingo import inclination_scan_fit as worker


class ToyFitter:
    names = ["velocity.V_sini", "velocity.R_v", "velocity.inc_v", "velocity.theta_v",
             "velocity.x0_v", "velocity.y0_v", "image.R.dx", "image.R.dy", "noise.ln_f"]
    mcmc_temper, n_mcmc_pix, _mcmc_lnvar_const = .5, 100, 7.

    def __init__(self, *args, **kwargs):
        self.target = np.array([124., 3., .6, 1., 40., 40., 1., -1., 0.])
        limits = [(0., 1500.), (.5, 40.), (np.deg2rad(5), np.deg2rad(85)),
                  (0., 2*np.pi), (10., 70.), (10., 70.), (-10., 10.), (-10., 10.), (-2., 2.)]
        self._mcmc_priors = [(n, dict(type="uniform", min=float(lo), max=float(hi)))
                             for n, (lo, hi) in zip(self.names, limits)]
        self._mcmc_priors[2][1].update(type="gaussian", mu=.6, sigma=.2)

    def prepare_mcmc(self):
        return dict(param_names=self.names, theta0=self.target.copy())

    def log_prior(self, point):
        if any(not spec["min"] <= value <= spec["max"]
               for (_, spec), value in zip(self._mcmc_priors, point)):
            return -np.inf
        return -.5*((point[2]-.6)/.2)**2

    def _chi2_to_lnL(self, chi2, ln_f):
        return self.mcmc_temper*(-.5*chi2*np.exp(-2*ln_f)
                                -self.n_mcmc_pix*ln_f-.5*self._mcmc_lnvar_const)

    def log_likelihood_batch(self, points):
        target = np.tile(self.target, (len(points), 1))
        target[:, 0] = 100.+40.*points[:, 2]
        scale = np.array([25., .5, .1, .1, .5, .5, .25, .25, 1.])
        chi2 = 100.+np.sum(((points[:, :-1]-target[:, :-1])/scale[:-1])**2, axis=1)
        return self._chi2_to_lnL(chi2, points[:, -1])


def test_fixed_inclination_refits_velocity_and_alignment_with_exact_prior_score():
    f = ToyFitter()
    s = worker.FixedInclinationSearch(f, 15., radius_pixels=8.)
    starts = np.tile(f.target, (2, 1))
    starts[:, 0], starts[:, 6:8] = [20., 300.], 0.
    points, scores = s.optimize(starts, iterations=80)
    np.testing.assert_array_equal(points[:, s.inc], np.repeat(np.deg2rad(15.), 2))
    np.testing.assert_allclose(points[:, s.amp], 100.+40.*np.deg2rad(15.), atol=1.5)
    np.testing.assert_allclose(points[:, 6:8], np.tile([1., -1.], (2, 1)), atol=.02)
    expected = f.log_likelihood_batch(points)+np.array([f.log_prior(p) for p in points])
    np.testing.assert_allclose(scores, expected)
    assert np.all(points[:, -1] > 0.)  # Noise is profiled, not held at the seed.
    with pytest.raises(ValueError, match="frees velocity"):
        s.evaluate(points, velocities=np.ones(2))


@pytest.mark.parametrize("grid,iterations", [
    ([], 10), ([np.nan], 10), ([np.inf], 10), ([0.], 10), ([86.], 10),
    ([5.], 0), ([5.], 1.5), ([5.], True),
])
def test_invalid_scan_settings_fail(grid, iterations):
    with pytest.raises(ValueError):
        worker.validate_settings(grid, iterations, dict(min=np.deg2rad(5), max=np.deg2rad(85)))


def test_grid_is_sorted_unique_and_prior_terms_match_fitter_convention():
    prior = dict(type="gaussian", min=np.deg2rad(5), max=np.deg2rad(85), mu=.6, sigma=.2)
    np.testing.assert_array_equal(worker.validate_settings([30, 5, 30, 85], 10, prior), [5, 30, 85])
    assert worker.inclination_log_prior(.4, prior) == pytest.approx(-.5)
    assert worker.inclination_log_prior(.4, dict(type="sin_i")) == pytest.approx(np.log(np.sin(.4)))
    assert worker.inclination_log_prior(.4, dict(type="log_uniform")) == pytest.approx(-np.log(.4))
    assert worker.inclination_log_prior(.4, dict(type="uniform")) == 0.
    assert np.isneginf(worker.inclination_log_prior(0., prior))


def test_checkpoint_resume_preserves_completed_angles_and_polish_cannot_worsen(tmp_path, monkeypatch):
    s = worker.FixedInclinationSearch(ToyFitter(), 10.)
    starts = np.tile(s.theta0, (2, 1))
    starts[:, s.amp] = [30, 300]
    incs = np.array([10., 20.])
    original = s.optimize
    calls = []

    def stop_at_second(starts, **kwargs):
        calls.append(s.fixed_inclination)
        if len(calls) == 2:
            raise InterruptedError("test interrupt")
        return original(starts, **kwargs)

    monkeypatch.setattr(s, "optimize", stop_at_second)
    with pytest.raises(InterruptedError):
        worker.scan(s, starts, incs, iterations=8, output=tmp_path)
    with np.load(tmp_path/"checkpoint.npz") as saved:
        initial_score = saved["log_posterior"][0]
    assert json.loads((tmp_path/"progress.json").read_text())["next_index"] == 1
    resumed_calls = []

    def record(starts, **kwargs):
        resumed_calls.append(s.fixed_inclination)
        return original(starts, **kwargs)

    monkeypatch.setattr(s, "optimize", record)
    points, scores, gains = worker.scan(s, starts, incs, iterations=8, output=tmp_path)
    assert resumed_calls[0] == np.deg2rad(20.)
    assert scores[0] >= initial_score
    assert len(gains) == 2 and min(gains) >= 0
    np.testing.assert_array_equal(points[:, s.inc], np.deg2rad(incs))
    # A completed checkpoint is a no-op, even if final rendering was interrupted.
    monkeypatch.setattr(s, "optimize", lambda *a, **k: pytest.fail("completed scan optimized again"))
    repeated, repeated_scores, _ = worker.scan(s, starts, incs, iterations=8, output=tmp_path)
    np.testing.assert_array_equal(repeated, points)
    np.testing.assert_array_equal(repeated_scores, scores)


@pytest.fixture
def saved_parent(tmp_path, monkeypatch):
    """Minimal hashed parent, using toy arrays only; no original run is altered."""
    parent, repo, cal = tmp_path/"source", tmp_path/"repo", tmp_path/"cal"
    parent.mkdir(); (repo/"dingo").mkdir(parents=True); cal.mkdir()
    for name in ("inclination_scan_fit.py", "three_file_plot.py", "fitting.py"):
        (repo/"dingo"/name).write_text(f"test {name}")
    (cal/"test.dat").write_text("test calibration")
    monkeypatch.setattr(worker, "REPO", repo)
    import dingo.grism
    monkeypatch.setattr(dingo.grism, "get_grism_cal_dir", lambda: cal)
    inputs = {}
    for name in ("image", "R", "C"):
        path = parent/f"{name}.fits"
        path.write_text(name)
        inputs[name] = dict(path=str(path), sha256=worker._sha(path))
    f = ToyFitter()
    s = worker.PosteriorSearch(f)
    points, scores = s.evaluate([f.target])
    worker._npz(parent/"velocity_profile.npz", points=points, log_posterior=scores,
                param_names=np.asarray(f.names))
    worker._npz(parent/"candidates.npz", candidate_ids=np.array([0]))
    worker._npz(parent/"comparison.npz", mask=np.ones((3, 3)))
    prior = dict(f._mcmc_priors)["velocity.inc_v"]
    config = dict(summary=dict(name="ID1"), shared_comparison=str(parent/"comparison.npz"),
                  mcmc=dict(priors={"velocity.inc_v": prior}))
    (parent/"search_config.yaml").write_text(yaml.safe_dump(config))
    identity = dict(source_id=1, config=worker._search_config(config), inputs=inputs,
                    posterior=dict(priors=config["mcmc"]["priors"]),
                    code={"fitting.py": worker._sha(repo/"dingo/fitting.py")},
                    calibration={"test.dat": worker._sha(cal/"test.dat")})
    worker._json(parent/"search_identity.json", identity)
    report = dict(source_id=1, redshift=1., pixel_scale_arcsec=.063, r_fit_pixels=40,
                  inclination_prior=prior, inclination_metadata=None, inclination_settings={},
                  noise_note="toy noise", display_mask_note="toy mask",
                  search_settings=dict(method="broad", profile_radius_pc=4000.),
                  search_identity_sha256=worker._sha(parent/"search_identity.json"),
                  trials=[dict(candidate_id=0, full_params=dict(zip(f.names, points[0])))],
                  **{f"{n}_sha256": worker._sha(parent/f"{n}.npz")
                     for n in ("candidates", "comparison", "velocity_profile")})
    worker._json(parent/"start_search.json", report)
    return parent


def test_parent_guards_detect_changed_inputs_and_configuration(saved_parent):
    report, identity, config = worker.load_source(saved_parent)
    assert report["source_id"] == 1
    path = saved_parent/"R.fits"
    original = path.read_bytes()
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="R input"):
        worker.load_source(saved_parent)
    path.write_bytes(original)
    config["mcmc"]["priors"]["velocity.inc_v"]["sigma"] = 1.
    (saved_parent/"search_config.yaml").write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="posterior configuration"):
        worker.load_source(saved_parent)


def test_run_outputs_scores_reuses_cache_and_keeps_parent_unchanged(saved_parent, tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "SharedComparisonFitter", ToyFitter)

    def render(f, search, points, scores, metadata, source_dir, path, kinds):
        images = np.zeros((len(points), 3, 3))
        worker._npz(path, candidate_ids=np.arange(len(points)), image=images[0],
                    observed_R=images[0], observed_C=images[0], rectified_R=images,
                    rectified_C=images, difference=images, mask=np.ones((3, 3), bool))
        return [dict(candidate_id=i, kind=kind, full_params=dict(zip(f.names, p)),
                     log_posterior=float(score), log_prior=f.log_prior(p),
                     log_likelihood=float(score-f.log_prior(p)))
                for i, (p, score, kind) in enumerate(zip(points, scores, kinds))]

    monkeypatch.setattr(worker, "records_and_pixels", render)
    before = {p.name: p.read_bytes() for p in saved_parent.iterdir()}
    args = SimpleNamespace(run_dir=saved_parent, output=tmp_path/"scan",
                           inclinations=[20., 10.], iterations=8, device="cpu")
    result = worker.run(args)
    assert result["schema"] == worker.SCHEMA and result["status"] == "complete"
    assert result["parent_search_sha256"] == worker._sha(saved_parent/"start_search.json")
    with np.load(args.output/"inclination_scan.npz") as saved:
        np.testing.assert_allclose(saved["log_likelihood"]+saved["log_prior"], saved["log_posterior"])
        np.testing.assert_allclose(saved["log_inclination_prior"]+saved["log_other_prior"], saved["log_prior"])
        assert saved["rectified_R"].shape == (2, 3, 3)
    monkeypatch.setattr(worker, "scan", lambda *a, **k: pytest.fail("cached scan optimized again"))
    assert worker.run(args) == result
    assert {p.name: p.read_bytes() for p in saved_parent.iterdir()} == before
    altered = dict(result, identity={"wrong": True})
    worker._json(args.output/"report.json", altered)
    with pytest.raises(ValueError, match="different cache identity"):
        worker.run(args)
    worker._json(args.output/"report.json", result)
    args.iterations = 9
    with pytest.raises(ValueError, match="different inputs/settings/seeds/code"):
        worker.run(args)


@pytest.mark.parametrize("location", ["same", "child", "parent"])
def test_output_must_be_separate_from_source_tree(saved_parent, location):
    destination = {"same": saved_parent, "child": saved_parent/"scan",
                   "parent": saved_parent.parent}[location]
    with pytest.raises(ValueError, match="separate inclination-scan directory"):
        worker.run(SimpleNamespace(run_dir=saved_parent, output=destination))


def test_saved_map_seed_binds_to_report_checksum(saved_parent):
    report = json.loads((saved_parent/"start_search.json").read_text())
    s = worker.PosteriorSearch(ToyFitter())
    no_map, provenance = worker.seed_points(saved_parent, report, s)
    posterior = dict(source_id=1, status="complete", comparison_sha256=report["comparison_sha256"],
                     initialization=dict(search_identity_sha256=report["search_identity_sha256"]),
                     best_fit=dict(zip(s.names, s.theta0)))
    worker._json(saved_parent/"report.json", posterior)
    with_map, updated = worker.seed_points(saved_parent, report, s)
    assert len(with_map) == len(no_map)+1
    assert updated["files_sha256"]["report.json"] == worker._sha(saved_parent/"report.json")
    posterior["comparison_sha256"] = "wrong"
    worker._json(saved_parent/"report.json", posterior)
    with pytest.raises(ValueError, match="different search"):
        worker.seed_points(saved_parent, report, s)


def test_actual_shared_cpu_model_holds_inclination_and_exact_prior_decomposition(tmp_path, monkeypatch):
    import torch
    from dingo.three_file_fit import REPO, build_config
    from dingo.three_file_search import SharedComparisonFitter, save_comparison
    monkeypatch.setenv("GRISM_CAL_DIR", str(REPO.parent/"grism_cal"))
    config = build_config(15665, REPO/"data", REPO/"data/ID15665/config_kinematics.yaml",
                          tmp_path, inclination_deg=32.1, inclination_sigma_deg=10.2, device="cpu")
    config["shared_comparison"] = str(tmp_path/"comparison.npz")
    path = tmp_path/"config.yaml"
    path.write_text(yaml.safe_dump(config))
    torch.set_num_threads(2)
    save_comparison(path, tmp_path/"comparison.npz", torch.device("cpu"))
    fitter = SharedComparisonFitter(str(path), device=torch.device("cpu"))
    search = worker.FixedInclinationSearch(fitter, 10.)
    points, scores = search.evaluate([search.theta0])
    point = points[0]
    assert point[search.inc] == np.deg2rad(10.)
    assert fitter.log_probability(point) == pytest.approx(scores[0], abs=.05)
    prior = dict(fitter._mcmc_priors)["velocity.inc_v"]
    reference = point.copy()
    reference[search.inc] = prior["mu"]
    # All other priors cancel; test against the actual fitter, not duplicated math.
    assert fitter.log_prior(point)-fitter.log_prior(reference) == pytest.approx(
        worker.inclination_log_prior(point[search.inc], prior))
