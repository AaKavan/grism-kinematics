"""A sampled-inclination alternative must never replace its search or scan."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dingo import inclination_posterior as posterior


def hashes(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob("*") if p.is_file()}


@pytest.fixture
def saved_parent(tmp_path, monkeypatch):
    results = tmp_path / "results"
    source, parent, scan, child = posterior._locations(15665, "broad_test", 30., 10., results)
    for path in (parent, scan):
        path.mkdir(parents=True)
        (path / "run.log").write_text("Keep this completed run unchanged")
        (path / "saved.npz").write_bytes(b"saved science")
    from dingo import inclination_scan
    monkeypatch.setattr(inclination_scan, "load", lambda *a, **k: SimpleNamespace(angles=np.array([5., 15., 30.])))
    report = dict(schema="dingo-three-file-v1", source_id=15665,
                  intrinsic_dispersion_kms=0., seed_kind="inclination_scan",
                  parent_run=str(parent), seed_inclination_deg=30.,
                  inclination_prior=dict(type="gaussian", mu=float(np.deg2rad(30.)),
                      sigma=float(np.deg2rad(10.)), min=float(np.deg2rad(5.)), max=float(np.deg2rad(85.))),
                  param_names=["velocity.V_sini", "velocity.inc_v", "noise.ln_f"],
                  fixed_params={"velocity.v_sys": 0.})
    def write_report():
        child.mkdir(parents=True, exist_ok=True)
        (child / "report.json").write_text(json.dumps(report))
    return SimpleNamespace(results=results, source=source, parent=parent,
                           scan=scan, child=child, report=report, write=write_report)


def test_read_only_load_needs_no_parent_or_worker(saved_parent, monkeypatch):
    saved_parent.write()
    from dingo import inclination_scan
    monkeypatch.setattr(inclination_scan, "load", lambda *a, **k: pytest.fail("Reading a saved posterior must not reload its parent scan"))
    monkeypatch.setattr(posterior, "run_command", lambda *a, **k: pytest.fail("Loading must never sample"))
    before = hashes(saved_parent.results)
    result = posterior.load(15665, run_name="broad_test", inclination_deg=30., results=saved_parent.results)
    assert result.path == saved_parent.child
    assert hashes(saved_parent.results) == before


@pytest.mark.parametrize("mutation", ["parent", "source", "seed", "kind", "prior_mean", "prior_sigma",
                                      "prior_support", "prior_type", "missing_inclination", "fixed_inclination"])
def test_load_rejects_a_different_scientific_interpretation(saved_parent, mutation):
    report = saved_parent.report
    if mutation == "parent":
        report["parent_run"] = str(saved_parent.parent.with_name("other_run"))
    elif mutation == "source":
        report["source_id"] = 6170
    elif mutation == "seed":
        report["seed_inclination_deg"] = 15.
    elif mutation == "kind":
        report["seed_kind"] = "candidate"
    elif mutation.startswith("prior_"):
        key, value = {"prior_mean": ("mu", np.deg2rad(32.)),
                      "prior_sigma": ("sigma", np.deg2rad(5.)),
                      "prior_support": ("min", np.deg2rad(20.)),
                      "prior_type": ("type", "uniform")}[mutation]
        report["inclination_prior"][key] = value
    elif mutation == "missing_inclination":
        report["param_names"].remove("velocity.inc_v")
    else:
        report["fixed_params"]["velocity.inc_v"] = float(np.deg2rad(30.))
    saved_parent.write()
    with pytest.raises(ValueError):
        posterior.load(15665, run_name="broad_test", inclination_deg=30., results=saved_parent.results)


def test_fit_dispatches_scan_seeded_worker_with_default_ten_degree_prior(saved_parent, monkeypatch, capsys):
    calls = []
    def worker(arguments, log_path, *, device):
        calls.append((arguments, log_path, device))
        saved_parent.write()
    monkeypatch.setattr(posterior, "run_command", worker)
    parent_before, scan_before = hashes(saved_parent.parent), hashes(saved_parent.scan)
    result = posterior.fit(15665, run_name="broad_test", inclination_deg=30.,
                           warmup=20, samples=40, walkers=32, seed=7,
                           device="cpu", results=saved_parent.results)
    args, log, device = calls[0]
    assert len(calls) == 1
    assert args[:2] == ["-m", "dingo.three_file_inclination_fit"]
    values = dict(zip(args[2::2], args[3::2]))
    assert values == {"--parent-run": str(saved_parent.parent), "--scan-dir": str(saved_parent.scan),
                      "--output": str(saved_parent.child), "--inclination-deg": "30.0",
                      "--inclination-sigma-deg": "10.0", "--warmup": "20", "--draws": "40",
                      "--walkers": "32", "--seed": "7", "--device": "cpu"}
    assert device == "cpu" and log == saved_parent.child / "run.log"
    assert result.path == saved_parent.child
    assert hashes(saved_parent.parent) == parent_before
    assert hashes(saved_parent.scan) == scan_before
    assert "Inclination is sampled" in capsys.readouterr().out


def test_override_resets_only_identified_child(saved_parent, monkeypatch):
    saved_parent.write()
    (saved_parent.child / "old_checkpoint.npz").write_bytes(b"old draws")
    (saved_parent.child / "search_identity.json").write_text(json.dumps(dict(
        schema=posterior.SCHEMA, source_id=15665,
        provenance={"parent_run": str(saved_parent.parent)})))
    before = hashes(saved_parent.parent), hashes(saved_parent.scan)
    def worker(*args, **kwargs):
        assert not (saved_parent.child / "old_checkpoint.npz").exists()
        saved_parent.write()
    monkeypatch.setattr(posterior, "run_command", worker)
    posterior.fit(15665, run_name="broad_test", inclination_deg=30., override=True,
                  results=saved_parent.results)
    assert (hashes(saved_parent.parent), hashes(saved_parent.scan)) == before


@pytest.mark.parametrize("identity", [None, {"schema": "other"},
    {"schema": posterior.SCHEMA, "source_id": 6170},
    {"schema": posterior.SCHEMA, "source_id": 15665, "provenance": {"parent_run": "other"}}])
def test_override_refuses_unidentified_or_unrelated_outputs(saved_parent, monkeypatch, identity):
    saved_parent.write()
    if identity is not None:
        (saved_parent.child / "search_identity.json").write_text(json.dumps(identity))
    before = hashes(saved_parent.results)
    monkeypatch.setattr(posterior, "run_command", lambda *a, **k: pytest.fail("Invalid reset must not run"))
    with pytest.raises(ValueError, match="identified inclination-prior run"):
        posterior.fit(15665, run_name="broad_test", inclination_deg=30., override=True,
                      results=saved_parent.results)
    assert hashes(saved_parent.results) == before


def test_missing_scan_angle_fails_before_override_or_worker(saved_parent, monkeypatch):
    _, _, _, child = posterior._locations(15665, "broad_test", 25., 10., saved_parent.results)
    child.mkdir()
    (child / "keep.txt").write_text("unrelated data")
    before = hashes(saved_parent.results)
    monkeypatch.setattr(posterior, "run_command", lambda *a, **k: pytest.fail("Missing scan row must not run"))
    with pytest.raises(ValueError, match="No saved scan row"):
        posterior.fit(15665, run_name="broad_test", inclination_deg=25., override=True,
                      results=saved_parent.results)
    assert hashes(saved_parent.results) == before


def test_alternative_final_folder_is_distinct_and_does_not_create_files(tmp_path):
    working = tmp_path / "working"
    root = posterior.final_root(working, run_name="broad_test", inclination_deg=30.)
    assert root == working / "inclination_prior_runs" / "broad_test_i30_s10"
    assert root / "ID15665" != working / "ID15665"
    other = posterior.final_root(working, run_name="broad_test", inclination_deg=30., inclination_sigma_deg=5.)
    assert other != root
    assert not working.exists()
