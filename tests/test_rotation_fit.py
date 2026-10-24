"""Worker commits, recovery and diagnostic gates for rotation-only inference."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from dingo import rotation_fit as worker


def chunk(path, index, draws=3):
    name = f"draws_{index:05d}.npz"
    worker.atomic_npz(path / name,
        sample_vrot=np.full((2, draws), 100.+index),
        stats_diverging=np.zeros((2, draws), dtype=bool))
    return dict(artifact=name, sha256=worker.sha256(path/name), draws=draws)


def test_checkpoint_pointer_survives_interruption_before_commit(tmp_path, monkeypatch):
    first = chunk(tmp_path, 0)
    worker.save_checkpoint(tmp_path, {"counter": np.array([3, 3])}, [first], "identity", 3)
    original_pointer = json.loads((tmp_path / "checkpoint.json").read_text())
    second = chunk(tmp_path, 1)
    original_writer = worker.write_json

    def interrupted(path, value):
        if path.name == "checkpoint.json":
            raise InterruptedError("Process ended before pointer commit")
        original_writer(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(worker, "write_json", interrupted)
        with pytest.raises(InterruptedError):
            worker.save_checkpoint(tmp_path, {"counter": np.array([6, 6])},
                                   [first, second], "identity", 6)
    assert json.loads((tmp_path / "checkpoint.json").read_text()) == original_pointer
    state, committed = worker.restore_checkpoint(tmp_path, "identity")
    np.testing.assert_array_equal(state["counter"], [3, 3])
    assert committed == [first]
    assert (tmp_path / original_pointer["state"]).exists()

    # Orphan state/draw files do not advance the committed chain or duplicate draws.
    worker.save_checkpoint(tmp_path, {"counter": np.array([6, 6])}, [first, second], "identity", 6)
    state, committed = worker.restore_checkpoint(tmp_path, "identity")
    np.testing.assert_array_equal(state["counter"], [6, 6])
    assert not (tmp_path / original_pointer["state"]).exists()
    samples, stats = worker.collected_chunks(tmp_path, committed)
    np.testing.assert_array_equal(samples["vrot"], [[100]*3+[101]*3]*2)
    assert stats["diverging"].shape == (2, 6)


def test_resume_rejects_identity_or_changed_state_and_draws(tmp_path):
    part = chunk(tmp_path, 0)
    worker.save_checkpoint(tmp_path, {"counter": np.array([3, 3])}, [part], "first", 3)
    with pytest.raises(ValueError, match="target/settings differ"):
        worker.restore_checkpoint(tmp_path, "different")
    (tmp_path / part["artifact"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="posterior chunk changed"):
        worker.restore_checkpoint(tmp_path, "first")
    state_path = tmp_path / json.loads((tmp_path / "checkpoint.json").read_text())["state"]
    state_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="sampler checkpoint"):
        worker.restore_checkpoint(tmp_path, "first")


def independent_chains():
    rng = np.random.default_rng(124)
    shape = (4, 2000)
    samples = dict(vrot=rng.normal(200, 10, shape), inclination=rng.normal(1., .02, shape),
        log_rturn=rng.normal(-2., .1, shape),
        continuum_coefficients=np.exp(rng.normal(-3., .2, (*shape, 2, 2, 3))))
    stats = dict(diverging=np.zeros(shape, dtype=bool), num_steps=np.full(shape, 15),
                 accept_prob=np.full(shape, .9), is_warmup=np.zeros(shape, dtype=bool))
    return samples, stats


def test_gate_accepts_independent_chains_and_rejects_stuck_kinematics():
    samples, stats = independent_chains()
    result = worker.convergence_diagnostics(samples, stats, max_depth=9)
    assert result["convergence_gate_passed"]
    assert "rotation_profile" in result["parameters"] and "vsini" in result["parameters"]
    samples["vrot"] = samples["vrot"] + np.arange(4)[:, None]*100
    result = worker.convergence_diagnostics(samples, stats, max_depth=9)
    assert not result["convergence_gate_passed"]
    assert result["parameters"]["vrot"]["max_rank_rhat"] > 1.01


def test_one_bad_nuisance_coordinate_blocks_pooled_intervals():
    samples, stats = independent_chains()
    samples["continuum_coefficients"][0, :, 1, 1, 2] *= 100
    result = worker.convergence_diagnostics(samples, stats, max_depth=9)
    assert not result["convergence_gate_passed"]
    assert result["parameters"]["vrot"]["max_rank_rhat"] < 1.01
    assert result["parameters"]["continuum_coefficients"]["max_rank_rhat"] > 1.01


@pytest.mark.parametrize("failure", ["warmup", "nonfinite", "missing", "shape"])
def test_gate_rejects_invalid_retained_sampler_statistics(failure):
    samples, stats = independent_chains()
    if failure == "warmup":
        stats["is_warmup"][0, 0] = True
    elif failure == "nonfinite":
        stats["accept_prob"][0, 0] = np.nan
    elif failure == "missing":
        stats.pop("is_warmup")
    else:
        stats["num_steps"] = stats["num_steps"][:, :-1]
    result = worker.convergence_diagnostics(samples, stats, max_depth=9)
    assert not result["convergence_gate_passed"]
    assert result["sampling_input_errors"]


def test_initial_state_write_recovers_if_report_commit_is_interrupted(tmp_path, monkeypatch):
    initial = [{"vrot": np.asarray(200.)}]
    calls = []
    monkeypatch.setattr(worker, "optimize_starts", lambda *a: None)

    def build_initial(*args):
        calls.append(True)
        return initial, [{"chain": 0, "vrot_kms": 200.}]

    monkeypatch.setattr(worker, "chain_initial_values", build_initial)
    report = {"source_id": 21636}
    (tmp_path / "rotation.json").write_text(json.dumps(report))

    def interrupted(*args):
        raise InterruptedError("Initial values file committed before report commit")

    with monkeypatch.context() as patch:
        patch.setattr(worker, "write_json", interrupted)
        with pytest.raises(InterruptedError):
            worker.initialize_chains(None, None, tmp_path, report, 1, 17, 100)
    assert (tmp_path / "initial_values.pkl").exists()
    report = json.loads((tmp_path / "rotation.json").read_text())
    assert "initial_values_sha256" not in report
    recovered = worker.initialize_chains(None, None, tmp_path, report, 1, 17, 100)
    assert len(calls) == 2 and recovered[0]["vrot"] == 200.
    assert report["initial_values_sha256"] == worker.sha256(tmp_path / "initial_values.pkl")
    worker.initialize_chains(None, None, tmp_path, report, 1, 17, 100)
    assert len(calls) == 2  # Committed initial positions load without regeneration.
    (tmp_path / "initial_values.pkl").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Initial chain states"):
        worker.initialize_chains(None, None, tmp_path, report, 1, 17, 100)


def test_chain_initialization_rejects_changed_artifact_or_wrong_source(tmp_path):
    artifact = tmp_path / "fit_0000.npz"
    np.savez(artifact, source_id=3700, parameters=np.zeros(20))
    report = dict(source_id=21636, fit_candidates=[dict(artifact=artifact.name,
        artifact_sha256=worker.sha256(artifact), objective=1.)])
    with pytest.raises(ValueError, match="different source"):
        worker.chain_initial_values(None, report, tmp_path, 1, 18)
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact is missing or changed"):
        worker.chain_initial_values(None, report, tmp_path, 1, 18)


@pytest.mark.parametrize("failure", ["divergence", "too_few_chains"])
def test_gate_checks_sampler_failures_as_well_as_parameter_mixing(failure):
    samples, stats = independent_chains()
    if failure == "divergence":
        stats["diverging"][0, 0] = True
    else:
        samples = {k: v[:2] for k, v in samples.items()}
        stats = {k: v[:2] for k, v in stats.items()}
    assert not worker.convergence_diagnostics(samples, stats, max_depth=9)["convergence_gate_passed"]


def test_tree_depth_saturation_warns_without_vetoing_well_mixed_samples():
    samples, stats = independent_chains()
    stats["num_steps"][:, :40] = 511
    result = worker.convergence_diagnostics(samples, stats, max_depth=9)
    assert result["convergence_gate_passed"]
    assert result["max_depth_fraction"] == pytest.approx(.02)
    assert result["sampling_warnings"]
    assert "efficiency" in result["sampling_warnings"][0]


def test_prediction_uses_sampled_continuum_without_reprofiling():
    import jax.numpy as jnp

    class Model:
        background_sigma = jnp.array([.1, .2])
        engine = SimpleNamespace(
            line_map=SimpleNamespace(east=jnp.array([0., .9]), north=jnp.array([0., 0.])),
            describe=lambda vector, speed: dict(vrot_0p5arcsec_kms=speed))
        renderer = SimpleNamespace(render=lambda vector, speed: (None, jnp.array([.8, .2]), jnp.array([0., speed])))
        continuum_model = SimpleNamespace(continuum=lambda coeff: coeff[:, :, :])

        def components(self, parameters):
            raise AssertionError("A posterior prediction must not profile its continuum")

        def vector_from_sites(self, sites):
            return jnp.array([sites["vrot"]])

        def line_from_sites(self, sites):
            return jnp.full((2, 2, 2), sites["vrot"] / 1000)

    target = SimpleNamespace(source_id=21636, exposures=[SimpleNamespace(science=np.ones((2, 2)),
        error=np.full((2, 2), .2), valid=np.ones((2, 2), dtype=bool)) for _ in range(2)])
    coefficients = np.arange(8).reshape(2, 2, 2)/10
    sites = dict(vrot=np.asarray(200.), continuum_coefficients=coefficients, background_z=np.array([1., 2.]))
    arrays, description = worker.prediction(Model(), target, sites=sites)
    np.testing.assert_allclose(arrays["continuum"], coefficients)
    np.testing.assert_allclose(arrays["continuum_coefficients"], coefficients)
    expected = (1. - .2 - coefficients - np.array([.1, .4])[:, None, None])/.2
    np.testing.assert_allclose(arrays["residual"], expected, atol=5e-7)
    assert description["vrot_0p5arcsec_kms"] == 200.
    assert description["source_edge_flux_fraction"] == pytest.approx(.2)
