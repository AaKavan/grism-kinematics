"""Recovery reporting uses saved arrays only, never inference or truth starts."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from dingo.direct_recovery import PARAMETER_NAMES, compare_recovery
from dingo.simple_simulation import SCHEMA, SOURCE_ID


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fixture(tmp_path, *, fixed_rdy=None, pa=35.):
    """Minimal saved-file contract: no FITS decoding or source generation."""
    mock = tmp_path / "mock"
    inputs = mock / "inputs" / f"ID{SOURCE_ID}"
    inputs.mkdir(parents=True)
    copied = tmp_path / "blind_inputs" / f"ID{SOURCE_ID}"
    copied.mkdir(parents=True)
    hashes, records = {}, {}
    for name in ("image.fits", "R.fits", "C.fits", "config_kinematics.yaml"):
        (inputs / name).write_bytes(f"synthetic bytes for {name}".encode())
        hashes[name] = _sha(inputs / name)
        (copied / name).write_bytes((inputs / name).read_bytes())
        records[name] = {"path": str(copied / name), "sha256": hashes[name]}
    # The display image is intentionally NOT the true source brightness map.
    (copied / "image.fits").write_bytes(b"blank display placeholder")
    records["image.fits"]["sha256"] = _sha(copied / "image.fits")

    truth_vector = np.array([200., 5., np.deg2rad(pa), 0., 0., np.deg2rad(60.)])
    truth = dict(zip(PARAMETER_NAMES, truth_vector.tolist()))
    truth.update({"velocity.x0_v": 40.3, "velocity.y0_v": 40.7})
    np.savez(mock / "truth.npz", image=np.ones((3, 3)))
    (mock / "truth.json").write_text(json.dumps(dict(
        schema=SCHEMA, source_id=SOURCE_ID, truth=truth, input_sha256=hashes,
        truth_npz_sha256=_sha(mock / "truth.npz"), noise_sigma=.02)))

    run = tmp_path / "run"
    run.mkdir()
    rng = np.random.default_rng(31)
    theta = truth_vector + rng.normal(size=(4, 30, 6)) * [.5, .1, .02, .1, .1, .02]
    theta[..., 2] %= 2 * np.pi
    model = {}
    if fixed_rdy is not None:
        theta[..., 4] = fixed_rdy
        model["fixed_rdy"] = fixed_rdy
    info = dict(source_id=SOURCE_ID, input_records=records, parameter_names=list(PARAMETER_NAMES),
                direct_model=model, fixed_center_pixels=[40.3, 40.7],
                inclination_prior_deg={"mean": 60., "sigma": 5.},
                mcmc={"fingerprint": "fixture-target"})
    (run / "run_info.txt").write_text(json.dumps(info))
    arrays = dict(sample_theta=theta, fingerprint=np.array("fixture-target"),
                  stats_is_warmup=np.zeros((4, 30), dtype=bool))
    np.savez(run / "posterior.npz", **arrays)
    diagnostics = dict(chains=4, draws_per_chain=30, convergence_checks_passed=False)
    (run / "posterior_diagnostics.json").write_text(json.dumps(diagnostics))
    return mock, run, info, arrays


def test_saved_recovery_quantiles_and_convergence_are_separate(tmp_path):
    mock, run, _, arrays = _fixture(tmp_path)
    result = compare_recovery(mock, run)
    try:
        expected = np.quantile(arrays["sample_theta"][..., 0], [.025, .16, .5, .84, .975])
        row = result["summary"].loc["V_inf"]
        np.testing.assert_allclose(row[["p2_5", "p16", "median", "p84", "p97_5"]].astype(float), expected)
        assert row["truth"] == 200.
        assert bool(row["truth_in_68"])
        # Truth is inside these fabricated intervals; this does not erase the failure.
        assert result["info"]["convergence_checks_passed"] is False
        assert "not a repeated-realization coverage validation" in result["info"]["interpretation"]
        assert len(result["comparison"].axes) == 6
        for name in ("truth_comparison.csv", "recovery_comparison.png", "recovery_info.json"):
            assert (run / name).is_file()
        saved = json.loads((run / "recovery_info.json").read_text())
        assert saved["parameters"][0]["truth_in_68"] is True
    finally:
        plt.close("all")


def test_fixed_registration_has_no_invented_uncertainty(tmp_path):
    mock, run, _, _ = _fixture(tmp_path, fixed_rdy=.271)
    result = compare_recovery(mock, run)
    try:
        row = result["summary"].loc["R_dy"]
        assert row["status"] == "fixed"
        assert row["median"] == .271 and row["truth"] == 0.
        assert np.isnan(row["p16"]) and np.isnan(row["p97_5"])
        assert row["truth_in_68"] is None and row["truth_in_95"] is None
        saved = json.loads((run / "recovery_info.json").read_text())
        assert saved["parameters"][4]["p16"] is None
    finally:
        plt.close("all")


def test_pa_truth_and_intervals_share_the_wrap_branch(tmp_path):
    mock, run, _, _ = _fixture(tmp_path, pa=359.8)
    result = compare_recovery(mock, run)
    try:
        row = result["summary"].loc["cutout_PA"]
        assert abs(row["median"] - row["truth"]) < 1.
        assert row["p97_5"] - row["p2_5"] < 10.
        assert result["summary"].loc["inclination", "truth"] == pytest.approx(60.)
    finally:
        plt.close("all")


@pytest.mark.parametrize("changed", ["source_id", "parameter_order", "R_hash", "changed_file",
                                     "fingerprint", "shape", "warmup", "fixed_draws"])
def test_mismatched_or_invalid_saved_recovery_is_rejected(tmp_path, changed):
    mock, run, info, arrays = _fixture(tmp_path)
    if changed == "source_id":
        info["source_id"] = 6170
    elif changed == "parameter_order":
        info["parameter_names"] = list(reversed(info["parameter_names"]))
    elif changed == "R_hash":
        info["input_records"]["R.fits"]["sha256"] = "another noise realization"
    elif changed == "changed_file":
        Path(info["input_records"]["R.fits"]["path"]).write_bytes(b"changed")
    elif changed == "fingerprint":
        arrays["fingerprint"] = np.array("another target")
    elif changed == "shape":
        arrays["sample_theta"] = arrays["sample_theta"][..., :5]
    elif changed == "warmup":
        arrays["stats_is_warmup"][0, 0] = True
    elif changed == "fixed_draws":
        info["direct_model"]["fixed_rdy"] = .271
    (run / "run_info.txt").write_text(json.dumps(info))
    np.savez(run / "posterior.npz", **arrays)
    with pytest.raises(ValueError):
        compare_recovery(mock, run)
    assert not (run / "recovery_info.json").exists()


def test_stale_diagnostics_do_not_claim_convergence(tmp_path):
    mock, run, _, _ = _fixture(tmp_path)
    (run / "posterior_diagnostics.json").write_text(json.dumps(dict(
        chains=4, draws_per_chain=1000, convergence_checks_passed=True)))
    result = compare_recovery(mock, run)
    try:
        assert result["info"]["convergence_checks_passed"] is None
        assert "No matching" in result["info"]["diagnostics_note"]
    finally:
        plt.close("all")
