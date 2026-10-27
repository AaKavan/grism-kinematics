"""Compare a saved direct-source posterior with injected truth after fitting.

This module never constructs a model, chooses starts, optimizes, or samples.
The simulation truth is read only to label the completed recovery. Inference
should use ``direct_workflow`` with its ordinary R/C input files and explicit
user-supplied priors, exactly as for an observed source.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import numpy as np

from .direct_posterior_review import (
    LABELS, PARAMETERS, UNITS, _check_fixed_draws, _display_theta, _fixed_parameters,
)
from .simple_simulation import load_mock


PARAMETER_NAMES = (
    "velocity.V_rot", "velocity.R_v", "velocity.theta_v",
    "image.R.dx", "image.R.dy", "velocity.inc_v",
)


def _sha256(path):
    """Verify that recorded inference inputs still contain the same bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_inputs(mock, info):
    """Match the fitted R/C observations and configuration to this simulation.

    The display-only image may be a blank placeholder in a blind recovery.
    It is deliberately excluded; its pixels are not part of the likelihood.
    """
    if info.get("source_id") != mock.source_id:
        raise ValueError("Recovery source ID does not match the simulation.")
    if tuple(info.get("parameter_names", ())) != PARAMETER_NAMES:
        raise ValueError("Saved recovery parameter names/order are not the six direct-source parameters.")
    records = info.get("input_records", {})
    for name in ("R.fits", "C.fits", "config_kinematics.yaml"):
        record = records.get(name, {})
        expected = mock.metadata["input_sha256"].get(name)
        if not expected or record.get("sha256") != expected:
            raise ValueError(f"Recovery {name} does not match this simulation.")
        path = Path(record.get("path", ""))
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"Recorded recovery input changed or is missing: {name}.")


def _saved_diagnostics(directory, info, shape):
    """Read existing diagnostics, checking that their draw counts still match.

    Missing or outdated diagnostics are unavailable, never a successful check.
    Calling this helper does not recompute diagnostics or modify their files.
    """
    path = directory / "posterior_diagnostics.json"
    report = (json.loads(path.read_text()) if path.is_file()
              else info.get("mcmc", {}).get("diagnostics", {}))
    if (report.get("chains"), report.get("draws_per_chain")) != tuple(shape):
        return None, "No matching saved convergence diagnostics; inspect the posterior first."
    passed = report.get("convergence_checks_passed")
    if not isinstance(passed, bool):
        return None, "Saved convergence status is unavailable."
    return passed, ("Saved numerical convergence checks passed."
                    if passed else "Saved numerical convergence checks failed; intervals are provisional.")


def compare_recovery(mock_directory, run_directory):
    """Save truth-versus-posterior intervals for one existing direct-source run.

    Parameters
    ----------
    mock_directory : path-like or MockData
        Simulation folder containing ``truth.json`` and ``truth.npz``. Truth
        is opened only here, after the inference call has completed.
    run_directory : path-like or DirectRun
        Direct-source output folder containing ``run_info.txt`` and retained
        ``posterior.npz`` draws. Its R/C/configuration hashes must match the
        simulation. Run ``direct.show_posterior`` first to refresh diagnostics.

    Returns
    -------
    dict
        ``summary`` is a pandas table with truth, median, central 68% and 95%
        endpoints, units and inclusion flags. ``comparison`` is a six-panel
        matplotlib figure. ``info`` records provenance and convergence status.
        All chains enter the quantiles; no favorable chain is selected.

    Notes
    -----
    Writes ``truth_comparison.csv``, ``recovery_comparison.png`` and
    ``recovery_info.json`` inside the run folder. A fixed coordinate has a point
    value but no uncertainty interval or inclusion flag. A truth falling inside
    an interval in this one realization is not a coverage validation. Numerical
    convergence and truth inclusion are reported as separate facts.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    mock = load_mock(getattr(mock_directory, "directory", mock_directory))
    directory = Path(getattr(run_directory, "directory", run_directory)).expanduser().resolve()
    info = json.loads((directory / "run_info.txt").read_text())
    _validate_inputs(mock, info)
    with np.load(directory / "posterior.npz", allow_pickle=False) as saved:
        if "sample_theta" not in saved:
            raise ValueError("No retained posterior draws are available for recovery comparison.")
        theta = saved["sample_theta"]
        if (theta.ndim != 3 or theta.shape[-1] != 6 or not all(theta.shape)
                or not np.isfinite(theta).all()):
            raise ValueError("Recovery draws must have finite, nonempty shape [chain, draw, 6].")
        if ("fingerprint" not in saved
                or str(saved["fingerprint"]) != info.get("mcmc", {}).get("fingerprint")):
            raise ValueError("Saved posterior fingerprint does not match its run record.")
        if "stats_is_warmup" in saved:
            warmup = saved["stats_is_warmup"]
            if warmup.shape != theta.shape[:2] or np.any(warmup):
                raise ValueError("Recovery comparison requires retained draws only, without warmup.")
    fixed = _fixed_parameters(info)
    _check_fixed_draws(theta, fixed)

    values = _display_theta(theta)
    truth = np.array([mock.truth[name] for name in PARAMETER_NAMES], dtype=float)
    truth[[2, 5]] *= 180 / np.pi
    # Put truth on precisely the same periodic PA branch as the displayed draws.
    anchor = np.rad2deg(np.angle(np.mean(np.exp(1j * theta[..., 2])))) % 360
    truth[2] = anchor + (truth[2] - anchor + 180) % 360 - 180
    quantiles = np.quantile(values.reshape(-1, 6), [.025, .16, .5, .84, .975], axis=0)
    rows = []
    for index, (name, unit) in enumerate(zip(PARAMETERS, UNITS)):
        low95, low68, median, high68, high95 = quantiles[:, index]
        is_fixed = index in fixed
        rows.append(dict(
            parameter=name, truth=float(truth[index]), median=float(median), unit=unit,
            status="fixed" if is_fixed else "sampled",
            p16=np.nan if is_fixed else float(low68),
            p84=np.nan if is_fixed else float(high68),
            p2_5=np.nan if is_fixed else float(low95),
            p97_5=np.nan if is_fixed else float(high95),
            truth_in_68=None if is_fixed else bool(low68 <= truth[index] <= high68),
            truth_in_95=None if is_fixed else bool(low95 <= truth[index] <= high95),
        ))
    summary = pd.DataFrame(rows).set_index("parameter")
    passed, diagnostics_note = _saved_diagnostics(directory, info, theta.shape[:2])

    fig, axes = plt.subplots(3, 2, figsize=(10, 7), layout="constrained")
    for index, ax in enumerate(axes.flat):
        row = summary.iloc[index]
        ax.axvline(row["truth"], color="tab:orange", ls="--", label="Injected truth")
        if index in fixed:
            ax.plot(row["median"], 0, "s", color="black", label="Fixed assumption")
        else:
            ax.plot([row["p2_5"], row["p97_5"]], [0, 0], color="black", lw=1, label="95% interval")
            ax.plot([row["p16"], row["p84"]], [0, 0], color="black", lw=5, label="68% interval")
            ax.plot(row["median"], 0, "o", color="black", label="Pooled median")
        ax.set(xlabel=LABELS[index], yticks=[], ylim=(-.4, .4))
        ax.margins(x=.15)
        ax.legend(fontsize=7, loc="upper right")
    status = "checks passed" if passed else "provisional; checks failed or unavailable"
    fig.suptitle(f"One simulated realization: {status}\nOrange = injected truth; black = saved posterior")

    record = dict(
        schema="dingo-direct-recovery-comparison-v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        mock_directory=str(mock.directory), run_directory=str(directory),
        source_id=mock.source_id, posterior_fingerprint=info["mcmc"]["fingerprint"],
        chains=int(theta.shape[0]), draws_per_chain=int(theta.shape[1]),
        convergence_checks_passed=passed, diagnostics_note=diagnostics_note,
        truth_used_for="Post-fit reporting only; this helper performs no inference.",
        interpretation="One realization, not a repeated-realization coverage validation. "
                       "Truth inclusion does not override failed convergence checks.",
        fixed_center_pixels=info.get("fixed_center_pixels"),
        inclination_prior_deg=info.get("inclination_prior_deg"),
        fixed_R_dy=info.get("direct_model", {}).get("fixed_rdy"),
        parameters=[{key: (None if isinstance(value, float) and not np.isfinite(value) else value)
                     for key, value in row.items()} for row in rows],
    )
    summary.to_csv(directory / "truth_comparison.csv")
    fig.savefig(directory / "recovery_comparison.png", dpi=150, bbox_inches="tight")
    (directory / "recovery_info.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(diagnostics_note)
    print("Truth inclusion is for this realization only; it is not a coverage validation.")
    print("Recovery comparison saved in:", directory)
    return dict(summary=summary, comparison=fig, info=record)
