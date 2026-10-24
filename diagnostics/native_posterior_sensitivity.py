"""Bounded importance-reweighting check of the turnover-radius prior.

This changes Uniform(r_min,r_max) to LogUniform on the SAME support, with
every other prior and the likelihood unchanged. Weights are proportional to
1/r_turn. This is a sensitivity estimate from existing posterior draws, not
an independently sampled posterior or a coverage experiment.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def weighted_quantiles(values, weights, probabilities=(.05, .5, .95)):
    values = np.asarray(values)
    weights = np.asarray(weights)
    if values.shape[0] != weights.size or np.any(weights <= 0) or not np.isfinite(weights).all():
        raise ValueError("One finite positive weight is required per draw")
    flat = values.reshape(len(weights), -1)
    result = []
    for column in flat.T:
        order = np.argsort(column)
        ordered_weights = weights[order]
        positions = (np.cumsum(ordered_weights)-.5*ordered_weights)/ordered_weights.sum()
        result.append(np.interp(probabilities, positions, column[order]))
    return np.asarray(result).T.reshape(len(probabilities), *values.shape[1:])


def run(output):
    output = Path(output)
    report = json.loads((output/"posterior.json").read_text())
    if not report.get("completed") or not report["diagnostics"]["convergence_gate_passed"]:
        raise ValueError("Finish sampling and pass chain diagnostics before summarizing prior sensitivity")
    if report["prior"]["turnover_prior"] != "uniform":
        raise ValueError("This comparison expects the uniform-turnover baseline")
    with np.load(output/"samples.npz") as saved:
        samples = {k: saved[k] for k in ("v_ref", "r_turn", "inclination")}
    with np.load(output/"profiles.npz") as saved:
        radii = saved["radii_arcsec"]
        profiles = {k: saved[k] for k in saved.files if k != "radii_arcsec"}
    weights = 1/samples["r_turn"]
    kish_ess = weights.sum()**2/np.sum(weights**2)
    result = {"method": "Self-normalized importance reweighting: log-uniform / uniform = constant / r_turn",
        "postprocessing_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "support_arcsec": [report["prior"]["turnover_min"], report["prior"]["turnover_max"]],
        "other_priors_unchanged": True, "kish_weight_ess": float(kish_ess),
        "kish_weight_ess_by_chain": (weights.sum(1)**2/np.sum(weights**2, axis=1)).tolist(),
        "largest_normalized_weight": float(weights.max()/weights.sum()),
        "baseline_min_bulk_ess": report["diagnostics"]["min_bulk_ess"],
        "baseline_min_tail_ess": report["diagnostics"]["min_tail_ess"],
        "note": "Kish weight ESS ignores MCMC autocorrelation. Reweighted intervals are sensitivity estimates, not independently validated posterior intervals; emission-map and inclination priors remain untested.",
        "quantiles_5_50_95": {}, "per_chain_reweighted_quantiles_5_50_95": {}}
    for name, values in (samples | profiles).items():
        flat = values.reshape(-1, *values.shape[2:])
        result["quantiles_5_50_95"][name] = {
            "uniform_turnover": np.quantile(flat, [.05, .5, .95], axis=0).tolist(),
            "log_uniform_turnover": weighted_quantiles(flat, weights.ravel()).tolist()}
        result["per_chain_reweighted_quantiles_5_50_95"][name] = [
            weighted_quantiles(v, w).tolist() for v, w in zip(values, weights)]
    (output/"turnover_prior_sensitivity.json").write_text(json.dumps(result, indent=2))
    (output/"turnover_prior_sensitivity_code.py").write_bytes(Path(__file__).read_bytes())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, name in zip(axes, ("deprojected_velocity_kms", "rotation_only_mass_msun")):
        for key, label in (("uniform_turnover", "Uniform turnover"),
                           ("log_uniform_turnover", "Log-uniform turnover: reweighted")):
            q = np.asarray(result["quantiles_5_50_95"][name][key])
            line, = ax.plot(radii, q[1], label=label)
            ax.fill_between(radii, q[0], q[2], alpha=.15, color=line.get_color())
        v, rt, _ = report["truth_vref_rturn_theta"]
        speed = v*np.arctan(radii/rt)/np.arctan(.5/rt)/np.sin(report["truth_inclination_rad"])
        truth = speed if name == "deprojected_velocity_kms" else (
            radii*speed**2*report["distance"]["kpc_per_arcsec"]/4.30091727003628e-6)
        ax.plot(radii, truth, "k--", label="Injected")
        ax.set(xlabel="Test radius [arcsec]", ylabel=("Deprojected rotation [km/s]" if name.startswith("deprojected")
                                                     else "Rotation-only spherical-equivalent mass [Msun]"))
    axes[0].legend(fontsize=8)
    fig.suptitle("Synthetic R+C: turnover-prior sensitivity; 5–95% pointwise bands")
    fig.tight_layout()
    fig.savefig(output/"turnover_prior_sensitivity.png", dpi=170)
    plt.close(fig)
    print(json.dumps({k: v for k, v in result.items() if "quantiles" not in k}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
