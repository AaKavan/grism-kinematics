"""Compare joint R+C emission-prior experiments on the same saved mock.

Correlation-length changes use fresh NUTS fits. Scalar hyperprior changes are
explicit importance-reweighting estimates, with overlap diagnostics and chain
identity retained. No prior is selected because it reproduces the injection.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from diagnostics.native_posterior_sensitivity import weighted_quantiles


def lognormal_log_ratio(values, old_log_mean, old_log_sd, new_log_mean, new_log_sd):
    log_values = np.log(values)
    return (np.log(old_log_sd/new_log_sd)
            -.5*((log_values-new_log_mean)/new_log_sd)**2
            +.5*((log_values-old_log_mean)/old_log_sd)**2)


def weight_diagnostics(log_weights):
    weights = np.exp(log_weights-np.max(log_weights))
    kish = weights.sum()**2/np.sum(weights**2)
    return weights, {"kish_weight_ess": float(kish), "total_draws": weights.size,
        "kish_weight_ess_by_chain": (weights.sum(1)**2/np.sum(weights**2, axis=1)).tolist(),
        "weight_fraction_by_chain": (weights.sum(1)/weights.sum()).tolist(),
        "largest_normalized_weight": float(weights.max()/weights.sum()),
        "adequate_empirical_overlap": bool(kish >= .2*weights.size and weights.max()/weights.sum() < .01),
        "note": "Kish ESS ignores MCMC autocorrelation; empirical overlap does not guarantee unsampled tails are covered"}


def load_run(path):
    r = json.loads((path/"posterior.json").read_text())
    with np.load(path/"samples.npz") as a:
        samples = dict(a)
    with np.load(path/"profiles.npz") as a:
        profiles = dict(a)
    return r, samples, profiles


def shape_moments(east, north, flux):
    """Centroid, minor/major second-moment ratio and sky angle; no inclination."""
    flux = flux/flux.sum(axis=-1, keepdims=True)
    x, y = flux @ east, flux @ north
    xx, yy = flux @ (east*east)-x*x, flux @ (north*north)-y*y
    xy = flux @ (east*north)-x*y
    difference = np.sqrt((xx-yy)**2+4*xy**2)
    q = np.sqrt(np.maximum(xx+yy-difference, 0)/(xx+yy+difference))
    return np.stack([x, y, q, .5*np.arctan2(2*xy, xx-yy)], axis=-1)


def summarize_map_shape(samples, prior, quadrature):
    import torch
    from dingo.native_posterior import log_field_basis
    from dingo.native_fitting import BilinearLineMap
    torch.set_num_threads(4)
    size = int(np.sqrt(samples["map_z"].shape[-1]+1))
    grid = BilinearLineMap(size, quadrature, flux_unit=1., dtype=torch.float64)
    basis, _, _ = log_field_basis(size, .63, prior["correlation_arcsec"])
    z = samples["map_z"].reshape(-1, size*size-1)
    sigma, width = (samples[k].ravel() for k in ("map_sigma", "envelope_width"))
    axis = np.linspace(-.63, .63, size)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    log_shape = (z @ basis.T)*sigma[:, None]-(x*x+y*y).ravel()[None, :]/(2*width[:, None]**2)
    controls = np.exp(log_shape-log_shape.max(1, keepdims=True)).reshape(-1, size, size)
    flux = grid(torch.as_tensor(controls)).numpy()
    flux /= flux.sum(1, keepdims=True)
    moments = shape_moments(grid.east.numpy(), grid.north.numpy(), flux)
    covariance = float(np.corrcoef(moments[:, 2], samples["v_ref"].ravel())[0, 1])
    return {"fields": ["centroid_east_arcsec", "centroid_north_arcsec", "second_moment_axis_ratio", "angle_rad"],
            "quantiles_5_50_95": np.quantile(moments, [.05, .5, .95], axis=0).tolist(),
            "axis_ratio_speed_correlation": covariance,
            "note": "Truncated-field brightness moments, including faint wings; this axis ratio is not used as an inclination constraint"}, moments


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    baseline, samples, profiles = load_run(args.baseline)
    if not baseline["completed"] or not baseline["diagnostics"]["convergence_gate_passed"]:
        raise ValueError("The baseline must have completed and passed diagnostics")
    radii = profiles.pop("radii_arcsec")
    v, rt, _ = baseline["truth_vref_rturn_theta"]
    injected_velocity = v*np.arctan(radii/rt)/np.arctan(.5/rt)/np.sin(baseline["truth_inclination_rad"])
    injected_mass = radii*injected_velocity**2*baseline["distance"]["kpc_per_arcsec"]/4.30091727003628e-6
    combined = {k: samples[k] for k in ("v_ref", "r_turn", "inclination", "map_sigma", "envelope_width")}
    combined.update(profiles)
    combined["mass_ratio_to_injection"] = profiles["rotation_only_mass_msun"]/injected_mass
    result = {"baseline": str(args.baseline), "all_inference_joint_R_and_C": True,
        "radii_arcsec": radii.tolist(), "injected_deprojected_velocity_kms": injected_velocity.tolist(),
        "injected_rotation_only_mass_msun": injected_mass.tolist(), "hyperprior_reweighting": {},
        "correlation_reweighting_overlap_only": {}, "fits": [],
        "input_report_sha256": {}, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    p = baseline["prior"]
    source_dir = Path(baseline["source_run"])
    source = json.loads((source_dir/"recovery.json").read_text())
    with np.load(source_dir/"injection.npz") as a:
        truth_shape = shape_moments(a["truth_east"], a["truth_north"], a["truth_grid_flux"])
    result["injected_map_moments"] = truth_shape.tolist()
    shape_draws = {}
    changes = [
        ("amplitude_median1", "map_sigma", p["log_map_sigma_mean"], p["log_map_sigma_sd"], np.log(1.), .5),
        ("amplitude_median4", "map_sigma", p["log_map_sigma_mean"], p["log_map_sigma_sd"], np.log(4.), .5),
        ("amplitude_logsd1", "map_sigma", p["log_map_sigma_mean"], p["log_map_sigma_sd"], p["log_map_sigma_mean"], 1.),
        ("envelope_median0125", "envelope_width", np.log(p["envelope_width_median"]), p["envelope_log_width_sd"], np.log(.125), .5),
        ("envelope_median05", "envelope_width", np.log(p["envelope_width_median"]), p["envelope_log_width_sd"], np.log(.5), .5),
    ]
    for label, site, mean, sd, newmean, newsd in changes:
        weights, check = weight_diagnostics(lognormal_log_ratio(samples[site], mean, sd, newmean, newsd))
        row = {"changed_site": site, "old_log_mean_sd": [mean, sd], "new_log_mean_sd": [newmean, newsd],
               "overlap": check, "quantiles_5_50_95": None, "per_chain_quantiles_5_50_95": None}
        if check["adequate_empirical_overlap"]:
            row["quantiles_5_50_95"] = {name: weighted_quantiles(values.reshape(-1, *values.shape[2:]), weights.ravel()).tolist()
                                                for name, values in combined.items()}
            row["per_chain_quantiles_5_50_95"] = {name: [weighted_quantiles(x, w).tolist() for x, w in zip(values, weights)]
                                                          for name, values in combined.items()}
        result["hyperprior_reweighting"][label] = row
    # With unchanged amplitude and envelope, transporting z preserves each
    # physical map. The Gaussian density ratio then tests support overlap;
    # its constant Jacobian drops out of normalized importance weights.
    from dingo.native_posterior import log_field_basis
    size = int(np.sqrt(samples["map_z"].shape[-1]+1))
    _, _, oldscale = log_field_basis(size, .63, p["correlation_arcsec"])
    for ell in (.05, .20):
        _, _, newscale = log_field_basis(size, .63, ell)
        log_weights = -.5*np.sum(samples["map_z"]**2*((oldscale/newscale)**2-1), axis=-1)
        _, check = weight_diagnostics(log_weights)
        result["correlation_reweighting_overlap_only"][str(ell)] = check
    for path in [args.baseline, *args.runs]:
        report, saved, prof = load_run(path)
        if report["source_run"] != baseline["source_run"] or report["train_exposures"] != baseline["train_exposures"]:
            raise ValueError("Comparison requires identical source data and training exposures")
        if report.get("noiseless", False):
            raise ValueError("Do not mix a noiseless control into a same-data prior comparison")
        if report.get("oracle_known_map", False):
            if report["prior"] != baseline["prior"]:
                raise ValueError("Oracle kinematic priors must match the baseline")
        else:
            for key in baseline["prior"]:
                if key != "correlation_arcsec" and report["prior"][key] != baseline["prior"][key]:
                    raise ValueError(f"Unexpected prior change in full correlation fit: {key}")
            if report["input_sha256"] != baseline["input_sha256"]:
                raise ValueError("Input hashes differ between correlation fits")
        np.testing.assert_array_equal(prof["radii_arcsec"], radii)
        passed = report["completed"] and report["diagnostics"]["convergence_gate_passed"]
        row = {"directory": str(path), "correlation_arcsec": report["prior"]["correlation_arcsec"],
               "oracle_known_map": report.get("oracle_known_map", False), "completed": report["completed"],
               "interpolated_map": report.get("interpolated_map", False),
               "diagnostics": {k: v for k, v in report["diagnostics"].items() if k != "parameters"},
               "quantiles_5_50_95": None, "truth_mass_percentile": None}
        if passed:
            values = {k: saved[k] for k in ("v_ref", "r_turn", "inclination")}
            values.update({k: prof[k] for k in ("deprojected_velocity_kms", "rotation_only_mass_msun")})
            values["mass_ratio_to_injection"] = prof["rotation_only_mass_msun"]/injected_mass
            row["quantiles_5_50_95"] = {k: np.quantile(x.reshape(-1, *x.shape[2:]), [.05, .5, .95], axis=0).tolist() for k, x in values.items()}
            row["truth_mass_percentile"] = (prof["rotation_only_mass_msun"] < injected_mass).mean((0, 1)).tolist()
            if not row["oracle_known_map"]:
                row["emission_shape"], moments = summarize_map_shape(saved, report["prior"], source["arguments"]["quadrature"])
                shape_draws[path.name] = {"v_ref": saved["v_ref"].ravel(), "axis_ratio": moments[:, 2],
                                         "correlation_arcsec": row["correlation_arcsec"]}
        result["fits"].append(row)
        result["input_report_sha256"][str(path/"posterior.json")] = hashlib.sha256((path/"posterior.json").read_bytes()).hexdigest()
    (args.output/"comparison.json").write_text(json.dumps(result, indent=2))
    (args.output/"comparison_code.py").write_bytes(Path(__file__).read_bytes())
    plot(result, args.output)
    plot_shapes(shape_draws, truth_shape, baseline["truth_vref_rturn_theta"][0], args.output)
    for row in result["fits"]:
        print(json.dumps({k: row[k] for k in ("directory", "oracle_known_map", "quantiles_5_50_95")}), flush=True)
    return result


def plot_shapes(draws, truth_shape, truth_speed, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(draws), figsize=(4.4*len(draws), 4), squeeze=False, sharex=True, sharey=True)
    for ax, row in zip(axes[0], draws.values()):
        ax.scatter(row["axis_ratio"][::2], row["v_ref"][::2], s=4, alpha=.2)
        ax.plot(truth_shape[2], truth_speed, marker="*", color="red", ms=13, label="Injected")
        ax.set(xlabel="Emission-map second-moment axis ratio", ylabel="Projected speed at 0.5 arcsec [km/s]",
               title=f"Correlation {row['correlation_arcsec']:.2f} arcsec")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Same joint R+C data: brightness shape and rotation are coupled")
    fig.tight_layout()
    fig.savefig(output/"map_shape_rotation.png", dpi=170)
    plt.close(fig)


def plot(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    radii = np.asarray(result["radii_arcsec"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for row in result["fits"]:
        if row["quantiles_5_50_95"] is None:
            continue
        label = (("Known map: 17/49 grid" if row["interpolated_map"] else "Known map: exact 97 grid")
                 if row["oracle_known_map"] else f"Map correlation {row['correlation_arcsec']:.2f} arcsec")
        for ax, key in zip(axes, ("deprojected_velocity_kms", "mass_ratio_to_injection")):
            q = np.asarray(row["quantiles_5_50_95"][key])
            line, = ax.plot(radii, q[1], label=label)
            ax.fill_between(radii, q[0], q[2], color=line.get_color(), alpha=.12)
    axes[0].plot(radii, result["injected_deprojected_velocity_kms"], "k--", label="Injected")
    axes[1].axhline(1., color="k", linestyle="--")
    axes[0].set(ylabel="Deprojected rotation [km/s]", xlabel="Test radius [arcsec]")
    axes[1].set(ylabel="Rotation-only mass / injected mass", xlabel="Test radius [arcsec]")
    axes[0].legend(fontsize=8)
    fig.suptitle("Same synthetic R+C data: emission-map sensitivity; central 90% pointwise bands")
    fig.tight_layout()
    fig.savefig(output/"map_prior_profiles.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
