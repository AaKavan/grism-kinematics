"""Inspect saved joint direct R+C posterior draws without rerunning a fit.

The corner plot and rotation band pool all retained chains. Residual images
come from one actual joint draw of kinematics AND emission; independently
taking the median of each object would generally not describe a sampled fit.
"""
from pathlib import Path
import json
import warnings

import numpy as np


PARAMETERS = ("V_inf", "R_t", "cutout_PA", "R_dx", "R_dy", "inclination")
UNITS = ("km/s", "pixel", "deg", "pixel", "pixel", "deg")
LABELS = (r"$V_\infty$ [km/s]", r"$R_t$ [pixel]", "Cutout PA [deg]",
          r"$dx_R$ [pixel]", r"$dy_R$ [pixel]", r"$i$ [deg]")
SCOPE = ("Conditional on the fixed kinematic center, calibration, pixel grid, "
         "independent HalfNormal emission-pixel prior and independent Gaussian "
         "input noise. No PSF or intrinsic velocity dispersion is modeled. "
         "Passing these numerical checks does not validate these assumptions "
         "or establish that all posterior modes have been found.")


def _directory(run_or_path):
    path = Path(getattr(run_or_path, "directory", run_or_path))
    return path.parent if path.name == "posterior.npz" else path


def _clean_json(value):
    """JSON null denotes unavailable diagnostics, never a successful number."""
    if isinstance(value, dict):
        return {key: _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _clean_json(value.tolist())
    if isinstance(value, np.generic):
        return _clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _display_theta(theta):
    """Convert angles to degrees and unwrap PA around the pooled circular mean."""
    values = np.array(theta, dtype=float, copy=True)
    values[..., [2, 5]] *= 180 / np.pi
    angle = values[..., 2]
    anchor = np.rad2deg(np.angle(np.mean(np.exp(1j*np.deg2rad(angle))))) % 360
    values[..., 2] = anchor + (angle-anchor+180) % 360 - 180
    return values


def _fixed_parameters(info):
    """Read explicit fixed coordinates; never infer them from stuck chains.

    The six-column physical theta representation is retained even when a
    coordinate is fixed. Older runs have no fixed_rdy setting and continue to
    treat all six coordinates as sampled.
    """
    value = info.get("direct_model", {}).get("fixed_rdy")
    if value is None:
        return {}
    if (isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value) or not -10 <= value <= 10):
        raise ValueError("Saved fixed_rdy must be a finite pixel offset within [-10,10].")
    return {4: float(value)}


def _check_fixed_draws(theta, fixed):
    """A fixed value is an assertion about every draw, not an axis to hide."""
    for index, value in fixed.items():
        if not np.allclose(theta[..., index], value, rtol=0., atol=1e-12):
            raise ValueError(f"Saved {PARAMETERS[index]} draws disagree with the explicit fixed value {value}.")


def _fixed_note(fixed):
    if not fixed:
        return ""
    return "Conditional on " + ", ".join(
        f"{PARAMETERS[index]} fixed at {value:+.6g} pixels" for index, value in fixed.items())


def _rank_statistics(values):
    """Compute worst rank-Rhat and bulk/tail ESS across event coordinates.

    A coordinate that is constant within any chain is flagged explicitly.
    Numerical ESS routines can otherwise return misleading finite values for
    a stuck chain, particularly if distinct chains are stuck at distinct points.
    """
    import arviz_stats as azs

    values = np.asarray(values, dtype=float)
    constant = np.any(np.ptp(values, axis=1) == 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rhat = np.asarray(azs.rhat(values, method="rank"))
        bulk = np.asarray(azs.ess(values, method="bulk"))
        tail = np.asarray(azs.ess(values, method="tail", prob=(.05, .95)))
    finite = (np.isfinite(rhat).all() and np.isfinite(bulk).all()
              and np.isfinite(tail).all())
    return dict(max_rank_rhat=float(np.max(rhat)), min_bulk_ess=float(np.min(bulk)),
                min_tail_ess=float(np.min(tail)), constant_chain_coordinate=bool(constant),
                finite=bool(finite))


def diagnostics(run_or_path):
    """Save numerical mixing checks for kinematics and the sampled source map.

    Source-map checks include every pixel, plus total flux, core/outer flux,
    quadrant fluxes and first spatial moments. Pixel diagnostics are evaluated
    in blocks to limit temporary memory. Warmup-only, too-short, nonfinite or
    stuck chains never pass. These checks do not certify scientific accuracy.
    """
    directory = _directory(run_or_path)
    info = json.loads((directory / "run_info.txt").read_text())
    report = dict(convergence_checks_passed=False, scope=SCOPE, parameters={}, morphology={},
                  gate="At least four chains and 100 retained draws per chain; all checked "
                       "rank Rhat < 1.01, bulk/tail ESS >= 400; no constant chain coordinate; "
                       "no divergences; BFMI >= 0.3. Depth saturation is reported separately.",
                  warnings=[])

    def save():
        result = _clean_json(report)
        temporary = directory / "posterior_diagnostics.pending.json"
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary.replace(directory / "posterior_diagnostics.json")
        return result

    try:
        fixed = _fixed_parameters(info)
    except ValueError as error:
        report["warnings"].append(str(error))
        return save()
    report["fixed_parameters"] = {PARAMETERS[index]: value for index, value in fixed.items()}
    if fixed:
        report["scope"] = _fixed_note(fixed) + ". " + SCOPE
        report["gate"] += " Explicitly fixed coordinates are verified, then excluded from mixing checks."

    with np.load(directory / "posterior.npz", allow_pickle=False) as archive:
        if "sample_theta" not in archive:
            report["warnings"].append("No retained posterior draws yet; the checkpoint may contain only warmup.")
            report.update(chains=info.get("mcmc", {}).get("chains", 0), draws_per_chain=0)
            return save()
        theta = archive["sample_theta"]
        required = ("sample_emission", "stats_energy", "stats_potential_energy", "stats_num_steps",
                    "stats_accept_prob", "stats_diverging", "stats_iteration", "stats_is_warmup")
        missing = [name for name in required if name not in archive]
        if missing:
            report["warnings"].append("Missing saved arrays: " + ", ".join(missing))
            return save()
        emission = archive["sample_emission"]
        stats = {name.removeprefix("stats_"): archive[name] for name in required if name.startswith("stats_")}

    if theta.ndim != 3 or theta.shape[-1] != 6 or not all(theta.shape):
        report["warnings"].append("Kinematic draws must have shape [chain,draw,6] with nonempty axes.")
        return save()
    chains, draws = theta.shape[:2]
    report.update(chains=chains, draws_per_chain=draws)
    consistent = (emission.ndim == 4 and emission.shape[:2] == (chains, draws)
                  and all(a.shape == (chains, draws) for a in stats.values()))
    finite = consistent and all(np.isfinite(a).all() for a in (theta, emission, *stats.values()))
    report["all_saved_values_finite"] = bool(finite)
    if not finite or np.any(stats["is_warmup"]):
        report["warnings"].append("Nonfinite/inconsistent arrays or warmup values were found in the posterior.")
        return save()
    try:
        _check_fixed_draws(theta, fixed)
    except ValueError as error:
        report["warnings"].append(str(error))
        return save()
    if np.any(emission < 0):
        report["warnings"].append("Negative emission values violate the saved posterior model.")
        return save()
    depth = info.get("mcmc", {}).get("max_tree_depth")
    depth_fraction = float(np.mean(stats["num_steps"] >= 2**int(depth)-1)) if depth else None
    report.update(divergences=int(np.sum(stats["diverging"])),
                  mean_acceptance=float(np.mean(stats["accept_prob"])),
                  mean_leapfrog_steps=float(np.mean(stats["num_steps"])),
                  max_tree_depth_fraction=depth_fraction)
    if depth_fraction is not None and depth_fraction > .01:
        report["warnings"].append("More than 1% of retained trajectories hit maximum tree depth; inspect efficiency.")
    if report["divergences"]:
        report["warnings"].append("Divergent transitions can bias the sampled posterior.")
    if chains < 2 or draws < 4:
        report["warnings"].append("Too few chains or retained draws to compute reliable mixing diagnostics.")
        return save()

    with np.errstate(divide="ignore", invalid="ignore"):
        bfmi = np.mean(np.diff(stats["energy"], axis=1)**2, axis=1) / np.var(stats["energy"], axis=1)
    report["bfmi_by_chain"] = bfmi.tolist()
    values = _display_theta(theta)
    for index, name in enumerate(PARAMETERS):
        if index not in fixed:
            report["parameters"][name] = _rank_statistics(values[..., index])

    height, width = emission.shape[-2:]
    yy, xx = np.indices((height, width))
    cx, cy = info.get("fixed_center_pixels", ((width-1)/2, (height-1)/2))
    radius = np.hypot(xx-cx, yy-cy)
    # These regions are diagnostics only, never likelihood masks or assumptions
    # about where real emission ends. The core is half the cutout half-width.
    core = radius <= min(height, width)/4
    summaries = {"total_flux": emission.sum(axis=(-2, -1)),
                 "core_flux": emission[..., core].sum(axis=-1),
                 "outer_flux": emission[..., ~core].sum(axis=-1),
                 "x_first_moment": (emission*(xx-cx)).sum(axis=(-2, -1)),
                 "y_first_moment": (emission*(yy-cy)).sum(axis=(-2, -1))}
    for name, mask in (("lower_left_flux", (xx < cx) & (yy < cy)),
                       ("lower_right_flux", (xx >= cx) & (yy < cy)),
                       ("upper_left_flux", (xx < cx) & (yy >= cy)),
                       ("upper_right_flux", (xx >= cx) & (yy >= cy))):
        if mask.any():
            summaries[name] = emission[..., mask].sum(axis=-1)
    for name, values in summaries.items():
        report["morphology"][name] = _rank_statistics(values)

    pixels = emission.reshape(chains, draws, -1)
    blocks = [_rank_statistics(pixels[..., begin:begin+256])
              for begin in range(0, pixels.shape[-1], 256)]
    report["morphology"]["all_pixels"] = dict(
        number_of_pixels=pixels.shape[-1],
        max_rank_rhat=max(block["max_rank_rhat"] for block in blocks),
        min_bulk_ess=min(block["min_bulk_ess"] for block in blocks),
        min_tail_ess=min(block["min_tail_ess"] for block in blocks),
        constant_chain_coordinate=any(block["constant_chain_coordinate"] for block in blocks),
        finite=all(block["finite"] for block in blocks))
    checks = [*report["parameters"].values(), *report["morphology"].values()]
    finite = all(check["finite"] for check in checks) and np.isfinite(bfmi).all()
    report["all_diagnostics_finite"] = bool(finite)
    report["convergence_checks_passed"] = bool(
        chains >= 4 and draws >= 100 and finite and report["divergences"] == 0
        and np.min(bfmi) >= .3 and all(
            check["max_rank_rhat"] < 1.01 and check["min_bulk_ess"] >= 400
            and check["min_tail_ess"] >= 400 and not check["constant_chain_coordinate"]
            for check in checks))
    if not report["convergence_checks_passed"]:
        report["warnings"].append("Numerical convergence checks did not pass; displayed intervals are provisional.")
    return save()


def _corner(values, source_id, indices=None):
    """Black contour corner with one pooled central 68% interval per diagonal."""
    import matplotlib.pyplot as plt
    import corner

    indices = tuple(range(6)) if indices is None else tuple(indices)
    count = len(indices)
    flat = values[..., indices].reshape(-1, count)
    lower, median, upper = np.quantile(flat, [.16, .5, .84], axis=0)
    ranges = []
    for column in flat.T:
        low, high = float(column.min()), float(column.max())
        if low == high:
            padding = max(1e-4, abs(low)*1e-4)
            low, high = low-padding, high+padding
        ranges.append((low, high))
    fig = corner.corner(flat, labels=[LABELS[index] for index in indices], range=ranges, bins=35, color="black",
                        plot_datapoints=False, plot_density=False, fill_contours=False,
                        levels=(.68, .95), smooth=1., smooth1d=1.,
                        hist_kwargs={"color": "black", "linewidth": 1.2},
                        contour_kwargs={"colors": "black"},
                        label_kwargs={"fontsize": 10}, max_n_ticks=4)
    axes = np.asarray(fig.axes).reshape(count, count)
    for index in range(count):
        errors = [median[index]-lower[index], upper[index]-median[index]]
        positive_errors = [error for error in errors if error > 0]
        precision = (max(2, min(6, 1-int(np.floor(np.log10(min(positive_errors))))))
                     if positive_errors else 2)
        axes[index, index].set_title(
            f"{median[index]:.{precision}f} −{errors[0]:.{precision}f} +{errors[1]:.{precision}f}",
            fontsize=10)
        axes[index, index].axvline(median[index], color="black", ls="--", lw=1.)
    fig.suptitle(f"ID{source_id}: all retained chains pooled; median and 16th–84th percentiles", y=1.01)
    return fig


def show_posterior(run_or_path):
    """Open pooled intervals, traces, corners and one coherent sampled R+C fit.

    No source FITS, calibration tables, JAX initialization or sampler is needed.
    ``posterior_prediction.npz`` must have been saved from the same posterior.
    Every figure is saved in the run directory, including PDF and PNG corners.
    Displayed intervals remain visible when diagnostics fail, but are labeled
    provisional. Fixed center uncertainty and observational systematics are not
    included in these model-conditional intervals.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import pandas as pd
    from ._simple_mcmc_plots import _brightness_norm

    directory = _directory(run_or_path)
    info = json.loads((directory / "run_info.txt").read_text())
    with np.load(directory / "posterior.npz", allow_pickle=False) as archive:
        if "sample_theta" not in archive or archive["sample_theta"].shape[1] == 0:
            raise ValueError("No retained MCMC draws yet. Complete warmup before plotting a posterior.")
        theta = archive["sample_theta"].copy()
        fingerprint = str(archive["fingerprint"]) if "fingerprint" in archive else None
        prediction = {key.removeprefix("prediction_"): archive[key].copy()
                      for key in archive.files if key.startswith("prediction_")}
    if theta.ndim != 3 or theta.shape[-1] != 6 or not np.isfinite(theta).all():
        raise ValueError("Invalid kinematic posterior draws; inspect posterior.npz.")
    fixed = _fixed_parameters(info)
    _check_fixed_draws(theta, fixed)
    active = tuple(index for index in range(6) if index not in fixed)
    fixed_note = _fixed_note(fixed)
    if not prediction:
        # Compatibility with the first development checkpoints, which kept the
        # representative draw only in a separate file.
        with np.load(directory / "posterior_prediction.npz", allow_pickle=False) as archive:
            prediction = {key: archive[key].copy() for key in archive.files}
    if fingerprint is not None and str(prediction.get("fingerprint")) != fingerprint:
        raise ValueError("Saved posterior prediction belongs to a different target/checkpoint.")
    chain, draw = int(prediction["chain_index"]), int(prediction["draw_index"])
    if not (0 <= chain < theta.shape[0] and 0 <= draw < theta.shape[1]):
        raise ValueError("Saved representative draw indices are outside the posterior.")
    if not np.array_equal(theta[chain, draw], prediction["theta"]):
        raise ValueError("Saved prediction and representative joint posterior draw do not agree.")

    report = diagnostics(directory)
    values = _display_theta(theta)
    low, median, high = np.quantile(values.reshape(-1, 6), [.16, .5, .84], axis=0)
    summary = pd.DataFrame(dict(median=median, minus_16th=median-low,
                                plus_84th=high-median, unit=UNITS), index=PARAMETERS)
    summary.index.name = "parameter (all chains pooled)"
    vhalf = np.quantile(theta[..., 0]/2, [.16, .5, .84])
    summary.loc["V_at_R_t"] = [vhalf[1], vhalf[1]-vhalf[0], vhalf[2]-vhalf[1], "km/s"]
    if fixed:
        summary["status"] = "sampled"
        summary.loc["V_at_R_t", "status"] = "derived"
        for index, value in fixed.items():
            name = PARAMETERS[index]
            summary.loc[name, "median"] = value
            summary.loc[name, ["minus_16th", "plus_84th"]] = np.nan
            summary.loc[name, "status"] = "fixed"
    summary.attrs["scope"] = report["scope"]
    summary.to_csv(directory / "posterior_summary.txt", sep="\t")
    status = "numerical checks passed" if report["convergence_checks_passed"] else "provisional: mixing checks did not pass"
    corner = _corner(values, info["source_id"], active)
    corner.suptitle(f"ID{info['source_id']}: {status}\n"
                   "All chains pooled: median and 16th–84th percentiles"
                   + ("\n" + fixed_note if fixed_note else ""), y=1.03)

    traces, axes = plt.subplots(3, 2, figsize=(12, 8), layout="constrained")
    shades = plt.get_cmap("Greys")(np.linspace(.4, .95, theta.shape[0]))
    for index, ax in zip(active, axes.flat):
        for number, color in enumerate(shades):
            ax.plot(values[number, :, index], color=color, lw=.7, alpha=.8,
                    label=f"Chain {number+1}")
        ax.axhline(median[index], color="black", ls="--", lw=1)
        ax.set(xlabel="Retained step (warmup excluded)", ylabel=LABELS[index])
    for ax in list(axes.flat)[len(active):]:
        ax.set_axis_off()
    axes[0, 0].legend(fontsize=8, ncols=2)
    traces.suptitle(f"ID{info['source_id']}: {status}\nDashed lines: pooled medians; PA unwrapped at 0/360°"
                    + ("\n" + fixed_note if fixed_note else ""))

    residuals, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
    norm = _brightness_norm(*(prediction[f"{kind}_{p}"] for kind in ("observed", "predicted") for p in ("R", "C")))
    for row, pupil in enumerate(("R", "C")):
        observed, predicted = prediction[f"observed_{pupil}"], prediction[f"predicted_{pupil}"]
        residual = np.where(prediction[f"valid_{pupil}"],
                            (observed-predicted)/info["noise_sigma"][pupil], np.nan)
        for col, pixels in enumerate((observed, predicted, residual)):
            artist = axes[row, col].imshow(pixels, origin="lower", interpolation="none",
                cmap="RdBu_r" if col == 2 else "afmhot", norm=Normalize(-5, 5) if col == 2 else norm)
            axes[row, col].plot(*prediction["projected_centers"][row], "+", color="cyan", ms=13, mew=1.5)
            axes[row, col].set(title=f"{pupil}: " + ("input", "sampled prediction", "(input − prediction) / noise")[col],
                               xlabel="x [pixel]", ylabel="y [pixel]")
            if col == 2:
                residuals.colorbar(artist, ax=axes[row, col], shrink=.8, extend="both")
                count = int(np.count_nonzero(np.abs(residual) > 5))
                axes[row, col].text(.02, .02, f"{count} pixels exceed ±5", transform=axes[row, col].transAxes,
                    fontsize=9, bbox=dict(facecolor="white", alpha=.85, edgecolor="none"))
    representative = prediction["theta"]
    residuals.suptitle(f"ID{info['source_id']}: one joint draw, chain {chain+1}, retained step {draw+1}\n"
                      f"V_inf={representative[0]:.1f} km/s; i={np.rad2deg(representative[5]):.1f}°; "
                      "this is not the pooled median parameter vector"
                      + ("\n" + fixed_note if fixed_note else ""))

    emission, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    ax.imshow(prediction["emission"], origin="lower", cmap="afmhot", interpolation="none",
              norm=_brightness_norm(prediction["emission"]))
    ax.plot(*info["fixed_center_pixels"], "+", color="cyan", ms=15)
    ax.set(title="Shared emission map from the same joint draw",
           xlabel="Source x [pixel]", ylabel="Source y [pixel]")

    radii = np.linspace(0., 20., 201)
    flat = theta.reshape(-1, 6)
    curves = 2/np.pi*flat[:, 0, None]*np.arctan(radii[None, :]/flat[:, 1, None])
    lower_curve, median_curve, upper_curve = np.quantile(curves, [.16, .5, .84], axis=0)
    rotation, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    ax.fill_between(radii, lower_curve, upper_curve, color="0.8", label="16th–84th percentiles")
    ax.plot(radii, median_curve, color="black", label="Pooled median at each radius")
    ax.set(xlabel="Disk radius [pixel]", ylabel="Deprojected rotation [km/s]",
           title=f"Rotation posterior; {status}\nDisplay range is not a measured emission extent"
                 + ("\n" + fixed_note if fixed_note else ""))
    ax.legend(fontsize=9)
    views = dict(summary=summary, corner=corner, traces=traces, residuals=residuals,
                 emission=emission, rotation_curve=rotation)
    for name, value in views.items():
        if hasattr(value, "savefig"):
            value.savefig(directory / f"posterior_{name}.png", dpi=150, bbox_inches="tight")
    corner.savefig(directory / "posterior_corner.pdf", bbox_inches="tight")
    print("Posterior:", status)
    print("Retained draws:", theta.shape[0], "chains ×", theta.shape[1], "steps per chain")
    print("Divergences:", report.get("divergences"),
          "Max-depth fraction:", report.get("max_tree_depth_fraction"))
    if report.get("parameters"):
        print("Kinematic diagnostics (rank R-hat, bulk ESS, tail ESS):")
        for name, check in report["parameters"].items():
            print(f"  {name}: {check['max_rank_rhat']}, "
                  f"{check['min_bulk_ess']}, {check['min_tail_ess']}")
    for warning in report.get("warnings", []):
        print(warning)
    print("Residual panels use one actual joint draw; table/corner/rotation intervals pool all chains.")
    print(report["scope"])
    print("Saved figures and intervals:", directory)
    return views
