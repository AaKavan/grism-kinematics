"""Small real-source interface for experimental direct R+C profile fitting.

Each run has explicit starting values, one free nonnegative emission map shared
between R and C, a fixed center and a fitted inclination with a Gaussian prior.
``fit`` saves a local profile fit. ``sample`` then samples the joint kinematics
and source-map posterior, and ``show_posterior`` displays its uncertainties and
diagnostics. The original observed pixels are never rectified during fitting.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil

import numpy as np

from .direct_source import SourceModel
from . import injection_flexible

SCHEMA = "dingo-real-direct-profile-v1"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot save {type(value).__name__} to the run record.")


@dataclass
class DirectRun:
    """One source/configuration, independent of every other notebook run."""
    model: SourceModel
    directory: Path
    theta: np.ndarray
    info: dict
    result: dict | None = None

    def save_info(self):
        temporary = self.directory / "run_info.pending.txt"
        temporary.write_text(json.dumps(self.info, indent=2, default=_json_value) + "\n")
        temporary.replace(self.directory / "run_info.txt")


def _validate_run_identity(run):
    """Reject an old notebook handle after its named folder was replaced."""
    run.model.validate_inputs()
    info = json.loads((run.directory / "run_info.txt").read_text())
    if (info.get("schema") != SCHEMA or info.get("source_id") != run.model.source_id
            or info.get("input_records") != run.model.input_records
            or info.get("direct_model") != run.model.configuration["direct_model"]):
        raise ValueError("This run handle no longer matches its saved folder. "
                         "Use direct.load_run(folder) or prepare a different run_name.")
    return info


def load_run(directory, *, device="cpu"):
    """Reopen a prepared run after a kernel restart, without deleting or fitting.

    Set GRISM_CAL_DIR as in the notebook setup. Inputs, calibration directory,
    model settings and fixed noise values must still match the saved run. Then
    ``sample(run, resume=True, overwrite=False, ...)`` can restore its checkpoint.
    For plots alone, ``show_posterior(directory)`` needs no source data or handle.
    """
    directory = Path(directory).expanduser().resolve()
    info = json.loads((directory / "run_info.txt").read_text())
    if info.get("schema") != SCHEMA or "direct_model" not in info:
        raise ValueError("This folder lacks the required direct-source model record.")
    configuration = info["direct_model"]
    source_directory = Path(info["input_records"]["R.fits"]["path"]).parent
    model = SourceModel(
        info["source_id"], data_dir=source_directory.parent,
        center=configuration["center_pixels"],
        inclination_mean=configuration["inclination_mean_deg"],
        inclination_sigma=configuration["inclination_sigma_deg"],
        noise_sigma=tuple(configuration["noise_sigma"][p] for p in ("R", "C")),
        max_velocity=configuration["bounds"][0][1], device=device,
        fixed_rdy=configuration.get("fixed_rdy"))
    # Reuse the original fixed noise, and preserve how it was originally chosen.
    # Re-estimating it on reopening would silently change the posterior target.
    model.noise_source = configuration["noise_source"]
    model.configuration["direct_model"]["noise_source"] = model.noise_source
    fit_path = directory / "fit.npz"
    if fit_path.exists():
        with np.load(fit_path, allow_pickle=False) as saved:
            theta = saved["theta"].copy()
    else:
        theta = np.array(info["initial_theta"], dtype=float)
    run = DirectRun(model, directory, theta, info)
    _validate_run_identity(run)
    print("Reopened without refitting:", directory)
    return run


def prepare(source_id, *, data_dir, output_root, run_name,
            center, inclination_mean, inclination_sigma=5.,
            start_velocity=450., start_turnover=10.09, start_pa=286.64,
            start_offset=(1.0625, .2852), start_inclination=None,
            noise_sigma=None, max_velocity=1000., device="cpu", overwrite=False,
            fixed_rdy=None):
    """Load a source and create one explicitly named output folder.

    All angles supplied here are degrees; center and offsets are zero-based
    cutout pixels. The center is fixed. Inclination is fitted with the supplied
    Gaussian prior; a width of 5 degrees is an assumption, not a measured error.
    ``noise_sigma`` optionally supplies positive R,C input noise values. With
    None, each grism gets its own robust background noise estimate.

    ``fixed_rdy`` is an optional fixed R y-registration in pixels. It overrides
    start_offset[1] in both the starting fit and MCMC. None samples/fits it as
    before. With a fixed value, all intervals are conditional on that chosen
    relative alignment; velocity, inclination and R dx remain free.

    The imaging FITS is used only for display. It is not an emission template.
    No old fit or morphology result is needed. Overwriting is restricted to
    this interface's marked output folders and never modifies the inputs.
    """
    if not run_name or Path(run_name).name != run_name or run_name in (".", ".."):
        raise ValueError("run_name must be a single folder name.")
    destination = Path(output_root).expanduser().resolve() / str(int(source_id)) / run_name
    if destination.is_symlink():
        raise ValueError("The output folder must not be a symlink.")
    directory = destination.resolve()
    source = Path(data_dir).expanduser().resolve() / f"ID{int(source_id)}"
    if directory == source or source in directory.parents or directory in source.parents:
        raise ValueError("Choose an output folder separate from the source inputs.")
    if directory.exists():
        if not overwrite:
            raise FileExistsError("Run folder exists. Change run_name or set overwrite=True.")
        marker = directory / "run_info.txt"
        try:
            existing = json.loads(marker.read_text())
        except (OSError, ValueError) as error:
            raise ValueError("Refusing to replace a folder without this interface's run record.") from error
        if existing.get("schema") != SCHEMA:
            raise ValueError("Refusing to replace a run created by a different interface.")
    # Validate the new inputs before removing any explicitly overwritten run.
    model = SourceModel(source_id, data_dir=data_dir, center=center,
                        inclination_mean=inclination_mean, inclination_sigma=inclination_sigma,
                        noise_sigma=noise_sigma, max_velocity=max_velocity, device=device,
                        fixed_rdy=fixed_rdy)
    inc = inclination_mean if start_inclination is None else start_inclination
    theta = np.array([start_velocity, start_turnover, np.deg2rad(start_pa),
                      *start_offset, np.deg2rad(inc)], dtype=float)
    if theta.shape == (6,) and model.fixed_rdy is not None:
        theta[4] = model.fixed_rdy
    if theta.shape != (6,) or not np.isfinite(model.log_prior(theta)):
        raise ValueError("Starting parameters must lie within the documented bounds.")
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    info = dict(schema=SCHEMA, source_id=int(source_id), created_utc=_now(), status="prepared",
                output_directory=str(directory), device=str(model.device),
                method="direct R+C predictions; shared nonnegative emission map profiled at each trial",
                input_records=model.input_records, noise_sigma=model.sigma_by_pupil,
                direct_model=model.configuration["direct_model"],
                noise_model="separate fixed input-pixel Gaussian noise; coadd correlations unmodeled",
                source_metadata=model.configuration.get("summary", {}),
                fixed_center_pixels=list(center), fixed_C_registration_pixels=[0., 0.],
                inclination_prior_deg=dict(mean=float(inclination_mean), sigma=float(inclination_sigma)),
                parameter_names=list(model.param_names), initial_theta=theta,
                parameter_bounds=model.theta_bounds,
                morphology="independent nonnegative pixels; no Sersic shape or smoothing prior",
                intrinsic_dispersion_kms=0., psf_model="none in this experimental prototype",
                posterior_errors_available=False)
    run = DirectRun(model, directory, theta, info)
    run.save_info()
    print(f"ID{int(source_id)}: center fixed at {tuple(center)} pixels; fitted i prior {inclination_mean:g} +/- {inclination_sigma:g} degrees.")
    if model.fixed_rdy is not None:
        print(f"R dy fixed at {model.fixed_rdy:+.6g} pixels; velocity errors will be conditional on this alignment.")
    print("R/C noise:", model.sigma_by_pupil)
    print("Saving to:", directory)
    return run


def _centers(model, theta):
    """Project the fixed source center into the two raw grism cutouts."""
    import torch
    zero = torch.tensor(0., device=model.device)
    centers = {}
    center = model.center
    with torch.no_grad():
        for pupil in ("R", "C"):
            mdx, mdy = model.forward_models[pupil](zero, zero, model.wavelength)
            dx, dy = theta[3:5] if pupil == "R" else (0., 0.)
            cutout = model.cutouts[pupil]
            centers[pupil] = (center[0] + cutout[0] - dx - float(mdx),
                              center[1] + cutout[1] - dy - float(mdy))
    return centers


def show_inputs(run):
    """Display imaging/R/C and projected kinematic centers at the chosen start."""
    import matplotlib.pyplot as plt
    from ._simple_mcmc_plots import _brightness_norm
    model = run.model
    _validate_run_identity(run)
    arrays = [model.data[p].detach().cpu().numpy() for p in ("R", "C")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
    axes[0].imshow(model.image, origin="lower", cmap="afmhot", interpolation="none",
                   norm=_brightness_norm(model.image))
    axes[0].set_title("Imaging reference (display only)")
    centers = _centers(model, run.theta)
    norm = _brightness_norm(*arrays)
    for ax, pupil, pixels in zip(axes[1:], ("R", "C"), arrays):
        ax.imshow(pixels, origin="lower", cmap="afmhot", norm=norm, interpolation="none")
        cx, cy = centers[pupil]
        ax.plot(cx, cy, "+", color="cyan", markersize=16, markeredgewidth=2)
        ax.set_title(f"{pupil}: projected center ({cx:.2f}, {cy:.2f})")
    for ax in axes:
        ax.set(xlabel="x [pixel, zero-based]", ylabel="y [pixel, zero-based]")
    fig.suptitle(f"ID{model.source_id}: inputs; cyan + = projected kinematic center")
    fig.savefig(run.directory / "inputs.png", dpi=150, bbox_inches="tight")
    return fig


def _save_result(run, result):
    """Atomically checkpoint a coherent map, parameters and both predictions."""
    model = run.model
    run.theta = np.array(result["theta"], copy=True)
    run.result = result
    payload = {k: result[k] for k in ("theta", "emission", "predicted_R", "predicted_C")}
    for pupil in ("R", "C"):
        payload[f"observed_{pupil}"] = model.data[pupil].detach().cpu().numpy()
        payload[f"valid_{pupil}"] = model.valid_by_pupil[pupil].detach().cpu().numpy()
    payload["projected_centers"] = np.array(list(_centers(model, run.theta).values()))
    temporary = run.directory / "fit.pending.npz"
    np.savez_compressed(temporary, **payload)
    temporary.replace(run.directory / "fit.npz")
    run.info["fit"] = {k: v for k, v in result.items()
                       if k not in ("emission", "predicted_R", "predicted_C")}
    run.save_info()


def fit(run, *, max_evaluations=150, progress=True):
    """Fit kinematics and a free shared map, saving every improved solution.

    Inclination is optimized under its Gaussian prior; no inclination draws or
    other MCMC samples are produced. This call returns the best local point
    found and saves whether the optimizer and gradient criteria were met.
    """
    run.info = _validate_run_identity(run)
    start = run.theta.copy()
    run.info.update(status="fitting", fit_started_utc=_now(), max_evaluations=int(max_evaluations))
    run.save_info()
    try:
        result = injection_flexible.fit(
            run.model, start_velocity=start[0], start_turnover=start[1],
            start_pa=np.rad2deg(start[2]), start_offset=tuple(start[3:5]),
            start_inclination=np.rad2deg(start[5]), max_evaluations=max_evaluations,
            progress=progress, callback=lambda result: _save_result(run, result))
        run.info.update(status="completed", fit_finished_utc=_now())
        _save_result(run, result)
    except BaseException as error:
        run.info.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        stopped_utc=_now(), error=str(error))
        run.save_info()
        raise
    print("Saved best fit:", run.directory)
    print("Optimizer:", result["success"], result["message"])
    print("Gradient criterion met:", result.get("gradient_converged"))
    return result


def show_results(run_or_directory):
    """Open saved best-fit images and values, without rerunning any fitter.

    A rotation curve is shown without an uncertainty band. Unknown emission-map
    uncertainty has not been marginalized, so no corner plot or mass errors
    would be statistically justified by these profile fits alone.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import pandas as pd
    from ._simple_mcmc_plots import _brightness_norm

    # A notebook can reload this module after creating a handle. Accept its
    # directory without requiring identity with the newly reloaded class.
    directory = Path(getattr(run_or_directory, "directory", run_or_directory))
    info = json.loads((directory / "run_info.txt").read_text())
    if info.get("schema") != SCHEMA:
        raise ValueError("This is not a direct-source profile run.")
    with np.load(directory / "fit.npz", allow_pickle=False) as z:
        saved = {k: z[k].copy() for k in z.files}
    theta = saved["theta"]
    values = theta * np.array([1., 1., 180/np.pi, 1., 1., 180/np.pi])
    summary = pd.DataFrame(dict(value=np.r_[values, theta[0]/2],
                               unit=["km/s", "pixel", "deg", "pixel", "pixel", "deg", "km/s"]),
                           index=["V_inf", "R_t", "cutout_PA", "R_dx", "R_dy", "inclination", "V_at_R_t"])
    summary.to_csv(directory / "parameter_summary.txt", sep="\t")
    residuals, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
    norm = _brightness_norm(*(saved[f"{kind}_{p}"] for kind in ("observed", "predicted") for p in ("R", "C")))
    for row, pupil in enumerate(("R", "C")):
        data, model = saved[f"observed_{pupil}"], saved[f"predicted_{pupil}"]
        sigma = info["noise_sigma"][pupil]
        difference = np.where(saved[f"valid_{pupil}"], (data-model)/sigma, np.nan)
        for col, pixels in enumerate((data, model, difference)):
            im = axes[row, col].imshow(pixels, origin="lower", interpolation="none",
                    cmap="RdBu_r" if col == 2 else "afmhot", norm=Normalize(-5,5) if col == 2 else norm)
            axes[row, col].plot(*saved["projected_centers"][row], "+", color="cyan", markersize=13, markeredgewidth=1.5)
            axes[row, col].set(title=f"{pupil}: " + ["input", "prediction", "(input − prediction) / noise"][col],
                              xlabel="x [pixel]", ylabel="y [pixel]")
            if col == 2:
                residuals.colorbar(im, ax=axes[row,col], shrink=.8, extend="both")
        # Clipping only affects the display; every finite input pixel is scored.
        count = int(np.count_nonzero(np.abs(difference) > 5))
        axes[row, 2].text(.02, .02, f"{count} pixels exceed +/-5", transform=axes[row,2].transAxes,
                          fontsize=9, bbox=dict(facecolor="white", alpha=.85, edgecolor="none"))
    residuals.suptitle(f"ID{info['source_id']}: direct R+C profile fit; V_inf={theta[0]:.1f} km/s, i={values[5]:.1f} deg\n"
                      f"Fixed source center {tuple(info['fixed_center_pixels'])}; cyan + = projected center")
    emission, ax = plt.subplots(figsize=(6,5), layout="constrained")
    ax.imshow(saved["emission"], origin="lower", cmap="afmhot", interpolation="none",
              norm=_brightness_norm(saved["emission"]))
    ax.plot(*info["fixed_center_pixels"], "+", color="cyan", markersize=15)
    ax.set(title="Shared fitted emission map", xlabel="Source x [pixel]", ylabel="Source y [pixel]")
    radii = np.linspace(0, 20, 201)
    curve = 2/np.pi*theta[0]*np.arctan(radii/theta[1])
    rotation, ax = plt.subplots(figsize=(7,4.5), layout="constrained")
    ax.plot(radii, curve, color="black")
    ax.set(xlabel="Disk radius [pixel]", ylabel="Deprojected rotation [km/s]",
           title="Local best-fit curve; uncertainties not computed\nDisplay range is not a measured emission extent")
    views = dict(summary=summary, residuals=residuals, emission=emission, rotation_curve=rotation)
    for name, value in views.items():
        if hasattr(value,"savefig"):
            value.savefig(directory / f"{name}.png", dpi=150, bbox_inches="tight")
    result = info.get("fit", {})
    print("Status:", info["status"], "Optimizer:", result.get("message", "best checkpoint"))
    print("Gradient criterion met:", result.get("gradient_converged"),
          "Inner optimality:", result.get("optimality"))
    print("This table contains profile-fit values. Use sample and show_posterior for joint MCMC intervals.")
    print("Results:", directory)
    return views


def sample(run, *, warmup=500, draws=1000, chains=4, seed=42,
           emission_prior_scale=None, target_accept=.9, max_tree_depth=8,
           checkpoint_every=25, device="auto", progress=True,
           overwrite=False, resume=False):
    """Run joint MCMC after ``fit``, with source-map uncertainty included.

    ``warmup`` and ``draws`` are steps per chain. The latter excludes warmup.
    Every emission pixel has an independent HalfNormal brightness prior;
    ``emission_prior_scale`` sets its scale in FITS units. None uses ten times
    the larger of positive data peak and noise. This explicit brightness prior
    imposes no Sérsic profile or smoothing, but its influence must be tested.

    NumPyro NUTS uses JAX on the requested CPU/GPU. State and draws are saved to
    this run's ``posterior.npz`` every ``checkpoint_every`` steps. ``resume``
    continues the saved state; ``draws`` is the requested total. ``overwrite``
    starts fresh MCMC while preserving the preceding profile fit.
    """
    from .direct_sampling import sample as sample_joint
    return sample_joint(
        run, warmup=warmup, draws=draws, chains=chains, seed=seed,
        emission_prior_scale=emission_prior_scale, target_accept=target_accept,
        max_tree_depth=max_tree_depth, checkpoint_every=checkpoint_every,
        device=device, progress=progress, overwrite=overwrite, resume=resume)


def show_posterior(run_or_directory):
    """Open saved corner, trace, residual and rotation-uncertainty plots.

    This does not run fitting or MCMC. Inspect mixing diagnostics before using
    the pooled 16th/50th/84th percentiles as scientific uncertainty intervals.
    """
    from .direct_posterior_review import show_posterior as show
    return show(run_or_directory)
