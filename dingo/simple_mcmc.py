"""A direct, local fit and MCMC interface for the original three-file model.

There is no candidate grid, parent run, or imaging-fit dependency. The same
deterministic log posterior is used for local optimization and sampling:

    log P = -sum((R_rect - C_rect)**2) / (2 * variance) + log prior + constant.

The fixed variance is (sigma_R**2 + sigma_C**2) * noise_correlation_area.
This preserves the original unweighted residual ordering. It is an explicitly
approximate noise prescription, not a propagation of the coadd covariance.
Posterior intervals are conditional on this prescription, the disk model, and
the stated priors; neither a finished optimizer nor a corner plot validates them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import torch
import yaml

from .fitting import KinematicsFitter
from .three_file_fit import build_config, _npz, _plain, _sha


SCHEMA = "dingo-simple-mcmc-v1"
NOISE_NOTE = (
    "Constant-noise Gaussian for the original full-image squared R-C residual. "
    "Noise is estimated from each input stack and held fixed. Coadd and "
    "rectification covariance are not fully propagated. Intervals are conditional "
    "on this approximation and the priors; completion is not convergence."
)


def _now():
    return datetime.now(timezone.utc).isoformat()


class _SimpleFitter(KinematicsFitter):
    """Bind fixed geometry to checkpoints while preserving old free-fit resumes."""

    def mcmc_fingerprint(self):
        fingerprint = super().mcmc_fingerprint()
        fixed = self.config.get("fixed_parameters", {})
        if fixed:
            fingerprint["fixed_parameters"] = {
                name: float(self.velocity_cfg[name].value) for name in sorted(fixed)
            }
        return fingerprint


def _complete_point(info, names, theta):
    """Combine fitted coordinates and explicitly fixed values for rendering."""
    values = dict(info.get("fixed_parameters", {}))
    values.update(zip(names, map(float, theta)))
    return values


def _fixed_note(info):
    """Label fixed geometry in user units without inventing error bars for it."""
    labels = {"velocity.x0_v": ("x0", "px", 1.),
              "velocity.y0_v": ("y0", "px", 1.),
              "velocity.inc_v": ("i", "deg", 180. / np.pi)}
    pieces = []
    for name, value in info.get("fixed_parameters", {}).items():
        label, unit, scale = labels[name]
        pieces.append(f"{label}={value * scale:g} {unit}")
    return "Fixed: " + ", ".join(pieces) if pieces else ""


def _prepare_fitter(config_path, device):
    """Freeze numerical support at a common reference, then restore the start.

    Constant residual variance is independent of the start. The inverse-solver
    convergence footprint must also be independent of it, so all experiments
    for the same data compare the same target distribution.
    """
    fitter = _SimpleFitter(str(config_path), device=torch.device(device))
    velocity = fitter.config["velocity"]
    initial = {f"velocity.{key}": value for key, value in velocity.items() if key != "model"}
    initial.update({f"image.R.{axis}": fitter.config["image"]["R"][axis] for axis in ("dx", "dy")})
    fitter.set_params_dict({"velocity.V_rot": 0.0, "image.R.dx": 0.0, "image.R.dy": 0.0})
    fitter.prepare_mcmc()
    fitter.set_params_dict(initial)
    state = fitter.prepare_mcmc()
    return fitter, state


@dataclass
class Run:
    """One independent experiment, with its own explicit start and output folder."""

    source_id: int
    directory: Path
    fitter: KinematicsFitter
    theta: np.ndarray
    image: np.ndarray
    info: dict = field(default_factory=dict)

    def _save_info(self):
        """Keep a human-readable, machine-readable record even after interruption."""
        target = self.directory / "run_info.txt"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(_plain(self.info), indent=2, allow_nan=False) + "\n")
        temporary.replace(target)


def prepare(source_id, *, data_dir, output_root, run_name=None,
            start_velocity=300.0, start_inclination=60.0, start_pa=0.0,
            start_turnover=3.0, start_center=None, start_offset=(0.0, 0.0),
            inclination_mean=60.0, inclination_sigma=5.0,
            center_mean=None, center_sigma=5.0,
            fixed_x0=None, fixed_y0=None, fixed_inclination=None,
            max_velocity=1000.0, noise_correlation_area=1.0, noise_sigma=None,
            device="cuda", overwrite=False):
    """Prepare an independent run using only source inputs and calibration.

    Parameters
    ----------
    source_id : int
        Reads data_dir/ID<source_id>/{image,R,C}.fits and config_kinematics.yaml.
        Only source/calibration metadata are taken from that YAML. Its old fitted
        values, old MCMC settings, and old inclination prior are not inherited.
    output_root : path
        Products go to output_root/<source_id>/<run_name>/. A timestamp is used
        when run_name is omitted. No other experiment is changed.
    start_velocity : float
        Deprojected asymptotic V_inf in km/s, NOT V sin(i) or V at a finite radius.
    start_inclination, start_pa : float
        Initial disk inclination and DINGO orientation in degrees. PA follows
        the existing cutout-coordinate convention; it is not relabeled sky PA.
    start_turnover : float
        Arctangent turnover radius in stack pixels.
    start_center : (float, float) or None
        Absolute zero-based x0,y0 pixels. None uses the R/C cutout center.
        This starting point does not set the center-prior mean.
    start_offset : (float, float)
        R-to-C registration dx,dy in pixels. C offsets stay fixed at zero.
    inclination_mean, inclination_sigma : float
        Explicit Gaussian prior in degrees, truncated to 5..85 degrees. These
        are assumptions supplied by you; this function does not run PySersic.
        The starting inclination and prior mean are deliberately separate.
        These two settings are ignored when fixed_inclination is supplied.
    center_mean : (float, float) or None
        Means of independent Gaussian priors on x0,y0, in absolute zero-based
        R/C stack pixels. None uses the geometric cutout center, independently
        of start_center. No imaging centroid or coordinate conversion is
        inferred: supply a measured center in this coordinate frame if known.
    center_sigma : float or (float, float)
        Positive Gaussian widths in R/C pixels. A scalar applies to both axes;
        a pair supplies sigma_x,sigma_y. The default 5 preserves earlier runs.
        Both local optimization and MCMC use these priors, with the existing
        hard bounds of cutout center +/-30 pixels. These widths express an
        assumption about the center, including possible stellar/gas offsets.
    fixed_x0, fixed_y0 : float or None
        Hold either center coordinate at this absolute zero-based pixel value
        during BOTH local fitting and MCMC. None leaves that coordinate free.
        A fixed value supersedes its start_center coordinate and center prior.
    fixed_inclination : float or None
        Hold i at this value in degrees during BOTH fitting and sampling. None
        samples i with the Gaussian prior above. No inclination prior or error
        is applied in fixed mode; velocity intervals are conditional on it.
    noise_correlation_area : float
        Fixed multiplicative factor on the residual variance; default 1. It is
        an assumption, not a measured correction for correlated image noise.
    noise_sigma : positive float or (float, float), optional
        Explicit input-pixel noise standard deviations, in the same intensity
        units as R.fits and C.fits. A scalar supplies the same value to both;
        a pair supplies sigma_R, sigma_C. None preserves the usual background
        estimates from the input images. Supplying known simulation noise does
        not account for covariance introduced by rectification. Zero is not a
        valid likelihood variance; use a residual-score scan for noiseless data.
    overwrite : bool
        Delete ONLY an existing folder marked as a run created by this module.
        Leave False to protect it; change run_name for another experiment.
    """
    from astropy.io import fits

    if isinstance(source_id, bool) or int(source_id) != source_id or source_id < 0:
        raise ValueError("source_id must be a nonnegative integer.")
    source_id = int(source_id)
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this kernel. Select the grism kernel or device='cpu'.")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    source = Path(data_dir).expanduser().resolve() / f"ID{source_id}"
    required = [source / f"{name}.fits" for name in ("image", "R", "C")]
    required.append(source / "config_kinematics.yaml")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing source inputs:\n" + "\n".join(missing))
    if not np.isfinite(noise_correlation_area) or noise_correlation_area <= 0:
        raise ValueError("noise_correlation_area must be positive.")
    supplied_noise = None
    if noise_sigma is not None:
        try:
            supplied_noise = np.asarray(noise_sigma, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("noise_sigma must be a positive scalar or an R,C pair.") from error
        if supplied_noise.ndim == 0:
            supplied_noise = np.repeat(supplied_noise, 2)
        if (supplied_noise.shape != (2,) or not np.isfinite(supplied_noise).all()
                or np.any(supplied_noise <= 0)
                or any(isinstance(value, (bool, np.bool_))
                       for value in np.asarray(noise_sigma, dtype=object).flat)):
            raise ValueError("noise_sigma must contain positive finite standard deviations: one value or an R,C pair.")
    if not np.isfinite(max_velocity) or max_velocity <= 0:
        raise ValueError("max_velocity must be positive.")
    run_name = run_name or datetime.now(timezone.utc).strftime("manual_%Y%m%d_%H%M%S_%f")
    if Path(run_name).name != run_name or run_name in (".", ".."):
        raise ValueError("run_name must be a single folder name.")
    directory = Path(output_root).expanduser().resolve() / str(source_id) / run_name
    if directory.is_symlink():
        raise ValueError("The output run folder must not be a symlink.")
    directory = directory.resolve()
    if directory == source or source in directory.parents or directory in source.parents:
        raise ValueError("Choose an output folder separate from the source input folder.")

    # build_config starts from metadata, not any old fitted parameter values.
    config = build_config(source_id, source.parent, required[-1], directory,
                          inclination_deg=inclination_mean if fixed_inclination is None else None,
                          inclination_sigma_deg=inclination_sigma if fixed_inclination is None else None,
                          device=device, max_velocity=max_velocity,
                          velocity_parameter="V_rot")
    shape = fits.getdata(source / "R.fits").shape
    if shape[0] != shape[1] or shape[0] != 2 * config["summary"]["r_fit"] + 1:
        raise ValueError("The original rectifier needs square, odd R/C cutouts matching r_fit.")
    cutout_center = ((shape[1] - 1) / 2, (shape[0] - 1) / 2)
    center = cutout_center if start_center is None else start_center
    if len(center) != 2 or len(start_offset) != 2:
        raise ValueError("start_center and start_offset must each contain x,y.")

    # Keep the prior separate from the optimizer start: changing a starting
    # point must not silently change the distribution being sampled.
    try:
        center_means = np.asarray(cutout_center if center_mean is None else center_mean,
                                  dtype=float)
        center_widths = np.asarray(center_sigma, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("center_mean needs x,y; center_sigma needs a positive width or x,y widths.") from error
    if center_means.shape != (2,) or not np.isfinite(center_means).all():
        raise ValueError("center_mean must contain two finite x,y pixel coordinates.")
    if center_widths.ndim == 0:
        center_widths = np.repeat(center_widths, 2)
    if (center_widths.shape != (2,) or not np.isfinite(center_widths).all()
            or np.any(center_widths <= 0) or isinstance(center_sigma, (bool, np.bool_))):
        raise ValueError("center_sigma must be positive and finite: one width or two x,y widths in pixels.")
    for axis, mean, width in zip(("x", "y"), center_means, center_widths):
        prior = config["mcmc"]["priors"][f"velocity.{axis}0_v"]
        if not prior["min"] <= mean <= prior["max"]:
            raise ValueError(f"center_mean {axis}={mean:g} outside [{prior['min']}, {prior['max']}].")
        prior.update(mu=float(mean), sigma=float(width))

    initial = {
        "velocity.V_rot": float(start_velocity),
        "velocity.inc_v": float(np.deg2rad(start_inclination)),
        "velocity.theta_v": float(np.deg2rad(start_pa) % (2 * np.pi)),
        "velocity.R_v": float(start_turnover),
        "velocity.x0_v": float(center[0]), "velocity.y0_v": float(center[1]),
        "image.R.dx": float(start_offset[0]), "image.R.dy": float(start_offset[1]),
    }
    fixed = {}
    for name, value, factor in (("velocity.x0_v", fixed_x0, 1.),
                                 ("velocity.y0_v", fixed_y0, 1.),
                                 ("velocity.inc_v", fixed_inclination, np.pi / 180.)):
        if value is not None:
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(value):
                raise ValueError(f"Fixed {name} must be a finite number or None.")
            fixed[name] = float(value) * factor
            initial[name] = fixed[name]
    for name, value in initial.items():
        prior = config["mcmc"]["priors"][name]
        if not np.isfinite(value) or not prior["min"] <= value <= prior["max"]:
            raise ValueError(f"Starting {name}={value} outside [{prior['min']}, {prior['max']}].")
        if name.startswith("velocity."):
            config["velocity"][name.split(".")[1]] = value
        else:
            config["image"]["R"][name.split(".")[-1]] = value

    # Keep one free-parameter declaration. We do not run the legacy Adam stages.
    config["fitting"] = config["fitting"][:1]
    mc = config["mcmc"]
    mc.update(likelihood="adam", resume=False, xy_iters=100,
              checkpoint=str(directory / "_checkpoints" / "chain.npz"))
    mc["noise"] = dict(fit_ln_f=False, corr_area=float(noise_correlation_area))
    if supplied_noise is not None:
        mc["noise"].update(sigma_R=float(supplied_noise[0]), sigma_C=float(supplied_noise[1]))
    mc["priors"].pop("noise.ln_f")
    mc["init_scatter"].pop("noise.ln_f")
    if fixed:
        # Expand grouped overrides before changing individual fit flags, so
        # fixing x0 alone does not accidentally fix y0 or lose its bounds.
        for stage in config["fitting"]:
            expanded = {name.strip(): dict(settings)
                        for group, settings in stage["override"].items()
                        for name in group.split(",")}
            for name in fixed:
                expanded[name]["fit"] = False
            stage["override"] = expanded
        for name in fixed:
            mc["priors"].pop(name, None)
            mc["init_scatter"].pop(name, None)
        config["fixed_parameters"] = fixed

    # Validate all settings before replacing a previous run.
    if directory.exists():
        info_path = directory / "run_info.txt"
        marked = info_path.is_file() and json.loads(info_path.read_text()).get("schema") == SCHEMA
        if not overwrite:
            raise FileExistsError(f"Run already exists: {directory}. Use another run_name or overwrite=True.")
        if not marked:
            raise ValueError("Refusing to replace a folder not created by simple_mcmc.")
    # Evaluate the replacement before deleting any explicitly overwritten run.
    with tempfile.TemporaryDirectory(prefix="dingo_prepare_") as scratch:
        trial_config = Path(scratch) / "config.yaml"
        trial_config.write_text(yaml.safe_dump(config, sort_keys=False))
        fitter, state = _prepare_fitter(trial_config, device)
        theta = state["theta0"].copy()
        score = fitter.log_probability(theta)
        if not np.isfinite(score):
            raise ValueError("The explicit starting model cannot be evaluated. Check its parameters.")
    noise_note = NOISE_NOTE
    if supplied_noise is not None:
        noise_note = noise_note.replace("Noise is estimated from each input stack and held fixed.",
                                        "Noise standard deviations are supplied explicitly and held fixed.")
    info = dict(schema=SCHEMA, source_id=source_id, created_utc=_now(), status="preparing",
                output_directory=str(directory), device=device, noise_note=noise_note,
                start_parameters=initial, priors=mc["priors"], fixed_parameters=fixed,
                inputs={p.name: {"path": str(p), "sha256": _sha(p)} for p in required},
                configuration=config)
    image = fits.getdata(source / "image.fits")
    if directory.exists():
        shutil.rmtree(directory)
    (directory / "_checkpoints").mkdir(parents=True)
    config_path = directory / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    fitter.config_path = str(config_path)
    info.update(status="prepared", param_names=state["param_names"], initial_log_posterior=float(score),
                noise=dict(sigma_R=fitter.sigma_R, sigma_C=fitter.sigma_C,
                           sigma_source="supplied" if supplied_noise is not None else "estimated_from_inputs",
                           residual_variance=fitter.mcmc_var_base,
                           correlation_area=float(noise_correlation_area),
                           support_reference="zero velocity and zero R registration"))
    run = Run(source_id, directory, fitter, theta, image, info)
    run._save_info()
    inclination_note = (f"fixed i={fixed_inclination:g} deg" if fixed_inclination is not None else
                        f"initial i={start_inclination:g} deg; prior i={inclination_mean:g} +/- {inclination_sigma:g} deg")
    print(f"ID{source_id}: explicit V_inf={start_velocity:g} km/s; {inclination_note}.\nSaving to {directory}")
    for axis in ("x", "y"):
        prior = mc["priors"].get(f"velocity.{axis}0_v")
        if prior is not None:
            print(f"Sampled {axis}0: Gaussian prior {prior['mu']:g} +/- {prior['sigma']:g} pixels (1 sigma).")
    if fixed:
        print(_fixed_note(info) + ". Excluded from optimization and MCMC; intervals are conditional on these values.")
    return run


def show_inputs(run):
    """Display the direct image and the two unmodified input stacks."""
    from ._simple_mcmc_plots import input_figure
    r, c = run.fitter.get_true_images()
    return input_figure(run.image, r, c)


def open_run(directory, *, device=None):
    """Recreate an existing experiment after restarting the notebook kernel.

    This loads the saved explicit configuration and MCMC starting point. It
    never optimizes, samples, or replaces products. Use sample(..., resume=True)
    with the SAME warmup and walker count to continue its checkpoint. Merely
    viewing results needs only show_results(directory), without this function.
    """
    from astropy.io import fits

    directory = Path(directory).expanduser().resolve()
    info = json.loads((directory / "run_info.txt").read_text())
    if info.get("schema") != SCHEMA:
        raise ValueError("This folder is not a simple_mcmc experiment.")
    for name, item in info["inputs"].items():
        if _sha(item["path"]) != item["sha256"]:
            raise ValueError(f"{name} has changed since this experiment; prepare a separate run.")
    device = device or info["device"]
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Choose an available device: 'cpu' or 'cuda'.")
    fitter, state = _prepare_fitter(directory / "config.yaml", device)
    point = info.get("sampling", {}).get("starting_parameters", info.get("mcmc_start", info["start_parameters"]))
    theta = np.array([point[name] for name in state["param_names"]])
    fitter.config["mcmc"]["vectorize"] = device == "cuda"
    if not np.isfinite(fitter.log_probability(theta)):
        raise ValueError("The saved starting point is no longer evaluable.")
    info["device"] = device
    return Run(info["source_id"], directory, fitter, theta,
               fits.getdata(info["inputs"]["image.fits"]["path"]), info)


def show_fit(run, *, label="Current model"):
    """Render exactly the parameter vector currently selected for MCMC."""
    from ._simple_mcmc_plots import residual_figure
    fitter = run.fitter
    score = fitter.log_probability(run.theta)
    if not np.isfinite(score):
        raise ValueError("Current parameters do not produce a finite posterior.")
    r, c, _, _ = fitter.get_fitting_results()
    p = _complete_point(run.info, fitter.mcmc_param_names, run.theta)
    title = (f"ID{run.source_id} — {label}: V_inf={p['velocity.V_rot']:.1f} km/s, "
             f"i={np.rad2deg(p['velocity.inc_v']):.1f} deg")
    if run.info.get("fixed_parameters"):
        title += "\n" + _fixed_note(run.info)
    return residual_figure(r, c, (p["velocity.x0_v"], p["velocity.y0_v"]), title)


def fit_local(run, *, max_evaluations=3000, progress=True):
    """Improve ONE explicit start, using the same log posterior as MCMC.

    Powell's derivative-free local optimizer uses scaled coordinates and the
    stated prior bounds. It does not scan a velocity grid or certify a global
    optimum. Its completion flag and evaluation count are saved. The best point
    actually evaluated is retained even when the evaluation budget is reached.
    """
    from scipy.optimize import minimize
    from tqdm.auto import tqdm

    if isinstance(max_evaluations, bool) or int(max_evaluations) != max_evaluations or max_evaluations < 1:
        raise ValueError("max_evaluations must be a positive integer.")
    if (run.directory / "samples.npz").exists() or Path(run.fitter.config["mcmc"]["checkpoint"]).exists():
        raise ValueError("This experiment already has a chain. Prepare another named run before changing its start.")
    fitter = run.fitter
    bounds = np.array([[p["min"], p["max"]] for _, p in fitter._mcmc_priors])
    lower, scale = bounds[:, 0], bounds[:, 1] - bounds[:, 0]
    best = run.theta.copy()
    best_score = float(fitter.log_probability(best))
    before = best_score
    started = time.monotonic()
    bar = tqdm(total=int(max_evaluations), desc="Local fit evaluations", disable=not progress)
    run.info.update(status="local_fit_running")
    run._save_info()

    def objective(unit_point):
        nonlocal best, best_score
        theta = lower + scale * unit_point
        score = float(fitter.log_probability(theta))
        if np.isfinite(score) and score > best_score:
            best, best_score = theta.copy(), score
        bar.update(1)
        if bar.n % 25 == 0:
            bar.set_postfix(best_logP=f"{best_score:.2f}", refresh=False)
        return -score if np.isfinite(score) else 1e100

    try:
        result = minimize(objective, (best - lower) / scale, method="Powell",
                          bounds=[(0.0, 1.0)] * len(best),
                          options=dict(maxfev=int(max_evaluations), xtol=1e-4, ftol=1e-8))
        record = dict(success=bool(result.success), message=str(result.message),
                      evaluations=int(result.nfev), log_posterior_before=before,
                      log_posterior_after=best_score, elapsed_seconds=time.monotonic() - started)
    except BaseException as exc:
        run.info.update(status="local_fit_interrupted", error=str(exc))
        run._save_info()
        raise
    finally:
        bar.close()
    run.theta = best
    fitter.set_params_dict(dict(zip(fitter.mcmc_param_names, best)))
    fitter.log_probability(best)  # Ensure the cached images match the retained point.
    run.info.update(status="local_fit_finished", local_optimizer=record,
                    mcmc_start=dict(zip(fitter.mcmc_param_names, best)))
    run._save_info()
    print(f"Local fit: {record['message']} ({record['evaluations']} evaluations).")
    if not result.success:
        print("The local optimizer has not established convergence; inspect this start before interpreting MCMC.")
    return record


def sample(run, *, warmup=1000, draws=4000, walkers=64, seed=42, resume=False, progress=True):
    """Sample around run.theta, with progress and checkpoints every 50 steps.

    Uses emcee's ensemble proposals and the existing PyTorch likelihood. With
    device='cuda' the likelihood is evaluated in GPU batches; emcee itself runs
    on the CPU. The requested draws are per walker, AFTER warmup. Walkers within
    one ensemble are not independent chains. A resume reuses this experiment's
    checkpoint; it is not a new independent run or guaranteed bitwise restart.
    """
    for key, value, minimum in (("warmup", warmup, 0), ("draws", draws, 1), ("walkers", walkers, 20), ("seed", seed, 0)):
        if isinstance(value, bool) or int(value) != value or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}.")
    fitter = run.fitter
    checkpoint = Path(fitter.config["mcmc"]["checkpoint"])
    if not resume and (checkpoint.exists() or (run.directory / "samples.npz").exists()):
        raise FileExistsError("This run already has samples. Use resume=True to extend it, or prepare a new run.")
    previous = run.info.get("sampling", {})
    if resume and previous and any(previous[k] != v for k, v in (("warmup", warmup), ("walkers", walkers))):
        raise ValueError("Keep warmup and walkers unchanged when resuming; increase draws to extend the chain.")
    fitter.set_params_dict(dict(zip(fitter.mcmc_param_names, run.theta)))
    mc = fitter.config["mcmc"]
    mc.update(nburn=int(warmup), nsteps=int(draws), nwalkers=int(walkers), seed=int(seed), resume=bool(resume))
    run.info.update(status="sampling", sampling_started_utc=_now(),
                    sampling=dict(warmup=int(warmup), draws=int(draws), walkers=int(walkers), seed=int(seed),
                                  resume=bool(resume), starting_parameters=dict(zip(fitter.mcmc_param_names, run.theta))))
    (run.directory / "config.yaml").write_text(yaml.safe_dump(fitter.config, sort_keys=False))
    run.info["configuration"] = fitter.config
    run._save_info()
    started = time.monotonic()
    try:
        fitter.fit_MCMC(progress=progress)
    except BaseException as exc:
        run.info.update(status="sampling_failed", error=str(exc), sampling_stopped_utc=_now())
        run._save_info()
        raise

    # Never relabel unfinished warmup as posterior samples. The lower-level
    # sampler can return after KeyboardInterrupt, so use the REQUESTED burn-in.
    raw_chain = fitter._mcmc_raw_chain
    raw_log_prob = fitter._mcmc_raw_log_prob
    chain, log_prob = raw_chain[warmup:], raw_log_prob[warmup:]
    completed = len(raw_chain) >= warmup + draws
    if fitter.mcmc_sampler.iteration:
        acceptance = fitter.mcmc_sampler.acceptance_fraction
    elif checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            acceptance = saved["acceptance_fraction"].copy()
    else:
        acceptance = np.full(walkers, np.nan)
    _npz(run.directory / "samples.npz", chain=chain, log_prob=log_prob,
         param_names=np.asarray(fitter.mcmc_param_names, dtype=str),
         fixed_names=np.asarray(list(run.info.get("fixed_parameters", {})), dtype=str),
         fixed_values=np.asarray(list(run.info.get("fixed_parameters", {}).values()), dtype=float),
         warmup=np.array(warmup), completed=np.array(completed),
         acceptance_fraction=acceptance, source_id=np.array(run.source_id))
    run.info.update(status="finished" if completed else "interrupted", sampling_finished_utc=_now(),
                    elapsed_sampling_seconds=time.monotonic() - started,
                    retained_steps_per_walker=len(chain), acceptance_mean=float(np.mean(acceptance)),
                    acceptance_scope="most recently sampled segment (includes its warmup, if any)")
    fitter._mcmc_acceptance = acceptance
    if len(chain):
        # All reported residuals come from ONE actual maximum-posterior draw.
        # Marginal medians belong in the table, not mixed into a synthetic fit.
        index = np.unravel_index(np.argmax(log_prob), log_prob.shape)
        theta_map = chain[index].copy()
        score = fitter.log_probability(theta_map)
        r, c, vr, vc = fitter.get_fitting_results()
        _npz(run.directory / "prediction.npz", rectified_R=r, rectified_C=c, difference=r-c,
             velocity_R=vr, velocity_C=vc, theta=theta_map,
             fixed_names=np.asarray(list(run.info.get("fixed_parameters", {})), dtype=str),
             fixed_values=np.asarray(list(run.info.get("fixed_parameters", {}).values()), dtype=float),
             param_names=np.asarray(fitter.mcmc_param_names, dtype=str), log_posterior=np.array(score))
        run.info["displayed_draw"] = _complete_point(run.info, fitter.mcmc_param_names, theta_map)
        if len(chain) >= 4:
            run.info["diagnostics"] = fitter.get_mcmc_diagnostics()
    run._save_info()
    print(f"{run.info['status']}: {len(chain)} retained steps per walker.\nResults: {run.directory}")
    return run.directory


def show_results(directory, *, save=True):
    """Open saved samples and plots without fitting, calibrations, or prior runs.

    Returns a dictionary containing a summary table and Matplotlib figures.
    The table and diagonal titles use pooled median/16th/84th percentiles.
    Residuals use one actual maximum-posterior draw, labeled with its own values.
    With save=True, corner.png and corner.pdf are written directly into this
    run directory alongside samples.npz; no fitting or sampling is started.
    """
    from ._simple_mcmc_plots import posterior_figures, residual_figure

    directory = Path(directory).expanduser().resolve()
    info = json.loads((directory / "run_info.txt").read_text())
    with np.load(directory / "samples.npz", allow_pickle=False) as saved:
        chain, log_prob = saved["chain"], saved["log_prob"]
        names = saved["param_names"].tolist()
    if not len(chain):
        raise ValueError("Sampling stopped before warmup finished. The checkpoint is saved; there are no posterior draws yet.")
    views = posterior_figures(chain, names, log_prob, info["source_id"])
    with np.load(directory / "prediction.npz", allow_pickle=False) as saved:
        p = _complete_point(info, saved["param_names"].tolist(), saved["theta"])
        title = (f"ID{info['source_id']} — maximum-posterior draw: "
                 f"V_inf={p['velocity.V_rot']:.1f} km/s, i={np.rad2deg(p['velocity.inc_v']):.1f} deg")
        if info.get("fixed_parameters"):
            title += "\n" + _fixed_note(info)
        views["residuals"] = residual_figure(saved["rectified_R"], saved["rectified_C"],
            (p["velocity.x0_v"], p["velocity.y0_v"]), title)
    print(f"Status: {info['status']}; retained steps/walker: {len(chain)}; "
          f"latest segment mean acceptance: {info.get('acceptance_mean', float('nan')):.3f}")
    print("These are conditional posterior summaries. Check traces, bounds and independent starts before quoting errors.")
    if info.get("fixed_parameters"):
        note = _fixed_note(info)
        print(note + ". These quantities were not sampled; their uncertainty is not included.")
        for name in ("traces", "corner"):
            figure = views[name]
            previous_title = figure._suptitle.get_text() if figure._suptitle else ""
            figure.suptitle(previous_title + "\n" + note, fontsize=10,
                           y=1.02 if name == "corner" else .995)
    if "diagnostics" in info:
        print("Autocorrelation time per parameter:", info["diagnostics"].get("tau"))
    if save:
        for name, value in views.items():
            if hasattr(value, "savefig"):
                value.savefig(directory / f"{name}.png", dpi=150, bbox_inches="tight")
        views["corner"].savefig(directory / "corner.pdf", bbox_inches="tight")
        views["summary"].to_csv(directory / "parameter_summary.txt", sep="\t", index_label="parameter")
        print(f"Corner plots saved directly in {directory}: corner.png and corner.pdf")
    return views
