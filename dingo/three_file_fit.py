"""Isolated worker for the original image.fits / R.fits / C.fits workflow.

This deliberately wraps KinematicsFitter instead of the native-exposure model.
R and C are continuum-subtracted inputs: a common rotation field rectifies them
back into a common frame, and their disagreement is fitted. There is no sampled
gas brightness map, continuum, background, or intrinsic line dispersion here.
The direct image is retained for visual context, not used as a gas template.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import time

import numpy as np
import yaml

LOG = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parents[1]
NOISE_NOTE = (
    "Approximate stacked-image likelihood: input noise is estimated from R/C; "
    "a frozen rectification variance and effective correlation area are used, "
    "with one fitted log noise scale. Stack/subtraction covariance is not fully "
    "propagated. The variance and comparison mask are frozen at the chosen "
    "candidate for its local MCMC. Uncertainties are conditional on that "
    "selection and this noise prescription; sampling checks do not validate "
    "the scientific error coverage."
)


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_plain(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _npz(path, **values):
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **values)
    temporary.replace(path)


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_inclination_metadata(metadata, source_id, mean, sigma):
    """Bind optional imaging provenance to this source and numerical prior."""
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise ValueError("inclination_metadata must be a dictionary.")
    try:
        value = json.loads(json.dumps(metadata, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("Inclination provenance must contain finite JSON values.") from exc
    if value.get("source_id") != source_id:
        raise ValueError("The imaging inclination prior belongs to a different source.")
    for key, expected in (("inclination_deg", mean), ("inclination_sigma_deg", sigma)):
        supplied = value.get(key)
        if (expected is None or isinstance(supplied, bool)
                or not isinstance(supplied, (int, float)) or not np.isfinite(expected)
                or not np.isclose(supplied, expected, rtol=1e-12, atol=1e-12)):
            raise ValueError(f"Imaging provenance disagrees with the supplied {key}.")
    return value


def _search_config(config):
    """Only settings used by the optimizer, independent of subsequent MCMC."""
    value = copy.deepcopy(config)
    value.pop("mcmc")
    return value


def _guard_identity(path, identity, *, description):
    """Reject incompatible reuse before overwriting any saved run products."""
    if path.exists() and json.loads(path.read_text()) != _plain(identity):
        raise ValueError(f"This three-file {description} has different inputs/settings/code. "
                         "Use OVERRIDE=True or a new RUN_NAME.")
    _json(path, identity)


def _choose_candidate(search, candidate=None):
    """Candidate IDs are original trial indices, never a velocity-based ranking."""
    candidate = search.get("selected") if candidate is None else candidate
    if not isinstance(candidate, (int, np.integer)) or isinstance(candidate, (bool, np.bool_)):
        raise ValueError("Choose an integer candidate ID from the saved candidate table.")
    trials = search["trials"]
    if candidate < 0 or candidate >= len(trials):
        raise ValueError(f"Candidate {candidate} is not in the saved search.")
    trial = trials[candidate]
    if trial["loss"] is None or not np.isfinite(trial["loss"]):
        raise ValueError(f"Candidate {candidate} did not return a finite fit.")
    if trial.get("candidate_id", candidate) != candidate:
        raise ValueError("The saved candidate IDs are inconsistent with their trials.")
    return int(candidate), trial


def _export_candidates(search, path, run_config, source_dir, device, fitter_class):
    """Save every finite refinement for identical-scale, read-only inspection.

    The shared display mask only excludes nonfinite pixels. It is not a
    candidate-specific noise/coverage mask and does not hide finite residuals.
    """
    import torch
    from astropy.io import fits

    ids, rectified_r, rectified_c = [], [], []
    fitter = fitter_class(str(run_config), device=device)
    observed_r, observed_c = fitter.get_true_images()
    for index, trial in enumerate(search["trials"]):
        trial["candidate_id"] = index
        if trial["loss"] is None or not np.isfinite(trial["loss"]):
            continue
        fitter.set_params_dict(trial["full_params"])
        fitter._reset_state()
        with torch.no_grad():
            fitter.loss()
        r, c, _, _ = fitter.get_fitting_results()
        if not np.isfinite(r).all() or not np.isfinite(c).all():
            raise RuntimeError(f"Candidate {index} has nonfinite corrected images; inspect its parameters.")
        ids.append(index)
        rectified_r.append(np.array(r, copy=True))
        rectified_c.append(np.array(c, copy=True))
        velocity = trial["params"]
        inc = float(velocity["inc_v"])
        projected = (float(velocity["V_sini"]) if "V_sini" in velocity
                     else float(velocity["V_rot"])*math.sin(inc))
        trial["velocity_summary"] = {
            "V_sini_kms": projected,
            "V_inf_kms": projected / math.sin(inc),
            "inclination_deg": math.degrees(inc),
            "R_v_pixels": float(velocity["R_v"]),
        }
    if not ids:
        raise RuntimeError("No finite candidates are available for visual inspection.")
    r, c = np.stack(rectified_r), np.stack(rectified_c)
    _npz(path, candidate_ids=np.asarray(ids, dtype=int),
         image=fits.getdata(source_dir / "image.fits"),
         observed_R=observed_r, observed_C=observed_c,
         rectified_R=r, rectified_C=c, difference=r-c,
         display_mask=np.all(np.isfinite(r) & np.isfinite(c), axis=0))
    search["candidates_sha256"] = _sha(path)
    search["display_mask_note"] = "Common finite-pixel mask only; not a likelihood/noise mask."
    return search


def velocity_definition(parameter):
    """Describe the sampled amplitude, not just a transformed plotting axis."""
    if parameter == "V_rot":
        return "V_rot is deprojected asymptotic rotation speed; V_sini=V_rot*sin(i) is derived"
    if parameter == "V_sini":
        return "V_sini is projected asymptotic amplitude; V_inf=V_sini/sin(i) is derived"
    raise ValueError("velocity_parameter must be 'V_rot' or 'V_sini'.")


def build_config(source_id, data_dir, config_path, output, *, warmup=1000,
                 draws=2000, walkers=64, seed=42, inclination_deg=None,
                 inclination_sigma_deg=None, optimizer_steps=(2500, 1500),
                 device="cuda", max_velocity=1000.0, velocity_parameter="V_sini"):
    """Copy source metadata, use explicit common parameter/settings definitions.

    Sample either the projected asymptotic amplitude V_sini (legacy default)
    or the deprojected asymptotic speed V_rot, alongside inclination. The same
    uniform 0..max_velocity bound applies to the chosen amplitude; these are
    different prior measures, so saved chains cannot be interchanged.
    Coordinates including turnover radius are in input-stack pixels.
    A soft imaging prior is optional; without it inclination is isotropic.
    """
    velocity_definition(velocity_parameter)  # Validate before reading any inputs.
    amplitude_key = f"velocity.{velocity_parameter}"
    source_id = int(source_id)
    if warmup < 0 or draws < 1 or walkers < 20:
        raise ValueError("Need warmup >= 0, draws >= 1 and at least 20 walkers.")
    if len(optimizer_steps) != 2 or min(optimizer_steps) < 1:
        raise ValueError("optimizer_steps needs two positive iteration counts.")
    data_dir, output = Path(data_dir).resolve(), Path(output).resolve()
    source_dir = data_dir / f"ID{source_id}"
    config = yaml.safe_load(Path(config_path).read_text())
    summary = copy.deepcopy(config["summary"])
    if summary.get("name") != f"ID{source_id}":
        raise ValueError("Source ID disagrees with the source configuration.")
    from astropy.io import fits
    arrays = {name: fits.getdata(source_dir / f"{name}.fits")
              for name in ("image", "R", "C")}
    if any(a.ndim != 2 or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("The three files must contain finite 2D images.")
    if arrays["R"].shape != arrays["C"].shape:
        raise ValueError("The legacy rectifier requires equal R/C image shapes.")
    rfit = int(summary.get("r_fit", (arrays["R"].shape[0] - 1) // 2))
    summary.update(mode="kinematics", unit="pix", r_fit=rfit)
    inc_lo, inc_hi = np.deg2rad([5.0, 85.0]).tolist()
    inc = float(np.deg2rad(60.0 if inclination_deg is None else inclination_deg))
    inc_prior = {"type": "sin_i", "min": inc_lo, "max": inc_hi}
    opt_inc_lo, opt_inc_hi = inc_lo, inc_hi
    if inclination_deg is not None:
        if not 5 < inclination_deg < 85:
            raise ValueError("Inclination prior mean must be between 5 and 85 degrees.")
        if (inclination_sigma_deg is None or not np.isfinite(inclination_sigma_deg)
                or inclination_sigma_deg <= 0):
            raise ValueError("An inclination prior needs a positive sigma in degrees.")
        sigma = float(np.deg2rad(inclination_sigma_deg))
        inc_prior.update(type="gaussian", mu=inc, sigma=sigma)
        # Adam does not include the MCMC prior. Keep its starting-region search
        # within two prior sigmas; MCMC retains the full 5..85 degree support.
        opt_inc_lo, opt_inc_hi = max(inc_lo, inc - 2*sigma), min(inc_hi, inc + 2*sigma)
    elif inclination_sigma_deg is not None:
        raise ValueError("Inclination sigma needs an inclination mean.")
    image = {}
    for pupil in ("R", "C"):
        image[pupil] = {"module": config["image"][pupil]["module"],
                        "path": [str(source_dir / f"{pupil}.fits"), 0], "dx": 0.0, "dy": 0.0}
    velocity = {"model": "arctan", velocity_parameter: 200.0, "R_v": 2.0,
                "x0_v": float(rfit), "y0_v": float(rfit), "theta_v": 0.0, "inc_v": inc}
    priors = {
        amplitude_key: {"type": "uniform", "min": 0.0, "max": float(max_velocity)},
        "velocity.R_v": {"type": "log_uniform", "min": 0.5, "max": float(rfit)},
        "velocity.inc_v": inc_prior,
        "velocity.theta_v": {"type": "uniform", "min": 0.0, "max": 2*math.pi},
        "noise.ln_f": {"type": "uniform", "min": -2.0, "max": 2.0},
    }
    for axis in ("x", "y"):
        priors[f"velocity.{axis}0_v"] = {"type": "gaussian", "mu": float(rfit),
                                        "sigma": 5.0, "min": float(rfit-30), "max": float(rfit+30)}
        priors[f"image.R.d{axis}"] = {"type": "gaussian", "mu": 0.0, "sigma": 2.0,
                                     "min": -10.0, "max": 10.0}
    stages = []
    for idx, steps in enumerate(optimizer_steps):
        overrides = {
            "image.C.dx, image.C.dy": {"fit": False},
            "image.R.dx, image.R.dy": {"lr": 0.5 if idx == 0 else 0.003, "min": -10.0, "max": 10.0},
            amplitude_key: {"lr": 5.0 if idx == 0 else 0.3, "min": 0.0, "max": float(max_velocity)},
            "velocity.R_v": {"min": 0.5, "max": float(rfit)},
            "velocity.x0_v, velocity.y0_v": {"min": float(rfit-30), "max": float(rfit+30)},
            "velocity.inc_v": {"min": opt_inc_lo, "max": opt_inc_hi},
            "velocity.theta_v": {"fit": idx == 0},
        }
        stages.append({"method": "Adam", "maxiter": int(steps),
                       "scheduler": "StepLR" if idx == 0 else "ReduceLROnPlateau",
                       "default": {"lr": 0.05 if idx == 0 else 0.003,
                                   "min": -1e10, "max": 1e10, "fit": True},
                       "override": overrides})
    return {"summary": summary, "image": image, "velocity": velocity, "fitting": stages,
            "mcmc": {"likelihood": "gaussian", "resume": True, "stage": 0,
                     "vectorize": device.startswith("cuda"), "nwalkers": int(walkers),
                     "nsteps": int(draws), "nburn": int(warmup), "seed": int(seed),
                     "checkpoint": str(output / "checkpoint.npz"), "checkpoint_every": 50,
                     "log_cadence": 25, "xy_iters": 40, "xy_tol": 5e-3,
                     "reseed_stuck": False, "noise": {"corr_area": "auto", "fit_ln_f": True},
                     "init_scatter": {amplitude_key: 2.0, "velocity.R_v": 0.05,
                                      "velocity.x0_v": 0.05, "velocity.y0_v": 0.05,
                                      "velocity.theta_v": 0.01, "velocity.inc_v": 0.01,
                                      "image.R.dx": 0.02, "image.R.dy": 0.02, "noise.ln_f": 0.05},
                     "priors": priors}}


def run(args):
    amplitude = getattr(args, "velocity_parameter", "V_sini")
    definition = velocity_definition(amplitude)
    metadata_json = getattr(args, "inclination_metadata_json", None)
    inclination_metadata = validate_inclination_metadata(
        json.loads(metadata_json) if metadata_json is not None else None,
        args.source, args.inclination_deg, args.inclination_sigma_deg)
    import torch
    from .fitting import KinematicsFitter
    from .startpoint import find_start
    from .grism import get_grism_cal_dir

    started = time.monotonic()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if "GRISM_CAL_DIR" not in os.environ and (REPO.parent / "grism_cal").is_dir():
        os.environ["GRISM_CAL_DIR"] = str(REPO.parent / "grism_cal")
    data_dir = Path(args.data_dir).resolve()
    source_dir = data_dir / f"ID{args.source}"
    config_path = Path(args.config or source_dir / "config_kinematics.yaml").resolve()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access it. Choose device='cpu' or the GPU-enabled environment.")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    config = build_config(args.source, data_dir, config_path, output, warmup=args.warmup,
                          draws=args.draws, walkers=args.walkers, seed=args.seed,
                          inclination_deg=args.inclination_deg,
                          inclination_sigma_deg=args.inclination_sigma_deg,
                          optimizer_steps=tuple(args.optimizer_steps), device=args.device,
                          max_velocity=args.max_velocity, velocity_parameter=amplitude)
    inputs = {name: {"path": str(source_dir / f"{name}.fits"),
                     "sha256": _sha(source_dir / f"{name}.fits")}
              for name in ("image", "R", "C")}
    search_settings = {"amp_step": args.amp_step, "n_theta": args.n_theta,
                       "n_keep": args.n_keep, "max_velocity": args.max_velocity,
                       "optimizer_steps": list(args.optimizer_steps)}
    search_identity = {"source_id": args.source, "config": _search_config(config), "inputs": inputs,
                       "pixel_scale_arcsec": args.pixel_scale, "search": search_settings,
                       "inclination_prior": config["mcmc"]["priors"]["velocity.inc_v"],
                       "inclination_metadata": inclination_metadata,
                       "code": {name: _sha(REPO / "dingo" / name) for name in
                                ("three_file_fit.py", "fitting.py", "kinematics.py", "startpoint.py", "grism.py", "utils.py")}}
    calibration = Path(get_grism_cal_dir()).resolve()
    search_identity["calibration"] = {str(p.relative_to(calibration)): _sha(p)
                                      for p in sorted(calibration.rglob("*.dat"))}
    search_identity_path = output / "search_identity.json"
    identity_path = output / "identity.json"
    if identity_path.exists() and not search_identity_path.exists():
        raise ValueError("This saved run predates visual candidate selection. Preserve it and use a new RUN_NAME.")
    _guard_identity(search_identity_path, search_identity, description="search")
    # Search settings may be reused with a different later MCMC budget. Keep
    # this separate so a rejected sampling resume cannot overwrite its config.
    run_config = output / "search_config.yaml"
    run_config.write_text(yaml.safe_dump(config, sort_keys=False))
    device = torch.device(args.device)
    LOG.info("ID%s: three input files; no continuum/background parameters; device=%s", args.source, device)
    LOG.info("%s", NOISE_NOTE)
    start_path = output / "start_search.json"
    if start_path.exists():
        search = json.loads(start_path.read_text())
        LOG.info("Reusing the saved automatic velocity/angle search in this run.")
    else:
        LOG.info("Automatic %s scan: 0..%.0f km/s every %.0f, %d angles; refining %d minima.",
                 amplitude, args.max_velocity, args.amp_step, args.n_theta, args.n_keep)
        evaluations = len(np.arange(0, args.max_velocity, args.amp_step)) * args.n_theta
        class ScanFitter(KinematicsFitter):
            scan_done = 0
            def loss(self):
                value = super().loss()
                self.scan_done += 1
                if self.scan_done == 1 or self.scan_done % 32 == 0 or self.scan_done == evaluations:
                    LOG.info("Velocity/angle grid %d/%d", self.scan_done, evaluations)
                return value

        built = 0
        def make_fitter(path):
            nonlocal built
            built += 1
            if built > 1:
                LOG.info("Refining grid minimum %d/%d", built-1, args.n_keep)
            return (ScanFitter if built == 1 else KinematicsFitter)(path, device=device)

        _, _, trials = find_start(str(run_config), device=device,
                                  amp_max=args.max_velocity, amp_step=args.amp_step,
                                  n_theta=args.n_theta, n_keep=args.n_keep,
                                  make_fitter=make_fitter)
        finite_trials = [i for i, trial in enumerate(trials) if np.isfinite(trial["loss"])]
        if not finite_trials:
            _json(start_path, {"trials": trials, "selected": None})
            raise RuntimeError("No automatic-search refinement returned a finite fit.")
        selected = min(finite_trials, key=lambda i: trials[i]["loss"])
        search = {"amplitude": definition, "velocity_parameter": amplitude,
                  "velocity_definition": definition, "trials": trials,
                  "selected": int(selected)}
        _json(start_path, search)
    if search["selected"] is None:
        raise RuntimeError("The saved automatic search has no finite fit; use a fresh run after inspecting its trials.")
    candidates_path = output / "candidates.npz"
    if candidates_path.exists() and "candidates_sha256" in search:
        if _sha(candidates_path) != search["candidates_sha256"]:
            raise ValueError("Saved candidate images changed; inspect this run before continuing.")
    else:
        search.update(schema="dingo-three-file-candidates-v1", source_id=args.source,
                      redshift=float(config["summary"]["z"]),
                      pixel_scale_arcsec=args.pixel_scale, r_fit_pixels=config["summary"]["r_fit"],
                      search_settings=search_settings, search_identity_sha256=_sha(search_identity_path),
                      inclination_prior=config["mcmc"]["priors"]["velocity.inc_v"],
                      inclination_settings={"inclination_deg": args.inclination_deg,
                          "inclination_sigma_deg": args.inclination_sigma_deg if args.inclination_deg is not None else 10.},
                      inclination_metadata=inclination_metadata,
                      selection_note="Candidates found by the velocity/angle search and local refinement; "
                                     "this finite search does not guarantee that all solutions were found.")
        search = _export_candidates(search, candidates_path, run_config, source_dir, device, KinematicsFitter)
        _json(start_path, search)
    if search.get("search_identity_sha256") != _sha(search_identity_path):
        raise ValueError("The saved candidates disagree with the search identity. Use a new RUN_NAME.")
    if args.search_only:
        LOG.info("Saved %d finite candidates for residual/velocity inspection; no MCMC was run.",
                 sum(t["loss"] is not None and np.isfinite(t["loss"]) for t in search["trials"]))
        return search

    chosen_id, trial = _choose_candidate(search, args.candidate)
    identity_config = copy.deepcopy(config)
    # A search is reusable when the user chooses MCMC lengths/walkers afterward.
    # Once sampling starts, only increasing the production length is safe.
    identity_config["mcmc"].pop("nsteps")
    identity = {"search_identity": search_identity, "config": identity_config,
                "chosen_candidate_id": chosen_id, "chosen_params": trial["full_params"]}
    _guard_identity(identity_path, identity, description="sampling run (including its chosen candidate)")
    report_path = output / "report.json"
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if previous.get("status") == "complete" and previous.get("draws") == args.draws:
            for name in ("samples", "predictions"):
                if _sha(output / f"{name}.npz") != previous.get(f"{name}_sha256"):
                    raise ValueError(f"Saved {name} changed; inspect this run before resuming.")
            LOG.info("Requested %d retained steps already saved in %s; loading without further sampling.",
                     args.draws, output)
            return previous
    run_config = output / "run_config.yaml"
    run_config.write_text(yaml.safe_dump(config, sort_keys=False))
    fitter = KinematicsFitter(str(run_config), device=device)
    fitter.set_params_dict(trial["full_params"])
    LOG.info("Chosen candidate %d: sampled %s=%.1f km/s, V_inf=%.1f km/s, R_v=%.3f px, inverse R-C loss=%.6g",
             chosen_id, amplitude, trial["params"][amplitude], trial["velocity_summary"]["V_inf_kms"],
             trial["params"]["R_v"], trial["loss"])
    fitter.fit_MCMC()
    raw_chain, raw_logp = fitter._mcmc_raw_chain, fitter._mcmc_raw_log_prob
    # The legacy sampler catches Ctrl-C. Never relabel an interrupted warmup
    # as posterior draws when exporting the simple workflow's results.
    if raw_chain.shape[0] <= args.warmup:
        raise InterruptedError("Stopped during warmup; checkpoint saved, no retained posterior yet.")
    retained, logp = raw_chain[args.warmup:], raw_logp[args.warmup:]
    complete = len(retained) >= args.draws
    names = list(fitter.mcmc_param_names)
    if fitter.mcmc_sampler.iteration:
        acceptance = fitter.mcmc_sampler.acceptance_fraction
    else:
        with np.load(output / "checkpoint.npz", allow_pickle=False) as saved:
            acceptance = saved["acceptance_fraction"]
    fitter._mcmc_acceptance = acceptance
    _npz(output / "samples.npz", chain=retained, log_prob=logp,
         param_names=np.asarray(names, dtype=str), warmup=np.asarray(args.warmup),
         acceptance_fraction=acceptance)
    diag = fitter.get_mcmc_diagnostics()
    ratios = np.asarray(list(diag["n_steps_over_tau"].values()), dtype=float)
    shifts = np.asarray(list(diag["split_half_shift"].values()), dtype=float)
    sampling_pass = bool(complete and np.isfinite(ratios).all() and np.all(ratios >= 50)
                         and np.isfinite(shifts).all() and np.all(shifts <= 0.2)
                         and diag["frac_neg_inf"] == 0)
    theta = fitter.set_params_to_mcmc("map")
    rectified_r, rectified_c, vz_r, vz_c = fitter.get_fitting_results()
    observed_r, observed_c = fitter.get_true_images()
    from astropy.io import fits
    _npz(output / "predictions.npz", image=fits.getdata(source_dir / "image.fits"),
         observed_R=observed_r, observed_C=observed_c,
         rectified_R=rectified_r, rectified_C=rectified_c,
         difference=rectified_r-rectified_c, velocity_R=vz_r, velocity_C=vz_c,
         mask=fitter.mcmc_mask.detach().cpu().numpy(),
         variance=fitter.mcmc_var_map.detach().cpu().numpy(),
         theta=theta, param_names=np.asarray(names, dtype=str))
    fixed = {key: float(cfg.tensor.detach().cpu()) for key, cfg in fitter._mcmc_stage_cfg().items()
             if isinstance(cfg.tensor, torch.Tensor) and cfg.tensor.numel() == 1 and key not in names}
    report = {"schema": "dingo-three-file-v1", "source_id": args.source,
              "status": "complete" if complete else "interrupted", "sampling_pass": sampling_pass,
              "model": "Original DINGO inverse rectification of line-only R+C stacks",
              "intrinsic_dispersion_kms": 0.0, "velocity_parameter": amplitude,
              "velocity_definition": definition,
              "radius_definition": "R_v is arctan turnover radius in input stack pixels",
              "pixel_scale_arcsec": args.pixel_scale, "pixel_scale_source": "explicit notebook/worker setting",
              "redshift": float(config["summary"]["z"]), "r_fit_pixels": config["summary"]["r_fit"],
              "inputs": inputs, "sampler": "emcee ensemble MCMC with PyTorch likelihood",
              "device": args.device, "walkers": retained.shape[1], "warmup": args.warmup,
              "draws": retained.shape[0], "param_names": names, "fixed_params": fixed,
              "inclination_prior": config["mcmc"]["priors"]["velocity.inc_v"],
              "inclination_metadata": inclination_metadata,
              "settings": {"pixel_scale_arcsec": args.pixel_scale, "warmup": args.warmup,
                           "draws": args.draws, "walkers": args.walkers, "device": args.device,
                           "velocity_parameter": amplitude},
              "initialization": search, "best_fit": dict(zip(names, theta)),
              "diagnostics": diag, "percentiles_16_50_84": fitter.get_mcmc_results(),
              "noise_note": NOISE_NOTE, "noise": {"sigma_R": fitter.sigma_R, "sigma_C": fitter.sigma_C,
                                                   "correlation_area": fitter.mcmc_corr_area,
                                                   "effective_pixels": fitter.mcmc_n_eff},
              "chosen_candidate_id": chosen_id,
              "selection_method": "explicit candidate" if args.candidate is not None else "automatic minimum residual",
              "selection_notes": args.selection_notes,
              "selection_note": "MCMC starts around the selected candidate; intervals are conditional on that "
                                "choice and the local fixed noise/mask prescription. Alternative candidates "
                                "are retained separately and are not pooled. No hard boundary isolates this "
                                "mode: inspect traces for excursions before reporting local intervals.",
              "elapsed_seconds": time.monotonic()-started,
              "samples_sha256": _sha(output / "samples.npz"),
              "predictions_sha256": _sha(output / "predictions.npz")}
    _json(output / "report.json", report)
    LOG.info("Saved %d retained steps per walker to %s; sampling checks %s.",
             retained.shape[0], output, "passed" if sampling_pass else "not yet passed")
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=int, required=True)
    p.add_argument("--data-dir", default=str(REPO / "data"))
    p.add_argument("--config")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--walkers", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp-step", type=float, default=25.0)
    p.add_argument("--max-velocity", type=float, default=1000.0,
                   help="Uniform upper bound for the chosen sampled asymptotic amplitude (km/s).")
    p.add_argument("--velocity-parameter", choices=("V_sini", "V_rot"), default="V_sini",
                   help="Sample projected V_sini (legacy) or deprojected V_rot alongside inclination.")
    p.add_argument("--n-theta", type=int, default=16)
    p.add_argument("--n-keep", type=int, default=4)
    p.add_argument("--optimizer-steps", nargs=2, type=int, default=[2500, 1500])
    p.add_argument("--inclination-deg", type=float)
    p.add_argument("--inclination-sigma-deg", type=float)
    p.add_argument("--inclination-metadata-json",
                   help="Source-bound imaging prior provenance as a JSON object.")
    p.add_argument("--pixel-scale", type=float, default=0.063)
    p.add_argument("--search-only", action="store_true",
                   help="Save candidate residual images/velocities and stop before MCMC.")
    p.add_argument("--candidate", type=int,
                   help="Original trial ID selected by visual inspection; absent uses the optimizer recommendation.")
    p.add_argument("--selection-notes", default="",
                   help="Reason for choosing this candidate, saved with the posterior report.")
    return p


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    args = parser().parse_args(argv)
    if args.amp_step <= 0 or args.max_velocity <= args.amp_step or args.n_theta < 2 or args.n_keep < 1:
        raise ValueError("The automatic velocity/angle search needs a positive range, step, at least two angles and one refinement.")
    if args.pixel_scale <= 0:
        raise ValueError("pixel_scale must be positive, in arcsec/pixel.")
    run(args)


if __name__ == "__main__":
    main()
