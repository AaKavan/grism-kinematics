"""Optional fixed-inclination diagnostic for a saved broad three-file search.

The parent search and MCMC are read-only. This worker optimizes nuisance
parameters at each inclination; it does not sample or produce credible regions.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from .candidate_search import PosteriorSearch, diverse_indices
from .three_file_fit import REPO, _json, _npz, _sha, _search_config
from .three_file_search import SharedComparisonFitter, records_and_pixels
from .three_file_plot import physical_scale

LOG = logging.getLogger(__name__)
SCHEMA = "dingo-inclination-scan-v1"
SCORE_NOTE = (
    "At each fixed inclination, all remaining parameters maximize the saved "
    "posterior density, with the uniform-prior log noise scale profiled "
    "analytically. The inclination-prior term is constant in that optimization. "
    "The likelihood shown is evaluated at the nuisance-posterior optimum; "
    "other nuisance priors remain, so this is not a pure maximum-likelihood "
    "profile. Neither curve is a marginalized inclination posterior or a "
    "confidence/credible interval. Finite multistart optimization does not "
    "guarantee the global optimum."
)


def validate_settings(inclinations, iterations, prior):
    """Return a deterministic grid, respecting the actual saved prior support."""
    values = np.asarray(inclinations, dtype=float)
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
            or isinstance(iterations, (bool, np.bool_))
            or not isinstance(iterations, (int, np.integer)) or iterations < 1):
        raise ValueError("Provide a nonempty finite inclination grid and positive integer iterations.")
    lo, hi = np.rad2deg([prior.get("min", 0.), prior.get("max", np.pi/2)])
    if np.any(values < lo-1e-10) or np.any(values > hi+1e-10):
        raise ValueError(f"Inclinations must be inside this run's saved support: {lo:g}–{hi:g} degrees.")
    # Clip only floating point roundoff at the saved bounds.
    return np.unique(np.clip(values, lo, hi))


def inclination_log_prior(radians, spec):
    """The exact per-parameter term used by KinematicsFitter.log_prior.

    As in the fitter, constant normalization factors are omitted.
    """
    value = float(radians)
    if not spec.get("min", -np.inf) <= value <= spec.get("max", np.inf):
        return -np.inf
    kind = spec["type"]
    if kind == "uniform":
        return 0.
    if kind == "gaussian":
        return -.5*((value-spec["mu"])/spec["sigma"])**2
    if kind == "sin_i":
        return float(np.log(np.sin(value))) if 0 < value < np.pi/2 else -np.inf
    if kind == "log_uniform":
        return float(-np.log(value)) if value > 0 else -np.inf
    raise ValueError(f"Unsupported inclination prior: {kind}")


class FixedInclinationSearch(PosteriorSearch):
    """Use the unchanged broad-search objective, fixing only inclination."""

    def __init__(self, fitter, inclination_deg, **kwargs):
        super().__init__(fitter, **kwargs)
        self.scales[self.inc] = 0.
        self.set_inclination(inclination_deg)

    def set_inclination(self, inclination_deg):
        spec = dict(self.fitter._mcmc_priors)["velocity.inc_v"]
        degree = validate_settings([inclination_deg], 1, spec)[0]
        self.fixed_inclination = float(np.clip(np.deg2rad(degree), *self.bounds[self.inc]))

    def constrain(self, points, velocities=None):
        if velocities is not None:
            raise ValueError("An inclination scan frees velocity; do not also fix its amplitude.")
        constrained = super().constrain(points)
        constrained[:, self.inc] = self.fixed_inclination
        return constrained


def _check_hash(path, expected, description):
    if not path.is_file() or expected is None or _sha(path) != expected:
        raise ValueError(f"Saved {description} is missing or changed: {path}")


def load_source(run_dir):
    """Verify the original source/config/code/weights before fitting anything."""
    run_dir = Path(run_dir).resolve()
    search = json.loads((run_dir/"start_search.json").read_text())
    if search.get("search_settings", {}).get("method") != "broad":
        raise ValueError("Inclination scans require a saved broad candidate search.")
    _check_hash(run_dir/"search_identity.json", search.get("search_identity_sha256"), "search identity")
    identity = json.loads((run_dir/"search_identity.json").read_text())
    for name in ("candidates", "comparison", "velocity_profile"):
        _check_hash(run_dir/f"{name}.npz", search.get(f"{name}_sha256"), name)
    for name, info in identity["inputs"].items():
        _check_hash(Path(info["path"]), info["sha256"], f"{name} input")
    for name, checksum in identity["code"].items():
        _check_hash(REPO/"dingo"/name, checksum, f"source code {name}")
    from .grism import get_grism_cal_dir
    cal = Path(get_grism_cal_dir())
    actual_cal = {str(p.relative_to(cal)): _sha(p) for p in sorted(cal.rglob("*.dat"))}
    if actual_cal != identity["calibration"]:
        raise ValueError("The grism calibration differs from this saved search.")
    config = yaml.safe_load((run_dir/"search_config.yaml").read_text())
    if _search_config(config) != identity["config"]:
        raise ValueError("The saved search configuration changed.")
    posterior = {key: config["mcmc"][key] for key in identity["posterior"]}
    if posterior != identity["posterior"]:
        raise ValueError("The saved posterior configuration changed.")
    if Path(config["shared_comparison"]).resolve() != run_dir/"comparison.npz":
        raise ValueError("The configuration points to a different comparison model.")
    if (search["source_id"] != identity["source_id"]
            or search["inclination_prior"] != config["mcmc"]["priors"]["velocity.inc_v"]):
        raise ValueError("Source or inclination-prior metadata is inconsistent.")
    return search, identity, config


def seed_points(run_dir, report, search):
    """Candidates plus diverse velocity-profile branches and optional saved MAP."""
    paths = {"start_search.json": _sha(run_dir/"start_search.json"),
             "velocity_profile.npz": _sha(run_dir/"velocity_profile.npz")}
    starts = [[trial["full_params"][name] for name in search.names] for trial in report["trials"]]
    provenance = [f"candidate {t['candidate_id']}" for t in report["trials"]]
    with np.load(run_dir/"velocity_profile.npz", allow_pickle=False) as saved:
        names = saved["param_names"].tolist()
        if set(names) != set(search.names):
            raise ValueError("Velocity-profile parameter names disagree with this fitter.")
        points = saved["points"][:, [names.index(n) for n in search.names]]
        scores = saved["log_posterior"]
        for index in diverse_indices(points, scores, search, 8):
            starts.append(points[index])
            provenance.append(f"velocity profile row {index}")
    posterior_path = run_dir/"report.json"
    if posterior_path.exists():
        posterior = json.loads(posterior_path.read_text())
        if posterior.get("status") == "complete" and posterior.get("best_fit"):
            if (posterior.get("source_id") != report["source_id"]
                    or posterior.get("comparison_sha256") != report["comparison_sha256"]
                    or posterior.get("initialization", {}).get("search_identity_sha256")
                    != report["search_identity_sha256"]):
                raise ValueError("Saved posterior MAP belongs to a different search/comparison.")
            paths["report.json"] = _sha(posterior_path)
            starts.append([posterior["best_fit"][n] for n in search.names])
            provenance.append("completed posterior MAP")
    starts = np.asarray(starts, dtype=float)
    if not len(starts) or not np.isfinite(starts).all():
        raise ValueError("Saved search did not supply finite starting parameters.")
    return starts, {"files_sha256": paths, "starts": provenance}


def scan(search, starts, inclinations, *, iterations, output):
    """Per-angle checkpoints, followed by two neighboring-branch cross-checks."""
    n, ndim = len(inclinations), len(search.names)
    points, scores = np.full((n, ndim), np.nan), np.full(n, -np.inf)
    stage, next_index, gains = 0, 0, []
    checkpoint_path, progress_path = output/"checkpoint.npz", output/"progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        _check_hash(checkpoint_path, progress["checkpoint_sha256"], "scan checkpoint")
        with np.load(checkpoint_path, allow_pickle=False) as saved:
            if (saved["points"].shape != points.shape
                    or saved["param_names"].tolist() != search.names
                    or not np.array_equal(saved["inclinations_deg"], inclinations)):
                raise ValueError("Inclination checkpoint dimensions/settings do not match.")
            points, scores = saved["points"].copy(), saved["log_posterior"].copy()
        stage, next_index = progress["stage"], progress["next_index"]
        gains = progress.get("cross_check_max_logp_gains", [])
        LOG.info("Resuming inclination scan: stage %d/3, next angle %d/%d",
                 stage+1, next_index+1, n)

    def save(stage_index, index, **active):
        _npz(checkpoint_path, points=points, log_posterior=scores,
             param_names=np.asarray(search.names), inclinations_deg=inclinations, **active)
        _json(progress_path, dict(stage=stage_index, next_index=index,
              cross_check_max_logp_gains=gains, checkpoint_sha256=_sha(checkpoint_path)))

    for current_stage in range(stage, 3):
        before = scores.copy()
        # Use a snapshot so a pass tries neighbors in both directions symmetrically.
        previous = points.copy()
        for index in range(next_index if current_stage == stage else 0, n):
            search.set_inclination(inclinations[index])
            if current_stage == 0:
                seeds = starts
            else:
                adjacent = previous[[max(0, index-1), index, min(n-1, index+1)]]
                # Keep the current best explicitly; cross-checking cannot worsen it.
                seeds = np.vstack([adjacent, points[index], starts])
                seeds = np.unique(seeds, axis=0)
            LOG.info("Inclination %g deg (%d/%d), %s: %d starts",
                     inclinations[index], index+1, n,
                     "initial scan" if current_stage == 0 else f"cross-check {current_stage}/2", len(seeds))
            def callback(active_points, active_scores, step):
                save(current_stage, index, active_points=active_points,
                     active_log_posterior=active_scores, active_iteration=np.asarray(step))
            trial, values = search.optimize(seeds, iterations=iterations,
                label=f"i={inclinations[index]:g} deg", callback=callback)
            best = int(np.argmax(values))
            if not np.isfinite(values[best]):
                raise RuntimeError(f"No finite fit at inclination {inclinations[index]:g} degrees.")
            if values[best] > scores[index]:
                points[index], scores[index] = trial[best], values[best]
            save(current_stage, index+1)
        if current_stage > 0:
            # A resumed pass measures its remaining improvements; label that explicitly.
            gain = float(np.max(np.maximum(scores-before, 0.)))
            gains.append(gain)
            LOG.info("Inclination cross-check %d: largest further lnP gain %.4f", current_stage, gain)
        save(current_stage+1, 0)
        next_index = 0
    return points, scores, gains


def run(args):
    started = time.monotonic()
    run_dir, output = Path(args.run_dir).resolve(), Path(args.output).resolve()
    if output == run_dir or output in run_dir.parents or run_dir in output.parents:
        raise ValueError("Use a separate inclination-scan directory outside the source run and its parents.")
    os.environ.setdefault("GRISM_CAL_DIR", str(REPO.parent/"grism_cal"))
    report, source_identity, config = load_source(run_dir)
    inclinations = validate_settings(args.inclinations, args.iterations, report["inclination_prior"])
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use the GPU environment or device='cpu'.")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    fitter = SharedComparisonFitter(str(run_dir/"search_config.yaml"), device=torch.device(args.device))
    radius_pc = report["search_settings"]["profile_radius_pc"]
    base = PosteriorSearch(fitter, radius_pixels=radius_pc/physical_scale(report)["pc_per_pixel"])
    starts, provenance = seed_points(run_dir, report, base)
    settings = dict(inclinations_deg=inclinations.tolist(), iterations=args.iterations,
                    cross_check_passes=2, device=args.device, profile_radius_pc=radius_pc)
    identity = dict(schema=SCHEMA, run_dir=str(run_dir),
                    search_identity_sha256=report["search_identity_sha256"],
                    comparison_sha256=report["comparison_sha256"],
                    candidates_sha256=report["candidates_sha256"],
                    settings=settings, seeds=provenance,
                    code={name: _sha(REPO/"dingo"/name) for name in
                          ("inclination_scan_fit.py", "three_file_plot.py")})
    identity_path, result_path = output/"identity.json", output/"report.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("This inclination scan has different inputs/settings/seeds/code. "
                             "Use a new scan name or explicitly overwrite only the scan.")
    elif output.exists() and any(p.name != "run.log" for p in output.iterdir()):
        raise ValueError("The scan output contains files without a matching identity.")
    output.mkdir(parents=True, exist_ok=True)
    _json(identity_path, identity)
    if result_path.exists():
        complete = json.loads(result_path.read_text())
        if complete.get("status") == "complete":
            if complete.get("identity") != identity:
                raise ValueError("The completed inclination report has a different cache identity.")
            _check_hash(output/"inclination_scan.npz", complete.get("inclination_scan_sha256"), "inclination scan")
            LOG.info("Reusing completed inclination scan. No optimization or MCMC was run.")
            return complete
    _json(output/"status.json", dict(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
                                     settings=settings, score_note=SCORE_NOTE))
    search = FixedInclinationSearch(fitter, inclinations[0], radius_pixels=base.radius_pixels)
    points, scores, gains = scan(search, starts, inclinations, iterations=args.iterations, output=output)
    # Keep records/pixels identical to the broad candidate renderer.
    path = output/"inclination_scan.npz"
    rows = records_and_pixels(fitter, search, points, scores, report,
        Path(source_identity["inputs"]["image"]["path"]).parent, path,
        ["fixed inclination diagnostic"]*len(points))
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    for row, theta in zip(rows, points):
        row["log_inclination_prior"] = inclination_log_prior(theta[search.inc], report["inclination_prior"])
        row["log_other_prior"] = row["log_prior"]-row["log_inclination_prior"]
    arrays.update(inclinations_deg=inclinations, points=points, param_names=np.asarray(search.names),
                  **{key: np.asarray([r[key] for r in rows]) for key in
                     ("log_likelihood", "log_prior", "log_posterior", "log_inclination_prior", "log_other_prior")})
    _npz(path, **arrays)
    metadata = {k: report[k] for k in
                ("source_id", "redshift", "pixel_scale_arcsec", "r_fit_pixels", "inclination_prior",
                 "inclination_metadata", "inclination_settings", "noise_note", "display_mask_note")}
    result = dict(metadata, schema=SCHEMA, status="complete", settings=settings, trials=rows,
                  param_names=search.names, selected=int(np.argmax(arrays["log_posterior"])),
                  score_note=SCORE_NOTE, search_run_dir=str(run_dir), identity=identity,
                  cross_check_max_logp_gains=gains,
                  cross_check_note="Gains cover work in each uninterrupted pass; after resume, "
                                   "a partial pass gain covers only its remaining angles.",
                  elapsed_seconds=time.monotonic()-started, completed_utc=datetime.now(timezone.utc).isoformat(),
                  inclination_scan_sha256=_sha(path))
    result.update(scan_sha256=result["inclination_scan_sha256"],
                  parent_search_sha256=_sha(run_dir/"start_search.json"),
                  finished_utc=result["completed_utc"])
    _json(result_path, result)
    _json(output/"status.json", dict(status="complete", completed_utc=result["completed_utc"]))
    LOG.info("Saved %d fixed-inclination fits to %s. No MCMC was run.", len(rows), output)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--inclinations", nargs="+", type=float, required=True)
    p.add_argument("--iterations", type=int, default=80)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return p


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    run(parser().parse_args(argv))


if __name__ == "__main__":
    main()
