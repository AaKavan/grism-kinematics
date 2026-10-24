"""Start a separate, free-inclination posterior from an inclination-scan row.

Only the inclination prior changes. The original three-file rectifier, shared
comparison weights, other priors and sampled parameters are inherited intact.
Neither the parent search nor the fixed-inclination diagnostic is overwritten.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from .inclination_scan_fit import load_source, _check_hash, inclination_log_prior
from .three_file_fit import REPO, _json, _npz, _sha, _guard_identity
from .three_file_search import sample

LOG = logging.getLogger(__name__)
SCHEMA = "dingo-three-file-inclination-prior-run-v1"
SELECTION_NOTE = (
    "Initialized from a fixed-inclination scan fit, then inclination and all "
    "other previously sampled parameters are free. The supplied Gaussian "
    "inclination prior replaces the parent's prior; it is not multiplied by "
    "it. The posterior can move away from the selected inclination. Its "
    "intervals include sampled inclination uncertainty, conditional on this "
    "user-specified prior and the approximate stacked-image noise model."
)


def _separate_output(output, *sources):
    for source in sources:
        if output == source or output in source.parents or source in output.parents:
            raise ValueError("Use a separate posterior directory outside the parent search and scan trees.")


def _validate_settings(args, prior):
    for key, minimum in (("warmup", 0), ("draws", 1), ("walkers", 20), ("seed", 0)):
        value = getattr(args, key)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}.")
    mean, sigma = args.inclination_deg, args.inclination_sigma_deg
    if (isinstance(mean, (bool, np.bool_)) or isinstance(sigma, (bool, np.bool_))
            or not np.isfinite([mean, sigma]).all() or sigma <= 0):
        raise ValueError("The Gaussian inclination prior needs a finite mean and positive sigma in degrees.")
    lo, hi = np.rad2deg([prior["min"], prior["max"]])
    if not lo < mean < hi:
        raise ValueError(f"The Gaussian inclination mean must be inside {lo:g}–{hi:g} degrees.")
    if args.device not in ("cpu", "cuda"):
        raise ValueError("Use device='cpu' or device='cuda'.")


def load_scan_seed(parent, scan_dir, parent_report, angle):
    """Verify the saved scan and return one actual fitted row, without refitting."""
    from .inclination_scan import ScanResults
    scan = ScanResults(scan_dir, parent_report["source_id"])
    report = scan.report
    identity = json.loads((scan_dir/"identity.json").read_text())
    if report.get("identity") != identity:
        raise ValueError("The scan report and saved identity disagree.")
    if (Path(report["search_run_dir"]).resolve() != parent
            or Path(identity["run_dir"]).resolve() != parent):
        raise ValueError("The inclination scan belongs to a different parent search.")
    _check_hash(parent/"start_search.json", report.get("parent_search_sha256"), "parent search")
    for key in ("search_identity_sha256", "comparison_sha256", "candidates_sha256"):
        if identity.get(key) != parent_report.get(key):
            raise ValueError(f"The inclination scan has different parent {key}.")
    for name, checksum in identity["code"].items():
        _check_hash(REPO/"dingo"/name, checksum, f"scan source code {name}")
    if not np.isfinite(angle):
        raise ValueError("The scan inclination must be finite.")
    indices = np.flatnonzero(np.isclose(scan.angles, angle, rtol=0, atol=1e-8))
    if len(indices) != 1:
        raise ValueError(f"No saved scan row at {angle:g} degrees; choose from {scan.angles.tolist()}.")
    index = int(indices[0])
    provenance = dict(parent_run=str(parent), scan_dir=str(scan_dir),
        parent_search_sha256=_sha(parent/"start_search.json"),
        scan_report_sha256=_sha(scan_dir/"report.json"),
        scan_identity_sha256=_sha(scan_dir/"identity.json"),
        scan_sha256=_sha(scan_dir/"inclination_scan.npz"),
        scan_row_index=index, seed_inclination_deg=float(scan.angles[index]))
    return copy.deepcopy(report["trials"][index]), scan.arrays, provenance


def run(args):
    started = time.monotonic()
    parent = Path(args.parent_run).expanduser().resolve()
    scan_dir = Path(args.scan_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    _separate_output(output, parent, scan_dir)
    os.environ.setdefault("GRISM_CAL_DIR", str(REPO.parent/"grism_cal"))
    original, parent_identity, config = load_source(parent)
    _validate_settings(args, original["inclination_prior"])
    scan_angle = getattr(args, "scan_inclination_deg", None)
    scan_angle = args.inclination_deg if scan_angle is None else scan_angle
    trial, arrays, provenance = load_scan_seed(parent, scan_dir, original, scan_angle)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use the GPU environment or device='cpu'.")
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    # Preserve the entire likelihood and other priors. Explicit bounds retain
    # the parent's support; Gaussian sigma never becomes a hard interval.
    config = copy.deepcopy(config)
    prior = dict(type="gaussian", mu=float(np.deg2rad(args.inclination_deg)),
                 sigma=float(np.deg2rad(args.inclination_sigma_deg)),
                 **{k: original["inclination_prior"][k] for k in ("min", "max")})
    config["mcmc"]["priors"]["velocity.inc_v"] = prior
    config["shared_comparison"] = str(output/"comparison.npz")
    config["mcmc"].update(nsteps=int(args.draws), nburn=int(args.warmup),
        nwalkers=int(args.walkers), seed=int(args.seed), resume=True,
        vectorize=args.device == "cuda", checkpoint=str(output/"checkpoint.npz"))
    config["mcmc"]["noise"]["ln_f"] = trial["full_params"]["noise.ln_f"]
    metadata = dict(kind="user_gaussian_inclination_prior", source_id=original["source_id"],
        inclination_deg=float(args.inclination_deg),
        inclination_sigma_deg=float(args.inclination_sigma_deg),
        description="User-specified Gaussian inclination prior for a separate scan-initialized run.",
        inherited_imaging_metadata=copy.deepcopy(original.get("inclination_metadata")),
        imaging_context_note="Inherited imaging fits are context; they do not determine this user-selected mean or width.")
    identity_config = copy.deepcopy(config)
    identity_config["mcmc"].pop("nsteps")  # Production may be extended safely.
    identity = dict(schema=SCHEMA, source_id=original["source_id"],
        provenance=provenance, config=identity_config, inclination_metadata=metadata,
        original_inclination_prior=original["inclination_prior"],
        comparison_sha256=original["comparison_sha256"],
        code={"three_file_inclination_fit.py": _sha(Path(__file__))},
        parent_identity=parent_identity)
    identity_path = output/"search_identity.json"
    if not identity_path.exists() and output.exists() and any(p.name != "run.log" for p in output.iterdir()):
        raise ValueError("Posterior output contains files without this scan-initialized run's identity.")
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("This scan-initialized posterior has different settings, prior or inputs. Use a new run name.")
    output.mkdir(parents=True, exist_ok=True)
    _guard_identity(identity_path, identity, description="scan-initialized posterior")
    weights = output/"comparison.npz"
    if weights.exists():
        _check_hash(weights, original["comparison_sha256"], "shared comparison")
    else:
        if (output/"start_search.json").exists():
            raise ValueError("The saved child comparison model is missing.")
        shutil.copyfile(parent/"comparison.npz", weights)
    _json(output/"comparison_identity.json", {"sha256": original["comparison_sha256"]})

    # The scan's likelihood and all other prior contributions are unchanged at
    # this seed. Only the inclination term is replaced; no optimization occurs.
    old_term = inclination_log_prior(trial["full_params"]["velocity.inc_v"], original["inclination_prior"])
    new_term = inclination_log_prior(trial["full_params"]["velocity.inc_v"], prior)
    trial.update(candidate_id=0, scan_row_index=provenance["scan_row_index"],
                 kind="inclination scan initialization; inclination sampled with Gaussian prior")
    trial["log_prior"] += new_term-old_term
    trial["log_posterior"] = trial["log_likelihood"]+trial["log_prior"]
    trial["log_inclination_prior"] = new_term
    trial["log_other_prior"] = trial["log_prior"]-new_term
    start_path = output/"start_search.json"
    if start_path.exists():
        search = json.loads(start_path.read_text())
        _check_hash(output/"candidates.npz", search.get("candidates_sha256"), "seed images")
        if (search.get("search_identity_sha256") != _sha(identity_path)
                or search.get("trials") != [trial]):
            raise ValueError("The saved scan initialization changed.")
    else:
        index = provenance["scan_row_index"]
        pixels = {k: arrays[k] for k in ("image", "observed_R", "observed_C", "mask")}
        pixels.update({k: arrays[k][index:index+1] for k in ("rectified_R", "rectified_C", "difference")})
        _npz(output/"candidates.npz", candidate_ids=np.array([0]), **pixels)
        search = {k: copy.deepcopy(original[k]) for k in
                  ("source_id", "redshift", "pixel_scale_arcsec", "r_fit_pixels", "noise_note", "display_mask_note")}
        search.update(schema="dingo-three-file-candidates-v1", trials=[trial], selected=0,
            search_settings=dict(method="inclination_scan_seed"), inclination_prior=prior,
            inclination_settings=dict(inclination_deg=float(args.inclination_deg),
                                      inclination_sigma_deg=float(args.inclination_sigma_deg)),
            inclination_metadata=metadata, seed_kind="inclination_scan", **provenance,
            original_inclination_prior=original["inclination_prior"], selection_note=SELECTION_NOTE,
            search_identity_sha256=_sha(identity_path), candidates_sha256=_sha(output/"candidates.npz"),
            comparison_sha256=original["comparison_sha256"])
        _json(start_path, search)
    (output/"search_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    LOG.info("ID%s: initialize at scanned i=%g deg; SAMPLE inclination with Gaussian prior %g +/- %g deg (%g..%g deg support)",
        original["source_id"], scan_angle, args.inclination_deg, args.inclination_sigma_deg,
        *np.rad2deg([prior["min"], prior["max"]]))
    LOG.info("%s", SELECTION_NOTE)
    sampling_args = SimpleNamespace(**vars(args), source=original["source_id"],
        pixel_scale=original["pixel_scale_arcsec"], candidate=0, selection_notes=SELECTION_NOTE)
    source_dir = Path(parent_identity["inputs"]["image"]["path"]).parent
    result = sample(sampling_args, config, output, source_dir, parent_identity["inputs"],
                    metadata, search, identity, started)
    if "velocity.inc_v" not in result["param_names"]:
        raise RuntimeError("Inclination was not sampled; do not use this posterior.")
    result.update(seed_kind="inclination_scan", **provenance,
        selection_method="inclination scan initialization with sampled inclination",
        selection_note=SELECTION_NOTE, original_inclination_prior=original["inclination_prior"])
    _json(output/"report.json", result)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-run", required=True)
    p.add_argument("--scan-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--inclination-deg", type=float, required=True)
    p.add_argument("--inclination-sigma-deg", type=float, required=True)
    p.add_argument("--scan-inclination-deg", type=float)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--draws", type=int, default=4000)
    p.add_argument("--walkers", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return p


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    run(parser().parse_args(argv))


if __name__ == "__main__":
    main()
