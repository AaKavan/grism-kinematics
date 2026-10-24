"""Initialize the unchanged three-file posterior at a user-specified V and i.

The parent broad search supplies the other starting parameters, inclination
prior, calibration, and frozen comparison weights. No optimization is run.
Both V_rot and inclination remain sampled, with the parent's original priors.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from .inclination_scan_fit import load_source, _check_hash
from .three_file_fit import REPO, _choose_candidate, _json, _sha
from .three_file_search import SharedComparisonFitter, records_and_pixels, sample

LOG = logging.getLogger(__name__)
SCHEMA = "dingo-three-file-manual-start-v1"
SELECTION_NOTE = (
    "Initialized around the supplied deprojected asymptotic V_inf and inclination, "
    "without optimization. Both are free during MCMC. Other starting parameters "
    "come from the recorded parent candidate. The parent's inclination prior "
    "and shared comparison model are unchanged; the starting inclination is not "
    "a new prior mean. Initialization does not restrict the chain to one solution: "
    "inspect convergence and compare independent starts before interpreting intervals."
)
_RUNTIME_FIELDS = ("nsteps", "nburn", "nwalkers", "seed", "resume", "vectorize",
                   "checkpoint", "backend", "checkpoint_every", "log_cadence")


def _validate_settings(args, config):
    for key, minimum in (("warmup", 0), ("draws", 1), ("walkers", 20), ("seed", 0)):
        value = getattr(args, key)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}.")
    if args.device not in ("cpu", "cuda"):
        raise ValueError("Use device='cpu' or device='cuda'.")
    if "V_rot" not in config["velocity"] or "V_sini" in config["velocity"]:
        raise ValueError("Manual V/i starts require a parent that samples V_rot and inclination separately.")
    priors = config["mcmc"]["priors"]
    for attr, name, scale in (("start_velocity", "velocity.V_rot", 1.),
                              ("start_inclination", "velocity.inc_v", np.pi/180)):
        value = getattr(args, attr)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
            raise ValueError(f"{attr} must be a finite number.")
        spec = priors[name]
        lo, hi = spec.get("min", -np.inf)/scale, spec.get("max", np.inf)/scale
        # Interior starts leave room for the required small Gaussian walker cloud.
        if not lo < value < hi or (name == "velocity.V_rot" and value <= 0):
            raise ValueError(f"{attr} must lie strictly inside the parent's prior support ({lo:g}, {hi:g}).")
    candidate = getattr(args, "base_candidate", None)
    if candidate is not None and (isinstance(candidate, (bool, np.bool_))
            or not isinstance(candidate, (int, np.integer))):
        raise ValueError("base_candidate must be an integer candidate ID or None for the highest posterior candidate.")


def _base_candidate(original, candidate):
    if candidate is None:
        eligible = [t for t in original["trials"] if t.get("log_posterior") is not None
                    and np.isfinite(t["log_posterior"]) and t.get("loss") is not None
                    and np.isfinite(t["loss"])]
        if not eligible:
            raise ValueError("The parent search contains no finite candidate.")
        candidate = max(eligible, key=lambda t: t["log_posterior"])["candidate_id"]
    return _choose_candidate(original, candidate)


def _runtime_config(original, args, output, full):
    config = copy.deepcopy(original)
    config["shared_comparison"] = str(output/"comparison.npz")
    config["velocity"].update(V_rot=full["velocity.V_rot"], inc_v=full["velocity.inc_v"])
    mc = config["mcmc"]
    mc.update(nsteps=int(args.draws), nburn=int(args.warmup), nwalkers=int(args.walkers),
              seed=int(args.seed), resume=True, vectorize=args.device == "cuda",
              checkpoint=str(output/"checkpoint.npz"), stage=0, init="adam")
    # The historical 'adam' label means a Gaussian cloud around current values;
    # fit_MCMC itself does no Adam optimization. 'prior' would ignore our seed.
    if mc.get("backend"):
        mc["backend"] = str(output/"chain.h5")
    mc["noise"]["ln_f"] = full["noise.ln_f"]
    return config


def _render_seed(config, parent, full, metadata, source_dir, device, directory):
    """Evaluate one exact seed; never optimize or draw posterior samples."""
    temporary_config = copy.deepcopy(config)
    temporary_config["shared_comparison"] = str(parent/"comparison.npz")
    config_path = directory/"seed_config.yaml"
    config_path.write_text(yaml.safe_dump(temporary_config, sort_keys=False))
    fitter = SharedComparisonFitter(str(config_path), device=torch.device(device))
    fitter.set_params_dict(full)
    state = fitter.prepare_mcmc()
    if state is None or not {"velocity.V_rot", "velocity.inc_v"}.issubset(state["param_names"]):
        raise ValueError("The manual-start posterior must freely sample both V_rot and inclination.")
    if "velocity.V_sini" in state["param_names"]:
        raise ValueError("The manual-start posterior cannot also sample V_sini.")
    # Avoid replacing the requested double-precision values with tensor roundoff.
    theta = np.asarray([full[name] for name in state["param_names"]], dtype=float)
    score = fitter.log_probability(theta)
    if not np.isfinite(score):
        raise ValueError("The requested V/i with this base candidate gives a nonfinite posterior. "
                         "Choose another starting point or base candidate.")
    rows = records_and_pixels(fitter, None, theta[None, :], np.asarray([score]), metadata,
        source_dir, directory/"candidates.npz", ["manual V/i initialization; both parameters sampled"])
    return rows, list(state["param_names"])


def run(args):
    started = time.monotonic()
    parent = Path(args.parent_run).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output == parent or output in parent.parents or parent in output.parents:
        raise ValueError("Use a separate posterior directory outside the parent search tree.")
    os.environ.setdefault("GRISM_CAL_DIR", str(REPO.parent/"grism_cal"))
    original, parent_identity, original_config = load_source(parent)
    _validate_settings(args, original_config)
    candidate_id, base = _base_candidate(original, getattr(args, "base_candidate", None))
    full = copy.deepcopy(base["full_params"])
    full.update({"velocity.V_rot": float(args.start_velocity),
                 "velocity.inc_v": float(np.deg2rad(args.start_inclination))})
    if "velocity.V_sini" in full:
        raise ValueError("The parent candidate uses V_sini; choose a V_rot parent search.")
    config = _runtime_config(original_config, args, output, full)
    provenance = dict(parent_run=str(parent), base_candidate_id=candidate_id,
        parent_search_sha256=_sha(parent/"start_search.json"),
        start_velocity_kms=float(args.start_velocity), start_inclination_deg=float(args.start_inclination))
    science_config = copy.deepcopy(config)
    for key in _RUNTIME_FIELDS:
        science_config["mcmc"].pop(key, None)
    identity = dict(schema=SCHEMA, source_id=original["source_id"], provenance=provenance,
        config=science_config, seed_full_params=full, parent_identity=parent_identity,
        comparison_sha256=original["comparison_sha256"],
        code={"three_file_manual_fit.py": _sha(Path(__file__)),
              "inclination_scan_fit.py": _sha(REPO/"dingo/inclination_scan_fit.py")})
    identity_path = output/"search_identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("This manual-start run has different starting values, base candidate, inputs or code. "
                             "Use a new run name or OVERRIDE=True.")
    elif output.exists() and any(p.name != "run.log" for p in output.iterdir()):
        raise ValueError("Manual-start output contains files without its preparation identity.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use the GPU environment or device='cpu'.")
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    manifest_path = output/"preparation.json"
    if identity_path.exists():
        if not manifest_path.is_file():
            raise ValueError("Manual-start preparation is incomplete. Use OVERRIDE=True to prepare again.")
        manifest = json.loads(manifest_path.read_text())
        for name in ("start_search.json", "candidates.npz", "comparison.npz", "search_identity.json"):
            _check_hash(output/name, manifest.get("files_sha256", {}).get(name), f"manual-start {name}")
        search = json.loads((output/"start_search.json").read_text())
        _check_hash(output/"comparison.npz", original["comparison_sha256"], "shared comparison")
    else:
        # Invalid seeds leave no half-created run; evaluate and render in scratch.
        source_dir = Path(parent_identity["inputs"]["image"]["path"]).parent
        with tempfile.TemporaryDirectory(prefix="dingo_manual_seed_") as scratch:
            scratch = Path(scratch)
            rows, names = _render_seed(config, parent, full, original, source_dir, args.device, scratch)
            output.mkdir(parents=True, exist_ok=True)
            _json(identity_path, identity)
            shutil.copyfile(parent/"comparison.npz", output/"comparison.npz")
            shutil.copyfile(scratch/"candidates.npz", output/"candidates.npz")
        search = {key: copy.deepcopy(original[key]) for key in
                  ("source_id", "redshift", "pixel_scale_arcsec", "r_fit_pixels", "noise_note", "display_mask_note")}
        search.update(schema="dingo-three-file-candidates-v1", manual_start_schema=SCHEMA,
            trials=rows, selected=0, search_settings={"method": "manual_start"},
            inclination_prior=copy.deepcopy(original["inclination_prior"]),
            inclination_settings=copy.deepcopy(original.get("inclination_settings", {})),
            inclination_metadata=copy.deepcopy(original.get("inclination_metadata")),
            seed_kind="manual_start", **provenance, param_names=names,
            selection_note=SELECTION_NOTE, prepared_utc=datetime.now(timezone.utc).isoformat(),
            search_identity_sha256=_sha(identity_path), candidates_sha256=_sha(output/"candidates.npz"),
            comparison_sha256=original["comparison_sha256"])
        _json(output/"start_search.json", search)
        _json(manifest_path, dict(schema=SCHEMA, files_sha256={name: _sha(output/name) for name in
            ("start_search.json", "candidates.npz", "comparison.npz", "search_identity.json")}))
        _json(output/"comparison_identity.json", {"sha256": original["comparison_sha256"]})
        (output/"search_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    LOG.info("ID%s: manual V_inf=%g km/s, i=%g deg; other starting values from candidate %d.",
        original["source_id"], args.start_velocity, args.start_inclination, candidate_id)
    LOG.info("%s", SELECTION_NOTE)
    if getattr(args, "prepare_only", False):
        # Reopening a prepared seed must not rewrite a running sampler's config.
        if not (output/"search_config.yaml").exists():
            (output/"search_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        return search

    sampling_args = SimpleNamespace(**vars(args), source=original["source_id"],
        pixel_scale=original["pixel_scale_arcsec"], candidate=0, selection_notes=SELECTION_NOTE)
    source_dir = Path(parent_identity["inputs"]["image"]["path"]).parent
    result = sample(sampling_args, config, output, source_dir, parent_identity["inputs"],
                    original.get("inclination_metadata"), search, identity, started)
    if not {"velocity.V_rot", "velocity.inc_v"}.issubset(result["param_names"]):
        raise RuntimeError("V_rot and inclination were not both sampled; do not use this posterior.")
    result.update(seed_kind="manual_start", manual_start_schema=SCHEMA, **provenance,
        selection_method="manual V/i initialization with both parameters sampled", selection_note=SELECTION_NOTE)
    _json(output/"report.json", result)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-run", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--start-velocity", type=float, required=True)
    p.add_argument("--start-inclination", type=float, required=True)
    p.add_argument("--base-candidate", type=int)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--draws", type=int, default=8000)
    p.add_argument("--walkers", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return p


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    run(parser().parse_args(argv))


if __name__ == "__main__":
    main()
