"""Manual V initialization with inclination held constant throughout MCMC.

Reuse a verified broad search's inputs, comparison weights and other priors.
The parent's inclination prior is recorded for provenance, but is not applied
to this conditional fit. V_rot and the other original parameters remain free.
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

from astropy.io import fits
import numpy as np
import torch
import yaml

from .inclination_scan_fit import load_source, _check_hash
from .three_file_fit import REPO, _json, _npz, _sha
from .three_file_manual_fit import _base_candidate, _runtime_config, _validate_settings, _RUNTIME_FIELDS
from .three_file_plot import velocity_summary
from .three_file_search import SharedComparisonFitter, sample

LOG = logging.getLogger(__name__)
SCHEMA = "dingo-three-file-manual-fixed-inclination-v1"
PARAM_NAMES = frozenset(("velocity.V_rot", "velocity.R_v", "velocity.x0_v", "velocity.y0_v",
    "velocity.theta_v", "image.R.dx", "image.R.dy", "noise.ln_f"))
SELECTION_NOTE = (
    "Inclination is fixed exactly at the supplied value and is not sampled. "
    "V_inf and the other original parameters remain free. Initialization uses "
    "the supplied V_inf and the recorded parent candidate's other parameters, "
    "without optimization. The parent's imaging inclination prior is recorded "
    "but not applied. Velocity intervals are conditional on this fixed inclination "
    "and exclude inclination uncertainty; inspect convergence and compare fixed "
    "inclinations and independent starts before interpreting them."
)


def _fixed_config(original, args, output, full):
    config = _runtime_config(original, args, output, full)
    inc = float(full["velocity.inc_v"])
    # Every stage uses the identical geometry, including initialization and MAP
    # rendering. A zero-width Gaussian would still sample a ninth parameter.
    for stage in config["fitting"]:
        overrides = stage.setdefault("override", {})
        for key in list(overrides):
            parts = [p.strip() for p in key.split(",")]
            if "velocity.inc_v" in parts:
                settings = overrides.pop(key)
                for part in parts:
                    if part != "velocity.inc_v":
                        overrides[part] = copy.deepcopy(settings)
        overrides["velocity.inc_v"] = {"fit": False, "min": 0., "max": float(np.pi/2)}
    config["velocity"]["inc_v"] = inc
    config["mcmc"].setdefault("init_scatter", {}).pop("velocity.inc_v", None)
    # This is descriptive metadata consumed by the shared report writer. The
    # fitter builds priors only for fitted parameters, so this key is never
    # interpreted as a sampled prior.
    config["mcmc"]["priors"]["velocity.inc_v"] = {"type": "fixed", "value": inc}
    return config


def _render_seed(config, parent, full, metadata, source_dir, device, directory):
    """Evaluate the exact requested eight-dimensional seed without optimization."""
    temporary_config = copy.deepcopy(config)
    temporary_config["shared_comparison"] = str(parent/"comparison.npz")
    config_path = directory/"seed_config.yaml"
    config_path.write_text(yaml.safe_dump(temporary_config, sort_keys=False))
    fitter = SharedComparisonFitter(str(config_path), device=torch.device(device))
    fitter.set_params_dict(full)
    state = fitter.prepare_mcmc()
    if state is None or set(state["param_names"]) != PARAM_NAMES:
        raise ValueError("Fixed-inclination sampling requires the original eight free parameters, excluding inclination.")
    for stage in fitter.param_config_lists["velocity"]:
        cfg = stage["velocity.inc_v"]
        if cfg.fit or not np.isclose(float(cfg.tensor.detach().cpu()), full["velocity.inc_v"], rtol=0, atol=1e-7):
            raise ValueError("Inclination was not fixed consistently in every fitting stage.")
    theta = np.asarray([full[name] for name in state["param_names"]], dtype=float)
    score = fitter.log_probability(theta)
    if not np.isfinite(score):
        raise ValueError("The requested V/fixed-i with this base candidate gives a nonfinite posterior. "
                         "Choose another starting velocity, fixed inclination or base candidate.")
    r, c, _, _ = fitter.get_fitting_results()
    seed_full = dict(zip(state["param_names"], map(float, theta)))
    seed_full.update({"velocity.inc_v": float(full["velocity.inc_v"]),
                     "velocity.v_sys": 0., "image.C.dx": 0., "image.C.dy": 0.})
    params = {k.split(".", 1)[1]: v for k, v in seed_full.items() if k.startswith("velocity.")}
    v = velocity_summary(params, metadata)
    prior = float(fitter.log_prior(theta))
    row = dict(candidate_id=0, kind="manual V initialization; inclination fixed", params=params,
        full_params=seed_full, loss=float(np.sum((r-c)**2)), log_posterior=float(score),
        log_prior=prior, log_likelihood=float(score-prior),
        velocity_summary=dict(V_sini_kms=v["v_sini"], V_inf_kms=v["v_inf"],
                              inclination_deg=v["inclination_deg"], R_v_pixels=params["R_v"]),
        flux_retained=fitter._flux_retained())
    observed_r, observed_c = fitter.get_true_images()
    _npz(directory/"candidates.npz", candidate_ids=np.asarray([0]),
         image=fits.getdata(source_dir/"image.fits"), observed_R=observed_r, observed_C=observed_c,
         rectified_R=r[None, ...], rectified_C=c[None, ...], difference=(r-c)[None, ...],
         mask=fitter.mcmc_mask.cpu().numpy())
    return [row], list(state["param_names"])


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
        raise ValueError("Fixed-i V starts require a V_rot parent search.")
    config = _fixed_config(original_config, args, output, full)
    provenance = dict(parent_run=str(parent), base_candidate_id=candidate_id,
        parent_search_sha256=_sha(parent/"start_search.json"),
        start_velocity_kms=float(args.start_velocity), start_inclination_deg=float(args.start_inclination),
        inclination_mode="fixed", fixed_inclination_deg=float(args.start_inclination))
    fixed_prior = copy.deepcopy(config["mcmc"]["priors"]["velocity.inc_v"])
    parent_prior = copy.deepcopy(original["inclination_prior"])
    metadata = dict(kind="user_fixed_inclination", fixed_inclination_deg=float(args.start_inclination),
                    inherited_imaging_metadata=copy.deepcopy(original.get("inclination_metadata")))
    science_config = copy.deepcopy(config)
    for key in _RUNTIME_FIELDS:
        science_config["mcmc"].pop(key, None)
    identity = dict(schema=SCHEMA, source_id=original["source_id"], provenance=provenance,
        config=science_config, seed_full_params=full, parent_identity=parent_identity,
        comparison_sha256=original["comparison_sha256"],
        code={name: _sha(REPO/"dingo"/name) for name in
              ("three_file_fixed_i_fit.py", "three_file_manual_fit.py", "inclination_scan_fit.py")})
    identity_path = output/"search_identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("This fixed-inclination run has different starting values, base candidate, inputs or code. "
                             "Use a new run name or OVERRIDE=True.")
    elif output.exists() and any(p.name != "run.log" for p in output.iterdir()):
        raise ValueError("Fixed-inclination output contains files without its preparation identity.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use the GPU environment or device='cpu'.")
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    manifest_path = output/"preparation.json"
    if identity_path.exists():
        if not manifest_path.is_file():
            raise ValueError("Fixed-inclination preparation is incomplete. Use OVERRIDE=True to prepare again.")
        manifest = json.loads(manifest_path.read_text())
        for name in ("start_search.json", "candidates.npz", "comparison.npz", "search_identity.json"):
            _check_hash(output/name, manifest.get("files_sha256", {}).get(name), f"fixed-inclination {name}")
        search = json.loads((output/"start_search.json").read_text())
        _check_hash(output/"comparison.npz", original["comparison_sha256"], "shared comparison")
    else:
        source_dir = Path(parent_identity["inputs"]["image"]["path"]).parent
        with tempfile.TemporaryDirectory(prefix="dingo_fixed_i_seed_") as scratch:
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
            inclination_prior=fixed_prior, parent_inclination_prior=parent_prior,
            inclination_settings={"inclination_mode": "fixed", "fixed_inclination_deg": float(args.start_inclination)},
            inclination_metadata=metadata, seed_kind="manual_start", **provenance, param_names=names,
            selection_note=SELECTION_NOTE, prepared_utc=datetime.now(timezone.utc).isoformat(),
            search_identity_sha256=_sha(identity_path), candidates_sha256=_sha(output/"candidates.npz"),
            comparison_sha256=original["comparison_sha256"])
        _json(output/"start_search.json", search)
        _json(manifest_path, dict(schema=SCHEMA, files_sha256={name: _sha(output/name) for name in
            ("start_search.json", "candidates.npz", "comparison.npz", "search_identity.json")}))
        _json(output/"comparison_identity.json", {"sha256": original["comparison_sha256"]})
        (output/"search_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    LOG.info("ID%s: manual V_inf=%g km/s; inclination FIXED at %g deg; other starts from candidate %d.",
             original["source_id"], args.start_velocity, args.start_inclination, candidate_id)
    LOG.info("%s", SELECTION_NOTE)
    if getattr(args, "prepare_only", False):
        if not (output/"search_config.yaml").exists():
            (output/"search_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        return search

    sampling_args = SimpleNamespace(**vars(args), source=original["source_id"],
        pixel_scale=original["pixel_scale_arcsec"], candidate=0, selection_notes=SELECTION_NOTE)
    source_dir = Path(parent_identity["inputs"]["image"]["path"]).parent
    result = sample(sampling_args, config, output, source_dir, parent_identity["inputs"],
                    metadata, search, identity, started)
    if set(result["param_names"]) != PARAM_NAMES:
        raise RuntimeError("The fixed-inclination posterior sampled the wrong parameter set; do not use it.")
    actual_inc = result.get("fixed_params", {}).get("velocity.inc_v")
    if actual_inc is None or not np.isclose(actual_inc, full["velocity.inc_v"], rtol=0, atol=1e-7):
        raise RuntimeError("Saved predictions do not use the requested fixed inclination; do not use this posterior.")
    result.update(seed_kind="manual_start", manual_start_schema=SCHEMA, **provenance,
        inclination_prior=fixed_prior, parent_inclination_prior=parent_prior, inclination_metadata=metadata,
        selection_method="manual V initialization with fixed inclination", selection_note=SELECTION_NOTE)
    _json(output/"report.json", result)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-run", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--start-velocity", type=float, required=True)
    p.add_argument("--start-inclination", type=float, required=True, help="Inclination held fixed, in degrees.")
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
