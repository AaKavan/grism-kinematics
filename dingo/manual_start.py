"""Choose initial V_inf and either a sampled or fixed inclination.

This companion to rotation_only.ipynb uses a completed broad search for the
remaining starting coordinates and its frozen likelihood. Each start has a
separate run directory; viewing saved output never starts a fit.
"""
from numbers import Real
from pathlib import Path

import numpy as np

from .real_workflow import read_json, run_command
from .three_file import CandidateResults, ThreeFileResults, _paths


SCHEMA = "dingo-three-file-manual-start-v1"
FIXED_SCHEMA = "dingo-three-file-manual-fixed-inclination-v1"


def _locations(source_id, *, parent_run, start_velocity_kms, start_inclination_deg,
               base_candidate_id=None, run_label="manual_v1", results=None, fix_inclination=False):
    if not isinstance(fix_inclination, bool):
        raise ValueError("fix_inclination must be True or False.")
    for name, value in (("start_velocity_kms", start_velocity_kms),
                        ("start_inclination_deg", start_inclination_deg)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")
    if start_velocity_kms <= 0 or not 5 < start_inclination_deg < 85:
        raise ValueError("Use a positive starting V_inf and an inclination strictly inside 5–85 degrees.")
    if base_candidate_id is not None and (isinstance(base_candidate_id, (bool, np.bool_))
            or not isinstance(base_candidate_id, (int, np.integer)) or base_candidate_id < 0):
        raise ValueError("base_candidate_id must be a nonnegative candidate ID or None.")
    source, parent = _paths(source_id, parent_run, results)
    _paths(source_id, run_label, results)  # Also validate the optional experiment label.
    def token(value):
        return str(float(value)).removesuffix(".0").replace(".", "p").replace("+", "")
    base = "best" if base_candidate_id is None else str(base_candidate_id)
    name = f"{parent_run}_{run_label}_v{token(start_velocity_kms)}_i{token(start_inclination_deg)}_c{base}"
    if fix_inclination:
        name += "_fixedi"
    _, output = _paths(source_id, name, results)
    return source, parent, output


def final_root(working_dir, **location):
    """Export root for this experiment; result.export adds the ID<source> folder."""
    _, _, output = _locations(**location)
    return Path(working_dir).expanduser().resolve() / output.name


def _check_report(report, parent, start_velocity_kms, start_inclination_deg, base_candidate_id,
                  fix_inclination=False):
    if (report.get("seed_kind") != "manual_start"
            or report.get("manual_start_schema") != (FIXED_SCHEMA if fix_inclination else SCHEMA)
            or report.get("inclination_mode", "sampled") != ("fixed" if fix_inclination else "sampled")
            or Path(report.get("parent_run", "")).resolve() != parent.resolve()
            or not np.isclose(report.get("start_velocity_kms", np.nan), start_velocity_kms, rtol=0, atol=1e-10)
            or not np.isclose(report.get("start_inclination_deg", np.nan), start_inclination_deg, rtol=0, atol=1e-10)
            or isinstance(report.get("base_candidate_id"), bool)
            or not isinstance(report.get("base_candidate_id"), int)
            or report.get("base_candidate_id", -1) < 0
            or (base_candidate_id is not None and report.get("base_candidate_id") != base_candidate_id)):
        raise ValueError("The saved manual start does not match these starting values, parent or base candidate.")
    if fix_inclination:
        prior = report.get("inclination_prior", {})
        if (not np.isclose(report.get("fixed_inclination_deg", np.nan), start_inclination_deg, rtol=0, atol=1e-10)
                or prior.get("type") != "fixed"
                or not np.isclose(prior.get("value", np.nan), np.deg2rad(start_inclination_deg), rtol=0, atol=1e-10)):
            raise ValueError("The saved fixed inclination disagrees with the requested angle.")


def load_start(source_id, *, parent_run, start_velocity_kms, start_inclination_deg,
               base_candidate_id=None, run_label="manual_v1", results=None, fix_inclination=False):
    """Read the exact saved starting residual; no search, optimization or sampling."""
    source, parent, output = _locations(source_id, parent_run=parent_run,
        start_velocity_kms=start_velocity_kms, start_inclination_deg=start_inclination_deg,
        base_candidate_id=base_candidate_id, run_label=run_label, results=results, fix_inclination=fix_inclination)
    result = CandidateResults(output, source.source_id)
    _check_report(result.report, parent, start_velocity_kms, start_inclination_deg, base_candidate_id, fix_inclination)
    if result.velocity_parameter != "V_rot":
        raise ValueError("The manual-start experiment must use deprojected V_rot.")
    return result


def load(source_id, *, parent_run, start_velocity_kms, start_inclination_deg,
         base_candidate_id=None, run_label="manual_v1", results=None, fix_inclination=False):
    """Read this experiment's saved posterior without needing its parent files."""
    source, parent, output = _locations(source_id, parent_run=parent_run,
        start_velocity_kms=start_velocity_kms, start_inclination_deg=start_inclination_deg,
        base_candidate_id=base_candidate_id, run_label=run_label, results=results, fix_inclination=fix_inclination)
    result = ThreeFileResults(output, source.source_id)
    _check_report(result.report, parent, start_velocity_kms, start_inclination_deg, base_candidate_id, fix_inclination)
    names, fixed = result.report.get("param_names", []), result.report.get("fixed_params", {})
    if result.velocity_parameter != "V_rot" or "velocity.V_rot" in fixed:
        raise ValueError("Deprojected V_rot must be sampled in this experiment.")
    if fix_inclination:
        if ("velocity.inc_v" in names
                or not np.isclose(fixed.get("velocity.inc_v", np.nan), np.deg2rad(start_inclination_deg),
                                  rtol=0, atol=1e-6)):
            raise ValueError("Inclination must be fixed at the requested angle, not sampled.")
    elif "velocity.inc_v" not in names or "velocity.inc_v" in fixed:
        raise ValueError("Both deprojected V_rot and inclination must be sampled in this experiment.")
    return result


def _preflight(source, parent, start_velocity_kms, start_inclination_deg, base_candidate_id):
    import yaml
    if not (parent / "start_search.json").is_file():
        raise FileNotFoundError(f"No saved candidate search at {parent}. Run the search in notebooks/legacy/rotation_only.ipynb "
                                "for this source first; the manual notebook reuses its prior and geometry starts.")
    candidates = CandidateResults(parent, source.source_id)
    if candidates.report.get("search_settings", {}).get("method") != "broad" or candidates.velocity_parameter != "V_rot":
        raise ValueError("Choose a saved broad search that samples deprojected V_rot and inclination.")
    if base_candidate_id is not None and base_candidate_id not in candidates.candidate_ids:
        raise ValueError(f"Base candidate {base_candidate_id} is not in the parent's candidate table.")
    config = yaml.safe_load((parent / "search_config.yaml").read_text())
    for name, value in (("velocity.V_rot", start_velocity_kms),
                        ("velocity.inc_v", np.deg2rad(start_inclination_deg))):
        prior = config["mcmc"]["priors"][name]
        if not prior["min"] < value < prior["max"]:
            raise ValueError(f"The starting {name} is outside the saved prior support.")
    return candidates.report["inclination_prior"]


def _launch(source_id, *, parent_run, start_velocity_kms, start_inclination_deg,
            base_candidate_id=None, run_label="manual_v1", warmup=1000, samples=8000,
            walkers=64, seed=42, override=False, device="cuda", results=None, prepare_only=False,
            fix_inclination=False):
    location = dict(source_id=source_id, parent_run=parent_run, start_velocity_kms=start_velocity_kms,
        start_inclination_deg=start_inclination_deg, base_candidate_id=base_candidate_id,
        run_label=run_label, results=results, fix_inclination=fix_inclination)
    source, parent, output = _locations(**location)
    if not isinstance(override, bool) or device not in ("cpu", "cuda"):
        raise ValueError("Use a boolean override and device='cpu' or 'cuda'.")
    for name, value, minimum in (("warmup", warmup, 0), ("samples", samples, 1),
                                 ("walkers", walkers, 20), ("seed", seed, 0)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    for path in (source.results / "three_file_sources", source.directory, output):
        if path.is_symlink():
            raise ValueError("The manual-run directory must not be symlinked.")
    prior = _preflight(source, parent, start_velocity_kms, start_inclination_deg, base_candidate_id)
    if override and output.exists() and any(p.name != "run.log" for p in output.iterdir()):
        identity_path = output / "search_identity.json"
        identity = read_json(identity_path) if identity_path.is_file() else {}
        if (identity.get("schema") != (FIXED_SCHEMA if fix_inclination else SCHEMA)
                or identity.get("source_id") != source.source_id
                or identity.get("provenance", {}).get("parent_run") != str(parent)):
            raise ValueError("Override can replace only this source's identified manual-start run.")
    worker = "dingo.three_file_fixed_i_fit" if fix_inclination else "dingo.three_file_manual_fit"
    arguments = ["-m", worker, "--parent-run", str(parent),
        "--output", str(output), "--start-velocity", str(float(start_velocity_kms)),
        "--start-inclination", str(float(start_inclination_deg)),
        "--warmup", str(warmup), "--draws", str(samples), "--walkers", str(walkers),
        "--seed", str(seed), "--device", device]
    if base_candidate_id is not None:
        arguments += ["--base-candidate", str(base_candidate_id)]
    if prepare_only:
        arguments.append("--prepare-only")
    print(f"ID{source.source_id}: manual starting V_inf={start_velocity_kms:g} km/s, "
          f"i={start_inclination_deg:g}°. "
          + ("Inclination is FIXED; V_inf and the other parameters remain free."
             if fix_inclination else "Both remain free during MCMC."), flush=True)
    if fix_inclination:
        print("The parent inclination prior is replaced by an exact fixed angle. "
              "Velocity errors exclude uncertainty in that angle.", flush=True)
    elif prior.get("type") == "gaussian":
        print(f"Unchanged inclination prior: {np.rad2deg(prior['mu']):.2f}° ± "
              f"{np.rad2deg(prior['sigma']):.2f}° (Gaussian 1σ before truncation).", flush=True)
    else:
        print(f"Unchanged inclination prior: {prior}", flush=True)
    if prepare_only:
        print("Rendering the exact starting residual; no optimization or MCMC in this cell.", flush=True)
    else:
        print(f"{walkers} walkers × {warmup:,} warmup + {samples:,} retained steps. "
              "An existing checkpoint resumes where it stopped.", flush=True)
    print(f"Progress and checkpoints: {output}", flush=True)
    from .run_output import output_run
    with output_run(output, source, kind="rotation", override=override):
        run_command(arguments, output / "run.log", device=device)
    return (load_start if prepare_only else load)(**location)


def prepare(source_id, **kwargs):
    """Save and display-ready the exact starting fit, without optimizing or sampling."""
    return _launch(source_id, prepare_only=True, **kwargs)


def fit(source_id, **kwargs):
    """Run V_inf MCMC, sampling i by default or holding it with fix_inclination=True."""
    return _launch(source_id, prepare_only=False, **kwargs)
