"""Sample a separate posterior from a scan fit, with inclination remaining free.

The Gaussian mean also selects the saved scan row used for initialization.
Only the worker samples; loading a result never starts or resets a run.
"""
from pathlib import Path
from numbers import Real

import numpy as np

from .real_workflow import read_json, run_command
from .three_file import DEFAULT_WORKING, ThreeFileResults, _paths


SCHEMA = "dingo-three-file-inclination-prior-run-v1"


def _settings(inclination_deg, inclination_sigma_deg):
    for name, value in (("inclination_deg", inclination_deg),
                        ("inclination_sigma_deg", inclination_sigma_deg)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value)):
            raise ValueError(f"{name} must be a finite number in degrees.")
    if not 5 < inclination_deg < 85 or inclination_sigma_deg <= 0:
        raise ValueError("Use a Gaussian mean inside 5–85 degrees and a positive sigma.")
    return float(inclination_deg), float(inclination_sigma_deg)


def _locations(source_id, run_name, inclination_deg, inclination_sigma_deg, results):
    mean, sigma = _settings(inclination_deg, inclination_sigma_deg)
    source, parent = _paths(source_id, run_name, results)
    def token(value):
        return str(value).removesuffix(".0").replace(".", "p").replace("+", "")
    name = f"{run_name}_i{token(mean)}_s{token(sigma)}"
    _, output = _paths(source_id, name, results)
    return source, parent, parent.with_name(parent.name + "_inclination_scan"), output


def final_root(working_dir=DEFAULT_WORKING, *, run_name, inclination_deg,
               inclination_sigma_deg=10.):
    """Root passed to result.export; preserves the original WORKING_DIR/ID folder."""
    _, _, _, output = _locations(1, run_name, inclination_deg, inclination_sigma_deg, None)
    return Path(working_dir).expanduser() / "inclination_prior_runs" / output.name


def load(source_id, *, run_name, inclination_deg, inclination_sigma_deg=10., results=None):
    """Read this exact Gaussian-prior run without accessing or changing its parent."""
    source, parent, _, output = _locations(
        source_id, run_name, inclination_deg, inclination_sigma_deg, results)
    result = ThreeFileResults(output, source.source_id)
    report = result.report
    prior = report.get("inclination_prior", {})
    if (report.get("seed_kind") != "inclination_scan"
            or Path(report.get("parent_run", "")).resolve() != parent.resolve()
            or not np.isclose(report.get("seed_inclination_deg", np.nan), inclination_deg,
                              rtol=0, atol=1e-8)
            or prior.get("type") != "gaussian"
            or not np.allclose([prior.get("mu", np.nan), prior.get("sigma", np.nan),
                                prior.get("min", np.nan), prior.get("max", np.nan)],
                               np.deg2rad([inclination_deg, inclination_sigma_deg, 5., 85.]),
                               rtol=0, atol=1e-12)
            or "velocity.inc_v" not in report.get("param_names", [])
            or "velocity.inc_v" in report.get("fixed_params", {})):
        raise ValueError("The saved posterior does not match this scan seed and sampled Gaussian inclination prior.")
    return result


def fit(source_id, *, run_name, inclination_deg, inclination_sigma_deg=10.,
        warmup=1000, samples=4000, walkers=64, seed=42, override=False,
        device="cuda", results=None):
    """Initialize from a saved scan row and sample all original free parameters.

    The supplied Gaussian replaces the parent inclination prior. It does not
    fix inclination or bound it to one sigma. The existing 5–85 degree support
    is retained. Progress, checkpoints and final draws live in a separate run.
    """
    source, parent, scan_dir, output = _locations(
        source_id, run_name, inclination_deg, inclination_sigma_deg, results)
    if not isinstance(override, bool) or device not in ("cpu", "cuda"):
        raise ValueError("Use a boolean override and device='cpu' or 'cuda'.")
    for name, value, minimum in (("warmup", warmup, 0), ("samples", samples, 1),
                                 ("walkers", walkers, 20), ("seed", seed, 0)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    for path in (source.results / "three_file_sources", source.directory, output):
        if path.is_symlink():
            raise ValueError("The posterior directory must not be symlinked.")
    # Check the requested scan row before a reset or worker invocation.
    from .inclination_scan import load as load_scan
    scan = load_scan(source_id, run_name=run_name, results=results)
    if not np.any(np.isclose(scan.angles, inclination_deg, rtol=0, atol=1e-8)):
        raise ValueError(f"No saved scan row at {inclination_deg:g} degrees; choose from {scan.angles.tolist()}.")
    if override and output.exists() and any(p.name != "run.log" for p in output.iterdir()):
        identity_path = output / "search_identity.json"
        identity = read_json(identity_path) if identity_path.is_file() else {}
        if (identity.get("schema") != SCHEMA or identity.get("source_id") != source.source_id
                or identity.get("provenance", {}).get("parent_run") != str(parent)):
            raise ValueError("Override can replace only this source's identified inclination-prior run.")
    arguments = ["-m", "dingo.three_file_inclination_fit",
                 "--parent-run", str(parent), "--scan-dir", str(scan_dir),
                 "--output", str(output), "--inclination-deg", str(float(inclination_deg)),
                 "--inclination-sigma-deg", str(float(inclination_sigma_deg)),
                 "--warmup", str(warmup), "--draws", str(samples),
                 "--walkers", str(walkers), "--seed", str(seed), "--device", device]
    print(f"ID{source.source_id}: start from the {inclination_deg:g}° scan fit.\n"
          f"Inclination is sampled with Gaussian prior {inclination_deg:g}° ± {inclination_sigma_deg:g}° "
          "(1σ; truncated to 5–85°).\n"
          f"{walkers} walkers × {warmup:,} warmup + {samples:,} retained steps.\n"
          f"Progress and checkpoints: {output}", flush=True)
    from .run_output import output_run
    with output_run(output, source, kind="rotation", override=override):
        run_command(arguments, output / "run.log", device=device)
    return load(source_id, run_name=run_name, inclination_deg=inclination_deg,
                inclination_sigma_deg=inclination_sigma_deg, results=results)
