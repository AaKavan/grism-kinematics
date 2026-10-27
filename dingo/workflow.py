"""Explicit, per-source workflows for the original rotation-only R+C model.

Each :class:`Run` identifies one experiment, not a global current source. The
functions below prepare inputs, search, sample, inspect and export separately.
They delegate all scientific calculations to the existing three-file workers.
Importing this module does not initialize a GPU or launch a fit.

Example: search, inspect, then choose a candidate explicitly::

    from dingo import workflow as wf

    run = wf.prepare_source(3700, working_dir="/path/to/results",
                            inclination_sigma_deg=5.0)
    imaging = wf.fit_morphology(run)
    candidates = wf.find_candidates(run)
    wf.show_candidates(run)
    chosen_id = int(input("Candidate ID selected after inspecting residuals: "))
    posterior = wf.fit_source(run, candidate_id=chosen_id, draws=8000)
    wf.show_results(run)
    wf.save_results(run)

A manual experiment uses ``start_velocity_kms`` and optionally
``start_inclination_deg`` in ``prepare_source``. An existing ``parent_search``
can provide the other starting coordinates; it is read, never overwritten.
Changing a starting inclination does NOT change the inclination prior.

The sampled amplitude is deprojected asymptotic V_inf (backend name V_rot).
Intrinsic dispersion is exactly zero. Inclination can have an imaging-centered
Gaussian, an explicitly chosen Gaussian, or an exact fixed value. The default
Gaussian width is a USER-SELECTED TOTAL 1-sigma width of 5 degrees, truncated to
5--85 degrees. It is not an additional scatter added to the imaging uncertainty.
Fixed-inclination intervals exclude inclination uncertainty.

Sampling uses emcee ensemble proposals with a PyTorch likelihood, batched on
the GPU for device='cuda'. Walkers interact; they are not independent chains.
Finishing a requested number of steps does not certify convergence or establish
that all solutions were found. Visual review and independent starts remain
necessary. The likelihood retains the existing approximate stacked-image noise
model. Mass profiles are spherical-equivalent rotation-supported estimates,
without pressure support or lensing corrections.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import re
import shutil
import time

from .real_workflow import REPO, read_json, sha256, source_number, write_json
from .progress import run_command as _stream_command


SCHEMA = "dingo-workflow-v1"
_DETAILS_MARKER = "\n--- POSTERIOR DETAILS ---\n"


def run_command(arguments, log_path, device="cuda"):
    """Show a live progress bar for a worker inside a child-held checkpoint lock.

    The child lock remains held even if its notebook kernel disappears. This
    lets a later session distinguish a live orphan worker from a stale file
    before resuming or explicitly deleting its outputs.
    """
    output = Path(log_path).parent
    # Manual workers require an otherwise empty first output directory. Keep
    # their child-held lock beside that directory, inside the same run cache.
    lock_path = output.parent / "_worker_locks" / output.name / "worker.lock"
    _no_symlinks(lock_path)
    wrapped = ["-m", "dingo._workflow_worker", "--lock", str(lock_path), "--", *arguments]
    return _stream_command(wrapped, log_path, device=device)


@dataclass(frozen=True)
class Run:
    """Immutable handle to one source/experiment; settings live in run.json.

    Obtain a handle with ``prepare_source`` or ``load_run``. Keeping several
    handles in a notebook lets sources run back to back without global state.
    Properties read current metadata; changing a returned dict changes no files.
    """

    source_id: int
    directory: Path

    @property
    def configuration(self) -> dict:
        """A fresh copy of the scientific choices for this experiment."""
        return _record(self)["configuration"]

    @property
    def search_directory(self) -> Path:
        """Local search checkpoints or an explicitly attached external search."""
        parent = self.configuration["parent_search"]
        return Path(parent) if parent else self.directory / "_checkpoints" / "search"

    @property
    def posterior_directory(self) -> Path:
        """MCMC checkpoints, progress log and the worker's posterior report."""
        return self.directory / "_checkpoints" / "posterior"


@dataclass(frozen=True)
class BatchResult:
    """Outcome of one sequential batch entry, retaining ordinary exceptions."""

    run: Run
    result: object | None = None
    error: Exception | None = None

    @property
    def succeeded(self) -> bool:
        """Whether the requested sampling finished (not a convergence verdict)."""
        return self.error is None and self.result is not None


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _number(name, value, *, positive=True) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive.")
    return float(value)


def _integer(name, value, minimum=1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _angle(name, value) -> float:
    value = _number(name, value)
    if not 5 < value < 85:
        raise ValueError(f"{name} must lie strictly between 5 and 85 degrees.")
    return value


def _path(value) -> Path:
    return Path(value).expanduser().resolve()


def _no_symlinks(path):
    """Do not permit an output path to redirect writes/deletion through a link."""
    path = Path(os.path.abspath(Path(path).expanduser()))
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError(f"Workflow output paths must not be symlinked: {component}")


def _record(run: Run) -> dict:
    path = run.directory / "run.json"
    record = read_json(path)
    if (record.get("schema") != SCHEMA or record.get("source_id") != run.source_id
            or record.get("run_name") != run.directory.name):
        raise ValueError(f"This is not the identified workflow run for ID{run.source_id}: {path}")
    return record


def _write_record(run: Run, record: dict):
    """Update machine/human records, preserving the exported posterior details."""
    _no_symlinks(run.directory)
    for name in ("run.json", "run.json.tmp", "run_info.txt", "run_info.txt.tmp"):
        _no_symlinks(run.directory / name)
    record["updated_utc"] = _utc()
    write_json(run.directory / "run.json", record)
    path = run.directory / "run_info.txt"
    suffix = ""
    if path.exists() and _DETAILS_MARKER in path.read_text():
        suffix = _DETAILS_MARKER + path.read_text().split(_DETAILS_MARKER, 1)[1]
    heading = (
        f"DINGO ID{run.source_id} | {run.directory.name}\n"
        f"Status: {record['status']} (completion is not a convergence verdict)\n"
        f"Directory: {run.directory}\n"
        "Model: original inverse R+C rectification; intrinsic dispersion = 0 km/s.\n"
        "Velocity: sampled deprojected asymptotic V_inf; radius: stack pixels.\n"
        "Sampler: emcee ensemble with PyTorch likelihood; CUDA batches likelihood evaluations.\n"
        "Gaussian inclination sigma is the total prior width; fixed-i errors are conditional.\n"
        "Residual image: highest posterior saved draw; corner: pooled medians/intervals.\n"
        "Review all walkers and independent starts before interpreting pooled intervals.\n\n"
    )
    temporary = path.with_suffix(".txt.tmp")
    temporary.write_text(heading + json.dumps(record, indent=2, allow_nan=False) + "\n" + suffix)
    temporary.replace(path)


@contextmanager
def _locked(run: Run):
    """Keep the lock outside the deletable run so overwrite cannot bypass it."""
    _no_symlinks(run.directory)
    run.directory.parent.mkdir(parents=True, exist_ok=True)
    lock_path = run.directory.parent / f".{run.directory.name}.lock"
    _no_symlinks(lock_path)
    with lock_path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"This run is already active: {run.directory}") from exc
        try:
            # A fitting subprocess can outlive a crashed notebook. Its own lock
            # prevents another session from resuming/deleting active files.
            for lock in [*run.directory.rglob("worker.lock"), *run.directory.rglob(".fit.lock")]:
                _no_symlinks(lock)
                with lock.open("r+") as worker:
                    try:
                        fcntl.flock(worker, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError as exc:
                        raise RuntimeError(f"A fitting worker is still active: {lock}") from exc
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _check_inputs(record):
    for name, info in record["inputs"].items():
        path = Path(info["path"])
        if not path.is_file() or sha256(path) != info["sha256"]:
            raise ValueError(f"The recorded {name} input is missing or changed: {path}. Use a new run.")


def _input_mask_records(input_masks, input_paths):
    """Validate source-pixel DS9 regions and record their exact content identity.

    The returned masks describe pixels to exclude, not zero-valued emission.
    Validation happens before any run directory is created or overwritten.
    """
    if input_masks is None:
        return {}
    if not isinstance(input_masks, Mapping) or set(input_masks) - {"R", "C"}:
        raise ValueError("input_masks must map 'R' and/or 'C' to DS9 region paths.")
    from astropy.io import fits
    from .input_masks import read_ds9_exclusion

    records = {}
    for pupil, value in sorted(input_masks.items()):
        path = _path(value)
        with fits.open(input_paths[pupil]) as hdus:
            mask = read_ds9_exclusion(path, hdus[0].data.shape, header=hdus[0].header)
        if not mask.any():
            raise ValueError(f"The {pupil} region excludes no input pixels: {path}")
        if mask.all():
            raise ValueError(f"The {pupil} region excludes the entire input image: {path}")
        records[pupil] = dict(path=str(path), sha256=sha256(path))
    return records


def _check_stage(record, name, settings):
    """Allow only an increased retained-step target when resuming sampling."""
    previous = record["stages"].get(name)
    if previous is None or name == "export":
        return
    old, new = dict(previous["settings"]), dict(settings)
    if name == "sampling":
        old_draws, new_draws = old.pop("draws"), new.pop("draws")
        if new_draws < old_draws:
            raise ValueError("draws is a total target and cannot decrease for this run.")
    if old != new:
        raise ValueError(f"The {name} settings changed. Use a new run name or explicitly overwrite this run.")


@contextmanager
def _operation(run: Run, name: str, settings: dict, *, check_inputs=True):
    """Record failures/interruptions as well as successful stage completion."""
    with _locked(run):
        record = _record(run)
        if check_inputs:
            _check_inputs(record)
        _check_stage(record, name, settings)
        previous = record["stages"].get(name, {})
        stage = dict(settings=settings, status="running", started_utc=_utc(),
                     attempt=int(previous.get("attempt", 0)) + 1)
        record["stages"][name] = stage
        record["status"] = f"{name}: running"
        _write_record(run, record)
        started = time.monotonic()
        try:
            yield record, stage
        except BaseException as exc:
            stage["status"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, InterruptedError)) else "failed"
            stage["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        else:
            stage["status"] = "complete"
        finally:
            stage["finished_utc"] = _utc()
            stage["elapsed_seconds"] = time.monotonic() - started
            record["status"] = f"{name}: {stage['status']}"
            _write_record(run, record)


def prepare_source(
    source_id, *, working_dir, run_name=None, run_number=1,
    inclination_mode="imaging_prior", inclination_deg=None,
    inclination_sigma_deg=5.0, start_velocity_kms=None,
    start_inclination_deg=None, parent_search=None, data_dir=None,
    config_path=None, device="cuda", notes="", overwrite=False, input_masks=None,
) -> Run:
    """Create or reopen one identified experiment, without fitting anything.

    Parameters
    ----------
    source_id : int or str
        Catalog number, e.g. 15665 or 'ID15665'.
    working_dir : path
        Main output directory. Files go in ``working_dir/source_id/run_name``.
    run_name : str, optional
        Safe directory basename. By default it describes source, inclination
        mode, starting velocity and run_number. Use a new name for a new trial.
    inclination_mode : {'imaging_prior', 'gaussian_prior', 'fixed'}
        Imaging-derived Gaussian mean; user-supplied Gaussian mean; or exact i.
    inclination_deg : float, optional
        Prior mean for gaussian_prior, or exact angle for fixed. Leave unset
        for imaging_prior. This is separate from the starting inclination.
    inclination_sigma_deg : float, default 5.0
        TOTAL untruncated Gaussian 1-sigma width in degrees. Ignored when fixed.
    start_velocity_kms, start_inclination_deg : float, optional
        Manual V_inf/i initial values. Neither is a new prior. Without V_inf,
        fit_source requires an explicitly selected candidate. Other coordinates
        always come from a candidate search, which can be shared between trials.
    parent_search : path, optional
        Existing broad-search directory containing start_search.json. It must
        belong to this source, data and requested prior; it is never modified.
    data_dir, config_path : path, optional
        Root containing ID*/image.fits, R.fits, C.fits; source metadata YAML.
        Defaults to repository data and its source-specific YAML, respectively.
    device : {'cuda', 'cpu'}
        Likelihood execution device. CUDA requires the grism environment/GPU.
    input_masks : mapping, optional
        DS9 exclusion regions in original R/C input-pixel coordinates, for
        example ``{'C': '/path/to/C_mask.reg'}``. These exclusions apply to
        candidate search and MCMC; the original FITS files remain unchanged.
    notes : str
        Scientific rationale or observational notes, saved verbatim.
    overwrite : bool, default False
        Delete ONLY this workflow-identified run (including its checkpoints)
        and start fresh. Never deletes inputs, siblings or an external search.

    Returns
    -------
    Run
        An independent handle. Repeating identical settings reopens the run;
        incompatible settings raise before changing existing files.
    """
    import yaml

    sid = source_number(source_id)
    run_number = _integer("run_number", run_number)
    if inclination_mode not in {"imaging_prior", "gaussian_prior", "fixed"}:
        raise ValueError("inclination_mode must be imaging_prior, gaussian_prior or fixed.")
    sigma = _number("inclination_sigma_deg", inclination_sigma_deg)
    if inclination_mode == "imaging_prior":
        if inclination_deg is not None:
            raise ValueError("Use gaussian_prior to specify a prior mean, or leave inclination_deg unset.")
    else:
        inclination_deg = _angle("inclination_deg", inclination_deg)
    if start_velocity_kms is not None:
        start_velocity_kms = _number("start_velocity_kms", start_velocity_kms)
    if start_inclination_deg is not None:
        start_inclination_deg = _angle("start_inclination_deg", start_inclination_deg)
    if (inclination_mode == "fixed" and start_inclination_deg is not None
            and not math.isclose(start_inclination_deg, inclination_deg, abs_tol=1e-10)):
        raise ValueError("A fixed-inclination start must equal inclination_deg.")
    if device not in {"cpu", "cuda"} or not isinstance(overwrite, bool):
        raise ValueError("Use device='cpu' or 'cuda' and a boolean overwrite.")
    if not isinstance(notes, str):
        raise ValueError("notes must be text.")
    token = lambda value: f"{value:g}".replace(".", "p")
    if run_name is None:
        inc = (f"fixinc_i{token(inclination_deg)}" if inclination_mode == "fixed"
               else f"priorinc_s{token(sigma)}")
        start = "search" if start_velocity_kms is None else f"vstart{token(start_velocity_kms)}"
        run_name = f"{sid}_{inc}_{start}_run{run_number:02d}"
    if not isinstance(run_name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_name):
        raise ValueError("run_name must contain letters, digits, underscores or dashes only.")

    raw_directory = Path(working_dir).expanduser() / str(sid) / run_name
    _no_symlinks(raw_directory)
    run = Run(sid, raw_directory.resolve())
    data = _path(data_dir or REPO / "data")
    config = _path(config_path or data / f"ID{sid}" / "config_kinematics.yaml")
    source_config = yaml.safe_load(config.read_text())
    if source_number(source_config["summary"]["name"]) != sid:
        raise ValueError("Configuration belongs to a different source.")
    paths = {name: data / f"ID{sid}" / f"{name}.fits" for name in ("image", "R", "C")}
    paths["config"] = config
    mask_records = _input_mask_records(input_masks, paths)
    paths.update({f"mask_{pupil}": Path(info["path"]) for pupil, info in mask_records.items()})
    parent = _path(parent_search) if parent_search is not None else None
    for path in [*paths.values(), *([parent] if parent else [])]:
        if path.resolve().is_relative_to(run.directory):
            raise ValueError("Inputs and parent_search must be outside the run that overwrite may delete.")
    if parent is not None and run.directory.is_relative_to(parent):
        raise ValueError("The new run must be outside its parent search tree.")
    inputs = {name: dict(path=str(path.resolve()), sha256=sha256(path)) for name, path in paths.items()}
    configuration = dict(data_dir=str(data), config_path=str(config), device=device,
        inclination_mode=inclination_mode, inclination_deg=inclination_deg,
        inclination_sigma_deg=sigma, start_velocity_kms=start_velocity_kms,
        start_inclination_deg=start_inclination_deg,
        parent_search=str(parent) if parent else None)
    # Keep unmasked records byte-compatible with the original interface.
    if mask_records:
        configuration["input_masks"] = mask_records
    record = dict(schema=SCHEMA, source_id=sid, run_name=run_name,
                  configuration=configuration, inputs=inputs, created_utc=_utc(),
                  status="prepared", stages={}, notes=notes,
                  interface_sha256=sha256(Path(__file__)))
    if parent is not None:
        candidates = _validate_search(run, record, parent)
        record["parent_search"] = dict(directory=str(parent),
            sha256=sha256(parent / "start_search.json"),
            settings=candidates.report["search_settings"],
            inclination_prior=candidates.report["inclination_prior"],
            inclination_metadata=candidates.report.get("inclination_metadata"))
    with _locked(run):
        if run.directory.exists():
            previous = _record(run) if (run.directory / "run.json").is_file() else None
            if previous is None:
                raise ValueError("Refusing to replace a directory without this workflow's run.json.")
            if not overwrite:
                if previous["configuration"] != configuration or previous["inputs"] != inputs:
                    raise ValueError("This run has different settings/inputs. Use a new name or overwrite=True.")
                return run
            # Also protect against a surviving worker after its notebook died.
            with ExitStack() as stack:
                locks = list(run.directory.rglob("worker.lock")) + list(run.directory.rglob(".fit.lock"))
                for path in locks:
                    _no_symlinks(path)
                    stream = stack.enter_context(path.open("r+"))
                    try:
                        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError as exc:
                        raise RuntimeError(f"A worker is still active: {path}") from exc
                shutil.rmtree(run.directory)
        (run.directory / "_checkpoints").mkdir(parents=True)
        _write_record(run, record)
    print(f"ID{sid}: {run.directory}", flush=True)
    return run


def with_input_masks(run: Run, *, C=None, R=None, run_name=None) -> Run:
    """Create/reopen a masked experiment while preserving the original run.

    ``C`` and ``R`` are DS9 files drawn on the original grism cutouts. Existing
    masks for unspecified grisms are retained. The default new directory name
    includes a mask-identity suffix; rerunning the same cell reuses that run.
    A changed region creates another sibling experiment instead of reusing an
    unmasked search or deleting results. No search or sampling is started.

    Assign the returned handle, e.g. ``run = with_input_masks(run, C=path)``.
    Subsequent ``find_candidates`` and ``fit_source`` calls use its exclusions.
    Any attached imaging prior is retained because the imaging data did not
    change. External candidate searches are not inherited by the new run.
    """
    if C is None and R is None:
        raise ValueError("Provide a C and/or R DS9 region file.")
    record = _record(run)
    # A revised region is allowed here, but changed science/config inputs are
    # not. Existing mask hashes are checked only if reusing that exact run.
    base_inputs = {key: value for key, value in record["inputs"].items()
                   if not key.startswith("mask_")}
    _check_inputs({"inputs": base_inputs})
    choice = record["configuration"]
    paths = {pupil: info["path"] for pupil, info in choice.get("input_masks", {}).items()}
    paths.update({pupil: value for pupil, value in (("R", R), ("C", C)) if value is not None})
    masks = _input_mask_records(paths, {pupil: base_inputs[pupil]["path"] for pupil in ("R", "C")})
    if masks == choice.get("input_masks") and (run_name is None or run_name == run.directory.name):
        _check_inputs(record)
        return run

    origin = record.get("input_mask_origin", dict(directory=str(run.directory), run_name=run.directory.name))
    if run_name is None:
        token = hashlib.sha256(json.dumps(masks, sort_keys=True).encode()).hexdigest()[:10]
        run_name = f"{origin['run_name']}_masked_{token}"
    if run_name == run.directory.name:
        raise ValueError("Use a different run_name for changed masks; existing results are preserved.")
    new_run = prepare_source(
        run.source_id, working_dir=run.directory.parent.parent, run_name=run_name,
        inclination_mode=choice["inclination_mode"], inclination_deg=choice["inclination_deg"],
        inclination_sigma_deg=choice["inclination_sigma_deg"],
        start_velocity_kms=choice["start_velocity_kms"],
        start_inclination_deg=choice["start_inclination_deg"],
        data_dir=choice["data_dir"], config_path=choice["config_path"], device=choice["device"],
        input_masks=paths, notes=record.get("notes", ""),
    )
    with _locked(new_run):
        new_record = _record(new_run)
        new_record.setdefault("input_mask_origin", origin)
        attached = record.get("imaging")
        if attached is not None and "imaging" not in new_record:
            new_record["imaging"] = attached
        _write_record(new_run, new_record)
    print("Input masks apply to candidate search and MCMC.", flush=True)
    print(f"Masked run: {new_run.directory}", flush=True)
    return new_run


def load_run(directory) -> Run:
    """Read an existing run handle; do not launch jobs or require raw inputs."""
    directory = _path(directory)
    record = read_json(directory / "run.json")
    run = Run(source_number(record["source_id"]), directory)
    _record(run)
    return run


def _validate_search(run, record, directory):
    """Check source, amplitude, data identity and actual prior before reuse.

    The worker additionally checks code, calibration and every saved array hash
    before fitting. This lighter check does not initialize PyTorch on import.
    """
    from .three_file import CandidateResults

    candidates = CandidateResults(directory, run.source_id)
    report = candidates.report
    pinned = record.get("parent_search")
    if pinned is not None and sha256(directory / "start_search.json") != pinned["sha256"]:
        raise ValueError("The external search changed since this run was prepared. Use a new run.")
    if report.get("search_settings", {}).get("method") != "broad" or candidates.velocity_parameter != "V_rot":
        raise ValueError("Choose a broad search using deprojected V_rot and inclination.")
    identity_path = directory / "search_identity.json"
    if not identity_path.is_file() or sha256(identity_path) != report.get("search_identity_sha256"):
        raise ValueError("The parent search identity is missing or changed.")
    identity = read_json(identity_path)
    if identity.get("source_id") != run.source_id:
        raise ValueError("The parent search identity belongs to a different source.")
    for name in ("image", "R", "C"):
        if identity.get("inputs", {}).get(name) != record["inputs"][name]:
            raise ValueError(f"The parent search used different {name} inputs.")
    for pupil in ("R", "C"):
        name = f"mask_{pupil}"
        if identity.get("inputs", {}).get(name) != record["inputs"].get(name):
            raise ValueError(f"The parent search used a different {pupil} input mask.")
    import yaml
    source_config = yaml.safe_load(Path(record["configuration"]["config_path"]).read_text())
    wanted = dict(source_config["summary"])
    # The existing worker copies the source metadata and normalizes these three
    # fields; compare that normalized summary, not the unmodified input YAML.
    if "r_fit" in wanted:
        r_fit = int(wanted["r_fit"])
    else:
        from astropy.io import fits
        r_fit = (fits.getdata(record["inputs"]["R"]["path"]).shape[0] - 1) // 2
    wanted.update(mode="kinematics", unit="pix", r_fit=r_fit)
    if identity.get("config", {}).get("summary") != wanted:
        raise ValueError("The parent search used different source metadata.")
    for pupil in ("R", "C"):
        if identity.get("config", {}).get("image", {}).get(pupil, {}).get("module") != source_config["image"][pupil]["module"]:
            raise ValueError(f"The parent search used a different grism {pupil} module.")
    settings = candidates.inclination_settings()
    choice = record["configuration"]
    if choice["inclination_mode"] != "fixed":
        if settings["inclination_deg"] is None or not math.isclose(
                settings["inclination_sigma_deg"], choice["inclination_sigma_deg"], abs_tol=1e-10):
            raise ValueError("The saved inclination prior width differs from this run. Make a new search.")
        if choice["inclination_mode"] == "gaussian_prior":
            if not math.isclose(settings["inclination_deg"], choice["inclination_deg"], abs_tol=1e-10):
                raise ValueError("The saved prior mean differs from the requested Gaussian mean.")
        else:
            metadata = settings.get("inclination_metadata") or {}
            if not metadata.get("kind", "").startswith("imaging_") or metadata.get("source_id") != run.source_id:
                raise ValueError("An imaging_prior run requires recorded imaging-derived prior provenance.")
            attached = record.get("imaging", {}).get("prior")
            if attached is not None and metadata != attached:
                raise ValueError("This search used a different imaging prior.")
    return candidates


def use_imaging(run: Run, imaging_directory):
    """Attach accepted source-specific PySersic results without refitting them.

    In imaging_prior mode only the inclination prior is supplied to the grism
    fit. For fixed/gaussian_prior modes imaging provides context and does not
    change the specified angle/prior. Gas emission shape and center are not
    forced to follow the stars. New morphology fits use one full-image Sérsic
    model. Legacy full/core pairs remain readable as alternative fits to the
    same image; see inclination_prior.
    """
    from .rotation_imaging import ImagingResults
    from .three_file_imaging import inclination_prior

    directory = _path(imaging_directory)
    result = ImagingResults(directory, run.source_id)
    prior = inclination_prior(result, total_sigma_deg=run.configuration["inclination_sigma_deg"])
    settings = dict(directory=str(directory), prior=prior)
    with _operation(run, "attach_imaging", settings) as (record, stage):
        if "search" in record["stages"] or "sampling" in record["stages"]:
            if record.get("imaging") != settings:
                raise ValueError("Attach imaging before searching/sampling; use a new run to change its prior.")
        record["imaging"] = settings
    return result


def fit_morphology(run: Run, *, refit=False, display=True, warmup=1000,
                   draws=2000, seed=735, fit_radius=2.0, cutout_size=181,
                   chunk=50, science=None, error=None, psf=None, bundle=None):
    """Fit or reuse this source's shared, single-Sérsic F150W posterior.

    Unlike ``fit_imaging``, which retains the legacy pair of per-run fits, this
    entry point stores imaging under ``<source>/morph_fits/F150W/fit_001``.
    Different kinematic experiments can reuse it without resampling PySersic.
    Each imaging fit has posterior samples, native plots, ``parameters.txt``
    and a readable ``pysersic_<ID>_info.cat`` with settings and timestamps.

    ``refit=True`` creates another immutable imaging version. An experiment
    that already attached a prior stays pinned to its recorded imaging version;
    to change that prior, first prepare a new kinematic run or explicitly
    overwrite that kinematic run. Kinematic overwrite never deletes morphology.

    Parameters are imaging sampler targets per chain, independent of the
    kinematic sampler. Optional science/error/PSF paths select calibrated F150W
    inputs; the three-file ``image.fits`` is not substituted for a calibrated
    imaging mosaic. With ``display=True`` the fitting image appears before
    sampling, followed by residuals, native corner and parameter tables.

    Returns an ``ImagingResults`` object after attaching its inclination prior.
    Incomplete or diagnostically unsuccessful imaging is saved for inspection
    but cannot silently become the inclination prior of the grism fit.
    """
    from .morphology import fit_morphology as fit_shared_morphology

    if not isinstance(refit, bool) or not isinstance(display, bool):
        raise ValueError("refit and display must be booleans.")
    record = _record(run)
    _check_inputs(record)
    attached = record.get("imaging")
    if attached is not None:
        if refit:
            raise ValueError(
                "This kinematic run already records an imaging prior. Choose a new "
                "run_number/run_name or overwrite this kinematic run before refitting "
                "morphology. Existing morphology versions will be preserved."
            )
        result = load_imaging(run)
        result.require_ready()
        print(f"ID{run.source_id}: using this run's recorded morphology: {result.path}", flush=True)
        if display:
            show_imaging(run)
        return result

    directory = run.directory.parent / "morph_fits" / "F150W"
    result = fit_shared_morphology(
        run.source_id, directory=directory,
        config_path=record["configuration"]["config_path"],
        device=record["configuration"]["device"], refit=refit, display=display,
        warmup=warmup, draws=draws, seed=seed, fit_radius=fit_radius,
        cutout_size=cutout_size, chunk=chunk,
        science=science, error=error, psf=psf, bundle=bundle,
    )
    result.require_ready()
    return use_imaging(run, result.path)


def fit_imaging(run: Run, *, warmup=1000, draws=2000, seed=735,
                fit_radius=2.0, cutout_size=181, chunk=50,
                science=None, error=None, psf=None, bundle=None):
    """Fit calibrated F150W imaging with the existing PySersic worker.

    Uses the full and core-masked fits, with native progress/checkpoints. This
    function performs no grism search. Raw imaging paths are optional when the
    survey bundle can supply them. An accepted fit becomes the imaging prior;
    unsuccessful fits remain available via load_imaging/show_imaging. Use a new
    run (or overwrite the whole identified run) to change imaging settings.
    """
    from .rotation_imaging import prepare_imaging
    from .three_file_imaging import inclination_prior

    settings = dict(warmup=_integer("warmup", warmup), draws=_integer("draws", draws),
        seed=_integer("seed", seed, 0), chunk=_integer("chunk", chunk),
        fit_radius=_number("fit_radius", fit_radius), cutout_size=_integer("cutout_size", cutout_size))
    for name, value in dict(science=science, error=error, psf=psf, bundle=bundle).items():
        settings[name] = str(_path(value)) if value is not None else None
    directory = run.directory / "_checkpoints" / "imaging"
    with _operation(run, "imaging", settings) as (record, stage):
        prior_in_use = "search" in record["stages"] or "sampling" in record["stages"]
        attached = record.get("imaging")
        if prior_in_use and (attached is None or Path(attached["directory"]) != directory):
            raise ValueError("Fit imaging before the search. To replace attached imaging, prepare a new run.")
        result = prepare_imaging(run.source_id, run=True, override=False,
            results=run.directory / "_checkpoints", imaging=directory,
            config=record["configuration"]["config_path"], device=record["configuration"]["device"],
            warmup=warmup, samples=draws, seed=seed, chunk=chunk,
            fit_radius=fit_radius, cutout_size=cutout_size,
            science=science, error=error, psf=psf, bundle=bundle)
        result.require_ready()
        prior = inclination_prior(result, total_sigma_deg=record["configuration"]["inclination_sigma_deg"])
        if prior_in_use and prior != attached["prior"]:
            raise ValueError("The imaging prior already used by this run changed. Prepare a new run.")
        record["imaging"] = dict(directory=str(result.path), prior=prior)
    return result


def load_imaging(run: Run):
    """Read attached or local PySersic output, including an incomplete fit."""
    from .rotation_imaging import ImagingResults
    directory = _record(run).get("imaging", {}).get("directory")
    return ImagingResults(directory or run.directory / "_checkpoints" / "imaging", run.source_id)


def _display(items):
    """Display once in a notebook while returning figures for further use."""
    from IPython.display import display
    import matplotlib.pyplot as plt
    for item in items.values():
        display(item)
        if isinstance(item, plt.Figure):
            plt.close(item)


def show_inputs(run: Run, *, display=True) -> dict:
    """Show the three observed cutouts recorded for this source, without fitting.

    Each panel uses its own asinh stretch so the input structures are visible.
    Axes are zero-based cutout pixels; no model or residual is shown here.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import AsinhNorm
    from astropy.io import fits

    record = _record(run)
    _check_inputs(record)
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for axis, name in zip(axes, ("image", "R", "C")):
        data = np.asarray(fits.getdata(record["inputs"][name]["path"]), dtype=float)
        if data.ndim != 2 or not np.isfinite(data).any():
            plt.close(figure)
            raise ValueError(f"{name}.fits must contain a 2-D image with finite pixels.")
        low, high = np.quantile(data[np.isfinite(data)], [.02, .995])
        span = max(float(high-low), 1e-12)
        norm = AsinhNorm(linear_width=span/20, vmin=low, vmax=low+span)
        axis.imshow(data, origin="lower", norm=norm,
                    cmap=plt.get_cmap("magma").with_extremes(bad="white"))
        axis.set(title=f"ID{run.source_id}: {name}.fits",
                 xlabel="Cutout x [pixel]", ylabel="Cutout y [pixel]")
    views = {"inputs": figure}
    if display:
        _display(views)
    return views


def show_input_masks(run: Run, *, display=True) -> dict:
    """Display the saved input exclusions with red/orange grism heatmaps.

    Axes refer to original, zero-based cutout pixels. Masked pixels appear as
    missing measurements, not measured zeros. This is a preview of the input
    regions; the fitter additionally accounts for their remapped footprint.
    """
    from astropy.io import fits
    from .input_masks import read_ds9_exclusion, plot_input_masks

    record = _record(run)
    _check_inputs(record)
    masks = {}
    for pupil, info in record["configuration"].get("input_masks", {}).items():
        with fits.open(record["inputs"][pupil]["path"]) as hdus:
            masks[pupil] = read_ds9_exclusion(info["path"], hdus[0].data.shape, header=hdus[0].header)
        print(f"{pupil}: excluding {int(masks[pupil].sum())} of {masks[pupil].size} input pixels; {info['path']}")
    source_dir = Path(record["inputs"]["C"]["path"]).parent
    views = {"input_masks": plot_input_masks(source_dir, masks)}
    if display:
        _display(views)
    return views


def show_imaging(run: Run, *, variant="full_pixel", display=True) -> dict:
    """Show fitting image, fit, native PySersic corner and q0=0/0.2 inclinations.

    Read-only; call before candidate search to inspect the imaging assumption.
    Returned figure/table objects remain usable even when displayed and closed.
    """
    result = load_imaging(run)
    views = dict(image=result.input_figure(variant=variant), summary=result.summary())
    if (result.path / variant / "samples.npz").exists():
        views.update(fit=result.figure(), parameters=result.parameter_summary(variant=variant),
            inclination=result.inclination_summary(variant=variant, q0_values=(0., .2)),
            corner=result.corner(variant=variant, save=False))
    if display:
        _display(views)
    return views


def _search_prior(record):
    choice = record["configuration"]
    if choice["inclination_mode"] == "fixed":
        # The existing broad-search worker varies i. Exact fixed-i evaluation
        # happens in prepare_start/fit_source, not in these exploratory rows.
        return None
    if choice["inclination_mode"] == "gaussian_prior":
        return dict(inclination_deg=choice["inclination_deg"],
                    inclination_sigma_deg=choice["inclination_sigma_deg"])
    if "imaging" not in record:
        raise ValueError("Run fit_imaging or use_imaging before finding candidates for imaging_prior.")
    return record["imaging"]["prior"]


def find_candidates(run: Run, *, velocity_step=25., n_angles=16, n_candidates=4,
                    refine_iterations=100, profile_iterations=80, profile_step=25.,
                    profile_fine_step=1., profile_max=1000., max_velocity=1000.,
                    seed=42, pixel_scale_arcsec=.063):
    """Run/reopen a broad velocity/geometry search; do not start MCMC.

    ``max_velocity`` caps V_inf in both the search and its subsequent MCMC
    (default 1000 km/s). ``velocity_step`` is the coarse V_inf grid spacing.
    ``profile_fine_step``
    refines selected intervals of the finite-radius velocity profile, not every
    possible V_inf at 1 km/s. A finite search cannot guarantee all solutions.
    All saved candidates are available for inspection; fit_source requires an
    explicit candidate unless a manual velocity was supplied.

    For fixed-i runs, this initial broad search still varies i with the worker's
    isotropic prior. Its rows are exploratory starts, NOT fixed-i fits. Use
    prepare_start to see the exact fixed-i residual before sampling.

    An attached parent_search is read-only: its original search settings are
    reused, and these optional grid arguments do not rerun or alter it.
    """
    record = _record(run)
    if record["configuration"]["parent_search"]:
        print("Reusing the external search with its recorded grid settings.", flush=True)
        return _validate_search(run, record, run.search_directory)
    prior = _search_prior(record)
    settings = dict(velocity_step=_number("velocity_step", velocity_step),
        n_angles=_integer("n_angles", n_angles), n_candidates=_integer("n_candidates", n_candidates),
        refine_iterations=_integer("refine_iterations", refine_iterations),
        profile_iterations=_integer("profile_iterations", profile_iterations),
        profile_step=_number("profile_step", profile_step), profile_fine_step=_number("profile_fine_step", profile_fine_step),
        profile_max=_number("profile_max", profile_max), max_velocity=_number("max_velocity", max_velocity),
        seed=_integer("seed", seed, 0), pixel_scale_arcsec=_number("pixel_scale_arcsec", pixel_scale_arcsec),
        prior=prior)
    if profile_max > max_velocity:
        raise ValueError("profile_max must not exceed max_velocity.")
    with _operation(run, "search", settings) as (record, stage):
        output = run.search_directory
        _no_symlinks(output)
        output.mkdir(parents=True, exist_ok=True)
        choice = record["configuration"]
        arguments = ["-m", "dingo.three_file_search", "--source", str(run.source_id),
            "--data-dir", choice["data_dir"], "--config", choice["config_path"],
            "--output", str(output), "--device", choice["device"],
            "--velocity-parameter", "V_rot", "--search-only"]
        flags = dict(velocity_step="amp-step", n_angles="n-theta", n_candidates="n-keep",
            refine_iterations="refine-iterations", profile_iterations="profile-iterations",
            profile_step="profile-step", profile_fine_step="profile-fine-step", profile_max="profile-max",
            max_velocity="max-velocity", seed="seed", pixel_scale_arcsec="pixel-scale")
        for name, flag in flags.items():
            arguments += [f"--{flag}", str(settings[name])]
        for pupil, info in choice.get("input_masks", {}).items():
            arguments += [f"--mask-{pupil.lower()}", info["path"]]
        if prior:
            arguments += ["--inclination-deg", str(prior["inclination_deg"]),
                          "--inclination-sigma-deg", str(prior["inclination_sigma_deg"])]
            if choice["inclination_mode"] == "imaging_prior":
                arguments += ["--inclination-metadata-json", json.dumps(prior, allow_nan=False)]
        else:
            print("Exploratory search varies i; the subsequent fixed-i run will evaluate your exact angle.", flush=True)
        print(f"Candidate-search progress: {output / 'run.log'}", flush=True)
        run_command(arguments, output / "run.log", device=choice["device"])
        result = _validate_search(run, record, output)
        stage["search_sha256"] = sha256(output / "start_search.json")
    return result


def load_candidates(run: Run):
    """Read saved candidate images without launching or modifying a search."""
    from .three_file import CandidateResults
    return CandidateResults(run.search_directory, run.source_id)


def show_candidates(run: Run, *, display=True) -> dict:
    """Show ALL candidate residuals and the matching table with original IDs."""
    candidates = load_candidates(run)
    views = dict(table=candidates.table(), residuals=candidates.figure())
    if display:
        _display(views)
    return views


def _selection(run, candidate_id):
    record = _record(run)
    candidates = _validate_search(run, record, run.search_directory)
    choice = record["configuration"]
    if candidate_id is None:
        previous = record.get("selection")
        if previous:
            candidate_id = previous["candidate_id"]
        elif choice["start_velocity_kms"] is None:
            raise ValueError("Inspect show_candidates, then supply candidate_id explicitly.")
        else:
            eligible = [row for row in candidates.report["trials"]
                        if row.get("log_posterior") is not None and row.get("loss") is not None
                        and math.isfinite(row["log_posterior"]) and math.isfinite(row["loss"])]
            if not eligible:
                raise ValueError("No finite parent candidates are available.")
            candidate_id = max(eligible, key=lambda row: row["log_posterior"])["candidate_id"]
    candidate_id = _integer("candidate_id", candidate_id, 0)
    if candidate_id not in candidates.candidate_ids:
        raise ValueError(f"Candidate {candidate_id} is absent. Use one of {candidates.candidate_ids}.")
    params = candidates.report["trials"][candidate_id]["params"]
    velocity = choice["start_velocity_kms"] or params["V_rot"]
    inclination = (choice["inclination_deg"] if choice["inclination_mode"] == "fixed"
                   else choice["start_inclination_deg"] or math.degrees(params["inc_v"]))
    selection = dict(candidate_id=candidate_id, start_velocity_kms=velocity,
        start_inclination_deg=inclination, parent_search=str(run.search_directory),
        parent_search_sha256=sha256(run.search_directory / "start_search.json"))
    if record.get("selection") is not None and record["selection"] != selection:
        raise ValueError("This run already has a different start. Prepare another named run.")
    return selection


def _fit_arguments(run, selection, warmup, draws, walkers, seed):
    choice = run.configuration
    worker = "dingo.three_file_fixed_i_fit" if choice["inclination_mode"] == "fixed" else "dingo.three_file_manual_fit"
    return ["-m", worker, "--parent-run", str(run.search_directory),
        "--output", str(run.posterior_directory), "--base-candidate", str(selection["candidate_id"]),
        "--start-velocity", str(selection["start_velocity_kms"]),
        "--start-inclination", str(selection["start_inclination_deg"]),
        "--warmup", str(warmup), "--draws", str(draws), "--walkers", str(walkers),
        "--seed", str(seed), "--device", choice["device"]]


def prepare_start(run: Run, *, candidate_id=None):
    """Render the EXACT selected/manual start, without optimizing or sampling.

    Returns CandidateResults for the initialized run (one row, ID 0). Its parent
    candidate ID is separately recorded in run.json. The choice is frozen for
    this experiment; use another run to compare another start.
    """
    from .three_file import CandidateResults
    selection = _selection(run, candidate_id)
    with _operation(run, "start", selection) as (record, stage):
        record["selection"] = selection
        _write_record(run, record)
        _no_symlinks(run.posterior_directory)
        run.posterior_directory.mkdir(parents=True, exist_ok=True)
        args = _fit_arguments(run, selection, 1000, 8000, 64, 42) + ["--prepare-only"]
        run_command(args, run.posterior_directory / "run.log", device=run.configuration["device"])
        result = CandidateResults(run.posterior_directory, run.source_id)
    return result


def fit_source(run: Run, *, candidate_id=None, warmup=1000, draws=8000,
               walkers=64, seed=42, notes=""):
    """Run/resume one selected start with a progress bar and persistent checkpoints.

    ``draws`` is the total retained target PER WALKER, excluding warmup. Calling
    again with the same settings reuses completed output; increasing only draws
    extends the run. Changing warmup, walkers, seed, start or priors requires a
    new named run. Checkpoints/logs remain under this run's _checkpoints folder.

    A manual velocity needs a parent search for other coordinates. If no manual
    velocity is given, candidate_id is required on the first call. An initial
    V_inf or i does not constrain the posterior around that value. This method
    neither removes wandering walkers nor declares a paper-ready measurement.
    Use save_results separately after inspection to publish final files/plots.
    """
    settings = dict(warmup=_integer("warmup", warmup, 0), draws=_integer("draws", draws),
                    walkers=_integer("walkers", walkers, 20), seed=_integer("seed", seed, 0))
    if not isinstance(notes, str):
        raise ValueError("notes must be text.")
    selection = _selection(run, candidate_id)
    settings["selection"] = selection
    with _operation(run, "sampling", settings) as (record, stage):
        record["selection"] = selection
        if notes:
            record.setdefault("sampling_notes", []).append(dict(utc=_utc(), text=notes))
        _write_record(run, record)
        _no_symlinks(run.posterior_directory)
        run.posterior_directory.mkdir(parents=True, exist_ok=True)
        args = _fit_arguments(run, selection, warmup, draws, walkers, seed)
        print(f"ID{run.source_id}: V_inf start {selection['start_velocity_kms']:g} km/s; "
              f"i start {selection['start_inclination_deg']:g} degrees.\n"
              f"{walkers} walkers x {warmup:,} warmup + {draws:,} retained steps.\n"
              f"Progress/checkpoints: {run.posterior_directory}", flush=True)
        run_command(args, run.posterior_directory / "run.log", device=run.configuration["device"])
        result = load_results(run)
        stage["completed_draws"] = result.report["draws"]
        stage["sampling_pass"] = result.report.get("sampling_pass", False)
        stage["samples_sha256"] = result.report.get("samples_sha256")
        stage["inclination_prior"] = result.report.get("inclination_prior")
        stage["param_names"] = result.report.get("param_names")
        stage["device"] = result.report.get("device")
        if result.report.get("status") != "complete" or result.report["draws"] < draws:
            raise InterruptedError("Worker stopped before the requested retained-step target; resume this run.")
    return result


def load_results(run: Run):
    """Read saved posterior output; never start work or silently use older exports.

    Prefer checkpoint output if present. A completed exported bundle can be read
    without raw observations once checkpoint products have been archived away.
    After a failed extension, available draws may be from the earlier target;
    the run status and report's actual draw count remain explicit.
    """
    from .three_file import ThreeFileResults
    if (run.posterior_directory / "report.json").exists():
        result = ThreeFileResults(run.posterior_directory, run.source_id)
    elif run.posterior_directory.exists() and any(run.posterior_directory.iterdir()):
        raise FileNotFoundError(f"No readable posterior report yet; check {run.posterior_directory / 'run.log'}")
    else:
        result = ThreeFileResults(run.directory, run.source_id, final_bundle=True)
    record = _record(run)
    choice = record["configuration"]
    if result.velocity_parameter != "V_rot":
        raise ValueError("This workflow requires deprojected V_rot samples, not V_sini.")
    if choice["inclination_mode"] == "fixed":
        if result.fixed_inclination_deg is None or not math.isclose(
                result.fixed_inclination_deg, choice["inclination_deg"], abs_tol=1e-5):
            raise ValueError("The saved posterior has a different fixed inclination.")
    else:
        prior = result.report.get("inclination_prior", {})
        if result.fixed_inclination_deg is not None or prior.get("type") != "gaussian":
            raise ValueError("The saved posterior must sample inclination with the requested Gaussian prior.")
        if not math.isclose(math.degrees(prior["sigma"]), choice["inclination_sigma_deg"], abs_tol=1e-8):
            raise ValueError("The saved posterior has a different inclination prior width.")
        mean = (choice["inclination_deg"] if choice["inclination_mode"] == "gaussian_prior"
                else record.get("imaging", {}).get("prior", {}).get("inclination_deg"))
        if mean is None and record.get("parent_search"):
            mean = math.degrees(record["parent_search"]["inclination_prior"]["mu"])
        if mean is not None and not math.isclose(math.degrees(prior["mu"]), mean, abs_tol=1e-8):
            raise ValueError("The saved posterior has a different inclination prior mean.")
    selection = record.get("selection")
    if selection:
        expected = {"base_candidate_id": selection["candidate_id"],
                    "parent_run": selection["parent_search"],
                    "start_velocity_kms": selection["start_velocity_kms"],
                    "start_inclination_deg": selection["start_inclination_deg"]}
        if any(result.report.get(key) != value for key, value in expected.items()):
            raise ValueError("The saved posterior does not match this run's selected start.")
    return result


def show_results(run: Run, *, comparison_radius_pc=None, display=True) -> dict:
    """Read/display residuals, pooled monochrome corner, diagnostics and ALL walkers.

    The residual map uses an actual maximum-posterior saved draw. The corner and
    velocity table summarize pooled draws, which can differ in a multimodal run.
    No walker is silently discarded. Pixel centers/turnover keep the display
    close to the original notebook. Returns figures/tables for separate cells.
    """
    from ._workflow_products import all_walker_traces
    result = load_results(run)
    parameters = ["vrot", "rturn_pixels", "inclination", "theta", "x0", "y0"]
    views = dict(status=f"Workflow: {_record(run)['status']}\n{result.status}\n"
                 "Residuals: maximum-posterior draw. Table/corner: pooled median and 16th–84th percentiles.",
                 velocity=result.velocity_summary(radius_pc=comparison_radius_pc),
                 diagnostics=result.diagnostics(), residuals=result.fit_figure(radius_pc=comparison_radius_pc),
                 traces=all_walker_traces(result, parameters), corner=result.corner(parameters, save=False))
    if display:
        _display(views)
    return views


def save_results(run: Run, *, comparison_radius_pc=None, radii_pc=None) -> Path:
    """Export final results.npz, run_info.txt and plots into this run directory.

    Checkpoints stay in _checkpoints. A manifest protects unrelated files and
    removes stale *previously owned* plots on re-export. Supply explicit radii_pc
    to export a rotation-supported mass plot/profile. These radii are not
    automatically certified as observed or constrained; choose them from data
    coverage. Rotation, inclination and turnover correlations are propagated
    using paired draws. No convergence or mode-selection guarantee is implied.
    """
    from ._workflow_products import export_run_products
    # Export choices may change without altering posterior identity. Keep the
    # latest export settings rather than treating plot radius as a fitting prior.
    settings = dict(comparison_radius_pc=comparison_radius_pc,
                    radii_pc=None if radii_pc is None else [float(r) for r in radii_pc])
    with _operation(run, "export", settings, check_inputs=False) as (record, stage):
        result = load_results(run)
        if result.report.get("status") != "complete":
            raise ValueError("Only completed posterior output can be exported; inspect or resume this run.")
        target = record["stages"].get("sampling", {}).get("settings", {}).get("draws", 0)
        if result.report["draws"] < target:
            raise ValueError("Available draws predate the requested extension. Resume before exporting.")
        stage["samples_sha256"] = result.report.get("samples_sha256")
        # The NPZ holds a timestamped snapshot taken during export; run.json and
        # run_info.txt are updated with actual export completion after publish.
        snapshot = dict(record, metadata_snapshot_utc=_utc(),
                        metadata_note="Snapshot during export. run.json/run_info.txt record export completion.")
        path = export_run_products(result, run.directory, snapshot,
            comparison_radius_pc=comparison_radius_pc, radii_pc=radii_pc)
    print(f"Saved final files: {run.directory}", flush=True)
    return path


def fit_batch(runs: Mapping[str, Run], *, candidate_ids=None, **fit_kwargs) -> dict[str, BatchResult]:
    """Fit named run handles sequentially, keeping failures separate by source.

    ``candidate_ids`` maps the same labels to explicit choices; manual runs can
    omit it. Shared keyword arguments are passed to fit_source. Ordinary errors
    are returned as BatchResult.error and later jobs proceed. KeyboardInterrupt
    propagates immediately, stopping the batch. Nothing runs in the background.
    Saving/reviewing stays explicit; this function does not select good sources.
    """
    if not isinstance(runs, Mapping) or not all(isinstance(run, Run) for run in runs.values()):
        raise ValueError("runs must map your labels to Run handles.")
    candidate_ids = {} if candidate_ids is None else candidate_ids
    if not isinstance(candidate_ids, Mapping) or set(candidate_ids) - set(runs):
        raise ValueError("candidate_ids must map existing run labels to candidate IDs.")
    if "candidate_id" in fit_kwargs:
        raise ValueError("Use candidate_ids to choose a candidate separately for each run.")
    outcomes = {}
    for label, run in runs.items():
        print(f"\nBatch: {label} (ID{run.source_id})", flush=True)
        try:
            result = fit_source(run, candidate_id=candidate_ids.get(label), **fit_kwargs)
        except Exception as exc:
            outcomes[label] = BatchResult(run, error=exc)
            print(f"{label} failed: {type(exc).__name__}: {exc}", flush=True)
        else:
            outcomes[label] = BatchResult(run, result=result)
    return outcomes
