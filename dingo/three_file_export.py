"""Publish final arrays, plots and a readable run summary outside the run cache."""
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from zipfile import BadZipFile
from zoneinfo import ZoneInfo

import numpy as np

from .real_workflow import REPO, sha256


DEFAULT_WORKING = REPO.parent / "working"
SCHEMA = "dingo-three-file-final-v1"


def _samples_deprojected_velocity(report):
    """Identify the stored chain's amplitude without relabeling older runs."""
    return "velocity.V_rot" in report.get("param_names", [])


def _fixed_inclination_deg(report):
    value = report.get("fixed_params", {}).get("velocity.inc_v")
    if "velocity.inc_v" not in report.get("param_names", []) and value is not None:
        return float(np.rad2deg(value))
    return None


def read_metadata(path):
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata_json"].item()))
    if metadata.get("schema") != SCHEMA:
        raise ValueError("This file is not a DINGO final-results bundle.")
    return metadata


def _check_destination(destination, source_id):
    """Only replace a source folder previously published by this exporter."""
    if destination.is_symlink():
        raise ValueError("The final source directory must not be a symlink.")
    if not destination.exists():
        return
    if not destination.is_dir() or not (destination / "results.npz").is_file():
        raise ValueError(f"Choose an empty working location; unrelated files exist in {destination}.")
    metadata = read_metadata(destination / "results.npz")
    if metadata["report"]["source_id"] != source_id:
        raise ValueError("The final source folder belongs to a different source.")
    owned = set(metadata["export"]["files"])
    if any(p.is_symlink() or not p.is_file() or p.name not in owned
           for p in destination.iterdir()):
        raise ValueError(f"Unrelated files in {destination}; move them before updating final results.")


def _imaging_products(result):
    """Include only intact optional imaging, without changing the saved prior.

    Imaging files may have been replaced since the kinematic fit completed.
    Their absence or failed validation must not prevent exporting that fit,
    and changed imaging must never be presented as the fit's original prior.
    """
    record = result.report.get("inclination_metadata") or {}
    fixed = record.get("kind") == "user_fixed_inclination"
    inherited = fixed or record.get("kind") == "user_gaussian_inclination_prior"
    while record.get("kind") in ("user_gaussian_inclination_prior", "user_fixed_inclination"):
        record = record.get("inherited_imaging_metadata") or {}
    sources = record.get("sources", [])
    if not sources:
        return None, {}, {"status": "no imaging posterior recorded for this run"}
    from .rotation_imaging import ImagingResults
    checks = []

    def omitted(status, detail):
        return None, {}, {"status": status, "detail": detail,
                          "recorded_sources": sources, "file_checks": checks,
                          "note": "All imaging arrays and plots omitted; the saved kinematic "
                                  "posterior and inclination prior are unchanged."}

    def check_file(path, expected, *, kind, variant=None):
        check = dict(path=str(path), kind=kind, variant=variant,
                     expected_sha256=expected, actual_sha256=None)
        checks.append(check)
        try:
            check["actual_sha256"] = sha256(path)
        except OSError:
            check["status"] = "unavailable"
            raise
        check["status"] = "unchanged" if check["actual_sha256"] == expected else "changed"
        if check["status"] == "changed":
            raise ValueError(f"Imaging {kind} changed: {path}; expected SHA256 {expected}, "
                             f"found {check['actual_sha256']}.")

    try:
        roots = {Path(item["path"]).parent.parent.resolve() for item in sources}
        if len(roots) != 1:
            raise ValueError("Recorded imaging variants do not describe one source's imaging fit.")
        root = roots.pop()
        for item in sources:
            expected_path = root / item["variant"] / "samples.npz"
            if Path(item["path"]).resolve() != expected_path.resolve():
                raise ValueError("Recorded imaging sample path does not match its variant.")
            check_file(item["path"], item["samples_sha256"], kind="samples", variant=item["variant"])
        # A declared single fit has one posterior; unmarked historical output
        # still requires both. Check recorded files first so missing originals
        # retain their unavailable status even if their directory is gone.
        imaging = ImagingResults(root, result.source_id)
        selected = dict(imaging.variants)
        if (len(sources) != len(selected)
                or {item["variant"] for item in sources} != selected.keys()):
            raise ValueError("Recorded imaging variants do not match the imaging directory's declared selection.")
        provenance_path = imaging.path / "input_provenance.json"
        provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
        inputs = dict(provenance.get("input_sha256", {}))
        for report in imaging.reports.values():
            inputs.update(report.get("input_sha256", {}))
        for path, expected in inputs.items():
            check_file(path, expected, kind="input")
        imaging.require_ready()
        arrays = {}
        for name, _ in imaging.variants:
            for key, value in imaging._physical_samples(name).items():
                arrays[f"imaging_{name}_samples_{key}"] = value
            fit_path = imaging.path / name / "fit.npz"
            expected = imaging.reports.get(name, {}).get("fit_sha256")
            if expected:
                check_file(fit_path, expected, kind="prediction", variant=name)
            with np.load(fit_path, allow_pickle=False) as fit:
                data, model, residual, mask, error = (
                    fit[key] for key in ("image", "model", "residual", "mask", "error"))
                if data.ndim != 2 or any(value.shape != data.shape
                                        for value in (model, residual, mask, error)):
                    raise ValueError(f"Imaging prediction arrays have inconsistent shapes: {fit_path}")
                valid = ~mask.astype(bool)
                if (not valid.any() or not np.all(error[valid] > 0)
                        or not np.allclose(residual[valid], (data[valid]-model[valid])/error[valid],
                                           rtol=1e-6, atol=1e-6)):
                    raise ValueError(f"Imaging prediction residuals are invalid: {fit_path}")
                for key in fit.files:
                    arrays[f"imaging_{name}_fit_{key}"] = fit[key].copy()
            summary = imaging.inclination_summary(name)
            arrays[f"imaging_{name}_inclination_q0"] = summary.index.to_numpy()
            columns = ["median_deg", "minus_1sigma_deg", "plus_1sigma_deg", "incompatible_fraction"]
            arrays[f"imaging_{name}_inclination_summary"] = summary[columns].to_numpy(dtype=float)
            arrays[f"imaging_{name}_inclination_summary_columns"] = np.asarray(columns)
        return imaging, arrays, {"status": ("included as context; inclination fixed by user, not an imaging prior"
                                            if fixed else
                                            "included as context; Gaussian mean and width selected by user"
                                            if inherited else "included"),
                                 "reports": imaging.reports, "input_provenance": imaging._check_inputs(),
                                 "recorded_sources": sources, "file_checks": checks}
    except OSError as exc:
        # Kinematic samples and their saved prior remain usable without the
        # original imaging files. Never substitute another imaging run.
        return omitted("original imaging files unavailable", str(exc))
    except (ValueError, KeyError, EOFError, BadZipFile) as exc:
        return omitted("omitted: original imaging files changed or incomplete", str(exc))


def _save_plot(figure, directory, name):
    import matplotlib.pyplot as plt
    try:
        figure.savefig(directory / f"{name}.png", dpi=160, bbox_inches="tight")
    finally:
        plt.close(figure)


def _render_plots(result, directory, radius_pc, imaging):
    plots = {
        "inputs": result.input_figure,
        "fit": lambda: result.fit_figure(radius_pc=radius_pc),
        "kinematic_center": lambda: result.center_figure(radius_pc=radius_pc),
        "corner": result.corner,
        "traces": result.traces,
        ("rotation_curve" if _samples_deprojected_velocity(result.report)
         else "projected_rotation_curve"): result.rotation_curve,
    }
    if imaging is not None:
        plots.update(pysersic_input=imaging.input_figure, pysersic_fit=imaging.figure,
                     pysersic_corner=imaging.corner, pysersic_inclination=imaging.inclination_figure)
    for name, make_figure in plots.items():
        _save_plot(make_figure(), directory, name)


def _run_details(result):
    """Keep the fit completion time separate from the later export time."""
    report = result.report
    finished = report.get("completed_utc") or report.get("finished_utc")
    basis = "recorded run timestamp"
    if not finished:
        finished = datetime.fromtimestamp((result.path/"report.json").stat().st_mtime,
                                         timezone.utc).isoformat()
        basis = "estimated from report.json modification time (no explicit completion timestamp recorded)"
    config = {}
    if (result.path/"run_config.yaml").exists():
        import yaml
        config = yaml.safe_load((result.path/"run_config.yaml").read_text()) or {}
    return dict(finished_utc=finished, finished_time_basis=basis,
                seed=report.get("settings", {}).get("seed", config.get("mcmc", {}).get("seed")),
                elapsed_seconds_last_invocation=report.get("elapsed_seconds"))


def _run_info_text(metadata, summary):
    report, details, exported = metadata["report"], metadata["run_info"], metadata["export"]
    fixed = _fixed_inclination_deg(report)
    def local_time(value):
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return stamp.astimezone(ZoneInfo("America/Phoenix")).strftime("%Y-%m-%d %H:%M:%S %Z")
    walkers, warmup, draws = (report[k] for k in ("walkers", "warmup", "draws"))
    selection = (f"Initialized from inclination scan: {report['seed_inclination_deg']:g} deg; inclination sampled"
                 if report.get("seed_kind") == "inclination_scan" else
                 f"Selected candidate: {report.get('chosen_candidate_id', 'not recorded')}")
    if report.get("seed_kind") == "manual_start":
        selection = (f"Manual starting V_inf: {report['start_velocity_kms']:g} km/s; "
                     + (f"starting inclination: {report['start_inclination_deg']:g} deg; both sampled"
                        if fixed is None else f"fixed inclination: {fixed:g} deg; V_inf sampled"))
    lines = [f"DINGO RUN INFORMATION — ID{report['source_id']}", "",
        f"Run: {Path(exported['original_run_directory']).name}",
        selection,
        f"Status: {report['status']}",
        f"Finished: {local_time(details['finished_utc'])} (America/Phoenix)",
        f"Finished (UTC): {details['finished_utc']}",
        f"Completion time source: {details['finished_time_basis']}",
        f"Final files exported: {local_time(exported['created_utc'])} (America/Phoenix)", "",
        f"Sampler: {report.get('sampler', 'not recorded')}",
        f"Device: {report.get('device', 'not recorded')}",
        f"Ensemble walkers: {walkers} (interacting; not independent chains)",
        f"Warmup / burn-in steps per walker: {warmup}",
        f"Retained steps per walker: {draws}",
        f"Total steps per walker: {warmup + draws}",
        f"Total retained draws: {walkers * draws} (correlated, not independent samples)",
        f"Random seed: {details['seed'] if details['seed'] is not None else 'not recorded'}"]
    elapsed = details.get("elapsed_seconds_last_invocation")
    if elapsed is not None:
        lines.append(f"Elapsed time: {elapsed:.1f} seconds ({elapsed/60:.2f} minutes; last worker invocation)")
    lines += ["", f"Model: {report.get('model', 'R+C rotation')}",
        f"Intrinsic dispersion: {report['intrinsic_dispersion_kms']:g} km/s (fixed)",
        f"Redshift: {report['redshift']:g}",
        f"Stack pixel scale: {report['pixel_scale_arcsec']:g} arcsec/pixel",
        f"Sampled parameters ({len(report['param_names'])}): " + ", ".join(report['param_names']),
        f"Sampling checks: {'PASSED' if report.get('sampling_pass') else 'NOT PASSED — provisional intervals'}"]
    prior = report.get("inclination_prior", {})
    if fixed is not None:
        lines += [f"Fixed inclination: {fixed:g} deg (not sampled).",
                  "The parent inclination prior is recorded as provenance and is not applied.",
                  "Intervals are conditional on this fixed inclination; they exclude inclination uncertainty."]
    elif prior.get("type") == "gaussian":
        lines.append(f"Gas inclination prior: Gaussian mu={np.rad2deg(prior['mu']):.3f} deg, "
                     f"sigma={np.rad2deg(prior['sigma']):.3f} deg before truncation to "
                     f"{np.rad2deg(prior['min']):g}–{np.rad2deg(prior['max']):g} deg")
    elif prior:
        lines.append(f"Gas inclination prior: {prior['type']}")
    if report.get("seed_kind") == "inclination_scan":
        lines += ["User-selected Gaussian replaces the parent inclination prior; it does not fix inclination.",
                  "Intervals depend on this prior; initializing at the scan fit does not constrain the posterior to it.",
                  f"Parent run (preserved): {report['parent_run']}",
                  f"Scan fit (preserved): {report['scan_dir']}"]
    if report.get("seed_kind") == "manual_start":
        lines += [("Manual starting values do not change the prior or fix either parameter."
                   if fixed is None else
                   "The starting velocity remains free; inclination is held fixed throughout MCMC."),
                  f"Other initial parameters from parent candidate: {report['base_candidate_id']}",
                  f"Parent run (preserved): {report['parent_run']}"]
    centers = metadata.get("kinematic_centers")
    if centers:
        x, y = centers["points"]["rectified"]
        lines += [f"Displayed fit's kinematic center (zero-based corrected-image pixels): x={x:.5f}, y={y:.5f}",
                  f"Offset from nominal cutout reference [pc]: {centers['offset_pc']}",
                  "This reference is not a measured stellar center; direct-image alignment is assumed."]
    lines += ["", "RESULTS — median (+upper / -lower 1-sigma; 16th–84th percentiles)"]
    if _samples_deprojected_velocity(report):
        lines += ["Primary result: deprojected rotation V(r); Vinf is the sampled asymptotic amplitude.",
                  ("Vinf and inclination are separate sampled parameters; Vinf sin(i) is derived from each paired draw."
                   if fixed is None else
                   "Vinf is sampled; Vinf sin(i) uses the fixed inclination for every draw.")]
    else:
        lines.append("Primary result: projected rotation V(r) sin(i); Vinf sin(i) is its asymptotic amplitude.")
    for name, row in summary.iterrows():
        lines.append(f"{name}: {row['median']:.3f} (+{row['plus_1sigma']:.3f} / -{row['minus_1sigma']:.3f})")
    lines += ["", "Intervals are conditional on the selected solution and approximate stacked-image noise model.",
        ("Inclination and the other fitted parameters remain free; this is not a one-parameter fit."
         if fixed is None else
         "Inclination is fixed; the other fitted parameters remain free. This is not a one-parameter fit."),
        "Both projected and deprojected velocity profiles, and derived mass arrays, are retained in results.npz.",
        "The comparison radius is not a measured limit of resolved emission.",
        f"PySersic export: {metadata['imaging']['status']}",
        f"Original run/checkpoint directory: {exported['original_run_directory']}",
        "results.npz contains the full metadata, diagnostics, samples and final images."]
    if report.get("selection_notes"):
        lines.append(f"Selection notes: {report['selection_notes']}")
    if metadata["imaging"].get("detail"):
        lines += [f"PySersic export detail: {metadata['imaging']['detail']}",
                  metadata["imaging"]["note"]]
    return "\n".join(lines) + "\n"


def export_final(result, working_dir=DEFAULT_WORKING, *, radius_pc=None):
    """Update WORKING_DIR/ID<source> with results.npz, run_info.txt and PNG plots.

    Only completed runs are exported. Sampling checks and model limitations are
    retained in the bundle, including when a completed run is provisional.
    The original checkpoint/run directory is never changed. A failed export
    leaves the previous final folder intact. No fitting is performed here.
    """
    if result.report.get("status") != "complete":
        raise ValueError("This run has not completed; its checkpoints remain in the run cache.")
    if getattr(result, "_final_bundle", None) is not None:
        raise ValueError("This result is already in a final folder; export from the original run cache.")
    root = Path(working_dir).expanduser().resolve()
    destination = root / f"ID{result.source_id}"
    if destination.is_relative_to(result.path) or result.path.is_relative_to(destination):
        raise ValueError("Keep the final working directory separate from the run cache.")
    _check_destination(destination, result.source_id)
    # These readers verify checksums and source/shape before anything is published.
    result._samples()
    pixels = result._prediction_pixels()
    from .rotation import _checked
    _checked(result.path / "samples.npz", result.report.get("samples_sha256"), "Posterior samples")
    with np.load(result.path / "samples.npz", allow_pickle=False) as samples:
        arrays = {key: samples[key].copy() for key in samples.files}
    arrays.update({f"fit_{key}": value for key, value in pixels.items()})
    profiles = result.radial_profiles()
    arrays.update(radius_pc=profiles["radius_pc"],
                  vsini_kms_p16_p50_p84=profiles["vsini_kms"],
                  vrot_kms_p16_p50_p84=profiles["vrot_kms"],
                  mass_msun_p16_p50_p84=profiles["mass_msun"],
                  quantiles=np.asarray([.16, .5, .84]))
    summary = result.velocity_summary(radius_pc=radius_pc)
    arrays.update(summary_names=summary.index.to_numpy(dtype=str),
                  summary_columns=np.asarray(summary.columns, dtype=str),
                  summary_values=summary.to_numpy(dtype=float))
    imaging, imaging_arrays, imaging_metadata = _imaging_products(result)
    if imaging_metadata.get("detail"):
        print(f"PySersic export: {imaging_metadata['status']}. "
              f"{imaging_metadata['note']}", flush=True)
    arrays.update(imaging_arrays)
    deprojected = _samples_deprojected_velocity(result.report)
    fixed = result.fixed_inclination_deg
    metadata = {
        "schema": SCHEMA, "report": result.report, "imaging": imaging_metadata,
        "kinematic_centers": result.center_geometry(),
        "run_info": _run_details(result),
        "export": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "original_run_directory": str(result.path),
            "comparison_radius_pc": float(profiles["reference_radius_pc"] if radius_pc is None else radius_pc),
            "physical_scale": result.physical_scale(),
            "primary_velocity": ("Vrot(r), deprojected rotation at disk-plane radius r" if deprojected
                                 else "Vrot(r) sin(i), projected rotation at disk-plane radius r"),
            "sampled_velocity_parameter": "velocity.V_rot" if deprojected else "velocity.V_sini",
            "sampled_velocity_definition": ("Vinf, deprojected asymptotic rotation speed" if deprojected
                                            else "Vinf sin(i), projected asymptotic rotation speed"),
            "inclination_mode": "sampled" if fixed is None else "fixed",
            "fixed_inclination_deg": fixed,
            "inclination_uncertainty_note": ("Inclination is sampled in the posterior."
                if fixed is None else
                "Intervals are conditional on the fixed inclination; they exclude inclination uncertainty."),
            "chain_axes": ["retained_step", "walker", "parameter"],
            "chain_parameter_units": {name: ("km/s" if name.endswith(("V_sini", "V_rot")) else
                "radian" if name.endswith(("inc_v", "theta_v")) else
                "dimensionless" if name == "noise.ln_f" else "pixel")
                for name in result.report["param_names"]},
            "profile_note": ("Pointwise 16/50/84 percentiles from paired draws. "
                            + ("Deprojected speed uses sampled Vinf; projected speed is Vinf sin(i) from each draw. "
                               if deprojected else "Projected speed uses sampled Vinf sin(i). ")
                            + "Inclination remains in the disk geometry. Illustrative radius range; "
                            "observed radial coverage not established. Mass is spherical-equivalent r Vrot^2/G, "
                            "without pressure support or lensing correction."),
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    print(f"Saving final results and plots to {destination}", flush=True)
    with tempfile.TemporaryDirectory(prefix=".dingo_export_", dir=root.parent) as temporary:
        stage = Path(temporary) / destination.name
        stage.mkdir()
        _render_plots(result, stage, radius_pc, imaging)
        (stage/"run_info.txt").write_text(_run_info_text(metadata, summary), encoding="utf-8")
        metadata["export"]["files"] = ["results.npz", *sorted(p.name for p in stage.iterdir())]
        arrays["metadata_json"] = np.asarray(json.dumps(metadata, allow_nan=False))
        np.savez_compressed(stage / "results.npz", **arrays)
        _check_destination(destination, result.source_id)
        backup = Path(temporary) / "previous"
        if destination.exists():
            destination.rename(backup)
        try:
            stage.rename(destination)
        except BaseException:
            if backup.exists():
                backup.rename(destination)
            raise
    print(f"Final files: {destination} ({len(metadata['export']['files'])} files)", flush=True)
    return destination
