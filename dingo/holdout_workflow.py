"""Notebook orchestration for the declared real ID21636 prediction experiment.

Imports are CPU-only. GPU fitting and CPU assessment each run in their own
process; opening saved results never initializes JAX or CUDA. This is a local
optimization diagnostic, not posterior sampling or a generic source pipeline.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from uuid import uuid4

import numpy as np


REPO = Path(__file__).resolve().parents[1]
KINEMATICS = REPO.parent
DEFAULT_SOURCE = KINEMATICS / "results/real_sources/ID21636/morphology_controls_v1"
DEFAULT_CACHE_ROOT = KINEMATICS / "results/real_sources/ID21636"
DEFAULT_WORKING_ROOT = KINEMATICS / "working/holdout_runs"
DEFAULT_REFERENCE = KINEMATICS / "working/ID21636_validation_2026-10-21/results.npz"


def _json(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_results(path=DEFAULT_REFERENCE):
    """Copy a portable final NPZ into memory without changing any saved files."""
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path /= "results.npz"
    if not path.is_file():
        raise FileNotFoundError(f"No completed results file: {path}")
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    if int(arrays.get("source_id", -1)) != 21636:
        raise ValueError("This notebook is the declared ID21636 experiment.")
    report = json.loads(str(arrays["report_json"]))
    if not report.get("completed"):
        raise ValueError("The saved held-out assessment is incomplete.")
    return arrays


def _worker_environment(gpu):
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", XLA_PYTHON_CLIENT_PREALLOCATE="false",
               MPLBACKEND="Agg", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               MKL_NUM_THREADS="4")
    # A CPU-only plotting kernel must not silently force the training child to CPU.
    if gpu:
        for key in ("JAX_PLATFORMS", "JAX_PLATFORM_NAME"):
            if env.get(key, "").strip().lower() == "cpu":
                env.pop(key)
    else:
        env.update(JAX_PLATFORMS="cpu", JAX_PLATFORM_NAME="cpu", CUDA_VISIBLE_DEVICES="")
    return env


def _stop_child(process):
    """Stop only the worker process group created by this notebook call."""
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            process.wait()
            return
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=8)


def _run_worker(command, *, gpu, log_path, stage, heartbeat_seconds=10.):
    """Stream worker output and keep silent compilation/assessment visible."""
    started = time.monotonic()
    print(f"\n{stage}", flush=True)
    print(f"Log: {log_path}", flush=True)
    process = subprocess.Popen(command, cwd=REPO, env=_worker_environment(gpu),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        start_new_session=True)
    messages = queue.Queue()

    def read_stdout():
        try:
            for line in process.stdout:
                messages.put(line)
        finally:
            messages.put(None)

    reader = threading.Thread(target=read_stdout, daemon=True)
    reader.start()
    try:
        with Path(log_path).open("a", encoding="utf-8") as log:
            log.write(f"\n{datetime.now(timezone.utc).isoformat()} {stage}\n")
            while True:
                try:
                    line = messages.get(timeout=heartbeat_seconds)
                except queue.Empty:
                    elapsed = time.monotonic()-started
                    message = f"  {stage}: running, {elapsed:.0f} seconds elapsed\n"
                    print(message, end="", flush=True)
                    log.write(message); log.flush()
                    continue
                if line is None:
                    break
                print(line, end="", flush=True)
                log.write(line); log.flush()
            code = process.wait()
        if code:
            raise RuntimeError(f"{stage} failed (exit {code}). Full output: {log_path}")
    except BaseException:
        _stop_child(process)
        raise
    finally:
        process.stdout.close()
        reader.join(timeout=2)
    elapsed = time.monotonic()-started
    print(f"{stage} finished in {elapsed:.1f} seconds.", flush=True)
    return elapsed


def _read_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def export_results(cache_directory, destination):
    """Export completed assessment to one NPZ, four plots and run_info.txt.

    Existing final files are never replaced. Full observations and training
    predictions are included for inspecting any exposure without recalculation.
    """
    cache = Path(cache_directory).resolve()
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Final directory already contains files: {destination}")
    report = _json(cache/"holdout.json")
    if int(report.get("source_id", -1)) != 21636 or not report.get("completed"):
        raise ValueError("Export needs a completed real ID21636 assessment.")
    covariance = _json(cache/"covariance_assessment.json")
    centers = _json(cache/"projected_centers.json")
    if report.get("projected_centers_sha256") and _sha(cache/"projected_centers.json") != report["projected_centers_sha256"]:
        raise ValueError("Saved projected centers changed after image assessment.")
    for input_path, digest in covariance.get("input_sha256", {}).items():
        if _sha(input_path) != digest:
            raise ValueError(f"Covariance input changed: {input_path}")
    observations = _read_npz(cache/"observations.npz")
    held, train = observations["held_indices"], observations["train_indices"]
    fits = report["fits"]
    if [f["start"] for f in fits] != report["settings"]["starts"]:
        raise ValueError("Saved candidates do not match the declared starts.")
    for f in fits:
        for file_key, hash_key in (("artifact", "artifact_sha256"),
                                   ("heldout_artifact", "heldout_artifact_sha256")):
            if _sha(cache/f[file_key]) != f[hash_key]:
                raise ValueError(f"Fit artifact changed: {cache/f[file_key]}")
    if _sha(cache/"observations.npz") != report["observation_sha256"]:
        raise ValueError("Saved observations changed after training.")
    fitted = [_read_npz(cache/f["artifact"]) for f in fits]
    predictions = [_read_npz(cache/f["heldout_artifact"]) for f in fits]
    nominal = observations["line_centers_xy"]
    exposure_names = np.array([Path(e["path"]).name for e in report["exposures"]])
    source = _json(Path(report["source_report"]))
    origin = _json(Path(source["settings"]["run"])/"dispersion.json")
    baseline = _json(Path(origin["settings"]["baseline"])/"coupled.json")
    grid = _json(Path(baseline["settings"]["grid"])/"grid.json")
    from astropy.cosmology import Planck18
    import astropy.units as u
    redshift = float(grid["source_summary"]["z"])
    radius_pc = .5 * Planck18.kpc_proper_per_arcmin(redshift).to_value(u.pc/u.arcsec)
    arrays = dict(source_id=np.array(21636), labels=np.array([f["start"] for f in fits]),
        report_json=np.array(json.dumps(report)),
        covariance_assessment_json=np.array(json.dumps(covariance)),
        projected_centers_json=np.array(json.dumps(centers)),
        source_summary_json=np.array(json.dumps(grid["source_summary"])),
        cache_directory=np.array(str(cache)), redshift=np.array(redshift),
        radius_reference_pc=np.array(radius_pc), radius_reference_arcsec=np.array(.5),
        parameters=np.stack([a["parameters"] for a in fitted]),
        initial_parameters=np.stack([a["initial_parameters"] for a in fitted]),
        vrot_kms=np.array([f["vrot_0p5arcsec_kms"] for f in fits]),
        vproj_kms=np.array([f["vproj_0p5arcsec_kms"] for f in fits]),
        inclination_deg=np.array([f["inclination_deg"] for f in fits]),
        held_exposure_indices=held, train_exposure_indices=train,
        pupils=observations["pupils"][held], all_pupils=observations["pupils"],
        exposure_names=exposure_names[held], all_exposure_names=exposure_names,
        science=observations["science"][held], error=observations["error"][held],
        valid=observations["valid"][held], all_science=observations["science"],
        all_error=observations["error"], all_valid=observations["valid"],
        line_centers_xy=nominal[held], all_line_centers_xy=nominal,
        source_east=fitted[0]["source_east"], source_north=fitted[0]["source_north"],
        source_flux=np.stack([a["source_flux"] for a in fitted]),
        source_velocity=np.stack([a["source_velocity"] for a in fitted]),
        predicted_line=np.stack([a["line"] for a in predictions]),
        predicted_total=np.stack([a["total"] for a in predictions]),
        residual_ERR=np.stack([a["residual_raw_ERR"] for a in predictions]),
        all_predicted_line=np.stack([a["line"] for a in fitted]),
        training_predicted_total=np.stack([a["total"] for a in fitted]),
        training_continuum=np.stack([a["continuum"] for a in fitted]),
        training_background=np.stack([a["background"] for a in fitted]),
        training_residual_ERR=np.stack([a["residual"] for a in fitted]),
        projected_kinematic_centers_xy=np.stack([a["projected_kinematic_centers_xy"] for a in predictions]),
        all_projected_kinematic_centers_xy=np.stack([centers[f["start"]]["centers_xy"] for f in fits]),
        sideband_prediction=predictions[0]["sideband_prediction"],
        sideband_prediction_variance=predictions[0]["sideband_prediction_variance"],
        line_window=predictions[0]["line_window"], core=predictions[0]["core"])
    for a in predictions:
        np.testing.assert_array_equal(a["held_indices"], held)
        np.testing.assert_array_equal(a["sideband_prediction"], arrays["sideband_prediction"])
    plot_pairs = [("heldout_saved_lower_RC.png", "lower_velocity_RC.png"),
                  ("heldout_saved_higher_RC.png", "higher_velocity_RC.png"),
                  ("heldout_saved_lower_all.png", "lower_velocity_all_exposures.png"),
                  ("heldout_saved_higher_all.png", "higher_velocity_all_exposures.png")]
    for original, _ in plot_pairs:
        if not (cache/original).is_file():
            raise FileNotFoundError(cache/original)
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination/"results.partial.npz"
    np.savez_compressed(temporary, **arrays)
    temporary.replace(destination/"results.npz")
    for original, final in plot_pairs:
        shutil.copy2(cache/original, destination/final)
    lines = ["ID21636 — joint R+C withheld-exposure prediction experiment", "",
        "Optimization and prediction assessment; NOT MCMC. No posterior errors.",
        f"Training started (UTC): {report['started_utc']}",
        f"Training finished (UTC): {report.get('training_finished_utc')}",
        f"Image assessment finished (UTC): {report.get('assessment_finished_utc')}",
        f"Covariance assessment finished (UTC): {covariance.get('completed_utc')}",
        f"Export finished (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"Device: {report.get('device')}", f"Full diagnostics and log: {cache}", "",
        f"Reference radius: 0.5 arcsec = {radius_pc:.2f} pc at z={redshift:g} (Planck18).",
        report['split_policy'], report['holdout_policy'], "",
        "Settings:", json.dumps(report['settings'], indent=2), "", "Results:"]
    for f in fits:
        lines.append(f"{f['start']}: Vrot={f['vrot_0p5arcsec_kms']:.3f} km/s; "
            f"Vsin(i)={f['vproj_0p5arcsec_kms']:.3f} km/s; i={f['inclination_deg']:.3f} deg; "
            f"turnover={f['turnover_arcsec']:.5f} arcsec; J={f['objective']:.5f}; "
            f"stationarity_passed={f['stationarity_passed']}; bounds={f['boundary_parameters']}")
    lines += ["", "Conditional covariance-aware prediction scores:",
              json.dumps(covariance['totals'], indent=2), "", "Interpretation:",
              covariance['interpretation'], *report['limitations'], *covariance['limitations'],
              "Bound hits do not establish measured parameters. Inspect residuals and source-map edges.",
              "Training arrays apply only at train_exposure_indices; withheld total predictions use sidebands.",
              "results.npz embeds complete reports, observations, masks, centers and candidate predictions."]
    (destination/"run_info.txt").write_text("\n".join(lines)+"\n")
    return destination/"results.npz"


def run_id21636(source_dir=DEFAULT_SOURCE, cache_root=DEFAULT_CACHE_ROOT,
                working_dir=DEFAULT_WORKING_ROOT, *, maxiter=1600, refine=2400,
                python=None, run_name=None):
    """Run the four declared starts, then assess frozen held-out predictions.

    Every call creates new timestamped directories. Interrupting the notebook
    stops its own worker; logs/checkpoints remain in cache. Successful finals
    are saved beneath working_dir/ID21636/<unique run name>/results.npz.
    """
    for value in (maxiter, refine):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError("maxiter and refine must be positive integers.")
    source_dir = Path(source_dir).expanduser().resolve()
    report = _json(source_dir/"dispersion.json")
    if report.get("source_id") != 21636 or not report.get("completed"):
        raise ValueError("Use the completed ID21636 morphology-controls source directory.")
    prefix = "heldout" if run_name is None else str(run_name)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,70}", prefix):
        raise ValueError("run_name must be a short filename label, without slashes.")
    tag = f"{prefix}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_{uuid4().hex[:6]}"
    cache = Path(cache_root).expanduser().resolve()/tag
    final = Path(working_dir).expanduser().resolve()/"ID21636"/tag
    if final.exists():
        raise FileExistsError(final)
    cache.mkdir(parents=True, exist_ok=False)
    snapshots = cache/"code"
    snapshots.mkdir()
    for relative in ("diagnostics/native_real_holdout.py", "diagnostics/native_holdout_centers.py",
                     "diagnostics/native_real_holdout_plot.py", "diagnostics/native_holdout_covariance.py",
                     "dingo/holdout_workflow.py", "dingo/native_holdout_continuum.py"):
        shutil.copy2(REPO/relative, snapshots/Path(relative).name)
    executable = str(python or sys.executable)
    log = cache/"run.log"
    stages = [
        ("1/4 GPU training — four starts", True,
         [executable, "-u", "-m", "diagnostics.native_real_holdout", "--source", str(source_dir),
          "--output", str(cache), "--maxiter", str(maxiter), "--refine", str(refine)]),
        ("2/4 Project fitted centers", False,
         [executable, "-u", "-m", "diagnostics.native_holdout_centers", str(cache)]),
        ("3/4 Withheld-image prediction assessment", False,
         [executable, "-u", "-m", "diagnostics.native_real_holdout_plot", str(cache)]),
        ("4/4 Correlated continuum uncertainty", False,
         [executable, "-u", "-m", "diagnostics.native_holdout_covariance", str(cache)])]
    status = dict(started_utc=datetime.now(timezone.utc).isoformat(), source_id=21636,
                  cache_directory=str(cache), final_directory=str(final), completed=False,
                  maxiter=int(maxiter), refine=int(refine), stages=[])
    status_path = cache/"workflow.json"

    def save_status():
        status_path.write_text(json.dumps(status, indent=2)+"\n")

    print(f"Fresh ID21636 experiment. Final results: {final}\nCache/log: {cache}", flush=True)
    save_status()
    try:
        for label, gpu, command in stages:
            status['active_stage'] = label; save_status()
            elapsed = _run_worker(command, gpu=gpu, log_path=log, stage=label)
            status['stages'].append(dict(stage=label, seconds=elapsed)); save_status()
        result = export_results(cache, final)
    except BaseException as exc:
        status.update(failed_utc=datetime.now(timezone.utc).isoformat(),
                      error=f"{type(exc).__name__}: {exc}")
        save_status()
        print(f"Run stopped. Logs/checkpoints are preserved at {cache}", flush=True)
        raise
    status.pop('active_stage', None)
    status.update(completed=True, finished_utc=datetime.now(timezone.utc).isoformat())
    save_status()
    print(f"\nFinished. Open: {result}\nRun details: {final/'run_info.txt'}", flush=True)
    return result
