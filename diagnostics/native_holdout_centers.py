"""Project saved ID21636 kinematic centers through exact exposure calibration.

CPU-only reconstruction; no optimization, resampling or fit parameter changes.
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np


def project_centers(target, parameters, *, nmap, registration_sigma, projector=None):
    """Use the same calibrated polynomial and additive shifts as the renderer."""
    if projector is None:
        from dingo.native_forward import projection_coefficients
        projector = projection_coefficients
    p = np.asarray(parameters, dtype=float)
    count = len(target.exposures)
    regstart = 11+int(nmap)
    if p.ndim != 1 or len(p) != regstart+4*count:
        raise ValueError("Saved parameter layout differs from the coupled flexible model.")
    shifts = p[regstart:regstart+2*count].reshape(count, 2)*registration_sigma
    center = p[6:8]*.1
    delta_wave = target.wavelength_um*(100*p[4])/299792.458
    positions = []
    for e, shift in zip(target.exposures, shifts):
        xy = e.source_detector_xy(np.array([center[0]]), np.array([center[1]]))
        coefficients = projector(e.calibration, xy, target.wavelength_um, e.origin_xy)[0]
        result = coefficients[:, -1].copy()
        for degree in range(coefficients.shape[-1]-2, -1, -1):
            result = result*delta_wave+coefficients[:, degree]
        positions.append(result+shift)
    return np.asarray(positions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    output = args.directory.resolve()
    destination = output/"projected_centers.json"
    if destination.exists():
        raise FileExistsError(f"Preserving saved center projections: {destination}")
    read = lambda p: json.loads(Path(p).read_text())
    report = read(output/"holdout.json")
    if report.get("source_id") != 21636 or not report.get("training_completed"):
        raise ValueError("Expected completed ID21636 training fits.")
    source = read(report["source_report"])
    origin = read(Path(source["settings"]["run"])/"dispersion.json")
    baseline = read(Path(origin["settings"]["baseline"])/"coupled.json")
    grid = read(Path(baseline["settings"]["grid"])/"grid.json")
    from dingo.exposures import load_native_target
    from dingo.native_real import RealVelocityProfile
    settings = grid["settings"]
    target = load_native_target(settings["bundle"], settings["config"], settings["size"])
    if target.source_id != 21636:
        raise ValueError("Reconstructed another source.")
    if [str(e.path) for e in target.exposures] != [e["path"] for e in report["exposures"]]:
        raise ValueError("Exposure ordering differs from the saved training run.")
    nmap = int(settings["controls"])**2-1
    registration_sigma = inspect.signature(RealVelocityProfile).parameters["registration_sigma"].default
    results = {}
    for fit in report["fits"]:
        with np.load(output/fit["artifact"], allow_pickle=False) as a:
            positions = project_centers(target, a["parameters"], nmap=nmap,
                                       registration_sigma=registration_sigma)
        results[fit["start"]] = dict(centers_xy=positions.tolist(),
            definition="Fitted sky kinematic center projected at systemic wavelength through exact exposure calibration, plus fitted detector registration. Coordinates are zero-based cutout (x,y).",
            nmap=nmap, registration_sigma_pixels=float(registration_sigma))
    destination.write_text(json.dumps(results, indent=2)+"\n")
    print(f"Projected centers for {len(results)} candidates: {destination}", flush=True)


if __name__ == "__main__":
    main()
