"""Noiseless image error from interpolating the known injected brightness.

These are constructive representations at the true kinematics, not optimized
maps or posterior intervals. Small errors establish that a good map exists in
the control-grid representation; they do not establish its posterior weight.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from dingo.exposures import load_native_target
from dingo.native_forward import NativeLineProjector, projected_arctan_velocity
from dingo.native_fitting import BilinearLineMap, StripContinuumLikelihood
from diagnostics.native_map_prior_comparison import shape_moments

REPO = Path(__file__).resolve().parents[1]


def run(baseline_path, output):
    torch.set_num_threads(4)
    baseline = json.loads((baseline_path/"posterior.json").read_text())
    source_dir = Path(baseline["source_run"])
    source = json.loads((source_dir/"recovery.json").read_text())
    settings = source["arguments"]
    if settings["fit_blur"] != settings["truth_blur"]:
        raise ValueError("The representation check requires matched response kernels")
    target = load_native_target(settings["bundle"], REPO/f"data/ID{source['source_geometry']}/config_kinematics.yaml", settings["size"])
    paths = [str(e.path) for e in target.exposures]
    train, held = ([paths.index(p) for p in source[k]] for k in ("train_exposures", "held_exposures"))
    with np.load(source_dir/"injection.npz") as a:
        true_line, error, valid, flux = (a[k] for k in ("true_line", "error", "valid", "truth_grid_flux"))
        east, north = a["truth_east"], a["truth_north"]
    truth_grid = BilinearLineMap(2, settings["truth_grid"], dtype=torch.float64)
    brightness = (flux/truth_grid.area.numpy()).reshape(settings["truth_grid"], -1)
    yy, xx = np.mgrid[-4:5, -4:5]
    kernel = np.exp(-.5*(xx*xx+yy*yy)/settings["fit_blur"]**2)
    likelihood = StripContinuumLikelihood(np.zeros_like(true_line), error, valid, [e.pupil for e in target.exposures], degree=settings["degree"])
    v, rt, theta = source["truth_vref_rturn_theta"]
    inc = source["truth_inclination_rad"]
    xp, yp = np.cos(theta)*east+np.sin(theta)*north, -np.sin(theta)*east+np.cos(theta)*north
    radius = np.sqrt(xp*xp+(yp/np.cos(inc))**2)
    report = {"source_run": str(source_dir), "fixed_true_kinematics": True, "constructive_not_optimized": True,
        "truth_moments": shape_moments(east, north, flux).tolist(),
        "enclosed_injected_flux_by_deprojected_radius": {r: float(flux[radius <= r].sum()/flux.sum()) for r in (.1,.15,.2,.3,.4,.5,.6)},
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "injection_sha256": hashlib.sha256((source_dir/"injection.npz").read_bytes()).hexdigest(), "representations": []}
    for size, quadrature, interpolation in ((17, 49, "brightness"), (17, 49, "log_brightness"),
                                            (17, 97, "log_brightness"), (33, 97, "brightness"),
                                            (97, 97, "brightness")):
        stride = (settings["truth_grid"]-1)//(size-1)
        if stride*(size-1) != settings["truth_grid"]-1:
            raise ValueError("Control nodes must be present in the saved truth grid")
        grid = BilinearLineMap(size, quadrature, flux_unit=1., dtype=torch.float64)
        projected = NativeLineProjector.from_target(target, grid.east.numpy(), grid.north.numpy(), kernel,
            np.ones(len(paths)), spectral_samples=7, dtype=torch.float64, deposition=settings["deposition"])
        with torch.no_grad():
            controls = torch.as_tensor(brightness[::stride, ::stride].copy())
            if interpolation == "brightness":
                particles = grid(controls)[0]
            else:
                log_field = torch.nn.functional.interpolate(controls.log()[None, None],
                    size=(quadrature, quadrature), mode="bilinear", align_corners=True)[0, 0]
                particles = log_field.exp().ravel()*grid.area
            particles *= flux.sum()/particles.sum()
            velocity = projected_arctan_velocity(grid.east, grid.north, v, rt, theta, inc)
            line = projected(particles, velocity, sigma_v=45.)[0]
            residual = likelihood.residualize((line-torch.as_tensor(true_line))*likelihood.inverse_error)
            chi = residual.square().sum((-2, -1)).numpy()
            max_error = float(np.max(np.abs((line.numpy()-true_line)[valid]/error[valid])))
        report["representations"].append({"controls": size, "quadrature": quadrature, "interpolation": interpolation,
            "max_pixel_difference_in_error_units": max_error,
            "train_projected_squared_error": float(chi[train].sum()),
            "held_projected_squared_error": float(chi[held].sum()),
            "map_moments": shape_moments(grid.east.numpy(), grid.north.numpy(), particles.numpy()).tolist()})
    if report["representations"][-1]["max_pixel_difference_in_error_units"] > 1e-9:
        raise AssertionError("Exact-grid representation does not reproduce the saved injection")
    output.mkdir(parents=True, exist_ok=True)
    (output/"representation.json").write_text(json.dumps(report, indent=2))
    (output/"representation_code.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    run(args.baseline, args.output)
