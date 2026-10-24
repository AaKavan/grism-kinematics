"""Local R/C information comparison, with a shared unknown emission map.

This is a tangent-space experiment at a specified synthetic source, not a
posterior, a calibrated confidence interval, or a measurement of that source.
The response, registration, center and intrinsic width remain known.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dingo.exposures import load_native_target
from dingo.native_fitting import BilinearLineMap, StripContinuumLikelihood
from dingo.native_forward import NativeLineProjector, projected_arctan_velocity

REPO = Path(__file__).resolve().parents[1]


def profile_information(design, derivatives, rcond=1e-8):
    """Remove linear nuisance directions by SVD, without Schur cancellation.

    Rows are whitened data, columns are nuisance/physical parameters. For a
    joint fit, stack the R and C rows BEFORE removing the shared map. Adding
    individually profiled matrices instead allows an independent map per grism.
    """
    u, singular, _ = torch.linalg.svd(design, full_matrices=False)
    keep = singular > singular[0] * rcond
    u = u[:, keep]
    residual = derivatives - u @ (u.T @ derivatives)
    return residual.T @ residual, {
        "rank": int(keep.sum()), "columns": design.shape[1],
        "relative_smallest_singular": float(singular[-1] / singular[0]),
        "orthogonality_error": float(torch.linalg.norm(u.T @ residual)),
    }


def log_mass_gradient(q, radii):
    """d log[r V(r)^2 / G] / d[Vref/200, rt/.2, theta, inclination].

    Vref is projected speed at 0.5 arcsec; V(r) is deprojected rotation.
    The proportionality is illustrative and assumes circular motion, fixed
    distance and fixed radius. It omits pressure support and mass geometry.
    """
    v, rt, _, inc = np.asarray(q) * [200., .2, 1., 1.]
    radii = np.asarray(radii)
    derivative_rt = (-radii / (rt**2 + radii**2) / np.arctan(radii / rt)
                     + .5 / (rt**2 + .5**2) / np.arctan(.5 / rt))
    return np.stack([np.full_like(radii, 400. / v), .4 * derivative_rt,
                     np.zeros_like(radii), np.full_like(radii, -2. / np.tan(inc))], axis=-1)


def propagated_sigma(information, gradients, rcond=1e-12):
    """Propagate local errors; unidentified combinations have infinite error.

    A pseudoinverse alone would misleadingly give zero error in a null space.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(information)
    keep = eigenvalues > max(eigenvalues[-1], 0.) * rcond
    if eigenvalues[0] < -max(eigenvalues[-1], 1.) * 1e-10:
        raise ValueError("Information matrix is not positive semidefinite")
    coordinates = np.asarray(gradients) @ eigenvectors
    variance = (coordinates[..., keep]**2 / eigenvalues[keep]).sum(axis=-1)
    null = np.linalg.norm(coordinates[..., ~keep], axis=-1)
    unidentified = null > 1e-7 * np.maximum(np.linalg.norm(gradients, axis=-1), 1e-30)
    return np.where(unidentified, np.inf, np.sqrt(variance))


def summarize_information(information, q, radii):
    result = {}
    for name, sigma_deg in (("fixed_inclination", 0.), ("inclination_prior_5deg", 5.),
                            ("inclination_prior_15deg", 15.), ("free_inclination", None)):
        f = np.array(information, copy=True)
        gradients = log_mass_gradient(q, radii)
        if sigma_deg == 0:
            f, gradients = f[:3, :3], gradients[:, :3]
        elif sigma_deg is not None:
            f[3, 3] += 1. / np.deg2rad(sigma_deg)**2
        sigma = propagated_sigma(f, gradients)
        parameter_sigma = propagated_sigma(f, np.eye(len(f)))
        result[name] = {
            "sigma_log_rotation_only_mass": sigma.tolist(),
            "sigma_projected_vref_kms": float(parameter_sigma[0] * 200.),
            "sigma_turnover_arcsec": float(parameter_sigma[1] * .2),
            "sigma_theta_deg": float(np.rad2deg(parameter_sigma[2])),
            "sigma_inclination_deg": (0. if sigma_deg == 0 else float(np.rad2deg(parameter_sigma[3]))),
            "information_eigenvalues": np.linalg.eigvalsh(f).tolist(),
        }
    return result


def run(args):
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA access is blocked or unavailable")
    target = load_native_target(args.bundle, REPO / f"data/ID{args.source}/config_kinematics.yaml", args.size)
    pupils = np.array([e.pupil for e in target.exposures])
    valid = np.stack([e.valid for e in target.exposures])
    error = np.stack([e.error for e in target.exposures])
    error = np.where(valid, error, 1.)
    likelihood = StripContinuumLikelihood(np.zeros_like(error), error, valid, pupils, degree=2, device=args.device)
    train = np.arange(len(pupils)) % 3 != 2
    selections = {"R": train & (pupils == "R"), "C": train & (pupils == "C"), "RC": train}
    q = np.array([210./200., .22/.2, .9, .9])
    radii = np.array([.1, .2, .3, .5])
    y, x = np.mgrid[-4:5, -4:5]
    kernel = np.exp(-.5*(x*x+y*y))
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "completed": False, "synthetic_local_diagnostic_only": True,
        "source_geometry": args.source, "q_names": ["vref/200", "rturn/.2", "theta", "inclination"],
        "q": q.tolist(), "radii_arcsec": radii.tolist(),
        "snr_all_exposures_after_continuum": args.snr,
        "n_exposures": {k: int(v.sum()) for k, v in selections.items()},
        "training_paths": {k: [str(e.path) for e, keep in zip(target.exposures, v) if keep]
                           for k, v in selections.items()},
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "code_sha256": {f: hashlib.sha256((REPO/f).read_bytes()).hexdigest() for f in
                        ("diagnostics/grism_information.py", "dingo/native_forward.py", "dingo/native_fitting.py", "dingo/exposures.py")},
        "limitations": ["Local linearization; not a posterior or repeated-noise coverage test",
                        "Map perturbations unconstrained locally; positivity boundary and large nonlinear excursions omitted",
                        "Known response, center, registration and intrinsic width; independent supplied pixel errors",
                        "Inclination prior centered on truth; free map is not tied to inclination or imaging",
                        "Rotation-only mass proxy, with no pressure correction or distance uncertainty",
                        "Equal-SNR controls rescale single-grism errors ideally, not measured observing-time efficiency"],
        "grids": [],
    }
    for size in args.grids:
        t0 = time.perf_counter()
        grid = BilinearLineMap(size, args.quadrature, device=args.device, dtype=torch.float64)
        projector = NativeLineProjector.from_target(target, grid.east.cpu().numpy(), grid.north.cpu().numpy(),
            kernel, np.ones(len(pupils)), spectral_samples=7, device=args.device, dtype=torch.float64, deposition="cubic")
        axis = torch.linspace(-.63, .63, size, device=args.device, dtype=torch.float64)
        north, east = torch.meshgrid(axis, axis, indexing="ij")
        controls = (torch.exp(-.5*((east/.23)**2+(north/.14)**2))
                    + .6*torch.exp(-.5*(((east-.24)/.07)**2+((north+.12)/.08)**2)))

        def prediction(parameters, maps=controls):
            v = projected_arctan_velocity(grid.east, grid.north, parameters[0]*200.,
                                          parameters[1]*.2, parameters[2], parameters[3])
            return projector(grid(maps), v, sigma_v=45.)

        def whiten(line):
            return likelihood.residualize(line * likelihood.inverse_error)

        with torch.no_grad():
            amplitude = args.snr / float(torch.linalg.norm(whiten(prediction(q))))
            controls *= amplitude
            expected = whiten(prediction(q))[0]
            derivatives = []
            for step in (1e-3, 5e-4):
                columns = []
                for i in range(4):
                    delta = np.eye(4)[i] * step
                    columns.append((whiten(prediction(q+delta))-whiten(prediction(q-delta)))[0] / (2*step))
                derivatives.append(torch.stack(columns, dim=-1))
            derivative_error = float(torch.linalg.norm(derivatives[1]-derivatives[0])/torch.linalg.norm(derivatives[1]))
            kin = derivatives[1]
            # A common scale does not change the map's column space and keeps
            # singular values comfortably away from numerical underflow.
            basis = torch.eye(size*size, device=args.device, dtype=torch.float64).reshape(-1, size, size)*amplitude
            columns = []
            for start in range(0, len(basis), args.batch):
                columns.append(whiten(prediction(q, basis[start:start+args.batch])).flatten(1).T)
            design = torch.cat(columns, dim=1).reshape(*error.shape, size*size)
            entry = {"control_grid": size, "finite_difference_relative_error": derivative_error,
                     "snr_by_selection": {k: float(torch.linalg.norm(expected[v])) for k, v in selections.items()},
                     "by_rank_tolerance": {}}
            matrices = {}
            for tolerance in args.rcond:
                modes = {}
                for label, selection in selections.items():
                    print(f"grid={size} rcond={tolerance:g} mode={label}: projecting {size*size} map directions", flush=True)
                    f, diagnostics = profile_information(design[selection].flatten(0, 2), kin[selection].flatten(0, 2), tolerance)
                    matrices[label] = f.cpu().numpy()
                    modes[label] = {"information": matrices[label].tolist(), "map_svd": diagnostics}
                # This diagnostic deliberately permits two unrelated maps. It
                # isolates how much the shared morphology constraint contributes.
                matrices["RC_independent_maps"] = matrices["R"] + matrices["C"]
                for label in ("R", "C"):
                    scale = (entry["snr_by_selection"]["RC"] / entry["snr_by_selection"][label])**2
                    matrices[f"{label}_equal_snr"] = matrices[label] * scale
                    modes[f"{label}_equal_snr"] = {"information": matrices[f"{label}_equal_snr"].tolist(), "information_scale": scale}
                modes["RC_independent_maps"] = {"information": matrices["RC_independent_maps"].tolist()}
                for label, f in matrices.items():
                    modes[label]["constraints"] = summarize_information(f, q, radii)
                # Joint information should exceed separately profiled maps,
                # up to truncation/numerical error, if map ranks are resolved.
                difference = matrices["RC"] - matrices["RC_independent_maps"]
                entry["by_rank_tolerance"][str(tolerance)] = {"modes": modes,
                    "shared_minus_independent_eigenvalues": np.linalg.eigvalsh(difference).tolist()}
            np.savez_compressed(args.output/f"grid{size}_derivatives.npz", q=q, radii_arcsec=radii,
                whitened_kinematic_derivatives=kin.cpu().numpy(), expected_whitened_line=expected.cpu().numpy(),
                controls=controls.cpu().numpy(), pupils=pupils, train=train)
        entry["seconds"] = time.perf_counter()-t0
        report["grids"].append(entry)
        (args.output/"information.json").write_text(json.dumps(report, indent=2, allow_nan=True))
        print(f"grid={size} completed in {entry['seconds']:.1f}s", flush=True)
    report["completed"] = True
    (args.output/"information.json").write_text(json.dumps(report, indent=2, allow_nan=True))
    plot(report, args.output)
    return report


def plot(report, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharey=True)
    entry = report["grids"][-1]
    modes = entry["by_rank_tolerance"][str(report["arguments"]["rcond"][0])]["modes"]
    for ax, condition in zip(axes, ("fixed_inclination", "inclination_prior_5deg")):
        for label, color, style in (("R", "C0", "-"), ("C", "C1", "-"), ("RC", "C2", "-"),
                                     ("R_equal_snr", "C0", "--"), ("C_equal_snr", "C1", "--")):
            ax.plot(report["radii_arcsec"], modes[label]["constraints"][condition]["sigma_log_rotation_only_mass"],
                    marker="o", color=color, ls=style, label=label.replace("_", " "))
        ax.set(xlabel="Radius [arcsec] (test coordinates)", title=condition.replace("_", " "), yscale="log")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Local sigma(log rotation-only mass proxy)")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Synthetic ID{report['source_geometry']} geometry; {entry['control_grid']}x{entry['control_grid']} free map; not posterior intervals")
    fig.tight_layout()
    fig.savefig(output/"information.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=int, default=3700)
    parser.add_argument("--size", type=int, default=81)
    parser.add_argument("--quadrature", type=int, default=49)
    parser.add_argument("--grids", type=int, nargs="+", default=[13, 17])
    parser.add_argument("--snr", type=float, default=100.)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--rcond", type=float, nargs="+", default=[1e-8, 1e-6, 1e-10])
    run(parser.parse_args())


if __name__ == "__main__":
    main()
