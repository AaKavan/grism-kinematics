"""Conditional injection tests with a free line map and fitted SCI continuum.

Real exposure geometry, errors and masks; synthetic line and continuum data.
Inclination, center, PSF/LSF, throughput and registration are still known inputs.
The default generator uses an analytic clumpy map on a finer source grid than
the fit. Outputs are experimental diagnostics, never galaxy measurements.
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
from dingo.native_forward import NativeLineProjector, projected_arctan_velocity
from dingo.native_fitting import BilinearLineMap, NativeMapFitter, StripContinuumLikelihood

REPO = Path(__file__).resolve().parents[1]


def run(args, target=None):
    torch.set_num_threads(args.threads)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA access is blocked or unavailable in this process")
    dtype = torch.float64 if args.precision == "float64" else torch.float32
    if target is None:
        target = load_native_target(args.bundle, REPO / f"data/ID{args.source}/config_kinematics.yaml", args.size)
    pupils = [e.pupil for e in target.exposures]
    valid, error = np.stack([e.valid for e in target.exposures]), np.stack([e.error for e in target.exposures])
    error = np.where(valid, error, 1.)
    n = len(pupils)
    train = np.where(np.arange(n) % 3 != 2)[0]
    held = np.setdiff1d(np.arange(n), train)
    grisms = getattr(args, "grisms", "RC")
    train = np.array([i for i in train if pupils[i] in grisms], dtype=int)
    if len(train) == 0:
        raise ValueError("No training exposures for selected grisms")
    y, x = np.mgrid[-4:5, -4:5]
    kernel = np.exp(-.5*(x*x+y*y)/args.fit_blur**2)
    truth_kernel = np.exp(-.5*(x*x+y*y)/args.truth_blur**2)
    truth = np.array([args.truth_vref, .22, .9])
    truth_grid = BilinearLineMap(2, args.truth_grid, half_extent=.63, device=device, dtype=dtype)
    truth_model = NativeLineProjector.from_target(
        target, truth_grid.east.cpu().numpy(), truth_grid.north.cpu().numpy(), truth_kernel,
        np.ones(n), spectral_samples=7, device=device, dtype=dtype, deposition=args.deposition)
    east, north = truth_grid.east, truth_grid.north
    with torch.no_grad():
        morphology = torch.exp(-.5*((east/.23)**2+(north/.14)**2))
        if args.morphology == "clumpy":
            morphology += .6*torch.exp(-.5*(((east-.24)/.07)**2+((north+.12)/.08)**2))
        elif args.morphology == "two_clumps":
            morphology = (.15*morphology
                          + torch.exp(-.5*(((east-.18)/.08)**2+((north+.09)/.07)**2))
                          + .8*torch.exp(-.5*(((east+.2)/.07)**2+((north-.07)/.08)**2)))
        flux = morphology*truth_grid.area
        flux /= flux.sum()
        velocity = projected_arctan_velocity(east, north, *truth, args.truth_inclination)
        unit_image = truth_model(flux, velocity, sigma_v=45.)[0].double()
        # S/N after removing the continuum subspace, across all exposures.
        template_score = StripContinuumLikelihood(np.zeros_like(error), error, valid, pupils,
                                                   degree=args.degree, device=device)
        amplitude = args.snr / np.sqrt(float(template_score.chi_square(unit_image)))
        true_line = unit_image.cpu().numpy()*amplitude
    rng = np.random.default_rng(args.seed)
    yy, xx = np.meshgrid(np.linspace(-1, 1, args.size), np.linspace(-1, 1, args.size), indexing="ij")
    continuum = np.zeros_like(error)
    for e, pupil in enumerate(pupils):
        along, across = (xx, yy) if pupil == "R" else (yy, xx)
        shape = .7 + 4*np.exp(-.5*(across/.2)**2)
        continuum[e] = np.median(error[e][valid[e]]) * shape * (
            1 + .3*along + .4*(1.5*along**2-.5))
    data = true_line + continuum + args.noise_scale*rng.normal(size=error.shape)*error
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "injection.npz", science=data, error=error, valid=valid,
                        true_line=true_line, true_continuum=continuum,
                        truth_grid_flux=(flux*amplitude).cpu().numpy(),
                        truth_east=east.cpu().numpy(), truth_north=north.cpu().numpy())
    report = {
        "not_science_measurements": True, "source_geometry": args.source,
        "truth_vref_rturn_theta": truth.tolist(), "radius_ref_arcsec": .5,
        "fixed_inclination_rad": args.fit_inclination, "fixed_sigma_v_kms": 45.,
        "truth_inclination_rad": args.truth_inclination, "morphology": args.morphology,
        "truth_blur_sigma_pixels": args.truth_blur, "fit_blur_sigma_pixels": args.fit_blur,
        "generator": "analytic brightness on finer quadrature; response/inclination may differ as recorded",
        "truth_quadrature": args.truth_grid, "fit_quadrature": args.quadrature,
        "continuum_injection_degree": 2, "continuum_fit_degree": args.degree,
        "noise_scale": args.noise_scale, "seed": args.seed, "snr_after_continuum": args.snr,
        "fixed_map_flux_unit": amplitude, "device": device, "precision": args.precision,
        "likelihood_precision": "float64", "torch_version": torch.__version__,
        "deposition": args.deposition,
        "response_note": "deposition basis convolved with a fixed sampled Gaussian kernel; not a calibrated empirical PSF/LSF",
        "device_name": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "train_exposures": [str(target.exposures[i].path) for i in train],
        "held_exposures": [str(target.exposures[i].path) for i in held],
        "training_grisms": grisms,
        "unused_exposures": [str(target.exposures[i].path) for i in range(n) if i not in train and i not in held],
        "snr_normalization_note": "All exposures, unchanged by training-grism selection",
        "held_score_note": "line prediction held fixed; polynomial continuum refit on held-out exposures",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "code_sha256": {name: hashlib.sha256((REPO/name).read_bytes()).hexdigest() for name in
                        ("dingo/exposures.py", "dingo/native_forward.py", "dingo/native_fitting.py",
                         "diagnostics/native_flexible_check.py", "diagnostics/native_recovery_suite.py")},
        "fits": [], "best_by_setting": [], "reference_fits": [], "comparisons": [],
    }
    starts = [[40., .08, .2], [120., .35, 1.7], [300., .6, 3.3],
              [600., .12, 5.6], [1000., 1., 2.1], [1300., .3, -.7]][:args.starts]
    for grid_size in args.grids:
        line_map = BilinearLineMap(grid_size, args.quadrature, flux_unit=amplitude, device=device, dtype=dtype)
        projector = NativeLineProjector.from_target(
            target, line_map.east.cpu().numpy(), line_map.north.cpu().numpy(), kernel,
            np.ones(n), spectral_samples=7, device=device, dtype=dtype, deposition=args.deposition)
        fitter = NativeMapFitter(projector, line_map, data, error, valid, pupils,
                                 train_indices=train, held_indices=held, continuum_degree=args.degree,
                                 inclination=args.fit_inclination)
        axis = np.linspace(-.63, .63, grid_size)
        sy, sx = np.meshgrid(axis, axis, indexing="ij")
        initial_maps = [np.ones((grid_size, grid_size)),
                        .1+np.exp(-.5*((sx/.35)**2+(sy/.25)**2)),
                        .05+np.exp(-.5*(((sx+.2)/.15)**2+((sy-.15)/.18)**2))]
        for strength in args.strengths:
            setting = []
            for i, start in enumerate(starts):
                initial = initial_maps[i % len(initial_maps)]
                with torch.no_grad():
                    unit = float(line_map(torch.as_tensor(initial, device=device, dtype=dtype)).sum())
                initial = initial*amplitude/unit
                t0 = time.perf_counter()
                def progress(count, score):
                    print(f"grid={grid_size} lambda={strength:g} start={i} eval={count} objective={score:.6f}", flush=True)
                fitted = fitter.fit(start, initial_map=initial, strength=strength, maxiter=args.maxiter,
                                    gtol=args.gtol, callback=progress)
                entry = fitted["report"] | {"grid": grid_size, "start_index": i,
                                            "initial_map": i % len(initial_maps), "seconds": time.perf_counter()-t0}
                tag = f"grid{grid_size}_lambda{strength:g}_start{i}"
                np.savez_compressed(args.output / f"{tag}.npz", vector=fitted["vector"], controls=fitted["controls"],
                                    line=fitted["line"], continuum=fitted["continuum"], report_json=json.dumps(entry))
                entry["artifact"] = f"{tag}.npz"
                report["fits"].append(entry); setting.append(entry)
                (args.output / "recovery.json").write_text(json.dumps(report, indent=2))
                print(json.dumps(entry), flush=True)
            best = min(setting, key=lambda e: e["objective"])
            report["best_by_setting"].append(best)
            for entry in setting:
                entry["delta_objective_within_setting"] = entry["objective"] - best["objective"]
            (args.output / "recovery.json").write_text(json.dumps(report, indent=2))
            if args.profile_reference or getattr(args, "profile_speeds", None):
                # At zero rotation the angle and turnover radius do not exist
                # as identifiable parameters. Remove them from the null fit.
                references = []
                if args.profile_reference:
                    references.append(("null", {"v_ref": 0., "r_turn": .22, "theta": .9}))
                if args.profile_reference and args.truth_vref != 0:
                    references.append(("truth_speed", {"v_ref": args.truth_vref}))
                for speed in dict.fromkeys(getattr(args, "profile_speeds", None) or []):
                    if any(fixed["v_ref"] == speed for _, fixed in references):
                        continue
                    fixed = {"v_ref": speed}
                    if speed == 0:
                        fixed.update(r_turn=.22, theta=.9)
                    references.append((f"speed_{speed:g}", fixed))
                with np.load(args.output / best["artifact"]) as state:
                    warm_map = state["controls"].copy()
                for label, fixed in references:
                    # Two starts avoid treating a trapped constrained fit as
                    # evidence against the injected value. Neither map is truth.
                    ref_results = []
                    for j, (ref_start, ref_map) in enumerate([
                            (best["fitted_vref_rturn_theta"], warm_map),
                            ([120., .35, 1.7], initial_maps[1])]):
                        t0 = time.perf_counter()
                        fitted = fitter.fit(ref_start, initial_map=ref_map, strength=strength,
                                            fixed=fixed, maxiter=args.maxiter, gtol=args.gtol)
                        entry = fitted["report"] | {"grid": grid_size, "reference": label,
                            "reference_start": j, "seconds": time.perf_counter()-t0}
                        tag = f"grid{grid_size}_lambda{strength:g}_{label}_{j}.npz"
                        np.savez_compressed(args.output / tag, vector=fitted["vector"],
                            controls=fitted["controls"], line=fitted["line"], continuum=fitted["continuum"],
                            report_json=json.dumps(entry))
                        entry["artifact"] = tag
                        ref_results.append(entry); report["reference_fits"].append(entry)
                    ref_best = min(ref_results, key=lambda e: e["objective"])
                    # A nested constraint cannot beat the unconstrained global
                    # optimum. If it does, start the free fit from that state.
                    if ref_best["objective"] < best["objective"] - 1e-5:
                        with np.load(args.output / ref_best["artifact"]) as state:
                            ref_map = state["controls"].copy()
                        t0 = time.perf_counter()
                        rescued = fitter.fit(ref_best["fitted_vref_rturn_theta"], ref_map,
                                              strength=strength, maxiter=args.maxiter, gtol=args.gtol)
                        entry = rescued["report"] | {"grid": grid_size, "start_index": -1,
                            "initial_map": f"release_{label}", "seconds": time.perf_counter()-t0}
                        tag = f"grid{grid_size}_lambda{strength:g}_release_{label}.npz"
                        np.savez_compressed(args.output / tag, vector=rescued["vector"],
                            controls=rescued["controls"], line=rescued["line"],
                            continuum=rescued["continuum"], report_json=json.dumps(entry))
                        entry["artifact"] = tag
                        report["fits"].append(entry); setting.append(entry)
                        best = min(setting, key=lambda e: e["objective"])
                        report["best_by_setting"][-1] = best
                    report["comparisons"].append({"grid": grid_size, "strength": strength,
                        "reference": label, "reference_best": ref_best})
                    (args.output / "recovery.json").write_text(json.dumps(report, indent=2))
                # A later rescue can change all comparisons for this setting.
                for entry in setting:
                    entry["delta_objective_within_setting"] = entry["objective"] - best["objective"]
                for comparison in report["comparisons"]:
                    if comparison["grid"] == grid_size and comparison["strength"] == strength:
                        reference = comparison["reference_best"]
                        delta = 2*(reference["objective"] - best["objective"])
                        comparison.update(twice_objective_difference=delta,
                            valid_optimization=bool(best["stationary"] and reference["stationary"] and delta >= -1e-4),
                            likelihood_ratio=bool(strength == 0),
                            free_best_stationary=best["stationary"])
                (args.output / "recovery.json").write_text(json.dumps(report, indent=2))
    plot_results(report, args.output, data, error, valid, pupils)
    report["completed"] = True
    (args.output / "recovery.json").write_text(json.dumps(report, indent=2))
    return report


def plot_results(report, output, data, error, valid, pupils):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    radii = np.linspace(.02, .6, 100)
    truth = report["truth_vref_rturn_theta"]
    axes[1].plot(radii, truth[0]*np.arctan(radii/truth[1])/np.arctan(.5/truth[1]), "k--", label="Injected")
    for best in report["best_by_setting"]:
        matches = [r for r in report["fits"] if r["grid"] == best["grid"] and r["strength"] == best["strength"]]
        label = f"{best['grid']}x{best['grid']}, lambda={best['strength']:g}"
        line, = axes[0].plot([r["start"][0] for r in matches], [r["fitted_vref_rturn_theta"][0] for r in matches], "o", label=label)
        v, rt, _ = best["fitted_vref_rturn_theta"]
        axes[1].plot(radii, v*np.arctan(radii/rt)/np.arctan(.5/rt), color=line.get_color(), label=label)
        axes[2].plot([r["start"][0] for r in matches], [r["held_chi_square"] for r in matches], "o", color=line.get_color())
    axes[0].axhline(truth[0], color="k", ls="--")
    axes[0].set(xlabel="Initial Vref [km/s]", ylabel="Fitted Vref [km/s]")
    axes[1].set(xlabel="Radius [arcsec]", ylabel="Projected rotation [km/s]")
    axes[2].set(xlabel="Initial Vref [km/s]", ylabel="Held-out chi-square (continuum re-fit)")
    axes[0].legend(fontsize=7); axes[1].legend(fontsize=7)
    fig.suptitle(f"Synthetic free-map recovery on ID{report['source_geometry']} geometry; inclination and blur fixed")
    fig.tight_layout(); fig.savefig(output / "recovery.png", dpi=160); plt.close(fig)
    for best in report["best_by_setting"]:
        with np.load(output / best["artifact"]) as fit:
            fig, axes = plt.subplots(2, 3, figsize=(10, 7))
            for j, pupil in enumerate(("R", "C")):
                e = pupils.index(pupil)
                for ax, arr, title in zip(axes[j], [data[e], fit["line"][e]+fit["continuum"][e],
                                                     data[e]-fit["line"][e]-fit["continuum"][e]],
                                          ["Synthetic SCI", "Line + fitted continuum", "Residual"]):
                    ax.imshow(np.where(valid[e], arr/error[e], np.nan), origin="lower", cmap="RdBu_r", vmin=-4, vmax=4)
                    ax.set_title(f"{pupil}: {title}"); ax.set_axis_off()
            fig.tight_layout()
            fig.savefig(output / f"best_grid{best['grid']}_lambda{best['strength']:g}.png", dpi=140); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source", type=int, default=3700)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    p.add_argument("--precision", choices=["float32", "float64"], default="float64")
    p.add_argument("--deposition", choices=["bilinear", "cubic"], default="cubic")
    p.add_argument("--threads", type=int, default=4); p.add_argument("--size", type=int, default=81)
    p.add_argument("--grids", type=int, nargs="+", default=[13, 17])
    p.add_argument("--quadrature", type=int, default=49); p.add_argument("--truth-grid", type=int, default=97)
    p.add_argument("--strengths", type=float, nargs="+", default=[0., .03, .3])
    p.add_argument("--degree", type=int, default=2); p.add_argument("--starts", type=int, default=6)
    p.add_argument("--maxiter", type=int, default=1000); p.add_argument("--gtol", type=float, default=1e-3)
    p.add_argument("--snr", type=float, default=100.); p.add_argument("--noise-scale", type=float, default=0.)
    p.add_argument("--seed", type=int, default=481)
    p.add_argument("--truth-vref", type=float, default=210.)
    p.add_argument("--truth-inclination", type=float, default=.9)
    p.add_argument("--fit-inclination", type=float, default=.9)
    p.add_argument("--truth-blur", type=float, default=1.)
    p.add_argument("--fit-blur", type=float, default=1.)
    p.add_argument("--morphology", choices=["clumpy", "smooth", "two_clumps"], default="clumpy")
    p.add_argument("--profile-reference", action="store_true")
    p.add_argument("--grisms", choices=["R", "C", "RC"], default="RC",
                   help="Training grisms only; injection normalization and six held-out exposures stay fixed")
    p.add_argument("--profile-speeds", type=float, nargs="+",
                   help="Fix each projected reference speed and re-fit all other parameters from two starts")
    p.add_argument("--seeds", type=int, nargs="+", help="Run a suite with independent noise seeds")
    p.add_argument("--scenarios", nargs="+", default=["matched", "null"],
                   choices=["matched", "null", "smooth", "two_clumps", "inclination", "blur"])
    args = p.parse_args()
    if not 1 <= args.starts <= 6 or args.snr <= 0 or args.noise_scale < 0:
        p.error("Require 1--6 starts, positive S/N, nonnegative noise scale")
    if (not 0 < args.truth_inclination < np.pi/2 or not 0 < args.fit_inclination < np.pi/2
            or min(args.truth_blur, args.fit_blur) <= 0 or not 0 <= args.truth_vref <= 1500):
        p.error("Require physical inclinations, positive blur, and 0--1500 km/s injected speed")
    if args.profile_speeds and any(not 0 <= v <= 1500 for v in args.profile_speeds):
        p.error("Profile speeds must be between 0 and 1500 km/s")
    if args.seeds:
        from diagnostics.native_recovery_suite import run_suite
        run_suite(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
