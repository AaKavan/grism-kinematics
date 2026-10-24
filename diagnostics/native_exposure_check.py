"""Prepare native cutouts and run a conditional injection/recovery diagnostic.

Run from the repository: python -m diagnostics.native_exposure_check --help
The validation uses synthetic data with real exposure geometry/masks/ERR. Its
line morphology, inclination, center, registration, width and PSF are known.
It cannot validate science posteriors or real-galaxy rotation measurements.
"""

import argparse
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import torch

from dingo.exposures import load_native_target
from dingo.native_forward import NativeLineProjector, NativeGaussianLikelihood, projected_arctan_velocity

REPO = Path(__file__).resolve().parents[1]


def prepare(args):
    records = []
    for sid in args.ids:
        target = load_native_target(args.bundle, REPO / f"data/ID{sid}/config_kinematics.yaml", args.size)
        target.save(args.output / f"ID{sid}_native.npz")
        offsets = np.array([e.metadata["wcs_minus_stats_xy"] for e in target.exposures])
        record = {"source_id": sid, "wavelength_um": target.wavelength_um,
                  "exposure_counts": {p: sum(e.pupil == p for e in target.exposures) for p in ("R", "C")},
                  "valid_fraction_range": [float(f([e.valid.mean() for e in target.exposures])) for f in (min, max)],
                  "wcs_minus_stats_max_norm_pixels": float(np.linalg.norm(offsets, axis=1).max()),
                  "excluded_exposures": target.excluded_exposures,
                  "exposures": [e.metadata for e in target.exposures]}
        records.append(record)
        for layer, label in [("science", "SCI / ERR"), ("emline", "EMLINE / SCI ERR (display only)")]:
            ncol = 6; nrow = int(np.ceil(len(target.exposures)/ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(15, 2.7*nrow), squeeze=False)
            for ax in axes.flat:
                ax.set_axis_off()
            for ax, e in zip(axes.flat, target.exposures):
                values = np.where(e.valid, getattr(e, layer)/e.error, np.nan)
                ax.imshow(values, origin="lower", vmin=-3, vmax=8, cmap="RdBu_r")
                ax.plot(e.line_xy[0]-e.origin_xy[0], e.line_xy[1]-e.origin_xy[1], "+", color="lime", ms=8)
                ax.set_title(f"{e.module}{e.pupil} {e.path.name[7:12]} {e.path.name[20:25]}\n{e.valid.mean():.1%} valid", fontsize=8)
            fig.suptitle(f"ID{sid}: native {label}; + is predicted systemic line center")
            fig.tight_layout()
            fig.savefig(args.output / f"ID{sid}_{layer}.png", dpi=140)
            plt.close(fig)
        print(json.dumps({k: v for k, v in record.items() if k not in ("exposures", "excluded_exposures")}
                         | {"excluded_count": len(target.excluded_exposures)}), flush=True)
    (args.output / "native_inventory.json").write_text(json.dumps(records, indent=2))


def validate(args):
    torch.set_num_threads(args.threads)
    dtype = torch.float64 if args.precision == "float64" else torch.float32
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not accessible in this process; run CPU or provide GPU access")
    target = load_native_target(args.bundle, REPO / f"data/ID{args.source}/config_kinematics.yaml", args.size)
    # Source-node integrated flux. Deliberately asymmetric, with a distinct clump.
    axis = np.linspace(-.63, .63, 25)
    north, east = np.meshgrid(axis, axis, indexing="ij")
    morphology = (np.exp(-.5*((east/.23)**2 + (north/.14)**2))
                  + .6*np.exp(-.5*(((east-.24)/.07)**2 + ((north+.12)/.08)**2)))
    morphology /= morphology.sum()
    east_t, north_t = [torch.as_tensor(a.ravel(), device=device, dtype=dtype) for a in (east, north)]
    kernel_y, kernel_x = np.mgrid[-4:5, -4:5]
    # Controlled, known detector blur. Not the as-yet unvalidated empirical PSF/LSF.
    kernel = np.exp(-.5*(kernel_x**2+kernel_y**2))
    model = NativeLineProjector.from_target(target, east.ravel(), north.ravel(), kernel,
                                           np.ones(len(target.exposures)), device=device, dtype=dtype,
                                           spectral_samples=7)
    flux = torch.as_tensor(morphology.ravel(), device=device, dtype=dtype)
    valid = np.stack([e.valid for e in target.exposures])
    err = np.stack([e.error for e in target.exposures])
    # Keep likelihood arithmetic in double precision even for a float32 model.
    # Otherwise rounding ~10^4 in log L can stop a fit before its gradient is small.
    error_t = torch.as_tensor(np.where(valid, err, 1.), device=device, dtype=torch.float64)
    mask = torch.as_tensor(valid, device=device)
    invvar = mask / error_t.square()
    nweight = invvar.sum((-2, -1))
    truth = np.array([210., .22, .9])
    def prediction(q):
        velocity = projected_arctan_velocity(east_t, north_t, q[..., 0]*200., q[..., 1]*.2,
                                             q[..., 2], .9, radius_ref=.5)
        return model(flux, velocity, sigma_v=45.)
    truth_q = torch.as_tensor(truth / [200., .2, 1.], device=device, dtype=dtype)
    with torch.no_grad():
        unit = prediction(truth_q)[0].double()
        amplitude = args.snr / torch.sqrt((unit.square()*invvar).sum())
        expected = amplitude * unit
    rng = np.random.default_rng(args.seed)
    background = torch.as_tensor(rng.normal(0, .3, (len(err), 1, 1)), dtype=torch.float64, device=device) * error_t.median()
    noise = torch.as_tensor(rng.normal(size=err.shape), dtype=torch.float64, device=device) * error_t
    synthetic = expected + background + args.noise_scale*noise
    likelihood = NativeGaussianLikelihood(synthetic, error_t, mask, device=device, dtype=torch.float64)
    data_mean = (likelihood.data * invvar).sum((-2, -1)) / nweight
    centered_data = likelihood.data - data_mean[:, None, None]

    def profiled_model(q):
        template = prediction(q)[0].double()
        mean = (template*invvar).sum((-2, -1)) / nweight
        centered = template - mean[:, None, None]
        # Exact profile over common positive line flux and independent constant
        # backgrounds, under the same Gaussian likelihood used for scoring.
        a = ((centered*centered_data*invvar).sum() / (centered.square()*invvar).sum()).clamp_min(0)
        b = data_mean - a*mean
        return a*template + b[:, None, None], a, b

    def objective(q_np):
        q = torch.tensor(q_np, dtype=dtype, device=device, requires_grad=True)
        fitted_image, _, _ = profiled_model(q)
        nll = -likelihood.relative(fitted_image)
        grad, = torch.autograd.grad(nll, q)
        return float(nll.detach()), grad.detach().cpu().numpy().astype(float)

    starts = [[40, .08, .2], [120, .35, 1.7], [300, .6, 3.3],
              [600, .12, 5.6], [1000, 1., 2.1], [1300, .3, -.7]]
    records = []
    bounds = [(-7.5, 7.5), (.1, 10), (-4*np.pi, 4*np.pi)]
    for start in starts:
        t0 = time.perf_counter()
        result = minimize(objective, np.array(start)/[200, .2, 1], jac=True, method="L-BFGS-B",
                          bounds=bounds,
                          options={"maxiter": args.maxiter, "ftol": 1e-13,
                                   "gtol": 1e-6 if dtype == torch.float64 else 1e-3, "maxls": 40})
        fitted = result.x*np.array([200, .2, 1])
        if fitted[0] < 0:
            fitted[0] *= -1; fitted[2] += np.pi
        fitted[2] %= 2*np.pi
        with torch.no_grad():
            _, a, b = profiled_model(torch.as_tensor(result.x, dtype=dtype, device=device))
        record = {"start": start, "fitted": fitted.tolist(), "negative_relative_log_likelihood": float(result.fun),
                  "success": bool(result.success), "message": str(result.message),
                  "evaluations": int(result.nfev), "seconds": time.perf_counter()-t0,
                  "profiled_line_flux": float(a), "profiled_background_per_exposure": b.cpu().tolist(),
                  "boundary_hits": [name for name, value, (low, high) in zip(
                      ("signed_v_ref", "r_turn", "theta"), result.x, bounds)
                                    if min(abs(value-low), abs(value-high)) < 1e-5*(high-low)]}
        records.append(record)
        print(json.dumps({k: v for k, v in record.items() if k != "profiled_background_per_exposure"}), flush=True)
    best_index = min(range(len(records)), key=lambda i: records[i]["negative_relative_log_likelihood"])
    best_score = records[best_index]["negative_relative_log_likelihood"]
    for record in records:
        record["delta_negative_log_likelihood"] = record["negative_relative_log_likelihood"] - best_score
    # Repeat the same model at batch sizes used by samplers; report measured time.
    timings = []
    for batch in (1, 16, 64):
        q = truth_q[None].expand(batch, -1)
        with torch.no_grad():
            prediction(q)
            if device == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(3): prediction(q)
            if device == "cuda": torch.cuda.synchronize()
        timings.append({"batch": batch, "seconds_per_forward": (time.perf_counter()-t0)/3})
    # Compare the projection to an independent CPU float64 evaluation using
    # the same cached geometry, to separate hardware/dtype from fit convergence.
    reference = NativeLineProjector.from_target(target, east.ravel(), north.ravel(), kernel,
                                                np.ones(len(target.exposures)), device="cpu",
                                                dtype=torch.float64, spectral_samples=7)
    with torch.no_grad():
        velocity_ref = projected_arctan_velocity(east_t.double().cpu(), north_t.double().cpu(),
                                                 *truth, .9, radius_ref=.5)
        reference_image = reference(flux.double().cpu(), velocity_ref, sigma_v=45.)
        candidate = prediction(truth_q).double().cpu()
        parity = {"relative_l2": float(torch.linalg.vector_norm(candidate-reference_image)/torch.linalg.vector_norm(reference_image)),
                  "max_absolute": float((candidate-reference_image).abs().max())}
    report = {"source_geometry": args.source, "device": device, "precision": args.precision,
              "likelihood_precision": "float64", "cpu_float64_image_comparison": parity,
              "truth_vref_kms_rturn_arcsec_theta_rad": truth.tolist(), "inclination_fixed_rad": .9,
              "radius_ref_arcsec": .5, "seed": args.seed, "injected_matched_filter_snr": args.snr,
              "noise_scale": args.noise_scale,
              "validation_scope": "conditional numerical recovery only; known morphology, inclination, center, registration, PSF, width; synthetic noise; no contamination/continuum structure",
              "not_science_measurements": True, "starts": records, "timings": timings}
    report["best_start_index"] = best_index
    report["best_fit"] = records[best_index]
    suffix = "_noiseless" if args.noise_scale == 0 else ""
    outfile = args.output / f"ID{args.source}_conditional_{device}_{args.precision}{suffix}.json"
    outfile.write_text(json.dumps(report, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for i, (title, val) in enumerate([("V sin(i) at 0.5 arcsec [km/s]", truth[0]), ("Turnover radius [arcsec]", truth[1])]):
        axes[i].plot([r["start"][0] for r in records], [r["fitted"][i] for r in records], "o")
        axes[i].axhline(val, color="k", ls="--", label="Injected truth")
        axes[i].set(xlabel="Initial projected speed [km/s]", ylabel=title)
        axes[i].legend()
    fig.suptitle(f"Conditional synthetic recovery — ID{args.source} exposure geometry")
    fig.tight_layout(); fig.savefig(outfile.with_suffix(".png"), dpi=160); plt.close(fig)
    print(json.dumps({"report": str(outfile), "timings": timings}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "validate"])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=81)
    parser.add_argument("--ids", type=int, nargs="+", default=[3700, 6170, 8910, 15665, 16887, 21619, 21636])
    parser.add_argument("--source", type=int, default=3700)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--precision", choices=["float32", "float64"], default="float64")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--seed", type=int, default=481)
    parser.add_argument("--snr", type=float, default=100.)
    parser.add_argument("--noise-scale", type=float, default=1.)
    args = parser.parse_args()
    if args.snr <= 0 or args.noise_scale < 0:
        parser.error("--snr must be positive and --noise-scale nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    (prepare if args.action == "prepare" else validate)(args)


if __name__ == "__main__":
    main()
