"""PSF-convolved F150W PySersic fits for an external inclination constraint.

The axis-ratio posterior is conditional on a single Sersic model and quoted
pixel errors. It is not, by itself, a gas inclination posterior. Central-mask
variants expose structural sensitivity; disk thickness and stellar/gas
misalignment must be propagated separately before fitting the grism data.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

from diagnostics.native_posterior_check import diagnostics
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import initialize_model
from numpyro.infer.initialization import init_to_value
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize
from astropy.io import fits
from pysersic import FitSingle
from pysersic.priors import SourceProperties
from pysersic.loss import gaussian_loss
from pysersic.rendering import PixelRenderer


def resample_effective_psf(psf, input_scale, output_scale):
    """Recenter the empirical core and sample its effective PSF on the image grid.

    An ePSF already describes the observed pixel response. Sample the smooth
    empirical array, rather than applying a second top-hat integration. Small
    negative empirical wings are clipped, with their fraction recorded.
    """
    psf = np.asarray(psf, dtype=float)
    if (psf.ndim != 2 or not np.isfinite(psf).all()
            or not np.isfinite([input_scale, output_scale]).all() or min(input_scale, output_scale) <= 0):
        raise ValueError("Invalid PSF or pixel scale")
    positive = np.maximum(psf, 0.)
    if positive.sum() <= 0:
        raise ValueError("PSF must have positive mass")
    yy, xx = np.indices(psf.shape)
    peak = np.unravel_index(np.argmax(positive), psf.shape)
    core = positive*((xx-peak[1])**2+(yy-peak[0])**2 <= (0.12/input_scale)**2)
    cx, cy = (core*xx).sum()/core.sum(), (core*yy).sum()/core.sum()
    half = int(np.floor(min(cx, cy, psf.shape[1]-1-cx, psf.shape[0]-1-cy)*input_scale/output_scale))
    if half < 1:
        raise ValueError("Insufficient PSF support around the measured core")
    axis = np.arange(-half, half+1)*output_scale/input_scale
    y, x = np.meshgrid(axis+cy, axis+cx, indexing="ij")
    result = np.maximum(map_coordinates(positive, [y, x], order=3, mode="constant"), 0.)
    result /= result.sum()
    return result, {"input_scale_arcsec": input_scale, "output_scale_arcsec": output_scale,
        "core_centroid_xy": [float(cx), float(cy)], "original_sum": float(psf.sum()),
        "clipped_negative_fraction": float(-psf[psf < 0].sum()/positive.sum()),
        "output_shape": list(result.shape), "method": "centered cubic sampling of effective PSF, then unit normalization"}


def run(args):
    if (args.output/"imaging.json").exists():
        raise FileExistsError("Choose a fresh imaging output directory")
    if args.device == "cuda" and jax.default_backend() != "gpu":
        raise RuntimeError("GPU requested but unavailable")
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((args.inputs/"input_provenance.json").read_text())
    source_id = int(provenance["source_id"])
    with np.load(args.inputs/"cutout.npz") as a:
        image, error = a["image"], a["error"]
        scale, center = float(a["pixel_scale_arcsec"]), a["center_xy"]
    with fits.open(args.inputs/"F150W_ePSF_original.fits") as h:
        psf, psf_report = resample_effective_psf(h[0].data, h[0].header["PIXSCALE"], scale)
    yy, xx = np.indices(image.shape)
    radius = np.hypot(xx-center[0], yy-center[1])*scale
    mask = (~np.isfinite(image) | ~np.isfinite(error) | (error <= 0)
            | (radius > args.fit_radius) | (radius < args.mask_core))
    # A dimensionless common scaling changes neither the likelihood nor q.
    unit = float(np.nanmedian(error[~mask]))
    data = np.where(np.isfinite(image), image/unit, 0.)
    rms = np.where(np.isfinite(error) & (error > 0), error/unit, 1.)
    props = SourceProperties(data, mask=mask)
    prior = props.generate_prior("sersic", sky_type="flat")
    # SourceProperties' default flux error is based on sqrt(flux). Our units
    # are surface brightness / median ERR, not Poisson counts. Use a broad,
    # positive flux prior and the same full-aperture scale for both masks.
    aperture = (radius <= args.fit_radius) & np.isfinite(data)
    flux_scale = float(np.maximum(data[aperture],0.).sum())
    prior.set_uniform_prior("flux", 0., 5*flux_scale)
    prior.sky_prior.update_prior("sky_back", 0., 3.)
    prior.set_uniform_prior("ellip", .01, .95)
    # Put the angle cut away from this source's measured major axis, rather
    # than choosing an interval specifically for the horizontal ID3700 disk.
    weight = np.where((radius < .8) & ~mask, np.maximum(data, 0.), 0.)
    if weight.sum() <= 0:
        raise ValueError('No positive source signal in the F150W fitting aperture.')
    dx, dy = xx-center[0], yy-center[1]
    covariance = np.array([[(weight*dx*dx).sum(), (weight*dx*dy).sum()],
                           [(weight*dx*dy).sum(), (weight*dy*dy).sum()]])/weight.sum()
    axis = np.linalg.eigh(covariance)[1][:, -1]
    theta_center = float(np.arctan2(axis[0], axis[1]) % np.pi)
    theta_low, theta_high = theta_center-np.pi/2, theta_center+np.pi/2
    prior.set_uniform_prior("theta", theta_low, theta_high)
    # HybridRenderer interpolates a table defined only for n >= .65. The
    # core-masked galaxy prefers lower n; use direct pixel rendering instead.
    prior.set_uniform_prior("n", .3, 6.)
    prior.set_truncated_gaussian_prior("r_eff", 10., 15., low=1., high=45.)
    prior.set_gaussian_prior("xc", float(center[0]), 5.)
    prior.set_gaussian_prior("yc", float(center[1]), 5.)
    fitter = FitSingle(data=data, rms=rms, psf=psf, mask=mask,
                       prior=prior, loss_func=gaussian_loss, renderer=PixelRenderer)
    model = fitter.build_model(return_model=False)
    info = initialize_model(jax.random.key(args.seed), model, init_strategy=init_to_value(values={
        "theta_base": .5, "ellip_base": .65, "n_base": .12, "r_eff_base": .2,
        "flux_base": .2, "xc_base": .013, "yc_base": .017, "sky_back": 0.}))
    initial, unravel = ravel_pytree(info.param_info.z)
    objective = jax.jit(jax.value_and_grad(lambda x: info.potential_fn(unravel(x))))
    initialization = {"nonfinite_trials_rejected": 0, "best_value": np.inf, "best_vector": None}
    def fun(x):
        value, grad = objective(jnp.asarray(x))
        value, grad = float(value), np.asarray(grad)
        if not np.isfinite(value) or not np.isfinite(grad).all():
            initialization["nonfinite_trials_rejected"] += 1
            # Optimization-only barrier. NUTS uses the unmodified model below.
            return 1e100, np.zeros_like(x)
        if value < initialization["best_value"]:
            initialization.update(best_value=value, best_vector=np.array(x, copy=True))
        return value, grad
    # A bounded trust radius avoids invalid large rendering excursions being
    # mistaken for relative-function convergence by a line-search optimizer.
    hessian = jax.jit(jax.hessian(lambda x: info.potential_fn(unravel(x))))
    opt = minimize(fun, np.asarray(initial), jac=True, hess=lambda x: np.asarray(hessian(jnp.asarray(x))),
                   method="trust-exact", options={"maxiter": 500, "gtol": 1e-4,
                       "initial_trust_radius": .1, "max_trust_radius": 1.})
    if initialization["best_vector"] is None:
        raise RuntimeError("No finite imaging initialization")
    opt.x = initialization.pop("best_vector")
    opt.fun, opt.jac = fun(opt.x)
    initialization.pop("best_vector", None)
    if abs(opt.jac).max() > .1:
        raise RuntimeError(f"Imaging initialization not stationary: gradient={abs(opt.jac).max()}")
    print(json.dumps({"initialization_objective": float(opt.fun), "success": bool(opt.success),
                      "gradient_inf": float(abs(opt.jac).max())}), flush=True)
    rng = np.random.default_rng(args.seed)
    starts = [unravel(jnp.asarray(opt.x+rng.normal(0, .01, len(opt.x)))) for _ in range(4)]
    initial = jax.tree.map(lambda *x: jnp.stack(x), *starts)
    report = {"completed": False, "source_id": source_id, "band": "F150W", "real_imaging": True,
        "theta_interval_rad": [theta_low, theta_high],
        "renderer": "PySersic PixelRenderer (direct Sersic, core Gauss-Legendre quadrature)",
        "initialization": initialization | {"optimizer_success": bool(opt.success),
            "message": str(opt.message), "gradient_inf": float(abs(opt.jac).max())},
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "psf": psf_report, "prior": repr(prior), "pixel_unit_scale": unit,
        "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                         (args.inputs/"cutout.npz", args.inputs/"F150W_ePSF_original.fits")},
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitations": ["Single Sersic description of a clumpy source", "Mosaic pixel covariance not modeled",
                        "Formal conditional statistical errors are not total inclination uncertainty",
                        "No gas/stellar alignment or intrinsic disk thickness inferred from this fit"]}
    (args.output/"imaging.json").write_text(json.dumps(report, indent=2))
    (args.output/"imaging_code.py").write_bytes(Path(__file__).read_bytes())
    # The Hessian executable is no longer needed; release compilation caches
    # before compiling the substantially larger vectorized NUTS transition.
    jax.clear_caches()
    sampler = MCMC(NUTS(model, dense_mass=True, target_accept_prob=.95, max_tree_depth=10),
        num_warmup=args.warmup, num_samples=args.samples, num_chains=4,
        chain_method="vectorized", progress_bar=False)
    fields = ("potential_energy", "energy", "num_steps", "accept_prob", "diverging")
    print(f"PySersic sampling: four chains, warmup {args.warmup}, draws {args.samples}", flush=True)
    t0 = time.perf_counter()
    sampler.run(jax.random.key(args.seed+1), init_params=initial, extra_fields=fields)
    saved = {k: np.asarray(v) for k, v in sampler.get_samples(group_by_chain=True).items()}
    extra = {k: np.asarray(v) for k, v in sampler.get_extra_fields(group_by_chain=True).items()}
    saved["q"] = 1-saved["ellip"]
    report["sampling_seconds"] = time.perf_counter()-t0
    report["diagnostics"] = diagnostics(saved, extra, 10)
    report["quantiles_5_50_95"] = {k: np.quantile(saved[k], [.05,.5,.95]).tolist()
                                  for k in ("q", "theta", "r_eff", "n", "xc", "yc")}
    report["angle_boundary_fraction"] = float(np.mean(
        np.minimum(saved["theta"]-theta_low, theta_high-saved["theta"]) < .01))
    report["orientation_parameterization_ok"] = report["angle_boundary_fraction"] < .01
    params = {k: jnp.asarray(np.median(v)) for k,v in saved.items()}
    rendered = np.asarray(fitter.renderer.render_source(params, "sersic"))*unit
    rendered += float(params["sky_back"])*unit
    residual = (image-rendered)/error
    report["residual_rms_inside_mask"] = float(np.sqrt(np.mean(residual[~mask]**2)))
    report["completed"] = True
    np.savez_compressed(args.output/"samples.npz", **saved)
    np.savez_compressed(args.output/"sampler_stats.npz", **extra)
    np.savez_compressed(args.output/"fit.npz", image=image, error=error, psf=psf, mask=mask, model=rendered, residual=residual)
    (args.output/"imaging.json").write_text(json.dumps(report, indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(12,4))
    for ax, values, title in zip(axes,[np.arcsinh(image/.05),np.arcsinh(rendered/.05),residual],
                                ["F150W data", "PySersic model", "(data-model)/ERR"]):
        ax.imshow(values,origin="lower",cmap="RdBu_r" if title.startswith("(") else "magma",
                  **({"vmin":-5,"vmax":5} if title.startswith("(") else {}))
        ax.contour(~mask,levels=[.5],colors="lime",linewidths=.6)
        ax.set_title(title)
    fig.suptitle(f"Real ID{source_id}; central exclusion {args.mask_core:.2f} arcsec; conditional morphology fit")
    fig.tight_layout(); fig.savefig(args.output/"fit.png",dpi=150); plt.close(fig)
    print(json.dumps({"output":str(args.output), "quantiles":report["quantiles_5_50_95"],
        "diagnostics":{k:v for k,v in report["diagnostics"].items() if k != "parameters"},
        "residual_rms":report["residual_rms_inside_mask"]}),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--mask-core",type=float,default=0.)
    p.add_argument("--fit-radius",type=float,default=1.2)
    p.add_argument("--warmup",type=int,default=500)
    p.add_argument("--samples",type=int,default=1000)
    p.add_argument("--seed",type=int,default=735)
    p.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    args=p.parse_args()
    if not 0 <= args.mask_core < args.fit_radius:
        p.error("Require 0 <= mask-core < fit-radius")
    run(args)
