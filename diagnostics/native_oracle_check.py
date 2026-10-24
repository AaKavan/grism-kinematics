"""Diagnostic joint R+C posterior with the injected emission shape known.

This is an oracle control, never a fit available for a real galaxy. The exact
97x97 injection quadrature is used, avoiding the flexible fit's map-grid
approximation. Total flux, speed, turnover, angle and inclination are sampled
with the baseline priors. It isolates the cost of learning an unknown map,
but does not separately identify morphology-prior bias versus marginalization.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

# Import the driver first to use its float64, GPU compiler and cache settings.
from diagnostics.native_posterior_check import REPO, summarize
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.util import unconstrain_fn
import torch

from dingo.exposures import load_native_target
from dingo.native_forward import NativeLineProjector, projected_arctan_velocity
from dingo.native_fitting import BilinearLineMap, StripContinuumLikelihood
from dingo.native_jax import JaxNativeProjector, JaxStripLikelihood, projected_velocity
from dingo.native_posterior import PosteriorPrior


class KnownMapPosterior:
    def __init__(self, projector, likelihood, east, north, flux, prior):
        if prior.turnover_prior != "uniform" or prior.inclination_error_deg is None:
            raise ValueError("This oracle control matches the baseline uniform-turnover / measured-inclination prior")
        self.projector, self.likelihood, self.prior = projector, likelihood, prior
        self.east, self.north, self.flux = map(jnp.asarray, (east, north, flux))

    def predict(self, v_ref, r_turn, theta, inclination, log_flux):
        velocity = projected_velocity(self.east, self.north, v_ref, r_turn, theta, inclination)
        return self.projector(self.flux*jnp.exp(log_flux), velocity, sigma_v=45.)

    def model(self, likelihood_weight=1.):
        p = self.prior
        v = numpyro.sample("v_ref", dist.Uniform(0., p.v_max))
        rt = numpyro.sample("r_turn", dist.Uniform(p.turnover_min, p.turnover_max))
        theta = numpyro.sample("theta", dist.Uniform(-np.pi, np.pi))
        inc = numpyro.sample("inclination", dist.TruncatedNormal(p.inclination_measurement_rad,
            np.deg2rad(p.inclination_error_deg), low=np.deg2rad(p.inclination_min_deg),
            high=np.deg2rad(p.inclination_max_deg)))
        log_flux = numpyro.sample("log_flux", dist.Normal(0., p.log_flux_sd))
        if likelihood_weight != 0:
            numpyro.factor("native_pixels", likelihood_weight*self.likelihood.relative(
                self.predict(v, rt, theta, inc, log_flux)))


def run(args):
    if (args.output/"posterior.json").exists():
        raise FileExistsError("Choose a fresh oracle output directory")
    if jax.default_backend() != "gpu":
        raise RuntimeError("This diagnostic requests the GPU")
    torch.set_num_threads(4)
    jax.config.update("jax_compilation_cache_dir", str(REPO.parent/"results/_jax_cache"))
    jax.config.update("jax_persistent_cache_enable_xla_caches", "all")
    baseline = json.loads((args.baseline/"posterior.json").read_text())
    source_dir = Path(baseline["source_run"])
    source = json.loads((source_dir/"recovery.json").read_text())
    settings = source["arguments"]
    if settings["fit_blur"] != settings["truth_blur"]:
        raise ValueError("Oracle control requires the matched response experiment")
    target = load_native_target(settings["bundle"], REPO/f"data/ID{source['source_geometry']}/config_kinematics.yaml", settings["size"])
    paths = [str(e.path) for e in target.exposures]
    selected = [paths.index(p) for p in baseline["train_exposures"]]
    with np.load(source_dir/"injection.npz") as a:
        data = a["true_line"]+a["true_continuum"] if args.noiseless else a["science"]
        data, error, valid = data[selected], a["error"][selected], a["valid"][selected]
        east, north, flux = (a[k] for k in ("truth_east", "truth_north", "truth_grid_flux"))
        injected = a["true_line"][selected]
    if args.interpolated_map:
        truth_size, control_size = settings["truth_grid"], source["best_by_setting"][0]["grid"]
        stride = (truth_size-1)//(control_size-1)
        if stride*(control_size-1) != truth_size-1:
            raise ValueError("Saved truth grid must contain the control nodes")
        truth_grid = BilinearLineMap(2, truth_size, dtype=torch.float64)
        brightness = (flux/truth_grid.area.numpy()).reshape(truth_size, truth_size)
        grid = BilinearLineMap(control_size, settings["quadrature"], flux_unit=1., dtype=torch.float64)
        particles = grid(torch.as_tensor(brightness[::stride, ::stride].copy()))[0].numpy()
        particles *= flux.sum()/particles.sum()
        east, north, flux = grid.east.numpy(), grid.north.numpy(), particles
    yy, xx = np.mgrid[-4:5, -4:5]
    kernel = np.exp(-.5*(xx*xx+yy*yy)/settings["fit_blur"]**2)
    full = NativeLineProjector.from_target(target, east, north, kernel, np.ones(len(paths)),
        spectral_samples=7, dtype=torch.float64, deposition=settings["deposition"])
    operator = NativeLineProjector(full.coefficients.numpy()[selected], full.wavelength_um, full.image_shape,
        full.kernels.numpy()[selected], full.response.numpy()[selected], spectral_samples=7, deposition=settings["deposition"])
    likelihood = StripContinuumLikelihood(data, error, valid, [target.exposures[i].pupil for i in selected], degree=settings["degree"])
    posterior = KnownMapPosterior(JaxNativeProjector(operator), JaxStripLikelihood(likelihood),
        east, north, flux, PosteriorPrior(**baseline["prior"]))
    truth = source["truth_vref_rturn_theta"]
    truth_inc = source["truth_inclination_rad"]
    reproduced = np.asarray(posterior.predict(*truth, truth_inc, 0.))
    mismatch = float(np.max(np.abs((reproduced-injected)[valid]/error[valid])))
    if not args.interpolated_map and mismatch > 1e-9:
        raise AssertionError(f"Oracle operator fails to reproduce the saved injection: {mismatch}")
    with torch.no_grad():
        velocity = projected_arctan_velocity(torch.as_tensor(east), torch.as_tensor(north), *truth, truth_inc)
        tline = operator(torch.as_tensor(flux), velocity, sigma_v=45.)[0].numpy()
    backend_mismatch = float(np.max(np.abs((reproduced-tline)[valid]/error[valid])))
    if backend_mismatch > 1e-9:
        raise AssertionError("Known-map JAX and Torch projections disagree")
    rng = np.random.default_rng(args.seed)
    starts, initial = [], []
    for c in range(4):
        high = c % 2 == 1
        q = {"v_ref": (500. if high else 210.)*np.exp(rng.normal(0., .01)),
             "r_turn": 1.999 if high else .2, "theta": .9+rng.normal(0., .01), "inclination": .9, "log_flux": 0.}
        starts.append({"chain": c, "branch": "high" if high else "low", "v_ref": q["v_ref"], "r_turn": q["r_turn"]})
        initial.append(unconstrain_fn(posterior.model, (), {}, q))
    initial = jax.tree.map(lambda *x: jnp.stack(x), *initial)
    report = {"completed": False, "synthetic_not_science_measurements": True, "oracle_known_map": True,
        "noiseless": args.noiseless, "interpolated_map": args.interpolated_map,
        "source_run": str(source_dir), "baseline": str(args.baseline),
        "source_geometry": source["source_geometry"], "train_exposures": baseline["train_exposures"],
        "truth_vref_rturn_theta": truth, "truth_inclination_rad": truth_inc, "distance": baseline["distance"],
        "starts": starts, "prior": baseline["prior"], "map_priors_used": False,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "backend_audit_max_pixel_difference_in_error_units": backend_mismatch,
        "representation_max_pixel_difference_in_error_units": mismatch,
        "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            args.baseline/"posterior.json", source_dir/"recovery.json", source_dir/"injection.npz")},
        "code_sha256": {name: hashlib.sha256((REPO/name).read_bytes()).hexdigest() for name in (
            "diagnostics/native_oracle_check.py", "diagnostics/native_posterior_check.py", "dingo/native_jax.py",
            "dingo/native_forward.py", "dingo/native_fitting.py", "dingo/native_posterior.py")},
        "versions": {"jax": jax.__version__, "numpyro": numpyro.__version__, "torch": torch.__version__},
        "limitations": ["Injected brightness shape is known: unavailable in real data",
                        "Known shape uses either exact injection particles or sampled controls, as recorded",
                        "Fixed response, width, center and registration; same kinematic priors as baseline",
                        "One synthetic realization, not a coverage study"]}
    args.output.mkdir(parents=True)
    for name in report["code_sha256"]:
        dst = args.output/"code_snapshot"/name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((REPO/name).read_bytes())
    (args.output/"posterior.json").write_text(json.dumps(report, indent=2))
    prior = Predictive(posterior.model, num_samples=512)(jax.random.key(args.seed+17), likelihood_weight=0.)
    np.savez_compressed(args.output/"prior_samples.npz", **{k: np.asarray(v) for k, v in prior.items()})
    kernel = NUTS(posterior.model, dense_mass=True, max_tree_depth=args.max_tree_depth, target_accept_prob=.9)
    mcmc = MCMC(kernel, num_warmup=args.warmup, num_samples=250, num_chains=4, chain_method="vectorized", progress_bar=False)
    fields = ("potential_energy", "energy", "num_steps", "accept_prob", "diverging")
    print(f"Oracle warmup: {args.warmup} iterations on four chains", flush=True)
    t0 = time.perf_counter()
    mcmc.warmup(jax.random.key(args.seed+1), init_params=initial, extra_fields=fields, collect_warmup=True)
    report["warmup_seconds"] = time.perf_counter()-t0
    report["warmup_completed"] = True
    np.savez_compressed(args.output/"warmup.npz", **{k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()})
    samples, extra = {}, {}
    for chunk in range(args.samples//250):
        print(f"Oracle retaining block {chunk+1} of {args.samples//250}", flush=True)
        t0 = time.perf_counter()
        mcmc.run(mcmc.post_warmup_state.rng_key, init_params=mcmc.post_warmup_state.z, extra_fields=fields)
        current = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
        current_extra = {k: np.asarray(v) for k, v in mcmc.get_extra_fields(group_by_chain=True).items()}
        samples = {k: np.concatenate([samples[k], v], axis=1) if k in samples else v for k, v in current.items()}
        extra = {k: np.concatenate([extra[k], v], axis=1) if k in extra else v for k, v in current_extra.items()}
        mcmc.post_warmup_state = mcmc.last_state
        report.setdefault("sampling_block_seconds", []).append(time.perf_counter()-t0)
        summarize(samples, extra, report, args.output, report["distance"]["kpc_per_arcsec"])
        print(json.dumps({k: v for k, v in report["diagnostics"].items() if k != "parameters"}), flush=True)
    report["completed"] = True
    (args.output/"posterior.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--noiseless", action="store_true")
    p.add_argument("--interpolated-map", action="store_true", help="Represent the known shape on the flexible fit's control/quadrature grid")
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--max-tree-depth", type=int, default=10)
    p.add_argument("--seed", type=int, default=741)
    args = p.parse_args()
    if args.samples < 250 or args.samples % 250 or args.warmup < 1:
        p.error("Require positive warmup and whole blocks of 250 retained draws")
    run(args)
