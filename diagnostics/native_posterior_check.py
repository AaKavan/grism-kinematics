"""GPU NumPyro posterior pilot on the saved synthetic native-exposure dataset.

This samples the emission map, its hyperparameters, flux, velocity curve, angle
and inclination. Four independent chains start in the two previously found
kinematic branches. Unconverged chains never produce an approved pooled mass
interval. Instrument response, center, registration and width remain fixed.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# The vectorized float64 sampler triggered a stalled GEMM-fusion compilation
# on the local GPU. Keep the same arithmetic with the library GEMM backend.
# An explicitly supplied XLA_FLAGS remains under the caller's control.
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")

import numpy as np
import torch
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import initialize_model, unconstrain_fn

from dingo.exposures import load_native_target
from dingo.native_forward import NativeLineProjector, projected_arctan_velocity
from dingo.native_fitting import BilinearLineMap, StripContinuumLikelihood
from dingo.native_jax import JaxNativeProjector, JaxLineMap, JaxStripLikelihood
from dingo.native_posterior import NativePosterior, PosteriorPrior, radial_profiles

REPO = Path(__file__).resolve().parents[1]


def reduce_adapted_step_size(state, scale):
    """Change only fixed integration steps after adaptation, preserving target/state."""
    if not np.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("Adapted step-size scale must be in (0, 1]")
    return state._replace(adapt_state=state.adapt_state._replace(
        step_size=state.adapt_state.step_size*scale))


def diagnostics(samples, extra, max_depth):
    import arviz_stats as azs
    stats = {}
    for name, values in samples.items():
        if name in ("native_pixels", "external_inclination"):
            continue
        values = np.asarray(values)
        if values.shape[1] < 4:
            continue
        rh = np.asarray(azs.rhat(values, method="rank"))
        bulk = np.asarray(azs.ess(values, method="bulk"))
        tail = np.asarray(azs.ess(values, method="tail", prob=(.05, .95)))
        stats[name] = {"max_rank_rhat": float(np.max(rh)), "min_bulk_ess": float(np.min(bulk)),
                       "min_tail_ess": float(np.min(tail)), "rhat": rh.tolist(),
                       "bulk_ess": bulk.tolist(), "tail_ess": tail.tolist()}
    energies = extra["energy"]
    bfmi = (np.diff(energies, axis=1)**2).mean(axis=1)/np.var(energies, axis=1)
    max_rhat = max(v["max_rank_rhat"] for v in stats.values())
    min_bulk = min(v["min_bulk_ess"] for v in stats.values())
    min_tail = min(v["min_tail_ess"] for v in stats.values())
    divergences = int(np.sum(extra["diverging"]))
    depth_fraction = float(np.mean(extra["num_steps"] >= 2**max_depth-1))
    all_finite = bool(np.isfinite(bfmi).all() and all(np.isfinite([
        v["max_rank_rhat"], v["min_bulk_ess"], v["min_tail_ess"]]).all() for v in stats.values()))
    passed = bool(all_finite and max_rhat < 1.01 and min_bulk >= 400 and min_tail >= 400
                  and divergences == 0 and depth_fraction < .01 and np.min(bfmi) > .3)
    return {"parameters": stats, "max_rank_rhat": max_rhat, "min_bulk_ess": min_bulk,
            "min_tail_ess": min_tail, "divergences": divergences, "max_depth_fraction": depth_fraction,
            "bfmi_by_chain": bfmi.tolist(), "mean_acceptance": float(np.mean(extra["accept_prob"])),
            "mean_leapfrog_steps": float(np.mean(extra["num_steps"])), "convergence_gate_passed": passed,
            "all_diagnostics_finite": all_finite,
            "gate": "all saved parameters and radial profiles: rank Rhat<1.01, bulk/tail ESS>=400, no divergences, <1% depth saturation, BFMI>0.3",
            "gate_note": "Passing is necessary, not proof of correct mode weights or scientific coverage"}


def summarize(samples, extra, report, output, kpc_per_arcsec):
    radii = np.array([.05, .1, .15, .2, .3, .4, .5])
    profiles = radial_profiles(samples["v_ref"], samples["r_turn"], samples["inclination"], radii,
                               kpc_per_arcsec=kpc_per_arcsec)
    # Persist draws before any optional statistics/plot library is called.
    np.savez_compressed(output/"samples.npz", **samples)
    np.savez_compressed(output/"sampler_stats.npz", **extra)
    np.savez_compressed(output/"profiles.npz", radii_arcsec=radii, **profiles)
    profiles_to_check = {k: v for k, v in profiles.items() if k != "mass_proxy_arcsec_kms2"}
    checked = samples | profiles_to_check
    diag = diagnostics(checked, extra, report["arguments"]["max_tree_depth"])
    report["diagnostics"] = diag
    report["retained_draws_per_chain"] = samples["v_ref"].shape[1]
    report["radii_arcsec"] = radii.tolist()
    report["per_chain_quantiles_5_50_95"] = {
        name: np.quantile(value, [.05, .5, .95], axis=1).swapaxes(0, 1).tolist()
        for name, value in (samples | profiles).items() if name != "map_z"}
    report["pooled_quantiles_5_50_95"] = ({
        name: np.quantile(value.reshape(-1, *value.shape[2:]), [.05, .5, .95], axis=0).tolist()
        for name, value in (samples | profiles).items() if name != "map_z"
    } if diag["convergence_gate_passed"] else None)
    report["high_turnover_fraction_by_chain"] = (samples["r_turn"] > .8).mean(1).tolist()
    report["turnover_near_upper_bound_fraction_by_chain"] = (samples["r_turn"] > .95*report["prior"]["turnover_max"]).mean(1).tolist()
    (output/"posterior.json").write_text(json.dumps(report, indent=2))
    plot(samples, profiles, radii, report, output)


def plot(samples, profiles, radii, report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for c in range(len(samples["v_ref"])):
        color = f"C{c}"
        for ax, key in zip(axes[0], ("v_ref", "r_turn", "inclination")):
            ax.plot(samples[key][c], lw=.6, alpha=.7, color=color, label=f"chain {c}: {report['starts'][c]['branch']}")
            ax.set(xlabel="Post-warmup draw", ylabel=key)
        for ax, key in zip(axes[1, :2], ("deprojected_velocity_kms", "mass_proxy_arcsec_kms2")):
            q = np.quantile(profiles[key][c], [.05, .5, .95], axis=0)
            ax.plot(radii, q[1], color=color)
            ax.fill_between(radii, q[0], q[2], color=color, alpha=.12)
        axes[1, 2].scatter(samples["r_turn"][c, ::4], samples["v_ref"][c, ::4], s=3, alpha=.3, color=color)
    truth = report["truth_vref_rturn_theta"]
    true = radial_profiles(truth[0], truth[1], report["truth_inclination_rad"], radii)
    axes[1, 0].plot(radii, true["deprojected_velocity_kms"], "k--", label="Injected")
    axes[1, 1].plot(radii, true["mass_proxy_arcsec_kms2"], "k--")
    axes[1, 0].set(xlabel="Test radius [arcsec]", ylabel="Deprojected rotation [km/s]")
    axes[1, 1].set(xlabel="Test radius [arcsec]", ylabel="Rotation-only mass proxy [arcsec (km/s)^2]")
    axes[1, 2].set(xlabel="Turnover [arcsec]", ylabel="Projected Vref [km/s]")
    axes[0, 0].legend(fontsize=7)
    gate = report["diagnostics"]["convergence_gate_passed"]
    fig.suptitle(f"Synthetic joint R+C posterior; per-chain 5–95% bands; convergence gate {'PASS' if gate else 'FAIL'}")
    fig.tight_layout()
    fig.savefig(output/"posterior.png", dpi=170)
    plt.close(fig)
    warmup_path = Path(report.get("warmup_reused_from", output))/"warmup.npz"
    if warmup_path.exists():
        with np.load(warmup_path) as saved:
            warmup = {k: saved[k] for k in ("v_ref", "r_turn", "inclination")}
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for c in range(len(samples["v_ref"])):
            for ax, key in zip(axes, warmup):
                ax.plot(warmup[key][c], color=f"C{c}", lw=.7, alpha=.8,
                        label=f"chain {c}: {report['starts'][c]['branch']} start")
                ax.set(xlabel="Adaptation iteration (excluded from inference)", ylabel=key)
        axes[0].legend(fontsize=8)
        fig.suptitle("Synthetic R+C: movement from the two initial kinematic branches")
        fig.tight_layout()
        fig.savefig(output/"initialization.png", dpi=170)
        plt.close(fig)


def run(args):
    torch.set_num_threads(args.threads)
    jax.config.update("jax_compilation_cache_dir", str(args.cache_dir))
    jax.config.update("jax_persistent_cache_enable_xla_caches", "all")
    if args.device == "cuda" and jax.default_backend() != "gpu":
        raise RuntimeError(f"JAX GPU requested, available devices: {jax.devices()}")
    if args.device == "cpu" and jax.default_backend() != "cpu":
        raise RuntimeError("Set JAX_PLATFORMS=cpu for an explicit CPU run")
    previous_report = None
    if args.resume:
        previous_report = json.loads((args.output/"posterior.json").read_text())
        for name, digest in previous_report["code_sha256"].items():
            if name.startswith("dingo/") and hashlib.sha256((REPO/name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Cannot resume after changing the posterior operator: {name}")
        if previous_report["versions"]["numpyro"] != numpyro.__version__ or previous_report["versions"]["jax"] != jax.__version__:
            raise ValueError("Resume requires the recorded sampler/library versions")
        for key in ("run_dir", "mode_dir", "all_exposures", "correlation", "inclination_error", "turnover_prior",
                    "turnover_max", "chains", "warmup", "max_tree_depth", "warmup_tree_depth", "target_accept", "dense_mass", "seed"):
            current = str(getattr(args, key)) if isinstance(getattr(args, key), Path) else getattr(args, key)
            if current != previous_report["arguments"].get(key):
                raise ValueError(f"Resume must retain {key}")
        if args.step_size_scale != previous_report["arguments"].get("step_size_scale", 1.):
            raise ValueError("Resume must retain step_size_scale")
    elif (args.output/"posterior.json").exists():
        raise FileExistsError("Choose a fresh output directory to preserve the previous experiment")
    adapted_report = None
    if args.from_checkpoint:
        if args.resume:
            raise ValueError("Use either --resume or --from-checkpoint")
        adapted_report = json.loads((args.from_checkpoint/"posterior.json").read_text())
        if not adapted_report.get("warmup_completed"):
            raise ValueError("An adapted checkpoint requires completed warmup")
        # A deeper NUTS tree changes the transition, not the target density.
        # Start a separate retained sample set so diagnostics still know the
        # actual tree limit used for every draw; preserve the original run.
        for key in ("run_dir", "mode_dir", "all_exposures", "correlation", "inclination_error", "turnover_prior",
                    "turnover_max", "chains", "warmup", "target_accept", "dense_mass", "seed"):
            current = str(getattr(args, key)) if isinstance(getattr(args, key), Path) else getattr(args, key)
            if current != adapted_report["arguments"][key]:
                raise ValueError(f"Adapted checkpoint must retain {key}")
        if args.max_tree_depth < adapted_report["arguments"]["max_tree_depth"]:
            raise ValueError("Use the same or a larger tree limit for the adapted restart")
    args.output.mkdir(parents=True, exist_ok=True)
    source = json.loads((args.run_dir/"recovery.json").read_text())
    settings = source["arguments"]
    target = load_native_target(settings["bundle"], REPO/f"data/ID{source['source_geometry']}/config_kinematics.yaml", settings["size"])
    paths = [str(e.path) for e in target.exposures]
    selected = list(range(len(paths))) if args.all_exposures else [paths.index(p) for p in source["train_exposures"]]
    pupils = [target.exposures[i].pupil for i in selected]
    with np.load(args.run_dir/"injection.npz") as injection:
        data, error, valid = [injection[k][selected] for k in ("science", "error", "valid")]
    grid_size = source["best_by_setting"][0]["grid"]
    grid = BilinearLineMap(grid_size, settings["quadrature"], flux_unit=source["fixed_map_flux_unit"], dtype=torch.float64)
    y, x = np.mgrid[-4:5, -4:5]
    kernel = np.exp(-.5*(x*x+y*y)/settings["fit_blur"]**2)
    full = NativeLineProjector.from_target(target, grid.east.numpy(), grid.north.numpy(), kernel,
        np.ones(len(paths)), spectral_samples=7, dtype=torch.float64, deposition=settings["deposition"])
    projector = NativeLineProjector(full.coefficients.numpy()[selected], full.wavelength_um, full.image_shape,
        full.kernels.numpy()[selected], full.response.numpy()[selected], spectral_samples=7, deposition=settings["deposition"])
    likelihood = StripContinuumLikelihood(data, error, valid, pupils, degree=settings["degree"])
    prior = PosteriorPrior(correlation_arcsec=args.correlation, turnover_prior=args.turnover_prior,
                           inclination_error_deg=None if args.inclination_error == 0 else args.inclination_error,
                           turnover_max=args.turnover_max)
    posterior = NativePosterior(JaxNativeProjector(projector), JaxLineMap(grid), JaxStripLikelihood(likelihood), prior=prior)
    low = source["best_by_setting"][0]
    high_source = json.loads((args.mode_dir/"modes.json").read_text())
    if Path(high_source["source_run"]).resolve() != args.run_dir.resolve():
        raise ValueError("Starting branches came from different datasets")
    high = max(high_source["fits"], key=lambda f: f["fitted_vref_rturn_theta"][0])
    with np.load(args.run_dir/low["artifact"]) as saved:
        low_controls = saved["controls"]
    with np.load(args.mode_dir/high["artifact"]) as saved:
        high_controls = saved["controls"]
    rng = np.random.default_rng(args.seed)
    initial_values, starts = [], []
    for c in range(args.chains):
        fit, controls, label = (low, low_controls, "low") if c % 2 == 0 else (high, high_controls, "high")
        initial = posterior.initialize_from_fit(fit, controls)
        initial["v_ref"] *= np.exp(rng.normal(0., .01))
        initial["theta"] += rng.normal(0., .01)
        initial["map_z"] += rng.normal(0., .03, len(initial["map_z"]))
        initial_values.append(initial)
        starts.append({"chain": c, "branch": label, "v_ref": float(initial["v_ref"]),
                       "r_turn": float(initial.get("r_turn", np.exp(initial.get("log_r_turn", 0.))))})
    # Full physical-distance conversion is explicit and remains a toy-source
    # assumption. It does not turn a synthetic recovery into a measured mass.
    from astropy.cosmology import Planck18
    kpc_per_arcsec = float(Planck18.kpc_proper_per_arcmin(target.summary["z"]).value/60.)
    report = {"completed": False, "synthetic_not_science_measurements": True,
        "source_run": str(args.run_dir), "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "source_geometry": source["source_geometry"], "truth_vref_rturn_theta": source["truth_vref_rturn_theta"],
        "truth_inclination_rad": source["truth_inclination_rad"], "prior": prior.to_dict(), "starts": starts,
        "train_exposures": [paths[i] for i in selected], "device": str(jax.devices()),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "versions": {"jax": jax.__version__, "numpyro": numpyro.__version__, "torch": torch.__version__},
        "distance": {"cosmology": "Planck18", "redshift_from_geometry_config": target.summary["z"], "kpc_per_arcsec": kpc_per_arcsec},
        "code_sha256": {p: hashlib.sha256((REPO/p).read_bytes()).hexdigest() for p in
            ("dingo/native_forward.py", "dingo/native_fitting.py", "dingo/native_jax.py", "dingo/native_posterior.py", "diagnostics/native_posterior_check.py")},
        "source_code_sha256": source["code_sha256"],
        "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                         (args.run_dir/"injection.npz", args.run_dir/"recovery.json", args.mode_dir/"modes.json")},
        "limitations": ["Synthetic data; imposed external inclination prior, not measured photometry",
                        "Proper log-field prior differs from earlier unconstrained positive-map profiling",
                        "Fixed center, registration, response, line width and errors; no contamination",
                        "Rotation-only spherical-equivalent mass proxy; no pressure support or distance uncertainty",
                        "Passing chain diagnostics does not establish interval coverage or rule out unseen modes"]}
    if previous_report is not None:
        if "input_sha256" in previous_report and previous_report["input_sha256"] != report["input_sha256"]:
            raise ValueError("Input data changed since the checkpoint")
        report = previous_report
        report["completed"] = False
        report.setdefault("resume_history", []).append({"requested_total_draws": args.samples,
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    else:
        for name, digest in report["code_sha256"].items():
            destination = args.output/"code_snapshot"/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            content = (REPO/name).read_bytes()
            assert hashlib.sha256(content).hexdigest() == digest
            destination.write_bytes(content)
    if adapted_report is not None:
        for name, digest in adapted_report["code_sha256"].items():
            if name.startswith("dingo/") and report["code_sha256"][name] != digest:
                raise ValueError(f"Adapted checkpoint operator changed: {name}")
        if report["input_sha256"] != adapted_report["input_sha256"] or report["versions"] != adapted_report["versions"]:
            raise ValueError("Adapted checkpoint requires identical inputs and library versions")
        content = (args.from_checkpoint/"sampler_state.pkl").read_bytes()
        (args.output/"adapted_initial_state.pkl").write_bytes(content)
        (args.output/"adapted_source_report.json").write_text(json.dumps(adapted_report, indent=2))
        report["adapted_from"] = {"directory": str(args.from_checkpoint),
            "checkpoint_sha256": hashlib.sha256(content).hexdigest(),
            "previous_tree_depth": adapted_report["arguments"]["max_tree_depth"],
            "previous_retained_draws_per_chain": adapted_report.get("retained_draws_per_chain", 0),
            "note": "Same posterior and adapted covariance; fresh retained arrays with this run's tree limit"}
    (args.output/"posterior.json").write_text(json.dumps(report, indent=2))
    # Verify the new backend on real geometry at both old fitted states.
    audit = []
    for fit, controls in ((low, low_controls), (high, high_controls)):
        v, rt, theta = fit["fitted_vref_rturn_theta"]
        tvel = projected_arctan_velocity(grid.east, grid.north, v, rt, theta, settings["fit_inclination"])
        tline = projector(grid(torch.as_tensor(controls)), tvel, sigma_v=45.)[0]
        from dingo.native_jax import projected_velocity
        jvel = projected_velocity(posterior.line_map.east, posterior.line_map.north, v, rt, theta, settings["fit_inclination"])
        jline = posterior.projector(posterior.line_map(jnp.asarray(controls)), jvel)
        difference = np.asarray(jline)-tline.numpy()
        max_error = float(np.max(np.abs(difference[valid]/error[valid])))
        log_difference = float(posterior.likelihood.relative(jline))-float(likelihood.relative(tline))
        if max_error > 1e-9 or abs(log_difference) > 1e-7:
            raise AssertionError("JAX native projection/likelihood disagrees with Torch")
        audit.append({"v_ref": v, "max_pixel_difference_in_error_units": max_error, "log_likelihood_difference": log_difference})
    model_info = initialize_model(jax.random.PRNGKey(args.seed), posterior.model,
                                  init_strategy=init_to_value(values=initial_values[0]))
    initial_params = [unconstrain_fn(posterior.model, (), {}, value) for value in initial_values]
    initial_params = jax.tree.map(lambda *x: jnp.stack(x), *initial_params)
    vg = jax.jit(jax.value_and_grad(model_info.potential_fn))
    q = jax.tree.map(lambda x: x[0], initial_params)
    t0 = time.perf_counter()
    value, grad = vg(q)
    jax.block_until_ready((value, grad))
    compile_seconds = time.perf_counter()-t0
    t0 = time.perf_counter()
    for _ in range(20):
        jax.block_until_ready(vg(q))
    seconds = (time.perf_counter()-t0)/20
    report["backend_audit"] = audit
    report["gradient_benchmark"] = {"compile_seconds": compile_seconds, "seconds_per_single_chain_gradient": seconds,
                                    "initial_potential": float(value)}
    if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves((value, grad))):
        raise ValueError("Nonfinite initial posterior gradient")
    print(json.dumps(report["gradient_benchmark"]), flush=True)
    # Draw the complete explicit prior, including its inclination distribution.
    prior_draws = Predictive(posterior.model, num_samples=512)(jax.random.PRNGKey(args.seed+17), likelihood_weight=0.)
    np.savez_compressed(args.output/"prior_samples.npz", **{k: np.asarray(v) for k, v in prior_draws.items()})
    (args.output/"posterior.json").write_text(json.dumps(report, indent=2))
    if args.audit_only:
        report["audit_completed"] = True
        (args.output/"posterior.json").write_text(json.dumps(report, indent=2))
        return report
    kinetic_sites = ["v_ref", "r_turn" if args.turnover_prior == "uniform" else "log_r_turn", "theta",
                     "inclination" if args.inclination_error else "cos_i", "log_flux", "map_sigma", "envelope_width"]
    kernel_options = {"dense_mass": True if args.dense_mass == "full" else [tuple(kinetic_sites)],
                      "target_accept_prob": args.target_accept}
    kernel = NUTS(posterior.model, **kernel_options, max_tree_depth=args.max_tree_depth)
    mcmc = MCMC(kernel, num_warmup=args.warmup, num_samples=args.chunk, num_chains=args.chains,
                chain_method="vectorized", progress_bar=False)
    extras = ("potential_energy", "energy", "num_steps", "accept_prob", "diverging")
    samples, extra = {}, {}
    completed_draws = 0
    if args.resume:
        # Only load the checkpoint in the explicitly requested existing run.
        # Operator, input, sampler settings and library versions were checked.
        with (args.output/"sampler_state.pkl").open("rb") as checkpoint:
            restored = pickle.load(checkpoint)
        if isinstance(restored, dict):
            mcmc.post_warmup_state = jax.device_put(restored["state"])
            samples, extra = restored["samples"], restored["stats"]
            completed_draws = samples["v_ref"].shape[1] if samples else 0
        else:
            # First prototype checkpoints contained only the adapted state.
            mcmc.post_warmup_state = jax.device_put(restored)
        if not isinstance(restored, dict) and (args.output/"samples.npz").exists():
            with np.load(args.output/"samples.npz") as saved:
                samples = dict(saved)
            with np.load(args.output/"sampler_stats.npz") as saved:
                extra = dict(saved)
            completed_draws = samples["v_ref"].shape[1]
        origin = np.asarray(report.get("initial_postwarmup_iterations_by_chain", 0))
        discarded = np.asarray(mcmc.post_warmup_state.i)-args.warmup-completed_draws-origin
        report["postwarmup_draws_not_in_retained_arrays_by_chain"] = np.broadcast_to(discarded, (args.chains,)).tolist()
        if args.samples <= completed_draws or (args.samples-completed_draws) % args.chunk:
            raise ValueError("Resume requires an additional whole number of blocks")
        print(f"Resuming adapted chains after {completed_draws} retained draws each", flush=True)
    elif args.from_checkpoint:
        with (args.output/"adapted_initial_state.pkl").open("rb") as checkpoint:
            restored = pickle.load(checkpoint)
        state = restored["state"]
        report["checkpoint_step_size_by_chain"] = np.asarray(state.adapt_state.step_size).tolist()
        state = reduce_adapted_step_size(state, args.step_size_scale)
        report["sampling_step_size_by_chain"] = np.asarray(state.adapt_state.step_size).tolist()
        mcmc.post_warmup_state = jax.device_put(state)
        report["initial_postwarmup_iterations_by_chain"] = np.broadcast_to(
            np.asarray(mcmc.post_warmup_state.i)-args.warmup, (args.chains,)).tolist()
        report["warmup_completed"] = True
        report["warmup_seconds"] = 0.
        report["warmup_reused_from"] = str(args.from_checkpoint)
        (args.output/"posterior.json").write_text(json.dumps(report, indent=2))
        print(f"Continuing adapted chains with tree limit {args.max_tree_depth}; fresh retained arrays", flush=True)
    else:
        warmup_depth = args.max_tree_depth if args.warmup_tree_depth is None else args.warmup_tree_depth
        # Restrict early trajectories before covariance adaptation. A new
        # sampling kernel then uses the requested full trajectory limit, with
        # exactly the adapted state and random keys. The posterior is unchanged.
        warmup_mcmc = mcmc if warmup_depth == args.max_tree_depth else MCMC(
            NUTS(posterior.model, **kernel_options, max_tree_depth=warmup_depth),
            num_warmup=args.warmup, num_samples=args.chunk, num_chains=args.chains,
            chain_method="vectorized", progress_bar=False)
        print(f"Warmup: {args.chains} chains, {args.warmup} iterations, tree limit {warmup_depth}; two starting branches", flush=True)
        t0 = time.perf_counter()
        warmup_mcmc.warmup(jax.random.PRNGKey(args.seed+1), init_params=initial_params, extra_fields=extras, collect_warmup=True)
        mcmc.post_warmup_state = warmup_mcmc.post_warmup_state
        report["warmup_seconds"] = time.perf_counter()-t0
        report["warmup_tree_depth"] = warmup_depth
        warmup = {k: np.asarray(v) for k, v in warmup_mcmc.get_samples(group_by_chain=True).items()}
        np.savez_compressed(args.output/"warmup.npz", **warmup)
        with (args.output/"sampler_state.pkl").open("wb") as checkpoint:
            pickle.dump({"state": jax.device_get(mcmc.post_warmup_state), "samples": {}, "stats": {}}, checkpoint)
        report["warmup_completed"] = True
        (args.output/"posterior.json").write_text(json.dumps(report, indent=2))
    for chunk in range((args.samples-completed_draws)//args.chunk):
        t0 = time.perf_counter()
        print(f"Retaining draws {completed_draws+chunk*args.chunk+1}–{completed_draws+(chunk+1)*args.chunk} of {args.samples} per chain", flush=True)
        mcmc.run(mcmc.post_warmup_state.rng_key, extra_fields=extras, init_params=mcmc.post_warmup_state.z)
        current = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
        current_extra = {k: np.asarray(v) for k, v in mcmc.get_extra_fields(group_by_chain=True).items()}
        samples = {k: np.concatenate([samples[k], v], axis=1) if k in samples else v for k, v in current.items()}
        extra = {k: np.concatenate([extra[k], v], axis=1) if k in extra else v for k, v in current_extra.items()}
        mcmc.post_warmup_state = mcmc.last_state
        # Local checkpoint of complete adapted state, including random keys.
        checkpoint_path = args.output/"sampler_state.pkl"
        temporary_path = checkpoint_path.with_suffix(".tmp")
        with temporary_path.open("wb") as checkpoint:
            pickle.dump({"state": jax.device_get(mcmc.last_state), "samples": samples, "stats": extra}, checkpoint)
        temporary_path.replace(checkpoint_path)
        report.setdefault("sampling_block_seconds", []).append(time.perf_counter()-t0)
        summarize(samples, extra, report, args.output, kpc_per_arcsec)
        print(json.dumps({k: v for k, v in report["diagnostics"].items() if k != "parameters"}), flush=True)
    report["completed"] = True
    (args.output/"posterior.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--mode-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--all-exposures", action="store_true")
    p.add_argument("--correlation", type=float, default=.10)
    p.add_argument("--inclination-error", type=float, default=5., help="External prior sigma in degrees; zero selects isotropic prior")
    p.add_argument("--turnover-prior", choices=["uniform", "log_uniform"], default="uniform")
    p.add_argument("--turnover-max", type=float, default=2.)
    p.add_argument("--warmup", type=int, default=800)
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--chunk", type=int, default=250)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--max-tree-depth", type=int, default=9)
    p.add_argument("--warmup-tree-depth", type=int, help="Optional smaller trajectory limit during covariance adaptation only")
    p.add_argument("--target-accept", type=float, default=.9)
    p.add_argument("--dense-mass", choices=["block", "full"], default="block")
    p.add_argument("--seed", type=int, default=741)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--resume", action="store_true", help="Continue this run's own saved adapted chains to --samples total draws")
    p.add_argument("--from-checkpoint", type=Path, help="Fresh sample set from an identical posterior's adapted state, permitting a larger tree limit")
    p.add_argument("--step-size-scale", type=float, default=1., help="For --from-checkpoint: reduce fixed integration steps, retaining the adapted covariance and target")
    p.add_argument("--cache-dir", type=Path, default=REPO.parent/"results/_jax_cache")
    p.add_argument("--predictive-only", action="store_true", help="Check predictions of the saved completed run")
    args = p.parse_args()
    if args.chains < 2 or args.samples < args.chunk or args.samples % args.chunk or args.chunk < 4 or args.warmup < 1:
        p.error("Use at least two chains, positive warmup, >=4 draws per block, and a whole number of blocks")
    if args.warmup_tree_depth is not None and not 1 <= args.warmup_tree_depth <= args.max_tree_depth:
        p.error("Warmup tree depth must be positive and no larger than the sampling limit")
    if not np.isfinite(args.step_size_scale) or not 0 < args.step_size_scale <= 1:
        p.error("Step-size scale must be in (0, 1]")
    if args.step_size_scale != 1. and not (args.from_checkpoint or args.resume):
        p.error("Step-size scaling requires an adapted checkpoint or continuation of that run")
    if args.predictive_only:
        from diagnostics.native_posterior_predictive import run as predictive_run
        predictive_run(args.output)
    else:
        run(args)
