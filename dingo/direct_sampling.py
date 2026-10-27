"""Joint NumPyro sampling of direct R+C kinematics and nonnegative emission.

The source map is a latent variable, not a map refitted or frozen inside a
kinematic likelihood. Warmup and draws advance the same NUTS state. Checkpoints
contain that state (including adaptation and RNG), so resuming does not start
new warmup or silently repeat draws.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import threading
import time

import numpy as np

# Do not reserve nearly all GPU memory before PyTorch releases its profile fit.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _heartbeat(message, enabled, *, announce=True):
    """Keep compilation and unusually slow sampling blocks visibly alive."""
    stopped = threading.Event()
    start = time.monotonic()

    def report():
        while not stopped.wait(20):
            print(f"{message}: {time.monotonic()-start:.0f} seconds elapsed", flush=True)

    if enabled:
        if announce:
            print(message, flush=True)
        worker = threading.Thread(target=report, daemon=True)
        worker.start()
    try:
        yield
    finally:
        stopped.set()
        if enabled:
            worker.join()


class JointSampler:
    """Small, file-independent, checkpointable multi-chain NUTS driver.

    ``initial_values`` is one dictionary of constrained values per chain.
    ``model.model`` defines the target and ``model.site_names`` names its small
    kinematic block. NUTS adapts a dense mass matrix for that block and a diagonal
    mass matrix for the thousands of emission pixels. It never constructs a
    dense pixel-by-pixel covariance matrix.
    """

    def __init__(self, model, initial_values, *, warmup, seed=42,
                 target_accept=.9, max_tree_depth=8):
        import jax
        import jax.numpy as jnp
        from numpyro.infer import NUTS
        from numpyro.infer.initialization import init_to_value
        from numpyro.infer.util import unconstrain_fn

        self.warmup = int(warmup)
        self.chains = len(initial_values)
        self.kernel = NUTS(
            model.model, init_strategy=init_to_value(values=initial_values[0]),
            dense_mass=[tuple(model.site_names)], target_accept_prob=target_accept,
            max_tree_depth=max_tree_depth)
        coordinates = [unconstrain_fn(model.model, (), {}, values)
                       for values in initial_values]
        if not all(np.isfinite(np.asarray(v)).all()
                   for point in coordinates for v in point.values()):
            raise ValueError("Initial values must be strictly inside their prior bounds.")
        initial = jax.tree.map(lambda *x: jnp.stack(x), *coordinates)
        # Legacy uint32 keys can be saved in a plain NPZ without pickling.
        keys = jax.random.split(jax.random.PRNGKey(seed), self.chains)
        self.state = jax.vmap(lambda key, z: self.kernel.init(
            key, self.warmup, z, (), {}))(keys, initial)
        jax.block_until_ready(self.state)
        self.constrain = self.kernel.postprocess_fn((), {})
        self.functions = {}

    @property
    def iterations(self):
        counts = np.asarray(self.state.i)
        if counts.shape != (self.chains,) or np.any(counts != counts[0]):
            raise ValueError("Chain iteration counters disagree.")
        return int(counts[0])

    def state_arrays(self):
        """Serialize array leaves only; no pickle or executable objects."""
        import jax
        return {f"state_{i:04d}": np.asarray(value)
                for i, value in enumerate(jax.tree.leaves(self.state))}

    def restore(self, arrays):
        """Restore a state after the caller verifies target/settings identity."""
        import jax
        leaves, structure = jax.tree.flatten(self.state)
        keys = sorted(k for k in arrays if k.startswith("state_"))
        if keys != [f"state_{i:04d}" for i in range(len(leaves))]:
            raise ValueError("Checkpoint has an incompatible sampler state.")
        restored = []
        for key, reference in zip(keys, leaves):
            value = arrays[key]
            if value.shape != reference.shape or value.dtype != reference.dtype:
                raise ValueError("Checkpoint state shapes or dtypes do not match.")
            restored.append(jax.device_put(value))
        self.state = jax.tree.unflatten(structure, restored)
        _ = self.iterations

    def advance(self, count, *, collect=True):
        """Advance global iterations, returning [chain,draw,...] host arrays.

        Adaptation stops at the original total warmup count even across chunks.
        The caller must not mix warmup and retained draws in one collected block.
        """
        import jax
        key = (int(count), bool(collect))
        if key not in self.functions:
            def transition(state, _):
                state = jax.vmap(lambda s: self.kernel.sample(s, (), {}))(state)
                stats = {name: getattr(state, name) for name in
                         ("energy", "potential_energy", "num_steps", "accept_prob", "diverging")}
                stats["iteration"] = state.i
                stats["is_warmup"] = state.i <= self.warmup
                samples = jax.vmap(self.constrain)(state.z) if collect else {}
                return state, (samples, stats)
            self.functions[key] = jax.jit(lambda state: jax.lax.scan(
                transition, state, xs=None, length=int(count)))
        before = self.iterations
        state, output = self.functions[key](self.state)
        jax.block_until_ready(state)
        self.state = state
        if self.iterations != before + count:
            raise RuntimeError("NUTS iteration counter did not advance as requested.")
        return jax.tree.map(lambda x: np.asarray(x).swapaxes(0, 1), output)


def _positive_integer(name, value, minimum=1):
    if isinstance(value, bool) or int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _atomic_npz(path, arrays):
    """Uncompressed saves keep checkpoints quick even with full pixel draws."""
    temporary = path.with_name(path.stem + ".pending.npz")
    np.savez(temporary, **arrays)
    temporary.replace(path)


def _fingerprint(record):
    text = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def _validate_retained_checkpoint(saved, *, chains, warmup, draws, site_names):
    """Reject truncated history before restoring or overwriting a checkpoint."""
    iterations = int(saved["iterations"])
    if iterations < 0 or int(saved["warmup"]) != warmup:
        raise ValueError("Checkpoint warmup/iteration counters are inconsistent.")
    retained = max(0, iterations-warmup)
    if retained > draws:
        raise ValueError("draws is a total; it cannot be smaller than the retained checkpoint.")
    sample_keys = {k for k in saved if k.startswith("sample_")}
    stat_keys = {k for k in saved if k.startswith("stats_")}
    expected_samples = {f"sample_{k}" for k in (*site_names, "theta", "emission")}
    expected_stats = {f"stats_{k}" for k in (
        "energy", "potential_energy", "num_steps", "accept_prob", "diverging", "iteration", "is_warmup")}
    if retained and (sample_keys != expected_samples or stat_keys != expected_stats):
        raise ValueError("Checkpoint is missing retained samples or sampler statistics.")
    if not retained and (sample_keys or stat_keys):
        raise ValueError("Warmup-only checkpoint must not contain posterior draws.")
    for key in sample_keys | stat_keys:
        if saved[key].shape[:2] != (chains, retained) or not np.isfinite(saved[key]).all():
            raise ValueError("Checkpoint draws are nonfinite or disagree with iteration counters.")
    if retained:
        expected = np.arange(warmup+1, iterations+1)
        if (np.any(saved["stats_is_warmup"])
                or not np.all(saved["stats_iteration"] == expected[None, :])):
            raise ValueError("Checkpoint retained iterations are duplicated, missing, or include warmup.")


def _select_device(request):
    """Report the actual JAX backend; PyTorch CUDA availability is insufficient."""
    import jax
    if request not in ("auto", "cpu", "gpu", "cuda"):
        raise ValueError("device must be 'auto', 'cpu', or 'gpu'.")
    if request == "auto":
        return jax.devices()[0]
    return jax.devices("gpu" if request in ("gpu", "cuda") else "cpu")[0]


def _save_prediction(run, posterior_model, samples, identity):
    """Save one actual joint draw near the pooled kinematic median.

    Combining a median parameter vector with a separately median source map
    need not describe any sampled galaxy. Select a real joint draw instead and
    label its chain and iteration, leaving pooled values for the summary table.
    """
    from .direct_workflow import _centers
    theta = samples["theta"]
    flat = theta.reshape(-1, 6).copy()
    anchor = np.arctan2(np.sin(flat[:, 2]).mean(), np.cos(flat[:, 2]).mean())
    flat[:, 2] = anchor + np.angle(np.exp(1j*(flat[:, 2]-anchor)))
    median = np.median(flat, axis=0)
    spread = np.std(flat, axis=0)
    distance = np.sum(((flat-median)/np.maximum(spread, 1e-10))**2, axis=1)
    chain, draw = np.unravel_index(int(np.argmin(distance)), theta.shape[:2])
    point = theta[chain, draw]
    emission = samples["emission"][chain, draw]
    predictions = posterior_model.predict(point, emission)
    payload = dict(theta=point, emission=emission, chain_index=np.array(chain),
                   draw_index=np.array(draw), fingerprint=np.array(identity),
                   projected_centers=np.array(list(_centers(run.model, point).values())))
    for pupil, predicted in zip(("R", "C"), predictions):
        payload[f"predicted_{pupil}"] = np.asarray(predicted)
        payload[f"observed_{pupil}"] = run.model.data[pupil].detach().cpu().numpy()
        payload[f"valid_{pupil}"] = run.model.valid_by_pupil[pupil].detach().cpu().numpy()
    _atomic_npz(run.directory / "posterior_prediction.npz", payload)
    return payload


def sample(run, *, warmup=500, draws=1000, chains=4, seed=42,
           emission_prior_scale=None, target_accept=.9, max_tree_depth=8,
           checkpoint_every=25, device="auto", progress=True,
           overwrite=False, resume=False):
    """Sample kinematics AND one shared unknown source map jointly.

    Parameters
    ----------
    run : DirectRun
        Prepared source with a completed ``direct.fit`` checkpoint. That fit
        supplies only starting values, never a fixed map in the likelihood.
    warmup, draws, chains : int
        Adaptation steps, retained steps per independent chain, and chain count.
        Short runs are debugging experiments, not automatically reportable errors.
    emission_prior_scale : float or None
        Every source pixel has an independent HalfNormal prior with this scale,
        in FITS intensity units. No Sérsic profile, imaging template, or spatial
        smoothing is used. None selects 10 times the maximum positive observed
        intensity (or input noise, whichever is larger). This data-based default
        is a fixed empirical choice, not an inferred scale. Its effect must be
        checked for science results; its exact value is saved in run_info.txt.
    target_accept, max_tree_depth : float, int
        NUTS adaptation target and trajectory budget. A high depth-hit fraction
        indicates inefficient exploration; more saved steps alone need not fix it.
    checkpoint_every : int
        Save complete sampler state and all retained draws every this many steps.
        A progress bar counts warmup/draws, with a heartbeat during compilation.
    device : str
        Actual JAX backend: auto, cpu or gpu. Reported separately from the Torch
        device used for the preceding profile fit.
    overwrite, resume : bool
        Mutually exclusive. Overwrite replaces this run's MCMC products only.
        Resume restores RNG/adaptation exactly and can extend ``draws`` (total,
        not additional). Warmup, priors, inputs and sampler settings must match.

    Returns
    -------
    pathlib.Path
        ``posterior.npz``: [chain,draw,parameter] theta, full emission draws,
        NUTS statistics, and a resumable state. Emission draws are stored in
        float32 to save space; likelihoods and checkpoint states use float64.
        ``posterior_diagnostics.json`` reports mixing, never a success guarantee.
    """
    warmup = _positive_integer("warmup", warmup)
    draws = _positive_integer("draws", draws)
    chains = _positive_integer("chains", chains)
    checkpoint_every = _positive_integer("checkpoint_every", checkpoint_every)
    max_tree_depth = _positive_integer("max_tree_depth", max_tree_depth)
    if max_tree_depth > 12 or not .5 < target_accept < 1:
        raise ValueError("Use max_tree_depth <= 12 and 0.5 < target_accept < 1.")
    if resume and overwrite:
        raise ValueError("Choose resume=True or overwrite=True, not both.")
    from .direct_workflow import _validate_run_identity
    run.info = _validate_run_identity(run)
    path = run.directory / "posterior.npz"
    if path.exists() and not (resume or overwrite):
        raise FileExistsError("MCMC results exist. Use resume=True, overwrite=True, or a new run_name.")
    if resume and not path.exists():
        raise FileNotFoundError("There is no posterior.npz checkpoint to resume.")
    with np.load(run.directory / "fit.npz", allow_pickle=False) as fit:
        start_theta = fit["theta"].copy()
        start_emission = fit["emission"].copy()
    data = [run.model.data[p].detach().cpu().numpy() for p in ("R", "C")]
    if emission_prior_scale is None:
        emission_prior_scale = 10 * max(
            max(float(np.max(a)) for a in data), max(run.model.sigma_by_pupil.values()))
    if (isinstance(emission_prior_scale, bool) or not np.isfinite(emission_prior_scale)
            or emission_prior_scale <= 0):
        raise ValueError("emission_prior_scale must be positive and finite.")

    from .direct_posterior_model import DirectPosteriorModel
    import jax
    from tqdm.auto import tqdm

    selected_device = _select_device(device)
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "jax", "jaxlib", "numpyro")}
    code = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("direct_sampling.py", "direct_posterior_model.py")}
    target = dict(inputs=run.model.input_records,
                  model=run.model.configuration["direct_model"],
                  source=run.model.configuration["summary"],
                  emission_prior_scale=float(emission_prior_scale),
                  warmup=warmup, chains=chains, seed=int(seed),
                  target_accept=float(target_accept), max_tree_depth=max_tree_depth,
                  versions=versions, code=code)

    # Always reread the record: notebook handles can outlive module reloads or
    # a previous sampling call. Do not replace newer saved metadata with a copy.
    run.info = json.loads((run.directory / "run_info.txt").read_text())
    saved = None
    if resume:
        with np.load(path, allow_pickle=False) as z:
            saved = {key: z[key].copy() for key in z.files}
    samples = {}
    stats = {}
    sampler = None
    start_time = time.monotonic()
    with jax.default_device(selected_device):
        posterior_model = DirectPosteriorModel(run.model, emission_prior_scale)
        # Calibration arrays are part of the target, not merely their path.
        target["calibration_hash"] = posterior_model.calibration_hash
        identity = _fingerprint(target)
        if saved is not None and str(saved["fingerprint"]) != identity:
            raise ValueError("Checkpoint target/settings/code changed. Start a new named run instead of resuming.")
        if saved is not None:
            _validate_retained_checkpoint(saved, chains=chains, warmup=warmup, draws=draws,
                                         site_names=posterior_model.parameter_sites)
            if (getattr(run.model, "fixed_rdy", None) is not None
                    and "sample_theta" in saved
                    and not np.all(saved["sample_theta"][..., 4] == run.model.fixed_rdy)):
                raise ValueError("Checkpoint draws disagree with the fixed R y-registration.")
        rng = np.random.default_rng(seed)
        initial = []
        lower, upper = run.model.theta_bounds
        for _ in range(chains):
            theta = start_theta + rng.normal(size=6) * [10., .25, .02, .02, .02, np.deg2rad(.5)]
            theta = np.clip(theta, lower + 1e-5*(upper-lower), upper - 1e-5*(upper-lower))
            if getattr(run.model, "fixed_rdy", None) is not None:
                theta[4] = run.model.fixed_rdy
            emission = np.maximum(start_emission, .1*min(run.model.sigma_by_pupil.values()))
            emission = emission * np.exp(rng.normal(0., .1, emission.shape))
            initial.append(posterior_model.initial_values(theta, emission))

        if overwrite:
            # Remove only known MCMC artifacts; profile fit and inputs survive.
            for name in ("posterior.npz", "posterior_diagnostics.json", "posterior_summary.txt",
                         "posterior_corner.png", "posterior_corner.pdf", "posterior_traces.png",
                         "posterior_residuals.png", "posterior_emission.png", "posterior_rotation_curve.png",
                         "posterior_prediction.npz"):
                (run.directory / name).unlink(missing_ok=True)
        previous_record = run.info.get("mcmc", {})
        record = dict(target, fingerprint=identity, backend=str(selected_device),
                      sampled_kinematic_sites=list(posterior_model.site_names),
                      sampler="NumPyro NUTS: joint kinematics and independent positive emission pixels",
                      draws_requested=draws, checkpoint_every=checkpoint_every,
                      started_utc=_now(), status="initializing", completed_warmup=0, retained_draws=0,
                      scope="Conditional on fixed center, calibration, pixelization, brightness prior and independent noise; no PSF or dispersion.")
        if run.model.fixed_rdy is not None:
            record["fixed_parameters"] = {"R_dy": run.model.fixed_rdy}
            record["scope"] += f" R dy is fixed at {run.model.fixed_rdy:+.6g} pixels; its uncertainty is excluded."
        if resume:
            record["started_utc"] = previous_record.get("started_utc", record["started_utc"])
            record["resumed_utc"] = [*previous_record.get("resumed_utc", []), _now()]
        run.info["mcmc"] = record
        run.info["status"] = "mcmc_initializing"
        run.info["posterior_errors_available"] = False
        run.save_info()
        print(f"MCMC: {chains} chains; {warmup} warmup + {draws} retained draws per chain.")
        print(f"JAX device: {selected_device}; float64 target; {np.prod(run.model.shape)} emission pixels sampled.")
        if run.model.fixed_rdy is not None:
            print(f"R dy fixed at {run.model.fixed_rdy:+.6g} pixels; five kinematic parameters sampled.")
        print(f"Pixel prior: independent HalfNormal(scale={emission_prior_scale:.6g}); no spatial shape prior.")
        print("Saving to:", path)

        def checkpoint(status):
            if sampler is None:
                return
            payload = dict(sampler.state_arrays(), fingerprint=np.array(identity),
                           iterations=np.array(sampler.iterations), warmup=np.array(warmup))
            payload.update({f"sample_{k}": v for k, v in samples.items()})
            payload.update({f"stats_{k}": v for k, v in stats.items()})
            if samples:
                # Partial retained checkpoints must be inspectable even if the
                # kernel dies before the requested total has been reached.
                prediction = _save_prediction(run, posterior_model, samples, identity)
                # Keep a coherent prediction in the SAME atomic archive as the
                # draws. A crash between writes of two separate files cannot
                # leave the primary checkpoint without its residual plot data.
                payload.update({f"prediction_{k}": v for k, v in prediction.items()})
            _atomic_npz(path, payload)
            record.update(status=status, updated_utc=_now(),
                          completed_warmup=min(warmup, sampler.iterations),
                          retained_draws=max(0, sampler.iterations-warmup),
                          elapsed_seconds_this_call=time.monotonic()-start_time)
            run.info["posterior_errors_available"] = bool(record["retained_draws"] > 0)
            run.info["status"] = f"mcmc_{status}"
            run.save_info()

        try:
            with _heartbeat("Initializing NUTS (first JAX compilation can take a while)", progress):
                sampler = JointSampler(posterior_model, initial, warmup=warmup, seed=seed,
                                       target_accept=target_accept, max_tree_depth=max_tree_depth)
            if saved is not None:
                sampler.restore(saved)
                if sampler.iterations != int(saved["iterations"]):
                    raise ValueError("Checkpoint state and saved iteration count disagree.")
                samples = {k.removeprefix("sample_"): v for k, v in saved.items() if k.startswith("sample_")}
                stats = {k.removeprefix("stats_"): v for k, v in saved.items() if k.startswith("stats_")}
                retained = max(0, sampler.iterations-warmup)
                if retained > draws:
                    raise ValueError("draws is a total; it cannot be smaller than the retained checkpoint.")
                if any(v.shape[:2] != (chains, retained) for v in [*samples.values(), *stats.values()]):
                    raise ValueError("Checkpoint draws and sampler iteration counters disagree.")
            checkpoint("warming_up" if sampler.iterations < warmup else "sampling")
            for end, phase, collect in ((warmup, "Warmup", False), (warmup+draws, "MCMC", True)):
                phase_start = 0 if not collect else warmup
                with tqdm(total=end-phase_start, initial=min(end-phase_start, max(0, sampler.iterations-phase_start)),
                          desc=phase, unit="step/chain", disable=not progress) as bar:
                    while sampler.iterations < end:
                        count = min(checkpoint_every, end-sampler.iterations)
                        with _heartbeat(f"{phase}: computing the next {count} steps per chain",
                                        progress, announce=False):
                            new_samples, new_stats = sampler.advance(count, collect=collect)
                        if collect:
                            if np.any(new_stats["is_warmup"]):
                                raise RuntimeError("Warmup draws cannot enter the posterior.")
                            # Preserve every map draw for diagnostics and reanalysis.
                            new_samples["emission"] = new_samples["emission"].astype(np.float32)
                            for k, v in new_samples.items():
                                samples[k] = np.concatenate((samples[k], v), axis=1) if k in samples else v
                            for k, v in new_stats.items():
                                stats[k] = np.concatenate((stats[k], v), axis=1) if k in stats else v
                        checkpoint("sampling" if collect else "warming_up")
                        bar.update(count)
                        bar.set_postfix(accept=f"{new_stats['accept_prob'].mean():.2f}",
                                        divergences=int(new_stats["diverging"].sum()))
            checkpoint("completed")
            record["finished_utc"] = _now()
            run.save_info()
        except BaseException as error:
            status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            # Preserve the last fully committed block. Initialization failures
            # have metadata only; a previous checkpoint is never fabricated.
            record.update(status=status, error=str(error), stopped_utc=_now())
            run.info["status"] = f"mcmc_{status}"
            run.save_info()
            raise
    from .direct_posterior_review import diagnostics
    report = diagnostics(path)
    record["diagnostics"] = report
    run.save_info()
    print("MCMC saved:", path)
    print("Convergence checks passed:", report["convergence_checks_passed"])
    return path
