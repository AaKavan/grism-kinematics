"""Checkpointed real-source, zero-intrinsic-dispersion R+C inference worker.

Use ``dingo.rotation.fit`` from the notebook. This module alone imports JAX.
The optimizer supplies starting states; only retained Gibbs/NUTS draws supply
posterior uncertainty. Historical optimization/posterior targets are untouched.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import threading
import time

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_FLAGS', '--xla_gpu_enable_triton_gemm=false')

import numpy as np

from .real_workflow import REPO, read_json, sha256, write_json


def atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


@contextlib.contextmanager
def heartbeat(message):
    """Explain compilation/long chunks while the main thread waits on the GPU."""
    start = time.monotonic()
    stop = threading.Event()
    def tick():
        while not stop.wait(20):
            print(f'  {message} ... {time.monotonic()-start:.0f}s elapsed; '
                  'the next completed block will report its step count.', flush=True)
    thread = threading.Thread(target=tick, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


def convergence_diagnostics(samples, stats, max_depth):
    """Check every sampled coordinate, including the nuisance continuum block."""
    import arviz_stats as azs
    counts = next(iter(samples.values())).shape[:2]
    output = dict(convergence_gate_passed=False, chains=counts[0], draws_per_chain=counts[1],
                  parameters={}, sampling_warnings=[],
                  gate='At least 4 chains; all rank Rhat < 1.01, bulk/tail ESS >= 400; '
                  'zero divergences. Tree-depth saturation is an efficiency warning, not a veto. '
                  'Necessary, not sufficient for valid inference.',
                  energy_note='Ordinary NUTS BFMI is not a calibrated diagnostic of this changing Gibbs conditional.')
    required_stats = {'diverging', 'num_steps', 'accept_prob', 'is_warmup'}
    failures = []
    if not required_stats.issubset(stats):
        failures.append('Missing required sampler statistics, including warmup membership')
    if any(np.asarray(v).shape != counts or not np.isfinite(v).all() for v in stats.values()):
        failures.append('Sampler statistics have incompatible dimensions or nonfinite values')
    if any(np.asarray(v).shape[:2] != counts or not np.isfinite(v).all() for v in samples.values()):
        failures.append('Posterior coordinates have incompatible dimensions or nonfinite values')
    if np.any(stats.get('is_warmup', False)):
        failures.append('Warmup transitions must not enter retained posterior diagnostics')
    if failures:
        output['sampling_input_errors'] = failures
        return output
    if counts[0] < 2 or counts[1] < 4:
        return output
    # Derived rotation values can mix worse than their individual coordinates.
    extra = dict(vsini=samples['vrot']*np.sin(samples['inclination']))
    radius = np.array([.05, .1, .2, .3, .5])
    rt = np.exp(samples['log_rturn'])
    extra['rotation_profile'] = (samples['vrot'][..., None] *
        np.arctan(radius/rt[..., None])/np.arctan(.5/rt)[..., None])
    for name, values in (samples | extra).items():
        rh = np.asarray(azs.rhat(values, method='rank'))
        bulk = np.asarray(azs.ess(values, method='bulk'))
        tail = np.asarray(azs.ess(values, method='tail', prob=(.05, .95)))
        output['parameters'][name] = dict(max_rank_rhat=float(np.max(rh)),
            min_bulk_ess=float(np.min(bulk)), min_tail_ess=float(np.min(tail)))
    summary = list(output['parameters'].values())
    rhat = max(s['max_rank_rhat'] for s in summary)
    bulk = min(s['min_bulk_ess'] for s in summary)
    tail = min(s['min_tail_ess'] for s in summary)
    finite = all(np.isfinite(list(s.values())).all() for s in summary)
    divergences = int(np.sum(stats['diverging']))
    depth = float(np.mean(stats['num_steps'] >= 2**max_depth-1))
    if depth >= .01:
        output['sampling_warnings'].append(
            f'{100*depth:.1f}% of retained transitions reached maximum tree depth. '
            'This warns about sampling efficiency; judge convergence using Rhat, ESS and divergences.')
    output.update(max_rank_rhat=rhat, min_bulk_ess=bulk, min_tail_ess=tail,
        all_diagnostics_finite=finite, divergences=divergences, max_depth_fraction=depth,
        mean_acceptance=float(np.mean(stats['accept_prob'])),
        mean_leapfrog_steps=float(np.mean(stats['num_steps'])),
        convergence_gate_passed=bool(counts[0] >= 4 and finite and rhat < 1.01 and bulk >= 400
                                     and tail >= 400 and divergences == 0))
    return clean_json(output)


def prediction(model, target, parameters=None, sites=None):
    """Render an actual draw, keeping its sampled continuum and background."""
    import jax.numpy as jnp
    if sites is None:
        objective, parts = model.components(jnp.asarray(parameters))
        parts = {k: np.asarray(v) for k, v in parts.items()}
        x, speed, _, _, _ = model.split(parameters)
    else:
        sites = {k: jnp.asarray(v) for k, v in sites.items()}
        x, speed = model.vector_from_sites(sites), sites['vrot']
        _, flux, velocity = model.renderer.render(x, speed)
        parts = dict(line=np.asarray(model.line_from_sites(sites)), source_flux=np.asarray(flux),
            source_velocity=np.asarray(velocity),
            continuum=np.asarray(model.continuum_model.continuum(sites['continuum_coefficients'])),
            continuum_coefficients=np.asarray(sites['continuum_coefficients']),
            background=np.asarray(model.background_sigma*sites['background_z']))
    total = parts['line']+parts['continuum']+parts['background'][:, None, None]
    data, error, valid = [np.stack([getattr(e, k) for e in target.exposures])
                          for k in ('science', 'error', 'valid')]
    residual = np.divide(data-total, error, out=np.full_like(data, np.nan), where=valid)
    parts.update(source_id=np.asarray(target.source_id), vector=np.asarray(x), residual=residual,
                 source_east=np.asarray(model.engine.line_map.east),
                 source_north=np.asarray(model.engine.line_map.north))
    if parameters is not None:
        parts['parameters'] = np.asarray(parameters)
    description = model.engine.describe(np.asarray(x), float(speed))
    description['chi_square'] = float(np.nansum(residual**2))
    edge = (abs(parts['source_east']) > .8) | (abs(parts['source_north']) > .8)
    description['source_edge_flux_fraction'] = float(parts['source_flux'][edge].sum())
    return parts, description


def historical_seed(model, target, source_dir):
    """Optional initializer only: accept a verified, matching flexible sigma=0 fit.

    The new target is always evaluated again. Old posterior samples are never
    reused, and a fresh source needs no historical fit directories.
    """
    paths = [source_dir/'morphology_controls_cross_v1/dispersion.json',
             source_dir/'morphology_controls_v1/dispersion.json',
             source_dir/'simple_dispersion_imaging_v1/dispersion.json',
             source_dir/'simple_dispersion_fine_v1/dispersion.json']
    for path in paths:
        if not path.exists():
            continue
        report = read_json(path)
        if report.get('source_id') != target.source_id:
            continue
        fits = [f for f in report.get('fits', []) if f.get('model') == 'flexible'
                and f.get('mode') == 'positive' and f.get('variant', 'baseline') == 'baseline'
                and f.get('fixed_sigma_kms') == 0 and f.get('spectral_scale', 1) == 1]
        if not fits:
            continue
        best = min(fits, key=lambda f: f['objective'])
        artifact = path.parent/best['artifact']
        native = path.parent/'native_data.npz'
        if (not artifact.resolve().is_relative_to(path.parent.resolve()) or not native.exists()
                or sha256(artifact) != best.get('artifact_sha256')):
            continue
        with np.load(native) as old:
            matched = all(np.array_equal(old[k], np.stack([getattr(e, k) for e in target.exposures]),
                                          equal_nan=True) for k in ('science', 'error', 'valid'))
        if not matched:
            continue
        with np.load(artifact) as old:
            p = old['parameters'].copy()
            if (int(old['source_id']) != target.source_id or len(p) != len(model.bounds)
                    or old['source_flux'].size != model.engine.line_map.east.size):
                continue
        if not np.isfinite(p).all():
            continue
        p[5] = 0.
        return p, dict(path=str(artifact), sha256=sha256(artifact), role='Initializer only; refitted under new target')
    return None, None


def optimize_starts(model, target, output, report, maxiter):
    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize
    engine = model.engine
    inc = float(engine.inclination_grid[np.argmax(np.asarray(engine.inclination_logpdf))])
    starts = []
    saved, record = historical_seed(model, target, output.parent)
    if saved is not None:
        starts.append(('verified_previous_zero_sigma', saved))
        shifted = saved.copy()
        shifted[0], shifted[2], shifted[1] = 6., shifted[2]+np.pi/2, np.log(.4)
        starts.append(('previous_map_separated_kinematics', shifted))
        report['historical_initializer'] = record
    for v, theta in ((80., 0.), (250., np.pi/2), (500., np.pi), (850., -np.pi/2)):
        x = engine.initial(theta, inc)
        x[4] = 0.
        starts.append((f'fresh_v{v:g}_angle{np.rad2deg(theta):g}', model.initial(x, v)))
    free = np.delete(np.arange(len(model.bounds)), model.sigma_index)
    bounds = [model.bounds[i] for i in free]
    def unpack(q):
        return jnp.zeros(len(model.bounds)).at[free].set(q)
    vg = jax.jit(jax.value_and_grad(lambda q: model.objective(unpack(q))))
    def function(q):
        val, grad = vg(jnp.asarray(q))
        val, grad = float(val), np.asarray(grad)
        if not np.isfinite(val) or not np.isfinite(grad).all():
            raise FloatingPointError('Nonfinite rotation objective or gradient.')
        return val, grad
    report.setdefault('fit_candidates', [])
    for i, (label, p) in enumerate(starts):
        if i < len(report['fit_candidates']):
            continue
        print(f'Fit initialization {i+1}/{len(starts)}: {label}; maximum {maxiter} optimizer iterations.', flush=True)
        tick = time.monotonic()
        count = [0]
        def callback(q):
            count[0] += 1
            if count[0] % 100 == 0:
                print(f'  initialization {i+1}: optimizer iteration {count[0]}/{maxiter}; '
                      f'{time.monotonic()-tick:.1f}s elapsed', flush=True)
        with heartbeat(f'Optimizing start {i+1}/{len(starts)} (first evaluation may compile)'):
            result = minimize(function, p[free], jac=True, bounds=bounds, method='L-BFGS-B',
                              callback=callback, options=dict(maxiter=maxiter, ftol=1e-12, gtol=.03, maxls=40))
        p = np.asarray(unpack(jnp.asarray(result.x)))
        arrays, description = prediction(model, target, parameters=p)
        artifact = f'fit_{i:04d}.npz'
        atomic_npz(output/artifact, **arrays)
        projected = result.jac.copy()
        for j, (low, high) in enumerate(bounds):
            if ((low is not None and result.x[j] <= low+1e-8 and projected[j] > 0) or
                    (high is not None and result.x[j] >= high-1e-8 and projected[j] < 0)):
                projected[j] = 0.
        entry = description | dict(start=label, objective=float(result.fun), iterations=int(result.nit),
            optimizer_success=bool(result.success), projected_gradient_inf=float(np.max(abs(projected))),
            message=str(result.message), artifact=artifact, artifact_sha256=sha256(output/artifact))
        report['fit_candidates'].append(entry)
        report['best_fit'] = min(report['fit_candidates'], key=lambda f: f['objective'])
        write_json(output/'rotation.json', clean_json(report))
        print(f'  Vrot(0.5 arcsec)={description["vrot_0p5arcsec_kms"]:.1f} km/s; '
              f'objective={result.fun:.3f}; projected gradient={entry["projected_gradient_inf"]:.3g}', flush=True)


def chain_initial_values(model, report, output, chains, seed):
    """Separated optimizer modes where competitive; independent perturbations."""
    rng = np.random.default_rng(seed)
    candidates = sorted(report['fit_candidates'], key=lambda f: f['objective'])
    competitive = [f for f in candidates if f['objective'] <= candidates[0]['objective']+25.]
    values, records = [], []
    for chain in range(chains):
        selected = competitive[chain % len(competitive)]
        artifact = output/selected['artifact']
        if (not artifact.resolve().is_relative_to(output.resolve()) or not artifact.is_file()
                or sha256(artifact) != selected.get('artifact_sha256')):
            raise ValueError('Optimizer initialization artifact is missing or changed.')
        with np.load(artifact, allow_pickle=False) as saved:
            if int(saved['source_id']) != report['source_id']:
                raise ValueError('Optimizer initialization belongs to a different source.')
            p = saved['parameters'].copy()
        p[0] *= np.exp(rng.normal(0, .08))
        p[1] += rng.normal(0, .10)
        p[2] += rng.normal(0, .025)
        p[3] += rng.normal(0, .025)
        p[6:8] += rng.normal(0, .05, 2)
        p[11:11+model.engine.nmap] += rng.normal(0, .015, model.engine.nmap)
        v = model.posterior_initial_values(p)
        # Uniform angular endpoints are excluded from finite HMC coordinates.
        v['theta'] = np.clip(v['theta'], -np.pi+1e-6, np.pi-1e-6)
        v['emission_angle'] = np.clip(v['emission_angle'], -np.pi/2+1e-6, np.pi/2-1e-6)
        values.append({k: np.asarray(x) for k, x in v.items()})
        records.append(dict(chain=chain, start_artifact=selected['artifact'],
            vrot_kms=float(v['vrot']), inclination_deg=float(np.rad2deg(v['inclination'])),
            optimizer_delta_objective=float(selected['objective']-candidates[0]['objective'])))
    return values, records


def initialize_chains(model, target, output, report, chains, seed, maxiter):
    """Load committed initial positions or regenerate an interrupted initial write.

    The report checksum commits initial_values.pkl. A file without that checksum
    is an orphan from an interruption before commit; deterministic initialization
    may replace it using the already checkpointed optimizer candidates.
    """
    path = output/'initial_values.pkl'
    if report.get('initial_values_sha256'):
        if not path.is_file() or sha256(path) != report['initial_values_sha256']:
            raise ValueError('Initial chain states are missing or changed.')
        with path.open('rb') as stream:
            return pickle.load(stream)
    optimize_starts(model, target, output, report, maxiter)
    with heartbeat('Preparing separated chain initial positions'):
        initial, report['starts'] = chain_initial_values(model, report, output, chains, seed)
    temporary = output/'initial_values.tmp'
    with temporary.open('wb') as stream:
        pickle.dump(initial, stream, protocol=5)
    temporary.replace(path)
    report['initial_values_sha256'] = sha256(path)
    write_json(output/'rotation.json', clean_json(report))
    return initial


def save_checkpoint(output, state, chunks, identity, iterations):
    """Commit a state and completed draw chunks before publishing their pointer."""
    import jax
    name = f'state_{iterations:08d}.pkl'
    temporary = output/(name+'.tmp')
    with temporary.open('wb') as stream:
        pickle.dump(dict(identity=identity, state=jax.device_get(state)), stream, protocol=5)
    temporary.replace(output/name)
    previous = read_json(output/'checkpoint.json') if (output/'checkpoint.json').exists() else None
    write_json(output/'checkpoint.json', dict(state=name, state_sha256=sha256(output/name),
               iterations=iterations, chunks=chunks, identity=identity))
    if previous and previous['state'] != name:
        (output/previous['state']).unlink(missing_ok=True)


def restore_checkpoint(output, identity):
    checkpoint = read_json(output/'checkpoint.json')
    if checkpoint['identity'] != identity:
        raise ValueError('Checkpoint target/settings differ. Choose a new run name.')
    path = output/checkpoint['state']
    if not path.resolve().is_relative_to(output.resolve()) or sha256(path) != checkpoint['state_sha256']:
        raise ValueError('Invalid or changed local sampler checkpoint.')
    # Only worker-created, checksummed checkpoints in this run are unpickled.
    with path.open('rb') as stream:
        saved = pickle.load(stream)
    if saved['identity'] != identity:
        raise ValueError('Sampler-state identity mismatch.')
    for part in checkpoint['chunks']:
        path = output/part['artifact']
        if not path.resolve().is_relative_to(output.resolve()) or sha256(path) != part['sha256']:
            raise ValueError('Saved posterior chunk changed.')
    return saved['state'], checkpoint['chunks']


def collected_chunks(output, chunks):
    sample_parts, stat_parts = {}, {}
    for part in chunks:
        with np.load(output/part['artifact']) as saved:
            for key in saved.files:
                dest, name = (stat_parts, key[6:]) if key.startswith('stats_') else (sample_parts, key[7:])
                dest.setdefault(name, []).append(saved[key])
    return ({k: np.concatenate(v, axis=1) for k, v in sample_parts.items()},
            {k: np.concatenate(v, axis=1) for k, v in stat_parts.items()})


def publish_draws(output, report, chunks, model, target, max_depth):
    samples, stats = collected_chunks(output, chunks)
    if not samples:
        return
    atomic_npz(output/'samples.npz', **samples)
    atomic_npz(output/'sampler_stats.npz', **stats)
    report.update(samples_sha256=sha256(output/'samples.npz'),
                  sampler_stats_sha256=sha256(output/'sampler_stats.npz'),
                  retained_draws_per_chain=samples['vrot'].shape[1])
    report['diagnostics'] = convergence_diagnostics(samples, stats, max_depth)
    report['chain_predictions'] = []
    for c in range(samples['vrot'].shape[0]):
        # A real middle draw, never a synthetic vector of marginal medians.
        draw = samples['vrot'].shape[1]//2
        arrays, description = prediction(model, target, sites={k: v[c, draw] for k, v in samples.items()})
        artifact = f'chain_{c}_prediction.npz'
        atomic_npz(output/artifact, **arrays, chain=np.asarray(c), draw=np.asarray(draw))
        report['chain_predictions'].append(description | dict(chain=c, draw=draw,
            artifact=artifact, artifact_sha256=sha256(output/artifact)))
    write_json(output/'rotation.json', clean_json(report))


def run(args):
    import jax
    jax.config.update('jax_enable_x64', True)
    import numpyro
    import torch
    from .rotation_data import prepare
    from .rotation_sampling import RotationSampler
    torch.set_num_threads(4)
    if args.device == 'cuda' and jax.default_backend() != 'gpu':
        raise RuntimeError('GPU requested but JAX has no GPU backend. Use the grism environment or device="cpu".')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    jax.config.update('jax_compilation_cache_dir', str(output/'jax_cache'))
    checkpoint_exists = (output/'rotation.json').exists()
    if checkpoint_exists and not args.resume:
        raise ValueError('This run already exists. Use resume=True or choose another run name.')
    print(f'Backend: {jax.default_backend()}; {jax.devices()}; NumPyro {numpyro.__version__}', flush=True)
    print('Preparing calibrated native R+C exposures and the accepted inclination prior.', flush=True)
    with heartbeat('Preparing real observations'):
        model, target, metadata = prepare(args.source_id, args.bundle, config=args.config,
            imaging=args.imaging, inclination=args.inclination, results=output.parents[2])
    settings = metadata.pop('settings', {}) | dict(chains=args.chains, warmup=args.warmup,
        seed=args.seed, max_tree_depth=args.max_tree_depth, target_accept=args.target_accept,
        continuum_sweeps=model.continuum_sweeps, optimizer_maxiter=args.optimizer_maxiter)
    code_hashes = metadata.pop('code_sha256', {})
    for name in ('rotation_fit', 'rotation_sampling', 'rotation_model'):
        code_hashes[f'dingo/{name}.py'] = sha256(REPO/f'dingo/{name}.py')
    versions = dict(jax=jax.__version__, numpyro=numpyro.__version__, numpy=np.__version__)
    identity_object = dict(source_id=args.source_id, settings=settings,
        array_sha256=metadata['array_sha256'], code_sha256=code_hashes,
        input_sha256=metadata.get('input_sha256', {}), versions=versions)
    identity = hashlib.sha256(json.dumps(clean_json(identity_object), sort_keys=True).encode()).hexdigest()
    if checkpoint_exists:
        report = read_json(output/'rotation.json')
        if report.get('identity') != identity:
            raise ValueError('Data, code, prior, library versions or adaptation settings changed. Use a new run name.')
        if report.get('completed') and report.get('retained_draws_per_chain', 0) >= args.samples:
            print('The requested run is already saved. Opening existing results; no new sampling.', flush=True)
            return
    else:
        target.save(output/'native_data.npz')
        report = metadata | dict(format='dingo_rotation_v1', source_id=args.source_id,
            completed=False, phase='initializing', settings=settings, identity=identity,
            code_sha256=code_hashes, versions=versions, started_unix=time.time(),
            model=dict(intrinsic_sigma_kms=0, emission='One flexible positive gas map shared by all R+C exposures',
                sampler='NumPyro NUTS within Gibbs; positive continuum sampled conditionally',
                priors=model.prior_description()),
            retained_draws_per_chain=0, diagnostics=dict(convergence_gate_passed=False),
            native_data_sha256=sha256(output/'native_data.npz'))
        write_json(output/'rotation.json', clean_json(report))
    report.update(completed=False, requested_draws_per_chain=args.samples)
    initial = initialize_chains(model, target, output, report, args.chains, args.seed, args.optimizer_maxiter)
    state, chunks = (restore_checkpoint(output, identity) if (output/'checkpoint.json').exists() else (None, []))
    with heartbeat('Initializing sampler and compiling model gradients'):
        sampler = RotationSampler(model, chains=args.chains, warmup=args.warmup,
            target_accept=args.target_accept, max_tree_depth=args.max_tree_depth,
            seed=args.seed, initial_values=initial, state=state)
    if state is None:
        save_checkpoint(output, sampler.state, chunks, identity, sampler.iterations)
    while sampler.iterations < args.warmup+args.samples:
        phase = 'warmup' if sampler.iterations < args.warmup else 'sampling'
        limit = args.warmup if phase == 'warmup' else args.warmup+args.samples
        n = min(args.chunk, limit-sampler.iterations)
        current = sampler.iterations if phase == 'warmup' else sampler.iterations-args.warmup
        total = args.warmup if phase == 'warmup' else args.samples
        report.update(phase=phase, current_iteration=sampler.iterations, updated_unix=time.time())
        write_json(output/'rotation.json', clean_json(report))
        print(f'{phase.capitalize()}: {current}/{total} per chain; advancing {n} steps '
              f'on {args.chains} chains. First block of each phase may compile.', flush=True)
        start = time.monotonic()
        with heartbeat(f'{phase.capitalize()} block starting at {current}/{total}'):
            draws, stats = sampler.advance(n, collect=phase == 'sampling')
        elapsed = time.monotonic()-start
        if phase == 'sampling':
            artifact = f'draws_{len(chunks):05d}.npz'
            atomic_npz(output/artifact, **{f'sample_{k}': v for k, v in draws.items()},
                       **{f'stats_{k}': v for k, v in stats.items()})
            chunks.append(dict(artifact=artifact, sha256=sha256(output/artifact), draws=n))
        save_checkpoint(output, sampler.state, chunks, identity, sampler.iterations)
        report.update(current_iteration=sampler.iterations, step_sizes=sampler.step_sizes.tolist())
        if phase == 'sampling':
            with heartbeat('Saving posterior draws and checking all sampled parameters'):
                publish_draws(output, report, chunks, model, target, args.max_tree_depth)
        else:
            write_json(output/'rotation.json', clean_json(report))
        print(f'  {phase}: {current+n}/{total} ({100*(current+n)/total:.1f}%) per chain; '
              f'{elapsed:.1f}s block, {elapsed/n:.2f}s/step; '
              f'accept={np.mean(stats["accept_prob"]):.3f}, '
              f'divergences={int(np.sum(stats["diverging"]))}; checkpoint saved.', flush=True)
    # A crash between committing a chunk and publishing combined arrays is recoverable.
    if report.get('retained_draws_per_chain', 0) != sum(c['draws'] for c in chunks):
        publish_draws(output, report, chunks, model, target, args.max_tree_depth)
    report.update(completed=True, phase='complete', finished_unix=time.time())
    write_json(output/'rotation.json', clean_json(report))
    print(f'Finished. Sampling checks passed: {report["diagnostics"]["convergence_gate_passed"]}. '
          f'Results: {output}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-id', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--config', type=Path)
    p.add_argument('--imaging', type=Path)
    p.add_argument('--inclination', choices=('imaging', 'isotropic'), default='imaging')
    p.add_argument('--warmup', type=int, default=1000)
    p.add_argument('--samples', type=int, default=2000)
    p.add_argument('--chains', type=int, default=4)
    p.add_argument('--chunk', type=int, default=50)
    p.add_argument('--seed', type=int, default=1018)
    p.add_argument('--max-tree-depth', type=int, default=9)
    p.add_argument('--target-accept', type=float, default=.9)
    p.add_argument('--optimizer-maxiter', type=int, default=1200)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if min(args.source_id, args.warmup, args.samples, args.chains, args.chunk,
           args.max_tree_depth, args.optimizer_maxiter) < 1 or not .5 < args.target_accept < 1 or args.max_tree_depth > 12:
        p.error('Positive counts, .5 < target_accept < 1 and max_tree_depth <= 12 are required.')
    os.environ.setdefault('JAX_PLATFORMS', 'cuda' if args.device == 'cuda' else 'cpu')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'worker.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('This source/run already has an active worker. See run.log.') from exc
        run(args)


if __name__ == '__main__':
    main()
