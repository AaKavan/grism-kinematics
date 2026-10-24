"""Fresh adaptation from saved real-posterior positions, with timed checkpoints.

Only starting positions are reused. Old adaptation and old draws are excluded.
The original posterior drivers remain unchanged so their checkpoints can resume.
"""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import io
from pathlib import Path
import pickle
import time

from diagnostics.native_real_posterior import (CODE, heartbeat, checkpoint, save_report,
    save_npz, summarize, save_representative_predictions)
from diagnostics.native_source_grid import build_engine
from dingo.native_real_posterior import RealNativePosterior
from dingo.real_workflow import REPO, read_json, sha256
import jax
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import initialize_model, unconstrain_fn
from numpyro.infer.initialization import init_to_value
import torch

SITES = (*RealNativePosterior.scalar_sites, 'map_z', 'registration_z')


def initial_positions(samples, chains):
    """Keep distinct, finite final positions; never replace other parameters by medians."""
    positions = {}
    shape = samples['vrot'].shape
    if len(shape) != 2 or shape[0] != chains or shape[1] < 4:
        raise ValueError('Initialization needs the same chain count and at least four saved draws.')
    for name in SITES:
        a = np.asarray(samples[name])
        if a.shape[:2] != shape or not np.isfinite(a[:, -1]).all():
            raise ValueError(f'Invalid final chain positions for {name}.')
        positions[name] = a[:, -1].copy()
    return positions


def duration(seconds):
    seconds = int(max(0, seconds))
    return f'{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}'


def progress(completed, total, elapsed, block_seconds):
    width = 24
    filled = int(width*completed/total)
    bar = '#' * filled + '-' * (width-filled)
    remaining_blocks = (total-completed)/block_seconds['chunk']
    # Exclude the first block once a later block is available: compilation distorts it.
    times = block_seconds['times']
    rate = np.median(times[-3:] if len(times) > 1 else times)
    if len(times) > 1:
        rate = np.median(times[max(1, len(times)-3):])
    eta = duration(remaining_blocks*rate)
    print(f'[{bar}] {completed}/{total} draws per chain | '
          f'elapsed this invocation {duration(elapsed)} | last checkpoint {duration(times[-1])} | '
          f'estimated sampling remaining ~{eta} (final rendering extra)', flush=True)


def run(args):
    if args.chains < 2 or args.warmup < 1 or args.chunk < 4 or args.samples < args.chunk or args.samples % args.chunk:
        raise ValueError('Use >=2 chains, positive warmup and whole chunks of >=4 retained draws.')
    if not 1 <= args.max_tree_depth <= 12 or not .5 < args.target_accept < 1:
        raise ValueError('Invalid trajectory depth or target acceptance.')
    torch.set_num_threads(4)
    jax.config.update('jax_compilation_cache_dir', str(REPO.parent/'results/_jax_cache'))
    if jax.default_backend() != ('gpu' if args.device == 'cuda' else 'cpu'):
        raise RuntimeError(f'Requested {args.device}, available {jax.devices()}')
    output = args.output.resolve(); parent = args.initial_posterior.resolve()
    if output == parent:
        raise ValueError('Fresh adaptation needs a separate output directory.')
    output.mkdir(parents=True, exist_ok=True)
    with (output/'.posterior.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('This output already has an active sampler.') from exc
        run_locked(args, output, parent)


def run_locked(args, output, parent):
    started = time.perf_counter()
    print(f'Started {datetime.now(timezone.utc).isoformat()} | device {jax.devices()}', flush=True)
    print(f'Output: {output}', flush=True)
    previous = None
    if (output/'posterior.json').exists():
        if not args.resume:
            raise FileExistsError('Use --resume or a fresh run tag.')
        previous = read_json(output/'posterior.json')
        if previous.get('initialized_from', {}).get('directory') != str(parent):
            raise ValueError('Resume must retain the same initialization source.')
        if not previous.get('warmup_completed'):
            raise ValueError('Warmup was interrupted before checkpointing. Use a fresh run tag.')
        if args.samples <= previous.get('retained_draws_per_chain', 0) and previous.get('completed'):
            raise ValueError('This run is complete. View it or increase the retained draw target.')
        original = read_json(output/'initial_source_report.json')
        with np.load(output/'initial_positions.npz') as a:
            positions = {k: a[k] for k in SITES}
    else:
        if args.resume:
            raise FileNotFoundError('No saved run to resume.')
        with (parent/'.posterior.lock').open('w') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('Initialization source is still running; wait for it to finish.') from exc
            original = read_json(parent/'posterior.json')
            # NPZ values suffice; no adapted sampler state is imported.
            with np.load(io.BytesIO((parent/'samples.npz').read_bytes())) as a:
                positions = initial_positions(a, args.chains)
    if original.get('real_observations') is not True or original['source_id'] != args.source_id:
        raise ValueError('Initialization must use this source\'s real-data posterior.')
    if original['settings']['chains'] != args.chains:
        raise ValueError('Keep the initialization source\'s chain count.')
    grid_dir = args.grid.resolve()
    if grid_dir != Path(original['settings']['grid']).resolve():
        raise ValueError('Initialization must use the same real-data grid.')
    for p, digest in original['code_sha256'].items():
        if sha256(REPO/p) != digest:
            raise ValueError(f'Initialization target code changed: {p}')
    for p, digest in original['input_sha256'].items():
        if sha256(p) != digest:
            raise ValueError(f'Initialization target input changed: {p}')
    versions = dict(jax=jax.__version__, numpyro=numpyro.__version__, torch=torch.__version__)
    if versions != original['versions']:
        raise ValueError('Use the same libraries as the saved posterior.')
    if not previous:
        save_report(output/'initial_source_report.json', original)
        save_npz(output/'initial_positions.npz', **positions)
    code_hashes = original['code_sha256'] | {p: sha256(REPO/p) for p in CODE}
    code_hashes['diagnostics/native_real_posterior_rewarm.py'] = sha256(Path(__file__))
    input_hashes = original['input_sha256'] | {
        str(output/p): sha256(output/p) for p in ('initial_source_report.json', 'initial_positions.npz')}
    settings = copy.deepcopy(original['settings'])
    settings.update(warmup=args.warmup, chains=args.chains, chunk=args.chunk,
                    max_tree_depth=args.max_tree_depth, target_accept=args.target_accept,
                    seed=args.seed, grid=str(grid_dir))
    if previous:
        for key, value in [('settings', settings), ('code_sha256', code_hashes),
                           ('input_sha256', input_hashes), ('versions', versions)]:
            if previous[key] != value:
                raise ValueError(f'Resume requires unchanged {key}; choose a fresh run tag.')
    grid = read_json(grid_dir/'grid.json')
    with heartbeat('Loading native R+C observations'):
        target, engine, likelihood, _, _, _ = build_engine(grid)
    if grid['array_sha256'] != original['array_sha256']:
        raise ValueError('Native input arrays differ from the initialization source.')
    anchor_fit = original['starts'][0]
    if sha256(grid_dir/anchor_fit['artifact']) != anchor_fit['artifact_sha256']:
        raise ValueError('The grid artifact defining the angular coordinates changed.')
    with np.load(grid_dir/anchor_fit['artifact']) as a:
        anchor = float(a['vector'][1])
    posterior = RealNativePosterior(engine, vmax=settings['vmax'], amplitude_scale=settings['amplitude_scale'],
        center_kind=grid['settings'].get('center', 'catalog'),
        center_mean=grid['center_mean_east_north_arcsec'], angle_anchor=anchor)
    if posterior.prior_description() != original['prior']:
        raise ValueError('Fresh adaptation must preserve the original posterior and priors.')
    initial_values = [{k: v[c] for k, v in positions.items()} for c in range(args.chains)]
    with heartbeat('Compiling and checking initial positions'):
        info = initialize_model(jax.random.key(args.seed), posterior.model,
                                init_strategy=init_to_value(values=initial_values[0]))
        initial = [unconstrain_fn(posterior.model, (), {}, p) for p in initial_values]
        initial = jax.tree.map(lambda *a: np.stack(a), *initial)
        value, grad = jax.block_until_ready(jax.jit(jax.vmap(jax.value_and_grad(info.potential_fn)))(initial))
        if not all(np.isfinite(v).all() for v in jax.tree.leaves((value, grad))):
            raise ValueError('Initial positions have a nonfinite potential or gradient.')
    report = copy.deepcopy(previous or original)
    if not previous:
        for key in ('adapted_from', 'adapted_potential_audit', 'diagnostics', 'retained_draws_per_chain',
                    'pooled_quantiles_5_50_95', 'per_chain_quantiles_5_50_95', 'sampling_block_seconds',
                    'warmup_seconds', 'adapted_step_size_by_chain', 'velocity_near_upper_bound_by_chain',
                    'gradient_seconds', 'checkpoint_seconds_including_plots', 'invocation_seconds'):
            report.pop(key, None)
        report.update(format='native_real_posterior_rewarm_v1', warmup_completed=False,
            initialized_from=dict(directory=str(parent), note='Final positions only; new warmup and new retained draws'),
            retained_draws_per_chain=0)
        for c, row in enumerate(report['starts']):
            row['initial_velocity'] = float(positions['vrot'][c])
            row['initial_center_arcsec'] = positions['center'][c].tolist()
        for name in code_hashes:
            dest = output/'code_snapshot'/name; dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((REPO/name).read_bytes())
    report.update(settings=settings, code_sha256=code_hashes, input_sha256=input_hashes,
                  versions=versions, completed=False, requested_draws_per_chain=args.samples, device=str(jax.devices()))
    save_report(output/'posterior.json', report)
    print('Starting velocities [km/s]: '+str(np.round(positions['vrot'], 2).tolist()), flush=True)
    print(f'{args.chains} chains | fresh warmup {args.warmup} each | retained {args.samples} each | '
          f'checkpoint every {args.chunk} | depth {args.max_tree_depth}', flush=True)
    sampler = MCMC(NUTS(posterior.model, dense_mass=[posterior.scalar_sites],
                        target_accept_prob=args.target_accept, max_tree_depth=args.max_tree_depth),
                   num_warmup=args.warmup, num_samples=args.chunk, num_chains=args.chains,
                   chain_method='vectorized', progress_bar=False)
    fields = ('potential_energy', 'energy', 'num_steps', 'accept_prob', 'diverging')
    samples, extra, completed = {}, {}, 0
    if previous:
        # Own local checkpoint, after code/input/settings/library verification.
        with (output/'sampler_state.pkl').open('rb') as stream:
            state = pickle.load(stream)
        sampler.post_warmup_state = jax.device_put(state['state'])
        samples, extra = state['samples'], state['stats']
        completed = samples['vrot'].shape[1] if samples else 0
        if args.samples < completed:
            raise ValueError('Requested total is below the checkpoint\'s saved draw count.')
        print(f'Resuming from {completed} retained draws per chain; warmup already complete.', flush=True)
    else:
        phase = time.perf_counter()
        with heartbeat(f'Fresh warmup ({args.warmup} steps per chain; includes compilation)'):
            sampler.warmup(jax.random.key(args.seed+1), init_params=initial,
                           extra_fields=fields, collect_warmup=True)
        save_npz(output/'warmup.npz', **{k: np.asarray(v) for k, v in sampler.get_samples(group_by_chain=True).items()})
        checkpoint(output/'sampler_state.pkl', sampler.post_warmup_state, {}, {})
        report.update(warmup_completed=True, warmup_seconds=time.perf_counter()-phase,
            adapted_step_size_by_chain=np.asarray(sampler.post_warmup_state.adapt_state.step_size).tolist())
        save_report(output/'posterior.json', report)
        print(f'Fresh warmup complete: {duration(report["warmup_seconds"])} | '
              f'total elapsed {duration(time.perf_counter()-started)}. Adaptation checkpoint saved.', flush=True)
    times = []
    while completed < args.samples:
        phase = time.perf_counter()
        print(f'Retaining {completed+1}–{completed+args.chunk}/{args.samples} draws per chain', flush=True)
        with heartbeat('Sampling current checkpoint block'):
            sampler.run(sampler.post_warmup_state.rng_key, extra_fields=fields, init_params=sampler.post_warmup_state.z)
        current = {k: np.asarray(v) for k, v in sampler.get_samples(group_by_chain=True).items()}
        stats = {k: np.asarray(v) for k, v in sampler.get_extra_fields(group_by_chain=True).items()}
        samples = {k: np.concatenate([samples[k], v], axis=1) if k in samples else v for k, v in current.items()}
        extra = {k: np.concatenate([extra[k], v], axis=1) if k in extra else v for k, v in stats.items()}
        sampler.post_warmup_state = sampler.last_state
        checkpoint(output/'sampler_state.pkl', sampler.last_state, samples, extra)
        completed += args.chunk
        report.setdefault('sampling_block_seconds', []).append(time.perf_counter()-phase)
        with heartbeat('Updating diagnostics and saved plots'):
            summarize(output, samples, extra, report)
        times.append(time.perf_counter()-phase)
        report.setdefault('checkpoint_seconds_including_plots', []).append(times[-1])
        save_report(output/'posterior.json', report)
        progress(completed, args.samples, time.perf_counter()-started, dict(chunk=args.chunk, times=times))
        d = report['diagnostics']
        print(f'Checks: max Rhat {d["max_rank_rhat"]:.3f} | min bulk ESS {d["min_bulk_ess"]:.1f} | '
              f'divergences {d["divergences"]} | trajectory cap {d["max_depth_fraction"]:.1%} | '
              f'passed {d["convergence_gate_passed"]}', flush=True)
    with heartbeat('Saving final chain predictions and diagnostics'):
        summarize(output, samples, extra, report)
        save_representative_predictions(output, posterior, samples, likelihood)
        report['completed'] = True
        report.setdefault('invocation_seconds', []).append(time.perf_counter()-started)
        save_report(output/'posterior.json', report)
        from diagnostics.native_real_posterior_plot import plot
        plot(output)
    print(f'Finished in {duration(time.perf_counter()-started)} this invocation | '
          f'convergence checks passed: {report["diagnostics"]["convergence_gate_passed"]}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--initial-posterior', type=Path, required=True)
    p.add_argument('--grid', type=Path, required=True)
    p.add_argument('--source-id', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--warmup', type=int, default=1500)
    p.add_argument('--samples', type=int, default=2000)
    p.add_argument('--chains', type=int, default=4)
    p.add_argument('--chunk', type=int, default=100)
    p.add_argument('--max-tree-depth', type=int, default=10)
    p.add_argument('--target-accept', type=float, default=.9)
    p.add_argument('--seed', type=int, default=1016)
    p.add_argument('--resume', action='store_true')
    run(p.parse_args())
