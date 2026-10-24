"""Longer NUTS trajectories from an unchanged real posterior's adapted state.

Save a fresh retained sample set; never pool a failed pilot into production
intervals. The original run and its diagnostics remain intact.
"""
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import pickle
import time

from diagnostics.native_real_posterior import (heartbeat, checkpoint, save_report,
    summarize, save_representative_predictions, finite_json)
from diagnostics.native_source_grid import build_engine
from dingo.native_real_posterior import RealNativePosterior
from dingo.real_workflow import REPO, read_json, sha256
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import initialize_model
from numpyro.infer.initialization import init_to_value
import torch


def run(args):
    if args.chunk < 4 or args.samples < args.chunk or args.samples % args.chunk:
        raise ValueError('Retained samples must be whole chunks of at least four draws.')
    if not 1 <= args.max_tree_depth <= 12:
        raise ValueError('Invalid tree depth.')
    torch.set_num_threads(4)
    jax.config.update('jax_compilation_cache_dir', str(REPO.parent/'results/_jax_cache'))
    if jax.default_backend() != ('gpu' if args.device == 'cuda' else 'cpu'):
        raise RuntimeError(f'Requested {args.device}, available {jax.devices()}')
    output = args.output.resolve(); parent = args.from_posterior.resolve()
    if parent == output:
        raise ValueError('Use a separate output for the deeper-trajectory sample set.')
    output.mkdir(parents=True, exist_ok=True)
    with (output/'.posterior.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('This output is already active.') from exc
        run_locked(args, output, parent)


def run_locked(args, output, parent):
    previous = None
    if (output/'posterior.json').exists():
        if not args.resume:
            raise FileExistsError('Use --resume or a fresh output directory.')
        previous = read_json(output/'posterior.json')
        if (previous['settings']['max_tree_depth'] != args.max_tree_depth
                or previous['settings']['chunk'] != args.chunk
                or previous['adapted_from']['directory'] != str(parent)):
            raise ValueError('Resume must retain trajectory settings and adapted source.')
        if args.samples <= previous.get('retained_draws_per_chain', 0):
            raise ValueError('Resume must request more total retained draws.')
        original = read_json(output/'adapted_source_report.json')
        content = (output/'adapted_initial_state.pkl').read_bytes()
    else:
        if args.resume:
            raise FileNotFoundError('No output checkpoint to resume.')
        # Take one consistent local snapshot. The parent's worker uses this lock.
        with (parent/'.posterior.lock').open('w') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('Wait for the source sampler to finish its checkpoint/run.') from exc
            original = read_json(parent/'posterior.json')
            content = (parent/'sampler_state.pkl').read_bytes()
    if original.get('real_observations') is not True or original['source_id'] != args.source_id:
        raise ValueError('Adapted source must be this source\'s real-data posterior.')
    if not original.get('warmup_completed'):
        raise ValueError('Adapted source has not completed warmup.')
    if original.get('retained_draws_per_chain', 0) < 4:
        raise ValueError('Retain at least one diagnostic block before extending this adapted source.')
    if args.max_tree_depth < original['settings']['max_tree_depth']:
        raise ValueError('Use the same or a larger trajectory limit.')
    for p, digest in original['code_sha256'].items():
        if sha256(REPO/p) != digest:
            raise ValueError(f'Adapted target implementation changed: {p}')
    for p, digest in original['input_sha256'].items():
        if sha256(p) != digest:
            raise ValueError(f'Adapted target input changed: {p}')
    versions = dict(jax=jax.__version__, numpyro=numpyro.__version__, torch=torch.__version__)
    if original['versions'] != versions:
        raise ValueError('Adapted state requires identical library versions.')
    code_hashes = original['code_sha256'] | {
        'diagnostics/native_real_posterior_extend.py': sha256(Path(__file__))}
    if previous and previous['code_sha256'] != code_hashes:
        raise ValueError('Continuation implementation changed since this output checkpoint.')
    s = original['settings']
    grid_dir = Path(s['grid'])
    if args.grid is not None and args.grid.resolve() != grid_dir.resolve():
        raise ValueError('Selected grid is different from the adapted posterior grid.')
    grid = read_json(grid_dir/'grid.json')
    target, engine, likelihood, _, _, _ = build_engine(grid)
    if grid['array_sha256'] != original['array_sha256']:
        raise ValueError('Native target changed since adaptation.')
    with np.load(grid_dir/original['starts'][0]['artifact']) as a:
        anchor = float(a['vector'][1])
    posterior = RealNativePosterior(engine, vmax=s['vmax'], amplitude_scale=s['amplitude_scale'],
        center_kind=grid['settings'].get('center', 'catalog'),
        center_mean=grid['center_mean_east_north_arcsec'], angle_anchor=anchor)
    if posterior.prior_description() != original['prior']:
        raise ValueError('Prior differs from adapted target.')
    if not previous:
        (output/'adapted_initial_state.pkl').write_bytes(content)
        save_report(output/'adapted_source_report.json', original)
    # Only local, explicitly selected checkpoints that passed all checks above.
    restored = pickle.loads((output/'sampler_state.pkl').read_bytes() if previous else content)
    state = jax.device_put(restored['state'])
    samples = restored['samples'] if previous else {}
    extra = restored['stats'] if previous else {}
    completed = samples['vrot'].shape[1] if samples else 0
    report = copy.deepcopy(previous or original)
    if not previous:
        for key in ('diagnostics', 'pooled_quantiles_5_50_95', 'per_chain_quantiles_5_50_95',
                    'retained_draws_per_chain', 'sampling_block_seconds'):
            report.pop(key, None)
        report['settings']['max_tree_depth'] = args.max_tree_depth
        report['settings']['chunk'] = args.chunk
        report['adapted_from'] = dict(directory=str(parent), checkpoint_sha256=hashlib.sha256(content).hexdigest(),
            original_retained_draws=original.get('retained_draws_per_chain', 0),
            original_depth=original['settings']['max_tree_depth'],
            note='Same target and adapted covariance/step sizes; fresh retained sample arrays')
        for name, digest in code_hashes.items():
            dest = output/'code_snapshot'/name; dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((REPO/name).read_bytes())
    report.update(completed=False, requested_draws_per_chain=args.samples, code_sha256=code_hashes)
    # Verify potential and gradient at the restored state before transitioning.
    initial = {k: v[0, -1] for k, v in restored['samples'].items()}
    with heartbeat('Adapted-target potential check'):
        info = initialize_model(jax.random.key(s['seed']), posterior.model,
                                init_strategy=init_to_value(values=initial))
        q = jax.tree.map(lambda x: x[0], state.z)
        value, gradient = jax.block_until_ready(jax.jit(jax.value_and_grad(info.potential_fn))(q))
        difference = abs(float(value-state.potential_energy[0]))
        max_gradient = max(float(np.max(abs(np.asarray(g)-np.asarray(old[0]))))
            for g, old in zip(jax.tree.leaves(gradient), jax.tree.leaves(state.z_grad)))
        if difference > 1e-6 or max_gradient > 1e-5:
            raise ValueError('Restored potential/gradient differs from current target.')
    report['adapted_potential_audit'] = dict(energy_difference=difference, max_gradient_difference=max_gradient)
    save_report(output/'posterior.json', report)
    print(f'Adapted target agrees: energy {difference:g}, gradient {max_gradient:g}; '
          f'depth {args.max_tree_depth}, fresh draws {completed}–{args.samples}', flush=True)
    sampler = MCMC(NUTS(posterior.model, dense_mass=[posterior.scalar_sites],
        target_accept_prob=s['target_accept'], max_tree_depth=args.max_tree_depth),
        num_warmup=s['warmup'], num_samples=args.chunk, num_chains=s['chains'],
        chain_method='vectorized', progress_bar=False)
    sampler.post_warmup_state = state
    fields = ('potential_energy', 'energy', 'num_steps', 'accept_prob', 'diverging')
    while completed < args.samples:
        start = time.perf_counter()
        print(f'Retaining deeper-trajectory draws {completed+1}–{completed+args.chunk} / {args.samples}', flush=True)
        with heartbeat('Deeper NUTS block'):
            sampler.run(sampler.post_warmup_state.rng_key, extra_fields=fields, init_params=sampler.post_warmup_state.z)
        current = {k: np.asarray(v) for k, v in sampler.get_samples(group_by_chain=True).items()}
        stats = {k: np.asarray(v) for k, v in sampler.get_extra_fields(group_by_chain=True).items()}
        samples = {k: np.concatenate([samples[k], v], axis=1) if k in samples else v for k, v in current.items()}
        extra = {k: np.concatenate([extra[k], v], axis=1) if k in extra else v for k, v in stats.items()}
        sampler.post_warmup_state = sampler.last_state
        checkpoint(output/'sampler_state.pkl', sampler.last_state, samples, extra)
        completed += args.chunk
        report.setdefault('sampling_block_seconds', []).append(time.perf_counter()-start)
        summarize(output, samples, extra, report)
        print(json.dumps(finite_json({k: v for k, v in report['diagnostics'].items() if k != 'parameters'})), flush=True)
    save_representative_predictions(output, posterior, samples, likelihood)
    report['completed'] = True
    save_report(output/'posterior.json', report)
    from diagnostics.native_real_posterior_plot import plot
    plot(output)
    print(f'Finished {output}; convergence gate: {report["diagnostics"]["convergence_gate_passed"]}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--from-posterior', type=Path, required=True)
    p.add_argument('--source-id', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--grid', type=Path)
    p.add_argument('--samples', type=int, default=1000)
    p.add_argument('--chunk', type=int, default=100)
    p.add_argument('--max-tree-depth', type=int, default=9)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--resume', action='store_true')
    run(p.parse_args())
