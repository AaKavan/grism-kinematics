"""Checkpointed multi-chain NUTS for a saved real-source joint R+C grid."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import pickle
import threading
import time

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_FLAGS', '--xla_gpu_enable_triton_gemm=false')

import numpy as np
import torch
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import initialize_model, unconstrain_fn

from diagnostics.native_source_grid import build_engine, CODE as GRID_CODE
from diagnostics.native_posterior_check import diagnostics
from dingo.native_real_posterior import RealNativePosterior, integrated_positive_amplitudes
from dingo.native_posterior import radial_profiles
from dingo.real_workflow import REPO, read_json, write_json, sha256

CODE = list(dict.fromkeys(GRID_CODE + ['dingo/native_real_posterior.py',
    'diagnostics/native_real_posterior.py', 'diagnostics/native_posterior_check.py']))


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_report(path, report):
    write_json(path, finite_json(report))


def save_npz(path, **arrays):
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


def checkpoint(path, state, samples, stats):
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as stream:
        pickle.dump(dict(state=jax.device_get(state), samples=samples, stats=stats), stream)
    temp.replace(path)


@contextmanager
def heartbeat(label):
    done = threading.Event()
    start = time.perf_counter()
    def pulse():
        while not done.wait(20):
            print(f'{label}: still running ({time.perf_counter()-start:.0f} s)', flush=True)
    worker = threading.Thread(target=pulse, daemon=True); worker.start()
    try:
        yield
    finally:
        done.set(); worker.join()


def starting_branches(grid):
    good = [f for f in grid['fits'] if f.get('stationarity_passed', False)]
    if not good:
        raise ValueError('No stationary grid solution is available for initialization.')
    best = min(good, key=lambda f: f['objective'])
    # Prefer the displaced-center, high-speed branch when the search found it.
    distinct = [f for f in good if abs(f['vrot_0p5arcsec_kms']-best['vrot_0p5arcsec_kms']) >= 75
                and np.linalg.norm(np.array(f['center_east_north_arcsec'])-best['center_east_north_arcsec']) >= .15]
    if not distinct:
        distinct = [f for f in good if abs(f['vrot_0p5arcsec_kms']-best['vrot_0p5arcsec_kms']) >= 75]
    if not distinct:
        raise ValueError('The grid needs a second well-separated stationary starting solution.')
    return best, min(distinct, key=lambda f: f['objective'])


def summarize(output, samples, extra, report):
    from astropy.cosmology import Planck18
    z = report['source_summary']['z']
    scale = float(Planck18.kpc_proper_per_arcmin(z).value/60)
    radii = np.array([.2, .3, .4, .5, .6, .8])
    profiles = radial_profiles(samples['vrot']*np.sin(samples['inclination']),
        np.exp(samples['log_rturn']), samples['inclination'], radii, kpc_per_arcsec=scale)
    save_npz(output/'samples.npz', **samples)
    save_npz(output/'sampler_stats.npz', **extra)
    save_npz(output/'profiles.npz', radii_arcsec=radii, **profiles)
    # Check map and registration coefficients too, not just attractive scalars.
    checked = samples | {k: v for k, v in profiles.items() if k != 'mass_proxy_arcsec_kms2'}
    checked['sin_theta'] = np.sin(samples['theta'])
    checked['cos_theta'] = np.cos(samples['theta'])
    diag = diagnostics(checked, extra, report['settings']['max_tree_depth'])
    report['diagnostics'] = diag
    report['retained_draws_per_chain'] = samples['vrot'].shape[1]
    report['distance'] = dict(cosmology='Planck18', redshift=z, kpc_per_arcsec=scale)
    report['radii_arcsec'] = radii.tolist()
    scalars = {k: v for k, v in (samples | profiles).items()
               if k not in ('map_z', 'registration_z', 'native_pixels', 'external_inclination')}
    report['per_chain_quantiles_5_50_95'] = {
        k: np.quantile(v, [.05, .5, .95], axis=1).swapaxes(0, 1).tolist() for k, v in scalars.items()}
    report['pooled_quantiles_5_50_95'] = ({
        k: np.quantile(v.reshape(-1, *v.shape[2:]), [.05, .5, .95], axis=0).tolist()
        for k, v in scalars.items()} if diag['convergence_gate_passed'] else None)
    report['velocity_near_upper_bound_by_chain'] = (samples['vrot'] > .95*report['settings']['vmax']).mean(1).tolist()
    save_report(output/'posterior.json', report)
    from diagnostics.native_real_posterior_plot import plot
    plot(output)


def save_representative_predictions(output, posterior, samples, likelihood):
    """Actual final draws, with conditional-mean amplitudes; not uncertainty maps."""
    from jax.scipy.special import log_ndtr
    render = jax.jit(posterior.render)
    for chain in range(samples['vrot'].shape[0]):
        p = {k: jnp.asarray(v[chain, -1]) for k, v in samples.items()}
        x = posterior.vector(p)
        unit, flux, velocity = render(x, p['vrot'])
        like = posterior.engine.likelihood
        template = like.residualize(jnp.where(like.valid, unit, 0.)*like.inverse_error)
        _, loc, sd = integrated_positive_amplitudes(template, like.residual_data, posterior.amplitude_scale)
        a = loc/sd
        amplitude = loc+sd*jnp.exp(-.5*a*a-.5*np.log(2*np.pi)-log_ndtr(a))
        line = np.asarray(unit*amplitude[:, None, None])
        continuum = likelihood.continuum(torch.from_numpy(line.copy())).numpy()
        residual = np.where(likelihood.valid.numpy(),
                            (likelihood.data.numpy()-line-continuum)*likelihood.inverse_error.numpy(), np.nan)
        save_npz(output/f'chain_{chain}_last_prediction.npz', vector=np.asarray(x),
            vrot=np.asarray(p['vrot']), line=line, continuum=continuum, residual=residual,
            source_flux=np.asarray(flux), source_velocity=np.asarray(velocity),
            amplitude_conditional_mean=np.asarray(amplitude), chain=np.asarray(chain),
            retained_draw_index=np.asarray(samples['vrot'].shape[1]-1))


def run_locked(args):
    output = args.output.resolve(); grid_dir = args.grid.resolve()
    grid = read_json(grid_dir/'grid.json')
    if grid.get('real_observations') is not True or grid['source_id'] != args.source_id:
        raise ValueError('A real grid belonging to the requested source is required.')
    settings = {k: v for k, v in vars(args).items() if k not in ('output', 'grid', 'resume', 'samples', 'audit_only', 'device')}
    settings['grid'] = str(grid_dir)
    code_hashes = {p: sha256(REPO/p) for p in CODE}
    input_hashes = {str(grid_dir/p): sha256(grid_dir/p) for p in ('grid.json', 'native_data.npz', 'inclination_prior.npz')}
    versions = dict(jax=jax.__version__, numpyro=numpyro.__version__, torch=torch.__version__)
    previous = None
    if (output/'posterior.json').exists():
        if not args.resume:
            raise FileExistsError('Use --resume for this unchanged run, or choose a new output directory.')
        previous = read_json(output/'posterior.json')
        for key, value in [('settings', settings), ('code_sha256', code_hashes),
                           ('input_sha256', input_hashes), ('versions', versions)]:
            if previous[key] != value:
                raise ValueError(f'Resume requires unchanged {key}; use a fresh run.')
        if not previous.get('warmup_completed'):
            raise ValueError('No completed warmup checkpoint. Use a fresh run.')
        if args.samples <= previous.get('retained_draws_per_chain', 0):
            raise ValueError('Resume must request more total draws than are already retained.')
    elif args.resume:
        raise FileNotFoundError('No posterior checkpoint exists to resume.')
    target, engine, likelihood, _, _, components = build_engine(grid)
    branches = starting_branches(grid)
    with np.load(grid_dir/branches[0]['artifact']) as a:
        anchor = float(a['vector'][1])
    posterior = RealNativePosterior(engine, vmax=args.vmax, amplitude_scale=args.amplitude_scale,
        center_kind=grid['settings'].get('center', 'catalog'),
        center_mean=grid['center_mean_east_north_arcsec'], angle_anchor=anchor)
    starts, initial_values = [], []
    rng = np.random.default_rng(args.seed)
    for c in range(args.chains):
        fit = branches[c % 2]
        with np.load(grid_dir/fit['artifact']) as a:
            initial = posterior.initial_values(a['vector'], fit['vrot_0p5arcsec_kms'])
        initial['map_z'] += rng.normal(0., .01, engine.nmap)
        initial['theta'] += rng.normal(0., .002)
        initial_values.append(initial)
        starts.append(dict(chain=c, branch='best grid' if c % 2 == 0 else 'alternative grid',
            grid_velocity=fit['vrot_0p5arcsec_kms'], artifact=fit['artifact'],
            artifact_sha256=sha256(grid_dir/fit['artifact']),
            center_arcsec=fit['center_east_north_arcsec'], grid_objective=fit['objective']))
    report = dict(format='native_real_posterior_v1', source_id=target.source_id,
        real_observations=True, completed=False, warmup_completed=False, source_summary=target.summary,
        settings=settings, requested_draws_per_chain=args.samples, code_sha256=code_hashes,
        input_sha256=input_hashes, versions=versions, starts=starts,
        prior=posterior.prior_description(), grid_settings=grid['settings'],
        array_sha256=grid['array_sha256'], device=str(jax.devices()),
        n_valid_pixels=grid['n_valid_pixels'], n_exposures=len(target.exposures),
        limitations=['Conditional on saved response, errors/masks, map variance/correlation and finite representation',
            'Imaging inclination is a provisional transferred prior; not a joint pixel-level stellar/gas likelihood',
            'Line amplitudes integrated with explicit HalfNormal prior; target differs from the profile objective',
            'Continuum integrated with fixed flat measure; no absolute model evidence',
            'Rotation-only spherical-equivalent mass proxy; no pressure-support or lensing correction',
            'Illustrative radii do not establish resolved radial information or detected outer disk',
            'Multiple chains and passing diagnostics cannot prove correct weights of disconnected modes'])
    if previous:
        if starts != previous['starts']:
            raise ValueError('Starting artifacts changed since the saved run.')
        report = previous
        report['requested_draws_per_chain'] = args.samples
        report['completed'] = False
    else:
        for name in CODE:
            path = output/'code_snapshot'/name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((REPO/name).read_bytes())
    save_report(output/'posterior.json', report)
    print(f'Real ID{target.source_id}: {len(target.exposures)} exposures, '
          f'{engine.nmap} sampled map coefficients; initial speeds {[s["grid_velocity"] for s in starts]}', flush=True)
    audit = []
    render = jax.jit(posterior.render)
    with heartbeat('Forward-model audit'):
        for fit in branches:
            with np.load(grid_dir/fit['artifact']) as a:
                x = jnp.asarray(a['vector'])
            speed = float(fit['vrot_0p5arcsec_kms'])
            _, parts = components(x, speed)
            unit, flux, velocity = render(x, speed)
            difference = (np.asarray(unit*parts[3][:, None, None])-np.asarray(parts[4]))
            max_error = float(np.max(abs(difference)*likelihood.inverse_error.numpy()))
            if max_error > 1e-8 or not np.allclose(flux, parts[5], atol=1e-13) or not np.allclose(velocity, parts[6], atol=1e-10):
                raise AssertionError('Posterior forward model disagrees with the saved grid operator.')
            audit.append(dict(speed=speed, max_pixel_difference_in_error_units=max_error))
    with heartbeat('Potential and gradient compilation'):
        info = initialize_model(jax.random.key(args.seed), posterior.model,
                                init_strategy=init_to_value(values=initial_values[0]))
        initial_params = [unconstrain_fn(posterior.model, (), {}, v) for v in initial_values]
        initial_params = jax.tree.map(lambda *v: jnp.stack(v), *initial_params)
        q = jax.tree.map(lambda x: x[0], initial_params)
        vg = jax.jit(jax.value_and_grad(info.potential_fn))
        value, gradient = jax.block_until_ready(vg(q))
        if not all(np.isfinite(v).all() for v in jax.tree.leaves((value, gradient))):
            raise ValueError('Nonfinite initial potential or gradient.')
        start = time.perf_counter()
        for _ in range(10):
            jax.block_until_ready(vg(q))
        seconds = (time.perf_counter()-start)/10
    report['forward_audit'] = audit
    report['gradient_seconds'] = seconds
    report['sampled_dimension'] = sum(v[0].size for v in jax.tree.leaves(initial_params))
    save_report(output/'posterior.json', report)
    print(f'Forward audit passed; {report["sampled_dimension"]} dimensions; '
          f'single-chain gradient {seconds:.4f} s', flush=True)
    if args.audit_only:
        report['audit_completed'] = True; save_report(output/'posterior.json', report)
        return
    kernel = NUTS(posterior.model, dense_mass=[posterior.scalar_sites],
                  target_accept_prob=args.target_accept, max_tree_depth=args.max_tree_depth)
    sampler = MCMC(kernel, num_warmup=args.warmup, num_samples=args.chunk,
                   num_chains=args.chains, chain_method='vectorized', progress_bar=False)
    fields = ('potential_energy', 'energy', 'num_steps', 'accept_prob', 'diverging')
    samples, extra, completed = {}, {}, 0
    if previous:
        # Only deserialize this run's own explicitly requested local checkpoint,
        # after source, input, code, settings and version checks above.
        with (output/'sampler_state.pkl').open('rb') as stream:
            state = pickle.load(stream)
        sampler.post_warmup_state = jax.device_put(state['state'])
        samples, extra = state['samples'], state['stats']
        completed = samples['vrot'].shape[1] if samples else 0
        if args.samples <= completed:
            raise ValueError('Resume must request more total draws than are already retained.')
    else:
        print(f'Warmup: {args.chains} chains × {args.warmup} steps; depth {args.max_tree_depth}', flush=True)
        start = time.perf_counter()
        with heartbeat('NUTS warmup'):
            sampler.warmup(jax.random.key(args.seed+1), init_params=initial_params,
                           extra_fields=fields, collect_warmup=True)
        warmup = {k: np.asarray(v) for k, v in sampler.get_samples(group_by_chain=True).items()}
        save_npz(output/'warmup.npz', **warmup)
        checkpoint(output/'sampler_state.pkl', sampler.post_warmup_state, {}, {})
        report['warmup_completed'] = True
        report['warmup_seconds'] = time.perf_counter()-start
        report['adapted_step_size_by_chain'] = np.asarray(sampler.post_warmup_state.adapt_state.step_size).tolist()
        save_report(output/'posterior.json', report)
    while completed < args.samples:
        print(f'Retaining draws {completed+1}–{completed+args.chunk} / {args.samples} per chain', flush=True)
        start = time.perf_counter()
        with heartbeat('NUTS retained block'):
            sampler.run(sampler.post_warmup_state.rng_key, extra_fields=fields,
                        init_params=sampler.post_warmup_state.z)
        current = {k: np.asarray(v) for k, v in sampler.get_samples(group_by_chain=True).items()}
        current_extra = {k: np.asarray(v) for k, v in sampler.get_extra_fields(group_by_chain=True).items()}
        samples = {k: np.concatenate([samples[k], v], axis=1) if k in samples else v for k, v in current.items()}
        extra = {k: np.concatenate([extra[k], v], axis=1) if k in extra else v for k, v in current_extra.items()}
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


def run(args):
    if args.chains < 2 or args.warmup < 1 or args.chunk < 4 or args.samples < args.chunk or args.samples % args.chunk:
        raise ValueError('Use >=2 chains, positive warmup and samples in whole chunks of >=4.')
    if not 1 <= args.max_tree_depth <= 12 or not .5 < args.target_accept < 1:
        raise ValueError('Invalid tree depth or acceptance target.')
    torch.set_num_threads(4)
    jax.config.update('jax_compilation_cache_dir', str(REPO.parent/'results/_jax_cache'))
    if jax.default_backend() != ('gpu' if args.device == 'cuda' else 'cpu'):
        raise RuntimeError(f'Requested {args.device}, available devices: {jax.devices()}')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'.posterior.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another worker is already using this posterior directory.') from exc
        run_locked(args)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grid', type=Path, required=True)
    p.add_argument('--source-id', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--warmup', type=int, default=800)
    p.add_argument('--samples', type=int, default=1000)
    p.add_argument('--chunk', type=int, default=100)
    p.add_argument('--chains', type=int, default=4)
    p.add_argument('--max-tree-depth', type=int, default=9)
    p.add_argument('--target-accept', type=float, default=.9)
    p.add_argument('--vmax', type=float, default=1000.)
    p.add_argument('--amplitude-scale', type=float, default=100.)
    p.add_argument('--seed', type=int, default=716)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--audit-only', action='store_true')
    run(p.parse_args())
