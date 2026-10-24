"""Checkpointed fixed-dispersion refits of real R+C exposures.

The objective is the saved velocity grid's penalized, amplitude-profiled
objective, not the amplitude-marginalized MCMC target. No posterior intervals
or pressure-supported masses are inferred by this diagnostic.
"""
import argparse
import copy
import fcntl
from pathlib import Path
import time

from diagnostics.native_source_grid import build_engine, CODE as GRID_CODE
from diagnostics.native_real_grid import projected_gradient
from diagnostics.native_real_posterior import heartbeat, save_npz, starting_branches
import jax
import jax.numpy as jnp
import numpy as np
import torch
from scipy.optimize import minimize

from dingo.native_real import separable_response_kernel
from dingo.real_workflow import REPO, read_json, write_json, sha256

VARIANTS = {
    'baseline': (1., 1.),
    'lsf_0p9': (.9, 1.),
    'lsf_1p1': (1.1, 1.),
    'psf_0p9': (1., .9),
    'psf_1p1': (1., 1.1),
}
CODE = list(dict.fromkeys(GRID_CODE + [
    'diagnostics/native_real_dispersion.py', 'diagnostics/native_real_posterior.py',
    'dingo/native_real_posterior.py', 'diagnostics/native_posterior_check.py']))


def pack_free(vector, speed):
    """Optimize speed and every nuisance coordinate except fixed sigma (x[4])."""
    return np.r_[speed/100., np.delete(np.asarray(vector), 4)]


def unpack_free(free, sigma):
    x = jnp.concatenate((free[1:5], jnp.reshape(sigma/100., (1,)), free[5:]))
    return x, 100*free[0]


def scaled_kernels(records, spectral_scale, spatial_scale):
    """Scale physical widths BEFORE removing cubic-deposition variance.

    Spectral means the complete empirical LSF (both mixture components).
    Spatial means the cross-dispersion Gaussian only: no second optical PSF
    is added along dispersion. These are sensitivity scenarios, not priors.
    """
    if not np.isfinite([spectral_scale, spatial_scale]).all() or min(spectral_scale, spatial_scale) <= 0:
        raise ValueError('Response scale factors must be finite and positive.')
    return np.stack([separable_response_kernel(
        r['pupil'], r['lsf_fraction'],
        np.asarray(r['along_sigma_pixels'])*spectral_scale,
        r['cross_sigma_pixels']*spatial_scale) for r in records])


def choose_best(fits):
    """Keep the lowest score visible even if its stationarity check failed."""
    groups = {}
    for f in fits:
        key = (f['variant'], f['fixed_sigma_kms'])
        if key not in groups or f['objective'] < groups[key]['objective']:
            groups[key] = f
    return list(groups.values())


def run_locked(args):
    output, grid_dir = args.output.resolve(), args.grid.resolve()
    grid = read_json(grid_dir/'grid.json')
    if not grid.get('completed') or grid.get('real_observations') is not True:
        raise ValueError('A completed real-source R+C grid is required.')
    if grid['source_id'] != args.source_id:
        raise ValueError('Grid belongs to a different source.')
    sigma_values = sorted(set(float(s) for s in args.sigmas))
    if not sigma_values or not np.isfinite(sigma_values).all() or min(sigma_values) < 0 or max(sigma_values) > 200:
        raise ValueError('Fixed dispersions must be finite and within 0–200 km/s.')
    if args.maxiter < 1 or args.gtol <= 0:
        raise ValueError('Positive optimizer limits required.')
    if args.device == 'cuda' and jax.default_backend() != 'gpu':
        raise RuntimeError('GPU requested but unavailable.')
    torch.set_num_threads(4)
    jax.config.update('jax_compilation_cache_dir', str(output/'jax_cache'))
    settings = dict(grid=str(grid_dir), source_id=args.source_id, sigmas=sigma_values,
                    variants=list(args.variants), maxiter=args.maxiter, gtol=args.gtol,
                    stationarity_threshold=.1)
    hashes = {p: sha256(REPO/p) for p in CODE}
    inputs = {str(grid_dir/p): sha256(grid_dir/p) for p in ('grid.json', 'native_data.npz', 'inclination_prior.npz')}
    branches = starting_branches(grid)
    for f in branches:
        inputs[str(grid_dir/f['artifact'])] = sha256(grid_dir/f['artifact'])
    path = output/'dispersion.json'
    if path.exists():
        report = read_json(path)
        for key, value in [('settings', settings), ('code_sha256', hashes), ('input_sha256', inputs)]:
            if report[key] != value:
                raise ValueError(f'Resume requires unchanged {key}; choose a new run name.')
        for f in report['fits']:
            if sha256(output/f['artifact']) != f['artifact_sha256']:
                raise ValueError('A checkpoint artifact changed; do not reuse this run.')
        if report.get('completed'):
            print(f'Already complete: {output}. Loading saved results needs no GPU refit.', flush=True)
            return
    else:
        report = dict(format='native_real_dispersion_v1', source_id=args.source_id,
            real_observations=True, completed=False, settings=settings, code_sha256=hashes,
            input_sha256=inputs, fits=[], best_by_sigma=[],
            started_unix=time.time(), device=str(jax.devices()),
            interpretation='Conditional penalized fixed-sigma fits; not MCMC, confidence intervals, or model evidence.',
            limitations=[
                'Same amplitude-profiled objective as the velocity grid; differs from the amplitude-integrated MCMC target.',
                'Response scale changes of +/-10% are illustrative sensitivity scenarios, not calibrated uncertainty bounds.',
                'Positive map, smoothness, finite support, masks, errors and inclination transfer remain assumptions.',
                'Two separated starts plus bidirectional continuation reduce but do not eliminate local-minimum risk.',
                'No pressure-support correction or mass estimate is calculated.'])
        for p in CODE:
            dest = output/'code'/p; dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((REPO/p).read_bytes())
        write_json(path, report)
    print(f'ID{args.source_id}: rebuilding the saved real R+C model on {jax.devices()}…', flush=True)
    with heartbeat('Loading/calibrating real data'):
        target, original, likelihood, nuisance_bounds, _, _ = build_engine(grid)
    pupils = np.array([e.pupil for e in target.exposures])
    if set(pupils) != {'R', 'C'}:
        raise ValueError('Both R and C exposures are required for this comparison.')
    science, error, valid = [np.stack([getattr(e, key) for e in target.exposures])
                             for key in ('science', 'error', 'valid')]
    # Store an independent pixel copy; viewing never depends on the grid being modified later.
    target.save(output/'native_data.npz')
    report.update(native_data_sha256=sha256(output/'native_data.npz'),
                  source_summary=target.summary, exposures=grid['exposures'],
                  n_valid_pixels=int(valid.sum()), n_exposures=len(pupils),
                  array_sha256=grid['array_sha256'], instrument_response=grid['instrument_response'])
    seeds = []
    for f in branches:
        with np.load(grid_dir/f['artifact']) as a:
            seeds.append(pack_free(a['vector'], f['vrot_0p5arcsec_kms']))
    report['starts'] = [dict(velocity=f['vrot_0p5arcsec_kms'], artifact=f['artifact'],
                            center=f['center_east_north_arcsec']) for f in branches]
    bounds = [(0., grid['settings']['vmax']/100.)] + [b for i, b in enumerate(nuisance_bounds) if i != 4]
    total = len(args.variants)*len(sigma_values)*2
    done = {(f['variant'], f['fixed_sigma_kms'], f['sweep']) for f in report['fits']}
    zero = float(np.sum(np.asarray(original.likelihood.residual_data)**2))
    for variant in args.variants:
        spectral, spatial = VARIANTS[variant]
        kernels = scaled_kernels(grid['instrument_response']['exposures'], spectral, spatial)
        if variant == 'baseline':
            # NativeLineProjector normalizes each already-normalized kernel
            # once more. Preserve its exact baseline, allowing only roundoff
            # in this independent reconstruction from calibration metadata.
            np.testing.assert_allclose(kernels, np.asarray(original.projector.kernels), rtol=1e-14, atol=1e-16)
            kernels = np.asarray(original.projector.kernels)
        else:
            kernels = kernels/kernels.sum(axis=(-2, -1), keepdims=True)
        model = copy.copy(original)
        model.projector = copy.copy(original.projector)
        model.projector.kernels = jnp.asarray(kernels)
        center_kind = grid['settings'].get('center', 'catalog')
        center_mu = jnp.asarray(grid['center_mean_east_north_arcsec'])
        def objective(free, sigma):
            x, speed = unpack_free(free, sigma)
            value = model.objective(x, speed)
            if center_kind != 'catalog':
                center = .1*x[5:7]
                value -= .5*jnp.sum((center/.15)**2)
                if center_kind == 'stellar':
                    value += .5*jnp.sum(((center-center_mu)/.05)**2)
            return value
        compiled = jax.jit(jax.value_and_grad(objective))
        components = jax.jit(model.components)
        print(f'{variant}: LSF width ×{spectral:g}, cross-dispersion PSF width ×{spatial:g}', flush=True)
        for sweep, values in [('up', sigma_values), ('down', sigma_values[::-1])]:
            previous = seeds[0 if sweep == 'up' else 1]
            for sigma in values:
                key = (variant, sigma, sweep)
                if key in done:
                    f = next(f for f in report['fits'] if (f['variant'], f['fixed_sigma_kms'], f['sweep']) == key)
                    with np.load(output/f['artifact']) as a:
                        previous = pack_free(a['vector'], f['vrot_0p5arcsec_kms'])
                    continue
                # Each sigma is tried from a continuation of each separated seed.
                # Up and down use distinct initial branches rather than silently
                # replacing the alternative seed with the lowest starting score.
                start = previous.copy()
                number = len(report['fits'])+1
                print(f'[{number}/{total}] {variant}, sigma={sigma:g} km/s, {sweep} start: '
                      f'V={100*start[0]:.1f} km/s', flush=True)
                began = time.perf_counter()
                def fun(p):
                    value, grad = compiled(jnp.asarray(p), jnp.asarray(sigma))
                    return float(value), np.asarray(grad)
                iterations = [0]
                def progress(p):
                    iterations[0] += 1
                    if iterations[0] % 100 == 0:
                        print(f'  [{number}/{total}] iteration {iterations[0]}, '
                              f'{time.perf_counter()-began:.0f}s, V={100*p[0]:.1f} km/s', flush=True)
                with heartbeat(f'[{number}/{total}] optimizer/compilation'):
                    opt = minimize(fun, start, jac=True, method='L-BFGS-B', bounds=bounds,
                        callback=progress, options=dict(maxiter=args.maxiter, gtol=args.gtol,
                                                       ftol=1e-13, maxls=40, maxcor=40))
                    pg = projected_gradient(opt.x, opt.jac, bounds)
                    if pg >= .1:
                        print('  Refining: projected gradient remains above 0.1.', flush=True)
                        opt = minimize(fun, opt.x, jac=True, method='L-BFGS-B', bounds=bounds,
                            callback=progress, options=dict(maxiter=args.maxiter, gtol=args.gtol,
                                                           ftol=1e-14, maxls=50, maxcor=40))
                        pg = projected_gradient(opt.x, opt.jac, bounds)
                if not np.isfinite(opt.fun) or not np.isfinite(opt.x).all() or not np.isfinite(opt.jac).all():
                    raise RuntimeError('Nonfinite optimizer result; earlier checkpoints remain available.')
                x, speed = unpack_free(jnp.asarray(opt.x), jnp.asarray(sigma))
                _, parts = components(x, speed)
                relative, incpen, otherpen, amplitudes, line, flux, velocity = map(np.asarray, parts)
                continuum = likelihood.continuum(torch.from_numpy(line.copy())).numpy()
                residual = np.where(valid, (science-line-continuum)/error, np.nan)
                chi = float(np.nansum(residual**2))
                if not np.isclose(chi, zero-2*float(relative), rtol=1e-10, atol=1e-6):
                    raise RuntimeError('Rendered residuals and objective chi-square disagree.')
                name = f'fit_{len(report["fits"]):04d}.npz'
                save_npz(output/name, source_id=np.asarray(args.source_id), vector=np.asarray(x),
                         vrot=np.asarray(speed), line=line, continuum=continuum, residual=residual,
                         source_flux=flux, source_velocity=velocity, amplitudes=amplitudes)
                f = model.describe(np.asarray(x), float(speed)) | dict(
                    variant=variant, fixed_sigma_kms=sigma, sweep=sweep,
                    spectral_scale=spectral, spatial_scale=spatial,
                    objective=float(opt.fun), chi_square=chi,
                    penalty=float(opt.fun)+float(relative),
                    chi_square_R=float(np.nansum(residual[pupils == 'R']**2)),
                    chi_square_C=float(np.nansum(residual[pupils == 'C']**2)),
                    stationarity_passed=pg < .1, projected_gradient_inf=pg,
                    optimizer_success=bool(opt.success), optimizer_message=str(opt.message),
                    iterations=iterations[0], seconds=time.perf_counter()-began,
                    artifact=name, artifact_sha256=sha256(output/name))
                report['fits'].append(f)
                report['best_by_sigma'] = choose_best(report['fits'])
                report['updated_unix'] = time.time()
                write_json(path, report)
                done.add(key)
                previous = opt.x
                print(f'  Saved {name}: V={float(speed):.2f}, score J={opt.fun:.3f}, '
                      f'gradient={pg:.3g}, check={"PASS" if pg < .1 else "FAIL"}, '
                      f'{f["seconds"]:.1f}s', flush=True)
        jax.clear_caches()
    report['completed'] = True
    report['all_selected_stationary'] = all(f['stationarity_passed'] for f in report['best_by_sigma'])
    report['finished_unix'] = time.time()
    write_json(path, report)
    print(f'Completed {total} fits. Selected-point checks: {report["all_selected_stationary"]}. '
          f'Results: {output}', flush=True)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'.dispersion.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('This dispersion run is already active.') from exc
        run_locked(args)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-id', type=int, required=True)
    p.add_argument('--grid', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sigmas', nargs='+', type=float, default=[0, 10, 20, 40, 80])
    p.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=list(VARIANTS))
    p.add_argument('--maxiter', type=int, default=1200)
    p.add_argument('--gtol', type=float, default=.01)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    run(p.parse_args())
