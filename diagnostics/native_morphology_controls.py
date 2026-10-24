"""Real R+C morphology-prior sensitivity, with an explicitly restrictive disk control.

Each flexible case starts from identical physical maps and separated kinematics.
Sigma is fixed and removed from optimization; other coordinates are refitted.
No objective comparison across morphology priors is a Bayes factor.
"""
import argparse
import copy
import fcntl
from pathlib import Path
import time

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
import torch

from diagnostics.native_simple_dispersion import (
    SIGMA_INDEX, pack_free, unpack_free, choose_best, boundary_parameters,
    snapshot_inputs, verify_hashes)
from diagnostics.native_simple_numerics import CODE as NUMERICS_CODE, rebuild_saved_target, refined_engine
from diagnostics.native_real_grid import projected_gradient
from diagnostics.native_real_posterior import heartbeat, save_npz
from dingo.native_coupled import CoupledNativeModel
from dingo.native_linked import LinkedExponentialCoupledModel
from dingo.native_morphology_controls import MORPHOLOGY_SCENARIOS, morphology_engine, transport_parameters
from dingo.real_workflow import REPO, read_json, write_json, sha256

CODE = list(dict.fromkeys(NUMERICS_CODE + ['dingo/native_linked.py',
    'dingo/native_morphology_controls.py', 'diagnostics/native_morphology_controls.py']))


def best_seed(report, model, sigma):
    candidates = [f for f in report['fits'] if f['model'] == model and f['variant'] == 'baseline'
                  and np.isclose(f['fixed_sigma_kms'], sigma, rtol=0, atol=1e-10)]
    if not candidates:
        raise ValueError(f'Missing saved {model} initializer at sigma={sigma:g}.')
    return min(candidates, key=lambda f: f['objective'])


def boundaries(parameters, model, name):
    if name == 'flexible':
        return boundary_parameters(parameters, model, name)
    labels = {0: 'rotation speed', 1: 'turnover radius', 3: 'inclination',
              4: 'systemic velocity', 6: 'shared center x', 7: 'shared center y',
              8: 'emission Re', 1+model.base_size: 'line flux'}
    result = []
    for i, label in labels.items():
        low, high = model.bounds[i]
        scale = max(1., high-low if low is not None and high is not None else abs(parameters[i]))
        if any(v is not None and abs(parameters[i]-v) <= 1e-4*scale for v in (low, high)):
            result.append(label)
    return result


def light_moments(flux, east, north):
    xy = np.stack((east, north), axis=1)
    center = flux @ xy / flux.sum()
    delta = xy-center
    cov = np.einsum('n,ni,nj->ij', flux/flux.sum(), delta, delta)
    eig, vectors = np.linalg.eigh(cov)
    return dict(brightness_centroid_arcsec=center.tolist(),
        brightness_moment_axis_ratio=float(np.sqrt(max(eig[0], 0)/eig[1])),
        brightness_moment_angle_deg=float(np.rad2deg(np.arctan2(*vectors[::-1, -1])) % 180))


def run_locked(args):
    run_dir, seeds_dir, output = args.run.resolve(), args.seeds.resolve(), args.output.resolve()
    if output in (run_dir, seeds_dir):
        raise ValueError('Use a new directory; saved scientific runs must remain unchanged.')
    source, seeds = [read_json(p/'dispersion.json') for p in (run_dir, seeds_dir)]
    for r in (source, seeds):
        if r.get('format') != 'native_simple_dispersion_v1' or not r.get('real_observations') or not r.get('completed'):
            raise ValueError('Both original and starting runs must be completed real-source scans.')
    sid = source['source_id']
    if seeds['source_id'] != sid or (args.source_id is not None and sid != args.source_id):
        raise ValueError('Source IDs disagree.')
    if source.get('refinement'):
        raise ValueError('--run must be the original scan; --seeds is its refined comparison.')
    if source['native_data_sha256'] != seeds['native_data_sha256']:
        raise ValueError('Starting fits must use exactly the same observed pixels.')
    if seeds['nuisance_settings'] != source['nuisance_settings']:
        raise ValueError('Starting nuisance target differs from original data reconstruction.')
    if args.device == 'cuda' and jax.default_backend() != 'gpu':
        raise RuntimeError('GPU requested but unavailable.')
    if args.device == 'cpu' and jax.default_backend() != 'cpu':
        raise RuntimeError('Set JAX_PLATFORMS=cpu for CPU execution.')
    sigmas = sorted(set(map(float, args.sigmas)))
    controls = int(source['grid_settings']['controls'])
    if not sigmas or not np.isfinite(sigmas).all() or min(sigmas) < 0 or max(sigmas) > 200:
        raise ValueError('Fixed widths must be finite in [0,200] km/s.')
    if args.spatial < controls or min(args.spectral, args.maxiter, args.refine) < 1:
        raise ValueError('Invalid quadrature or iteration limits.')
    plans = [('flexible', v) for v in args.variants]
    if not args.skip_linked:
        plans.append(('linked_exponential', 'baseline'))
    selections = {(m, s): best_seed(seeds, m, s) for s in sigmas
                  for m in (['flexible'] if args.skip_linked else ['flexible', 'exponential'])}
    inputs = dict(source['input_sha256'])
    for directory, report in ((run_dir, source), (seeds_dir, seeds)):
        verify_hashes(report['input_sha256'], 'Saved input')
        verify_hashes({str(REPO/p): h for p, h in report['code_sha256'].items()}, 'Saved code')
        inputs[str(directory/'dispersion.json')] = sha256(directory/'dispersion.json')
        inputs[str(directory/'native_data.npz')] = report['native_data_sha256']
    inputs[str(seeds_dir/'inclination_prior.npz')] = seeds['inclination_prior_sha256']
    for fit in selections.values():
        inputs[str(seeds_dir/fit['artifact'])] = fit['artifact_sha256']
    verify_hashes(inputs, 'Morphology input')
    settings = dict(run=str(run_dir), seeds=str(seeds_dir), baseline=source['settings']['baseline'],
        source_id=sid, variants=list(args.variants), skip_linked=args.skip_linked,
        sigmas=sigmas, spatial=args.spatial, spectral=args.spectral,
        maxiter=args.maxiter, refine=args.refine, stationarity_threshold=.1,
        inclination='exact frozen density from seeds', response='nominal fixed calibrated response',
        flexible_starts='Saved optimum plus V=600 km/s and PA+90 degrees on the same brightness map; '
                        'exact brightness transport to each prior; bidirectional sigma continuation.',
        linked_starts='Flexible geometry, independent-exponential geometry, light-aligned geometry, '
                      'opposite light-aligned velocity sense; bidirectional sigma continuation.')
    hashes = {name: sha256(REPO/name) for name in CODE}
    path = output/'dispersion.json'
    report = read_json(path) if path.exists() else dict(format='native_simple_dispersion_v1',
        morphology_controls=True, source_id=sid, real_observations=True, completed=False,
        settings=settings, input_sha256=inputs, code_sha256=hashes, fits=[], best_by_sigma=[],
        started_unix=time.time(), device=str(jax.devices()), initial_map_checks=[],
        interpretation='Conditional fixed-sigma nuisance refits, not MCMC or confidence intervals. '
            'Each morphology-prior curve has its own zero; cross-prior scores are not evidence. '
            'Flexible shared gas brightness is primary. Linked axisymmetric exponential is a restrictive control.',
        limitations=['A finite source square, regularization and a circular velocity field remain assumptions.',
            'The soft imaging inclination density does not constrain gas-brightness moments.',
            'Multiple starts and stationarity do not prove a global optimum.',
            'Numerical sampling and empirical PSF/LSF still have finite accuracy.'])
    for key, value in [('settings', settings), ('input_sha256', inputs), ('code_sha256', hashes)]:
        if report[key] != value:
            raise ValueError(f'Cannot resume changed {key}; select a new output directory.')
    for fit in report['fits']:
        verify_hashes({str(output/fit['artifact']): fit['artifact_sha256']}, 'Checkpoint')
    verify_hashes({str(output/p): h for p, h in report.get('input_snapshot_sha256', {}).items()}, 'Snapshot')
    for name, digest in hashes.items():
        destination = output/'code'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_bytes((REPO/name).read_bytes())
        verify_hashes({str(destination): digest}, 'Frozen code')
    if report['completed']:
        print('Already complete:', output, flush=True)
        return
    snapshots, snapshot_hashes = snapshot_inputs(output, inputs)
    report.update(input_snapshot_paths=snapshots, input_snapshot_sha256=snapshot_hashes)
    write_json(path, report)
    with heartbeat('Rebuilding the same R+C observations and calibrated projector'):
        target, engine, old_like, response, grid = rebuild_saved_target(source, run_dir)
        engine = copy.copy(engine)
        with np.load(seeds_dir/'inclination_prior.npz') as prior:
            if int(prior['source_id']) != sid:
                raise ValueError('Inclination prior belongs to a different source.')
            engine.inclination_grid = jnp.asarray(prior['inclination_grid'])
            engine.inclination_logpdf = jnp.asarray(prior['inclination_logpdf'])
        engine = refined_engine(engine, target, controls, args.spatial, args.spectral,
                                np.asarray(engine.projector.kernels))
    for filename, directory in [('native_data.npz', run_dir), ('inclination_prior.npz', seeds_dir)]:
        dest = output/filename
        if not dest.exists():
            dest.write_bytes((directory/filename).read_bytes())
        verify_hashes({str(dest): sha256(directory/filename)}, 'Frozen observation/prior')
    data, error, valid = [np.stack([getattr(e, k) for e in target.exposures]) for k in ('science', 'error', 'valid')]
    pupils = np.asarray([e.pupil for e in target.exposures])
    if set(pupils) != {'R', 'C'}:
        raise ValueError('Both grisms must be present.')
    nuisance = source['nuisance_settings']
    expected = len(args.variants)*(4*len(sigmas)-2)+(0 if args.skip_linked else 6*len(sigmas)-2)
    report.update(native_data_sha256=source['native_data_sha256'],
        inclination_prior_sha256=seeds['inclination_prior_sha256'], inclination_prior=seeds['inclination_prior'],
        source_summary=target.summary, exposures=[e.metadata for e in target.exposures],
        grid_settings=seeds['grid_settings'] | dict(quadrature=args.spatial),
        instrument_response=source['instrument_response'], calibration=source['calibration'],
        calibration_sha256=source['calibration_sha256'], nuisance_settings=nuisance,
        n_valid_pixels=int(valid.sum()), n_exposures=len(pupils),
        source_grid=dict(shape=[args.spatial, args.spatial], half_extent_arcsec=float(engine.half_extent),
                         spectral_quadrature_nodes=args.spectral),
        model_settings=report.get('model_settings', {}), scenario_settings=MORPHOLOGY_SCENARIOS,
        expected_fit_count=expected)
    done = {(f['model'], f['variant'], f['fixed_sigma_kms'], f['start']): f for f in report['fits']}
    write_json(path, report)

    def read_parameters(directory, fit):
        with np.load(directory/fit['artifact']) as saved:
            if int(saved['source_id']) != sid:
                raise ValueError('Initializer source mismatch.')
            return saved['parameters'].copy(), saved['source_flux'].copy()

    for name, variant in plans:
        trial_engine = morphology_engine(engine, variant) if name == 'flexible' else engine
        cls = CoupledNativeModel if name == 'flexible' else LinkedExponentialCoupledModel
        model = cls(trial_engine, target, response, **nuisance)
        free_bounds = [b for i, b in enumerate(model.bounds) if i != SIGMA_INDEX]
        compiled = jax.jit(jax.value_and_grad(lambda free, sigma: model.objective(unpack_free(free, sigma))))
        components = jax.jit(model.components)
        description = (dict(shape='Flexible positive common gas-brightness map; independent of stellar and '
                           'kinematic brightness moments', gas_geometry_independent_of_stars=True,
                           gas_geometry_independent_of_kinematic_geometry=True)
                       if name == 'flexible' else model.prior_description())
        report['model_settings'][name] = description | dict(bounds=[list(b) for b in model.bounds],
            parameter_count=len(model.bounds), free_parameters_per_fixed_sigma=len(free_bounds))

        def fit_one(start, sigma, label, sweep, initial_artifact):
            key = name, variant, sigma, label
            if key in done:
                return read_parameters(output, done[key])[0]
            start = np.asarray(start).copy(); start[SIGMA_INDEX] = sigma/100
            print(f'[{len(report["fits"])+1}/{expected}] ID{sid} {name}/{variant} '
                  f'sigma={sigma:g}, {label}, Vstart={100*start[0]:.1f}', flush=True)
            begin = time.perf_counter(); count = [0]; last = [0.]
            report['active'] = dict(model=name, variant=variant, fixed_sigma_kms=sigma,
                start=label, iteration=0, started_unix=time.time())
            write_json(path, report)
            def fun(free):
                value, gradient = compiled(jnp.asarray(free), jnp.asarray(sigma))
                return float(value), np.asarray(gradient)
            def progress(free):
                count[0] += 1
                elapsed = time.perf_counter()-begin
                if elapsed-last[0] >= 20:
                    print(f'  iteration {count[0]}, {elapsed:.0f}s, V={100*free[0]:.2f}', flush=True)
                    report['active'].update(iteration=count[0], seconds=elapsed,
                                             vrot_0p5arcsec_kms=float(100*free[0]))
                    write_json(path, report); last[0] = elapsed
            with heartbeat(f'Fitting {name}/{variant}/sigma={sigma:g}'):
                opt = minimize(fun, pack_free(start), jac=True, method='L-BFGS-B', bounds=free_bounds,
                    callback=progress, options=dict(maxiter=args.maxiter, gtol=.01, ftol=1e-12, maxls=40, maxcor=40))
                pg = projected_gradient(opt.x, opt.jac, free_bounds)
                if pg >= .1:
                    second = minimize(fun, opt.x, jac=True, method='L-BFGS-B', bounds=free_bounds,
                        callback=progress, options=dict(maxiter=args.refine, gtol=.002, ftol=1e-13, maxls=50, maxcor=60))
                    if np.isfinite(second.fun) and second.fun <= opt.fun:
                        opt = second
                    pg = projected_gradient(opt.x, opt.jac, free_bounds)
            if not np.isfinite(opt.fun) or not np.isfinite(opt.x).all() or not np.isfinite(opt.jac).all():
                raise RuntimeError('Nonfinite optimizer result; checkpoints preserved.')
            parameters = np.asarray(unpack_free(jnp.asarray(opt.x), jnp.asarray(sigma)))
            value, parts = components(jnp.asarray(parameters))
            parts = {k: np.asarray(v) for k, v in parts.items()}
            if nuisance['mode'] == 'coupled':
                parts['continuum'] = old_like.continuum(torch.from_numpy(parts['line'].copy())).numpy()
            total = parts['line']+parts['continuum']+parts['background'][:, None, None]
            residual = np.divide(data-total, error, out=np.full_like(data, np.nan), where=valid)
            chi2 = float(np.nansum(residual**2))
            np.testing.assert_allclose(chi2, parts['chi2'], rtol=1e-10, atol=1e-6)
            np.testing.assert_allclose(value, .5*chi2+parts['penalty'], rtol=1e-10, atol=1e-6)
            x, speed, flux, gz, bz = model.split(parameters)
            desc = trial_engine.describe(np.asarray(x), float(speed)) if name == 'flexible' else model.describe(parameters)
            tag = f'fit_{len(report["fits"]):04d}.npz'
            east, north = np.asarray(engine.line_map.east), np.asarray(engine.line_map.north)
            save_npz(output/tag, source_id=np.asarray(sid), parameters=parameters, initial_parameters=start,
                vector=np.asarray(x), residual=residual, source_east=east, source_north=north, **parts)
            edge = (np.abs(east) > .8) | (np.abs(north) > .8)
            entry = desc | light_moments(parts['source_flux'], east, north) | dict(model=name,
                mode=nuisance['mode'], variant=variant, fixed_sigma_kms=sigma, start=label,
                sweep=sweep, initial_artifact=initial_artifact, spectral_scale=1., spatial_scale=1.,
                objective=float(value), chi_square=chi2, penalty=float(parts['penalty']),
                map_penalty=float(.5*np.sum(np.asarray(x[10:10+engine.nmap])**2)) if name=='flexible' else None,
                fixed_sigma_prior_penalty=.5*(sigma/100)**2,
                chi_square_R=float(np.nansum(residual[pupils=='R']**2)),
                chi_square_C=float(np.nansum(residual[pupils=='C']**2)), line_flux_1e17=float(flux),
                source_edge_flux_fraction=float(parts['source_flux'][edge].sum()),
                gain_min_max=[float(parts['gains'].min()), float(parts['gains'].max())],
                background_standardized=np.asarray(bz).tolist(), projected_gradient_inf=float(pg),
                stationarity_passed=bool(pg < .1), optimizer_success=bool(opt.success),
                message=str(opt.message), iterations=count[0], seconds=time.perf_counter()-begin,
                boundary_parameters=boundaries(parameters, model, name), artifact=tag,
                artifact_sha256=sha256(output/tag))
            report['fits'].append(entry); report['best_by_sigma'] = choose_best(report['fits'])
            report['updated_unix'] = time.time(); report.pop('active', None)
            write_json(path, report); done[key] = entry
            print(f'  saved V={float(speed):.2f}, J={float(value):.3f}, gradient={pg:.3g}, '
                  f'{entry["seconds"]:.1f}s', flush=True)
            return parameters

        for sigma in sigmas:
            fit = selections['flexible', sigma]
            p, source_flux = read_parameters(seeds_dir, fit)
            origin = str(seeds_dir/fit['artifact'])
            if name == 'flexible':
                transported = transport_parameters(p, engine, trial_engine)
                # Check real quadrature brightness AND detector predictions at the transported start.
                reference = CoupledNativeModel(engine, target, response, **nuisance)
                with heartbeat('Checking identical physical starting maps across priors'):
                    old_value, old_parts = jax.jit(reference.components)(jnp.asarray(p))
                    new_value, new_parts = components(jnp.asarray(transported))
                np.testing.assert_allclose(old_parts['source_flux'], new_parts['source_flux'], rtol=1e-10, atol=1e-13)
                np.testing.assert_allclose(old_parts['line'], new_parts['line'], rtol=1e-10, atol=1e-13)
                penalty_delta = .5*(np.sum(transported[11:11+engine.nmap]**2)-np.sum(p[11:11+engine.nmap]**2))
                np.testing.assert_allclose(float(new_value-old_value), penalty_delta, rtol=1e-8, atol=1e-7)
                check_key = variant, sigma
                if not any((c['variant'], c['sigma']) == check_key for c in report['initial_map_checks']):
                    report['initial_map_checks'].append(dict(variant=variant, sigma=sigma,
                        max_flux_difference=float(np.max(np.abs(np.asarray(old_parts['source_flux']-new_parts['source_flux'])))),
                        max_line_difference=float(np.max(np.abs(np.asarray(old_parts['line']-new_parts['line'])))),
                        objective_delta=float(new_value-old_value), map_penalty_delta=float(penalty_delta)))
                    write_json(path, report)
                shifted = transported.copy(); shifted[0] = 6.; shifted[2] += np.pi/2
                starts = [('matched_saved', transported, origin), ('shifted_kinematics', shifted, origin)]
            else:
                exp_fit = selections['exponential', sigma]
                exp, _ = read_parameters(seeds_dir, exp_fit)
                from_exp = model.from_exponential(exp)
                aligned = from_exp.copy()
                aligned[0] = 3.5; aligned[2] = exp[10]
                aligned[3] = np.clip(np.arccos(exp[9]), model.bounds[3][0], model.bounds[3][1])
                aligned[6:8] = exp[11:13]
                opposite = aligned.copy(); opposite[2] += np.pi
                starts = [('flexible_geometry', model.from_flexible(p, source_flux), origin),
                    ('exponential_geometry', from_exp, str(seeds_dir/exp_fit['artifact'])),
                    ('light_aligned', aligned, str(seeds_dir/exp_fit['artifact'])),
                    ('light_opposite', opposite, str(seeds_dir/exp_fit['artifact']))]
            for label, start, initial_artifact in starts:
                fit_one(start, sigma, label, 'independent', initial_artifact)
        def best_at(sigma):
            return min((f for f in report['fits'] if f['model']==name and f['variant']==variant
                        and f['fixed_sigma_kms']==sigma), key=lambda f: f['objective'])
        for sweep, values in [('up', sigmas), ('down', sigmas[::-1])]:
            previous = best_at(values[0])
            for sigma in values[1:]:
                fit_one(read_parameters(output, previous)[0], sigma, f'continuation_{sweep}', sweep,
                        str(output/previous['artifact']))
                previous = best_at(sigma)
        jax.clear_caches()
    report.update(completed=True, finished_unix=time.time(),
                  all_selected_stationary=all(f['stationarity_passed'] for f in report['best_by_sigma']))
    report.pop('active', None); write_json(path, report)
    print(f'Completed {len(report["fits"])} fits: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--seeds', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-id', type=int)
    parser.add_argument('--variants', nargs='+', choices=list(MORPHOLOGY_SCENARIOS), default=list(MORPHOLOGY_SCENARIOS))
    parser.add_argument('--skip-linked', action='store_true')
    parser.add_argument('--sigmas', nargs='+', type=float, default=[0, 20, 40, 80])
    parser.add_argument('--spatial', type=int, default=121)
    parser.add_argument('--spectral', type=int, default=17)
    parser.add_argument('--maxiter', type=int, default=1200)
    parser.add_argument('--refine', type=int, default=4000)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    if len(set(args.variants)) != len(args.variants):
        raise ValueError('Duplicate scenarios are not allowed.')
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'.workflow.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('This comparison is already running.') from exc
        run_locked(args)


if __name__ == '__main__':
    main()
