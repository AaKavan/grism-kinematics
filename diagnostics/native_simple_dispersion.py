"""Fixed-dispersion real R+C refits with flexible and smooth gas emission.

The exponential profile resembles the old DINGO analytic emission model;
the native-exposure likelihood, calibration and continuum treatment remain
those of the selected coupled fit. These are conditional penalized profiles,
not MCMC, confidence intervals, or evidence for choosing a morphology.
"""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import time

from diagnostics.native_coupled_fit import CODE as COUPLED_CODE
from diagnostics.native_source_grid import build_engine
from diagnostics.native_real_grid import projected_gradient
from diagnostics.native_real_posterior import heartbeat, save_npz
from diagnostics.native_real_dispersion import scaled_kernels
from dingo.native_coupled import CoupledNativeModel, calibration_response
from dingo.native_fitting import StripContinuumLikelihood
from dingo.native_jax import JaxStripLikelihood
from dingo.native_quality import apply_detector_mask
from dingo.native_simple import ExponentialCoupledModel
from dingo.real_workflow import REPO, read_json, write_json, sha256
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
import torch


VARIANTS = {'baseline': 1., 'lsf_0p9': .9, 'lsf_1p1': 1.1}
CODE = list(dict.fromkeys(COUPLED_CODE + [
    'dingo/native_simple.py', 'diagnostics/native_simple_dispersion.py',
    'diagnostics/native_real_dispersion.py', 'dingo/real_workflow.py']))
SIGMA_INDEX = 5


def pack_free(parameters):
    """Remove sigma from optimization altogether; zero has no trapping gradient."""
    return np.delete(np.asarray(parameters, dtype=float), SIGMA_INDEX)


def unpack_free(free, sigma):
    return jnp.concatenate((free[:SIGMA_INDEX], jnp.reshape(sigma/100., (1,)), free[SIGMA_INDEX:]))


def choose_best(fits):
    """Keep the lowest objective visible even if it failed stationarity."""
    groups = {}
    for fit in fits:
        key = fit['model'], fit['variant'], fit['fixed_sigma_kms']
        if key not in groups or fit['objective'] < groups[key]['objective']:
            groups[key] = fit
    return list(groups.values())


def boundary_parameters(parameters, model, name):
    labels = {0:'rotation speed', 1:'turnover radius', 3:'inclination', 4:'systemic velocity',
              6:'kinematic center x', 7:'kinematic center y', 8:'emission Re' if name=='exponential' else 'envelope width',
              9:'emission axis ratio', 1+model.base_size:'line flux'}
    if name == 'exponential':
        labels.update({11:'emission center x', 12:'emission center y'})
    result = []
    for i,label in labels.items():
        low, high = model.bounds[i]
        scale = max(1., (high-low) if low is not None and high is not None else abs(parameters[i]))
        if (low is not None and abs(parameters[i]-low) <= 1e-4*scale) or (high is not None and abs(parameters[i]-high) <= 1e-4*scale):
            result.append(label)
    return result


def starting_branches(fits, mode):
    """Choose the best and a distinct stored basin, never just adjacent repeats."""
    good = [f for f in fits if f['mode'] == mode and np.isfinite(f['objective'])]
    stationary = [f for f in good if f.get('stationarity_passed')]
    pool = stationary if len(stationary) >= 2 else good
    if len(pool) < 2:
        raise ValueError('At least two stored baseline fits are needed for independent starts.')
    best = min(pool, key=lambda f: f['objective'])
    others = [f for f in pool if f['artifact'] != best['artifact']]
    def distance(fit):
        angle = abs((fit['theta_east_to_north_deg']-best['theta_east_to_north_deg']+180)%360-180)
        center = np.linalg.norm(np.asarray(fit['center_east_north_arcsec'])-best['center_east_north_arcsec'])
        return abs(fit['vrot_0p5arcsec_kms']-best['vrot_0p5arcsec_kms'])/75 + angle/90 + center/.15
    distinct = [f for f in others if distance(f) >= 1.]
    # Reject duplicate minima rather than falsely claiming independent basins.
    if not distinct:
        raise ValueError('Saved baselines contain no separated velocity/angle/center branch. '
                         'Run additional coupled starts before this comparison.')
    return best, min(distinct, key=lambda f: f['objective'])


def verify_hashes(mapping, label):
    for name, digest in mapping.items():
        if sha256(name) != digest:
            raise ValueError(f'{label} changed: {name}')


def verify_native_arrays(target, path):
    """Check the rebuilt, masked data against the actual fitted pixel archive."""
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved['metadata_json']))
        if metadata['source_id'] != target.source_id:
            raise ValueError('Baseline native data belongs to a different source.')
        for name in ('science', 'error', 'dq', 'valid', 'emline', 'origin_xy', 'direct_xy', 'line_xy'):
            actual = np.stack([getattr(exposure, name) for exposure in target.exposures])
            np.testing.assert_array_equal(actual, saved[name], err_msg=f'Baseline native {name} changed')


def snapshot_inputs(output, inputs):
    paths, hashes = {}, {}
    for i, (original, digest) in enumerate(sorted(inputs.items())):
        relative = f'inputs/{i:03d}_{Path(original).name}'
        dest = output/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes(Path(original).read_bytes())
        if sha256(dest) != digest:
            raise ValueError(f'Input snapshot changed: {dest}')
        paths[original], hashes[relative] = relative, digest
    return paths, hashes


def run_locked(args):
    baseline_dir, output = args.baseline.resolve(), args.output.resolve()
    baseline = read_json(baseline_dir/'coupled.json')
    if baseline.get('format') != 'native_coupled_v1' or not baseline.get('completed'):
        raise ValueError('A completed coupled baseline is required.')
    if baseline.get('real_observations') is not True or baseline['source_id'] != args.source_id:
        raise ValueError('The baseline is not the requested real source.')
    modes = {fit['mode'] for fit in baseline['fits']}
    if len(modes) != 1 or not modes.issubset({'positive', 'coupled'}):
        raise ValueError('Select a baseline directory with one positive or coupled continuum model.')
    mode = next(iter(modes))
    sigmas = sorted(set(float(s) for s in args.sigmas))
    if not sigmas or not np.isfinite(sigmas).all() or min(sigmas) < 0 or max(sigmas) > 200:
        raise ValueError('Fixed dispersions must be finite and within 0–200 km/s.')
    if args.maxiter < 1 or args.refine < 1:
        raise ValueError('Positive iteration limits are required.')
    if args.device == 'cuda' and jax.default_backend() != 'gpu':
        raise RuntimeError('GPU requested but unavailable.')
    verify_hashes(baseline['input_sha256'], 'Baseline input')
    verify_hashes({str(REPO/p): digest for p,digest in baseline['code_sha256'].items()}, 'Baseline code')
    verify_hashes(baseline.get('calibration_sha256', {}), 'Calibration')
    verify_hashes({str(baseline_dir/'native_data.npz'): baseline['native_data_sha256']}, 'Baseline data')
    for fit in baseline['fits']:
        verify_hashes({str(baseline_dir/fit['artifact']): fit['artifact_sha256']}, 'Baseline fit')
    branches = starting_branches(baseline['fits'], mode)
    grid_dir = Path(baseline['settings']['grid'])
    grid = read_json(grid_dir/'grid.json')
    settings = dict(baseline=str(baseline_dir), source_id=args.source_id,
        models=list(args.models), variants=list(args.variants), sigmas=sigmas,
        maxiter=args.maxiter, refine=args.refine, stationarity_threshold=.1,
        continuum_mode=mode, seed_policy='Two separated stored branches, bidirectional continuation; '
        'four fresh kinematic angles at each exponential endpoint.')
    inputs = dict(baseline['input_sha256']) | dict(baseline.get('calibration_sha256', {}))
    inputs[str(baseline_dir/'coupled.json')] = sha256(baseline_dir/'coupled.json')
    inputs[str(baseline_dir/'native_data.npz')] = baseline['native_data_sha256']
    for fit in baseline['fits']:
        inputs[str(baseline_dir/fit['artifact'])] = fit['artifact_sha256']
    hashes = {p: sha256(REPO/p) for p in CODE}
    path = output/'dispersion.json'
    report = read_json(path) if path.exists() else dict(format='native_simple_dispersion_v1',
        source_id=args.source_id, real_observations=True, completed=False, settings=settings,
        input_sha256=inputs, code_sha256=hashes, fits=[], best_by_sigma=[],
        started_unix=time.time(), device=str(jax.devices()),
        interpretation='J = chi-square/2 + prior penalties. Conditional nuisance-refitted fixed-sigma '
        'profiles; not MCMC, confidence intervals, or model evidence. Compare score differences '
        'within each model/response scenario; do not use raw cross-model J to select morphology.',
        limitations=['LSF width +/-10% is an illustrative sensitivity test, not a measured uncertainty.',
            'First-order stationarity does not prove a global optimum; every attempted candidate is retained.',
            'The exponential gas profile is independent of stellar light and gas kinematic geometry.',
            'Finite source support, inclination transfer, masks and continuum assumptions remain.'])
    for key, expected in (('settings', settings), ('input_sha256', inputs), ('code_sha256', hashes)):
        if report[key] != expected:
            raise ValueError(f'Cannot resume changed {key}; use a new run directory.')
    for fit in report['fits']:
        verify_hashes({str(output/fit['artifact']): fit['artifact_sha256']}, 'Checkpoint artifact')
    for p, digest in report.get('input_snapshot_sha256', {}).items():
        verify_hashes({str(output/p): digest}, 'Input snapshot')
    if report.get('native_data_sha256'):
        verify_hashes({str(output/'native_data.npz'): report['native_data_sha256']}, 'Saved native data')
    for p in CODE:
        dest = output/'code'/p
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes((REPO/p).read_bytes())
        if sha256(dest) != hashes[p]:
            raise ValueError(f'Code snapshot changed: {dest}')
    if report['completed']:
        print('Already complete:', output, flush=True)
        return
    snapshots, snapshot_hashes = snapshot_inputs(output, inputs)
    report.update(input_snapshot_paths=snapshots, input_snapshot_sha256=snapshot_hashes)
    write_json(path, report)
    print(f'ID{args.source_id}: reconstructing saved baseline on {jax.devices()}', flush=True)
    with heartbeat('Loading real R+C data and response'):
        target, engine, old_like, _, _, _ = build_engine(grid)
        detector_mask = baseline['settings'].get('detector_mask')
        if detector_mask:
            report['detector_mask'] = apply_detector_mask(target, Path(detector_mask))
            arrays = [np.stack([getattr(e,k) for e in target.exposures]) for k in ('science','error','valid')]
            old_like = StripContinuumLikelihood(*arrays, [e.pupil for e in target.exposures], degree=2)
            engine.likelihood = JaxStripLikelihood(old_like)
        verify_native_arrays(target, baseline_dir/'native_data.npz')
        response, calibration = calibration_response(target, engine.projector, grid['settings']['bundle'])
    cal_hashes = {r['sensitivity_path']: r['sensitivity_sha256'] for r in calibration['records']}
    if cal_hashes != baseline['calibration_sha256']:
        raise ValueError('Calibrated response differs from the baseline.')
    native_dest = output/'native_data.npz'
    if not native_dest.exists():
        native_dest.write_bytes((baseline_dir/'native_data.npz').read_bytes())
    pupils = np.asarray([e.pupil for e in target.exposures])
    if set(pupils) != {'R', 'C'}:
        raise ValueError('This comparison requires both R and C exposures.')
    data, error, valid = [np.stack([getattr(e,k) for e in target.exposures]) for k in ('science','error','valid')]
    nuisance = dict(mode=mode, calibration_sigma=baseline['settings']['calibration_sigma'],
        background_scale=baseline['settings']['background_scale'],
        continuum_scale=baseline['settings']['continuum_scale'],
        center_kind=grid['settings'].get('center','catalog'),
        center_mean=grid['center_mean_east_north_arcsec'])
    original = CoupledNativeModel(engine, target, response, **nuisance)
    branch_arrays = []
    for fit in branches:
        with np.load(baseline_dir/fit['artifact']) as saved:
            if int(saved['source_id']) != args.source_id:
                raise ValueError('Starting artifact belongs to a different source.')
            branch_arrays.append({key: saved[key].copy() for key in ('parameters', 'source_flux')})
    with heartbeat('Checking reconstructed baseline objective'):
        saved_objective = float(jax.jit(original.objective)(jnp.asarray(branch_arrays[0]['parameters'])))
    np.testing.assert_allclose(saved_objective, branches[0]['objective'], rtol=1e-11, atol=1e-6,
        err_msg='Rebuilt flexible baseline objective disagrees with saved fit')
    report.update(native_data_sha256=sha256(native_dest), calibration=calibration,
        calibration_sha256=cal_hashes, source_summary=target.summary,
        exposures=[e.metadata for e in target.exposures], grid_settings=grid['settings'],
        inclination_prior=baseline['inclination_prior'], instrument_response=grid['instrument_response'],
        n_valid_pixels=int(valid.sum()), n_exposures=len(pupils),
        source_grid=dict(shape=[grid['settings']['quadrature']]*2, half_extent_arcsec=float(engine.half_extent),
                         spectral_quadrature_nodes=int(len(engine.projector.quadrature))),
        baseline_objective_reproduced=saved_objective, baseline_starts=[dict(f) for f in branches],
        nuisance_settings=nuisance, model_settings={})
    report['expected_fit_count'] = len(args.variants)*sum(
        2*len(sigmas)+(4*len(set((sigmas[0],sigmas[-1]))) if model=='exponential' else 0)
        for model in args.models)
    done = {(f['model'],f['variant'],f['fixed_sigma_kms'],f['start']): f for f in report['fits']}
    write_json(path, report)

    for variant in args.variants:
        spectral = VARIANTS[variant]
        kernels = scaled_kernels(grid['instrument_response']['exposures'], spectral, 1.)
        if variant == 'baseline':
            np.testing.assert_allclose(kernels, np.asarray(engine.projector.kernels), rtol=1e-14, atol=1e-16)
            kernels = np.asarray(engine.projector.kernels)
        else:
            kernels /= kernels.sum(axis=(-2,-1), keepdims=True)
        varied_engine = copy.copy(engine)
        varied_engine.projector = copy.copy(engine.projector)
        varied_engine.projector.kernels = jnp.asarray(kernels)
        for name in args.models:
            cls = CoupledNativeModel if name == 'flexible' else ExponentialCoupledModel
            model = cls(varied_engine, target, response, **nuisance)
            seeds = [a['parameters'].copy() if name=='flexible' else
                     model.from_flexible(a['parameters'], source_flux=a['source_flux']) for a in branch_arrays]
            free_bounds = [bound for i,bound in enumerate(model.bounds) if i != SIGMA_INDEX]
            def objective(free, sigma):
                return model.objective(unpack_free(free, sigma))
            compiled = jax.jit(jax.value_and_grad(objective))
            components = jax.jit(model.components)
            report['model_settings'][name] = dict(parameter_count=len(model.bounds),
                free_parameters_per_fixed_sigma=len(free_bounds), bounds=[list(b) for b in model.bounds],
                kinematic_center_bound_arcsec=[.1*x for x in model.bounds[6]],
                shape='Independent elliptical exponential (Sersic n=1)' if name=='exponential' else 'Flexible positive log-brightness map',
                gas_geometry_independent_of_stars=True,
                gas_geometry_independent_of_kinematic_geometry=True,
                emission_shape_bounds=dict(radius_arcsec=np.exp(model.bounds[8]).tolist(),
                                           axis_ratio=list(model.bounds[9])),
                emission_center_bound_arcsec=[.1*x for x in model.bounds[11]] if name=='exponential' else None)
            if name == 'exponential':
                report['model_settings'][name]['prior_description'] = model.prior_description()
            write_json(path, report)

            def fit_one(start, sigma, label, sweep):
                key = name, variant, sigma, label
                if key in done:
                    with np.load(output/done[key]['artifact']) as saved:
                        return saved['parameters'].copy()
                number = len(report['fits'])+1
                print(f'[{number}/{report["expected_fit_count"]}] ID{args.source_id} {name} {variant} '
                      f'sigma={sigma:g} km/s {label}; initial V={100*start[0]:.1f}', flush=True)
                begin = time.perf_counter()
                count, last = [0], [0.]
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
                        write_json(path, report)
                        last[0] = elapsed
                with heartbeat(f'{name}/{variant}/sigma={sigma:g}: optimizer'):
                    opt = minimize(fun, pack_free(start), jac=True, method='L-BFGS-B', bounds=free_bounds,
                        callback=progress, options=dict(maxiter=args.maxiter,gtol=.01,ftol=1e-12,maxls=40,maxcor=40))
                    pg = projected_gradient(opt.x, opt.jac, free_bounds)
                    if pg >= .1:
                        opt = minimize(fun, opt.x, jac=True, method='L-BFGS-B', bounds=free_bounds,
                            callback=progress, options=dict(maxiter=args.refine,gtol=.002,ftol=1e-13,maxls=50,maxcor=60))
                        pg = projected_gradient(opt.x, opt.jac, free_bounds)
                if not np.isfinite(opt.fun) or not np.isfinite(opt.x).all() or not np.isfinite(opt.jac).all():
                    report['active']['error'] = 'Nonfinite optimization result; earlier checkpoints preserved.'
                    write_json(path, report)
                    raise RuntimeError(report['active']['error'])
                parameters = np.asarray(unpack_free(jnp.asarray(opt.x), jnp.asarray(sigma)))
                value, parts = components(jnp.asarray(parameters))
                parts = {key:np.asarray(val) for key,val in parts.items()}
                if mode == 'coupled':
                    parts['continuum'] = old_like.continuum(torch.from_numpy(parts['line'].copy())).numpy()
                total = parts['line']+parts['continuum']+parts['background'][:,None,None]
                residual = np.divide(data-total,error,out=np.full_like(data,np.nan),where=valid)
                chi2 = float(np.nansum(residual**2))
                np.testing.assert_allclose(chi2, parts['chi2'], rtol=1e-10, atol=1e-6)
                np.testing.assert_allclose(float(value), .5*chi2+float(parts['penalty']), rtol=1e-10, atol=1e-6)
                x,speed,flux,gz,bz = model.split(parameters)
                description = varied_engine.describe(np.asarray(x),float(speed)) if name=='flexible' else model.describe(parameters)
                tag = f'fit_{len(report["fits"]):04d}.npz'
                save_npz(output/tag, source_id=np.asarray(args.source_id), parameters=parameters,
                    initial_parameters=np.asarray(start), vector=np.asarray(x), residual=residual,
                    source_east=np.asarray(engine.line_map.east), source_north=np.asarray(engine.line_map.north), **parts)
                edge = (np.abs(np.asarray(engine.line_map.east))>.8)|(np.abs(np.asarray(engine.line_map.north))>.8)
                entry = description | dict(model=name, mode=mode, variant=variant, fixed_sigma_kms=sigma,
                    start=label, sweep=sweep, spectral_scale=spectral, spatial_scale=1.,
                    objective=float(value), chi_square=chi2, penalty=float(parts['penalty']),
                    fixed_sigma_prior_penalty=.5*(sigma/100.)**2,
                    chi_square_R=float(np.nansum(residual[pupils=='R']**2)),
                    chi_square_C=float(np.nansum(residual[pupils=='C']**2)),
                    line_flux_1e17=float(flux), source_edge_flux_fraction=float(parts['source_flux'][edge].sum()),
                    gain_min_max=[float(parts['gains'].min()),float(parts['gains'].max())],
                    background_standardized=np.asarray(bz).tolist(), projected_gradient_inf=float(pg),
                    stationarity_passed=bool(pg < .1), optimizer_success=bool(opt.success),
                    message=str(opt.message), iterations=count[0], seconds=time.perf_counter()-begin,
                    boundary_parameters=boundary_parameters(parameters,model,name),
                    artifact=tag, artifact_sha256=sha256(output/tag))
                report['fits'].append(entry)
                report['best_by_sigma'] = choose_best(report['fits'])
                report['updated_unix'] = time.time()
                report.pop('active', None)
                write_json(path, report)
                done[key] = entry
                print(f'  saved V={float(speed):.2f}, sigma={sigma:g}, J={float(value):.3f}, '
                      f'gradient={pg:.3g}, {entry["seconds"]:.1f}s', flush=True)
                return parameters

            # Fresh orientation starts are retained individually. They do not
            # replace either independent stored branch at the sweep endpoints.
            if name == 'exponential':
                for sigma in sorted(set((sigmas[0],sigmas[-1]))):
                    for angle in (0,90,180,270):
                        fresh = seeds[0].copy()
                        fresh[0], fresh[1], fresh[2] = 3., np.log(.25), np.deg2rad(angle)
                        fresh[4] = 0.  # systemic velocity starts independently too
                        fresh[6:8] = 0.
                        fresh[10] = np.deg2rad(angle)
                        fit_one(fresh, sigma, f'fresh_angle_{angle}', 'endpoint')
            for sweep, values, seed in (('up',sigmas,seeds[0]), ('down',sigmas[::-1],seeds[1])):
                previous = seed
                for index,sigma in enumerate(values):
                    previous = fit_one(previous,sigma,f'branch_{sweep}',sweep)
                    # Carry a newly found fresh-angle endpoint basin through
                    # the interior, while retaining the saved-branch endpoint
                    # attempt separately for inspection and resume provenance.
                    if index == 0 and name == 'exponential':
                        endpoint = min((f for f in report['fits'] if f['model']==name and
                            f['variant']==variant and f['fixed_sigma_kms']==sigma),key=lambda f:f['objective'])
                        with np.load(output/endpoint['artifact']) as saved:
                            previous = saved['parameters'].copy()
            jax.clear_caches()
    report.update(completed=True, finished_unix=time.time(),
        all_selected_stationary=all(f['stationarity_passed'] for f in report['best_by_sigma']))
    report.pop('active',None)
    write_json(path,report)
    print(f'Completed {len(report["fits"])} fits; selected stationarity checks '
          f'{report["all_selected_stationary"]}. Results: {output}',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-id',type=int,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--models',nargs='+',choices=['flexible','exponential'],default=['flexible','exponential'])
    parser.add_argument('--variants',nargs='+',choices=list(VARIANTS),default=list(VARIANTS))
    parser.add_argument('--sigmas',nargs='+',type=float,default=[0,10,20,40,80,120,200])
    parser.add_argument('--maxiter',type=int,default=1200)
    parser.add_argument('--refine',type=int,default=4000)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args()
    if len(set(args.models)) != len(args.models) or len(set(args.variants)) != len(args.variants):
        raise ValueError('Duplicate model or response scenarios are not allowed.')
    torch.set_num_threads(4)
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.workflow.lock').open('w') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('This dispersion run is already active in another process.') from exc
        run_locked(args)


if __name__=='__main__':
    main()
