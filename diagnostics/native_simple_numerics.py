"""Check source/spectral quadrature at saved real-source dispersion fits.

All physical and calibration parameters are held fixed. Only the same continuum
coefficients profiled by the original likelihood are reprofiled. These checks
measure numerical changes, not new parameter estimates or confidence intervals.
"""
import argparse
import copy
import fcntl
import gc
import hashlib
import json
from pathlib import Path
import time

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import torch

from diagnostics.native_simple_dispersion import (
    CODE as DISPERSION_CODE, verify_hashes, verify_native_arrays)
from diagnostics.native_source_grid import build_engine
from diagnostics.native_real_posterior import heartbeat, save_npz
from diagnostics.native_real_dispersion import scaled_kernels
from dingo.native_coupled import CoupledNativeModel, calibration_response
from dingo.native_fitting import BilinearLineMap, StripContinuumLikelihood
from dingo.native_forward import NativeLineProjector
from dingo.native_jax import JaxLineMap, JaxNativeProjector, JaxStripLikelihood
from dingo.native_quality import apply_detector_mask
from dingo.native_simple import B1, ExponentialCoupledModel
from dingo.real_workflow import REPO, read_json, write_json, sha256


CODE = list(dict.fromkeys(DISPERSION_CODE + ['diagnostics/native_simple_numerics.py']))


def selected_fits(report, model, variant, sigmas, include_best=True):
    """Select the lowest stored objective, retaining a failed convergence flag."""
    available = [f for f in report['fits'] if f['model'] == model and f['variant'] == variant]
    if not available:
        raise ValueError(f'No saved {model}/{variant} fits.')
    requested = set(float(x) for x in sigmas)
    if include_best:
        requested.add(float(min(available, key=lambda f: f['objective'])['fixed_sigma_kms']))
    result = []
    for sigma in sorted(requested):
        matches = [f for f in available if np.isclose(f['fixed_sigma_kms'], sigma, rtol=0, atol=1e-10)]
        if not matches:
            raise ValueError(f'No saved {model}/{variant} fit at sigma={sigma:g}. '
                             'This check uses a separately optimized saved candidate at each width.')
        result.append(min(matches, key=lambda f: f['objective']))
    return result


def rebuild_saved_target(run, run_dir):
    """Follow the dispersion runner's source, calibration and mask reconstruction."""
    baseline_dir = Path(run['settings']['baseline'])
    baseline = read_json(baseline_dir/'coupled.json')
    if not baseline.get('completed') or baseline['source_id'] != run['source_id']:
        raise ValueError('The original coupled baseline is incomplete or for another source.')
    verify_hashes(run['input_sha256'], 'Dispersion input')
    verify_hashes({str(REPO/p): digest for p, digest in run['code_sha256'].items()}, 'Dispersion code')
    verify_hashes(baseline['input_sha256'], 'Coupled input')
    verify_hashes({str(REPO/p): digest for p, digest in baseline['code_sha256'].items()}, 'Coupled code')
    verify_hashes(baseline.get('calibration_sha256', {}), 'Calibration')
    verify_hashes({str(run_dir/'native_data.npz'): run['native_data_sha256'],
                   str(baseline_dir/'native_data.npz'): baseline['native_data_sha256']}, 'Native arrays')
    grid = read_json(Path(baseline['settings']['grid'])/'grid.json')
    target, engine, old_like, _, _, _ = build_engine(grid)
    if target.source_id != run['source_id']:
        raise ValueError('Reconstructed target belongs to another source.')
    detector_mask = baseline['settings'].get('detector_mask')
    if detector_mask:
        apply_detector_mask(target, Path(detector_mask))
        arrays = [np.stack([getattr(e, k) for e in target.exposures]) for k in ('science', 'error', 'valid')]
        old_like = StripContinuumLikelihood(*arrays, [e.pupil for e in target.exposures], degree=2)
        engine.likelihood = JaxStripLikelihood(old_like)
    verify_native_arrays(target, baseline_dir/'native_data.npz')
    verify_native_arrays(target, run_dir/'native_data.npz')
    response, calibration = calibration_response(target, engine.projector, grid['settings']['bundle'])
    calibration_hashes = {r['sensitivity_path']: r['sensitivity_sha256'] for r in calibration['records']}
    if calibration_hashes != baseline['calibration_sha256'] or calibration_hashes != run['calibration_sha256']:
        raise ValueError('Reconstructed calibrated response differs from saved inputs.')
    return target, engine, old_like, response, grid


def refined_engine(engine, target, controls, spatial, spectral, kernels):
    """Recalibrate every refined sky node; never interpolate a detector image."""
    grid = BilinearLineMap(controls, spatial, half_extent=engine.half_extent, dtype=torch.float64)
    operator = NativeLineProjector.from_target(target, grid.east.numpy(), grid.north.numpy(), kernels,
        np.ones(len(target.exposures)), spectral_samples=spectral, dtype=torch.float64, deposition='cubic')
    refined = copy.copy(engine)
    refined.line_map = JaxLineMap(grid)
    refined.projector = JaxNativeProjector(operator)
    return refined


def difference_stats(current, reference, error, valid, pupils):
    delta = np.divide(current-reference, error, out=np.zeros_like(current), where=valid)
    def summarize(mask):
        values = delta[mask]
        return dict(rms=float(np.sqrt(np.mean(values**2))), max_abs=float(np.max(np.abs(values))),
                    sum_squares=float(np.sum(values**2)))
    return dict(all=summarize(valid), **{
        pupil: summarize(valid & (pupils == pupil)[:, None, None]) for pupil in ('R', 'C')})


def refresh_comparisons(report, output, error, valid, pupils):
    """Record raw-line and continuum-reprofiled detector differences separately."""
    records = report['evaluations']
    base = tuple(report['baseline_quadrature'])
    finest = (max(report['settings']['spatial']), max(report['settings']['spectral']))
    groups = sorted({r['fixed_sigma_kms'] for r in records})
    for sigma in groups:
        group = {tuple(r['quadrature']): r for r in records if r['fixed_sigma_kms'] == sigma}
        for record in group.values():
            comparisons = {}
            with np.load(output/record['artifact'], allow_pickle=False) as current:
                for label, grid in [('baseline', base), ('finest_available_requested', finest),
                    ('previous_spatial', (max([x for x in report['settings']['spatial'] if x < record['quadrature'][0]], default=0), record['quadrature'][1]))]:
                    if grid not in group:
                        continue
                    ref = group[grid]
                    with np.load(output/ref['artifact'], allow_pickle=False) as previous:
                        comparisons[label] = dict(quadrature=list(grid),
                            objective_delta=record['objective']-ref['objective'],
                            chi_square_delta=record['chi_square']-ref['chi_square'],
                            line_difference_ERR=difference_stats(current['line'], previous['line'], error, valid, pupils),
                            total_difference_ERR=difference_stats(current['total'], previous['total'], error, valid, pupils))
            record['comparisons'] = comparisons


def run_locked(args):
    run_dir, output = args.run.resolve(), args.output.resolve()
    if output == run_dir:
        raise ValueError('Numerical checks require a separate output directory; preserve the fitted run.')
    raw_report = (run_dir/'dispersion.json').read_bytes()
    source = json.loads(raw_report)
    if source.get('format') != 'native_simple_dispersion_v1' or source.get('real_observations') is not True:
        raise ValueError('Select a saved real-data native_simple_dispersion_v1 run.')
    if not source.get('completed') and not args.allow_partial:
        raise ValueError('The dispersion run is incomplete. Use --allow-partial only for an explicit checkpoint diagnostic.')
    if args.device == 'cuda' and jax.default_backend() != 'gpu':
        raise RuntimeError('GPU requested but unavailable.')
    if args.device == 'cpu' and jax.default_backend() != 'cpu':
        raise RuntimeError('Set JAX_PLATFORMS=cpu for an explicitly requested CPU check.')
    selected = selected_fits(source, args.model, args.variant, args.sigmas, not args.no_include_best)
    base_spatial = int(source['grid_settings']['quadrature'])
    base_spectral = 9
    spatial = sorted(set([base_spatial, *args.spatial]))
    spectral = sorted(set([base_spectral, *args.spectral]))
    controls = int(source['grid_settings']['controls'])
    if min(spatial) < controls or min(spectral) < 1:
        raise ValueError('Spatial quadrature must be >= controls, and spectral quadrature positive.')
    for fit in selected:
        verify_hashes({str(run_dir/fit['artifact']): fit['artifact_sha256']}, 'Selected fit')
    settings = dict(run=str(run_dir), model=args.model, variant=args.variant, spatial=spatial, spectral=spectral,
        selected_sigmas=[float(f['fixed_sigma_kms']) for f in selected],
        source_report_sha256=hashlib.sha256(raw_report).hexdigest())
    hashes = {p: sha256(REPO/p) for p in CODE}
    path = output/'numerics.json'
    report = read_json(path) if path.exists() else dict(format='native_simple_numerics_v1',
        real_observations=True, source_id=source['source_id'], completed=False, settings=settings,
        baseline_quadrature=[base_spatial, base_spectral], source_run_completed=bool(source['completed']),
        selected_fits=selected, evaluations=[], code_sha256=hashes, started_unix=time.time(), device=str(jax.devices()),
        interpretation='Numerical re-evaluation of fixed saved physical parameters; continuum coefficients alone '
            'are reprofiled. Objective shifts are not refitted profile shifts or parameter uncertainties.',
        limitations=['Finite source support and instrument response are unchanged.',
            'The finest requested grid is a comparison reference, not a guarantee of convergence.',
            'Each dispersion uses its separately saved preferred candidate; parameter optimization is not rerun.'])
    for name, value in [('settings', settings), ('code_sha256', hashes)]:
        if report[name] != value:
            raise ValueError(f'Cannot resume changed {name}; use a new output directory.')
    for record in report['evaluations']:
        verify_hashes({str(output/record['artifact']): record['artifact_sha256']}, 'Numerics checkpoint')
    if report['completed']:
        print('Already complete:', output, flush=True)
        return
    snapshot = output/'source_dispersion_snapshot.json'
    if snapshot.exists() and snapshot.read_bytes() != raw_report:
        raise ValueError('Source report snapshot changed; use a new directory.')
    snapshot.write_bytes(raw_report)
    for name in CODE:
        dest = output/'code'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes((REPO/name).read_bytes())
        verify_hashes({str(dest): hashes[name]}, 'Numerics code snapshot')
    write_json(path, report)
    with heartbeat('Rebuilding exact real-data target for numerical checks'):
        target, engine, old_like, response, grid = rebuild_saved_target(source, run_dir)
    data, error, valid = [np.stack([getattr(e, k) for e in target.exposures]) for k in ('science', 'error', 'valid')]
    pupils = np.asarray([e.pupil for e in target.exposures])
    if set(pupils) != {'R', 'C'}:
        raise ValueError('Both R and C observations are required.')
    report.update(n_valid_pixels=int(valid.sum()), n_exposures=len(pupils),
        source_support_half_extent_arcsec=float(engine.half_extent),
        input_sha256=dict(source['input_sha256']) | {str(run_dir/f['artifact']): f['artifact_sha256'] for f in selected},
        source_report_snapshot_sha256=sha256(snapshot), native_data_sha256=source['native_data_sha256'])
    save_npz(output/'native_data.npz', source_id=np.asarray(target.source_id), science=data, error=error,
             valid=valid, pupils=pupils)
    report['saved_native_data_sha256'] = sha256(output/'native_data.npz')
    kernels = np.asarray(engine.projector.kernels)
    if args.variant != 'baseline':
        kernels = scaled_kernels(source['instrument_response']['exposures'], selected[0]['spectral_scale'], 1.)
        kernels /= kernels.sum(axis=(-2, -1), keepdims=True)
    nuisance = source['nuisance_settings']
    done = {(r['fixed_sigma_kms'], *r['quadrature']) for r in report['evaluations']}
    cls = ExponentialCoupledModel if args.model == 'exponential' else CoupledNativeModel
    for nsky in spatial:
        for nspec in spectral:
            pending = [f for f in selected if (float(f['fixed_sigma_kms']), nsky, nspec) not in done]
            if not pending:
                continue
            print(f'ID{target.source_id}: {nsky}x{nsky} sky nodes, {nspec} spectral nodes', flush=True)
            with heartbeat('Constructing calibrated refined source projector'):
                refined = refined_engine(engine, target, controls, nsky, nspec, kernels)
                model = cls(refined, target, response, **nuisance)
                components = jax.jit(model.components)
            for fit in pending:
                start = time.perf_counter()
                sigma = float(fit['fixed_sigma_kms'])
                with np.load(run_dir/fit['artifact'], allow_pickle=False) as saved:
                    parameters = saved['parameters'].copy()
                    if int(saved['source_id']) != target.source_id:
                        raise ValueError('Selected parameter artifact belongs to another source.')
                if parameters.shape != (len(model.bounds),) or not np.isclose(100*parameters[5], sigma):
                    raise ValueError('Saved parameters do not match this model or fixed dispersion.')
                with heartbeat(f'Fixed parameters, sigma={sigma:g}: evaluating refined prediction'):
                    value, parts = components(jnp.asarray(parameters))
                    value = float(value)
                    parts = {key: np.asarray(val) for key, val in parts.items()}
                if nuisance['mode'] == 'coupled':
                    parts['continuum'] = old_like.continuum(torch.from_numpy(parts['line'].copy())).numpy()
                total = parts['line']+parts['continuum']+parts['background'][:, None, None]
                residual = np.divide(data-total, error, out=np.full_like(data, np.nan), where=valid)
                chi2 = float(np.sum(residual[valid]**2))
                np.testing.assert_allclose(chi2, parts['chi2'], rtol=1e-10, atol=1e-6)
                np.testing.assert_allclose(value, .5*chi2+float(parts['penalty']), rtol=1e-10, atol=1e-6)
                if (nsky, nspec) == (base_spatial, base_spectral):
                    np.testing.assert_allclose(value, fit['objective'], rtol=1e-11, atol=1e-6,
                        err_msg='Original-grid objective does not reproduce saved candidate')
                if not np.isfinite(value) or not np.isfinite(total[valid]).all():
                    raise ValueError('Nonfinite numerical evaluation.')
                tag = f'evaluation_{len(report["evaluations"]):03d}.npz'
                save_npz(output/tag, source_id=np.asarray(target.source_id), parameters=parameters,
                    total=total, residual=residual, source_east=np.asarray(refined.line_map.east),
                    source_north=np.asarray(refined.line_map.north), **parts)
                spacing = 2*engine.half_extent/(nsky-1)
                minor = (float(np.exp(parameters[8])*parameters[9]/B1) if args.model == 'exponential' else None)
                edge = (np.abs(np.asarray(refined.line_map.east)) > .8) | (np.abs(np.asarray(refined.line_map.north)) > .8)
                entry = dict(fixed_sigma_kms=sigma, quadrature=[nsky, nspec], objective=value, chi_square=chi2,
                    penalty=float(parts['penalty']), original_objective=float(fit['objective']),
                    objective_delta_from_original=value-float(fit['objective']),
                    original_stationarity_passed=bool(fit['stationarity_passed']),
                    source_fit_artifact=fit['artifact'], source_fit_sha256=fit['artifact_sha256'],
                    sky_spacing_arcsec=float(spacing), minor_exponential_scale_arcsec=minor,
                    sky_spacing_per_minor_scale=float(spacing/minor) if minor is not None else None,
                    source_edge_flux_fraction=float(parts['source_flux'][edge].sum()),
                    artifact=tag, artifact_sha256=sha256(output/tag), seconds=time.perf_counter()-start)
                report['evaluations'].append(entry)
                done.add((sigma, nsky, nspec))
                refresh_comparisons(report, output, error, valid, pupils)
                report['updated_unix'] = time.time()
                write_json(path, report)
                print(f'  sigma={sigma:g}: J={value:.6f}, delta saved J={value-fit["objective"]:+.6f}; '
                      f'{entry["seconds"]:.1f}s', flush=True)
            del components, model, refined
            jax.clear_caches()
            gc.collect()
    refresh_comparisons(report, output, error, valid, pupils)
    report.update(completed=True, finished_unix=time.time())
    write_json(path, report)
    print('Numerical checks complete:', output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--model', choices=['exponential', 'flexible'], default='exponential')
    parser.add_argument('--variant', choices=['baseline', 'lsf_0p9', 'lsf_1p1'], default='baseline')
    parser.add_argument('--sigmas', nargs='+', type=float, default=[0., 20., 80.])
    parser.add_argument('--spatial', nargs='+', type=int, default=[61, 121, 181])
    parser.add_argument('--spectral', nargs='+', type=int, default=[9, 17])
    parser.add_argument('--no-include-best', action='store_true')
    parser.add_argument('--allow-partial', action='store_true',
        help='Diagnose the explicit saved checkpoint of a still-incomplete run; do not resume after it changes.')
    args = parser.parse_args()
    if not np.isfinite(args.sigmas).all() or min(args.sigmas) < 0 or max(args.sigmas) > 200:
        raise ValueError('Request finite saved dispersions in 0–200 km/s.')
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'.workflow.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('This numerical check is already active in another process.') from exc
        run_locked(args)


if __name__ == '__main__':
    main()
